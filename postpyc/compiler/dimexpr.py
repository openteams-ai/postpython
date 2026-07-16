"""Dimension-expression mini-language for computed output core dimensions.

A gufunc layout signature may declare an output core dimension whose size is
computed from input core dimensions, using the named expression form::

    (n,d)->(m=n*(n-1)//2)      # ppspatial pdist, condensed form
    (n),(m)->(k=n+m-1)         # ppsignal convolve

This module owns the expression language end to end — parsing, evaluation,
and C rendering — so the interpreted runtime and the C backend cannot drift.

Grammar (deliberately small)::

    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '//') factor)*
    factor := INT | NAME | '(' expr ')'

``NAME`` matches ``[a-z][a-z0-9_]*`` (the same alphabet as layout-signature
core-dim names); ``INT`` is a decimal literal.

Restrictions (v1, widening later is compatible):

- ``/`` is rejected with a hint to use ``//``; ``%`` and unary minus are
  rejected outright.
- The divisor of ``//`` must be a positive integer literal, so evaluation
  can never divide by zero and emitted C can never SIGFPE.
- Python floor-division semantics are normative in BOTH execution modes:
  ``evaluate`` uses Python's ``//``, and ``to_c`` renders ``//`` through the
  ``__pp_floordiv_si`` helper the C backend emits (C's ``/`` truncates
  toward zero, which disagrees with floor for negative dividends).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator, Union


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimConst:
    value: int


@dataclass(frozen=True)
class DimName:
    name: str


@dataclass(frozen=True)
class DimBinOp:
    op: str            # '+', '-', '*', '//'
    left: "DimExpr"
    right: "DimExpr"


DimExpr = Union[DimConst, DimName, DimBinOp]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<name>[a-z][a-z0-9_]*)|(?P<int>[0-9]+)|(?P<op>//|[+\-*()])"
    r"|(?P<bad>\S))"
)


def _tokenize(text: str) -> Iterator[tuple[str, str]]:
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:  # only trailing whitespace remains
            break
        pos = m.end()
        if m.lastgroup == "bad":
            ch = m.group("bad")
            if ch == "/":
                raise ValueError(
                    f"invalid dimension expression {text!r}: "
                    "'/' is not supported; use floor division '//'"
                )
            if ch == "%":
                raise ValueError(
                    f"invalid dimension expression {text!r}: "
                    "'%' is not supported in dimension expressions"
                )
            raise ValueError(
                f"invalid dimension expression {text!r}: unexpected character {ch!r}"
            )
        kind = m.lastgroup
        assert kind is not None
        yield kind, m.group(kind)
    yield "end", ""


# ---------------------------------------------------------------------------
# Parser (recursive descent over the grammar above)
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._tokens = list(_tokenize(text))
        self._pos = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _next(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _error(self, message: str) -> ValueError:
        return ValueError(
            f"invalid dimension expression {self._text!r}: {message}"
        )

    def parse(self) -> DimExpr:
        expr = self._expr()
        kind, value = self._peek()
        if kind != "end":
            raise self._error(f"unexpected {value!r}")
        return expr

    def _expr(self) -> DimExpr:
        node = self._term()
        while self._peek() == ("op", "+") or self._peek() == ("op", "-"):
            _, op = self._next()
            node = DimBinOp(op, node, self._term())
        return node

    def _term(self) -> DimExpr:
        node = self._factor()
        while self._peek() == ("op", "*") or self._peek() == ("op", "//"):
            _, op = self._next()
            right = self._factor()
            if op == "//" and not (
                isinstance(right, DimConst) and right.value > 0
            ):
                raise self._error(
                    "the divisor of '//' must be a positive integer literal"
                )
            node = DimBinOp(op, node, right)
        return node

    def _factor(self) -> DimExpr:
        kind, value = self._next()
        if kind == "int":
            return DimConst(int(value))
        if kind == "name":
            return DimName(value)
        if (kind, value) == ("op", "("):
            node = self._expr()
            kind, value = self._next()
            if (kind, value) != ("op", ")"):
                raise self._error("expected ')'")
            return node
        if (kind, value) == ("op", "-"):
            raise self._error("unary minus is not supported")
        if kind == "end":
            raise self._error("unexpected end of expression")
        raise self._error(f"unexpected {value!r}")


def parse_dim_expr(text: str) -> DimExpr:
    """Parse a dimension expression, raising ValueError on any misuse."""
    if not text.strip():
        raise ValueError("invalid dimension expression: empty expression")
    return _Parser(text).parse()


# ---------------------------------------------------------------------------
# Queries and evaluation
# ---------------------------------------------------------------------------

def free_names(expr: DimExpr) -> set[str]:
    """The set of dimension names the expression references."""
    if isinstance(expr, DimName):
        return {expr.name}
    if isinstance(expr, DimBinOp):
        return free_names(expr.left) | free_names(expr.right)
    return set()


def evaluate(expr: DimExpr, sizes: dict[str, int]) -> int:
    """Evaluate with Python integer semantics ('//' floors, exact ints).

    The final result must fit in int64 — dimension sizes are int64 on every
    ABI surface — otherwise ValueError. Intermediates are exact Python ints,
    so there is no intermediate-overflow hazard on this path.
    """
    result = _evaluate(expr, sizes)
    if not (INT64_MIN <= result <= INT64_MAX):
        raise ValueError(
            f"computed dimension value {result} does not fit in int64"
        )
    return result


def _evaluate(expr: DimExpr, sizes: dict[str, int]) -> int:
    if isinstance(expr, DimConst):
        return expr.value
    if isinstance(expr, DimName):
        try:
            return sizes[expr.name]
        except KeyError:
            raise ValueError(
                f"unknown dimension name {expr.name!r} in dimension expression"
            ) from None
    left = _evaluate(expr.left, sizes)
    right = _evaluate(expr.right, sizes)
    if expr.op == "+":
        return left + right
    if expr.op == "-":
        return left - right
    if expr.op == "*":
        return left * right
    assert expr.op == "//"
    return left // right   # divisor is a positive literal by construction


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "//": 2}


def render(expr: DimExpr) -> str:
    """Canonical human-readable text (minimal parentheses, no spaces).

    Used for manifests, headers, and diagnostics; stable across parses:
    render(parse_dim_expr(s)) is a fixed point.
    """
    if isinstance(expr, DimConst):
        return str(expr.value)
    if isinstance(expr, DimName):
        return expr.name
    prec = _PRECEDENCE[expr.op]
    left = render(expr.left)
    if isinstance(expr.left, DimBinOp) and _PRECEDENCE[expr.left.op] < prec:
        left = f"({left})"
    right = render(expr.right)
    if isinstance(expr.right, DimBinOp):
        rp = _PRECEDENCE[expr.right.op]
        # Integer '+' and '*' chains are exactly associative; everything
        # else needs parens at equal precedence on the right.
        if rp < prec or (rp == prec and expr.op not in ("+", "*")):
            right = f"({right})"
        elif rp == prec and expr.op == "*" and expr.right.op == "//":
            right = f"({right})"
    return f"{left}{expr.op}{right}"


def to_c(expr: DimExpr, name_of: Callable[[str], str]) -> str:
    """Render as a C expression over int64 values.

    ``name_of`` maps a dimension name to its C lvalue (e.g. ``_pp_dim_n``).
    '//' renders through __pp_floordiv_si so compiled floor-division agrees
    with the interpreted evaluator for negative dividends. Fully
    parenthesized: C precedence can never reorder the tree.
    """
    if isinstance(expr, DimConst):
        return str(expr.value)
    if isinstance(expr, DimName):
        return name_of(expr.name)
    left = to_c(expr.left, name_of)
    right = to_c(expr.right, name_of)
    if expr.op == "//":
        return f"__pp_floordiv_si({left}, {right})"
    # Each binop renders as one parenthesized unit, so C operator
    # precedence can never reassociate the tree.
    return f"({left} {expr.op} {right})"


# C runtime helpers that ``to_c_checked`` output references. A translation unit
# that renders checked dimension expressions must emit this once.
CHECKED_ARITH_C = """\
/* int64 overflow-checked +, -, * for computed dimension sizes: set *ovf on
   overflow so the caller can raise instead of wrapping, matching the int64
   range check in dimexpr.evaluate. Floor division (//) is by a positive
   literal (parser-enforced) and cannot overflow, so it keeps __pp_floordiv_si. */
static int64_t __pp_ckd_mul_i64(int64_t a, int64_t b, int *ovf) {
    int64_t r; if (__builtin_mul_overflow(a, b, &r)) *ovf = 1; return r;
}
static int64_t __pp_ckd_add_i64(int64_t a, int64_t b, int *ovf) {
    int64_t r; if (__builtin_add_overflow(a, b, &r)) *ovf = 1; return r;
}
static int64_t __pp_ckd_sub_i64(int64_t a, int64_t b, int *ovf) {
    int64_t r; if (__builtin_sub_overflow(a, b, &r)) *ovf = 1; return r;
}
"""

_CHECKED_C_FUNC = {"+": "__pp_ckd_add_i64", "-": "__pp_ckd_sub_i64", "*": "__pp_ckd_mul_i64"}


def to_c_checked(expr: DimExpr, name_of: Callable[[str], str], overflow_flag: str) -> str:
    """Render as a C int64 expression whose +, -, * are overflow-checked.

    Every add/sub/mul goes through a ``__pp_ckd_*`` helper (see
    ``CHECKED_ARITH_C``) that sets the C ``int`` lvalue named by
    *overflow_flag* on int64 overflow, so the compiled evaluation raises on
    the same inputs ``dimexpr.evaluate`` rejects instead of silently wrapping
    (C signed overflow is undefined behaviour). ``//`` renders through
    ``__pp_floordiv_si`` exactly as :func:`to_c`.
    """
    if isinstance(expr, DimConst):
        return str(expr.value)
    if isinstance(expr, DimName):
        return name_of(expr.name)
    left = to_c_checked(expr.left, name_of, overflow_flag)
    right = to_c_checked(expr.right, name_of, overflow_flag)
    if expr.op == "//":
        return f"__pp_floordiv_si({left}, {right})"
    return f"{_CHECKED_C_FUNC[expr.op]}({left}, {right}, &{overflow_flag})"
