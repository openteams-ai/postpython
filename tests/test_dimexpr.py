"""Dimension-expression mini-language tests (postpyc.compiler.dimexpr)."""

import pytest

from postpyc.compiler.dimexpr import (
    DimBinOp,
    DimConst,
    DimName,
    evaluate,
    free_names,
    parse_dim_expr,
    render,
    to_c,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_pdist_expression():
    expr = parse_dim_expr("n*(n-1)//2")

    assert expr == DimBinOp(
        "//",
        DimBinOp("*", DimName("n"), DimBinOp("-", DimName("n"), DimConst(1))),
        DimConst(2),
    )


def test_parse_convolve_expression():
    expr = parse_dim_expr("n+m-1")

    # Left-associative: (n+m)-1
    assert expr == DimBinOp(
        "-", DimBinOp("+", DimName("n"), DimName("m")), DimConst(1)
    )


def test_parse_tolerates_whitespace():
    assert parse_dim_expr(" n * ( n - 1 ) // 2 ") == parse_dim_expr("n*(n-1)//2")


def test_parse_precedence():
    # a+b*c parses as a+(b*c)
    assert parse_dim_expr("a+b*c") == DimBinOp(
        "+", DimName("a"), DimBinOp("*", DimName("b"), DimName("c"))
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("n/2", "use floor division '//'"),
        ("n%2", "'%' is not supported"),
        ("-n", "unary minus is not supported"),
        ("n//0", "positive integer literal"),
        ("n//m", "positive integer literal"),
        ("n+", "unexpected end of expression"),
        ("(n", "expected '\\)'"),
        ("n)", "unexpected '\\)'"),
        ("", "empty expression"),
        ("2n", "unexpected 'n'"),
        ("n ? 2", "unexpected character '\\?'"),
    ],
)
def test_parse_rejections(text, match):
    with pytest.raises(ValueError, match=match):
        parse_dim_expr(text)


# ---------------------------------------------------------------------------
# Queries and evaluation
# ---------------------------------------------------------------------------

def test_free_names():
    assert free_names(parse_dim_expr("n*(n-1)//2")) == {"n"}
    assert free_names(parse_dim_expr("n+m-1")) == {"n", "m"}
    assert free_names(parse_dim_expr("3+4")) == set()


@pytest.mark.parametrize(
    ("text", "sizes", "expected"),
    [
        ("n*(n-1)//2", {"n": 4}, 6),
        ("n*(n-1)//2", {"n": 1}, 0),
        ("n*(n-1)//2", {"n": 0}, 0),
        ("n+m-1", {"n": 3, "m": 4}, 6),
        ("n+m-1", {"n": 0, "m": 0}, -1),
        # Floor division battery: Python semantics for negative dividends.
        ("(n-5)//2", {"n": 3}, -1),   # C trunc would give 0
        ("(n-5)//2", {"n": 0}, -3),   # C trunc would give -2
        ("(n-5)//2", {"n": 5}, 0),
        ("(n-5)//2", {"n": 9}, 2),
    ],
)
def test_evaluate(text, sizes, expected):
    assert evaluate(parse_dim_expr(text), sizes) == expected


def test_evaluate_unknown_name():
    with pytest.raises(ValueError, match="unknown dimension name 'q'"):
        evaluate(parse_dim_expr("q+1"), {"n": 3})


def test_evaluate_int64_range():
    # n*n overflows int64 for n = 2**32
    with pytest.raises(ValueError, match="does not fit in int64"):
        evaluate(parse_dim_expr("n*n"), {"n": 2**32})
    # ... but intermediates are exact Python ints: only the RESULT must fit.
    assert evaluate(parse_dim_expr("n*n-n*n+n"), {"n": 2**32}) == 2**32


def test_parenthesized_literal_divisor_is_accepted():
    # '(2)' unwraps to the literal 2, satisfying the divisor rule.
    assert evaluate(parse_dim_expr("n//(2)"), {"n": 7}) == 3


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["n*(n-1)//2", "n+m-1", "a+b*c", "(a+b)*c", "a-(b+c)", "(a-b)//3"],
)
def test_render_is_a_fixed_point(text):
    once = render(parse_dim_expr(text))
    assert render(parse_dim_expr(once)) == once


def test_render_canonical_forms():
    assert render(parse_dim_expr("n*(n-1)//2")) == "n*(n-1)//2"
    assert render(parse_dim_expr(" n + m - 1 ")) == "n+m-1"
    assert render(parse_dim_expr("(a+b)*c")) == "(a+b)*c"


def test_render_preserves_semantics():
    # Canonical text must evaluate identically to the original.
    for text in ["n*(n-1)//2", "a-(b+c)", "a+b*c", "(a-b)//3"]:
        expr = parse_dim_expr(text)
        again = parse_dim_expr(render(expr))
        sizes = {"n": 7, "a": 10, "b": 3, "c": 2}
        assert evaluate(expr, sizes) == evaluate(again, sizes)


def test_to_c_uses_floordiv_helper():
    c = to_c(parse_dim_expr("n*(n-1)//2"), lambda n: f"_pp_dim_{n}")

    assert c == "__pp_floordiv_si((_pp_dim_n * (_pp_dim_n - 1)), 2)"


def test_to_c_plain_arithmetic():
    c = to_c(parse_dim_expr("n+m-1"), lambda n: f"_pp_dim_{n}")

    assert c == "((_pp_dim_n + _pp_dim_m) - 1)"
