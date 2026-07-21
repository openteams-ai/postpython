"""Vectorized-function layout signature parser tests."""

import pytest

from postpyc.compiler.dimexpr import evaluate, render
from postpyc.compiler.frontend import compile_source
from postpyc.ufunc import guvectorize, parse_layout_signature


@pytest.mark.parametrize(
    ("signature", "inputs", "outputs"),
    [
        ("()->()", [[]], [[]]),
        ("(n)->(n)", [["n"]], [["n"]]),
        ("(n),(n)->()", [["n"], ["n"]], [[]]),
        ("(m,k),(k,n)->(m,n)", [["m", "k"], ["k", "n"]], [["m", "n"]]),
        (" (n) , (n) -> () ", [["n"], ["n"]], [[]]),
    ],
)
def test_valid_layout_signatures(signature, inputs, outputs):
    parsed = parse_layout_signature(signature)

    assert parsed.inputs == inputs
    assert parsed.outputs == outputs
    assert parsed.computed_dims == {}


# ---------------------------------------------------------------------------
# Computed output core dimensions — the named expression form (issue #38)
# ---------------------------------------------------------------------------

def test_computed_dim_pdist_signature():
    parsed = parse_layout_signature("(n,d)->(m=n*(n-1)//2)")

    assert parsed.inputs == [["n", "d"]]
    assert parsed.outputs == [["m"]]           # name only — plumbing-uniform
    assert list(parsed.computed_dims) == ["m"]
    assert render(parsed.computed_dims["m"]) == "n*(n-1)//2"
    assert evaluate(parsed.computed_dims["m"], {"n": 4, "d": 3}) == 6
    # The name-only rendering is what NumPy registration will consume.
    assert str(parsed) == "(n,d)->(m)"
    # Computed dims are ordinary core dims: pp_dim_m plumbing stays uniform.
    assert parsed.core_dims == ["n", "d", "m"]


def test_computed_dim_convolve_signature():
    parsed = parse_layout_signature("(n),(m)->(k=n+m-1)")

    assert parsed.inputs == [["n"], ["m"]]
    assert parsed.outputs == [["k"]]
    assert evaluate(parsed.computed_dims["k"], {"n": 3, "m": 4}) == 6
    assert str(parsed) == "(n),(m)->(k)"


def test_computed_dim_tolerates_whitespace():
    parsed = parse_layout_signature(" (n,d) -> ( m = n * (n-1) // 2 ) ")

    assert parsed.outputs == [["m"]]
    assert render(parsed.computed_dims["m"]) == "n*(n-1)//2"


def test_computed_dim_reusable_as_bare_name_in_other_outputs():
    # A computed name is a real core dim; later output groups may reuse it.
    parsed = parse_layout_signature("(n)->(m=n+1),(m)")

    assert parsed.outputs == [["m"], ["m"]]
    assert list(parsed.computed_dims) == ["m"]


@pytest.mark.parametrize(
    ("signature", "match"),
    [
        # Anonymous expressions must be named.
        ("(n)->(n*(n-1)//2)", "must be named"),
        # '((n))' stays invalid: not a name, not a 'name=expr'.
        ("(n)->((n))", "must be named"),
        # Unknown symbol inside an expression names the symbol.
        ("(n)->(m=q+1)", "unknown dimension 'q'"),
        # True division and modulo are rejected with guidance.
        ("(n)->(m=n/2)", "use floor division '//'"),
        ("(n)->(m=n%2)", "'%' is not supported"),
        # Division safety: literal-positive divisors only.
        ("(n)->(m=n//0)", "positive integer literal"),
        # Expressions are output-side only.
        ("(m=n)->(n)", "only allowed in output groups"),
        # Name hygiene.
        ("(n)->(n=n+1)", "collides with an input dimension"),
        ("(n)->(m=n)", "alias of input dimension 'n'"),
        ("(n)->(m=3)", "must reference at least one input dimension"),
        ("(n)->(m=n+1),(m=n+2)", "defined more than once"),
    ],
)
def test_invalid_computed_dim_signatures(signature, match):
    with pytest.raises(ValueError, match=match):
        parse_layout_signature(signature)


@pytest.mark.parametrize(
    "signature",
    [
        "(N)->()",
        "n)->()",
        "(n)->(m)",
        "(n),(n)->",
        "()->()junk",
        "(n,)->()",
        "(n,,m)->()",
        "(n)->(n),",
        "(n)->((n))",
    ],
)
def test_invalid_layout_signatures_are_rejected(signature):
    with pytest.raises(ValueError):
        parse_layout_signature(signature)


def test_guvectorize_decorator_rejects_invalid_signature():
    with pytest.raises(ValueError, match="@guvectorize: invalid signature"):

        @guvectorize([], "(N)->()")
        def identity(x: int, out: int) -> None:
            out = x


def test_compile_source_reports_invalid_guvectorize_signature():
    source = """\
from postpyc import guvectorize
from postyp import Float64

@guvectorize([], "(n)->(m)")
def f(x: Float64, out: Float64) -> None:
    out = x
"""

    _, errors = compile_source(source)

    assert [error.code for error in errors] == ["PP100"]
    assert "output dimension 'm' does not appear in an input" in errors[0].message


def test_compile_source_lowers_computed_dims_without_error():
    """Slice 38b: computed output core dims lower cleanly (no PP900)."""
    source = """\
from postpyc import guvectorize
from postyp import Array, Float64

@guvectorize([], "(n,d)->(m=n*(n-1)//2)")
def pdist(points: Array[Float64], out: Array[Float64]) -> None:
    out[0] = 0.0
"""

    module, errors = compile_source(source)

    assert errors == []
    pdist = module.get_function("pdist")
    # `m` is a real core dim: the kernel takes it as a trailing size param.
    assert "m" in pdist.ufunc_sig.core_dims
    assert "pp_dim_m" in {p.name for p in pdist.core_dim_params}


def test_guvectorize_decorator_accepts_computed_dim_signature():
    @guvectorize([], "(n,d)->(m=n*(n-1)//2)")
    def pdist(points, out) -> None:  # pragma: no cover - never invoked
        out[0] = 0.0

    parsed = pdist.__pp_ufunc_sig__
    assert parsed.outputs == [["m"]]
    assert "m" in parsed.computed_dims


def test_vectorize_kernels_never_carry_computed_dims():
    """Computed dims cannot reach the @vectorize path: the frontend always
    synthesizes a scalar `()->()` signature for it, so the whole computed-dim
    machinery (and its NumPy 2.1 requirement) is structurally out of reach for
    element-wise kernels rather than merely rejected after the fact.
    """
    source = """\
from postpyc import vectorize
from postyp import Float64

@vectorize
def scale(x: Float64, y: Float64) -> Float64:
    return x * y
"""
    module, errors = compile_source(source)
    assert errors == []
    fn = module.get_function("scale")
    assert str(fn.ufunc_sig) == "(),()->()"
    assert fn.ufunc_sig.computed_dims == {}
    assert fn.core_dim_params == []
