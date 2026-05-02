"""Generalized ufunc signature parser tests."""

import pytest

from postpython.compiler.frontend import compile_source, parse_gufunc_sig
from postpython.gufunc import gufunc


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
def test_valid_gufunc_signatures(signature, inputs, outputs):
    parsed = parse_gufunc_sig(signature)

    assert parsed.inputs == inputs
    assert parsed.outputs == outputs


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
def test_invalid_gufunc_signatures_are_rejected(signature):
    with pytest.raises(ValueError):
        parse_gufunc_sig(signature)


def test_gufunc_decorator_rejects_invalid_signature():
    with pytest.raises(ValueError, match="@gufunc: invalid signature"):

        @gufunc("(N)->()")
        def identity(x: int) -> int:
            return x


def test_compile_source_reports_invalid_gufunc_signature():
    source = """\
from postpython.gufunc import gufunc
from postyp import Float64

@gufunc("(n)->(m)")
def f(x: Float64) -> Float64:
    return x
"""

    _, errors = compile_source(source)

    assert [error.code for error in errors] == ["PP100"]
    assert "output dimension 'm' does not appear in an input" in errors[0].message
