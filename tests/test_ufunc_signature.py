"""Vectorized-function layout signature parser tests."""

import pytest

from postpython.compiler.frontend import compile_source
from postpython.ufunc import guvectorize, parse_layout_signature


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
from postpython import guvectorize
from postyp import Float64

@guvectorize([], "(n)->(m)")
def f(x: Float64, out: Float64) -> None:
    out = x
"""

    _, errors = compile_source(source)

    assert [error.code for error in errors] == ["PP100"]
    assert "output dimension 'm' does not appear in an input" in errors[0].message
