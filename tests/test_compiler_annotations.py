"""Compiler annotation validation tests."""

from postpython.compiler.frontend import compile_source


def error_codes(source: str) -> list[str]:
    _, errors = compile_source(source)
    return [error.code for error in errors]


def error_messages(source: str) -> list[str]:
    _, errors = compile_source(source)
    return [error.message for error in errors]


def test_unknown_parameter_annotation_is_rejected():
    source = """\
def f(x: dict) -> int:
    return 1
"""

    assert error_codes(source) == ["PP100"]
    assert "unsupported annotation for parameter `x`" in error_messages(source)[0]


def test_unknown_return_annotation_is_rejected():
    source = """\
def f(x: int) -> dict:
    return x
"""

    assert error_codes(source) == ["PP100"]
    assert "unsupported return annotation" in error_messages(source)[0]


def test_none_parameter_annotation_is_rejected():
    source = """\
def f(x: None) -> int:
    return 1
"""

    assert error_codes(source) == ["PP100"]
    assert "`None` is not a valid parameter annotation" in error_messages(source)[0]


def test_array_with_unknown_dtype_annotation_is_rejected():
    source = """\
from postyp import Array

def f(x: Array[dict]) -> None:
    pass
"""

    assert error_codes(source) == ["PP100"]
    assert "unsupported annotation for parameter `x`" in error_messages(source)[0]


def test_none_return_annotation_is_valid_void_return():
    source = """\
def f(x: int) -> None:
    pass
"""

    assert error_codes(source) == []
