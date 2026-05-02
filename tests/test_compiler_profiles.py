"""Compiler profile diagnostics for valid but unsupported POST features."""

from postpython.compiler.frontend import compile_source


def compile_errors(source: str) -> list:
    _, errors = compile_source(source)
    return errors


def test_dataframe_annotation_reports_unsupported_profile():
    source = """\
from postyp import DataFrame

def f(df: DataFrame) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "POST DataFrame profile" in errors[0].message


def test_series_annotation_reports_unsupported_profile():
    source = """\
from postyp import Series, Float64

def f(s: Series[Float64]) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "POST DataFrame profile" in errors[0].message


def test_optional_annotation_reports_unsupported_valid_annotation():
    source = """\
from typing import Optional

def f(x: Optional[int]) -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "`Optional` annotations" in errors[0].message


def test_top_level_expression_reports_unsupported_instead_of_being_ignored():
    source = """\
def f() -> int:
    return 1

f()
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "top-level executable expressions" in errors[0].message


def test_module_docstring_is_allowed_at_top_level():
    source = '''\
"""module docs"""

def f() -> int:
    return 1
'''

    assert compile_errors(source) == []


def test_class_definition_reports_unsupported_instead_of_being_ignored():
    source = """\
class Point:
    x: int

def f() -> int:
    return 1
"""

    errors = compile_errors(source)

    assert [error.code for error in errors] == ["PP900"]
    assert "class/dataclass definitions" in errors[0].message
