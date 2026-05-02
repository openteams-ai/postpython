"""Tests for the POST Python subset checker."""

import pytest
from postpython.checker import check_source, is_valid


def violations_for(source: str) -> set[str]:
    return {v.code for v in check_source(source)}


def only_violation(source: str) -> str:
    vs = check_source(source)
    assert len(vs) == 1, f"expected 1 violation, got {[v.code for v in vs]}"
    return vs[0].code


# ---------------------------------------------------------------------------
# Valid POST Python — should produce zero violations
# ---------------------------------------------------------------------------

VALID_TYPED = """\
def add(x: int, y: int) -> int:
    return x + y

class Point:
    x: int
    y: int
    def distance(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""

def test_valid_typed_source_has_no_violations():
    assert is_valid(VALID_TYPED)


# ---------------------------------------------------------------------------
# PP001-PP005  dynamic execution
# ---------------------------------------------------------------------------

def test_pp002_eval():
    assert "PP002" in violations_for("eval('1+1')")

def test_pp002_exec():
    assert "PP002" in violations_for("exec('x=1')")

def test_pp002_compile():
    assert "PP002" in violations_for("compile('x', '<str>', 'exec')")

def test_pp002_globals():
    assert "PP002" in violations_for("globals()")

def test_pp002_locals():
    assert "PP002" in violations_for("locals()")

def test_pp003_getattr():
    assert "PP003" in violations_for("getattr(obj, 'x')")

def test_pp003_setattr():
    assert "PP003" in violations_for("setattr(obj, 'x', 1)")

def test_method_names_matching_banned_builtins_are_allowed():
    src = """\
class Tool:
    def dir(self) -> int:
        return 1
    def compile(self) -> int:
        return 2
    def getattr(self) -> int:
        return 3

def f(tool: Tool) -> int:
    return tool.dir() + tool.compile() + tool.getattr()
"""
    codes = violations_for(src)
    assert "PP002" not in codes
    assert "PP003" not in codes

def test_pp004_dunder_import():
    assert "PP004" in violations_for("__import__('os')")

def test_pp005_dynamic_type():
    assert "PP005" in violations_for("type('Foo', (object,), {})")

def test_pp005_type_one_arg_ok():
    # type(x) for isinstance-style use is fine
    assert "PP005" not in violations_for("type(x)")

def test_method_named_type_with_three_args_is_allowed():
    src = """\
class Factory:
    def type(self, name: str, bases: int, attrs: int) -> int:
        return 1

def f(factory: Factory) -> int:
    return factory.type("Foo", 1, 2)
"""
    assert "PP005" not in violations_for(src)


# ---------------------------------------------------------------------------
# PP006-PP007  namespace statements
# ---------------------------------------------------------------------------

def test_pp006_global():
    src = "def f() -> None:\n    global x"
    assert "PP006" in violations_for(src)

def test_pp007_nonlocal():
    src = "def outer() -> None:\n    x: int = 1\n    def inner() -> None:\n        nonlocal x"
    assert "PP007" in violations_for(src)


# ---------------------------------------------------------------------------
# PP008  relative imports
# ---------------------------------------------------------------------------

def test_pp008_relative_import():
    assert "PP008" in violations_for("from . import foo")

def test_absolute_import_ok():
    assert "PP008" not in violations_for("from os import path")


# ---------------------------------------------------------------------------
# PP009-PP010  class features
# ---------------------------------------------------------------------------

def test_pp009_metaclass():
    src = "class Foo(metaclass=Meta):\n    pass"
    assert "PP009" in violations_for(src)

def test_pp010_multiple_inheritance():
    src = "class Foo(Bar, Baz):\n    pass"
    assert "PP010" in violations_for(src)

def test_single_base_ok():
    src = "class Foo(Bar):\n    pass"
    assert "PP010" not in violations_for(src)


# ---------------------------------------------------------------------------
# PP011-PP014  async
# ---------------------------------------------------------------------------

def test_pp011_async_def():
    src = "async def f() -> None:\n    pass"
    assert "PP011" in violations_for(src)

def test_pp012_await():
    # async def already triggers PP011; also check await independently
    src = "async def f() -> None:\n    await something()"
    codes = violations_for(src)
    assert "PP012" in codes

def test_pp013_async_for():
    src = "async def f() -> None:\n    async for x in it:\n        pass"
    assert "PP013" in violations_for(src)

def test_pp014_async_with():
    src = "async def f() -> None:\n    async with ctx() as c:\n        pass"
    assert "PP014" in violations_for(src)


# ---------------------------------------------------------------------------
# PP020-PP021  missing annotations
# ---------------------------------------------------------------------------

def test_pp020_unannotated_param():
    src = "def f(x) -> int:\n    return x"
    assert "PP020" in violations_for(src)

def test_pp021_missing_return_annotation():
    src = "def f(x: int):\n    return x"
    assert "PP021" in violations_for(src)

def test_fully_annotated_ok():
    src = "def f(x: int) -> int:\n    return x"
    assert not violations_for(src)


# ---------------------------------------------------------------------------
# PP022-PP024  variadic / starred
# ---------------------------------------------------------------------------

def test_pp022_varargs():
    src = "def f(*args: int) -> None:\n    pass"
    assert "PP022" in violations_for(src)

def test_pp023_kwargs():
    src = "def f(**kwargs: int) -> None:\n    pass"
    assert "PP023" in violations_for(src)

def test_pp024_splat_in_call():
    src = "def f() -> None:\n    g(*lst)"
    assert "PP024" in violations_for(src)

def test_starred_destructuring_assignment_is_not_pp024():
    src = """\
def f(xs: list[int]) -> int:
    first, *rest = xs
    return first
"""
    assert "PP024" not in violations_for(src)

def test_starred_list_display_is_not_pp024():
    src = """\
def f(xs: list[int]) -> int:
    ys = [0, *xs]
    return ys[0]
"""
    assert "PP024" not in violations_for(src)


# ---------------------------------------------------------------------------
# PP025  del
# ---------------------------------------------------------------------------

def test_pp025_del():
    assert "PP025" in violations_for("del x")


# ---------------------------------------------------------------------------
# PP030-PP032  generators / lambda
# ---------------------------------------------------------------------------

def test_pp030_yield():
    src = "def f() -> None:\n    yield 1"
    assert "PP030" in violations_for(src)

def test_pp031_yield_from():
    src = "def f() -> None:\n    yield from other()"
    assert "PP031" in violations_for(src)

def test_pp032_lambda():
    assert "PP032" in violations_for("f = lambda x: x + 1")


# ---------------------------------------------------------------------------
# PP000  syntax errors
# ---------------------------------------------------------------------------

def test_pp000_syntax_error():
    vs = check_source("def (:")
    assert len(vs) == 1
    assert vs[0].code == "PP000"


# ---------------------------------------------------------------------------
# Violation metadata
# ---------------------------------------------------------------------------

def test_violation_has_line_number():
    vs = check_source("eval('x')", filename="test.py")
    assert vs[0].lineno == 1
    assert vs[0].filename == "test.py"

def test_violation_str_format():
    vs = check_source("eval('x')", filename="test.py")
    s = str(vs[0])
    assert s.startswith("test.py:1:")
    assert "PP002" in s
