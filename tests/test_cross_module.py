"""Cross-module compilation and linking (spec §9.1).

POST module imports resolve to source files, dependencies compile as their
own translation units, and calls link against the compiled POST functions
— never silently against same-named libm symbols.
"""

import ctypes
import math
import shutil

import pytest

from post_py.build import build_file, BuildError
from post_py.compiler.backend.c_backend import emit_module
from post_py.compiler.frontend import compile_program, compile_source

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


HELPER = """\
from postyp import Float64

def _hidden(x: Float64) -> Float64:
    return x * 10.0

def double_it(x: Float64) -> Float64:
    return x * 2.0
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# compile_program: resolution, ordering, classification
# ---------------------------------------------------------------------------

def test_program_compiles_dependencies_first(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import double_it\n"
        "def quad(x: Float64) -> Float64:\n"
        "    return double_it(double_it(x))\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    assert [m.name for m in modules] == ["helper", "main"]
    assert modules[1].dependencies == ["helper"]
    assert modules[1].dep_modules[0] is modules[0]


def test_diamond_dependency_compiles_each_unit_once(tmp_path):
    _write(tmp_path, "base.py", (
        "from postyp import Float64\n"
        "def inc(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
    ))
    _write(tmp_path, "left.py", (
        "from postyp import Float64\n"
        "from base import inc\n"
        "def linc(x: Float64) -> Float64:\n"
        "    return inc(x)\n"
    ))
    _write(tmp_path, "right.py", (
        "from postyp import Float64\n"
        "from base import inc\n"
        "def rinc(x: Float64) -> Float64:\n"
        "    return inc(x) + 1.0\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from left import linc\n"
        "from right import rinc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return linc(x) + rinc(x)\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    names = [m.name for m in modules]
    assert names.count("base") == 1
    assert names[-1] == "main"


def test_import_cycle_reports_pp500(tmp_path):
    _write(tmp_path, "a.py", (
        "from postyp import Float64\n"
        "from b import g\n"
        "def f(x: Float64) -> Float64:\n"
        "    return g(x)\n"
    ))
    _write(tmp_path, "b.py", (
        "from postyp import Float64\n"
        "from a import f\n"
        "def g(x: Float64) -> Float64:\n"
        "    return f(x)\n"
    ))
    _, errors = compile_program(tmp_path / "a.py")
    assert any(e.code == "PP500" for e in errors), errors


def test_unresolved_import_is_boundary_and_calls_are_diagnosed(tmp_path):
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from no_such_module import mystery\n"
        "def f(x: Float64) -> Float64:\n"
        "    return mystery(x)\n"
    ))
    modules, errors = compile_program(main)
    assert "mystery" in modules[-1].boundary_imports
    assert any(e.code == "PP900" and "mystery" in e.message for e in errors)


def test_unknown_function_call_reports_pp502():
    _, errors = compile_source(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return no_such_fn(x)\n"
    )
    assert any(e.code == "PP502" for e in errors), errors


def test_private_cross_module_call_reports_pp503(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import _hidden\n"
        "def f(x: Float64) -> Float64:\n"
        "    return _hidden(x)\n"
    ))
    _, errors = compile_program(main)
    assert any(e.code == "PP503" for e in errors), errors


def test_duplicate_public_function_across_modules_reports_pp501(tmp_path):
    _write(tmp_path, "a.py", (
        "from postyp import Float64\n"
        "def transform(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
    ))
    _write(tmp_path, "b.py", (
        "from postyp import Float64\n"
        "def transform(x: Float64) -> Float64:\n"
        "    return x * 2.0\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from a import transform as ta\n"
        "from b import transform as tb\n"
        "def f(x: Float64) -> Float64:\n"
        "    return ta(x) + tb(x)\n"
    ))
    _, errors = compile_program(main)
    pp501 = [e for e in errors if e.code == "PP501"]
    assert pp501, errors
    assert "transform" in pp501[0].message
    assert "`a`" in pp501[0].message and "`b`" in pp501[0].message


def test_duplicate_private_helpers_are_fine(tmp_path):
    # Privates get internal linkage, so same-named helpers must not
    # trigger PP501.
    _write(tmp_path, "a.py", (
        "from postyp import Float64\n"
        "def _poly(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
        "def fa(x: Float64) -> Float64:\n"
        "    return _poly(x)\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from a import fa\n"
        "def _poly(x: Float64) -> Float64:\n"
        "    return x * 2.0\n"
        "def f(x: Float64) -> Float64:\n"
        "    return fa(x) + _poly(x)\n"
    ))
    _, errors = compile_program(main)
    assert errors == [], errors


def test_unused_stdlib_import_is_boundary_not_compiled(tmp_path):
    # `from fractions import Fraction` must classify as a CPython-boundary
    # import — never pull stdlib source in as a POST translation unit.
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from fractions import Fraction\n"
        "def g(x: Float64) -> Float64:\n"
        "    return x\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    assert [m.name for m in modules] == ["main"]
    assert "Fraction" in modules[0].boundary_imports


def test_called_stdlib_import_is_diagnosed_not_compiled(tmp_path):
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from fractions import Fraction\n"
        "def g(x: Float64) -> Float64:\n"
        "    return Fraction(x)\n"
    ))
    modules, errors = compile_program(main)
    assert [m.name for m in modules] == ["main"]
    assert any(e.code == "PP900" and "Fraction" in e.message for e in errors)


def test_search_paths_opt_in_resolves_external_module(tmp_path):
    external = tmp_path / "external-root"
    external.mkdir()
    (external / "extmod.py").write_text(
        "from postyp import Float64\n"
        "def scale(x: Float64) -> Float64:\n"
        "    return x * 5.0\n"
    )
    project = tmp_path / "project"
    project.mkdir()
    main = project / "main.py"
    main.write_text(
        "from postyp import Float64\n"
        "from extmod import scale\n"
        "def f(x: Float64) -> Float64:\n"
        "    return scale(x)\n"
    )

    # Without the search path: boundary import, call diagnosed.
    _, errors = compile_program(main)
    assert any(e.code == "PP900" and "scale" in e.message for e in errors)

    # With the opt-in search path: resolved, compiled, linked.
    modules, errors = compile_program(main, search_paths=[external])
    assert errors == [], errors
    assert [m.name for m in modules] == ["extmod", "main"]


def test_forward_reference_uses_callee_return_dtype():
    module, errors = compile_source(
        "from postyp import Float64, Bool\n"
        "def caller(x: Float64) -> Bool:\n"
        "    return check(x)\n"
        "def check(x: Float64) -> Bool:\n"
        "    return x > 0.0\n"
    )
    assert errors == [], errors
    c = emit_module(module)
    # The call result carries the callee's Bool return type, not a
    # promoted-from-arguments guess.
    assert "bool _ret" in c


# ---------------------------------------------------------------------------
# C emission: externs, static privates, symbol mangling
# ---------------------------------------------------------------------------

def test_emission_declares_externs_and_hides_privates(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import double_it\n"
        "def quad(x: Float64) -> Float64:\n"
        "    return double_it(double_it(x))\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    helper_c = emit_module(modules[0], dep_modules=modules[0].dep_modules)
    main_c = emit_module(modules[1], dep_modules=modules[1].dep_modules)

    # Private helpers get internal linkage in their own unit.
    assert "static double _hidden(double _x)" in helper_c
    # The importer declares the dependency's public function extern...
    assert "double double_it(double _x);" in main_c
    # ...but not the private one.
    assert "_hidden" not in main_c


def test_reserved_name_import_uses_mangled_symbol(tmp_path):
    _write(tmp_path, "helper.py", (
        "from postyp import Float64\n"
        "def erfc(x: Float64) -> Float64:\n"
        "    return x + 41.5\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import erfc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return erfc(x)\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    main_c = emit_module(modules[1], dep_modules=modules[1].dep_modules)
    # The call must target the POST symbol, not libm's erfc.
    assert "__pp_erfc(_x)" in main_c


def test_gufunc_call_across_modules_passes_core_dims(tmp_path):
    _write(tmp_path, "vec.py", (
        "from postyp import Array, Float64\n"
        "from post_py import guvectorize\n"
        "@guvectorize([], \"(n),(n)->()\")\n"
        "def vdot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:\n"
        "    acc: Float64 = 0.0\n"
        "    for i in range(len(a)):\n"
        "        acc += a[i] * b[i]\n"
        "    out[0] = acc\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Array, Float64\n"
        "from post_py import guvectorize\n"
        "from vec import vdot\n"
        "@guvectorize([], \"(n),(n)->()\")\n"
        "def scaled_dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:\n"
        "    vdot(a, b, out)\n"
        "    out[0] = out[0] * 2.0\n"
    ))
    modules, errors = compile_program(main)
    assert errors == [], errors
    main_c = emit_module(modules[1], dep_modules=modules[1].dep_modules)
    # The cross-module gufunc call receives the caller's core-dim value.
    assert "vdot(_a, _b, _out, _pp_dim_n)" in main_c


# ---------------------------------------------------------------------------
# Runtime: build, link, and call across translation units
# ---------------------------------------------------------------------------

@needs_cc
def test_two_module_program_builds_and_runs(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import double_it\n"
        "def quad(x: Float64) -> Float64:\n"
        "    return double_it(double_it(x))\n"
    ))
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    quad = lib.quad
    quad.argtypes = [ctypes.c_double]
    quad.restype = ctypes.c_double
    assert quad(2.5) == 10.0


@needs_cc
def test_import_alias_resolves_to_source_function(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import double_it as twice\n"
        "def f(x: Float64) -> Float64:\n"
        "    return twice(x)\n"
    ))
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(3.0) == 6.0


@needs_cc
def test_imported_reserved_name_links_to_post_function_not_libm(tmp_path):
    # A POST function named after a libm symbol must link to the POST
    # implementation. libm's erfc(0.5) ≈ 0.4795; POST's returns 42.0.
    _write(tmp_path, "helper.py", (
        "from postyp import Float64\n"
        "def erfc(x: Float64) -> Float64:\n"
        "    return x + 41.5\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from helper import erfc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return erfc(x)\n"
    ))
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(0.5) == 42.0
    assert abs(f(0.5) - math.erfc(0.5)) > 40.0  # decidedly not libm


@needs_cc
def test_vectorize_kernel_called_across_modules(tmp_path):
    _write(tmp_path, "kernels.py", (
        "from postyp import Float64\n"
        "from post_py import vectorize\n"
        "from post_py.math import exp\n"
        "@vectorize\n"
        "def sigmoid(x: Float64) -> Float64:\n"
        "    if x >= 0.0:\n"
        "        return 1.0 / (1.0 + exp(-x))\n"
        "    z: Float64 = exp(x)\n"
        "    return z / (1.0 + z)\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from kernels import sigmoid\n"
        "def centered(x: Float64) -> Float64:\n"
        "    return sigmoid(x) - 0.5\n"
    ))
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.centered
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(0.0) == 0.0
    assert abs(f(2.0) - (1.0 / (1.0 + math.exp(-2.0)) - 0.5)) < 1e-15


@needs_cc
def test_diamond_program_links_single_definition(tmp_path):
    _write(tmp_path, "base.py", (
        "from postyp import Float64\n"
        "def inc(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
    ))
    _write(tmp_path, "left.py", (
        "from postyp import Float64\n"
        "from base import inc\n"
        "def linc(x: Float64) -> Float64:\n"
        "    return inc(x)\n"
    ))
    _write(tmp_path, "right.py", (
        "from postyp import Float64\n"
        "from base import inc\n"
        "def rinc(x: Float64) -> Float64:\n"
        "    return inc(inc(x))\n"
    ))
    main = _write(tmp_path, "main.py", (
        "from postyp import Float64\n"
        "from left import linc\n"
        "from right import rinc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return linc(x) + rinc(x)\n"
    ))
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(1.0) == (1.0 + 1.0) + (1.0 + 2.0)


@needs_cc
def test_package_style_absolute_imports(tmp_path):
    # mypkg/_a.py and mypkg/_b.py with absolute `from mypkg._a import ...`,
    # mirroring the ppspecial layout.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_a.py").write_text(
        "from postyp import Float64\n"
        "def base(x: Float64) -> Float64:\n"
        "    return x * 3.0\n"
    )
    main = pkg / "_b.py"
    main.write_text(
        "from postyp import Float64\n"
        "from mypkg._a import base\n"
        "def f(x: Float64) -> Float64:\n"
        "    return base(x) + 1.0\n"
    )
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(2.0) == 7.0
