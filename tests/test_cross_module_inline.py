"""Cross-module inlining via static-inline replicas (issue #13, spec §9.1).

With ``cross_module_inline=True`` each translation unit receives
``static inline`` copies of the POST functions it imports (transitively),
so the C compiler can inline small kernels across module boundaries.
Observable behavior — results, public symbols, the pp_* ABI, ufunc
wrappers — must be identical to the default per-TU build.
"""

import ctypes
import math
import shutil

import pytest

from postpyc.build import build_file
from postpyc.cli import main
from postpyc.compiler.backend.c_backend import emit_module, transitive_dep_modules
from postpyc.compiler.frontend import compile_program

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


HELPER = """\
from postyp import Float64

def _hidden(x: Float64) -> Float64:
    return x * 10.0

def double_it(x: Float64) -> Float64:
    return _hidden(x) / 5.0
"""

MAIN = """\
from postyp import Float64
from helper import double_it

def _hidden(x: Float64) -> Float64:
    return x + 100.0

def quad(x: Float64) -> Float64:
    return _hidden(double_it(double_it(x))) - 100.0
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def _f64(lib, name, nargs=1):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_double] * nargs
    fn.restype = ctypes.c_double
    return fn


def _program(tmp_path):
    _write(tmp_path, "helper.py", HELPER)
    main_path = _write(tmp_path, "main.py", MAIN)
    modules, errors = compile_program(main_path)
    assert errors == [], errors
    return main_path, modules


# ---------------------------------------------------------------------------
# C emission
# ---------------------------------------------------------------------------

def test_importing_unit_gets_static_inline_replicas(tmp_path):
    _, modules = _program(tmp_path)
    main_c = emit_module(modules[1], dep_modules=modules[1].dep_modules,
                         inline_deps=True)
    # The imported public function is replicated with internal linkage...
    assert "static inline double double_it(double _x)" in main_c
    # ...and no extern declaration remains for it.
    assert "double double_it(double _x);" not in main_c
    # The replica of the dependency's private helper is renamed per unit,
    # so it cannot collide with the importer's own `_hidden`.
    assert "static inline double __ppi_helper_hidden(double _x)" in main_c
    assert "__ppi_helper_hidden(_x)" in main_c
    assert "static double _hidden(double _x)" in main_c  # importer's own


def test_dependency_unit_itself_is_unchanged(tmp_path):
    _, modules = _program(tmp_path)
    helper_default = emit_module(modules[0], dep_modules=modules[0].dep_modules)
    helper_inline = emit_module(modules[0], dep_modules=modules[0].dep_modules,
                                inline_deps=True)
    # No dependencies → nothing to replicate; external definitions stay.
    assert helper_inline == helper_default
    assert "double double_it(double _x)" in helper_default


def test_replicas_cover_transitive_dependencies(tmp_path):
    _write(tmp_path, "base.py", (
        "from postyp import Float64\n"
        "def inc(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
    ))
    _write(tmp_path, "mid.py", (
        "from postyp import Float64\n"
        "from base import inc\n"
        "def twice_inc(x: Float64) -> Float64:\n"
        "    return inc(inc(x))\n"
    ))
    entry = _write(tmp_path, "entry.py", (
        "from postyp import Float64\n"
        "from mid import twice_inc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return twice_inc(x) * 2.0\n"
    ))
    modules, errors = compile_program(entry)
    assert errors == [], errors
    order = [m.name for m in transitive_dep_modules(modules[-1].dep_modules)]
    assert order == ["base", "mid"]  # dependencies first
    entry_c = emit_module(modules[-1], dep_modules=modules[-1].dep_modules,
                          inline_deps=True)
    # mid's replica calls base's `inc`, so base must be replicated too.
    assert "static inline double inc(double _x)" in entry_c
    assert "static inline double twice_inc(double _x)" in entry_c


def test_reserved_name_replica_keeps_mangled_symbol(tmp_path):
    _write(tmp_path, "kern.py", (
        "from postyp import Float64\n"
        "def erfc(x: Float64) -> Float64:\n"
        "    return x + 41.5\n"
    ))
    entry = _write(tmp_path, "entry.py", (
        "from postyp import Float64\n"
        "from kern import erfc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return erfc(x)\n"
    ))
    modules, errors = compile_program(entry)
    assert errors == [], errors
    entry_c = emit_module(modules[1], dep_modules=modules[1].dep_modules,
                          inline_deps=True)
    # The replica shadows the mangled POST symbol, never libm's erfc.
    assert "static inline double __pp_erfc(double _x)" in entry_c
    assert "__pp_erfc(_x)" in entry_c


def test_package_style_dependency_replicates(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "_a.py").write_text(
        "from postyp import Float64\n"
        "def _shift(x: Float64) -> Float64:\n"
        "    return x + 0.5\n"
        "def base(x: Float64) -> Float64:\n"
        "    return _shift(x) * 3.0\n"
    )
    entry = pkg / "_b.py"
    entry.write_text(
        "from postyp import Float64\n"
        "from mypkg._a import base\n"
        "def f(x: Float64) -> Float64:\n"
        "    return base(x) + 1.0\n"
    )
    modules, errors = compile_program(entry)
    assert errors == [], errors
    entry_c = emit_module(modules[-1], dep_modules=modules[-1].dep_modules,
                          inline_deps=True)
    # Module names are file stems, so the unit tag is `_a`.
    assert "static inline double __ppi__a_shift(double _x)" in entry_c
    assert "static inline double base(double _x)" in entry_c
    assert "__ppi__a_shift(_x)" in entry_c


# ---------------------------------------------------------------------------
# Runtime equivalence
# ---------------------------------------------------------------------------

@needs_cc
def test_results_match_default_build(tmp_path):
    main_path, _ = _program(tmp_path)
    plain = ctypes.CDLL(str(build_file(main_path, output=tmp_path / "plain.so")))
    inlined = ctypes.CDLL(str(build_file(
        main_path, output=tmp_path / "inlined.so", cross_module_inline=True,
    )))
    for x in (0.0, -3.5, 2.5, 1e300):
        assert _f64(plain, "quad")(x) == _f64(inlined, "quad")(x)
    assert _f64(inlined, "quad")(2.5) == 10.0


@needs_cc
def test_public_symbols_and_abi_survive(tmp_path):
    main_path, _ = _program(tmp_path)
    lib = ctypes.CDLL(str(build_file(
        main_path, output=tmp_path / "out.so", cross_module_inline=True,
    )))
    # The dependency's public function keeps its external definition...
    assert _f64(lib, "double_it")(3.0) == 6.0
    # ...and the stable pp_* export ABI is intact.
    assert _f64(lib, "pp_quad")(2.5) == 10.0
    assert _f64(lib, "pp_double_it")(3.0) == 6.0


@needs_cc
def test_reserved_name_links_to_post_function_not_libm(tmp_path):
    _write(tmp_path, "kern.py", (
        "from postyp import Float64\n"
        "def erfc(x: Float64) -> Float64:\n"
        "    return x + 41.5\n"
    ))
    entry = _write(tmp_path, "entry.py", (
        "from postyp import Float64\n"
        "from kern import erfc\n"
        "def f(x: Float64) -> Float64:\n"
        "    return erfc(x)\n"
    ))
    lib = ctypes.CDLL(str(build_file(
        entry, output=tmp_path / "out.so", cross_module_inline=True,
    )))
    assert _f64(lib, "f")(0.5) == 42.0
    assert abs(_f64(lib, "f")(0.5) - math.erfc(0.5)) > 40.0


@needs_cc
def test_gufunc_called_across_modules_inlines_and_runs(tmp_path):
    _write(tmp_path, "vec.py", (
        "from postyp import Array, Float64\n"
        "from postpyc import guvectorize\n"
        "@guvectorize([], \"(n),(n)->()\")\n"
        "def vdot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:\n"
        "    acc: Float64 = 0.0\n"
        "    for i in range(len(a)):\n"
        "        acc += a[i] * b[i]\n"
        "    out[0] = acc\n"
    ))
    entry = _write(tmp_path, "entry.py", (
        "from postyp import Array, Float64\n"
        "from postpyc import guvectorize\n"
        "from vec import vdot\n"
        "@guvectorize([], \"(n),(n)->()\")\n"
        "def scaled_dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:\n"
        "    vdot(a, b, out)\n"
        "    out[0] = out[0] * 2.0\n"
    ))
    modules, errors = compile_program(entry)
    assert errors == [], errors
    entry_c = emit_module(modules[1], dep_modules=modules[1].dep_modules,
                          inline_deps=True)
    # The scalar/core kernel is replicated; the NumPy loop wrapper is not.
    assert "static inline void vdot(" in entry_c
    assert "vdot_ufunc_loop" not in entry_c

    lib = ctypes.CDLL(str(build_file(
        entry, output=tmp_path / "out.so", cross_module_inline=True,
    )))
    # Both ufunc loop wrappers keep their external symbols.
    assert lib.scaled_dot_ufunc_loop is not None
    assert lib.vdot_ufunc_loop is not None


@needs_cc
def test_duplicate_private_helpers_do_not_collide(tmp_path):
    # Two dependencies with same-named private helpers, both replicated
    # into the entry unit: the per-unit renaming must keep them apart.
    _write(tmp_path, "a.py", (
        "from postyp import Float64\n"
        "def _poly(x: Float64) -> Float64:\n"
        "    return x + 1.0\n"
        "def fa(x: Float64) -> Float64:\n"
        "    return _poly(x)\n"
    ))
    _write(tmp_path, "b.py", (
        "from postyp import Float64\n"
        "def _poly(x: Float64) -> Float64:\n"
        "    return x * 2.0\n"
        "def fb(x: Float64) -> Float64:\n"
        "    return _poly(x)\n"
    ))
    entry = _write(tmp_path, "entry.py", (
        "from postyp import Float64\n"
        "from a import fa\n"
        "from b import fb\n"
        "def f(x: Float64) -> Float64:\n"
        "    return fa(x) + fb(x)\n"
    ))
    lib = ctypes.CDLL(str(build_file(
        entry, output=tmp_path / "out.so", cross_module_inline=True,
    )))
    assert _f64(lib, "f")(3.0) == (3.0 + 1.0) + (3.0 * 2.0)


@needs_cc
def test_cli_flag_builds_working_artifact(tmp_path, capsys):
    main_path, _ = _program(tmp_path)
    out = tmp_path / "cli.so"
    code = main([
        "build", str(main_path), "--output", str(out), "--cross-module-inline",
    ])
    assert code == 0
    lib = ctypes.CDLL(str(out))
    assert _f64(lib, "quad")(2.5) == 10.0
