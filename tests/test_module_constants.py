"""Module-level constants (spec §3, §9.1; postpython#11).

Top-level typed constants fold at compile time: literals, constant
expressions, references to prior constants, compile-time imports from
postpython.math, and constants imported from other POST modules.
"""

import ctypes
import math
import shutil

import pytest

from postpython.build import build_file, build_source
from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_program, compile_source
from postyp import Bool, Complex128, Float64, Int64

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


def _module_ok(source: str):
    module, errors = compile_source(source)
    assert errors == [], errors
    return module


# ---------------------------------------------------------------------------
# Collection and folding
# ---------------------------------------------------------------------------

def test_annotated_literal_constant():
    module = _module_ok(
        "from postyp import Float64\n"
        "HALF: Float64 = 0.5\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * HALF\n"
    )
    assert module.constants["HALF"] == (Float64, 0.5)


def test_unannotated_constants_infer_dtype():
    module = _module_ok(
        "from postyp import Float64\n"
        "N = 4\n"
        "FLAG = True\n"
        "Z = 1.0 + 2.0j\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x\n"
    )
    assert module.constants["N"] == (Int64, 4)
    assert module.constants["FLAG"] == (Bool, True)
    assert module.constants["Z"] == (Complex128, 1.0 + 2.0j)


def test_constant_expressions_fold_exactly():
    module = _module_ok(
        "from postyp import Float64\n"
        "P_HIGH: Float64 = 1.0 - 0.02425\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * P_HIGH\n"
    )
    assert module.constants["P_HIGH"] == (Float64, 1.0 - 0.02425)


def test_constants_can_reference_prior_constants():
    module = _module_ok(
        "from postyp import Float64\n"
        "BASE: Float64 = 2.0\n"
        "SCALED: Float64 = BASE * 3.0 + 1.0\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * SCALED\n"
    )
    assert module.constants["SCALED"] == (Float64, 7.0)


def test_postpython_math_constants_fold_in():
    module = _module_ok(
        "from postyp import Float64\n"
        "from postpython.math import PI, E\n"
        "TWO_PI: Float64 = 2.0 * PI\n"
        "def circ(r: Float64) -> Float64:\n"
        "    return TWO_PI * r\n"
    )
    assert module.constants["PI"] == (Float64, math.pi)
    assert module.constants["E"] == (Float64, math.e)
    assert module.constants["TWO_PI"] == (Float64, 2.0 * math.pi)
    # exp stays a callable intrinsic, not a constant.
    assert "PI" not in module.intrinsic_imports


def test_function_alias_lines_are_skipped_not_fatal():
    module = _module_ok(
        "from postyp import Float64\n"
        "def lgamma_impl(x: Float64) -> Float64:\n"
        "    return x\n"
        "gammaln = lgamma_impl\n"
        "AFTER: Float64 = 1.5\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * AFTER\n"
    )
    assert "gammaln" not in module.constants
    assert module.constants["AFTER"] == (Float64, 1.5)


# ---------------------------------------------------------------------------
# Scoping semantics
# ---------------------------------------------------------------------------

def test_local_assignment_shadows_module_constant():
    c = emit_module(_module_ok(
        "from postyp import Float64\n"
        "K: Float64 = 100.0\n"
        "def f(x: Float64) -> Float64:\n"
        "    K: Float64 = 2.0\n"
        "    return x * K\n"
    ))
    # The local declaration is used; the module value 100.0 never appears.
    assert "100.0" not in c


def test_read_before_local_assignment_is_an_error():
    _, errors = compile_source(
        "from postyp import Float64\n"
        "K: Float64 = 100.0\n"
        "def f(x: Float64) -> Float64:\n"
        "    y: Float64 = K + x\n"   # K is assigned below → local everywhere
        "    K = 2.0\n"
        "    return y * K\n"
    )
    assert any(
        e.code == "PP100" and "before assignment" in e.message for e in errors
    ), errors


def test_unknown_name_still_diagnosed():
    _, errors = compile_source(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * MISSING\n"
    )
    assert any(e.code == "PP900" and "MISSING" in e.message for e in errors)


# ---------------------------------------------------------------------------
# Cross-module constants
# ---------------------------------------------------------------------------

def test_cross_module_constant_import(tmp_path):
    (tmp_path / "consts.py").write_text(
        "from postyp import Float64\n"
        "SQRT2: Float64 = 1.4142135623730951\n"
    )
    main = tmp_path / "main.py"
    main.write_text(
        "from postyp import Float64\n"
        "from consts import SQRT2\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * SQRT2\n"
    )
    modules, errors = compile_program(main)
    assert errors == [], errors
    assert modules[-1].constants["SQRT2"] == (Float64, 1.4142135623730951)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@needs_cc
def test_pi_circumference_runtime():
    lib = ctypes.CDLL(str(build_source(
        "from postyp import Float64\n"
        "from postpython.math import PI\n"
        "def circ(r: Float64) -> Float64:\n"
        "    return 2.0 * PI * r\n",
        filename="circ.py",
    )))
    circ = lib.circ
    circ.argtypes = [ctypes.c_double]
    circ.restype = ctypes.c_double
    assert circ(1.0) == 2.0 * math.pi
    assert circ(2.5) == 2.0 * math.pi * 2.5


@needs_cc
def test_module_scope_coefficient_table_pattern():
    # The ppspecial motif: polynomial coefficients at module scope,
    # Horner evaluation in a kernel.
    lib = ctypes.CDLL(str(build_source(
        "from postyp import Float64\n"
        "C0: Float64 = 2.0\n"
        "C1: Float64 = -3.0\n"
        "C2: Float64 = 0.5\n"
        "def poly(x: Float64) -> Float64:\n"
        "    p: Float64 = C2\n"
        "    p = p * x + C1\n"
        "    p = p * x + C0\n"
        "    return p\n",
        filename="poly.py",
    )))
    poly = lib.poly
    poly.argtypes = [ctypes.c_double]
    poly.restype = ctypes.c_double

    def reference(x):
        return (0.5 * x - 3.0) * x + 2.0

    for x in (-2.0, 0.0, 1.0, 3.5):
        assert poly(x) == reference(x)


@needs_cc
def test_cross_module_constant_runtime(tmp_path):
    (tmp_path / "consts.py").write_text(
        "from postyp import Float64\n"
        "FACTOR: Float64 = 2.5\n"
        "OFFSET: Float64 = FACTOR * 2.0\n"
    )
    main = tmp_path / "main.py"
    main.write_text(
        "from postyp import Float64\n"
        "from consts import FACTOR, OFFSET\n"
        "def f(x: Float64) -> Float64:\n"
        "    return x * FACTOR + OFFSET\n"
    )
    lib = ctypes.CDLL(str(build_file(main, output=tmp_path / "out.so")))
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(2.0) == 2.0 * 2.5 + 5.0
