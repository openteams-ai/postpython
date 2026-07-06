"""Floating-point semantics (spec §4.1.2).

Compiled code defaults to strict IEEE 754 behavior: NaN propagation,
infinities, signed zero, no FMA contraction. Fast-math is an explicit,
per-build opt-in that never leaks into the link step.
"""

import ctypes
import math
import shutil

import pytest

import postpyc.build as build_mod
from postpyc.build import BuildError, build_file, build_source
from postpyc.cli import main

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")

KERNELS = """\
from postyp import Bool, Float64
from postpyc.math import INF, NAN, isnan

def arith(x: Float64) -> Float64:
    return (x * 2.0 + 1.0) / 3.0

def sub(x: Float64, y: Float64) -> Float64:
    return x - y

def mul(x: Float64, y: Float64) -> Float64:
    return x * y

def invert(x: Float64) -> Float64:
    return 1.0 / x

def plus_zero(x: Float64) -> Float64:
    return x + 0.0

def negate(x: Float64) -> Float64:
    return -x

def sub_self(x: Float64) -> Float64:
    return x - x

def fma_shape(a: Float64, b: Float64, c: Float64) -> Float64:
    return a * b + c

def lt(x: Float64, y: Float64) -> Bool:
    return x < y

def le(x: Float64, y: Float64) -> Bool:
    return x <= y

def eq(x: Float64, y: Float64) -> Bool:
    return x == y

def ne(x: Float64, y: Float64) -> Bool:
    return x != y

def self_ne(x: Float64) -> Bool:
    return x != x

def self_eq(x: Float64) -> Bool:
    return x == x

def libm_isnan(x: Float64) -> Bool:
    return isnan(x)

def inf_const() -> Float64:
    return INF

def neg_inf_const() -> Float64:
    return -INF

def nan_const() -> Float64:
    return NAN
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    out = tmp_path_factory.mktemp("fp") / "kernels.so"
    return ctypes.CDLL(str(build_source(KERNELS, output=out)))


def _fn(lib, name, nargs, restype=ctypes.c_double):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_double] * nargs
    fn.restype = restype
    return fn


# ---------------------------------------------------------------------------
# NaN
# ---------------------------------------------------------------------------

@needs_cc
def test_nan_propagates_through_arithmetic(lib):
    arith = _fn(lib, "arith", 1)
    assert math.isnan(arith(math.nan))
    assert arith(1.0) == 1.0


@needs_cc
def test_nan_comparisons_are_unordered(lib):
    for name in ("lt", "le", "eq"):
        cmp = _fn(lib, name, 2, ctypes.c_bool)
        assert cmp(math.nan, 1.0) is False
        assert cmp(1.0, math.nan) is False
        assert cmp(math.nan, math.nan) is False
    ne = _fn(lib, "ne", 2, ctypes.c_bool)
    assert ne(math.nan, 1.0) is True
    assert ne(math.nan, math.nan) is True


@needs_cc
def test_self_comparison_is_a_nan_test(lib):
    # `x != x` / `x == x` must survive optimization as NaN tests and
    # agree with the libm-lowered isnan.
    self_ne = _fn(lib, "self_ne", 1, ctypes.c_bool)
    self_eq = _fn(lib, "self_eq", 1, ctypes.c_bool)
    libm_isnan = _fn(lib, "libm_isnan", 1, ctypes.c_bool)
    for x in (math.nan, 0.0, -0.0, 1.5, math.inf, -math.inf):
        expected = math.isnan(x)
        assert self_ne(x) is expected
        assert self_eq(x) is (not expected)
        assert libm_isnan(x) is expected


# ---------------------------------------------------------------------------
# Infinities
# ---------------------------------------------------------------------------

@needs_cc
def test_infinity_arithmetic(lib):
    mul = _fn(lib, "mul", 2)
    sub = _fn(lib, "sub", 2)
    assert mul(1e308, 10.0) == math.inf          # overflow rounds to inf
    assert mul(-1e308, 10.0) == -math.inf
    assert math.isnan(sub(math.inf, math.inf))   # inf - inf → NaN
    assert math.isnan(mul(0.0, math.inf))        # 0 * inf → NaN
    assert sub(math.inf, 1e308) == math.inf


@needs_cc
def test_infinity_ordering(lib):
    lt = _fn(lib, "lt", 2, ctypes.c_bool)
    assert lt(-math.inf, -1e308) is True
    assert lt(1e308, math.inf) is True
    assert lt(math.inf, math.inf) is False


@needs_cc
def test_sub_self_is_not_folded_to_zero(lib):
    sub_self = _fn(lib, "sub_self", 1)
    assert math.isnan(sub_self(math.inf))
    assert math.isnan(sub_self(math.nan))
    assert sub_self(1.5) == 0.0


@needs_cc
def test_nonfinite_constants_compile_and_evaluate(lib):
    # INF / NAN constants must emit valid C (INFINITY / NAN, not repr()).
    assert _fn(lib, "inf_const", 0)() == math.inf
    assert _fn(lib, "neg_inf_const", 0)() == -math.inf
    assert math.isnan(_fn(lib, "nan_const", 0)())


# ---------------------------------------------------------------------------
# Signed zero
# ---------------------------------------------------------------------------

@needs_cc
def test_signed_zero_semantics(lib):
    invert = _fn(lib, "invert", 1)
    negate = _fn(lib, "negate", 1)
    plus_zero = _fn(lib, "plus_zero", 1)

    assert invert(0.0) == math.inf
    assert invert(-0.0) == -math.inf             # -0.0 survives the call ABI
    assert invert(negate(0.0)) == -math.inf      # negation produces -0.0

    # `x + 0.0` must not be folded to `x`: IEEE requires -0.0 + 0.0 == +0.0.
    r = plus_zero(-0.0)
    assert r == 0.0 and math.copysign(1.0, r) == 1.0


# ---------------------------------------------------------------------------
# No FMA contraction in strict mode
# ---------------------------------------------------------------------------

@needs_cc
def test_no_fma_contraction_by_default(lib):
    # With a = 1 + 2**-27, a*a rounds to 1 + 2**-26 in binary64, so the
    # strictly evaluated a*a - (1 + 2**-26) is exactly 0. A contracted
    # fma(a, a, -(1 + 2**-26)) would yield the residue 2**-54.
    fma_shape = _fn(lib, "fma_shape", 3)
    a = 1.0 + 2.0 ** -27
    assert fma_shape(a, a, -(1.0 + 2.0 ** -26)) == 0.0


# ---------------------------------------------------------------------------
# Mode selection and flag plumbing
# ---------------------------------------------------------------------------

def _capture_commands(monkeypatch):
    commands: list[list[str]] = []
    real_run = build_mod._run

    def spy(cmd):
        commands.append(cmd)
        return real_run(cmd)

    monkeypatch.setattr(build_mod, "_run", spy)
    return commands


SIMPLE = (
    "from postyp import Float64\n"
    "def triple(x: Float64) -> Float64:\n"
    "    return x * 3.0\n"
)


@needs_cc
def test_strict_is_default_and_compile_only(tmp_path, monkeypatch):
    commands = _capture_commands(monkeypatch)
    build_source(SIMPLE, output=tmp_path / "s.so")
    compiles = [c for c in commands if "-c" in c]
    links = [c for c in commands if "-c" not in c]
    assert compiles and links
    for c in compiles:
        assert "-ffp-contract=off" in c
        assert "-ffast-math" not in c
    for c in links:  # FP flags never reach the link step (no crtfastmath)
        assert "-ffp-contract=off" not in c
        assert "-ffast-math" not in c


@needs_cc
def test_fast_math_is_explicit_opt_in(tmp_path, monkeypatch):
    commands = _capture_commands(monkeypatch)
    out = build_source(SIMPLE, output=tmp_path / "f.so", fp="fast")
    compiles = [c for c in commands if "-c" in c]
    for c in compiles:
        assert "-ffast-math" in c
        assert "-ffp-contract=off" not in c
    links = [c for c in commands if "-c" not in c]
    for c in links:
        assert "-ffast-math" not in c
    lib = ctypes.CDLL(str(out))
    triple = _fn(lib, "triple", 1)
    assert triple(2.0) == 6.0


@needs_cc
def test_build_file_threads_fp_mode(tmp_path, monkeypatch):
    src = tmp_path / "kern.py"
    src.write_text(SIMPLE)
    commands = _capture_commands(monkeypatch)
    build_file(src, output=tmp_path / "kern.so", fp="fast")
    assert any("-ffast-math" in c for c in commands if "-c" in c)


def test_unknown_fp_mode_is_a_build_error(tmp_path):
    with pytest.raises(BuildError, match="floating-point mode"):
        build_source(SIMPLE, output=tmp_path / "x.so", fp="loose")


@needs_cc
def test_cli_fp_flag(tmp_path, capsys, monkeypatch):
    src = tmp_path / "kern.py"
    src.write_text(SIMPLE)
    commands = _capture_commands(monkeypatch)
    out = tmp_path / "kern.so"
    assert main(["build", str(src), "--output", str(out), "--fp", "fast"]) == 0
    assert out.exists()
    assert any("-ffast-math" in c for c in commands if "-c" in c)
