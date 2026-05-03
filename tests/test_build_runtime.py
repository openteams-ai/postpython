"""End-to-end build tests: source → C → shared library → ctypes call.

Verifies that the semantic fixes for floor division, integer power, and
abs() produce correct runtime values (not just correct C source text).
"""

import ctypes
import shutil

import pytest

from postpython.build import build_source

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


def _build(source: str):
    return ctypes.CDLL(str(build_source(source, filename="rt.py")))


@needs_cc
def test_signed_floor_div_matches_python_semantics():
    lib = _build(
        "from postyp import Int64\n"
        "def fdiv(a: Int64, b: Int64) -> Int64:\n"
        "    return a // b\n"
    )
    fdiv = lib.fdiv
    fdiv.argtypes = [ctypes.c_int64, ctypes.c_int64]
    fdiv.restype = ctypes.c_int64

    # Cases where Python's floor differs from C's truncation:
    assert fdiv(-7, 2) == -7 // 2 == -4
    assert fdiv(7, -2) == 7 // -2 == -4
    assert fdiv(-7, -2) == -7 // -2 == 3
    # Positive divides should still match:
    assert fdiv(7, 2) == 3
    # Exact division:
    assert fdiv(-8, 2) == -4
    assert fdiv(-8, -2) == 4


@needs_cc
def test_integer_pow_preserves_full_int64_precision():
    lib = _build(
        "from postyp import Int64\n"
        "def ipow(a: Int64, b: Int64) -> Int64:\n"
        "    return a ** b\n"
    )
    ipow = lib.ipow
    ipow.argtypes = [ctypes.c_int64, ctypes.c_int64]
    ipow.restype = ctypes.c_int64

    # 7 ** 22 = 3909821048582988049 fits in int64 but exceeds double's 53-bit
    # mantissa, so a libm pow() round-trip would lose low-order bits.
    expected = 7 ** 22
    assert expected < 2 ** 63
    assert int(float(expected)) != expected  # confirms double can't hold it
    assert ipow(7, 22) == expected

    assert ipow(7, 3) == 343
    assert ipow(5, 0) == 1
    assert ipow(-3, 3) == -27


@needs_cc
def test_abs_int64_returns_unsigned_magnitude():
    lib = _build(
        "from postyp import Int64\n"
        "def myabs(x: Int64) -> Int64:\n"
        "    return abs(x)\n"
    )
    myabs = lib.myabs
    myabs.argtypes = [ctypes.c_int64]
    myabs.restype = ctypes.c_int64
    assert myabs(-42) == 42
    assert myabs(42) == 42
    assert myabs(0) == 0


@needs_cc
def test_abs_float64_returns_magnitude():
    lib = _build(
        "from postyp import Float64\n"
        "def myabs(x: Float64) -> Float64:\n"
        "    return abs(x)\n"
    )
    myabs = lib.myabs
    myabs.argtypes = [ctypes.c_double]
    myabs.restype = ctypes.c_double
    assert myabs(-3.5) == 3.5
    assert myabs(3.5) == 3.5
    assert myabs(0.0) == 0.0


@needs_cc
def test_chained_compare_evaluates_logical_and():
    lib = _build(
        "from postyp import Int64, Bool\n"
        "def in_range(x: Int64) -> Bool:\n"
        "    return 0 < x < 10\n"
    )
    in_range = lib.in_range
    in_range.argtypes = [ctypes.c_int64]
    in_range.restype = ctypes.c_bool
    assert in_range(5) is True
    assert in_range(0) is False
    assert in_range(10) is False
    assert in_range(-1) is False
    assert in_range(9) is True


@needs_cc
def test_bool_and_or_short_circuit_at_c_level():
    lib = _build(
        "from postyp import Bool\n"
        "def both(a: Bool, b: Bool) -> Bool:\n"
        "    return a and b\n"
        "def either(a: Bool, b: Bool) -> Bool:\n"
        "    return a or b\n"
    )
    both = lib.both
    both.argtypes = [ctypes.c_bool, ctypes.c_bool]
    both.restype = ctypes.c_bool
    either = lib.either
    either.argtypes = [ctypes.c_bool, ctypes.c_bool]
    either.restype = ctypes.c_bool

    for a in (False, True):
        for b in (False, True):
            assert both(a, b) is (a and b)
            assert either(a, b) is (a or b)
