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


@needs_cc
def test_int_division_is_true_division():
    lib = _build(
        "from postyp import Int64, Float64\n"
        "def divide(a: Int64, b: Int64) -> Float64:\n"
        "    return a / b\n"
    )
    divide = lib.divide
    divide.argtypes = [ctypes.c_int64, ctypes.c_int64]
    divide.restype = ctypes.c_double

    assert divide(7, 2) == 7 / 2 == 3.5
    assert divide(-1, 4) == -1 / 4 == -0.25
    assert divide(1, 3) == 1 / 3
    assert divide(-7, 2) == -3.5


@needs_cc
def test_sequential_loops_reusing_variable_compile_and_run():
    lib = _build(
        "from postyp import Int64\n"
        "def twice(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    return total\n"
    )
    twice = lib.twice
    twice.argtypes = [ctypes.c_int64]
    twice.restype = ctypes.c_int64
    assert twice(5) == 2 * sum(range(5)) == 20
    assert twice(0) == 0


@needs_cc
def test_loop_variable_shadowing_parameter_matches_python():
    def reference(n):
        total = 0
        for n in range(n):
            total += n
        return total

    lib = _build(
        "from postyp import Int64\n"
        "def shadow(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    for n in range(n):\n"
        "        total += n\n"
        "    return total\n"
    )
    shadow = lib.shadow
    shadow.argtypes = [ctypes.c_int64]
    shadow.restype = ctypes.c_int64
    assert shadow(5) == reference(5) == 10
    assert shadow(1) == reference(1) == 0


@needs_cc
def test_walrus_assigns_and_returns_value():
    def reference(x):
        if (y := x * 2.0) > 1.0:
            return y
        return x

    lib = _build(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    if (y := x * 2.0) > 1.0:\n"
        "        return y\n"
        "    return x\n"
    )
    f = lib.f
    f.argtypes = [ctypes.c_double]
    f.restype = ctypes.c_double
    assert f(1.0) == reference(1.0) == 2.0
    assert f(0.25) == reference(0.25) == 0.25


# ---------------------------------------------------------------------------
# NumPy gufunc ABI: the loop must honor outer AND inner (core-dim) steps
# ---------------------------------------------------------------------------

_DOT_SOURCE = (
    "from postyp import Array, Float64\n"
    "from postpython import guvectorize\n"
    "\n"
    "@guvectorize([], \"(n),(n)->()\")\n"
    "def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:\n"
    "    result: Float64 = 0.0\n"
    "    for i in range(len(a)):\n"
    "        result += a[i] * b[i]\n"
    "    out[0] = result\n"
)


def _call_dot_loop(lib, a_buf, b_buf, out_buf, outer_n, core_n, steps_list):
    """Invoke dot_ufunc_loop with the NumPy gufunc calling convention.

    steps_list = [outer_a, outer_b, outer_out, core_a, core_b] in bytes.
    """
    loop = lib.dot_ufunc_loop
    loop.restype = None

    args = (ctypes.c_void_p * 3)(
        ctypes.cast(a_buf, ctypes.c_void_p),
        ctypes.cast(b_buf, ctypes.c_void_p),
        ctypes.cast(out_buf, ctypes.c_void_p),
    )
    dimensions = (ctypes.c_ssize_t * 2)(outer_n, core_n)
    steps = (ctypes.c_ssize_t * len(steps_list))(*steps_list)
    loop(args, dimensions, steps, None)


@needs_cc
def test_gufunc_loop_honors_outer_and_broadcast_steps():
    lib = _build(_DOT_SOURCE)

    # a: two contiguous rows of 4; b: one row broadcast across the outer
    # loop via a zero outer step (NumPy's broadcasting convention).
    a = (ctypes.c_double * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    b = (ctypes.c_double * 4)(10, 20, 30, 40)
    out = (ctypes.c_double * 2)(0, 0)

    _call_dot_loop(
        lib, a, b, out,
        outer_n=2, core_n=4,
        steps_list=[32, 0, 8, 8, 8],
    )
    assert list(out) == [300.0, 700.0]


@needs_cc
def test_gufunc_loop_honors_non_contiguous_core_steps():
    lib = _build(_DOT_SOURCE)

    # a is logically [1, 3, 5, 7]: every other element of an 8-element
    # buffer, expressed through a 16-byte core step. A compact-layout
    # assumption would read [1, 2, 3, 4] and produce 10 instead of 16.
    a = (ctypes.c_double * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    b = (ctypes.c_double * 4)(1, 1, 1, 1)
    out = (ctypes.c_double * 1)(0)

    _call_dot_loop(
        lib, a, b, out,
        outer_n=1, core_n=4,
        steps_list=[0, 0, 0, 16, 8],
    )
    assert out[0] == 1 + 3 + 5 + 7 == 16.0
