"""Semantic-correctness tests: type promotion, true division, loop-variable
reuse, walrus lowering, and diagnose-instead-of-drop expression handling.
"""

import itertools

import pytest

from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_source
from postpython.compiler.typechecker import promote
from postyp import (
    Bool,
    Complex64, Complex128,
    Float16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
)

_NUMERIC = [
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128,
]


def _module(source: str):
    module, errors = compile_source(source)
    assert errors == [], errors
    return module


def _emit(source: str) -> str:
    return emit_module(_module(source))


def _errors(source: str):
    _, errors = compile_source(source)
    return errors


# ---------------------------------------------------------------------------
# promote() — symmetry and array-api conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", itertools.combinations(_NUMERIC, 2))
def test_promote_is_symmetric(a, b):
    assert promote(a, b) is promote(b, a)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Bool, Int64, Int64),
        (Bool, UInt8, UInt8),
        (Bool, Float32, Float32),
        (Bool, Bool, Bool),
        (UInt8, Int32, Int32),
        (UInt16, Int16, Int32),   # widen-signed with one extra bit
        (UInt32, Int8, Int64),
        (UInt64, Int64, Int64),   # saturates at Int64 (documented rule)
        (Int32, Int64, Int64),
        (UInt8, UInt32, UInt32),
        (Int64, Float16, Float16),
        (Float32, Float64, Float64),
        (Complex64, Float32, Complex64),
        (Complex64, Float64, Complex128),  # Float64 precision must survive
        (Complex64, Int64, Complex64),
        (Complex128, Float32, Complex128),
    ],
)
def test_promote_expected_results(a, b, expected):
    assert promote(a, b) is expected
    assert promote(b, a) is expected


# ---------------------------------------------------------------------------
# True division: int / int is Float64 with a C operand cast
# ---------------------------------------------------------------------------

def test_int_division_result_is_float64_in_c():
    c = _emit(
        "from postyp import Int64, Float64\n"
        "def halve(x: Int64) -> Float64:\n"
        "    return x / 2\n"
    )
    assert "double _v" in c
    # The left operand must be cast so C integer division never happens.
    assert "(double)(_x) /" in c


def test_float_division_is_unchanged():
    c = _emit(
        "from postyp import Float64\n"
        "def f(x: Float64, y: Float64) -> Float64:\n"
        "    return x / y\n"
    )
    assert "(double)(_x) / (_y);" in c


def test_floor_division_stays_integral():
    c = _emit(
        "from postyp import Int64\n"
        "def f(x: Int64, y: Int64) -> Int64:\n"
        "    return x // y\n"
    )
    assert "__pp_floordiv_si" in c


def test_aug_assign_division_promotes_to_float():
    c = _emit(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    total: Float64 = 10.0\n"
        "    total /= x\n"
        "    return total\n"
    )
    assert "(double)(_total) / (_x);" in c


# ---------------------------------------------------------------------------
# Loop-variable reuse
# ---------------------------------------------------------------------------

def test_sequential_loops_reusing_variable_declare_once():
    c = _emit(
        "from postyp import Int64\n"
        "def twice(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    return total\n"
    )
    declarations = [
        line for line in c.splitlines() if line.strip().startswith("int64_t _i =")
    ]
    assert len(declarations) == 1, declarations


def test_loop_variable_shadowing_parameter_does_not_redeclare():
    c = _emit(
        "from postyp import Int64\n"
        "def shadow(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    for n in range(3):\n"
        "        total += n\n"
        "    return total\n"
    )
    # The parameter already declares _n in the C signature; the loop must
    # not introduce a second declaration.
    declarations = [
        line for line in c.splitlines() if line.strip().startswith("int64_t _n =")
    ]
    assert declarations == []


def test_self_referential_range_snapshots_stop_value():
    # Python evaluates range(n) before rebinding n; the emitted C must
    # compare against a snapshot, not the reassigned loop variable.
    c = _emit(
        "from postyp import Int64\n"
        "def f(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    for n in range(n):\n"
        "        total += n\n"
        "    return total\n"
    )
    assert "_n < _n;" not in c


# ---------------------------------------------------------------------------
# Walrus operator
# ---------------------------------------------------------------------------

def test_walrus_lowers_to_assignment_and_value():
    module, errors = compile_source(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    if (y := x * 2.0) > 1.0:\n"
        "        return y\n"
        "    return x\n"
    )
    assert errors == [], errors
    c = emit_module(module)
    assert "double _y = " in c


def test_walrus_in_while_condition():
    errors = _errors(
        "from postyp import Int64\n"
        "def f(n: Int64) -> Int64:\n"
        "    total: Int64 = 0\n"
        "    while (m := n - 1) > 0:\n"
        "        total += m\n"
        "        n = m\n"
        "    return total\n"
    )
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Diagnose-instead-of-drop: silent miscompilations become PP900
# ---------------------------------------------------------------------------

def _assert_pp900(source: str, needle: str):
    errors = _errors(source)
    assert errors, f"expected a PP900 diagnostic mentioning {needle!r}"
    matching = [e for e in errors if e.code == "PP900" and needle in e.message]
    assert matching, f"no PP900 mentioning {needle!r} in: {[str(e) for e in errors]}"


def test_unknown_name_reports_pp900():
    _assert_pp900(
        "from postyp import Float64\n"
        "from postpython.math import PI\n"
        "def circ(r: Float64) -> Float64:\n"
        "    return 2.0 * PI * r\n",
        "PI",
    )


def test_attribute_call_reports_pp900():
    _assert_pp900(
        "import math\n"
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return math.sqrt(x)\n",
        "math.sqrt",
    )


def test_is_comparison_reports_pp900():
    _assert_pp900(
        "from postyp import Int64, Bool\n"
        "def f(x: Int64) -> Bool:\n"
        "    return x is 0\n",
        "Is",
    )


def test_bitwise_operator_reports_pp900():
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    return x | 1\n",
        "BitOr",
    )


def test_invert_operator_reports_pp900():
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    return ~x\n",
        "Invert",
    )


def test_tuple_value_assignment_reports_pp900():
    # The tuple RHS fails to lower first, which is diagnosed rather than
    # silently dropping the assignment.
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    a, b = x, x\n"
        "    return a\n",
        "Tuple",
    )


def test_tuple_target_assignment_reports_pp900():
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    (a, b) = x\n"
        "    return x\n",
        "assignment targets",
    )


def test_range_as_value_reports_pp900():
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    r = range(x)\n"
        "    return x\n",
        "range()",
    )


def test_fstring_reports_pp900():
    _assert_pp900(
        "from postyp import Int64, Str\n"
        "def f(x: Int64) -> Str:\n"
        "    return f\"{x}\"\n",
        "JoinedStr",
    )


def test_aug_assign_bitwise_reports_pp900():
    _assert_pp900(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    x |= 1\n"
        "    return x\n",
        "BitOr",
    )


def test_unary_plus_is_identity_not_negation():
    module, errors = compile_source(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    return +x\n"
    )
    assert errors == [], errors
    c = emit_module(module)
    assert "return _x;" in c
    assert "-_x" not in c
