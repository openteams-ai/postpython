"""Standard binary ufuncs — signature (),()->()

These mirror NumPy's element-wise binary ufuncs such as np.add,
np.multiply, np.power, np.maximum, np.minimum, np.copysign, np.hypot,
np.logical_and, np.logical_or, and the comparison ufuncs.

A @vectorize binary kernel takes two scalars and returns one.  Like all ufuncs,
broadcasting over arbitrary array shapes is handled by the runtime.
"""

from postyp import Float64, Int64, Bool
from postpython import vectorize


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

@vectorize
def pp_add(x: Float64, y: Float64) -> Float64:
    """x + y  (mirrors np.add)"""
    return x + y


@vectorize
def pp_subtract(x: Float64, y: Float64) -> Float64:
    """x - y  (mirrors np.subtract)"""
    return x - y


@vectorize
def pp_multiply(x: Float64, y: Float64) -> Float64:
    """x * y  (mirrors np.multiply)"""
    return x * y


@vectorize
def pp_divide(x: Float64, y: Float64) -> Float64:
    """x / y  (mirrors np.divide; undefined for y == 0)"""
    return x / y


@vectorize
def pp_power(base: Float64, exp: Float64) -> Float64:
    """base ** exp  (mirrors np.power)"""
    return base ** exp


@vectorize
def pp_mod(x: Float64, y: Float64) -> Float64:
    """x % y  (mirrors np.mod / np.remainder)"""
    return x % y


# ---------------------------------------------------------------------------
# Extrema
# ---------------------------------------------------------------------------

@vectorize
def pp_maximum(x: Float64, y: Float64) -> Float64:
    """Element-wise maximum (mirrors np.maximum; NaN-propagating)."""
    if x >= y:
        return x
    return y


@vectorize
def pp_minimum(x: Float64, y: Float64) -> Float64:
    """Element-wise minimum (mirrors np.minimum; NaN-propagating)."""
    if x <= y:
        return x
    return y


@vectorize
def pp_fmod(x: Float64, y: Float64) -> Float64:
    """C-style floating-point remainder: sign follows x (mirrors np.fmod)."""
    return x - (x // y) * y


# ---------------------------------------------------------------------------
# Geometry / utilities
# ---------------------------------------------------------------------------

@vectorize
def hypot(x: Float64, y: Float64) -> Float64:
    """√(x² + y²) without intermediate overflow (mirrors np.hypot)."""
    a: Float64 = x * x + y * y
    return a ** 0.5


@vectorize
def copysign(magnitude: Float64, sign_src: Float64) -> Float64:
    """Return magnitude with the sign of sign_src (mirrors np.copysign)."""
    if magnitude < 0.0:
        magnitude = -magnitude
    if sign_src < 0.0:
        return -magnitude
    return magnitude


@vectorize
def ldexp(x: Float64, n: Int64) -> Float64:
    """x * 2**n  (mirrors np.ldexp; integer exponent)."""
    result: Float64 = x
    i: Int64 = 0
    if n >= 0:
        for i in range(n):
            result = result * 2.0
    else:
        for i in range(-n):
            result = result * 0.5
    return result


# ---------------------------------------------------------------------------
# Comparison ufuncs (return Bool)
# ---------------------------------------------------------------------------

@vectorize
def pp_equal(x: Float64, y: Float64) -> Bool:
    """x == y  (mirrors np.equal)"""
    return x == y


@vectorize
def pp_not_equal(x: Float64, y: Float64) -> Bool:
    """x != y  (mirrors np.not_equal)"""
    return x != y


@vectorize
def pp_less(x: Float64, y: Float64) -> Bool:
    """x < y  (mirrors np.less)"""
    return x < y


@vectorize
def pp_less_equal(x: Float64, y: Float64) -> Bool:
    """x <= y  (mirrors np.less_equal)"""
    return x <= y


@vectorize
def pp_greater(x: Float64, y: Float64) -> Bool:
    """x > y  (mirrors np.greater)"""
    return x > y


@vectorize
def pp_greater_equal(x: Float64, y: Float64) -> Bool:
    """x >= y  (mirrors np.greater_equal)"""
    return x >= y


# ---------------------------------------------------------------------------
# Logical ufuncs (Bool inputs and outputs)
# ---------------------------------------------------------------------------

@vectorize
def pp_logical_and(x: Bool, y: Bool) -> Bool:
    """x and y  (mirrors np.logical_and)"""
    return x and y


@vectorize
def pp_logical_or(x: Bool, y: Bool) -> Bool:
    """x or y  (mirrors np.logical_or)"""
    return x or y


@vectorize
def pp_logical_xor(x: Bool, y: Bool) -> Bool:
    """x XOR y  (mirrors np.logical_xor)"""
    return (x or y) and not (x and y)


# ---------------------------------------------------------------------------
# Self-test (interpreted mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EPS: Float64 = 1e-9

    def close(a: Float64, b: Float64) -> Bool:
        diff: Float64 = a - b
        if diff < 0.0:
            diff = -diff
        return diff < EPS

    cases = [
        ("pp_add",       pp_add,       (3.0,  4.0),   7.0),
        ("pp_subtract",  pp_subtract,  (10.0, 3.0),   7.0),
        ("pp_multiply",  pp_multiply,  (3.0,  4.0),   12.0),
        ("pp_divide",    pp_divide,    (7.0,  2.0),   3.5),
        ("pp_power",     pp_power,     (2.0,  10.0),  1024.0),
        ("pp_maximum",   pp_maximum,   (3.0,  7.0),   7.0),
        ("pp_minimum",   pp_minimum,   (3.0,  7.0),   3.0),
        ("hypot",        hypot,        (3.0,  4.0),   5.0),
        ("copysign",     copysign,     (3.0, -1.0),   -3.0),
        ("ldexp",        ldexp,        (1.0,  3),     8.0),
    ]

    all_ok: Bool = True
    for name, fn, args, expected in cases:
        got: Float64 = fn(*args)
        ok: Bool = close(got, expected)
        status: str = "ok  " if ok else "FAIL"
        print(f"  {status}  {name}{args} = {got}  (expected {expected})")
        if not ok:
            all_ok = False

    print()

    # Comparison ufuncs.
    cmp_cases = [
        ("pp_equal",         pp_equal,         (1.0, 1.0),  True),
        ("pp_not_equal",     pp_not_equal,     (1.0, 2.0),  True),
        ("pp_less",          pp_less,          (1.0, 2.0),  True),
        ("pp_greater_equal", pp_greater_equal, (3.0, 3.0),  True),
        ("pp_logical_and",   pp_logical_and,   (True, False), False),
        ("pp_logical_or",    pp_logical_or,    (False, True), True),
        ("pp_logical_xor",   pp_logical_xor,   (True, True),  False),
    ]
    for name, fn, args, expected in cmp_cases:
        got = fn(*args)
        ok = (got == expected)
        status = "ok  " if ok else "FAIL"
        print(f"  {status}  {name}{args} = {got}  (expected {expected})")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All binary ufunc scalar tests passed.")
    else:
        print("Some tests FAILED.")
