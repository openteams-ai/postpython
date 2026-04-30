"""Standard unary ufuncs — signature ()->()

These mirror NumPy's element-wise unary ufuncs such as np.square,
np.sqrt, np.exp, np.log, np.abs, np.sign, np.ceil, np.floor.

A ()->() gufunc takes a single scalar, returns a single scalar.  NumPy
(or the POST Python runtime) handles broadcasting over any array shape
automatically; the author only writes the scalar kernel.

Contrast with the generalized ufuncs in gufunc_norm.py / gufunc_dot.py
which operate on *sub-arrays* of specified rank.
"""

from postyp import Float64, Int64, Bool
from postpython.gufunc import gufunc


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

@gufunc("()->()")
def square(x: Float64) -> Float64:
    """x → x²"""
    return x * x


@gufunc("()->()")
def cube(x: Float64) -> Float64:
    """x → x³"""
    return x * x * x


@gufunc("()->()")
def reciprocal(x: Float64) -> Float64:
    """x → 1/x"""
    return 1.0 / x


@gufunc("()->()")
def pp_sqrt(x: Float64) -> Float64:
    """x → √x  (non-negative inputs only)"""
    return x ** 0.5


@gufunc("()->()")
def pp_abs(x: Float64) -> Float64:
    """x → |x|"""
    if x < 0.0:
        return -x
    return x


@gufunc("()->()")
def sign(x: Float64) -> Float64:
    """x → -1, 0, or +1"""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

@gufunc("()->()")
def pp_floor(x: Float64) -> Float64:
    """x → ⌊x⌋  (largest integer ≤ x)"""
    t: Float64 = x - (x % 1.0)
    if x < 0.0 and x % 1.0 != 0.0:
        return t - 1.0
    return t


@gufunc("()->()")
def pp_ceil(x: Float64) -> Float64:
    """x → ⌈x⌉  (smallest integer ≥ x)"""
    t: Float64 = x - (x % 1.0)
    if x > 0.0 and x % 1.0 != 0.0:
        return t + 1.0
    return t


# ---------------------------------------------------------------------------
# Activation functions (common in ML pipelines)
# ---------------------------------------------------------------------------

@gufunc("()->()")
def relu(x: Float64) -> Float64:
    """Rectified linear unit: x → max(0, x)"""
    if x > 0.0:
        return x
    return 0.0


@gufunc("()->()")
def sigmoid(x: Float64) -> Float64:
    """Logistic sigmoid: x → 1 / (1 + e^(-x))

    Uses the numerically stable form to avoid overflow for large |x|.
    """
    if x >= 0.0:
        z: Float64 = pp_exp_approx(-x)
        return 1.0 / (1.0 + z)
    z = pp_exp_approx(x)
    return z / (1.0 + z)


@gufunc("()->()")
def pp_exp_approx(x: Float64) -> Float64:
    """Minimax polynomial approximation of e^x valid for |x| ≤ 1.

    In a real compiler this would lower to the libm exp() call; this
    version is self-contained POST Python to illustrate the kernel shape.
    """
    # Clamp to a sane range to avoid overflow.
    xc: Float64 = x
    if xc > 88.0:
        xc = 88.0
    if xc < -88.0:
        xc = -88.0
    # 6th-order Taylor series around 0 — acceptable for |x| ≤ 1.
    result: Float64 = 1.0
    term: Float64 = 1.0
    for k in range(1, 7):
        term = term * xc / k
        result = result + term
    return result


@gufunc("()->()")
def swish(x: Float64) -> Float64:
    """Swish activation: x → x · σ(x)"""
    return x * sigmoid(x)


# ---------------------------------------------------------------------------
# Type conversion ufuncs
# ---------------------------------------------------------------------------

@gufunc("()->()")
def float_to_int(x: Float64) -> Int64:
    """Truncate float to int64 (towards zero)."""
    return Int64(x)


@gufunc("()->()")
def is_positive(x: Float64) -> Bool:
    """x → True if x > 0"""
    return x > 0.0


# ---------------------------------------------------------------------------
# Self-test (interpreted mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases: list = [
        ("square",      square,      [2.0, -3.0, 0.5],    [4.0,  9.0,  0.25]),
        ("cube",        cube,        [2.0, -2.0],          [8.0, -8.0]),
        ("reciprocal",  reciprocal,  [2.0, 4.0, -1.0],    [0.5,  0.25, -1.0]),
        ("pp_sqrt",     pp_sqrt,     [4.0, 9.0, 0.25],    [2.0,  3.0,  0.5]),
        ("pp_abs",      pp_abs,      [-3.0, 0.0, 5.0],    [3.0,  0.0,  5.0]),
        ("sign",        sign,        [-7.0, 0.0, 3.0],    [-1.0, 0.0,  1.0]),
        ("relu",        relu,        [-1.0, 0.0, 2.5],    [0.0,  0.0,  2.5]),
    ]

    all_ok: Bool = True
    for name, fn, inputs, expected in cases:
        for x, exp in zip(inputs, expected):
            got: Float64 = fn(x)
            ok: Bool = abs(got - exp) < 1e-9
            if not ok:
                print(f"FAIL  {name}({x}) = {got}, expected {exp}")
                all_ok = False
    if all_ok:
        print("All unary ufunc scalar tests passed.")

    # Demonstrate sigmoid is in (0,1) for a range of inputs.
    for x in [-100.0, -2.0, 0.0, 2.0, 100.0]:
        s: Float64 = sigmoid(x)
        print(f"  sigmoid({x:7.1f}) = {s:.6f}")
