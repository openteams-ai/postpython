"""POST Python math library — scalar libm wrappers.

These are plain typed functions (not gufuncs).  The compiler lowers each
call to the corresponding libm symbol.  In interpreted mode they delegate
to Python's math module.

ppspecial and other numerical libraries import from here rather than
calling math directly, so the compiler can intercept and inline the calls.
"""

from __future__ import annotations

import math as _m

from postyp import Float64, Bool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI:      Float64 = _m.pi
E:       Float64 = _m.e
INF:     Float64 = float("inf")
NAN:     Float64 = float("nan")
LOG2:    Float64 = _m.log(2.0)
LOG_PI:  Float64 = _m.log(_m.pi)
SQRT2:   Float64 = _m.sqrt(2.0)
SQRT_PI: Float64 = _m.sqrt(_m.pi)
EULER:   Float64 = 0.5772156649015328606   # Euler–Mascheroni constant


# ---------------------------------------------------------------------------
# Exponential / logarithm
# ---------------------------------------------------------------------------

def exp(x: Float64) -> Float64:
    """e^x"""
    return _m.exp(x)


def exp2(x: Float64) -> Float64:
    """2^x"""
    return _m.pow(2.0, x)


def expm1(x: Float64) -> Float64:
    """e^x - 1, accurate for small x"""
    return _m.expm1(x)


def log(x: Float64) -> Float64:
    """Natural logarithm ln(x)"""
    return _m.log(x)


def log2(x: Float64) -> Float64:
    """Base-2 logarithm"""
    return _m.log2(x)


def log10(x: Float64) -> Float64:
    """Base-10 logarithm"""
    return _m.log10(x)


def log1p(x: Float64) -> Float64:
    """ln(1 + x), accurate for small x"""
    return _m.log1p(x)


# ---------------------------------------------------------------------------
# Power / root
# ---------------------------------------------------------------------------

def sqrt(x: Float64) -> Float64:
    """Square root"""
    return _m.sqrt(x)


def cbrt(x: Float64) -> Float64:
    """Cube root (preserves sign)"""
    if x < 0.0:
        return -((-x) ** (1.0 / 3.0))
    return x ** (1.0 / 3.0)


def pow(base: Float64, exp_: Float64) -> Float64:
    """base ** exp_"""
    return _m.pow(base, exp_)


def hypot(x: Float64, y: Float64) -> Float64:
    """sqrt(x^2 + y^2)"""
    return _m.hypot(x, y)


# ---------------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------------

def sin(x: Float64) -> Float64:
    return _m.sin(x)


def cos(x: Float64) -> Float64:
    return _m.cos(x)


def tan(x: Float64) -> Float64:
    return _m.tan(x)


def asin(x: Float64) -> Float64:
    return _m.asin(x)


def acos(x: Float64) -> Float64:
    return _m.acos(x)


def atan(x: Float64) -> Float64:
    return _m.atan(x)


def atan2(y: Float64, x: Float64) -> Float64:
    return _m.atan2(y, x)


def sincos_sin(x: Float64) -> Float64:
    """sin component; pair with sincos_cos for efficient simultaneous evaluation."""
    return _m.sin(x)


def sincos_cos(x: Float64) -> Float64:
    return _m.cos(x)


# ---------------------------------------------------------------------------
# Hyperbolic
# ---------------------------------------------------------------------------

def sinh(x: Float64) -> Float64:
    return _m.sinh(x)


def cosh(x: Float64) -> Float64:
    return _m.cosh(x)


def tanh(x: Float64) -> Float64:
    return _m.tanh(x)


def asinh(x: Float64) -> Float64:
    return _m.asinh(x)


def acosh(x: Float64) -> Float64:
    return _m.acosh(x)


def atanh(x: Float64) -> Float64:
    return _m.atanh(x)


# ---------------------------------------------------------------------------
# Rounding / sign / misc
# ---------------------------------------------------------------------------

def floor(x: Float64) -> Float64:
    return float(_m.floor(x))


def ceil(x: Float64) -> Float64:
    return float(_m.ceil(x))


def fabs(x: Float64) -> Float64:
    return _m.fabs(x)


def copysign(x: Float64, y: Float64) -> Float64:
    return _m.copysign(x, y)


def fmod(x: Float64, y: Float64) -> Float64:
    return _m.fmod(x, y)


def isfinite(x: Float64) -> Bool:
    return _m.isfinite(x)


def isinf(x: Float64) -> Bool:
    return _m.isinf(x)


def isnan(x: Float64) -> Bool:
    return _m.isnan(x)
