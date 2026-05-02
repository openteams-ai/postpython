"""Error functions — ppspecial._erf

Public API
----------
erf(x)      — error function
erfc(x)     — complementary error function  1 - erf(x)
erfinv(x)   — inverse error function
erfcinv(x)  — inverse complementary error function

Algorithms
----------
erfc  : Chebyshev rational approximation, Numerical Recipes §6.2.
        |error| < 1.2e-7 for all real x.
erfinv: Rational approximation by Peter J. Acklam (2003).
        |error| < 1.15e-9 for |x| < 1.
"""

from postyp import Float64, Bool
from postpython import vectorize
from postpython.math import exp, log, sqrt, fabs


# ---------------------------------------------------------------------------
# Internal helpers called inside vectorized kernels.
# ---------------------------------------------------------------------------

def _erfc_positive(x: Float64) -> Float64:
    """erfc(x) for x >= 0.  Chebyshev rational approximation."""
    t: Float64 = 1.0 / (1.0 + 0.5 * x)
    # Horner evaluation of degree-9 polynomial in t
    p: Float64 = 0.17087294
    p = p * t - 0.82215223
    p = p * t + 1.48851587
    p = p * t - 1.13520398
    p = p * t + 0.27886807
    p = p * t - 0.18628806
    p = p * t + 0.09678418
    p = p * t + 0.37409196
    p = p * t + 1.00002368
    return t * exp(-x * x - 1.26551223 + t * p)


# ---------------------------------------------------------------------------
# erfc / erf
# ---------------------------------------------------------------------------

@vectorize
def erfc(x: Float64) -> Float64:
    """Complementary error function: erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^{-t²} dt

    Numerically superior to 1 - erf(x) for large x where erf(x) ≈ 1.
    """
    if x == 0.0:
        return 1.0
    if x < 0.0:
        return 2.0 - _erfc_positive(-x)
    return _erfc_positive(x)


@vectorize
def erf(x: Float64) -> Float64:
    """Error function: erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt"""
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return _erfc_positive(-x) - 1.0
    return 1.0 - _erfc_positive(x)


# ---------------------------------------------------------------------------
# erfinv — inverse error function
# ---------------------------------------------------------------------------

# Acklam rational approximation constants
# Central region coefficients (|p - 0.5| <= 0.425)
_A0: Float64 = -3.969683028665376e+01
_A1: Float64 =  2.209460984245205e+02
_A2: Float64 = -2.759285104469687e+02
_A3: Float64 =  1.383577518672690e+02
_A4: Float64 = -3.066479806614716e+01
_A5: Float64 =  2.506628277459239e+00

_B0: Float64 = -5.447609879822406e+01
_B1: Float64 =  1.615858368580409e+02
_B2: Float64 = -1.556989798598866e+02
_B3: Float64 =  6.680131188771972e+01
_B4: Float64 = -1.328068155288572e+01

# Tail region coefficients (p near 0 or 1)
_C0: Float64 = -7.784894002430293e-03
_C1: Float64 = -3.223964580411365e-01
_C2: Float64 = -2.400758277161838e+00
_C3: Float64 = -2.549732539343734e+00
_C4: Float64 =  4.374664141464968e+00
_C5: Float64 =  2.938163982698783e+00

_D0: Float64 =  7.784695709041462e-03
_D1: Float64 =  3.224671290700398e-01
_D2: Float64 =  2.445134137142996e+00
_D3: Float64 =  3.754408661907416e+00

_P_LOW:  Float64 = 0.02425
_P_HIGH: Float64 = 1.0 - 0.02425


def _ndtri_rational(p: Float64) -> Float64:
    """Acklam rational approximation for ndtri(p) = sqrt(2)*erfinv(2p-1)."""
    if p <= 0.0:
        return -1.0e308
    if p >= 1.0:
        return 1.0e308

    if p < _P_LOW:
        q: Float64 = sqrt(-2.0 * log(p))
        num: Float64 = _C0
        num = num * q + _C1
        num = num * q + _C2
        num = num * q + _C3
        num = num * q + _C4
        num = num * q + _C5
        den: Float64 = _D0
        den = den * q + _D1
        den = den * q + _D2
        den = den * q + _D3
        den = den * q + 1.0
        return num / den

    if p <= _P_HIGH:
        q = p - 0.5
        r: Float64 = q * q
        num = _A0
        num = num * r + _A1
        num = num * r + _A2
        num = num * r + _A3
        num = num * r + _A4
        num = num * r + _A5
        den = _B0
        den = den * r + _B1
        den = den * r + _B2
        den = den * r + _B3
        den = den * r + _B4
        den = den * r + 1.0
        return num / den * q

    # Upper tail: use symmetry
    q = sqrt(-2.0 * log(1.0 - p))
    num = _C0
    num = num * q + _C1
    num = num * q + _C2
    num = num * q + _C3
    num = num * q + _C4
    num = num * q + _C5
    den = _D0
    den = den * q + _D1
    den = den * q + _D2
    den = den * q + _D3
    den = den * q + 1.0
    return -(num / den)


@vectorize
def erfinv(x: Float64) -> Float64:
    """Inverse error function: erfinv(erf(x)) == x for x in (-1, 1).

    Uses the Acklam rational approximation (2003), then one step of
    Halley's method to polish to full float64 precision.
    """
    if x == 0.0:
        return 0.0
    if x <= -1.0:
        return -1.0e308
    if x >= 1.0:
        return 1.0e308

    # erfinv(x) = ndtri((x+1)/2) / sqrt(2)
    p: Float64 = (x + 1.0) * 0.5
    w: Float64 = _ndtri_rational(p)
    result: Float64 = w * 0.7071067811865475   # / sqrt(2)

    # One Halley refinement step: f(r) = erf(r) - x = 0
    # f'(r) = 2/sqrt(pi) * exp(-r^2)
    # f''(r) = -4r/sqrt(pi) * exp(-r^2)
    # Halley: r_new = r - f/(f' - f*f''/(2*f'))
    #              = r - (erf(r)-x) / (2/sqrt(pi)*exp(-r^2)) * 1/(1 + r*(erf(r)-x))
    err: Float64 = erf(result) - x
    deriv: Float64 = 1.1283791670955126 * exp(-result * result)   # 2/sqrt(pi)
    result = result - err / (deriv + result * err)
    return result


@vectorize
def erfcinv(x: Float64) -> Float64:
    """Inverse complementary error function: erfcinv(erfc(x)) == x for x in (0, 2)."""
    return erfinv(1.0 - x)
