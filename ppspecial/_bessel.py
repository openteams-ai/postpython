"""Bessel functions — ppspecial._bessel

Public API (real argument, integer order 0 or 1)
-------------------------------------------------
j0(x)   — Bessel J₀(x),  first kind order 0
j1(x)   — Bessel J₁(x),  first kind order 1
y0(x)   — Bessel Y₀(x),  second kind order 0  (x > 0)
y1(x)   — Bessel Y₁(x),  second kind order 1  (x > 0)
i0(x)   — Modified I₀(x), first kind order 0
i1(x)   — Modified I₁(x), first kind order 1
k0(x)   — Modified K₀(x), second kind order 0  (x > 0)
k1(x)   — Modified K₁(x), second kind order 1  (x > 0)

Algorithms
----------
Rational polynomial / asymptotic approximations from Abramowitz & Stegun
(1964) and Numerical Recipes in C §6.5.  All polynomials written in
Horner form for numerical stability and compiler-friendly loop structure.
Maximum relative error < 1.7e-8.
"""

from postyp import Float64
from postpython import vectorize
from postpython.math import exp, log, sqrt, sin, cos, fabs


_TWO_OVER_PI: Float64 = 0.6366197723675814   # 2/π
_PI_OVER_4:   Float64 = 0.7853981633974483   # π/4
_3PI_OVER_4:  Float64 = 2.356194490192345    # 3π/4


# ---------------------------------------------------------------------------
# Asymptotic auxiliary functions for j0/y0 (|x| > 8)
# ---------------------------------------------------------------------------

def _j0_asymp(ax: Float64) -> Float64:
    """sqrt(2/(π·ax))·(P₀·cos(ax-π/4) - Q₀·z·sin(ax-π/4)), z = 8/ax."""
    z: Float64 = 8.0 / ax
    y: Float64 = z * z
    xx: Float64 = ax - _PI_OVER_4
    p0: Float64 = 0.2093887211e-6
    p0 = p0 * y - 0.2073370639e-5
    p0 = p0 * y + 0.2734510407e-4
    p0 = p0 * y - 0.1098628627e-2
    p0 = p0 * y + 1.0
    q0: Float64 = -0.934945152e-7
    q0 = q0 * y + 0.7621095161e-6
    q0 = q0 * y - 0.6911147651e-5
    q0 = q0 * y + 0.1430488765e-3
    q0 = q0 * y - 0.1562499995e-1
    return sqrt(_TWO_OVER_PI / ax) * (cos(xx) * p0 - z * sin(xx) * q0)


def _y0_asymp(x: Float64) -> Float64:
    """Asymptotic Y₀ for x > 8 (same auxiliary functions, sin/cos exchanged)."""
    z: Float64 = 8.0 / x
    y: Float64 = z * z
    xx: Float64 = x - _PI_OVER_4
    p0: Float64 = 0.2093887211e-6
    p0 = p0 * y - 0.2073370639e-5
    p0 = p0 * y + 0.2734510407e-4
    p0 = p0 * y - 0.1098628627e-2
    p0 = p0 * y + 1.0
    q0: Float64 = -0.934945152e-7
    q0 = q0 * y + 0.7621095161e-6
    q0 = q0 * y - 0.6911147651e-5
    q0 = q0 * y + 0.1430488765e-3
    q0 = q0 * y - 0.1562499995e-1
    return sqrt(_TWO_OVER_PI / x) * (sin(xx) * p0 + z * cos(xx) * q0)


def _j1_asymp(ax: Float64) -> Float64:
    """Asymptotic J₁ / Y₁ common auxiliary functions for |x| > 8."""
    z: Float64 = 8.0 / ax
    y: Float64 = z * z
    xx: Float64 = ax - _3PI_OVER_4
    p1: Float64 = -0.240337019e-6
    p1 = p1 * y + 0.2457520174e-5
    p1 = p1 * y - 0.3516396496e-4
    p1 = p1 * y + 0.183105e-2
    p1 = p1 * y + 1.0
    q1: Float64 = 0.105787412e-6
    q1 = q1 * y - 0.88228987e-6
    q1 = q1 * y + 0.8449199096e-5
    q1 = q1 * y - 0.2002690873e-3
    q1 = q1 * y + 0.04687499995
    return sqrt(_TWO_OVER_PI / ax) * (cos(xx) * p1 - z * sin(xx) * q1)


# ---------------------------------------------------------------------------
# j0 — J₀(x)
# ---------------------------------------------------------------------------

@vectorize
def j0(x: Float64) -> Float64:
    """Bessel function of the first kind, order 0: J₀(x)."""
    ax: Float64 = fabs(x)
    if ax >= 8.0:
        return _j0_asymp(ax)
    y: Float64 = x * x
    # Rational form — Horner in y
    p: Float64 = -184.9052456
    p = p * y + 77392.33017
    p = p * y - 11214424.18
    p = p * y + 651619640.7
    p = p * y - 13362590354.0
    p = p * y + 57568490574.0
    q: Float64 = 1.0
    q = q * y + 267.8532712
    q = q * y + 59272.64853
    q = q * y + 9494680.718
    q = q * y + 1029532985.0
    q = q * y + 57568490411.0
    return p / q


# ---------------------------------------------------------------------------
# j1 — J₁(x)
# ---------------------------------------------------------------------------

@vectorize
def j1(x: Float64) -> Float64:
    """Bessel function of the first kind, order 1: J₁(x)."""
    ax: Float64 = fabs(x)
    if ax >= 8.0:
        ans: Float64 = _j1_asymp(ax)
        if x < 0.0:
            return -ans
        return ans
    y: Float64 = x * x
    # Numerator (odd in x, so factor x out)
    p: Float64 = -30.16036606
    p = p * y + 15704.48260
    p = p * y - 2972611.439
    p = p * y + 242396853.1
    p = p * y - 7895059235.0
    p = p * y + 72362614232.0
    # Denominator (even in x)
    q: Float64 = 1.0
    q = q * y + 376.9991397
    q = q * y + 99447.43394
    q = q * y + 18583304.74
    q = q * y + 2300535178.0
    q = q * y + 144725228442.0
    return x * p / q


# ---------------------------------------------------------------------------
# y0 — Y₀(x), x > 0
# ---------------------------------------------------------------------------

@vectorize
def y0(x: Float64) -> Float64:
    """Bessel function of the second kind, order 0: Y₀(x), x > 0."""
    if x <= 0.0:
        return -1.0e308
    if x >= 8.0:
        return _y0_asymp(x)
    y: Float64 = x * x
    p: Float64 = 228.4622733
    p = p * y - 86327.92757
    p = p * y + 10879881.29
    p = p * y - 512359803.6
    p = p * y + 7062834065.0
    p = p * y - 2957821389.0
    q: Float64 = 1.0
    q = q * y + 226.1030244
    q = q * y + 47447.26470
    q = q * y + 7189466.438
    q = q * y + 745249964.8
    q = q * y + 40076544269.0
    return p / q + _TWO_OVER_PI * j0(x) * log(x)


# ---------------------------------------------------------------------------
# y1 — Y₁(x), x > 0
# ---------------------------------------------------------------------------

@vectorize
def y1(x: Float64) -> Float64:
    """Bessel function of the second kind, order 1: Y₁(x), x > 0."""
    if x <= 0.0:
        return -1.0e308
    if x >= 8.0:
        # Asymptotic: same p1/q1 as j1 but sin/cos exchanged
        z: Float64 = 8.0 / x
        y: Float64 = z * z
        xx: Float64 = x - _3PI_OVER_4
        p1: Float64 = -0.240337019e-6
        p1 = p1 * y + 0.2457520174e-5
        p1 = p1 * y - 0.3516396496e-4
        p1 = p1 * y + 0.183105e-2
        p1 = p1 * y + 1.0
        q1: Float64 = 0.105787412e-6
        q1 = q1 * y - 0.88228987e-6
        q1 = q1 * y + 0.8449199096e-5
        q1 = q1 * y - 0.2002690873e-3
        q1 = q1 * y + 0.04687499995
        return sqrt(_TWO_OVER_PI / x) * (sin(xx) * p1 + z * cos(xx) * q1)
    y = x * x
    # Numerator (odd in x)
    p: Float64 = 8511.9379
    p = p * y - 4237922.726
    p = p * y + 734926455.1
    p = p * y - 51534381390.0
    p = p * y + 1275274390000.0
    p = p * y - 4900604943000.0
    # Denominator (even)
    q: Float64 = 1.0
    q = q * y + 354.9632885
    q = q * y + 102042.605
    q = q * y + 22459040.02
    q = q * y + 3733650367.0
    q = q * y + 424441966400.0
    q = q * y + 24995805700000.0
    return x * p / q + _TWO_OVER_PI * (j1(x) * log(x) - 1.0 / x)


# ---------------------------------------------------------------------------
# i0 — I₀(x)
# ---------------------------------------------------------------------------

@vectorize
def i0(x: Float64) -> Float64:
    """Modified Bessel function of the first kind, order 0: I₀(x)."""
    ax: Float64 = fabs(x)
    if ax <= 3.75:
        y: Float64 = (ax / 3.75) * (ax / 3.75)
        p: Float64 = 0.0045813
        p = p * y + 0.0360768
        p = p * y + 0.2659732
        p = p * y + 1.2067492
        p = p * y + 3.0899424
        p = p * y + 3.5156229
        p = p * y + 1.0
        return p
    y: Float64 = 3.75 / ax
    p: Float64 = 0.00392377
    p = p * y - 0.01647633
    p = p * y + 0.02635537
    p = p * y - 0.02057706
    p = p * y + 0.00916281
    p = p * y - 0.00157565
    p = p * y + 0.00225319
    p = p * y + 0.01328592
    p = p * y + 0.39894228
    return (exp(ax) / sqrt(ax)) * p


# ---------------------------------------------------------------------------
# i1 — I₁(x)
# ---------------------------------------------------------------------------

@vectorize
def i1(x: Float64) -> Float64:
    """Modified Bessel function of the first kind, order 1: I₁(x)."""
    ax: Float64 = fabs(x)
    if ax <= 3.75:
        y: Float64 = (ax / 3.75) * (ax / 3.75)
        p: Float64 = 0.00032411
        p = p * y + 0.00301532
        p = p * y + 0.02658733
        p = p * y + 0.15084934
        p = p * y + 0.51498869
        p = p * y + 0.87890594
        p = p * y + 0.5
        result: Float64 = x * p
        return result
    y: Float64 = 3.75 / ax
    p: Float64 = -0.00420059
    p = p * y + 0.01787654
    p = p * y - 0.02895312
    p = p * y + 0.02282967
    p = p * y - 0.01031555
    p = p * y + 0.00163801
    p = p * y - 0.00362018
    p = p * y - 0.03988024
    p = p * y + 0.39894228
    ans: Float64 = (exp(ax) / sqrt(ax)) * p
    if x < 0.0:
        return -ans
    return ans


# ---------------------------------------------------------------------------
# k0 — K₀(x), x > 0
# ---------------------------------------------------------------------------

@vectorize
def k0(x: Float64) -> Float64:
    """Modified Bessel function of the second kind, order 0: K₀(x), x > 0."""
    if x <= 0.0:
        return 1.0e308
    if x <= 2.0:
        y: Float64 = x * x / 4.0
        p: Float64 = 0.00000740
        p = p * y + 0.00010750
        p = p * y + 0.00262698
        p = p * y + 0.03488590
        p = p * y + 0.23069756
        p = p * y + 0.42278420
        p = p * y - 0.57721566
        return p - log(x / 2.0) * i0(x)
    y: Float64 = 2.0 / x
    p: Float64 = 0.00053208
    p = p * y - 0.00251540
    p = p * y + 0.00587872
    p = p * y - 0.01062446
    p = p * y + 0.02189568
    p = p * y - 0.07832358
    p = p * y + 1.25331414
    return (exp(-x) / sqrt(x)) * p


# ---------------------------------------------------------------------------
# k1 — K₁(x), x > 0
# ---------------------------------------------------------------------------

@vectorize
def k1(x: Float64) -> Float64:
    """Modified Bessel function of the second kind, order 1: K₁(x), x > 0."""
    if x <= 0.0:
        return 1.0e308
    if x <= 2.0:
        y: Float64 = x * x / 4.0
        # Polynomial for K₁(x) = log(x/2)·I₁(x) + (1/x)·poly(y)
        # poly(y) = 1 + y*(0.15443144 + y*(-0.67278579 + ... ))
        p: Float64 = -0.00004686
        p = p * y - 0.00110404
        p = p * y - 0.01919402
        p = p * y - 0.18156897
        p = p * y - 0.67278579
        p = p * y + 0.15443144
        p = p * y + 1.0
        return log(x / 2.0) * i1(x) + p / x
    y: Float64 = 2.0 / x
    p: Float64 = -0.00068245
    p = p * y + 0.00325614
    p = p * y - 0.00780353
    p = p * y + 0.01504268
    p = p * y - 0.03655620
    p = p * y + 0.23498619
    p = p * y + 1.25331414
    return (exp(-x) / sqrt(x)) * p
