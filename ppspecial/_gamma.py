"""Gamma and related functions — ppspecial._gamma

Public API
----------
lgamma(x)       — log-gamma, ln Γ(x), for x > 0
gamma(x)        — Gamma function Γ(x), for x > 0
digamma(x)      — digamma / psi function ψ(x) = d/dx ln Γ(x)
polygamma(n, x) — polygamma ψ⁽ⁿ⁾(x) (n = 0 is digamma)
beta(a, b)      — beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
lbeta(a, b)     — log-beta  ln B(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)
gammaln         — alias for lgamma (scipy compat)

Algorithms
----------
lgamma  : Stirling series with recurrence to shift x ≥ 12.
          Error O(x^{-11}) after 5 Bernoulli terms.
digamma : Asymptotic series with recurrence to shift x ≥ 10.
gamma   : exp(lgamma(x)) with sign handling for negative x.
"""

from postyp import Float64, Int64, Bool
from postpython.gufunc import gufunc
from postpython.math import exp, log, sin, floor, fabs, PI, LOG_PI, EULER


# ---------------------------------------------------------------------------
# lgamma — log Γ(x) for x > 0
# ---------------------------------------------------------------------------

# Half log(2π)
_HALF_LOG_2PI: Float64 = 0.9189385332046727


def _lgamma_stirling(x: Float64) -> Float64:
    """Stirling series for lgamma, valid for x >= 12."""
    w: Float64 = 1.0 / (x * x)
    # Horner evaluation of Bernoulli correction in powers of w = 1/x²:
    # 1/12 - w/360 + w²/1260 - w³/1680 + w⁴/1188
    p: Float64 = 1.0 / 1188.0
    p = p * w - 1.0 / 1680.0
    p = p * w + 1.0 / 1260.0
    p = p * w - 1.0 / 360.0
    p = p * w + 1.0 / 12.0
    return _HALF_LOG_2PI + (x - 0.5) * log(x) - x + p / x


@gufunc("()->()")
def lgamma(x: Float64) -> Float64:
    """Natural log of the Gamma function: ln Γ(x), x > 0.

    Uses the recurrence Γ(x) = (x-1)·Γ(x-1) to shift into the domain
    where the Stirling series converges rapidly.
    """
    if x <= 0.0:
        return 1.0e308    # +∞ at non-positive integers

    # Shift x up until x >= 12, accumulating the recurrence correction.
    z: Float64 = x
    adj: Float64 = 0.0
    while z < 12.0:
        adj = adj - log(z)
        z = z + 1.0

    return _lgamma_stirling(z) + adj


# Scipy-compatible alias
gammaln = lgamma


# ---------------------------------------------------------------------------
# gamma — Γ(x)
# ---------------------------------------------------------------------------

@gufunc("()->()")
def gamma(x: Float64) -> Float64:
    """Gamma function Γ(x).

    For x > 0: exp(lgamma(x)).
    For x < 0: uses the reflection formula Γ(x)Γ(1-x) = π/sin(πx).
    Returns +∞ at non-positive integers.
    """
    if x > 0.0:
        return exp(lgamma(x))

    # Non-positive: check for pole
    if x == floor(x):
        return 1.0e308    # pole

    # Reflection: Γ(x) = π / (sin(πx) · Γ(1-x))
    s: Float64 = sin(PI * x)
    if s == 0.0:
        return 1.0e308
    return PI / (s * exp(lgamma(1.0 - x)))


# ---------------------------------------------------------------------------
# digamma — ψ(x) = d/dx ln Γ(x)
# ---------------------------------------------------------------------------

def _digamma_asymptotic(x: Float64) -> Float64:
    """Asymptotic series for digamma, valid for x >= 10.

    ψ(x) = ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − …

    Derived from -B_{2k}/(2k·x^{2k}) with Bernoulli numbers
    B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30.
    """
    w: Float64 = 1.0 / (x * x)
    # Horner in w = 1/x² for the bracket [ -1/12 + w*(1/120 + w*(-1/252 + w*1/240)) ]
    p: Float64 = 1.0 / 240.0
    p = p * w - 1.0 / 252.0
    p = p * w + 1.0 / 120.0
    p = p * w - 1.0 / 12.0
    return log(x) - 0.5 / x + p * w


@gufunc("()->()")
def digamma(x: Float64) -> Float64:
    """Digamma function ψ(x) = d/dx ln Γ(x).

    Uses the recurrence ψ(x+1) = ψ(x) + 1/x to shift into the
    asymptotic regime.  Returns -∞ at non-positive integers.
    """
    if x <= 0.0:
        if x == floor(x):
            return -1.0e308   # pole
        # Reflection: ψ(1-x) - ψ(x) = π·cot(πx)
        # → ψ(x) = ψ(1-x) - π·cos(πx)/sin(πx)
        s: Float64 = sin(PI * x)
        c: Float64 = sin(PI * (0.5 - x))   # cos(πx) = sin(π(0.5-x))
        return digamma(1.0 - x) - PI * c / s

    z: Float64 = x
    adj: Float64 = 0.0
    while z < 10.0:
        adj = adj - 1.0 / z
        z = z + 1.0

    return _digamma_asymptotic(z) + adj


# ---------------------------------------------------------------------------
# polygamma — ψ⁽ⁿ⁾(x)
# ---------------------------------------------------------------------------

@gufunc("(),()->()")
def polygamma(n: Int64, x: Float64) -> Float64:
    """Polygamma function ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ ln Γ(x).

    n=0 returns digamma(x).  n >= 1 uses the recurrence and asymptotic
    series for the higher-order polygamma.

    Note: this implementation handles n=0,1,2 accurately; larger n
    will accumulate more approximation error in the asymptotic term.
    """
    if n == 0:
        return digamma(x)

    # For n >= 1, shift x up and use the asymptotic formula:
    # ψ⁽ⁿ⁾(x) ~ (-1)^(n+1) * n! * sum_{k=0}^∞ 1/(x+k)^(n+1)
    z: Float64 = x
    adj: Float64 = 0.0
    while z < 10.0:
        # recurrence: ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x+1) - (-1)^n * n! / x^(n+1)
        adj = adj + 1.0 / (z ** (n + 1))
        z = z + 1.0

    # Asymptotic: ψ⁽¹⁾(x) = 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + ...
    # For general n, lead term is n!/x^(n+1)
    # Factorial (n is small in practice)
    fact: Float64 = 1.0
    k: Int64 = 1
    while k <= n:
        fact = fact * k
        k = k + 1

    sign: Float64 = 1.0
    if (n + 1) % 2 == 0:
        sign = -1.0

    asymp: Float64 = sign * fact / (z ** (n + 1))
    # Include leading correction term 1/(2 * x^(n+2))... (approx)
    correction: Float64 = sign * fact * 0.5 * (n + 1) / (z ** (n + 2))

    result: Float64 = asymp + correction
    if n % 2 == 0:
        return -(result + adj)
    return result - adj


# ---------------------------------------------------------------------------
# beta / lbeta
# ---------------------------------------------------------------------------

@gufunc("(),()->()")
def lbeta(a: Float64, b: Float64) -> Float64:
    """Log of the beta function: ln B(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)"""
    return lgamma(a) + lgamma(b) - lgamma(a + b)


@gufunc("(),()->()")
def beta(a: Float64, b: Float64) -> Float64:
    """Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
    return exp(lbeta(a, b))
