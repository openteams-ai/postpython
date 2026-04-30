"""Statistical distribution functions — ppspecial._stats

Public API
----------
ndtr(x)       — standard normal CDF  Φ(x) = P(Z ≤ x)
log_ndtr(x)   — log of the normal CDF, numerically stable for large |x|
ndtri(x)      — inverse normal CDF (probit): ndtri(ndtr(x)) == x
expit(x)      — logistic sigmoid 1/(1+e^{-x})  (alias: sigmoid)
logit(x)      — log-odds ln(x/(1-x)), inverse of expit
xlogy(x, y)   — x*ln(y) with the convention 0*ln(0) = 0
xlog1py(x, y) — x*ln(1+y), 0*ln(1+0) = 0
log_expit(x)  — ln(expit(x)), numerically stable

Algorithms
----------
ndtr   : erfc(-x/√2) / 2, inheriting erfc precision.
ndtri  : Acklam rational approximation + Halley polish (via erfinv).
expit  : numerically stable two-branch implementation.
"""

from postyp import Float64, Bool
from postpython.gufunc import gufunc
from postpython.math import exp, log, log1p, sqrt, fabs, isfinite
from ppspecial._erf import erfc, erfinv


_SQRT2:     Float64 = 1.4142135623730951   # √2
_SQRT2PI:   Float64 = 2.5066282746310002   # √(2π)
_LOG_SQRT2PI: Float64 = 0.9189385332046727 # ln(√(2π))


# ---------------------------------------------------------------------------
# ndtr — Φ(x)
# ---------------------------------------------------------------------------

@gufunc("()->()")
def ndtr(x: Float64) -> Float64:
    """Standard normal CDF: Φ(x) = P(Z ≤ x) = erfc(-x/√2) / 2."""
    return erfc(-x / _SQRT2) * 0.5


# ---------------------------------------------------------------------------
# log_ndtr — ln Φ(x)
# ---------------------------------------------------------------------------

@gufunc("()->()")
def log_ndtr(x: Float64) -> Float64:
    """Log of the standard normal CDF: ln Φ(x).

    Uses the log-erfc path for x < -1 to stay numerically accurate
    when Φ(x) is very small.
    """
    if x > 6.0:
        # Φ(x) ≈ 1 — use log1p to preserve precision
        tail: Float64 = -0.5 * x * x - _LOG_SQRT2PI - log(fabs(x))
        return log1p(-exp(tail))

    if x > -20.0:
        p: Float64 = ndtr(x)
        if p > 0.0:
            return log(p)
        return -1.0e308

    # Deep left tail: ln Φ(x) ≈ ln(φ(x)/|x|) where φ is the normal PDF
    # ln Φ(x) ≈ -x²/2 - ln(√(2π)) - ln|x| + series correction
    t: Float64 = 1.0 / (x * x)
    # Asymptotic: -x²/2 - ln(√2π|x|) - 1/(2x²) + 3/(4x⁴) - ...
    series: Float64 = 1.0 - t * (1.0 - t * (3.0 - t * 15.0))
    return -0.5 * x * x - _LOG_SQRT2PI - log(-x) + log(series)


# ---------------------------------------------------------------------------
# ndtri — inverse normal CDF
# ---------------------------------------------------------------------------

@gufunc("()->()")
def ndtri(x: Float64) -> Float64:
    """Inverse standard normal CDF (probit function): ndtri(ndtr(z)) == z.

    Implemented as √2 · erfinv(2x - 1), inheriting erfinv's Halley polish.
    """
    if x <= 0.0:
        return -1.0e308
    if x >= 1.0:
        return 1.0e308
    return _SQRT2 * erfinv(2.0 * x - 1.0)


# ---------------------------------------------------------------------------
# expit / logit
# ---------------------------------------------------------------------------

@gufunc("()->()")
def expit(x: Float64) -> Float64:
    """Logistic sigmoid: expit(x) = 1 / (1 + e^{-x}).

    Numerically stable for large |x|.
    """
    if x >= 0.0:
        z: Float64 = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


# Scipy compat alias
sigmoid = expit


@gufunc("()->()")
def log_expit(x: Float64) -> Float64:
    """ln(expit(x)) = ln(1 / (1 + e^{-x})), numerically stable.

    For x >> 0: ≈ -e^{-x} (uses log1p).
    For x << 0: ≈ x.
    """
    if x >= 0.0:
        return -log1p(exp(-x))
    return x - log1p(exp(x))


@gufunc("()->()")
def logit(x: Float64) -> Float64:
    """Log-odds: logit(x) = ln(x / (1 - x)), inverse of expit.

    Returns -∞ for x ≤ 0, +∞ for x ≥ 1.
    """
    if x <= 0.0:
        return -1.0e308
    if x >= 1.0:
        return 1.0e308
    return log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# xlogy / xlog1py — safe products with log
# ---------------------------------------------------------------------------

@gufunc("(),()->()")
def xlogy(x: Float64, y: Float64) -> Float64:
    """x · ln(y), with the convention that 0 · ln(0) = 0.

    Used in entropy and cross-entropy calculations.
    """
    if x == 0.0:
        return 0.0
    if y <= 0.0:
        return -1.0e308
    return x * log(y)


@gufunc("(),()->()")
def xlog1py(x: Float64, y: Float64) -> Float64:
    """x · ln(1 + y), with the convention that 0 · ln(1 + 0) = 0.

    More accurate than xlogy(x, 1+y) when |y| << 1.
    """
    if x == 0.0:
        return 0.0
    if y <= -1.0:
        return -1.0e308
    return x * log1p(y)
