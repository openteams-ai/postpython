"""ppspecial — POST Python reimplementation of scipy.special.

Each function is a @vectorize kernel, written in fully-typed POST Python.
The POST Python compiler lowers them to native shared-library code; in
interpreted mode they run via the pure-Python broadcast loop.

Function families implemented
------------------------------
Error functions  (_erf)    : erf, erfc, erfinv, erfcinv
Gamma functions  (_gamma)  : lgamma / gammaln, gamma, digamma, polygamma,
                              beta, lbeta
Bessel functions (_bessel) : j0, j1, y0, y1, i0, i1, k0, k1
Statistical      (_stats)  : ndtr, log_ndtr, ndtri, expit / sigmoid,
                              log_expit, logit, xlogy, xlog1py

Roadmap (not yet implemented)
------------------------------
Hypergeometric   : hyp1f1, hyp2f1, hyp0f1
Orthogonal poly  : eval_legendre, eval_hermite, eval_chebyshev*, eval_laguerre
Spherical harm.  : sph_harm
Elliptic integrals: ellipk, ellipe, ellipj
Airy functions   : airy, airye
Statistical dists: betainc, betaincinv, stdtr, stdtrit, chdtr, chdtri, ...
"""

from ppspecial._erf import (
    erf,
    erfc,
    erfinv,
    erfcinv,
)

from ppspecial._gamma import (
    lgamma,
    gammaln,   # alias for lgamma
    gamma,
    digamma,
    polygamma,
    beta,
    lbeta,
)

from ppspecial._bessel import (
    j0, j1,
    y0, y1,
    i0, i1,
    k0, k1,
)

from ppspecial._stats import (
    ndtr,
    log_ndtr,
    ndtri,
    expit,
    sigmoid,   # alias for expit
    log_expit,
    logit,
    xlogy,
    xlog1py,
)

__all__ = [
    # erf
    "erf", "erfc", "erfinv", "erfcinv",
    # gamma
    "lgamma", "gammaln", "gamma", "digamma", "polygamma", "beta", "lbeta",
    # bessel
    "j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1",
    # stats
    "ndtr", "log_ndtr", "ndtri",
    "expit", "sigmoid", "log_expit", "logit",
    "xlogy", "xlog1py",
]
