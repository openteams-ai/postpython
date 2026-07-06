"""ppdemo scalar kernels — a private POST translation unit.

Underscore-prefixed helpers get internal linkage (spec §9.1); the public
functions are re-exported through the package ``__init__`` manifest and
become part of the artifact's ``pp_*`` C ABI. The ``@vectorize`` kernel
lives here rather than in ``__init__.py`` because a namespace manifest
defines no compiled functions (spec §9.1) — kernels belong in modules.
"""

from postyp import Float64
from postpyc import vectorize
from postpyc.math import exp


def _clip01(x: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def smoothstep(x: Float64) -> Float64:
    t: Float64 = _clip01(x)
    return t * t * (3.0 - 2.0 * t)


def mix(x: Float64, y: Float64, t: Float64) -> Float64:
    return x + (y - x) * _clip01(t)


@vectorize
def logistic(x: Float64) -> Float64:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z: Float64 = exp(x)
    return z / (1.0 + z)
