"""Gaussian kernel functions — POST Python example source.

Demonstrates three @vectorize patterns:

  square(x)                  — elementwise x²
  gaussian(x, mu, sigma)     — normal PDF value at x
  relu(x)                    — rectified linear unit (conditional return)

All functions are valid POST Python: every parameter and local variable is
annotated, no reflection, no closures, no *args / **kwargs.
"""

from postyp import Float64
from postpython import vectorize
from postpython.math import exp


@vectorize
def square(x: Float64) -> Float64:
    return x * x


@vectorize
def gaussian(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    z: Float64 = (x - mu) / sigma
    return exp(-0.5 * z * z) / (sigma * 2.5066282746310002)


@vectorize
def relu(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0
