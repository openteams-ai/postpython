"""Runtime tests for the interpreted @gufunc broadcast loop."""

import math

import pytest

from postpython.gufunc import gufunc
from postpython.math import exp
from postyp import Float64

np = pytest.importorskip("numpy")


@gufunc("()->()")
def square(x: Float64) -> Float64:
    return x * x


@gufunc("(),(),()->()")
def gaussian(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    z: Float64 = (x - mu) / sigma
    return exp(-0.5 * z * z) / (sigma * 2.5066282746310002)


def test_unary_scalar_gufunc_broadcasts_over_array():
    xs = np.array([-2.0, -1.0, 0.0, 3.0])

    result = square(xs)

    assert np.allclose(result, xs * xs)


def test_ternary_scalar_gufunc_broadcasts_arrays_and_scalars():
    xs = np.linspace(-3.0, 3.0, 7)

    result = gaussian(xs, 0.0, 1.0)
    expected = np.exp(-0.5 * xs**2) / math.sqrt(2.0 * math.pi)

    assert np.allclose(result, expected)


def test_ternary_scalar_gufunc_broadcasts_mixed_shapes():
    xs = np.array([[-1.0], [0.0], [1.0]])
    mus = np.array([0.0, 1.0])

    result = gaussian(xs, mus, 1.0)
    expected = np.exp(-0.5 * ((xs - mus) ** 2)) / math.sqrt(2.0 * math.pi)

    assert result.shape == (3, 2)
    assert np.allclose(result, expected)
