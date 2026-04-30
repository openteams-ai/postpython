"""Runtime tests for the interpreted @gufunc broadcast loop."""

import math

import pytest

from postpython.gufunc import gufunc
from postpython.math import exp
from postyp import Array, Bool, Float64

np = pytest.importorskip("numpy")


@gufunc("()->()")
def square(x: Float64) -> Float64:
    return x * x


@gufunc("(),(),()->()")
def gaussian(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    z: Float64 = (x - mu) / sigma
    return exp(-0.5 * z * z) / (sigma * 2.5066282746310002)


@gufunc("()->()")
def is_positive(x: Float64) -> Bool:
    return x > 0.0


@gufunc("(n)->(n)")
def double_vector(x: Array[Float64], out: Array[Float64]) -> None:
    for i in range(len(x)):
        out[i] = 2.0 * x[i]


@gufunc("()->()")
def scalar_write_output(x: Float64, out: Float64) -> None:
    out = x + 1.0


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


def test_scalar_gufunc_preserves_bool_output_dtype():
    xs = np.array([-1.0, 0.0, 2.0])

    result = is_positive(xs)

    assert result.dtype == np.bool_
    assert np.array_equal(result, np.array([False, False, True]))


def test_return_style_output_argument_is_filled():
    xs = np.array([-2.0, -1.0, 0.0, 3.0])
    out = np.empty_like(xs)

    result = square(xs, out)

    assert result is out
    assert np.allclose(out, xs * xs)


def test_core_output_argument_is_filled():
    xs = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = np.zeros_like(xs)

    result = double_vector(xs, out)

    assert result is out
    assert np.allclose(out, 2.0 * xs)


def test_scalar_write_through_output_argument_fails_loudly():
    xs = np.array([1.0, 2.0])
    out = np.zeros_like(xs)

    with pytest.raises(TypeError, match="scalar write-through"):
        scalar_write_output(xs, out)
