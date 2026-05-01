"""Tests for ppspecial Bessel functions.

Reference values from scipy.special (pre-computed to avoid scipy dependency).
"""

import pytest
from ppspecial import j0, j1, y0, y1, i0, i1, k0, k1


def close(a, b, rtol=1e-4):
    if abs(b) < 1e-300:
        return abs(a - b) < 1e-10
    return abs(a - b) / abs(b) < rtol


# Reference values: scipy.special evaluated at test points
_J0_REFS = [
    (0.0,   1.0),
    (1.0,   0.7651976866),
    (2.0,   0.2238907791),
    (5.0,  -0.1775967713),
    (10.0, -0.2459357645),
]

_J1_REFS = [
    (0.0,   0.0),
    (1.0,   0.4400505857),
    (2.0,   0.5767248078),
    (5.0,  -0.3275791376),
    (10.0,  0.0434727462),
]

_I0_REFS = [
    (0.0,  1.0),
    (1.0,  1.2660658778),
    (2.0,  2.2795853023),
    (5.0,  27.2398718136),
]

_I1_REFS = [
    (0.0,  0.0),
    (1.0,  0.5651591040),
    (2.0,  1.5906368546),
    (5.0,  24.3356043100),
]

_K0_REFS = [
    (0.1,  2.4270690247),
    (1.0,  0.4210244382),
    (2.0,  0.1138938727),
    (5.0,  0.003691009966),   # scipy.special.k0(5)
]

_K1_REFS = [
    (0.1,  9.853844194),      # scipy.special.k1(0.1)
    (1.0,  0.6019072302),
    (2.0,  0.1398658818),
    (5.0,  0.0040446134),
]


@pytest.mark.parametrize("x, expected", _J0_REFS)
def test_j0(x, expected):
    assert close(j0(x), expected)


@pytest.mark.parametrize("x, expected", _J1_REFS)
def test_j1(x, expected):
    assert close(j1(x), expected)


def test_j0_antisymmetry():
    # J0 is even
    for x in [1.0, 3.0, 7.0]:
        assert close(j0(-x), j0(x))


def test_j1_antisymmetry():
    # J1 is odd
    for x in [1.0, 3.0, 7.0]:
        assert close(j1(-x), -j1(x))


@pytest.mark.parametrize("x, expected", _I0_REFS)
def test_i0(x, expected):
    assert close(i0(x), expected)


@pytest.mark.parametrize("x, expected", _I1_REFS)
def test_i1(x, expected):
    assert close(i1(x), expected)


def test_i0_even():
    for x in [1.0, 2.0, 4.0]:
        assert close(i0(-x), i0(x))


def test_i1_odd():
    for x in [1.0, 2.0, 4.0]:
        assert close(i1(-x), -i1(x))


@pytest.mark.parametrize("x, expected", _K0_REFS)
def test_k0(x, expected):
    assert close(k0(x), expected)


@pytest.mark.parametrize("x, expected", _K1_REFS)
def test_k1(x, expected):
    assert close(k1(x), expected)


def test_k0_positive_only():
    assert k0(0.0) > 1e100
    assert k0(-1.0) > 1e100


# Wronskian: J0·Y1 - J1·Y0 = 2/(π·x)
def test_wronskian_j0_j1():
    import math
    for x in [1.0, 2.0, 5.0, 10.0]:
        w = j0(x) * y1(x) - j1(x) * y0(x)
        expected = -2.0 / (math.pi * x)
        assert close(w, expected, rtol=1e-4)


# Recurrence: I0·K1 + I1·K0 = 1/x
def test_wronskian_i0_k0():
    for x in [0.5, 1.0, 2.0, 5.0]:
        w = i0(x) * k1(x) + i1(x) * k0(x)
        assert close(w, 1.0 / x, rtol=1e-4)
