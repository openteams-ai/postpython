"""Tests for ppspecial erf family — validated against known values."""

import math
import pytest
from ppspecial import erf, erfc, erfinv, erfcinv


def close(a, b, rtol=1e-6):
    if abs(b) < 1e-300:
        return abs(a - b) < 1e-12
    return abs(a - b) / abs(b) < rtol


class TestErfc:
    def test_erfc_zero(self):
        assert close(erfc(0.0), 1.0)

    def test_erfc_positive(self):
        assert close(erfc(1.0), 0.15729920705, rtol=1e-6)

    def test_erfc_negative(self):
        assert close(erfc(-1.0), 1.84270079295, rtol=1e-6)

    def test_erfc_large_positive(self):
        # Should go to 0
        assert erfc(8.0) < 1e-28

    def test_erfc_large_negative(self):
        # Should go to 2
        assert abs(erfc(-8.0) - 2.0) < 1e-28

    def test_erf_plus_erfc_equals_one(self):
        for x in [-3.0, -1.0, 0.0, 0.5, 1.0, 3.0]:
            assert close(erf(x) + erfc(x), 1.0, rtol=1e-12)


class TestErf:
    def test_erf_zero(self):
        assert erf(0.0) == 0.0

    def test_erf_one(self):
        assert close(erf(1.0), 0.84270079295, rtol=1e-6)

    def test_erf_antisymmetric(self):
        for x in [0.5, 1.0, 2.0]:
            assert close(erf(-x), -erf(x), rtol=1e-12)

    def test_erf_limits(self):
        assert close(erf(5.0), 1.0, rtol=1e-10)


class TestErfinv:
    def test_erfinv_zero(self):
        assert close(erfinv(0.0), 0.0, rtol=1e-10)

    def test_erfinv_roundtrip(self):
        for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]:
            assert close(erf(erfinv(x)), x, rtol=1e-10)

    def test_erfinv_antisymmetric(self):
        for x in [0.3, 0.7]:
            assert close(erfinv(-x), -erfinv(x), rtol=1e-10)

    def test_erfinv_boundary(self):
        assert erfinv(-1.0) < -1e200
        assert erfinv(1.0) > 1e200


class TestErfcinv:
    def test_erfcinv_one(self):
        assert close(erfcinv(1.0), 0.0, rtol=1e-10)

    def test_erfcinv_roundtrip(self):
        for x in [0.1, 0.5, 1.0, 1.5, 1.9]:
            assert close(erfc(erfcinv(x)), x, rtol=1e-9)
