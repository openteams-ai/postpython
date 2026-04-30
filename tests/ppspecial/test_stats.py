"""Tests for ppspecial statistical functions."""

import math
import pytest
from ppspecial import ndtr, log_ndtr, ndtri, expit, logit, xlogy, xlog1py, log_expit


def close(a, b, rtol=1e-6):
    if abs(b) < 1e-300:
        return abs(a - b) < 1e-12
    return abs(a - b) / abs(b) < rtol


class TestNdtr:
    def test_ndtr_zero(self):
        assert close(ndtr(0.0), 0.5)

    def test_ndtr_one(self):
        # Φ(1) ≈ 0.8413447461
        assert close(ndtr(1.0), 0.8413447461, rtol=1e-6)

    def test_ndtr_neg_one(self):
        assert close(ndtr(-1.0), 1.0 - ndtr(1.0), rtol=1e-10)

    def test_ndtr_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert close(ndtr(x) + ndtr(-x), 1.0, rtol=1e-10)

    def test_ndtr_large_pos(self):
        assert ndtr(8.0) > 1.0 - 1e-14

    def test_ndtr_large_neg(self):
        assert ndtr(-8.0) < 1e-14


class TestLogNdtr:
    def test_log_ndtr_zero(self):
        assert close(log_ndtr(0.0), math.log(0.5))

    def test_log_ndtr_matches_log_ndtr(self):
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert close(log_ndtr(x), math.log(ndtr(x)), rtol=1e-8)

    def test_log_ndtr_large_neg_finite(self):
        # Should not return -inf for moderate negative x
        v = log_ndtr(-10.0)
        assert v > -1e308
        assert v < 0.0


class TestNdtri:
    def test_ndtri_half(self):
        assert close(ndtri(0.5), 0.0, rtol=1e-10)

    def test_ndtri_roundtrip(self):
        for x in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert close(ndtr(ndtri(x)), x, rtol=1e-9)

    def test_ndtri_boundary(self):
        assert ndtri(0.0) < -1e200
        assert ndtri(1.0) > 1e200


class TestExpit:
    def test_expit_zero(self):
        assert close(expit(0.0), 0.5)

    def test_expit_large_pos(self):
        assert close(expit(100.0), 1.0, rtol=1e-10)

    def test_expit_large_neg(self):
        assert close(expit(-100.0), 0.0, rtol=1e-10)

    def test_expit_logit_roundtrip(self):
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert close(expit(logit(x)), x, rtol=1e-10)


class TestLogExpit:
    def test_log_expit_zero(self):
        assert close(log_expit(0.0), math.log(0.5))

    def test_log_expit_matches_log_expit(self):
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert close(log_expit(x), math.log(expit(x)), rtol=1e-9)

    def test_log_expit_large_neg_stable(self):
        # Should ≈ x for very negative x
        v = log_expit(-50.0)
        assert close(v, -50.0, rtol=1e-6)


class TestXlogy:
    def test_xlogy_zero_times_zero(self):
        assert xlogy(0.0, 0.0) == 0.0

    def test_xlogy_zero_times_anything(self):
        assert xlogy(0.0, 1.0) == 0.0
        assert xlogy(0.0, 100.0) == 0.0

    def test_xlogy_normal(self):
        assert close(xlogy(2.0, math.e), 2.0)

    def test_xlog1py_small_y(self):
        # x*log1p(y) ≈ x*y for small y
        for y in [1e-10, 1e-8, 1e-6]:
            assert close(xlog1py(1.0, y), math.log1p(y), rtol=1e-10)


class TestLogit:
    def test_logit_half(self):
        assert close(logit(0.5), 0.0)

    def test_logit_boundaries(self):
        assert logit(0.0) < -1e200
        assert logit(1.0) > 1e200

    def test_logit_antisymmetric(self):
        for p in [0.1, 0.3, 0.4]:
            assert close(logit(p), -logit(1.0 - p), rtol=1e-10)
