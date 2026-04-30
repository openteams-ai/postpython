"""Tests for ppspecial gamma family."""

import math
import pytest
from ppspecial import lgamma, gamma, digamma, beta, lbeta


def close(a, b, rtol=1e-6):
    if abs(b) < 1e-300:
        return abs(a - b) < 1e-12
    return abs(a - b) / abs(b) < rtol


class TestLgamma:
    def test_lgamma_one(self):
        assert close(lgamma(1.0), 0.0)

    def test_lgamma_two(self):
        assert close(lgamma(2.0), 0.0)

    def test_lgamma_half(self):
        # lgamma(0.5) = 0.5 * log(pi)
        assert close(lgamma(0.5), 0.5 * math.log(math.pi), rtol=1e-5)

    def test_lgamma_integer(self):
        # lgamma(n) = log((n-1)!) for integer n
        for n in [3, 5, 8, 10]:
            expected = math.log(math.factorial(n - 1))
            assert close(lgamma(float(n)), expected, rtol=1e-5)

    def test_lgamma_large(self):
        # Stirling: lgamma(100) ≈ 359.134
        assert close(lgamma(100.0), math.lgamma(100.0), rtol=1e-8)


class TestGamma:
    def test_gamma_one(self):
        assert close(gamma(1.0), 1.0)

    def test_gamma_half(self):
        # Γ(0.5) = √π
        assert close(gamma(0.5), math.sqrt(math.pi), rtol=1e-5)

    def test_gamma_integer_factorial(self):
        for n in [1, 2, 3, 4, 5, 6]:
            assert close(gamma(float(n)), math.factorial(n - 1), rtol=1e-5)

    def test_gamma_recurrence(self):
        # Γ(x+1) = x·Γ(x)
        for x in [0.3, 1.7, 2.5]:
            assert close(gamma(x + 1.0), x * gamma(x), rtol=1e-5)


class TestDigamma:
    def test_digamma_one(self):
        # ψ(1) = -γ  (Euler–Mascheroni constant)
        assert close(digamma(1.0), -0.5772156649, rtol=1e-5)

    def test_digamma_two(self):
        # ψ(2) = 1 - γ
        assert close(digamma(2.0), 1.0 - 0.5772156649, rtol=1e-5)

    def test_digamma_recurrence(self):
        # ψ(x+1) = ψ(x) + 1/x
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert close(digamma(x + 1.0), digamma(x) + 1.0 / x, rtol=1e-5)


class TestBeta:
    def test_beta_symmetric(self):
        for a, b in [(1.0, 2.0), (0.5, 3.0), (2.0, 5.0)]:
            assert close(beta(a, b), beta(b, a), rtol=1e-10)

    def test_beta_one_one(self):
        assert close(beta(1.0, 1.0), 1.0)

    def test_beta_half_half(self):
        # B(0.5, 0.5) = π
        assert close(beta(0.5, 0.5), math.pi, rtol=1e-5)

    def test_lbeta_matches_log_beta(self):
        for a, b in [(1.0, 2.0), (3.0, 4.0)]:
            assert close(lbeta(a, b), math.log(beta(a, b)), rtol=1e-10)
