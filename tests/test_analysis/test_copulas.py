"""Tests for the copula-based multivariate dependence module."""

import numpy as np
import pytest

from aquascope.analysis.copulas import (
    CopulaResult,
    JointProbability,
    compare_copulas,
    copula_density,
    copula_function,
    fit_copula,
    generate_copula_samples,
    generate_synthetic_data,
    joint_exceedance_probability,
    tail_dependence,
    to_pseudo_observations,
)


def _correlated_data(n: int = 200, rho: float = 0.7, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate bivariate normal data with known correlation."""
    rng = np.random.default_rng(seed)
    cov = [[1, rho], [rho, 1]]
    data = rng.multivariate_normal([0, 0], cov, size=n)
    return data[:, 0], data[:, 1]


class TestPseudoObservations:
    def test_to_pseudo_observations_range(self):
        x, y = _correlated_data()
        u, v = to_pseudo_observations(x, y)
        assert np.all(u > 0) and np.all(u < 1)
        assert np.all(v > 0) and np.all(v < 1)

    def test_to_pseudo_observations_correct_ranking(self):
        x = np.array([10.0, 30.0, 20.0, 50.0, 40.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        u, v = to_pseudo_observations(x, y)
        # x = 10 has rank 1 → u = 1/11 ≈ 0.0909
        assert abs(u[0] - 1.0 / 11.0) < 1e-10
        # x = 30 has rank 3 → u = 3/11
        assert abs(u[1] - 3.0 / 11.0) < 1e-10


class TestFitCopulas:
    def test_fit_gaussian_copula(self):
        x, y = _correlated_data(n=300, rho=0.6, seed=10)
        u, v = to_pseudo_observations(x, y)
        result = fit_copula(u, v, family="gaussian")
        assert result.family == "gaussian"
        assert 0.3 < result.parameter < 0.9

    def test_fit_clayton_copula(self):
        x, y = _correlated_data(n=300, rho=0.5, seed=20)
        u, v = to_pseudo_observations(x, y)
        result = fit_copula(u, v, family="clayton")
        assert result.family == "clayton"
        assert result.parameter > 0

    def test_fit_gumbel_copula(self):
        x, y = _correlated_data(n=300, rho=0.5, seed=30)
        u, v = to_pseudo_observations(x, y)
        result = fit_copula(u, v, family="gumbel")
        assert result.family == "gumbel"
        assert result.parameter >= 1.0

    def test_fit_frank_copula(self):
        x, y = _correlated_data(n=300, rho=0.5, seed=40)
        u, v = to_pseudo_observations(x, y)
        result = fit_copula(u, v, family="frank")
        assert result.family == "frank"
        assert result.parameter != 0

    def test_fit_best_selects_lowest_aic(self):
        x, y = _correlated_data(n=300, rho=0.6, seed=50)
        u, v = to_pseudo_observations(x, y)
        best = fit_copula(u, v, family="best")
        all_results = compare_copulas(u, v)
        assert best.family == all_results[0].family
        assert abs(best.aic - all_results[0].aic) < 1e-10


class TestCopulaFunction:
    def test_copula_function_values(self):
        """C(0.5, 0.5) for each family at a reasonable theta."""
        # Independence copula: C(0.5, 0.5) = 0.25
        c_gauss = copula_function(0.5, 0.5, "gaussian", 0.0)
        assert abs(c_gauss - 0.25) < 0.01

        c_clayton = copula_function(0.5, 0.5, "clayton", 2.0)
        assert 0.0 < c_clayton < 0.5

        c_gumbel = copula_function(0.5, 0.5, "gumbel", 1.0)
        # Gumbel theta=1 → independence
        assert abs(c_gumbel - 0.25) < 0.01

        c_frank = copula_function(0.5, 0.5, "frank", 0.0)
        assert abs(c_frank - 0.25) < 0.01

    def test_copula_density_positive(self):
        for fam, theta in [("gaussian", 0.5), ("clayton", 2.0), ("gumbel", 2.0), ("frank", 5.0)]:
            d = copula_density(0.5, 0.5, fam, theta)
            assert d > 0, f"Density should be positive for {fam}, got {d}"


class TestJointExceedance:
    def test_joint_exceedance(self):
        """P(both) + P(neither) = 1 - P(only one)."""
        copula = CopulaResult(
            family="gaussian", parameter=0.5, kendall_tau=0.3,
            spearman_rho=0.4, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        jp = joint_exceedance_probability(copula, 0.8, 0.8)
        assert isinstance(jp, JointProbability)
        # P(both) >= 0
        assert jp.prob_both_exceed >= 0
        # P(either) = 1 - C(u,v) should be >= P(both)
        assert jp.prob_either_exceed >= jp.prob_both_exceed

    def test_joint_return_period(self):
        copula = CopulaResult(
            family="clayton", parameter=2.0, kendall_tau=0.5,
            spearman_rho=0.6, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        jp = joint_exceedance_probability(copula, 0.9, 0.9)
        if jp.prob_both_exceed > 0:
            assert abs(jp.joint_return_period - 1.0 / jp.prob_both_exceed) < 1e-10

    def test_conditional_probability(self):
        copula = CopulaResult(
            family="gumbel", parameter=2.0, kendall_tau=0.5,
            spearman_rho=0.6, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        jp = joint_exceedance_probability(copula, 0.7, 0.7)
        assert 0 <= jp.prob_x_exceed_given_y <= 1


class TestSampling:
    def test_generate_copula_samples_shape(self):
        copula = CopulaResult(
            family="gaussian", parameter=0.5, kendall_tau=0.3,
            spearman_rho=0.4, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        u, v = generate_copula_samples(copula, n=500, seed=42)
        assert u.shape == (500,)
        assert v.shape == (500,)

    def test_generate_copula_samples_range(self):
        for fam, theta in [("gaussian", 0.5), ("clayton", 2.0), ("gumbel", 2.0), ("frank", 5.0)]:
            copula = CopulaResult(
                family=fam, parameter=theta, kendall_tau=0.3,
                spearman_rho=0.4, aic=10.0, log_likelihood=-5.0, n_samples=100,
            )
            u, v = generate_copula_samples(copula, n=500, seed=42)
            assert np.all(u > 0) and np.all(u < 1), f"{fam}: u out of (0,1)"
            assert np.all(v > 0) and np.all(v < 1), f"{fam}: v out of (0,1)"

    def test_generate_synthetic_data(self):
        copula = CopulaResult(
            family="gaussian", parameter=0.5, kendall_tau=0.3,
            spearman_rho=0.4, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        x, y = generate_synthetic_data(
            copula,
            marginal_x=("norm", (10, 2)),
            marginal_y=("expon", (0, 5)),
            n=2000,
            seed=42,
        )
        # Mean of x should be near 10
        assert abs(np.mean(x) - 10) < 1.0
        # Mean of expon(scale=5) = 5
        assert abs(np.mean(y) - 5) < 1.5


class TestTailDependence:
    def test_tail_dependence_clayton(self):
        copula = CopulaResult(
            family="clayton", parameter=2.0, kendall_tau=0.5,
            spearman_rho=0.6, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        td = tail_dependence(copula)
        assert td["lower"] > 0
        assert td["upper"] == 0.0

    def test_tail_dependence_gumbel(self):
        copula = CopulaResult(
            family="gumbel", parameter=2.0, kendall_tau=0.5,
            spearman_rho=0.6, aic=10.0, log_likelihood=-5.0, n_samples=100,
        )
        td = tail_dependence(copula)
        assert td["lower"] == 0.0
        assert td["upper"] > 0


class TestCompareCopulas:
    def test_compare_copulas_returns_sorted(self):
        x, y = _correlated_data(n=200, rho=0.5, seed=99)
        u, v = to_pseudo_observations(x, y)
        results = compare_copulas(u, v)
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].aic <= results[i + 1].aic


class TestEdgeCases:
    def test_insufficient_data(self):
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.5, 0.6])
        with pytest.raises(ValueError, match="at least"):
            fit_copula(u, v)

    def test_insufficient_data_pseudo_observations(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="at least"):
            to_pseudo_observations(x, y)
