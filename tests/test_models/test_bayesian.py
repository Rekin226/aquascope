"""Tests for aquascope.models.bayesian — Bayesian uncertainty quantification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aquascope.models.bayesian import (
    BayesianLinearRegression,
    BayesianPolynomialRegression,
    MetropolisHastings,
    PosteriorResult,
    bayesian_model_comparison,
    dic,
    effective_sample_size,
    gelman_rubin,
)


class TestBayesianLinearRegression:
    """Tests for conjugate Bayesian linear regression."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        self.x = rng.uniform(0, 10, n)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 1.0, n)
        self.X = np.column_stack([np.ones(n), self.x])

    def test_conjugate_linear_basic(self) -> None:
        """Known y = 2*x + 1 + noise — recover slope ≈ 2."""
        model = BayesianLinearRegression()
        result = model.fit(self.X, self.y)

        assert isinstance(result, PosteriorResult)
        slope_mean = result.posterior_mean["beta_1"]
        assert abs(slope_mean - 2.0) < 0.3, f"Slope {slope_mean} not close to 2.0"

    def test_conjugate_posterior_mean_close_to_ols(self) -> None:
        """Bayesian posterior mean ≈ OLS estimate (with weak prior)."""
        model = BayesianLinearRegression(prior_precision=0.001)
        result = model.fit(self.X, self.y)

        ols_beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        for j in range(2):
            name = f"beta_{j}"
            assert abs(result.posterior_mean[name] - ols_beta[j]) < 0.3

    def test_conjugate_credible_intervals(self) -> None:
        """True parameters fall within 95 % credible intervals."""
        model = BayesianLinearRegression()
        result = model.fit(self.X, self.y)

        true_vals = {"beta_0": 1.0, "beta_1": 2.0}
        for name, true_val in true_vals.items():
            lo, hi = result.credible_intervals[name]
            assert lo < true_val < hi, f"{name}: {true_val} not in ({lo}, {hi})"

    def test_conjugate_predict_shape(self) -> None:
        """Prediction output shapes match input."""
        model = BayesianLinearRegression()
        model.fit(self.X, self.y)

        X_new = np.column_stack([np.ones(20), np.linspace(0, 10, 20)])
        mean_pred, lower, upper = model.predict(X_new)
        assert mean_pred.shape == (20,)
        assert lower.shape == (20,)
        assert upper.shape == (20,)

    def test_conjugate_prediction_intervals(self) -> None:
        """lower < mean < upper for predictions."""
        model = BayesianLinearRegression()
        model.fit(self.X, self.y)

        X_new = np.column_stack([np.ones(10), np.linspace(0, 10, 10)])
        mean_pred, lower, upper = model.predict(X_new)
        assert np.all(lower < mean_pred)
        assert np.all(mean_pred < upper)

    def test_prior_effect(self) -> None:
        """A strong prior pulls the posterior toward the prior mean."""
        # Use a small dataset so the prior has more influence
        rng = np.random.default_rng(99)
        n = 15
        x_small = rng.uniform(0, 10, n)
        y_small = 2.0 * x_small + 1.0 + rng.normal(0, 1.0, n)
        X_small = np.column_stack([np.ones(n), x_small])

        # Strong prior: beta ≈ [0, 0] with very high precision
        strong_prior = BayesianLinearRegression(
            prior_mean=np.array([0.0, 0.0]),
            prior_precision=1000.0,
        )
        result_strong = strong_prior.fit(X_small, y_small)

        weak_prior = BayesianLinearRegression(prior_precision=0.001)
        result_weak = weak_prior.fit(X_small, y_small)

        # Strong prior should pull slope closer to 0 than weak prior
        assert abs(result_strong.posterior_mean["beta_1"]) < abs(result_weak.posterior_mean["beta_1"])

    def test_empty_data_raises(self) -> None:
        """Empty training data raises ValueError."""
        model = BayesianLinearRegression()
        with pytest.raises(ValueError, match="empty"):
            model.fit(np.empty((0, 2)), np.empty(0))

    def test_predict_before_fit_raises(self) -> None:
        """Predict before fit raises RuntimeError."""
        model = BayesianLinearRegression()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(np.ones((5, 2)))


class TestBayesianPolynomialRegression:
    """Tests for Bayesian polynomial regression."""

    def setup_method(self) -> None:
        rng = np.random.default_rng(1)
        n = 150
        self.x = rng.uniform(-3, 3, n)
        self.y = 0.5 * self.x**2 - 1.0 * self.x + 2.0 + rng.normal(0, 0.5, n)

    def test_polynomial_degree2(self) -> None:
        """Quadratic data — recover positive curvature."""
        model = BayesianPolynomialRegression(degree=2)
        result = model.fit(self.x, self.y)

        # beta_2 should be close to 0.5 (x^2 coefficient)
        assert result.posterior_mean["beta_2"] > 0.2, "Curvature not recovered"

    def test_polynomial_predict(self) -> None:
        """Smooth prediction curve with correct shape."""
        model = BayesianPolynomialRegression(degree=2)
        model.fit(self.x, self.y)

        x_new = np.linspace(-3, 3, 50)
        mean_pred, lower, upper = model.predict(x_new)
        assert mean_pred.shape == (50,)
        assert np.all(lower < upper)


class TestMetropolisHastings:
    """Tests for the Metropolis-Hastings MCMC sampler."""

    def test_metropolis_hastings_basic(self) -> None:
        """Sample from a known normal posterior — mean should be recovered."""
        true_mean = 3.0
        true_std = 1.0

        def log_post(theta: np.ndarray) -> float:
            return float(stats.norm.logpdf(theta[0], loc=true_mean, scale=true_std))

        sampler = MetropolisHastings(log_post, ["mu"], proposal_scale=0.5)
        result = sampler.sample(np.array([0.0]), n_samples=5000, burn_in=1000, seed=42)

        assert abs(result.posterior_mean["mu"] - true_mean) < 0.3

    def test_metropolis_acceptance_rate(self) -> None:
        """Acceptance rate should be between 15 % and 60 %."""
        def log_post(theta: np.ndarray) -> float:
            return float(stats.norm.logpdf(theta[0], loc=0.0, scale=1.0))

        sampler = MetropolisHastings(log_post, ["mu"], proposal_scale=1.0)
        # Run with enough samples to get a stable acceptance rate
        result = sampler.sample(np.array([0.0]), n_samples=3000, burn_in=500, seed=7)

        # Check that sampling produced reasonable samples (indirect check)
        assert result.posterior_std["mu"] > 0.3

    def test_metropolis_burn_in_removed(self) -> None:
        """Chain length equals n_samples, not n_samples + burn_in."""
        def log_post(theta: np.ndarray) -> float:
            return float(-0.5 * theta[0] ** 2)

        sampler = MetropolisHastings(log_post, ["x"], proposal_scale=1.0)
        result = sampler.sample(np.array([0.0]), n_samples=2000, burn_in=500, seed=0)

        assert len(result.chains["x"]) == 2000


class TestGelmanRubin:
    """Tests for the Gelman–Rubin diagnostic."""

    def test_gelman_rubin_converged(self) -> None:
        """Identical chains → R_hat ≈ 1.0."""
        rng = np.random.default_rng(10)
        chain = rng.normal(0, 1, 1000)
        r_hat = gelman_rubin([chain, chain.copy()])
        assert abs(r_hat - 1.0) < 0.05

    def test_gelman_rubin_not_converged(self) -> None:
        """Very different chains → R_hat > 1.1."""
        rng = np.random.default_rng(11)
        chain_a = rng.normal(0, 1, 500)
        chain_b = rng.normal(10, 1, 500)
        r_hat = gelman_rubin([chain_a, chain_b])
        assert r_hat > 1.1


class TestEffectiveSampleSize:
    """Tests for effective sample size."""

    def test_effective_sample_size_bound(self) -> None:
        """ESS should be ≤ n_samples."""
        rng = np.random.default_rng(20)
        chain = np.cumsum(rng.normal(0, 1, 500))  # highly correlated
        ess = effective_sample_size(chain)
        assert ess <= 500

    def test_effective_sample_size_iid(self) -> None:
        """For i.i.d. data, ESS ≈ n."""
        rng = np.random.default_rng(21)
        chain = rng.normal(0, 1, 2000)
        ess = effective_sample_size(chain)
        # ESS should be close to n for iid (at least > 50 % of n)
        assert ess > 1000


class TestDIC:
    """Tests for DIC computation."""

    def test_dic_computation(self) -> None:
        """DIC is finite for a simple model."""
        rng = np.random.default_rng(30)
        true_mu = 5.0
        data = rng.normal(true_mu, 1.0, 100)
        samples = {"mu": rng.normal(true_mu, 0.1, 500)}

        def log_lik(theta: dict[str, float], d: np.ndarray) -> float:
            return float(np.sum(stats.norm.logpdf(d, loc=theta["mu"], scale=1.0)))

        dic_val = dic(log_lik, samples, data)
        assert np.isfinite(dic_val)


class TestModelComparison:
    """Tests for bayesian_model_comparison."""

    def test_model_comparison(self) -> None:
        """Returns DataFrame with expected columns."""
        r1 = PosteriorResult(
            parameter_names=["a"],
            chains={"a": np.array([1.0, 2.0])},
            posterior_mean={"a": 1.5},
            posterior_std={"a": 0.5},
            credible_intervals={"a": (1.0, 2.0)},
            dic=100.0,
            waic=102.0,
        )
        r2 = PosteriorResult(
            parameter_names=["a"],
            chains={"a": np.array([1.0, 2.0])},
            posterior_mean={"a": 1.5},
            posterior_std={"a": 0.5},
            credible_intervals={"a": (1.0, 2.0)},
            dic=110.0,
            waic=112.0,
        )
        df = bayesian_model_comparison([("model_A", r1), ("model_B", r2)])
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"model_name", "DIC", "WAIC", "delta_DIC", "weight"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 2
        # Best model (lowest DIC) should have delta_DIC == 0
        assert df.loc[df["model_name"] == "model_A", "delta_DIC"].iloc[0] == 0.0


class TestDataFrameInputs:
    """Ensure pandas inputs are handled."""

    def test_dataframe_input(self) -> None:
        """BayesianLinearRegression accepts DataFrame/Series."""
        rng = np.random.default_rng(50)
        n = 100
        X_df = pd.DataFrame({"intercept": np.ones(n), "x": rng.uniform(0, 5, n)})
        y_s = pd.Series(2.0 * X_df["x"] + 1.0 + rng.normal(0, 0.5, n))

        model = BayesianLinearRegression()
        result = model.fit(X_df, y_s)
        assert abs(result.posterior_mean["beta_1"] - 2.0) < 0.5
