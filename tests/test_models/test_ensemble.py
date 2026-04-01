"""Tests for aquascope.models.ensemble — weighted, stacking, and adaptive ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from aquascope.models.ensemble import (
    AdaptiveEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
    ensemble_cross_validate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate simple regression data: y = 2*x1 + 3*x2 + noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    y = pd.Series(2 * X["x1"] + 3 * X["x2"] + rng.normal(0, 0.5, n), name="y")
    return X, y


def _make_models(n: int = 3) -> list[tuple[str, object]]:
    """Return n sklearn-compatible models wrapped in (name, instance) tuples."""
    models: list[tuple[str, object]] = [
        ("ridge", Ridge(alpha=1.0)),
        ("linear", LinearRegression()),
        ("tree", DecisionTreeRegressor(max_depth=4, random_state=42)),
    ]
    return models[:n]


def _split(X: pd.DataFrame, y: pd.Series, ratio: float = 0.7):
    """Simple train/val split."""
    n = int(len(y) * ratio)
    return X.iloc[:n], y.iloc[:n], X.iloc[n:], y.iloc[n:]


# ---------------------------------------------------------------------------
# WeightedEnsemble tests
# ---------------------------------------------------------------------------


class TestWeightedEnsemble:
    def test_weighted_equal(self):
        """Three models with equal weighting each receive 1/3."""
        models = _make_models(3)
        ens = WeightedEnsemble(models, weighting="equal")
        X, y = _make_regression_data()
        ens.fit(X, y)
        for w in ens.weights_.values():
            assert abs(w - 1.0 / 3) < 1e-8

    def test_weighted_performance(self):
        """Better-performing model should receive a higher weight."""
        X, y = _make_regression_data(300, seed=7)
        X_train, y_train, X_val, y_val = _split(X, y)

        # Linear model should outperform a shallow tree on linear data
        models = [
            ("linear", LinearRegression()),
            ("tree", DecisionTreeRegressor(max_depth=1, random_state=0)),
        ]
        ens = WeightedEnsemble(models, weighting="performance")
        ens.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        assert ens.weights_["linear"] > ens.weights_["tree"]

    def test_weighted_optimal(self):
        """Optimal weights should minimise validation RMSE."""
        X, y = _make_regression_data(300, seed=11)
        X_train, y_train, X_val, y_val = _split(X, y)
        models = _make_models(3)
        ens = WeightedEnsemble(models, weighting="optimal")
        ens.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Weights must sum to 1 and be non-negative
        assert abs(sum(ens.weights_.values()) - 1.0) < 1e-6
        assert all(w >= -1e-8 for w in ens.weights_.values())

    def test_weighted_predict_shape(self):
        """Output prediction shape matches input length."""
        X, y = _make_regression_data()
        models = _make_models(2)
        ens = WeightedEnsemble(models, weighting="equal")
        ens.fit(X, y)
        result = ens.predict(X)
        assert result.predictions.shape == (len(X),)

    def test_weighted_uncertainty_bounds(self):
        """Lower bound < prediction < upper bound (element-wise)."""
        X, y = _make_regression_data()
        models = _make_models(3)
        ens = WeightedEnsemble(models, weighting="equal")
        ens.fit(X, y)
        result = ens.predict(X)
        assert np.all(result.lower_bound <= result.predictions + 1e-10)
        assert np.all(result.upper_bound >= result.predictions - 1e-10)


# ---------------------------------------------------------------------------
# StackingEnsemble tests
# ---------------------------------------------------------------------------


class TestStackingEnsemble:
    def test_stacking_fit_predict(self):
        """Basic stacking works end-to-end."""
        X, y = _make_regression_data(200)
        models = _make_models(3)
        ens = StackingEnsemble(models, meta_learner="ridge", n_folds=3)
        ens.fit(X, y)
        result = ens.predict(X)
        assert result.predictions.shape == (len(X),)

    def test_stacking_meta_learner_ridge(self):
        """Meta-learner is a Ridge instance when configured."""
        models = _make_models(2)
        ens = StackingEnsemble(models, meta_learner="ridge")
        X, y = _make_regression_data(100)
        ens.fit(X, y)
        assert isinstance(ens._meta_learner, Ridge)

    def test_stacking_outperforms_average(self):
        """Stacking RMSE ≤ mean of individual model RMSEs on simple data."""
        X, y = _make_regression_data(300, seed=99)
        X_train, y_train, X_test, y_test = _split(X, y, 0.7)

        models = _make_models(3)
        ens = StackingEnsemble(models, meta_learner="ridge", n_folds=3)
        ens.fit(X_train, y_train)
        result = ens.predict(X_test)
        y_arr = np.asarray(y_test)

        ens_rmse = float(np.sqrt(np.mean((y_arr - result.predictions) ** 2)))

        individual_rmses = []
        for name, pred in result.individual_predictions.items():
            individual_rmses.append(float(np.sqrt(np.mean((y_arr - pred) ** 2))))

        assert ens_rmse <= np.mean(individual_rmses) + 1e-6


# ---------------------------------------------------------------------------
# AdaptiveEnsemble tests
# ---------------------------------------------------------------------------


class TestAdaptiveEnsemble:
    def test_adaptive_initial_equal(self):
        """Before any updates, all weights should be equal."""
        X, y = _make_regression_data()
        models = _make_models(3)
        ens = AdaptiveEnsemble(models, lookback=30, decay=0.95)
        ens.fit(X, y)
        for w in ens.weights_.values():
            assert abs(w - 1.0 / 3) < 1e-8

    def test_adaptive_update_shifts_weights(self):
        """After observing errors, the poor model gets a lower weight."""
        X, y = _make_regression_data(200, seed=21)
        X_train, y_train, X_val, y_val = _split(X, y)

        models = [
            ("good", LinearRegression()),
            ("bad", DecisionTreeRegressor(max_depth=1, random_state=0)),
        ]
        ens = AdaptiveEnsemble(models, lookback=50, decay=0.95)
        ens.fit(X_train, y_train)

        result = ens.update_and_predict(X_val, y_recent=y_val)

        # The better model should receive higher weight
        assert result.model_weights["good"] > result.model_weights["bad"]

    def test_adaptive_decay_effect(self):
        """Higher decay (slower forgetting) should produce different weights than lower decay."""
        X, y = _make_regression_data(200, seed=33)
        X_train, y_train, X_val, y_val = _split(X, y)

        def _run_with_decay(decay: float) -> dict[str, float]:
            models = [("lr", LinearRegression()), ("tree", DecisionTreeRegressor(max_depth=1, random_state=0))]
            ens = AdaptiveEnsemble(models, lookback=50, decay=decay)
            ens.fit(X_train, y_train)
            result = ens.update_and_predict(X_val, y_recent=y_val)
            return result.model_weights

        w_high = _run_with_decay(0.99)
        w_low = _run_with_decay(0.80)

        # With different decay values the weights should differ
        assert abs(w_high["lr"] - w_low["lr"]) > 1e-6


# ---------------------------------------------------------------------------
# EnsembleResult & utility tests
# ---------------------------------------------------------------------------


class TestEnsembleResultAndUtils:
    def test_ensemble_result_individual_predictions(self):
        """individual_predictions dict contains all model names."""
        X, y = _make_regression_data()
        models = _make_models(3)
        ens = WeightedEnsemble(models, weighting="equal")
        ens.fit(X, y)
        result = ens.predict(X)
        for name, _ in models:
            assert name in result.individual_predictions
            assert len(result.individual_predictions[name]) == len(X)

    def test_ensemble_evaluate_metrics(self):
        """evaluate() returns dict with expected metric keys."""
        X, y = _make_regression_data()
        models = _make_models(2)
        ens = WeightedEnsemble(models, weighting="equal")
        ens.fit(X, y)
        metrics = ens.evaluate(X, y)
        for key in ("nse", "kge", "rmse", "mae"):
            assert key in metrics

    def test_ensemble_cross_validate(self):
        """ensemble_cross_validate returns dict with expected keys."""
        X, y = _make_regression_data(200)
        models = _make_models(2)
        ens = WeightedEnsemble(models, weighting="equal")
        cv_metrics = ensemble_cross_validate(ens, X, y, n_folds=3)
        for key in ("nse", "kge", "rmse", "mae"):
            assert key in cv_metrics

    def test_single_model_ensemble(self):
        """Ensemble with one model should produce that model's predictions."""
        X, y = _make_regression_data(100)
        single_model = LinearRegression()
        single_model.fit(X, y)
        expected = single_model.predict(X)

        ens = WeightedEnsemble([("only", LinearRegression())], weighting="equal")
        ens.fit(X, y)
        result = ens.predict(X)
        np.testing.assert_allclose(result.predictions, expected, atol=1e-8)

    def test_empty_models_raises(self):
        """Empty model list raises ValueError."""
        with pytest.raises(ValueError):
            WeightedEnsemble([], weighting="equal")
        with pytest.raises(ValueError):
            StackingEnsemble([])
        with pytest.raises(ValueError):
            AdaptiveEnsemble([])

    def test_predict_before_fit_raises(self):
        """Calling predict on unfitted ensemble raises RuntimeError."""
        X, _ = _make_regression_data(10)
        ens = WeightedEnsemble(_make_models(1))
        with pytest.raises(RuntimeError):
            ens.predict(X)
