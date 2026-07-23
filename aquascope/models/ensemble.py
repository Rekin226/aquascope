"""Model ensemble and stacking methods for hydrological forecasting.

Provides three ensemble strategies:
- ``WeightedEnsemble`` — weighted average with equal, performance-based, or optimal weights
- ``StackingEnsemble`` — two-level stacking with configurable meta-learner
- ``AdaptiveEnsemble`` — time-varying weights based on recent prediction errors

All ensembles accept models with scikit-learn–compatible ``fit(X, y)`` and
``predict(X)`` interfaces and return :class:`EnsembleResult` dataclass instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from aquascope.models.base import BaseHydroModel

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble prediction.

    Attributes:
        predictions: Combined ensemble predictions.
        lower_bound: Uncertainty lower bound (mean − 1.96·σ), or ``None``.
        upper_bound: Uncertainty upper bound (mean + 1.96·σ), or ``None``.
        model_weights: Mapping of model name → weight used.
        individual_predictions: Mapping of model name → raw prediction array.
        metrics: Evaluation metrics dict when actuals are provided, else ``None``.
    """

    predictions: np.ndarray
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None
    model_weights: dict[str, float] = field(default_factory=dict)
    individual_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, float] | None = None


class WeightedEnsemble:
    """Weighted average ensemble of multiple models.

    Combines predictions from multiple models using learned or fixed weights.

    Parameters:
        models: List of ``(name, model_instance)`` tuples.  Each model must
            expose ``fit(X, y)`` and ``predict(X)`` methods.
        weighting: Strategy for computing weights.

            * ``"equal"`` — each model receives ``1/n``.
            * ``"performance"`` — weight by inverse RMSE on a validation set.
            * ``"optimal"`` — minimise ensemble RMSE via constrained optimisation.
    """

    def __init__(self, models: list[tuple[str, object]], weighting: str = "equal"):
        if not models:
            raise ValueError("At least one model is required")
        self.models = models
        self.weighting = weighting
        self.weights_: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit all constituent models and compute weights.

        Parameters:
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation features (required when *weighting* is not ``"equal"``).
            y_val: Validation targets (required when *weighting* is not ``"equal"``).

        Raises:
            ValueError: If validation data is missing for non-equal weighting.
        """
        if self.weighting != "equal" and (X_val is None or y_val is None):
            raise ValueError(f"X_val and y_val are required for weighting='{self.weighting}'")

        for name, model in self.models:
            model.fit(X_train, y_train)
            logger.info("Fitted base model '%s'", name)

        if self.weighting == "equal":
            n = len(self.models)
            self.weights_ = {name: 1.0 / n for name, _ in self.models}
        elif self.weighting == "performance":
            self._compute_performance_weights(X_val, y_val)
        elif self.weighting == "optimal":
            self._compute_optimal_weights(X_val, y_val)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting}")

        self._fitted = True
        logger.info("WeightedEnsemble fitted with weights: %s", self.weights_)

    def predict(self, X: pd.DataFrame) -> EnsembleResult:
        """Generate weighted ensemble prediction with uncertainty bounds.

        The ensemble prediction is ``ŷ = Σ(wᵢ · predᵢ)``.  Uncertainty bounds
        are derived from the weighted variance of individual predictions:
        ``lower/upper = ŷ ± 1.96 · √(weighted_variance)``.

        Parameters:
            X: Feature matrix for prediction.

        Returns:
            EnsembleResult with predictions, bounds, and per-model outputs.
        """
        if not self._fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        individual: dict[str, np.ndarray] = {}
        for name, model in self.models:
            individual[name] = np.asarray(model.predict(X)).ravel()

        weights_arr = np.array([self.weights_[name] for name, _ in self.models])
        preds_matrix = np.column_stack([individual[name] for name, _ in self.models])

        ensemble_pred = preds_matrix @ weights_arr
        weighted_var = np.sum(weights_arr * (preds_matrix - ensemble_pred[:, None]) ** 2, axis=1)
        std = np.sqrt(np.maximum(weighted_var, 0.0))

        return EnsembleResult(
            predictions=ensemble_pred,
            lower_bound=ensemble_pred - 1.96 * std,
            upper_bound=ensemble_pred + 1.96 * std,
            model_weights=dict(self.weights_),
            individual_predictions=individual,
        )

    def evaluate(self, X: pd.DataFrame, y_actual: pd.Series) -> dict[str, float]:
        """Evaluate ensemble using standard hydrological metrics.

        Parameters:
            X: Feature matrix.
            y_actual: Ground-truth target values.

        Returns:
            Dict with keys ``nse``, ``kge``, ``rmse``, ``mae``, etc.
        """
        result = self.predict(X)
        return BaseHydroModel._compute_metrics(np.asarray(y_actual), result.predictions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_performance_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Weight each model by its inverse RMSE on the validation set."""
        y_arr = np.asarray(y_val)
        inv_rmses: dict[str, float] = {}
        for name, model in self.models:
            pred = np.asarray(model.predict(X_val)).ravel()
            rmse = float(np.sqrt(np.mean((y_arr - pred) ** 2)))
            inv_rmses[name] = 1.0 / max(rmse, 1e-10)

        total = sum(inv_rmses.values())
        self.weights_ = {name: val / total for name, val in inv_rmses.items()}

    def _compute_optimal_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Find weights that minimise validation RMSE via constrained optimisation.

        Solves  ``min ‖y − P·w‖²``  s.t.  ``wᵢ ≥ 0, Σwᵢ = 1``.
        """
        from scipy.optimize import minimize

        y_arr = np.asarray(y_val).ravel()
        preds = []
        for name, model in self.models:
            preds.append(np.asarray(model.predict(X_val)).ravel())
        P = np.column_stack(preds)
        n_models = P.shape[1]

        def objective(w: np.ndarray) -> float:
            return float(np.mean((y_arr - P @ w) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        w0 = np.ones(n_models) / n_models
        res = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")

        names = [name for name, _ in self.models]
        self.weights_ = {name: float(w) for name, w in zip(names, res.x)}


class StackingEnsemble:
    """Stacking ensemble with a meta-learner.

    **Level 0** — base models generate out-of-fold predictions via K-fold CV.
    **Level 1** — a meta-learner (Ridge by default) is trained on stacked
    base-model predictions to produce the final output.

    Parameters:
        base_models: List of ``(name, model_instance)`` tuples.
        meta_learner: Meta-learner type — ``"ridge"``, ``"linear"``, or ``"rf"``.
        n_folds: Number of cross-validation folds for out-of-fold predictions.
    """

    def __init__(
        self,
        base_models: list[tuple[str, object]],
        meta_learner: str = "ridge",
        n_folds: int = 5,
    ):
        if not base_models:
            raise ValueError("At least one base model is required")
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.n_folds = n_folds
        self._meta_learner = None
        self._fitted_base_models: list[tuple[str, object]] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the stacking ensemble.

        Steps:
            1. For each CV fold, train each base model on in-fold data and
               predict on out-of-fold data.
            2. Stack out-of-fold predictions as meta-features.
            3. Train the meta-learner on ``meta-features → y``.
            4. Re-train every base model on the **full** dataset for use at
               prediction time.

        Parameters:
            X: Training feature matrix.
            y: Training target vector.
        """
        import copy

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        n_samples = len(y_arr)
        n_models = len(self.base_models)

        meta_features = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
            X_fold_train = pd.DataFrame(X_arr[train_idx], columns=X.columns if hasattr(X, "columns") else None)
            y_fold_train = pd.Series(y_arr[train_idx])
            X_fold_val = pd.DataFrame(X_arr[val_idx], columns=X.columns if hasattr(X, "columns") else None)

            for model_idx, (name, model) in enumerate(self.base_models):
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_fold_train, y_fold_train)
                meta_features[val_idx, model_idx] = np.asarray(fold_model.predict(X_fold_val)).ravel()

            logger.info("Stacking fold %d/%d complete", fold_idx + 1, self.n_folds)

        self._meta_learner = self._build_meta_learner()
        self._meta_learner.fit(meta_features, y_arr)

        # Re-train base models on all data for prediction
        self._fitted_base_models = []
        for name, model in self.base_models:
            full_model = copy.deepcopy(model)
            full_model.fit(X, y)
            self._fitted_base_models.append((name, full_model))

        self._fitted = True
        logger.info("StackingEnsemble fitted with %d base models and '%s' meta-learner", n_models, self.meta_learner_type)

    def predict(self, X: pd.DataFrame) -> EnsembleResult:
        """Generate stacking prediction.

        Steps:
            1. Obtain predictions from every base model.
            2. Stack them as meta-features.
            3. Pass through the meta-learner to get final predictions.

        Parameters:
            X: Feature matrix.

        Returns:
            EnsembleResult with predictions and individual model outputs.
        """
        if not self._fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        individual: dict[str, np.ndarray] = {}
        meta_features_list: list[np.ndarray] = []

        for name, model in self._fitted_base_models:
            pred = np.asarray(model.predict(X)).ravel()
            individual[name] = pred
            meta_features_list.append(pred)

        meta_features = np.column_stack(meta_features_list)
        ensemble_pred = np.asarray(self._meta_learner.predict(meta_features)).ravel()

        # Derive weights from meta-learner coefficients when available
        weights: dict[str, float] = {}
        if hasattr(self._meta_learner, "coef_"):
            coefs = self._meta_learner.coef_.ravel()
            total = np.sum(np.abs(coefs)) or 1.0
            for (name, _), c in zip(self._fitted_base_models, coefs):
                weights[name] = float(np.abs(c) / total)
        else:
            n = len(self._fitted_base_models)
            weights = {name: 1.0 / n for name, _ in self._fitted_base_models}

        return EnsembleResult(
            predictions=ensemble_pred,
            model_weights=weights,
            individual_predictions=individual,
        )

    def evaluate(self, X: pd.DataFrame, y_actual: pd.Series) -> dict[str, float]:
        """Evaluate stacking ensemble using standard hydrological metrics.

        Parameters:
            X: Feature matrix.
            y_actual: Ground-truth target values.

        Returns:
            Dict with keys ``nse``, ``kge``, ``rmse``, ``mae``, etc.
        """
        result = self.predict(X)
        return BaseHydroModel._compute_metrics(np.asarray(y_actual), result.predictions)

    def _build_meta_learner(self) -> object:
        """Instantiate the meta-learner."""
        if self.meta_learner_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.meta_learner_type == "linear":
            return LinearRegression()
        elif self.meta_learner_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")


class AdaptiveEnsemble:
    """Ensemble with time-varying weights based on recent performance.

    Tracks recent prediction errors and adjusts weights to favour models
    that perform well in the current regime.  Older errors are down-weighted
    via an exponential decay factor.

    Parameters:
        models: List of ``(name, model_instance)`` tuples.
        lookback: Number of recent observations used for weight computation.
        decay: Exponential decay factor for older errors (typically 0.90–0.99).
    """

    def __init__(
        self,
        models: list[tuple[str, object]],
        lookback: int = 30,
        decay: float = 0.95,
    ):
        if not models:
            raise ValueError("At least one model is required")
        self.models = models
        self.lookback = lookback
        self.decay = decay
        self.weights_: dict[str, float] = {}
        self._error_history: dict[str, list[float]] = {name: [] for name, _ in models}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all base models on training data.

        Initialises equal weights.

        Parameters:
            X: Training feature matrix.
            y: Training target vector.
        """
        for name, model in self.models:
            model.fit(X, y)
            logger.info("AdaptiveEnsemble: fitted model '%s'", name)

        n = len(self.models)
        self.weights_ = {name: 1.0 / n for name, _ in self.models}
        self._error_history = {name: [] for name, _ in self.models}
        self._fitted = True

    def update_and_predict(
        self,
        X_new: pd.DataFrame,
        y_recent: pd.Series | None = None,
    ) -> EnsembleResult:
        """Update weights based on recent errors and generate predictions.

        If *y_recent* is provided the method computes each model's recent
        weighted RMSE (with exponential decay), then normalises to obtain
        updated weights.  Otherwise the current weights are used as-is.

        Parameters:
            X_new: Feature matrix for prediction.
            y_recent: Recent ground-truth values for weight updating.

        Returns:
            EnsembleResult with predictions, bounds, weights, and per-model outputs.
        """
        if not self._fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Update error history and weights if actuals are available
        if y_recent is not None:
            y_arr = np.asarray(y_recent).ravel()
            for name, model in self.models:
                pred = np.asarray(model.predict(X_new)).ravel()[: len(y_arr)]
                errors = (y_arr - pred) ** 2
                self._error_history[name].extend(errors.tolist())

            self._update_weights()

        # Generate predictions
        individual: dict[str, np.ndarray] = {}
        for name, model in self.models:
            individual[name] = np.asarray(model.predict(X_new)).ravel()

        weights_arr = np.array([self.weights_[name] for name, _ in self.models])
        preds_matrix = np.column_stack([individual[name] for name, _ in self.models])

        ensemble_pred = preds_matrix @ weights_arr
        weighted_var = np.sum(weights_arr * (preds_matrix - ensemble_pred[:, None]) ** 2, axis=1)
        std = np.sqrt(np.maximum(weighted_var, 0.0))

        return EnsembleResult(
            predictions=ensemble_pred,
            lower_bound=ensemble_pred - 1.96 * std,
            upper_bound=ensemble_pred + 1.96 * std,
            model_weights=dict(self.weights_),
            individual_predictions=individual,
        )

    def _update_weights(self) -> None:
        """Recompute model weights from recent error history with exponential decay."""
        inv_wrmses: dict[str, float] = {}
        for name, _ in self.models:
            errors = self._error_history[name][-self.lookback:]
            if not errors:
                inv_wrmses[name] = 1.0
                continue

            n = len(errors)
            decay_weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
            decay_weights /= decay_weights.sum()
            wrmse = float(np.sqrt(np.sum(decay_weights * np.array(errors))))
            inv_wrmses[name] = 1.0 / max(wrmse, 1e-10)

        total = sum(inv_wrmses.values())
        self.weights_ = {name: val / total for name, val in inv_wrmses.items()}
        logger.debug("AdaptiveEnsemble weights updated: %s", self.weights_)


def ensemble_cross_validate(
    ensemble: WeightedEnsemble | StackingEnsemble,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
) -> dict[str, float]:
    """Cross-validate any ensemble and return mean metrics across folds.

    Parameters:
        ensemble: A ``WeightedEnsemble`` or ``StackingEnsemble`` instance (unfitted).
        X: Full feature matrix.
        y: Full target vector.
        n_folds: Number of CV folds.

    Returns:
        Dict mapping metric name → mean value across folds.
    """
    import copy

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in kf.split(X_arr):
        X_train = pd.DataFrame(X_arr[train_idx], columns=X.columns if hasattr(X, "columns") else None)
        y_train = pd.Series(y_arr[train_idx])
        X_val = pd.DataFrame(X_arr[val_idx], columns=X.columns if hasattr(X, "columns") else None)
        y_val = pd.Series(y_arr[val_idx])

        ens_copy = copy.deepcopy(ensemble)

        if isinstance(ens_copy, WeightedEnsemble):
            ens_copy.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            ens_copy.fit(X_train, y_train)

        pred = ens_copy.predict(X_val)
        metrics = BaseHydroModel._compute_metrics(np.asarray(y_val), pred.predictions)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    mean_metrics: dict[str, float] = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if key in m]
        mean_metrics[key] = round(float(np.mean(values)), 4)

    return mean_metrics
