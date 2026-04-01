"""Base model interface and hydrological metrics for all predictive models."""

from __future__ import annotations

import abc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseHydroModel(abc.ABC):
    """Abstract base for all AquaScope predictive models.

    All models must implement:
      - ``fit(df)`` — train on a time-series DataFrame
      - ``predict(horizon)`` — generate future predictions

    The normalised output of ``predict()`` always returns a DataFrame with:
      - ``yhat`` — predicted value
      - ``yhat_lower`` / ``yhat_upper`` — uncertainty bounds (if supported)
      - DatetimeIndex named ``datetime``

    Parameters
    ----------
    target_variable : str
        Column name to use as the prediction target.
    """

    MODEL_ID: str = "base"
    SUPPORTS_UNCERTAINTY: bool = False
    SUPPORTS_MULTIVARIATE: bool = False

    def __init__(self, target_variable: str = "value"):
        self.target_variable = target_variable
        self._is_fitted = False
        self._training_dates: pd.DatetimeIndex | None = None
        self._training_mean: float | None = None
        self._training_std: float | None = None

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> BaseHydroModel:
        """Train the model on historical data."""

    @abc.abstractmethod
    def predict(self, horizon: int = 7, **kwargs) -> pd.DataFrame:
        """Generate predictions for *horizon* days into the future."""

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model on a test DataFrame using standard hydro metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Test data with the same schema used for fitting.

        Returns
        -------
        dict
            Dictionary of metric name → value.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        y_true = df["value"].values if "value" in df.columns else df.iloc[:, 0].values
        pred = self.predict(horizon=len(y_true))
        y_pred = pred["yhat"].values[: len(y_true)]
        return self._compute_metrics(y_true, y_pred)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute standard hydrological skill metrics.

        Returns
        -------
        dict
            Keys: ``nse``, ``kge``, ``rmse``, ``mae``, ``r2``, ``n_samples``.
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        if len(y_true) == 0:
            return {}

        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)

        nse = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Kling-Gupta Efficiency
        r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
        alpha = float(y_pred.std() / y_true.std()) if y_true.std() > 0 else float("nan")
        beta = float(y_pred.mean() / y_true.mean()) if y_true.mean() != 0 else float("nan")
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return {
            "nse": round(float(nse), 4),
            "kge": round(float(kge), 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(float(r2), 4),
            "n_samples": len(y_true),
        }

    def _prepare_series(self, df: pd.DataFrame) -> pd.Series:
        """Extract the target column and ensure a sorted datetime index."""
        series = df["value"] if "value" in df.columns else df.iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        series = series.sort_index().dropna()
        self._training_dates = series.index
        self._training_mean = float(series.mean())
        self._training_std = float(series.std())
        return series

    def _future_dates(self, horizon: int, freq: str = "D") -> pd.DatetimeIndex:
        """Generate future dates starting from the end of training data."""
        if self._training_dates is None:
            raise RuntimeError("Model not fitted yet")
        last_date = self._training_dates[-1]
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq=freq)

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "unfitted"
        return f"<{self.__class__.__name__} [{fitted}] target='{self.target_variable}'>"
