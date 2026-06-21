"""Base model interface and hydrological metrics for all predictive models."""

from __future__ import annotations

import abc
import logging

import numpy as np
import pandas as pd

from aquascope.analysis import metrics

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
        mae = float(np.mean(np.abs(residuals)))

        nse_val = metrics.nse(y_true, y_pred)
        kge_val = metrics.kge(y_true, y_pred)
        rmse_val = metrics.rmse(y_true, y_pred)
        r2_val = metrics.r2(y_true, y_pred)

        return {
            "nse": round(nse_val, 4) if not np.isnan(nse_val) else float("nan"),
            "kge": round(kge_val, 4) if not np.isnan(kge_val) else float("nan"),
            "rmse": round(rmse_val, 4) if not np.isnan(rmse_val) else float("nan"),
            "mae": round(mae, 4),
            "r2": round(r2_val, 4) if not np.isnan(r2_val) else float("nan"),
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
