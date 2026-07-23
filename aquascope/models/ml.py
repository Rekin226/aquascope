"""
Machine-learning models for hydrological forecasting and anomaly detection.

- ``RandomForestModel`` — Multivariate regression with quantile uncertainty
- ``XGBoostModel`` — Gradient boosting for tabular water data
- ``IsolationForestModel`` — Unsupervised anomaly detection
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aquascope.models.base import BaseHydroModel
from aquascope.utils.imports import require

logger = logging.getLogger(__name__)


def make_lag_features(
    series: pd.Series,
    lags: list[int],
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create lag, rolling-statistic, and calendar features from a time series.

    Parameters
    ----------
    series : pd.Series
        Input series with a DatetimeIndex.
    lags : list[int]
        Autoregressive lag offsets.
    rolling_windows : list[int] | None
        Window sizes for rolling mean / std features.

    Returns
    -------
    pd.DataFrame
        Feature matrix (rows with NaN from lagging are dropped).
    """
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    if rolling_windows:
        for window in rolling_windows:
            df[f"rolling_mean_{window}"] = series.rolling(window).mean()
            df[f"rolling_std_{window}"] = series.rolling(window).std()
    df["dayofyear"] = series.index.dayofyear
    df["month"] = series.index.month
    df["season"] = series.index.month % 12 // 3
    return df.dropna()


class RandomForestModel(BaseHydroModel):
    """Random Forest regressor with lag features and quantile uncertainty.

    Uses recursive one-step-ahead forecasting: each prediction is appended
    to history and used to compute lags for the next step.

    Parameters
    ----------
    lags : list[int] | None
        Autoregressive lag offsets (default ``[1, 2, 3, 7, 14, 30]``).
    rolling_windows : list[int] | None
        Rolling-statistic window sizes (default ``[7, 30]``).
    n_estimators : int
        Number of trees.
    """

    MODEL_ID = "random_forest"
    SUPPORTS_UNCERTAINTY = True
    SUPPORTS_MULTIVARIATE = True

    def __init__(
        self,
        target_variable: str = "value",
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        super().__init__(target_variable)
        self.lags = lags or [1, 2, 3, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 30]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None
        self._model_lower = None
        self._model_upper = None
        self._feature_cols: list[str] | None = None
        self._last_known: pd.Series | None = None
        self.feature_importances_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> RandomForestModel:
        """Fit three Random Forest regressors (central, lower, upper)."""
        require("sklearn", feature="machine learning models")
        from sklearn.ensemble import RandomForestRegressor

        series = self._prepare_series(df)
        features = make_lag_features(series, self.lags, self.rolling_windows)
        self._feature_cols = [c for c in features.columns if c != "y"]

        x = features[self._feature_cols].values
        y = features["y"].values

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(x, y)

        # Approximate quantile forests for uncertainty bounds
        self._model_lower = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        self._model_upper = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        residuals = y - self._model.predict(x)
        sigma = residuals.std()
        self._model_lower.fit(x, y - 1.64 * sigma)
        self._model_upper.fit(x, y + 1.64 * sigma)

        self._last_known = series
        self._is_fitted = True

        self.feature_importances_ = pd.Series(
            self._model.feature_importances_, index=self._feature_cols
        ).sort_values(ascending=False)
        logger.info("RandomForestModel fitted: %d samples, %d features", len(y), len(self._feature_cols))
        return self

    def predict(self, horizon: int = 7, **kwargs) -> pd.DataFrame:
        """Recursive one-step-ahead forecast for *horizon* days."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        series = self._last_known.copy()
        predictions: list[tuple[float, float, float]] = []

        for _ in range(horizon):
            features = make_lag_features(series, self.lags, self.rolling_windows)
            if features.empty:
                break
            x = features[self._feature_cols].values[-1].reshape(1, -1)
            y_pred = float(self._model.predict(x)[0])
            y_lower = float(self._model_lower.predict(x)[0])
            y_upper = float(self._model_upper.predict(x)[0])
            predictions.append((y_pred, y_lower, y_upper))

            next_date = series.index[-1] + pd.Timedelta(days=1)
            series = pd.concat([series, pd.Series([y_pred], index=[next_date])])

        future_dates = self._future_dates(horizon)
        result = pd.DataFrame(
            predictions, index=future_dates[: len(predictions)], columns=["yhat", "yhat_lower", "yhat_upper"]
        )
        result.index.name = "datetime"
        return result

    def feature_importance(self) -> pd.Series:
        """Return feature importance scores (descending)."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model not fitted")
        return self.feature_importances_


class XGBoostModel(BaseHydroModel):
    """XGBoost gradient-boosted trees for hydrological forecasting.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step-size shrinkage.
    """

    MODEL_ID = "xgboost"
    SUPPORTS_UNCERTAINTY = False
    SUPPORTS_MULTIVARIATE = True

    def __init__(
        self,
        target_variable: str = "value",
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        super().__init__(target_variable)
        self.lags = lags or [1, 2, 3, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 30]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self._model = None
        self._feature_cols: list[str] | None = None
        self._last_known: pd.Series | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> XGBoostModel:
        """Fit XGBoost on lag-engineered features."""
        require("xgboost", feature="gradient boosting")
        from xgboost import XGBRegressor

        series = self._prepare_series(df)
        features = make_lag_features(series, self.lags, self.rolling_windows)
        self._feature_cols = [c for c in features.columns if c != "y"]

        x, y = features[self._feature_cols].values, features["y"].values

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._model.fit(x, y)
        self._last_known = series
        self._is_fitted = True
        logger.info("XGBoostModel fitted: %d samples", len(y))
        return self

    def predict(self, horizon: int = 7, **kwargs) -> pd.DataFrame:
        """Recursive one-step-ahead forecast for *horizon* days."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        series = self._last_known.copy()
        predictions: list[float] = []

        for _ in range(horizon):
            features = make_lag_features(series, self.lags, self.rolling_windows)
            if features.empty:
                break
            x = features[self._feature_cols].values[-1].reshape(1, -1)
            y_pred = float(self._model.predict(x)[0])
            predictions.append(y_pred)
            next_date = series.index[-1] + pd.Timedelta(days=1)
            series = pd.concat([series, pd.Series([y_pred], index=[next_date])])

        future_dates = self._future_dates(horizon)
        result = pd.DataFrame(
            {
                "yhat": predictions,
                "yhat_lower": [np.nan] * len(predictions),
                "yhat_upper": [np.nan] * len(predictions),
            },
            index=future_dates[: len(predictions)],
        )
        result.index.name = "datetime"
        return result


class IsolationForestModel(BaseHydroModel):
    """Isolation Forest for water-quality anomaly detection.

    Unsupervised — no labelled anomalies are needed.  Returns anomaly scores
    and binary labels for each observation.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies (0–0.5).
    """

    MODEL_ID = "isolation_forest"
    SUPPORTS_UNCERTAINTY = False
    SUPPORTS_MULTIVARIATE = True

    def __init__(
        self,
        target_variable: str = "value",
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        super().__init__(target_variable)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._series: pd.Series | None = None
        self._feature_cols: list[str] | None = None
        self._fitted_features: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> IsolationForestModel:
        """Fit the Isolation Forest on lag-engineered features."""
        require("sklearn", feature="anomaly detection")
        from sklearn.ensemble import IsolationForest

        series = self._prepare_series(df)
        self._series = series

        features = make_lag_features(series, lags=[1, 2, 3, 7], rolling_windows=[7, 30])
        self._feature_cols = [c for c in features.columns if c != "y"]
        x = features[self._feature_cols].values

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(x)
        self._fitted_features = features
        self._is_fitted = True
        logger.info("IsolationForestModel fitted on %d samples", len(x))
        return self

    def predict(self, horizon: int = 0, **kwargs) -> pd.DataFrame:
        """Return anomaly scores and labels for the fitted data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        features = self._fitted_features
        x = features[self._feature_cols].values

        scores = self._model.score_samples(x)
        labels = self._model.predict(x)

        result = pd.DataFrame(
            {
                "yhat": features["y"].values,
                "anomaly_score": -scores,
                "is_anomaly": labels == -1,
                "yhat_lower": np.nan,
                "yhat_upper": np.nan,
            },
            index=features.index,
        )
        result.index.name = "datetime"
        return result

    def get_anomalies(self) -> pd.DataFrame:
        """Return only the detected anomalous observations, sorted by score."""
        result = self.predict()
        return result[result["is_anomaly"]].sort_values("anomaly_score", ascending=False)
