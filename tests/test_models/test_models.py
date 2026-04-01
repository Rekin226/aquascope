"""Tests for aquascope.models — base, statistical, ML, and LSTM models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ts(n: int = 365, freq: str = "D", start: str = "2020-01-01") -> pd.DataFrame:
    """Create a synthetic daily time-series with seasonality and noise."""
    idx = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    values = 100 + 10 * np.sin(2 * np.pi * t / 365) + np.random.default_rng(42).normal(0, 2, n)
    return pd.DataFrame({"value": values}, index=idx)


class TestModelMap:
    def test_get_model_map_lazy_loads(self):
        from aquascope.models import get_model_map

        m = get_model_map()
        assert "prophet" in m or "arima" in m or "random_forest" in m
        assert len(m) >= 5

    def test_model_map_values_are_classes(self):
        from aquascope.models import get_model_map
        from aquascope.models.base import BaseHydroModel

        for cls in get_model_map().values():
            assert issubclass(cls, BaseHydroModel)


class TestBaseHydroModel:
    def test_metrics(self):
        from aquascope.models.base import BaseHydroModel

        obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
        metrics = BaseHydroModel._compute_metrics(obs, pred)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "nse" in metrics
        assert "kge" in metrics
        assert metrics["r2"] > 0.95
        assert metrics["nse"] > 0.95


class TestARIMAModel:
    def test_fit_predict(self):
        from aquascope.models.statistical import ARIMAModel

        df = _make_ts(200)
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(df)
        forecast = model.predict(horizon=7)
        assert len(forecast) == 7
        assert "yhat" in forecast.columns

    def test_evaluate(self):
        from aquascope.models.statistical import ARIMAModel

        df = _make_ts(200)
        model = ARIMAModel()
        model.fit(df)
        metrics = model.evaluate(df)
        assert "rmse" in metrics


class TestSPIModel:
    def test_fit_predict(self):
        from aquascope.models.statistical import SPIModel

        df = _make_ts(730)  # 2 years of daily data
        model = SPIModel(timescales=[1, 3])
        model.fit(df)
        result = model.predict()
        assert "SPI_1" in result.columns
        assert "SPI_3" in result.columns
        assert "drought_category" in result.columns

    def test_categorise(self):
        from aquascope.models.statistical import SPIModel

        assert SPIModel._categorise(-2.5) == "extremely_dry"
        assert SPIModel._categorise(-1.3) == "moderately_dry"
        assert SPIModel._categorise(0.5) == "normal"
        assert SPIModel._categorise(2.5) == "extremely_wet"


class TestRandomForestModel:
    def test_fit_predict(self):
        from aquascope.models.ml import RandomForestModel

        df = _make_ts(200)
        model = RandomForestModel(lags=[1, 2, 3, 7], n_estimators=10)
        model.fit(df)
        forecast = model.predict(horizon=5)
        assert len(forecast) == 5
        assert "yhat" in forecast.columns

    def test_feature_importance(self):
        from aquascope.models.ml import RandomForestModel

        df = _make_ts(200)
        model = RandomForestModel(lags=[1, 2, 3, 7], n_estimators=10)
        model.fit(df)
        fi = model.feature_importance()
        assert len(fi) > 0


class TestXGBoostModel:
    def test_fit_predict(self):
        from aquascope.models.ml import XGBoostModel

        df = _make_ts(200)
        model = XGBoostModel(lags=[1, 2, 3, 7], n_estimators=10)
        try:
            model.fit(df)
        except Exception:
            pytest.skip("XGBoost not available (libomp missing)")
        forecast = model.predict(horizon=5)
        assert len(forecast) == 5


class TestIsolationForestModel:
    def test_fit_and_anomalies(self):
        from aquascope.models.ml import IsolationForestModel

        df = _make_ts(200)
        # Inject clear anomalies
        df.iloc[50, 0] = 999.0
        df.iloc[100, 0] = -999.0

        model = IsolationForestModel(contamination=0.05)
        model.fit(df)
        anomalies = model.get_anomalies()
        assert len(anomalies) > 0
        assert "anomaly_score" in anomalies.columns


class TestMakeLagFeatures:
    def test_basic(self):
        from aquascope.models.ml import make_lag_features

        df = _make_ts(100)
        features = make_lag_features(df["value"], lags=[1, 2, 3, 4, 5])
        assert "y" in features.columns
        assert features.shape[1] >= 5
        assert len(features) > 0
