"""Tests for well data analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.groundwater.wells import (
    HydrographResult,
    RecessionResult,
    SeasonalResult,
    WellTrendResult,
    recession_analysis,
    seasonal_decomposition,
    storage_coefficient,
    trend_detection,
    well_hydrograph,
)


def _daily_index(n: int = 365, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


class TestWellHydrograph:
    def setup_method(self):
        self.idx = _daily_index(365)
        self.levels = pd.Series(
            10.0 + np.sin(2 * np.pi * np.arange(365) / 365) * 2.0,
            index=self.idx,
        )

    def test_returns_hydrograph_result(self):
        result = well_hydrograph(self.levels)
        assert isinstance(result, HydrographResult)

    def test_stats_keys(self):
        result = well_hydrograph(self.levels)
        for key in ("mean", "median", "std", "min", "max", "range", "count"):
            assert key in result.stats

    def test_correlation_with_precip(self):
        precip = pd.Series(np.random.default_rng(42).uniform(0, 10, 365), index=self.idx)
        result = well_hydrograph(self.levels, precip=precip)
        assert result.correlation is not None
        assert -1.0 <= result.correlation <= 1.0

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty"):
            well_hydrograph(pd.Series([], dtype=float))


class TestTrendDetection:
    def setup_method(self):
        self.idx = _daily_index(365)

    def test_increasing_trend(self):
        levels = pd.Series(np.linspace(5, 15, 365), index=self.idx)
        result = trend_detection(levels)
        assert isinstance(result, WellTrendResult)
        assert result.trend == "increasing"
        assert result.slope > 0

    def test_decreasing_trend(self):
        levels = pd.Series(np.linspace(15, 5, 365), index=self.idx)
        result = trend_detection(levels)
        assert result.trend == "decreasing"
        assert result.slope < 0

    def test_no_trend_in_noise(self):
        np.random.seed(99)
        levels = pd.Series(np.random.randn(365) * 0.01 + 10.0, index=self.idx)
        result = trend_detection(levels)
        # Should have small slope and high p-value
        assert abs(result.slope) < 0.1

    def test_method_recorded(self):
        levels = pd.Series(np.linspace(1, 10, 50), index=_daily_index(50))
        result = trend_detection(levels, method="mann_kendall")
        assert result.method == "mann_kendall"

    def test_unknown_method_raises(self):
        levels = pd.Series([1.0, 2.0, 3.0], index=_daily_index(3))
        with pytest.raises(ValueError, match="Unknown method"):
            trend_detection(levels, method="invalid")

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            trend_detection(pd.Series([1.0, 2.0], index=_daily_index(2)))


class TestSeasonalDecomposition:
    def setup_method(self):
        n = 730  # 2 years of daily data
        self.idx = _daily_index(n)
        t = np.arange(n)
        trend = 10.0 + 0.005 * t
        seasonal = 2.0 * np.sin(2 * np.pi * t / 365)
        self.levels = pd.Series(trend + seasonal, index=self.idx)

    def test_returns_seasonal_result(self):
        result = seasonal_decomposition(self.levels, period=365)
        assert isinstance(result, SeasonalResult)

    def test_components_sum_approximately_to_original(self):
        result = seasonal_decomposition(self.levels, period=365)
        reconstructed = result.trend + result.seasonal + result.residual
        # Only check where trend is not NaN
        valid = ~result.trend.isna()
        np.testing.assert_allclose(
            reconstructed.values[valid],
            self.levels.values[valid],
            atol=1e-10,
        )

    def test_seasonal_strength_positive(self):
        result = seasonal_decomposition(self.levels, period=365)
        assert result.strength > 0

    def test_short_series_raises(self):
        short = pd.Series([1.0, 2.0, 3.0], index=_daily_index(3))
        with pytest.raises(ValueError, match="must be >="):
            seasonal_decomposition(short, period=12)


class TestRecessionAnalysis:
    def setup_method(self):
        # Create a series with clear recession events
        n = 200
        self.idx = _daily_index(n)
        vals = np.zeros(n)
        # First recession: exponential decay from day 10 to day 60
        for i in range(10, 60):
            vals[i] = 20.0 * np.exp(-(i - 10) / 15.0)
        # Second recession: exponential decay from day 100 to day 150
        for i in range(100, 150):
            vals[i] = 15.0 * np.exp(-(i - 100) / 20.0)
        self.levels = pd.Series(vals, index=self.idx)

    def test_finds_recession_events(self):
        result = recession_analysis(self.levels, min_recession_days=5)
        assert isinstance(result, RecessionResult)
        assert len(result.events) >= 1

    def test_recession_constants_positive(self):
        result = recession_analysis(self.levels, min_recession_days=5)
        for tau in result.recession_constants:
            assert tau > 0

    def test_no_recession_raises(self):
        # Monotonically increasing series
        idx = _daily_index(50)
        levels = pd.Series(np.linspace(1, 50, 50), index=idx)
        with pytest.raises(ValueError, match="No recession"):
            recession_analysis(levels, min_recession_days=5)


class TestStorageCoefficient:
    def setup_method(self):
        self.idx = _daily_index(100)
        self.levels = pd.Series(np.zeros(100), index=self.idx)
        # Simulate a rise of 2m
        self.levels.iloc[:50] = 10.0
        self.levels.iloc[50:] = 12.0

    def test_estimates_sy(self):
        # recharge_events: (start_idx, end_idx, volume_m3)
        # Δh = 2m, area = 10 km² = 1e7 m², volume = 0.2 * 2 * 1e7 = 4e6
        events = [(40, 60, 4_000_000.0)]
        sy = storage_coefficient(self.levels, events, area_km2=10.0)
        assert 0.0 < sy < 1.0
        np.testing.assert_allclose(sy, 0.2, atol=0.01)

    def test_empty_events_raises(self):
        with pytest.raises(ValueError, match="No recharge events"):
            storage_coefficient(self.levels, [], area_km2=10.0)
