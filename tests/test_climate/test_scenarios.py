"""Tests for aquascope.climate.scenarios — climate scenario analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquascope.climate.scenarios import (
    drought_frequency,
    idf_adjustment,
    return_period_shift,
    scenario_comparison,
    wet_spell_analysis,
)


class TestReturnPeriodShift:
    def setup_method(self):
        np.random.seed(42)
        # Historical: 30 years of daily data drawn from GEV
        hist_idx = pd.date_range("1970-01-01", periods=365 * 30, freq="D")
        self.hist = pd.Series(
            np.random.gumbel(loc=50, scale=10, size=len(hist_idx)),
            index=hist_idx,
        )
        # Future: shifted distribution (higher extremes)
        fut_idx = pd.date_range("2070-01-01", periods=365 * 30, freq="D")
        self.future = pd.Series(
            np.random.gumbel(loc=60, scale=12, size=len(fut_idx)),
            index=fut_idx,
        )

    def test_shift_factors_generally_greater_than_one(self):
        result = return_period_shift(self.hist, self.future)
        # Future distribution is higher, so most shift factors should be > 1
        above_one = sum(1 for f in result.shift_factors if f > 1.0)
        assert above_one >= len(result.shift_factors) // 2

    def test_default_return_periods(self):
        result = return_period_shift(self.hist, self.future)
        assert result.return_periods == [2, 5, 10, 25, 50, 100]

    def test_custom_return_periods(self):
        result = return_period_shift(self.hist, self.future, return_periods=[5, 50])
        assert len(result.return_periods) == 2
        assert len(result.hist_quantiles) == 2


class TestIDFAdjustment:
    def test_scalar_factor(self):
        intensities = np.array([10.0, 20.0, 30.0])
        durations = np.array([5, 15, 60])
        result = idf_adjustment(intensities, durations, future_factor=1.2)
        np.testing.assert_allclose(result, [12.0, 24.0, 36.0])

    def test_array_factor(self):
        intensities = np.array([10.0, 20.0])
        durations = np.array([5, 60])
        factors = np.array([1.1, 1.3])
        result = idf_adjustment(intensities, durations, future_factor=factors)
        np.testing.assert_allclose(result, [11.0, 26.0])


class TestDroughtFrequency:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
        # Create series with dry spells: mostly ~5 mm, but insert drought
        vals = np.random.uniform(3, 8, len(idx))
        # Force a drought: 30 consecutive very low values
        vals[100:130] = 0.5
        vals[300:320] = 0.3
        self.precip = pd.Series(vals, index=idx)

    def test_detects_drought_events(self):
        result = drought_frequency(self.precip, threshold_percentile=10.0)
        assert result.n_events >= 2

    def test_max_duration_reasonable(self):
        result = drought_frequency(self.precip, threshold_percentile=10.0)
        assert result.max_duration >= 20

    def test_total_deficit_positive(self):
        result = drought_frequency(self.precip, threshold_percentile=10.0)
        assert result.total_deficit > 0


class TestWetSpellAnalysis:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        vals = np.random.exponential(2.0, 365)
        # Force a wet spell
        vals[50:60] = 15.0
        self.precip = pd.Series(vals, index=idx)

    def test_detects_wet_spells(self):
        result = wet_spell_analysis(self.precip, threshold_mm=1.0)
        assert result.n_spells > 0

    def test_max_duration_at_least_10(self):
        result = wet_spell_analysis(self.precip, threshold_mm=1.0)
        assert result.max_duration >= 10

    def test_mean_intensity_positive(self):
        result = wet_spell_analysis(self.precip, threshold_mm=1.0)
        assert result.mean_intensity > 0


class TestScenarioComparison:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2070-01-01", periods=365, freq="D")
        self.baseline = pd.Series(np.random.normal(20, 3, 365), index=idx)
        self.scenarios = {
            "ssp126": pd.Series(np.random.normal(21, 3, 365), index=idx),
            "ssp585": pd.Series(np.random.normal(24, 3, 365), index=idx),
        }

    def test_returns_dataframe(self):
        result = scenario_comparison(self.scenarios, self.baseline)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_columns_present(self):
        result = scenario_comparison(self.scenarios, self.baseline)
        expected_cols = {"scenario", "baseline_value", "scenario_value", "absolute_change", "percent_change"}
        assert expected_cols.issubset(set(result.columns))

    def test_ssp585_larger_change(self):
        result = scenario_comparison(self.scenarios, self.baseline)
        ssp126_row = result[result["scenario"] == "ssp126"].iloc[0]
        ssp585_row = result[result["scenario"] == "ssp585"].iloc[0]
        assert abs(ssp585_row["absolute_change"]) > abs(ssp126_row["absolute_change"])

    def test_invalid_metric_raises(self):
        try:
            scenario_comparison(self.scenarios, self.baseline, metric="invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass
