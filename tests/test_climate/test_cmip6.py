"""Tests for aquascope.climate.cmip6 — CMIP6 data processing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquascope.climate.cmip6 import SSP, CMIP6Processor


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="MS")


class TestSSP:
    def test_values(self):
        assert SSP.SSP126.value == "ssp126"
        assert SSP.SSP585.value == "ssp585"

    def test_descriptions_non_empty(self):
        for ssp in SSP:
            assert len(ssp.description) > 10


class TestComputeEnsembleStats:
    def setup_method(self):
        np.random.seed(0)
        idx = _monthly_index("2015-01", 120)
        self.models = {
            f"model_{i}": pd.DataFrame(
                {"tas": np.random.normal(288 + i * 0.5, 2, 120)}, index=idx
            )
            for i in range(5)
        }
        self.proc = CMIP6Processor("tas")

    def test_returns_correct_n_models(self):
        result = self.proc.compute_ensemble_stats(self.models)
        assert result.n_models == 5

    def test_mean_within_model_range(self):
        result = self.proc.compute_ensemble_stats(self.models)
        assert result.mean.mean() > 287
        assert result.mean.mean() < 292

    def test_p10_less_than_p90(self):
        result = self.proc.compute_ensemble_stats(self.models)
        assert (result.p10 <= result.p90).all()

    def test_empty_models_raises(self):
        try:
            self.proc.compute_ensemble_stats({})
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestTimeSlice:
    def setup_method(self):
        idx = _monthly_index("1950-01", 1800)  # 150 years
        self.data = pd.DataFrame({"tas": np.arange(1800, dtype=float)}, index=idx)
        self.proc = CMIP6Processor("tas")

    def test_historical_range(self):
        result = self.proc.time_slice(self.data, "historical")
        assert result.index.year.min() == 1950
        assert result.index.year.max() == 2014

    def test_end_century_range(self):
        result = self.proc.time_slice(self.data, "end_century")
        assert result.index.year.min() == 2071
        assert result.index.year.max() == 2099  # data ends before 2100 Dec

    def test_custom_period(self):
        result = self.proc.time_slice(self.data, "2000-2020")
        assert result.index.year.min() == 2000
        assert result.index.year.max() == 2020

    def test_invalid_period_raises(self):
        try:
            self.proc.time_slice(self.data, "invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestComputeAnomaly:
    def setup_method(self):
        idx = _monthly_index("1970-01", 600)  # 50 years
        self.data = pd.DataFrame({"tas": np.ones(600) * 15.0}, index=idx)
        self.proc = CMIP6Processor("tas")

    def test_anomaly_is_zero_for_constant(self):
        result = self.proc.compute_anomaly(self.data)
        assert np.allclose(result.values, 0.0)

    def test_anomaly_with_offset(self):
        data = self.data.copy()
        # Add 2°C after 2010
        mask = data.index.year >= 2011
        data.loc[mask, "tas"] = 17.0
        result = self.proc.compute_anomaly(data)
        # Baseline (1981-2010) mean is 15.0, so post-2010 anomaly ≈ 2.0
        assert abs(result.loc[mask].values.mean() - 2.0) < 0.01

    def test_empty_baseline_raises(self):
        try:
            self.proc.compute_anomaly(self.data, baseline_period=(2200, 2300))
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestAnnualCycle:
    def setup_method(self):
        idx = _monthly_index("2000-01", 120)
        # Create a seasonal pattern
        months = np.tile(np.arange(12), 10)
        self.data = pd.DataFrame({"tas": months.astype(float)}, index=idx)
        self.proc = CMIP6Processor("tas")

    def test_returns_12_rows(self):
        result = self.proc.annual_cycle(self.data)
        assert len(result) == 12

    def test_values_match_month_index(self):
        result = self.proc.annual_cycle(self.data)
        # Month 1 (Jan) → value 0.0, Month 7 (Jul) → value 6.0
        assert abs(result.iloc[0, 0] - 0.0) < 1e-10
        assert abs(result.iloc[6, 0] - 6.0) < 1e-10


class TestTrendAnalysis:
    def setup_method(self):
        np.random.seed(42)
        idx = _monthly_index("2000-01", 240)
        trend = np.linspace(0, 2, 240)  # 2°C over 20 years = 1°C/decade
        noise = np.random.normal(0, 0.1, 240)
        self.data = pd.DataFrame({"tas": 15.0 + trend + noise}, index=idx)
        self.proc = CMIP6Processor("tas")

    def test_positive_slope(self):
        result = self.proc.trend_analysis(self.data)
        assert result.slope > 0

    def test_significant_trend(self):
        result = self.proc.trend_analysis(self.data)
        assert result.p_value < 0.05

    def test_unit_per_decade_reasonable(self):
        result = self.proc.trend_analysis(self.data)
        # Expect ~1°C/decade
        assert 0.5 < result.unit_per_decade < 1.5

    def test_ci_contains_slope(self):
        result = self.proc.trend_analysis(self.data)
        assert result.ci_lower <= result.slope <= result.ci_upper
