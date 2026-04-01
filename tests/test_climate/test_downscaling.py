"""Tests for aquascope.climate.downscaling — statistical downscaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquascope.climate.downscaling import (
    bias_correction,
    delta_method,
    evaluate_downscaling,
    quantile_delta_mapping,
    quantile_mapping,
)


def _daily_series(mean: float, std: float, n: int, start: str) -> pd.Series:
    return pd.Series(
        np.random.normal(mean, std, n),
        index=pd.date_range(start, periods=n, freq="D"),
    )


class TestDeltaMethodAdditive:
    def setup_method(self):
        np.random.seed(42)
        self.obs = _daily_series(20.0, 3.0, 365, "2000-01-01")
        self.gcm_hist = _daily_series(22.0, 4.0, 365, "2000-01-01")
        self.gcm_future = _daily_series(25.0, 4.0, 365, "2050-01-01")

    def test_additive_shifts_mean(self):
        result = delta_method(self.obs, self.gcm_hist, self.gcm_future)
        expected_delta = self.gcm_future.mean() - self.gcm_hist.mean()
        assert abs(result.mean() - (self.obs.mean() + expected_delta)) < 0.01

    def test_preserves_variance(self):
        result = delta_method(self.obs, self.gcm_hist, self.gcm_future)
        assert abs(result.std() - self.obs.std()) < 0.01


class TestDeltaMethodMultiplicative:
    def setup_method(self):
        np.random.seed(42)
        self.obs = pd.Series(
            np.random.gamma(2.0, 5.0, 365),
            index=pd.date_range("2000-01-01", periods=365, freq="D"),
        )
        self.gcm_hist = pd.Series(
            np.random.gamma(2.0, 4.0, 365),
            index=pd.date_range("2000-01-01", periods=365, freq="D"),
        )
        self.gcm_future = pd.Series(
            np.random.gamma(2.0, 6.0, 365),
            index=pd.date_range("2050-01-01", periods=365, freq="D"),
        )

    def test_multiplicative_scales_mean(self):
        result = delta_method(self.obs, self.gcm_hist, self.gcm_future, method="multiplicative")
        factor = self.gcm_future.mean() / self.gcm_hist.mean()
        assert abs(result.mean() - self.obs.mean() * factor) < 0.1

    def test_invalid_method_raises(self):
        try:
            delta_method(self.obs, self.gcm_hist, self.gcm_future, method="invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestQuantileMapping:
    def setup_method(self):
        np.random.seed(42)
        self.obs = _daily_series(20.0, 3.0, 365, "2000-01-01")
        # GCM with +2 bias in mean and wider spread
        self.gcm_hist = _daily_series(22.0, 4.0, 365, "2000-01-01")
        self.gcm_future = _daily_series(24.0, 4.0, 365, "2050-01-01")

    def test_reduces_mean_bias(self):
        result = quantile_mapping(self.obs, self.gcm_hist, self.gcm_future)
        # Corrected future should be closer to obs distribution + climate signal
        raw_bias = abs(self.gcm_future.mean() - self.obs.mean())
        corrected_bias = abs(result.mean() - self.obs.mean())
        assert corrected_bias < raw_bias

    def test_output_length_matches_future(self):
        result = quantile_mapping(self.obs, self.gcm_hist, self.gcm_future)
        assert len(result) == len(self.gcm_future)

    def test_preserves_index(self):
        result = quantile_mapping(self.obs, self.gcm_hist, self.gcm_future)
        assert (result.index == self.gcm_future.index).all()


class TestQuantileDeltaMapping:
    def setup_method(self):
        np.random.seed(42)
        self.obs = _daily_series(20.0, 3.0, 365, "2000-01-01")
        self.gcm_hist = _daily_series(22.0, 4.0, 365, "2000-01-01")
        self.gcm_future = _daily_series(24.0, 4.0, 365, "2050-01-01")

    def test_qdm_output_length(self):
        result = quantile_delta_mapping(self.obs, self.gcm_hist, self.gcm_future)
        assert len(result) == len(self.gcm_future)

    def test_qdm_reduces_bias(self):
        result = quantile_delta_mapping(self.obs, self.gcm_hist, self.gcm_future)
        raw_bias = abs(self.gcm_future.mean() - self.obs.mean())
        corrected_bias = abs(result.mean() - self.obs.mean())
        assert corrected_bias < raw_bias


class TestBiasCorrection:
    def setup_method(self):
        np.random.seed(42)
        n = 730
        self.obs = _daily_series(20.0, 3.0, 365, "2000-01-01")
        self.gcm = pd.Series(
            np.random.normal(22.0, 4.0, n),
            index=pd.date_range("2000-01-01", periods=n, freq="D"),
        )

    def test_quantile_mapping_method(self):
        result = bias_correction(self.gcm, self.obs, method="quantile_mapping")
        assert len(result) == len(self.gcm) // 2

    def test_delta_method(self):
        result = bias_correction(self.gcm, self.obs, method="delta")
        assert len(result) == len(self.obs)

    def test_unknown_method_raises(self):
        try:
            bias_correction(self.gcm, self.obs, method="unknown")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestEvaluateDownscaling:
    def setup_method(self):
        np.random.seed(42)
        self.obs = _daily_series(20.0, 3.0, 100, "2000-01-01")
        self.good = self.obs + np.random.normal(0, 0.5, 100)
        self.bad = self.obs + 5.0  # systematic +5 bias

    def test_good_correction_low_rmse(self):
        metrics = evaluate_downscaling(self.obs, self.good)
        assert metrics.rmse < 1.0

    def test_bad_correction_high_bias(self):
        metrics = evaluate_downscaling(self.obs, self.bad)
        assert abs(metrics.bias - 5.0) < 0.01

    def test_correlation_near_one_for_offset(self):
        metrics = evaluate_downscaling(self.obs, self.bad)
        assert metrics.correlation > 0.99

    def test_percentile_errors_present(self):
        metrics = evaluate_downscaling(self.obs, self.good)
        assert 50 in metrics.percentile_errors
        assert 95 in metrics.percentile_errors

    def test_length_mismatch_raises(self):
        try:
            evaluate_downscaling(self.obs, self.obs.iloc[:50])
            assert False, "Expected ValueError"
        except ValueError:
            pass
