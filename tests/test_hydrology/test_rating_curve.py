"""Tests for aquascope.hydrology.rating_curve module.

Tests power-law rating curve fitting, segmented curves, prediction,
uncertainty estimation, shift detection, cross-validation, and export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_power_law_data(
    a: float = 2.0,
    b: float = 1.5,
    h0: float = 0.5,
    n: int = 50,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Q = a * (H - h0)^b data with optional noise."""
    rng = np.random.default_rng(seed)
    stage = np.linspace(h0 + 0.1, h0 + 5.0, n)
    discharge = a * (stage - h0) ** b
    if noise_std > 0:
        discharge += rng.normal(0, noise_std, n)
        discharge = np.maximum(discharge, 0.01)
    return stage, discharge


class TestFitRatingCurve:
    """Tests for fit_rating_curve."""

    def test_fit_basic_power_law(self):
        """Known Q = 2*(H-0.5)^1.5 — recover parameters."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage, discharge = _make_power_law_data(a=2.0, b=1.5, h0=0.5, noise_std=0.0)
        result = fit_rating_curve(stage, discharge)

        assert result.n_points == 50
        assert result.r_squared > 0.99
        assert result.rmse < 0.5
        np.testing.assert_allclose(result.a, 2.0, rtol=0.1)
        np.testing.assert_allclose(result.b, 1.5, rtol=0.1)
        np.testing.assert_allclose(result.h0, 0.5, atol=0.15)

    def test_fit_with_h0_fixed(self):
        """Provide h0, check that a and b are recovered."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage, discharge = _make_power_law_data(a=3.0, b=2.0, h0=1.0, noise_std=0.0)
        result = fit_rating_curve(stage, discharge, h0=1.0)

        np.testing.assert_allclose(result.a, 3.0, rtol=0.05)
        np.testing.assert_allclose(result.b, 2.0, rtol=0.05)
        assert result.h0 == 1.0
        assert result.r_squared > 0.999

    def test_fit_with_h0_estimated(self):
        """Let h0 be optimised — it should be close to the true value."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage, discharge = _make_power_law_data(a=2.0, b=1.5, h0=0.5, n=80, noise_std=0.01)
        result = fit_rating_curve(stage, discharge)

        np.testing.assert_allclose(result.h0, 0.5, atol=0.2)
        assert result.r_squared > 0.99

    def test_fit_with_noisy_data(self):
        """Moderate noise — result should still be reasonable."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage, discharge = _make_power_law_data(a=2.0, b=1.5, h0=0.5, n=100, noise_std=0.5)
        result = fit_rating_curve(stage, discharge)

        assert result.r_squared > 0.9
        assert result.n_points == 100
        assert len(result.residuals) == 100

    def test_stage_range(self):
        """stage_range should match the input data range."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage, discharge = _make_power_law_data()
        result = fit_rating_curve(stage, discharge)

        np.testing.assert_allclose(result.stage_range[0], np.min(stage), atol=1e-10)
        np.testing.assert_allclose(result.stage_range[1], np.max(stage), atol=1e-10)


class TestPrediction:
    """Tests for predict_discharge and predict_stage."""

    def setup_method(self):
        from aquascope.hydrology.rating_curve import fit_rating_curve

        self.stage, self.discharge = _make_power_law_data(a=2.0, b=1.5, h0=0.5, noise_std=0.0)
        self.result = fit_rating_curve(self.stage, self.discharge)

    def test_predict_discharge(self):
        """Round-trip: predicted discharge should match input."""
        from aquascope.hydrology.rating_curve import predict_discharge

        predicted = predict_discharge(self.result, self.stage)
        np.testing.assert_allclose(predicted, self.discharge, rtol=0.05)

    def test_predict_stage(self):
        """Inverse prediction should recover original stage values."""
        from aquascope.hydrology.rating_curve import predict_stage

        predicted_stage = predict_stage(self.result, self.discharge)
        np.testing.assert_allclose(predicted_stage, self.stage, rtol=0.05)

    def test_predict_discharge_pandas_series(self):
        """predict_discharge should accept a pandas Series."""
        from aquascope.hydrology.rating_curve import predict_discharge

        stage_series = pd.Series(self.stage)
        predicted = predict_discharge(self.result, stage_series)
        assert isinstance(predicted, np.ndarray)
        assert len(predicted) == len(self.stage)

    def test_predict_stage_roundtrip(self):
        """stage → discharge → stage should be identity."""
        from aquascope.hydrology.rating_curve import predict_discharge, predict_stage

        q = predict_discharge(self.result, self.stage)
        h_back = predict_stage(self.result, q)
        np.testing.assert_allclose(h_back, self.stage, rtol=0.05)


class TestSegmentedCurve:
    """Tests for fit_segmented_rating_curve."""

    def _make_two_segment_data(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Create data with two distinct power-law segments."""
        bp = 3.0
        h0 = 0.0
        # Low segment: Q = 1.0 * H^1.2
        stage_lo = np.linspace(0.5, bp, 25)
        q_lo = 1.0 * (stage_lo - h0) ** 1.2
        # High segment: Q = 0.5 * H^2.0
        stage_hi = np.linspace(bp, 6.0, 25)
        q_hi = 0.5 * (stage_hi - h0) ** 2.0
        stage = np.concatenate([stage_lo, stage_hi])
        discharge = np.concatenate([q_lo, q_hi])
        return stage, discharge, bp

    def test_segmented_curve(self):
        """Fit 2-segment curve and verify each segment."""
        from aquascope.hydrology.rating_curve import fit_segmented_rating_curve

        stage, discharge, bp = self._make_two_segment_data()
        result = fit_segmented_rating_curve(stage, discharge, n_segments=2, breakpoints=[bp])

        assert result.segments is not None
        assert len(result.segments) == 2
        assert result.r_squared > 0.95
        assert result.segments[0].stage_max <= result.segments[1].stage_min + 0.01

    def test_segmented_with_given_breakpoints(self):
        """Explicit breakpoints should be used as-is."""
        from aquascope.hydrology.rating_curve import fit_segmented_rating_curve

        stage, discharge, bp = self._make_two_segment_data()
        result = fit_segmented_rating_curve(stage, discharge, n_segments=2, breakpoints=[bp])

        assert result.segments is not None
        assert result.segments[0].stage_max == pytest.approx(bp)

    def test_segmented_predict(self):
        """Segmented curve prediction should use appropriate segments."""
        from aquascope.hydrology.rating_curve import fit_segmented_rating_curve, predict_discharge

        stage, discharge, bp = self._make_two_segment_data()
        result = fit_segmented_rating_curve(stage, discharge, n_segments=2, breakpoints=[bp])

        predicted = predict_discharge(result, stage)
        np.testing.assert_allclose(predicted, discharge, rtol=0.20)


class TestUncertainty:
    """Tests for rating_curve_uncertainty."""

    def test_uncertainty_bands(self):
        """Lower < predicted < upper for all points."""
        from aquascope.hydrology.rating_curve import (
            fit_rating_curve,
            predict_discharge,
            rating_curve_uncertainty,
        )

        stage, discharge = _make_power_law_data(noise_std=0.3, n=60)
        result = fit_rating_curve(stage, discharge)
        predicted = predict_discharge(result, stage)
        lower, upper = rating_curve_uncertainty(result, stage, confidence=0.95)

        assert np.all(lower <= predicted + 1e-10)
        assert np.all(upper >= predicted - 1e-10)
        assert np.all(upper > lower)


class TestShiftDetection:
    """Tests for detect_rating_shift."""

    def test_detect_shift(self):
        """Create data with a known shift and verify detection."""
        from aquascope.hydrology.rating_curve import detect_rating_shift

        rng = np.random.default_rng(123)
        n = 100
        timestamps = pd.date_range("2020-01-01", periods=n, freq="D")

        # First half: Q = 2 * H^1.5
        stage = rng.uniform(1.0, 5.0, n)
        discharge = np.empty(n)
        discharge[:50] = 2.0 * stage[:50] ** 1.5 + rng.normal(0, 0.1, 50)
        # Second half: Q = 5 * H^1.5 (shift in 'a')
        discharge[50:] = 5.0 * stage[50:] ** 1.5 + rng.normal(0, 0.1, 50)
        discharge = np.maximum(discharge, 0.01)

        shifts = detect_rating_shift(stage, discharge, timestamps, window_size=20)

        assert isinstance(shifts, list)
        assert len(shifts) > 0
        for s in shifts:
            assert "timestamp" in s
            assert "shift_magnitude" in s
            assert "p_value" in s

    def test_no_shift_in_stable_data(self):
        """Stable data should produce few or no shifts."""
        from aquascope.hydrology.rating_curve import detect_rating_shift

        n = 100
        timestamps = pd.date_range("2020-01-01", periods=n, freq="D")
        stage = np.linspace(1.0, 5.0, n)
        discharge = 2.0 * stage**1.5

        shifts = detect_rating_shift(stage, discharge, timestamps, window_size=20)
        assert isinstance(shifts, list)


class TestCrossValidation:
    """Tests for cross_validate_rating."""

    def test_cross_validate(self):
        """Verify returns dict with expected keys."""
        from aquascope.hydrology.rating_curve import cross_validate_rating

        stage, discharge = _make_power_law_data(noise_std=0.2, n=60)
        result = cross_validate_rating(stage, discharge, k_folds=5)

        assert isinstance(result, dict)
        assert "mean_rmse" in result
        assert "std_rmse" in result
        assert "mean_r2" in result
        assert "fold_results" in result
        assert len(result["fold_results"]) == 5
        assert result["mean_rmse"] >= 0
        assert result["mean_r2"] > 0.5


class TestExportHecRas:
    """Tests for export_hec_ras."""

    def test_export_hec_ras(self):
        """Export to file and verify content."""
        from pathlib import Path

        from aquascope.hydrology.rating_curve import export_hec_ras, fit_rating_curve

        stage, discharge = _make_power_law_data()
        result = fit_rating_curve(stage, discharge)

        output_path = Path("test_hec_ras_output.txt")
        try:
            export_hec_ras(result, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            lines = content.strip().split("\n")

            # Header lines start with #
            header_lines = [line for line in lines if line.startswith("#")]
            assert len(header_lines) >= 4

            # Data lines
            data_lines = [line for line in lines if not line.startswith("#") and line.strip()]
            assert len(data_lines) == 50

            # Each data line should have two numeric columns
            for line in data_lines:
                parts = line.split()
                assert len(parts) == 2
                float(parts[0])  # should not raise
                float(parts[1])  # should not raise
        finally:
            if output_path.exists():
                output_path.unlink()


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_insufficient_data(self):
        """Fewer than 5 points should raise ValueError."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage = np.array([1.0, 2.0, 3.0, 4.0])
        discharge = np.array([1.0, 4.0, 9.0, 16.0])

        with pytest.raises(ValueError, match="At least 5"):
            fit_rating_curve(stage, discharge)

    def test_negative_discharge_raises(self):
        """Negative discharge should raise ValueError."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        discharge = np.array([1.0, 4.0, -1.0, 16.0, 25.0])

        with pytest.raises(ValueError, match="non-negative"):
            fit_rating_curve(stage, discharge)

    def test_nan_in_stage_raises(self):
        """NaN in stage should raise ValueError."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        discharge = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

        with pytest.raises(ValueError, match="NaN"):
            fit_rating_curve(stage, discharge)

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        from aquascope.hydrology.rating_curve import fit_rating_curve

        stage = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        discharge = np.array([1.0, 4.0, 9.0])

        with pytest.raises(ValueError, match="equal length"):
            fit_rating_curve(stage, discharge)
