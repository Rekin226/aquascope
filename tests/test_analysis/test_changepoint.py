"""Tests for the change-point detection module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from aquascope.analysis.changepoint import (
    ChangePoint,
    ChangePointResult,
    binary_segmentation,
    cusum,
    mann_whitney_test,
    pelt,
    pettitt_test,
    plot_changepoints,
    regime_shift_detector,
)


def _step_data(n1: int = 100, n2: int = 100, mean1: float = 0.0, mean2: float = 5.0, seed: int = 42) -> np.ndarray:
    """Create synthetic data with a single mean shift."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(mean1, 1.0, n1), rng.normal(mean2, 1.0, n2)])


def _multi_step_data(seed: int = 42) -> np.ndarray:
    """Create synthetic data with three segments (two changepoints)."""
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(0.0, 0.5, 100)
    seg2 = rng.normal(5.0, 0.5, 100)
    seg3 = rng.normal(2.0, 0.5, 100)
    return np.concatenate([seg1, seg2, seg3])


def _stationary_data(n: int = 200, seed: int = 42) -> np.ndarray:
    """Create stationary data with no changepoints."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n)


class TestPELT:
    def test_pelt_single_changepoint(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=5.0)
        result = pelt(data, min_segment_length=10)
        assert isinstance(result, ChangePointResult)
        assert result.method == "pelt"
        assert result.n_changepoints >= 1
        # The detected changepoint should be near index 100
        indices = [cp.index for cp in result.changepoints]
        assert any(80 <= idx <= 120 for idx in indices)

    def test_pelt_multiple_changepoints(self):
        data = _multi_step_data()
        result = pelt(data, min_segment_length=10)
        assert result.n_changepoints >= 2
        indices = sorted(cp.index for cp in result.changepoints)
        # Should find changepoints near 100 and 200
        assert any(80 <= idx <= 120 for idx in indices)
        assert any(180 <= idx <= 220 for idx in indices)

    def test_pelt_no_changepoint(self):
        data = _stationary_data(200)
        result = pelt(data, min_segment_length=20)
        assert result.n_changepoints == 0
        assert len(result.segments) == 1

    def test_pelt_penalty_effect(self):
        data = _multi_step_data()
        result_low = pelt(data, penalty=1.0, min_segment_length=10)
        result_high = pelt(data, penalty=100.0, min_segment_length=10)
        assert result_high.n_changepoints <= result_low.n_changepoints

    def test_pelt_min_segment_length(self):
        # With a large min_segment_length, a short anomalous segment cannot be
        # isolated as its own segment (both of its boundaries can't be CPs).
        rng = np.random.default_rng(99)
        data = np.concatenate([
            rng.normal(0.0, 0.3, 100),
            rng.normal(10.0, 0.3, 5),  # very short segment
            rng.normal(0.0, 0.3, 100),
        ])
        result = pelt(data, min_segment_length=20)
        # The short segment is only 5 points, so PELT cannot place CPs at
        # *both* boundaries (~100 and ~105) since that would create a segment
        # shorter than min_segment_length.
        cp_indices = sorted(cp.index for cp in result.changepoints)
        pairs_around_blip = [idx for idx in cp_indices if 95 <= idx <= 110]
        assert len(pairs_around_blip) <= 1, (
            f"Short segment should not be isolated: found CPs at {pairs_around_blip}"
        )


class TestCUSUM:
    def test_cusum_detects_shift(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=5.0)
        result = cusum(data)
        assert isinstance(result, ChangePointResult)
        assert result.method == "cusum"
        assert result.n_changepoints >= 1

    def test_cusum_no_shift(self):
        # Truly constant data should produce zero changepoints
        data = np.ones(200)
        result = cusum(data)
        assert result.n_changepoints == 0
        assert len(result.segments) == 1

    def test_cusum_threshold_effect(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=3.0)
        result_low = cusum(data, threshold=1.0)
        result_high = cusum(data, threshold=100.0)
        assert result_high.n_changepoints <= result_low.n_changepoints


class TestBinarySegmentation:
    def test_binary_segmentation_single(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=5.0)
        result = binary_segmentation(data, max_changepoints=5, min_segment_length=10, significance=0.05)
        assert isinstance(result, ChangePointResult)
        assert result.method == "binary_segmentation"
        assert result.n_changepoints >= 1
        indices = [cp.index for cp in result.changepoints]
        assert any(80 <= idx <= 120 for idx in indices)

    def test_binary_segmentation_max_cp(self):
        data = _multi_step_data()
        result = binary_segmentation(data, max_changepoints=1, min_segment_length=10)
        assert result.n_changepoints <= 1


class TestMannWhitney:
    def test_mann_whitney_different(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(5.0, 1.0, 50)])
        u_stat, p_val = mann_whitney_test(data, 50)
        assert p_val < 0.05

    def test_mann_whitney_same(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 1.0, 100)
        u_stat, p_val = mann_whitney_test(data, 50)
        assert p_val > 0.05


class TestPettitt:
    def test_pettitt_detects_shift(self):
        data = _step_data(n1=50, n2=50, mean1=0.0, mean2=5.0)
        result = pettitt_test(data)
        assert result is not None
        assert isinstance(result, ChangePoint)
        assert 30 <= result.index <= 70
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_pettitt_no_shift(self):
        data = _stationary_data(100)
        result = pettitt_test(data)
        assert result is None


class TestRegimeShiftDetector:
    def test_regime_shift_detector(self):
        rng = np.random.default_rng(42)
        # Two clear regimes
        data = np.concatenate([
            rng.normal(0.0, 0.5, 60),
            rng.normal(5.0, 0.5, 60),
        ])
        shifts = regime_shift_detector(data, window_size=20, threshold=2.0)
        assert len(shifts) >= 1
        assert isinstance(shifts[0], ChangePoint)
        # Shift should be near index 60
        assert any(40 <= cp.index <= 80 for cp in shifts)


class TestChangePointResultSegments:
    def test_changepoint_result_segments(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=5.0)
        result = pelt(data, min_segment_length=10)
        assert len(result.segments) == result.n_changepoints + 1
        for seg in result.segments:
            assert "start" in seg
            assert "end" in seg
            assert "mean" in seg
            assert "variance" in seg
            assert seg["end"] > seg["start"]


class TestChangePointTimestamps:
    def test_changepoint_timestamps(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        values = np.concatenate([rng.normal(0.0, 0.5, 100), rng.normal(5.0, 0.5, 100)])
        series = pd.Series(values, index=dates)
        result = pelt(series, min_segment_length=10)
        assert result.n_changepoints >= 1
        for cp in result.changepoints:
            assert cp.timestamp is not None
            assert isinstance(cp.timestamp, pd.Timestamp)


class TestPlotChangepoints:
    def test_plot_changepoints(self):
        data = _step_data(n1=100, n2=100, mean1=0.0, mean2=5.0)
        result = pelt(data, min_segment_length=10)
        # Should run without error with Agg backend
        plot_changepoints(data, result, title="Test Plot")
