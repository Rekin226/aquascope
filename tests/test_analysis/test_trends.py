"""Unit tests for Mann-Kendall trend detection and Sen's slope estimation (aquascope/analysis/trends.py)."""

import numpy as np
import pandas as pd
import pytest

from aquascope.analysis.trends import (
    MannKendallResult,
    SensSlopeResult,
    mann_kendall,
    sens_slope,
)
from aquascope.api import (
    mann_kendall as api_mann_kendall,
)
from aquascope.api import (
    sens_slope as api_sens_slope,
)
from aquascope.api import (
    trend_analysis as api_trend_analysis,
)


class TestMannKendall:
    """Test suite for mann_kendall()."""

    def test_increasing_trend(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        res = mann_kendall(data)
        assert isinstance(res, MannKendallResult)
        assert res.trend == "increasing"
        assert res.h is True
        assert res.p_value < 0.05
        assert res.slope == pytest.approx(1.0)
        assert res.intercept == pytest.approx(1.0)
        assert res.n_samples == 10
        assert res.n_dropped_nans == 0

    def test_decreasing_trend(self):
        data = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        res = mann_kendall(data)
        assert res.trend == "decreasing"
        assert res.h is True
        assert res.p_value < 0.05
        assert res.slope == pytest.approx(-1.0)
        assert res.n_samples == 10

    def test_hand_computed_tied_series(self):
        # Validates exact S, var_s, Z, p-value, and Sen's slope on tied series
        data = [3, 3, 5, 4, 4, 7, 6, 9, 9, 12, 11, 15]
        res = mann_kendall(data)
        assert res.s_stat == 55
        assert res.var_s == pytest.approx(209.667, abs=1e-3)
        assert res.z_stat == pytest.approx(3.729315, abs=1e-5)
        assert res.p_value == pytest.approx(0.000192, abs=1e-5)
        assert res.slope == pytest.approx(1.0)
        assert res.intercept == pytest.approx(1.0)

    def test_no_trend(self):
        # Constant data has no trend
        data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        res = mann_kendall(data)
        assert res.trend == "no trend"
        assert res.h is False

    def test_nan_handling(self, caplog):
        series = pd.Series([1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        res = mann_kendall(series)
        assert res.n_dropped_nans == 2
        assert res.n_samples == 10
        assert res.trend == "increasing"

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient valid data points"):
            mann_kendall([1.0, 2.0])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown Mann-Kendall method"):
            mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0], method="invalid_method")

    def test_modified_variants(self):
        data = np.linspace(10, 50, 20) + np.sin(np.linspace(0, 4 * np.pi, 20))
        for method in ("hamed_rao", "yue_wang", "pre_whitening", "tfpw"):
            res = mann_kendall(data, method=method)
            assert res.method == method
            assert res.trend in ("increasing", "decreasing", "no trend")
            assert isinstance(res.p_value, float)


class TestSensSlope:
    """Test suite for sens_slope()."""

    def test_known_slope_and_intercept(self):
        x = np.arange(10, dtype=float)
        y = 3.0 * x + 10.0
        res = sens_slope(y)
        assert isinstance(res, SensSlopeResult)
        assert res.slope == pytest.approx(3.0)
        assert res.intercept == pytest.approx(10.0)
        assert res.n_samples == 10
        assert res.n_dropped_nans == 0

    def test_sens_slope_with_nans(self):
        y = np.array([10.0, np.nan, 13.0, 16.0, np.nan, 19.0, 22.0])
        res = sens_slope(y)
        assert res.n_dropped_nans == 2
        assert res.n_samples == 5
        assert res.slope == pytest.approx(3.0)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Insufficient valid data points"):
            sens_slope([5.0])


class TestAPIFacade:
    """Test suite for aquascope.api trend functions."""

    def test_api_trend_analysis(self):
        res = api_trend_analysis([1, 3, 5, 7, 9, 11, 13])
        assert res.trend == "increasing"

    def test_api_mann_kendall_alias(self):
        res = api_mann_kendall([1, 3, 5, 7, 9, 11, 13])
        assert res.trend == "increasing"

    def test_api_sens_slope(self):
        res = api_sens_slope([10, 12, 14, 16, 18])
        assert res.slope == pytest.approx(2.0)
