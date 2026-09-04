"""Tests for the high-level convenience API (``aquascope.api``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _daily_discharge(n: int = 2200, seed: int = 42) -> pd.Series:
    """Return a synthetic daily discharge series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    base = 50.0 + 10.0 * np.sin(np.linspace(0, 6 * np.pi, n))
    noise = rng.normal(0, 5, n)
    values = np.maximum(base + noise, 0.1)
    index = pd.date_range("2000-01-01", periods=n, freq="D")
    return pd.Series(values, index=index, name="discharge")


def _regime_shift_series(n: int = 300, seed: int = 42) -> np.ndarray:
    """Return a 1-D array with a mean-shift at the midpoint."""
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(loc=0.0, scale=1.0, size=n // 2)
    seg2 = rng.normal(loc=5.0, scale=1.0, size=n - n // 2)
    return np.concatenate([seg1, seg2])


# ---------------------------------------------------------------------------
# Flood frequency
# ---------------------------------------------------------------------------

class TestFloodAnalysis:
    """Tests for :func:`aquascope.api.flood_analysis`."""

    def test_flood_analysis_gev(self):
        from aquascope.api import flood_analysis
        from aquascope.hydrology.flood_frequency import FloodFreqResult

        q = _daily_discharge()
        result = flood_analysis(q, method="gev", return_periods=[10, 50, 100])

        assert isinstance(result, FloodFreqResult)
        assert result.distribution.upper() == "GEV"
        assert 10 in result.return_periods
        assert 50 in result.return_periods
        assert 100 in result.return_periods

    def test_flood_analysis_lp3(self):
        from aquascope.api import flood_analysis
        from aquascope.hydrology.flood_frequency import FloodFreqResult

        q = _daily_discharge()
        result = flood_analysis(q, method="lp3")

        assert isinstance(result, FloodFreqResult)
        assert result.distribution.upper() == "LP3"

    def test_flood_analysis_invalid_method(self):
        from aquascope.api import flood_analysis

        q = _daily_discharge()
        with pytest.raises(ValueError, match="Unknown flood-frequency method"):
            flood_analysis(q, method="nonexistent")

    @pytest.mark.parametrize("method", ["gumbel", "gev_lmoments", "gev", "lp3"])
    def test_daily_input_is_reduced_to_annual_maxima(self, method):
        """Every method must be fitted on annual maxima, not raw daily values.

        `gumbel` and `gev_lmoments` used to receive the full series, so a
        six-year daily record was fitted as ~2,200 annual maxima and the
        return levels described the distribution of daily flows.
        """
        from aquascope.api import flood_analysis

        q = _daily_discharge()  # 2,200 daily values spanning 7 calendar years
        result = flood_analysis(q, method=method)

        assert len(result.annual_max) == q.index.year.nunique()
        assert len(result.annual_max) < 10

    def test_all_methods_agree_on_the_annual_maximum_series(self):
        """The quantity fitted must not depend on which distribution is chosen."""
        from aquascope.api import flood_analysis

        q = _daily_discharge()
        counts = {
            method: len(flood_analysis(q, method=method).annual_max)
            for method in ("gumbel", "gev_lmoments", "gev", "lp3")
        }

        assert len(set(counts.values())) == 1, counts

    def test_annual_maxima_input_is_passed_through_unchanged(self):
        """A caller who already has annual maxima must not be resampled again.

        Without a DatetimeIndex there is nothing to resample on, and the values
        are already the quantity the fitting functions want.
        """
        from aquascope.api import flood_analysis

        maxima = np.array([120.0, 95.0, 143.0, 88.0, 131.0, 104.0, 117.0])
        result = flood_analysis(maxima, method="gumbel")

        assert len(result.annual_max) == len(maxima)

    def test_daily_gumbel_return_level_matches_a_direct_annual_max_fit(self):
        """The wrapper must agree with calling `fit_gumbel` on annual maxima."""
        from aquascope.api import flood_analysis
        from aquascope.hydrology.flood_frequency import fit_gumbel

        q = _daily_discharge()
        expected = fit_gumbel(q.resample("YS").max().dropna(), return_periods=[100])
        actual = flood_analysis(q, method="gumbel", return_periods=[100])

        assert actual.return_periods[100] == pytest.approx(expected.return_periods[100])


# ---------------------------------------------------------------------------
# Baseflow
# ---------------------------------------------------------------------------

class TestBaseflowAnalysis:
    """Tests for :func:`aquascope.api.baseflow_analysis`."""

    def test_baseflow_analysis_lyne_hollick(self):
        from aquascope.api import baseflow_analysis
        from aquascope.hydrology.baseflow import BaseflowResult

        q = _daily_discharge()
        result = baseflow_analysis(q, method="lyne_hollick")

        assert isinstance(result, BaseflowResult)
        assert result.method == "lyne_hollick"
        assert 0.0 <= result.bfi <= 1.0

    def test_baseflow_analysis_eckhardt(self):
        from aquascope.api import baseflow_analysis
        from aquascope.hydrology.baseflow import BaseflowResult

        q = _daily_discharge()
        result = baseflow_analysis(q, method="eckhardt")

        assert isinstance(result, BaseflowResult)
        assert result.method == "eckhardt"
        assert 0.0 <= result.bfi <= 1.0


# ---------------------------------------------------------------------------
# Flow duration
# ---------------------------------------------------------------------------

class TestFlowDuration:
    """Tests for :func:`aquascope.api.flow_duration`."""

    def test_flow_duration(self):
        from aquascope.api import flow_duration
        from aquascope.hydrology.flow_duration import FDCResult

        q = _daily_discharge()
        result = flow_duration(q)

        assert isinstance(result, FDCResult)
        assert len(result.exceedance) == len(result.discharge)
        assert 50 in result.percentiles


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

class TestComputeAllSignatures:
    """Tests for :func:`aquascope.api.compute_all_signatures`."""

    def test_compute_all_signatures(self):
        from aquascope.api import compute_all_signatures
        from aquascope.hydrology.signatures import SignatureReport

        q = _daily_discharge()
        result = compute_all_signatures(q)

        assert isinstance(result, SignatureReport)
        assert result.mean_flow > 0
        assert result.cv > 0


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------

class TestDetectChangepoints:
    """Tests for :func:`aquascope.api.detect_changepoints`."""

    def test_detect_changepoints_pelt(self):
        from aquascope.analysis.changepoint import ChangePointResult
        from aquascope.api import detect_changepoints

        data = _regime_shift_series()
        result = detect_changepoints(data, method="pelt")

        assert isinstance(result, ChangePointResult)
        assert result.method == "pelt"
        assert result.n_changepoints >= 1

    def test_detect_changepoints_pettitt(self):
        from aquascope.analysis.changepoint import ChangePointResult
        from aquascope.api import detect_changepoints

        data = _regime_shift_series()
        result = detect_changepoints(data, method="pettitt")

        assert isinstance(result, ChangePointResult)
        assert result.method == "pettitt"
        assert result.n_changepoints >= 1


# ---------------------------------------------------------------------------
# Copula
# ---------------------------------------------------------------------------

class TestFitCopula:
    """Tests for :func:`aquascope.api.fit_copula`."""

    def test_fit_copula_auto(self):
        from aquascope.analysis.copulas import CopulaResult
        from aquascope.api import fit_copula

        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = 0.6 * x + rng.normal(scale=0.5, size=200)

        result = fit_copula(x, y, family="auto")

        assert isinstance(result, CopulaResult)
        assert result.family in {"gaussian", "clayton", "gumbel", "frank"}

    def test_fit_copula_specific_family(self):
        from aquascope.analysis.copulas import CopulaResult
        from aquascope.api import fit_copula

        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        y = 0.6 * x + rng.normal(scale=0.5, size=200)

        result = fit_copula(x, y, family="gaussian")

        assert isinstance(result, CopulaResult)
        assert result.family == "gaussian"


# ---------------------------------------------------------------------------
# Bayesian regression
# ---------------------------------------------------------------------------

class TestBayesianRegression:
    """Tests for :func:`aquascope.api.bayesian_regression`."""

    def test_bayesian_regression_linear(self):
        from aquascope.api import bayesian_regression
        from aquascope.models.bayesian import PosteriorResult

        rng = np.random.default_rng(42)
        x_features = rng.normal(size=(50, 1))
        y = 3.0 * x_features[:, 0] + 1.0 + rng.normal(scale=0.5, size=50)

        result = bayesian_regression(x_features, y, degree=1)

        assert isinstance(result, PosteriorResult)
        assert len(result.posterior_mean) > 0


# ---------------------------------------------------------------------------
# Ensemble forecast
# ---------------------------------------------------------------------------

class TestEnsembleForecast:
    """Tests for :func:`aquascope.api.ensemble_forecast`."""

    def test_ensemble_forecast_stacking(self):
        from aquascope.api import ensemble_forecast

        rng = np.random.default_rng(42)
        n_train, n_test = 80, 20
        x_train = pd.DataFrame({"a": rng.normal(size=n_train), "b": rng.normal(size=n_train)})
        y_train = pd.Series(2 * x_train["a"] - x_train["b"] + rng.normal(scale=0.3, size=n_train))
        x_test = pd.DataFrame({"a": rng.normal(size=n_test), "b": rng.normal(size=n_test)})

        from sklearn.linear_model import LinearRegression, Ridge

        models = [("lr", LinearRegression()), ("ridge", Ridge(alpha=1.0))]

        preds = ensemble_forecast(models, x_train, y_train, x_test, method="stacking")

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (n_test,)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Tests for :func:`aquascope.api.generate_report`."""

    def test_generate_report(self):
        from aquascope.api import generate_report
        from aquascope.reporting.builder import ReportBuilder

        report = generate_report("Test Report", author="Tester", description="A test")

        assert isinstance(report, ReportBuilder)
        assert report.metadata.title == "Test Report"
        assert report.metadata.author == "Tester"


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

class TestRequire:
    """Tests for :func:`aquascope.utils.imports.require`."""

    def test_require_missing_module(self):
        from aquascope.utils.imports import require

        with pytest.raises(ImportError, match="pip install"):
            require("nonexistent_pkg_xyz", feature="Test feature")

    def test_require_existing_module(self):
        from aquascope.utils.imports import require

        mod = require("json", feature="JSON support")
        assert hasattr(mod, "dumps")
