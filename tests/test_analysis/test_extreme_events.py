"""Tests for aquascope.analysis.extreme_events — frequency analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.analysis.extreme_events import (
    DEFAULT_RETURN_PERIODS,
    compute_gev_parameters,
    estimate_return_periods,
    fit_distribution,
)
from aquascope.models.analysis import (
    DistributionFit,
    GEVParameters,
    ReturnPeriodResult,
)


def _annual_max_series(n: int = 60, start: str = "1960") -> pd.Series:
    """Synthetic series of yearly maxima drawn from a GEV-like distribution."""
    rng = np.random.default_rng(7)
    idx = pd.date_range(start, periods=n, freq="YE")
    values = stats_genextreme_sample(rng, n)
    return pd.Series(values, index=idx, name="peak_flow")


def stats_genextreme_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    from scipy import stats

    return stats.genextreme.rvs(-0.1, loc=100, scale=25, size=n, random_state=rng)


class TestComputeGevParameters:
    def test_returns_gev_parameters(self):
        params = compute_gev_parameters(_annual_max_series())
        assert isinstance(params, GEVParameters)
        assert params.scale > 0
        assert params.method == "mle"

    def test_accepts_raw_array(self):
        rng = np.random.default_rng(1)
        params = compute_gev_parameters(stats_genextreme_sample(rng, 50))
        assert isinstance(params, GEVParameters)


class TestFitDistribution:
    @pytest.mark.parametrize("dist", ["gev", "lp3", "gumbel"])
    def test_fit_each_distribution(self, dist):
        fit = fit_distribution(_annual_max_series(), distribution=dist)
        assert isinstance(fit, DistributionFit)
        assert fit.distribution == dist
        assert fit.n_samples >= 3
        assert 0.0 <= fit.ks_pvalue <= 1.0
        assert np.isfinite(fit.aic)

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError):
            fit_distribution(_annual_max_series(), distribution="weibull")

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_distribution(pd.Series([1.0, 2.0]))


class TestEstimateReturnPeriods:
    def test_returns_result_with_bounds(self):
        result = estimate_return_periods(_annual_max_series(), n_bootstrap=100)
        assert isinstance(result, ReturnPeriodResult)
        assert len(result.return_levels) == len(DEFAULT_RETURN_PERIODS)
        assert isinstance(result.fit, DistributionFit)

    def test_return_levels_increase_with_period(self):
        result = estimate_return_periods(_annual_max_series(), n_bootstrap=100)
        levels = result.return_levels
        assert levels == sorted(levels), "higher return periods must give higher magnitudes"

    def test_bounds_bracket_estimate(self):
        result = estimate_return_periods(_annual_max_series(), n_bootstrap=200)
        for lo, est, hi in zip(result.lower_bound, result.return_levels, result.upper_bound):
            assert lo <= est <= hi

    def test_reproducible_with_seed(self):
        a = estimate_return_periods(_annual_max_series(), n_bootstrap=100, random_state=11)
        b = estimate_return_periods(_annual_max_series(), n_bootstrap=100, random_state=11)
        assert a.return_levels == b.return_levels
        assert a.lower_bound == b.lower_bound
