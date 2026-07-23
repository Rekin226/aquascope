"""Tests for extended extreme value theory functions in flood_frequency.

Covers Gumbel, Weibull-min, GPD, L-moments, non-stationary GEV,
regional frequency analysis, and goodness-of-fit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gumbel_data(n: int = 100, loc: float = 50.0, scale: float = 10.0, seed: int = 42) -> np.ndarray:
    """Generate Gumbel-distributed data."""
    rng = np.random.default_rng(seed)
    return rng.gumbel(loc=loc, scale=scale, size=n)


def _weibull_data(n: int = 100, shape: float = 2.0, scale: float = 5.0, seed: int = 42) -> np.ndarray:
    """Generate Weibull-distributed data (minima-like)."""
    rng = np.random.default_rng(seed)
    return rng.weibull(shape, size=n) * scale


def _gev_data(n: int = 100, shape: float = -0.1, loc: float = 50.0, scale: float = 10.0, seed: int = 42) -> np.ndarray:
    """Generate GEV-distributed data using scipy."""
    from scipy.stats import genextreme

    return genextreme.rvs(shape, loc=loc, scale=scale, size=n, random_state=seed)


def _gpd_data(n: int = 200, shape: float = 0.2, scale: float = 5.0, seed: int = 42) -> np.ndarray:
    """Generate Generalised Pareto exceedances."""
    from scipy.stats import genpareto

    return genpareto.rvs(shape, scale=scale, size=n, random_state=seed)


# ---------------------------------------------------------------------------
# Gumbel
# ---------------------------------------------------------------------------

class TestFitGumbel:

    def test_fit_gumbel_basic(self):
        from aquascope.hydrology.flood_frequency import fit_gumbel

        data = _gumbel_data(200, loc=50, scale=10)
        result = fit_gumbel(data)
        assert result.distribution == "Gumbel"
        assert len(result.params) == 2
        # Check recovered location is in a reasonable range
        fitted_loc = result.params[0]
        assert 40 < fitted_loc < 60

    def test_fit_gumbel_return_periods(self):
        from aquascope.hydrology.flood_frequency import fit_gumbel

        data = _gumbel_data(200)
        result = fit_gumbel(data, return_periods=[10.0, 100.0])
        assert set(result.return_periods.keys()) == {10.0, 100.0}
        assert result.return_periods[100.0] > result.return_periods[10.0]

    def test_gumbel_vs_gev(self):
        """Gumbel result should be similar to GEV with shape ≈ 0."""
        from aquascope.hydrology.flood_frequency import fit_gev_lmoments, fit_gumbel

        data = _gumbel_data(500, loc=100, scale=15, seed=99)
        gumbel_res = fit_gumbel(data, return_periods=[100.0])
        gev_res = fit_gev_lmoments(data, return_periods=[100.0])
        gumbel_100 = gumbel_res.return_periods[100.0]
        gev_100 = gev_res.return_periods[100.0]
        np.testing.assert_allclose(gumbel_100, gev_100, rtol=0.15)


# ---------------------------------------------------------------------------
# Weibull minimum
# ---------------------------------------------------------------------------

class TestFitWeibullMin:

    def test_fit_weibull_min_basic(self):
        from aquascope.hydrology.flood_frequency import fit_weibull_min

        data = _weibull_data(200, shape=2.0, scale=5.0)
        result = fit_weibull_min(data)
        assert result.distribution == "Weibull_min"
        assert len(result.params) == 3
        # Return levels for low flow should be small
        assert result.return_periods[100.0] < result.return_periods[2.0]


# ---------------------------------------------------------------------------
# GPD
# ---------------------------------------------------------------------------

class TestFitGPD:

    def test_fit_gpd_basic(self):
        from aquascope.hydrology.flood_frequency import fit_gpd

        exceedances = _gpd_data(200, shape=0.2, scale=5.0) + 10  # threshold = 10
        result = fit_gpd(exceedances, threshold=10.0, total_observations=1000)
        assert result.distribution == "GPD"
        # Higher return period → higher level
        assert result.return_periods[100.0] > result.return_periods[10.0]

    def test_fit_gpd_return_levels(self):
        from aquascope.hydrology.flood_frequency import fit_gpd

        exceedances = _gpd_data(300, shape=0.1, scale=3.0) + 20
        result = fit_gpd(exceedances, threshold=20.0, total_observations=5000, return_periods=[10.0, 50.0])
        assert 10.0 in result.return_periods
        assert 50.0 in result.return_periods
        assert result.return_periods[50.0] > result.return_periods[10.0]

    def test_gpd_insufficient_exceedances(self):
        from aquascope.hydrology.flood_frequency import fit_gpd

        small = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
        with pytest.warns(UserWarning, match="unreliable"):
            fit_gpd(small, threshold=10.0)


# ---------------------------------------------------------------------------
# POT threshold selection
# ---------------------------------------------------------------------------

class TestSelectPOTThreshold:

    def test_percentile(self):
        from aquascope.hydrology.flood_frequency import select_pot_threshold

        rng = np.random.default_rng(42)
        data = rng.exponential(10, 1000)
        thr = select_pot_threshold(data, method="percentile")
        expected = float(np.percentile(data, 95))
        np.testing.assert_allclose(thr, expected, rtol=1e-10)

    def test_mean_residual(self):
        from aquascope.hydrology.flood_frequency import select_pot_threshold

        rng = np.random.default_rng(42)
        data = rng.exponential(10, 1000)
        thr = select_pot_threshold(data, method="mean_residual")
        assert thr > np.median(data)
        assert thr < np.max(data)

    def test_sqrt_rule(self):
        from aquascope.hydrology.flood_frequency import select_pot_threshold

        data = np.arange(1.0, 101.0)
        thr = select_pot_threshold(data, method="sqrt_rule")
        expected = float(np.mean(data) + 1.5 * np.std(data, ddof=1))
        np.testing.assert_allclose(thr, expected, rtol=1e-10)

    def test_unknown_method_raises(self):
        from aquascope.hydrology.flood_frequency import select_pot_threshold

        with pytest.raises(ValueError, match="Unknown"):
            select_pot_threshold(np.array([1, 2, 3]), method="unknown")


# ---------------------------------------------------------------------------
# L-moments
# ---------------------------------------------------------------------------

class TestLmoments:

    def test_lmoments_known_values(self):
        """For a uniform(0,1) sample the theoretical L1=0.5, L2≈1/6."""
        from aquascope.hydrology.flood_frequency import lmoments_from_sample

        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, 10000)
        lmom = lmoments_from_sample(data)
        np.testing.assert_allclose(lmom["L1"], 0.5, atol=0.02)
        np.testing.assert_allclose(lmom["L2"], 1 / 6, atol=0.02)
        # Uniform has L-skewness = 0
        np.testing.assert_allclose(lmom["t3"], 0.0, atol=0.03)

    def test_lmoments_too_few(self):
        from aquascope.hydrology.flood_frequency import lmoments_from_sample

        with pytest.raises(ValueError, match="≥4"):
            lmoments_from_sample(np.array([1.0, 2.0, 3.0]))

    def test_fit_gev_lmoments_basic(self):
        from aquascope.hydrology.flood_frequency import fit_gev_lmoments

        data = _gev_data(200, shape=-0.1, loc=50, scale=10)
        result = fit_gev_lmoments(data)
        assert result.distribution == "GEV_Lmom"
        assert result.return_periods[100.0] > result.return_periods[2.0]

    def test_fit_gev_lmoments_vs_mle(self):
        """L-moment and MLE GEV fits on same data should give similar 100-yr levels."""
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import fit_gev_lmoments

        data = _gev_data(300, shape=-0.15, loc=100, scale=20, seed=7)
        lmom_res = fit_gev_lmoments(data, return_periods=[100.0])

        # Quick MLE fit for comparison
        shape, loc, scale = genextreme.fit(data)
        mle_100 = float(genextreme.ppf(0.99, shape, loc=loc, scale=scale))
        lmom_100 = lmom_res.return_periods[100.0]
        np.testing.assert_allclose(lmom_100, mle_100, rtol=0.25)


# ---------------------------------------------------------------------------
# Non-stationary GEV
# ---------------------------------------------------------------------------

class TestNonStationaryGEV:

    def test_fit_nonstationary_gev_basic(self):
        from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

        rng = np.random.default_rng(42)
        years = np.arange(1970, 2020, dtype=float)
        trend = 0.5 * (years - years.mean())
        data = 50 + trend + rng.gumbel(0, 8, size=len(years))
        result = fit_nonstationary_gev(data, years)
        assert result.loc_trend != 0.0
        assert len(result.return_levels) > 0

    def test_nonstationary_trend_significant(self):
        """Strong positive trend → trend_significant should be True."""
        from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

        rng = np.random.default_rng(123)
        years = np.arange(1950, 2020, dtype=float)
        trend = 1.0 * (years - years.mean())
        data = 100 + trend + rng.gumbel(0, 5, size=len(years))
        result = fit_nonstationary_gev(data, years)
        assert result.trend_significant is True

    def test_nonstationary_no_trend(self):
        """Stationary data → trend_significant should be False."""
        from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

        rng = np.random.default_rng(77)
        years = np.arange(1960, 2020, dtype=float)
        data = 50 + rng.gumbel(0, 10, size=len(years))
        result = fit_nonstationary_gev(data, years)
        assert result.trend_significant is False

    def test_nonstationary_aic_bic(self):
        from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

        rng = np.random.default_rng(42)
        years = np.arange(1970, 2020, dtype=float)
        data = 50 + rng.gumbel(0, 8, size=len(years))
        result = fit_nonstationary_gev(data, years)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)


# ---------------------------------------------------------------------------
# Regional frequency analysis
# ---------------------------------------------------------------------------

class TestRegionalFrequency:

    def _make_sites(self, n_sites: int = 4, n_years: int = 40, seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        sites = {}
        for i in range(n_sites):
            loc = 50 + i * 10
            sites[f"site_{i}"] = rng.gumbel(loc=loc, scale=8, size=n_years)
        return sites

    def test_regional_frequency_basic(self):
        from aquascope.hydrology.flood_frequency import regional_frequency_analysis

        sites = self._make_sites()
        result = regional_frequency_analysis(sites)
        assert len(result.growth_curve) > 0
        assert len(result.index_flood) == 4
        assert result.growth_curve[100.0] > result.growth_curve[2.0]
        for sid in sites:
            assert sid in result.regional_return_levels

    def test_regional_discordancy(self):
        """One outlier site should have a higher discordancy statistic."""
        from aquascope.hydrology.flood_frequency import regional_frequency_analysis

        rng = np.random.default_rng(99)
        # Normal sites: Gumbel with similar L-moment ratios
        sites = {
            "normal_1": rng.gumbel(50, 8, 40),
            "normal_2": rng.gumbel(55, 8, 40),
            "normal_3": rng.gumbel(48, 8, 40),
        }
        # Outlier site: heavily skewed exponential → very different t3/t4
        sites["outlier"] = rng.exponential(2, 40) ** 3
        result = regional_frequency_analysis(sites)
        # The outlier should have the highest discordancy
        max_d_site = max(result.discordancy, key=result.discordancy.get)
        assert max_d_site == "outlier"


# ---------------------------------------------------------------------------
# Goodness-of-fit tests
# ---------------------------------------------------------------------------

class TestGoodnessOfFit:

    def test_anderson_darling_good_fit(self):
        """GEV data tested against fitted GEV — should not reject."""
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import anderson_darling_test

        data = _gev_data(200, shape=-0.1, loc=50, scale=10, seed=10)
        params = genextreme.fit(data)
        result = anderson_darling_test(data, "gev", params)
        assert result.test_name == "Anderson-Darling"
        assert result.reject_h0 is False

    def test_anderson_darling_bad_fit(self):
        """Uniform data tested against a GEV fit — should reject."""
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import anderson_darling_test

        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 200)
        params = genextreme.fit(data)
        result = anderson_darling_test(data, "gev", params)
        assert result.reject_h0 is True

    def test_cramer_von_mises_good_fit(self):
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import cramer_von_mises_test

        data = _gev_data(200, shape=-0.1, loc=50, scale=10, seed=20)
        params = genextreme.fit(data)
        result = cramer_von_mises_test(data, "gev", params)
        assert result.test_name == "Cramer-von-Mises"
        assert result.reject_h0 is False

    def test_cramer_von_mises_bad_fit(self):
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import cramer_von_mises_test

        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 200)
        params = genextreme.fit(data)
        result = cramer_von_mises_test(data, "gev", params)
        assert result.reject_h0 is True

    def test_ppcc_good_fit(self):
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import probability_plot_correlation

        data = _gev_data(200, shape=-0.1, loc=50, scale=10, seed=30)
        params = genextreme.fit(data)
        ppcc = probability_plot_correlation(data, "gev", params)
        assert ppcc > 0.98

    def test_ppcc_bad_fit(self):
        from scipy.stats import genextreme

        from aquascope.hydrology.flood_frequency import probability_plot_correlation

        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 200)
        params = genextreme.fit(data)
        ppcc = probability_plot_correlation(data, "gev", params)
        assert ppcc < 0.98
