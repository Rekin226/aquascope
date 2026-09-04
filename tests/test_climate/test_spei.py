"""SPEI (#309): the log-logistic fit, the Thornthwaite PET helper, the shared core with SPI, and the case that
motivates the index: under warming, SPEI runs drier than SPI on the same rainfall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aquascope.climate.indices import (
    fit_generalized_logistic_lmoments,
    fit_log_logistic_lmoments,
    standardized_precipitation_evapotranspiration_index,
    standardized_precipitation_index,
    thornthwaite_pet,
)


def _monthly(n_years: int = 40, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Forty years of seasonal precipitation (mm) and mean temperature (deg C) at a temperate site."""
    idx = pd.date_range("1985-01-01", periods=n_years * 12, freq="MS")
    rng = np.random.default_rng(seed)
    phase = np.arange(len(idx)) % 12
    seasonal = 80 + 60 * np.sin(2 * np.pi * phase / 12)
    precip = pd.Series(rng.gamma(shape=2.0, scale=seasonal / 2.0), index=idx)
    temp = pd.Series(10 + 8 * np.sin(2 * np.pi * (phase - 3) / 12) + rng.normal(0, 1, len(idx)), index=idx)
    return precip, temp


class TestThornthwaite:
    def test_annual_total_is_plausible_and_summer_beats_winter(self):
        _, temp = _monthly()
        pet = thornthwaite_pet(temp, 51.4)
        annual = pet.groupby(pet.index.year).sum().mean()
        assert 450 < annual < 850, annual
        assert pet[pet.index.month == 7].mean() > 5 * pet[pet.index.month == 1].mean()
        assert (pet >= 0).all() and pet.name == "pet"

    def test_freezing_months_give_zero(self):
        idx = pd.date_range("2000-01-01", periods=24, freq="MS")
        temp = pd.Series(np.where(idx.month.isin([12, 1, 2]), -5.0, 15.0), index=idx)
        pet = thornthwaite_pet(temp, 45.0)
        assert (pet[idx.month.isin([12, 1, 2])] == 0).all() and (pet[~idx.month.isin([12, 1, 2])] > 0).all()

    def test_hemispheres_mirror_through_the_day_length_correction(self):
        idx = pd.date_range("2000-01-01", periods=12, freq="MS")
        temp = pd.Series(15.0, index=idx)
        north = thornthwaite_pet(temp, 45.0)
        south = thornthwaite_pet(temp, -45.0)
        assert north[idx.month == 7].iloc[0] == pytest.approx(south[idx.month == 1].iloc[0], rel=0.02)
        assert north[idx.month == 7].iloc[0] > north[idx.month == 12].iloc[0]

    def test_needs_a_datetime_index(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            thornthwaite_pet(pd.Series([10.0, 12.0]), 40.0)


class TestLogLogisticFit:
    def test_recovers_known_parameters(self):
        alpha, beta, gamma = 120.0, 4.0, -60.0
        sample = stats.fisk.rvs(c=beta, loc=gamma, scale=alpha, size=4000, random_state=3)
        a, b, g = fit_log_logistic_lmoments(sample)
        assert a == pytest.approx(alpha, rel=0.08) and b == pytest.approx(beta, rel=0.08)
        assert g == pytest.approx(gamma, abs=12)

    def test_refuses_what_it_cannot_fit(self):
        with pytest.raises(ValueError, match="four"):
            fit_log_logistic_lmoments([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="spread"):
            fit_log_logistic_lmoments([5.0] * 30)  # degenerate: no spread
        left = -stats.fisk.rvs(c=4.0, loc=0.0, scale=100.0, size=2000, random_state=1)  # left-skewed
        with pytest.raises(ValueError, match="not positive"):
            fit_log_logistic_lmoments(left)
        xi, alpha, k = fit_generalized_logistic_lmoments(left)
        assert k > 0 and alpha > 0, "the generalized logistic covers the mirror image"

    def test_generalized_logistic_reduces_to_the_logistic_for_symmetric_data(self):
        sample = stats.logistic.rvs(loc=5.0, scale=2.0, size=5000, random_state=7)
        xi, alpha, k = fit_generalized_logistic_lmoments(sample)
        assert xi == pytest.approx(5.0, abs=0.3) and alpha == pytest.approx(2.0, rel=0.1) and abs(k) < 0.05


class TestSPEI:
    def test_standard_normal_distribution(self):
        precip, temp = _monthly()
        spei = standardized_precipitation_evapotranspiration_index(precip, thornthwaite_pet(temp, 51.4), 3).dropna()
        assert abs(spei.mean()) < 0.2 and 0.7 < spei.std() < 1.3 and spei.name == "spei"

    def test_tracks_spi_when_the_climate_is_stationary(self):
        precip, temp = _monthly()
        spi = standardized_precipitation_index(precip, scale=3)
        spei = standardized_precipitation_evapotranspiration_index(precip, thornthwaite_pet(temp, 51.4), 3)
        both = pd.concat([spi, spei], axis=1).dropna()
        assert both["spi"].corr(both["spei"]) > 0.9

    def test_dry_period_is_negative(self):
        precip, temp = _monthly()
        dry = precip.index.year.isin([2010, 2011])
        precip[dry] *= 0.1
        spei = standardized_precipitation_evapotranspiration_index(precip, thornthwaite_pet(temp, 51.4), 6).dropna()
        assert spei[spei.index.year.isin([2010, 2011])].mean() < -0.8

    def test_warming_makes_spei_drier_than_spi_on_the_same_rainfall(self):
        """The reason the index exists: three degrees of warming over the record, rainfall unchanged."""
        precip, temp = _monthly()
        warming = temp + np.linspace(0.0, 3.0, len(temp))
        spi = standardized_precipitation_index(precip, scale=12).dropna()
        spei = standardized_precipitation_evapotranspiration_index(precip, thornthwaite_pet(warming, 51.4), 12)
        spei = spei.dropna()
        recent = slice("2015-01-01", None)
        early = slice(None, "1995-12-01")
        assert spei[recent].mean() < spi[recent].mean() - 0.25
        assert spei[early].mean() > spi[early].mean() + 0.1, "the early, cooler years read wetter in SPEI"
        assert abs(spi[recent].mean()) < 0.3, "SPI cannot see a drought that rainfall did not cause"

    def test_timescale_changes_output(self):
        precip, temp = _monthly()
        pet = thornthwaite_pet(temp, 51.4)
        s3 = standardized_precipitation_evapotranspiration_index(precip, pet, 3).dropna()
        s12 = standardized_precipitation_evapotranspiration_index(precip, pet, 12).dropna()
        assert s12.index.min() > s3.index.min() and not np.allclose(s3.reindex(s12.index).dropna(), s12.dropna())

    def test_uses_only_the_shared_months(self):
        precip, temp = _monthly()
        pet = thornthwaite_pet(temp, 51.4).iloc[120:]
        spei = standardized_precipitation_evapotranspiration_index(precip, pet, 1).dropna()
        assert spei.index.min() >= pet.index.min()

    def test_bad_inputs_raise(self):
        precip, temp = _monthly()
        pet = thornthwaite_pet(temp, 51.4)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            standardized_precipitation_evapotranspiration_index(pd.Series([1.0, 2.0]), pet, 1)
        with pytest.raises(ValueError, match="share no months"):
            standardized_precipitation_evapotranspiration_index(precip, pet.shift(600, freq="MS"), 1)
        with pytest.raises(ValueError, match="scale"):
            standardized_precipitation_evapotranspiration_index(precip, pet, 0)

    def test_spi_is_unchanged_by_the_shared_core(self):
        precip, _ = _monthly(30)
        spi = standardized_precipitation_index(precip, scale=3).dropna()
        assert abs(spi.mean()) < 0.2 and 0.7 < spi.std() < 1.3 and spi.name == "spi"
        precip.iloc[::5] = 0.0
        assert np.isfinite(standardized_precipitation_index(precip, scale=1).dropna()).all()
