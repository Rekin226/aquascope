"""Tests for the Standardized Precipitation Index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.climate.indices import drought_class, standardized_precipitation_index


def _monthly_precip(n_years: int = 30, seed: int = 0) -> pd.Series:
    idx = pd.date_range("1990-01-01", periods=n_years * 12, freq="MS")
    rng = np.random.default_rng(seed)
    # Seasonal gamma-ish precip (wet summers), always non-negative.
    seasonal = 80 + 60 * np.sin(2 * np.pi * (np.arange(len(idx)) % 12) / 12)
    precip = rng.gamma(shape=2.0, scale=seasonal / 2.0)
    return pd.Series(precip, index=idx)


class TestSPI:
    def test_standard_normal_distribution(self):
        spi = standardized_precipitation_index(_monthly_precip(), scale=3).dropna()
        assert abs(spi.mean()) < 0.2
        assert 0.7 < spi.std() < 1.3

    def test_dry_period_is_negative(self):
        precip = _monthly_precip()
        dry = (precip.index.year >= 2010) & (precip.index.year <= 2011)
        precip[dry] *= 0.1  # strong dry anomaly
        spi = standardized_precipitation_index(precip, scale=6).dropna()
        dry_spi = spi[spi.index.year.isin([2010, 2011])]
        assert dry_spi.mean() < -0.8

    def test_scale_changes_output(self):
        precip = _monthly_precip()
        spi3 = standardized_precipitation_index(precip, scale=3)
        spi12 = standardized_precipitation_index(precip, scale=12)
        # Different accumulation -> different series (and SPI-12 starts later).
        assert spi12.dropna().index.min() > spi3.dropna().index.min()

    def test_non_datetime_index_raises(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            standardized_precipitation_index(pd.Series([1.0, 2.0, 3.0]))

    def test_bad_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            standardized_precipitation_index(_monthly_precip(), scale=0)

    def test_handles_zeros(self):
        precip = _monthly_precip()
        precip.iloc[::5] = 0.0  # inject dry months
        spi = standardized_precipitation_index(precip, scale=1)
        assert np.isfinite(spi.dropna()).all()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.nan, "unknown"),
        (2.0, "extremely_wet"),
        (1.5, "very_wet"),
        (1.0, "moderately_wet"),
        (-1.0, "moderately_dry"),
        (-1.5, "severely_dry"),
        (-2.0, "extremely_dry"),
    ],
)
def test_drought_class_boundaries(value, expected):
    assert drought_class(value) == expected
