"""Tests for the Standardised Groundwater Index and drought events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.groundwater.drought import (
    drought_events,
    standardised_groundwater_index,
)


def _monthly(n_years: int = 25, start: str = "1995-01-15") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_years * 12, freq="MS")


class TestSGI:
    def setup_method(self):
        self.idx = _monthly(25)
        rng = np.random.default_rng(0)
        seasonal = 2.0 * np.sin(2 * np.pi * np.arange(len(self.idx)) / 12)
        self.levels = pd.Series(10.0 + seasonal + rng.normal(0, 1, len(self.idx)),
                                index=self.idx)

    def test_standard_normal_distribution(self):
        sgi = standardised_groundwater_index(self.levels)
        s = sgi.dropna()
        # Normal scores -> mean ~0, std ~1.
        assert abs(s.mean()) < 0.15
        assert 0.8 < s.std() < 1.2

    def test_removes_seasonality(self):
        # SGI is standardised per calendar month, so the strong seasonal cycle
        # in the raw levels should not survive into the index.
        sgi = standardised_groundwater_index(self.levels).dropna()
        by_month = sgi.groupby(sgi.index.month).mean()
        assert by_month.abs().max() < 0.5  # raw seasonal swing was +/-2 m

    def test_rank_preserving_within_month(self):
        # Within one calendar month, the lowest level gets the lowest SGI.
        sgi = standardised_groundwater_index(self.levels)
        jan = self.levels[self.levels.index.month == 1]
        jan_sgi = sgi[sgi.index.month == 1]
        assert jan_sgi.loc[jan.idxmin()] == jan_sgi.min()
        assert jan_sgi.loc[jan.idxmax()] == jan_sgi.max()

    def test_drought_year_is_negative(self):
        # Depress two years of levels; their SGI should be strongly negative.
        levels = self.levels.copy()
        drought = (levels.index.year >= 2015) & (levels.index.year <= 2016)
        levels[drought] -= 5.0
        sgi = standardised_groundwater_index(levels)
        assert sgi[drought].mean() < -1.0

    def test_non_datetime_index_raises(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            standardised_groundwater_index(pd.Series([1.0, 2.0, 3.0]))

    def test_too_short_raises(self):
        s = pd.Series([1.0, 2.0], index=_monthly(1)[:2])
        with pytest.raises(ValueError, match="at least"):
            standardised_groundwater_index(s, min_per_month=5)

    def test_short_month_left_nan(self):
        # 4 years -> each calendar month has 4 obs; with min_per_month=5 all NaN.
        s = pd.Series(np.arange(48.0), index=_monthly(4))
        sgi = standardised_groundwater_index(s, min_per_month=5)
        assert sgi.isna().all()


class TestDroughtEvents:
    def test_detects_below_threshold_run(self):
        idx = _monthly(3)
        vals = np.zeros(36)
        vals[10:16] = -1.5  # a 6-month drought
        events = drought_events(pd.Series(vals, index=idx), threshold=-1.0)
        assert len(events) == 1
        assert events[0].duration == 6
        assert events[0].peak == pytest.approx(-1.5)
        assert events[0].severity == pytest.approx(-9.0)

    def test_no_event_when_above_threshold(self):
        idx = _monthly(2)
        assert drought_events(pd.Series(np.zeros(24), index=idx)) == []

    def test_trailing_event_closed(self):
        idx = _monthly(1)
        vals = np.zeros(12)
        vals[9:] = -2.0  # runs to the end of the series
        events = drought_events(pd.Series(vals, index=idx), threshold=-1.0)
        assert len(events) == 1
        assert events[0].duration == 3
