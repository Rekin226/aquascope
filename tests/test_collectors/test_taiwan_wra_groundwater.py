"""Tests for the Taiwan WRA annual groundwater collector (normalise + sentinel
handling). No network: normalise is exercised on sample records."""

from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.taiwan_wra import TaiwanWRAGroundwaterCollector
from aquascope.schemas.water_data import DataSource

SAMPLE = [
    {
        "wellidentifier": "W1",
        "year": "2020",
        "annualaveragewaterlevel": "12.50",
        "annualmaximumdailywaterlevel": "15.00",
        "annualminimumdailywaterlevel": "9.00",
    },
    {  # missing average (empty) + sentinel max
        "wellidentifier": "W2",
        "year": "2021",
        "annualaveragewaterlevel": "",
        "annualmaximumdailywaterlevel": "-999998.00",
        "annualminimumdailywaterlevel": "-999998.00",
    },
    {  # sentinel average
        "wellidentifier": "W3",
        "year": "2021",
        "annualaveragewaterlevel": "-999998.00",
        "annualmaximumdailywaterlevel": "8.0",
        "annualminimumdailywaterlevel": "2.0",
    },
]


def _collector(**kw):
    kw.setdefault("with_metadata", False)
    return TaiwanWRAGroundwaterCollector(client=MagicMock(), **kw)


# A 井況 metadata row for well W1 (TWD97 coords near Tainan, Chianan Plain).
META_ROWS = [
    {
        "wellidentifier": "W1",
        "wellname": "Test Well 1",
        "groundwaterzone": "嘉南平原",
        "welldepth": "39.0",
        "locationbytwd97": "208675.89 2475257.40",
    },
    {  # corrupt coordinates -> location should be rejected
        "wellidentifier": "W3",
        "wellname": "Bad Coords",
        "groundwaterzone": "屏東平原",
        "welldepth": "20.0",
        "locationbytwd97": "1.0 1.0",
    },
]


class TestGroundwaterCollector:
    def test_average_statistic_and_datetime(self):
        recs = list(_collector().normalise(SAMPLE))
        # Only W1 has a real average; W2/W3 are sentinel/empty -> dropped.
        assert len(recs) == 1
        r = recs[0]
        assert r.station_id == "W1"
        assert r.water_level_m == 12.5
        assert r.unit == "m"
        assert r.source == DataSource.TAIWAN_WRA
        assert r.measurement_datetime == datetime(2020, 7, 1)

    def test_minimum_statistic(self):
        recs = list(_collector(statistic="minimum").normalise(SAMPLE))
        # W1 (9.0) and W3 (2.0) have real minima; W2 is sentinel.
        levels = sorted(r.water_level_m for r in recs)
        assert levels == [2.0, 9.0]

    def test_drop_missing_is_default(self):
        recs = list(_collector(statistic="maximum").normalise(SAMPLE))
        # W1 (15.0) and W3 (8.0) real; W2 max is sentinel -> dropped.
        assert sorted(r.water_level_m for r in recs) == [8.0, 15.0]
        assert all(r.water_level_m > -9999 for r in recs)

    def test_na_value_zero_injects_zeros(self):
        recs = list(_collector(na_value=0.0).normalise(SAMPLE))
        # All three well-years emitted; the two missing averages become 0.0.
        assert len(recs) == 3
        assert sum(1 for r in recs if r.water_level_m == 0.0) == 2

    def test_na_value_nan_keeps_placeholder(self):
        recs = list(_collector(na_value=float("nan")).normalise(SAMPLE))
        assert len(recs) == 3
        assert sum(1 for r in recs if math.isnan(r.water_level_m)) == 2

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError):
            _collector(statistic="median")

    def test_fetch_raw_uses_annual_dataset(self):
        client = MagicMock()
        client.get_json.return_value = SAMPLE
        col = TaiwanWRAGroundwaterCollector(client=client, with_metadata=False)
        out = col.fetch_raw()
        assert out == SAMPLE
        # called with the dangling annual-groundwater UUID
        assert "f3ae2889" in client.get_json.call_args[0][0]


class TestMetadataEnrichment:
    def test_join_fills_aquifer_depth_location(self):
        # Two get_json calls: metadata first, then the annual series.
        client = MagicMock()
        client.get_json.side_effect = [META_ROWS, SAMPLE]
        col = TaiwanWRAGroundwaterCollector(with_metadata=True, client=client)
        recs = list(col.collect())
        by_id = {r.station_id: r for r in recs}
        w1 = by_id["W1"]
        assert w1.aquifer_name == "嘉南平原"
        assert w1.well_depth_m == 39.0
        assert w1.station_name == "Test Well 1"
        assert w1.location is not None
        # TWD97 (208675, 2475257) -> a valid point in southern Taiwan
        assert 21.9 < w1.location.latitude < 25.3
        assert 119.5 < w1.location.longitude < 122.0

    def test_corrupt_coords_rejected_metadata_kept(self):
        client = MagicMock()
        client.get_json.side_effect = [META_ROWS, SAMPLE]
        col = TaiwanWRAGroundwaterCollector(statistic="minimum", with_metadata=True, client=client)
        recs = {r.station_id: r for r in col.collect()}
        w3 = recs["W3"]
        assert w3.location is None  # out-of-Taiwan coords filtered
        assert w3.aquifer_name == "屏東平原"  # but zone/depth still populated
        assert w3.well_depth_m == 20.0
