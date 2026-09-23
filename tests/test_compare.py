"""aquascope.compare: the Compare view's engine (My places, item 7 of the study-on-the-map arc)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope import compare
from aquascope.compare import compare_series, compare_stations


def _flow(start: str, years: int, scale: float, seed: int) -> pd.Series:
    idx = pd.date_range(start, periods=int(365.25 * years), freq="D")
    rng = np.random.default_rng(seed)
    season = 1 + 0.6 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    return pd.Series(scale * season * rng.lognormal(0, 0.5, len(idx)), index=idx)


def _item(key: str, s: pd.Series | None, *, area=None, variable="discharge", unit="m3/s", **kw) -> dict:
    return {"key": key, "label": key.upper(), "series": s, "variable": variable, "unit": unit, "area_km2": area, **kw}


def test_normalises_by_area_when_every_gauge_has_one() -> None:
    small = _flow("2000-01-01", 20, 1.0, 1)
    big = _flow("2005-01-01", 20, 100.0, 2)
    out = compare_series([_item("a", small, area=10.0), _item("b", big, area=1000.0)])
    assert out["normalised"] is True
    assert out["unit"] == "mm/d"
    assert "Normalised by catchment area" in out["basis"]
    # Same runoff depth: the two gauges' specific flows land on one scale.
    ma, mb = (p["mean"] for p in out["places"])
    assert ma == pytest.approx(mb, rel=0.2)
    assert ma == pytest.approx(small.mean() * 86.4 / 10.0, rel=1e-3)
    # The hydrographs are cut to the shared window and aligned on one index.
    h = out["hydrograph"]
    assert h["window"] == {"start": "2005-01-01", "end": small.index.max().date().isoformat(), "overlap": True}
    assert h["t"][0] == "2005-01-01"
    assert set(h["series"]) == {"a", "b"}
    assert all(len(v) == len(h["t"]) for v in h["series"].values())
    json.dumps(out)  # the page gets it as JSON


def test_falls_back_to_raw_and_says_which_gauge_lacks_an_area() -> None:
    a, b = _flow("2000-01-01", 15, 1, 1), _flow("2000-01-01", 15, 5, 2)
    out = compare_series([_item("a", a, area=10.0), _item("b", b)])
    assert out["normalised"] is False
    assert out["unit"] == "m3/s"
    assert "not normalised" in out["basis"] and "B" in out["basis"]


def test_curves_match_the_station_page_code() -> None:
    from aquascope.explore import _annual_max
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments
    from aquascope.hydrology.flow_duration import flow_duration_curve

    s = _flow("1990-01-01", 25, 3.0, 7)
    out = compare_series([_item("a", s), _item("b", _flow("1990-01-01", 25, 4.0, 8))])
    fdc = out["fdc"]["a"]
    assert fdc["q95"] == pytest.approx(flow_duration_curve(s).percentiles[95], rel=1e-4)
    assert len(fdc["exceedance"]) == len(fdc["q"]) <= 2 * compare.FDC_POINTS
    ffa = out["ffa"]["a"]
    ref = fit_gev_lmoments(_annual_max(s), return_periods=[100])
    assert ffa["q100"] == pytest.approx(ref.return_periods[100], rel=1e-4)
    assert ffa["n_years"] == len(ffa["empirical"]["q"])
    assert ffa["q"] == sorted(ffa["q"])  # a return-level curve rises with T
    assert {m["name"] for m in out["methods"]} >= {"Flow-duration curve", "GEV fitted by L-moments"}


def test_short_records_get_no_flood_curve_but_a_reason() -> None:
    out = compare_series([_item("a", _flow("2020-01-01", 3, 1, 1)), _item("b", _flow("2020-01-01", 3, 2, 2))])
    assert "error" in out["ffa"]["a"]
    assert any("no flood curve" in n for n in out["notes"])
    assert out["fdc"]["a"]["q"]


def test_disjoint_records_span_both_and_say_so() -> None:
    out = compare_series([_item("a", _flow("1980-01-01", 5, 1, 1)), _item("b", _flow("2000-01-01", 5, 1, 2))])
    assert out["hydrograph"]["window"]["overlap"] is False
    assert out["hydrograph"]["window"]["start"] == "1980-01-01"
    assert any("do not share a full year" in n for n in out["notes"])


def test_long_records_are_binned_for_the_browser() -> None:
    out = compare_series([_item("a", _flow("1950-01-01", 60, 1, 1)), _item("b", _flow("1950-01-01", 60, 2, 2))])
    h = out["hydrograph"]
    assert h["bin_days"] > 1 and len(h["t"]) <= compare.MAX_HYDRO_POINTS
    assert any("-day means" in n for n in out["notes"])


def test_other_variables_and_failures_are_left_out_with_a_note() -> None:
    out = compare_series([
        _item("a", _flow("2000-01-01", 12, 1, 1)),
        _item("b", _flow("2000-01-01", 12, 1, 2)),
        _item("c", _flow("2000-01-01", 12, 1, 3), variable="water_level", unit="m"),
        _item("d", None, error="the agency cannot be reached from a browser"),
    ])
    by = {p["key"]: p for p in out["places"]}
    assert by["a"]["compared"] and by["b"]["compared"]
    assert not by["c"]["compared"] and "water_level" in by["c"]["error"]
    assert not by["d"]["compared"]
    assert out["n_compared"] == 2
    assert set(out["hydrograph"]["series"]) == {"a", "b"}
    assert sum("left out" in n for n in out["notes"]) == 2


def test_water_levels_compare_raw_without_flow_curves() -> None:
    out = compare_series([
        _item("a", _flow("2000-01-01", 5, 1, 1), variable="water_level", unit="m", area=10),
        _item("b", _flow("2000-01-01", 5, 1, 2), variable="water_level", unit="m", area=20),
    ])
    assert out["normalised"] is False and out["unit"] == "m"
    assert out["fdc"] == {} and out["ffa"] == {}


def test_nothing_usable() -> None:
    out = compare_series([_item("a", None), _item("b", pd.Series(dtype=float))])
    assert out["hydrograph"] is None and out["n_compared"] == 0
    assert len(out["notes"]) == 2


def test_compare_stations_checks_the_count() -> None:
    assert "error" in compare_stations([{"source": "usgs", "station_id": "1"}])
    assert "error" in compare_stations([{"source": "usgs", "station_id": str(i)} for i in range(6)])


def test_compare_stations_fetches_each_and_passes_the_area(monkeypatch) -> None:
    import aquascope.explore as explore

    series = {"1": _flow("2000-01-01", 12, 1, 1), "2": _flow("2000-01-01", 12, 3, 2)}
    calls = []

    def fake_fetch(source, station_id, *, years=None, period_start=None, **_kw):
        calls.append((source, station_id, years, period_start))
        if station_id == "3":
            raise explore.BrowserUnreachableError("no CORS")
        return {"series": series[station_id], "variable": "discharge", "unit": "m3/s", "note": ""}

    monkeypatch.setattr(explore, "fetch_series", fake_fetch)
    out = compare_stations([
        {"source": "usgs", "station_id": "1", "area_km2": 10, "period_start": "2000-01-01"},
        {"source": "usgs", "station_id": "2", "area_km2": 30, "label": "Two"},
        {"source": "opw", "station_id": "3"},
    ], years=12)
    assert [c[1] for c in calls] == ["1", "2", "3"]
    assert calls[0][2] == 12 and calls[0][3] == "2000-01-01"
    assert out["normalised"] is True  # the gauge that failed does not block normalisation
    assert out["places"][1]["label"] == "Two"
    assert "browser" in out["places"][2]["error"]
    assert out["places"][0]["key"] == "usgs/1"
