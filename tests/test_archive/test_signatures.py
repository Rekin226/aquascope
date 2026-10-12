"""Per-station flow signatures (signatures.parquet) and the map filter they feed."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import signatures as sig
from aquascope.archive.observations import load_manifest, obs_path, save_manifest, series_to_csv_gz


def _series(start="1960-01-01", end="2019-12-31", *, trend=0.0, seed=0, peak_doy=60, gaps=None):
    """Daily flow with a seasonal cycle, noise and, optionally, annual peaks growing ``trend`` m3/s per year."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    season = 4 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - peak_doy + 91) / 365.25)
    base = 10 + season + rng.gamma(2, 0.5, len(idx))
    q = pd.Series(base, index=idx)
    # one flood a year on the peak day, with a size that can trend
    for y in range(idx[0].year, idx[-1].year + 1):
        day = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=peak_doy - 1 + int(rng.integers(-5, 6)))
        if day in q.index:
            q[day] = 80 + trend * (y - idx[0].year) + rng.normal(0, 3)
    if gaps is not None:
        q = q[~gaps(q.index)]
    return q


def test_a_long_record_with_growing_floods_gets_every_signature():
    row = sig.station_signatures(_series(trend=1.5), source="usgs", station_id="USGS-1", latitude=38.9, longitude=-77.1)
    assert set(row) == set(sig.COLUMNS)
    assert row["record_start"] == "1960-01-01" and row["record_end"] == "2019-12-31"
    assert row["n_days"] == 21915 and row["completeness"] == 1.0
    assert 59.9 < row["record_years"] < 60.1 and row["data_years"] == row["record_years"]
    assert row["q5"] > row["q50"] > row["q95"] > 0, "exceedance convention: q5 is the high flow"
    assert 0 < row["bfi"] < 1
    assert row["n_amax_years"] == 60
    assert row["amax_trend"] == "rising" and row["amax_mk_p"] < 0.001
    assert row["amax_sen_slope"] == pytest.approx(1.5, rel=0.15), "Sen slope in m3/s per calendar year"
    assert row["q100"] > max(80 + 1.5 * 59, 0) * 0.9
    assert abs(row["amax_doy_mean"] - 60) < 5 and row["amax_doy_r"] > 0.95
    assert row["notes"] is None
    assert row["latitude"] == 38.9 and row["unit"] == "m3/s"


def test_stationary_floods_say_none_and_falling_floods_say_falling():
    assert sig.station_signatures(_series(), source="s", station_id="a")["amax_trend"] == "none"
    assert sig.station_signatures(_series(trend=-1.0), source="s", station_id="b")["amax_trend"] == "falling"


def test_seasonality_wraps_around_new_year():
    """Peaks either side of 1 January average to about 1 January, not to mid-year."""
    s = _series(peak_doy=3, seed=4)
    row = sig.station_signatures(s, source="s", station_id="w")
    doy = row["amax_doy_mean"]
    assert doy < 15 or doy > 350


def test_a_short_record_gets_a_row_with_reasons_not_an_error():
    row = sig.station_signatures(_series("2015-01-01", "2020-12-31"), source="s", station_id="short")
    assert row["q_mean"] is not None and row["bfi"] is not None
    assert row["q100"] is None and row["amax_trend"] is None and row["amax_mk_p"] is None
    assert "q100: needs 10 complete years, has 6" in row["notes"]
    assert "flood trend: needs 8 complete years, has 6" in row["notes"]
    assert row["amax_doy_mean"] is not None  # 5 years are enough for seasonality


def test_gappy_years_do_not_give_annual_maxima():
    # every other year keeps only its first 200 days: those years drop out of the annual-maxima sample
    gaps = lambda idx: (idx.year % 2 == 1) & (idx.dayofyear > 200)  # noqa: E731
    row = sig.station_signatures(_series("1980-01-01", "1999-12-31", gaps=gaps), source="s", station_id="g")
    assert row["n_amax_years"] == 10
    assert row["completeness"] < 0.9 and row["data_years"] < row["record_years"]
    assert row["q100"] is not None


@pytest.mark.parametrize("q", [
    pd.Series([], dtype=float, index=pd.DatetimeIndex([])),
    pd.Series([np.nan, np.inf, -3.0], index=pd.date_range("2000-01-01", periods=3)),
    pd.Series([1.0, 2.0], index=pd.date_range("2000-01-01", periods=2)),
    pd.Series(np.zeros(4000), index=pd.date_range("2000-01-01", periods=4000)),
    pd.Series(np.full(9000, 5.0), index=pd.date_range("1990-01-01", periods=9000)),
])
def test_degenerate_records_never_raise(q):
    row = sig.station_signatures(q, source="s", station_id="x")
    assert set(row) == set(sig.COLUMNS)
    for k in ("q_mean", "bfi", "q100", "amax_mk_p", "amax_doy_mean"):
        v = row[k]
        assert v is None or (isinstance(v, float) and math.isfinite(v))
    if row["q100"] is None:
        assert row["notes"]


def test_zero_flow_record_explains_the_missing_bfi_and_q100():
    row = sig.station_signatures(pd.Series(np.zeros(4000), index=pd.date_range("2000-01-01", periods=4000)),
                                 source="s", station_id="z")
    assert row["bfi"] is None and "bfi: no flow" in row["notes"]
    assert row["q100"] is None and "q100:" in row["notes"]


# ── the table, from mirrored files ─────────────────────────────────────────


def _mirror(root, source, sid, s, *, manifest=True):
    p = obs_path(root, "discharge", source, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(series_to_csv_gz(s))
    if manifest:
        m = load_manifest(root)
        entry = m["sources"].setdefault(f"{source}/discharge", {"source": source, "variable": "discharge",
                                                                 "stations": {}})
        entry["stations"][sid] = {"n": int(len(s)), "file": str(p.relative_to(root)).replace("\\", "/")}
        save_manifest(root, m)


def test_build_signatures_reads_the_manifest_and_the_catalog_coordinates(tmp_path):
    _mirror(tmp_path, "usgs", "USGS-01646500", _series(trend=1.0))
    _mirror(tmp_path, "uk_ea", "a/b", _series("2018-01-01", "2020-12-31"))       # id with a slash
    _mirror(tmp_path, "hubeau_hydrometrie", "H1", _series("1990-01-01", "2010-12-31"), manifest=False)
    stations = [{"source": "usgs", "station_id": "USGS-01646500", "latitude": 38.95, "longitude": -77.13}]
    df = sig.build_signatures(tmp_path, stations=stations)
    assert list(df.columns) == sig.COLUMNS
    assert len(df) == 3
    by = df.set_index(["source", "station_id"])
    assert by.loc[("usgs", "USGS-01646500"), "latitude"] == 38.95
    assert by.loc[("usgs", "USGS-01646500"), "amax_trend"] == "rising"
    assert ("uk_ea", "a/b") in by.index, "the manifest keeps the real id, not the file-safe one"
    assert ("hubeau_hydrometrie", "H1") in by.index, "files without a manifest entry are still found"
    assert pd.isna(by.loc[("uk_ea", "a/b"), "q100"])
    assert df["n_days"].dtype == "int64" and df["q100"].dtype == "float64"


def test_an_unreadable_file_is_a_row_with_a_reason(tmp_path):
    p = obs_path(tmp_path, "discharge", "usgs", "BAD")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not gzip")
    df = sig.build_signatures(tmp_path)
    assert len(df) == 1 and "unreadable" in df.iloc[0]["notes"]


def test_build_signatures_file_writes_parquet_with_metadata(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    _mirror(tmp_path, "usgs", "USGS-1", _series(trend=1.0))
    summary = sig.build_signatures_file(tmp_path)
    assert summary["n_stations"] == 1 and summary["n_rising"] == 1 and summary["n_with_q100"] == 1
    table = pq.read_table(tmp_path / "signatures.parquet")
    meta = json.loads(table.schema.metadata[b"aquascope"])
    assert meta["kind"] == "signatures" and meta["columns"] == sig.COLUMNS
    # empty values are nulls, not NaN: DuckDB orders NaN above every number, so `bfi >= 0.6` would match it
    assert table.column("notes").null_count == 1
    back = sig.load_signatures(path=tmp_path / "signatures.parquet")
    assert back.iloc[0]["station_id"] == "USGS-1"


def test_build_signatures_file_with_nothing_mirrored_writes_nothing(tmp_path):
    summary = sig.build_signatures_file(tmp_path)
    assert summary["file"] is None and not (tmp_path / "signatures.parquet").exists()


# ── harvest wiring ─────────────────────────────────────────────────────────


def _fake_catalogs(**kwargs):
    from aquascope.registry import StationCatalog
    from aquascope.schemas.station import Station

    st = Station(source="usgs", station_id="USGS-1", name="Potomac", latitude=38.95, longitude=-77.13,
                 variables=("discharge",))
    return {"usgs": StationCatalog(source="usgs", stations=[st], seconds=0.1)}


def test_harvest_stations_writes_signatures_next_to_the_catalog(tmp_path):
    pytest.importorskip("pyarrow")
    from aquascope.archive import harvest_stations

    _mirror(tmp_path, "usgs", "USGS-1", _series(trend=1.0))
    with patch("aquascope.archive.harvest.station_catalogs", side_effect=_fake_catalogs):
        report = harvest_stations(tmp_path)
    assert report.files["signatures.parquet"] == "signatures.parquet"
    df = pd.read_parquet(tmp_path / "signatures.parquet")
    assert df.iloc[0]["latitude"] == 38.95, "coordinates come from the fresh catalog"
    assert "signatures.parquet" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_harvest_stations_flag_off_or_no_obs_skips_signatures(tmp_path):
    pytest.importorskip("pyarrow")
    from aquascope.archive import harvest_stations

    with patch("aquascope.archive.harvest.station_catalogs", side_effect=_fake_catalogs):
        report = harvest_stations(tmp_path / "empty")
    assert "signatures.parquet" not in report.files
    _mirror(tmp_path / "off", "usgs", "USGS-1", _series("2000-01-01", "2003-12-31"))
    with patch("aquascope.archive.harvest.station_catalogs", side_effect=_fake_catalogs):
        report = harvest_stations(tmp_path / "off", write_signatures=False)
    assert not (tmp_path / "off" / "signatures.parquet").exists()


def test_a_signatures_failure_never_sinks_the_harvest(tmp_path):
    pytest.importorskip("pyarrow")
    from aquascope.archive import harvest_stations

    _mirror(tmp_path, "usgs", "USGS-1", _series("2000-01-01", "2003-12-31"))
    with patch("aquascope.archive.harvest.station_catalogs", side_effect=_fake_catalogs), \
            patch("aquascope.archive.signatures.build_signatures_file", side_effect=RuntimeError("boom")):
        report = harvest_stations(tmp_path)
    assert (tmp_path / "stations.parquet").exists() and "signatures.parquet" not in report.files


def test_cli_harvest_signatures_builds_the_file(tmp_path, capsys, monkeypatch):
    pytest.importorskip("pyarrow")
    import sys

    from aquascope.cli import main

    _mirror(tmp_path, "usgs", "USGS-1", _series(trend=1.0))
    monkeypatch.setattr(sys, "argv", ["aquascope", "harvest", "signatures", "--out", str(tmp_path)])
    with patch("aquascope.archive.publish_folder") as publish:
        main()
    publish.assert_not_called()
    assert (tmp_path / "signatures.parquet").exists()
    assert "1 stations" in capsys.readouterr().out


def test_publish_uploads_signatures():
    """publish_folder's allow-list must carry the new file."""
    import inspect

    from aquascope.archive import publish

    assert '"*.parquet"' in inspect.getsource(publish.publish_folder)


# ── the filter ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("gauges with 50+ years and a rising flood trend", {"min_years": 50, "flood_trend": "rising"}),
    ("at least forty years of record, baseflow index above 0.6", {"min_years": 40, "bfi_min": 0.6}),
    ("flashy rivers where peak flows are falling", {"flood_trend": "falling", "bfi_max": 0.3}),
    ("BFI between 0.3 and 0.5 with no trend", {"flood_trend": "none", "bfi_min": 0.3, "bfi_max": 0.5}),
    ("groundwater-fed streams, 30 or more years", {"min_years": 30, "bfi_min": 0.6}),
    ("a 60-year record", {"min_years": 60}),
    ("show me the Thames", {}),
])
def test_parse_filter_question(text, expected):
    assert sig.parse_filter_question(text) == expected


def test_normalize_filter_drops_nonsense_and_orders_the_bfi_range():
    assert sig.normalize_filter({"min_years": -3, "flood_trend": "sideways", "bfi_min": 2}) == {}
    assert sig.normalize_filter({"bfi_min": 0.7, "bfi_max": 0.2}) == {"bfi_min": 0.2, "bfi_max": 0.7}
    assert sig.normalize_filter({"flood_trend": "Increasing"}) == {"flood_trend": "rising"}


def test_filter_signatures_never_matches_a_missing_value():
    df = pd.DataFrame({
        "source": ["a", "a", "a"], "station_id": ["1", "2", "3"],
        "data_years": [60.0, 20.0, np.nan], "amax_trend": ["rising", None, "rising"], "bfi": [0.7, 0.2, np.nan],
    })
    assert list(sig.filter_signatures(df, {"min_years": 50})["station_id"]) == ["1"]
    assert list(sig.filter_signatures(df, {"flood_trend": "rising"})["station_id"]) == ["1", "3"]
    assert list(sig.filter_signatures(df, {"bfi_max": 0.5})["station_id"]) == ["2"]
    assert len(sig.filter_signatures(df, {})) == 3


def test_filter_gauges_combines_words_and_fields(tmp_path):
    df = pd.DataFrame([
        {**dict.fromkeys(sig.COLUMNS), "source": "usgs", "station_id": "1", "data_years": 70.0,
         "amax_trend": "rising", "bfi": 0.4},
        {**dict.fromkeys(sig.COLUMNS), "source": "usgs", "station_id": "2", "data_years": 55.0,
         "amax_trend": "rising", "bfi": 0.8},
        {**dict.fromkeys(sig.COLUMNS), "source": "uk_ea", "station_id": "3", "data_years": 30.0,
         "amax_trend": "rising", "bfi": 0.5},
    ])
    with patch.object(sig, "load_signatures", return_value=df):
        out = sig.filter_gauges(question="50+ years with rising floods", bfi_max=0.6)
    assert out["filter"] == {"min_years": 50, "flood_trend": "rising", "bfi_max": 0.6}
    assert out["n_match"] == 1 and out["stations"][0]["station_id"] == "1" and out["n_total"] == 3
    assert "50+ years" in out["description"]


def test_filter_gauges_spec_only_and_unreadable_table():
    assert sig.filter_gauges(question="40+ years", spec_only=True) == {
        "filter": {"min_years": 40}, "description": "40+ years of data"}
    with patch.object(sig, "load_signatures", side_effect=ImportError("no pyarrow")):
        out = sig.filter_gauges(min_years=10)
    assert out["filter"] == {"min_years": 10} and "could not be read" in out["error"]
    assert "No filter recognised" in sig.filter_gauges(question="hello", spec_only=True)["note"]


def test_filter_gauges_is_an_mcp_and_analyst_tool():
    from aquascope.ai_engine import analyst

    spec = {s.name: s for s in analyst._tool_specs()}["filter_gauges"]
    assert spec.func(question="30+ years", spec_only=True)["filter"] == {"min_years": 30}
