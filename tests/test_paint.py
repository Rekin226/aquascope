"""Paint the globe (#657): aquascope.paint, its faces and its agreement with the single-station study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquascope import paint
from aquascope.analysis.trends import mann_kendall, sens_slope
from aquascope.archive.observations import series_to_csv_gz
from aquascope.explore import _annual_max, analyze_series

FIXTURES = Path(__file__).resolve().parents[1] / "explorer" / "tests" / "fixtures"


def daily(start: str, end: str, seed: int = 0, trend: float = 0.0, drop_years: tuple[int, ...] = ()) -> pd.Series:
    """A synthetic daily record: a seasonal cycle, noise, an optional trend, and whole years removed."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx)) / 365.25
    v = 20 + 10 * np.sin(2 * np.pi * idx.day_of_year / 365.25) + rng.gamma(2.0, 3.0, len(idx)) + trend * t
    s = pd.Series(v, index=idx)
    if drop_years:
        s = s[~s.index.year.isin(drop_years)]
    return s


# ── the check: the same numbers as the single-station study ─────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_calendar_years_match_the_single_station_trend_exactly(seed):
    gaps = (1975, 1990) if seed % 2 else ()
    s = daily("1961-03-10", "2020-08-20", seed=seed, trend=0.05 * (seed - 3), drop_years=gaps)
    # The annual maxima are the single-station study's, value for value.
    am = paint.annual_series(s, stat="max", start_month=1, min_days=292)
    ref = _annual_max(s)
    assert list(am.index) == list(ref.index.year) and np.array_equal(am.to_numpy(), ref.to_numpy())
    got = paint.flood_trend(s, year_start_month=1, min_days=292, min_years=8)
    mk = mann_kendall(ref.values)
    assert got["p"] == float(mk.p_value) and got["n_years"] == mk.n_samples
    one = analyze_series(s, "discharge", "m3/s")["ffa"]["amax_trend"]
    assert round(got["p"], 4) == one["p_value"] and got["tau"] == one["tau"] and got["n_years"] == one["n_years"]
    if not got["missing_years"]:
        # With no year missing, Sen's slope against the year is the package's sens_slope.
        assert got["slope_per_year"] == pytest.approx(sens_slope(ref.values).slope, rel=1e-3)
        assert got["slope_per_year"] == pytest.approx(one["sens_slope_per_year"], rel=1e-3)
    else:
        # With gaps the slope is per calendar year (a missing year stays a gap in time); on positions it would match.
        assert paint.sen_slope_years(range(len(ref)), ref.values) == pytest.approx(sens_slope(ref.values).slope)
        # #668: the single-station panel now reads the same calendar year axis, so both numbers agree.
        assert one["sens_slope_per_year"] == pytest.approx(got["slope_per_year"], rel=1e-3)
        on_positions = sens_slope(ref.values).slope
        assert one["sens_slope_per_year"] != pytest.approx(on_positions, rel=1e-2)


def test_sen_slope_counts_a_missing_year_as_time():
    assert paint.sen_slope_years([2000, 2001, 2003], [0.0, 1.0, 3.0]) == pytest.approx(1.0)
    assert sens_slope([0.0, 1.0, 3.0]).slope == pytest.approx(1.5)


# ── years ───────────────────────────────────────────────────────────────────


def test_a_water_year_is_named_by_the_year_it_ends_in():
    idx = pd.DatetimeIndex(["2019-09-30", "2019-10-01", "2020-09-30"])
    assert list(paint.year_labels(idx, 10)) == [2019, 2020, 2020]
    assert list(paint.year_labels(idx, 1)) == [2019, 2019, 2020]
    # A November flood counts for the next water year.
    s = pd.Series(1.0, index=pd.date_range("2018-10-01", "2020-09-30", freq="D"))
    s["2019-11-15"] = 50.0
    am = paint.annual_series(s, start_month=10, min_days=330)
    assert am.loc[2020] == 50.0 and am.loc[2019] == 1.0


def test_only_complete_years_count():
    s = pd.Series(1.0, index=pd.date_range("2000-10-01", "2002-09-30", freq="D"))
    s = s.drop(pd.date_range("2001-03-01", periods=36, freq="D"))   # water year 2001 keeps 329 days
    am = paint.annual_series(s, start_month=10, min_days=330)
    assert list(am.index) == [2002]
    assert list(paint.annual_series(s, start_month=10, min_days=329).index) == [2001, 2002]


def test_the_year_starts_by_hemisphere_and_question():
    assert paint.start_month_for("flood_trend", 51.5) == 10
    assert paint.start_month_for("flood_trend", -33.9) == 4
    assert paint.start_month_for("low_flow_trend", 51.5) == 4
    assert paint.start_month_for("flood_trend", 51.5, 9) == 9
    with pytest.raises(ValueError):
        paint.start_month_for("flood_trend", 0, 13)
    assert paint.year_words(10) == "October to September" and paint.year_words(1) == "calendar years"


def test_low_flows_are_the_annual_minimum_seven_day_mean():
    s = pd.Series(10.0, index=pd.date_range("2000-04-01", "2002-03-31", freq="D"))
    s["2000-08-01":"2000-08-07"] = 3.0   # one 7-day window at 3
    s["2001-07-10"] = 0.0                # a single day low: its 7-day mean is (6*10 + 0) / 7
    am = paint.annual_series(s, stat="min7", start_month=4, min_days=330)
    assert am.loc[2001] == pytest.approx(3.0) and am.loc[2002] == pytest.approx(60 / 7)


def test_a_rising_record_paints_up_and_a_short_one_is_too_short():
    up = paint.flood_trend(daily("1970-10-01", "2020-09-30", seed=9, trend=0.6), latitude=45.0)
    assert up["status"] == "painted" and up["value"] > 0 and up["direction"] == "up" and up["year_start_month"] == 10
    assert up["value"] == pytest.approx(1000 * up["slope_per_year"] / up["mean"], rel=1e-2)
    short = paint.flood_trend(daily("2010-10-01", "2020-09-30", seed=9), latitude=45.0)
    assert short["status"] == "too_short" and "needs 20" in short["note"]
    window = paint.flood_trend(daily("1950-10-01", "2020-09-30", seed=3), start_year=1990, min_years=20)
    assert window["first_year"] == 1990 and window["n_years"] == 31


# ── field significance ──────────────────────────────────────────────────────


def _rows(ps, taus=None):
    taus = taus or [0.3] * len(ps)
    return [{"key": f"s/{i}", "status": "painted", "p": p, "tau": t} for i, (p, t) in enumerate(zip(ps, taus))]


def test_false_discovery_rate_at_twice_the_global_level():
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    f = paint.field_significance(_rows(ps))
    # BH at 0.10: the largest k with p(k) <= k/10 * 0.10 is k = 6 (0.06 <= 0.06), so the six smallest survive.
    assert f["alpha_fdr"] == 0.1 and f["n_significant"] == 6 and f["p_threshold"] == 0.06 and f["field_significant"]
    g = paint.field_significance(_rows(ps), correlated=False)
    # BH at 0.05: only p(2) = 0.008 <= 2/10 * 0.05.
    assert g["alpha_fdr"] == 0.05 and g["n_significant"] == 2
    none = paint.field_significance(_rows([0.2, 0.5, 0.9]))
    assert not none["field_significant"] and "could be chance" in none["verdict"]
    assert paint.field_significance([])["n_tested"] == 0


def test_the_field_counts_directions():
    f = paint.field_significance(_rows([0.001, 0.002, 0.003, 0.5], taus=[0.4, -0.4, 0.3, 0.1]))
    assert (f["up"], f["down"]) == (2, 1) and f["local"] == {"up": 2, "down": 1, "expected_by_chance": 0.2}


# ── which gauges: the same rules as the Explorer, on one fixture ────────────


def test_plan_matches_the_shared_fixture():
    fix = json.loads((FIXTURES / "paint-plan.json").read_text())
    p = fix["params"]
    out = paint.plan(fix["stations"], fix["signatures"], bbox=p["bbox"], min_years=p["min_years"],
                     max_gauges=p["max_gauges"])
    exp = fix["expected"]
    assert [r["key"] for r in out["read"]] == exp["read"]
    assert [{k: r[k] for k in ("key", "status", "record_years", "note")} for r in out["rows"]] == exp["rows"]
    assert out["counts"] == exp["counts"]


def test_the_question_list_matches_the_explorer_fixture_and_the_map_commands():
    assert json.loads((FIXTURES / "paint-questions.json").read_text()) == paint.questions()
    from aquascope.map_commands import PAINT_QUESTIONS

    assert list(PAINT_QUESTIONS) == list(paint.QUESTIONS)


def test_plan_across_the_antimeridian():
    st = [{"source": "s", "station_id": "a", "lat": 0.0, "lon": 179.5}, {"source": "s", "station_id": "b", "lat": 0.0,
                                                                         "lon": -179.5},
          {"source": "s", "station_id": "c", "lat": 0.0, "lon": 0.0}]
    sig = [{"source": "s", "station_id": k, "record_years": 30} for k in "abc"]
    out = paint.plan(st, sig, bbox=[179.0, -1.0, -179.0, 1.0])
    assert sorted(r["key"] for r in out["read"]) == ["s/a", "s/b"]


# ── the whole painting ──────────────────────────────────────────────────────


def _archive(seed_by_id):
    def reader(source, station_id):
        if station_id == "broken":
            raise OSError("boom")
        seed, trend = seed_by_id[station_id]
        return daily("1960-10-01", "2020-09-30", seed=seed, trend=trend)
    return reader


def test_paint_end_to_end_with_coverage_said():
    ids = ["u1", "u2", "u3", "u4", "broken", "short", "gone"]
    stations = [{"source": "usgs", "station_id": sid, "name": sid, "lat": 40 + i * 0.1, "lon": -75.0,
                 "variables": ["discharge"]} for i, sid in enumerate(ids)]
    sigs = [{"source": "usgs", "station_id": sid, "record_years": 60.0, "variable": "discharge"}
            for sid in ["u1", "u2", "u3", "u4", "broken"]] + \
        [{"source": "usgs", "station_id": "short", "record_years": 9.0, "variable": "discharge"}]
    seen = []
    res = paint.paint("flood_trend", bbox=[-76, 39, -74, 41], stations=stations, signatures=sigs,
                      reader=_archive({"u1": (1, 0.8), "u2": (2, 0.7), "u3": (3, 0.0), "u4": (4, -0.1)}),
                      on_row=seen.append)
    assert res["counts"] == {"painted": 4, "too_short": 1, "no_archive": 1, "over_cap": 0, "failed": 1}
    assert len(seen) == 5 and res["field"]["n_tested"] == 4
    by = {r["key"]: r for r in res["rows"]}
    assert by["usgs/u1"]["value"] > 0 and by["usgs/broken"]["status"] == "failed"
    assert all(r["fdr_significant"] is None for r in res["rows"] if r["status"] != "painted")
    assert {f["properties"]["status"] for f in res["geojson"]["features"]} == {"painted", "too_short", "no_archive",
                                                                              "failed"}
    assert res["licences"] == [{"source": "usgs", "label": res["licences"][0]["label"], "license": "US-PD",
                                "attribution": res["licences"][0]["attribution"]}]
    m = res["methods"]
    for words in ("Mann-Kendall", "Sen's (1968)", "Benjamini and Hochberg, 1995", "Wilks (2016)",
                  "October to September", "1 not mirrored by the Archive", "US-PD", "alpha_FDR = 2 alpha_global = 0.1"):
        assert words in m, words
    assert "south of" not in m
    assert any("BAMS-D-15-00267.1" in r for r in res["references"])
    csv = paint.to_csv(res)
    assert csv.splitlines()[0].split(",")[:6] == ["source", "station_id", "name", "latitude", "longitude", "status"]
    assert "References" in paint.methods_text(res) and "Caveat" in paint.methods_text(res)
    assert res["question"]["unit"] == "% per decade" and "Not painted" in res["headline"]


def test_paint_needs_a_box_and_a_known_question():
    with pytest.raises(ValueError):
        paint.paint("flood_trend", stations=None, bbox=None)
    with pytest.raises(ValueError):
        paint.paint("rainbow", bbox=[0, 0, 1, 1], stations=[], signatures=[])


def test_baseflow_index_comes_from_signatures_without_reading_a_record():
    stations = [{"source": "uk_ea", "station_id": k, "lat": 52.0, "lon": -1.0} for k in ("a", "b", "c")]
    sigs = [{"source": "uk_ea", "station_id": "a", "record_years": 40, "bfi": 0.61, "record_start": "1980-01-01",
             "record_end": "2020-01-01", "data_years": 39.5},
            {"source": "uk_ea", "station_id": "b", "record_years": 40, "bfi": None, "notes": "bfi: failed"},
            {"source": "uk_ea", "station_id": "c", "record_years": 12, "bfi": 0.3}]

    def reader(*_):
        raise AssertionError("no record is read for the baseflow index")

    res = paint.paint("baseflow_index", bbox=[-2, 51, 0, 53], stations=stations, signatures=sigs, reader=reader)
    by = {r["key"]: r for r in res["rows"]}
    assert by["uk_ea/a"]["value"] == 0.61 and by["uk_ea/a"]["first_year"] == 1980
    assert by["uk_ea/b"]["status"] == "too_short" and "failed" in by["uk_ea/b"]["note"]
    assert by["uk_ea/c"]["status"] == "too_short"
    assert res["field"] is None and "Lyne and Hollick (1979)" in res["methods"]
    assert paint.class_of("baseflow_index", 0.61) == 3


def test_gauge_rows_reads_the_archive_file_bytes_and_says_what_failed():
    s = daily("1980-10-01", "2020-09-30", seed=4)
    data = series_to_csv_gz(s)
    rows = paint.gauge_rows("flood_trend", [
        {"key": "usgs/x", "source": "usgs", "station_id": "x", "latitude": 40.0, "longitude": -75.0, "csv_gz": data},
        {"key": "usgs/y", "source": "usgs", "station_id": "y", "csv_gz": b"not gzip"},
        {"key": "usgs/z", "source": "usgs", "station_id": "z", "fetch_error": "could not download (503)"},
    ], {"min_years": 20})
    assert rows[0]["status"] == "painted" and rows[0]["n_years"] == 40
    assert rows[1]["status"] == "failed" and rows[2]["note"] == "could not download (503)"


def test_paint_is_experimental():
    from aquascope import experimental

    assert experimental.is_experimental("aquascope.paint")
    assert "paint" in experimental.CLI and "paint_map" in experimental.MCP_TOOLS


def test_cli_paint(monkeypatch, capsys, tmp_path):
    from aquascope import cli

    seen = {}

    def fake(question, *, bbox, params, max_gauges, on_row):
        seen.update(question=question, bbox=bbox, params=params, max_gauges=max_gauges)
        rows = [{"key": "usgs/a", "source": "usgs", "station_id": "a", "status": "painted", "value": 3.0, "p": 0.01,
                 "tau": 0.3, "latitude": 40.0, "longitude": -75.0}]
        return paint.summarize(question, rows, params=params, bbox=bbox)

    monkeypatch.setattr(paint, "paint", fake)
    out = tmp_path / "t.csv"
    monkeypatch.setattr("sys.argv", ["aquascope", "paint", "--bbox=-8,49.8,2,59", "--min-years", "30",
                                     "--start-year", "1970", "--independent", "-o", str(out)])
    cli.main()
    assert seen["bbox"] == [-8.0, 49.8, 2.0, 59.0] and seen["params"]["min_years"] == 30
    assert seen["params"]["start_year"] == 1970 and seen["params"]["correlated"] is False
    text = capsys.readouterr().out
    assert "Trend in annual floods" in text and "usgs/a" in text and out.read_text().startswith("source,")


def test_mcp_paint_map(monkeypatch):
    from aquascope import mcp_server

    def fake(question, *, bbox, params, max_gauges):
        assert max_gauges == 300 and params == {"min_years": 25, "start_year": 1980}
        return paint.summarize(question, [], params=params, bbox=bbox)

    monkeypatch.setattr(paint, "paint", fake)
    out = mcp_server.paint_map([-8, 49.8, 2, 59], min_years=25, start_year=1980, max_gauges=999)
    assert "geojson" not in out and "table" not in out and out["counts"]["painted"] == 0
    assert mcp_server.paint_map([1, 2])["error"]
