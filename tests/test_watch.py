"""Watch a river (#521): the digest, thresholds and the Atom writer, with every network read replaced."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aquascope import nownext, watch

TODAY = date(2026, 10, 8)
ATOM = "{http://www.w3.org/2005/Atom}"


def _series(end: str = "2026-10-07", start: str = "1980-01-01", bump: float | None = None) -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(1)
    s = pd.Series(50 + 30 * np.sin(2 * np.pi * idx.dayofyear.to_numpy() / 365.25) + rng.normal(0, 3, len(idx)),
                  index=idx)
    if bump is not None:
        s.iloc[-5:] = bump
    return s


@pytest.fixture
def offline(monkeypatch):
    """No network: no top-up, no flood mirror, no live forecast unless a test sets one."""
    monkeypatch.setattr(nownext, "top_up", lambda s, *a, **k: (s, ""))
    monkeypatch.setattr("aquascope.context.events.flood_history", lambda *a, **k: {
        "news": {"available": True, "recent": [], "latest": None}})
    monkeypatch.setattr("aquascope.context.events.flood_history_area", lambda *a, **k: {
        "news": {"available": True, "recent": [], "latest": None}})

    def no_forecast(*a, **k):
        raise AssertionError("no live forecast expected")

    monkeypatch.setattr(nownext, "forecast", no_forecast)
    return monkeypatch


# ── items and thresholds ─────────────────────────────────────────────────────


def test_parse_item_reads_ids_and_dicts():
    g = watch.parse_item("usgs/USGS-01646500")
    assert (g["kind"], g["id"], g["station_id"]) == ("gauge", "usgs/USGS-01646500", "USGS-01646500")
    r = watch.parse_item("river:760021611")
    assert (r["kind"], r["river_id"], r["id"]) == ("reach", 760021611, "river:760021611")
    a = watch.parse_item("area:-77.5,38.1,-76.8,39")
    assert a["bbox"] == [-77.5, 38.1, -76.8, 39.0] and a["id"] == "area:-77.5,38.1,-76.8,39"
    d = watch.parse_item({"kind": "gauge", "source": "uk_ea", "station_id": "3400TH", "threshold": "10y",
                          "lat": "51.4", "river_id": "7"})
    assert d["threshold"] == {"return_period": 10.0} and d["lat"] == 51.4 and d["river_id"] == 7
    assert watch.parse_item({"id": "river:5"})["kind"] == "reach"
    for bad in ("nonsense", "area:1,2,3", "river:abc", {"kind": "lake"}, "area:0,10,1,5"):
        with pytest.raises(ValueError):
            watch.parse_item(bad)


@pytest.mark.parametrize("spec,want", [
    ("10y", {"return_period": 10.0}), ("10-year", {"return_period": 10.0}), ("T25", {"return_period": 25.0}),
    ("2 year flow", {"return_period": 2.0}), ("300", {"value": 300.0}), (1.5, {"value": 1.5}),
    ("1,200", {"value": 1200.0}), ({"value": 4}, {"value": 4.0}), ({"return_period": 5}, {"return_period": 5.0}),
    (None, None), ("", None), ("none", None),
])
def test_parse_threshold(spec, want):
    assert watch.parse_threshold(spec) == want


def test_parse_threshold_refuses_nonsense():
    for bad in ("high", "0.5y", "5000y"):
        with pytest.raises(ValueError):
            watch.parse_threshold(bad)


def test_threshold_flow_from_a_value_a_table_a_record_and_the_default():
    v = watch.threshold_flow({"value": 300})
    assert v["value"] == 300 and v["label"] == "your threshold (300 m³/s)"
    t = watch.threshold_flow({"return_period": 10}, table={"return_periods": [2, 10], "q": [100.0, 250.0]})
    assert t["value"] == 250.0 and t["label"] == "the 10-year flow (250 m³/s)"
    d = watch.threshold_flow(None, record=_series())
    assert d["return_period"] == 2.0 and d["default"] is True and 50 < d["value"] < 120
    short = watch.threshold_flow({"return_period": 10}, record=_series(start="2024-01-01"))
    assert short["value"] is None and "10 complete years" in short["error"]


# ── the digest ───────────────────────────────────────────────────────────────


def test_a_gauge_says_new_days_latest_value_and_status_change(offline):
    s = _series(bump=140.0)
    d = watch.watch_digest([{"id": "usgs/1", "name": "Big River", "series": s, "unit": "m3/s"}],
                           {"usgs/1": {"date": "2026-10-01", "class": "normal"}}, today=TODAY, forecast="off",
                           floods=False, archive=False)
    item = d["items"][0]
    assert item["new_days"] == 6 and item["latest"] == {"date": "2026-10-07", "value": 140.0}
    assert item["status"]["class"] == "much_above" and item["status"]["previous"]["class"] == "normal"
    assert item["status"]["changed"] is True and "now much above normal" in item["alerts"]
    assert item["line"].startswith("6 new days of data since 1 October, latest 140 m³/s on 7 October.")
    assert "Now much above normal (was normal on 1 October)." in item["line"]
    assert item["seen"] == {"date": "2026-10-08", "class": "much_above", "value_date": "2026-10-07"}
    assert d["n_changed"] == 1 and d["n_alerts"] == 1
    assert d["summary"].startswith("Since 1 October: 1 of 1 watched place changed, 1 with something to look at")


def test_a_plain_since_date_places_the_previous_status_from_the_record(offline):
    s = _series(bump=140.0)
    d = watch.watch_digest([{"id": "usgs/1", "series": s}], "2026-10-01", today=TODAY, forecast="off",
                           floods=False, archive=False)
    prev = d["items"][0]["status"]["previous"]
    assert prev["date"] == "2026-10-01" and prev["class"] in {"normal", "above", "below", "much_below", "much_above"}


def test_nothing_new_is_said_calmly(offline):
    s = _series(end="2026-09-20")
    d = watch.watch_digest([{"id": "usgs/1", "series": s}], {"usgs/1": "2026-10-01"}, today=TODAY,
                           forecast="off", floods=False, archive=False)
    item = d["items"][0]
    assert item["new_days"] == 0 and item["changed"] is False
    assert item["line"].startswith("No new data since 1 October; the latest value is from 20 September.")
    assert "The latest status, on 20 September, was" in item["line"]
    assert d["summary"] == "Nothing changed at your 1 watched place since 1 October."


def test_a_threshold_is_checked_on_the_latest_value_and_the_archived_forecast(offline):
    s = _series(bump=95.0)
    issued = [{"source": "usgs", "station_id": "1", "model": "geoglows", "issue_date": "2026-10-08",
               "valid_date": f"2026-10-{d:02d}", "mean": 50.0, "mean_c": v, "gauge_q2": 90.0, "gauge_q10": 150.0}
              for d, v in ((8, 80.0), (9, 120.0), (10, 160.0), (11, 140.0))]
    issued.append({**issued[0], "model": "glofas", "mean_c": None, "mean": 999.0})
    d = watch.watch_digest([{"id": "usgs/1", "series": s, "unit": "m3/s", "threshold": "10y"}], "2026-10-01",
                           today=TODAY, issued=issued, snapshot=[], floods=False, archive=False)
    item = d["items"][0]
    fc = item["forecast"]
    assert fc["from"] == "the forecast archive" and fc["scale"] == "corrected to the gauge"
    assert fc["peak"] == 160.0 and fc["peak_date"] == "2026-10-10" and fc["exceeds"] is True
    assert fc["first_above"] == "2026-10-10" and fc["threshold"]["value"] == 150.0
    assert item["latest_above"] is False
    assert "peaks at 160 m³/s on 10 October, above the 10-year flow (150 m³/s)." in item["line"]
    assert "forecast above the 10-year flow (150 m³/s)" in item["alerts"]


def test_without_a_threshold_the_forecast_is_checked_against_the_two_year_flow(offline):
    issued = [{"source": "usgs", "station_id": "1", "model": "geoglows", "issue_date": "2026-10-08",
               "valid_date": "2026-10-09", "mean": 5.0, "mean_c": 60.0, "gauge_q2": 90.0}]
    d = watch.watch_digest([{"id": "usgs/1", "series": _series()}], "2026-10-01", today=TODAY, issued=issued,
                           snapshot=[], floods=False, archive=False)
    fc = d["items"][0]["forecast"]
    assert fc["exceeds"] is False and fc["threshold"]["default"] is True
    assert "peaks at 60.0 m³/s on 9 October, under the 2-year flow (90.0 m³/s)." in d["items"][0]["line"]


def test_a_live_forecast_is_asked_when_the_archive_does_not_cover_the_gauge(offline):
    seen = {}

    def live(lat, lon, **kw):
        seen.update(lat=lat, lon=lon, **kw)
        return {"river_id": 7, "geoglows": {"date": ["2026-10-09"], "mean": [10.0], "initialized": "2026-10-07"},
                "correction": {"forecast": {"date": ["2026-10-09", "2026-10-10"], "mean": [70.0, 200.0]}},
                "gauge_thresholds": {"return_periods": [2, 5], "q": [90.0, 180.0]}}

    offline.setattr(nownext, "forecast", live)
    d = watch.watch_digest([{"id": "usgs/1", "series": _series(), "lat": 40.0, "lon": -75.0, "threshold": "5y",
                             "river_id": 7}], "2026-10-01", today=TODAY, issued=[], snapshot=[], floods=False,
                           archive=False)
    fc = d["items"][0]["forecast"]
    assert seen["river_id"] == 7 and seen["glofas"] is False and seen["obs"] is not None
    assert fc["from"] == "GEOGLOWS v2, asked now" and fc["exceeds"] is True and fc["threshold"]["value"] == 180.0


def test_a_level_gauge_gets_no_flow_forecast(offline):
    d = watch.watch_digest([{"id": "uk_ea/x", "series": _series(), "variable": "water_level", "unit": "m"}],
                           "2026-10-01", today=TODAY, issued=[], snapshot=[], floods=False, archive=False)
    item = d["items"][0]
    assert item["forecast"] is None and any("records water level" in n for n in item["notes"])


def test_the_snapshot_gives_status_when_the_record_cannot_be_read(offline, monkeypatch):
    from aquascope import explore

    def boom(*a, **k):
        raise RuntimeError("agency down")

    monkeypatch.setattr(explore, "fetch_series", boom)
    snap = [{"source": "usgs", "station_id": "1", "value_date": "2026-10-07", "value": 12.0, "percentile": 95.0,
             "class": "much_above", "n_years": 40}]
    d = watch.watch_digest(["usgs/1"], {"usgs/1": {"date": "2026-10-01", "class": "above"}}, today=TODAY,
                           snapshot=snap, issued=[], forecast="archive", floods=False, archive=False)
    item = d["items"][0]
    assert item["status"]["class"] == "much_above" and item["status"]["from"] == "the daily status snapshot"
    assert item["latest"] == {"date": "2026-10-07", "value": 12.0} and item["new_days"] is None
    assert any("agency down" in n for n in item["notes"])


def test_new_flood_events_nearby_are_those_since_the_last_visit(offline):
    events = [{"start": "2026-10-05", "lat": 40.0, "lon": -75.0}, {"start": "2026-09-01", "lat": 40.1, "lon": -75.0}]
    offline.setattr("aquascope.context.events.flood_history", lambda lat, lon, radius_km, limit: {
        "news": {"available": True, "recent": events, "latest": "2026-10-05"}})
    d = watch.watch_digest([{"id": "usgs/1", "series": _series(), "lat": 40.0, "lon": -75.0}], "2026-10-01",
                           today=TODAY, forecast="off", archive=False)
    item = d["items"][0]
    assert item["floods"]["n_new"] == 1 and "new flood event nearby" in item["alerts"]
    assert "1 new flood event in the news within 25 km, latest 5 October." in item["line"]


def test_a_missing_flood_mirror_is_a_note_not_a_change(offline):
    offline.setattr("aquascope.context.events.flood_history", lambda *a, **k: {
        "news": {"available": False, "note": "The Groundsource mirror is not published yet."}})
    d = watch.watch_digest([{"id": "usgs/1", "series": _series(end="2026-09-01"), "lat": 1.0, "lon": 1.0}],
                           "2026-10-01", today=TODAY, forecast="off", archive=False)
    assert d["items"][0]["changed"] is False and d["notes"] == ["The Groundsource mirror is not published yet."]


def test_a_reach_gets_its_modelled_status_and_forecast(offline):
    offline.setattr(nownext, "forecast", lambda lat, lon, **kw: {
        "river_id": kw["river_id"], "geoglows": {"date": ["2026-10-08", "2026-10-09"], "mean": [30.0, 45.0]},
        "thresholds": {"return_periods": [2, 5], "q": [40.0, 60.0]},
        "status": {"class": "above", "label": "above normal", "percentile": 80.0, "date": "2026-10-08"}})
    d = watch.watch_digest([{"id": "river:7", "lat": 1.0, "lon": 2.0}], {"river:7": {"date": "2026-10-01",
                                                                                     "class": "normal"}},
                           today=TODAY, floods=False, archive=False)
    item = d["items"][0]
    assert item["kind"] == "reach" and item["name"] == "River reach 7"
    assert item["status"]["modelled"] is True and item["status"]["changed"] is True
    assert item["forecast"]["exceeds"] is True
    assert "Now modelled flow above normal (was normal on 1 October)." in item["line"]
    assert "above the reach's simulated 2-year flow (40.0 m³/s)" in item["line"]


def test_an_area_counts_its_gauges_above_normal_and_new_floods(offline):
    snap = [{"source": "usgs", "station_id": str(i), "class": c, "lat": 38.5, "lon": -77.0}
            for i, c in enumerate(["above", "much_above", "normal"])]
    snap.append({"source": "usgs", "station_id": "far", "class": "above", "lat": 10.0, "lon": 10.0})
    offline.setattr("aquascope.context.events.flood_history_area", lambda *a, **k: {
        "news": {"available": True, "recent": [{"start": "2026-10-03"}], "latest": "2026-10-03"}})
    d = watch.watch_digest(["area:-77.5,38.1,-76.8,39"], {"area:-77.5,38.1,-76.8,39": {"date": "2026-10-01",
                                                                                       "high": 1}},
                           today=TODAY, snapshot=snap, archive=False)
    item = d["items"][0]
    assert item["gauges"]["n"] == 3 and item["gauges"]["high"] == 2 and item["gauges"]["rose"] is True
    assert item["line"].startswith("2 of 3 gauges with a live record are above normal today (was 1).")
    assert "in this area, latest 3 October" in item["line"] and item["seen"]["high"] == 2


def test_one_failing_item_does_not_hide_the_others(offline, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(watch, "_reach", broken)
    d = watch.watch_digest(["river:1", {"id": "usgs/1", "series": _series()}, "nonsense"], "2026-10-01",
                           today=TODAY, forecast="off", floods=False, archive=False)
    assert [i["id"] for i in d["items"]] == ["river:1", "usgs/1"]
    assert d["items"][0]["error"] == "boom" and d["errors"][0]["item"] == "nonsense"


def test_no_last_visit_looks_back_a_week_and_bad_forecast_mode_is_refused(offline):
    d = watch.watch_digest([{"id": "usgs/1", "series": _series()}], None, today=TODAY, forecast="off",
                           floods=False, archive=False)
    assert d["since"] == "2026-10-01"
    with pytest.raises(ValueError):
        watch.watch_digest([], None, forecast="maybe")
    assert watch.digest_summary([]) == "Nothing is watched yet."


def test_the_digest_is_json(offline):
    import json

    d = watch.watch_digest([{"id": "usgs/1", "series": _series()}], "2026-10-01", today=TODAY, forecast="off",
                           floods=False, archive=False)
    json.dumps(d)


# ── Atom ─────────────────────────────────────────────────────────────────────


def test_atom_feed_is_valid_rfc4287():
    text = watch.atom_feed(
        feed_id="https://example.org/feeds/usgs/1.xml", title="Big River & Co: AquaScope watch",
        updated="2026-10-08T06:41:00Z", self_url="https://example.org/feeds/usgs/1.xml",
        link="https://example.org/#s=usgs/1", rights="CC BY 4.0",
        entries=[{"id": "https://example.org/feeds/usgs/1.xml#status-2026-10-07", "title": "Big River: normal",
                  "updated": "2026-10-07", "summary": "Flow was normal <today>."},
                 {"id": "https://example.org/feeds/usgs/1.xml#forecast-2026-10-08", "title": "Big River: forecast",
                  "updated": "2026-10-08T06:41:00+00:00", "link": "https://example.org/#s=usgs/1"}])
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert_valid_atom(text)
    root = ET.fromstring(text.split("\n", 1)[1])
    entries = root.findall(f"{ATOM}entry")
    assert [e.findtext(f"{ATOM}updated") for e in entries] == ["2026-10-08T06:41:00Z", "2026-10-07T00:00:00Z"]
    assert root.findtext(f"{ATOM}title") == "Big River & Co: AquaScope watch"
    assert entries[1].findtext(f"{ATOM}summary") == "Flow was normal <today>."


def assert_valid_atom(text: str) -> None:
    """The RFC 4287 must-haves: feed id, title, updated and author; each entry an id, a title and an updated
    (RFC 3339, UTC here); a self link; ids unique."""
    import re

    root = ET.fromstring(text.split("\n", 1)[1] if text.startswith("<?xml") else text)
    assert root.tag == f"{ATOM}feed"
    stamp = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    for tag in ("id", "title", "updated"):
        assert len(root.findall(f"{ATOM}{tag}")) == 1 and root.findtext(f"{ATOM}{tag}")
    assert stamp.match(root.findtext(f"{ATOM}updated"))
    assert root.find(f"{ATOM}author/{ATOM}name").text
    assert any(lk.get("rel") == "self" and lk.get("href") for lk in root.findall(f"{ATOM}link"))
    ids = []
    for e in root.findall(f"{ATOM}entry"):
        for tag in ("id", "title", "updated"):
            assert len(e.findall(f"{ATOM}{tag}")) == 1 and e.findtext(f"{ATOM}{tag}")
        assert stamp.match(e.findtext(f"{ATOM}updated"))
        ids.append(e.findtext(f"{ATOM}id"))
    assert len(ids) == len(set(ids))


def test_a_gauge_given_by_id_takes_its_name_and_position_from_the_catalog(offline):
    offline.setattr("aquascope.archive.catalog.load_stations", lambda *a, **k: [
        {"source": "usgs", "station_id": "1", "name": "Little River", "latitude": 40.0, "longitude": -75.0}])
    seen = {}

    def floods(lat, lon, **kw):
        seen.update(lat=lat, lon=lon)
        return {"news": {"available": True, "recent": [], "latest": None}}

    offline.setattr("aquascope.context.events.flood_history", floods)
    offline.setattr(watch, "load_snapshot", lambda repo_id=None: [])
    d = watch.watch_digest([{"id": "usgs/1", "series": _series()}], "2026-10-01", today=TODAY, issued=[],
                           forecast="off")
    assert d["items"][0]["name"] == "Little River" and seen == {"lat": 40.0, "lon": -75.0}


def test_an_area_id_in_a_dict_is_an_area():
    item = watch.parse_item({"id": "area:-77.5,38.1,-76.8,39", "threshold": None})
    assert item["kind"] == "area" and item["bbox"] == [-77.5, 38.1, -76.8, 39.0]
