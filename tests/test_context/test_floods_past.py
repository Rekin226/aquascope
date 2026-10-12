"""Floods past (#547): the monthly half-degree flood grid, built from tiny frames and read back from a fake web."""

from __future__ import annotations

import gzip
import json

import pytest

from aquascope.context import floods_past as fp
from aquascope.context._common import cell_key, mirror_url

pd = pytest.importorskip("pandas")

GS_HEAD = ["uuid", "start_date", "end_date", "lat", "lon", "west", "south", "east", "north", "area_km2"]
MS_HEAD = ["lat", "lon", "year", "month", "n"]


# ── months and cells ─────────────────────────────────────────────────────────


def test_months_parse_step_and_span():
    assert fp.month_of("2024-07-14") == "2024-07" and fp.month_of("2024-07") == "2024-07"
    with pytest.raises(ValueError):
        fp.month_of("2024-13")
    assert fp.add_months("2024-11", 3) == "2025-02" and fp.add_months("2024-01", -1) == "2023-12"
    assert fp.months_between("2024-11", "2025-01") == ["2024-11", "2024-12", "2025-01"]
    assert fp.months_between("2025-01", "2024-12") == ["2024-12", "2025-01"]
    assert fp.span_label(["2024-11", "2024-12"]) == "Nov 2024 to Dec 2024"


def test_the_window_is_a_month_a_range_or_the_latest_twelve():
    assert fp.resolve_window("2021-07") == ["2021-07"]
    assert fp.resolve_window(start="2021-06", end="2021-08") == ["2021-06", "2021-07", "2021-08"]
    latest = fp.resolve_window(last="2026-02")
    assert latest[0] == "2025-03" and latest[-1] == "2026-02" and len(latest) == 12
    with pytest.raises(ValueError, match="at most 60"):
        fp.resolve_window(start="2000-01", end="2010-01")
    with pytest.raises(ValueError):
        fp.resolve_window()


def test_cells_are_half_open_half_degrees():
    assert fp.cell_of(-90, -180) == (0, 0) and fp.cell_of(90, 180) == (359, 719)
    row, col = fp.cell_of(50.6, 5.6)
    assert fp.cell_centre(row, col) == (50.75, 5.75)
    assert fp.cell_bbox(row, col) == (5.5, 50.5, 6.0, 51.0)
    # a point on the line between two cells belongs to the one above and to the right of it
    assert fp.cell_of(50.5, 5.5) == (row, col)
    assert fp.merge_cells([[[1, 2, 3, 0]], [[1, 2, 1, 7], [4, 4, 0, 9]]]) == {(1, 2): [4, 7], (4, 4): [0, 9]}


# ── building the grid ────────────────────────────────────────────────────────

NEWS = pd.DataFrame({
    "start_date": ["2021-07-14", "2021-07-15", "2021-07-29", "2021-08-01", "not a date", None],
    "lat": [50.61, 50.62, 50.55, 50.61, 50.6, 50.6],
    "lon": [5.61, 5.70, 5.52, 5.61, 5.6, 5.6],
})
RADAR = pd.DataFrame({
    "lat": [50.625, 50.575, -3.025], "lon": [5.625, 5.875, 30.025],
    "year": [2021, 2021, 2019], "month": [7, 7, 12], "n": [120, 30, 5],
})


def test_the_grid_counts_news_and_radar_apart_per_cell_and_month():
    g = fp.grid_frame(NEWS, RADAR)
    assert list(g.columns) == ["month", "row", "col", "lat", "lon", "news", "radar"]
    row = g[(g.month == "2021-07")].iloc[0]
    assert (row.news, row.radar, row.lat, row.lon) == (3, 150, 50.75, 5.75)
    assert g[g.month == "2021-08"].iloc[0].news == 1 and g[g.month == "2021-08"].iloc[0].radar == 0
    assert g[g.month == "2019-12"].iloc[0].radar == 5
    assert g.news.sum() == 4  # the undated rows are left out, not guessed
    payloads = fp.month_payloads(g)
    assert payloads["2021-07"] == {"month": "2021-07", "deg": 0.5, "cells": [[281, 371, 3, 150]]}
    assert json.loads(gzip.decompress(fp.encode_month(payloads["2021-07"]))) == payloads["2021-07"]
    idx = fp.index_payload(g, news_first="2021-07-14", news_last="2021-08-01")
    assert [m["month"] for m in idx["months"]] == ["2019-12", "2021-07", "2021-08"]
    assert idx["months"][1] == {"month": "2021-07", "cells": 1, "news": 3, "radar": 150, "news_max": 3,
                                "radar_max": 150}
    assert idx["radar"]["first"] == "2019-12" and idx["news"]["last"] == "2021-08-01"
    assert {s["licence"] for s in idx["sources"]} == {"CC-BY-4.0", "MIT"}
    assert fp.grid_frame(None, None).empty


# ── reading it back ──────────────────────────────────────────────────────────


def _publish(web, frame, *, cells=True):
    deg = fp.GRID_DEG
    idx = fp.index_payload(frame, news_first="2021-07-14", news_last="2021-08-01")
    web.add_json(mirror_url("floods/monthly/index.json"), idx)
    for month, payload in fp.month_payloads(frame, deg).items():
        web.files[mirror_url(f"floods/monthly/months/{month}.json.gz")] = fp.encode_month(payload)
    if cells:
        key = cell_key(50.6, 5.6)
        web.add_csv_gz(mirror_url(f"floods/groundsource/cells/{key}.csv.gz"), GS_HEAD, [
            ["a", "2021-07-14", "2021-07-16", 50.61, 5.61, 5.5, 50.5, 5.7, 50.7, 120.5],
            ["b", "2021-07-15", "2021-07-15", 50.62, 5.70, 5.6, 50.6, 5.8, 50.7, 3.2],
            ["c", "2021-07-29", "", 50.55, 5.52, 5.5, 50.5, 5.6, 50.6, ""],
            ["d", "2021-08-01", "2021-08-01", 50.61, 5.61, 5.5, 50.5, 5.7, 50.7, 9.0],
            ["e", "2021-07-20", "2021-07-20", 50.4, 5.61, 5.5, 50.3, 5.7, 50.5, 1.0],  # the cell below
        ])
        web.add_csv_gz(mirror_url(f"floods/microsoft/cells/{key}.csv.gz"), MS_HEAD, [
            [50.625, 5.625, 2021, 7, 120], [50.575, 5.875, 2021, 7, 30], [50.425, 5.625, 2021, 7, 999],
        ])
        web.add_json(mirror_url("manifest.json"), {"datasets": {
            "groundsource": {"folder": "floods/groundsource", "cells": [key]},
            "microsoft_floods": {"folder": "floods/microsoft", "cells": [key]}}})


def test_worldwide_by_month_with_hotspots(web):
    _publish(web, fp.grid_frame(NEWS, RADAR), cells=False)
    res = fp.flood_events_month("2021-07")
    assert res["ok"] and res["available"] and res["months"] == ["2021-07"] and res["bbox"] is None
    assert res["news"]["n_events"] == 3 and res["radar"]["detections"] == 150
    assert res["news"]["by_month"] == {"2021-07": 3} and res["radar"]["covered"]
    assert res["news"]["hotspots"][0] == {"lat": 50.75, "lon": 5.75, "news": 3, "radar": 150,
                                          "bbox": [5.5, 50.5, 6.0, 51.0]}
    assert res["summary"] == ("3 flood events in the news worldwide, Jul 2021; Sentinel-1 radar made 150 flood "
                              "detections (20 m pixels).")
    assert {s["key"] for s in res["sources"]} == {"groundsource", "microsoft_floods"}
    assert res["listed"] is False and "events" not in res["news"]


def test_the_default_is_the_latest_twelve_months_and_says_radar_stopped(web):
    late = pd.DataFrame({"start_date": ["2026-01-05"], "lat": [10.1], "lon": [10.1]})
    frame = fp.grid_frame(late, None)
    idx = fp.index_payload(frame, news_first="2026-01-05", news_last="2026-02-03")
    web.add_json(mirror_url("floods/monthly/index.json"), idx)
    for month, payload in fp.month_payloads(frame).items():
        web.files[mirror_url(f"floods/monthly/months/{month}.json.gz")] = fp.encode_month(payload)
    res = fp.flood_events_month()
    assert res["months"][0] == "2025-03" and res["months"][-1] == "2026-02"
    assert res["news"]["n_events"] == 1
    assert "radar covers Oct 2014 to Sep 2024 only" in res["summary"] and not res["radar"]["covered"]
    # a month with no file is never requested
    assert web.requested("months/2025-06") == 0
    # past the end of the news record, the answer says the record stops rather than "0 flood events"
    after = fp.flood_events_month("2026-05")
    assert after["summary"].startswith("No news record worldwide for May 2026: news runs from Jan 2026 to Feb 2026")


def test_a_clicked_cell_lists_its_events_and_agrees_with_the_grid(web):
    _publish(web, fp.grid_frame(NEWS, RADAR))
    res = fp.flood_events_month(start="2021-07", end="2021-07", bbox=fp.cell_bbox(*fp.cell_of(50.6, 5.6)))
    news, radar = res["news"], res["radar"]
    assert res["listed"] and news["n_events"] == 3 == news["events_found"]
    assert [e["start"] for e in news["events"]] == ["2021-07-29", "2021-07-15", "2021-07-14"]
    assert news["events"][2] == {"start": "2021-07-14", "end": "2021-07-16", "lat": 50.61, "lon": 5.61,
                                 "area_km2": 120.5, "source": "news"}
    assert news["events"][0]["end"] is None and news["events"][0]["area_km2"] is None
    assert radar["detections"] == 150 and radar["by_month"] == {"2021-07": 150}
    assert res["summary"].startswith("3 flood events in the news in this box, Jul 2021")
    short = fp.flood_events_month("2021-07", bbox=(5.5, 50.5, 6.0, 51.0), limit=1)
    assert len(short["news"]["events"]) == 1 and short["news"]["events_found"] == 3


def test_a_large_box_gets_counts_but_no_event_list(web):
    _publish(web, fp.grid_frame(NEWS, RADAR))
    res = fp.flood_events_month("2021-07", bbox=(-10, 40, 20, 60))
    assert res["news"]["n_events"] == 3 and not res["listed"] and "events" not in res["news"]


def test_before_the_grid_is_published(web):
    res = fp.flood_events_month("2021-07")
    assert res["ok"] and res["available"] is False and res["months"] == ["2021-07"]
    assert "not published" in res["summary"]
    assert fp.flood_events_month()["months"] == []


def test_a_bad_box_or_window_is_a_value_error(web):
    _publish(web, fp.grid_frame(NEWS, RADAR), cells=False)
    with pytest.raises(ValueError):
        fp.flood_events_month("2021-07", bbox=(6, 50, 5, 51))
    with pytest.raises(ValueError):
        fp.flood_events_month(start="2000-01", end="2020-01")


def test_a_network_failure_is_an_answer(web, monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(fp, "get_bytes", boom)
    res = fp.flood_events_month("2021-07")
    assert res["ok"] is False and "503" in res["error"]
