"""The forecast track record read back and said (#785): the card's lines, the ranking, the MCP tool."""

from __future__ import annotations

import pandas as pd
import pytest

from aquascope import forecast_skill as fs


def _row(model: str, variant: str, factor: float | None, n: int, *, sid: str = "A", source: str = "usgs",
         lo: int = 1, hi: int = 3, river_id: int | None = 7, lat: float | None = 45.0, lon: float | None = -122.0,
         trust: str | None = None) -> dict:
    enough = n >= 30
    if trust is None:
        trust = "unknown" if not enough or factor is None else ("good" if factor <= 2 else "fair" if factor <= 4
                                                                else "poor")
    return {"source": source, "station_id": sid, "river_id": river_id, "lat": lat, "lon": lon, "model": model,
            "variant": variant, "lead_min": lo, "lead_max": hi, "window_days": 30, "n_pairs": n,
            "first_valid": "2026-10-19", "last_valid": "2026-11-17", "enough_pairs": enough,
            "median_abs_log10_error": None, "factor": factor, "median_log10_error": None, "crps_log10": None,
            "n_threshold": 0, "hits": 0, "misses": 0, "false_alarms": 0, "trust": trust}


ROWS = [
    _row("geoglows", "raw", 4.8, 84),
    _row("geoglows", "corrected", 1.84, 84),
    _row("glofas", "raw", 2.25, 41),
    _row("glofas", "raw", 1.3, 12, lo=2, hi=2),
]


def test_the_card_says_each_model_once_corrected_when_it_can():
    block = fs.track_lines(ROWS)
    assert block["head"] == "Forecast track record here, last 30 days, 1 to 3 days ahead:"
    assert [line["text"] for line in block["lines"]] == [
        "GEOGLOWS, corrected: within a factor of 1.8 (84 pairs, 19 Oct to 17 Nov).",
        "GloFAS: within a factor of 2.3 (41 pairs, 19 Oct to 17 Nov).",   # a tie rounds up, as in the Explorer
    ]
    assert [line["trust"] for line in block["lines"]] == ["good", "fair"]
    raw = fs.track_lines(ROWS, variant="raw")
    assert raw["lines"][0]["text"].startswith("GEOGLOWS: within a factor of 4.8")


def test_not_enough_pairs_says_so_and_nothing_says_nothing():
    thin = [_row("glofas", "raw", 1.2, 12)]
    assert fs.track_lines(thin)["lines"] == [
        {"model": "glofas", "variant": "raw", "text": "GloFAS: not enough pairs yet (12 of 30).", "trust": "unknown"}]
    assert fs.track_lines([]) is None
    assert fs.track_lines([_row("glofas", "raw", 1.2, 40, lo=2, hi=2)]) is None   # no row in the card's window


def test_ranking_orders_by_verified_factor_where_enough_pairs_exist():
    rank = fs.rank_models(ROWS + [_row("glofas", "corrected", 1.5, 10)])
    assert [(r["model"], r["variant"], r["rank"]) for r in rank["ranked"]] == [
        ("geoglows", "corrected", 1), ("glofas", "raw", 2), ("geoglows", "raw", 3)]
    assert [(r["model"], r["variant"]) for r in rank["not_enough"]] == [("glofas", "corrected")]
    assert rank["sentence"] == ("Verified against this gauge over the last 30 days, 1 to 3 days ahead, GEOGLOWS, "
                                "corrected did best: within a factor of 1.8 (84 pairs), then GloFAS 2.3; GEOGLOWS 4.8.")
    none = fs.rank_models([_row("glofas", "raw", 1.2, 12)])
    assert none["ranked"] == [] and "No forecast has enough verified pairs here yet" in none["sentence"]


GAUGES = pd.DataFrame(ROWS + [_row("geoglows", "raw", 2.0, 50, sid="Z", source="uk_ea", lat=51.5, lon=-0.1,
                                   river_id=None),
                              _row("geoglows", "raw", 2.0, 50, sid="A", source="uk_ea", lat=None, lon=None)])
NETWORK = pd.DataFrame([
    {"model": "geoglows", "variant": "raw", "lead_day": d, "month": m, "n_pairs": 5000, "factor": 4.2}
    for d in (0, 1, 2, 3, 4) for m in ("all", "2026-10")])


def test_the_tool_answers_a_station_with_its_rows_ranking_and_the_network():
    res = fs.forecast_skill("A", source="usgs", gauges=GAUGES, network=NETWORK)
    assert res["gauge"] == {"source": "usgs", "station_id": "A", "river_id": 7, "lat": 45.0, "lon": -122.0}
    assert len(res["rows"]) == 4 and res["ranking"]["ranked"][0]["variant"] == "corrected"
    assert res["sentence"] == res["ranking"]["sentence"]
    assert sorted(r["lead_day"] for r in res["network"]) == [1, 2, 3]
    assert {r["month"] for r in res["network"]} == {"all"}
    assert res["min_pairs"] == 30 and res["trust_bands"][0]["trust"] == "good"


def test_the_tool_narrows_by_model_variant_and_lead():
    res = fs.forecast_skill("A", source="usgs", model="glofas", lead_days=2, gauges=GAUGES, network=NETWORK)
    assert [(r["model"], r["lead_min"], r["lead_max"]) for r in res["rows"]] == [("glofas", 2, 2)]
    assert res["network"] == []            # no GloFAS rows in this network table
    res = fs.forecast_skill("A", source="usgs", lead_days="1-3", variant="raw", gauges=GAUGES, network=NETWORK)
    assert {(r["model"], r["variant"]) for r in res["rows"]} == {("geoglows", "raw"), ("glofas", "raw")}
    assert fs._lead_window([3, 1]) == (1, 3) and fs._lead_window(None) is None
    with pytest.raises(ValueError):
        fs._lead_window([1, 2, 3])


def test_the_tool_finds_the_nearest_gauge_and_says_when_it_cannot():
    res = fs.forecast_skill(lat=45.01, lon=-122.0, gauges=GAUGES, network=NETWORK)
    assert res["gauge"]["station_id"] == "A" and res["gauge"]["distance_km"] == pytest.approx(1.11, abs=0.01)
    far = fs.forecast_skill(lat=0.0, lon=0.0, gauges=GAUGES, network=NETWORK)
    assert "within 25 km" in far["error"]
    two = fs.forecast_skill("A", gauges=GAUGES, network=NETWORK)
    assert two["sources"] == ["uk_ea", "usgs"] and "pass source" in two["error"]
    assert "No verified forecasts" in fs.forecast_skill("Q", gauges=GAUGES, network=NETWORK)["error"]
    assert "not published yet" in fs.forecast_skill("A", gauges=pd.DataFrame(), network=NETWORK)["error"]
    with pytest.raises(ValueError):
        fs.forecast_skill(gauges=GAUGES, network=NETWORK)


def test_gauge_rows_by_station_or_by_reach():
    assert len(fs.gauge_rows("usgs", "A", frame=GAUGES)) == 4
    assert {r["source"] for r in fs.gauge_rows(river_id=7, frame=GAUGES)} == {"usgs", "uk_ea"}
    assert fs.gauge_rows("usgs", "A", frame=pd.DataFrame()) == []


def test_the_evidence_ladder_carries_the_verified_ranking(monkeypatch):
    from aquascope import evidence

    obs = pd.Series([10.0] * 2000, index=pd.date_range("2015-01-01", periods=2000, freq="D"))
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, models=["nwm"], published=[], verified=ROWS)
    assert res["forecasts"]["ranked"][0]["label"] == "GEOGLOWS, corrected"
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, models=["nwm"], published=[])
    assert "forecasts" not in res
    # without published or verified rows, the published table is read for the gauge
    monkeypatch.setattr(fs, "gauge_rows", lambda source, station_id: ROWS)
    monkeypatch.setattr(evidence, "published_rows", lambda source, station_id: [])
    from aquascope import explore

    monkeypatch.setattr(explore, "fetch_series", lambda source, station_id, variable=None: {
        "series": obs, "unit": "m3/s", "variable": "discharge"})
    res = evidence.model_skill("usgs", "A", lat=1.0, lon=2.0, area_km2=100.0, models=["nwm"])
    assert res["forecasts"]["ranked"][0]["model"] == "geoglows"


def test_the_mcp_tool_is_registered_and_passes_through(monkeypatch):
    from aquascope import mcp_server

    seen = {}

    def fake(station_id, **kw):
        seen.update(kw, station_id=station_id)
        return {"ok": True}

    monkeypatch.setattr(fs, "forecast_skill", fake)
    assert mcp_server.forecast_skill("A", source="usgs", lead_days=[1, 3]) == {"ok": True}
    assert seen == {"station_id": "A", "source": "usgs", "lat": None, "lon": None, "model": None,
                    "lead_days": [1, 3], "variant": None}
    from aquascope.experimental import MCP_TOOLS

    assert "forecast_skill" in MCP_TOOLS
