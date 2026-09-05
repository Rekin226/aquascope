"""``assess_site`` (#306): the data reconnaissance behind "what can be answered here".

The catalog is in memory and BasinATLAS and the similarity search are
replaced by fakes; what is checked is the contract the CLI, the MCP tool, the
Analyst and the Explorer card all read (docs/solve-design.md).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aquascope import explore
from aquascope.archive import catalog
from aquascope.methods import DEFENSIBLE, MARGINAL, NOT_DEFENSIBLE, method_ids


def _row(source, sid, name, lat, lon, variables, start, end=None, river=None):
    return {"source": source, "station_id": sid, "name": name, "latitude": lat, "longitude": lon,
            "variables": list(variables), "period_start": start, "period_end": end, "url": None, "river": river,
            "country": None, "agency": None, "license": None, "redistributable": True, "extra": {}}


KINGSTON = _row("uk_ea", "8496ce69", "Kingston", 51.4150, -0.3080, ["discharge", "water_level"], "1883-01-01",
                river="River Thames")
RAIN = _row("uk_ea", "rain1", "Teddington rain", 51.43, -0.33, ["precipitation"], "1990-01-01", "2015-06-30")
SEINE_SHORT = _row("hubeau_hydrometrie", "F1", "La Seine à Paris", 48.85, 2.35, ["discharge"], "2019-01-01",
                   "2024-01-01", river="La Seine")
MARNE_12 = _row("hubeau_hydrometrie", "F2", "La Marne à Gournay", 48.86, 2.58, ["discharge"], "2012-01-01",
                "2024-01-01", river="La Marne")
STUB = _row("usgs", "USGS-9", "Stub Creek", 40.0, -105.0, ["discharge"], "2023-01-01", "2023-05-31")
CATALOG = [KINGSTON, RAIN, SEINE_SHORT, MARNE_12, STUB]

CATCHMENT = {
    "sub_basin": {"hybas_id": 2120001, "up_area": 9948.0},
    "upstream": {"n_sub_basins": 400, "note": "n"},
    "attributes": {"area_km2": 9900.0, "upstream_area_km2": 9948.0, "elevation_m": {"value": 110.0},
                   "aridity_index": {"value": 1.2}, "degree_of_regulation_pct": {"value": 3.0}},
}
DONORS = {"k": 10, "n_candidates": 3000, "stations": [{"source": "x", "station_id": str(i)} for i in range(10)]}


@pytest.fixture
def small_catalog():
    catalog.set_catalog(CATALOG)
    try:
        yield CATALOG
    finally:
        catalog.set_catalog(None)


@pytest.fixture
def archive():
    with patch("aquascope.mcp_server.describe_catchment", return_value=CATCHMENT) as desc, \
         patch("aquascope.mcp_server.similar_basins", return_value=DONORS) as sim:
        yield desc, sim


def _row_for(res, method):
    return next(r for r in res["sufficiency"] if r["method"] == method)


def test_gauged_site_with_a_long_record_supports_at_site_flood_frequency(small_catalog, archive):
    res = explore.assess_site(51.415, -0.308, problem="flood_risk", return_period=100)
    json.dumps(res)  # the contract is plain JSON
    assert set(res) == {"point", "stations", "catchment", "context", "sufficiency", "notes"}
    assert res["point"] == {"lat": 51.415, "lon": -0.308}
    # nearest first, true catalog spans
    assert [s["station_id"] for s in res["stations"]] == ["8496ce69", "rain1"]
    assert res["stations"][0]["distance_km"] == 0.0 and res["stations"][0]["years"] > 140
    assert res["stations"][0]["period_start"] == "1883-01-01"
    ctx = res["context"]
    assert not ctx["ungauged"] and ctx["years_by_variable"]["discharge"] > 140
    assert ctx["resolution_by_variable"] == {"discharge": "daily", "water_level": "daily", "precipitation": "daily"}
    assert ctx["area_km2"] == 9948.0 and ctx["donors"] == 10 and ctx["return_period"] == 100.0
    assert set(ctx["available"]) == {"forcing", "glofas", "temperature"}
    assert res["catchment"]["upstream_area_km2"] == 9948.0 and res["catchment"]["dams"] == 3.0
    ffa = _row_for(res, "at_site_flood_frequency")
    assert ffa["status"] == DEFENSIBLE and ffa["station"] == {"source": "uk_ea", "station_id": "8496ce69"}
    assert ffa["tool"] == "analyze_station" and ffa["label"]
    # defensible first
    order = [r["status"] for r in res["sufficiency"]]
    assert order == sorted(order, key=[DEFENSIBLE, MARGINAL, NOT_DEFENSIBLE].index)
    # the default fetch serves the full record now (#270), so no note claims a 40-year cap
    assert not any("last 40 years" in n for n in res["notes"])
    assert any("daily is assumed" in n for n in res["notes"])
    assert any("ends in 2015" in n for n in res["notes"])  # the rain gauge stopped reporting


def test_the_station_a_span_came_from_is_listed_even_at_a_dense_site(archive):
    """At a dense site the well that gives the groundwater years can be beyond the nearest 25; the playbook
    that reads the context and then asks for the station must find it in the list."""
    from aquascope import playbooks as pbk

    dense = [_row("uk_ea", f"g{i}", f"Gauge {i}", 53.0 + 0.001 * i, -2.0, ["discharge"], "1990-01-01")
             for i in range(30)]
    well = _row("uk_ea", "well", "The well", 53.04, -2.0, ["groundwater_level"], "1995-01-01")
    catalog.set_catalog(dense + [well])
    try:
        res = explore.assess_site(53.0, -2.0)
    finally:
        catalog.set_catalog(None)
    assert res["context"]["years_by_variable"]["groundwater_level"] > 30
    assert len(res["stations"]) == explore._MAX_STATIONS_LISTED + 1
    assert res["stations"][-1]["station_id"] == "well", "beyond the nearest 25, listed all the same"
    study = pbk.plan("groundwater_decline", res)
    assert study.plan["branch"] == "well" and study.plan["station"]["station_id"] == "well"


def test_problem_filters_the_table_and_unknown_problem_raises(small_catalog, archive):
    res = explore.assess_site(51.415, -0.308, problem="flood_risk")
    assert {r["method"] for r in res["sufficiency"]} == set(method_ids("flood_risk"))
    with pytest.raises(ValueError, match="unknown problem"):
        explore.assess_site(51.415, -0.308, problem="lava")


def test_gauged_site_with_a_short_record_is_not_defensible_for_flood_frequency(small_catalog, archive):
    res = explore.assess_site(48.85, 2.35, radius_km=10, return_period=100)
    assert res["context"]["years_by_variable"] == {"discharge": 5.0}
    ffa = _row_for(res, "at_site_flood_frequency")
    assert ffa["status"] == NOT_DEFENSIBLE and "below the 10-year floor" in ffa["reason"]
    assert ffa["station"] == {"source": "hubeau_hydrometrie", "station_id": "F1"}
    assert _row_for(res, "flow_duration")["status"] == DEFENSIBLE  # five years is enough for an FDC
    assert _row_for(res, "trend_mann_kendall")["status"] == NOT_DEFENSIBLE


def test_return_period_far_beyond_the_record_is_marginal(small_catalog, archive):
    res = explore.assess_site(48.86, 2.58, radius_km=10, problem="flood_risk", return_period=100)
    ffa = _row_for(res, "at_site_flood_frequency")
    assert ffa["status"] == MARGINAL and "T = 100" in ffa["reason"]
    assert ffa["station"]["station_id"] == "F2"


def test_ungauged_site_points_at_the_regionalisation_path(small_catalog, archive):
    res = explore.assess_site(-20.0, 130.0)
    assert res["stations"] == [] and res["context"]["ungauged"] is True
    assert res["context"]["years_by_variable"] == {}
    for method in ("at_site_flood_frequency", "flow_duration", "spi", "sgi", "gr4j_calibration", "trend_mann_kendall"):
        row = _row_for(res, method)
        assert row["status"] == NOT_DEFENSIBLE and row["station"] is None, method
    for method in ("similar_basins", "regionalize_signatures", "glofas_cross_check"):
        assert _row_for(res, method)["status"] == DEFENSIBLE, method
    assert any("regionalisation path" in n for n in res["notes"])
    assert any(n.startswith("No catalog gauge within 50 km") for n in res["notes"])
    # a point with nothing in range does not get one "too far" note per variable
    assert not any("beyond the 50 km radius" in n for n in res["notes"])


def test_nearest_gauge_beyond_the_radius_is_noted(small_catalog, archive):
    res = explore.assess_site(48.85, 2.35, radius_km=10, problem="drought")
    assert _row_for(res, "spi")["status"] == NOT_DEFENSIBLE
    too_far = [n for n in res["notes"] if "Nearest precipitation gauge" in n]
    assert len(too_far) == 1 and "Teddington rain" in too_far[0] and "beyond the 10 km radius" in too_far[0]


def test_suspiciously_short_catalog_span_is_flagged(small_catalog, archive):
    res = explore.assess_site(40.0, -105.0, radius_km=5)
    assert res["context"]["years_by_variable"]["discharge"] < 1
    assert any("suspiciously short" in n for n in res["notes"])


def test_catchment_failure_is_a_note_not_an_error(small_catalog):
    with patch("aquascope.mcp_server.describe_catchment", return_value={"error": "basins files unreachable"}), \
         patch("aquascope.mcp_server.similar_basins", return_value={"error": "no catchment"}):
        res = explore.assess_site(51.415, -0.308, problem="ungauged_flow")
    assert res["catchment"] == {"error": "basins files unreachable"}
    assert res["context"]["area_km2"] is None and res["context"]["donors"] is None
    assert any("Catchment not described" in n for n in res["notes"])
    assert any("Donor search not available" in n for n in res["notes"])
    assert _row_for(res, "regionalize_signatures")["status"] == NOT_DEFENSIBLE


def test_a_lumped_model_is_refused_above_its_area_ceiling(small_catalog):
    big = {**CATCHMENT, "attributes": {"area_km2": 101033.0, "upstream_area_km2": 101033.0}}
    with patch("aquascope.mcp_server.describe_catchment", return_value=big), \
         patch("aquascope.mcp_server.similar_basins", return_value=DONORS):
        res = explore.assess_site(51.415, -0.308, problem="climate_change")
    gr4j = _row_for(res, "gr4j_calibration")
    assert gr4j["status"] == NOT_DEFENSIBLE and "ceiling" in gr4j["reason"]


def test_caller_hints_skip_the_archive_lookups(small_catalog, archive):
    desc, sim = archive
    res = explore.assess_site(51.415, -0.308, area_km2=250.0, donors=4)
    assert not desc.called and not sim.called
    assert res["catchment"] == {"area_km2": 250.0, "upstream_area_km2": 250.0, "source": "caller"}
    assert res["context"]["area_km2"] == 250.0 and res["context"]["donors"] == 4
    assert any("supplied by the caller" in n for n in res["notes"])
