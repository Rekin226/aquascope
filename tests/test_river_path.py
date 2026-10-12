"""aquascope.river_path: the dams on a trace, the dams upstream of a reach and the countries crossed (#516).

No network: the Global Dam Watch mirror rows, the borders and the GEOGLOWS network are all served from memory.
"""

from __future__ import annotations

import math

import pytest

from aquascope import river_path, rivers
from aquascope.context._common import cell_key
from tests.test_rivers import _fresh_caches, small_network  # noqa: F401 - fixtures used below

COLUMNS = ["gdw_id", "name", "reservoir", "river", "country", "year", "height_m", "capacity_mcm", "area_km2",
           "main_use", "dor_pc", "catchment_km2", "lat", "lon", "grand_id"]


def _row(gdw_id: int, name: str, lat: float, lon: float, *, cap: str = "", use: str = "", dor: str = "",
         catch: str = "", reservoir: str = "") -> dict[str, str]:
    values = [str(gdw_id), name, reservoir, "Aare", "Switzerland", "1920", "", cap, "", use, dor, catch,
              str(lat), str(lon), ""]
    return dict(zip(COLUMNS, values))


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _topo(boxes: dict[str, tuple[float, float, float, float]]) -> dict:
    """An unquantised TopoJSON with one square polygon per country."""
    arcs, geoms = [], []
    for k, (name, box) in enumerate(boxes.items()):
        arcs.append(_square(*box))
        geoms.append({"type": "Polygon", "arcs": [[k]], "id": str(100 + k), "properties": {"name": name}})
    return {"type": "Topology", "arcs": arcs, "objects": {"countries": {"type": "GeometryCollection",
                                                                       "geometries": geoms}}}


# ── countries ────────────────────────────────────────────────────────────────


def test_parse_countries_decodes_quantised_delta_arcs_and_reversed_rings():
    topo = {
        "type": "Topology", "transform": {"scale": [0.5, 0.5], "translate": [10.0, 20.0]},
        # a 1 x 1 degree square from (10, 20), delta-encoded; the second country walks the same arc backwards
        "arcs": [[[0, 0], [2, 0], [0, 2], [-2, 0], [0, -2]]],
        "objects": {"countries": {"type": "GeometryCollection", "geometries": [
            {"type": "Polygon", "arcs": [[0]], "id": "1", "properties": {"name": "Squareland"}},
            {"type": "MultiPolygon", "arcs": [[[-1]]], "id": "2", "properties": {"name": "Backwards"}},
            {"type": None, "id": "3"},
        ]}},
    }
    countries = river_path.parse_countries_topojson(topo)
    assert [c.name for c in countries] == ["Squareland", "Backwards"]
    square = countries[0]
    assert square.bbox == (10.0, 20.0, 11.0, 21.0) and square.code == "1"
    assert square.contains(10.5, 20.5) and not square.contains(11.5, 20.5) and not square.contains(10.5, 21.5)
    assert countries[1].contains(10.2, 20.9)


def test_a_hole_is_not_inside_the_country():
    topo = {"type": "Topology", "arcs": [_square(0, 0, 4, 4), _square(1, 1, 3, 3)],
            "objects": {"countries": {"type": "GeometryCollection", "geometries": [
                {"type": "Polygon", "arcs": [[0], [1]], "properties": {"name": "Ring"}}]}}}
    ring = river_path.parse_countries_topojson(topo)[0]
    assert ring.contains(0.5, 0.5) and not ring.contains(2.0, 2.0)


def test_countries_along_names_them_in_the_order_the_water_enters():
    countries = river_path.parse_countries_topojson(_topo({"Upland": (7.0, 46.0, 8.0, 48.0),
                                                           "Lowland": (8.0, 46.0, 10.0, 48.0)}))
    coords = [[7.5, 47.0], [8.5, 47.0], [9.5, 47.0], [10.5, 47.0]]  # the last third is over the sea
    res = river_path.countries_along(coords, countries=countries)
    assert [c["name"] for c in res["countries"]] == ["Upland", "Lowland"]
    km_per_deg = 111.32 * math.cos(math.radians(47.0))
    up, low = res["countries"]
    assert up["first_km"] == 0.0 and up["km"] == pytest.approx(0.5 * km_per_deg, rel=0.02)
    assert low["first_km"] == pytest.approx(0.5 * km_per_deg, rel=0.02) and low["km"] == pytest.approx(
        2.0 * km_per_deg, rel=0.02)
    assert res["summary"] == "Crosses Upland and Lowland." and "public domain" in res["licence"]


def test_countries_along_without_the_borders_says_so(monkeypatch):
    def offline():
        raise RuntimeError("jsDelivr unreachable")

    monkeypatch.setattr(river_path, "_countries", offline)
    res = river_path.countries_along([[7.5, 47.0], [8.5, 47.0]])
    assert res["available"] is False and res["countries"] == [] and "jsDelivr" in res["error"]


def test_countries_are_read_once_from_the_cdn(monkeypatch):
    import json

    seen = []

    def fake_text(url):
        seen.append(url)
        return json.dumps(_topo({"Only": (0.0, 0.0, 1.0, 1.0)}))

    monkeypatch.setattr(rivers, "_fetch_text", fake_text)
    assert [c.name for c in river_path._countries()] == ["Only"]
    river_path._countries()
    assert seen == [river_path.COUNTRIES_URL] and "world-atlas@2.0.2" in seen[0]


# ── dams along a path ────────────────────────────────────────────────────────


PATH = [[8.00, 47.0], [8.01, 47.0], [8.02, 47.0], [8.03, 47.0]]


def test_dams_along_places_each_dam_on_the_path_with_its_facts():
    rows = [
        _row(2, "Lower weir", 47.0003, 8.025, catch="40"),
        _row(1, "Big dam", 47.0005, 8.005, cap="25", use="Hydroelectricity", dor="0.8", catch="30"),
        _row(3, "Far away", 47.2, 8.5, cap="900"),
        _row(4, "Side stream", 47.001, 8.015, cap="5", catch="2"),
        _row(5, "", 47.0, 8.01, reservoir="Lake"),
        dict(_row(6, "No place", 47.0, 8.01), lat=""),
        _row(7, "Parked at 0, 0", 0.0, 0.0, cap="99"),
    ]
    res = river_path.dams_along(PATH, reach_of_segment=[11, 12, 13], dam_km=2.0, min_catchment_km2=20.0, rows=rows)
    assert res["available"] is True and res["n_dams"] == 3
    # GDW leaves a name out more often than not; the reservoir's name stands in
    assert [d["name"] for d in res["dams"]] == ["Big dam", "Lake", "Lower weir"]
    big = res["dams"][0]
    assert big["capacity_mcm"] == 25.0 and big["purpose"] == "Hydroelectricity" and big["dor_pc"] == 0.8
    assert big["dor_source"] == "Global Dam Watch" and big["river_id"] == 11
    assert big["along_km"] == pytest.approx(0.005 * 111.32 * math.cos(math.radians(47.0)), abs=0.1)
    assert big["distance_km"] == pytest.approx(0.0005 * 110.574, abs=0.01)
    assert res["dams"][1]["reservoir"] == "Lake" and res["dams"][2]["river_id"] == 13
    assert res["side_streams_left_out"] == 1 and res["total_capacity_mcm"] == 25.0
    assert res["summary"] == ("3 dams within 2 km of the path, holding about 25 million m3; the largest is Big dam "
                              "at km 0.")
    assert "CC BY 4.0" in res["source"]["attribution"]


def test_a_dam_behind_the_start_is_upstream_not_on_the_way():
    rows = [_row(1, "Behind", 47.0, 7.99, cap="5"), _row(2, "At the start", 47.0001, 8.0, cap="3"),
            _row(3, "On the way", 47.0003, 8.015, cap="7")]
    res = river_path.dams_along(PATH, rows=rows)
    assert [d["name"] for d in res["dams"]] == ["At the start", "On the way"]
    assert res["behind_the_start"] == 1


def test_a_dam_parked_at_0_0_by_an_old_mirror_is_ignored():
    rows = [_row(7, "Parked", 0.0, 0.0, cap="99"), _row(8, "Real", 0.0005, 0.0005, cap="1")]
    res = river_path.dams_along([[0.0, 0.0], [0.001, 0.001]], rows=rows)
    assert [d["name"] for d in res["dams"]] == ["Real"]


def test_dams_along_reads_the_cells_the_path_touches(monkeypatch):
    asked = []

    def fake_rows(keys):
        asked.append(list(keys))
        return [_row(1, "Big dam", 47.0005, 8.005, cap="25")]

    monkeypatch.setattr(river_path, "_dam_rows", fake_rows)
    res = river_path.dams_along(PATH)
    # the path starts at 8.00 E, so its 2 km buffer reaches into the cell west of it too
    assert asked == [[cell_key(47.0, 7.99), cell_key(47.0, 8.0)]] and res["n_dams"] == 1


def test_path_cells_cover_the_buffer_across_a_cell_edge():
    keys = river_path.path_cells([[9.999, 47.0], [10.001, 47.0]], pad_km=2.0)
    assert keys[0] == cell_key(47.0, 9.999) and cell_key(47.0, 10.001) in keys and len(keys) == 2


def test_dams_along_before_the_mirror_is_published():
    res = river_path.dams_along(PATH)  # tests/conftest.py: the mirror reads as not published
    assert res["available"] is False and res["dams"] == [] and res["n_dams"] is None
    assert "not published" in res["note"] and "not available yet" in res["summary"]


def test_dams_along_when_the_mirror_read_fails(monkeypatch):
    def boom(keys):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(river_path, "_dam_rows", boom)
    res = river_path.dams_along(PATH)
    assert res["available"] is False and "HTTP 500" in res["error"]


# ── the trace carries them ───────────────────────────────────────────────────


def test_the_trace_lists_the_dams_the_countries_and_the_dams_upstream(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: [
        _row(1, "Big dam", 47.0005, 8.025, cap="25", use="Hydroelectricity", catch="20")])
    monkeypatch.setattr(river_path, "_countries", lambda: river_path.parse_countries_topojson(
        _topo({"Upland": (7.9, 46.9, 8.015, 47.1), "Lowland": (8.015, 46.9, 8.2, 47.1)})))
    res = rivers.trace_downstream(230000001, stations=[])
    assert [d["name"] for d in res["dams"]] == ["Big dam"] and res["dams"][0]["river_id"] == 230000003
    assert res["dams_info"]["n_dams"] == 1 and "dams" not in res["dams_info"]
    assert [c["name"] for c in res["countries"]] == ["Upland", "Lowland"]
    assert res["countries_info"]["summary"] == "Crosses Upland and Lowland."
    assert res["upstream_dams"]["available"] is True and res["upstream_dams"]["n_dams"] == 0
    assert "1 dam." in res["message"]


def test_the_trace_stands_without_the_mirror_or_the_borders(small_network):  # noqa: F811
    res = rivers.trace_downstream(230000001, stations=[])
    assert res["dams"] == [] and res["dams_info"]["available"] is False
    assert res["countries"] == [] and res["countries_info"]["available"] is False
    assert res["upstream_dams"]["available"] is False and "not published" in res["upstream_dams"]["note"]
    assert any("Global Dam Watch mirror is not published" in n for n in res["notes"])
    assert res["n_reaches"] == 3 and len(res["geometry"]["coordinates"]) >= 2


def test_the_trace_can_leave_the_path_context_out(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: pytest.fail("no dams asked for"))
    res = rivers.trace_downstream(230000001, stations=[], path_context=False)
    assert res["dams"] == [] and res["dams_info"] is None and res["upstream_dams"] is None


# ── dams upstream ────────────────────────────────────────────────────────────


def test_upstream_dams_keeps_the_dams_that_drain_to_the_reach(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: [
        _row(1, "Above", 47.0004, 8.005, cap="6.311", catch="9"),  # on reach 230000001, upstream of 230000002
        _row(2, "Below", 47.0004, 8.025, cap="500", catch="15"),  # on reach 230000003, downstream
        _row(3, "Too big", 47.0004, 8.006, cap="70", catch="5000"),  # drains more than the reach: not above it
        _row(4, "Elsewhere", 48.5, 9.5, cap="80"),  # outside the basin's cells: never matched
    ])
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: {"230000002": [1.0, 3.0, None],
                                                                          "datetime": ["1940", "1941", "1942"]})
    res = river_path.upstream_dams(230000002)
    assert res["available"] is True and res["complete"] is True and res["regulated"] is True
    assert [d["name"] for d in res["dams"]] == ["Above"] and res["dams"][0]["river_id"] == 230000001
    assert res["dams"][0]["on_this_reach"] is False and res["upstream_area_km2"] == pytest.approx(18.0)
    assert res["mean_flow_cms"] == 2.0 and "modelled" in res["mean_flow_source"]
    # 6.311 million m3 against 2 m3/s for a year (63.1 million m3)
    assert res["degree_of_regulation_pct"] == pytest.approx(10.0, abs=0.01)
    assert res["summary"].startswith("Regulated upstream: 1 dam in Global Dam Watch drain")
    assert "10 % of a year's mean flow" in res["summary"] and "Approximate" in res["note"]


def test_upstream_dams_without_the_flow_or_with_none_upstream(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: [_row(2, "Below", 47.0004, 8.025, cap="500")])
    monkeypatch.setattr(rivers, "_fetch_json", lambda *a, **k: pytest.fail("no flow asked for"))
    res = river_path.upstream_dams(230000002, with_flow=False)
    assert res["regulated"] is False and res["n_dams"] == 0 and "degree_of_regulation_pct" not in res
    assert res["summary"] == "No dams in Global Dam Watch upstream of this reach."


def test_upstream_dams_in_a_basin_too_large_to_search(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: pytest.fail("no read for a basin this large"))
    res = river_path.upstream_dams(230000002, max_cells=0)
    assert res["complete"] is False and res["regulated"] is None and "too large" in res["summary"]


def test_upstream_dams_counts_the_dams_past_the_tile_cap(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: [_row(1, "Above", 47.0004, 8.005, cap="6")])
    res = river_path.upstream_dams(230000002, with_flow=False, max_tiles=0)
    assert res["unchecked"] == 1 and res["complete"] is False and "were not checked" in res["summary"]
    # nothing found among the dams checked is not "unregulated" while one went unchecked
    assert res["regulated"] is None and "among those checked" in res["summary"]


def test_upstream_dams_from_a_hillside(monkeypatch):
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: {
        "snapped": False, "message": "No stream within 1 km of this point."})
    res = river_path.upstream_dams(lat=47.0, lon=8.0)
    assert res["available"] is False and res["summary"].startswith("No stream")
    with pytest.raises(ValueError):
        river_path.upstream_dams()


def test_the_rivers_module_exposes_upstream_dams(small_network, monkeypatch):  # noqa: F811
    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: [])
    assert rivers.upstream_dams(230000002, with_flow=False)["n_dams"] == 0
