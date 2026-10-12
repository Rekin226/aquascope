"""The place-context layers (#520) on synthetic GeoTIFFs and mirror files: no network anywhere."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from aquascope import context
from aquascope.context import events, gauges, rasters
from aquascope.context._common import cell_key, cells_for_bbox, mirror_url
from aquascope.context.projection import homolosine
from aquascope.registry import CONTEXT_LAYERS
from tests.cog_builder import build_tiff

LAT, LON = 45.05, 5.05


def _strips(arr: np.ndarray, x0: float, dx: float, y0: float, dy: float, **kw) -> bytes:
    """A GSW-like file: one LZW strip per row with the horizontal predictor."""
    return build_tiff(arr, tile=None, rows_per_strip=1, compression=5, predictor=2,
                      geotransform=(x0, dx, y0, dy), **kw)


def _assert_licensed(res: dict) -> None:
    assert res["sources"], res
    for s in res["sources"]:
        assert s["licence"] and s["attribution"] and s["homepage"].startswith("https://")
    assert res["attribution"]
    assert isinstance(res["summary"], str) and res["summary"]


# ── surface water ────────────────────────────────────────────────────────────


def _gsw(web, occurrence: np.ndarray, change: np.ndarray) -> None:
    for layer, arr in (("occurrence", occurrence), ("change", change)):
        web.files[rasters.gsw_url(layer, LAT, LON)] = _strips(arr, 0.0, 0.1, 50.0, -0.1)


def test_gsw_file_names_follow_the_top_left_corner():
    assert rasters.gsw_url("occurrence", 45.05, 5.05).endswith("/occurrence/occurrence_0E_50N_v1_5_2024.tif")
    assert rasters.gsw_url("change", -3.2, -61.0).endswith("/change/change_70W_0N_v1_5_2024.tif")
    assert rasters.gsw_url("occurrence", -12.0, 131.0).endswith("occurrence_130E_10S_v1_5_2024.tif")


def test_surface_water_reads_occurrence_change_and_the_neighbourhood(web):
    occ = np.zeros((100, 100), dtype="uint8")
    occ[49, 50] = 60  # the pixel holding (45.05, 5.05)
    occ[45, 52] = 90  # four rows up, two columns across: inside the 300 m search
    chg = np.full((100, 100), 100, dtype="uint8")
    chg[49, 50] = 120
    _gsw(web, occ, chg)
    res = context.surface_water(LAT, LON)
    assert res["ok"] and res["occurrence_pct"] == 60 and res["change_norm_pct"] == 20
    assert res["nearby_max_occurrence_pct"] == 90
    assert "60 % of the time" in res["summary"]
    assert "more often since 2000 than in 1984-1999 (normalised change +20 %)" in res["summary"]
    _assert_licensed(res)
    assert res["sources"][0]["key"] == "surface_water"


def test_the_change_is_worded_as_the_normalised_difference_it_is():
    assert rasters._change_words(None) == ""
    assert rasters._change_words(100) == ", all of it since 2000 (never in 1984-1999)"
    assert rasters._change_words(-100) == ", none of it since 2000"
    assert rasters._change_words(0) == ", as often since 2000 as in 1984-1999"
    assert rasters._change_words(-7).endswith("less often since 2000 than in 1984-1999 (normalised change -7 %)")
    assert "percentage point" not in rasters._change_words(20)


def test_the_neighbourhood_rows_come_in_one_request(web):
    occ = np.zeros((100, 100), dtype="uint8")
    _gsw(web, occ, np.full((100, 100), 253, dtype="uint8"))
    res = context.surface_water(LAT, LON)
    assert res["occurrence_pct"] == 0 and res["change_status"] == "not water"
    assert res["summary"].startswith("Never mapped as water")
    # header + offsets block + the point's strip, the change file, then one span for the eleven rows around it
    assert web.requested("occurrence_0E_50N") <= 5


def test_surface_water_with_no_tile_is_open_sea(web):
    res = context.surface_water(LAT, LON)
    assert res["ok"] and res["occurrence_pct"] is None
    assert "open sea" in res["summary"]


def test_surface_water_area_samples_a_grid(web):
    occ = np.zeros((100, 100), dtype="uint8")
    occ[:, 55:] = 40  # only the eastern column of the 3 x 3 grid (lon 5.67)
    _gsw(web, occ, np.full((100, 100), 100, dtype="uint8"))
    res = context.surface_water_area(4.0, 44.0, 6.0, 46.0)
    assert res["samples"] == 9 and res["points_ever_wet"] == 3
    assert "3 of 9 sample points" in res["summary"]


# ── flood hazard ─────────────────────────────────────────────────────────────

HAZARD_NAME = "ID105_N50_E000"


def _hazard(web, depths: dict[int, float | None]) -> None:
    web.files[f"{rasters.HAZARD_BASE}/all_urls.txt"] = (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP10/"
        f"{HAZARD_NAME}_RP10_depth.tif\n"
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP10/"
        f"{HAZARD_NAME}_RP10_depth_reclass.tif\n"
    ).encode()
    for rp, d in depths.items():
        arr = np.full((40, 40), -9999.0, dtype="float32")
        if d is not None:
            arr[19, 20] = d  # (45.05, 5.05) on a 0.25 degree grid from (0, 50)
        web.files[f"{rasters.HAZARD_BASE}/depth-rp{rp}/{HAZARD_NAME}/{HAZARD_NAME}_RP{rp}_depth.tif"] = build_tiff(
            arr, tile=(16, 16), compression=8, predictor=3, geotransform=(0.0, 0.25, 50.0, -0.25), nodata=-9999)


def test_flood_hazard_depths_by_return_period(web):
    _hazard(web, {10: None, 20: 0.5, 50: 1.1, 75: 1.4, 100: 1.8, 200: 2.4, 500: 3.0})
    res = context.flood_hazard(LAT, LON)
    assert res["ok"] and res["first_flooded_return_period"] == 20
    assert res["depth_m"]["10"] is None and res["depth_m"]["100"] == pytest.approx(1.8)
    assert "from the 20-year flood" in res["summary"] and "1.8 m deep at the 100-year flood" in res["summary"]
    _assert_licensed(res)


def test_a_dry_point_costs_one_hazard_file(web):
    _hazard(web, {rp: None for rp in rasters.RETURN_PERIODS})
    res = context.flood_hazard(LAT, LON)
    assert res["first_flooded_return_period"] is None and "Outside the modelled flood extent" in res["summary"]
    assert web.requested("depth-rp500") >= 1
    assert all(web.requested(f"depth-rp{rp}/") == 0 for rp in (10, 20, 50, 75, 100, 200))


def test_flood_hazard_outside_every_tile(web):
    _hazard(web, {})
    res = context.flood_hazard(-60.5, 120.0)
    assert res["depth_m"] == {} and "No flood hazard map" in res["summary"]


# ── soil ─────────────────────────────────────────────────────────────────────


def _soil(web, values: dict[str, int]) -> None:
    x, y = homolosine(LON, LAT)
    for prop, value in values.items():
        depths = rasters.TEXTURE_DEPTHS if prop in ("clay", "sand") else rasters.AWC_DEPTHS
        for depth, _ in depths:
            arr = np.full((10, 10), value, dtype="int16")
            web.files[rasters.soilgrids_url(prop, depth)] = build_tiff(
                arr, tile=(16, 16), compression=8, geotransform=(x - 5000.0, 1000.0, y + 5000.0, -1000.0),
                nodata=-32768, epsg=None)


def test_texture_class_follows_the_usda_triangle():
    assert rasters.texture_class(40, 40, 20) == "loam"
    assert rasters.texture_class(90, 5, 5) == "sand"
    assert rasters.texture_class(10, 30, 60) == "clay"
    assert rasters.texture_class(10, 70, 20) == "silt loam"
    assert rasters.texture_class(65, 15, 20) == "sandy clay loam"


def test_soil_texture_and_available_water(web):
    _soil(web, {"clay": 200, "sand": 400, "wv0033": 300, "wv1500": 150})
    res = context.soil(LAT, LON)
    assert res["ok"] and res["texture"] == "loam"
    assert (res["sand_pct"], res["silt_pct"], res["clay_pct"]) == (40.0, 40.0, 20.0)
    # (0.300 - 0.150) m3/m3 over one metre is 150 mm
    assert res["available_water_mm"] == 150
    assert len(res["layers"]) == 5 and res["layers"][0]["field_capacity"] == 0.3
    assert "Loam topsoil" in res["summary"] and "150 mm" in res["summary"]
    _assert_licensed(res)


def test_soil_where_soilgrids_has_nothing(web):
    _soil(web, {"clay": -32768, "sand": -32768, "wv0033": -32768, "wv1500": -32768})
    res = context.soil(LAT, LON)
    assert res["available_water_mm"] is None and "No SoilGrids estimate" in res["summary"]


# ── actual ET ────────────────────────────────────────────────────────────────


def _wapor(web, years: dict[int, int]) -> None:
    web.add_json(f"https://storage.googleapis.com/storage/v1/b/{rasters.WAPOR_BUCKET}/o", {"items": [
        {"name": rasters.WAPOR_PREFIX},
        *[{"name": f"{rasters.WAPOR_PREFIX}WAPOR-3.L1-AETI-A.{y}.json"} for y in years],
        *[{"name": f"{rasters.WAPOR_PREFIX}WAPOR-3.L1-AETI-A.{y}.tif"} for y in years],
    ]})
    for y, raw in years.items():
        arr = np.full((100, 100), raw, dtype="int32")
        web.files[rasters.wapor_url(y)] = build_tiff(arr, compression=8, geotransform=(0.0, 0.1, 50.0, -0.1),
                                                     nodata=-9999, scale=0.1)


def test_actual_et_latest_years_with_the_scale_factor(web):
    _wapor(web, {2022: 5120, 2023: 4880})
    res = context.actual_et(LAT, LON, years=2)
    assert res["years_available"] == [2022, 2023]
    assert res["annual_mm"] == {"2022": 512.0, "2023": 488.0}
    assert res["latest_year"] == 2023 and res["mean_mm"] == 500.0
    assert "488 mm" in res["summary"] and "2023" in res["summary"]
    _assert_licensed(res)


def test_actual_et_on_nodata(web):
    _wapor(web, {2023: -9999})
    res = context.actual_et(LAT, LON)
    assert res["annual_mm"] == {"2023": None} and "No WaPOR estimate" in res["summary"]


def test_actual_et_area_mean(web):
    _wapor(web, {2023: 6000})
    res = context.actual_et_area(4.0, 44.0, 6.0, 46.0)
    assert res["year"] == 2023 and res["mean_mm"] == 600.0


# ── flood history and dams (the Archive mirror) ─────────────────────────────

GS_HEAD = ["uuid", "start_date", "end_date", "lat", "lon", "west", "south", "east", "north", "area_km2"]
MS_HEAD = ["lat", "lon", "year", "month", "n"]
DAM_HEAD = ["gdw_id", "name", "reservoir", "river", "country", "year", "height_m", "capacity_mcm", "area_km2",
            "main_use", "dor_pc", "catchment_km2", "lat", "lon", "grand_id"]


def _mirror(web, *, dams: bool = True, floods: bool = True) -> None:
    cell = cell_key(LAT, LON)
    datasets = {}
    if floods:
        datasets["groundsource"] = {"folder": "floods/groundsource", "cells": [cell]}
        datasets["microsoft_floods"] = {"folder": "floods/microsoft", "cells": [cell]}
        web.add_csv_gz(mirror_url(f"floods/groundsource/cells/{cell}.csv.gz"), GS_HEAD, [
            ["a", "2021-05-01", "2021-05-03", 45.10, 5.10, 5.0, 45.0, 5.2, 45.2, 12.5],
            ["b", "2019-11-20", "2019-11-21", 45.20, 5.00, 4.9, 45.1, 5.1, 45.3, 3.0],
            ["c", "2020-01-01", "2020-01-02", 44.10, 4.10, 4.0, 44.0, 4.2, 44.2, 1.0],  # about 125 km away
        ])
        web.add_csv_gz(mirror_url(f"floods/microsoft/cells/{cell}.csv.gz"), MS_HEAD, [
            [45.075, 5.025, 2018, 3, 40], [45.075, 5.025, 2021, 5, 12], [44.125, 4.125, 2016, 1, 9],
        ])
        arr = np.zeros((30, 30), dtype="uint8")
        arr[29, 20] = 4  # (45.05, 5.05) in the N45E003 tile: flooded in three distinct months
        web.files[f"{events.MS_BASE}/N45/N45E003/N45E003-recurrence-80m-buffer.tif"] = build_tiff(
            arr, compression=5, geotransform=(3.0, 0.1, 48.0, -0.1))
    if dams:
        datasets["dams"] = {"folder": "dams", "cells": [cell]}
        web.add_csv_gz(mirror_url(f"dams/cells/{cell}.csv.gz"), DAM_HEAD, [
            [1, "Big Dam", "Big Lake", "Isere", "France", 1960, 80, 250.0, 12, "Hydroelectricity", 30, 5000,
             45.15, 5.15, 7],
            [2, "Small Weir", "", "Drac", "France", "", "", "", "", "", "", "", 45.06, 5.06, ""],
            [3, "Far Dam", "", "", "France", 1990, 20, 10.0, "", "", "", "", 44.2, 4.2, ""],
        ])
    web.add_json(mirror_url("manifest.json"), {"datasets": datasets})


def test_cells_cover_a_box():
    assert cell_key(45.05, 5.05) == "n44_e004" and cell_key(-0.5, -0.5) == "s02_w002"
    assert cells_for_bbox(3.5, 44.5, 4.5, 45.5) == ["n44_e002", "n44_e004"]


def test_flood_history_news_and_radar(web):
    _mirror(web)
    res = context.flood_history(LAT, LON, radius_km=25)
    news, radar = res["news"], res["radar"]
    assert news["available"] and news["n_events"] == 2
    assert news["latest"] == "2021-05-01" and news["by_year"] == {"2019": 1, "2021": 1}
    assert news["recent"][0]["start"] == "2021-05-01" and news["recent"][0]["distance_km"] < 10
    assert radar["months"] == {"2018-03": 40, "2021-05": 12} and radar["months_detected"] == 2
    assert radar["pixel"] == {"months": 3, "status": "flooding detected", "tile": "N45E003"}
    assert "2 flood events in the news within 25 km" in res["summary"]
    assert {s["key"] for s in res["sources"]} == {"groundsource", "microsoft_floods"}
    assert "CC-BY-4.0" in [s["licence"] for s in res["sources"]] and "MIT" in [s["licence"] for s in res["sources"]]


def test_flood_history_before_the_mirror_is_published(web):
    res = context.flood_history(LAT, LON)
    assert res["ok"] and res["news"]["available"] is False
    assert "not published to the Archive yet" in res["news"]["note"]
    assert res["radar"]["available"] is False
    # the live per-pixel radar read still answers: no file for this tile means nothing detected
    assert res["radar"]["pixel"]["months"] == 0
    assert res["summary"] == ("The flood-event mirror is not published yet; Sentinel-1 radar saw no flooding at "
                              "this exact spot from Oct 2014 to Sep 2024.")


def test_flood_history_area_lists_points_for_the_map(web):
    _mirror(web)
    res = context.flood_history_area(4.0, 44.0, 6.0, 46.0)
    assert res["news"]["n_events"] == 3 and len(res["points"]) == 3
    assert res["points"][0] == [45.1, 5.1, "2021-05-01"]
    assert res["radar"]["months_detected"] == 3


def test_dams_nearest_first_with_capacity(web):
    _mirror(web)
    res = context.dams(LAT, LON, radius_km=50)
    assert res["n_dams"] == 2 and [d["name"] for d in res["nearest"]] == ["Small Weir", "Big Dam"]
    assert res["total_capacity_mcm"] == 250.0
    assert res["nearest"][1]["year"] == 1960 and res["nearest"][0]["year"] is None
    assert res["summary"].startswith("2 dams within 50 km; nearest: Small Weir on the Drac")
    assert res["see_also"]["url"] == "https://www.globalwaterwatch.earth/" and "linked only" in res["see_also"]["note"]
    _assert_licensed(res)


def test_dams_area_and_the_unpublished_mirror(web):
    _mirror(web)
    res = context.dams_area(4.0, 44.0, 6.0, 46.0)
    assert res["n_dams"] == 3 and res["largest"][0]["name"] == "Big Dam"
    from aquascope.context import _common

    _common.clear_memo()
    web.files.clear()
    res = context.dams(LAT, LON)
    assert res["available"] is False and "not available yet" in res["summary"]


def test_a_dam_parked_at_0_0_has_no_position():
    from aquascope.context.events import _dam

    # the first GDW mirror read LAT_DAM/LONG_DAM, which are 0 for most barriers; no dam stands at 0, 0
    assert _dam({"name": "Hoover", "lat": "0", "lon": "0"})["lat"] is None
    assert _dam({"name": "Hoover", "lat": "36.0", "lon": "-114.7"})["lat"] == 36.0


# ── rain gauge ───────────────────────────────────────────────────────────────


def _station_line(sid: str, lat: float, lon: float, elev: float, name: str) -> str:
    return f"{sid:<11} {lat:8.4f} {lon:9.4f} {elev:6.1f}    {name:<30}"


def _record(web, sid: str, years: list[int], *, element: str = "PRCP") -> None:
    rows = []
    for y in years:
        for doy in range(365):
            d = date.fromordinal(date(y, 1, 1).toordinal() + doy)
            rows.append([sid, d.strftime("%Y%m%d"), element, 10 if doy else 523, "", "", "E", ""])
    rows.append([sid, f"{years[-1]}0101", element, 99999, "", "X", "E", ""])  # a flagged value is dropped
    web.add_csv_gz(f"{gauges.GHCN_BASE}/csv.gz/by_station/{sid}.csv.gz",
                   ["ID", "DATE", "ELEMENT", "VALUE", "M", "Q", "S", "T"], rows)
    # add_csv_gz writes a header; GHCN files have none, and the reader skips it as a non-PRCP row


def test_parse_stations_txt_fixed_width():
    rows = gauges.parse_stations_txt(_station_line("GM000001234", 50.9, 6.95, -999.9, "KOLN-BOTANISCHER GARTEN"))
    assert rows == [{"id": "GM000001234", "lat": 50.9, "lon": 6.95, "elev_m": None,
                     "name": "KOLN-BOTANISCHER GARTEN", "first_year": None, "last_year": None}]


def test_rain_gauge_skips_a_station_without_precipitation(web):
    web.files[f"{gauges.GHCN_BASE}/ghcnd-stations.txt"] = "\n".join([
        _station_line("TMAXONLY001", 45.06, 5.06, 200.0, "NEAR BUT NO RAIN"),
        _station_line("RAIN0000002", 45.20, 5.20, 300.0, "GRENOBLE TEST"),
        _station_line("FARAWAY0003", 10.00, 10.00, 10.0, "FAR AWAY"),
    ]).encode()
    _record(web, "TMAXONLY001", [2020], element="TMAX")
    _record(web, "RAIN0000002", [2021, 2022])
    res = context.rain_gauge(LAT, LON)
    assert res["ok"] and res["station"]["id"] == "RAIN0000002" and res["skipped"] == ["TMAXONLY001"]
    assert res["index"] == "ghcnd-stations.txt"
    rec = res["record"]
    assert rec["first_date"] == "2021-01-01" and rec["complete_years"] == 2 and rec["flagged_days_dropped"] == 1
    # 364 days of 1.0 mm and one of 52.3 mm
    assert rec["annual_mm"] == {"2021": 416.3, "2022": 416.3} and rec["mean_annual_mm"] == 416.3
    assert rec["wettest_day"]["mm"] == 52.3
    assert "Grenoble Test (RAIN0000002)" in res["summary"] and "416 mm a year" in res["summary"]
    _assert_licensed(res)


def test_rain_gauge_reads_the_mirror_index_first(web):
    web.add_json(mirror_url("manifest.json"), {"datasets": {"ghcn": {"folder": "ghcn", "cells": []}}})
    web.add_csv_gz(mirror_url("ghcn/prcp_stations.csv.gz"), ["id", "lat", "lon", "elev_m", "name", "first_year",
                                                             "last_year"],
                   [["RAIN0000002", 45.2, 5.2, 300, "GRENOBLE TEST", 2021, 2022]])
    _record(web, "RAIN0000002", [2021, 2022])
    res = context.rain_gauge(LAT, LON)
    assert res["index"] == "mirror" and res["station"]["first_year"] == 2021
    assert web.requested("ghcnd-stations.txt") == 0
    area = context.layer("rain_gauge", LAT, LON)  # same function by name
    assert area["station"]["id"] == "RAIN0000002"


def test_no_rain_gauge_in_reach(web):
    web.files[f"{gauges.GHCN_BASE}/ghcnd-stations.txt"] = _station_line("FARAWAY0003", 10.0, 10.0, 1.0, "X").encode()
    res = context.rain_gauge(LAT, LON, max_km=50)
    assert res["station"] is None and "No GHCN-Daily rain gauge" in res["summary"]


def test_rain_gauges_area_counts_without_opening_records(web):
    web.files[f"{gauges.GHCN_BASE}/ghcnd-stations.txt"] = "\n".join([
        _station_line("RAIN0000002", 45.2, 5.2, 300.0, "A"), _station_line("RAIN0000004", 45.4, 5.4, 300.0, "B"),
    ]).encode()
    res = context.rain_gauges_area(5.0, 45.0, 5.3, 45.3)
    assert res["n_stations"] == 1 and web.requested("by_station") == 0


# ── the bundle ───────────────────────────────────────────────────────────────


def test_place_context_runs_the_named_layers(web):
    _mirror(web)
    res = context.place_context(LAT, LON, layers="dams,flood_history")
    assert set(res["layers"]) == {"dams", "flood_history"}
    assert len(res["summary"]) == 2 and any("Global Dam Watch" in a for a in res["attribution"])
    with pytest.raises(ValueError, match="unknown context layer"):
        context.place_context(LAT, LON, layers=["nope"])
    with pytest.raises(ValueError):
        context.place_context(95.0, 0.0)


def test_area_context_and_box_checks(web):
    _mirror(web)
    res = context.area_context(4.0, 44.0, 6.0, 46.0, layers=["dams"])
    assert res["bbox"] == [4.0, 44.0, 6.0, 46.0] and res["layers"]["dams"]["n_dams"] == 3
    with pytest.raises(ValueError, match="west, south, east, north"):
        context.area_context(6.0, 44.0, 4.0, 46.0)


def test_a_failing_layer_says_so_and_keeps_its_licence(web, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("the host is down")

    monkeypatch.setattr(rasters, "_soil_at", boom)
    res = context.soil(LAT, LON)
    assert res["ok"] is False and "the host is down" in res["summary"] and res["sources"][0]["key"] == "soil"


def test_every_layer_is_registered_with_a_licence():
    for name, fn in context.LAYERS.items():
        assert callable(fn) and name in context.AREA_LAYERS
    keys = {"groundsource", "microsoft_floods", "surface_water", "flood_hazard", "dams", "soil", "actual_et",
            "rain_gauge"}
    assert keys <= set(CONTEXT_LAYERS)
    for meta in CONTEXT_LAYERS.values():
        assert meta.license and meta.attribution and meta.homepage.startswith("https://")
        if meta.mirrored:
            assert meta.redistributable, f"{meta.key} is mirrored without a licence that allows it"
    gww = CONTEXT_LAYERS["global_water_watch"]
    assert not gww.mirrored and not gww.redistributable


def test_layer_and_area_layer_by_name(web):
    _mirror(web)
    assert context.area_layer("dams", 4.0, 44.0, 6.0, 46.0)["n_dams"] == 3
    with pytest.raises(ValueError, match="unknown context layer"):
        context.area_layer("rivers", 4.0, 44.0, 6.0, 46.0)
    with pytest.raises(ValueError, match="unknown context layer"):
        context.layer("rivers", LAT, LON)
