"""Subbasins by monthly water balance (#557): the months, the balance, the stores read, a run, and what is read back."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from aquascope import waterbalance as wb

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "water-balance.yml"


# ── months ──────────────────────────────────────────────────────────────────


def test_month_helpers():
    assert wb.month_range("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
    assert wb.month_hours("2024-02") == 29 * 24 and wb.month_seconds("2023-02") == 28 * 86400
    assert wb.next_month("2025-12") == "2026-01"
    # ERA5T to 5 October: September is the newest whole month (its last hour is 1 October 00:00)
    assert wb.last_complete_month("2026-10-05") == "2026-09"
    assert wb.last_complete_month("2026-01-01") == "2025-12"
    assert wb.month_label("2026-09") == "September 2026"


def test_month_hours_are_01_00_on_the_first_to_00_00_on_the_next_month():
    start = int((datetime(2020, 1, 1) - datetime(1900, 1, 1)).total_seconds() // 3600)
    hours = np.arange(start, start + 24 * 70)
    idx = wb.month_hour_indexes(hours, "2020-02")
    assert idx[0] == 31 * 24 + 1 and len(idx) == 29 * 24 and idx[-1] == (31 + 29) * 24
    assert len(wb.month_hour_indexes(hours, "2020-02", max_hours=6)) == 6
    with pytest.raises(RuntimeError):
        wb.hour_index(hours, datetime(2019, 1, 1))


# ── the balance ─────────────────────────────────────────────────────────────


def _basins(n=3, down=(2, -1, -1), outlets=((11,), (12,), (13, 14)), area=(1000.0, 2000.0, 500.0)):
    """Three subbasins: 0 drains into 2; 1 drains to the sea; 2 has two outlet reaches."""
    return wb.Basins(ids=[101, 102, 103][:n], area_km2=list(area), lat=[10.0, 20.0, 10.5], lon=[100.0, 110.0, 101.0],
                     outlets=[list(o) for o in outlets], down=list(down),
                     cells=np.array([0, 1, 2, 3, 4, 5]), cell_basin=np.array([0, 0, 1, 1, 2, 2]),
                     cell_weight=np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0]))


def test_runoff_is_the_outlet_flow_minus_what_comes_in_spread_over_the_subbasin():
    b = _basins()
    assert b.up == [[], [], [0]] and b.runoff_ok == [True, True, True]
    q_out = wb.outlet_flow(b.outlets, {11: 10.0, 12: 5.0, 13: 20.0, 14: 4.0})
    assert q_out == [10.0, 5.0, 24.0]
    rows = wb.balance_month("2021-04", [100.0, 50.0, 80.0], [60.0, 30.0, 40.0], q_out, b)
    secs = 30 * 86400
    assert rows[0]["q_mm"] == pytest.approx(10 * secs / 1e9 * 1000)            # 25.92 mm
    assert rows[2]["q_in_cms"] == 10.0
    assert rows[2]["q_mm"] == pytest.approx((24 - 10) * secs / 0.5e9 * 1000)    # its own 14 m3/s over 500 km2
    assert rows[0]["closure_mm"] == pytest.approx(100 - 60 - rows[0]["q_mm"])
    assert rows[1]["q_in_cms"] == 0.0


def test_no_runoff_without_an_outlet_here_or_upstream_and_a_loss_is_kept():
    b = _basins(outlets=((), (12,), (13,)))
    assert b.runoff_ok == [False, True, False]                 # 2's inflow from 0 cannot be taken off
    q_out = wb.outlet_flow(b.outlets, {12: 5.0, 13: 2.0})
    rows = wb.balance_month("2021-04", [10.0, None, 10.0], [5.0, 4.0, 5.0], q_out, b)
    assert rows[0]["q_mm"] is None and rows[2]["q_mm"] is None and rows[2]["closure_mm"] is None
    assert rows[1]["closure_mm"] is None and rows[1]["p_mm"] is None             # no rain value, no closure
    # a river that leaves smaller than it came in: negative runoff, shown as such
    b2 = _basins()
    neg = wb.balance_month("2021-04", [1.0] * 3, [1.0] * 3, wb.outlet_flow(b2.outlets, {11: 30.0, 12: 1, 13: 5, 14: 5}),
                           b2)
    assert neg[2]["q_mm"] < 0
    assert "lost" in wb.basin_sentence(neg[2])


def _month_rows(b, flows, months, p=(100.0, 100.0, 100.0)):
    rows = []
    for m in months:
        rows += wb.balance_month(m, list(p), [10.0] * 3, wb.outlet_flow(b.outlets, flows), b)
    return rows


def test_runoff_that_cannot_be_trusted_is_withheld_with_why_and_the_flows_are_kept():
    b = _basins(area=(1000.0, 2000.0, 500.0))
    months = wb.month_range("2021-01", "2021-12")
    # a sane chain: 0 makes 10 m3/s, 2 adds 4 of its own
    ok = _month_rows(b, {11: 10.0, 12: 5.0, 13: 10.0, 14: 4.0}, months[:1])
    assert wb.runoff_checks(ok, b) == {}
    # 2's paired outlet carries a fifth of what flows in: not the river leaving it (a delta, another river)
    delta = _month_rows(b, {11: 10.0, 12: 5.0, 13: 1.0, 14: 1.0}, months[:1])
    assert wb.runoff_checks(delta, b) == {103: "outlet"}
    # a big river through a small subbasin: the inflow is metres of water a month over it
    big = _basins(area=(100000.0, 2000.0, 50.0))
    through = _month_rows(big, {11: 1000.0, 12: 5.0, 13: 1000.0, 14: 30.0}, months[:1])
    assert wb.runoff_checks(through, big) == {103: "through"}
    # over a year, 0 runs off more than its rain: its reach drains more, and 2 takes that as its inflow
    wet = _month_rows(b, {11: 100.0, 12: 5.0, 13: 100.0, 14: 4.0}, months)
    assert wb.runoff_checks(wet[:3], b) == {}                   # one month is not a record
    checks = wb.runoff_checks(wet, b)
    assert checks == {101: "ratio", 103: "upstream"}
    wb.withhold_runoff(wet, checks)
    r0 = next(r for r in wet if r["hybas_id"] == 101)
    assert r0["q_mm"] is None and r0["closure_mm"] is None and r0["q_out_cms"] == 100.0 and r0["p_mm"] == 100.0
    assert "no runoff worked out here" in wb.basin_sentence(r0)
    rows = {r["hybas_id"]: r for r in wb.basins_rows(b, checks)}
    assert rows[101]["runoff_ok"] is False and rows[101]["runoff_check"] == "ratio"
    assert rows[102]["runoff_ok"] is True and rows[102]["runoff_check"] is None
    assert set(wb.RUNOFF_CHECKS) == {"outlet", "through", "ratio", "upstream"}


def test_a_missing_reach_flow_leaves_the_subbasin_without_an_outlet_flow():
    assert wb.outlet_flow([[1, 2], [], [3]], {1: 1.0, 3: 2.0}) == [None, None, 2.0]


def test_normals_need_enough_years_and_anomalies_follow_them():
    rows = []
    for y in range(1991, 2021):
        rows.append({"hybas_id": 1, "month": f"{y}-07", "p_mm": float(y - 1990), "et_mm": 10.0, "q_mm": None,
                     "closure_mm": None})
    rows.append({"hybas_id": 1, "month": "2024-07", "p_mm": 40.0, "et_mm": 12.0, "q_mm": 3.0, "closure_mm": 25.0})
    rows.append({"hybas_id": 2, "month": "2024-07", "p_mm": 40.0, "et_mm": 12.0, "q_mm": 3.0, "closure_mm": 25.0})
    norms = wb.normals(rows)
    assert norms[(1, 7)]["p"] == pytest.approx(15.5) and norms[(1, 7)]["q"] is None
    wb.add_anomalies(rows, norms)
    assert rows[-2]["p_anom"] == pytest.approx(24.5) and rows[-2]["et_anom"] == pytest.approx(2.0)
    assert rows[-2]["q_anom"] is None and rows[-1]["p_anom"] is None
    few = wb.normals(rows[:5])                                  # 5 years of 30: no normal
    assert few[(1, 7)]["p"] is None


def test_the_sentence_carries_the_numbers_of_the_row():
    row = {"month": "2026-09", "p_mm": 84.4, "et_mm": 61.2, "q_mm": 12.0, "closure_mm": 11.2, "p_anom": -20.0}
    s = wb.basin_sentence(row)
    assert s.startswith("September 2026: 84 mm of rain, 61 mm evaporated, 12 mm ran off")
    assert "+11 mm" in s and "20 mm below the 1991-2020 normal" in s
    assert "no runoff" in wb.basin_sentence({"month": "2026-09", "p_mm": 1.0, "et_mm": 1.0, "q_mm": None})


# ── reading the stores ──────────────────────────────────────────────────────


def _arr(shape, chunks, dtype, dims, **attrs):
    return ({"shape": shape, "chunks": chunks, "dtype": dtype, "compressor": None, "fill_value": None,
             "filters": None, "order": "C", "zarr_format": 2}, {"_ARRAY_DIMENSIONS": dims, **attrs})


def test_era5_month_averages_each_subbasin_by_cell_weight_and_refuses_a_gap():
    b = _basins()
    meta = {}
    for v in wb.ERA5_VARS.values():
        meta[f"{v}/.zarray"], meta[f"{v}/.zattrs"] = _arr([100, 2, 3], [1, 2, 3], "<f4", ["time", "lat", "lon"])
    start = int((datetime(2021, 2, 1) - datetime(1900, 1, 1)).total_seconds() // 3600)
    hours = np.arange(start, start + 24 * 40)
    grid_p = np.array([1, 1, 1, 2, 3, 3], dtype="<f4") * 1e-4          # metres per hour
    grid_e = np.full(6, -5e-5, dtype="<f4")
    store = {}
    for i in range(len(hours)):
        store[f"{wb.ERA5_ZARR}/total_precipitation/{i}.0.0"] = grid_p.tobytes()
        store[f"{wb.ERA5_ZARR}/evaporation/{i}.0.0"] = grid_e.tobytes()
    layout = {"meta": meta, "hours": hours}
    res = wb.era5_month(store.get, layout, b, "2021-02", workers=2)
    n = 28 * 24
    assert res["hours"] == n and res["missing"] == {"p": 0, "e": 0}
    assert res["p"][0] == pytest.approx(0.1 * n)                       # 1e-4 m = 0.1 mm an hour
    assert res["p"][1] == pytest.approx((1 * 1 + 2 * 3) / 4 * 0.1 * n)  # weighted 1:3
    assert res["et"][2] == pytest.approx(0.05 * n)
    del store[f"{wb.ERA5_ZARR}/evaporation/30.0.0"]
    gap = wb.era5_month(store.get, layout, b, "2021-02", workers=2)
    assert gap["missing"]["e"] == 1 and np.isnan(gap["et"]).all() and not np.isnan(gap["p"]).any()


def _q_store(months=30, reaches=(11, 12, 13, 14, 99)):
    base = wb.MONTHLY_Q_ZARR
    meta = {}
    meta["Q/.zarray"], meta["Q/.zattrs"] = _arr([months, len(reaches)], [20, 2], "<f4", ["time", "river_id"])
    meta["river_id/.zarray"], meta["river_id/.zattrs"] = _arr([len(reaches)], [len(reaches)], "<i4", ["river_id"])
    meta["time/.zarray"], meta["time/.zattrs"] = _arr([months], [months], "<i8", ["time"])
    q = np.zeros((40, 6), dtype="<f4")
    for j in range(len(reaches)):
        q[:months, j] = np.arange(months) + 10 * (j + 1)
    q[3, 4] = np.nan
    days = []
    d = datetime(2019, 1, 1)
    for _ in range(months):
        days.append((d - datetime(1940, 1, 1)).days)
        d = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
    store = {f"{base}/.zmetadata": json.dumps({"metadata": meta}).encode(),
             f"{base}/river_id/0": np.array(reaches, dtype="<i4").tobytes(),
             f"{base}/time/0": np.array(days, dtype="<i8").tobytes()}
    for t in range(2):
        for c in range(3):
            store[f"{base}/Q/{t}.{c}"] = np.ascontiguousarray(q[t * 20:(t + 1) * 20, c * 2:(c + 1) * 2]).tobytes()
    return store


def test_geoglows_flows_read_only_the_chunks_of_the_reaches_and_months_asked():
    store = _q_store()
    asked = []

    def fetch(url):
        asked.append(url)
        return store.get(url)

    res = wb.geoglows_flows(fetch, [11, 13, 404], ["2019-01", "2020-12", "2021-01"])
    assert res["found"] == 2 and res["last_month"] == "2021-06"
    assert res["flows"]["2019-01"] == {11: 10.0, 13: 30.0}
    assert res["flows"]["2020-12"] == {11: 10.0 + 23, 13: 30.0 + 23}
    assert res["flows"]["2021-01"][13] == 30.0 + 24
    qs = [u for u in asked if "/Q/" in u]
    assert sorted(qs) == sorted(f"{wb.MONTHLY_Q_ZARR}/Q/{t}.{c}" for t in (0, 1) for c in (0, 1))
    assert 99 not in wb.geoglows_flows(store.get, [99], ["2019-04"])["flows"]["2019-04"]   # NaN is no flow


# ── a run ───────────────────────────────────────────────────────────────────


def _square(x, y, s=1.0):
    import shapely

    return shapely.box(x, y, x + s, y + s)


def _run(tmp_path, monkeypatch, **kw):
    b = _basins()
    b.geometry = [_square(100, 10), _square(110, 20), _square(101, 10)]
    monkeypatch.setattr(wb, "era5_layout", lambda fetch: {"meta": {}, "hours": None, "final_until": "2019-04-01",
                                                          "era5t_until": "2019-05-03"})
    done = []

    def fake_month(fetch, layout, basins, month, workers=32, max_hours=None):
        done.append(month)
        k = int(month[5:7])
        return {"month": month, "p": np.array([100.0 + k, 50.0, 80.0]), "et": np.array([60.0, 30.0, np.nan]),
                "hours": max_hours or wb.month_hours(month), "missing": {"p": 0, "e": 0}}

    monkeypatch.setattr(wb, "era5_month", fake_month)
    store = _q_store()
    info = wb.run(tmp_path, fetch=store.get, basins=b, start="2019-01", **kw)
    return info, done


def test_a_run_writes_the_table_the_year_files_and_the_outlines(tmp_path, monkeypatch):
    info, done = _run(tmp_path, monkeypatch, previous=tmp_path / "none")
    assert done == ["2019-04", "2019-03", "2019-02", "2019-01"]     # newest first
    assert info["months"] == 4 and info["last_month"] == "2019-04" and not info["smoke"]
    root = tmp_path / wb.FOLDER
    rows = pq.read_table(root / "monthly.parquet").to_pylist()
    assert len(rows) == 12
    r = next(x for x in rows if x["hybas_id"] == 101 and x["month"] == "2019-01")
    assert r["q_out_cms"] == 10.0 and r["q_mm"] == pytest.approx(10 * 31 * 86400 / 1e9 * 1000, rel=1e-5)
    assert r["closure_mm"] == pytest.approx(101 - 60 - r["q_mm"], rel=1e-5) and r["era5t"] is False
    assert next(x for x in rows if x["month"] == "2019-04")["era5t"] is True    # after the final ERA5
    assert next(x for x in rows if x["hybas_id"] == 103)["closure_mm"] is None  # no ET: no closure
    year = json.loads((root / "years" / "2019.json").read_text())
    assert year["months"] == ["2019-01", "2019-02", "2019-03", "2019-04"] and year["era5t"][-1] is True
    assert year["p_mm"][0] == [101.0, 50.0, 80.0] and year["et_mm"][0][2] is None
    assert year["p_anom"][0] == [None, None, None]                       # no 1991-2020 years here
    fc = json.loads((root / "basins.geojson").read_text())
    assert [f["properties"]["k"] for f in fc["features"]] == [0, 1, 2]
    assert fc["features"][0]["properties"]["id"] == "101"
    m = json.loads((root / "manifest.json").read_text())
    assert m["computed"] == ["2019-01", "2019-02", "2019-03", "2019-04"] and "storage" in m["closure"]
    assert "ERA5" in m["licence"]["era5"] and m["normal_period"] == "1991-2020"
    basins = pq.read_table(root / "basins.parquet").to_pylist()
    assert basins[2]["up_ids"] == [101] and basins[0]["down_id"] == 103


def test_a_second_run_computes_only_what_is_missing_or_preliminary(tmp_path, monkeypatch):
    _run(tmp_path / "a", monkeypatch, previous=tmp_path / "none", end="2019-02")
    info, done = _run(tmp_path / "b", monkeypatch, previous=tmp_path / "a", refresh_months=0)
    assert done == ["2019-04", "2019-03"] and info["months"] == 4
    _, again = _run(tmp_path / "c", monkeypatch, previous=tmp_path / "b", refresh_months=1)
    assert again == ["2019-04"]                                          # ERA5T and the newest month only


def test_the_time_budget_stops_a_run_and_says_what_it_skipped(tmp_path, monkeypatch):
    info, done = _run(tmp_path, monkeypatch, previous=tmp_path / "none", time_budget_s=-1)
    assert done == [] and info["skipped_for_time"] == ["2019-01", "2019-02", "2019-03", "2019-04"]


def test_a_smoke_run_is_marked_and_never_published(tmp_path, monkeypatch):
    info, done = _run(tmp_path, monkeypatch, smoke_hours=6)
    assert info["smoke"] and done == ["2019-04"]
    called = []
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", lambda *a, **k: called.append(k))
    with pytest.raises(ValueError, match="smoke"):
        wb.publish(tmp_path)
    assert not called


def test_publish_uploads_only_the_waterbalance_folder(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, previous=tmp_path / "none")
    (tmp_path / "forecasts").mkdir()
    (tmp_path / "forecasts" / "x.parquet").write_bytes(b"x")
    seen = {}

    def fake(folder, repo_id, **kw):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen.update(kw)
        return "https://huggingface.co/commit/1"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake)
    assert wb.publish(tmp_path).endswith("/1")
    assert seen["allow_patterns"] == ["waterbalance/*", "waterbalance/years/*"]
    assert all(f.startswith("waterbalance/") for f in seen["files"])
    assert "waterbalance/years/2019.json" in seen["files"]


# ── what is read back ───────────────────────────────────────────────────────


def test_water_balance_reads_a_subbasin_a_point_and_a_month(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, previous=tmp_path / "none")
    one = wb.water_balance(103, local=tmp_path)
    assert one["found"] and one["month"] == "2019-04" and len(one["series"]) == 4
    assert one["up_ids"] == [101] and one["area_km2"] == 500.0
    assert one["sentence"].startswith("April 2019: 80 mm of rain")
    at = wb.water_balance(lat=20.5, lon=110.5, month="2019-02", local=tmp_path)
    assert at["hybas_id"] == 102 and at["series"][-1]["month"] == "2019-02"
    assert wb.water_balance(lat=-40, lon=0, local=tmp_path)["found"] is False
    month = wb.water_balance(month="2019-03", local=tmp_path)
    assert month["n"] == 3 and month["counts"]["et"] == 2 and month["medians"]["p"] == 80.0
    box = wb.water_balance(month="2019-03", bbox=[99, 9, 102, 12], local=tmp_path)
    assert box["n"] == 2 and "in this box" in box["sentence"]
    with pytest.raises(ValueError):
        wb.water_balance(variable="snow", local=tmp_path)
    assert wb.water_balance(local=tmp_path / "nothing")["available"] is False


def test_point_in_polygon_respects_holes():
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                                                [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]]}
    assert wb.feature_contains(geom, 3, 3) and not wb.feature_contains(geom, 1.5, 1.5)
    multi = {"type": "MultiPolygon", "coordinates": [[[[10, 10], [11, 10], [11, 11], [10, 10]]]]}
    assert wb.feature_contains(multi, 10.8, 10.2) and not wb.feature_contains(multi, 3, 3)


def test_outline_drops_specks_but_keeps_the_biggest_part():
    import shapely

    g = shapely.MultiPolygon([_square(0, 0, 2), _square(5, 5, 0.05)])
    o = wb.outline(g)
    assert o.geom_type == "Polygon" and o.area == pytest.approx(4)
    assert wb.outline(_square(0, 0, 0.05)).area > 0


def test_build_basins_routes_level_4_through_level_12(monkeypatch):
    import pandas as pd
    import shapely

    polys = [_square(0, 0, 2), _square(2, 0, 2), _square(10, 10, 0.1)]
    basins = pd.DataFrame({"HYBAS_ID": [1, 2, 3], "geometry": shapely.to_wkb(polys)})
    pairs = pd.DataFrame({"HYBAS_ID": [1, 2, 2], "LINKNO": [501, 502, 503]})
    topo = pd.DataFrame({"hybas_id": [11, 12, 21, 31], "next_down": [12, 21, 0, 0],
                         "sub_area": [10.0, 20.0, 30.0, 1.0], "up_area": [10.0, 30.0, 60.0, 1.0],
                         "lat": [0.5, 1.5, 1.0, 10.05], "lon": [0.5, 1.5, 3.0, 10.05]})

    def parquet(df):
        import io

        buf = io.BytesIO()
        df.to_parquet(buf)
        return buf.getvalue()

    store = {wb.BASINS_URL: parquet(basins), wb.PAIRS_URL: parquet(pairs)}
    b = wb.build_basins(store.get, topology=topo)
    assert b.ids == [1, 2, 3] and b.area_km2 == [30.0, 30.0, 1.0]
    assert b.down == [1, -1, -1] and b.up == [[], [0], []]
    assert b.outlets == [[501], [502, 503], []] and b.runoff_ok == [True, True, False]
    counts = np.bincount(b.cell_basin, minlength=3)
    # cell centres strictly inside 2 x 2 degrees: 7 x 7; a speck smaller than a cell gets the nearest one
    assert counts[0] == 49 and counts[2] == 1
    assert np.all(b.cell_weight > 0.98)


# ── the workflow ────────────────────────────────────────────────────────────


def test_the_workflow_never_publishes_a_smoke_run_and_writes_only_waterbalance():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert on["schedule"][0]["cron"].split()[2] != "*"                   # monthly
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["publish"]["default"] is True and inputs["smoke_hours"]["default"] == ""
    job = wf["jobs"]["waterbalance"]
    assert job["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}" and job["timeout-minutes"] < 360
    publish = next(s for s in job["steps"] if "waterbalance publish" in s.get("run", ""))
    assert "smoke_hours == ''" in publish["if"] and "github.event_name == 'schedule'" in publish["if"]
    assert "pull_request" not in publish["if"] or "github.event_name != 'pull_request'" in publish["if"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "harvest" not in runs and "warnings publish" not in runs and "numcodecs" in runs
