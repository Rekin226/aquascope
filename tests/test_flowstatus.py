"""Flow against normal (#559): the classes, the browser file, the climatology and the daily step, on tiny stores."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from aquascope import cli, mcp_server
from aquascope import flowstatus as fs
from aquascope.archive import warnings as fw

pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(__file__).resolve().parents[1]

# ── the classes ─────────────────────────────────────────────────────────────


def test_classify_uses_the_hydrosos_bounds_and_ties_go_to_normal():
    p = (10.0, 20.0, 40.0, 60.0)
    assert fs.classify(5, *p) == 1 and fs.classify(15, *p) == 2 and fs.classify(30, *p) == 3
    assert fs.classify(50, *p) == 4 and fs.classify(70, *p) == 5
    assert fs.classify(10, *p) == 2 and fs.classify(20, *p) == 3 and fs.classify(40, *p) == 3
    assert fs.classify(60, *p) == 4


def test_classify_leaves_dry_channels_and_gaps_unclassed():
    assert fs.classify(0, 0, 0, 0, 0) == 0                 # dry nine days in ten: no normal to compare with
    assert fs.classify(None, 1, 2, 3, 4) == 0
    assert fs.classify(float("nan"), 1, 2, 3, 4) == 0
    assert fs.classify(3, 1, 2, float("nan"), 4) == 0
    assert fs.classify(0, 0, 0, 0, 0.5) == 3                # mostly dry, but it flows sometimes: 0 is normal


def test_classify_array_matches_classify():
    rng = np.random.default_rng(1)
    p = np.sort(rng.gamma(2, 10, (4, 300)), axis=0)
    p[:, :5] = 0
    q = rng.gamma(2, 12, 300)
    q[7] = np.nan
    got = fs.classify_array(q, p)
    assert list(got) == [fs.classify(q[i], *p[:, i]) for i in range(300)]
    assert set(got[5:]) <= {0, 1, 2, 3, 4, 5} and got[7] == 0


# ── the browser file ────────────────────────────────────────────────────────


def test_the_browser_file_round_trips_and_leaves_out_unclassed_reaches():
    ids = np.array([820425136, 5, 3, 128, 127, 16384, 2**31 - 1])
    cls = np.array([5, 1, 0, 2, 3, 4, 3], dtype="uint8")
    blob = fs.encode(ids, cls)
    assert blob[:2] == b"\x1f\x8b" and fs.encode(ids, cls) == blob      # gzip, the same bytes every time
    got_ids, got_cls = fs.decode(blob)
    assert list(got_ids) == [5, 127, 128, 16384, 820425136, 2**31 - 1]
    assert list(got_cls) == [1, 3, 2, 4, 5, 3]
    raw = __import__("gzip").decompress(blob)
    assert raw[:5] == b"AQFS\x01" and int.from_bytes(raw[5:9], "little") == 6
    assert raw[9:11] == bytes([5, 122])                   # the first gap from 0, then 127 - 5
    with pytest.raises(ValueError):
        fs.encode([1, 1], [3, 3])
    with pytest.raises(ValueError):
        fs.decode(b"nope")


def test_varints_are_leb128():
    assert fs._varints([0, 1, 127, 128, 300, 16384]) == bytes([0, 1, 127, 0x80, 1, 0xAC, 2, 0x80, 0x80, 1])


def test_sentence_says_modelled_and_takes_its_numbers_from_the_counts():
    m = {"issue_date": "2026-10-11"}
    counts = {"much_below": 10, "below": 15, "normal": 50, "above": 15, "much_above": 10}
    s = fs.sentence(m, counts)
    assert "of 100 river reaches" in s and "25% run above normal" in s and "10% much above" in s
    assert "Modelled, not measured" in s
    assert fs.sentence({}) .startswith("No Flow against normal")


def test_in_bbox_handles_the_antimeridian():
    lat = np.array([0, 0, 0, 50])
    lon = np.array([175, -175, 0, 175])
    assert list(fs.in_bbox(lat, lon, [170, -10, -170, 10])) == [True, True, False, False]
    with pytest.raises(ValueError):
        fs.in_bbox(lat, lon, [1, 2, 3])


# ── the climatology over a fake flow duration store ─────────────────────────

N, WIDTH = 8, 4
P_EXCEED = list(range(0, 101, 5))                         # 21 probabilities, coarser than the real 101


def _zarr(shape, chunks, dtype, dims):
    return ({"shape": shape, "chunks": chunks, "dtype": dtype, "compressor": None, "fill_value": None,
             "filters": None, "order": "C", "zarr_format": 2}, {"_ARRAY_DIMENSIONS": dims})


def _fdc_store() -> dict[str, bytes]:
    """8 reaches in two chunks; reach i's flow exceeded p% of the time in month m is (100 - p) * (i + 1) * m."""
    rid = np.array([2008, 2001, 2002, 2003, 2004, 2005, 2006, 2007], dtype="<i8")   # not sorted, as in the store
    vals = np.zeros((12, len(P_EXCEED), N), dtype="<f4")
    for m in range(12):
        for j, pe in enumerate(P_EXCEED):
            vals[m, j, :] = (100 - pe) * (np.arange(N) + 1) * (m + 1)
    vals[:, :, 6] = 0                                     # reach 2006: a dry channel
    meta = {}
    for name, shape, chunks, dtype, dims in [
        ("daily_monthly", [12, len(P_EXCEED), N], [12, len(P_EXCEED), WIDTH], "<f4",
         ["month", "p_exceed", "river_id"]),
        ("p_exceed", [len(P_EXCEED)], [len(P_EXCEED)], "<i8", ["p_exceed"]),
        ("month", [12], [12], "<i4", ["month"]), ("river_id", [N], [N], "<i8", ["river_id"])]:
        meta[f"{name}/.zarray"], meta[f"{name}/.zattrs"] = _zarr(shape, chunks, dtype, dims)
    z = fs.FDC_ZARR
    out = {f"{z}/.zmetadata": json.dumps({"metadata": meta}).encode(),
           f"{z}/p_exceed/0": np.array(P_EXCEED, dtype="<i8").tobytes(),
           f"{z}/month/0": np.arange(1, 13, dtype="<i4").tobytes(), f"{z}/river_id/0": rid.tobytes()}
    for c in range(N // WIDTH):
        out[f"{z}/daily_monthly/0.0.{c}"] = np.ascontiguousarray(vals[:, :, c * WIDTH:(c + 1) * WIDTH]).tobytes()
    return out


def _fdc_tables(rid):
    order = np.full(len(rid), 6, dtype="int16")
    order[2] = 3                                          # reach 2002: too small to class
    return {"order": order, "area_km2": np.full(len(rid), 1e4), "lat": np.linspace(10, 17, len(rid)),
            "lon": np.linspace(100, 107, len(rid))}


def _climatology(tmp_path, **kw):
    store = _fdc_store()
    asked = []

    def fetch(url):
        asked.append(url)
        return store.get(url)

    info = fs.build_climatology(tmp_path, fetch=fetch, tables=_fdc_tables, workers=1, **kw)
    return info, asked


def test_the_climatology_keeps_four_percentiles_a_month_for_the_big_reaches(tmp_path):
    info, asked = _climatology(tmp_path)
    assert info["n"] == 7 and not info["smoke"] and info["chunks"]["read"] == 2
    root = tmp_path / fs.CLIM_FOLDER
    oct_ = pq.read_table(root / "month-10.parquet").to_pandas()
    assert list(oct_.columns) == ["river_id", "p10", "p25", "p75", "p90"]
    assert list(oct_["river_id"]) == [2001, 2003, 2004, 2005, 2006, 2007, 2008]      # sorted, 2002 left out
    row = oct_.set_index("river_id").loc[2001]                 # store index 1: (100 - p_exceed) * 2 * 10
    assert (row["p10"], row["p25"], row["p75"], row["p90"]) == (200.0, 500.0, 1500.0, 1800.0)
    reaches = pq.read_table(root / "reaches.parquet").to_pandas()
    assert set(reaches.columns) == {"river_id", "lat", "lon", "strahler_order"} and len(reaches) == 7
    m = json.loads((root / "manifest.json").read_text())
    assert m["percentiles"] == [10, 25, 75, 90] and m["base_period"] == "1950 to 2019"
    assert m["files"]["months"][0] == "flowstatus/climatology/month-01.parquet" and "CC BY 4.0" in m["licence"]
    assert sum("/daily_monthly/" in u for u in asked) == 2


def test_a_climatology_smoke_run_is_marked_and_never_published(tmp_path, monkeypatch):
    info, asked = _climatology(tmp_path, max_chunks=1)
    assert info["smoke"] and info["chunks"]["planned"] == 1 and info["n"] == 3
    called = []
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", lambda *a, **k: called.append(k))
    with pytest.raises(ValueError, match="smoke"):
        fs.publish_climatology(tmp_path)
    box, _ = _climatology(tmp_path / "box", bbox=[99, 9, 101.5, 11.5])    # reaches 0 and 1 only
    assert box["smoke"] and box["n"] == 2 and not called


def test_publish_climatology_uploads_only_its_folder(tmp_path, monkeypatch):
    _climatology(tmp_path)
    (tmp_path / "forecasts").mkdir()
    (tmp_path / "forecasts" / "x.parquet").write_bytes(b"x")
    seen = {}

    def fake(folder, repo_id, **kw):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen.update(kw)
        return "https://huggingface.co/commit/2"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake)
    assert fs.publish_climatology(tmp_path).endswith("/2")
    assert all(f.startswith("flowstatus/climatology/") for f in seen["files"]) and len(seen["files"]) == 14
    assert all(p.startswith("flowstatus/climatology/") for p in seen["allow_patterns"])


# ── the daily step ──────────────────────────────────────────────────────────


def _day0(tmp_path, flows: dict[int, float], issue="2026-10-11", smoke=False) -> Path:
    path = tmp_path / "scratch" / "day0.parquet"
    fw.write_day0(path, list(flows), list(flows.values()), {"run": "2026101100", "issue_date": issue,
                                                             "smoke": smoke})
    return path


def _daily(tmp_path, monkeypatch, flows, **kw):
    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {
        "history": [{"issue_date": "2026-10-10", "file": "x", "counts": {}}]})
    clim = tmp_path / "clim"
    _climatology(clim)
    day0 = _day0(tmp_path, flows, **kw)
    return fs.daily(tmp_path / "out", day0=day0, climatology=clim)


# October percentiles for reach 2001: p10 200, p25 500, p75 1500, p90 1800
FLOWS = {2001: 100.0, 2003: 3000.0, 2004: 4000.0, 2005: 9000.0, 2006: 5.0, 2008: 50000.0, 2002: 1.0, 9999: 4.0}


def test_daily_classes_every_climatology_reach_and_writes_the_browser_file(tmp_path, monkeypatch):
    info = _daily(tmp_path, monkeypatch, FLOWS)
    # 2003: p25 1000 .. p75 3000, 3000 is normal; 2004: p75 3750, p90 4500 -> above; 2005: p90 5400 -> much above;
    # 2008 (index 0): p90 900 -> much above; 2006 is dry; 2007 has no flow; 2002 and 9999 are not in the climatology
    assert info["issue_date"] == "2026-10-11" and info["reaches"] == 7 and info["n"] == 5
    assert info["counts"] == {"much_below": 1, "below": 0, "normal": 1, "above": 1, "much_above": 2}
    assert info["no_flow"] == 1 and info["dry"] == 1 and not info["smoke"]
    root = tmp_path / "out" / fs.FOLDER
    ids, cls = fs.decode((root / "latest.bin.gz").read_bytes())
    assert dict(zip(ids.tolist(), cls.tolist())) == {2001: 1, 2003: 3, 2004: 4, 2005: 5, 2008: 5}
    assert (root / "2026-10-11.bin.gz").read_bytes() == (root / "latest.bin.gz").read_bytes()
    table = pq.read_table(root / "latest.parquet").to_pandas()
    assert len(table) == 6 and set(table.columns) == {"river_id", "flow_cms", "class"}
    m = json.loads((root / "manifest.json").read_text())
    assert m["month"] == 10 and m["licence"]["this_file"] == "CC BY 4.0" and m["not"].startswith("Modelled")
    assert [h["issue_date"] for h in m["history"]] == ["2026-10-10", "2026-10-11"]
    assert "1950 to 2019" in m["method"] and m["sizes"]["latest.bin.gz"] == len((root / "latest.bin.gz").read_bytes())


def test_a_smoke_day0_makes_a_smoke_issue_that_is_never_published(tmp_path, monkeypatch):
    info = _daily(tmp_path, monkeypatch, FLOWS, smoke=True)
    assert info["smoke"]
    called = []
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", lambda *a, **k: called.append(k))
    with pytest.raises(ValueError, match="smoke"):
        fs.publish(tmp_path / "out")
    assert not called


def test_publish_uploads_the_day_and_not_the_climatology(tmp_path, monkeypatch):
    _daily(tmp_path, monkeypatch, FLOWS)
    _climatology(tmp_path / "out")                          # a climatology in the same folder stays behind
    seen = {}

    def fake(folder, repo_id, **kw):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen.update(kw)
        return "https://huggingface.co/commit/3"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake)
    fs.publish(tmp_path / "out")
    assert seen["files"] == ["flowstatus/2026-10-11.bin.gz", "flowstatus/latest.bin.gz", "flowstatus/latest.parquet",
                             "flowstatus/manifest.json"]


def test_daily_without_a_climatology_says_how_to_make_one(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.archive.forecasts._get", lambda url: None)
    with pytest.raises(FileNotFoundError, match="flow-climatology"):
        fs.daily(tmp_path, day0=_day0(tmp_path, FLOWS))


def test_flow_vs_normal_reads_counts_a_box_and_one_reach(tmp_path, monkeypatch):
    _daily(tmp_path, monkeypatch, FLOWS)
    for name in ("flowstatus/climatology",):
        src = tmp_path / "clim" / name
        dst = tmp_path / "out" / name
        dst.mkdir(parents=True)
        for p in src.iterdir():
            dst.joinpath(p.name).write_bytes(p.read_bytes())
    local = tmp_path / "out"
    res = fs.flow_vs_normal(local=local)
    assert res["available"] and res["n"] == 5 and res["counts"]["much_above"] == 2 and "Modelled" in res["sentence"]
    box = fs.flow_vs_normal([99, 9, 101.5, 11.5], local=local)   # store reaches 0 (2008) and 1 (2001)
    assert box["n"] == 2 and box["counts"]["much_below"] == 1 and "in this box" in box["sentence"]
    one = fs.flow_vs_normal(river_id=2001, local=local)["reach"]
    assert one["class"] == "much_below" and one["flow_cms"] == 100.0 and one["p25"] == 500.0
    dry = fs.flow_vs_normal(river_id=2006, local=local)["reach"]
    assert dry["classed"] is False and "not classed" in dry["why"]
    assert fs.flow_vs_normal(local=tmp_path / "nothing")["available"] is False
    with pytest.raises(ValueError):
        fs.flow_vs_normal([1, 2, 3], local=local)


# ── the daily step reading the forecast itself ─────────────────────────────


def _forecast_store() -> dict[str, bytes]:
    members, steps, n, width = 3, 16, 8, 4
    rivid = np.array([2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008], dtype="<i4")
    q = np.full((members + 1, steps, n), 3000.0, dtype="<f4")
    q[:members, :8, 0] = 100.0                           # 2001: much below on day 0
    q[:members, 8:, 0] = 99999.0                         # ... whatever day 1 does
    q[members, :, 0] = 1e6                               # the high-res run is left out
    meta = {}
    for name, shape, chunks, dtype, dims in [
        ("Qout", [members + 1, steps, n], [members + 1, steps, width], "<f4", ["ensemble", "time", "rivid"]),
        ("rivid", [n], [n], "<i4", ["rivid"]), ("time", [steps], [steps], "<i4", ["time"]),
        ("ensemble", [members + 1], [members + 1], "<i8", ["ensemble"])]:
        meta[f"{name}/.zarray"], meta[f"{name}/.zattrs"] = _zarr(shape, chunks, dtype, dims)
    meta["time/.zattrs"]["units"] = "seconds since 2026-10-11"
    fb = fw.FORECAST_BUCKET + "/2026101100.zarr"
    out = {f"{fb}/.zmetadata": json.dumps({"metadata": meta}).encode(), f"{fb}/rivid/0": rivid.tobytes(),
           f"{fb}/time/0": (np.arange(steps) * 3 * 3600).astype("<i4").tobytes(),
           f"{fb}/ensemble/0": np.array([1, 2, 3, fw.HIGH_RES_MEMBER], dtype="<i8").tobytes()}
    for c in range(n // width):
        out[f"{fb}/Qout/0.0.{c}"] = np.ascontiguousarray(q[:, :, c * width:(c + 1) * width]).tobytes()
    return out


def test_daily_reads_the_first_forecast_day_itself(tmp_path, monkeypatch):
    clim = tmp_path / "clim"
    _climatology(clim)
    store = _forecast_store()
    info = fs.daily(tmp_path / "out", climatology=clim, run_name="2026101100", fetch=store.get, workers=1,
                    max_chunks=2)
    assert info["smoke"] and info["issue_date"] == "2026-10-11"
    ids, cls = fs.decode((tmp_path / "out" / fs.FOLDER / "latest.bin.gz").read_bytes())
    got = dict(zip(ids.tolist(), cls.tolist()))
    assert got[2001] == 1 and 2006 not in got           # 100 m3/s against p10 200; the dry reach unclassed
    assert got[2003] == 3                                 # 3000 is reach 2003's 75th percentile: normal


def test_the_warnings_run_writes_the_first_day_for_flowstatus(tmp_path, monkeypatch):
    from tests.test_archive.test_warnings import _store, _tables

    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {})
    store = _store()
    day0 = tmp_path / "scratch" / "day0.parquet"
    fw.run(tmp_path, fetch=store.get, tables=_tables, gauges={}, today=date(2026, 10, 10), workers=1,
           day0_out=day0)
    t = pq.read_table(day0).to_pandas().set_index("river_id")["flow_cms"]
    assert len(t) == 7 and 1008 not in t.index           # order 3, not read
    assert t[1001] == 50.0 and t[1003] == 50.0             # day 0 of members only (1003's high-res is left out)
    info = json.loads(day0.with_suffix(".json").read_text())
    assert info["issue_date"] == "2026-10-09" and info["run"] == "2026100900" and not info["smoke"]


# ── the faces ───────────────────────────────────────────────────────────────

R = {"available": True, "issue_date": "2026-10-11", "month": 10, "n": 100,
     "counts": {"much_below": 10, "below": 15, "normal": 50, "above": 15, "much_above": 10},
     "sentence": "GEOGLOWS forecast of 2026-10-11: of 100 river reaches, 25% run above normal.",
     "not": "Modelled, not measured.",
     "reach": {"river_id": 7, "class": "above", "classed": True, "flow_cms": 12.5, "p10": 1.0, "p25": 2.0,
               "p75": 10.0, "p90": 20.0}}


def test_the_cli_prints_the_sentence_the_classes_and_a_reach(monkeypatch, capsys):
    seen = {}

    def fake(bbox, **kw):
        seen.update(bbox=bbox, **kw)
        return R

    monkeypatch.setattr(fs, "flow_vs_normal", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "flowstatus", "--bbox", "1", "2", "3", "4", "--river-id", "7"])
    cli.main()
    out = capsys.readouterr().out
    assert seen == {"bbox": [1.0, 2.0, 3.0, 4.0], "river_id": 7, "local": None}
    assert "25% run above normal" in out and "much below normal" in out and "50%" in out
    assert "Reach 7: above normal, 12.5 m3/s today" in out and "2 to 10 m3/s" in out and "Modelled" in out
    monkeypatch.setattr(fs, "flow_vs_normal", lambda bbox, **kw: {"available": False, "sentence": "Nothing yet."})
    monkeypatch.setattr(sys, "argv", ["aquascope", "flowstatus", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["available"] is False


def test_the_mcp_tool_wraps_the_engine(monkeypatch):
    pytest.importorskip("mcp")
    assert "bbox" in mcp_server.flow_vs_normal([1, 2, 3])["error"]
    monkeypatch.setattr(fs, "flow_vs_normal", lambda bbox, **kw: {**R, "seen": [bbox, kw]})
    assert mcp_server.flow_vs_normal([1, 2, 3, 4], river_id=7)["seen"] == [[1, 2, 3, 4], {"river_id": 7}]
    names = {t.name for t in asyncio.run(mcp_server.build_server().list_tools())}
    assert "flow_vs_normal" in names


# ── the workflows ───────────────────────────────────────────────────────────


def test_the_daily_step_rides_on_the_flood_warnings_run_and_never_publishes_a_smoke_run():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "flood-warnings.yml").read_text(encoding="utf-8"))
    steps = wf["jobs"]["warnings"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "--day0 build/scratch/day0.parquet" in runs
    daily_step = next(s for s in steps if "aquascope.flowstatus daily" in s.get("run", ""))
    assert daily_step.get("continue-on-error") is True
    pub = next(s for s in steps if "aquascope.flowstatus publish" in s.get("run", ""))
    assert "inputs.smoke == ''" in pub["if"] and pub.get("continue-on-error") is True


def test_the_climatology_workflow_is_run_by_hand_and_writes_only_its_folder():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "flow-climatology.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert set(on) == {"workflow_dispatch"}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["smoke"]["default"] == "" and inputs["publish"]["default"] is True
    job = next(iter(wf["jobs"].values()))
    assert job["timeout-minutes"] < 360
    pub = next(s for s in job["steps"] if "publish-climatology" in s.get("run", ""))
    assert "inputs.smoke == ''" in pub["if"] and "inputs.publish" in pub["if"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "warnings publish" not in runs and "harvest" not in runs
