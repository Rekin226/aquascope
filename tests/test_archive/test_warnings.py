"""Floods ahead (#546): the classing, the daily reduction, a whole run over a small fake forecast, what is published."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="the archive extra")

from aquascope.archive import warnings as fw  # noqa: E402

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "flood-warnings.yml"
Q = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]   # 2- to 100-year flows of every fake reach


# ── the pure part ───────────────────────────────────────────────────────────


def test_classify_takes_the_largest_return_period_reached():
    assert fw.classify(99, Q) == 0
    assert fw.classify(100, Q) == 2
    assert fw.classify(350, Q) == 10
    assert fw.classify(10_000, Q) == 100
    assert fw.classify(250, {2: 100, 5: 200, 10: 300}) == 5


def test_classify_refuses_a_missing_or_empty_2_year_flow():
    assert fw.classify(500, [float("nan")] + Q[1:]) == 0
    assert fw.classify(500, [0.0] + Q[1:]) == 0
    assert fw.classify(None, Q) == 0
    assert fw.classify(float("nan"), Q) == 0


def test_classify_array_matches_classify():
    peaks = np.array([50, 100, 250, 450, 600, np.nan, 700])
    q = np.tile(np.array(Q)[:, None], (1, len(peaks)))
    q[0, 6] = np.nan
    got = fw.classify_array(peaks, q)
    assert got.tolist() == [0, 2, 5, 25, 100, 0, 0]
    assert got.tolist() == [fw.classify(p, q[:, i]) for i, p in enumerate(peaks)]


def test_daily_means_average_the_three_hourly_members_per_utc_day():
    seconds = np.arange(0, 2 * 86400, 3600)                # hourly steps, two days
    values = np.full((2, len(seconds), 1), np.nan, dtype="float32")
    values[:, ::3, 0] = 10.0                              # members are 3-hourly
    values[0, 24::3, 0] = 30.0                            # member 0 is wetter on day 1
    got = fw.daily_means(values, seconds, days=3)
    assert got.shape == (2, 3, 1)
    assert got[:, 0, 0].tolist() == [10.0, 10.0]
    assert got[:, 1, 0].tolist() == [30.0, 10.0]
    assert np.isnan(got[:, 2, 0]).all()                  # no step on the third day


def test_summarise_gives_peak_day_class_share_and_daily_classes():
    seconds = np.arange(0, 3 * 86400, 3 * 3600)
    n_members = 4
    values = np.full((n_members, len(seconds), 2), 50.0, dtype="float32")
    values[:, 8:16, 0] = 250.0                           # reach 0: every member at 250 on day 1
    values[:2, 16:, 1] = 220.0                           # reach 1: two members at 220 on day 2, two at 50
    q = np.tile(np.array(Q)[:, None], (1, 2))
    s = fw.summarise(values, seconds, q, days=3)
    assert s["rp"].tolist() == [5, 2]
    assert s["peak_day"].tolist() == [1, 2]
    assert s["peak"][0] == pytest.approx(250) and s["peak"][1] == pytest.approx(135)
    assert s["share"].tolist() == [1.0, 0.5]
    assert s["daily"][:, 0].tolist() == [0, 5, 0]
    assert fw.daily_string(s["daily"][:, 0]) == "020"


def test_evenly_spreads_a_smoke_run_over_the_list():
    assert fw.evenly(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert fw.evenly([1, 2], 5) == [1, 2] and fw.evenly([1, 2], None) == [1, 2] and fw.evenly([1], 0) == []


def test_sentence_says_model_output_and_the_biggest_class():
    m = {"issue_date": "2026-10-09", "counts": {"2": 3, "5": 1, "10": 2, "25": 0, "50": 0, "100": 0}}
    line = fw.sentence(m)
    assert "6 river reaches" in line and "2 of them the 10-year flow or more" in line
    assert "largest class is the 10-year flow" in line and "not an official warning" in line
    assert "expects no checked river reach" in fw.sentence({**m, "counts": {"2": 0}})
    assert fw.sentence({}) == "No Floods ahead issue is published yet."
    for text in (line, fw.METHOD, fw.NOT, fw.ABOUT):
        assert "\u2014" not in text


def test_forecast_layout_refuses_a_changed_store():
    meta = _forecast_meta(n=8, chunk=4)
    assert fw.forecast_layout(meta)["start"] == "2026-10-09"
    bad = json.loads(json.dumps(meta))
    bad["Qout/.zarray"]["chunks"] = [1, 40, 4]
    with pytest.raises(RuntimeError, match="split members"):
        fw.forecast_layout(bad)
    bad = json.loads(json.dumps(meta))
    bad["Qout/.zattrs"]["_ARRAY_DIMENSIONS"] = ["time", "ensemble", "rivid"]
    with pytest.raises(RuntimeError, match="layout"):
        fw.forecast_layout(bad)


def test_decode_chunk_reads_an_uncompressed_chunk_and_refuses_other_codecs():
    arr = {"chunks": [2, 3], "dtype": "<f4", "compressor": None, "order": "C"}
    got = fw.decode_chunk(np.arange(6, dtype="<f4").tobytes(), arr)
    assert got.shape == (2, 3) and got[1, 2] == 5
    with pytest.raises(ValueError, match="compressor"):
        fw.decode_chunk(b"", {**arr, "compressor": {"id": "gzip"}})


def test_to_geojson_ranks_by_class_and_leaves_out_reaches_without_a_position():
    base = {"peak_cms": 10.0, "q2": 5.0, "peak_date": "2026-10-10", "share": 0.5, "strahler_order": 6,
            "daily": "0" * 15, "gauges": []}
    rows = [{**base, "river_id": 1, "lat": 10.0, "lon": 20.0, "rp": 2},
            {**base, "river_id": 2, "lat": float("nan"), "lon": 20.0, "rp": 100},
            {**base, "river_id": 3, "lat": -5.0, "lon": 30.0, "rp": 25}]
    fc, truncated = fw.to_geojson(rows, cap=5)
    assert [f["properties"]["river_id"] for f in fc["features"]] == [3, 1]
    assert not truncated
    json.dumps(fc, allow_nan=False)                     # what the browser reads must be strict JSON
    _, truncated = fw.to_geojson(rows, cap=1)
    assert truncated


# ── a whole run over a fake forecast ────────────────────────────────────────

MEMBERS, STEPS, DAY_STEPS = 3, 40, 8                    # 3 members + the high-res run, 5 days of 3-hourly steps


def _arr(shape, chunks, dtype, dims):
    return ({"shape": shape, "chunks": chunks, "dtype": dtype, "compressor": None, "fill_value": None,
             "filters": None, "order": "C", "zarr_format": 2}, {"_ARRAY_DIMENSIONS": dims})


def _forecast_meta(n: int, chunk: int) -> dict:
    meta = {}
    for name, shape, chunks, dtype, dims in [
        ("Qout", [MEMBERS + 1, STEPS, n], [MEMBERS + 1, STEPS, chunk], "<f4", ["ensemble", "time", "rivid"]),
        ("rivid", [n], [n], "<i4", ["rivid"]), ("time", [STEPS], [STEPS], "<i4", ["time"]),
        ("ensemble", [MEMBERS + 1], [MEMBERS + 1], "<i8", ["ensemble"])]:
        meta[f"{name}/.zarray"], meta[f"{name}/.zattrs"] = _arr(shape, chunks, dtype, dims)
    meta["time/.zattrs"]["units"] = "seconds since 2026-10-09"
    return meta


def _store(n: int = 8, chunk: int = 4) -> dict[str, bytes]:
    """A forecast of 8 reaches in two chunks and their return periods, as the URLs the run asks for."""
    rivid = np.arange(1001, 1001 + n, dtype="<i4")
    seconds = (np.arange(STEPS) * 3 * 3600).astype("<i4")
    q = np.full((MEMBERS + 1, STEPS, n), 50.0, dtype="<f4")
    q[:MEMBERS, DAY_STEPS:2 * DAY_STEPS, 0] = 250.0       # reach 1001: 5-year on day 1
    q[:MEMBERS, 3 * DAY_STEPS:, 5] = 650.0                # reach 1006: 100-year from day 3
    q[MEMBERS, :, 2] = 5000.0                             # reach 1003: only the high-res run is high (left out)
    q[:MEMBERS, :, 7] = 900.0                             # reach 1008: high, but order 3 (not checked)
    fb = fw.FORECAST_BUCKET + "/2026100900.zarr"
    out = {
        f"{fb}/.zmetadata": json.dumps({"metadata": _forecast_meta(n, chunk)}).encode(),
        f"{fb}/rivid/0": rivid.tobytes(), f"{fb}/time/0": seconds.tobytes(),
        f"{fb}/ensemble/0": np.array([*range(1, MEMBERS + 1), fw.HIGH_RES_MEMBER], dtype="<i8").tobytes(),
    }
    for c in range(n // chunk):
        out[f"{fb}/Qout/0.0.{c}"] = np.ascontiguousarray(q[:, :, c * chunk:(c + 1) * chunk]).tobytes()
    rp = fw.RETURN_PERIODS_ZARR
    rmeta = {}
    rmeta["return_period/.zarray"], rmeta["return_period/.zattrs"] = _arr([6], [6], "<i8", ["return_period"])
    rmeta["river_id/.zarray"], rmeta["river_id/.zattrs"] = _arr([n], [n], "<i4", ["river_id"])
    rmeta["gumbel_daily/.zarray"], rmeta["gumbel_daily/.zattrs"] = _arr([6, n], [6, 4], "<f8",
                                                                          ["return_period", "river_id"])
    thr = np.tile(np.array(Q)[:, None], (1, n))
    thr[:, 4] = np.nan                                    # reach 1005: no threshold
    out.update({f"{rp}/.zmetadata": json.dumps({"metadata": rmeta}).encode(),
                f"{rp}/return_period/0": np.array(fw.RETURN_PERIODS, dtype="<i8").tobytes(),
                f"{rp}/river_id/0": rivid.tobytes()})
    for c in range(n // 4):
        out[f"{rp}/gumbel_daily/0.{c}"] = np.ascontiguousarray(thr[:, c * 4:(c + 1) * 4]).tobytes()
    out[f"{fw.FORECAST_BUCKET}/?list-type=2&delimiter=/&start-after=20260831"] = (
        b"<ListBucketResult><CommonPrefixes><Prefix>2026100800.zarr/</Prefix></CommonPrefixes>"
        b"<CommonPrefixes><Prefix>2026100900.zarr/</Prefix></CommonPrefixes>"
        b"<CommonPrefixes><Prefix>2026101000.zarr/</Prefix></CommonPrefixes></ListBucketResult>")
    return out


def _tables(rivid):
    n = len(rivid)
    order = np.full(n, 6, dtype="int16")
    order[7] = 3
    return {"order": order, "area_km2": np.full(n, 12345.0), "lat": np.linspace(10, 17, n),
            "lon": np.linspace(100, 107, n)}


def _run(tmp_path, gauges=None, **kw):
    store = _store()
    asked = []

    def fetch(url):
        asked.append(url)
        return store.get(url)

    info = fw.run(tmp_path, fetch=fetch, tables=_tables, gauges=gauges or {1006: ["usgs/X"]},
                  today=date(2026, 10, 10), workers=1, **kw)
    return info, asked


def test_a_run_writes_the_reaches_expected_to_flood(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {
        "history": [{"issue_date": "2026-10-08", "file": "x", "n": 1, "counts": {}}]})
    info, asked = _run(tmp_path)
    # the newest run without metadata (10 Oct) is passed over for the one with it
    assert info["run"] == "2026100900" and info["issue_date"] == "2026-10-09"
    assert info["n"] == 2 and info["counts"]["5"] == 1 and info["counts"]["100"] == 1
    assert info["checked"] == 6 and info["no_threshold"] == 1 and not info["smoke"]
    root = tmp_path / "forecasts" / "warnings"
    rows = {r["river_id"]: r for r in pq.read_table(root / "latest.parquet").to_pylist()}
    assert set(rows) == {1001, 1006}                      # not 1003 (high-res only) nor 1008 (order 3)
    a, b = rows[1001], rows[1006]
    assert a["rp"] == 5 and a["peak_date"] == "2026-10-10" and a["share"] == 1.0
    assert a["daily"] == "020000000000000"
    assert b["rp"] == 100 and b["peak_date"] == "2026-10-12" and b["daily"] == "000660000000000"
    assert b["gauges"] == ["usgs/X"]
    assert b["q2"] == 100 and b["q100"] == 600 and b["strahler_order"] == 6
    assert (root / "2026-10-09.parquet").exists()
    fc = json.loads((root / "latest.geojson").read_text())
    assert [f["properties"]["river_id"] for f in fc["features"]] == [1006, 1001]   # highest class first
    assert fc["features"][0]["properties"]["gauges"] == "usgs/X"
    # every return-period flow rides along, for the card's threshold lines (#556)
    assert fc["features"][0]["properties"]["q100"] == 600 and "q5" in fc["features"][1]["properties"]
    m = json.loads((root / "manifest.json").read_text())
    assert m["valid_to"] == "2026-10-23" and m["min_strahler_order"] == fw.DEFAULT_MIN_ORDER
    assert m["not"].startswith("Model output, not an official warning")
    assert m["thresholds"]["variable"] == "gumbel_daily" and "CC BY-NC-SA" in m["licence"]["this_file"]
    assert [h["issue_date"] for h in m["history"]] == ["2026-10-08", "2026-10-09"]
    assert sum("/Qout/" in u for u in asked) == 2           # both chunks hold an order-5+ reach


def test_a_reach_with_a_gauge_is_checked_whatever_its_order(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {})
    info, _ = _run(tmp_path, gauges={1008: ["uk_ea/Y"]})
    rows = {r["river_id"]: r for r in pq.read_table(tmp_path / fw.FOLDER / "latest.parquet").to_pylist()}
    assert set(rows) == {1001, 1006, 1008} and rows[1008]["gauges"] == ["uk_ea/Y"]
    assert rows[1008]["strahler_order"] == 3 and rows[1008]["rp"] == 100
    assert json.loads((tmp_path / fw.FOLDER / "manifest.json").read_text())["gauge_reaches"] == 1


def test_a_smoke_run_is_marked_and_never_published(tmp_path, monkeypatch):
    info, asked = _run(tmp_path, max_chunks=1)
    assert info["smoke"] and info["chunks"]["planned"] == 1 and info["chunks"]["needed"] == 2
    assert sum("/Qout/" in u for u in asked) == 1
    called = []
    monkeypatch.setattr("aquascope.archive.publish.publish_folder", lambda *a, **k: called.append(k))
    with pytest.raises(ValueError, match="smoke"):
        fw.publish(tmp_path)
    assert not called


def test_publish_uploads_only_the_warnings_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {})
    _run(tmp_path)
    (tmp_path / "forecasts" / "status").mkdir(parents=True)
    (tmp_path / "forecasts" / "status" / "latest.parquet").write_bytes(b"x")
    seen = {}

    def fake(folder, repo_id, **kw):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
        seen.update(kw)
        return "https://huggingface.co/commit/1"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake)
    assert fw.publish(tmp_path).endswith("/1")
    assert seen["allow_patterns"] == ["forecasts/warnings/*"]
    assert all(f.startswith("forecasts/warnings/") for f in seen["files"]) and len(seen["files"]) == 4


def test_flood_warnings_reads_an_issue_and_filters_by_box(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.archive.forecasts.read_published_json", lambda path, repo_id=None: {})
    _run(tmp_path)
    res = fw.flood_warnings(local=tmp_path)
    assert res["available"] and res["n"] == 2 and res["reaches"][0]["river_id"] == 1006
    assert "not an official warning" in res["sentence"]
    box = fw.flood_warnings([99, 9, 101, 11], local=tmp_path)    # only reach 1001 at (10, 100)
    assert [r["river_id"] for r in box["reaches"]] == [1001] and "in this box" in box["sentence"]
    assert fw.flood_warnings(min_rp=25, local=tmp_path)["n"] == 1
    assert fw.flood_warnings([170, -10, -170, 10], local=tmp_path)["n"] == 0     # across the antimeridian
    with pytest.raises(ValueError):
        fw.flood_warnings([1, 2, 3], local=tmp_path)
    assert fw.flood_warnings(local=tmp_path / "nothing")["available"] is False


def test_the_workflow_never_publishes_a_smoke_run_and_writes_only_warnings():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert on["schedule"][0]["cron"].endswith("* * *")
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["publish"]["default"] is True and inputs["smoke"]["default"] == ""
    job = wf["jobs"]["warnings"]
    assert job["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}" and job["timeout-minutes"] < 360
    steps = job["steps"]
    publish = next(s for s in steps if "warnings publish" in s.get("run", ""))
    assert "inputs.smoke == ''" in publish["if"] and "github.event_name == 'schedule'" in publish["if"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "harvest" not in runs and "forecasts publish" not in runs and "numcodecs" in runs
