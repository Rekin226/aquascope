"""aquascope.archive.skill: the monthly skill run, its Zarr readers on synthetic stores, no network (#518)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import skill

# ── a Zarr v2 store in memory, uncompressed, served through the _get seam ────


class FakeStore:
    def __init__(self, base: str):
        self.base = base
        self.meta: dict = {}
        self.files: dict[str, bytes] = {}
        self.reads: list[str] = []

    def array(self, name, data, chunks, attrs=None):
        data = np.asarray(data)
        self.meta[f"{name}/.zarray"] = {"chunks": list(chunks), "shape": list(data.shape), "dtype": data.dtype.str,
                                        "compressor": None, "fill_value": None, "filters": None, "order": "C",
                                        "zarr_format": 2}
        self.meta[f"{name}/.zattrs"] = dict(attrs or {})
        grid = [range((s + c - 1) // c) for s, c in zip(data.shape, chunks)]
        for idx in np.ndindex(*[len(g) for g in grid]):
            block = np.zeros(chunks, dtype=data.dtype)
            sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, data.shape))
            part = data[sl]
            block[tuple(slice(0, n) for n in part.shape)] = part
            self.files[f"{self.base}/{name}/{'.'.join(map(str, idx))}"] = block.tobytes()

    def get(self, url):
        self.reads.append(url)
        if url == f"{self.base}/.zmetadata":
            return json.dumps({"metadata": self.meta}).encode()
        if url not in self.files:
            raise FileNotFoundError(url)
        return self.files[url]


@pytest.fixture
def stores(monkeypatch):
    out: dict[str, FakeStore] = {}

    def get(url):
        for base, st in out.items():
            if url.startswith(base):
                return st.get(url)
        raise FileNotFoundError(url)

    monkeypatch.setattr(skill, "_get", get)
    return out


def test_usgs_site_numbers_come_out_of_catalog_ids():
    assert skill.usgs_site_number("USGS-01013500") == "01013500"
    assert skill.usgs_site_number("CA574-09527500") == "09527500"
    assert skill.usgs_site_number("abc") is None


def test_geoglows_daily_reads_two_chunks_per_reach(stores):
    st = stores[skill.GEOGLOWS_ZARR] = FakeStore(skill.GEOGLOWS_ZARR)
    ids = np.array([300, 100, 200, 400], dtype="<i4")           # not sorted, as in the real store
    n_t = 5
    q = np.arange(n_t * 4, dtype="<f4").reshape(n_t, 4)
    st.array("river_id", ids, [3])
    st.array("time", np.arange(n_t, dtype="<f8") * 86400.0, [3])
    st.array("Q", q, [3, 2])
    out = skill.geoglows_daily([200, 999])
    assert list(out) == [200]
    s = out[200]
    assert s.index[0] == pd.Timestamp("1940-01-01") and len(s) == n_t
    np.testing.assert_allclose(s.to_numpy(), q[:, 2])
    assert not any("Q/0.0" in u or "Q/1.0" in u for u in st.reads)     # only the chunk column of reach 200


def test_grrr_daily_finds_the_basin_by_its_hybas_id(stores):
    st = stores[skill.GRRR_ZARR] = FakeStore(skill.GRRR_ZARR)
    gid = np.array(["hybas_1000000001", "hybas_1000000002", "hybas_1000000003"], dtype="<U16")
    flow = np.array([[1, 2, 3, 4], [10, 20, 30, 40], [5, 5, 5, 5]], dtype="<f4")
    st.array("gauge_id", gid, [2])
    st.array("time", np.arange(4, dtype="<i8"), [4])
    st.array("streamflow", flow, [2, 3])
    dropped: dict = {}
    out = skill.grrr_daily([1000000002, 1000000003, 42], dropped=dropped)
    np.testing.assert_allclose(out[1000000002].to_numpy(), [10, 20, 30, 40])
    np.testing.assert_allclose(out[1000000003].to_numpy(), [5, 5, 5, 5])
    assert out[1000000002].index[0] == pd.Timestamp("1980-01-01") and 42 not in out
    capped = skill.grrr_daily([1000000002, 1000000003], max_chunks=1, dropped=dropped)
    assert list(capped) == [1000000002] and dropped["grrr_chunks_over_cap"] == 1


def _nwm_store(stores) -> FakeStore:
    st = stores[skill.NWM_ZARR] = FakeStore(skill.NWM_ZARR)
    fid = np.array([11, 22, 33, 44], dtype="<i8")
    st.array("feature_id", fid, [4])
    st.array("gage_id", np.array([b"", b"01013500", b"", b""], dtype="|S15"), [4])
    hours = 48
    raw = np.full((hours, 4), 1000, dtype="<i4")           # 10 m3/s at scale 0.01
    raw[:24, 1] = 2000                                     # reach 22: 20 m3/s on day one, 10 on day two
    raw[5, 1] = -999900                                    # a missing hour
    st.array("streamflow", raw, [24, 2], attrs={"scale_factor": 0.01, "missing_value": -999900})
    st.array("time", np.arange(hours, dtype="<i8") + 23, [24])   # hours since 1979-02-01T01: day one at 00:00
    return st


def test_nwm_feature_ids_use_the_models_own_gauge_link_then_nldi(stores, monkeypatch):
    _nwm_store(stores)
    calls = []
    real_get = skill._get

    def get(url):
        if "nldi" in url:
            calls.append(url)
            return json.dumps({"features": [{"properties": {"comid": 33}}]}).encode()
        return real_get(url)

    monkeypatch.setattr(skill, "_get", get)
    out = skill.nwm_feature_ids(["01013500", "07010000"])
    assert out == {"01013500": (22, "nwm_gage"), "07010000": (33, "nldi_comid")}
    assert len(calls) == 1 and "USGS-07010000" in calls[0]


def test_nwm_daily_averages_hours_into_days_and_caps_the_chunks(stores):
    _nwm_store(stores)
    dropped: dict = {}
    out = skill.nwm_daily([22, 44], years=1, max_columns=1, workers=2, dropped=dropped)
    assert list(out) == [22]                               # 22 and 44 sit in different column chunks; one kept
    s = out[22]
    assert s.index[0] == pd.Timestamp("1979-02-02") and s.iloc[0] == pytest.approx(20.0) and s.iloc[1] == 10.0
    assert dropped == {"nwm_columns_over_cap": 1, "nwm_gauges_over_cap": 1}


def test_nwm_daily_drops_the_padding_past_the_end_of_the_record(stores):
    st = stores[skill.NWM_ZARR] = FakeStore(skill.NWM_ZARR)
    st.array("feature_id", np.array([11, 22], dtype="<i8"), [2])
    hours = 40                                             # the second time chunk is padded to 24 rows
    st.array("streamflow", np.full((hours, 2), 500, dtype="<i4"), [24, 2], attrs={"scale_factor": 0.01})
    st.array("time", np.arange(hours, dtype="<i8") + 23, [24])
    s = skill.nwm_daily([22], years=1, workers=1)[22]
    assert s.index.min() == pd.Timestamp("1979-02-02")    # no zero-flow day at the epoch from the padding
    assert (s == 5.0).all()


def test_nwm_daily_reads_the_chunks_left_out_last_month_first(stores):
    _nwm_store(stores)
    dropped: dict = {}
    out = skill.nwm_daily([22, 44], years=1, max_columns=1, workers=1, dropped=dropped,
                          last_scored={22: "2026-09-03T05:41:00+00:00"})
    assert list(out) == [44]                               # 44 was never scored, so its chunk goes first


# ── the build, end to end on a tiny archive ─────────────────────────────────


def _archive(n_years: int = 12) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2005-01-01", periods=int(n_years * 365.25), freq="D")
    rng = np.random.default_rng(0)
    q = 30 + 20 * np.sin(2 * np.pi * idx.dayofyear / 365.25) + rng.gamma(2, 3, len(idx))
    catalog = [
        {"source": "usgs", "station_id": "USGS-01013500", "latitude": 47.0, "longitude": -68.0,
         "extra": {"catchment_area_km2": 1000.0}},
        {"source": "uk_ea", "station_id": "abc", "latitude": 51.4, "longitude": -0.3, "extra": {}},
        {"source": "uk_ea", "station_id": "short", "latitude": 51.0, "longitude": 0.0, "extra": {}},
    ]
    obs = pd.concat([
        pd.DataFrame({"source": "usgs", "station_id": "USGS-01013500", "date": idx, "value": q}),
        pd.DataFrame({"source": "uk_ea", "station_id": "abc", "date": idx, "value": q * 2}),
        pd.DataFrame({"source": "uk_ea", "station_id": "short", "date": idx[:100], "value": q[:100]}),
    ], ignore_index=True)
    catch = pd.DataFrame([
        {"source": "usgs", "station_id": "USGS-01013500", "hybas_id": 7000000001, "up_area": 1100.0,
         "area_km2": 1000.0},
        {"source": "uk_ea", "station_id": "abc", "hybas_id": 2000000001, "up_area": 9950.0, "area_km2": 9950.0},
    ])
    return catalog, obs, catch, pd.Series(q, index=idx)


def _patch_build(monkeypatch, tmp_path):
    from aquascope import evidence

    catalog, obs, catch, q = _archive()
    monkeypatch.setattr(skill, "_load_inputs", lambda archive: (catalog, obs, catch))
    monkeypatch.setattr(evidence, "geoglows_site", lambda lat, lon, area: {
        "site_id": "110000001", "distance_m": 50.0, "model_area_km2": 1010.0, "area_ratio": 1.01, "match": "area"})
    monkeypatch.setattr(skill, "geoglows_daily", lambda ids, **kw: {110000001: q * 1.4})
    monkeypatch.setattr(skill, "grrr_daily", lambda ids, max_chunks, dropped, **kw: {7000000001: q * 1.02,
                                                                               2000000001: q * 2.1})
    monkeypatch.setattr(skill, "nwm_feature_ids", lambda sites: {"01013500": (22, "nwm_gage")})
    monkeypatch.setattr(skill, "nwm_daily", lambda fids, **kw: {22: q * 0.9})
    return q


def test_build_scores_every_ci_model_and_writes_the_table_and_manifest(monkeypatch, tmp_path):
    _patch_build(monkeypatch, tmp_path)
    m = skill.build(tmp_path / "archive", tmp_path / "out")
    df = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")
    assert list(df.columns) == skill._COLUMNS
    usgs = df[df["source"] == "usgs"].set_index("model")
    assert set(usgs.index) == {"geoglows", "grrr", "nwm"}
    assert usgs.loc["grrr", "is_best"] and usgs.loc["grrr", "grade"] == "A"
    assert usgs.loc["grrr", "site_id"] == "hybas_7000000001" and usgs.loc["grrr", "area_ratio"] == pytest.approx(1.1)
    assert usgs.loc["nwm", "match"] == "nwm_gage" and usgs.loc["geoglows", "match"] == "area"
    assert df[df["source"] == "uk_ea"]["model"].tolist() == ["geoglows", "grrr"]   # no NWM outside the US
    assert m["n_gauges"] == 2 and m["dropped"]["short_record"] == 1 and m["smoke"] is False
    assert m["rows_by_model"] == {"geoglows": 2, "nwm": 1, "grrr": 2}
    assert "glofas" in m["not_scored_here"]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["n_rows"] == 5


def test_a_capped_run_keeps_last_months_rows_for_the_gauges_it_skipped(monkeypatch, tmp_path):
    _patch_build(monkeypatch, tmp_path)
    skill.build(tmp_path / "a", tmp_path / "out")
    first = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")
    m = skill.build(tmp_path / "a", tmp_path / "out", max_gauges=1)
    df = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")
    assert m["scored_this_run"] == 1 and m["dropped"]["over_cap"] == 1
    assert len(df) == len(first)                           # the skipped gauge's rows carried over
    stamps = df.groupby(["source", "station_id"])["computed_at"].first()
    assert stamps.nunique() in (1, 2)


def test_a_smoke_build_is_never_published(monkeypatch, tmp_path):
    _patch_build(monkeypatch, tmp_path)
    m = skill.build(tmp_path / "a", tmp_path / "out", smoke=True)
    assert m["smoke"] is True and m["scored_this_run"] <= skill.SMOKE_GAUGES
    with pytest.raises(RuntimeError, match="never published"):
        skill.publish(tmp_path / "out")


def test_publish_uploads_only_the_skill_folder(monkeypatch, tmp_path):
    _patch_build(monkeypatch, tmp_path)
    skill.build(tmp_path / "a", tmp_path / "out")
    seen = {}

    def fake_publish(folder, repo_id, token=None, commit_message=None):
        seen["files"] = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        seen["msg"] = commit_message
        return "https://hf/commit"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert skill.publish(tmp_path / "out", token="t") == "https://hf/commit"
    assert seen["files"] == ["skill/manifest.json", "skill/model_skill.parquet"]
    assert seen["msg"].startswith("model skill: 5 rows")


def test_a_model_left_out_this_run_keeps_last_months_row(monkeypatch, tmp_path):
    q = _patch_build(monkeypatch, tmp_path)
    skill.build(tmp_path / "a", tmp_path / "out")
    first = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")

    def nwm_down(fids, **kw):
        raise RuntimeError("store unreachable")

    monkeypatch.setattr(skill, "nwm_daily", nwm_down)
    monkeypatch.setattr(skill, "grrr_daily", lambda ids, **kw: {7000000001: q * 3.0, 2000000001: q * 2.1})
    m = skill.build(tmp_path / "a", tmp_path / "out")
    df = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")
    usgs = df[df["source"] == "usgs"].set_index("model")
    assert set(usgs.index) == {"geoglows", "grrr", "nwm"} and m["failures"]["nwm_read_failed"] == 1
    old_nwm = first[(first["source"] == "usgs") & (first["model"] == "nwm")].iloc[0]
    assert usgs.loc["nwm", "computed_at"] == old_nwm["computed_at"]          # carried, not rescored
    assert usgs.loc["nwm", "kge"] == pytest.approx(old_nwm["kge"])
    # GRRR got worse this month, so the best mark moves to the carried NWM row: one best per gauge
    assert usgs.loc["nwm", "is_best"] and not usgs.loc["grrr", "is_best"]
    assert df.groupby(["source", "station_id"])["is_best"].sum().max() == 1


def test_a_gauge_the_archive_no_longer_mirrors_is_let_go(monkeypatch, tmp_path):
    _patch_build(monkeypatch, tmp_path)
    skill.build(tmp_path / "a", tmp_path / "out")
    catalog, obs, catch, _q = _archive()
    obs = obs[obs["station_id"] != "abc"]
    monkeypatch.setattr(skill, "_load_inputs", lambda archive: (catalog, obs, catch))
    skill.build(tmp_path / "a", tmp_path / "out")
    df = pd.read_parquet(tmp_path / "out" / "model_skill.parquet")
    assert "abc" not in set(df["station_id"])
