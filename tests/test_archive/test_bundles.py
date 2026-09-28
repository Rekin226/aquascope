"""Archive Phase 2: Parquet bundles rolled up from the per-station files."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from aquascope.archive import bundles  # noqa: E402
from aquascope.archive import observations as obs  # noqa: E402


def _write_station(root, variable, source, sid, n, start="2000-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    s = pd.Series(np.arange(n, dtype=float), index=idx)
    p = obs.obs_path(root, variable, source, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(obs.series_to_csv_gz(s))
    return s


def test_failed_bundle_preserves_previous_bytes_and_does_not_block_other_sources(tmp_path):
    _write_station(tmp_path, "discharge", "usgs", "A", 20)
    _write_station(tmp_path, "precipitation", "uk_ea", "B", 20)
    bundles.build_bundles(tmp_path)
    previous = (tmp_path / "obs/discharge/usgs.parquet").read_bytes()
    build = bundles.build_bundle

    def fail_one(root, variable, source):
        if source == "usgs":
            raise RuntimeError("simulated per-source serialization failure")
        return build(root, variable, source)

    with patch.object(bundles, "build_bundle", side_effect=fail_one):
        infos = bundles.build_bundles(tmp_path)
    assert [i.source for i in infos] == ["uk_ea"]
    assert (tmp_path / "obs/discharge/usgs.parquet").read_bytes() == previous
    manifest = obs.load_manifest(tmp_path)
    assert manifest["bundle_status"]["usgs/discharge"]["retained_previous"] is True
    assert manifest["bundle_status"]["usgs/discharge"]["status"] == "failed"
    assert manifest["bundle_status"]["uk_ea/precipitation"]["status"] == "ok"


def test_build_bundles_writes_one_parquet_per_pair_and_records_manifest(tmp_path):
    a = _write_station(tmp_path, "discharge", "usgs", "USGS-1", 400)
    b = _write_station(tmp_path, "discharge", "usgs", "USGS-2", 10, start="2010-06-01")
    _write_station(tmp_path, "precipitation", "uk_ea", "R1", 30)
    (tmp_path / "obs" / "discharge" / "hubeau_hydrometrie").mkdir(parents=True)  # empty folder: no bundle

    infos = bundles.build_bundles(tmp_path)
    assert [(i.variable, i.source, i.n_stations, i.n_rows) for i in infos] == [
        ("discharge", "usgs", 2, 410), ("precipitation", "uk_ea", 1, 30),
    ]
    usgs = infos[0]
    assert usgs.file == "obs/discharge/usgs.parquet" and usgs.unit == "m3/s"
    assert usgs.first == "2000-01-01" and usgs.last == "2010-06-10"

    df = bundles.read_bundle(tmp_path / usgs.file)
    assert list(df.columns) == ["station_id", "date", "value"]
    assert df["station_id"].tolist()[:2] == ["USGS-1", "USGS-1"] and str(df["date"].dtype).startswith("datetime64")
    back = bundles.to_series(df, "USGS-1")
    assert len(back) == 400 and back.iloc[-1] == a.iloc[-1] and back.index[0] == a.index[0]
    assert bundles.read_bundle(tmp_path / usgs.file, station_ids=["USGS-2"]).shape[0] == len(b)

    import pyarrow.parquet as pq

    meta = json.loads(pq.read_schema(tmp_path / usgs.file).metadata[b"aquascope"])
    assert meta["variable"] == "discharge" and meta["license"] == "US-PD"

    manifest = obs.load_manifest(tmp_path)
    assert set(manifest["bundles"]) == {"usgs/discharge", "uk_ea/precipitation"}
    assert manifest["bundles"]["usgs/discharge"]["n_rows"] == 410

    # rebuilding after another station lands replaces the bundle (the folder is the source of truth)
    _write_station(tmp_path, "discharge", "usgs", "USGS-3", 5)
    infos = bundles.build_bundles(tmp_path, variables=["discharge"], sources=["usgs"])
    assert infos[0].n_stations == 3 and obs.load_manifest(tmp_path)["bundles"]["usgs/discharge"]["n_stations"] == 3


def test_build_bundles_on_empty_tree(tmp_path):
    assert bundles.build_bundles(tmp_path) == []
    assert bundles.build_bundle(tmp_path, "discharge", "usgs") is None


def test_corrupt_station_does_not_silently_shrink_existing_bundle(tmp_path):
    _write_station(tmp_path, "discharge", "usgs", "A", 20)
    _write_station(tmp_path, "discharge", "usgs", "B", 30)
    bundles.build_bundles(tmp_path)
    path = tmp_path / "obs/discharge/usgs.parquet"
    previous = path.read_bytes()
    obs.obs_path(tmp_path, "discharge", "usgs", "B").write_bytes(b"broken gzip")
    assert bundles.build_bundles(tmp_path) == []
    assert path.read_bytes() == previous
    assert obs.load_manifest(tmp_path)["bundle_status"]["usgs/discharge"]["status"] == "failed"


def test_load_observations_downloads_and_handles_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AQUASCOPE_CACHE_DIR", str(tmp_path / "cache"))
    _write_station(tmp_path, "discharge", "usgs", "USGS-1", 12)
    bundles.build_bundles(tmp_path)
    payload = (tmp_path / "obs" / "discharge" / "usgs.parquet").read_bytes()

    def fake_download(url, dest, refresh):
        if "hubeau" in url:
            import httpx

            req = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("nf", request=req, response=httpx.Response(404, request=req))
        dest.write_bytes(payload)
        return dest

    with patch("aquascope.archive.catalog._download", side_effect=fake_download):
        df = bundles.load_observations("usgs")  # variable defaults to discharge
        assert df.shape[0] == 12 and df["station_id"].iloc[0] == "USGS-1"
        empty = bundles.load_observations("hubeau_hydrometrie", "discharge")
        assert empty.empty and list(empty.columns) == ["station_id", "date", "value"]
    assert bundles.bundle_url("uk_ea", "groundwater_level").endswith("/obs/groundwater_level/uk_ea.parquet")
