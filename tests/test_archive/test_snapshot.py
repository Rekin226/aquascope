import json
import zipfile

import pytest

from aquascope.archive.harvest import write_stations_parquet
from aquascope.archive.snapshot import prepare
from aquascope.schemas.station import Station


def test_deposit_preparation_keeps_source_rights_and_immutable_identity(tmp_path):
    archive = tmp_path / "archive"
    (archive / "obs/discharge").mkdir(parents=True)
    write_stations_parquet([Station(source="usgs", station_id="A", latitude=1, longitude=2)],
                           archive / "stations.parquet")
    (archive / "health.json").write_text('{}', encoding="utf-8")
    (archive / "obs/discharge/usgs.parquet").write_bytes(b"test bundle")
    (archive / "obs/manifest.json").write_text(json.dumps({"bundles": {
        "usgs/discharge": {"source": "usgs", "file": "obs/discharge/usgs.parquet"},
    }}), encoding="utf-8")
    metadata = prepare(archive, tmp_path / "deposit", revision="a" * 40, snapshot_date="2026-09-23")
    assert metadata["doi"] is None and metadata["concept_doi"] is None
    assert metadata["agency_credits_and_rights"][0]["agency"]
    assert metadata["agency_credits_and_rights"][0]["license"]
    assert all(len(f["sha256"]) == 64 for f in metadata["files"])
    with zipfile.ZipFile(tmp_path / "deposit/aquascope-gauges-2026-09-23.zip") as z:
        assert z.testzip() is None
        assert "obs/manifest.json" in z.namelist() and "snapshot-metadata.json" in z.namelist()
    with pytest.raises(ValueError, match="immutable"):
        prepare(archive, tmp_path / "bad", revision="main", snapshot_date="2026-09-23")
