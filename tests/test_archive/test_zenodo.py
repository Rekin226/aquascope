"""The Archive's Zenodo deposit (#444): the bundle from a tiny Archive tree, and both deposit paths with mocked HTTP."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import httpx
import pytest

pytest.importorskip("pyarrow")

from aquascope.archive import zenodo  # noqa: E402
from aquascope.archive.bundles import build_bundles  # noqa: E402
from aquascope.archive.harvest import write_stations_parquet  # noqa: E402
from aquascope.schemas.station import Station  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40


def _csv(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(rows.encode()))


def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    write_stations_parquet([
        Station(source="usgs", station_id="A", latitude=1, longitude=2),
        Station(source="usgs", station_id="B", latitude=1, longitude=3),
        Station(source="uk_ea", station_id="C", latitude=51, longitude=0),
        Station(source="brazil_ana", station_id="D", latitude=-10, longitude=-50),
    ], archive / "stations.parquet")
    (archive / "stations.geojson").write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    (archive / "health.json").write_text("{}", encoding="utf-8")
    (archive / "README.md").write_text("# card\n", encoding="utf-8")
    _csv(archive / "obs/discharge/usgs/A.csv.gz", "date,value\n2020-01-01,1.5\n")
    _csv(archive / "obs/discharge/uk_ea/C.csv.gz", "date,value\n2020-01-01,2.0\n")
    _csv(archive / "obs/water_level/usgs/A.csv.gz", "date,value\n2020-01-01,0.4\n")
    (archive / "obs/manifest.json").write_text("{}", encoding="utf-8")
    build_bundles(archive)
    for stray in (archive / "obs").glob("*"):
        if stray.is_file() and stray.suffix != ".json":
            stray.unlink()
    return archive


def test_the_bundle_packs_obs_per_source_and_variable_and_says_what_is_mirrored(tmp_path):
    archive = _archive(tmp_path)
    out = tmp_path / "deposit"
    manifest = zenodo.build(archive, out, revision=SHA, snapshot_date="2026-10-25")

    names = {f["name"] for f in manifest["files"]}
    assert {"stations.parquet", "stations.geojson", "health.json", "README.md", "obs_manifest.json",
            "SOURCES.md"} <= names
    tars = sorted(n for n in names if n.endswith(".tar"))
    assert tars == ["obs_discharge_uk_ea.tar", "obs_discharge_usgs.tar", "obs_water_level_usgs.tar"]
    with tarfile.open(out / "obs_discharge_usgs.tar") as tar:
        members = tar.getnames()
    assert "obs/discharge/usgs/A.csv.gz" in members and "obs/discharge/usgs.parquet" in members

    for f in manifest["files"]:
        assert f["sha256"] == hashlib.sha256((out / f["name"]).read_bytes()).hexdigest()
        assert f["bytes"] == (out / f["name"]).stat().st_size
    on_disk = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk["snapshot_date"] == "2026-10-25" and on_disk["revision"] == SHA

    assert manifest["catalogued"] == {"stations": 3, "sources": 2}
    assert manifest["mirrored"]["stations"] == 2 and manifest["mirrored"]["sources"] == 2
    assert manifest["mirrored"]["per_variable"] == {"discharge": 2, "water_level": 1}
    scopes = {c["source"]: c["scope"] for c in manifest["sources"]}
    assert set(scopes) == {"usgs", "uk_ea"} and scopes["usgs"].startswith("catalogue and mirrored")

    sources_md = (out / "SOURCES.md").read_text(encoding="utf-8")
    for key in ("usgs", "uk_ea"):
        assert key in sources_md
    assert "Environment Agency" in sources_md and "OGL-UK-3.0" in sources_md


def test_the_metadata_is_a_dataset_with_honest_counts_and_links(tmp_path):
    meta = zenodo.build(_archive(tmp_path), tmp_path / "d", revision=SHA, snapshot_date="2026-10-25")[
        "zenodo_metadata"]
    assert meta["upload_type"] == "dataset" and meta["access_right"] == "open"
    assert meta["version"] == "2026-10-25" and meta["license"] == zenodo.RECORD_LICENSE
    assert "Catalogued and mirrored are different numbers" in meta["description"]
    assert "3 gauging stations" in meta["description"] and "mirrored for 2 of them" in meta["description"]
    for derived in ("basins/", "forecasts/", "context/"):
        assert derived in meta["description"]
    rel = {r["relation"]: r["identifier"] for r in meta["related_identifiers"]}
    assert rel["isSupplementTo"] == "10.5281/zenodo.21903143"
    assert rel["isDerivedFrom"].endswith(f"/tree/{SHA}")
    assert {c["type"] for c in meta["contributors"]} == {"DataCollector"}
    written = json.loads((tmp_path / "d/zenodo-metadata.json").read_text(encoding="utf-8"))
    assert written == {"metadata": meta}


def test_rows_of_sources_without_a_verified_licence_are_left_out_and_said(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    archive = _archive(tmp_path)
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, 1]},
                 "properties": {"source": src, "station_id": sid}}
                for src, sid, lon in (("usgs", "A", 2), ("brazil_ana", "D", -50), ("not_a_source", "X", 9))]
    (archive / "stations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                                              encoding="utf-8")
    pq.write_table(pa.table({"source": ["usgs", "brazil_ana", "brazil_ana"], "station_id": ["A", "D", "E"],
                             "q_mean": [1.0, 2.0, 3.0]}), archive / "signatures.parquet")
    out = tmp_path / "d"
    manifest = zenodo.build(archive, out, revision=SHA, snapshot_date="2026-10-25")

    stations = pq.read_table(out / "stations.parquet")
    assert "brazil_ana" not in stations.column("source").to_pylist() and stations.num_rows == 3
    geo = json.loads(stations.schema.metadata[b"geo"])
    assert geo["columns"]["geometry"]["bbox"] == [0.0, 1.0, 3.0, 51.0], "bbox of the rows that remain"
    assert pq.read_table(out / "signatures.parquet").column("source").to_pylist() == ["usgs"]
    kept = json.loads((out / "stations.geojson").read_text(encoding="utf-8"))["features"]
    assert [f["properties"]["source"] for f in kept] == ["usgs"]

    left = {x["source"]: x for x in manifest["left_out"]}
    assert set(left) == {"brazil_ana", "not_a_source"}
    assert left["brazil_ana"]["rows_left_out"] == {"stations.parquet": 1, "stations.geojson": 1,
                                                   "signatures.parquet": 2}
    assert left["brazil_ana"]["stations_left_out"] == 1
    assert manifest["catalogued"] == {"stations": 3, "sources": 2}
    assert "brazil_ana" not in {c["source"] for c in manifest["sources"]}
    text = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert "Left out until their reuse terms are verified" in text and "brazil_ana" in text
    assert "left out until their reuse terms are verified" in manifest["zenodo_metadata"]["description"].lower()
    assert "brazil_ana" in manifest["zenodo_metadata"]["description"]
    assert not any(c["name"].startswith("Agência") for c in manifest["zenodo_metadata"]["contributors"])


def test_creators_match_citation_cff():
    yaml = pytest.importorskip("yaml")
    author = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))["authors"][0]
    creator = zenodo.CREATORS[0]
    assert creator["name"] == f"{author['family-names']}, {author['given-names']}"
    assert creator["orcid"] == author["orcid"].rsplit("/", 1)[-1]
    assert creator["affiliation"] == author["affiliation"]


def test_a_source_that_may_not_be_redistributed_is_refused(tmp_path):
    archive = _archive(tmp_path)
    _csv(archive / "obs/discharge/brazil_ana/D.csv.gz", "date,value\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="not approved for redistribution"):
        zenodo.build(archive, tmp_path / "d", revision=SHA)


def test_build_refuses_a_mutable_revision_and_a_non_empty_folder(tmp_path):
    archive = _archive(tmp_path)
    with pytest.raises(ValueError, match="40-character"):
        zenodo.build(archive, tmp_path / "d", revision="main")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy/x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        zenodo.build(archive, tmp_path / "busy", revision=SHA)


def test_dry_run_prints_sizes_and_metadata_and_sends_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("the build must not use the network"))
    assert zenodo.main(["build", "--archive", str(_archive(tmp_path)), "--out", str(tmp_path / "d"),
                        "--revision", SHA, "--date", "2026-10-25", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "obs_discharge_usgs.tar" in text and "MANIFEST.json" in text
    assert '"upload_type": "dataset"' in text and "nothing was sent to Zenodo" in text


# ── deposit, against a fake Zenodo ──────────────────────────────────────────────────────────────────────────


class FakeZenodo:
    """Answers the deposit API calls the way https://developers.zenodo.org/ documents them."""

    def __init__(self, api: str, *, concept: str = "100", latest: int = 102, old_files: int = 2):
        self.api, self.concept, self.latest = api, concept, latest
        self.calls: list[tuple[str, str]] = []
        self.uploads: dict[str, bytes] = {}
        self.metadata: dict | None = None
        self.old_files = [{"id": f"old{i}", "filename": f"old{i}.tar"} for i in range(old_files)]
        self.auth: set[str] = set()

    def _dep(self, dep_id: int) -> dict:
        return {"id": dep_id, "conceptrecid": self.concept, "files": self.old_files,
                "links": {"bucket": f"{self.api}/files/bucket-{dep_id}", "html": f"https://z/deposit/{dep_id}",
                          "latest_draft": f"{self.api}/deposit/depositions/{dep_id}"}}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("Authorization", ""))
        method, path = request.method, request.url.path.removeprefix("/api")
        self.calls.append((method, path))
        if method == "POST" and path == "/deposit/depositions":
            self.old_files = []
            return httpx.Response(201, json=self._dep(201))
        if method == "GET" and path == f"/records/{self.concept}":
            return httpx.Response(200, json={"id": self.latest, "conceptrecid": self.concept})
        if method == "POST" and path == f"/deposit/depositions/{self.latest}/actions/newversion":
            body = self._dep(self.latest)
            body["links"]["latest_draft"] = f"{self.api}/deposit/depositions/301"
            return httpx.Response(201, json=body)
        if method == "GET" and path == "/deposit/depositions/301":
            return httpx.Response(200, json=self._dep(301))
        if method == "DELETE" and path.startswith("/deposit/depositions/301/files/"):
            return httpx.Response(204)
        if method == "PUT" and path.startswith("/files/bucket-"):
            data = request.read()
            name = path.rsplit("/", 1)[-1]
            self.uploads[name] = data
            return httpx.Response(201, json={"key": name, "size": len(data),
                                             "checksum": "md5:" + hashlib.md5(data).hexdigest()})  # noqa: S324
        if method == "PUT" and path.startswith("/deposit/depositions/"):
            self.metadata = json.loads(request.read())
            return httpx.Response(200, json={**self._dep(int(path.rsplit("/", 1)[-1])), "conceptrecid": self.concept})
        if method == "POST" and path.endswith("/actions/publish"):
            return httpx.Response(202, json={"doi": "10.5072/zenodo.301", "conceptdoi": "10.5072/zenodo.100",
                                             "doi_url": "https://doi.org/10.5072/zenodo.301"})
        return httpx.Response(404, json={"message": "not found", "status": 404})


def _built(tmp_path: Path) -> Path:
    out = tmp_path / "deposit"
    zenodo.build(_archive(tmp_path), out, revision=SHA, snapshot_date="2026-10-25")
    return out


def test_a_first_deposit_makes_a_draft_uploads_every_file_and_sets_the_metadata(tmp_path):
    bundle = _built(tmp_path)
    fake = FakeZenodo(zenodo.SANDBOX_API)
    client = httpx.Client(transport=httpx.MockTransport(fake))
    result = zenodo.deposit(bundle, token="t0k", sandbox=True, client=client)

    expected = {p.name for p in zenodo.bundle_files(bundle)}
    assert set(fake.uploads) == expected and "MANIFEST.json" in expected
    assert fake.uploads["MANIFEST.json"] == (bundle / "MANIFEST.json").read_bytes()
    sent = json.loads((bundle / "zenodo-metadata.json").read_text(encoding="utf-8"))
    sent["metadata"]["publication_date"] = fake.metadata["metadata"]["publication_date"]
    assert fake.metadata == sent
    assert fake.calls[0] == ("POST", "/deposit/depositions")
    assert not any(path.endswith("/publish") for _, path in fake.calls), "a draft unless asked"
    assert fake.auth == {"Bearer t0k"}
    assert result["published"] is False and result["doi"] is None and result["id"] == 201


def test_a_new_version_starts_from_the_latest_version_drops_old_files_and_can_publish(tmp_path):
    bundle = _built(tmp_path)
    fake = FakeZenodo(zenodo.ZENODO_API)
    client = httpx.Client(transport=httpx.MockTransport(fake))
    result = zenodo.deposit(bundle, token="t", sandbox=False, concept_id="10.5281/zenodo.100", publish=True,
                            client=client)
    order = [c for c in fake.calls if not c[1].startswith("/files/")]
    assert order[:3] == [("GET", "/records/100"), ("POST", "/deposit/depositions/102/actions/newversion"),
                         ("GET", "/deposit/depositions/301")]
    assert ("DELETE", "/deposit/depositions/301/files/old0") in order
    assert ("DELETE", "/deposit/depositions/301/files/old1") in order
    assert order[-2:] == [("PUT", "/deposit/depositions/301"), ("POST", "/deposit/depositions/301/actions/publish")]
    assert result["published"] and result["doi"] == "10.5072/zenodo.301"


def test_a_version_record_is_not_taken_for_a_concept(tmp_path):
    bundle = _built(tmp_path)
    fake = FakeZenodo(zenodo.ZENODO_API, concept="100")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"id": 102, "conceptrecid": "100"})))
    with pytest.raises(zenodo.ZenodoError, match="not a concept record"):
        zenodo.deposit(bundle, token="t", concept_id="102", client=client)
    assert fake.calls == []


def test_a_missing_token_and_a_refused_token_fail_clearly(tmp_path, monkeypatch):
    bundle = _built(tmp_path)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    with pytest.raises(zenodo.ZenodoError, match="ZENODO_TOKEN"):
        zenodo.deposit(bundle)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"message": "The server could not verify", "status": 401})))
    with pytest.raises(zenodo.ZenodoError, match="401.*token was refused"):
        zenodo.deposit(bundle, token="bad", client=client)


def test_a_changed_bundle_file_is_not_uploaded(tmp_path):
    bundle = _built(tmp_path)
    (bundle / "health.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="health.json"):
        zenodo.deposit(bundle, token="t", client=httpx.Client(transport=httpx.MockTransport(FakeZenodo("x"))))


def test_a_wrong_checksum_from_zenodo_fails_the_upload(tmp_path):
    bundle = _built(tmp_path)
    fake = FakeZenodo(zenodo.SANDBOX_API)

    def corrupt(request):
        resp = fake(request)
        if request.url.path.startswith("/api/files/"):
            return httpx.Response(201, json={"checksum": "md5:0"})
        return resp

    with pytest.raises(zenodo.ZenodoError, match="checksum"):
        zenodo.deposit(bundle, token="t", client=httpx.Client(transport=httpx.MockTransport(corrupt)))


def test_the_workflow_is_manual_sandbox_first_draft_first_and_never_writes_to_hugging_face():
    yaml = pytest.importorskip("yaml")
    text = (ROOT / ".github/workflows/archive-zenodo.yml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    on = doc.get("on", doc.get(True))
    assert set(on) == {"workflow_dispatch"}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["sandbox"]["default"] is True and inputs["publish"]["default"] is False
    assert inputs["concept_id"]["default"] == ""
    assert "ZENODO_SANDBOX_TOKEN" in text and "secrets.ZENODO_TOKEN" in text
    assert "HF_TOKEN" not in text and "publish_folder" not in text
    assert doc["permissions"] == {"contents": "read"}
