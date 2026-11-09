"""Sample model results in the Archive (#591): the copy, its check, its manifest and what a publish uploads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from aquascope.archive import samples

ROOT = Path(__file__).resolve().parents[2]


def fake_sample(monkeypatch, data: bytes) -> dict:
    s = dict(samples.HEC_RAS, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(samples.SAMPLES, s["id"], s)
    return s


def test_build_copies_checks_and_writes_the_manifest(tmp_path, monkeypatch):
    data = b"\x89HDF\r\n\x1a\n" + b"x" * 1000
    src = tmp_path / "plan.hdf"
    src.write_bytes(data)
    s = fake_sample(monkeypatch, data)
    info = samples.build(tmp_path / "out", source=src)
    folder = tmp_path / "out" / "samples" / "hec-ras"
    assert (folder / s["name"]).read_bytes() == data
    m = json.loads((folder / "manifest.json").read_text())
    assert m == info
    assert m["files"] == [{"path": f"samples/hec-ras/{s['name']}", "name": s["name"], "bytes": len(data),
                           "sha256": s["sha256"]}]
    assert m["source"]["licence"] == "CC BY 4.0" and m["source"]["creator"] == "Zandsalimi, Zanko"
    assert "GNN4CF" in m["attribution"] and "CC BY 4.0" in m["attribution"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", m["mirrored_at"])


def test_build_refuses_a_file_that_is_not_the_recorded_one(tmp_path, monkeypatch):
    fake_sample(monkeypatch, b"the recorded file")
    src = tmp_path / "plan.hdf"
    src.write_bytes(b"something else")
    with pytest.raises(ValueError, match="changed at the source"):
        samples.build(tmp_path / "out", source=src)
    assert not list((tmp_path / "out" / "samples" / "hec-ras").glob("*.hdf"))
    assert not (tmp_path / "out" / "samples" / "hec-ras" / "manifest.json").exists()


def test_publish_uploads_only_samples(tmp_path, monkeypatch):
    data = b"plan"
    src = tmp_path / "plan.hdf"
    src.write_bytes(data)
    fake_sample(monkeypatch, data)
    samples.build(tmp_path / "out", source=src)
    (tmp_path / "out" / "stray.parquet").write_bytes(b"not mine")
    seen = {}

    def fake_publish(folder, repo_id, **kw):
        seen["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        seen["patterns"] = kw["allow_patterns"]
        return "https://huggingface.co/commit/1"

    monkeypatch.setattr("aquascope.archive.publish.publish_folder", fake_publish)
    assert samples.publish(tmp_path / "out") == "https://huggingface.co/commit/1"
    assert seen["files"] == ["samples/hec-ras/Flood_model.p01_BC12_S6_R6.hdf", "samples/hec-ras/manifest.json"]
    assert all(p.startswith("samples/") for p in seen["patterns"])


def test_publish_without_a_build_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="build first"):
        samples.publish(tmp_path)


def test_the_recorded_sample_is_the_one_assessed_in_592():
    doc = (ROOT / "docs" / "assessments" / "hec-ras.md").read_text(encoding="utf-8")
    s = samples.HEC_RAS
    assert s["source"]["url"].endswith("GNN4CF_dataset/raw_hecras/test_multi_boundary/" + s["name"])
    assert "32aac3af553c4e1fbb997ec58d9de9b6" in s["source"]["url"] and "32aac3af553c4e1fbb997ec58d9de9b6" in doc
    assert f"SHA-256 `{s['sha256'][:8]}…{s['sha256'][-7:]}`" in doc
    assert "(29.9 MB" in doc and round(s["bytes"] / 1e6, 1) == 29.9


def test_the_explorer_reads_the_same_path_and_credit():
    js = (ROOT / "explorer" / "src" / "model-samples-core.js").read_text(encoding="utf-8")
    s = samples.HEC_RAS
    assert f"samples/{s['folder']}/manifest.json" in js
    assert s["source"]["resource"] in js
    assert s["source"]["licence"] in js and "Zandsalimi" in js


def test_the_workflow_writes_only_samples_and_a_smoke_run_never_publishes():
    yaml = pytest.importorskip("yaml")
    text = (ROOT / ".github" / "workflows" / "mirror-samples.yml").read_text()
    doc = yaml.safe_load(text)
    on = doc.get("on", doc.get(True))
    assert set(on) == {"workflow_dispatch"} and "smoke" in on["workflow_dispatch"]["inputs"]
    steps = doc["jobs"]["mirror"]["steps"]
    publish = [st for st in steps if "samples publish" in (st.get("run") or "")]
    assert len(publish) == 1 and "!inputs.smoke" in publish[0]["if"]
    assert "secrets.HF_TOKEN" in text
