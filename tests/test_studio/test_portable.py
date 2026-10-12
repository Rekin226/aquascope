"""Completed studies round-trip their evidence and bytes without executing a plan."""

import json

import pytest

from aquascope.claims import content_digest
from aquascope.studio import portable
from aquascope.studio.workspace import Artifact, Workspace


def test_round_trip_keeps_inputs_report_findings_and_artifacts():
    ws = Workspace(status="done", site={"lat": 1, "lon": 2}, tables={"mine.csv": "date,value\n2020-01-01,3"})
    ws.report = {"answer": "Stored answer"}
    ws.findings = {"decision": {"evidence": {"result_id": "s1.ffa.fits.lp3.q.0"}}}
    ws.artifacts = [Artifact(id="figure", kind="figure", name="figures/result.png", data=b"figure"),
                    Artifact(id="report-html", kind="document", name="report.html", data=b"<p>Stored answer</p>"),
                    Artifact(id="bundle", kind="bundle", name="bundle.zip", data=b"recursive bundle")]
    text = portable.dumps(ws)
    restored = portable.loads(text)
    assert restored.tables == ws.tables and restored.report == ws.report and restored.findings == ws.findings
    assert restored.artifact("figure").data == b"figure"
    assert restored.artifact("report-html").data == b"<p>Stored answer</p>"
    assert restored.artifact("bundle") is None
    assert "recursive bundle" not in text


def test_tampering_and_unsupported_versions_are_rejected():
    obj = json.loads(portable.dumps(Workspace()))
    obj["workspace"]["status"] = "done"
    with pytest.raises(ValueError, match="checksum"):
        portable.loads(json.dumps(obj))
    obj["version"] = 9
    with pytest.raises(ValueError, match="version"):
        portable.loads(json.dumps(obj))


def test_unsafe_paths_are_rejected_even_with_a_matching_checksum():
    obj = json.loads(portable.dumps(Workspace()))
    obj["workspace"]["artifacts"] = [{"id": "bad", "name": "../report.html", "data": ""}]
    obj["sha256"] = content_digest(obj["workspace"])
    with pytest.raises(ValueError, match="Unsafe"):
        portable.loads(json.dumps(obj))


def test_export_limit_is_checked(monkeypatch):
    monkeypatch.setattr(portable, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="limit"):
        portable.dumps(Workspace())


def test_lightweight_workspace_cannot_silently_export_empty_artifacts():
    ws = Workspace(artifacts=[Artifact(id="figure", kind="figure", name="figure.png")])
    with pytest.raises(ValueError, match="missing artifact bytes"):
        portable.dumps(ws)
