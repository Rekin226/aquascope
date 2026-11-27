"""The CLI rescores saved plan evidence without discarding other benchmark records."""

import json
import sys
from pathlib import Path

import pytest

from aquascope import cli
from aquascope.gym import plans as gp


def _rescore(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "rescore", *map(str, args)])
    cli.main()


def _old_plan():
    return {"case_id": "gw_regional_potomac", "playbook": "groundwater_decline", "decline_expected": False,
            "agent": "file", "provider": None, "model": None, "declined": True,
            "declined_reason": "No monitored well is available.", "score": 0, "extra_evidence": {"note": "é"}}


def test_mixed_phases_rescore_only_recognized_plan_fields(tmp_path, monkeypatch):
    path = tmp_path / "mixed.jsonl"
    out = tmp_path / "rescored.jsonl"
    other = ['{"task_id":"phase1","agent":"tree"}', json.dumps({**_old_plan(), "phase": "other"}),
             json.dumps({**_old_plan(), "case_id": "future"}), json.dumps({**_old_plan(), "error": "offline"})]
    path.write_bytes(("\r\n".join([other[0], json.dumps(_old_plan()), "", *other[1:]]) + "\r\n").encode())
    original = path.read_bytes()
    _rescore(monkeypatch, path, "--out", out)
    assert path.read_bytes() == original
    lines = out.read_bytes().decode().split("\r\n")
    assert [lines[0], *lines[3:-1]] == other and lines[2] == "" and lines[-1] == ""
    rescored = json.loads(lines[1])
    assert rescored["score"] is None and rescored["decline_correct"] is None
    assert rescored["extra_evidence"] == {"note": "é"}
    before_retry = out.read_bytes()
    _rescore(monkeypatch, out, "--out", out)
    assert out.read_bytes() == before_retry


def test_multiple_inputs_and_out_leave_historical_runs_untouched(tmp_path, monkeypatch):
    runs = Path(gp.__file__).parent / "results"
    sources = [runs / "2026-09-03/tree.jsonl", runs / "2026-09-08/plans-tree.jsonl"]
    copies = [tmp_path / src.name for src in sources]
    for src, copy in zip(sources, copies):
        copy.write_bytes(src.read_bytes())
    out = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit):
        _rescore(monkeypatch, *copies, "--out", out)
    assert not out.exists()
    assert all(copy.read_bytes() == src.read_bytes() for copy, src in zip(copies, sources))


def test_later_invalid_json_leaves_all_originals_intact(tmp_path, monkeypatch):
    first, bad = tmp_path / "first.jsonl", tmp_path / "bad.jsonl"
    original = json.dumps(_old_plan()).encode() + b"\n"
    first.write_bytes(original)
    bad.write_bytes(b"{invalid\n")
    with pytest.raises((ValueError, SystemExit)):
        _rescore(monkeypatch, first, bad)
    assert first.read_bytes() == original and bad.read_bytes() == b"{invalid\n"
    assert set(tmp_path.iterdir()) == {first, bad}


def test_multiple_in_place_rescore_and_duplicate_input(tmp_path, monkeypatch):
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for path in paths:
        path.write_text(json.dumps(_old_plan()), encoding="utf-8")
    original = paths[0].read_bytes()
    with pytest.raises(SystemExit, match="more than once"):
        _rescore(monkeypatch, paths[0], paths[0])
    assert paths[0].read_bytes() == original
    _rescore(monkeypatch, *paths)
    for path in paths:
        assert json.loads(path.read_bytes())["score"] is None


def test_failed_replace_preserves_the_input_and_removes_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    original = json.dumps(_old_plan()).encode()
    path.write_bytes(original)

    def fail_replace(self, target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(SystemExit, match="replace failed"):
        _rescore(monkeypatch, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_rescore_keeps_a_result_appended_while_scoring(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    original = json.dumps(_old_plan()).encode() + b"\n"
    path.write_bytes(original)
    appended = b'{"task_id": "new"}\n'
    score = gp.score_plan

    def score_and_append(*args, **kwargs):
        with path.open("ab") as f:
            f.write(appended)
        return score(*args, **kwargs)

    monkeypatch.setattr(gp, "score_plan", score_and_append)
    with pytest.raises(SystemExit, match="changed during rescoring"):
        _rescore(monkeypatch, path)
    assert path.read_bytes() == original + appended
