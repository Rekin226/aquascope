"""The recorder of the Explorer's recorded studies: the files, the index and meta shapes, freshness, the budget,
the trimming of an oversized workspace, an honest decline, and the CLI verbs. A fake Studio, no network."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from aquascope import cli
from aquascope.studio import showcase
from aquascope.studio.showcase import Case, headline, record, synthetic_flows, usd_for
from aquascope.studio.workspace import Artifact, Workspace

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

CASE = Case(id="kingston-flood", title="Design flood at Kingston", lat=51.415, lon=-0.308,
            problem="Design flow for a road crossing, 100-year return period", kind="flood_risk",
            site="Thames at Kingston", intake={"return_period": 100})
OTHER = Case(id="own-table-flood", title="Your own table", lat=40.2, lon=-8.0,
             problem="Use my attached table of daily flows", kind="flood_risk", site="A stream",
             upload=showcase.TABLE_NAME)


class Reply:
    def __init__(self, kind: str, text: str = "") -> None:
        self.kind, self.text, self.payload = kind, text, {}


class FakeStudio:
    """A Studio that asks one question, plans, and reports (or declines, or errors), with a filled workspace."""

    made: list[dict[str, Any]] = []

    def __init__(self, *, declined: bool = False, huge: bool = False, keyless: bool = False, **kwargs: Any) -> None:
        FakeStudio.made.append(kwargs)
        self.declined, self.huge = declined, huge
        ws = Workspace()
        ws.site = {"lat": kwargs["lat"], "lon": kwargs["lon"]}
        ws.brief.problem = "p"
        ws.brief.intake = dict(kwargs.get("intake") or {})
        ws.brief.playbook = "flood_risk"
        if not keyless:
            ws.model, ws.provider = kwargs.get("model"), kwargs.get("provider")
        for key, value in (kwargs.get("data") or {}).items():
            ws.add_table(f"upload:{key}", value)
        self.workspace = ws
        self.said: list[str] = []

    def say(self, text: str) -> Reply:
        self.said.append(text)
        ws = self.workspace
        if len(self.said) == 1:
            ws.status = "intake"
            return Reply("questions", "One question. Say just go.")
        if self.declined:
            ws.status, ws.declined_reason = "declined", "The requested return period is beyond the record."
            return Reply("declined", "Declined: " + ws.declined_reason)
        ws.study = {"version": 3, "title": "t", "question": "q", "author": "playbook", "created": "2026",
                    "aquascope_version": "0", "model": None,
                    "plan": {"playbook": "flood_risk", "branch": "at_site", "author": "playbook"},
                    "steps": [{"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 1, "lon": 2}},
                              {"id": "s2", "tool": "flood_frequency",
                               "arguments": {"source": "uk_ea", "station_id": "x"}}]}
        ws.status = "review"
        return Reply("plan", "Plan (playbook)")

    def approve(self, edits: Any = None) -> Reply:
        ws = self.workspace
        n = 400_000 if self.huge else 3
        ws.run = {"ok": True, "results": [
            {"id": "s1", "tool": "describe_catchment", "result": {"area_km2": 9990.7}, "gates": []},
            {"id": "s2", "tool": "flood_frequency", "result": {"q100": 520.0, "annual_maxima": list(range(n))},
             "gates": [{"check": "min_years", "passed": True}]}],
            "gates": [{"passed": True}, {"passed": True}, {"passed": False}], "failed_gates": [{"passed": False}],
            "stopped_at": None, "stop_reason": None, "replans": 0}
        ws.critique = {"ok": True, "checks": [{"name": "a", "passed": True}, {"name": "b", "passed": False}],
                       "issues": [], "not_established": ["the tail beyond 200 years"]}
        ws.report = {"title": "Design flood at Kingston", "answer": "**The 100-year flood is about 520 m3/s.** "
                     "The record is long.", "key_numbers": [], "sections": [], "not_established": ["the tail"],
                     "references": [], "footer": {}}
        if ws.model:
            ws.charge("methodologist", 30_000, 2_000)
            ws.charge("author", 20_000, 3_000)
        ws.add_artifact(Artifact(id="fig-s2-frequency_curve", kind="figure", name="figures/s2_frequency_curve.png",
                                 data=PNG, media_type="image/png", caption="The curve", step="s2"))
        ws.add_artifact(Artifact(id="fig-s2-frequency_curve-svg", kind="figure",
                                 name="figures/s2_frequency_curve.svg", data=b"<svg/>", media_type="image/svg+xml"))
        ws.status = "done"
        return Reply("report", ws.report["answer"])


@pytest.fixture(autouse=True)
def _reset():
    FakeStudio.made = []
    yield


def _factory(**flags: Any):
    return lambda **kw: FakeStudio(**flags, **kw)


def test_record_writes_the_files_the_meta_and_the_index(tmp_path):
    lines: list[str] = []
    written = record([CASE, OTHER], tmp_path, studio_factory=_factory(), on_event=lines.append,
                     model="claude-sonnet-5", provider="anthropic")
    assert [m["id"] for m in written] == ["kingston-flood", "own-table-flood"]
    kw = FakeStudio.made[0]
    assert kw["lat"] == 51.415 and kw["provider"] == "anthropic" and kw["model"] == "claude-sonnet-5"
    assert kw["intake"] == {"return_period": 100} and kw["data"] is None
    assert list(FakeStudio.made[1]["data"]) == [showcase.TABLE_NAME]
    case_dir = tmp_path / "kingston-flood"
    assert {p.name for p in case_dir.iterdir()} == {"workspace.json", "report.md", "study.yaml", "figures",
                                                     "meta.json"}
    assert (case_dir / "figures" / "s2_frequency_curve.png").read_bytes() == PNG
    assert not (case_dir / "figures" / "s2_frequency_curve.svg").exists()
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["model"] == "claude-sonnet-5" and meta["steps"] == 2
    assert meta["gates"] == {"passed": 2, "total": 3} and meta["checks"] == {"passed": 1, "total": 2}
    assert meta["tokens"] == {"prompt": 50_000, "completion": 5_000, "calls": 2, "total": 55_000,
                              "by_role": {"methodologist": {"calls": 1, "prompt_tokens": 30_000,
                                                            "completion_tokens": 2_000},
                                          "author": {"calls": 1, "prompt_tokens": 20_000,
                                                     "completion_tokens": 3_000}}}
    assert meta["usd"] == pytest.approx(0.15) and meta["usd_per_mtoken"] == [2.0, 10.0]
    assert meta["headline"] == "The 100-year flood is about 520 m3/s."
    assert meta["playbook"] == "flood_risk" and meta["branch"] == "at_site" and meta["trimmed"] == []
    assert meta["files"] == ["workspace.json", "report.md", "study.yaml", "figures/s2_frequency_curve.png"]
    assert meta["figures"][0]["name"] == "s2_frequency_curve.png" and meta["figures"][0]["step"] == "s2"
    assert meta["site"] == {"lat": 51.415, "lon": -0.308, "name": "Thames at Kingston"}
    assert meta["recorded"].endswith("+00:00") and meta["seconds"] >= 0
    ws = json.loads((case_dir / "workspace.json").read_text(encoding="utf-8"))
    assert ws["status"] == "done" and ws["artifacts"][0].get("data") is None and ws["study"]["version"] == 3
    assert "520 m³/s" in (case_dir / "report.md").read_text(encoding="utf-8")
    assert "flood_frequency" in (case_dir / "study.yaml").read_text(encoding="utf-8")
    # the upload travels in the workspace as CSV text
    ws2 = json.loads((tmp_path / "own-table-flood" / "workspace.json").read_text(encoding="utf-8"))
    assert f"upload:{showcase.TABLE_NAME}" in ws2["tables"] and ws2["tables"][f"upload:{showcase.TABLE_NAME}"]\
        .startswith("date,flow_m3s")
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert set(index) == {"generated", "aquascope_version", "note", "studies"}
    assert [s["id"] for s in index["studies"]] == ["kingston-flood", "own-table-flood"]
    row = index["studies"][0]
    assert row["headline"] == meta["headline"] and row["usd"] == meta["usd"] and row["date"] == meta["recorded"][:10]
    assert row["figures"] == ["s2_frequency_curve.png"] and row["files"] == meta["files"]
    assert {"title", "kind", "site", "shows", "status", "model", "steps", "gates", "checks"} <= set(row)
    assert index["studies"][1]["upload"] == showcase.TABLE_NAME
    assert any("run total" in line for line in lines)


def test_rerun_skips_fresh_recordings_and_only_forces_them(tmp_path):
    record([CASE, OTHER], tmp_path, studio_factory=_factory())
    assert len(FakeStudio.made) == 2
    again = record([CASE, OTHER], tmp_path, studio_factory=_factory(), fresh_for_days=30)
    assert again == [] and len(FakeStudio.made) == 2
    stale = json.loads((tmp_path / "kingston-flood" / "meta.json").read_text(encoding="utf-8"))
    stale["recorded"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(timespec="seconds")
    (tmp_path / "kingston-flood" / "meta.json").write_text(json.dumps(stale), encoding="utf-8")
    third = record([CASE, OTHER], tmp_path, studio_factory=_factory(), fresh_for_days=30)
    assert [m["id"] for m in third] == ["kingston-flood"]
    forced = record([CASE, OTHER], tmp_path, studio_factory=_factory(), only=["own-table-flood"])
    assert [m["id"] for m in forced] == ["own-table-flood"]
    assert record([CASE], tmp_path, studio_factory=_factory(), fresh_for_days=0) != []


def test_the_budget_stops_the_run(tmp_path):
    lines: list[str] = []
    written = record([CASE, OTHER], tmp_path, studio_factory=_factory(), max_usd=0.1, on_event=lines.append)
    assert [m["id"] for m in written] == ["kingston-flood"]
    assert any("budget reached" in line and "own-table-flood" in line for line in lines)
    studies = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["studies"]
    assert [s["id"] for s in studies] == ["kingston-flood"]


def test_an_oversized_workspace_is_trimmed_and_says_so(tmp_path):
    (meta,) = record([CASE], tmp_path, studio_factory=_factory(huge=True))
    assert meta["trimmed"] == ["run.results[s2].result.annual_maxima: 400000 entries kept to 50"]
    ws = json.loads((tmp_path / "kingston-flood" / "workspace.json").read_text(encoding="utf-8"))
    assert len(ws["run"]["results"][1]["result"]["annual_maxima"]) == 50
    assert (tmp_path / "kingston-flood" / "workspace.json").stat().st_size < showcase.MAX_WORKSPACE_BYTES


def test_a_decline_is_a_valid_recording(tmp_path):
    (meta,) = record([CASE], tmp_path, studio_factory=_factory(declined=True))
    assert meta["status"] == "declined" and meta["headline"] == "The requested return period is beyond the record."
    names = {p.name for p in (tmp_path / "kingston-flood").iterdir()}
    assert names == {"workspace.json", "meta.json"} and meta["steps"] == 0 and meta["usd"] == 0
    assert showcase.already_recorded(tmp_path, fresh_for_days=30) == {"kingston-flood"}


def test_a_crash_before_the_studio_exists_is_reported_not_raised(tmp_path):
    def broken(**kw: Any):
        raise RuntimeError("no key")

    lines: list[str] = []
    written = record([CASE], tmp_path, studio_factory=broken, on_event=lines.append)
    assert written == [{"id": "kingston-flood", "status": "error", "error": "RuntimeError: no key", "usd": 0.0}]
    assert any("failed: RuntimeError: no key" in line for line in lines)
    assert not (tmp_path / "kingston-flood").exists()


def test_keyless_recording_costs_nothing(tmp_path):
    (meta,) = record([CASE], tmp_path, studio_factory=_factory(keyless=True), provider=None, model=None)
    assert meta["model"] is None and meta["usd"] == 0 and meta["tokens"]["total"] == 0


def test_helpers():
    assert usd_for({"a": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}, "claude-sonnet-5") == 2.0
    assert usd_for({"a": {"prompt_tokens": 0, "completion_tokens": 1_000_000}}, "unknown", (1.0, 2.0)) == 2.0
    assert headline("") == "" and headline("## No stop here") == "No stop here"
    assert headline("_The_ **uk_ea** gauge (`hubeau_hydrometrie`) is long. Next.") == \
        "The uk_ea gauge (hubeau_hydrometrie) is long."
    assert headline("First. second lower-case continues. Third") == "First. second lower-case continues."
    assert headline("x" * 300).endswith("...") and len(headline("x" * 300)) == 240
    df = synthetic_flows(years=2)
    assert list(df.columns) == ["date", "flow_m3s"] and len(df) == 730 and float(df.flow_m3s.min()) > 0
    assert df.equals(synthetic_flows(years=2)), "deterministic"
    assert len(showcase.CASES) == 12 and len({c.id for c in showcase.CASES}) == 12
    assert {"flood_risk", "drought_status", "supply_reliability", "groundwater_decline", "ungauged_flow",
            "water_quality"} <= {c.kind for c in showcase.CASES}
    assert sum(1 for c in showcase.CASES if c.upload) == 1


def test_diagnose_and_the_cli_list(tmp_path, monkeypatch, capsys):
    assert showcase.diagnose(tmp_path).startswith("no recordings")
    record([CASE], tmp_path, studio_factory=_factory())
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio-showcase", "list", "--out", str(tmp_path)])
    cli.main()
    out = capsys.readouterr().out
    assert "kingston-flood" in out and "done" in out and "1 recording(s), 55,000 tokens, 0.15 USD" in out


def test_the_cli_record_verb_drives_the_recorder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(showcase, "_default_factory", lambda **kw: FakeStudio(**kw))
    monkeypatch.setattr(showcase, "CASES", [CASE])
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio-showcase", "record", "--out", str(tmp_path), "-q",
                                      "--model", "claude-sonnet-5"])
    cli.main()
    out = capsys.readouterr().out
    assert "recorded 1/1 this run, 0.15 USD" in out and (tmp_path / "index.json").exists()
    assert FakeStudio.made[0]["model"] == "claude-sonnet-5"
