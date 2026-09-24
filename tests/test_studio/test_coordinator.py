"""The Coordinator: keyless end to end, questions, review edits, follow-ups, checkpoints, the model crew, and
what happens when a role fails."""

from __future__ import annotations

import json

import pytest

from aquascope.study import loads, run_study
from tests.test_studio.conftest import PROBLEM, FakeModel, fake_tools, patched
from tests.test_studio.test_methodologist import VALID_PLAN


def test_keyless_end_to_end_say_plan_approve_report_export(studio_factory, tmp_path):
    events: list = []
    s, calls = studio_factory(on_event=events.append)
    r = s.say(PROBLEM)
    assert r.kind == "plan" and s.workspace.status == "review" and "[s3] flood_frequency" in r.text
    assert r.payload["study"]["version"] == 3 and r.payload["plan"]["branch"] == "at_site"
    assert calls == [], "nothing runs before the approval"
    r2 = s.approve()
    ws = s.workspace
    assert r2.kind == "report" and ws.status == "done" and "100-year return level" in r2.text[:120]
    assert r2.text.startswith("design flow:") and "(established)" in r2.text[:160]
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    report = r2.payload["report"]
    assert [x["id"] for x in report["sections"]][:7] == ["summary", "decision", "findings", "problem", "site_data",
                                                          "methodology", "results-s1"]
    assert any(k["label"].startswith("100-year return level") and k["value"] == 520 for k in report["key_numbers"])
    assert report["not_established"] == [] and report["critique"]["checks_passed"] == report["critique"]["checks"]
    assert ws.ledger == {} and ws.model is None
    statuses = [e["detail"] for e in events if e["event"] == "status"]
    assert statuses == ["scouting", "planning", "review", "running", "critique", "authoring", "done"]
    assert [m.kind for m in ws.messages] == ["text", "brief", "plan", "report"]
    assert any(e["event"] == "deliverables_unavailable" for e in events)
    paths = s.export(tmp_path / "out")
    assert set(paths) == {"report.md", "study.yaml", "report.json", "workspace.json", "study_map.geojson"}
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Methodology" in md and "520 m3/s" in md and "Model calls: 0" in md
    # the study replays with no model and lands on the same results
    back = loads((tmp_path / "out" / "study.yaml").read_text(encoding="utf-8"))
    assert back.version == 3 and back.plan["objective"] == PROBLEM
    with patched():
        rerun = run_study(back)
    assert rerun.ok and [r["sha256"] for r in rerun.results] == [r["sha256"] for r in ws.run["results"]]
    saved = json.loads((tmp_path / "out" / "workspace.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done" and saved["study"]["version"] == 3


def test_questions_round_trip_to_a_plan(studio_factory):
    s, calls = studio_factory()
    r = s.say("Can the river supply the town with 3 ML/day reliably?")
    assert r.kind == "questions" and s.workspace.status == "intake"
    assert [q["id"] for q in r.questions] == ["decision"] and "just go" in r.text, \
        "the demand is in the text; what is decided is not"
    r2 = s.say("just go")
    assert r2.kind == "plan" and s.workspace.brief.playbook == "supply_reliability"
    assert s.workspace.brief.intake["demand_ml_day"] == 3.0 and s.workspace.brief.intake["use"] == "municipal"
    assert s.workspace.brief.decision is None, "no default is invented for the decision"
    r3 = s.approve()
    assert r3.kind == "report" and [c[0] for c in calls][-1] == "supply_reliability"
    assert "reliable" in r3.text or any(k["label"] == "Verdict" for k in r3.payload["report"]["key_numbers"])


def test_no_playbook_and_no_model_declines_in_the_playbooks_words(studio_factory):
    s, _ = studio_factory()
    r = s.say("Tell me about the water here")
    assert r.kind == "questions" and r.questions[0]["id"] == "playbook"
    r2 = s.say("just go")
    assert r2.kind == "declined" and s.workspace.status == "declined" and "flood_risk" in r2.text
    assert s.say("anything").kind == "declined" and s.approve().kind == "declined"


def test_the_brief_can_change_at_review_and_edits_are_revalidated(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    r = s.say("make it a 50-year return period")
    assert r.kind == "plan" and s.workspace.brief.intake["return_period"] == 50 and "T = 50 year" in r.text
    bad = s.approve(edits={"s2": {"arguments": {"nope": 1}}})
    assert bad.kind == "plan" and bad.payload["errors"] and s.workspace.status == "review"
    good = s.approve(edits={"s1": None, "s3": {"arguments": {"bootstrap_ci": False}}})
    assert good.kind == "report" and [c[0] for c in calls] == ["analyze_station", "flood_frequency", "anywhere"]
    assert calls[1][1] == {"source": "uk_ea", "station_id": "3400TH", "bootstrap_ci": False}
    assert s.workspace.study["plan"]["edited"] is True


def test_say_approve_word_at_review_runs(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    assert s.say("approve").kind == "report" and len(calls) == 4


def test_follow_up_question_and_change(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    s.approve()
    q = s.follow_up("how sure can we be?")
    assert q.kind == "question" or q.kind == "answer"
    assert "520" in q.text and s.workspace.follow_ups[-1]["kind"] == "question" and len(calls) == 4
    c = s.follow_up("redo it with a 50-year return period")
    assert c.kind == "report" and s.workspace.status == "done"
    assert s.workspace.brief.intake["return_period"] == 50 and s.workspace.follow_ups[-1]["kind"] == "change"
    assert [x[0] for x in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere",
                                     "flood_frequency", "anywhere"]
    assert any("50-year" in k["label"] for k in c.payload["report"]["key_numbers"])
    assert "T = 50" in next(g["detail"] for g in s.workspace.run["gates"] if g["check"] == "max_return_period_factor")
    assert s.say("and a 20-year return period?").kind == "report", "say() after the report is a follow-up"
    assert s.workspace.follow_ups[-1]["kind"] == "change" and s.workspace.brief.intake["return_period"] == 20


def test_checkpoint_and_resume_mid_way(studio_factory):
    from aquascope.studio import Studio

    s, calls = studio_factory()
    s.say(PROBLEM)
    frozen = s.to_dict()
    assert frozen["status"] == "review" and frozen["study"]["steps"]
    text = json.dumps(frozen)
    calls2: list = []
    with patched(tools=fake_tools(calls2)):
        s2 = Studio.from_dict(json.loads(text), tools=fake_tools(calls2))
        r = s2.approve()
    assert r.kind == "report" and calls == [] and len(calls2) == 4
    assert s2.workspace.id == s.workspace.id and s2.workspace.brief.problem == PROBLEM
    with patched(tools=fake_tools(calls2)):
        s3 = Studio.from_dict(s2.to_dict(with_artifacts=False), tools=fake_tools(calls2))
        assert s3.workspace.status == "done" and s3.follow_up("what is the 100-year flow?").kind == "answer"
    with pytest.raises(ValueError, match="site"):
        Studio()


def test_a_failing_run_still_ends_in_a_report(studio_factory):
    def boom(**kw):
        raise RuntimeError("agency down")

    s, calls = studio_factory(tools=fake_tools([], analyze_station=boom))
    s.say(PROBLEM)
    r = s.approve()
    ws = s.workspace
    assert r.kind == "report" and ws.status == "done" and not ws.run["ok"]
    assert any("agency down" in n and "did not run" in n for n in r.payload["not_established"])
    assert not any("stopped at" in n for n in r.payload["not_established"]), "a failed step no longer stops the study"
    assert ws.run["stopped_at"] is None and not ws.run["results"][1]["ok"]
    assert len(ws.run["results"]) == len(ws.study["steps"]), "every planned step ran or was skipped"
    assert ws.run["summary"]["failed"] >= 1 and [f["id"] for f in ws.run["failed_steps"]][0] == "s2"
    s2_text = next(x["text"] for x in r.payload["report"]["sections"] if x["id"] == "results-s2")
    assert "failed: RuntimeError: agency down" in s2_text


def test_the_model_crew_end_to_end(studio_factory):
    client = FakeModel({
        "consultant": [{"decision": "size the crossing", "quantities": ["the 100-year flow with a band"],
                        "kind": "flood_risk", "playbook": "flood_risk", "intake": {"return_period": 100},
                        "assumptions": ["the gauge is representative"], "questions": [], "ready": True}],
        "methodologist": [VALID_PLAN],
        "author": [{"title": "Design flow at Kingston",
                    "answer": "About 520 m3/s at uk_ea 3400TH by GEV (90 % band 420 to 650 m3/s).",
                    "sections": {"summary": "The 100-year flow at Kingston is 520 m3/s.",
                                 "recommendations": "Adopt 548 m3/s (LP3) as the design value with the band."}}],
        "critic": [{"issues": []}],
    })
    s, calls = studio_factory(client=client)
    r = s.say("A culvert on the Thames at Kingston, 100-year")
    ws = s.workspace
    assert r.kind == "plan" and ws.brief.source == "model" and ws.study["author"] == "methodologist"
    assert len(ws.study["steps"]) == 4
    r2 = s.approve()
    assert r2.kind == "report" and "About 520" in r2.text and "(established)" in r2.text
    assert r2.payload["grade"] == "established" and r2.payload["decision"]["value"] == 520
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    assert set(ws.ledger) == {"consultant", "methodologist", "interpreter", "author", "critic"}
    assert r2.payload["report"]["footer"]["prose"] == "model" and r2.payload["report"]["footer"]["model"] == "fake"
    assert r2.payload["report"]["footer"]["tokens"]["author"]["calls"] == 1
    assert ws.report["critique"]["issues"] == [] and ws.report["not_established"] == []
    assert all(len(req["system"]) < 2500 for req in client.requests), "prompts stay tight"
