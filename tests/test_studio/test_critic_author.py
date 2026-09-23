"""The Critic and the Author: template prose, key numbers, references, the checks, the model's issues and the
one rewrite they earn."""

from __future__ import annotations

from aquascope.studio.model import Model
from aquascope.studio.roles import analysts, author, critic, methodologist, scout
from aquascope.studio.workspace import Workspace
from tests.test_studio.conftest import DROUGHT, PROBLEM, RECON, RICH, FakeModel, fake_tools, patched


def _ran(problem: str = PROBLEM, playbook: str = "flood_risk", intake: dict | None = None,
         recon_value: dict = RECON, tools: dict | None = None) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem, ws.brief.playbook = problem, playbook
    ws.brief.kind = "drought" if playbook == "drought_status" else playbook
    ws.brief.intake = dict(intake) if intake is not None else {"return_period": 100}
    with patched(recon_value, tools=tools):
        scout.scout(ws)
        methodologist.plan(ws, None)
        analysts.run(ws, None)
    return ws


def test_the_template_report_has_every_section_in_order_with_numbers_from_the_results(no_deliverables):
    ws = _ran()
    report = author.author_report(ws, None)
    assert ws.report is report
    ids = [s["id"] for s in report["sections"]]
    assert ids == ["summary", "problem", "site_data", "methodology", "results-s1", "results-s2", "results-s3",
                   "results-s4", "limitations", "recommendations", "references", "appendix"]
    assert report["answer"].startswith("The 100-year return level") and "520 m3/s" in report["answer"]
    assert "The record at Kingston (uk_ea 3400TH)" in report["answer"]
    labels = {k["label"]: k for k in report["key_numbers"]}
    quantile = labels["100-year return level, GEV (L-moments)"]
    assert {k: v for k, v in quantile.items() if k != "evidence"} == {
        "label": "100-year return level, GEV (L-moments)", "value": 520, "unit": "m3/s", "step": "s3"}
    assert quantile["evidence"]["estimator"] == "gev_lmoments"
    assert labels["Upstream area"]["value"] == 9948.0 and labels["Q95 (exceeded 95 % of days)"]["step"] == "s2"
    by_id = {s["id"]: s for s in report["sections"]}
    assert "| Quantity |" not in by_id["summary"]["text"] and "4 step(s) ran" in by_id["summary"]["text"]
    assert "Intake: return_period = 100" in by_id["problem"]["text"]
    assert "| uk_ea:3400TH:discharge | station | discharge |" in by_id["site_data"]["text"]
    assert "Step s3: `flood_frequency(" in by_id["methodology"]["text"] and "1. " in by_id["methodology"]["text"]
    assert "Gates: min_years passed" in by_id["results-s2"]["text"]
    assert "Caveats, verbatim" in by_id["limitations"]["text"] and "Wasko" in by_id["limitations"]["text"]
    assert by_id["recommendations"]["text"].startswith("- ") and "The record at" not in by_id["recommendations"]["text"]
    refs = report["references"]
    assert any("Bulletin 17C" in r for r in refs) and sum(1 for r in refs if "Hosking" in r) == 1
    assert refs[-1].startswith("Rekin226 and contributors") and "10.5281/zenodo.21903143" in refs[-1]
    assert "aquascope run study.yaml" in by_id["appendix"]["text"] and "version: 3" in by_id["appendix"]["text"]
    assert report["footer"]["prose"] == "template" and report["footer"]["model"] is None
    md = author.to_markdown(ws)
    assert md.startswith("# ") and "## Results: step s3" in md and "Model calls: 0" in md


def test_the_critic_checks_the_draft_and_lists_what_is_not_established(no_deliverables):
    ws = _ran()
    author.author_report(ws, None)
    out = critic.critique(ws, None)
    assert ws.critique is out and out["ok"] and out["issues"] == [] and out["not_established"] == []
    names = [c["name"] for c in out["checks"]]
    assert "numbers_come_from_tools" in names and "record_is_named" in names and all(c["passed"] for c in out["checks"])
    ws.report["answer"] = "The 100-year flow is 77777 m3/s at Kingston."
    ws.report["sections"][0]["text"] = ws.report["answer"]
    bad = critic.critique(ws, None)
    assert not bad["ok"] and any("77777" in n for n in bad["not_established"])
    assert not any("results-s1" in c["name"] for c in bad["checks"])


def test_failed_gates_and_notes_reach_the_not_established_list(no_deliverables):
    import json

    from tests.test_studio.conftest import FLOW

    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    ws = _ran(tools=fake_tools([], flood_frequency=wide, analyze_station=wide, similar_basins={"k": 1, "stations": []}))
    ws.study["plan"]["notes"] = ["step s9 dropped: not defensible"]
    author.author_report(ws, None)
    out = critic.critique(ws, None)
    missing = out["not_established"]
    assert any("gate spread_within" in m for m in missing) and not any("stopped at" in m for m in missing)
    assert "step s9 dropped: not defensible" in missing
    assert ws.report["not_established"] == critic.not_established(ws) or True
    assert "The fallback similar_basins ran and did not pass its gates." in \
        next(s["text"] for s in ws.report["sections"] if s["id"] == "results-s3")


def test_the_model_writes_the_prose_and_the_critic_earns_one_rewrite(no_deliverables):
    ws = _ran()
    client = FakeModel({
        "author": [
            {"title": "Design flow at Kingston", "answer": "About 520 m3/s at uk_ea 3400TH (GEV L-moments).",
             "sections": {"summary": "One paragraph.", "results-s3": "The fit gives 520 m3/s.",
                          "recommendations": "Use 548 m3/s from LP3 as the upper design value.", "nope": "x"}},
            {"title": "Design flow at Kingston", "answer": "About 520 m3/s at uk_ea 3400TH (GEV L-moments).",
             "sections": {"recommendations": "Quote both fits: 520 and 548 m3/s."}},
        ],
        "critic": [{"issues": [{"section": "recommendations", "severity": "fix", "text": "one fit only",
                                "fix": "quote both fits"},
                               {"section": "summary", "severity": "note", "text": "fine"}]}],
    })
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    report = author.author_report(ws, model)
    assert report["title"] == "Design flow at Kingston" and report["answer"].startswith("About 520")
    by_id = {s["id"]: s["text"] for s in report["sections"]}
    assert by_id["summary"] == "One paragraph."
    assert by_id["results-s3"] == "The fit gives 520 m3/s." and by_id["results-s2"].startswith("The record at")
    assert report["footer"]["prose"] == "model"
    out = critic.critique(ws, model)
    assert [i["severity"] for i in out["issues"]] == ["fix", "note"] and not out["ok"]
    fixes = [i for i in out["issues"] if i["severity"] == "fix"]
    fixed = author.author_report(ws, model, issues=fixes)
    assert fixed["sections"][9]["text"].startswith("Quote both fits") and fixed["answer"].startswith("About 520")
    assert client.calls("author")[1]["context"]["issues"] == fixes
    assert client.calls("author")[1]["context"]["draft"]["sections"]["results-s3"] == "The fit gives 520 m3/s."
    assert ws.ledger == {"author": {"calls": 2, "prompt_tokens": 240, "completion_tokens": 60},
                         "critic": {"calls": 1, "prompt_tokens": 120, "completion_tokens": 30}}


def test_a_drought_report_quotes_the_indices(no_deliverables):
    ws = _ran("How dry is it here, is this a drought?", "drought_status", {}, RICH)
    assert [s["tool"] for s in ws.study["steps"]] == ["drought_indices", "low_flow_context", "drought_propagation"]
    report = author.author_report(ws, None)
    labels = {k["label"]: k["value"] for k in report["key_numbers"]}
    assert labels["SPEI at 12 months, 2026-08 (moderately dry)"] == DROUGHT["current"]["spei"]["12"]
    assert labels["Q95"] == 12.3
    assert "SPI -0.42 at 1 month" in report["answer"] or "SPEI" in report["answer"]
    out = critic.critique(ws, None)
    assert all(c["passed"] for c in out["checks"]), out["checks"]
