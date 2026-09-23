"""The keyless Consultant's questions from the gaps, and the fuller key numbers."""

from __future__ import annotations

import json

from aquascope.studio.roles import analysts, author, consultant, methodologist, scout
from aquascope.studio.workspace import Workspace
from tests.test_studio.conftest import (
    DROUGHT,
    FLOW,
    PROBLEM,
    RECON,
    RICH,
    SIGNATURES,
    SUPPLY,
    UNGAUGED,
    fake_tools,
    patched,
)

TWO_COLUMNS = "\n".join(["date,flow_m3s,stage_m"] + [
    f"20{10 + i // 365:02d}-{(i % 365) // 31 + 1:02d}-{i % 28 + 1:02d},{20 + (i * 7) % 90},{0.5 + (i % 40) / 100}"
    for i in range(0, 365 * 12, 3)]) + "\n"


def _ws(**tables) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    for k, v in tables.items():
        ws.add_table(k, v)
    return ws


# ── the questions ──


def test_a_flood_question_without_a_return_period_asks_with_the_default():
    ws = _ws()
    msg = consultant.consult(ws, None, "A culvert on the Hogsmill")
    qs = ws.brief.open_questions
    assert msg.kind == "questions" and [q.id for q in qs] == ["return_period"] and qs[0].default == 100
    assert ws.brief.decision == "design flow", "culvert names the decision"
    consultant.consult(ws, None, "200 years")
    assert ws.brief.ready and ws.brief.intake["return_period"] == 200


def test_no_decision_is_asked_with_the_playbooks_options():
    ws = _ws()
    consultant.consult(ws, None, "What is the 1 in 50 flood here?")
    qs = ws.brief.open_questions
    assert [q.id for q in qs] == ["decision"] and qs[0].options == ["design flow", "risk screening", "insurance",
                                                                    "inundation extent"]
    assert qs[0].default == "design flow" and ws.brief.intake["return_period"] == 50
    consultant.consult(ws, None, "insurance")
    assert ws.brief.decision == "insurance" and ws.brief.intake["decision"] == "insurance" and ws.brief.ready
    ws2 = _ws()
    consultant.consult(ws2, None, "What flow can I expect from this ungauged stream for irrigation?")
    assert ws2.brief.playbook == "ungauged_flow" and ws2.brief.decision == "irrigation offtake"
    assert ws2.brief.ready, "the purpose field is the decision and the text names it"
    ws3 = _ws()
    msg = consultant.consult(ws3, None, "What flow can I expect from this ungauged stream?")
    assert msg.kind == "questions" and ws3.brief.open_questions[0].id == "purpose"
    assert ws3.brief.open_questions[0].text == "What the flow is for?"


def test_a_drought_question_without_a_period_asks_for_one_and_just_go_lists_the_defaults():
    ws = _ws()
    msg = consultant.consult(ws, None, "How dry is it here, is this a drought?")
    assert msg.kind == "questions" and [q.id for q in ws.brief.open_questions] == ["decision", "period"]
    period = ws.brief.open_questions[1]
    assert period.default == "now" and "the last 12 months" in period.options
    consultant.consult(ws, None, "just go")
    assert ws.brief.ready and ws.brief.period == "now" and ws.brief.decision is None
    assert any(a.startswith("Which period is the drought question about?: now (the default")
               for a in ws.brief.assumptions)
    assert "The client asked to proceed on the defaults." in ws.brief.assumptions
    ws2 = _ws()
    consultant.consult(ws2, None, "Is this area in drought now? We may need restrictions.")
    assert ws2.brief.ready and ws2.brief.period == "now" and ws2.brief.decision == "drought restrictions"
    ws3 = _ws()
    consultant.consult(ws3, None, "Drought status for irrigation planning")
    consultant.consult(ws3, None, "the last 3 months")
    assert ws3.brief.period == "the last 3 months" and ws3.brief.intake["timescales"] == [1, 3, 12]


def test_an_upload_with_two_numeric_columns_asks_which_and_the_plan_loads_it():
    ws = _ws(**{"upload:gauge.csv": TWO_COLUMNS})
    msg = consultant.consult(ws, None, "Study my own record: flood frequency for a 50-year design")
    q = ws.brief.open_questions
    assert msg.kind == "questions" and [x.id for x in q] == ["value_column"]
    assert q[0].options == ["flow_m3s", "stage_m"] and q[0].default == "flow_m3s" and "upload:gauge.csv" in q[0].text
    consultant.consult(ws, None, "the flow_m3s one")
    assert ws.brief.ready and ws.brief.intake["value_column"] == "flow_m3s"
    with patched(UNGAUGED):
        scout.scout(ws)
        study = methodologist.plan(ws, None)
    upload = ws.inventory.dataset("upload:gauge.csv")
    assert upload.variable == "discharge" and upload.quality["mapping"]["value_column"] == "flow_m3s", \
        "the Scout reads the column the client named"
    assert study.steps[0].tool == "load_table" and study.steps[0].arguments == {"table": "upload:gauge.csv",
                                                                                 "value_column": "flow_m3s"}
    with patched(UNGAUGED):
        run = analysts.run(ws, None)
    assert run.ok and run.results[0]["result"]["mapping"]["value_column"] == "flow_m3s"
    other = _ws(**{"upload:gauge.csv": TWO_COLUMNS})
    with patched(UNGAUGED):
        scout.scout(other)
    assert other.inventory.dataset("upload:gauge.csv").quality["mapping"]["value_column"] != "flow_m3s", \
        "left to the guess, the stage column is taken for the record"
    one = _ws(**{"upload:flows.csv": "date,flow\n2020-01-01,1\n2020-01-02,2\n"})
    consultant.consult(one, None, PROBLEM)
    assert one.brief.ready, "one numeric column needs no question"


def test_at_most_three_questions_and_the_demand_pair_is_one():
    ws = _ws(**{"upload:gauge.csv": TWO_COLUMNS})
    consultant.consult(ws, None, "Can the river supply the town reliably?")
    ids = [q.id for q in ws.brief.open_questions]
    assert ids == ["demand_m3s", "decision", "value_column"] and len(ids) <= consultant.MAX_QUESTIONS
    consultant.consult(ws, None, "3 ML/day, a screening, flow_m3s")
    assert ws.brief.ready and ws.brief.intake["demand_ml_day"] == 3.0 and "demand_m3s" not in ws.brief.intake
    assert ws.brief.questions[0].answer == "" and ws.brief.decision == "a screening of the source"


def test_the_model_path_is_unchanged_by_the_gap_rules(monkeypatch):
    import aquascope.explore
    from aquascope.studio.model import Model
    from tests.test_studio.conftest import FakeModel

    monkeypatch.setattr(aquascope.explore, "assess_site", lambda *a, **k: {"stations": [], "context": {}},
                        raising=False)
    ws = _ws()
    client = FakeModel({"consultant": [{"decision": "size a culvert", "kind": "flood_risk", "playbook": "flood_risk",
                                        "intake": {}, "questions": [], "ready": True}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    msg = consultant.consult(ws, model, "A culvert on the Hogsmill")
    assert msg.kind == "brief" and ws.brief.ready and ws.brief.source == "model", "the model decides what to ask"


# ── the key numbers ──


def _ran(problem: str, playbook: str, intake: dict, recon_value: dict, tools: dict | None = None) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem, ws.brief.playbook = problem, playbook
    ws.brief.kind = {"drought_status": "drought", "supply_reliability": "supply_reliability"}.get(playbook, playbook)
    ws.brief.intake = dict(intake)
    with patched(recon_value, tools=tools):
        scout.scout(ws)
        methodologist.plan(ws, None)
        analysts.run(ws, None)
    return ws


def test_both_fits_at_every_return_period_after_the_headline(no_deliverables):
    ws = _ran(PROBLEM, "flood_risk", {"return_period": 100}, RECON)
    report = author.author_report(ws, None)
    labels = [k["label"] for k in report["key_numbers"]]
    head = labels.index("100-year return level, GEV (L-moments)")
    assert labels[head:head + 7] == ["100-year return level, GEV (L-moments)", "100-year return level, Log-Pearson III",
                                     "100-year LP3 90 % interval, low", "100-year LP3 90 % interval, high",
                                     "100-year return level, GEV (MLE with L-moments fallback)",
                                     "100-year GEV bootstrap 90 % interval, low",
                                     "100-year GEV bootstrap 90 % interval, high"]
    by = {k["label"]: k for k in report["key_numbers"]}
    for t, gev, lp3 in zip(FLOW["ffa"]["return_periods"], FLOW["ffa"]["fits"]["gev_lmoments"]["q"],
                           FLOW["ffa"]["fits"]["lp3"]["q"]):
        assert by[f"{t}-year return level, GEV (L-moments)"]["value"] == gev
        assert by[f"{t}-year return level, Log-Pearson III"]["value"] == lp3
    two = by["2-year return level, GEV (L-moments)"]
    assert two["step"] == "s3" and two["unit"] == "m3/s"
    assert labels.index("Sen's slope") < labels.index("2-year return level, GEV (L-moments)"), "the other T come last"
    assert sum(1 for lab in labels if "interval" in lab) == 4, "the intervals at the headline only"
    md = author.to_markdown(ws)
    assert "| 50-year return level, Log-Pearson III | 500 | m3/s | s3 |" in md


def test_every_drought_timescale_with_its_class(no_deliverables):
    ws = _ran("How dry is it here, is this a drought?", "drought_status", {}, RICH)
    report = author.author_report(ws, None)
    rows = [k for k in report["key_numbers"] if k["step"] == "s1" and k["label"].startswith("SP")]
    assert [k["label"] for k in rows] == [
        "SPI at 12 months, 2026-08 (near normal)", "SPEI at 12 months, 2026-08 (moderately dry)",
        "SPI at 1 months, 2026-08 (near normal)", "SPEI at 1 months, 2026-08 (near normal)",
        "SPI at 3 months, 2026-08 (moderately dry)", "SPEI at 3 months, 2026-08 (moderately dry)"]
    assert [k["value"] for k in rows] == [-0.87, -1.05, -0.42, -0.61, -1.13, -1.36]
    assert rows[1]["class"] == "moderately dry" and rows[0]["class"] == "near normal"
    assert DROUGHT["current"]["spei"]["3"] == -1.36
    assert author._drought_class(-2.5) == "extremely dry" and author._drought_class(1.7) == "very wet"
    assert author._drought_class("x") is None


def test_signature_bands_and_reliability_rows(no_deliverables):
    ws = _ran("What flow can I expect here?", "ungauged_flow", {"statistic": "all"}, UNGAUGED)
    report = author.author_report(ws, None)
    by = {k["label"]: k for k in report["key_numbers"]}
    assert by["Q95"]["value"] == 0.12 and by["Q95 band, low"]["value"] == 0.08 and by["Q95 band, high"]["value"] == 0.2
    assert by["Q95 band, high"]["unit"] == "mm/d" and by["mean flow band, low"]["value"] == 0.5
    assert SIGNATURES["estimates"]["q95_mm"]["high"] == 0.2
    by_year = json.loads(json.dumps(SUPPLY))
    by_year["reliability"]["by_year"] = {"2010": 0.9, "2011": 0.26}
    ws2 = _ran("Can the river supply the town reliably?", "supply_reliability", {"demand_m3s": 2}, RECON,
               tools=fake_tools([], supply_reliability=by_year))
    report2 = author.author_report(ws2, None)
    by2 = {k["label"]: k for k in report2["key_numbers"]}
    assert by2["Days the demand is met"]["value"] == 61 and by2["Years without a shortfall"]["value"] == 20
    assert by2["Volume delivered"]["value"] == 97 and by2["Days short in the worst year (2011)"]["value"] == 270
    assert by2["Days the demand is met in 2010"]["value"] == 90 and by2["Days the demand is met in 2011"]["value"] == 26
    assert by2["Verdict"]["value"] == "reliable"
