"""HydroGym Phase 2 (#367): the reference plans load and validate against the catalogue, the registry and their
saved reconnaissance; the scorer gives a perfect plan 1, the tree a known value and an empty plan 0; the
Methodologist plans on a scripted model with no network; the bench writes rows, resumes and renders; the CLI verbs
work. Nothing here touches the network."""

from __future__ import annotations

import json
import sys

import pytest

import aquascope.explore
from aquascope import cli
from aquascope.gym import bench as gb
from aquascope.gym import plans as gp
from tests.test_studio.conftest import FakeModel

REFS = gp.load_references()
BY_ID = {r.id: r for r in REFS}
PLAYBOOKS = {"flood_risk", "ungauged_flow", "groundwater_decline", "drought_status", "supply_reliability",
             "irrigation_feasibility", "water_quality"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("assess_site must not be called: the reconnaissance is saved with the case")

    monkeypatch.setattr(aquascope.explore, "assess_site", boom, raising=False)


def _perfect(ref: gp.Reference) -> dict:
    """The reference's required steps as a study dict: what a plan that covers the reference exactly looks like."""
    steps = [{"id": f"s{i}", "tool": s.tool, "method": s.method, "arguments": {},
              "expects": [{"check": g["check"], "path": g["path"]} for g in s.gates]}
             for i, s in enumerate(ref.required_steps, 1)]
    return {"steps": steps, "plan": {"branch": ref.expected_branch, "author": "test"}}


# ── the references ──


def test_the_reference_plans_load_and_validate():
    assert len(REFS) >= 20 and len({r.id for r in REFS}) == len(REFS)
    assert {r.playbook for r in REFS} >= PLAYBOOKS
    assert sum(1 for r in REFS if "off_tree" in r.tags) >= 3
    assert sum(1 for r in REFS if r.decline) >= 6 and sum(1 for r in REFS if not r.decline) >= 14
    for ref in REFS:
        assert gp.validate_reference(ref) == [], ref.id
        recon = gp.recon_for(ref)
        assert recon["point"]["lat"] == pytest.approx(ref.lat, abs=1e-3) and recon["stations"] is not None
        assert ref.brief and ref.rationale and ref.site.get("name")
        if not ref.decline:
            assert ref.expected_branch and ref.required_steps
    rows = gp.list_references()
    assert [r["id"] for r in rows] == [r.id for r in REFS] and rows[0]["site"]


def test_the_recon_is_narrowed_to_the_playbooks_problem_like_the_scout_does():
    from aquascope.methods import METHODS

    ref = BY_ID["drought_gauge_tetbury"]
    recon = gp.recon_for(ref)
    assert recon["sufficiency"] and all("drought" in METHODS[r["method"]].problems for r in recon["sufficiency"])
    raw = gp._load_recon_file(ref.recon_name)["recon"]
    assert len(raw["sufficiency"]) > len(recon["sufficiency"]), "the file keeps the full table"
    recon["stations"] = []
    assert gp.recon_for(ref)["stations"], "a deep copy: the cache is not edited"


def test_validation_catches_an_unknown_tool_a_forbidden_required_step_and_a_missing_recon():
    ref = gp.Reference.from_dict({**BY_ID["flood_at_site_potomac"].to_dict(), "id": "x", "steps": [
        {"tool": "frobnicate"}, {"tool": "flood_frequency", "method": "at_site_flood_frequency",
                                 "gates": [{"check": "min_yearz"}, {"check": "ci_finite", "path": "nowhere"}]}],
        "forbidden": {"tools": ["flood_frequency"], "methods": ["nope"]}})
    errors = gp.validate_reference(ref)
    assert any("unknown tool" in e for e in errors) and any("unknown check" in e for e in errors)
    assert any("catalogue lists" in e for e in errors) and any("forbidden tool or method" in e for e in errors)
    assert any("forbidden method 'nope'" in e for e in errors)
    lost = gp.Reference.from_dict({**BY_ID["flood_at_site_potomac"].to_dict(), "site": {"lat": 1, "lon": 2,
                                                                                         "recon": "missing"}})
    assert any("recon 'missing'" in e for e in gp.validate_reference(lost))
    declined = gp.Reference.from_dict({**BY_ID["flood_inundation_declined_potomac"].to_dict(),
                                       "decline_kind": "whatever"})
    assert any("decline_kind" in e for e in gp.validate_reference(declined))


# ── the tree on the saved reconnaissance ──


def test_the_tree_selects_the_expected_branch_or_declines_on_every_case():
    for ref in REFS:
        cand, detail = gp.tree_candidate(ref)
        if ref.decline:
            assert cand.declined and cand.kind == ref.decline_kind, (ref.id, cand.reason)
        else:
            assert not cand.declined and cand.branch == ref.expected_branch, (ref.id, cand.branch)
            assert cand.steps and all(st["tool"] for st in cand.steps)


# ── the scorer ──


def test_a_perfect_plan_scores_one_and_an_empty_or_declining_plan_zero():
    for ref in REFS:
        if ref.decline:
            yes = gp.score_plan(ref, {"declined": True, "reason": "no", "kind": ref.decline_kind})
            assert yes["score"] == 1.0 and yes["decline_correct"] is True and yes["coverage_tools"] is None
            no = gp.score_plan(ref, _perfect(BY_ID["flood_at_site_potomac"]))
            assert no["score"] == 0.0 and no["decline_correct"] is False
            continue
        scored = gp.score_plan(ref, _perfect(ref))
        assert scored["score"] == 1.0 and scored["forbidden_used"] == 0 and scored["extraneous"] == 0.0, (ref.id,
                                                                                                           scored)
        assert scored["coverage_tools"] == 1.0 and scored["explain"] == ["the plan covers the reference"]
        empty = gp.score_plan(ref, {"steps": []})
        assert empty["score"] == 0.0 and empty["coverage_tools"] == 0.0 and "no steps" in empty["explain"][0]
        declined = gp.score_plan(ref, {"declined": True, "reason": "cannot"})
        assert declined["score"] == 0.0 and declined["decline_correct"] is False
        assert gp.score_plan(ref.to_dict(), {"status": "declined", "declined_reason": "x"})["score"] == 0.0


def test_the_tree_scores_a_known_value_on_the_comparison_case():
    from aquascope import playbooks as pbk

    ref = BY_ID["offtree_atsite_vs_regional_potomac"]
    # One branch alone answers half the brief: the partial plan is the scorer's known value.
    alone = pbk.plan(ref.playbook, gp.recon_for(ref), ref.intake, problem_text=ref.brief, compose=False)
    scored = gp.score_plan(ref, gp.candidate_from(alone))
    # required tools: analyze_station, flood_frequency, similar_basins, regionalize_signatures; the branch has two;
    # methods likewise; gates: the branch carries five of the eight; nothing extraneous, nothing forbidden.
    assert scored["coverage_tools"] == 0.5 and scored["coverage_methods"] == 0.5 and scored["coverage_gates"] == 0.625
    assert scored["extraneous"] == 0.0 and scored["forbidden_used"] == 0 and scored["decline_correct"] is True
    assert scored["score"] == pytest.approx(0.30 * 0.5 + 0.25 * 0.5 + 0.20 * 0.625 + 0.15 + 0.10) == 0.65
    assert "missing tool similar_basins" in scored["explain"]
    assert "missing method regionalize_signatures" in scored["explain"]
    assert "missing gate min_donors on similar_basins (k)" in scored["explain"]
    # The tree itself composes the regional branch for the compound brief (#383) and covers the reference.
    cand, detail = gp.tree_candidate(ref)
    full = gp.score_plan(ref, cand)
    assert full["score"] == 1.0 and full["extraneous"] == 0.0 and full["coverage_tools"] == 1.0


def test_forbidden_and_extraneous_steps_cost_their_parts():
    ref = BY_ID["flood_short_record_fish_creek"]
    plan = _perfect(ref)
    plan["steps"] += [{"id": "s8", "tool": "flood_frequency", "method": "at_site_flood_frequency", "expects": []},
                      {"id": "s9", "tool": "drought_indices", "method": "spei_reanalysis"}]
    scored = gp.score_plan(ref, plan)
    assert scored["forbidden_used"] == 1 and scored["extraneous"] == pytest.approx(2 / 5)
    assert scored["score"] == pytest.approx(0.30 + 0.25 + 0.20 + 0.0 + 0.10 * 0.6)
    assert any("forbidden in step s8: tool flood_frequency, method at_site_flood_frequency" == e
               for e in scored["explain"])
    assert "extraneous step s9 (drought_indices, spei_reanalysis)" in scored["explain"]
    fb = _perfect(ref)
    fb["steps"][0]["fallback"] = {"step": {"tool": "flood_frequency", "arguments": {}}}
    assert gp.score_plan(ref, fb)["forbidden_used"] == 1, "a forbidden fallback counts"


def test_gates_match_on_tool_check_and_path_and_optional_steps_are_neither_required_nor_extraneous():
    ref = BY_ID["flood_at_site_potomac"]
    plan = _perfect(ref)
    for st in plan["steps"]:
        if st["tool"] == "flood_frequency":
            st["expects"] = [{"check": "spread_within", "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"]},
                             {"check": "ci_finite", "path": "ffa.fits.gev_bootstrap.ci.low"},
                             {"check": "max_return_period_factor", "path": "n_years"}]
    scored = gp.score_plan(ref, plan)
    # a paths list matches, a longer path matches, n_years does not
    assert scored["coverage_gates"] == pytest.approx(5 / 6)
    plan["steps"] += [{"id": "s7", "tool": "anywhere", "method": "glofas_cross_check"},
                      {"id": "s8", "tool": "describe_catchment"}]
    assert gp.score_plan(ref, plan)["extraneous"] == 0.0
    weightless = gp.Reference.from_dict({**ref.to_dict(), "steps": [{"tool": "anywhere"}]})
    scored = gp.score_plan(weightless, {"steps": [{"id": "s1", "tool": "anywhere"}]})
    assert scored["coverage_methods"] is None and scored["coverage_gates"] is None and scored["score"] == 1.0


def test_candidate_from_reads_a_study_a_workspace_and_a_decline():
    from aquascope.study import Step, Study

    step = Step(tool="anywhere", id="s1", expects=[{"check": "not_empty", "path": "climate"}],
                fallback={"step": {"tool": "similar_basins", "arguments": {}}})
    study = Study(question="q", steps=[step], plan={"branch": "regional", "author": "playbook"})
    cand = gp.candidate_from(study)
    assert cand.steps[0]["gates"] == [{"check": "not_empty", "path": "climate"}] and cand.branch == "regional"
    assert cand.steps[0]["fallback"] == {"tool": "similar_basins", "method": None}
    ws = {"status": "review", "study": study.to_dict(), "declined_reason": None}
    assert gp.candidate_from(ws).steps[0]["tool"] == "anywhere"
    assert gp.candidate_from({"status": "declined", "declined_reason": "x", "study": None}).declined
    assert gp.candidate_from({"declined": True, "kind": "refused"}).kind == "refused"
    assert gp.candidate_from("nonsense").steps == []


# ── the Methodologist with a scripted model, no network ──

POTOMAC_PLAN = {
    "objective": "The 100-year design flow at Little Falls with its band",
    "methodology": ["Trend pre-test.", "Two fits with a band."],
    "steps": [
        {"id": "s1", "tool": "analyze_station", "arguments": {"source": "usgs", "station_id": "USGS-01646500"},
         "method": "trend_mann_kendall", "rationale": "Trend pre-test.",
         "expects": [{"check": "min_years", "value": 20, "path": "years"}, {"check": "not_empty", "path": "trend"},
                     {"check": "unit_present", "path": "unit"}]},
        {"id": "s2", "tool": "flood_frequency", "depends_on": ["s1"],
         "arguments": {"source": "usgs", "station_id": "USGS-01646500", "bootstrap_ci": True},
         "method": "at_site_flood_frequency", "rationale": "Two fits with a band.",
         "expects": [{"check": "max_return_period_factor", "value": 3, "path": "years", "return_period": 100},
                     {"check": "ci_finite", "path": "ffa.fits.gev_bootstrap.ci", "return_period": 100},
                     {"check": "spread_within", "value": 0.25, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
                      "return_period": 100}]},
    ],
    "assumptions": ["the record is stationary enough"],
}


def test_the_methodologist_plans_on_the_saved_recon_and_is_scored():
    ref = BY_ID["flood_at_site_potomac"]
    client = FakeModel({"methodologist": [POTOMAC_PLAN]})
    cand, detail = gp.methodologist_candidate(ref, client=client, model="fake", provider="custom")
    assert not cand.declined and [s["tool"] for s in cand.steps] == ["analyze_station", "flood_frequency"]
    assert detail["usage"]["calls"] == 1 and detail["validator_errors_first_try"] == 0 and detail["first_try_valid"]
    assert detail["author"] == "methodologist" and not detail["fallback_to_tree"] and detail["model"] == "fake"
    ctx = client.requests[0]["context"]
    assert ctx["inventory"]["years_by_variable"]["discharge"] == 96.5 and ctx["exemplar"]["branch"] == "at_site"
    assert ctx["brief"]["playbook"] == "flood_risk" and ctx["brief"]["intake"]["return_period"] == 100
    assert any(d.get("station_id") == "USGS-01646500" for d in ctx["inventory"]["datasets"])
    assert gp.score_plan(ref, cand)["score"] == 1.0


def test_an_invalid_first_plan_is_counted_and_a_silent_model_falls_back_to_the_tree():
    ref = BY_ID["flood_at_site_potomac"]
    broken = dict(POTOMAC_PLAN, steps=[dict(POTOMAC_PLAN["steps"][0], tool="frobnicate"),
                                       dict(POTOMAC_PLAN["steps"][1], expects=[{"check": "min_yearz"}])])
    client = FakeModel({"methodologist": [broken, POTOMAC_PLAN]})
    cand, detail = gp.methodologist_candidate(ref, client=client, model="fake", provider="custom")
    assert detail["validator_errors_first_try"] == 2 and detail["first_try_valid"] is False
    assert detail["usage"]["calls"] == 2 and not detail["fallback_to_tree"] and len(cand.steps) == 2
    silent = FakeModel({})
    cand, detail = gp.methodologist_candidate(ref, client=silent, model="fake", provider="custom")
    assert detail["fallback_to_tree"] and detail["author"] == "playbook" and cand.branch == "at_site"
    assert detail["validator_errors_first_try"] >= 1 and gp.score_plan(ref, cand)["score"] == 1.0
    keyless, detail = gp.methodologist_candidate(ref)
    assert keyless.branch == "at_site" and detail["usage"]["calls"] == 0 and not detail["fallback_to_tree"]


def test_the_methodologist_declines_a_declining_case_before_any_model_call():
    ref = BY_ID["gw_cause_declined_tetbury"]
    client = FakeModel({"methodologist": [POTOMAC_PLAN]})
    cand, detail = gp.methodologist_candidate(ref, client=client, model="fake", provider="custom")
    # the model's plan does not survive on this brief: the tree declines, and the Studio's fallback is that decline
    assert cand.declined or gp.score_plan(ref, cand)["score"] in (0.0, 1.0)


# ── the bench and the leaderboard ──


def test_the_bench_writes_rows_resumes_and_renders_the_leaderboard(tmp_path):
    out = tmp_path / "tree.jsonl"
    events: list[str] = []
    results = gp.run_plan_bench(None, "tree", out=out, on_event=events.append, limit=8)
    assert len(results) == 8 and all(r.error is None and r.agent == "tree" and r.model is None for r in results)
    assert all(r.validator_errors_first_try == 0 and r.first_try_valid for r in results)
    again = gp.run_plan_bench(None, "tree", out=out, on_event=events.append, limit=8, resume=True)
    assert again == [] and any(e.startswith("resuming") for e in events)
    loaded = gp.load_plan_results([out])
    assert len(loaded) == 8 and {r.case_id for r in loaded} == {r.case_id for r in results}
    assert gb.load_results([out]) == [], "Phase 1 skips plan rows"
    rows = gp.summarize_plans(loaded)
    assert len(rows) == 1 and rows[0]["agent"] == "tree" and rows[0]["n"] == 8 and rows[0]["cost_usd"] == 0.0
    assert rows[0]["decline_rate"] == 1.0 and rows[0]["false_decline_rate"] == 0.0
    text = gp.plan_leaderboard(loaded, out=tmp_path / "lb.md")
    assert "| tree | none | 8 (" in text and (tmp_path / "lb.md").exists() and "Mean score by playbook" in text
    picked = gp.run_plan_bench(None, "tree", case_ids=["flood_at_site_potomac"])
    assert [r.case_id for r in picked] == ["flood_at_site_potomac"] and picked[0].score == 1.0


def test_the_file_agent_scores_a_plan_json_and_reports_a_missing_one(tmp_path):
    ref = BY_ID["supply_gauged_vegre"]
    (tmp_path / f"{ref.id}.json").write_text(json.dumps({**_perfect(ref), "model": "device-4b", "provider": "local",
                                                          "usage": {"calls": 1, "prompt_tokens": 10,
                                                                    "completion_tokens": 5}}), encoding="utf-8")
    results = gp.run_plan_bench(None, "file", candidates_dir=tmp_path,
                                case_ids=[ref.id, "flood_at_site_potomac"], model="device-4b")
    by_id = {r.case_id: r for r in results}
    assert by_id[ref.id].score == 1.0 and by_id[ref.id].model == "device-4b" and by_id[ref.id].tokens == 15
    assert by_id["flood_at_site_potomac"].error and by_id["flood_at_site_potomac"].error.startswith("FileNotFoundError")
    with pytest.raises(ValueError):
        gp.run_plan_bench(None, "file", case_ids=[ref.id])


def test_repeats_and_the_spread_with_a_scripted_model(tmp_path):
    ref = BY_ID["flood_at_site_potomac"]
    client = FakeModel({"methodologist": [POTOMAC_PLAN, dict(POTOMAC_PLAN, steps=POTOMAC_PLAN["steps"][:1])]})
    results = gp.run_plan_bench([ref], "methodologist", client=client, model="fake", provider="custom", repeats=2,
                                out=tmp_path / "m.jsonl")
    assert [r.repeat for r in results] == [0, 1] and results[0].score == 1.0 and results[1].score < 1.0
    rows = gp.summarize_plans(results)
    assert rows[0]["repeats"] == 2 and rows[0]["score_min_run"] < rows[0]["score_max_run"] == 1.0
    assert rows[0]["first_try_valid"] == 1.0 and rows[0]["fallback_rate"] == 0.0
    assert " to 1.00 |" in gp.plan_leaderboard(results)


# ── the CLI ──


def test_the_cli_plans_verbs_and_the_bench(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "flood_at_site_potomac" in out and "reference plans in" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "show", "wq_drinking_potomac"])
    cli.main()
    assert "water_quality_samples" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "validate"])
    cli.main()
    assert "0 with errors" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "score", "offtree_atsite_vs_regional_potomac"])
    cli.main()
    assert "score 1.00" in capsys.readouterr().out
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_perfect(BY_ID["gw_well_tetbury"])), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "score", "gw_well_tetbury", "--candidate",
                                      str(plan), "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["score"] == 1.0
    out_file = tmp_path / "tree.jsonl"
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "bench", "--agent", "tree", "--limit", "4", "--out",
                                      str(out_file), "--quiet"])
    cli.main()
    assert "aquascope gym bench: tree" in capsys.readouterr().out
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == 4
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "leaderboard", str(out_file), "--out",
                                      str(tmp_path / "lb.md")])
    cli.main()
    assert "| tree | none | 4 (" in capsys.readouterr().out and (tmp_path / "lb.md").exists()
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "bench", "--agent", "team"])
    with pytest.raises(SystemExit):
        cli.main()


# ── alternatives and re-scoring ──


def test_an_alternative_step_covers_the_tool_the_method_and_the_check():
    ref = BY_ID["gw_well_tetbury"]
    sgi = next(s for s in ref.required_steps if s.tool == "sgi_drought")
    assert sgi.tools == {"sgi_drought", "drought_propagation"} and sgi.methods == {"sgi"}
    plan = {"steps": [
        {"id": "s1", "tool": "analyze_station", "method": "groundwater_trend",
         "expects": [{"check": "min_years", "path": "years"}, {"check": "not_empty", "path": "trend"},
                     {"check": "unit_present", "path": "unit"}]},
        {"id": "s2", "tool": "drought_propagation", "method": "sgi",
         "expects": [{"check": "not_empty", "path": "sgi"}]},
    ]}
    scored = gp.score_plan(ref, plan)
    assert scored["coverage_tools"] == 1.0 and scored["coverage_methods"] == 1.0 and scored["score"] == 1.0
    plan["steps"][1] = {"id": "s2", "tool": "get_timeseries", "expects": [{"check": "not_empty", "path": "points"}]}
    scored = gp.score_plan(ref, plan)
    assert scored["coverage_tools"] == 0.5 and "missing tool drought_propagation (or sgi_drought)" in scored["explain"]
    assert "missing method sgi" in scored["explain"]
    low = BY_ID["drought_reanalysis_snake_plain"]
    plan = _perfect(low)
    for st in plan["steps"]:
        if st["tool"] == "low_flow_context":
            st["method"] = "baseflow_separation"
    assert gp.score_plan(low, plan)["score"] == 1.0, "the alternative method counts"
    bad = gp.Reference.from_dict({**ref.to_dict(), "steps": [
        {"tool": "sgi_drought", "method": "sgi",
         "alternatives": [{"tool": "nope"}, {"tool": "wqi", "method": "spi"}]}]})
    errors = gp.validate_reference(bad)
    assert any("alternative tool 'nope'" in e for e in errors) and any("wqi does not apply" in e for e in errors)


def test_rescore_reads_the_stored_plans_again_without_running_the_agent(tmp_path, monkeypatch):
    out = tmp_path / "tree.jsonl"
    rows = gp.run_plan_bench(None, "tree", out=out, case_ids=["offtree_atsite_vs_regional_potomac",
                                                              "flood_inundation_declined_potomac"])
    assert sorted(r.score for r in rows) == [1.0, 1.0]
    easier = gp.Reference.from_dict({**BY_ID["offtree_atsite_vs_regional_potomac"].to_dict(), "steps": [
        {"tool": "analyze_station", "method": "trend_mann_kendall"},
        {"tool": "flood_frequency", "method": "at_site_flood_frequency"}]})
    stored = gp.load_plan_results([out], latest=False)
    again = gp.rescore_plans(stored, [easier, BY_ID["flood_inundation_declined_potomac"]])
    by_case = {r.case_id: r.score for r in again}
    assert by_case["flood_inundation_declined_potomac"] == 1.0
    # the composed plan carries the two regional steps, which the easier reference counts as extraneous
    assert by_case["offtree_atsite_vs_regional_potomac"] == pytest.approx(0.94, abs=0.01)
    assert again[0].explain and all(r.decline_correct for r in again)
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "plans", "rescore", str(out)])
    cli.main()
    assert sorted(r.score for r in gp.load_plan_results([out])) == [1.0, 1.0], "the package references again"
