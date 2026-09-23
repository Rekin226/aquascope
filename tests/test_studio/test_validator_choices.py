"""The validator refuses a value a tool would refuse, and names the choices (#413)."""

from __future__ import annotations

from aquascope.studio import catalogue
from aquascope.studio.workspace import Workspace


def test_an_invalid_choice_is_rejected_and_the_choices_are_named():
    step = {"id": "s1", "tool": "regionalize_signatures",
            "arguments": {"lat": 51.4, "lon": -0.3, "k": 10, "method": "regionalization"}}
    errors = catalogue.validate_step(step)
    assert errors and "similarity" in errors[0] and "regression" in errors[0] and "both" in errors[0]
    assert step["arguments"]["method"] == "regionalization", "never substituted"
    assert "notes" not in step


def test_every_declared_choice_validates_and_a_reference_is_left_to_the_runner():
    for entry in catalogue.entries().values():
        for arg, schema in entry.arguments.items():
            values = schema.get("enum") if isinstance(schema, dict) else None
            if not values:
                continue
            for v in values:
                args = {k: 1 for k in entry.required if k != "from_step"}
                args[arg] = v
                if "from_step" in entry.required:
                    args["from_step"] = "s0"
                errors = catalogue.validate_step({"id": "sx", "tool": entry.id, "arguments": args}, known_ids={"s0"})
                assert not any(arg in e and "is not one of" in e for e in errors), (entry.id, arg, v, errors)
    step = {"id": "s2", "tool": "regionalize_signatures",
            "arguments": {"lat": 1, "lon": 2, "method": "{{ result.s1.method }}"}}
    assert not [e for e in catalogue.validate_step(step, known_ids={"s1"}) if "is not one of" in e]


def test_the_workbench_tools_declare_their_closed_sets():
    assert catalogue.get("wqi").arguments["use"]["enum"] == ["drinking", "irrigation", "aquatic life"]
    assert catalogue.get("baseflow").arguments["method"]["enum"] == ["lyne_hollick", "eckhardt", "ukih"]
    assert catalogue.get("return_periods").arguments["distribution"]["enum"] == ["gev", "lp3", "gumbel"]
    assert catalogue.get("irrigation").arguments["method"]["enum"] == ["single", "dual"]


def test_a_step_whose_only_fault_is_its_fallback_keeps_its_place_without_the_fallback():
    from aquascope.studio.roles.methodologist import _prune

    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    steps = [
        {"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 51.415, "lon": -0.308}, "rationale": "r"},
        {"id": "s2", "tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": 100}, "rationale": "r",
         "expects": [{"check": "not_empty", "path": "glofas"}],
         "fallback": {"step": {"tool": "regionalize_signatures",
                               "arguments": {"lat": 51.415, "lon": -0.308, "k": 10, "method": "regionalization"},
                               "rationale": "r"}}},
    ]
    kept, notes = _prune(steps, ws)
    assert [s["id"] for s in kept] == ["s1", "s2"] and "fallback" not in kept[1]
    assert any("fallback dropped" in n and "regionalization" in n for n in notes)


def test_the_recorded_kingston_fallback_of_2026_09_07_is_now_refused():
    """The Kingston study recorded on 2026-09-07 carried this fallback into a run and died on the tool's own
    ValueError; the same step is refused by the validator now (the recording has since been redone)."""
    s4 = {"id": "s4", "tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": 100},
          "fallback": {"step": {"tool": "regionalize_signatures",
                                "arguments": {"lat": 51.415, "lon": -0.308, "k": 10, "method": "regionalization"},
                                "rationale": "a regionalized cross-check",
                                "expects": [{"check": "not_empty", "path": "estimates"}]}}}
    errors = catalogue.validate_step(s4)
    assert any(e.startswith("fallback of step s4") and "is not one of" in e for e in errors)


def test_a_spread_gate_written_as_one_comma_string_without_a_value_still_evaluates():
    from aquascope.gates import evaluate

    fits = {"gev_lmoments": {"q": [250, 520]}, "lp3": {"q": [252, 548]}}
    payload = {"ffa": {"return_periods": [2, 100], "fits": fits}}
    gate = {"check": "spread_within", "path": "ffa.fits.gev_lmoments.q, ffa.fits.lp3.q", "return_period": 100}
    out = evaluate([gate], payload)[0]
    assert out["passed"] and "spread 5%" in out["detail"] and "25% allowed" in out["detail"]
    declared = next(g for g in catalogue.get("flood_frequency").gates if g["check"] == "spread_within")
    assert declared["paths"] == ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"]


def test_a_cross_check_gate_without_a_reference_is_pointed_at_the_flood_step_or_dropped():
    steps = [
        {"id": "s1", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "3400TH"},
         "expects": [{"check": "max_return_period_factor", "path": "years", "value": 3, "return_period": 100}]},
        {"id": "s2", "tool": "anywhere", "arguments": {"lat": 51.4, "lon": -0.3},
         "expects": [{"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T", "value": 0.5}]},
        {"id": "s3", "tool": "anywhere", "arguments": {"lat": 51.4, "lon": -0.3},
         "expects": [{"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                      "reference": "{{ result.<the flood_frequency step>.ffa.fits.gev_lmoments.q_by_T }}"}]},
    ]
    assert catalogue.validate_plan(steps) == []
    g2 = steps[1]["expects"][0]
    assert g2["reference"] == "{{ result.s1.ffa.fits.gev_lmoments.q_by_T }}" and g2["return_period"] == 100
    assert steps[1]["depends_on"] == ["s1"]
    assert steps[2]["expects"][0]["reference"] == "{{ result.s1.ffa.fits.gev_lmoments.q_by_T }}"
    alone = [{"id": "s1", "tool": "anywhere", "arguments": {"lat": 51.4, "lon": -0.3},
              "expects": [{"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T"}]}]
    assert catalogue.validate_plan(alone) == [] and alone[0]["expects"] == []
    assert any("dropped" in n for n in alone[0]["notes"])
