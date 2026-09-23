"""The Methodologist: the plan, a version-3 study composed for the data at hand.

Keyless, the plan is the playbook tree's (``playbooks.plan``), promoted to
version 3 with an objective, a methodology (one sentence per step) and the
brief's assumptions. With a model, one call composes the steps from the
catalogue, given the brief, the inventory, the sufficiency table, the gate
vocabulary and the tree's own plan as an exemplar; the plan then passes
:func:`aquascope.studio.catalogue.validate_plan` (the tool exists, the
arguments are its own, the gates are known, a method the registry calls not
defensible here is refused). Errors get one repair call; a plan that still
fails falls back to the tree when a playbook applies and is declined
otherwise, with the errors listed. A plan a caller's own model wrote (the
Explorer's on-device model) goes through :func:`adopt`, the same validator,
repair by pruning and fall-back to the tree. A follow-up after the report
goes through :func:`change`: keyless, a rule table maps the request to
catalogue steps appended to the plan (another gauge, the donors, the flow
duration curve, a trend, the drought indices, baseflow, the ERA5 cell);
with a model, the steps it writes. Nothing runs here.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from aquascope.studio import catalogue
from aquascope.studio.model import Model
from aquascope.studio.prompts import METHODOLOGIST, METHODOLOGIST_CHANGE, METHODOLOGIST_REPAIR
from aquascope.studio.workspace import Workspace
from aquascope.study import Step, Study

__all__ = ["FOLLOW_UP_RULES", "adopt", "change", "change_context", "plan", "plan_context", "plan_text", "revise",
           "sufficiency_for_validation", "wants_upload"]

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]+)\.")
_STEP_IN_ERROR = re.compile(r"^(?:fallback of )?step ([^:]+):")
_UPLOAD_WORDS = re.compile(
    r"\b(my|own|this|these|attached|uploaded)\s+(?:\w+\s+){0,2}(record|data|file|table|upload|series|csv)\b|"
    r"\b(upload|csv)\b", re.I,
)
MIN_STEPS, MAX_STEPS = 1, 12
#: Steps that frame a study but are not an analysis on their own.
_FRAMING_TOOLS = frozenset({"describe_catchment", "find_stations", "assess_site", "load_table", "eda", "quality"})
#: The workbench step per problem kind over an attached table (F): tool, method, the key its gate checks.
_UPLOAD_STEPS: dict[str, list[tuple[str, str | None, str]]] = {
    "flood_risk": [("return_periods", "at_site_flood_frequency", "return_levels"),
                   ("flow_duration", "flow_duration", "percentiles")],
    "supply_reliability": [("flow_duration", "flow_duration", "percentiles"), ("signatures", None, "signatures"),
                           ("baseflow", "baseflow_separation", "bfi")],
    "ungauged_flow": [("flow_duration", "flow_duration", "percentiles"), ("signatures", None, "signatures"),
                      ("baseflow", "baseflow_separation", "bfi")],
    "groundwater_decline": [("sgi_drought", "sgi", "current"), ("recharge", "recharge_wtf", "value_mm_per_year")],
    "water_quality": [("who_screen", None, "rows"), ("wqi", "water_quality_index", "ccme")],
}
_GENERIC_UPLOAD_STEPS: list[tuple[str, str | None, str]] = [("eda", None, "n_records"), ("quality", None, "n_records"),
                                                          ("signatures", None, "signatures")]


# ── helpers ──────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _first_sentence(text: str | None) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return m.group(1) if m else text


def _outputs_for(step: Step) -> list[dict[str, Any]]:
    entry = catalogue.get(step.tool)
    if entry is None:
        return []
    sid = step.id or step.tool
    out = [{"kind": "figure", "id": f"{sid}_{f}", "caption": f"{f.replace('_', ' ')} from {step.tool}"}
           for f in entry.figures]
    out += [{"kind": "table", "id": f"{sid}_{t}", "caption": f"{t.replace('_', ' ')} from {step.tool}"}
            for t in entry.tables]
    return out


def _playbook(ws: Workspace) -> Any | None:
    from aquascope import playbooks as pbk

    pid = ws.brief.playbook
    if not pid:
        return None
    try:
        return pbk.load(pid)
    except pbk.PlaybookError:
        return None


def _recon(ws: Workspace) -> dict[str, Any]:
    return dict(ws.inventory.recon) if ws.inventory else {"point": dict(ws.site or {}), "stations": [],
                                                           "context": {"years_by_variable": {}}, "sufficiency": []}


def _promote(ws: Workspace, study: Study, *, author: str) -> Study:
    """Make a study version 3: objective, methodology, assumptions and outputs on every step."""
    b = ws.brief
    study.version = 3
    study.plan = dict(study.plan or {})
    study.plan["author"] = author
    study.plan.setdefault("objective", _first_sentence(b.problem) or b.decision)
    study.plan.setdefault("decision", b.decision)
    study.plan.setdefault("methodology", [_first_sentence(s.rationale) or f"{s.tool}" for s in study.steps])
    study.plan.setdefault("assumptions", list(b.assumptions))
    for s in study.steps:
        if not s.outputs:
            s.outputs = _outputs_for(s)
    if b.problem:
        study.question = b.problem
    if study.problem is not None:
        study.problem.setdefault("text", b.problem)
    return study


def _tree(ws: Workspace, *, branch: str | None = None) -> Study:
    """The playbook tree's plan for this site, promoted to version 3. Raises ``playbooks.Declined``."""
    from aquascope import playbooks as pbk

    pb = _playbook(ws)
    if pb is None:
        raise pbk.Declined("no playbook applies", kind="no_playbook")
    study = pbk.plan(pb, _recon(ws), dict(ws.brief.intake), branch=branch, problem_text=ws.brief.problem)
    study = _promote(ws, study, author="playbook")
    ws.brief.intake = dict((study.problem or {}).get("params") or ws.brief.intake)
    return study


def _caveats_and_citations(ws: Workspace) -> tuple[list[str], list[str]]:
    from aquascope import playbooks as pbk

    pb = _playbook(ws)
    if pb is None:
        return [], []
    try:
        ctx = pbk.evaluation_context(pb, _recon(ws), dict(ws.brief.intake))
        return pbk.caveats_for(pb, ctx), list(pb.citations)
    except (pbk.Declined, pbk.PlaybookError):
        return [], list(pb.citations)


def _uploads(ws: Workspace) -> list[Any]:
    inv = ws.inventory
    return [d for d in (inv.uploads() if inv else []) if (d.quality or {}).get("verdict") != "unreadable"]


def sufficiency_for_validation(ws: Workspace) -> list[dict[str, Any]] | None:
    """The registry's verdicts the validator judges a ``method`` against: the site's table, and for a method whose
    variable an attached table carries, the better of the site's verdict and the table's own record."""
    from aquascope.methods import METHODS, SiteContext, assess_method

    inv = ws.inventory
    if inv is None:
        return None
    rows = {r.get("method"): dict(r) for r in inv.sufficiency if isinstance(r, dict)}
    order = {"defensible": 0, "marginal": 1, "not_defensible": 2, "not defensible": 2}
    rp = ws.brief.intake.get("return_period")
    for d in _uploads(ws):
        if not d.variable or not d.years:
            continue
        ctx = SiteContext(years_by_variable={d.variable: float(d.years)},
                          resolution_by_variable={d.variable: "monthly" if d.resolution == "monthly" else "daily"},
                          return_period=float(rp) if isinstance(rp, (int, float)) else None)
        for m in METHODS.values():
            if m.variable != d.variable:
                continue
            verdict = assess_method(m, ctx)
            verdict["station"] = {"source": "upload", "station_id": d.id}
            old = rows.get(m.id)
            if old is None or order.get(str(verdict["status"]), 2) < order.get(str(old.get("status")), 2):
                rows[m.id] = verdict
    return list(rows.values())


#: Table tools that describe a table rather than analyse it: listed only when the client attached one.
_GENERIC_TABLE_TOOLS = frozenset({"eda", "quality", "preprocess", "insights"})


def _catalogue_for(problem: str | None, *, uploads: bool) -> list[dict[str, Any]]:
    """The catalogue as the Methodologist reads it: the entries that serve the problem, the generic site and
    station tools, the table tools (a step such as water_quality_samples, get_timeseries or load_table feeds
    them with from_step), ``about`` cut short. Tokens matter."""
    from aquascope.methods import METHODS

    rows: list[dict[str, Any]] = []
    for row in catalogue.compact(problem):
        entry = catalogue.get(row["tool"])
        if entry is None or entry.kind == "recon" or entry.id == "find_stations":
            continue
        if not uploads and (entry.kind == "none" or entry.id in _GENERIC_TABLE_TOOLS):
            continue
        if entry.methods and problem:
            serves = any(problem in METHODS[m].problems for m in entry.methods if m in METHODS)
            if not serves and entry.kind in ("station", "site"):
                continue
        row = dict(row)
        row["about"] = row["about"][:110]
        rows.append(row)
    return rows


def _inventory_compact(ws: Workspace) -> dict[str, Any]:
    inv = ws.inventory
    if inv is None:
        return {}
    keep = ("id", "kind", "variable", "source", "station_id", "name", "distance_km", "years", "resolution", "n")
    datasets = []
    for d in inv.datasets:
        row = {k: v for k, v in d.to_dict().items() if k in keep}
        if d.quality:
            row["quality"] = d.quality.get("verdict")
            if d.quality.get("unit"):
                row["unit"] = d.quality["unit"]
            if d.quality.get("columns"):
                row["columns"] = d.quality["columns"][:12]
        datasets.append(row)
    catchment = inv.catchment or {}
    return {
        "site": inv.site,
        "datasets": datasets,
        "years_by_variable": inv.years_by_variable,
        "catchment": {k: catchment.get(k) for k in ("upstream_area_km2", "area_km2", "dams", "sub_basin")
                      if catchment.get(k) is not None},
        "donors": inv.donors,
        "sufficiency": [{k: r.get(k) for k in ("method", "status", "reason")} for r in inv.sufficiency],
        "notes": inv.notes[:6],
    }


def _brief_compact(ws: Workspace) -> dict[str, Any]:
    b = ws.brief
    return {k: v for k, v in b.to_dict().items()
            if k in ("problem", "decision", "quantities", "period", "horizon", "constraints", "kind", "playbook",
                     "intake", "assumptions") and v not in (None, [], {})}


def _placeholder_errors(step: dict[str, Any]) -> list[str]:
    """Placeholders other than ``{{ result.<id>.<path> }}`` in the arguments, the gates or the fallback."""
    errors: list[str] = []
    sid = step.get("id") or "?"
    stack: list[tuple[str, Any]] = [("arguments", step.get("arguments")), ("expects", step.get("expects")),
                                    ("fallback", step.get("fallback"))]
    while stack:
        where, item = stack.pop()
        if isinstance(item, str):
            for m in _PLACEHOLDER.finditer(item):
                if m.group(1) != "result":
                    errors.append(f"step {sid}: {where} carries the placeholder '{{{{ {m.group(1)}.… }}}}'; write "
                                  "the concrete value from the inventory")
        elif isinstance(item, dict):
            stack.extend((where, v) for v in item.values())
        elif isinstance(item, list):
            stack.extend((where, v) for v in item)
    return errors


def _normalise(steps: Any) -> list[dict[str, Any]]:
    """Steps as the runner reads them: ids, mappings and lists where the shape demands them."""
    out: list[dict[str, Any]] = []
    if not isinstance(steps, list):
        return out
    for i, raw in enumerate(steps, 1):
        if not isinstance(raw, dict):
            continue
        step: dict[str, Any] = {
            "id": str(raw.get("id") or f"s{i}"),
            "tool": str(raw.get("tool") or ""),
            "arguments": dict(raw["arguments"]) if isinstance(raw.get("arguments"), dict) else {},
            "rationale": str(raw.get("rationale") or ""),
            "expects": [g for g in (raw.get("expects") or []) if isinstance(g, dict)],
            "depends_on": [str(d) for d in (raw.get("depends_on") or []) if d],
            "outputs": [o for o in (raw.get("outputs") or []) if isinstance(o, dict) and o.get("id")],
        }
        if raw.get("method"):
            step["method"] = str(raw["method"])
        fb = raw.get("fallback")
        if isinstance(fb, dict) and isinstance(fb.get("step"), dict):
            fstep = fb["step"]
            step["fallback"] = {"step": {"tool": str(fstep.get("tool") or ""),
                                         "arguments": dict(fstep.get("arguments") or {}),
                                         "rationale": str(fstep.get("rationale") or ""),
                                         "expects": [g for g in (fstep.get("expects") or []) if isinstance(g, dict)]}}
        elif fb == "stop":
            step["fallback"] = "stop"
        out.append(step)
    return out


def _unwrap(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    """A repair reply that echoes the context's shape (``{"plan": {...}}``) is the plan inside it."""
    if isinstance(obj, dict) and "steps" not in obj and isinstance(obj.get("plan"), dict):
        return obj["plan"]
    return obj


def _fix_methods(steps: list[dict[str, Any]], kind: str | None) -> list[str]:
    """A ``method`` the tool does not apply is not fatal: it becomes the entry's first method that serves the
    problem, or goes; one note per change (A)."""
    from aquascope.methods import METHODS

    notes: list[str] = []
    for step in steps:
        method = step.get("method")
        if not method:
            continue
        entry = catalogue.get(str(step.get("tool") or ""))
        if entry is None:
            continue
        fits = method in METHODS and (not entry.methods or method in entry.methods)
        if fits:
            continue
        replacement = next((m for m in entry.methods if m in METHODS and (not kind or kind in METHODS[m].problems)),
                           None)
        if replacement:
            step["method"] = replacement
            notes.append(f"step {step.get('id')}: method {method!r} is not one {entry.id} applies; "
                         f"{replacement!r} stands in")
        else:
            step.pop("method", None)
            notes.append(f"step {step.get('id')}: method {method!r} is not one {entry.id} applies; dropped")
    return notes


def _fix_arguments(steps: list[dict[str, Any]]) -> list[str]:
    """A ``return_period`` (singular) argument on a tool that takes ``return_periods`` (or ``periods``) is the
    return period asked, not a mistake worth the step: it lands in the list, with a note. Other arguments the
    tool does not take are left for the validator."""
    notes: list[str] = []
    for step in steps:
        args = step.get("arguments")
        entry = catalogue.get(str(step.get("tool") or ""))
        if not isinstance(args, dict) or entry is None or "return_period" not in args:
            continue
        if "return_period" in entry.arguments:
            continue
        target = next((k for k in ("return_periods", "periods") if k in entry.arguments), None)
        if target is None:
            continue
        value = args.pop("return_period")
        try:
            number = float(value)
        except (TypeError, ValueError):
            notes.append(f"step {step.get('id')}: return_period {value!r} is not a number; dropped")
            continue
        current = args.get(target) if isinstance(args.get(target), list) else []
        merged = sorted({float(x) for x in current if isinstance(x, (int, float))} | {number})
        args[target] = [int(x) if x.is_integer() else x for x in merged]
        notes.append(f"step {step.get('id')}: return_period {value!r} became {target}={args[target]}")
    return notes


def _stations_of(ws: Workspace) -> set[tuple[str, str]] | None:
    """The (source, station_id) pairs the inventory knows, uploads included; None when there is no inventory."""
    inv = ws.inventory
    if inv is None or not inv.datasets:
        return None
    return {(str(d.source), str(d.station_id)) for d in inv.datasets if d.source and d.station_id}


def _errors_of(steps: list[dict[str, Any]], ws: Workspace) -> list[str]:
    errors = catalogue.validate_plan(steps, sufficiency=sufficiency_for_validation(ws), stations=_stations_of(ws))
    for st in steps:
        errors += _placeholder_errors(st)
    return errors


def _check(obj: dict[str, Any] | None, ws: Workspace) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """The steps, the validator's errors and the notes (methods repaired) for a model's plan object."""
    obj = _unwrap(obj)
    if not obj:
        return [], ["the model returned no plan"], []
    steps = _normalise(obj.get("steps"))
    if not steps:
        return [], ["the plan has no steps"], []
    if len(steps) > MAX_STEPS:
        return steps, [f"the plan has {len(steps)} steps; at most {MAX_STEPS}"], []
    notes = [*_fix_arguments(steps), *_fix_methods(steps, ws.brief.kind)]
    return steps, _errors_of(steps, ws), notes


def _has_analysis(steps: list[dict[str, Any]]) -> bool:
    return any(str(st.get("tool")) not in _FRAMING_TOOLS for st in steps)


def _prune(steps: list[dict[str, Any]], ws: Workspace) -> tuple[list[dict[str, Any]], list[str]]:
    """The plan minus the steps the validator names (and the steps that then lose what they depend on),
    until what remains validates. Returns the kept steps and one note per removed step."""
    kept = [dict(st) for st in steps]
    notes: list[str] = []
    for _ in range(len(steps) + 1):
        errors = _errors_of(kept, ws)
        if not errors:
            break
        bad: dict[str, str] = {}
        only_fallback: dict[str, str] = {}
        for e in errors:
            m = _STEP_IN_ERROR.match(e)
            if not m:
                continue
            if e.startswith("fallback of "):
                only_fallback.setdefault(m.group(1), e)
            else:
                bad.setdefault(m.group(1), e)
        # A step whose only fault is its fallback keeps its place and loses the fallback (#413).
        stripped = {sid: e for sid, e in only_fallback.items() if sid not in bad}
        for st in kept:
            if str(st.get("id")) in stripped and st.get("fallback") is not None:
                st.pop("fallback", None)
                notes.append(f"step {st.get('id')}: fallback dropped: {stripped[str(st.get('id'))]}")
        if stripped and not bad:
            continue
        if not bad:
            return [], notes + errors
        notes += [f"step {sid} removed: {reason}" for sid, reason in bad.items()]
        kept = [st for st in kept if str(st.get("id")) not in bad]
    return kept, notes


def _study_from(ws: Workspace, obj: dict[str, Any], steps: list[dict[str, Any]], *,
                base: Study | None = None, notes: list[str] | None = None) -> Study:
    """A version-3 study from the model's plan object (``base`` keeps an earlier plan's block on a change)."""
    b = ws.brief
    site = dict(ws.site or {})
    caveats, citations = _caveats_and_citations(ws)
    model_cites = [str(c) for c in (obj.get("citations") or []) if isinstance(c, str)]
    plan_block: dict[str, Any] = dict(base.plan or {}) if base else {}
    plan_block.update({
        "author": "methodologist",
        "playbook": b.playbook,
        "objective": str(obj.get("objective") or plan_block.get("objective") or b.decision
                         or _first_sentence(b.problem)),
        "decision": str(obj.get("decision") or plan_block.get("decision") or b.decision or ""),
        "methodology": [str(m) for m in (obj.get("methodology") or [])] or [_first_sentence(s["rationale"])
                                                                             for s in steps],
        "assumptions": list(dict.fromkeys([*b.assumptions, *[str(a) for a in (obj.get("assumptions") or [])]])),
        "alternatives": obj.get("alternatives") if isinstance(obj.get("alternatives"), list)
        else plan_block.get("alternatives") or [],
        "limitations_expected": [str(x) for x in (obj.get("limitations_expected") or [])]
        or plan_block.get("limitations_expected") or [],
        "citations": list(dict.fromkeys([*citations, *model_cites])),
        "caveats": caveats,
    })
    plan_block["rationale"] = plan_block["objective"]
    if notes:
        plan_block["notes"] = list(dict.fromkeys([*(plan_block.get("notes") or []), *notes]))
    if ws.inventory and ws.inventory.notes:
        plan_block["recon_notes"] = list(ws.inventory.notes[:8])
    where = f"{site.get('lat')}, {site.get('lon')}" if site else "the site"
    study = Study(
        question=b.problem, title=str(obj.get("title") or f"{plan_block['objective'][:60]}: {where}"),
        steps=[Step.from_dict(s) for s in steps], author="methodologist", model=ws.model, version=3,
        problem={k: v for k, v in {"kind": b.kind, "site": site, "params": dict(b.intake), "text": b.problem}.items()
                 if v is not None},
        plan=plan_block, created=_now(),
    )
    for s in study.steps:
        if not s.outputs:
            s.outputs = _outputs_for(s)
    return study


def _uploads_compact(ws: Workspace) -> list[dict[str, Any]]:
    out = []
    for d in _uploads(ws):
        q = d.quality or {}
        out.append({"id": d.id, "variable": d.variable, "unit": q.get("unit"), "years": d.years, "n": d.n,
                    "resolution": d.resolution, "columns": (q.get("columns") or [])[:12], "quality": q.get("verdict")})
    return out


def plan_context(ws: Workspace) -> tuple[str, dict[str, Any]]:
    """The system prompt and the context the Methodologist sends a model to compose the plan: the uploads, the
    brief, the inventory, the catalogue cut to the problem, the gate vocabulary and the tree's plan as an
    exemplar. A page runs the same prompt on a device model and hands the plan back through :func:`adopt`."""
    from aquascope import playbooks as pbk
    from aquascope.gates import CHECKS

    exemplar: dict[str, Any] | None = None
    try:
        tree = _upload_tree(ws) if wants_upload(ws) else _tree(ws)
        exemplar = {"playbook": tree.plan.get("playbook"), "branch": tree.plan.get("branch"),
                    "rationale": tree.plan.get("rationale"),
                    "steps": [st.to_dict() for st in tree.steps], "caveats": tree.plan.get("caveats")}
    except pbk.Declined as exc:
        exemplar = {"playbook": ws.brief.playbook, "declined": exc.reason}
    except pbk.PlaybookError:
        exemplar = None
    uploads = _uploads_compact(ws)
    return METHODOLOGIST, {
        "uploads": uploads or None,
        "brief": _brief_compact(ws),
        "inventory": _inventory_compact(ws),
        "catalogue": _catalogue_for(ws.brief.kind, uploads=bool(uploads)),
        "gates": CHECKS,
        "exemplar": exemplar,
    }


def change_context(ws: Workspace, request: str) -> tuple[str, dict[str, Any]]:
    """The system prompt and the context the Methodologist sends a model for a change after the report."""
    from aquascope.gates import CHECKS

    base = ws.study_obj()
    results = {r.get("id"): {"ok": r.get("ok"),
                             "gates": [f"{g.get('check')}: {'ok' if g.get('passed') else 'failed'}"
                                       for g in (r.get("gates") or [])]}
               for r in (ws.run or {}).get("results") or []}
    uploads = bool(ws.inventory and ws.inventory.uploads())
    return METHODOLOGIST_CHANGE, {
        "request": request, "brief": _brief_compact(ws),
        "steps": [{**s.to_dict(), "outcome": results.get(s.id)} for s in (base.steps if base else [])],
        "inventory": _inventory_compact(ws),
        "catalogue": _catalogue_for(ws.brief.kind, uploads=uploads), "gates": CHECKS,
    }


def _model_plan(ws: Workspace, model: Model) -> tuple[Study | None, list[str]]:
    from aquascope.gates import CHECKS

    system, context = plan_context(ws)
    uploads = context["uploads"]
    obj = model.call_json("methodologist", system, context)
    if isinstance(obj, dict) and obj.get("decline"):
        reason = str(obj.get("reason") or "the Methodologist found no method that can establish what the brief asks")
        _decline(ws, reason)
        return None, [reason]
    steps, errors, notes = _check(obj, ws)
    if errors and obj is not None:
        ws.event("methodologist", "invalid", "; ".join(errors[:6]))
        repaired = _unwrap(model.call_json("methodologist", METHODOLOGIST_REPAIR, {
            "plan": obj, "errors": errors, "catalogue": context["catalogue"], "gates": list(CHECKS),
            "inventory": context["inventory"], "uploads": uploads or None,
        }))
        steps2, errors2, notes2 = _check(repaired, ws)
        if repaired is not None and steps2 and not errors2:
            obj, steps, errors, notes = repaired, steps2, [], notes + notes2
            ws.event("methodologist", "repaired", f"{len(steps)} steps pass the validator")
        else:
            kept, pruned = _prune(steps, ws)
            if kept and _has_analysis(kept):
                ws.event("methodologist", "pruned", f"{len(steps) - len(kept)} invalid step(s) removed, "
                         f"{len(kept)} kept: " + "; ".join(pruned[:4]))
                steps, errors, notes = kept, [], notes + pruned
            else:
                errors = errors2 if repaired is not None and errors2 else errors
    if errors:
        return None, errors
    return _study_from(ws, obj or {}, steps, notes=notes), []


def _playbook_rule_decline(ws: Workspace) -> str | None:
    """The sentence the playbook prints when one of its own decline rules holds for this brief, else None.
    Data-driven declines (no branch for the record, a method the registry refuses) are not these: a model may
    still find a defensible route there."""
    from aquascope import playbooks as pbk

    pb = _playbook(ws)
    if pb is None:
        return None
    try:
        said = pbk.declines_for(pb, _recon(ws), dict(ws.brief.intake))
    except Exception:  # noqa: BLE001 - a rule that cannot be evaluated does not decline
        return None
    return str(said[0]) if said else None


def _decline(ws: Workspace, reason: str) -> None:
    """Decline, unless the reason is data the user could bring: then the study waits for it (#419)."""
    from aquascope.studio.requests import data_request_for

    request = data_request_for(ws, reason)
    if request is not None and not ws.brief.intake.get("_no_request"):
        _wait(ws, request)
        return
    ws.declined_reason = reason
    ws.set_status("declined")
    ws.event("methodologist", "declined", reason)
    ws.say("methodologist", f"Declined: {reason}", kind="declined", payload={"reason": reason})


def _wait(ws: Workspace, request: dict[str, Any]) -> None:
    """Park the study at ``waiting`` with the request the Consultant relays to the user."""
    ws.pending_request = request
    ws.set_status("waiting")
    ws.event("methodologist", "data_request", str(request.get("what")))
    text = request_text(request)
    ws.say("consultant", text, kind="data_request", payload=request)


def request_text(request: dict[str, Any]) -> str:
    """The request as the Consultant says it."""
    lines = [f"Before this can be planned, the crew needs {request.get('what')}.",
             f"Why: {request.get('why')}.", f"What it changes: {request.get('effect')}."]
    if request.get("can_continue"):
        lines.append("Drop the table in, or say \"continue without\" to go on at the lower grade.")
    else:
        lines.append("Drop the table in; without it the study cannot answer this brief.")
    return " ".join(lines)


def _announce(ws: Workspace, study: Study) -> None:
    ws.set_study(study)
    text = plan_text(ws.study)
    plan = study.plan or {}
    ws.event("methodologist", "plan", f"{plan.get('author')}: {len(study.steps)} step(s)"
             + (f", playbook {plan.get('playbook')}" if plan.get("playbook") else "")
             + (f", branch {plan.get('branch')}" if plan.get("branch") else ""))
    ws.say("methodologist", text, kind="plan", payload={"study": ws.study})


# ── the public functions ─────────────────────────────────────────────────────


def plan(ws: Workspace, model: Model | None) -> Study | None:
    """Write the plan into ``ws.study`` and announce it; on a decline set the status and return None."""
    from aquascope import playbooks as pbk

    known = sorted(p["id"] for p in pbk.list_playbooks() if "error" not in p)
    study: Study | None = None
    if model:
        rule = _playbook_rule_decline(ws)
        if rule:
            # The playbook declines by one of its own rules (an inundation map, a cause without pumping data, a
            # reservoir yield, a health verdict): that is domain knowledge, not a want of a plan, so no model
            # gets to override it.
            _decline(ws, rule)
            return None
        study, errors = _model_plan(ws, model)
        if study is None:
            if ws.status == "declined":
                return None
            if ws.brief.playbook:
                ws.event("methodologist", "fallback", "the model's plan did not pass the validator, the tree is used: "
                         + "; ".join(errors[:4]))
                try:
                    study = _tree(ws)
                    study.plan["model_plan_rejected"] = errors[:8]
                except pbk.Declined as exc:
                    _decline(ws, exc.reason)
                    return None
            else:
                _decline(ws, "The model's plan did not pass the validator and no playbook covers this problem: "
                         + "; ".join(errors[:6]))
                return None
    elif ws.brief.playbook:
        try:
            study = _upload_tree(ws) if wants_upload(ws) else _tree(ws)
        except pbk.Declined as exc:
            _decline(ws, exc.reason)
            return None
    else:
        _decline(ws, f"No playbook covers this problem; say which of {', '.join(known)} it is, or add a model.")
        return None
    _announce(ws, study)
    return study


def adopt(ws: Workspace, obj: dict[str, Any], *, source: str = "device") -> tuple[Study | None, list[str], str]:
    """A plan a caller's own model wrote, taken the way a model's is: the validator with its repairs (a wrong
    method replaced or dropped, a guessed gate path corrected), the invalid steps pruned, the tree when nothing
    valid remains. Returns ``(study, errors, used)``: ``used`` is ``proposed`` or ``tree``, ``errors`` the
    validator's findings on the proposal; ``study`` is None when the tree declines too (the plan at review
    stands then). The study is announced and written to ``ws.study``."""
    from aquascope import playbooks as pbk

    if isinstance(obj, dict) and obj.get("decline"):
        reason = str(obj.get("reason") or f"the {source} model found no method that can establish what the brief asks")
        _decline(ws, reason)
        return None, [], "declined"
    steps, errors, notes = _check(obj if isinstance(obj, dict) else None, ws)
    study: Study | None = None
    if errors:
        ws.event("methodologist", "invalid", f"{source}: " + "; ".join(errors[:6]))
        kept, pruned = _prune(steps, ws)
        if kept and _has_analysis(kept):
            ws.event("methodologist", "pruned", f"{len(steps) - len(kept)} invalid step(s) removed, {len(kept)} kept")
            steps, notes = kept, [*notes, *pruned]
        else:
            steps = []
    if steps:
        study = _study_from(ws, obj, steps, notes=notes)
        study.author = source
        study.plan["author"] = source
        study.plan["proposal"] = {"source": source, "errors": errors[:8], "used": "proposed"}
        _announce(ws, study)
        return study, errors, "proposed"
    ws.event("methodologist", "fallback", f"the {source} plan did not pass the validator, the tree is used: "
             + "; ".join(errors[:4]))
    try:
        study = _upload_tree(ws) if wants_upload(ws) else _tree(ws)
    except (pbk.Declined, pbk.PlaybookError) as exc:
        reason = exc.reason if isinstance(exc, pbk.Declined) else str(exc)
        ws.event("methodologist", "declined", f"the tree cannot replace the {source} plan: {reason}")
        return None, errors, "tree"
    study.plan["proposal"] = {"source": source, "errors": errors[:8], "used": "tree"}
    study.plan["model_plan_rejected"] = errors[:8]
    _announce(ws, study)
    return study, errors, "tree"


def wants_upload(ws: Workspace) -> bool:
    """Plan on an attached table when the text points at it (my/own/this record, data, file, table, upload, csv)
    or when no gauge within reach carries the variable the playbook needs."""
    uploads = _uploads(ws)
    if not uploads:
        return False
    if _UPLOAD_WORDS.search(ws.brief.problem or ""):
        return True
    pb = _playbook(ws)
    if pb is None or not pb.variable:
        return False
    return pb.variable not in (ws.inventory.years_by_variable if ws.inventory else {})


def _upload_tree(ws: Workspace) -> Study:
    """The keyless plan over an attached table: load_table, then the workbench steps for the problem kind, each
    with a gate on its result key, the registry consulted on every method. Raises ``Declined`` when nothing
    defensible remains."""
    from aquascope import playbooks as pbk
    from aquascope.methods import METHODS, SiteContext, assess_method

    pb = _playbook(ws)
    if pb is None:
        raise pbk.Declined("no playbook applies", kind="no_playbook")
    uploads = _uploads(ws)
    upload = next((d for d in uploads if d.variable == pb.variable), None) or \
        next((d for d in uploads if d.variable), None) or uploads[0]
    intake = pbk.fill_intake(pb, dict(ws.brief.intake))
    ws.brief.intake = dict(intake)
    site = dict(ws.site or {})
    rp = intake.get("return_period")
    years = float(upload.years or 0.0)
    ctx = SiteContext(years_by_variable={upload.variable: years} if upload.variable else {},
                      resolution_by_variable={upload.variable: "monthly" if upload.resolution == "monthly"
                                              else "daily"} if upload.variable else {},
                      return_period=float(rp) if isinstance(rp, (int, float)) else None)
    load_args: dict[str, Any] = {"table": upload.id}
    for key in ("value_column", "datetime_column"):
        chosen = ws.brief.intake.get(key)
        if chosen and chosen in ((upload.quality or {}).get("columns") or []):
            load_args[key] = chosen      # the column the client named at intake
    steps: list[Step] = [Step(
        tool=catalogue.LOAD_TABLE, id="s1", arguments=load_args,
        rationale=f"The attached table {upload.id} ({upload.variable or 'rows'}"
                  + (f", {years:g} years" if years else "") + ") is the record this study is about.",
        expects=[{"check": "not_empty", "path": "n"}]
                + ([{"check": "min_years", "value": 1, "path": "years"}] if upload.variable else []),
    )]
    notes: list[str] = []
    recipe = list(_UPLOAD_STEPS.get(pb.id) or [])
    if pb.id == "drought_status":
        # workbench.spei needs a temperature or PET column and a from_step frame carries the value column only:
        # the table is profiled and the indices come from the ERA5 cell.
        recipe = [("eda", None, "n_records"), ("quality", None, "n_records")]
        notes.append("SPI and SPEI over an attached table need a temperature series the table loader does not "
                     "carry; the indices are computed for the ERA5 cell instead.")
    if not upload.variable and pb.id != "water_quality":
        recipe = [("eda", None, "n_records"), ("quality", None, "n_records")]
    if not recipe:
        recipe = _GENERIC_UPLOAD_STEPS
    n = 1
    for tool, method, key in recipe:
        if method and method in METHODS and upload.variable:
            verdict = assess_method(METHODS[method], ctx)
            if verdict.get("status") == "not_defensible":
                notes.append(f"step {tool} dropped: {method} is not defensible on the table: {verdict.get('reason')}")
                continue
            if verdict.get("status") == "marginal":
                notes.append(f"{METHODS[method].label} is marginal on the table: {verdict.get('reason')}")
        n += 1
        args: dict[str, Any] = {"from_step": "s1"}
        if tool == "return_periods":
            periods = sorted({2, 5, 10, 25, 50, 100, *([int(rp)] if isinstance(rp, (int, float)) else [])})
            args.update({"distribution": "gev", "periods": periods})
        if tool == "wqi":
            args["use"] = str(intake.get("use") or "drinking")
        steps.append(Step(tool=tool, id=f"s{n}", arguments=args, method=method, depends_on=["s1"],
                          rationale=f"{catalogue.get(tool).about if catalogue.get(tool) else tool} on the table.",
                          expects=[{"check": "not_empty", "path": key}]))
        if tool == "return_periods":
            n += 1
            steps.append(Step(tool=tool, id=f"s{n}", arguments={**args, "distribution": "lp3"}, method=method,
                              depends_on=["s1"], rationale="A second distribution (Log-Pearson III) on the same "
                              "maxima, so the spread between fits can be quoted.",
                              expects=[{"check": "not_empty", "path": key}]))
    if pb.id == "drought_status" and site.get("lat") is not None:
        n += 1
        args = {"lat": site["lat"], "lon": site["lon"], "years": 40}
        if intake.get("timescales"):
            args["timescales"] = list(intake["timescales"])
        steps.append(Step(tool="drought_indices", id=f"s{n}", arguments=args, method="spei_reanalysis",
                          rationale="SPI and SPEI for the ERA5 cell, the indices the table cannot yield alone.",
                          expects=[{"check": "not_empty", "path": "indices"}]))
    if not any(st.tool not in _FRAMING_TOOLS for st in steps):
        raise pbk.Declined("The attached table supports no defensible analysis for this problem: "
                           + "; ".join(notes), kind="refused", playbook=pb.id)
    caveats, citations = _caveats_and_citations(ws)
    where = f"{site.get('lat')}, {site.get('lon')}" if site else "the site"
    study = Study(
        question=ws.brief.problem, title=f"{pb.title} on {upload.id}: {where}", steps=steps, author="playbook",
        version=3,
        problem={"kind": pb.problem, "site": site, "params": dict(intake), "text": ws.brief.problem},
        plan={"playbook": pb.id, "branch": "upload", "upload": upload.id,
              "rationale": f"The brief points at the attached table {upload.id}"
                           + (f" ({years:g} years of {upload.variable})" if upload.variable else "")
                           + "; the study runs on it rather than on a gauge within reach.",
              "caveats": caveats, "citations": citations, "notes": notes},
        created=_now(),
    )
    return _promote(ws, study, author="playbook")


def revise(ws: Workspace, model: Model | None, edits: dict[str, Any] | list[dict[str, Any]]) -> Study:
    """Apply the user's edits to the plan and validate it again.

    ``edits`` is either a replacement list of steps, ``{"steps": [...]}``, or a
    mapping of step id to overrides (``{"s3": {"arguments": {"k": 8}, "expects": [...]}}``;
    ``{"s2": None}`` drops a step). Raises ``ValueError`` with the validator's
    errors when the edited plan is not acceptable; the plan is left unchanged then.
    """
    if not ws.study:
        raise ValueError("there is no plan to edit")
    steps = [dict(s) for s in ws.study.get("steps") or []]
    if isinstance(edits, list):
        steps = _normalise(edits)
    elif isinstance(edits, dict) and isinstance(edits.get("steps"), list):
        steps = _normalise(edits["steps"])
    elif isinstance(edits, dict):
        kept: list[dict[str, Any]] = []
        for s in steps:
            sid = str(s.get("id"))
            if sid not in edits:
                kept.append(s)
                continue
            override = edits[sid]
            if override is None:
                continue
            if not isinstance(override, dict):
                raise ValueError(f"the edit for step {sid} must be a mapping or null")
            s = dict(s)
            if isinstance(override.get("arguments"), dict):
                s["arguments"] = {**(s.get("arguments") or {}), **_split_overrides(s, override["arguments"])}
            for key in ("expects", "depends_on", "outputs"):
                if isinstance(override.get(key), list):
                    s[key] = override[key]
            for key in ("rationale", "method", "tool"):
                if override.get(key) is not None:
                    s[key] = override[key]
            if "fallback" in override:
                s["fallback"] = override["fallback"]
            kept.append(s)
        steps = _normalise(kept)
    errors = _errors_of(steps, ws)
    if not steps:
        errors.append("the edited plan has no steps")
    if errors:
        ws.event("methodologist", "invalid", "edit refused: " + "; ".join(errors[:6]))
        raise ValueError("; ".join(errors))
    study = Study.from_dict(ws.study)
    study.steps = [Step.from_dict(s) for s in steps]
    study.plan = dict(study.plan or {})
    study.plan["edited"] = True
    study.plan["methodology"] = [_first_sentence(s.rationale) or s.tool for s in study.steps]
    study.results = {}
    for s in study.steps:
        if not s.outputs:
            s.outputs = _outputs_for(s)
    ws.set_study(study)
    ws.event("methodologist", "edited", f"{len(study.steps)} step(s) after the user's edits")
    ws.say("methodologist", plan_text(ws.study), kind="plan", payload={"study": ws.study})
    return study


def _split_overrides(step: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """An override that is not one of the tool's arguments but a key of one of the step's gates (``return_period``)
    lands on those gates; the rest are arguments, which the validator then judges."""
    entry = catalogue.get(str(step.get("tool") or ""))
    args: dict[str, Any] = {}
    for key, value in overrides.items():
        gates = [g for g in (step.get("expects") or []) if isinstance(g, dict) and key in g]
        if entry is not None and key not in entry.arguments and gates:
            step["expects"] = [dict(g, **{key: value}) if key in g else g for g in step["expects"]]
            continue
        args[key] = value
    return args


def change(ws: Workspace, model: Model | None, request: str, *, intake: dict[str, Any] | None = None) -> Study | None:
    """A change after the report: with a model, steps added or replaced for the request; keyless, the tree
    planned again with the changed intake and the steps the rule table adds for the request
    (:data:`FOLLOW_UP_RULES`), with ids continuing the sequence and the validator's say. A request no rule
    covers is declined with the honest reason: it cannot be added without a model. Returns the new study
    (announced) or None with the reason logged."""
    from aquascope import playbooks as pbk

    if intake:
        ws.brief.intake.update(intake)
        ws.event("methodologist", "intake", "changed: " + ", ".join(f"{k}={v}" for k, v in intake.items()))
    base = ws.study_obj()
    study: Study | None = None
    if model and base is not None:
        from aquascope.gates import CHECKS

        system, context = change_context(ws, request)
        obj = model.call_json("methodologist", system, context)
        steps, errors, notes = _check(obj, ws)
        if errors and obj is not None:
            ws.event("methodologist", "invalid", "; ".join(errors[:6]))
            repaired = _unwrap(model.call_json("methodologist", METHODOLOGIST_REPAIR, {
                "plan": obj, "errors": errors, "gates": list(CHECKS)}))
            steps2, errors2, notes2 = _check(repaired, ws)
            if repaired is not None and steps2 and not errors2:
                obj, steps, errors, notes = repaired, steps2, [], notes + notes2
            else:
                kept, pruned = _prune(steps, ws)
                if kept and _has_analysis(kept):
                    steps, errors, notes = kept, [], notes + pruned
        if not errors:
            study = _study_from(ws, obj or {}, steps, base=base, notes=notes)
            study.plan["changed_for"] = request
            if isinstance((obj or {}).get("methodology"), list):
                study.plan["methodology"] = [str(m) for m in obj["methodology"]]
        else:
            ws.event("methodologist", "fallback", "the change did not pass the validator: " + "; ".join(errors[:4]))
    if study is None and model and base is not None:
        # The model's change did not pass the validator: the tree with the changed intake, as a plan would.
        if not ws.brief.playbook:
            ws.event("methodologist", "declined", "no playbook to plan the change from and no valid model plan")
            return None
        try:
            study = _upload_tree(ws) if wants_upload(ws) else _tree(ws)
        except pbk.Declined as exc:
            ws.event("methodologist", "declined", exc.reason)
            return None
        study.plan["changed_for"] = request
    if study is None:
        added = _rule_steps(ws, request, base) if base is not None else ([], [])
        if not intake and not added[0]:
            reason = ("I cannot add that without a model. Keyless, a follow-up can " + _FOLLOW_UP_HELP
                      + (" (" + "; ".join(added[1]) + ")" if added[1] else "") + ".")
            ws.event("methodologist", "declined", reason)
            return None
        if intake and not ws.brief.playbook:
            ws.event("methodologist", "declined", "no playbook to plan the change from and no valid model plan")
            return None
        if intake:
            try:
                study = _upload_tree(ws) if wants_upload(ws) else _tree(ws)
            except pbk.Declined as exc:
                ws.event("methodologist", "declined", exc.reason)
                return None
            # The steps earlier follow-ups added come along, renumbered after the tree's.
            carried = [st.to_dict() for st in base.steps if st.id in ((base.plan or {}).get("added") or [])] \
                if base is not None else []
            if carried:
                study = _append_steps(ws, study, _renumbered(carried, study))
            added = _rule_steps(ws, request, study)
        else:
            study = base
        if added[0]:
            study = _append_steps(ws, study, added[0])
        study.plan["changed_for"] = request
    _announce(ws, study)
    return study


# ── keyless follow-ups: a rule table from the client's words to catalogue steps ─────────────────────────────

#: ``(pattern, rule)``: every pattern the request matches names a rule that writes steps, in this order.
FOLLOW_UP_RULES: list[tuple[str, str]] = [
    (r"\b(add|include|also|another|next|nearest|upstream|downstream|second)\b.*\b(gauge|station|record)s?\b|"
     r"\b(upstream|nearest|another|next|second)\s+(gauge|station)\b", "another_gauge"),
    (r"\bcompar\w*\b.*\b(donors?|regional|neighbou?rs?|similar)\b|\b(donors?|regionali[sz]|neighbou?ring basins?)\b",
     "donors"),
    (r"flow[- ]duration|\bfdc\b|\bq95\b|\bq50\b|percentiles?", "flow_duration"),
    (r"\btrends?\b|mann[- ]kendall|sen'?s slope|stationar", "trend"),
    (r"\b(spi|spei|drought)\b", "drought"),
    (r"base ?flow|\bbfi\b|recession", "baseflow"),
    (r"glofas|era5|reanalysis|climate normal|cross[- ]check", "anywhere"),
]
_FOLLOW_UP_HELP = ("add another gauge (the nearest, an upstream one), compare with the donors, add the flow duration "
                   "curve, a trend test, the drought indices (SPI, SPEI), the baseflow, the ERA5 cell, or change a "
                   "return period (\"200-year\")")


def _station_steps(study: Study) -> set[tuple[str, str]]:
    return {(str(st.arguments.get("source")), str(st.arguments.get("station_id"))) for st in study.steps
            if st.arguments.get("station_id")}


def _stations(ws: Workspace, variable: str | None) -> list[Any]:
    """The inventory's stations carrying ``variable`` (or any), nearest first."""
    inv = ws.inventory
    rows = [d for d in (inv.datasets if inv else []) if d.kind == "station" and d.station_id
            and (variable is None or d.variable == variable)]
    return sorted(rows, key=lambda d: (d.distance_km if d.distance_km is not None else 1e9))


def _main_station(ws: Workspace, study: Study, variable: str | None) -> Any | None:
    """The station the plan already works on (its first station step), else the nearest with the variable."""
    used = [st for st in study.steps if st.arguments.get("station_id")]
    if used:
        st = used[0]
        found = next((d for d in _stations(ws, None) if d.source == st.arguments.get("source")
                      and d.station_id == str(st.arguments.get("station_id"))), None)
        if found is not None:
            return found
    rows = _stations(ws, variable)
    return rows[0] if rows else None


def _variable(ws: Workspace) -> str:
    pb = _playbook(ws)
    return pb.variable if pb is not None and pb.variable else "discharge"


def _rule_steps(ws: Workspace, request: str, study: Study) -> tuple[list[dict[str, Any]], list[str]]:
    """The steps the keyless rules add for ``request`` over ``study``, with ids continuing the sequence, and the
    reasons a matched rule could not be applied."""
    rules = [rule for pat, rule in FOLLOW_UP_RULES if re.search(pat, request, re.I)]
    if not rules:
        return [], []
    site = dict(ws.site or {})
    variable = _variable(ws)
    ids = [st.id for st in study.steps if st.id]
    n = max([int(m.group(1)) for m in (re.match(r"^s(\d+)$", i) for i in ids) if m] or [len(ids)])
    steps: list[dict[str, Any]] = []
    reasons: list[str] = []
    tools_in_plan = {st.tool for st in study.steps}
    table_step = next((st.id for st in study.steps if st.tool == catalogue.LOAD_TABLE), None)

    def add(tool: str, arguments: dict[str, Any], rationale: str, expects: list[dict[str, Any]],
            method: str | None = None, depends_on: list[str] | None = None) -> None:
        nonlocal n
        n += 1
        step: dict[str, Any] = {"id": f"s{n}", "tool": tool, "arguments": arguments, "rationale": rationale,
                                "expects": expects, "depends_on": list(depends_on or []), "outputs": []}
        if method:
            step["method"] = method
        steps.append(step)

    for rule in dict.fromkeys(rules):
        if rule == "another_gauge":
            used = _station_steps(study)
            nxt = next((d for d in _stations(ws, variable) if (str(d.source), str(d.station_id)) not in used), None)
            if nxt is None:
                reasons.append(f"no other {variable} gauge within reach")
                continue
            add("analyze_station", {"source": nxt.source, "station_id": nxt.station_id},
                f"The next {variable} gauge in the inventory, {nxt.name or nxt.station_id} ({nxt.source} "
                f"{nxt.station_id}, {nxt.distance_km} km), analysed the same way for comparison.",
                [{"check": "min_years", "value": 1, "path": "years"}, {"check": "unit_present", "path": "unit"}])
        elif rule == "donors":
            if site.get("lat") is None:
                reasons.append("no site for the donor search")
                continue
            if "similar_basins" not in tools_in_plan:
                add("similar_basins", {"lat": site["lat"], "lon": site["lon"], "k": 10},
                    "Donor catchments by similarity, for a regional comparison.",
                    [{"check": "min_donors", "value": 3, "path": "k"}], method="similar_basins")
            if "regionalize_signatures" not in tools_in_plan:
                add("regionalize_signatures", {"lat": site["lat"], "lon": site["lon"], "k": 10},
                    "The flow signatures transferred from the donors, with their bands.",
                    [{"check": "not_empty", "path": "estimates"}], method="regionalize_signatures")
        elif rule == "flow_duration":
            if table_step:
                add("flow_duration", {"from_step": table_step}, "The flow duration curve of the attached table.",
                    [{"check": "not_empty", "path": "percentiles"}], method="flow_duration", depends_on=[table_step])
            else:
                st = _main_station(ws, study, "discharge")
                if st is None:
                    reasons.append("no discharge gauge for a flow duration curve")
                    continue
                add("low_flow_context", {"source": st.source, "station_id": st.station_id},
                    f"The flow duration curve and low-flow statistics at {st.name or st.station_id}.",
                    [{"check": "not_empty", "path": "fdc"}, {"check": "min_years", "value": 1, "path": "years"}])
        elif rule == "trend":
            st = _main_station(ws, study, variable)
            if st is None:
                reasons.append(f"no {variable} gauge for a trend test")
                continue
            from aquascope.trend_series import is_flood_question

            series = ("annual maxima" if is_flood_question(ws.brief.kind, ws.brief.problem, ws.brief.playbook)
                      else "annual means")
            add("analyze_station", {"source": st.source, "station_id": st.station_id},
                f"Mann-Kendall trend test with Sen's slope on the {series} at {st.name or st.station_id}.",
                [{"check": "not_empty", "path": "trend"}], method="trend_mann_kendall")
        elif rule == "drought":
            if site.get("lat") is None:
                reasons.append("no site for the drought indices")
                continue
            args: dict[str, Any] = {"lat": site["lat"], "lon": site["lon"], "years": 40}
            if isinstance(ws.brief.intake.get("timescales"), list):
                args["timescales"] = list(ws.brief.intake["timescales"])
            add("drought_indices", args, "SPI and SPEI for the ERA5 cell at the site.",
                [{"check": "not_empty", "path": "indices"}], method="spei_reanalysis")
        elif rule == "baseflow":
            if table_step:
                add("baseflow", {"from_step": table_step}, "Baseflow separation on the attached table.",
                    [{"check": "not_empty", "path": "bfi"}], method="baseflow_separation", depends_on=[table_step])
            else:
                st = _main_station(ws, study, "discharge")
                if st is None:
                    reasons.append("no discharge gauge for a baseflow index")
                    continue
                add("low_flow_context", {"source": st.source, "station_id": st.station_id},
                    f"The baseflow index and low-flow context at {st.name or st.station_id}.",
                    [{"check": "not_empty", "path": "bfi"}, {"check": "min_years", "value": 1, "path": "years"}],
                    method="baseflow_separation")
        elif rule == "anywhere":
            if site.get("lat") is None:
                reasons.append("no site for the ERA5 cell")
                continue
            add("anywhere", {"lat": site["lat"], "lon": site["lon"], "years": 20},
                "The ERA5 cell's climate normals and the GloFAS discharge as an independent cross-check.",
                [{"check": "not_empty", "path": "climate"}])
    return steps, reasons


def _renumbered(steps: list[dict[str, Any]], study: Study) -> list[dict[str, Any]]:
    """``steps`` with ids continuing ``study``'s sequence (their references to each other renamed too)."""
    ids = [st.id for st in study.steps if st.id]
    n = max([int(m.group(1)) for m in (re.match(r"^s(\d+)$", i) for i in ids) if m] or [len(ids)])
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for step in steps:
        n += 1
        names[str(step.get("id"))] = f"s{n}"
        out.append({**step, "id": f"s{n}"})
    for step in out:
        step["depends_on"] = [names.get(str(d), str(d)) for d in step.get("depends_on") or []]
        args = dict(step.get("arguments") or {})
        if args.get("from_step") in names:
            args["from_step"] = names[str(args["from_step"])]
        step["arguments"] = args
    return out


def _append_steps(ws: Workspace, study: Study, added: list[dict[str, Any]]) -> Study:
    """``study`` with the rule's steps appended, validated against the catalogue (an invalid one is dropped with
    a note); the plan block extended."""
    kept: list[dict[str, Any]] = []
    notes: list[str] = []
    existing = [st.to_dict() for st in study.steps]
    for step in added:
        errors = _errors_of([*existing, *kept, step], ws)
        if errors:
            notes.append(f"step {step['id']} ({step['tool']}) not added: " + "; ".join(errors[:2]))
            continue
        kept.append(step)
    study.steps = [*study.steps, *[Step.from_dict(s) for s in kept]]
    study.results = {}
    study.plan = dict(study.plan or {})
    study.plan["methodology"] = [*(study.plan.get("methodology") or []),
                                 *[_first_sentence(s["rationale"]) for s in kept]]
    study.plan["added"] = [*(study.plan.get("added") or []), *[s["id"] for s in kept]]
    if notes:
        study.plan["notes"] = list(dict.fromkeys([*(study.plan.get("notes") or []), *notes]))
    for s in study.steps:
        if not s.outputs:
            s.outputs = _outputs_for(s)
    ws.event("methodologist", "added", f"{len(kept)} step(s) by the keyless rules: "
             + ", ".join(f"{s['id']} {s['tool']}" for s in kept))
    return study


def plan_text(study: dict[str, Any] | None) -> str:
    """The plan as a numbered checklist: objective, then each step with tool, arguments, method, gates and
    rationale, then the caveats count. What the CLI prints and a face can show."""
    if not study:
        return "No plan."
    plan = study.get("plan") or {}
    steps = study.get("steps") or []
    head = f"Plan ({plan.get('author') or study.get('author')}"
    if plan.get("playbook"):
        head += f", playbook {plan['playbook']}"
    if plan.get("branch"):
        head += f", branch {plan['branch']}"
    lines = [f"{head}, {len(steps)} step(s))"]
    if plan.get("objective"):
        lines.append(f"  Objective: {plan['objective']}")
    for n in (plan.get("notes") or []):
        lines.append(f"  note: {n}")
    for i, s in enumerate(steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in (s.get("arguments") or {}).items())
        lines.append(f"  {i}. [{s.get('id')}] {s.get('tool')}({args})"
                     + (f"  method {s['method']}" if s.get("method") else ""))
        if s.get("rationale"):
            lines.append(f"     {s['rationale']}")
        for g in s.get("expects") or []:
            where = g.get("path") or ", ".join(g.get("paths") or [])
            value = f" {g['value']}" if g.get("value") is not None else ""
            lines.append(f"     gate {g.get('check')}{value} on {where}")
        fb = s.get("fallback")
        if isinstance(fb, dict) and isinstance(fb.get("step"), dict):
            lines.append(f"     fallback: {fb['step'].get('tool')}")
        elif isinstance(fb, dict) and fb.get("branch"):
            lines.append(f"     fallback: replan on branch {fb['branch']}")
    for a in (plan.get("assumptions") or [])[:5]:
        lines.append(f"  assumes: {a}")
    if plan.get("caveats"):
        lines.append(f"  {len(plan['caveats'])} caveat(s) will be printed verbatim in the report.")
    lines.append("  Edit a step with 's3.return_period=200' style overrides, or approve to run.")
    return "\n".join(lines)
