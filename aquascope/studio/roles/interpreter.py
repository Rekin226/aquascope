"""The Interpreter: the engineer's step between the numbers and the report (#417).

After the Analysts and before the Author, one role reads the results, the
gates and the brief's decision and writes findings, not prose: claims that
each point at the result they come from (``basis`` paths the engine
resolves), consistency notes between estimates, a decision block (the
answer, its band, its grade, the conditions and what would change it), the
data the study would ask for, and the assumptions it made. The Author writes
from the findings; the Critic checks the prose against them and the findings
against the results.

Keyless, the findings are a rule table over the key numbers and the gates.
With a model, one stateless call returns the same shape and the engine
validates it: a finding whose basis resolves to nothing is dropped and
counted, a claim whose number is not at its basis is dropped, a grade the
model raises above the rule's is lowered (a model may be more cautious than
the rules, never less), and a decision value that is at no basis is replaced
by the rules' own. The model decides and judges; it never computes or
transcribes a number the tools did not return.

Grades (#418): ``established`` (at-site data, every gate of the step
passed), ``indicative`` (a fallback ran, a donor transfer, a method the
sufficiency table calls marginal, another required step failed, or a gate
that compares with the headline number failed, a cross-check included),
``screening`` (regional or reanalysis data only, no in-situ record),
``not_established`` (the step that carries the answer failed).
"""

from __future__ import annotations

import re
from typing import Any

from aquascope.gates import resolve_path
from aquascope.studio.model import Model, compact
from aquascope.studio.prompts import INTERPRETER
from aquascope.studio.workspace import Workspace

__all__ = ["GRADES", "decision_text", "find_path", "grade_for_step", "grade_for_study", "headline_gates", "interpret",
           "interpreter_context", "resolve_basis", "rules_findings", "validate_findings"]

#: From the most to the least trusted; a model may move a grade down this list, never up.
GRADES = ("established", "indicative", "screening", "not_established")

#: Tools whose numbers come from regional transfer or reanalysis rather than a record at the place.
_SCREENING_TOOLS = frozenset({"anywhere", "similar_basins", "regionalize_signatures", "describe_catchment"})
_SCREENING_METHODS = frozenset({"spei_reanalysis", "regionalize_signatures", "similar_basins", "glofas_cross_check"})
#: Tools whose failure does not lower the answer's grade: they frame or cross-check, they do not carry it.
_SIDE_TOOLS = frozenset({"anywhere", "describe_catchment", "similar_basins", "assess_site"})

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", re.I)
_MAX_FINDINGS = 24


def _rank(grade: str | None) -> int:
    return GRADES.index(grade) if grade in GRADES else len(GRADES)


def _lower(a: str | None, b: str | None) -> str:
    """The less trusted of two grades."""
    return a if _rank(a) >= _rank(b) else (b or "not_established")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) <= max(abs(b), 1e-9) * tol + 1e-9


# ── basis paths ──


def find_path(payload: Any, value: float, *, tol: float = 0.005, _prefix: str = "") -> str | None:
    """The dotted path of the first number in ``payload`` within ``tol`` of ``value`` (depth-first); None when
    the number is nowhere in it. Lists are indexed ``[i]``; bulk keys (series, points) are skipped."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in ("series", "points", "samples", "daily", "monthly_series", "png", "svg", "data"):
                continue
            if _is_number(v) and _close(float(v), value, tol):
                return f"{_prefix}{k}"
            if isinstance(v, (dict, list)):
                found = find_path(v, value, tol=tol, _prefix=f"{_prefix}{k}.")
                if found:
                    return found
    elif isinstance(payload, list):
        # list positions as dotted indices ("q.5", "ci.5.0"): the one form resolve_path reads at any depth
        for i, v in enumerate(payload[:64]):
            if _is_number(v) and _close(float(v), value, tol):
                return f"{_prefix}{i}"
            if isinstance(v, (dict, list)):
                found = find_path(v, value, tol=tol, _prefix=f"{_prefix}{i}.")
                if found:
                    return found
    return None


def _result_of(ws: Workspace, sid: str) -> Any:
    """The payload of a step (``s3``) or of its fallback (``s3.fallback``); None when neither ran."""
    for r in (ws.run or {}).get("results") or []:
        if str(r.get("id")) == sid:
            return r.get("result")
        if sid == f"{r.get('id')}.fallback" and isinstance(r.get("fallback"), dict):
            return r["fallback"].get("result")
    return None


def resolve_basis(ws: Workspace, basis: str) -> Any:
    """The value at a basis path ``<step id>[.fallback].<path>``; None when the step, the path or the value
    is not there."""
    text = str(basis or "").strip()
    if not text:
        return None
    head, _, rest = text.partition(".")
    sid = head
    if rest.startswith("fallback."):
        sid, rest = f"{head}.fallback", rest[len("fallback."):]
    elif rest == "fallback":
        sid, rest = f"{head}.fallback", ""
    payload = _result_of(ws, sid)
    if payload is None:
        return None
    if not rest:
        return payload
    return resolve_path(payload, rest)


# ── grades ──


def _record(ws: Workspace, sid: str) -> dict[str, Any] | None:
    for r in (ws.run or {}).get("results") or []:
        if str(r.get("id")) == sid:
            return r
    return None


def _sufficiency(ws: Workspace) -> dict[str, dict[str, Any]]:
    rows = ws.inventory.sufficiency if ws.inventory else []
    return {str(r.get("method")): r for r in rows if isinstance(r, dict) and r.get("method")}


def _established(rec: dict[str, Any]) -> bool:
    if rec.get("skipped") or not rec.get("ok"):
        return False
    if rec.get("gates_passed", True):
        return True
    fb = rec.get("fallback")
    return bool(rec.get("fallback_used") and isinstance(fb, dict) and fb.get("ok") and fb.get("gates_passed"))


def grade_for_step(ws: Workspace, sid: str) -> str:
    """The grade a step's numbers carry, from what ran and the gates (see the module docstring)."""
    rec = _record(ws, sid.replace(".fallback", ""))
    if rec is None or rec.get("skipped") or not rec.get("ok"):
        return "not_established"
    via_fallback = sid.endswith(".fallback") or (not rec.get("gates_passed", True) and _established(rec))
    if not _established(rec):
        return "not_established"
    tool = str(rec.get("tool") or "")
    step = next((s for s in (ws.study or {}).get("steps") or [] if str(s.get("id")) == rec.get("id")), {})
    method = str(step.get("method") or "")
    if tool in _SCREENING_TOOLS or method in _SCREENING_METHODS:
        return "screening"
    if via_fallback:
        return "indicative"
    row = _sufficiency(ws).get(method)
    if row and str(row.get("status") or "") in ("marginal", "marginally_defensible"):
        return "indicative"
    return "established"


def _branch_is_screening(ws: Workspace) -> bool:
    plan = (ws.study or {}).get("plan") or {}
    if str(plan.get("branch") or "") in ("regional", "reanalysis", "demand_only"):
        return True
    steps = (ws.study or {}).get("steps") or []
    return bool(steps) and all(str(s.get("tool")) in _SCREENING_TOOLS or str(s.get("method") or "")
                               in _SCREENING_METHODS for s in steps)


#: Key numbers that frame a study rather than answer it: never the headline.
_FRAMING_LABELS = re.compile(r"^(upstream area|record length|mean of the record|catchment area|area)\b|"
                             r"interval|\bcount\b|^donor|^n\b|p-value|\btau\b", re.I)
_STOP_WORDS = frozenset({"the", "and", "with", "its", "for", "from", "over", "per", "record", "band", "its",
                         "confidence", "interval", "year", "years", "a", "an", "of", "at", "in", "on", "to",
                         "gauge", "station", "river", "data", "series", "value", "values", "number"})
#: What each problem kind's answer is called in the key numbers, so a brief with no usable quantity words still
#: gets the right headline (a flood question gets a return level, never the record's mean; an irrigation
#: question gets the demand or the reliability, never the flood fit the record also carries).
#: What a brief's word is called in a key-number label.
_SYNONYMS: dict[str, set[str]] = {
    "trend": {"slope", "sen's"}, "trends": {"slope"}, "slope": {"trend"},
    "flood": {"return", "level"}, "return": {"level"}, "discharge": {"return", "flow"},
    "reliability": {"reliab", "days", "met"}, "reliable": {"reliab", "days", "met"}, "reliably": {"reliab", "days"},
    "drought": {"spi", "spei", "sgi", "class"}, "index": {"spi", "spei", "sgi", "wqi"},
    "demand": {"demand", "requirement"}, "requirement": {"demand"},
    "low-flow": {"q95"}, "low": {"q95"}, "quality": {"wqi", "index"},
}
_KIND_ANSWERS: dict[str, str] = {
    "flood_risk": r"return level",
    "ungauged_flow": r"q95|mean flow|q05|signature|flow",
    "drought": r"spi|spei|sgi|drought|class",
    "drought_status": r"spi|spei|sgi|drought|class",
    "groundwater_decline": r"slope|trend|sgi|recharge|level",
    "supply_reliability": r"reliab|days|years|deficit|demand",
    "irrigation": r"demand|requirement|reliab|peak",
    "irrigation_feasibility": r"demand|requirement|reliab|peak",
    "water_quality": r"index|wqi|exceed|class",
}


def _headline(ws: Workspace, key: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The key number that answers the brief: the one whose label shares the most words with the brief's
    quantities, then the one the problem kind's answer is called, never a framing number (an area, a record
    length, a count, an interval bound). None when nothing answers: the decision then says so rather than
    quoting a number that is not an answer."""
    if not key:
        return None
    candidates = [kn for kn in key if not _FRAMING_LABELS.search(str(kn.get("label") or ""))]

    def words_of(text: str) -> set[str]:
        out = {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 2 and w not in _STOP_WORDS}
        for w in list(out):
            out |= _SYNONYMS.get(w, set())
        return out

    quantities = list(ws.brief.quantities)
    first = words_of(quantities[0]) if quantities else set()
    rest = {w for q in quantities[1:] for w in words_of(q)}
    kind_pattern = _KIND_ANSWERS.get(str(ws.brief.kind or ""), None) or _KIND_ANSWERS.get(str(ws.brief.playbook or ""))
    if kind_pattern:
        # the problem kind says what an answer is called: a flood question is answered by a return level or
        # nothing, never by the record's mean flow because the brief happened to say "flow"
        candidates = [kn for kn in candidates if re.search(kind_pattern, str(kn.get("label") or ""), re.I)]
    best, best_score = None, 0.0
    for kn in candidates:
        label = str(kn.get("label") or "").lower()
        # the brief's first quantity is what it asks for first; a word from it counts double
        score = float(sum(2 for w in first if re.search(rf"(?<![a-z0-9]){re.escape(w)}", label)))
        score += float(sum(1 for w in rest - first if re.search(rf"(?<![a-z0-9]){re.escape(w)}", label)))
        if score > best_score:
            best, best_score = kn, score
    if best is None and candidates and kind_pattern:
        best = candidates[0]    # the kind's own vocabulary, in the plan's order, when the brief's words say nothing
    return best


def _primary_step(ws: Workspace, key: list[dict[str, Any]]) -> str | None:
    """The step that carries the answer: the one behind the headline key number, else the last analysis step
    that established anything."""
    head = _headline(ws, key)
    if head is not None and head.get("step"):
        return str(head["step"]).replace(".fallback", "")
    for r in reversed((ws.run or {}).get("results") or []):
        if _established(r) and str(r.get("tool")) not in _SIDE_TOOLS:
            return str(r.get("id"))
    return None


#: Gates that compare one step's number with another's: when one fails against the headline's step, the
#: headline itself is in question, whichever step the gate sits on.
_COMPARISON_CHECKS = frozenset({"cross_check_ratio", "spread_within"})


def headline_gates(ws: Workspace, primary: str | None) -> list[dict[str, Any]]:
    """The failed (not skipped) gates that bear on the headline number: every failed gate of the primary step,
    and a failed comparison gate on any other step that compares with the primary step (its reference reads
    the primary's result, or the step depends on it). A side step's cross-check is optional, but when it ran
    and disagreed with the answer, the answer cannot be called established (the live USGS 01013500 study
    said "established" above a failed GloFAS cross-check and a "Not established" box)."""
    if not primary:
        return []
    steps = {str(s.get("id")): s for s in (ws.study or {}).get("steps") or [] if isinstance(s, dict)}
    out: list[dict[str, Any]] = []
    for r in (ws.run or {}).get("results") or []:
        sid = str(r.get("id"))
        failed = [g for g in r.get("gates") or [] if isinstance(g, dict) and not g.get("passed")
                  and not g.get("skipped")]
        if not failed:
            continue
        if sid == primary:
            out.extend({**g, "step": sid} for g in failed)
            continue
        step = steps.get(sid) or {}
        refs_primary = any(isinstance(e, dict) and f"result.{primary}." in str(e.get("reference") or "")
                           for e in step.get("expects") or [])
        depends = primary in [str(d) for d in step.get("depends_on") or []]
        for g in failed:
            if str(g.get("check")) in _COMPARISON_CHECKS and (refs_primary or depends):
                out.append({**g, "step": sid})
    return out


def grade_for_study(ws: Workspace, key: list[dict[str, Any]] | None = None) -> tuple[str, str | None]:
    """The grade of the study's answer and the step it rests on: the primary step's grade, lowered to
    indicative when another step that carries results failed, when a gate that bears on the headline failed
    (:func:`headline_gates`) or when the run stopped early, and to screening when the whole plan is regional
    or reanalysis. This is the one verdict: the decision, the report's grade and the badge all read it."""
    from aquascope.studio.roles.author import key_numbers

    key = key if key is not None else key_numbers(ws.study_obj(), (ws.run or {}).get("results") or [])
    primary = _primary_step(ws, key)
    grade = grade_for_step(ws, primary) if primary else "not_established"
    if _branch_is_screening(ws):
        grade = _lower(grade, "screening")
    for r in (ws.run or {}).get("results") or []:
        if str(r.get("id")) != primary and not _established(r) and str(r.get("tool")) not in _SIDE_TOOLS:
            grade = _lower(grade, "indicative")
    if headline_gates(ws, primary) or (ws.run or {}).get("stop_reason"):
        grade = _lower(grade, "indicative")
    return grade, primary


# ── the rules ──


_DATA_REQUESTS: dict[str, list[tuple[str, str, str, str]]] = {
    # playbook: (intake flag or gate check that triggers it, what, why, effect on the grade)
    "groundwater_decline": [
        ("attribute_cause", "abstraction (pumping) records for the wells around the site",
         "a decline can only be attributed to pumping, drought or land use with the abstraction history",
         "with them the cause can be tested and the answer becomes established; without them the study reports "
         "the trend and stops at what the levels show"),
    ],
    "flood_risk": [
        ("min_years", "a longer discharge record, or the agency's annual peak series",
         "the at-site fit needs about twenty complete years of annual maxima",
         "a longer record moves the fit from indicative to established"),
        ("max_return_period_factor", "the annual peak series of a nearby long gauge, or a regional growth curve",
         "the return period asked is beyond what the record can carry without extrapolation",
         "a regional curve keeps the estimate indicative but bounded"),
    ],
    "supply_reliability": [
        ("storage", "the reservoir's capacity and operating rule",
         "a run-of-river screening cannot answer a stored-supply question",
         "with them a storage-yield analysis is possible"),
    ],
    "irrigation_feasibility": [
        ("min_years", "a longer gauge record, or the agency's low-flow statistics",
         "the supply check rests on the dry years the record holds",
         "a longer record moves the reliability from indicative to established"),
    ],
}


def _data_requests(ws: Workspace) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    pb = str(ws.brief.playbook or "")
    intake = ws.brief.intake or {}
    failed = {str(g.get("check")) for g in (ws.run or {}).get("failed_gates") or []}
    for trigger, what, why, effect in _DATA_REQUESTS.get(pb, []):
        if intake.get(trigger) or trigger in failed:
            out.append({"what": what, "why": why, "effect_on_grade": effect})
    for n in ((ws.study or {}).get("plan") or {}).get("notes") or []:
        text = str(n)
        if re.search(r"\bbring\b|\bneeds?\b|\brequires?\b", text, re.I) and "removed" not in text and len(out) < 4:
            out.append({"what": text, "why": "the plan named it", "effect_on_grade": "see the plan's note"})
    return out[:4]


def _consistency(ws: Workspace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in (ws.run or {}).get("gates") or []:
        check = str(g.get("check") or "")
        if check not in ("spread_within", "cross_check_ratio", "fit_envelopes_max", "trend_on_series"):
            continue
        detail = str(g.get("detail") or "")
        m = re.search(r"ratio (\d+\.\d+)|spread (\d+)%", detail)
        ratio = None
        if m:
            ratio = float(m.group(1)) if m.group(1) else round(1 + float(m.group(2)) / 100, 2)
        out.append({"a": f"{g.get('step')}: {check}", "b": str(g.get("path") or g.get("paths") or ""),
                    "ratio": ratio, "agree": None if g.get("skipped") else bool(g.get("passed")), "note": detail})
    return out


def _interval_for(key: list[dict[str, Any]], label: str) -> list[float] | None:
    """The [low, high] key numbers that belong to a headline label: the interval whose label shares the most
    words with it ("100-year return level, GEV (L-moments)" pairs with "100-year GEV bootstrap 90 % interval",
    not with the LP3 one)."""
    words = {w for w in re.findall(r"[a-z0-9-]+", label.lower()) if len(w) > 1 and w not in ("return", "level")}
    first = label.split(" ")[0].lower()

    def pick(kind: str) -> dict[str, Any] | None:
        rows = [k for k in key if f"interval, {kind}" in str(k.get("label", "")).lower()
                and first in str(k.get("label", "")).lower() and _is_number(k.get("value"))]
        if not rows:
            return None
        return max(rows, key=lambda k: sum(1 for w in words if w in str(k.get("label", "")).lower()))

    low, high = pick("low"), pick("high")
    if low is not None and high is not None:
        return [float(low["value"]), float(high["value"])]
    return None


#: Where a key number's value lives, by what its label says: the sub-trees searched first, so a return level
#: is anchored under ``ffa`` even when the record maximum happens to equal it.
_LABEL_HINTS: list[tuple[str, tuple[str, ...]]] = [
    (r"return level|interval|annual max|bootstrap|record maximum", ("ffa",)),
    (r"mann-kendall|sen's slope|\btau\b|trend", ("trend", "ffa")),
    (r"^q\d|exceeded|median flow|flow.duration", ("fdc",)),
    (r"^record length|years of", ("years",)),
    (r"^mean of the record|minimum|maximum of", ("stats",)),
    (r"upstream area|elevation|slope|regulation|reservoir|land|population|aridity|precipitation",
     ("attributes", "catchment", "climate")),
    (r"spi|spei|drought|timescale", ("current", "indices", "events")),
    (r"reliab|demand|deficit", ("reliability", "demand", "season")),
    (r"donor|signature|q95 band|mean flow band", ("estimates", "skill", "stations")),
]


def _basis_for(payload: Any, label: str, value: float) -> str | None:
    """The basis path of a key number: the sub-trees its label points at first, then the whole payload."""
    if not isinstance(payload, dict):
        return find_path(payload, value)
    for pattern, keys in _LABEL_HINTS:
        if re.search(pattern, label, re.I):
            for k in keys:
                if k in payload:
                    if _is_number(payload[k]) and _close(float(payload[k]), value, 0.005):
                        return k
                    found = find_path(payload[k], value, _prefix=f"{k}.")
                    if found:
                        return found
    return find_path(payload, value)


def decision_text(decision: dict[str, Any]) -> str:
    """The decision block as one sentence for the report's answer."""
    if not decision:
        return ""
    value, unit = decision.get("value"), decision.get("unit") or ""
    band, grade = decision.get("band"), decision.get("grade")
    head = str(decision.get("answer") or "").strip()
    if head:
        return head
    parts = []
    if _is_number(value):
        parts.append(f"{value:g} {unit}".strip())
    if isinstance(band, (list, tuple)) and len(band) == 2 and all(_is_number(x) for x in band):
        parts.append(f"(band {band[0]:g} to {band[1]:g} {unit})".strip())
    if grade:
        parts.append(f"graded {grade.replace('_', ' ')}")
    return " ".join(parts) + "." if parts else ""


def rules_findings(ws: Workspace) -> dict[str, Any]:
    """The keyless Interpreter: one finding per key number with the path its value sits at, consistency from
    the comparison gates, the decision from the brief and the primary number, the data requests from the
    playbook's rules."""
    from aquascope.studio.roles.author import key_numbers

    study = ws.study_obj()
    results = (ws.run or {}).get("results") or []
    key = key_numbers(study, results) if study else []
    findings: list[dict[str, Any]] = []
    for kn in key[:_MAX_FINDINGS]:
        sid = str(kn.get("step") or "")
        value = kn.get("value")
        payload = _result_of(ws, sid)
        path = (_basis_for(payload, str(kn.get("label") or ""), float(value))
                if _is_number(value) and payload is not None else None)
        if path is None:
            continue
        unit = kn.get("unit") or ""
        findings.append({
            "id": f"f{len(findings) + 1}",
            "claim": (f"{kn.get('label')}: {value:g} {unit}".strip() if _is_number(value)
                      else f"{kn.get('label')}: {value}"),
            "basis": [f"{sid}.{path}"],
            "grade": grade_for_step(ws, sid),
        })
    grade, primary = grade_for_study(ws, key)
    headline = _headline(ws, key)
    decision: dict[str, Any] = {"answer": "", "value": None, "unit": None, "band": None, "grade": grade,
                                "basis": [], "conditions": [], "what_would_change_it": []}
    if headline is not None and _is_number(headline.get("value")):
        f_head = next((f for f in findings if f["basis"][0].startswith(f"{headline.get('step')}.")
                       and f["claim"].startswith(str(headline.get("label")))), None)
        band = _interval_for(key, str(headline.get("label")))
        decision.update({"value": float(headline["value"]), "unit": headline.get("unit") or "", "band": band,
                         "basis": list(f_head["basis"]) if f_head else []})
        what = ws.brief.decision or (study.plan or {}).get("objective") if study else ws.brief.decision
        band_text = f", band {band[0]:g} to {band[1]:g} {headline.get('unit') or ''}".rstrip() if band else ""
        decision["answer"] = (f"{(what or 'The answer').strip().rstrip('.')}: {headline.get('label')} "
                              f"{float(headline['value']):g} {headline.get('unit') or ''}".rstrip()
                              + f"{band_text} ({grade.replace('_', ' ')}).")
    else:
        answers = [kn for kn in key if not _FRAMING_LABELS.search(str(kn.get("label") or ""))][:3]
        have = "; ".join(f"{kn.get('label')} {kn.get('value')} {kn.get('unit') or ''}".strip() for kn in answers
                         if _is_number(kn.get("value")))
        decision["answer"] = (f"No number in the results answers the decision ({grade.replace('_', ' ')})"
                              + (f"; the study established {have}." if have else "."))
    run = ws.run or {}
    for g in (run.get("failed_gates") or [])[:3]:
        decision["conditions"].append(f"step {g.get('step')} did not pass {g.get('check')}: {g.get('detail')}")
    for c in ((study.plan or {}).get("caveats") or [])[:2] if study else []:
        decision["conditions"].append(str(c).split(". ")[0].rstrip(".") + ".")
    for f in (run.get("failed_steps") or []):
        if not f.get("skipped"):
            decision["what_would_change_it"].append(f"{f.get('tool')} ({f.get('id')}) establishing its result: "
                                                    f"{f.get('reason')}")
    suff = _sufficiency(ws)
    for s in (study.steps if study else []):
        row = suff.get(str(s.method or ""))
        if row and str(row.get("status") or "") == "marginal" and len(decision["what_would_change_it"]) < 4:
            decision["what_would_change_it"].append(f"a longer record for {s.method}: {row.get('reason')}")
    requests = _data_requests(ws)
    for r in requests:
        if len(decision["what_would_change_it"]) < 5:
            decision["what_would_change_it"].append(f"{r['what']}: {r['effect_on_grade']}")
    plan_assumptions = [str(a) for a in ((study.plan or {}).get("assumptions") or [])] if study else []
    assumptions = list(dict.fromkeys([*ws.brief.assumptions, *plan_assumptions]))
    return {"findings": findings, "consistency": _consistency(ws), "decision": decision,
            "data_requests": requests, "assumptions": assumptions, "next_steps": [],
            "written_by": "rules", "dropped": 0, "primary_step": primary}


# ── the model ──


def interpreter_context(ws: Workspace) -> tuple[str, dict[str, Any]]:
    """The system prompt and the compact context the Interpreter sends a model."""
    from aquascope.studio.roles.author import key_numbers

    study = ws.study_obj()
    results = (ws.run or {}).get("results") or []
    draft = rules_findings(ws)
    context = {
        "brief": {k: v for k, v in ws.brief.to_dict().items()
                  if k in ("problem", "decision", "quantities", "kind", "playbook", "intake", "assumptions")},
        "steps": [{"id": r.get("id"), "tool": r.get("tool"), "method": next(
            (s.get("method") for s in (ws.study or {}).get("steps") or [] if s.get("id") == r.get("id")), None),
            "ok": r.get("ok"), "skipped": r.get("skipped"), "gates": r.get("gates"),
            "failed_reason": r.get("failed_reason"), "result": compact(r.get("result"), max_list=16),
            "fallback": (compact({k: v for k, v in (r.get("fallback") or {}).items()
                                  if k in ("tool", "ok", "gates", "result")}, max_list=16)
                         if r.get("fallback") else None)}
                  for r in results],
        "key_numbers": key_numbers(study, results) if study else [],
        "sufficiency": [{k: r.get(k) for k in ("method", "status", "reason")}
                        for r in (ws.inventory.sufficiency if ws.inventory else []) if isinstance(r, dict)][:16],
        "run_summary": (ws.run or {}).get("summary"),
        "grades": list(GRADES),
        "rule_grade": draft["decision"]["grade"],
        "draft": {k: draft[k] for k in ("findings", "consistency", "decision", "data_requests")},
    }
    return INTERPRETER, context


def validate_findings(ws: Workspace, obj: dict[str, Any], *, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """A model's findings, held to the results: a finding whose basis resolves to nothing, or whose claim
    carries a number that is at none of its bases, is dropped and counted; a grade above the rule's for
    its step is lowered; the decision's value must sit at a basis or the rules' decision stands; grades
    only ever go down from the rule's."""
    rules = rules or rules_findings(ws)
    dropped = 0
    reanchored = 0
    findings: list[dict[str, Any]] = []
    for raw in (obj.get("findings") or [])[:_MAX_FINDINGS]:
        if not isinstance(raw, dict) or not raw.get("claim"):
            dropped += 1
            continue
        bases = [str(b) for b in (raw.get("basis") or []) if isinstance(b, str) and b.strip()]
        claimed = _claimed_numbers(str(raw["claim"]))
        values: list[float] = []
        kept_bases: list[str] = []
        for b in bases:
            v = resolve_basis(ws, b)
            if v is None and claimed:
                # A path a model guessed wrong is re-anchored when the claim's number is in that step's result:
                # the number exists and the tool computed it, the path was the model's typo.
                parts = b.split(".")
                sid, head = parts[0], (parts[1] if len(parts) > 1 else None)
                payload = _result_of(ws, sid)
                found = None
                if isinstance(payload, dict) and head in payload:
                    # the model's path started right: look under that key first ("s3.ffa.gev.q100" -> under ffa)
                    found = next((find_path(payload[head], c, _prefix=f"{head}.") for c in claimed
                                  if find_path(payload[head], c, _prefix=f"{head}.")), None)
                if found is None and payload is not None:
                    found = next((find_path(payload, c) for c in claimed if find_path(payload, c)), None)
                if found:
                    b, v = f"{sid}.{found}", resolve_basis(ws, f"{sid}.{found}")
                    reanchored += 1
            if _is_number(v):
                values.append(float(v))
                kept_bases.append(b)
            elif isinstance(v, (list, dict)):
                values.extend(float(x) for x in _walk(v))
                kept_bases.append(b)
        bases = kept_bases
        if not bases or not values:
            dropped += 1
            continue
        if claimed and not all(any(_close(c, v) for v in values) for c in claimed):
            dropped += 1
            continue
        sid = bases[0].split(".")[0]
        rule_grade = grade_for_step(ws, sid)
        grade = str(raw.get("grade") or rule_grade)
        findings.append({"id": f"f{len(findings) + 1}", "claim": str(raw["claim"]).strip(), "basis": bases,
                         "grade": _lower(grade if grade in GRADES else rule_grade, rule_grade)})
    if not findings:
        out = dict(rules)
        out["written_by"], out["dropped"] = "rules", dropped
        out["note"] = "the model's findings did not resolve; the rules stand"
        return out
    decision = dict(rules["decision"])
    raw_d = obj.get("decision") if isinstance(obj.get("decision"), dict) else {}
    value = raw_d.get("value")
    if _is_number(value) and any(any(_close(float(value), v) for v in _basis_values(ws, f["basis"])) for f in findings):
        decision["value"] = float(value)
        decision["unit"] = str(raw_d.get("unit") or decision.get("unit") or "")
        band = raw_d.get("band")
        if isinstance(band, (list, tuple)) and len(band) == 2 and all(_is_number(x) for x in band):
            decision["band"] = [float(band[0]), float(band[1])]
    if isinstance(raw_d.get("answer"), str) and raw_d["answer"].strip():
        claimed = _claimed_numbers(raw_d["answer"])
        pool = [v for f in findings for v in _basis_values(ws, f["basis"])]
        pool += [x for x in (decision.get("band") or []) if _is_number(x)]
        if all(any(_close(c, v) for v in pool) for c in claimed):
            decision["answer"] = raw_d["answer"].strip()
    decision["grade"] = _lower(str(raw_d.get("grade") or rules["decision"]["grade"]), rules["decision"]["grade"])
    if decision["grade"] != str(raw_d.get("grade") or "") and decision.get("answer"):
        # the grade word in the answer is the engine's, wherever the sentence carries it
        decision["answer"] = re.sub(r"\((established|indicative|screening|not established)\)",
                                    f"({decision['grade'].replace('_', ' ')})", decision["answer"], count=1)
    for k in ("conditions", "what_would_change_it"):
        got = [str(x) for x in (raw_d.get(k) or []) if isinstance(x, str) and x.strip()]
        if got:
            decision[k] = got[:6]
    requests = [{"what": str(r.get("what")), "why": str(r.get("why") or ""),
                 "effect_on_grade": str(r.get("effect_on_grade") or "")}
                for r in (obj.get("data_requests") or []) if isinstance(r, dict) and r.get("what")][:4]
    consistency = [dict(c) for c in (obj.get("consistency") or []) if isinstance(c, dict) and c.get("note")][:8]
    assumptions = [str(a) for a in (obj.get("assumptions") or []) if isinstance(a, str)] or rules["assumptions"]
    next_steps = [dict(s) for s in (obj.get("next_steps") or []) if isinstance(s, dict) and s.get("tool")][:4]
    return {"findings": findings, "consistency": consistency or rules["consistency"], "decision": decision,
            "data_requests": requests or rules["data_requests"], "assumptions": assumptions,
            "next_steps": next_steps, "written_by": "model", "dropped": dropped, "reanchored": reanchored,
            "primary_step": rules.get("primary_step")}


#: What is not a claim in a sentence: a return period ("100-year", "T = 100"), a percentage, a date, a year, a
#: timescale ("SPI-12", "3-month"), a station id with digits, a confidence level.
_NOT_A_CLAIM = re.compile(r"\b\d+\s*-?\s*(?:year|yr|month|day)s?\b|\bT\s*=\s*\d+|\d+(?:\.\d+)?\s*%|"
                          r"\b\d{4}-\d{2}-\d{2}\b|\b(?:spi|spei|sgi)-?\d+\b|\b[A-Za-z]+\d+[A-Za-z\d]*\b", re.I)


def _claimed_numbers(text: str) -> list[float]:
    """The numbers a sentence claims about the data: what is left after the labels a reader does not check."""
    cleaned = _NOT_A_CLAIM.sub(" ", text)
    out: list[float] = []
    for t in _NUMBER.findall(cleaned):
        try:
            v = float(t.replace(",", ""))
        except ValueError:
            continue
        if 1800 <= v <= 2100 and v.is_integer():
            continue
        out.append(v)
    return out


def _walk(x: Any) -> list[float]:
    out: list[float] = []
    stack = [x]
    while stack and len(out) < 200:
        item = stack.pop()
        if _is_number(item):
            out.append(float(item))
        elif isinstance(item, dict):
            stack.extend(v for k, v in item.items() if k not in ("series", "points", "samples"))
        elif isinstance(item, (list, tuple)):
            stack.extend(item[:64])
    return out


def _basis_values(ws: Workspace, bases: list[str]) -> list[float]:
    out: list[float] = []
    for b in bases:
        v = resolve_basis(ws, b)
        if _is_number(v):
            out.append(float(v))
        elif isinstance(v, (list, dict)):
            out.extend(_walk(v))
    return out


def interpret(ws: Workspace, model: Model | None) -> dict[str, Any]:
    """Write ``ws.findings`` from the run: the rules, refined by the model when one is present."""
    if not ws.run:
        ws.findings = None
        return {}
    rules = rules_findings(ws)
    out = rules
    if model:
        system, context = interpreter_context(ws)
        obj = model.call_json("interpreter", system, context)
        if isinstance(obj, dict):
            out = validate_findings(ws, obj, rules=rules)
        else:
            out = dict(rules)
            out["note"] = "the model gave no findings; the rules stand"
    ws.findings = out
    d = out["decision"]
    ws.event("interpreter", "findings", f"{len(out['findings'])} finding(s) ({out['written_by']}), "
             f"grade {d.get('grade')}" + (f", {out['dropped']} dropped by the checks" if out.get("dropped") else "")
             + (f", {len(out['data_requests'])} data request(s)" if out.get("data_requests") else ""))
    return out
