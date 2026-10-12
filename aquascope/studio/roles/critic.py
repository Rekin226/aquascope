"""The Critic: an independent pass over the draft report against the results.

Deterministic first: :func:`aquascope.ai_engine.verify.verify` over the
report's answer and sections against the tool results (with the gates' own
words in the pool, as the Solve team does), the failed gates, the steps that
did not run, the plan's notes. Those make the "what this study does not
establish" list. With a model, one more call reads the sections and the
compact results and returns issues with a section, a severity (``fix`` or
``note``) and the fix. The Author is called again once when anything is a
``fix``, and every failed deterministic check is a fix (:func:`fixes_for`):
the check's detail and the repair it asks for go to the Author with the
model's issues, once; keyless, the Author's template repair drops the
sentences the checks refuse. A report whose critique is still not ok after
that opens with :func:`notice`, one line naming the failed checks.
"""

from __future__ import annotations

from typing import Any

from aquascope.studio.model import Model, compact
from aquascope.studio.prompts import CRITIC
from aquascope.studio.workspace import Workspace

__all__ = ["CHECK_FIXES", "check_issues", "critique", "failed_checks", "findings_checks", "fixes_for",
           "not_established", "notice", "tool_results"]

#: The repair each deterministic check asks of the Author when it fails. Every failed check is a "fix", the
#: trend check included: a "significant" claim against a p above 0.05 is a contradiction, not a note.
CHECK_FIXES: dict[str, str] = {
    "tools_were_used": "No result exists: say so and quote no number.",
    "numbers_come_from_tools": "Remove or replace every number that is in no result; quote only numbers from the "
                               "steps' results.",
    "years_traceable": "Drop the years that are in no result, or say they are general knowledge.",
    "flood_estimate_carries_uncertainty": "Quote the return level with its confidence interval from the result.",
    "trend_matches_the_test": "Call the trend significant only when the test's p is below 0.05 and not significant "
                              "otherwise; keep the p-value as the test reported it.",
    "units_are_named": "Name the unit of the record next to the numbers.",
    "record_is_named": "Name the station or record the numbers come from.",
    "findings_resolve": "Drop or re-anchor the findings whose basis paths point at no result.",
    "decision_in_answer": "Open the answer with the decision's value, its band and its grade word.",
    "stationarity_matches_the_maxima": "Say what the Mann-Kendall test on the annual maxima found, and treat the "
                                       "record as stationary only when it found no trend.",
}


def tool_results(ws: Workspace) -> list[dict[str, Any]]:
    """The run's results in the shape ``verify`` reads, fallbacks included, plus the gates pool."""
    seen: list[dict[str, Any]] = []
    run = ws.run or {}
    for r in run.get("results") or []:
        if r.get("skipped"):
            continue
        seen.append({"name": r.get("tool"), "arguments": r.get("arguments") or {}, "payload": r.get("result"),
                     "ok": bool(r.get("ok"))})
        fb = r.get("fallback")
        if isinstance(fb, dict) and fb.get("tool"):
            seen.append({"name": fb["tool"], "arguments": fb.get("arguments") or {}, "payload": fb.get("result"),
                         "ok": bool(fb.get("ok"))})
    gates = [g for r in (run.get("results") or []) for g in (r.get("gates") or [])]
    all_gates = run.get("gates") or gates
    pool = {"gates": gates, "plan": (ws.study or {}).get("plan") or {},
            # the counts the Author writes ("3 steps ran, 7 of 7 gates passed") are results too
            "n_steps": len(run.get("results") or []), "n_gates": len(all_gates),
            "n_passed": sum(1 for g in all_gates if g.get("passed")),
            "n_failed": sum(1 for g in all_gates if not g.get("passed")),
            "n_not_established": len(not_established(ws)), "n_replans": run.get("replans") or 0}
    seen.append({"name": "gates", "arguments": {}, "payload": pool, "ok": True})
    datasets = [d.to_dict() for d in ws.inventory.datasets] if ws.inventory else []
    stations = [d for d in datasets if d.get("kind") == "station"]
    short = [d for d in stations if (d.get("years") or 0) < 1]
    seen.append({"name": "inventory", "arguments": {}, "ok": True, "payload": {
        "datasets": datasets, "n_datasets": len(datasets), "n_stations": len(stations),
        "n_short": len(short), "n_listed": len(datasets) - len(short)}})
    return seen


def not_established(ws: Workspace) -> list[str]:
    """What the run did not establish, from the gates, the failed steps, the stop and the plan's notes."""
    out: list[str] = []
    run = ws.run or {}
    for g in run.get("failed_gates") or []:
        out.append(f"Step {g.get('step')}, gate {g.get('check')}: {g.get('detail')}")
    for g in run.get("gates") or []:
        if g.get("skipped"):
            out.append(f"Step {g.get('step')}, gate {g.get('check')} skipped: {g.get('detail')}")
    for r in run.get("results") or []:
        if not r.get("ok"):
            out.append(f"Step {r.get('id')} ({r.get('tool')}) did not run: {r.get('error')}")
    if run.get("stop_reason"):
        out.append(f"The study stopped at {run.get('stopped_at')}: {run['stop_reason']}")
    for n in ((ws.study or {}).get("plan") or {}).get("notes") or []:
        out.append(str(n))
    return out


def failed_checks(critique: dict[str, Any] | None) -> list[str]:
    """The names of the deterministic checks that failed."""
    return [str(c.get("name") or "check") for c in (critique or {}).get("checks") or [] if not c.get("passed")]


def check_issues(critique: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The failed deterministic checks as issues for the Author's fix round: severity ``fix``, the check's name
    under ``check``, its detail as the text, and the repair it calls for."""
    out: list[dict[str, Any]] = []
    for c in (critique or {}).get("checks") or []:
        if c.get("passed"):
            continue
        name = str(c.get("name") or "check")
        out.append({"section": "*", "severity": "fix", "check": name, "text": str(c.get("detail") or name),
                    "fix": CHECK_FIXES.get(name, "Correct the draft so the check passes.")})
    return out


def fixes_for(critique: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The Author's fix list: the model's ``fix`` issues, then every failed check, each once."""
    issues = [dict(i) for i in (critique or {}).get("issues") or [] if i.get("severity") == "fix"]
    return issues + check_issues(critique)


def notice(critique: dict[str, Any] | None) -> str | None:
    """One line for the top of a report whose critique is not ok, naming the failed checks; None when it is ok.
    Written without digits, so the checks never read the notice itself as a claim."""
    if critique is None or critique.get("ok", True):
        return None
    failed = failed_checks(critique)
    if failed:
        return (f"Notice: this report did not pass the Critic's checks ({', '.join(failed)}); read its numbers "
                "with the list of what this study does not establish.")
    sections = sorted({str(i.get("section") or "summary") for i in critique.get("issues") or []
                       if i.get("severity") == "fix"})
    return (f"Notice: the Critic's fix requests on {', '.join(sections) or 'the draft'} were not all resolved; "
            "read the report with the list of what this study does not establish.")


def findings_checks(ws: Workspace) -> list[Any]:
    """Two deterministic checks over the Interpreter's findings (#417): every basis path resolves to a value
    in the results, and the report's answer carries the decision's value and its grade word."""
    from aquascope.ai_engine.verify import Check, _numbers, normalise
    from aquascope.studio.roles.interpreter import resolve_basis

    out: list[Any] = []
    findings = ws.findings or {}
    rows = findings.get("findings") or []
    if rows:
        bad = [f.get("id") for f in rows if not any(resolve_basis(ws, b) is not None for b in (f.get("basis") or []))]
        out.append(Check("findings_resolve", not bad,
                         "" if not bad else f"Finding(s) {', '.join(str(b) for b in bad)} point at no result."))
    decision = findings.get("decision") or {}
    value, grade = decision.get("value"), decision.get("grade")
    answer = normalise(str((ws.report or {}).get("answer") or ""))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and answer:
        head = answer.split("\n\n")[-1] if answer.startswith("Notice:") else answer
        nums = _numbers(head, claims_only=True) + _numbers(head)   # a percentage is a claim here (82.5 %)
        has_value = any(abs(n - float(value)) <= max(abs(float(value)), 1e-9) * 0.02 + 1e-9 for n in nums)
        has_grade = bool(grade) and str(grade).replace("_", " ") in head.lower()
        out.append(Check("decision_in_answer", has_value and has_grade,
                         "" if has_value and has_grade else
                         ("The answer does not quote the decision's value " if not has_value else "")
                         + ("The answer does not name the answer's grade " if not has_grade else "")
                         + f"({value:g}, {grade})."))
    return out


def _draft(ws: Workspace) -> str:
    report = ws.report or {}
    answer = str(report.get("answer") or "")
    head = report.get("notice")
    if isinstance(head, str) and head and answer.startswith(head):
        answer = answer[len(head):].lstrip()
    parts = [answer]
    for s in report.get("sections") or []:
        if s.get("id") in ("appendix", "references"):
            continue
        parts.append(str(s.get("text") or ""))
    return "\n\n".join(p for p in parts if p)


def critique(ws: Workspace, model: Model | None) -> dict[str, Any]:
    """Write ``ws.critique`` (``checks``, ``issues``, ``not_established``) from the draft in ``ws.report``."""
    from aquascope.ai_engine.verify import verify

    draft = _draft(ws)
    results = tool_results(ws)
    checks = verify(draft, results, question=ws.brief.problem)
    checks.checks.extend(findings_checks(ws))
    missing = not_established(ws)
    for c in checks.failed:
        missing.append(c.detail or c.name)
    issues: list[dict[str, Any]] = []
    if model:
        report = ws.report or {}
        obj = model.call_json("critic", CRITIC, {
            "problem": ws.brief.problem, "answer": report.get("answer"),
            "sections": [{"id": s.get("id"), "title": s.get("title"), "text": s.get("text")}
                         for s in report.get("sections") or [] if s.get("id") not in ("appendix", "references")],
            "key_numbers": report.get("key_numbers"),
            "results": [{"id": r.get("id"), "tool": r.get("tool"), "ok": r.get("ok"), "error": r.get("error"),
                         "gates": r.get("gates"), "result": compact(r.get("result"))}
                        for r in (ws.run or {}).get("results") or []],
            "not_established": missing,
            "checks": [c.to_dict() for c in checks.failed],
        })
        for raw in (obj or {}).get("issues") or []:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            severity = "fix" if str(raw.get("severity") or "").lower() == "fix" else "note"
            issues.append({"section": str(raw.get("section") or "summary"), "severity": severity,
                           "text": str(raw["text"]), "fix": str(raw.get("fix") or "")})
    ws.critique = {"ok": checks.ok and not any(i["severity"] == "fix" for i in issues),
                   "checks": checks.to_dict()["checks"], "issues": issues, "not_established": missing,
                   "failed": [c.name for c in checks.failed]}
    ws.event("critic", "checks", f"{len(checks.checks) - len(checks.failed)} of {len(checks.checks)} checks passed"
             + (f" (failed: {', '.join(c.name for c in checks.failed)})" if checks.failed else "")
             + (f"; {len(issues)} issue(s) from the model" if issues else ""))
    return ws.critique
