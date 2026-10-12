"""Evaluate finished studies: how well the crew did, where its time went, and how runs compare.

A study bundle (``aquascope studio --out DIR``, a recorded showcase study, the Explorer's export)
carries its whole run in ``workspace.json``: the events of every role with a timestamp, each step
and gate, the Critic's checks, the report, and the model ledger. Everything here reads that file and
nothing else; no model is called and nothing is fetched.

- :func:`scorecard`: one study, scored. Outcome (status, grade, headline), run health (steps and
  gates passed, failed and skipped, fallbacks, replans), the Critic's checks, report quality on the
  six axes of :mod:`aquascope.gym.reports`, plan accuracy against a HydroGym reference case when
  one is named, and cost (time, tokens, USD, by role).
- :func:`trace`: one study as a timeline. The coordinator's phases with their durations, each step
  with its tool, time and gates, the model calls per role, and every event.
- :func:`stats`: many studies at once, grouped by playbook, model, date or grade: the grade mix,
  mean report score, gate failure and skip rates by check, time and cost.

The CLI face is ``aquascope eval score | trace | stats``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

#: The coordinator's phases, in the order a study goes through them.
PHASES = ("scouting", "planning", "review", "running", "critique", "authoring", "done")

#: What ``stats`` can group by.
GROUP_BY = ("playbook", "model", "date", "grade", "none")


# ── loading ──────────────────────────────────────────────────────────────────


def load_study(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Read a study from its bundle directory or its ``workspace.json``.

    Returns ``(workspace, meta, directory)``. ``meta`` is the recording's ``meta.json`` when one sits
    beside the workspace (showcase studies carry their model, tokens, cost and seconds there), else ``{}``.
    """
    p = Path(path)
    ws_path = p / "workspace.json" if p.is_dir() else p
    if not ws_path.exists():
        raise FileNotFoundError(f"no workspace.json at {p}")
    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    meta_path = ws_path.parent / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    return ws, meta, ws_path.parent


def find_studies(paths: list[str | Path]) -> list[Path]:
    """Every study directory under ``paths`` (a directory holding ``workspace.json``), sorted, no repeats."""
    found: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name == "workspace.json":
            found.add(p.parent)
        elif (p / "workspace.json").exists():
            found.add(p)
        elif p.is_dir():
            found.update(ws.parent for ws in p.rglob("workspace.json"))
    return sorted(found)


# ── small helpers ────────────────────────────────────────────────────────────


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds(start: Any, end: Any) -> float | None:
    a, b = _time(start), _time(end)
    return round((b - a).total_seconds(), 1) if a and b else None


def _gate_state(gate: dict[str, Any]) -> str:
    if gate.get("skipped"):
        return "skipped"
    return "passed" if gate.get("passed") else "failed"


def _results(ws: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((ws.get("run") or {}).get("results")) or [])


def _ledger(ws: dict[str, Any], meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ledger = ws.get("ledger") or {}
    if not ledger:
        ledger = ((meta.get("tokens") or {}).get("by_role")) or {}
    return {role: dict(row) for role, row in ledger.items() if isinstance(row, dict)}


def _headline(answer: str | None, limit: int = 160) -> str:
    text = " ".join(str(answer or "").split())
    first = text.split(". ")[0]
    return first if len(first) <= limit else first[: limit - 1] + "…"


# ── trace ────────────────────────────────────────────────────────────────────


def trace(ws: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """One study as a timeline: phases, steps, model calls per role and every event.

    Phase durations run from one coordinator status to the next. ``review`` includes any time spent
    waiting for the person to approve the plan, so it is reported but kept out of ``slowest_phase``.
    """
    meta = meta or {}
    events = [e for e in (ws.get("events") or []) if isinstance(e, dict)]
    t0 = _time(events[0].get("at")) if events else _time(ws.get("created"))
    rows = []
    for e in events:
        at = _time(e.get("at"))
        rows.append({
            "t": round((at - t0).total_seconds(), 1) if at and t0 else None,
            "role": e.get("role"), "step": e.get("step"), "event": e.get("event"), "detail": e.get("detail") or "",
        })

    marks = [(e.get("detail"), e.get("at")) for e in events if e.get("role") == "coordinator"
             and e.get("event") == "status"]
    end_at = events[-1].get("at") if events else None
    phases: list[dict[str, Any]] = []
    for i, (name, at) in enumerate(marks):
        if name == "done":
            continue
        nxt = marks[i + 1][1] if i + 1 < len(marks) else end_at
        phases.append({"phase": name, "seconds": _seconds(at, nxt) or 0.0})

    starts: dict[str, Any] = {}
    step_seconds: dict[str, float | None] = {}
    for e in events:
        if e.get("role") != "runner" or not e.get("step"):
            continue
        if e.get("event") == "start":
            starts[e["step"]] = e.get("at")
        elif e.get("event") in ("done", "failed", "error") and e["step"] in starts:
            step_seconds[e["step"]] = _seconds(starts[e["step"]], e.get("at"))

    steps: list[dict[str, Any]] = []
    for r in _results(ws):
        gates = [g for g in (r.get("gates") or []) if isinstance(g, dict)]
        states = Counter(_gate_state(g) for g in gates)
        steps.append({
            "id": r.get("id"), "tool": r.get("tool"), "seconds": step_seconds.get(str(r.get("id"))),
            "ok": bool(r.get("ok")), "skipped": bool(r.get("skipped")), "fallback_used": bool(r.get("fallback_used")),
            "gates": {"passed": states["passed"], "failed": states["failed"], "skipped": states["skipped"]},
            "not_passed": [{"check": g.get("check"), "state": _gate_state(g),
                            "detail": str(g.get("detail") or "").removeprefix("skipped: ")}
                           for g in gates if _gate_state(g) != "passed"],
            "error": r.get("error"),
        })

    timed = [p for p in phases if p["phase"] != "review"]
    slowest_step = max((s for s in steps if s["seconds"] is not None), key=lambda s: s["seconds"], default=None)
    run = ws.get("run") or {}
    total = (_seconds(events[0].get("at"), end_at) if events else None) or meta.get("seconds")
    return {
        "study": ws.get("id"),
        "question": (ws.get("brief") or {}).get("problem") or (ws.get("study") or {}).get("question"),
        "seconds": total,
        "run_seconds": _seconds(run.get("started"), run.get("finished")),
        "phases": phases,
        "slowest_phase": max(timed, key=lambda p: p["seconds"], default=None),
        "steps": steps,
        "slowest_step": slowest_step and {"id": slowest_step["id"], "tool": slowest_step["tool"],
                                          "seconds": slowest_step["seconds"]},
        "model_calls": _ledger(ws, meta),
        "events": rows,
    }


# ── scorecard ────────────────────────────────────────────────────────────────


def _report_reference(study_dir: Path | None):
    """The report reference whose ``study`` is this directory's name, if the package has one."""
    if study_dir is None:
        return None
    from aquascope.gym import reports as gru

    return next((r for r in gru.load_references() if r.study == study_dir.name), None)


def scorecard(ws: dict[str, Any], meta: dict[str, Any] | None = None, *, study_dir: Path | None = None,
              plan_case: str | None = None, report_reference: Any = None) -> dict[str, Any]:
    """One study, scored with no model: outcome, run health, Critic, report quality, plan accuracy, cost.

    ``plan_case`` names a HydroGym Phase 2 reference case (``aquascope gym plans list``); the study's plan
    is then scored against it. ``report_reference`` is a report reference; by default the one whose
    ``study`` matches ``study_dir``'s name is used when the package has one.
    """
    meta = meta or {}
    brief = ws.get("brief") or {}
    study = ws.get("study") or {}
    report = ws.get("report") or {}
    run = ws.get("run") or {}
    critique = ws.get("critique") or {}
    results = _results(ws)

    gates = Counter(_gate_state(g) for r in results for g in (r.get("gates") or []) if isinstance(g, dict))
    summary = run.get("summary") or {}
    checks = [c for c in (critique.get("checks") or []) if isinstance(c, dict)]

    from aquascope.gym import reports as gru

    reference = report_reference if report_reference is not None else _report_reference(study_dir)
    try:
        quality = gru.score_report(ws, reference=reference)
        report_q = {"mean": quality.get("mean"), "dimensions": quality.get("dimensions") or {},
                    "reference": (quality.get("reference") or {}).get("score") if reference else None,
                    "numbers_without_evidence": len((quality.get("metrics") or {}).get("numbers_without_evidence")
                                                    or [])}
    except Exception as exc:  # a malformed or partial bundle still gets the rest of its card
        report_q = {"mean": None, "dimensions": {}, "reference": None, "error": str(exc)}

    plan_q = None
    if plan_case:
        from aquascope.gym import plans as gp

        ref = gp.load_reference(plan_case)
        scored = gp.score_plan(ref, gp.candidate_from(ws))
        plan_q = {"case": ref.id, "score": scored.get("score"), "tools": scored.get("coverage_tools"),
                  "methods": scored.get("coverage_methods"), "gates": scored.get("coverage_gates"),
                  "extraneous": scored.get("extraneous"), "forbidden": scored.get("forbidden_used"),
                  "decline_correct": scored.get("decline_correct"), "explain": scored.get("explain") or []}

    ledger = _ledger(ws, meta)
    prompt = sum(int(r.get("prompt_tokens") or 0) for r in ledger.values())
    completion = sum(int(r.get("completion_tokens") or 0) for r in ledger.values())
    usd_rows = [float(r["cost_usd"]) for r in ledger.values() if r.get("cost_usd") is not None]
    usd = meta.get("usd") if meta.get("usd") is not None else (round(sum(usd_rows), 4) if usd_rows else None)
    t = trace(ws, meta)

    return {
        "study": ws.get("id"),
        "dir": str(study_dir) if study_dir else None,
        "question": brief.get("problem") or study.get("question"),
        "playbook": brief.get("playbook"),
        "branch": (study.get("plan") or {}).get("branch"),
        "date": str(ws.get("created") or "")[:10],
        "model": ws.get("model") or meta.get("model"),
        "provider": ws.get("provider") or meta.get("provider"),
        "aquascope_version": study.get("aquascope_version"),
        "outcome": {"status": ws.get("status"), "grade": report.get("grade"),
                    "headline": _headline(report.get("answer")), "declined": ws.get("declined_reason")},
        "run": {"planned": summary.get("planned", len(results)), "ok": summary.get("ok"),
                "failed": summary.get("failed"), "skipped": summary.get("skipped"),
                "fallbacks": sum(1 for r in results if r.get("fallback_used")), "replans": run.get("replans", 0),
                "gates": {"passed": gates["passed"], "failed": gates["failed"], "skipped": gates["skipped"]}},
        "critic": {"passed": sum(1 for c in checks if c.get("passed")), "total": len(checks),
                   "issues": len(critique.get("issues") or []),
                   "not_established": len(critique.get("not_established") or [])},
        "report": report_q,
        "plan": plan_q,
        "cost": {"seconds": t["seconds"], "slowest_phase": t["slowest_phase"], "slowest_step": t["slowest_step"],
                 "calls": sum(int(r.get("calls") or 0) for r in ledger.values()),
                 "prompt_tokens": prompt, "completion_tokens": completion, "usd": usd, "by_role": ledger},
    }


def score_path(path: str | Path, *, plan_case: str | None = None) -> dict[str, Any]:
    """:func:`scorecard` for a bundle directory or a ``workspace.json``."""
    ws, meta, d = load_study(path)
    return scorecard(ws, meta, study_dir=d, plan_case=plan_case)


# ── stats ────────────────────────────────────────────────────────────────────


def _group_key(card: dict[str, Any], by: str) -> str:
    if by == "none":
        return "all"
    if by == "grade":
        return card["outcome"].get("grade") or "-"
    if by == "model":
        return card.get("model") or "keyless"
    return str(card.get(by) or "-")


def _mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def _median(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def stats(cards: list[dict[str, Any]], *, by: str = "playbook", top: int = 3) -> dict[str, Any]:
    """Aggregate scorecards: one row per group, plus the gate checks that fail or skip most often overall.

    Each row has ``n``, the grade mix, the mean report score, the share of gates passed / failed /
    skipped, the median seconds per study, tokens and USD (total and per study, model runs only).
    """
    if by not in GROUP_BY:
        raise ValueError(f"group by one of {', '.join(GROUP_BY)}, not {by!r}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cards:
        groups[_group_key(c, by)].append(c)

    rows = []
    for key in sorted(groups):
        cs = groups[key]
        gates: Counter[str] = Counter()
        for c in cs:
            gates.update(c["run"]["gates"])
        n_gates = sum(gates.values()) or 1
        usd = [c["cost"]["usd"] for c in cs if c["cost"]["usd"] is not None]
        rows.append({
            by if by != "none" else "group": key,
            "n": len(cs),
            "grades": dict(Counter(c["outcome"].get("grade") or "-" for c in cs)),
            "report_mean": _mean([c["report"].get("mean") for c in cs]),
            "plan_mean": _mean([(c.get("plan") or {}).get("score") for c in cs]),
            "gates_passed": round(gates["passed"] / n_gates, 3),
            "gates_failed": round(gates["failed"] / n_gates, 3),
            "gates_skipped": round(gates["skipped"] / n_gates, 3),
            "critic_passed": _mean([c["critic"]["passed"] / c["critic"]["total"] for c in cs
                                    if c["critic"]["total"]]),
            "median_seconds": _median([c["cost"]["seconds"] for c in cs]),
            "tokens": sum(c["cost"]["prompt_tokens"] + c["cost"]["completion_tokens"] for c in cs),
            "usd": round(sum(usd), 4) if usd else None,
            "usd_per_study": round(sum(usd) / len(usd), 4) if usd else None,
        })
    return {"by": by, "studies": len(cards), "rows": rows, "gate_problems": gate_problems(cards, top=top)}


def gate_problems(cards: list[dict[str, Any]], *, top: int = 3) -> dict[str, list[dict[str, Any]]]:
    """The gate checks that failed or were skipped most often, with how many studies each touched.

    Reads the workspaces again through the cards' ``dir`` so the counts are per check, not per step.
    """
    failed: Counter = Counter()
    skipped: Counter = Counter()
    for c in cards:
        if not c.get("dir"):
            continue
        try:
            ws, _meta, _d = load_study(c["dir"])
        except (OSError, ValueError):
            continue
        seen_f: set[str] = set()
        seen_s: set[str] = set()
        for r in _results(ws):
            for g in r.get("gates") or []:
                state = _gate_state(g) if isinstance(g, dict) else "passed"
                if state == "failed":
                    seen_f.add(str(g.get("check")))
                elif state == "skipped":
                    seen_s.add(str(g.get("check")))
        failed.update(seen_f)
        skipped.update(seen_s)
    as_rows = lambda counts: [{"check": k, "studies": v} for k, v in counts.most_common(top)]  # noqa: E731
    return {"failed": as_rows(failed), "skipped": as_rows(skipped)}


def stats_rows_csv(result: dict[str, Any]) -> str:
    """The rows of :func:`stats` as CSV, the grade mix flattened to ``grade_<name>`` columns."""
    import csv
    import io

    grades = sorted({g for r in result["rows"] for g in r["grades"]})
    key = result["by"] if result["by"] != "none" else "group"
    cols = [key, "n", *[f"grade_{g}" for g in grades], "report_mean", "plan_mean", "gates_passed",
            "gates_failed", "gates_skipped", "critic_passed", "median_seconds", "tokens", "usd", "usd_per_study"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in result["rows"]:
        w.writerow([r.get(key), r["n"], *[r["grades"].get(g, 0) for g in grades]]
                   + [r[c] for c in cols[2 + len(grades):]])
    return buf.getvalue()
