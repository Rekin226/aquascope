"""The Analysts: run the plan with its gates, draw the figures as results land, replan once.

The runner is :func:`aquascope.study.run_study`, unchanged: every step in
order, its gates after it, a fallback once, a stop with the reason. What
this module adds is the table loader for the client's uploads
(``load_table``), the figure and table makers called after every step result
(when the deliverables package is importable; skipped with an event
otherwise), and the bounded replan of the Solve team: a branch replan
through the playbook when the plan asks for one, else a Specialist's
proposal from the model, validated against the catalogue before it runs.

The steps run sequentially. ``run_study`` owns the dependency order, the
result references (``{{ result.s2.x }}``) and the stop-on-gate semantics; a
parallel pass over independent steps would have to reimplement those, and
the network fetch behind each step is what takes the time, not Python. So
no thread pool here, in Pyodide or out of it.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any

from aquascope import study_map
from aquascope.studio import catalogue
from aquascope.studio.model import Model, compact
from aquascope.studio.prompts import SPECIALIST
from aquascope.studio.workspace import Artifact, Workspace
from aquascope.study import Study, StudyRun, run_study

__all__ = ["KINDS_BY_METHOD", "analyze_station_full", "load_table", "prior_run", "run"]

#: The figure kinds a station step draws when its plan names a method (E); no method draws every kind of the tool.
KINDS_BY_METHOD: dict[str, list[str]] = {
    "trend_mann_kendall": ["series", "trend"],
    "at_site_flood_frequency": ["annual_maxima", "frequency_curve"],
    "flow_duration": ["fdc"],
    "low_flow_frequency": ["fdc"],
    "groundwater_trend": ["series", "trend"],
}
#: Payload keys stripped before a result goes into the workspace (the figures read them first).
_BULK_KEYS = ("series",)


# ── the station analysis with its series kept for the figures ───────────────


def analyze_station_full(source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False,
                         variable: str | None = None, return_periods: list[float] | None = None) -> dict[str, Any]:
    """``aquascope.explore.analyze_station`` with the daily series and the full flow-duration curve kept in the
    payload (the runner's own ``analyze_station`` drops them, so the hydrograph, trend and FDC figures never
    drew); the bootstrap band as the runner adds it. The Analysts strip the series before the payload is stored."""
    from aquascope import mcp_server as registry
    from aquascope.explore import analyze_station as _analyze
    from aquascope.explore import flood_ci

    sources = getattr(registry, "SOURCES", None)
    variables = getattr(registry, "VARIABLES", None)
    if sources is not None and source not in sources:
        return {"error": f"unknown source {source!r}"}
    if variable and variables is not None and variable not in variables:
        return {"error": f"unknown variable {variable!r}; allowed: {list(variables)}"}
    store: dict[str, Any] = {}
    extra = {"return_periods": return_periods} if return_periods else {}
    res = _analyze(source, station_id, years=int(years) if years else None, store=store, variable=variable, **extra)
    if bootstrap_ci and res.get("ffa") and store.get("series") is not None:
        try:
            ci = flood_ci(store["series"], **extra)
            res["ffa"]["fits"]["gev_bootstrap"] = {
                k: ci[k] for k in ("q", "ci", "params", "n_bootstrap", "n_bootstrap_discarded") if k in ci
            }
            res.setdefault("methods", []).append(ci["method"])
        except Exception as exc:  # noqa: BLE001 - the band is optional
            res.setdefault("notes", []).append(f"bootstrap CI failed: {exc}")
    return res


def _name_stations(ws: Workspace, run: StudyRun) -> None:
    """``station_name`` (and ``name`` when the agency gave none) from the inventory on every station payload,
    so captions, tables and the template prose say "Kingston (uk_ea 8496ce69...)" and not the id alone (D)."""
    inv = ws.inventory
    if inv is None:
        return
    names = {(d.source, d.station_id): d.name for d in inv.datasets if d.station_id and d.name
             and d.name != d.station_id}
    for r in run.results:
        payloads = [r.get("result")]
        fb = r.get("fallback")
        if isinstance(fb, dict):
            payloads.append(fb.get("result"))
        for p in payloads:
            if not isinstance(p, dict) or not p.get("source") or not p.get("station_id"):
                continue
            name = names.get((str(p["source"]), str(p["station_id"])))
            if name:
                p.setdefault("station_name", name)
                if not p.get("name"):
                    p["name"] = name


def _report_the_asked_trend(ws: Workspace, run: StudyRun, study: Study) -> None:
    """A flood question ("is it getting worse?") is answered with the Mann-Kendall test on the annual maxima,
    not on the annual means: every station payload is marked with the trend the report quotes
    (``trend_reported``), so the key numbers, the sentences and the trend figure use the same series."""
    from aquascope.trend_series import is_flood_question, mark_reported_trend

    plan = study.plan or {}
    flood = is_flood_question(ws.brief.kind, ws.brief.problem, ws.brief.playbook or plan.get("playbook"))
    if not flood:
        return
    for r in run.results:
        mark_reported_trend(r.get("result"), flood=True)
        fb = r.get("fallback")
        if isinstance(fb, dict):
            mark_reported_trend(fb.get("result"), flood=True)


def _ask_for_the_return_period(ws: Workspace, study: Study) -> None:
    """A flood step reports the fits at 2, 5, 10, 25, 50 and 100 years unless told otherwise; when the brief
    asks for another T (200 years, 20 years) the step is told, so the gates and the prose find it."""
    rp = (ws.brief.intake or {}).get("return_period")
    try:
        rp_f = float(rp)
    except (TypeError, ValueError):
        return
    if rp_f < 1.01:
        return
    from aquascope.explore import RETURN_PERIODS

    for step in study.steps:
        if step.tool not in ("analyze_station", "flood_frequency"):
            continue
        given = step.arguments.get("return_periods")
        periods = [float(x) for x in (given or RETURN_PERIODS) if isinstance(x, (int, float))]
        if rp_f in periods:
            continue
        periods.append(rp_f)
        step.arguments["return_periods"] = [int(v) if float(v).is_integer() else v for v in sorted(set(periods))]
    for step in study.steps:
        for gate in step.expects or []:
            if isinstance(gate, dict) and gate.get("check") == "max_return_period_factor" \
                    and gate.get("return_period") is None:
                gate["return_period"] = int(rp_f) if rp_f.is_integer() else rp_f


def _inherit_units(ws: Workspace, run: StudyRun, study: Study) -> None:
    """A workbench step that took its table from an earlier step (``from_step``) inherits that payload's unit
    and variable when it reports none, so "496.7" becomes "496.7 m3/s" in the prose, the tables and the figures."""
    by_id: dict[str, dict[str, Any]] = {}
    for r in run.results:
        if isinstance(r.get("result"), dict) and r.get("id"):
            by_id[str(r["id"])] = r["result"]
    for step in study.steps:
        src = (step.arguments or {}).get("from_step")
        if not src or not step.id:
            continue
        source, target = by_id.get(str(src)), by_id.get(str(step.id))
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        for key in ("unit", "variable"):
            if source.get(key) and not target.get(key):
                target[key] = source[key]


def _strip_bulk(run: StudyRun, study: Study) -> None:
    """Drop the daily series from the stored results (a workspace must not carry 50k points) and rehash."""
    for r in run.results:
        recs = [r]
        if isinstance(r.get("fallback"), dict):
            recs.append(r["fallback"])
        for rec in recs:
            p = rec.get("result")
            if not isinstance(p, dict) or not any(k in p for k in _BULK_KEYS):
                continue
            for k in _BULK_KEYS:
                p.pop(k, None)
            blob = json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
            rec["sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        sid = str(r.get("id"))
        if sid in study.results and r.get("sha256"):
            study.results[sid]["sha256"] = r["sha256"]


# ── the table loader ────────────────────────────────────────────────────────


def load_table(ws: Workspace, table: str, value_column: str | None = None,
               datetime_column: str | None = None) -> dict[str, Any]:
    """A client's upload as a step payload: ``{"series": {"t", "v"}, "n", "years", "unit", "columns", ...}`` for a
    datetime/value table, ``{"samples": [rows], "n", "columns"}`` for anything else."""
    import pandas as pd

    from aquascope import ingest

    csv = ws.tables.get(table)
    if csv is None:
        alt = next((k for k in ws.tables if k.endswith(table) or table.endswith(k)), None)
        if alt is None:
            return {"error": f"no table {table!r} in the workspace; the uploads are {sorted(ws.tables) or 'none'}"}
        table, csv = alt, ws.tables[alt]
    df = pd.read_csv(io.StringIO(csv), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    columns = list(df.columns)
    out: dict[str, Any] = {"table": table, "columns": columns, "source": "upload", "station_id": table,
                           "name": table, "unit": None}
    try:
        mapping = ingest.guess_mapping(df)
        if value_column:
            if value_column not in df.columns:
                return {"error": f"no column {value_column!r} in {table}; the columns are {columns}", **out}
            from aquascope.studio.roles.scout import choose_column

            mapping = choose_column(df, mapping, value_column)
        if datetime_column:
            if datetime_column not in df.columns:
                return {"error": f"no column {datetime_column!r} in {table}; the columns are {columns}", **out}
            mapping.datetime_column = datetime_column
        raw = ingest.apply_mapping(df, mapping)
        series, qa = ingest.qa_series(raw, n_rows_in=len(df))
    except (ValueError, TypeError, KeyError) as exc:
        rows = [{k: (None if v != v else v) for k, v in r.items()} for r in df.to_dict("records")]
        out.update({"n": len(rows), "samples": rows, "note": f"not a datetime/value series ({exc}); the rows are "
                                                             "passed as samples"})
        return out
    out.update({
        "variable": mapping.variable, "unit": mapping.unit or None, "n": int(qa.n_values),
        "years": round(qa.n_days_span / 365.25, 2) if qa.n_days_span else 0.0,
        "start": qa.start, "end": qa.end,
        "series": {"t": [t.isoformat() for t in series.index], "v": [float(v) for v in series.values]},
        "stats": {"mean": float(series.mean()), "min": float(series.min()), "max": float(series.max())} if len(series)
        else {},
        "qa": qa.to_dict(), "mapping": mapping.to_dict(),
    })
    return out


# ── figures and tables per step ─────────────────────────────────────────────


def _kind_of(item: Any, sid: str) -> str | None:
    """The figure kind of a maker's artifact: ``meta["kind"]`` or the ``fig-{step}-{kind}`` id."""
    meta = getattr(item, "meta", None) if not isinstance(item, dict) else item.get("meta")
    if isinstance(meta, dict) and meta.get("kind"):
        return str(meta["kind"])
    ident = str(getattr(item, "id", None) if not isinstance(item, dict) else item.get("id") or "")
    prefix = f"fig-{sid}-"
    if ident.startswith(prefix):
        return ident[len(prefix):].removesuffix("-svg")
    return None


def _as_artifact(item: Any) -> Artifact | None:
    if isinstance(item, Artifact):
        return item
    if isinstance(item, dict) and item.get("id"):
        return Artifact.from_dict(item)
    return None


def _kinds_for(study: Study, sid: str) -> list[str] | None:
    step = study.step_by_id(sid)
    if step is None or not step.method:
        return None
    return KINDS_BY_METHOD.get(step.method)


def _draw(ws: Workspace, run: StudyRun, drawn: set[str], on_artifact: Any, study: Study | None = None) -> None:
    """Call the deliverables' makers on every result not drawn yet (the figure kinds limited by the step's
    method when it names one); an event when they cannot be."""
    try:
        from aquascope.studio.deliverables.figures import figures_for
        from aquascope.studio.deliverables.tables import tables_for
    except ImportError as exc:
        if "deliverables" not in drawn:
            drawn.add("deliverables")
            ws.event("analyst", "figures_skipped", f"deliverables not importable: {exc}")
        return
    for r in run.results:
        entries = [(r.get("id"), r)]
        fb = r.get("fallback")
        if isinstance(fb, dict) and fb.get("tool"):
            entries.append((f"{r.get('id')}.fallback", fb))
        for sid, rec in entries:
            if not sid or sid in drawn or not rec.get("ok") or not isinstance(rec.get("result"), dict):
                continue
            drawn.add(sid)
            payload = rec["result"]
            made: list[Any] = []
            kinds = _kinds_for(study, sid) if study is not None else None
            try:
                if kinds is not None:
                    try:
                        figs = figures_for(sid, rec.get("tool"), payload, unit=payload.get("unit"), site=ws.site,
                                           kinds=kinds)
                    except TypeError:  # a maker without the kinds argument: filter what it drew
                        figs = [a for a in (figures_for(sid, rec.get("tool"), payload, unit=payload.get("unit"),
                                                        site=ws.site) or [])
                                if _kind_of(a, sid) in kinds]
                else:
                    figs = figures_for(sid, rec.get("tool"), payload, unit=payload.get("unit"), site=ws.site)
                made += list(figs or [])
            except Exception as exc:  # noqa: BLE001 - a figure that cannot be drawn is a note, not a stop
                ws.event("analyst", "figures_skipped", f"{type(exc).__name__}: {exc}", step=sid)
            try:
                made += list(tables_for(sid, rec.get("tool"), payload) or [])
            except Exception as exc:  # noqa: BLE001
                ws.event("analyst", "figures_skipped", f"tables: {type(exc).__name__}: {exc}", step=sid)
            n = 0
            for item in made:
                art = _as_artifact(item)
                if art is None:
                    continue
                art.step = art.step or sid
                ws.add_artifact(art)
                n += 1
                if on_artifact:
                    try:
                        on_artifact(art)
                    except Exception:  # noqa: BLE001 - a face's callback must not stop the study
                        pass
            if n:
                ws.event("analyst", "figures", f"{n} artifact(s)", step=sid)


# ── the run ─────────────────────────────────────────────────────────────────


def prior_run(ws: Workspace) -> StudyRun | None:
    """The last run as a StudyRun, so a re-run reuses the steps that passed (same id, tool and arguments)."""
    if not ws.run or not ws.run.get("results"):
        return None
    study = ws.study_obj() or Study(question=ws.brief.problem)
    return StudyRun(study=study, results=[dict(r) for r in ws.run["results"]],
                    started=str(ws.run.get("started") or ""), finished=str(ws.run.get("finished") or ""),
                    ok=bool(ws.run.get("ok")), stopped_at=ws.run.get("stopped_at"),
                    stop_reason=ws.run.get("stop_reason"))


def _reusable(prior: StudyRun | None, study: Study) -> StudyRun | None:
    """The prior run without the results whose step now carries different gates (a new return period changes
    the gate, not the arguments; the runner would otherwise keep the old gate outcome)."""
    if prior is None:
        return None
    keep = []
    for r in prior.results:
        old = prior.study.step_by_id(str(r.get("id")))
        new = study.step_by_id(str(r.get("id")))
        if old is not None and new is not None and old.expects != new.expects:
            continue
        keep.append(r)
    if len(keep) == len(prior.results):
        return prior
    return StudyRun(study=prior.study, results=keep, started=prior.started, finished=prior.finished, ok=prior.ok,
                    stopped_at=prior.stopped_at, stop_reason=prior.stop_reason)


def _carry(old: Study, new: Study) -> Study:
    """A replanned study keeps the version-3 plan block of the one it replaces."""
    from aquascope.studio.roles.methodologist import _outputs_for

    new.version = 3
    new.plan = dict(new.plan or {})
    for key in ("author", "objective", "decision", "methodology", "assumptions", "alternatives",
                "limitations_expected", "citations"):
        if key not in new.plan and (old.plan or {}).get(key) is not None:
            new.plan[key] = old.plan[key]
    new.plan.setdefault("methodology", [s.rationale or s.tool for s in new.steps])
    new.question = old.question
    for s in new.steps:
        if not s.outputs:
            s.outputs = _outputs_for(s)
    return new


def run(ws: Workspace, model: Model | None, *, tools: dict[str, Any] | None = None, on_artifact: Any = None,
        max_replans: int = 1, prior: StudyRun | None = None, reuse: list[str] | None = None) -> StudyRun:
    """Run ``ws.study`` with gates, figures and one bounded replan; write ``ws.run`` and the study's results."""
    from aquascope import playbooks as pbk

    study = ws.study_obj()
    if study is None or not study.steps:
        raise ValueError("there is no plan to run")
    text = ws.brief.problem
    intake = dict(ws.brief.intake)
    recon = dict(ws.inventory.recon) if ws.inventory else {}
    plan = study.plan or {}
    known = {p["id"] for p in pbk.list_playbooks() if "error" not in p}
    pb = pbk.load(plan["playbook"]) if plan.get("playbook") in known else None
    kind = plan.get("playbook")

    def say(event: dict[str, Any]) -> None:
        ws.event(str(event.get("role") or "runner"), str(event.get("event") or ""), str(event.get("detail") or ""),
                 step=event.get("step"))

    callables = catalogue.callables({**(tools or {}), catalogue.LOAD_TABLE: functools.partial(load_table, ws)})
    if "analyze_station" not in (tools or {}):
        # Only the registry's own analyze_station is swapped for the one that keeps the series; a caller's tool
        # (the browser's, a test's) stands.
        try:
            from aquascope.mcp_server import analyze_station as registry_analyze
        except ImportError:  # pragma: no cover
            registry_analyze = None
        if callables.get("analyze_station") is registry_analyze:
            callables["analyze_station"] = analyze_station_full
    prior = _reusable(prior, study)
    drawn: set[str] = set()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ws.event("analyst", "start", f"{len(study.steps)} step(s)")
    _ask_for_the_return_period(ws, study)
    run_ = run_study(study, on_event=say, prior=prior, tools=callables, reuse=reuse)  # reuse: a steered rerun
    _name_stations(ws, run_)
    _inherit_units(ws, run_, study)
    _report_the_asked_trend(ws, run_, study)  # study-trust-fixes: flood questions quote the annual-maxima trend
    _draw(ws, run_, drawn, on_artifact, study)
    study_map.publish(ws, run_.results, on_artifact)  # the study on the map (study_map.geojson)
    replans = 0
    #: Recovery attempts per step id: each failed step gets its branch replan or its Specialist fallback at most
    #: ``max_replans`` times, then it stays not established and the crew moves to the next failed step.
    attempted: dict[str, int] = {}

    def rerun(new_study: Study) -> StudyRun:
        nonlocal study
        study = new_study
        out = run_study(study, on_event=say, prior=run_, tools=callables)
        _name_stations(ws, out)
        _inherit_units(ws, out, study)
        _report_the_asked_trend(ws, out, study)
        _draw(ws, out, drawn, on_artifact, study)
        study_map.publish(ws, out.results, on_artifact)
        return out

    while not run_.stop_reason:
        if run_.replan and attempted.get(str(run_.replan["step"]), 0) < max_replans:
            sid = str(run_.replan["step"])
            attempted[sid] = attempted.get(sid, 0) + 1
            branch = run_.replan["branch"]
            if pb is None:
                ws.event("analyst", "replan_declined", "no playbook to fill the branch from", step=sid)
                run_.replan = None
                continue
            try:
                new = pbk.plan(pb, recon, intake, branch=branch, problem_text=text)
            except pbk.Declined as exc:
                ws.event("analyst", "replan_declined", exc.reason, step=sid)
                run_.replan = None
                continue
            new = _carry(study, new)
            new.plan["replanned_from"] = {"branch": plan.get("branch"), "step": sid,
                                          "reason": run_.replan.get("reason")}
            ws.event("analyst", "replan", f"branch {branch} after {run_.replan.get('reason')}", step=sid)
            replans += 1
            run_ = rerun(new)
            continue
        if not model:
            break
        pending = [f for f in run_.failed_steps
                   if not f.get("skipped") and attempted.get(str(f["id"]), 0) < max_replans]
        if not pending:
            break
        target = str(pending[0]["id"])
        attempted[target] = attempted.get(target, 0) + 1
        failed = next((r for r in run_.results if r.get("id") == target), None)
        step = study.step_by_id(target)
        if failed is None or step is None:
            continue
        from aquascope.ai_engine.team import SPECIALIST_PROMPTS, _recon_summary

        proposal = model.call_json("analyst", f"{SPECIALIST_PROMPTS.get(kind, SPECIALIST_PROMPTS['default'])}\n"
                                   f"{SPECIALIST}", {
            "problem": text, "playbook": kind, "site": ws.site,
            "failed_step": {"id": step.id, "tool": step.tool, "arguments": step.arguments,
                            "rationale": step.rationale},
            "failed_gates": [g for g in failed.get("gates") or [] if not g.get("passed")],
            "error": failed.get("error"),
            "result": compact(failed.get("result")),
            "earlier_fallback": compact(failed.get("fallback")) if failed.get("fallback") else None,
            "recon": _recon_summary(recon),
            "tools": [{"tool": e["tool"], "arguments": list(e["arguments"]), "gates": e["gates"],
                       "allowed": e.get("allowed") or {}}
                      for e in catalogue.compact(ws.brief.kind)[:20]],
        }, step=step.id)
        if not proposal or not proposal.get("tool"):
            ws.event("analyst", "no_fallback", (proposal or {}).get("rationale")
                     or "the specialist proposed no usable fallback", step=step.id)
            continue
        fb_step = {"tool": str(proposal["tool"]), "arguments": dict(proposal.get("arguments") or {}),
                   "rationale": str(proposal.get("rationale") or "proposed by the specialist after the gate failed"),
                   "expects": [g for g in (proposal.get("expects") or []) if isinstance(g, dict)]}
        ids = {s.id for s in study.steps if s.id}
        from aquascope.studio.roles.methodologist import sufficiency_for_validation

        errors = catalogue.validate_step({"id": f"{step.id}.fallback", **fb_step}, known_ids=ids,
                                         sufficiency=sufficiency_for_validation(ws))
        if errors:
            ws.event("analyst", "no_fallback", "the proposal did not pass the validator: " + "; ".join(errors[:3]),
                     step=step.id)
            continue
        step.fallback = {"step": fb_step}
        study.plan = dict(study.plan or {})
        study.plan.setdefault("replans", []).append({"step": step.id, "reason": pending[0]["reason"],
                                                     "fallback": fb_step})
        ws.event("analyst", "replan", f"fallback {fb_step['tool']}: {fb_step['rationale']}", step=step.id)
        replans += 1
        run_ = rerun(study)

    _strip_bulk(run_, study)
    ws.set_study(study)
    ws.run = {
        "ok": bool(run_.ok), "results": [dict(r) for r in run_.results], "gates": run_.gates,
        "failed_gates": run_.failed_gates, "stopped_at": run_.stopped_at, "stop_reason": run_.stop_reason,
        "started": started, "finished": run_.finished or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replans": replans, "failed_steps": run_.failed_steps, "summary": run_.summary,
    }
    summ = run_.summary
    ws.event("analyst", "gates", f"{len(run_.gates) - len(run_.failed_gates)} of {len(run_.gates)} gates passed; "
             f"{summ['ok']} of {summ['planned']} step(s) established"
             + (f", {summ['failed']} failed" if summ["failed"] else "")
             + (f", {summ['skipped']} skipped" if summ["skipped"] else "")
             + (f"; stopped at {run_.stopped_at}" if run_.stop_reason else ""))
    return run_
