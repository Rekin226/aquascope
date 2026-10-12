"""The Study Desk: a finished study as a document a hydrologist revises, checks and signs.

A report is never right the first time. The reviewer pushes back on its assumptions: use the 200-year flood,
leave out the 2008 maximum (the rating was revised after it), quote Log-Pearson III rather than GEV, use only
the last 50 years. The Desk turns each of those into a *lever* in plain words, applies it to the study (the
steps it touches rerun with their gates, every other result stands), writes the documents again and records a
*revision* with the answer before and after, so the change history travels in the report. Beside the levers it
computes a *sensitivity table* (the design value under the reasonable alternatives) straight from the stored
annual maxima, without a fetch. And it keeps the *review*: comments with their responses, and the sign-off
that fills the prepared, checked and approved lines of the document control.

Everything is a plain function over a :class:`~aquascope.studio.workspace.Workspace` (state in
``ws.desk`` and ``ws.house_style``), so the CLI (``aquascope desk``), the Explorer's worker and the MCP tools
are thin faces over the same code. The reruns go through :class:`aquascope.studio.coordinator.Studio`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aquascope.studio.workspace import Workspace

__all__ = ["LEVERS", "comment", "desk_state", "estimator_of", "levers", "record_revision", "resolve",
           "sensitivity", "sign", "split_changes"]

#: The study-level levers: id -> (label, the step parameter it sets, kind). ``rerun`` levers change step
#: arguments and rerun those steps; ``document`` levers change what the documents quote.
LEVERS: dict[str, tuple[str, str | None, str]] = {
    "return_period": ("Design return period (years)", "return_period", "rerun"),
    "years": ("Record window: only the last N years", "years", "rerun"),
    "exclude_years": ("Leave out these years' floods", "exclude_years", "rerun"),
    "estimator": ("Distribution quoted as the answer", None, "document"),
}

#: The steps a rerun lever applies to.
FLOOD_TOOLS = ("flood_frequency", "analyze_station")

ESTIMATORS = {"gev_lmoments": "GEV (L-moments)", "lp3": "Log-Pearson III", "gev_bootstrap": "GEV (maximum likelihood)"}

#: The statuses a document passes through as people sign it.
STATUS_BY_ROLE = {"prepared_by": "DRAFT", "checked_by": "CHECKED", "approved_by": "ISSUED"}
ROLE_WORDS = {"prepared_by": "Prepared", "checked_by": "Checked", "approved_by": "Approved"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def desk_state(ws: Workspace) -> dict[str, Any]:
    """``ws.desk`` with its lists in place (created on first use)."""
    if not isinstance(ws.desk, dict):
        ws.desk = {}
    ws.desk.setdefault("revisions", [])
    ws.desk.setdefault("comments", [])
    return ws.desk


def _flood_steps(ws: Workspace) -> list[Any]:
    study = ws.study_obj()
    if study is None:
        return []
    return [s for s in study.steps if s.tool in FLOOD_TOOLS]


def _ffa_payload(ws: Workspace) -> dict[str, Any] | None:
    """The payload of the flood fit the documents quote: the flood_frequency step's, else the first with ffa."""
    best = None
    for r in (ws.run or {}).get("results") or []:
        p = r.get("result") if isinstance(r.get("result"), dict) else None
        if not p or not isinstance(p.get("ffa"), dict):
            continue
        if r.get("tool") == "flood_frequency":
            return p
        best = best or p
    return best


def estimator_of(ws: Workspace) -> str:
    """The fit the documents quote as the answer: the Desk's choice, else GEV (L-moments)."""
    chosen = (ws.desk or {}).get("estimator") if isinstance(ws.desk, dict) else None
    return str(chosen) if chosen in ESTIMATORS else "gev_lmoments"


# ── the levers ──────────────────────────────────────────────────────────────


def levers(ws: Workspace) -> list[dict[str, Any]]:
    """The levers this study offers, each ``{"id", "label", "kind", "type", "value", "choices"?, "help",
    "steps"}``, with the current value; ``exclude_years`` also lists the annual maxima as ``candidates``
    (largest first) so a page can offer them as chips. Empty for a study with no flood fit."""
    steps = _flood_steps(ws)
    payload = _ffa_payload(ws)
    if not steps or payload is None:
        return []
    from aquascope.studio.steering import controls_for

    ids = [str(s.id) for s in steps]
    out: list[dict[str, Any]] = []
    first = steps[-1]
    controls = {c.param: c for c in controls_for(first)}
    params = ((ws.study or {}).get("problem") or {}).get("params") or {}
    if "return_period" in controls:
        c = controls["return_period"]
        out.append({"id": "return_period", "label": LEVERS["return_period"][0], "kind": "rerun", "type": "choice",
                    "value": params.get("return_period") or (ws.brief.intake or {}).get("return_period") or 100,
                    "choices": list(c.choices), "help": c.help, "steps": ids})
    if "years" in controls:
        out.append({"id": "years", "label": LEVERS["years"][0], "kind": "rerun", "type": "integer",
                    "value": first.arguments.get("years"), "min": 10, "max": 200,
                    "help": "leave empty for the full record", "steps": ids})
    if "exclude_years" in controls:
        am = payload.get("annual_max") if isinstance(payload.get("annual_max"), dict) else {}
        ex = payload.get("annual_max_excluded") if isinstance(payload.get("annual_max_excluded"), dict) else {}
        pairs = list(zip(am.get("year") or [], am.get("v") or [])) + list(zip(ex.get("year") or [], ex.get("v") or []))
        pairs = [(int(y), float(v)) for y, v in pairs if v is not None]
        pairs.sort(key=lambda p: -p[1])
        out.append({"id": "exclude_years", "label": LEVERS["exclude_years"][0], "kind": "rerun", "type": "years",
                    "value": list(first.arguments.get("exclude_years") or []),
                    "candidates": [{"year": y, "value": round(v, 4)} for y, v in pairs[:12]],
                    "help": "a year whose maximum is not to be trusted: a dam break, a rating revised afterwards",
                    "steps": ids})
    fits = [k for k, v in ((payload.get("ffa") or {}).get("fits") or {}).items()
            if k in ESTIMATORS and isinstance(v, dict) and v.get("q")]
    if len(fits) > 1:
        out.append({"id": "estimator", "label": LEVERS["estimator"][0], "kind": "document", "type": "choice",
                    "value": estimator_of(ws), "choices": fits, "labels": {k: ESTIMATORS[k] for k in fits},
                    "help": "every fit stays in the tables; this one leads the answer", "steps": []})
    return out


def split_changes(ws: Workspace, changes: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any],
                                                                   list[str]]:
    """``(step_changes, document_changes, errors)``: the rerun levers as steering changes per flood step (only
    where the step declares the control and the value differs), the document levers as they are, and the
    reasons a lever was refused."""
    from aquascope.studio.steering import _coerce, _current, controls_for

    step_changes: dict[str, dict[str, Any]] = {}
    doc: dict[str, Any] = {}
    errors: list[str] = []
    study = ws.study_obj()
    for key, value in (changes or {}).items():
        if key not in LEVERS:
            errors.append(f"{key} is not a lever; the levers are {', '.join(LEVERS)}")
            continue
        label, param, kind = LEVERS[key]
        if kind == "document":
            if key == "estimator":
                fits = next((lv for lv in levers(ws) if lv["id"] == "estimator"), None)
                allowed = fits["choices"] if fits else []
                v = str(value)
                hit = next((k for k in allowed if v.lower() in (k, ESTIMATORS[k].lower())), None)
                if hit is None:
                    errors.append(f"the distribution must be one of {', '.join(allowed) or 'the fits that ran'}")
                elif hit != estimator_of(ws):
                    doc[key] = hit
            continue
        touched = False
        for step in _flood_steps(ws):
            controls = {c.param: c for c in controls_for(step)}
            control = controls.get(str(param))
            if control is None:
                continue
            try:
                new = _coerce(control, value)
            except ValueError as exc:
                errors.append(str(exc))
                break
            old = _current(step, control, study)
            if (new is None and control.argument not in (step.arguments or {})) or new == old or \
                    (isinstance(new, list) and sorted(new) == sorted(old or [])):
                touched = True
                continue
            step_changes.setdefault(str(step.id), {})[str(param)] = new
            touched = True
        if not touched:
            errors.append(f"{label.lower()} does not apply to this study")
    return step_changes, doc, errors


def apply_estimator(ws: Workspace) -> None:
    """Point the Interpreter's decision at the Desk's chosen fit (value, basis, answer), so the board, the answer
    and the documents quote the same number. The grade is the Interpreter's and stays."""
    chosen = (ws.desk or {}).get("estimator") if isinstance(ws.desk, dict) else None
    decision = (ws.findings or {}).get("decision") if isinstance(ws.findings, dict) else None
    if chosen not in ESTIMATORS or not isinstance(decision, dict):
        return
    for r in (ws.run or {}).get("results") or []:
        p = r.get("result") if isinstance(r.get("result"), dict) else None
        if not p or not isinstance(p.get("ffa"), dict):
            continue
        ffa = p["ffa"]
        fit = (ffa.get("fits") or {}).get(chosen)
        params = ((ws.study or {}).get("problem") or {}).get("params") or {}
        t = params.get("return_period") or (ws.brief.intake or {}).get("return_period") or 100
        try:
            i = [float(x) for x in ffa.get("return_periods") or []].index(float(t))
        except (TypeError, ValueError):
            continue
        if not isinstance(fit, dict) or not fit.get("q") or fit["q"][i] is None:
            continue
        value = float(fit["q"][i])
        tt = int(float(t)) if float(t).is_integer() else t
        decision["value"] = round(value, 4)
        decision["basis"] = [f"{r.get('id')}.ffa.fits.{chosen}.q.{i}"]
        decision["answer"] = (f"design flow: {tt}-year return level, {ESTIMATORS[chosen]} "
                              f"{value:.4g} {p.get('unit') or ''} ({decision.get('grade', '')}).").replace("  ", " ")
        if r.get("tool") == "flood_frequency":
            break


def describe(changes: dict[str, Any], before: dict[str, Any]) -> list[str]:
    """Each lever change as a line a revision history can carry: "Left out the 2008 annual maximum"."""
    out: list[str] = []
    for key, value in changes.items():
        if key == "return_period":
            out.append(f"Design return period changed from {before.get('t') and int(float(before['t']))} to "
                       f"{int(float(value))} years")
        elif key == "years":
            out.append(f"Record window limited to the last {int(value)} years" if value else
                       "Full record used again")
        elif key == "exclude_years":
            years = [int(y) for y in (value or [])]
            out.append(("Left out the annual maxima of " + ", ".join(str(y) for y in years)) if years else
                       "Every complete year used again")
        elif key == "estimator":
            out.append(f"The answer now quotes {ESTIMATORS.get(str(value), value)}")
    return out


# ── the record a revision refits ────────────────────────────────────────────

#: The payload fields a refit carries over from the study's first fetch (who served the record, under what licence).
_CARRY = ("source", "station_id", "agency", "license", "attribution", "requested", "archive_revision",
          "station_name", "name", "software_revision")


def _stored_series(ws: Workspace, source: str, station_id: str) -> tuple[Any, dict[str, Any]] | None:
    """The daily record the study already analysed for ``(source, station_id)``, as a pandas Series, with the
    payload it came from: read from the step's record table (``tab-<step>-series``, the full observations the
    fit used). None when the study holds no such table (a workspace saved without its files)."""
    import io

    import pandas as pd

    for r in (ws.run or {}).get("results") or []:
        p = r.get("result") if isinstance(r.get("result"), dict) else None
        if not p or str(p.get("source")) != str(source) or str(p.get("station_id")) != str(station_id):
            continue
        art = ws.artifact(f"tab-{r.get('id')}-series")
        if art is None or not art.data:
            continue
        try:
            frame = pd.read_csv(io.BytesIO(art.data))
            if "datetime" not in frame.columns or "value" not in frame.columns or frame.empty:
                continue
            idx = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
            series = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(), index=idx).dropna()
            series = series[~series.index.isna()].sort_index()
        except Exception:  # noqa: BLE001 - an unreadable table means: fetch as before
            continue
        if not series.empty:
            return series, p
    return None


def stored_record_tools(ws: Workspace) -> dict[str, Any]:
    """``analyze_station`` and ``flood_frequency`` that refit the record the study already holds instead of
    fetching it again, so a revision changes the assumptions and never the data (an agency that serves a longer
    record today, or a rate-limited request that falls back to the archive's copy, would otherwise move the
    answer under the reviewer's feet). ``years`` cuts the stored record to its last N years; ``exclude_years``
    and ``return_periods`` go to the fit. A station the study holds no record table for is fetched as before."""
    from aquascope.explore import analyze_series, flood_ci
    from aquascope.mcp_server import _flood_result
    from aquascope.studio.roles.analysts import analyze_station_full

    def refit(source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False,
              variable: str | None = None, return_periods: list[float] | None = None,
              exclude_years: list[int] | None = None) -> dict[str, Any] | None:
        got = _stored_series(ws, source, station_id)
        if got is None:
            return None
        series, before = got
        if years:
            import pandas as pd

            series = series[series.index >= series.index.max() - pd.DateOffset(years=int(years))]
        out = {k: before.get(k) for k in _CARRY if before.get(k) is not None}
        out.update(analyze_series(series, str(before.get("variable") or variable or "discharge"),
                                  str(before.get("unit") or ""), return_periods=return_periods,
                                  exclude_years=exclude_years))
        out["fetch_note"] = (str(before.get("fetch_note") or "").rstrip() +
                             " Refitted on the Study Desk from the record the study already holds; nothing was "
                             "fetched again.").strip()
        out["observations"] = {"t": [t.isoformat() for t in series.index], "v": [float(v) for v in series.values]}
        if bootstrap_ci and out.get("ffa"):
            try:
                ci = flood_ci(series, return_periods=return_periods, exclude_years=exclude_years)
                out["ffa"]["fits"]["gev_bootstrap"] = {
                    k: ci[k] for k in ("q", "ci", "params", "n_bootstrap", "n_bootstrap_discarded", "estimator",
                                       "interval_method", "ci_level") if k in ci}
                out.setdefault("methods", []).append(ci["method"])
            except Exception as exc:  # noqa: BLE001 - the band is optional
                out.setdefault("notes", []).append(f"bootstrap CI failed: {exc}")
        return out

    def analyze_station(source: str, station_id: str, **kw: Any) -> dict[str, Any]:
        res = refit(source, station_id, **kw)
        return res if res is not None else analyze_station_full(source, station_id, **kw)

    def flood_frequency(source: str, station_id: str, **kw: Any) -> dict[str, Any]:
        res = refit(source, station_id, **kw)
        if res is None:
            from aquascope.studio.roles.analysts import flood_frequency_full

            return flood_frequency_full(source, station_id, **kw)
        result = _flood_result(res)
        result["observations"] = res["observations"]
        return result

    return {"analyze_station": analyze_station, "flood_frequency": flood_frequency}


# ── revisions ───────────────────────────────────────────────────────────────


def _headline(ws: Workspace) -> dict[str, Any]:
    from aquascope.studio.document.compose import _design_t, _fits_at
    from aquascope.studio.document.facts import facts_of

    f = facts_of(ws)
    t = _design_t(f)
    at = _fits_at(f, t)
    main = at.get(estimator_of(ws)) or at.get("gev_lmoments") or next(iter(at.values()), None)
    return {"t": t, "value": main.get("q") if main else f.headline.get("value"),
            "label": main.get("label") if main else "", "unit": (f.ffa or {}).get("unit") or f.headline.get("unit"),
            "grade": f.grade}


def _letter(n: int) -> str:
    """A, B, ..., Z, AA, AB: revision letters."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def record_revision(ws: Workspace, description: str, *, before: dict[str, Any] | None = None,
                    by: str | None = None) -> dict[str, Any]:
    """Append a revision ``{"rev", "at", "by", "description", "before", "after"}`` (``before`` and ``after`` are
    the headline: return period, value, unit, grade)."""
    state = desk_state(ws)
    after = _headline(ws)
    rev = {"rev": _letter(len(state["revisions"])), "at": _now(), "by": by or "",
           "description": description, "before": before, "after": after}
    state["revisions"].append(rev)
    return rev


def baseline(ws: Workspace) -> None:
    """The study as the crew delivered it is revision A (recorded once, the first time the Desk opens it)."""
    state = desk_state(ws)
    if not state["revisions"]:
        record_revision(ws, "First issue, as the crew delivered it", by=(ws.house_style or {}).get("prepared_by")
                        or "AquaScope Studio")


# ── sign-off and review ─────────────────────────────────────────────────────


def sign(ws: Workspace, role: str, name: str, *, date: str | None = None) -> dict[str, Any]:
    """Fill a sign-off line: ``role`` is prepared_by, checked_by or approved_by. The document's status moves
    with it (DRAFT, CHECKED, ISSUED), and a revision records the signature."""
    if role not in STATUS_BY_ROLE:
        raise ValueError(f"role must be one of {', '.join(STATUS_BY_ROLE)}")
    name = " ".join(str(name or "").split())
    if not name:
        raise ValueError("a sign-off needs a name")
    baseline(ws)
    style = dict(ws.house_style or {})
    style[role] = name
    style[role.replace("_by", "_on")] = (date or _now())[:10]
    order = list(STATUS_BY_ROLE)
    signed = [r for r in order if style.get(r)]
    style["status"] = STATUS_BY_ROLE[signed[-1]] if signed else "DRAFT"
    ws.house_style = style
    rev = record_revision(ws, f"{ROLE_WORDS[role]} by {name}", before=None, by=name)
    return rev


def comment(ws: Workspace, text: str, *, section: str = "", author: str = "") -> dict[str, Any]:
    """Add a review comment ``{"id", "section", "author", "text", "at", "status": "open", "response"}``."""
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("a comment needs text")
    state = desk_state(ws)
    item = {"id": f"c{len(state['comments']) + 1}", "section": section, "author": author, "text": text,
            "at": _now(), "status": "open", "response": ""}
    state["comments"].append(item)
    return item


def resolve(ws: Workspace, comment_id: str, response: str = "", *, by: str = "") -> dict[str, Any]:
    """Close a comment with the response (what was changed, or why not)."""
    state = desk_state(ws)
    item = next((c for c in state["comments"] if c["id"] == comment_id), None)
    if item is None:
        raise ValueError(f"there is no comment {comment_id!r}")
    item["status"] = "resolved"
    item["response"] = " ".join(str(response or "").split())
    item["resolved_by"] = by
    item["resolved_at"] = _now()
    return item


# ── sensitivity ─────────────────────────────────────────────────────────────


def _refit_q(values: list[float], t: float, estimator: str = "gev_lmoments") -> float | None:
    """The T-year quantile of ``values`` (annual maxima) refitted with the reported distribution: Log-Pearson III
    for lp3, GEV by L-moments otherwise (a GEV by maximum likelihood is refitted by L-moments, which is quick
    and close; the basis column says so)."""
    try:
        if estimator == "lp3":
            import pandas as pd

            from aquascope.hydrology.flood_frequency import fit_lp3

            idx = pd.to_datetime([f"{1900 + i}-01-01" for i in range(len(values))])
            r = fit_lp3(pd.Series(values, index=idx), return_periods=[t])
        else:
            from aquascope.hydrology.flood_frequency import fit_gev_lmoments

            r = fit_gev_lmoments(values, return_periods=[t])
        return float(r.return_periods[t])
    except Exception:  # noqa: BLE001 - a case that cannot be fitted is left out, not raised
        return None


def sensitivity(ws: Workspace) -> list[dict[str, Any]]:
    """The design value under the reasonable alternatives, computed from the annual maxima the study stored
    (no fetch): ``[{"case", "value", "change_pct", "n", "basis"}]``, the base first. The cases: each other
    fitted distribution, the largest flood left out, only the most recent 30 and 50 years, and the excluded
    years put back. Refits use GEV by L-moments. Empty for a study with no flood fit."""
    payload = _ffa_payload(ws)
    if payload is None:
        return []
    ffa = payload["ffa"]
    head = _headline(ws)
    t = head.get("t")
    base = head.get("value")
    if t is None or base is None:
        return []
    am = payload.get("annual_max") if isinstance(payload.get("annual_max"), dict) else {}
    years = [int(y) for y in am.get("year") or []]
    vals = [float(v) for v in am.get("v") or [] if v is not None]
    rows: list[dict[str, Any]] = []

    def add(case: str, value: float | None, n: int | None, basis: str) -> None:
        if value is None:
            return
        rows.append({"case": case, "value": value, "n": n,
                     "change_pct": (value - base) / base * 100 if base else None, "basis": basis})

    chosen = estimator_of(ws)
    add(f"As reported: {ESTIMATORS.get(chosen, chosen)}", base, ffa.get("n_years"), "the study's fit")
    try:
        i = [float(x) for x in ffa.get("return_periods") or []].index(float(t))
    except ValueError:
        i = None
    for key, label in ESTIMATORS.items():
        fit = (ffa.get("fits") or {}).get(key)
        if key == chosen or i is None or not isinstance(fit, dict) or not fit.get("q") or fit["q"][i] is None:
            continue
        add(f"{label} instead", float(fit["q"][i]), ffa.get("n_years"), "the study's fit")
    refit = "lp3" if chosen == "lp3" else "gev_lmoments"
    basis = f"{ESTIMATORS[refit]} refit"
    if len(vals) == len(years) and len(vals) >= 10:
        k = max(range(len(vals)), key=lambda j: vals[j])
        rest = vals[:k] + vals[k + 1:]
        add(f"Without the largest flood ({years[k]})", _refit_q(rest, t, refit), len(rest), basis)
        order = sorted(range(len(years)), key=lambda j: years[j])
        for n in (50, 30):
            if len(vals) >= n + 10:
                recent = [vals[j] for j in order[-n:]]
                add(f"Most recent {n} years only ({years[order[-n]]} to {years[order[-1]]})",
                    _refit_q(recent, t, refit), n, basis)
        ex = payload.get("annual_max_excluded") if isinstance(payload.get("annual_max_excluded"), dict) else None
        if ex and ex.get("v"):
            back = vals + [float(v) for v in ex["v"] if v is not None]
            add(f"Excluded years put back ({', '.join(str(y) for y in ex.get('year') or [])})",
                _refit_q(back, t, refit), len(back), basis)
    return rows


# ── the faces ───────────────────────────────────────────────────────────────


def view(ws: Workspace) -> dict[str, Any]:
    """Everything a Desk panel draws: the levers, the sensitivity table, the revisions, the comments and the
    sign-off, as plain JSON."""
    style = ws.house_style or {}
    rows = []
    try:
        rows = sensitivity(ws)
    except Exception:  # noqa: BLE001 - the panel draws without it
        rows = []
    unit = None
    payload = _ffa_payload(ws)
    if payload:
        from aquascope.viz.publication import unit_text

        unit = unit_text(payload.get("unit")) or None
    state = ws.desk if isinstance(ws.desk, dict) else {}
    return {
        "levers": levers(ws),
        "sensitivity": [{**r, "value": round(float(r["value"]), 4)} for r in rows],
        "unit": unit,
        "revisions": list(state.get("revisions") or []),
        "comments": list(state.get("comments") or []),
        "signoff": {role: style.get(role) or "" for role in STATUS_BY_ROLE},
        "status": style.get("status") or "DRAFT",
        "available": bool(levers(ws)) or ws.status == "done",
    }


def studio_op(studio: Any, a: dict[str, Any]) -> dict[str, Any]:
    """One Desk action for a face (the browser worker, the MCP server): ``a["action"]`` is ``view``, ``revise``
    (``changes``, ``by``, ``note``), ``sign`` (``role``, ``name``), ``comment`` (``text``, ``section``,
    ``author``) or ``resolve`` (``id``, ``response``, ``by``). Returns ``{"desk": view}`` plus ``"reply"`` (the
    :class:`~aquascope.studio.coordinator.Reply`) when the documents were rewritten, or ``{"error": ...}``."""
    ws = studio.ws
    action = str(a.get("action") or "view")
    try:
        if action == "view":
            return {"desk": view(ws)}
        if action == "revise":
            reply = studio.revise(dict(a.get("changes") or {}), by=a.get("by") or None, note=a.get("note") or None)
            if reply.payload.get("errors"):
                return {"error": reply.text, "desk": view(ws)}
            return {"desk": view(ws), "reply": reply}
        if action == "sign":
            reply = studio.sign(str(a.get("role") or ""), str(a.get("name") or ""))
            if reply.payload.get("errors"):
                return {"error": reply.text, "desk": view(ws)}
            return {"desk": view(ws), "reply": reply}
        if action in ("comment", "resolve"):
            if action == "comment":
                comment(ws, str(a.get("text") or ""), section=str(a.get("section") or ""),
                        author=str(a.get("author") or ""))
            else:
                resolve(ws, str(a.get("id") or ""), str(a.get("response") or ""), by=str(a.get("by") or ""))
            studio._build_deliverables()
            return {"desk": view(ws), "reply": studio._report_reply()}
    except ValueError as exc:
        return {"error": str(exc), "desk": view(ws)}
    return {"error": f"unknown desk action {action!r}"}
