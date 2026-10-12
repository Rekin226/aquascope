"""What a study established, read once from the workspace into plain fields the documents are written from.

The workspace is the crew's working memory: payloads in the shapes the tools
return, gates with their raw details, the Interpreter's decision. A report
needs something else: the site by name, the record with its period and
completeness, the design value with the interval that belongs to it, each
check as a sentence with a verdict, the grade with the reason for it. This
module does that reading, defensively (a missing key gives None or an empty
list, never an exception), so the composer can be written as prose.

Nothing here formats for a page or invents a value: every number is one a
tool returned, and the composer quotes it through
:mod:`aquascope.studio.document.text`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from aquascope.studio.workspace import Workspace

__all__ = ["GRADE_MEANING", "GRADE_WORDS", "Facts", "facts_of"]

#: The grade words as a report prints them.
GRADE_WORDS = {
    "established": "Established",
    "indicative": "Indicative",
    "screening": "Screening",
    "not_established": "Not established",
}

#: What each grade means, worded from the Interpreter's rules (``roles/interpreter.py``: ``grade_for_step``
#: and ``grade_for_study``), so a reader can tell what a grade licenses.
GRADE_MEANING = {
    "established": "The answer rests on a measured record at the site; every check on it passed and the method "
                   "was rated defensible here.",
    "indicative": "The answer rests on a measured record and its checks passed, but something limits it: the "
                  "method was rated marginal at this site, a supporting check failed, the primary method fell "
                  "back to another, or the record shows a change in regime.",
    "screening": "The answer rests on modelled, reanalysis or transferred data rather than a record at the "
                 "site. It is suitable for scoping and comparison, not for design.",
    "not_established": "The step that should answer the question failed or did not pass its checks. No number "
                       "from it should be used.",
}

#: Tool names as a methods section names the work they did.
TOOL_WORDS = {
    "describe_catchment": "catchment description (BasinATLAS)",
    "assess_site": "site assessment (what data exist here)",
    "analyze_station": "record analysis",
    "flood_frequency": "flood frequency analysis",
    "anywhere": "reanalysis climate and modelled discharge (ERA5, GloFAS)",
    "reach_record": "simulated discharge for the river reach (GEOGLOWS v2, modelled)",
    "similar_basins": "donor catchment search",
    "regionalize_signatures": "transfer of flow signatures from donor catchments",
    "supply_reliability": "supply reliability analysis",
    "low_flow_context": "low-flow analysis",
    "drought_indices": "drought indices (SPI/SPEI)",
    "drought_propagation": "drought propagation",
    "sgi_drought": "groundwater drought index (SGI)",
    "wqi": "water quality index",
    "iwqi": "irrigation water quality index",
    "who_screen": "screening against WHO guideline values",
    "water_quality_samples": "water quality samples",
    "regional_flood": "regional flood frequency",
    "get_timeseries": "record retrieval",
    "crop_water_demand": "crop water demand (FAO-56)",
    "climate_projection": "climate projection (CMIP6)",
    "catchment_model": "rainfall-runoff model",
    "recharge": "groundwater recharge",
    "pot_flood": "peaks-over-threshold flood frequency",
    "nonstationary_flood": "nonstationary flood frequency",
    "change_points": "change-point analysis",
}

#: Estimators by the name a report gives them.
FIT_WORDS = {
    "gev_lmoments": "GEV (L-moments)",
    "lp3": "Log-Pearson III",
    "gev_bootstrap": "GEV (maximum likelihood)",
}


@dataclass
class Check:
    step: str
    name: str
    sentence: str
    verdict: str        # "passed", "failed", "skipped"
    detail: str


@dataclass
class Facts:
    question: str = ""
    decision: str = ""
    playbook: str = ""
    kind: str = ""
    return_period: float | None = None
    asks_trend: bool = False

    lat: float | None = None
    lon: float | None = None
    site_name: str = ""
    site_id: str = ""
    site_label: str = ""

    record: dict[str, Any] = field(default_factory=dict)
    catchment: dict[str, Any] = field(default_factory=dict)
    headline: dict[str, Any] = field(default_factory=dict)
    grade: str = "not_established"
    grade_reason: str = ""
    conditions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    would_change: list[str] = field(default_factory=list)
    data_requests: list[dict[str, str]] = field(default_factory=list)

    ffa: dict[str, Any] = field(default_factory=dict)
    trend: dict[str, Any] = field(default_factory=dict)
    change: dict[str, Any] = field(default_factory=dict)
    fdc: dict[str, Any] = field(default_factory=dict)
    climate: dict[str, Any] = field(default_factory=dict)
    glofas: dict[str, Any] = field(default_factory=dict)
    regional: dict[str, Any] = field(default_factory=dict)

    checks: list[Check] = field(default_factory=list)
    consistency: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    nearby: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    #: The fit the answer quotes (the Study Desk's choice, else GEV by L-moments).
    estimator: str = "gev_lmoments"
    #: The Desk: revisions, review comments, the sensitivity of the answer.
    revisions: list[dict[str, Any]] = field(default_factory=list)
    comments: list[dict[str, Any]] = field(default_factory=list)
    sensitivity: list[dict[str, Any]] = field(default_factory=list)
    prose: dict[str, str] = field(default_factory=dict)
    version: str = ""
    created: str = ""
    workspace_id: str = ""

    @property
    def established_anything(self) -> bool:
        return any(s.get("ok") for s in self.steps)

    @property
    def gauged(self) -> bool:
        return bool(self.record.get("station_id"))


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _results(ws: Workspace) -> list[dict[str, Any]]:
    return [r for r in (ws.run or {}).get("results") or [] if isinstance(r, dict)]


def _payload(r: dict[str, Any]) -> dict[str, Any]:
    p = r.get("result")
    if isinstance(p, dict) and not p.get("error"):
        return p
    fb = r.get("fallback")
    if isinstance(fb, dict) and isinstance(fb.get("result"), dict) and r.get("fallback_used"):
        return fb["result"]
    return p if isinstance(p, dict) else {}


def station_label(source: str | None, station_id: str | None) -> str:
    """``USGS 01013500`` from ``("usgs", "USGS-01013500")``; ``UK EA 8496ce69`` style otherwise."""
    if not station_id:
        return ""
    sid = str(station_id)
    src = str(source or "")
    prefix = src.upper().replace("_", " ")
    if sid.upper().startswith(src.upper() + "-"):
        sid = sid[len(src) + 1:]
    try:
        from aquascope.registry import SOURCES

        info = SOURCES.get(src)
        short = getattr(info, "short", None) if info is not None else None
        if short:
            prefix = str(short)
    except Exception:  # noqa: BLE001 - the registry is a nicety here
        pass
    if len(sid) > 16 and re.fullmatch(r"[0-9a-f\-]+", sid):
        sid = sid[:8]
    return f"{prefix} {sid}".strip()


def _clean_reason(text: Any) -> str:
    """A planner's terse reason as a sentence a reader can follow: snake_case names spelled out, doubled
    colons split, raw error plumbing reduced to what failed."""
    t = " ".join(str(text or "").split())
    if not t:
        return ""
    t = re.sub(r"\b(at_site_flood_frequency)\b", "at-site flood frequency", t)
    t = re.sub(r"\b([a-z]+(?:_[a-z0-9]+)+)\b", lambda m: m.group(1).replace("_", " "), t)
    t = re.sub(r"DataSourceError:\s*CURL error:\s*Could not resolve host:\s*(\S+)",
               r"the data service \1 could not be reached (no network connection)", t)
    t = re.sub(r"CURL error[^;.]*", "a network error", t)
    return t


def _sufficiency_reason(ws: Workspace, method: str | None) -> tuple[str, str]:
    """(status, reason) of the site-sufficiency row for ``method``."""
    inv = ws.inventory
    rows = list(getattr(inv, "sufficiency", None) or []) if inv else []
    for r in rows:
        if isinstance(r, dict) and str(r.get("method")) == str(method):
            return str(r.get("status") or ""), _clean_reason(r.get("reason") or r.get("why") or "")
    return "", ""


def _catchment(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
    out: dict[str, Any] = {"rows": []}
    area = _num(attrs.get("upstream_area_km2")) or _num(attrs.get("area_km2"))
    if area is not None:
        out["area_km2"] = area
        out["rows"].append(("Upstream catchment area", area, "km²"))
    wanted = [("elevation_m", "Mean elevation"), ("slope_deg", "Mean slope"),
              ("precipitation_mm_yr", "Mean annual precipitation"), ("pet_mm_yr", "Potential evapotranspiration"),
              ("aridity_index", "Aridity index (P/PET)"), ("temperature_c", "Mean annual air temperature"),
              ("snow_cover_pct", "Snow cover extent"), ("forest_pct", "Forest cover"), ("cropland_pct", "Cropland"),
              ("pasture_pct", "Pasture"), ("urban_pct", "Urban extent"), ("irrigated_pct", "Irrigated area"),
              ("lake_pct", "Lake area"), ("wetland_pct", "Wetlands"), ("karst_pct", "Karst"),
              ("regulation_pct", "Degree of regulation by reservoirs"), ("dams", "Dams upstream"),
              ("discharge_m3s", "Mean annual natural discharge (modelled)")]
    for key, label in wanted:
        a = attrs.get(key)
        if isinstance(a, dict):
            v, u = _num(a.get("value")), a.get("unit")
        else:
            v, u = _num(a), None
        if v is None:
            continue
        u = {"degrees": "°", "P/PET": "", "mm/yr": "mm/yr", "%": "%"}.get(str(u), u)
        out[key] = v
        out["rows"].append((label, v, u or ""))
    reg = attrs.get("regulation") or attrs.get("dor_pc_pva")
    if isinstance(reg, dict) and _num(reg.get("value")) is not None and "regulation_pct" not in out:
        out["regulation_pct"] = _num(reg.get("value"))
    sb = payload.get("sub_basin") if isinstance(payload.get("sub_basin"), dict) else {}
    if sb.get("hybas_id"):
        out["hybas_id"] = sb["hybas_id"]
    out["n_sub_basins"] = attrs.get("n_sub_basins")
    return out


def _ffa(payload: dict[str, Any]) -> dict[str, Any]:
    ffa = payload.get("ffa") if isinstance(payload.get("ffa"), dict) else None
    if not ffa or not isinstance(ffa.get("fits"), dict):
        return {}
    periods = [_num(t) for t in ffa.get("return_periods") or []]
    fits: dict[str, dict[str, Any]] = {}
    for name, fit in ffa["fits"].items():
        if not isinstance(fit, dict) or not isinstance(fit.get("q"), list):
            continue
        ci = fit.get("ci") if isinstance(fit.get("ci"), list) else None
        fits[name] = {
            "label": FIT_WORDS.get(name, name.replace("_", " ")),
            "q": [_num(v) for v in fit["q"]],
            "ci": [[_num(c[0]), _num(c[1])] if isinstance(c, (list, tuple)) and len(c) == 2 else [None, None]
                   for c in ci] if ci else None,
            "level": _num(fit.get("ci_level")),
            "interval_method": str(fit.get("interval_method") or "").replace("_", " "),
            "estimator": fit.get("estimator"),
            "n_bootstrap": fit.get("n_bootstrap"),
        }
    out = {"T": periods, "fits": fits, "n_years": ffa.get("n_years"),
           "record_max": ffa.get("record_max") if isinstance(ffa.get("record_max"), dict) else {},
           "unit": payload.get("unit") or "m3/s"}
    am = payload.get("annual_max")
    if isinstance(am, dict) and isinstance(am.get("year"), list):
        out["years"] = [int(y) for y in am["year"] if isinstance(y, (int, float))]
    return out


def _trend(payload: dict[str, Any]) -> dict[str, Any]:
    from aquascope.trend_series import reported_trend

    tr = None
    try:
        tr = reported_trend(payload)
    except Exception:  # noqa: BLE001
        tr = None
    if not isinstance(tr, dict) or tr.get("unavailable"):
        ffa = payload.get("ffa") if isinstance(payload.get("ffa"), dict) else {}
        tr = ffa.get("amax_trend") if isinstance(ffa.get("amax_trend"), dict) else payload.get("trend")
    if not isinstance(tr, dict) or _num(tr.get("p_value")) is None:
        return {}
    return {"on": tr.get("on") or "annual series", "p": _num(tr.get("p_value")), "tau": _num(tr.get("tau")),
            "slope": _num(tr.get("sens_slope_per_year")), "n": tr.get("n_years"),
            "verdict": str(tr.get("trend") or "").replace("_", " "), "unit": payload.get("unit")}


def facts_of(ws: Workspace) -> Facts:  # noqa: C901 - one pass over the workspace, branch per payload kind
    """Read the workspace into :class:`Facts`."""
    from aquascope import __version__

    f = Facts()
    b = ws.brief
    f.question = str(b.problem or "").strip()
    f.decision = str(b.decision or (b.intake or {}).get("decision") or "").strip()
    f.playbook = str(b.playbook or ((ws.study or {}).get("plan") or {}).get("playbook") or "")
    f.kind = str(b.kind or "")
    f.return_period = _num((b.intake or {}).get("return_period"))
    f.asks_trend = "trend" in f.decision.lower()
    site = ws.site or (ws.inventory.site if ws.inventory else None) or {}
    f.lat, f.lon = _num(site.get("lat", site.get("latitude"))), _num(site.get("lon", site.get("longitude")))
    f.version = str(__version__)
    f.created = str(ws.created or "")[:10]
    f.workspace_id = str(ws.id or "")
    f.model, f.provider = str(ws.model or ""), str(ws.provider or "")
    desk = ws.desk if isinstance(ws.desk, dict) else {}
    if desk.get("estimator") in FIT_WORDS:
        f.estimator = str(desk["estimator"])
    f.revisions = [r for r in desk.get("revisions") or [] if isinstance(r, dict)]
    f.comments = [c for c in desk.get("comments") or [] if isinstance(c, dict)]

    study = ws.study or {}
    steps_by_id = {str(s.get("id")): s for s in study.get("steps") or [] if isinstance(s, dict)}
    for r in _results(ws):
        sid = str(r.get("id"))
        step = steps_by_id.get(sid, {})
        res = r.get("result")
        # A stored run says "ok"; an older or hand-built one may only carry the result: a payload without an
        # error counts as having run.
        ran = bool(r.get("ok", isinstance(res, dict) and not res.get("error") and not r.get("error")))
        gates_ok = r.get("gates_passed")
        if gates_ok is None:
            gates_ok = not any(isinstance(g, dict) and g.get("passed") is False and not g.get("skipped")
                               for g in r.get("gates") or [])
        f.steps.append({"id": sid, "tool": str(r.get("tool") or ""), "method": step.get("method"),
                        "ok": (ran and not r.get("skipped") and bool(gates_ok)) or bool(r.get("fallback_used")),
                        "ran": ran, "rationale": step.get("rationale") or r.get("rationale"),
                        "arguments": r.get("arguments") or step.get("arguments") or {},
                        "fallback_used": bool(r.get("fallback_used")),
                        "error": _clean_reason(r.get("error") or (_payload(r).get("error") if not ran else ""))})
        for g in r.get("gates") or []:
            if not isinstance(g, dict):
                continue
            from aquascope.gates import plain

            verdict = "skipped" if g.get("skipped") or str(g.get("detail") or "").startswith("skipped") else \
                ("passed" if g.get("passed") else "failed")
            f.checks.append(Check(step=sid, name=str(g.get("check")), sentence=plain(g),
                                  verdict=verdict, detail=_clean_reason(g.get("detail"))))
        if not ran or (not gates_ok and not r.get("fallback_used")):
            why = r.get("error") or r.get("stop_reason") or ""
            failed = [g for g in r.get("gates") or [] if isinstance(g, dict) and not g.get("passed")
                      and not g.get("skipped")]
            if failed and not why:
                why = "; ".join(str(g.get("detail") or g.get("check")) for g in failed)
            f.failures.append({"step": sid, "tool": str(r.get("tool") or ""),
                               "what": TOOL_WORDS.get(str(r.get("tool")), str(r.get("tool"))),
                               "why": _clean_reason(why) or "it returned no usable result"})

        p = _payload(r)
        tool = str(r.get("tool") or "")
        if not p:
            continue
        if tool == "describe_catchment":
            f.catchment = _catchment(p)
        if p.get("station_id") and not f.record:
            f.record = {k: p.get(k) for k in ("source", "station_id", "agency", "variable", "unit", "start", "end",
                                              "years", "n", "license", "attribution", "station_name", "name",
                                              "data_snapshot")}
            f.record["complete_years"] = (p.get("eligibility") or {}).get("complete_years") \
                if isinstance(p.get("eligibility"), dict) else None
            f.record["coverage_rule"] = (p.get("eligibility") or {}).get("coverage_rule") \
                if isinstance(p.get("eligibility"), dict) else None
            f.record["sampling"] = p.get("sampling") if isinstance(p.get("sampling"), dict) else {}
            f.record["stats"] = p.get("stats") if isinstance(p.get("stats"), dict) else {}
        if isinstance(p.get("ffa"), dict):
            got = _ffa(p)
            if got and (tool == "flood_frequency" or not f.ffa):
                f.ffa = {**got, "step": sid}
            ch = p["ffa"].get("amax_change")
            if isinstance(ch, dict) and not f.change:
                f.change = {"test": ch.get("test") or "Pettitt", "year": ch.get("change_year"),
                            "p": _num(ch.get("p_value")), "significant": bool(ch.get("significant")),
                            "before": _num(ch.get("mean_before")), "after": _num(ch.get("mean_after"))}
        if not f.trend:
            f.trend = _trend(p)
            if f.trend:
                f.trend["step"] = sid
        if isinstance(p.get("fdc"), dict) and not f.fdc:
            fd = p["fdc"]
            f.fdc = {k: _num(fd.get(k)) for k in ("q95", "q50", "q10")}
            f.fdc["unit"] = p.get("unit")
        if isinstance(p.get("climate"), dict) and not f.climate:
            cl = p["climate"]
            f.climate = {"p": _num(cl.get("precipitation_mm_per_year")), "et0": _num(cl.get("et0_mm_per_year")),
                         "t": _num(cl.get("temperature_mean_c")), "aridity": _num(cl.get("aridity_index")),
                         "aridity_class": cl.get("aridity_class"), "years": p.get("years"), "start": p.get("start"),
                         "end": p.get("end"), "source": cl.get("source") or "ERA5 via Open-Meteo"}
        if isinstance(p.get("glofas"), dict) and not f.glofas:
            g = p["glofas"]
            cell = g.get("cell") if isinstance(g.get("cell"), dict) else {}
            f.glofas = {"comparable": g.get("comparable"), "note": _clean_reason(g.get("note")),
                        "offset_km": _num(cell.get("offset_km")), "ratio": _num(cell.get("ratio")),
                        "mean": _num(cell.get("mean_flow")) or _num((g.get("stats") or {}).get("mean")
                                                                   if isinstance(g.get("stats"), dict) else None),
                        "step": sid, "ffa": _ffa(g), "start": g.get("start"), "end": g.get("end"),
                        "years": g.get("years")}
            am = g.get("annual_max") if isinstance(g.get("annual_max"), dict) else {}
            vals = [_num(v) for v in am.get("v") or []]
            vals = [v for v in vals if v is not None]
            if vals:
                f.glofas["amax_mean"] = sum(vals) / len(vals)
        if tool in ("similar_basins", "regionalize_signatures"):
            reg = f.regional
            if isinstance(p.get("estimates"), dict):
                reg["estimates"] = p["estimates"]
                reg["skill"] = ((p.get("skill") or {}).get("by_signature") or {}) \
                    if isinstance(p.get("skill"), dict) else {}
            k = p.get("k") or (p.get("similarity") or {}).get("k") if isinstance(p.get("similarity"), dict) \
                else p.get("k")
            if k:
                reg["k"] = k
            try:
                from aquascope.studio.deliverables._payload import stations_of

                donors = stations_of(p)
            except ImportError:        # a core install without the deliverables
                donors = []
            if donors and not reg.get("donors"):
                reg["donors"] = donors

    area = _num(f.catchment.get("area_km2"))
    if area and f.regional.get("estimates"):
        # Depths per day over the catchment as flows (Q = depth x area / 86.4 for mm/d and km2 to m3/s): the
        # arithmetic is stated wherever the converted value is quoted.
        conv = []
        skill = f.regional.get("skill") or {}
        for key, e in f.regional["estimates"].items():
            if not isinstance(e, dict) or str(e.get("unit")) != "mm/d" or _num(e.get("value")) is None:
                continue
            k = area / 86.4
            conv.append({"key": key, "label": str(e.get("label") or key), "value": _num(e["value"]) * k,
                         "low": (_num(e.get("low")) or 0) * k if _num(e.get("low")) is not None else None,
                         "high": _num(e.get("high")) * k if _num(e.get("high")) is not None else None,
                         "mm": _num(e["value"]), "n_donors": e.get("n_donors"),
                         "nse": _num((skill.get(key) or {}).get("nse")) if isinstance(skill.get(key), dict) else None})
        f.regional["converted"] = conv
        f.regional["area_km2"] = area
    name = f.record.get("station_name") or f.record.get("name") or ""
    f.site_name = str(name)
    f.site_id = station_label(f.record.get("source"), f.record.get("station_id"))
    if f.site_name:
        f.site_label = f"{f.site_name} ({f.site_id})" if f.site_id else f.site_name
    elif f.site_id:
        f.site_label = f"station {f.site_id}"
    else:
        from aquascope.studio.document.text import coord

        f.site_label = f"the site at {coord(f.lat, f.lon)}" if f.lat is not None else "the site"

    findings = ws.findings or {}
    decision = findings.get("decision") if isinstance(findings.get("decision"), dict) else {}
    f.grade = str(decision.get("grade") or (ws.report or {}).get("grade") or "not_established")
    if decision:
        ev = decision.get("evidence") if isinstance(decision.get("evidence"), dict) else {}
        interval = ev.get("interval") if isinstance(ev.get("interval"), dict) else {}
        label = ""
        basis = (decision.get("basis") or [""])[0] if decision.get("basis") else ""
        for kn in (ws.report or {}).get("key_numbers") or []:
            if isinstance(kn, dict) and _num(kn.get("value")) == _num(decision.get("value")):
                label = str(kn.get("label") or "")
                break
        f.headline = {"value": _num(decision.get("value")), "unit": decision.get("unit"),
                      "label": label, "basis": basis, "estimator": ev.get("estimator"),
                      "band": decision.get("band"),
                      "interval": interval.get("bounds"), "interval_level": _num(interval.get("level")),
                      "interval_method": interval.get("method"),
                      "answer": str(decision.get("answer") or "")}
        f.conditions = [_clean_reason(x) for x in decision.get("conditions") or [] if x]
        f.limitations = [_clean_reason(x) for x in decision.get("limitations") or [] if x]
        f.would_change = [_clean_reason(x) for x in decision.get("what_would_change_it") or [] if x]
    if not f.headline.get("answer") and (ws.report or {}).get("answer"):
        # No Interpreter decision (an older workspace, or a face that wrote its own report): the Author's answer.
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", str(ws.report["answer"]))
        f.headline.setdefault("value", None)
        f.headline["answer"] = re.split(r"(?<=[.!?])\s+", answer.strip())[0]
        f.headline["source"] = "author"
        if f.grade == "not_established" and any(s.get("ok") for s in f.steps):
            try:
                from aquascope.studio.roles.interpreter import grade_for_study

                f.grade = grade_for_study(ws)[0]
            except Exception:  # noqa: BLE001 - a thin workspace keeps the cautious grade
                f.grade = str((ws.report or {}).get("grade") or "not_established")
    f.data_requests = [{"what": _clean_reason(r.get("what")), "why": _clean_reason(r.get("effect_on_grade")
                                                                                   or r.get("why"))}
                       for r in findings.get("data_requests") or [] if isinstance(r, dict) and r.get("what")]
    f.consistency = [x for x in findings.get("consistency") or [] if isinstance(x, dict)]

    primary = str(findings.get("primary_step") or "")
    method = steps_by_id.get(primary, {}).get("method") if primary else None
    status, reason = _sufficiency_reason(ws, method)
    f.grade_reason = _grade_reason(f, status, reason)

    plan = study.get("plan") or {}
    f.caveats = [_clean_reason(x) for x in plan.get("caveats") or [] if x]
    f.references = list((ws.report or {}).get("references") or [])
    rep = ws.report or {}
    for s in rep.get("sections") or []:
        if isinstance(s, dict) and s.get("id") and s.get("text"):
            f.prose[str(s["id"])] = str(s["text"])
    f.prose["_written_by"] = "model" if any(v not in ("template", "device") and v for v in
                                            (rep.get("written_by") or {}).values()) else "template"

    if ws.inventory:
        for d in ws.inventory.datasets:
            if d.station_id and d.kind == "station":
                f.nearby.append({"name": d.name or d.station_id, "id": station_label(d.source, d.station_id),
                                 "variable": d.variable, "km": d.distance_km, "years": d.years,
                                 "start": d.start, "end": d.end, "source": d.source})
    return f


def _grade_reason(f: Facts, status: str, reason: str) -> str:
    g = f.grade
    failed = [ch for ch in f.checks if ch.verdict == "failed"]
    if g == "established":
        return "Every check on the answer passed and the method was rated defensible at this site."
    if g == "indicative":
        if status in ("marginal", "marginally_defensible") and reason:
            return f"The method was rated marginal at this site: {reason}."
        if failed:
            return "A supporting check did not pass: " + failed[0].sentence + "."
        return "One of the conditions for an established answer was not met (see the checks)."
    if g == "screening":
        return ("The answer rests on modelled, reanalysis or transferred data, not on a measured record at the "
                "site.")
    if f.failures:
        first = f.failures[0]
        return f"The {first['what']} did not complete: {first['why']}."
    return "No step established a result that answers the question."
