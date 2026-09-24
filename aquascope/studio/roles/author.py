"""The Author: the report, from the results and the critique, never from memory.

Keyless, the prose is the Solve team's template sentences per step
(``team._sentences_for`` and its sentence makers), the key numbers are
harvested from the payloads, the references are the registry citations of
the methods used plus the playbook's and the software citation. With a
model, ONE call writes the prose of every section from the compact results,
the brief, the "not established" list and the caveats; the numbers must be
in the results (the Critic checks). Every sentence the model wrote passes
the Critic's number check before it is kept (:func:`_checked`): one whose
numbers or years are in no result is dropped and counted in
``report["dropped"]`` and the footer. With ``issues`` from the Critic (the
model's fixes and the failed deterministic checks) one more call applies the
fixes; keyless, the template repair drops the sentences the checks refuse.
The report's sections, in order: summary, problem, site and data,
methodology, results (one per step), limitations, recommendations,
references, appendix. Prose a caller's own model wrote arrives through
:func:`narrate` and goes through the same sentence check, and
``report["written_by"]`` says who wrote which section.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from aquascope import __version__
from aquascope.studio.model import Model, compact
from aquascope.studio.prompts import AUTHOR, AUTHOR_FIX
from aquascope.studio.workspace import Workspace
from aquascope.study import Study

__all__ = ["author_report", "footer_line", "key_numbers", "narrate", "references", "report_context", "to_markdown"]

#: The concept DOI in CITATION.cff; it resolves to the latest archived version.
SOFTWARE_DOI = "10.5281/zenodo.21903143"

SECTION_TITLES: dict[str, str] = {
    "summary": "Summary",
    "decision": "The decision",
    "findings": "Findings",
    "problem": "Problem and decision",
    "site_data": "Site and data",
    "methodology": "Methodology",
    "results": "Results",
    "limitations": "Limitations and what this study does not establish",
    "recommendations": "Recommendations",
    "references": "References",
    "appendix": "Appendix: reproducibility",
}

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: "p = 4.3e-07" or "p = 0.000" in a template sentence: below 0.001 a p-value is reported as "p < 0.001".
_P_EQUALS = re.compile(r"\bp = (-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)")


def p_text(p: Any) -> Any:
    """A p-value as a report writes it: the number, or ``"< 0.001"`` below that threshold."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or p != p:
        return p
    return "< 0.001" if 0 <= p < 0.001 else p


def _small_p(text: str) -> str:
    """Every ``p = X`` with X below 0.001 rewritten as ``p < 0.001``."""
    def swap(m: re.Match[str]) -> str:
        try:
            value = float(m.group(1))
        except ValueError:
            return m.group(0)
        return "p < 0.001" if 0 <= value < 0.001 else m.group(0)

    return _P_EQUALS.sub(swap, text)


def _sig(x: Any, digits: int = 4) -> Any:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if isinstance(x, int):
        return x
    return float(f"{x:.{digits}g}")


def _fmt(x: Any) -> str:
    from aquascope.ai_engine.team import _fmt as fmt

    return fmt(x)


# ── key numbers ─────────────────────────────────────────────────────────────


def _rp_index(payload: dict[str, Any], rp: Any) -> int | None:
    periods = (payload.get("ffa") or {}).get("return_periods") or []
    try:
        return [float(p) for p in periods].index(float(rp))
    except (ValueError, TypeError):
        return None


def _t(x: Any) -> Any:
    """A return period as a label writes it: 100, not 100.0."""
    return int(x) if isinstance(x, (int, float)) and float(x).is_integer() else x


def _drought_class(value: Any) -> str | None:
    """The McKee et al. (1993) class of an SPI or SPEI value, in words."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
        return None
    for floor, name in ((2.0, "extremely wet"), (1.5, "very wet"), (1.0, "moderately wet")):
        if value >= floor:
            return name
    for floor, name in ((-1.0, "near normal"), (-1.5, "moderately dry"), (-2.0, "severely dry")):
        if value > floor:
            return name
    return "extremely dry"


def _numbers_for(sid: str, tool: str, p: dict[str, Any], rp: Any) -> list[dict[str, Any]]:  # noqa: C901
    out: list[dict[str, Any]] = []
    unit = p.get("unit") or ""

    def add(label: str, value: Any, unit_: str | None = None, **more: Any) -> None:
        if value is None or (isinstance(value, float) and value != value):
            return
        out.append({"label": label, "value": _sig(value), "unit": unit_ if unit_ is not None else unit, "step": sid,
                    **more})

    if tool in ("analyze_station", "flood_frequency", "get_timeseries", "load_table"):
        if p.get("years") is not None:
            add("Record length", p.get("years"), "years")
        stats = p.get("stats") or {}
        if stats.get("mean") is not None and tool != "flood_frequency":
            add("Mean of the record", stats["mean"])
        fits = (p.get("ffa") or {}).get("fits") or {}
        periods = (p.get("ffa") or {}).get("return_periods") or []
        idx = _rp_index(p, rp)
        gev = fits.get("gev_lmoments") or {}
        lp3 = fits.get("lp3") or {}
        boot = fits.get("gev_bootstrap") or {}
        others: list[dict[str, Any]] = []
        if fits and idx is not None:
            if gev.get("q"):
                add(f"{rp}-year return level, GEV (L-moments)", gev["q"][idx])
            if lp3.get("q"):
                add(f"{rp}-year return level, Log-Pearson III", lp3["q"][idx])
                ci = (lp3.get("ci") or [None] * (idx + 1))[idx]
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    add(f"{rp}-year LP3 90 % interval, low", ci[0])
                    add(f"{rp}-year LP3 90 % interval, high", ci[1])
            if boot.get("ci"):
                ci = (boot.get("ci") or [None] * (idx + 1))[idx]
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    add(f"{rp}-year GEV bootstrap 90 % interval, low", ci[0])
                    add(f"{rp}-year GEV bootstrap 90 % interval, high", ci[1])
        if fits and periods:
            # Both fits at every other return period the step reported, after the headline's rows.
            for i, period in enumerate(periods):
                if i == idx:
                    continue
                for name, fit in (("GEV (L-moments)", gev), ("Log-Pearson III", lp3)):
                    q = fit.get("q") or []
                    if i < len(q) and q[i] is not None:
                        others.append({"label": f"{_t(period)}-year return level, {name}", "value": _sig(q[i]),
                                       "unit": unit, "step": sid})
        fdc = p.get("fdc") or {}
        for key, label in (("q95", "Q95 (exceeded 95 % of days)"), ("q50", "Q50 (median flow)"), ("q10", "Q10")):
            if fdc.get(key) is not None:
                add(label, fdc[key])
        from aquascope.trend_series import reported_trend

        trend = reported_trend(p) or {}  # the annual maxima for a flood question, the annual mean otherwise
        if isinstance(trend, dict) and trend.get("p_value") is not None:
            add(f"Mann-Kendall p-value ({trend.get('on') or 'annual mean'})", p_text(trend["p_value"]), "")
            add("Sen's slope", trend.get("sens_slope_per_year"), f"{unit} per year" if unit else "per year")
        out += others
    elif tool == "describe_catchment":
        attrs = p.get("attributes") or {}
        area = attrs.get("upstream_area_km2") or attrs.get("area_km2") or p.get("upstream_area_km2")
        add("Upstream area", area, "km2")
    elif tool == "similar_basins":
        add("Donor gauges", p.get("k") or len(p.get("stations") or []), "")
    elif tool == "regionalize_signatures":
        est = p.get("estimates") or {}
        for key in ("q_mean_mm", "q95_mm", "q05_mm", "q_annual_max_mm", "runoff_ratio", "baseflow_index"):
            e = est.get(key)
            if isinstance(e, dict) and e.get("value") is not None:
                label, u = str(e.get("label") or key), str(e.get("unit") or "")
                add(label, e["value"], u)
                if e.get("low") is not None and e.get("high") is not None:
                    add(f"{label} band, low", e["low"], u)
                    add(f"{label} band, high", e["high"], u)
    elif tool == "anywhere":
        cl = p.get("climate") or {}
        add("ERA5 precipitation", cl.get("precipitation_mm_per_year"), "mm per year")
        add("ERA5 reference evapotranspiration", cl.get("et0_mm_per_year"), "mm per year")
        add("Aridity index", cl.get("aridity_index"), "")
        g = p.get("glofas") or {}
        add("GloFAS mean discharge (cell)", (g.get("stats") or {}).get("mean"), "m3/s")
    elif tool == "drought_indices":
        cur = p.get("current") or {}
        head = p.get("headline_timescale")
        scales = [str(t) for t in (p.get("timescales") or [])] or sorted(
            {k for name in ("spi", "spei") for k in (cur.get(name) or {})}, key=lambda k: float(k))
        ordered = ([str(head)] if head is not None and str(head) in scales else []) + \
            [t for t in scales if str(t) != str(head)]
        for t in ordered:
            for name in ("spi", "spei"):
                v = (cur.get(name) or {}).get(t)
                if v is not None:
                    cls = _drought_class(v)
                    add(f"{name.upper()} at {t} months, {cur.get('date')}" + (f" ({cls})" if cls else ""), v, "",
                        **({"class": cls} if cls else {}))
        temp = p.get("temperature") or {}
        add("ERA5 temperature trend", temp.get("trend_c_per_decade"), "C per decade")
    elif tool == "drought_propagation":
        sgi = p.get("sgi") or {}
        add(f"SGI now ({sgi.get('date')})", sgi.get("current"), "")
        add("SGI worst on record", sgi.get("worst"), "")
        best = (p.get("propagation") or {}).get("best") or {}
        add("Rainfall-to-groundwater lag", best.get("lag_months"), "months")
    elif tool == "low_flow_context":
        fdc = p.get("fdc") or {}
        add("Q95", fdc.get("q95"))
        add("Q50", fdc.get("q50"))
        add("7Q10", (p.get("low_flow") or {}).get("7q10"))
        add("Baseflow index", p.get("bfi"), "")
    elif tool == "supply_reliability":
        rel = p.get("reliability") or {}
        if rel.get("daily") is not None:
            add("Days the demand is met", 100 * float(rel["daily"]), "%")
        if rel.get("annual") is not None:
            add("Years without a shortfall", 100 * float(rel["annual"]), "%")
        if rel.get("volumetric") is not None:
            add("Volume delivered", 100 * float(rel["volumetric"]), "%")
        add("Flow the river must carry", p.get("required_flow_m3s"), "m3/s")
        add("Demand", p.get("demand_m3s"), "m3/s")
        if p.get("verdict"):
            out.append({"label": "Verdict", "value": str(p["verdict"]), "unit": "", "step": sid})
        worst = rel.get("worst_year") or {}
        if isinstance(worst, dict) and worst.get("year") is not None and worst.get("days_short") is not None:
            add(f"Days short in the worst year ({worst['year']})", worst["days_short"], "days")
        by_year = rel.get("by_year")
        rows = ([{"year": k, "value": v} for k, v in by_year.items()] if isinstance(by_year, dict)
                else [r for r in by_year if isinstance(r, dict)] if isinstance(by_year, list) else [])
        for r in rows:
            value = r.get("value", r.get("reliability", r.get("daily")))
            if r.get("year") is not None and isinstance(value, (int, float)):
                add(f"Days the demand is met in {r['year']}", 100 * float(value), "%")
    elif tool == "crop_water_demand":
        d = p.get("demand") or {}
        add("Gross irrigation", d.get("gross_irrigation_mm"), "mm")
        add("Net irrigation", d.get("net_irrigation_mm"), "mm")
        add("Mean demand over the season", d.get("mean_m3s"), "m3/s")
        add("Peak-month demand", d.get("peak_month_m3s"), "m3/s")
    elif tool == "sgi_drought":
        add("SGI now", p.get("current"), "")
        add("SGI worst", p.get("worst"), "")
    elif tool == "recharge":
        add("Recharge (water-table fluctuation)", p.get("value_mm_per_year"), "mm per year")
    elif tool == "water_quality_samples":
        add("Samples", p.get("n_samples"), "")
    elif tool == "wqi":
        ccme = p.get("ccme") or {}
        add("CCME WQI", ccme.get("score"), "of 100")
    elif tool == "who_screen":
        add("WHO guideline alerts", p.get("n_alerts"), "")
    elif tool == "return_periods":
        periods = p.get("return_periods") or []
        levels = p.get("return_levels") or []
        dist = str(p.get("distribution") or "").upper()
        try:
            idx = [float(x) for x in periods].index(float(rp))
        except (ValueError, TypeError):
            idx = None
        if idx is not None and idx < len(levels):
            add(f"{rp}-year return level, {dist or 'fit'} on the table", levels[idx])
            lo, hi = (p.get("lower_bound") or []), (p.get("upper_bound") or [])
            if idx < len(lo) and idx < len(hi):
                add(f"{rp}-year {dist} interval, low", lo[idx])
                add(f"{rp}-year {dist} interval, high", hi[idx])
        add("Years of annual maxima", p.get("n_years"), "years")
    elif tool == "flow_duration":
        pct = p.get("percentiles") or {}
        wanted = {95: "Q95", 50: "Q50", 10: "Q10"}
        for key, value in (pct.items() if isinstance(pct, dict) else []):
            try:
                number = float(str(key).lower().lstrip("q"))
            except ValueError:
                continue
            if number in wanted and value is not None:
                add(wanted[int(number)], value)
    return out


def key_numbers(study: Study, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The numbers the report quotes, harvested from the payloads of the steps that ran."""
    params = (study.problem or {}).get("params") or {}
    rp = params.get("return_period") or 100
    out: list[dict[str, Any]] = []
    for r in results:
        recs = [(str(r.get("id")), str(r.get("tool")), r.get("result"), bool(r.get("ok")))]
        fb = r.get("fallback")
        if isinstance(fb, dict) and fb.get("tool"):
            recs.append((f"{r.get('id')}.fallback", str(fb["tool"]), fb.get("result"), bool(fb.get("ok"))))
        for sid, tool, payload, ok in recs:
            if ok and isinstance(payload, dict) and not payload.get("error"):
                out += _numbers_for(sid, tool, payload, rp)
    # Two steps on the same record quote the same number (analyze_station and flood_frequency both carry the
    # fit): keep one row, attributed to the later step, which is the one the plan names for it.
    seen: dict[tuple[str, Any], int] = {}
    deduped: list[dict[str, Any]] = []
    for kn in out:
        k = (kn["label"], kn["value"])
        if k in seen:
            deduped[seen[k]]["step"] = kn["step"]
            continue
        seen[k] = len(deduped)
        deduped.append(kn)
    return deduped


# ── references ──────────────────────────────────────────────────────────────


def software_citation() -> str:
    from aquascope.reporting.builder import ReportBuilder

    builder = ReportBuilder("AquaScope Studio report", author="Rekin226 and contributors")
    builder.metadata.doi = SOFTWARE_DOI
    return builder.software_citation()


_DOI = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+", re.I)
_YEAR_IN = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_AUTHOR = re.compile(r"([A-Z][A-Za-z'\-]+)")


def _ref_key(text: str) -> str:
    """One key per work: the DOI when there is one, else the first author's surname and the year."""
    m = _DOI.search(text)
    if m:
        return "doi:" + m.group(0).rstrip(".").lower()
    year = _YEAR_IN.search(text)
    author = _AUTHOR.search(text)
    if year and author:
        return f"{author.group(1).lower()}:{year.group(1)}"
    return " ".join(text.lower().split())


def references(ws: Workspace) -> list[str]:
    """The works behind the study, once each: the registry citations of the methods the steps name, the
    payloads' own method citations, the playbook's citations and the software citation. Two entries for the
    same work (a DOI, or the same first author and year) collapse to the fuller one."""
    from aquascope.methods import METHODS

    study = ws.study_obj()
    order: list[str] = []
    best: dict[str, str] = {}

    def add(text: str | None) -> None:
        text = " ".join(str(text or "").split())
        if not text:
            return
        # A registry line may cite two works ("England et al. (2019) ...; Hosking (1990) ..."): split them.
        parts = [p.strip() for p in text.split(";")] if _YEAR_IN.search(text) and text.count("(") >= 2 \
            and "; " in text and not _DOI.search(text) else [text]
        for part in parts:
            if not part:
                continue
            key = _ref_key(part)
            if key not in best:
                order.append(key)
                best[key] = part
            elif len(part) > len(best[key]):
                best[key] = part

    for s in (study.steps if study else []):
        m = METHODS.get(s.method or "")
        if m is not None and m.citation:
            add(m.citation)
    for r in (ws.run or {}).get("results") or []:
        payloads = [r.get("result")] + ([r["fallback"].get("result")] if isinstance(r.get("fallback"), dict) else [])
        for p in payloads:
            for m in ((p or {}).get("methods") or []) if isinstance(p, dict) else []:
                if isinstance(m, dict) and m.get("citation"):
                    add(m["citation"])
                elif isinstance(m, str):
                    add(m)
    for c in ((study.plan or {}).get("citations") or []) if study else []:
        add(c)
    add(software_citation())
    return [best[k] for k in order]


# ── template prose ──────────────────────────────────────────────────────────


def _md_table(rows: list[list[Any]], header: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) if c is not None else "" for c in row) + " |")
    return "\n".join(lines)


_STOP = frozenset({"the", "and", "with", "its", "for", "from", "over", "a", "an", "of", "at", "in", "on", "to", "per",
                   "requested", "their", "current", "status"})


def _answer_from(prose: str, key: list[dict[str, Any]], quantities: list[str] | None = None) -> str:
    """The finding in two to four sentences: the paragraph of the template prose that speaks of the brief's
    quantities and quotes the most key numbers (the last one when tied, the main step usually comes last)."""
    from aquascope.ai_engine.verify import _numbers

    values = {float(k["value"]) for k in key if isinstance(k.get("value"), (int, float))
              and not isinstance(k.get("value"), bool)}
    words = {w for q in (quantities or []) for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) > 2
             and w not in _STOP}
    best, best_score = None, -1.0
    for para in prose.split("\n\n"):
        if para.lower().startswith("plan ") or not re.search(r"\d", para):
            continue
        found = set(_numbers(para, claims_only=True))
        low = para.lower()
        score = sum(1 for v in values if any(abs(v - f) <= max(abs(v), 1e-9) * 0.005 + 1e-9 for f in found))
        score += 3 * sum(1 for w in words if re.search(rf"\b{re.escape(w)}", low))
        if score >= best_score:
            best, best_score = para, score
    if best is not None:
        sentences = [s for s in _SENTENCE.split(best.strip()) if s]
        # The sentences that speak of the brief's quantities lead (the 100-year flow before the record's mean),
        # so the answer opens with the number asked for; the order among equals is the paragraph's own.
        def asked(sentence: str) -> int:
            low = sentence.lower()
            return sum(1 for w in words if re.search(rf"\b{re.escape(w)}", low))

        sentences = sorted(sentences, key=lambda sentence: -asked(sentence))
        return " ".join(sentences[:4])
    if key:
        kn = key[0]
        return f"{kn['label']}: {_fmt(kn['value'])} {kn.get('unit') or ''}".strip() + "."
    return "No step produced a number; see what the study does not establish."


def _record_names(ws: Workspace) -> list[tuple[str, str, str]]:
    """``(source, station_id, name)`` for every named record the run touched or the inventory lists."""
    out: dict[tuple[str, str], str] = {}
    for r in (ws.run or {}).get("results") or []:
        for p in (r.get("result"), (r.get("fallback") or {}).get("result") if isinstance(r.get("fallback"), dict)
                  else None):
            if isinstance(p, dict) and p.get("source") and p.get("station_id"):
                name = p.get("station_name") or p.get("name")
                if isinstance(name, str) and name.strip() and name.strip() != str(p["station_id"]):
                    out.setdefault((str(p["source"]), str(p["station_id"])), name.strip())
    for d in (ws.inventory.datasets if ws.inventory else []):
        if d.source and d.station_id and d.name and d.name != d.station_id:
            out.setdefault((str(d.source), str(d.station_id)), d.name)
    return [(k[0], k[1], v) for k, v in out.items()]


def name_records(text: str, names: list[tuple[str, str, str]]) -> str:
    """"Kingston (uk_ea 8496ce69...)" the first time a record is named, "Kingston" after that."""
    for source, sid, name in names:
        full = f"{name} ({source} {sid})"
        first = [full not in text]
        pat = re.compile(rf"(?:\b(?:station|gauge|well|rain gauge)\s+)?(?<![(\w])"
                         rf"{re.escape(source)}\s+{re.escape(sid)}(?![\w-])")

        def swap(m: re.Match[str]) -> str:
            if first[0]:
                first[0] = False
                return full
            return name

        text = pat.sub(swap, text)
    return text


def _step_prose(sid: str, r: dict[str, Any], study: Study) -> str:
    from aquascope.ai_engine.team import _sentences_for

    lines: list[str] = []
    if r.get("skipped"):
        lines.append(f"Step {sid} ({r.get('tool')}) was skipped: {r.get('error')}.")
    elif not r.get("ok"):
        lines.append(f"Step {sid} ({r.get('tool')}) failed: {r.get('error')}.")
    else:
        payload = r.get("result")
        if isinstance(payload, dict):
            lines += _sentences_for(str(r.get("tool")), payload, study)
        if not lines:
            from aquascope.study import _summarise

            lines.append(f"{r.get('tool')} returned: {_summarise(payload)}.")
    gates = r.get("gates") or []
    if gates:
        lines.append("Gates: " + "; ".join(f"{g.get('check')} {'passed' if g.get('passed') else 'FAILED'}"
                                            f" ({g.get('detail')})" for g in gates) + ".")
    fb = r.get("fallback")
    if r.get("fallback_used") and isinstance(fb, dict):
        state = "passed its gates" if fb.get("ok") and fb.get("gates_passed") else (
            "failed: " + str(fb.get("error")) if not fb.get("ok") else "did not pass its gates")
        lines.append(f"The fallback {fb.get('tool')} ran and {state}.")
        if isinstance(fb.get("result"), dict):
            lines += _sentences_for(str(fb.get("tool")), fb["result"], study)
    return _small_p(" ".join(lines))


def _summary_paragraph(ws: Workspace, study: Study, results: list[dict[str, Any]], key: list[dict[str, Any]],
                       missing: list[str], answer: str) -> str:
    """One paragraph: what was asked, on what record, what ran and how the gates went, the headline number."""
    plan = study.plan or {}
    run = ws.run or {}
    records = []
    for r in results:
        p = r.get("result")
        if isinstance(p, dict) and p.get("source") and p.get("station_id"):
            label = p.get("station_name") or p.get("name")
            label = f"{label} ({p['source']} {p['station_id']})" if label else f"{p['source']} {p['station_id']}"
            if p.get("years"):
                label += f", {_fmt(p['years'])} years"
            if label not in records:
                records.append(label)
    gates = run.get("gates") or []
    passed = sum(1 for g in gates if g.get("passed"))
    bits = [f"{plan.get('objective') or ws.brief.problem}."]
    if records:
        bits.append("The record: " + "; ".join(records[:3]) + ".")
    bits.append(f"{len(results)} step(s) ran" + (f" ({plan.get('author')} plan"
                + (f", playbook {plan['playbook']}" if plan.get("playbook") else "") + ")")
                + (f"; {passed} of {len(gates)} gates passed" if gates else "")
                + (f"; the study stopped at {run.get('stopped_at')}" if run.get("stop_reason") else "") + ".")
    head = [s for s in _SENTENCE.split(answer) if re.search(r"\d", s)]
    if head:
        bits.append(head[-1] if len(head[-1]) > 40 else head[0])
    if missing:
        bits.append(f"{len(missing)} point(s) are listed under what this study does not establish.")
    return " ".join(bits)


def _site_rows(ws: Workspace, study: Study) -> tuple[list[list[Any]], int]:
    """The inventory rows worth a line (a year of record or more, or used by the plan) and how many short
    records were left out (G)."""
    inv = ws.inventory
    if inv is None:
        return [], 0
    used = {(str(st.arguments.get("source")), str(st.arguments.get("station_id"))) for st in study.steps
            if st.arguments.get("station_id")}
    rows: list[list[Any]] = []
    short = 0
    for d in inv.datasets:
        listed = d.kind != "station" or (d.years or 0) >= 1 or (str(d.source), str(d.station_id)) in used
        if not listed:
            short += 1
            continue
        who = f"{d.source} {d.station_id}" if d.station_id else (d.source or "")
        end = d.end or ("present" if d.start else "?")
        span = f"{d.start or '?'} to {end}" if (d.start or d.end) else ""
        rows.append([d.id, d.kind, d.variable or "", who, d.name or "", f"{d.years:g}" if d.years else "",
                     d.resolution or "", d.distance_km if d.distance_km is not None else "", span,
                     (d.quality or {}).get("verdict") or ""])
    return rows, short


def _recommendations(ws: Workspace, study: Study, missing: list[str]) -> list[str]:
    """Two to four bullets from the gates' outcomes, the registry's marginal verdicts and the playbook's caveats;
    never the answer again (H)."""
    from aquascope.methods import METHODS

    run = ws.run or {}
    plan = study.plan or {}
    out: list[str] = []
    decision = (ws.findings or {}).get("decision") or {}
    if decision.get("answer"):
        out.append(f"Adopt this as the answer to the decision: {decision['answer']}")
        for c in (decision.get("what_would_change_it") or [])[:2]:
            out.append(f"To firm it up: {c}.")
        for r in ((ws.findings or {}).get("data_requests") or [])[:1]:
            out.append(f"Obtain {r.get('what')}: {r.get('effect_on_grade') or r.get('why')}.")
    for g in (run.get("failed_gates") or [])[:2]:
        if len(out) >= 4:
            break
        out.append(f"Before relying on step {g.get('step')}, settle the failed gate {g.get('check')}: "
                   f"{g.get('detail')}. A longer record or another source would.")
    if run.get("replans") or any(r.get("fallback_used") for r in run.get("results") or []):
        out.append("A fallback ran after a failed gate; its numbers are indicative, not a substitute for the "
                   "step that failed.")
    suff = {r.get("method"): r for r in (ws.inventory.sufficiency if ws.inventory else []) if isinstance(r, dict)}
    for st in study.steps:
        row = suff.get(st.method)
        if row and row.get("status") == "marginal" and len(out) < 3:
            label = METHODS[st.method].label if st.method in METHODS else st.method
            out.append(f"{label} is marginal here ({row.get('reason')}); a longer record would firm it up.")
    for g in run.get("gates") or []:
        if g.get("check") == "spread_within" and g.get("passed") and len(out) < 3:
            out.append(f"Quote both fits with their intervals: {g.get('detail')}.")
            break
    for c in (plan.get("caveats") or []):
        if len(out) >= 5:
            break
        out.append(f"Read the numbers with this caveat: {_first_sentence(str(c))}")
    if not out:
        out.append("Re-run the study file when the record is updated; the gates will say whether the estimate "
                   "moved.")
    return out[:5]


_SENTENCE_END = re.compile(r"(?<!\bal)(?<!\bet)(?<!\bvs)(?<![A-Z])[.!?](?=\s+[A-Z(]|$)")


def _first_sentence(text: str) -> str:
    """The first sentence, with "et al.", "vs." and initials not taken for its end."""
    text = " ".join(text.split())
    m = _SENTENCE_END.search(text)
    return text[:m.end()] if m else text


def _template_sections(ws: Workspace, study: Study, results: list[dict[str, Any]], key: list[dict[str, Any]],
                       missing: list[str], refs: list[str], answer: str) -> dict[str, str]:
    b = ws.brief
    plan = study.plan or {}
    site = ws.site or {}
    sections: dict[str, str] = {}
    sections["summary"] = _summary_paragraph(ws, study, results, key, missing, answer)
    findings = ws.findings or {}
    decision = findings.get("decision") or {}
    if findings.get("findings") or decision:
        lines = []
        if decision.get("answer"):
            lines.append(str(decision["answer"]))
        if decision.get("conditions"):
            lines.append("It holds under these conditions: " + "; ".join(str(c) for c in decision["conditions"]) + ".")
        if decision.get("what_would_change_it"):
            lines.append("What would change it: " + "; ".join(str(c) for c in decision["what_would_change_it"]) + ".")
        for r in findings.get("data_requests") or []:
            lines.append(f"The crew would ask for {r.get('what')}: {r.get('effect_on_grade') or r.get('why')}.")
        sections["decision"] = " ".join(lines)
        rows = [f"- [{str(f.get('grade') or '').replace('_', ' ')}] {f.get('claim')} "
                f"(from {', '.join(f.get('basis') or [])})" for f in findings.get("findings") or []]
        for c in findings.get("consistency") or []:
            rows.append(f"- {'Agrees' if c.get('agree') else 'Disagrees'}: {c.get('note')}")
        sections["findings"] = "\n".join(rows) if rows else ""

    parts = [b.problem]
    if b.decision:
        parts.append(f"Decision: {b.decision}.")
    if b.quantities:
        parts.append("Quantities wanted: " + "; ".join(b.quantities) + ".")
    if b.period or b.horizon:
        parts.append(f"Period: {b.period or 'not stated'}; horizon: {b.horizon or 'not stated'}.")
    if b.constraints:
        parts.append("Constraints: " + "; ".join(b.constraints) + ".")
    if b.intake:
        parts.append("Intake: " + ", ".join(f"{k} = {v}" for k, v in b.intake.items() if v is not None) + ".")
    if b.assumptions:
        parts.append("Assumed: " + "; ".join(b.assumptions) + ".")
    sections["problem"] = " ".join(parts)

    inv = ws.inventory
    rows, short = _site_rows(ws, study)
    head = (f"Site: {site.get('lat')}, {site.get('lon')}. "
            + ("The datasets within reach or attached:" if rows else "No inventory."))
    tbl = _md_table(rows, ["Id", "Kind", "Variable", "Source", "Name", "Years", "Resolution", "km", "Period",
                           "Quality"])
    tail = (f"and {short} more short record(s) within reach, under a year of record each, not listed."
            if short else "")
    notes = "\n".join(f"- {n}" for n in (inv.notes[:8] if inv else []))
    sections["site_data"] = "\n\n".join(x for x in (head, tbl, tail, notes) if x)

    meth = [f"Objective: {plan.get('objective')}." if plan.get("objective") else ""]
    meth.append("\n".join(f"{i}. {m}" for i, m in enumerate(plan.get("methodology") or [], 1)))
    for st in study.steps:
        args = ", ".join(f"{k}={v!r}" for k, v in st.arguments.items())
        gates = "; ".join(f"{g.get('check')}" + (f" {g['value']}" if g.get("value") is not None else "")
                          + f" on {g.get('path') or ', '.join(g.get('paths') or [])}" for g in st.expects)
        meth.append(f"Step {st.id}: `{st.tool}({args})`" + (f", method {st.method}" if st.method else "")
                    + (f"; gates: {gates}" if gates else "; no gate") + ".")
    if plan.get("assumptions"):
        meth.append("Assumptions: " + "; ".join(str(a) for a in plan["assumptions"]) + ".")
    if plan.get("alternatives"):
        alts = [f"{a.get('method')}: {a.get('why_not')}" if isinstance(a, dict) else str(a)
                for a in plan["alternatives"]]
        meth.append("Alternatives considered: " + "; ".join(alts) + ".")
    if plan.get("notes"):
        meth.append("Notes from planning: " + "; ".join(str(n) for n in plan["notes"]) + ".")
    sections["methodology"] = "\n\n".join(m for m in meth if m)

    for r in results:
        sid = str(r.get("id"))
        sections[f"results-{sid}"] = _step_prose(sid, r, study)
    if not results:
        sections["results-none"] = "No step ran."

    lim = ["\n".join(f"- {m}" for m in missing) if missing else "Every gate and check passed."]
    if plan.get("caveats"):
        lim.append("Caveats, verbatim from the playbook:\n" + "\n".join(f"- {c}" for c in plan["caveats"]))
    if plan.get("limitations_expected"):
        lim.append("Expected at planning:\n" + "\n".join(f"- {c}" for c in plan["limitations_expected"]))
    sections["limitations"] = "\n\n".join(lim)

    sections["recommendations"] = "\n".join(f"- {r}" for r in _recommendations(ws, study, missing))

    sections["references"] = "\n".join(f"{i}. {r}" for i, r in enumerate(refs, 1))

    ledger = ", ".join(f"{role} {v.get('calls', 0)} call(s), "
                       f"{v.get('prompt_tokens', 0) + v.get('completion_tokens', 0)} tokens"
                       for role, v in ws.ledger.items()) or "no model calls"
    sections["appendix"] = "\n\n".join([
        "Re-run the same steps with no model: `aquascope run study.yaml`. Resume the workspace: "
        "`aquascope studio --resume workspace.json`.",
        f"Model: {ws.model or 'none'} via {ws.provider or 'none'}; ledger: {ledger}. aquascope {__version__}.",
        "```yaml\n" + study.to_yaml() + "```",
    ])
    return sections


# ── the report ──────────────────────────────────────────────────────────────


def _section_list(ws: Workspace, texts: dict[str, str], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = {a.step: [] for a in ws.artifacts}
    figs: dict[str | None, list[str]] = {}
    for a in ws.artifacts:
        if a.kind == "figure" and a.id.endswith("-svg") and ws.artifact(a.id[:-4]) is not None:
            continue    # the SVG twin of a PNG figure is the same figure, listed once
        if a.kind not in ("figure", "table"):
            continue
        (figs if a.kind == "figure" else tables).setdefault(a.step, []).append(a.id)
    out: list[dict[str, Any]] = []
    for sid in ("summary", "decision", "findings", "problem", "site_data", "methodology"):
        if sid in ("decision", "findings") and not texts.get(sid):
            continue    # only when the Interpreter ran
        out.append({"id": sid, "title": SECTION_TITLES[sid], "text": texts.get(sid, ""),
                    "figures": figs.get(sid, []), "tables": tables.get(sid, [])})
    result_ids = [str(r.get("id")) for r in results] or ["none"]
    for rid in result_ids:
        step_ids = [rid, f"{rid}.fallback"]
        out.append({"id": f"results-{rid}", "title": f"Results: step {rid}", "text": texts.get(f"results-{rid}", ""),
                    "figures": [f for s in step_ids for f in figs.get(s, [])],
                    "tables": [t for s in step_ids for t in tables.get(s, [])]})
    for sid in ("limitations", "recommendations", "references", "appendix"):
        out.append({"id": sid, "title": SECTION_TITLES[sid], "text": texts.get(sid, ""),
                    "figures": figs.get(sid, []), "tables": tables.get(sid, [])})
    return out


def _draft(ws: Workspace) -> dict[str, Any]:
    """The template report before any model: the study, the results, the key numbers, the references, what is
    not established, the answer, the section texts and the title."""
    study = ws.study_obj() or Study(question=ws.brief.problem, version=3, plan={})
    run = ws.run or {}
    results = list(run.get("results") or [])
    key = key_numbers(study, results)
    refs = references(ws)
    plan = study.plan or {}
    missing = list((ws.critique or {}).get("not_established") or [])
    if not missing:
        from aquascope.studio.roles.critic import not_established

        missing = not_established(ws)
    from aquascope.ai_engine.team import _template_answer
    from aquascope.studio.roles.analysts import prior_run

    quantities = ws.brief.quantities
    if not quantities:
        from aquascope.studio.roles.consultant import _rules_quantities

        quantities = _rules_quantities(ws.brief.playbook, ws.brief.intake)
    names = _record_names(ws)
    answer = name_records(_answer_from(_template_answer(study, prior_run(ws)), key, quantities), names)
    decision = (ws.findings or {}).get("decision") or {}
    if decision.get("answer"):
        from aquascope.studio.roles.interpreter import decision_text

        answer = f"{decision_text(decision)} {answer}".strip()
    texts = _template_sections(ws, study, results, key, missing, refs, answer)
    for sid in list(texts):
        if sid.startswith("results-") or sid == "summary":
            texts[sid] = name_records(texts[sid], names)
    site = ws.site or {}
    title = str(plan.get("objective") or ws.brief.decision or ws.brief.problem)[:80]
    if site:
        title += f" ({site.get('lat')}, {site.get('lon')})"
    return {"study": study, "results": results, "key": key, "refs": refs, "plan": plan, "missing": missing,
            "answer": answer, "texts": texts, "title": title, "site": site}


def report_context(ws: Workspace, *, issues: list[dict[str, Any]] | None = None,
                   draft: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """The system prompt and the context the Author sends a model for the prose (with ``issues``, the fix
    round over the draft in ``ws.report``). A page runs the same prompt on a device model and hands the
    sections back through :func:`narrate`."""
    d = draft or _draft(ws)
    plan, texts, results = d["plan"], d["texts"], d["results"]
    previous = ws.report or {}
    context: dict[str, Any] = {
        "brief": {k: v for k, v in ws.brief.to_dict().items()
                  if k in ("problem", "decision", "quantities", "kind", "playbook", "intake", "assumptions")},
        "site": d["site"],
        "plan": {k: plan.get(k) for k in ("objective", "methodology", "assumptions", "caveats",
                                          "limitations_expected", "branch", "playbook")},
        "steps": [{"id": r.get("id"), "tool": r.get("tool"), "arguments": r.get("arguments"),
                   "rationale": r.get("rationale"), "ok": r.get("ok"), "error": r.get("error"),
                   "gates": r.get("gates"), "fallback_used": r.get("fallback_used"),
                   "result": compact(r.get("result"), max_list=24),
                   "fallback": compact({k: v for k, v in (r.get("fallback") or {}).items()
                                        if k in ("tool", "arguments", "ok", "gates", "result")}, max_list=24)
                   if r.get("fallback") else None} for r in results],
        "key_numbers": d["key"],
        "findings": {k: v for k, v in (ws.findings or {}).items()
                     if k in ("findings", "consistency", "decision", "data_requests", "assumptions")} or None,
        "numbers_rule": "every number in steps[*].result and steps[*].fallback.result may be quoted; "
                        "key_numbers is the summary table's subset, not a whitelist; the answer opens with the "
                        "decision's answer and grade, and the findings section walks the findings",
        "not_established": d["missing"],
        "inventory": [{k: v for k, v in ds.to_dict().items() if k in ("id", "kind", "variable", "source",
                                                                       "station_id", "name", "years")}
                      for ds in (ws.inventory.datasets if ws.inventory else [])][:12],
        "section_ids": [k for k in texts if k not in ("references", "appendix")],
    }
    system = AUTHOR
    if issues:
        system = AUTHOR_FIX
        context["draft"] = {"title": previous.get("title"), "answer": previous.get("answer"),
                            "sections": {s["id"]: s["text"] for s in previous.get("sections") or []
                                         if s["id"] not in ("references", "appendix")}}
        context["issues"] = issues
    return system, context


def author_report(ws: Workspace, model: Model | None, *, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write ``ws.report``; with ``issues`` (from the Critic: its ``fix`` issues and the failed checks) the draft
    is rewritten once with the fixes. Every sentence the model writes passes the Critic's number check first;
    keyless, a fix round is the template's own repair: the sentences the checks refuse are dropped. The
    sentences dropped are counted in ``report["dropped"]`` (across the draft and the fix round)."""
    from aquascope.studio.roles.critic import tool_results

    d = _draft(ws)
    study, results, key, refs, plan, missing = d["study"], d["results"], d["key"], d["refs"], d["plan"], d["missing"]
    answer, texts, title = d["answer"], d["texts"], d["title"]
    prose_by = "template"
    written_by: dict[str, str] = {"answer": "template", **{sid: "template" for sid in texts}}
    before = int((ws.report or {}).get("dropped") or 0) if issues else 0
    dropped = 0
    if model:
        system, context = report_context(ws, issues=issues, draft=d)
        obj = model.call_json("author", system, context)
        if obj:
            pool = tool_results(ws)
            if isinstance(obj.get("title"), str) and obj["title"].strip():
                title = obj["title"].strip()[:120]
            if isinstance(obj.get("answer"), str) and obj["answer"].strip():
                text, k = _checked(obj["answer"].strip(), pool, ws.brief.problem)
                dropped += k
                if text:
                    answer = _with_decision(ws, text)
                    written_by["answer"] = "model"
            written = obj.get("sections") if isinstance(obj.get("sections"), dict) else {}
            n = 0
            for sid, text in written.items():
                if sid in texts and sid not in ("references", "appendix") and isinstance(text, str) and text.strip():
                    text, k = _checked(text.strip(), pool, ws.brief.problem)
                    dropped += k
                    if not text:
                        continue    # nothing survived the check: the template's text stands
                    texts[sid] = text
                    written_by[sid] = "model"
                    n += 1
            prose_by = "model"
            ws.event("author", "prose", f"model wrote {n} section(s)" + (" after the Critic's fixes" if issues else "")
                     + (f"; {dropped} sentence(s) dropped by the checks" if dropped else ""))
        else:
            ws.event("author", "template", "the model gave no usable prose; template prose stands")
    elif issues:
        # Keyless, the template's own repair: the sentences whose numbers or years the checks refuse are dropped,
        # and every failed check is listed under what the study does not establish either way.
        names = sorted({str(i.get("check")) for i in issues if i.get("check")})
        if {"numbers_come_from_tools", "years_traceable"} & set(names):
            pool = tool_results(ws)
            text, k = _checked(answer, pool, ws.brief.problem)
            dropped += k
            if text:
                answer = text
            for sid in list(texts):
                if sid in ("references", "appendix"):
                    continue
                text, k = _checked(texts[sid], pool, ws.brief.problem)
                dropped += k
                if text:
                    texts[sid] = text
        what = ", ".join(names) or "the Critic's issues"
        ws.event("author", "template", f"template repair for {what}: {dropped} sentence(s) dropped")
    else:
        ws.event("author", "template", f"{len(texts)} section(s) from the template")
    if dropped:
        ws.event("critic", "dropped", f"{dropped} sentence(s) quoted numbers or years in no result")
    dropped += before
    findings = ws.findings or {}
    report = {
        "title": title,
        "answer": answer,
        "grade": (findings.get("decision") or {}).get("grade"),
        "decision": findings.get("decision"),
        "findings": findings.get("findings") or [],
        "data_requests": findings.get("data_requests") or [],
        "key_numbers": key,
        "sections": _section_list(ws, texts, results),
        "not_established": missing,
        "recommendations": _bullets(texts["recommendations"]),
        "references": refs,
        "caveats": list(plan.get("caveats") or []),
        "written_by": written_by,
        "dropped": dropped,
        "footer": {
            "model": ws.model, "provider": ws.provider, "prose": prose_by,
            "tokens": {k: dict(v) for k, v in ws.ledger.items()}, "total_tokens": ws.tokens,
            "total_usd": ws.total_usd, "budget": dict(ws.budget) if ws.budget else None, "dropped": dropped,
            "aquascope_version": __version__,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "workspace": ws.id,
            "plan_author": plan.get("author") or study.author,
            "written_by": written_by,
        },
    }
    ws.report = report
    return report


def _with_decision(ws: Workspace, answer: str) -> str:
    """The answer with the decision's value and grade in front when the prose does not carry them: the grade
    is the engine's verdict, and it is not the model's to leave out (#418)."""
    from aquascope.ai_engine.verify import _numbers, normalise
    from aquascope.studio.roles.interpreter import decision_text

    decision = (ws.findings or {}).get("decision") or {}
    if not decision.get("answer"):
        return answer
    value, grade = decision.get("value"), str(decision.get("grade") or "")
    text = normalise(answer)
    has_grade = bool(grade) and grade.replace("_", " ") in text.lower()
    has_value = not isinstance(value, (int, float)) or isinstance(value, bool) or any(
        abs(n - float(value)) <= max(abs(float(value)), 1e-9) * 0.02 + 1e-9
        for n in _numbers(text, claims_only=True) + _numbers(text))
    if has_grade and has_value:
        return answer
    return f"{decision_text(decision)} {answer}".strip()


def _bullets(text: str) -> list[str]:
    """The recommendations as a list: the "- " lines, else the sentences."""
    lines = [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]
    if lines:
        return lines
    return [x.strip() for x in _SENTENCE.split(" ".join(text.split())) if x.strip()]


# ── prose from a caller's own model ─────────────────────────────────────────


def _checked(text: str, results: list[dict[str, Any]], question: str) -> tuple[str, int]:
    """``text`` with every sentence whose numbers (or years) are in no tool result removed, and the count.
    Paragraphs and bullet lines are kept as they are."""
    from aquascope.ai_engine.verify import verify

    dropped = 0
    lines_out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            lines_out.append("")
            continue
        m = re.match(r"^(\s*(?:[-*]|\d+[.)])\s+)", line)
        lead, body = (m.group(1), line[m.end():]) if m else ("", line)
        kept: list[str] = []
        for sentence in _SENTENCE.split(body.strip()):
            if not sentence.strip():
                continue
            v = verify(sentence, results, question=question)
            bad = [c for c in v.failed if c.name in ("numbers_come_from_tools", "years_traceable")]
            if bad:
                dropped += 1
                continue
            kept.append(sentence.strip())
        if kept:
            lines_out.append(lead + " ".join(kept))
    return "\n".join(lines_out).strip(), dropped


def narrate(ws: Workspace, sections: dict[str, str], *, source: str = "device") -> dict[str, Any]:
    """Replace the prose of the named sections of ``ws.report`` (ids as in its ``sections``, plus ``answer``)
    with what a caller's model wrote, after the Critic's number check on every sentence. Returns
    ``{"dropped", "written_by", "written", "ignored"}``; ``ws.report["written_by"]`` and the footer say who wrote
    which section."""
    from aquascope.studio.roles.critic import tool_results

    report = ws.report
    if not report:
        raise ValueError("there is no report to narrate")
    results = tool_results(ws)
    by_id = {s.get("id"): s for s in report.get("sections") or [] if isinstance(s, dict)}
    written_by = dict(report.get("written_by") or {})
    dropped = 0
    written: list[str] = []
    ignored: list[str] = []
    for sid, text in sections.items():
        if sid != "answer" and (sid not in by_id or sid in ("references", "appendix")):
            ignored.append(sid)
            continue
        new_text, n = _checked(str(text), results, ws.brief.problem)
        dropped += n
        if n:
            ws.event("critic", "dropped", f"{n} sentence(s) of {sid} quoted numbers in no result", step=None)
        if not new_text:
            ignored.append(sid)
            continue
        if sid == "answer":
            report["answer"] = _with_decision(ws, new_text)
        else:
            by_id[sid]["text"] = new_text
            if sid == "recommendations":
                report["recommendations"] = _bullets(new_text)
        written_by[sid] = source
        written.append(sid)
    report["written_by"] = written_by
    report["dropped"] = int(report.get("dropped") or 0) + dropped
    footer = report.get("footer")
    if isinstance(footer, dict):
        footer["written_by"] = written_by
        footer["dropped"] = report["dropped"]
    ws.event("author", "narrated", f"{source} wrote {len(written)} section(s)"
             + (f", {dropped} sentence(s) dropped" if dropped else "")
             + (f", ignored {', '.join(ignored)}" if ignored else ""))
    return {"dropped": dropped, "written_by": written_by, "written": written, "ignored": ignored}


def to_markdown(ws: Workspace) -> str:
    """The report as Markdown: title, the answer, every section with its figures and tables referenced by
    their bundle names, and the footer."""
    r = ws.report or {}
    by_id = {a.id: a for a in ws.artifacts}
    lines = [f"# {r.get('title') or ws.brief.problem or 'Study'}", "", str(r.get("answer") or ""), ""]
    table = _md_table([[k["label"], _fmt(k["value"]), k.get("unit") or "", k["step"]]
                       for k in (r.get("key_numbers") or [])[:40]], ["Quantity", "Value", "Unit", "Step"])
    if table:
        lines += [table, ""]
    for s in r.get("sections") or []:
        lines += [f"## {s.get('title')}", "", str(s.get("text") or ""), ""]
        for fid in s.get("figures") or []:
            a = by_id.get(fid)
            if a is not None:
                lines += [f"![{a.caption or a.id}]({a.name})", ""]
        for tid in s.get("tables") or []:
            a = by_id.get(tid)
            if a is not None:
                lines += [f"Table: {a.caption or a.id} ({a.name})", ""]
    lines += ["---", footer_line(ws)]
    return "\n".join(lines) + "\n"


def _usd(x: float) -> str:
    return f"{x:.2f} USD" if x >= 0.1 else f"{x:.4f} USD"


def footer_line(ws: Workspace) -> str:
    """The one line under every document: who produced the report and who wrote which part, the ledger in
    calls, tokens and USD, the spend ceiling when it was reached, the sentences the checks dropped, and the
    honesty sentence. A footer that is already a string (an older workspace) is returned as it is."""
    r = ws.report or {}
    f = r.get("footer")
    if isinstance(f, str):
        return f
    f = f if isinstance(f, dict) else {}
    model, provider = f.get("model", ws.model), f.get("provider", ws.provider)
    who = (f"model {model} via {provider}" if model
           else "no model: the playbook tree filled the plan and a template wrote the prose")
    by: dict[str, list[str]] = {}
    for sid, src in (f.get("written_by") or r.get("written_by") or {}).items():
        if src not in ("template", "model"):
            by.setdefault(str(src), []).append(str(sid))
    if by:
        who += "; " + "; ".join(f"{src} wrote {', '.join(ids)}" for src, ids in by.items())
    ledger = f.get("tokens") if isinstance(f.get("tokens"), dict) else ws.ledger
    calls = sum(int(v.get("calls", 0)) for v in ledger.values() if isinstance(v, dict))
    tokens = f.get("total_tokens") if f.get("total_tokens") is not None else ws.tokens
    usd = f.get("total_usd") if "total_usd" in f else ws.total_usd
    spend = (f" ({tokens} tokens" + (f", {_usd(float(usd))}" if usd is not None else "") + ")") if tokens else ""
    parts = [f"Produced by aquascope {f.get('aquascope_version', __version__)} (`aquascope studio`), "
             f"plan by {f.get('plan_author') or 'playbook'}, {who}, {f.get('date', '')}. "
             f"Model calls: {calls}{spend}."]
    budget = f.get("budget") if isinstance(f.get("budget"), dict) else ws.budget
    if budget:
        parts.append(f"The spend ceiling of {_usd(float(budget.get('max_usd') or 0))} was reached "
                     f"({_usd(float(budget.get('spent_usd') or 0))} after the {budget.get('role')}'s call); the roles "
                     "ran keyless from there.")
    dropped = f.get("dropped") if f.get("dropped") is not None else r.get("dropped")
    if dropped:
        parts.append(f"{dropped} sentence(s) were dropped by the checks: their numbers were in no result.")
    parts.append("Numbers come from the tool results and were checked by the gates and the Critic; the study file "
                 "re-runs the same steps with `aquascope run`.")
    return " ".join(parts)


def report_json(ws: Workspace) -> str:
    return json.dumps(ws.report or {}, ensure_ascii=False, indent=2, default=str)
