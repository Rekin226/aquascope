"""The tables of a study: one CSV artifact per table name a step yields, from its payload.

The catalogue names the tables each tool yields (``return_levels``,
``indices_monthly``, ``donors``, ...); :func:`tables_for` builds them and
returns :class:`Artifact` rows the report, the workbook and the bundle
carry. Each maker reads the payload defensively and yields nothing when the
keys it needs are absent. :func:`frame_of` turns a table artifact back into
a DataFrame (pandas is imported there, not at module import).
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Callable
from typing import Any

from aquascope.studio.deliverables._payload import (
    annual_maxima_of,
    fdc_of,
    flatten,
    frame_records,
    indices_of,
    is_scalar,
    num,
    numbers,
    record_name,
    records,
    return_levels_of,
    series_of,
    stations_of,
)
from aquascope.studio.workspace import MEDIA_TYPES, Artifact

logger = logging.getLogger(__name__)

Table = tuple[list[str], list[list[Any]]]

#: Captions per table name; ``{record}`` is the record the payload names.
CAPTIONS: dict[str, str] = {
    "series": "The record at {record} (datetime, value).",
    "summary": "Summary of the record at {record}.",
    "annual_maxima": "Annual maxima at {record}.",
    "return_levels": "Return levels at {record} by return period, with the confidence band.",
    "fit_spread": "Spread between the GEV and Log-Pearson III return levels at {record}.",
    "fdc_percentiles": "Flow-duration percentiles at {record}.",
    "trend": "Mann-Kendall trend test and Sen slope at {record}.",
    "glofas_summary": "GloFAS modelled discharge for the grid cell at {record} (indicative).",
    "monthly_climate": "Mean monthly precipitation and reference evapotranspiration for the ERA5 cell at {record}.",
    "indices_monthly": "Monthly SPI and SPEI at {record} per timescale.",
    "drought_events": "Drought classes, worst months and event counts per timescale at {record}.",
    "index_divergence": "Divergence between SPEI and SPI per timescale at {record}.",
    "sgi_monthly": "Monthly Standardised Groundwater Index at {record}.",
    "propagation_lag": "Cross-correlation lag between SPI and SGI per accumulation at {record}.",
    "events": "Drought events at {record}.",
    "donors": "Donor gauges selected for {record}.",
    "signatures": "Flow signatures at {record}.",
    "stations": "Stations found.",
    "stations_within_reach": "Catalogue stations within reach of the site.",
    "sufficiency": "The registry's verdict per method at the site.",
    "catchment_attributes": "Catchment attributes from BasinATLAS for {record}.",
    "low_flow_stats": "Low-flow statistics at {record}.",
    "reliability": "Supply reliability at {record}.",
    "demand_monthly": "Crop water demand per season at {record}.",
    "et0_monthly": "Reference evapotranspiration by calendar month.",
    "demand": "Season totals of the crop water requirement.",
    "schedule": "The daily irrigation schedule.",
    "et0": "Daily reference evapotranspiration.",
    "samples": "Water-quality samples at {record}.",
    "sample_counts": "Samples per parameter at {record}.",
    "who_screen": "WHO drinking-water guideline screen per parameter.",
    "wqi": "Water quality index and its components.",
    "iwqi": "Irrigation water quality after FAO 29.",
    "quality_issues": "Data-quality findings on the table.",
    "insights": "Quality score and next steps for the table.",
    "baseflow": "Total flow and separated baseflow.",
    "recharge_events": "Water-table fluctuation recharge estimate and its inputs.",
    "recession_segments": "Recession segments of the hydrograph.",
    "drawdown": "Theis drawdown and its inputs.",
}


# ── generic makers ────────────────────────────────────────────────────────


def _kv(pairs: list[tuple[str, Any]]) -> Table | None:
    rows = [[k, v] for k, v in pairs if v is not None and v != ""]
    return (["item", "value"], rows) if rows else None


def _scalars(payload: dict[str, Any], *, include: tuple[str, ...] = (), depth: int = 2,
             anchors: tuple[str, ...] = ()) -> Table | None:
    """A key/value table of the payload's scalars (bulk keys skipped), plus the nested blocks in ``include``.

    With ``anchors``, the table is made only when the payload carries at least one of those keys: a table named
    ``demand`` is not built from a payload that has no demand in it.
    """
    if anchors and not any(payload.get(k) is not None for k in anchors):
        return None
    pairs = flatten(payload, depth=depth)
    for key in include:
        block = payload.get(key)
        if isinstance(block, dict):
            pairs.extend(flatten(block, prefix=f"{key}.", depth=depth, skip=frozenset()))
    return _kv(pairs)


# ── the makers by table name ──────────────────────────────────────────────


def _series(payload: dict[str, Any]) -> Table | None:
    got = series_of(payload)
    if got:
        return ["datetime", "value"], [[t, v] for t, v in zip(*got)]
    if isinstance(payload.get("preview"), dict):
        return frame_records(payload["preview"])
    return None


def _summary(payload: dict[str, Any]) -> Table | None:
    params = payload.get("parameters")
    if isinstance(params, list) and params and all(isinstance(p, dict) for p in params):
        return records(params)
    pairs = [(k, payload.get(k)) for k in ("source", "station_id", "variable", "unit", "n", "start", "end", "years",
                                            "n_records", "n_stations", "n_parameters", "time_span_years",
                                            "completeness_pct")]
    pairs = [(k, v) for k, v in pairs if is_scalar(v) and v is not None]
    pairs.extend(flatten(payload.get("stats") or {}, prefix="stats.", depth=0, skip=frozenset()))
    if isinstance(payload.get("date_range"), list):
        pairs.append(("date_range", " to ".join(str(x) for x in payload["date_range"])))
    if isinstance(payload.get("sources"), list):
        pairs.append(("sources", "; ".join(str(s) for s in payload["sources"])))
    return _kv(pairs) or _scalars(payload, anchors=("n", "stats", "n_records", "variable", "years"))


def _annual_maxima(payload: dict[str, Any]) -> Table | None:
    am = annual_maxima_of(payload)
    return (["year", "value"], [[y, v] for y, v in zip(*am)]) if am else None


def _return_levels(payload: dict[str, Any]) -> Table | None:
    rl = return_levels_of(payload)
    if not rl:
        return None
    n = len(rl["T"])

    def col(name: str) -> list[Any]:
        vals = rl.get(name)
        return list(vals) + [None] * (n - len(vals)) if isinstance(vals, list) else [None] * n

    gev = col("gev") if rl["gev"] is not None else col("boot")
    rows = [[t, g, lp, lo, hi] for t, g, lp, lo, hi in zip(rl["T"], gev, col("lp3"), col("lower"), col("upper"))]
    return ["T", "GEV", "LP3", "lower", "upper"], rows


def _fit_spread(payload: dict[str, Any]) -> Table | None:
    rl = return_levels_of(payload)
    if not rl or rl["gev"] is None or rl["lp3"] is None:
        return None
    rows = []
    for t, g, lp in zip(rl["T"], rl["gev"], rl["lp3"]):
        if g is None or lp is None or (g + lp) == 0:
            continue
        rows.append([t, g, lp, round(abs(g - lp) / ((g + lp) / 2) * 100.0, 1)])
    return (["T", "GEV", "LP3", "spread_pct"], rows) if rows else None


def _fdc_percentiles(payload: dict[str, Any]) -> Table | None:
    fdc = fdc_of(payload)
    if not fdc or not fdc["percentiles"]:
        return None
    return ["exceedance_pct", "value"], [[k, v] for k, v in fdc["percentiles"].items()]


def _trend(payload: dict[str, Any]) -> Table | None:
    from aquascope.trend_series import reported_trend

    tr = reported_trend(payload) or payload.get("trend")  # the annual maxima for a flood question
    if not isinstance(tr, dict):
        tr = payload.get("temperature")
    return _kv(flatten(tr, depth=1, skip=frozenset())) if isinstance(tr, dict) else None


def _glofas_summary(payload: dict[str, Any]) -> Table | None:
    g = payload.get("glofas")
    if not isinstance(g, dict):
        return None
    pairs = flatten(g, depth=1)
    rl = return_levels_of(g)
    if rl and rl["gev"] is not None:
        pairs.extend((f"return_level_T{int(t)}_gev", q) for t, q in zip(rl["T"], rl["gev"]) if t is not None)
    fdc = fdc_of(g)
    if fdc:
        pairs.extend((f"q{int(k)}", v) for k, v in fdc["percentiles"].items())
    return _kv(pairs)


def _monthly_climate(payload: dict[str, Any]) -> Table | None:
    clim = payload.get("climate") if isinstance(payload.get("climate"), dict) else payload
    p = numbers(clim.get("monthly_precipitation_mm"))
    et0 = numbers(clim.get("monthly_et0_mm"))
    temp = numbers(clim.get("monthly_temperature_c"))
    if len(p) != 12 and len(et0) != 12:
        return None
    cols = ["month"] + (["precipitation_mm"] if len(p) == 12 else []) + (["et0_mm"] if len(et0) == 12 else []) + (
        ["temperature_c"] if len(temp) == 12 else [])
    rows = []
    for m in range(12):
        row: list[Any] = [m + 1]
        if len(p) == 12:
            row.append(p[m])
        if len(et0) == 12:
            row.append(et0[m])
        if len(temp) == 12:
            row.append(temp[m])
        rows.append(row)
    return cols, rows


def _indices_monthly(payload: dict[str, Any]) -> Table | None:
    panels = [p for p in indices_of(payload) if p.get("timescale") is not None]
    if not panels:
        return None
    cols = ["date"]
    per_date: dict[str, dict[str, Any]] = {}
    for p in panels:
        for name in ("spi", "spei"):
            if p.get(name) is None:
                continue
            col = f"{name}_{p['timescale']}"
            cols.append(col)
            for d, v in zip(p["dates"], p[name]):
                per_date.setdefault(d[:10], {})[col] = v
    rows = [[d] + [per_date[d].get(c) for c in cols[1:]] for d in sorted(per_date)]
    return cols, rows


def _drought_events(payload: dict[str, Any]) -> Table | None:
    rows_in = payload.get("indices")
    if isinstance(rows_in, list) and rows_in:
        cols = ["timescale", "index", "current", "class", "date", "worst", "worst_date", "events", "n"]
        rows = []
        for r in rows_in:
            if not isinstance(r, dict):
                continue
            for name in ("spi", "spei"):
                d = r.get(name)
                if isinstance(d, dict):
                    rows.append([r.get("timescale"), name.upper()] + [d.get(k) for k in cols[2:]])
        return (cols, rows) if rows else None
    return _events(payload)


def _events(payload: dict[str, Any]) -> Table | None:
    ev = payload.get("events")
    if isinstance(ev, list) and ev and all(isinstance(e, dict) for e in ev):
        return records(ev, ["start", "end", "duration", "severity", "peak"])
    sgi = payload.get("sgi")
    if isinstance(sgi, dict) and isinstance(sgi.get("last_event"), dict):
        return records([sgi["last_event"]], ["start", "end", "duration", "severity", "peak"])
    return None


def _index_divergence(payload: dict[str, Any]) -> Table | None:
    rows_in = payload.get("indices")
    if not isinstance(rows_in, list):
        return None
    cols = ["timescale", "current", "mean_last_10y", "months_spei_drier_pct", "correlation", "n"]
    rows = [[r.get("timescale")] + [r["divergence"].get(k) for k in cols[1:]]
            for r in rows_in if isinstance(r, dict) and isinstance(r.get("divergence"), dict)]
    return (cols, rows) if rows else None


def _sgi_monthly(payload: dict[str, Any]) -> Table | None:
    s = payload.get("series")
    if isinstance(s, dict) and isinstance(s.get("index"), list) and isinstance(s.get("sgi"), list):
        cols = ["date", "sgi"] + (["spi"] if isinstance(s.get("spi"), list) else [])
        spi = numbers(s.get("spi")) if "spi" in cols else None
        rows = [[str(d)[:10], v] + ([spi[i]] if spi is not None else [])
                for i, (d, v) in enumerate(zip(s["index"], numbers(s["sgi"])))]
        return cols, rows
    sgi = payload.get("sgi")
    if isinstance(sgi, dict) and isinstance(sgi.get("index"), list) and isinstance(sgi.get("values"), list):
        return ["date", "sgi"], [[str(d)[:10], v] for d, v in zip(sgi["index"], numbers(sgi["values"]))]
    return None


def _propagation_lag(payload: dict[str, Any]) -> Table | None:
    prop = payload.get("propagation")
    if not isinstance(prop, dict) or not isinstance(prop.get("by_timescale"), dict):
        return None
    best = prop.get("best") if isinstance(prop.get("best"), dict) else {}
    rows = []
    for scale, r in prop["by_timescale"].items():
        if not isinstance(r, dict):
            continue
        rows.append([num(scale), r.get("lag_months"), r.get("correlation"), r.get("n"), r.get("error"),
                     str(best.get("timescale")) == str(scale)])
    return (["timescale", "lag_months", "correlation", "n", "error", "best"], rows) if rows else None


def _donors(payload: dict[str, Any]) -> Table | None:
    donors = stations_of(payload)
    if not donors:
        return None
    cols = ["source", "station_id", "name", "latitude", "longitude", "distance_km", "score", "similarity_distance",
            "up_area_km2", "period_start", "period_end"]
    return records(donors, [c for c in cols if any(c in d for d in donors)])


def _signatures(payload: dict[str, Any]) -> Table | None:
    est = payload.get("estimates")
    if not isinstance(est, dict) or not est:
        sim = payload.get("similarity")
        est = sim.get("estimates") if isinstance(sim, dict) else None
    if isinstance(est, dict) and est and all(isinstance(v, dict) for v in est.values()):
        skill = ((payload.get("skill") or {}).get("by_signature") or {}) if isinstance(payload.get("skill"),
                                                                                       dict) else {}
        cols = ["signature", "label", "value", "low", "high", "unit", "n_donors", "nse"]
        rows = [[name, e.get("label"), e.get("value"), e.get("low"), e.get("high"), e.get("unit"), e.get("n_donors"),
                 (skill.get(name) or {}).get("nse") if isinstance(skill.get(name), dict) else None]
                for name, e in est.items()]
        return cols, rows
    sig = payload.get("signatures")
    if isinstance(sig, dict) and sig:
        return _kv([(k, v) for k, v in sig.items() if is_scalar(v)])
    return None


def _stations(payload: dict[str, Any]) -> Table | None:
    stations = stations_of(payload)
    if not stations:
        return None
    rows = [{k: v for k, v in s.items() if k not in ("lat", "lon", "label")} for s in stations]
    return records(rows)


def _sufficiency(payload: dict[str, Any]) -> Table | None:
    rows = payload.get("sufficiency")
    if not isinstance(rows, list) or not rows:
        recon = payload.get("recon")
        rows = recon.get("sufficiency") if isinstance(recon, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        st = r.get("station")
        row = dict(r)
        row["station"] = f"{st.get('source')} {st.get('station_id')}" if isinstance(st, dict) else None
        out.append(row)
    return records(out)


def _catchment_attributes(payload: dict[str, Any]) -> Table | None:
    attrs = payload.get("attributes")
    if not isinstance(attrs, dict) or not attrs:
        return None
    rows = []
    for key, v in attrs.items():
        if isinstance(v, dict):
            rows.append([key, v.get("label"), v.get("value"), v.get("unit"), v.get("source"), v.get("note")])
        elif is_scalar(v):
            rows.append([key, None, v, None, None, None])
    return (["attribute", "label", "value", "unit", "source", "note"], rows) if rows else None


def _low_flow_stats(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("fdc", "stats", "low_flow", "recent"), anchors=("low_flow", "bfi", "n_days"))


def _reliability(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("reliability", "fdc", "signatures_m3s", "low_flow"),
                    anchors=("reliability", "required_flow_m3s", "verdict"))


def _demand_monthly(payload: dict[str, Any]) -> Table | None:
    seasons = payload.get("per_season")
    if isinstance(seasons, list) and seasons:
        return records(seasons)
    sched = frame_records(payload.get("schedule"))
    if sched and "date" in sched[0]:
        cols, rows = sched
        ci = {c: i for i, c in enumerate(cols)}
        wanted = [c for c in ("eto", "etc", "effective_rain", "net_irrigation", "gross_irrigation") if c in ci]
        months: dict[str, list[float]] = {}
        for r in rows:
            m = months.setdefault(str(r[ci["date"]])[:7], [0.0] * len(wanted))
            for i, c in enumerate(wanted):
                v = num(r[ci[c]])
                if v is not None:
                    m[i] += v
        return ["month"] + [f"{c}_mm" for c in wanted], [[k] + [round(x, 2) for x in months[k]]
                                                          for k in sorted(months)]
    return None


def _et0_monthly(payload: dict[str, Any]) -> Table | None:
    eto = payload.get("eto")
    got = series_of({"series": eto}) if isinstance(eto, dict) else None
    if got:
        sums: dict[int, list[float]] = {m: [] for m in range(1, 13)}
        for d, v in zip(*got):
            if v is not None:
                sums[int(d[5:7])].append(v)
        return ["month", "et0_mm_per_day", "n_days"], [[m, round(sum(x) / len(x), 3) if x else None, len(x)]
                                                         for m, x in sums.items()]
    clim = payload.get("climate") if isinstance(payload.get("climate"), dict) else payload
    et0 = numbers(clim.get("monthly_et0_mm"))
    if len(et0) == 12:
        return ["month", "et0_mm"], [[m + 1, et0[m]] for m in range(12)]
    return None


def _demand(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("totals_mm", "demand", "season", "eto"), anchors=("totals_mm", "demand", "crop"))


def _schedule(payload: dict[str, Any]) -> Table | None:
    return frame_records(payload.get("schedule"))


def _et0(payload: dict[str, Any]) -> Table | None:
    eto = payload.get("eto")
    got = series_of({"series": eto}) if isinstance(eto, dict) else None
    return (["datetime", "et0_mm"], [[t, v] for t, v in zip(*got)]) if got else None


def _samples(payload: dict[str, Any]) -> Table | None:
    return records(payload.get("samples"), ["datetime", "parameter", "value", "unit"])


def _sample_counts(payload: dict[str, Any]) -> Table | None:
    per = payload.get("parameters")
    if isinstance(per, dict) and per and all(isinstance(v, dict) for v in per.values()):
        cols = ["parameter", "n", "unit", "start", "end", "min", "median", "max"]
        return cols, [[name] + [v.get(k) for k in cols[1:]] for name, v in per.items()]
    counts = payload.get("sample_counts")
    if isinstance(counts, dict) and counts:
        units = payload.get("units") if isinstance(payload.get("units"), dict) else {}
        return ["parameter", "n", "unit"], [[k, v, units.get(k)] for k, v in counts.items()]
    return None


def _who_screen(payload: dict[str, Any]) -> Table | None:
    return records(payload.get("rows"), ["parameter", "rule", "n", "n_exceed", "pct", "status"])


def _wqi(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("ccme", "nsf", "period"), anchors=("ccme", "nsf", "score"))


def _iwqi(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("sar", "components"), depth=2, anchors=("restriction", "sar", "components"))


def _quality_issues(payload: dict[str, Any]) -> Table | None:
    rows: list[list[Any]] = []
    for key in ("null_counts", "outlier_counts"):
        block = payload.get(key)
        if isinstance(block, dict):
            rows.extend([key, k, v] for k, v in block.items() if is_scalar(v))
    gaps = payload.get("temporal_gaps")
    if isinstance(gaps, list):
        rows.extend(["temporal_gap", str(g)[:120], None] for g in gaps[:50])
    elif isinstance(gaps, dict):
        rows.extend(["temporal_gap", k, v] for k, v in gaps.items() if is_scalar(v))
    for key in ("unit_issues", "recommended_steps"):
        block = payload.get(key)
        if isinstance(block, list):
            rows.extend([key, str(x), None] for x in block)
    for key in ("n_records", "n_duplicates", "completeness_pct"):
        if is_scalar(payload.get(key)) and payload.get(key) is not None:
            rows.append(["count", key, payload[key]])
    return (["kind", "item", "value"], rows) if rows else None


def _insights(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, depth=2, anchors=("score", "quality_score", "next_steps", "who", "summary"))


def _baseflow(payload: dict[str, Any]) -> Table | None:
    s = payload.get("series")
    if not isinstance(s, dict) or not isinstance(s.get("index"), list) or not isinstance(s.get("total"), list):
        return None
    base = numbers(s.get("baseflow") or [])
    total = numbers(s["total"])
    if len(base) != len(total):
        return None
    return ["datetime", "total", "baseflow"], [[str(d), t, b] for d, t, b in zip(s["index"], total, base)]


def _recharge_events(payload: dict[str, Any]) -> Table | None:
    ev = payload.get("events") or payload.get("rises")
    if isinstance(ev, list) and ev and all(isinstance(e, dict) for e in ev):
        return records(ev)
    return _scalars(payload, include=("metadata",), anchors=("value_mm_per_year", "uncertainty"))


def _recession_segments(payload: dict[str, Any]) -> Table | None:
    seg = records(payload.get("segments"))
    if seg:
        return seg
    return _scalars(payload, anchors=("recession_constant", "half_life_days"))


def _drawdown(payload: dict[str, Any]) -> Table | None:
    return _scalars(payload, include=("inputs",), anchors=("drawdown_m",))


MAKERS: dict[str, Callable[[dict[str, Any]], Table | None]] = {
    "series": _series,
    "summary": _summary,
    "annual_maxima": _annual_maxima,
    "return_levels": _return_levels,
    "fit_spread": _fit_spread,
    "fdc_percentiles": _fdc_percentiles,
    "trend": _trend,
    "glofas_summary": _glofas_summary,
    "monthly_climate": _monthly_climate,
    "indices_monthly": _indices_monthly,
    "drought_events": _drought_events,
    "index_divergence": _index_divergence,
    "sgi_monthly": _sgi_monthly,
    "propagation_lag": _propagation_lag,
    "events": _events,
    "donors": _donors,
    "signatures": _signatures,
    "stations": _stations,
    "stations_within_reach": _stations,
    "sufficiency": _sufficiency,
    "catchment_attributes": _catchment_attributes,
    "low_flow_stats": _low_flow_stats,
    "reliability": _reliability,
    "demand_monthly": _demand_monthly,
    "et0_monthly": _et0_monthly,
    "demand": _demand,
    "schedule": _schedule,
    "et0": _et0,
    "samples": _samples,
    "sample_counts": _sample_counts,
    "who_screen": _who_screen,
    "wqi": _wqi,
    "iwqi": _iwqi,
    "quality_issues": _quality_issues,
    "insights": _insights,
    "baseflow": _baseflow,
    "recharge_events": _recharge_events,
    "recession_segments": _recession_segments,
    "drawdown": _drawdown,
}


def names() -> list[str]:
    """Every table name a maker exists for."""
    return sorted(MAKERS)


def names_of(tool: str) -> list[str]:
    """The table names the catalogue lists for a tool."""
    from aquascope.studio import catalogue

    entry = catalogue.get(tool)
    return list(entry.tables) if entry else []


def make(name: str, payload: dict[str, Any]) -> Table | None:
    """``(columns, rows)`` for one table name, or None when the payload holds nothing for it."""
    maker = MAKERS.get(name)
    if maker is None or not isinstance(payload, dict):
        return None
    return maker(payload)


def csv_bytes(columns: list[str], rows: list[list[Any]]) -> bytes:
    """The table as UTF-8 CSV with a header row; None cells are empty."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue().encode("utf-8")


def tables_for(step_id: str, tool: str, payload: dict[str, Any], *, names: list[str] | None = None,
               site: dict[str, Any] | None = None) -> list[Artifact]:
    """One CSV artifact per table name the tool yields (or per ``names``) that the payload supports.

    Ids are ``tab-{step_id}-{name}``, names ``tables/{step_id}_{name}.csv``; ``meta`` carries the columns, the
    row count and the table name. A maker that finds nothing yields no artifact; one that trips on an
    unexpected shape is logged and skipped.
    """
    wanted = list(names) if names is not None else names_of(tool)
    out: list[Artifact] = []
    if not isinstance(payload, dict) or payload.get("error"):
        return out
    record = record_name(payload, site)
    for name in wanted:
        try:
            table = make(name, payload)
        except Exception as exc:  # noqa: BLE001 - a table that cannot be built is not a failed study
            logger.warning("table %s for step %s (%s) skipped: %s", name, step_id, tool, exc)
            table = None
        if not table or not table[1]:
            continue
        columns, rows = table
        caption = CAPTIONS.get(name, f"{name.replace('_', ' ')} for {{record}}.").format(record=record)
        out.append(Artifact(id=f"tab-{step_id}-{name}", kind="table", name=f"tables/{step_id}_{name}.csv",
                            data=csv_bytes(columns, rows), media_type=MEDIA_TYPES["csv"], caption=caption,
                            step=step_id, meta={"columns": list(columns), "rows": len(rows), "name": name,
                                                "tool": tool}))
    return out


def frame_of(artifact: Artifact) -> Any:
    """A table artifact's CSV as a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(io.BytesIO(artifact.data))
