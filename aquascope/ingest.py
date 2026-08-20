"""Ingest anything: turn an agency CSV / Excel export into a clean daily series with a QA report.

Every hydrologist has done this by hand a hundred times: guess the date
column, guess the value column, guess the unit, notice the gaps and the
sentinel values three figures later. ``ingest()`` does the guessing with
heuristics (always available) or with an LLM (when a key is configured, it
proposes the mapping and the heuristics validate it), applies the mapping
deterministically, and writes a QA report the file's owner can check.

Nothing here talks to an agency; nothing here needs a key.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquascope.schemas.station import VARIABLES

logger = logging.getLogger(__name__)

SENTINELS = (-999, -9999, -999.0, -9999.0, -99999, 9999, 99999, -999998, -999999)

# Column-name hints -> (variable, unit). Order matters (first match wins).
NAME_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"discharge|streamflow|\bflow\b|\bq\b|debit|débit|caudal|流量", re.I), "discharge", "m3/s"),
    (re.compile(r"stage|water[_ ]?level|gauge[_ ]?height|gage[_ ]?height|hauteur|nivel|水位|\bh\b", re.I),
     "water_level", "m"),
    (re.compile(r"precip|rain|pluie|lluvia|降雨|雨量", re.I), "precipitation", "mm"),
    (re.compile(r"ground ?water|piezo|well|nappe|地下水", re.I), "groundwater_level", "m"),
    (re.compile(r"\bet0\b|evapotrans|\beto\b", re.I), "evapotranspiration", "mm"),
    (re.compile(r"storage|reservoir|volume|蓄水", re.I), "reservoir_storage", "hm3"),
]
UNIT_HINTS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"cfs|ft3/s|ft³/s|cubic feet", re.I), "m3/s", 0.028316846592),
    (re.compile(r"m3/s|m³/s|cms|cumec", re.I), "m3/s", 1.0),
    (re.compile(r"l/s|litre|liter", re.I), "m3/s", 0.001),
    (re.compile(r"\bmm\b", re.I), "mm", 1.0),
    (re.compile(r"\bcm\b", re.I), "m", 0.01),
    (re.compile(r"\bft\b|feet", re.I), "m", 0.3048),
    (re.compile(r"\binch|\bin\b", re.I), "mm", 25.4),
]


@dataclass
class Mapping:
    datetime_column: str
    value_column: str
    variable: str = "discharge"
    unit: str = ""
    to_si_factor: float = 1.0
    station_id: str | None = None
    station_column: str | None = None
    date_format: str | None = None
    decimal: str = "."
    confidence: float = 0.5
    rationale: str = ""
    method: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAReport:
    n_rows_in: int
    n_values: int
    start: str | None
    end: str | None
    n_days_span: int
    n_days_with_data: int
    coverage_pct: float
    n_duplicates_dropped: int
    n_sentinels_dropped: int
    n_negative: int
    n_spikes_flagged: int
    gaps: list[dict[str, Any]] = field(default_factory=list)
    yearly_coverage: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── reading ────────────────────────────────────────────────────────────────


def read_table(path: str | Path, *, sheet: str | int | None = None) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX/JSON with a few agency-export quirks handled (comment lines, ; and tab delimiters)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(p, sheet_name=sheet or 0)
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return pd.json_normalize(data if isinstance(data, list) else data.get("data", data))
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    sample = "\n".join(lines[:50])
    delim = ","
    for cand in ("\t", ";", "|"):
        if sample.count(cand) > sample.count(delim):
            delim = cand
    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines)), sep=delim, engine="python", dtype=str,
                     skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ── mapping ────────────────────────────────────────────────────────────────


def _looks_like_datetime(series: pd.Series) -> float:
    s = series.dropna().astype(str).head(200)
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", format="mixed") if hasattr(pd, "to_datetime") else None
    return float(parsed.notna().mean()) if parsed is not None else 0.0


def _looks_numeric(series: pd.Series) -> float:
    s = series.dropna().astype(str).head(500).str.replace(",", ".", regex=False)
    if s.empty:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").notna().mean())


def guess_mapping(df: pd.DataFrame, *, variable: str | None = None) -> Mapping:
    """Heuristic mapping: the most datetime-like column, the most plausible numeric column, hints from names/units."""
    dt_scores = {c: _looks_like_datetime(df[c]) for c in df.columns}
    dt_col = max(dt_scores, key=dt_scores.get)
    if dt_scores[dt_col] < 0.6:
        # maybe separate year/month/day columns
        raise ValueError(
            "Could not find a date/time column (best guess "
            f"'{dt_col}' parses {dt_scores[dt_col]:.0%} of rows). Pass --date-column."
        )
    candidates = [c for c in df.columns if c != dt_col]
    numeric = {c: _looks_numeric(df[c]) for c in candidates}
    var_guess, unit_guess, factor = variable or "", "", 1.0
    best, best_score = None, -1.0
    for c in candidates:
        score = numeric[c]
        if score < 0.5:
            continue
        for pat, var, unit in NAME_HINTS:
            if pat.search(c):
                score += 1.0
                if not variable:
                    var_guess, unit_guess = var, unit
                break
        for pat, unit, f in UNIT_HINTS:
            if pat.search(c):
                unit_guess, factor = unit, f
                score += 0.2
                break
        if re.search(r"flag|qual|code|status|remark|comment|id\b", c, re.I):
            score -= 0.8
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise ValueError("Could not find a numeric value column. Pass --value-column.")
    if not var_guess:
        var_guess = "discharge"
    station_col = next(
        (c for c in candidates
         if c != best and re.search(r"station|site|gauge|gage|code|id$", c, re.I)
         and df[c].nunique(dropna=True) <= max(1, len(df) // 20)),  # an id repeats; a value column does not
        None,
    )
    return Mapping(
        datetime_column=dt_col, value_column=best, variable=var_guess, unit=unit_guess or {"discharge": "m3/s",
        "water_level": "m", "precipitation": "mm", "groundwater_level": "m"}.get(var_guess, ""),
        to_si_factor=factor, station_column=station_col, confidence=min(0.95, 0.4 + best_score / 3),
        rationale=f"date column '{dt_col}' ({dt_scores[dt_col]:.0%} parse), value column '{best}' by name/unit hints",
        method="heuristic",
    )


LLM_MAPPING_PROMPT = """You map a water-data table to a fixed schema. Reply with JSON only:
{"datetime_column": str, "value_column": str, "variable": one of %s, "unit": str,
 "to_si_factor": number (multiply values to get m3/s, m or mm), "station_column": str|null,
 "station_id": str|null, "date_format": str|null, "decimal": "." or ",", "rationale": str}
Columns and first rows follow. Choose the observed value column (not flags/quality codes).
"""


def llm_mapping(df: pd.DataFrame, *, client: Any, model: str, description: str = "") -> Mapping | None:
    """Ask an OpenAI-compatible model for the mapping; validated by the heuristics before use."""
    head = df.head(8).to_csv(index=False)
    prompt = LLM_MAPPING_PROMPT % list(VARIABLES) + f"\nUser note: {description or '(none)'}\n\n{head}"
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception as exc:  # noqa: BLE001 - the heuristic path takes over
        logger.warning("LLM mapping failed (%s); using heuristics", exc)
        return None
    if data.get("datetime_column") not in df.columns or data.get("value_column") not in df.columns:
        logger.warning("LLM mapping named columns that do not exist; using heuristics")
        return None
    if data.get("variable") not in VARIABLES:
        data["variable"] = "discharge"
    return Mapping(
        datetime_column=data["datetime_column"], value_column=data["value_column"], variable=data["variable"],
        unit=str(data.get("unit") or ""), to_si_factor=float(data.get("to_si_factor") or 1.0),
        station_column=data.get("station_column"), station_id=data.get("station_id"),
        date_format=data.get("date_format"), decimal=str(data.get("decimal") or "."),
        confidence=0.8, rationale=str(data.get("rationale") or ""), method=f"llm:{model}",
    )


# ── applying + QA ──────────────────────────────────────────────────────────


def apply_mapping(df: pd.DataFrame, mapping: Mapping, *, station: str | None = None) -> pd.Series:
    """Deterministically build a tz-naive, sorted, de-duplicated series in SI units from a mapping."""
    frame = df
    if mapping.station_column and station:
        frame = frame[frame[mapping.station_column].astype(str) == str(station)]
    raw_t = frame[mapping.datetime_column].astype(str).str.strip()
    t = pd.to_datetime(raw_t, format=mapping.date_format, errors="coerce", utc=False) if mapping.date_format else \
        pd.to_datetime(raw_t, errors="coerce", format="mixed")
    raw_v = frame[mapping.value_column].astype(str).str.strip()
    if mapping.decimal == ",":
        raw_v = raw_v.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    v_raw = pd.to_numeric(raw_v, errors="coerce")
    sentinel = v_raw.isin(SENTINELS)  # before scaling, so -9999 cfs is still -9999
    v = v_raw.mask(sentinel) * float(mapping.to_si_factor)
    s = pd.Series(v.to_numpy(), index=pd.DatetimeIndex(t))
    s = s[~s.index.isna()]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.sort_index()
    s.attrs["n_sentinels"] = int(sentinel.sum())
    return s


def qa_series(s: pd.Series, *, n_rows_in: int) -> tuple[pd.Series, QAReport]:
    """Drop sentinels/duplicates, flag spikes and negatives, measure coverage and gaps."""
    n0 = int(len(s))
    n_sent = int(s.attrs.get("n_sentinels", 0))
    dup = int(s.index.duplicated(keep="last").sum())
    s = s[~s.index.duplicated(keep="last")]
    sentinel_mask = s.isin(SENTINELS)  # a mapping applied without apply_mapping (factor 1) still gets caught
    n_sent += int(sentinel_mask.sum())
    s = s[~sentinel_mask].dropna()
    negative = int((s < 0).sum())
    spikes = 0
    if len(s) > 30:
        med = float(np.nanmedian(s.values))
        mad = float(np.nanmedian(np.abs(s.values - med))) or 1e-9
        spikes = int((np.abs(s.values - med) / (1.4826 * mad) > 12).sum())
    warnings: list[str] = []
    if negative and n_sent == 0:
        warnings.append(
            f"{negative} negative values: sentinel codes the table did not declare, or a level datum below zero"
        )
    if spikes:
        warnings.append(f"{spikes} values more than 12 robust sigmas from the median (unit mix-up or spikes?)")
    daily = s.resample("D").mean().dropna() if len(s) else s
    span = int((s.index.max() - s.index.min()).days) + 1 if len(s) else 0
    coverage = float(len(daily) / span * 100) if span else 0.0
    gaps: list[dict[str, Any]] = []
    if len(daily) > 1:
        idx = daily.index
        deltas = (idx[1:] - idx[:-1]).days
        for start, d in zip(idx[:-1][deltas > 30], deltas[deltas > 30]):
            gaps.append({"from": start.date().isoformat(), "days": int(d) - 1})
        gaps.sort(key=lambda g: -g["days"])
    yearly = {}
    if len(daily):
        for y, grp in daily.groupby(daily.index.year):
            days_in_year = 366 if pd.Timestamp(year=int(y), month=12, day=31).is_leap_year else 365
            yearly[str(int(y))] = round(len(grp) / days_in_year * 100, 1)
    if coverage < 80:
        warnings.append(f"coverage is {coverage:.0f}% of the span; check the gaps list before computing statistics")
    report = QAReport(
        n_rows_in=n_rows_in, n_values=int(len(s)), start=daily.index.min().date().isoformat() if len(daily) else None,
        end=daily.index.max().date().isoformat() if len(daily) else None, n_days_span=span,
        n_days_with_data=int(len(daily)), coverage_pct=round(coverage, 1), n_duplicates_dropped=dup,
        n_sentinels_dropped=n_sent, n_negative=negative, n_spikes_flagged=spikes, gaps=gaps[:20],
        yearly_coverage=yearly, warnings=warnings,
    )
    logger.info("QA: %d -> %d values, coverage %.0f%%, %d gaps > 30 d", n0, len(s), coverage, len(gaps))
    return s, report


def ingest(
    path: str | Path,
    *,
    variable: str | None = None,
    date_column: str | None = None,
    value_column: str | None = None,
    unit: str | None = None,
    station: str | None = None,
    sheet: str | int | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Read + map + QA one file. Returns ``{"mapping", "series", "qa", "analysis"}``.

    ``analysis`` is :func:`aquascope.explore.analyze_series` on the cleaned
    series (hydrograph, FDC, flood frequency when the record allows).
    """
    from aquascope.explore import analyze_series

    df = read_table(path, sheet=sheet)
    mapping: Mapping | None = None
    if llm_client is not None and not (date_column and value_column):
        mapping = llm_mapping(df, client=llm_client, model=llm_model or "gpt-4o-mini", description=description)
    if mapping is None:
        mapping = guess_mapping(df, variable=variable)
    if date_column:
        mapping.datetime_column = date_column
    if value_column:
        mapping.value_column = value_column
    if variable:
        mapping.variable = variable
    if unit:
        mapping.unit = unit
        for pat, si_unit, f in UNIT_HINTS:
            if pat.search(unit):
                mapping.unit, mapping.to_si_factor = si_unit, f
                break
    raw = apply_mapping(df, mapping, station=station)
    series, qa = qa_series(raw, n_rows_in=len(df))
    analysis = analyze_series(series, mapping.variable, mapping.unit) if len(series) else {}
    return {"mapping": mapping.to_dict(), "series": series, "qa": qa.to_dict(), "analysis": analysis}


def ingest_text(
    text: str | bytes,
    filename: str = "upload.csv",
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`ingest` for data that is already in memory (a browser upload, a paste).

    The Explorer has no filesystem to hand a path to, so the bytes are written
    to a temporary file and read back through the same reader, which keeps one
    code path for CSV, Excel and the rest.
    """
    import tempfile

    suffix = Path(filename).suffix or ".csv"
    data = text.encode("utf-8") if isinstance(text, str) else text
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        return ingest(tmp, **kwargs)
    finally:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()


def write_outputs(result: dict[str, Any], out_stem: str | Path) -> dict[str, str]:
    """Write ``<stem>.csv`` (date,value), ``<stem>.qa.json`` and ``<stem>.qa.md``; return the paths."""
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    s: pd.Series = result["series"]
    csv_path = stem.with_suffix(".csv")
    pd.DataFrame({"date": s.index.strftime("%Y-%m-%d %H:%M"), "value": s.values}).to_csv(csv_path, index=False)
    qa_json = stem.with_suffix(".qa.json")
    qa_json.write_text(json.dumps({"mapping": result["mapping"], "qa": result["qa"]}, indent=2, default=str))
    m, q = result["mapping"], result["qa"]
    md = [
        f"# Ingest report: {stem.name}", "",
        f"- Mapping ({m['method']}, confidence {m['confidence']:.0%}): date `{m['datetime_column']}`, value "
        f"`{m['value_column']}` → {m['variable']} in {m['unit']} (×{m['to_si_factor']}). {m['rationale']}",
        f"- Rows in: {q['n_rows_in']:,}; values kept: {q['n_values']:,}; span {q['start']} → {q['end']} "
        f"({q['n_days_span']:,} days, {q['coverage_pct']}% with data)",
        f"- Dropped: {q['n_duplicates_dropped']} duplicate timestamps, {q['n_sentinels_dropped']} sentinel values; "
        f"flagged: {q['n_negative']} negative, {q['n_spikes_flagged']} spikes",
    ]
    if q["gaps"]:
        md.append("- Largest gaps: " + ", ".join(f"{g['from']} ({g['days']} d)" for g in q["gaps"][:5]))
    for w in q["warnings"]:
        md.append(f"- Warning: {w}")
    a = result.get("analysis") or {}
    if a.get("ffa"):
        g = a["ffa"]["fits"].get("gev_lmoments", {}).get("q")
        if g:
            md.append(
                f"- Flood frequency (GEV L-moments, {a['ffa']['n_years']} annual maxima): Q100 = {g[-1]} {m['unit']}"
            )
    md_path = stem.with_suffix(".qa.md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "qa_json": str(qa_json), "qa_md": str(md_path)}
