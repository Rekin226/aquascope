"""Per-station flow signatures for every mirrored gauge: ``signatures.parquet``.

One row per station with archived daily discharge, a fixed set of numbers a
map can be filtered on ("gauges with 50+ years and a rising flood trend"):

    source, station_id, variable, latitude, longitude, unit
    record_start, record_end, record_years   first and last day, span in years
    n_days, data_years, completeness         valid days, n_days / 365.25, n_days / span
    q_mean, q5, q50, q95                     m3/s; q5 is exceeded 5 % of days (high flow), q95 95 % (low flow)
    bfi                                      baseflow index, Lyne-Hollick filter (alpha 0.925, 3 passes)
    n_amax_years                             calendar years with 80 %+ of days, the annual-maxima sample
    amax_mk_p, amax_sen_slope, amax_trend    Mann-Kendall p and Sen slope (m3/s per year) of the annual maxima;
                                             trend is rising, falling or none at the 5 % level
    q100                                     100-year flood, GEV fitted by L-moments to the annual maxima
    amax_doy_mean, amax_doy_r                circular mean day of year of the annual maxima and its concentration
    notes                                    why any value above is empty ("q100: needs 10 complete years, has 6")

This differs from ``basins/station_signatures.parquet`` (:mod:`aquascope.archive.regionalize`), which is the donor
table for regionalisation: that one needs a catchment area and 10 years, and is in mm/d. This table covers every
mirrored discharge station, in m3/s, and a short record gets a row with empty cells and a reason, never an error.

Everything here runs on numpy, scipy and pandas only; writing the parquet needs ``pyarrow`` (the ``archive``
extra). The filter half (:func:`parse_filter_question`, :func:`filter_signatures`, :func:`filter_gauges`) is what
the MCP server, the Analyst and the Explorer's filter bar share.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquascope import __version__

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
SIGNATURES_FILE = "signatures.parquet"

MIN_DAYS = 365           # below this, no flow statistics at all
YEAR_COVERAGE = 292      # days a calendar year needs (80 %) to give an annual maximum
MIN_TREND_YEARS = 8      # matches analyze_station's amax_trend
MIN_FFA_YEARS = 10       # matches explore.MIN_YEARS_FOR_FFA
MIN_SEASON_YEARS = 5
TREND_ALPHA = 0.05

COLUMNS: list[str] = [
    "source", "station_id", "variable", "latitude", "longitude", "unit",
    "record_start", "record_end", "record_years", "n_days", "data_years", "completeness",
    "q_mean", "q5", "q50", "q95", "bfi",
    "n_amax_years", "amax_mk_p", "amax_sen_slope", "amax_trend", "q100",
    "amax_doy_mean", "amax_doy_r", "notes",
]
_STRING_COLUMNS = {"source", "station_id", "variable", "unit", "record_start", "record_end", "amax_trend", "notes"}
_INT_COLUMNS = {"n_days", "n_amax_years"}

TRENDS = ("rising", "falling", "none")


def signatures_url(repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{SIGNATURES_FILE}"


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def empty_row(source: str, station_id: str, *, latitude: float | None = None, longitude: float | None = None,
              variable: str = "discharge", unit: str = "m3/s") -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row.update(source=source, station_id=station_id, variable=variable, unit=unit,
               latitude=_num(latitude), longitude=_num(longitude), n_days=0, n_amax_years=0)
    return row


def _annual_maxima(daily: pd.Series) -> pd.Series:
    """Maximum of each calendar year with at least :data:`YEAR_COVERAGE` valid days, indexed by the date it fell on."""
    by_year = daily.groupby(daily.index.year)
    out: dict[pd.Timestamp, float] = {}
    for _year, grp in by_year:
        if grp.count() < YEAR_COVERAGE:
            continue
        when = grp.idxmax()
        out[when] = float(grp.loc[when])
    return pd.Series(out, dtype=float).sort_index()


def _circular_doy(dates: pd.DatetimeIndex) -> tuple[float, float]:
    """Circular mean day of year and resultant length (0 = spread over the year, 1 = same day every year)."""
    theta = 2.0 * np.pi * (dates.dayofyear.to_numpy(dtype=float) - 0.5) / 365.25
    c, s = float(np.cos(theta).mean()), float(np.sin(theta).mean())
    r = math.hypot(c, s)
    ang = math.atan2(s, c) % (2.0 * np.pi)
    doy = ang * 365.25 / (2.0 * np.pi) + 0.5
    return doy, r


def station_signatures(
    q: pd.Series,
    *,
    source: str,
    station_id: str,
    latitude: float | None = None,
    longitude: float | None = None,
    variable: str = "discharge",
    unit: str = "m3/s",
) -> dict[str, Any]:
    """The signature row for one station from its daily discharge series (DatetimeIndex, m3/s).

    Never raises for a bad record: every value that cannot be computed is None and the reason is in ``notes``.
    """
    row = empty_row(source, station_id, latitude=latitude, longitude=longitude, variable=variable, unit=unit)
    notes: list[str] = []

    try:
        s = pd.Series(pd.to_numeric(q, errors="coerce"), index=pd.DatetimeIndex(q.index))
    except (TypeError, ValueError) as exc:
        row["notes"] = f"record: unreadable ({type(exc).__name__})"
        return row
    s = s[np.isfinite(s.to_numpy(dtype=float))]
    negative = int((s < 0).sum())
    s = s[s >= 0]
    if negative:
        notes.append(f"record: dropped {negative} negative values")
    if s.empty:
        notes.append("record: no valid values")
        row["notes"] = "; ".join(notes)
        return row
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    daily = s.resample("D").mean().dropna()

    start, end = daily.index.min(), daily.index.max()
    span_days = int((end - start).days) + 1
    n = int(len(daily))
    row.update(
        record_start=start.date().isoformat(), record_end=end.date().isoformat(),
        record_years=round(span_days / 365.25, 2), n_days=n, data_years=round(n / 365.25, 2),
        completeness=round(n / span_days, 4) if span_days else None,
    )

    if n < MIN_DAYS:
        notes.append(f"flows: need {MIN_DAYS} days of data, have {n}")
    else:
        values = daily.to_numpy(dtype=float)
        row["q_mean"] = _num(values.mean())
        # exceedance convention: q5 is exceeded on 5 % of days, so it is the 95th percentile
        row["q5"] = _num(np.percentile(values, 95))
        row["q50"] = _num(np.percentile(values, 50))
        row["q95"] = _num(np.percentile(values, 5))
        if values.sum() <= 0:
            notes.append("bfi: no flow in the record")
        else:
            try:
                from aquascope.hydrology.baseflow import lyne_hollick

                row["bfi"] = _num(round(lyne_hollick(daily).bfi, 4))
            except Exception as exc:  # noqa: BLE001 - one odd record must not sink the table
                notes.append(f"bfi: failed ({type(exc).__name__})")

    am = _annual_maxima(daily)
    n_am = int(len(am))
    row["n_amax_years"] = n_am

    if n_am < MIN_TREND_YEARS:
        notes.append(f"flood trend: needs {MIN_TREND_YEARS} complete years, has {n_am}")
    else:
        try:
            from aquascope.analysis.trends import mann_kendall

            mk = mann_kendall(am.to_numpy(), alpha=TREND_ALPHA)
            # Sen's slope per calendar year, not per sample: the complete years need not be consecutive
            years = am.index.year.to_numpy(dtype=float)
            vals = am.to_numpy()
            i, j = np.triu_indices(len(vals), k=1)
            slope = float(np.median((vals[j] - vals[i]) / (years[j] - years[i])))
            row["amax_mk_p"] = _num(round(float(mk.p_value), 5))
            row["amax_sen_slope"] = _num(slope)
            row["amax_trend"] = {"increasing": "rising", "decreasing": "falling"}.get(str(mk.trend), "none")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"flood trend: failed ({type(exc).__name__})")

    if n_am < MIN_FFA_YEARS:
        notes.append(f"q100: needs {MIN_FFA_YEARS} complete years, has {n_am}")
    elif float(am.max()) <= 0:
        notes.append("q100: annual maxima are all zero")
    else:
        try:
            from aquascope.hydrology.flood_frequency import fit_gev_lmoments

            q100 = _num(fit_gev_lmoments(am.to_numpy(), return_periods=[100]).return_periods[100])
            if q100 is None or q100 <= 0:
                notes.append("q100: the GEV fit gave no usable value")
            else:
                row["q100"] = q100
        except Exception as exc:  # noqa: BLE001
            notes.append(f"q100: failed ({type(exc).__name__})")

    if n_am < MIN_SEASON_YEARS:
        notes.append(f"seasonality: needs {MIN_SEASON_YEARS} complete years, has {n_am}")
    else:
        doy, r = _circular_doy(pd.DatetimeIndex(am.index))
        row["amax_doy_mean"] = round(doy, 1)
        row["amax_doy_r"] = round(r, 3)

    row["notes"] = "; ".join(notes) or None
    return row


# ── the table ───────────────────────────────────────────────────────────────


def _coords(out: Path, stations: Any = None) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """(source, station_id) -> (lat, lon), from ``stations`` (Station objects or dicts) or ``out/stations.parquet``."""
    rows: list[Any] = []
    if stations is not None:
        rows = list(stations)
    else:
        path = out / "stations.parquet"
        if path.exists():
            try:
                rows = pd.read_parquet(path, columns=["source", "station_id", "latitude", "longitude"]).to_dict(
                    "records")
            except Exception as exc:  # noqa: BLE001 - coordinates are a nicety; the numbers still stand
                logger.warning("signatures: could not read %s (%s); coordinates left empty", path, exc)
    out_map: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for r in rows:
        get = r.get if isinstance(r, dict) else (lambda k, _r=r: getattr(_r, k, None))
        out_map[(str(get("source")), str(get("station_id")))] = (_num(get("latitude")), _num(get("longitude")))
    return out_map


def _station_files(out: Path, variable: str) -> list[tuple[str, str, Path]]:
    """(source, station_id, file) for every mirrored station of ``variable``: the manifest first, else the folder."""
    from aquascope.archive.observations import load_manifest

    found: dict[tuple[str, str], Path] = {}
    manifest = load_manifest(out)
    for entry in manifest.get("sources", {}).values():
        if entry.get("variable") != variable:
            continue
        src = entry.get("source")
        for sid, meta in (entry.get("stations") or {}).items():
            f = meta.get("file")
            if src and f and meta.get("n") and (out / f).exists():
                found[(src, sid)] = out / f
    root = out / "obs" / variable
    if root.exists():
        known = {p.resolve() for p in found.values()}
        for p in sorted(root.glob("*/*.csv.gz")):
            if p.resolve() in known:
                continue
            found.setdefault((p.parent.name, p.name[: -len(".csv.gz")]), p)
    return [(src, sid, p) for (src, sid), p in sorted(found.items())]


def build_signatures(
    out_dir: str | Path,
    *,
    stations: Any = None,
    variable: str = "discharge",
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Signatures for every mirrored station under ``out_dir/obs/<variable>/``, one row each, in :data:`COLUMNS`."""
    from aquascope.archive.observations import ARCHIVE_UNITS, read_csv_gz

    out = Path(out_dir)
    coords = _coords(out, stations)
    unit = ARCHIVE_UNITS.get(variable, "")
    rows: list[dict[str, Any]] = []
    for src, sid, path in _station_files(out, variable):
        if sources and src not in sources:
            continue
        lat, lon = coords.get((src, sid), (None, None))
        try:
            series = read_csv_gz(path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not sink the table
            row = empty_row(src, sid, latitude=lat, longitude=lon, variable=variable, unit=unit)
            row["notes"] = f"record: unreadable file ({type(exc).__name__})"
            rows.append(row)
            continue
        rows.append(station_signatures(series, source=src, station_id=sid, latitude=lat, longitude=lon,
                                       variable=variable, unit=unit))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in COLUMNS:
        if col in _STRING_COLUMNS:
            df[col] = df[col].astype(object)
        elif col in _INT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    logger.info("signatures: %d stations (%s)", len(df), variable)
    return df


def write_signatures_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the table as Parquet (zstd) with an ``aquascope`` metadata entry naming the version and columns."""
    from aquascope.utils.imports import require

    pa = require("pyarrow", feature="archive parquet output", group="archive")
    pq = require("pyarrow.parquet", feature="archive parquet output", group="archive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df[COLUMNS], preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"aquascope"] = json.dumps({"version": __version__, "kind": "signatures", "columns": COLUMNS}).encode()
    pq.write_table(table.replace_schema_metadata(meta), str(path), compression="zstd")
    return path


def build_signatures_file(
    out_dir: str | Path,
    *,
    stations: Any = None,
    path: str | Path | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Build ``signatures.parquet`` next to ``stations.parquet`` from the mirrored discharge files.

    Returns a summary dict (``file``, ``n_stations``, ``n_with_q100``, ``n_with_trend``); with no mirrored
    discharge it writes nothing and says so.
    """
    out = Path(out_dir)
    df = build_signatures(out, stations=stations, sources=sources)
    if df.empty:
        return {"file": None, "n_stations": 0, "note": f"no mirrored discharge under {out / 'obs' / 'discharge'}"}
    target = Path(path) if path else out / SIGNATURES_FILE
    write_signatures_parquet(df, target)
    return {
        "file": str(target),
        "n_stations": int(len(df)),
        "n_with_flows": int(df["q_mean"].notna().sum()),
        "n_with_trend": int(df["amax_trend"].notna().sum()),
        "n_with_q100": int(df["q100"].notna().sum()),
        "n_rising": int((df["amax_trend"] == "rising").sum()),
        "n_falling": int((df["amax_trend"] == "falling").sum()),
    }


def load_signatures(*, repo_id: str = DEFAULT_REPO_ID, refresh: bool = False,
                    path: str | Path | None = None) -> pd.DataFrame:
    """The published ``signatures.parquet`` (cached a day), or a local one with ``path``."""
    if path is None:
        from aquascope.archive.catalog import _download, cache_dir

        path = _download(signatures_url(repo_id), cache_dir() / f"{repo_id.replace('/', '__')}__{SIGNATURES_FILE}",
                         refresh)
    return pd.read_parquet(path)


# ── the filter: shared by the MCP tool, the Analyst and the Explorer's filter bar ──

FILTER_FIELDS = ("min_years", "flood_trend", "bfi_min", "bfi_max")

_WORD_NUMBERS = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                 "eighty": 80, "ninety": 90, "hundred": 100}


def normalize_filter(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the known fields with valid values; drop the rest. ``flood_trend`` "any" means no condition."""
    spec = spec or {}
    out: dict[str, Any] = {}
    y = _num(spec.get("min_years"))
    if y is not None and y > 0:
        out["min_years"] = round(y, 1)
    t = str(spec.get("flood_trend") or "").strip().lower()
    t = {"increasing": "rising", "up": "rising", "decreasing": "falling", "down": "falling",
         "no trend": "none", "stationary": "none"}.get(t, t)
    if t in TRENDS:
        out["flood_trend"] = t
    lo, hi = _num(spec.get("bfi_min")), _num(spec.get("bfi_max"))
    if lo is not None and 0 < lo <= 1:
        out["bfi_min"] = round(lo, 3)
    if hi is not None and 0 <= hi < 1:
        out["bfi_max"] = round(hi, 3)
    if "bfi_min" in out and "bfi_max" in out and out["bfi_min"] > out["bfi_max"]:
        out["bfi_min"], out["bfi_max"] = out["bfi_max"], out["bfi_min"]
    return out


def parse_filter_question(text: str) -> dict[str, Any]:
    """Read a filter out of plain words, with rules, no model: "50+ years, rising floods, baseflow index over 0.6".

    Recognised: a record length ("at least 40 years", "30+ years", "a 50-year record", "longer than fifty
    years"); a flood trend ("rising", "increasing", "falling", "declining", "no trend"); a baseflow index
    ("BFI above 0.6", "baseflow index between 0.3 and 0.5", "groundwater-fed" = 0.6 and up, "flashy" = 0.3 and
    below). Returns the normalised filter dict; empty when nothing was recognised.
    """
    t = " " + str(text or "").lower().replace("–", "-") + " "
    for word, n in _WORD_NUMBERS.items():
        t = re.sub(rf"\b{word}\b", str(n), t)
    spec: dict[str, Any] = {}

    at_least = r"(?:at least|more than|over|longer than|>=?|minimum of|min(?:imum)?)"
    m = (re.search(r"(\d{1,3})\s*\+\s*(?:years?|yrs?)", t)
         or re.search(rf"{at_least}\s*(\d{{1,3}})\s*(?:years?|yrs?)", t)
         or re.search(r"(\d{1,3})\s*(?:or more|and more|and up)\s*(?:years?|yrs?)", t)
         or re.search(r"(\d{1,3})\s*(?:-|\s)?\s*(?:years?|yr)\s*(?:long|of (?:data|record)|record)", t))
    if m:
        spec["min_years"] = int(m.group(1))

    trendish = re.search(r"flood|peak|maxim|annual max|trend|floods", t)
    if re.search(r"\bno (?:flood )?trend\b|\bwithout (?:a )?trend\b|\bstationary\b|\btrend-free\b", t):
        spec["flood_trend"] = "none"
    elif trendish and re.search(r"\b(?:rising|increasing|upward|growing|getting (?:bigger|worse)|more severe)\b", t):
        spec["flood_trend"] = "rising"
    elif trendish and re.search(r"\b(?:falling|decreasing|declining|downward|shrinking|getting smaller)\b", t):
        spec["flood_trend"] = "falling"

    bfi_word = r"(?:bfi|baseflow index|base flow index|baseflow)"
    num = r"(0?\.\d+|1(?:\.0+)?)"
    m = re.search(rf"{bfi_word}\s*(?:of\s*)?(?:between|from)\s*{num}\s*(?:and|to|-)\s*{num}", t)
    if m:
        spec["bfi_min"], spec["bfi_max"] = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(rf"{bfi_word}\s*(?:is\s*|of\s*)?(?:above|over|greater than|more than|at least|>=?)\s*{num}", t)
        if m:
            spec["bfi_min"] = float(m.group(1))
        m = re.search(rf"{bfi_word}\s*(?:is\s*|of\s*)?(?:below|under|less than|at most|<=?)\s*{num}", t)
        if m:
            spec["bfi_max"] = float(m.group(1))
    if "bfi_min" not in spec and "bfi_max" not in spec:
        if re.search(r"groundwater[- ]fed|baseflow[- ]dominated|high baseflow|spring[- ]fed", t):
            spec["bfi_min"] = 0.6
        elif re.search(r"\bflashy\b|low baseflow|quick ?flow[- ]dominated", t):
            spec["bfi_max"] = 0.3
    return normalize_filter(spec)


def describe_filter(spec: dict[str, Any]) -> str:
    """The filter in a few plain words, the same phrasing the filter bar uses."""
    spec = normalize_filter(spec)
    parts = []
    if "min_years" in spec:
        parts.append(f"{spec['min_years']:g}+ years of data")
    if "flood_trend" in spec:
        parts.append({"rising": "rising flood trend", "falling": "falling flood trend",
                      "none": "no flood trend"}[spec["flood_trend"]])
    if "bfi_min" in spec and "bfi_max" in spec:
        parts.append(f"BFI {spec['bfi_min']:g} to {spec['bfi_max']:g}")
    elif "bfi_min" in spec:
        parts.append(f"BFI {spec['bfi_min']:g} and up")
    elif "bfi_max" in spec:
        parts.append(f"BFI up to {spec['bfi_max']:g}")
    return ", ".join(parts) or "no filter"


def filter_signatures(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Rows of the signatures table that meet ``spec``. A missing value never matches a condition on it."""
    spec = normalize_filter(spec)
    mask = pd.Series(True, index=df.index)
    if "min_years" in spec:
        mask &= pd.to_numeric(df["data_years"], errors="coerce") >= spec["min_years"]
    if "flood_trend" in spec:
        mask &= df["amax_trend"] == spec["flood_trend"]
    if "bfi_min" in spec:
        mask &= pd.to_numeric(df["bfi"], errors="coerce") >= spec["bfi_min"]
    if "bfi_max" in spec:
        mask &= pd.to_numeric(df["bfi"], errors="coerce") <= spec["bfi_max"]
    return df[mask.fillna(False)]


def filter_gauges(
    question: str | None = None,
    min_years: float | None = None,
    flood_trend: str | None = None,
    bfi_min: float | None = None,
    bfi_max: float | None = None,
    limit: int = 25,
    spec_only: bool = False,
) -> dict[str, Any]:
    """Which mirrored gauges meet a condition on their flow signatures, and the filter for the map.

    Give the condition as words (question: "50+ years with a rising flood trend") or as fields: min_years (years
    of daily data), flood_trend (rising | falling | none: Mann-Kendall on the annual maxima at the 5 % level),
    bfi_min / bfi_max (baseflow index, 0 to 1). Fields override what the words say. Returns the normalised filter
    (the Explorer applies it to the map), how many gauges match, and up to ``limit`` of them, longest records
    first, each with its signatures. spec_only returns the filter without reading the table.
    """
    spec = parse_filter_question(question) if question else {}
    spec.update(normalize_filter({"min_years": min_years, "flood_trend": flood_trend,
                                  "bfi_min": bfi_min, "bfi_max": bfi_max}))
    spec = normalize_filter(spec)
    out: dict[str, Any] = {"filter": spec, "description": describe_filter(spec)}
    if question and not spec:
        out["note"] = ("No filter recognised in the question. Try a record length (40+ years), a flood trend "
                       "(rising, falling, no trend) or a baseflow index range (BFI above 0.6).")
    if spec_only:
        return out
    try:
        df = load_signatures()
    except Exception as exc:  # noqa: BLE001 - the table is optional; the filter still stands
        out["error"] = (f"the signatures table could not be read here ({type(exc).__name__}: {str(exc)[:120]}); "
                        "the filter itself is valid, and the Explorer applies it to its map")
        return out
    hits = filter_signatures(df, spec).sort_values("data_years", ascending=False)
    limit = max(1, min(int(limit or 25), 200))
    keep = ["source", "station_id", "latitude", "longitude", "record_start", "record_end", "data_years",
            "completeness", "q_mean", "q100", "bfi", "amax_trend", "amax_mk_p", "amax_sen_slope", "amax_doy_mean"]
    stations = [{k: (_num(v) if isinstance(v, float) else v) for k, v in r.items()}
                for r in hits[keep].head(limit).to_dict("records")]
    out.update({"n_total": int(len(df)), "n_match": int(len(hits)), "n_returned": len(stations),
                "stations": stations,
                "method": "Signatures computed weekly from the Archive's mirrored daily discharge "
                          "(aquascope.archive.signatures): Mann-Kendall and Sen slope on annual maxima of years "
                          "with 80 %+ of days, GEV by L-moments for Q100, Lyne-Hollick baseflow index."})
    return out


__all__ = [
    "COLUMNS", "FILTER_FIELDS", "SIGNATURES_FILE", "build_signatures", "build_signatures_file", "describe_filter",
    "filter_gauges", "filter_signatures", "load_signatures", "normalize_filter", "parse_filter_question",
    "signatures_url", "station_signatures", "write_signatures_parquet",
]
