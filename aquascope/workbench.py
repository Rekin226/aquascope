"""Analyses of *your own* data, as plain functions the whole project can share.

The dashboard grew ten Streamlit pages that each wired a few widgets to an
aquascope function and drew the result. That logic was reachable from exactly
one place: a Streamlit server. This module lifts it out, so the same code runs

* in the browser (the Explorer's Pyodide worker),
* in the MCP server and the Analyst's tool loop,
* in the CLI and in a notebook,
* and, still, behind the Streamlit pages, which now call in here.

Every function takes a DataFrame (or a Series, or plain numbers) plus keyword
parameters and returns a **JSON-serialisable dict**: no numpy arrays, no
Timestamps, no NaN or infinity, so a result can cross the worker boundary or
become a tool result without a second thought. Nothing here touches the
network, imports Streamlit, or plots: fetching is the collectors' job, drawing
is the page's.

The three column-picking rules the dashboard had grown (one per page) are one
resolver here, :func:`pick_column`, with the same defaults.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TOOLS",
    "DataProfile",
    "aquifer_drawdown",
    "baseflow",
    "datetime_indexed",
    "eda",
    "flood_frequency",
    "flow_duration",
    "insights",
    "irrigation",
    "jsonable",
    "pick_column",
    "profile",
    "quality",
    "recession",
    "recharge",
    "reference_et",
    "return_periods",
    "run",
    "sgi_drought",
    "signatures",
    "who_screen",
]

# ── making results safe to serialise ────────────────────────────────────────


def jsonable(obj: Any) -> Any:
    """Convert pandas/numpy results into something ``json.dumps`` accepts.

    NaN and infinity become ``None``: they are not JSON, and in these results
    they always mean "no value" (a month with too few observations, a Q5/Q95
    ratio with a zero denominator).
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {"index": [jsonable(i) for i in obj.index], "values": [jsonable(v) for v in obj.to_numpy()]}
    if isinstance(obj, pd.DataFrame):
        return {"columns": [str(c) for c in obj.columns],
            "rows": [[jsonable(v) for v in row] for row in obj.to_numpy()]}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return jsonable(asdict(obj))
    return obj


def _series_payload(s: pd.Series, *, max_points: int = 4000) -> dict[str, Any]:
    """A series as parallel index/value lists, thinned so a long record stays sendable."""
    s = s.dropna()
    step = max(1, len(s) // max_points)
    s = s.iloc[::step]
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        index = [t.isoformat() for t in idx]
    else:
        index = [jsonable(i) for i in idx]
    return {"index": index, "values": [jsonable(v) for v in s.to_numpy()], "n": int(len(s)), "step": step}


# ── the shape of a table ────────────────────────────────────────────────────

_DATETIME_CANDIDATES = (
    "sample_datetime", "reading_datetime", "observation_datetime", "forecast_datetime",
    "date", "datetime", "timestamp", "time",
)
_STATION_CANDIDATES = ("station_name", "station_id", "site_id", "site_name", "station")
_PARAM_CANDIDATES = ("parameter", "variable", "characteristic_name")
_VALUE_CANDIDATES = ("value", "result_value", "reading_value")
_DISCHARGE_HINTS = ("discharge", "flow", "streamflow", "q_cms")
_LEVEL_HINTS = ("level", "gwl", "water_level", "head", "wtr_level")
_NON_VALUE = ("latitude", "longitude", "lat", "lon", "elevation")


@dataclass
class DataProfile:
    """What a table appears to contain, worked out from its column names."""

    n_records: int = 0
    datetime_col: str | None = None
    station_col: str | None = None
    param_col: str | None = None
    value_col: str | None = None
    discharge_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    numeric_cols: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    n_stations: int = 0
    date_min: str | None = None
    date_max: str | None = None
    span_years: float = 0.0
    completeness_pct: float = 100.0

    @property
    def has_time(self) -> bool:
        return self.datetime_col is not None

    @property
    def has_geo(self) -> bool:
        return bool(self.lat_col and self.lon_col)

    @property
    def has_params(self) -> bool:
        return bool(self.param_col and self.value_col)


def _first_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def profile(df: pd.DataFrame) -> DataProfile:
    """Guess which column is time, station, parameter, value, discharge, lat and lon."""
    prof = DataProfile(n_records=int(len(df)))
    if df is None or df.empty:
        return prof
    cols = list(df.columns)
    prof.numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]

    prof.datetime_col = _first_match(cols, _DATETIME_CANDIDATES)
    if prof.datetime_col is None:
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                prof.datetime_col = str(c)
                break
    prof.station_col = _first_match(cols, _STATION_CANDIDATES)
    prof.param_col = _first_match(cols, _PARAM_CANDIDATES)
    prof.value_col = _first_match(cols, _VALUE_CANDIDATES)
    if prof.value_col is None and prof.numeric_cols:
        prof.value_col = next(
            (c for c in prof.numeric_cols if str(c).lower() not in _NON_VALUE), prof.numeric_cols[0]
        )
    prof.discharge_col = next(
        (c for c in prof.numeric_cols if any(h in str(c).lower() for h in _DISCHARGE_HINTS)), None
    )
    prof.lat_col = _first_match(cols, ("latitude", "lat"))
    prof.lon_col = _first_match(cols, ("longitude", "lon", "lng"))

    if prof.param_col:
        try:
            prof.parameters = sorted(df[prof.param_col].dropna().astype(str).unique())[:50]
        except Exception:  # noqa: BLE001 - a column of unhashable cells
            prof.parameters = []
    if prof.station_col:
        try:
            prof.n_stations = int(df[prof.station_col].nunique())
        except Exception:  # noqa: BLE001
            prof.n_stations = 0
    if prof.datetime_col:
        try:
            dt = pd.to_datetime(df[prof.datetime_col], errors="coerce", utc=True).dropna()
            if not dt.empty:
                prof.date_min = dt.min().isoformat()
                prof.date_max = dt.max().isoformat()
                prof.span_years = float((dt.max() - dt.min()).days / 365.25)
        except Exception:  # noqa: BLE001
            pass
    cells = len(df) * max(len(df.columns), 1)
    if cells:
        prof.completeness_pct = float(100.0 * (1 - df.isna().sum().sum() / cells))
    return prof


def datetime_indexed(df: pd.DataFrame, column: str, prof: DataProfile | None = None) -> pd.Series:
    """``df[column]`` with a DatetimeIndex when the table has a usable date column."""
    series = df[column].dropna()
    prof = prof or profile(df)
    dt_col = prof.datetime_col or _first_match(list(df.columns), _DATETIME_CANDIDATES)
    if not dt_col:
        return series
    try:
        idx = pd.to_datetime(df.loc[series.index, dt_col], errors="coerce")
        series = series[idx.notna().to_numpy()]
        series.index = pd.DatetimeIndex(idx.dropna())
    except Exception:  # noqa: BLE001 - unparseable dates: keep the original index
        return df[column].dropna()
    return series


def pick_column(df: pd.DataFrame, column: str | None = None, *, prefer: str = "value",
                prof: DataProfile | None = None) -> str:
    """Which numeric column an analysis should run on.

    ``prefer`` reproduces the three rules the dashboard had grown:
    ``discharge`` (the Hydrology Lab), ``level`` (the Groundwater page) and
    ``value`` (everything else).
    """
    prof = prof or profile(df)
    if column:
        if column not in df.columns:
            raise ValueError(f"No column {column!r}; columns are {list(df.columns)}")
        return column
    numeric = prof.numeric_cols
    if not numeric:
        raise ValueError("This table has no numeric column to analyse.")
    if prefer == "discharge":
        return prof.discharge_col if prof.discharge_col in numeric else numeric[0]
    if prefer == "level":
        hinted = next((c for c in numeric if any(h in str(c).lower() for h in _LEVEL_HINTS)), None)
        if hinted:
            return hinted
        return prof.discharge_col if prof.discharge_col in numeric else numeric[0]
    return prof.value_col if prof.value_col in numeric else numeric[0]


# ── clean and describe ──────────────────────────────────────────────────────


def eda(df: pd.DataFrame) -> dict[str, Any]:
    """Exploratory summary: record counts, per-parameter statistics, correlations."""
    from aquascope.analysis.eda import generate_eda_report

    rep = generate_eda_report(df)
    return {
        "n_records": int(rep.n_records),
        "n_stations": int(rep.n_stations),
        "n_parameters": int(rep.n_parameters),
        "date_range": list(rep.date_range) if rep.date_range else None,
        "time_span_years": jsonable(rep.time_span_years),
        "completeness_pct": jsonable(rep.completeness_pct),
        "sources": list(rep.sources),
        "parameters": [jsonable(asdict(p)) for p in rep.parameters],
        "correlations": jsonable(rep.correlation_matrix) if rep.correlation_matrix is not None else None,
        "methods": [{
            "name": "Exploratory data analysis",
            "text": "Record, station and parameter counts; per-parameter mean, standard deviation, quartiles and "
                    "outliers by the 1.5 x IQR rule; completeness as the share of non-missing cells.",
            "citation": "Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.",
        }],
    }


def quality(df: pd.DataFrame) -> dict[str, Any]:
    """Duplicates, gaps, outliers, unit problems, and what to do about them."""
    from aquascope.analysis.quality import assess_quality

    rep = assess_quality(df)
    return {
        "n_records": int(rep.n_records),
        "n_duplicates": int(rep.n_duplicates),
        "completeness_pct": jsonable(rep.completeness_pct),
        "null_counts": jsonable(rep.null_counts),
        "outlier_counts": jsonable(rep.outlier_counts),
        "temporal_gaps": jsonable(rep.temporal_gaps),
        "unit_issues": list(rep.unit_issues),
        "recommended_steps": list(rep.recommended_steps),
    }


PREPROCESS_STEPS = ("remove_duplicates", "fill_missing", "remove_outliers", "normalize", "resample_daily")


def preprocess(df: pd.DataFrame, steps: list[str] | None = None, *, preview_rows: int = 20) -> dict[str, Any]:
    """Apply cleaning steps in order and report what changed (the frame comes back too)."""
    from aquascope.analysis.quality import preprocess as _preprocess

    chosen = list(steps) if steps else ["remove_duplicates", "fill_missing", "remove_outliers"]
    unknown = [s for s in chosen if s not in PREPROCESS_STEPS]
    cleaned = _preprocess(df, steps=chosen)
    return {
        "steps": chosen,
        "unknown_steps": unknown,
        "n_before": int(len(df)),
        "n_after": int(len(cleaned)),
        "columns": [str(c) for c in cleaned.columns],
        "preview": jsonable(cleaned.head(preview_rows)),
        "frame": cleaned,          # not JSON: for callers that keep working with it
    }


def insights(df: pd.DataFrame) -> dict[str, Any]:
    """A quality score, a WHO quick screen and what to look at next."""
    prof = profile(df)
    score = 100.0
    notes: list[str] = []
    missing = 100.0 - prof.completeness_pct
    if missing > 0:
        score -= min(40.0, missing)
        notes.append(f"{missing:.1f}% missing values")
    try:
        n_dup = int(df.duplicated().sum())
    except Exception:  # noqa: BLE001 - unhashable cells
        n_dup = 0
    if n_dup:
        score -= min(20.0, 100.0 * n_dup / max(len(df), 1))
        notes.append(f"{n_dup} duplicate rows")
    if prof.has_time and prof.span_years < 1 / 12:
        score -= 10
        notes.append("very short time span")
    if not prof.has_time:
        score -= 10
        notes.append("no datetime column detected")

    screen = who_screen(df)
    alerts = sum(1 for r in screen["rows"] if r["status"] != "OK")

    suggestions: list[dict[str, str]] = []
    if alerts:
        suggestions.append({"label": "Review quality alerts", "key": "alerts",
                            "reason": f"{alerts} parameter(s) exceed WHO guidelines"})
    if prof.discharge_col and prof.has_time:
        if prof.span_years >= 3:
            suggestions.append({"label": "Flood frequency analysis", "key": "extremes",
                                "reason": f"{prof.span_years:.0f} years of discharge"})
        suggestions.append({"label": "Baseflow and flow signatures", "key": "hydrology",
                            "reason": "a discharge column was detected"})
    if int(round(max(0.0, score))) < 85:
        suggestions.append({"label": "Clean and preprocess", "key": "analysis", "reason": "; ".join(notes)})
    if prof.has_time and prof.value_col:
        suggestions.append({"label": "Plot the time series", "key": "visualize", "reason": "time and value columns"})
    if prof.has_geo:
        suggestions.append({"label": "Map the stations", "key": "visualize", "reason": "coordinates present"})

    return {
        "profile": jsonable(asdict(prof)),
        "quality_score": int(max(0, round(score))),
        "quality_notes": notes,
        "n_duplicates": n_dup,
        "who_alerts": alerts,
        "who_checked": len(screen["rows"]),
        "suggestions": suggestions[:4],
    }


# ── water quality ───────────────────────────────────────────────────────────

WHO_GUIDELINES: dict[str, tuple[float, float, str]] = {
    "ph": (6.5, 8.5, "pH units"),
    "dissolved_oxygen": (5.0, float("inf"), "mg/L"),
    "turbidity": (0.0, 5.0, "NTU"),
    "nitrate": (0.0, 50.0, "mg/L"),
    "e_coli": (0.0, 0.0, "CFU/100mL"),
    "arsenic": (0.0, 0.01, "mg/L"),
    "lead": (0.0, 0.01, "mg/L"),
    "mercury": (0.0, 0.001, "mg/L"),
}


def who_screen(df: pd.DataFrame) -> dict[str, Any]:
    """Share of samples outside the WHO drinking-water guideline, per parameter."""
    prof = profile(df)
    rows: list[dict[str, Any]] = []
    if not prof.has_params:
        return {"rows": [], "note": "This table has no parameter/value columns to screen."}
    for name in df[prof.param_col].astype(str).str.lower().unique():
        limits = WHO_GUIDELINES.get(name)
        if not limits:
            continue
        lo, hi, unit = limits
        subset = df[df[prof.param_col].astype(str).str.lower() == name][prof.value_col].dropna()
        if subset.empty:
            continue
        if math.isinf(hi):
            n_exceed = int((subset < lo).sum())
            rule = f"at least {lo} {unit}"
        elif lo == 0:
            n_exceed = int((subset > hi).sum())
            rule = f"at most {hi} {unit}"
        else:
            n_exceed = int(((subset < lo) | (subset > hi)).sum())
            rule = f"{lo} to {hi} {unit}"
        n = int(len(subset))
        pct = round(100.0 * n_exceed / n, 1)
        rows.append({
            "parameter": name, "rule": rule, "n": n, "n_exceed": n_exceed, "pct": pct,
            "status": "Alert" if pct > 10 else "Warning" if n_exceed else "OK",
        })
    return {
        "rows": rows,
        "n_alerts": sum(1 for r in rows if r["status"] == "Alert"),
        "n_warnings": sum(1 for r in rows if r["status"] == "Warning"),
        "methods": [{
            "name": "WHO drinking-water guideline screen",
            "text": "Share of samples outside the WHO guideline range for each recognised parameter; "
                    "over 10 % is reported as an alert, any exceedance as a warning.",
            "citation": "World Health Organization (2022). Guidelines for drinking-water quality, 4th edition, "
                        "incorporating the first and second addenda.",
        }],
    }


# ── hydrology ───────────────────────────────────────────────────────────────


def flow_duration(df: pd.DataFrame, column: str | None = None, *,
    percentiles: list[float] | None = None) -> dict[str, Any]:
    """Flow-duration curve and its percentiles."""
    from aquascope.hydrology import flow_duration_curve

    col = pick_column(df, column, prefer="discharge")
    q = df[col].dropna()
    res = flow_duration_curve(q, percentiles=percentiles) if percentiles else flow_duration_curve(q)
    step = max(1, len(res.exceedance) // 1500)
    return {
        "column": col,
        "n": int(len(q)),
        "percentiles": {str(k): jsonable(v) for k, v in res.percentiles.items()},
        "exceedance": jsonable(res.exceedance[::step]),
        "discharge": jsonable(res.discharge[::step]),
        "methods": [{
            "name": "Flow-duration curve",
            "text": "Daily flows ranked and plotted against the percentage of time they are exceeded; "
                    "percentiles read from the ranked series.",
            "citation": "Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves I: new interpretation and "
                        "confidence intervals. J. Water Resour. Plann. Manage., 120(4), 485-504.",
        }],
    }


BASEFLOW_METHODS = ("lyne_hollick", "eckhardt", "ukih")


def baseflow(df: pd.DataFrame, column: str | None = None, *, method: str = "lyne_hollick",
             alpha: float = 0.925, n_passes: int = 3, bfi_max: float = 0.8,
             block_size: int = 5) -> dict[str, Any]:
    """Split a hydrograph into baseflow and quickflow."""
    from aquascope import hydrology

    if method not in BASEFLOW_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(BASEFLOW_METHODS)}")
    col = pick_column(df, column, prefer="discharge")
    prof = profile(df)
    q = datetime_indexed(df, col, prof) if prof.has_time else df[col].dropna()
    if method == "lyne_hollick":
        res = hydrology.lyne_hollick(q, alpha=alpha, n_passes=n_passes)
    elif method == "eckhardt":
        res = hydrology.eckhardt(q, alpha=alpha, bfi_max=bfi_max)
    else:
        res = hydrology.ukih(q, block_size=block_size)
    frame = res.df
    return {
        "column": col,
        "method": res.method,
        "bfi": jsonable(res.bfi),
        "series": {
            "index": [t.isoformat() if isinstance(t, pd.Timestamp) else jsonable(t) for t in frame.index[::max(1,
                len(frame) // 4000)]],
            "total": jsonable(frame["total"].to_numpy()[::max(1, len(frame) // 4000)]),
            "baseflow": jsonable(frame["baseflow"].to_numpy()[::max(1, len(frame) // 4000)]),
        },
        "parameters": {"alpha": alpha, "n_passes": n_passes, "bfi_max": bfi_max, "block_size": block_size},
        "methods": [{
            "name": f"Baseflow separation ({res.method})",
            "text": "Recursive digital filter (Lyne-Hollick or Eckhardt) or the UKIH smoothed-minima method; "
                    "the baseflow index is the baseflow volume over the total volume.",
            "citation": "Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling; "
                        "Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. "
                        "Hydrol. Process. 19, 507-515; Institute of Hydrology (1980). Low flow studies report 3.",
        }],
    }


def recession(df: pd.DataFrame, column: str | None = None, *, min_length: int = 5) -> dict[str, Any]:
    """Recession segments and the recession constant."""
    from aquascope.hydrology import recession_analysis

    col = pick_column(df, column, prefer="discharge")
    prof = profile(df)
    q = datetime_indexed(df, col, prof) if prof.has_time else df[col].dropna()
    res = recession_analysis(q, min_length=min_length)
    segments = [{
        "start": jsonable(s.start), "end": jsonable(s.end), "n_days": int(len(s.discharge)),
        "q_start": jsonable(s.discharge[0]) if len(s.discharge) else None,
        "q_end": jsonable(s.discharge[-1]) if len(s.discharge) else None,
    } for s in res.segments]
    return {
        "column": col,
        "recession_constant": jsonable(res.recession_constant),
        "r_squared": jsonable(res.r_squared),
        "half_life_days": jsonable(res.half_life_days),
        "n_segments": len(segments),
        "segments": segments[:50],
        "methods": [{
            "name": "Recession analysis",
            "text": "Falling limbs longer than the minimum length are extracted and fitted with an exponential "
                    "storage-outflow relation; the constant is the mean decay rate and the half-life follows from it.",
            "citation": "Tallaksen, L. M. (1995). A review of baseflow recession analysis. J. Hydrol. 165, 349-370.",
        }],
    }


def flood_frequency(df: pd.DataFrame, column: str | None = None) -> dict[str, Any]:
    """GEV flood frequency on annual maxima, with bootstrap confidence limits."""
    from aquascope.hydrology import fit_gev

    col = pick_column(df, column, prefer="discharge")
    q = datetime_indexed(df, col)
    res = fit_gev(q)
    return {
        "column": col,
        "distribution": res.distribution,
        "params": jsonable(list(res.params)),
        "return_periods": {str(k): jsonable(v) for k, v in res.return_periods.items()},
        "confidence_intervals": {str(k): jsonable(list(v)) for k, v in (res.confidence_intervals or {}).items()},
        "annual_max": _series_payload(res.annual_max) if res.annual_max is not None else None,
        "methods": [{
            "name": "GEV flood frequency",
            "text": "Annual maxima fitted to a Generalized Extreme Value distribution; return levels for "
                    "T = 2 to 500 years with 90 % bootstrap confidence limits (1,000 resamples, seed 42).",
            "citation": "Hosking, J. R. M. (1990). L-moments. J. R. Stat. Soc. B, 52(1), 105-124; "
                        "Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.",
        }],
    }


def signatures(df: pd.DataFrame, column: str | None = None) -> dict[str, Any]:
    """The twenty-odd flow signatures of a daily discharge record."""
    from aquascope.hydrology import compute_signatures

    col = pick_column(df, column, prefer="discharge")
    q = datetime_indexed(df, col)
    if not isinstance(q.index, pd.DatetimeIndex) or len(q) < 365:
        raise ValueError("Flow signatures need at least one year of dated daily discharge.")
    rep = compute_signatures(q)
    return {
        "column": col,
        "signatures": jsonable(asdict(rep)),
        "methods": [{
            "name": "Flow signatures",
            "text": "Magnitude, variability, timing and shape indices of a daily flow record (mean and median flow, "
                    "Q5 and Q95, baseflow index, high- and low-flow frequency and duration, seasonality, flashiness).",
            "citation": "Olden, J. D., & Poff, N. L. (2003). Redundancy and the choice of hydrologic indices. "
                        "River Res. Applic. 19, 101-121; Addor, N. et al. (2018). Water Resour. Res. 54, 8792-8812.",
        }],
    }


DISTRIBUTIONS = ("gev", "lp3", "gumbel")


def return_periods(df: pd.DataFrame, column: str | None = None, *, distribution: str = "gev",
                   periods: list[float] | None = None, confidence_level: float = 0.95,
                   n_bootstrap: int = 300) -> dict[str, Any]:
    """Return levels for a chosen distribution, with confidence limits and the empirical points."""
    from aquascope.analysis.extreme_events import estimate_return_periods

    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution!r}; choose from {list(DISTRIBUTIONS)}")
    col = pick_column(df, column, prefer="discharge")
    series = datetime_indexed(df, col)
    rp = tuple(float(t) for t in sorted(periods or [2, 5, 10, 25, 50, 100]))
    n_years = int(series.resample("YE").max().dropna().shape[0]) if isinstance(series.index,
        pd.DatetimeIndex) else int(series.dropna().shape[0])
    if n_years < 3:
        raise ValueError(f"Only {n_years} year(s) of data; frequency analysis needs at least three.")
    res = estimate_return_periods(
        series, distribution=distribution, return_periods=rp,
        confidence_level=confidence_level, n_bootstrap=int(n_bootstrap),
    )
    # The observed points, by the Weibull plotting position.
    amax = (series.resample("YE").max().dropna() if isinstance(series.index,
        pd.DatetimeIndex) else series.dropna()).to_numpy()
    amax_sorted = np.sort(amax)
    n = amax_sorted.size
    ranks = np.arange(1, n + 1)
    empirical_t = (n + 1) / (n + 1 - ranks)
    return {
        "column": col,
        "distribution": res.distribution,
        "return_periods": jsonable(res.return_periods),
        "return_levels": jsonable(res.return_levels),
        "lower_bound": jsonable(res.lower_bound),
        "upper_bound": jsonable(res.upper_bound),
        "confidence_level": jsonable(res.confidence_level),
        "n_years": n_years,
        "fit": jsonable(asdict(res.fit)),
        "empirical": {"return_period": jsonable(empirical_t), "value": jsonable(amax_sorted)},
        "methods": [{
            "name": f"Frequency analysis ({res.distribution.upper()})",
            "text": "Annual maxima fitted to the chosen distribution by maximum likelihood; confidence limits from a "
                    "parametric bootstrap (seed 42); observed points placed by the Weibull formula (n+1)/(n+1-rank).",
            "citation": "Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer; "
                        "England, J. F. Jr. et al. (2018). Bulletin 17C. USGS Techniques and Methods 4-B5.",
        }],
    }


# ── agriculture ─────────────────────────────────────────────────────────────


def reference_et(weather: pd.DataFrame, *, latitude: float, elevation: float) -> dict[str, Any]:
    """FAO-56 Penman-Monteith reference evapotranspiration from a daily weather table."""
    from aquascope.agri.eto import penman_monteith_series

    eto = penman_monteith_series(weather, latitude=latitude, elevation=elevation)
    return {
        "eto": _series_payload(eto),
        "mean_mm_per_day": jsonable(float(eto.mean())),
        "total_mm": jsonable(float(eto.sum())),
        "methods": [{
            "name": "FAO-56 Penman-Monteith reference ET0",
            "text": "Daily reference evapotranspiration for a hypothetical grass reference crop from temperature, "
                    "humidity, wind speed and solar radiation.",
            "citation": "Allen, R. G., Pereira, L. S., Raes, D., Smith, M. (1998). Crop evapotranspiration: "
                        "guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56.",
        }],
    }


def irrigation(weather: pd.DataFrame, *, latitude: float, elevation: float, crop: str,
               planting_date: str | date, precipitation: pd.Series | None = None,
               efficiency: float = 0.7, method: str = "single", kc_max: float = 1.20,
               few: float = 1.0, kr: float = 1.0) -> dict[str, Any]:
    """A season's crop water requirement and irrigation schedule."""
    from aquascope.agri import irrigation_schedule
    from aquascope.agri.eto import penman_monteith_series

    eto = penman_monteith_series(weather, latitude=latitude, elevation=elevation)
    if precipitation is None:
        precipitation = weather["precipitation"] if "precipitation" in weather.columns else pd.Series(0.0,
            index=weather.index)
    planting = pd.to_datetime(planting_date).date() if not isinstance(planting_date, date) else planting_date
    sched = irrigation_schedule(
        eto, precipitation, crop, planting, efficiency=efficiency,
        method=method, kc_max=kc_max, few=few, kr=kr,
    )
    totals = {c: jsonable(float(sched[c].sum())) for c in ("etc", "effective_rain", "net_irrigation",
        "gross_irrigation") if c in sched}
    return {
        "crop": crop,
        "method": method,
        "planting_date": planting.isoformat(),
        "season_days": int(len(sched)),
        "eto_mean_mm_per_day": jsonable(float(eto.mean())),
        "totals_mm": totals,
        "schedule": jsonable(sched),
        "methods": [{
            "name": "FAO-56 crop water requirement and irrigation schedule",
            "text": "Reference ET0 by Penman-Monteith, crop evapotranspiration through the crop coefficient "
                    "(single Kc, or dual Kcb and Ke), effective rainfall subtracted, and the net demand divided by "
                    "the irrigation efficiency to give the gross depth.",
            "citation": "Allen, R. G. et al. (1998). FAO Irrigation and Drainage Paper 56.",
        }],
    }


# ── groundwater ─────────────────────────────────────────────────────────────


def sgi_drought(df: pd.DataFrame, column: str | None = None, *, min_per_month: int = 5,
                threshold: float = -1.0) -> dict[str, Any]:
    """Standardised Groundwater Index and the droughts it identifies."""
    from aquascope.groundwater.drought import drought_events, standardised_groundwater_index

    col = pick_column(df, column, prefer="level")
    levels = datetime_indexed(df, col)
    sgi = standardised_groundwater_index(levels, min_per_month=min_per_month)
    events = drought_events(sgi, threshold=threshold)
    latest = sgi.dropna()
    return {
        "column": col,
        "sgi": _series_payload(sgi),
        "current": jsonable(latest.iloc[-1]) if not latest.empty else None,
        "worst": jsonable(sgi.min()),
        "threshold": threshold,
        "events": [{
            "start": jsonable(e.start), "end": jsonable(e.end), "duration": int(e.duration),
            "severity": jsonable(e.severity), "peak": jsonable(e.peak),
        } for e in events],
        "methods": [{
            "name": "Standardised Groundwater Index",
            "text": "Groundwater levels transformed to a standard normal variate month by month; a drought runs "
                    "while the index stays at or below the threshold, with severity as the summed deficit.",
            "citation": "Bloomfield, J. P., & Marchant, B. P. (2013). Analysis of groundwater drought building on "
                        "the standardised precipitation index approach. Hydrol. Earth Syst. Sci. 17, 4769-4787.",
        }],
    }


def recharge(df: pd.DataFrame, column: str | None = None, *, specific_yield: float = 0.15) -> dict[str, Any]:
    """Water-table fluctuation recharge estimate."""
    from aquascope.groundwater.recharge import water_table_fluctuation

    col = pick_column(df, column, prefer="level")
    levels = datetime_indexed(df, col)
    res = water_table_fluctuation(levels, specific_yield=specific_yield)
    return {
        "column": col,
        "method": res.method,
        "value_mm_per_year": jsonable(res.value_mm_per_year),
        "uncertainty": jsonable(res.uncertainty),
        "metadata": jsonable(res.metadata),
        "methods": [{
            "name": "Water-table fluctuation recharge",
            "text": "Recharge as the specific yield times the sum of water-table rises over the record, divided by "
                    "its length. Assumes rises come from recharge alone.",
            "citation": "Healy, R. W., & Cook, P. G. (2002). Using groundwater levels to estimate recharge. "
                        "Hydrogeology Journal 10, 91-109.",
        }],
    }


def aquifer_drawdown(*, transmissivity: float, storativity: float, pumping_rate: float,
                     distance: float, time_days: float) -> dict[str, Any]:
    """Theis drawdown at a distance from a well after a time of pumping."""
    from aquascope.groundwater.aquifer import theis_drawdown

    s = theis_drawdown(transmissivity, storativity, pumping_rate, distance, time_days)
    return {
        "drawdown_m": jsonable(float(np.atleast_1d(s)[0])),
        "inputs": {
            "transmissivity_m2_per_day": transmissivity, "storativity": storativity,
            "pumping_rate_m3_per_day": pumping_rate, "distance_m": distance, "time_days": time_days,
        },
        "methods": [{
            "name": "Theis solution",
            "text": "Drawdown in a confined aquifer of infinite extent from a fully penetrating well pumped at a "
                    "constant rate, s = Q/(4 pi T) W(u) with u = r^2 S / (4 T t).",
            "citation": "Theis, C. V. (1935). The relation between the lowering of the piezometric surface and the "
                        "rate and duration of discharge of a well using ground-water storage. Eos 16, 519-524.",
        }],
    }


# ── the registry the CLI, MCP and the browser share ─────────────────────────

TOOLS: dict[str, dict[str, Any]] = {
    "eda": {"func": eda, "needs": "frame", "summary": "Exploratory summary of a table."},
    "quality": {"func": quality, "needs": "frame", "summary": "Duplicates, gaps, outliers and what to fix."},
    "preprocess": {"func": preprocess, "needs": "frame", "summary": "Clean a table with a list of steps."},
    "insights": {"func": insights, "needs": "frame", "summary": "Quality score, WHO screen and next steps."},
    "who_screen": {"func": who_screen, "needs": "frame", "summary": "WHO drinking-water guideline screen."},
    "flow_duration": {"func": flow_duration, "needs": "frame", "summary": "Flow-duration curve and percentiles."},
    "baseflow": {"func": baseflow, "needs": "frame", "summary": "Baseflow separation and the baseflow index."},
    "recession": {"func": recession, "needs": "frame", "summary": "Recession segments and constant."},
    "flood_frequency": {"func": flood_frequency, "needs": "frame", "summary": "GEV flood frequency on annual maxima."},
    "signatures": {"func": signatures, "needs": "frame", "summary": "Flow signatures of a daily record."},
    "return_periods": {"func": return_periods, "needs": "frame", "summary": "Return levels for GEV, LP3 or Gumbel."},
    "reference_et": {"func": reference_et, "needs": "weather", "summary": "FAO-56 reference evapotranspiration."},
    "irrigation": {"func": irrigation, "needs": "weather",
        "summary": "Crop water requirement and irrigation schedule."},
    "sgi_drought": {"func": sgi_drought, "needs": "frame", "summary": "Groundwater drought index and events."},
    "recharge": {"func": recharge, "needs": "frame", "summary": "Water-table fluctuation recharge."},
    "aquifer_drawdown": {"func": aquifer_drawdown, "needs": "none", "summary": "Theis drawdown at a distance."},
}


def run(name: str, df: pd.DataFrame | None = None, **params: Any) -> dict[str, Any]:
    """Run a workbench analysis by name (what the MCP tool and the browser call)."""
    spec = TOOLS.get(name)
    if not spec:
        raise ValueError(f"Unknown analysis {name!r}; choose from {sorted(TOOLS)}")
    if spec["needs"] == "none":
        return spec["func"](**params)
    if df is None:
        raise ValueError(f"{name} needs a table of data.")
    return spec["func"](df, **params)
