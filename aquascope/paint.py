"""Paint the globe (#657): one question, answered at every gauge in view, with field significance.

Studies answer for one place; hydrologists publish maps of answers ("are floods getting bigger across
Europe?"). :func:`paint` asks one question of every gauge in a box that the AquaScope Archive mirrors, and
says, besides each gauge's own answer, whether the pattern as a whole is more than chance.

One engine, three faces: the Explorer's "Paint this view" (the worker calls :func:`gauge_rows` batch by batch
as the page downloads the records, then :func:`summarize`), ``aquascope paint`` and the MCP tool ``paint_map``
all run these functions. Only numpy, scipy and pandas are needed, so it runs in Pyodide.

The questions (:data:`QUESTIONS`, one registry, each a plain function of one gauge):

* ``flood_trend``: the trend in the annual maximum daily flow over complete hydrological years
  (October to September in the north, April to March in the south): Mann-Kendall p-value, Sen's slope in
  per cent of the mean annual maximum per decade, and the years used.
* ``low_flow_trend``: the same for the annual minimum 7-day mean flow, over climate years (April to March in
  the north, October to September in the south), so a dry season is never cut in two.
* ``baseflow_index``: the Lyne-Hollick baseflow index the Archive already holds in ``signatures.parquet``
  (no record is read).

Field significance (trend questions): the Benjamini-Hochberg false discovery rate over the gauges' p-values
at ``alpha_FDR = 2 * alpha_global``, which Wilks (2016) recommends for spatially correlated tests (nearby
gauges share floods), or at ``alpha_global`` with ``correlated=False``. The field is significant at
``alpha_global`` when at least one gauge survives.

Coverage is said, never implied away: a gauge in the box that the Archive does not mirror is ``no_archive``, one
whose record is shorter than the minimum is ``too_short``, and gauges past the per-view cap are ``over_cap``.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Gauges read for one painting at most (the longest Archive records first); the rest are ``over_cap``.
MAX_GAUGES = 300
#: Complete years a gauge needs before its trend is tested (the user can raise or lower it).
MIN_YEARS = 20
#: Days a hydrological year needs to count as complete (330 of 365, about 90 %).
MIN_DAYS = 330
#: The global (field) significance level.
ALPHA_GLOBAL = 0.05
#: Per-gauge statuses, in the order the legend lists them.
STATUSES = ("painted", "too_short", "no_archive", "over_cap", "failed")

#: The references every methods paragraph may cite, by key.
REFERENCES: dict[str, str] = {
    "mann1945": "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica, 13(3), 245-259.",
    "kendall1975": "Kendall, M. G. (1975). Rank Correlation Methods (4th ed.). Charles Griffin, London.",
    "sen1968": "Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. Journal of the "
               "American Statistical Association, 63(324), 1379-1389.",
    "bh1995": "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful "
              "approach to multiple testing. Journal of the Royal Statistical Society B, 57(1), 289-300.",
    "wilks2016": "Wilks, D. S. (2016). \"The stippling shows statistically significant grid points\": how research "
                 "results are routinely overstated and overinterpreted, and what to do about it. Bulletin of the "
                 "American Meteorological Society, 97(12), 2263-2273. https://doi.org/10.1175/BAMS-D-15-00267.1",
    "lyne1979": "Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling. Institution "
                "of Engineers Australia National Conference, Perth, 89-93.",
    "aquascope": "AquaScope (software and Archive). https://doi.org/10.5281/zenodo.21903143",
}

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


# ── the registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """One question the map can be painted with.

    ``kind`` is "trend" (a value, a p-value and field significance) or "value" (a value only); ``needs`` is
    "daily" (each gauge's record is read) or "signatures" (the Archive's per-gauge table answers it).
    ``breaks`` are the legend's class edges in ``unit``; ``year_start`` the month a year starts in, north and
    south of the equator.
    """

    id: str
    label: str
    short: str
    kind: str
    needs: str
    unit: str
    breaks: tuple[float, ...]
    diverging: bool
    describe: str
    year_start: tuple[int, int] = (10, 4)
    references: tuple[str, ...] = ()
    compute: Callable[..., dict[str, Any]] | None = field(default=None, compare=False, repr=False)

    def public(self) -> dict[str, Any]:
        """The question as JSON (what the Explorer's picker and legend need)."""
        return {"id": self.id, "label": self.label, "short": self.short, "kind": self.kind, "needs": self.needs,
                "unit": self.unit, "breaks": list(self.breaks), "diverging": self.diverging,
                "describe": self.describe, "year_start": {"north": self.year_start[0], "south": self.year_start[1]}}


# ── small helpers ───────────────────────────────────────────────────────────


def _num(x: Any, digits: int = 4) -> float | None:
    """JSON-safe number: NaN and inf become None; ``digits`` significant figures."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    if v == 0:
        return 0.0
    mag = max(0, digits - 1 - int(math.floor(math.log10(abs(v)))))
    return round(v, mag)


def _key(source: str, station_id: str) -> str:
    return f"{source}/{station_id}"


def start_month_for(question: str | Question, latitude: float | None, override: int | None = None) -> int:
    """The month a year starts in for ``question`` at ``latitude`` (``override`` wins when it is 1 to 12)."""
    if override is not None:
        m = int(override)
        if not 1 <= m <= 12:
            raise ValueError("year_start_month must be 1 to 12")
        return m
    q = QUESTIONS[question] if isinstance(question, str) else question
    south = latitude is not None and float(latitude) < 0
    return q.year_start[1] if south else q.year_start[0]


def year_words(start_month: int) -> str:
    """"October to September" for a year starting in October; "calendar years" for January."""
    if start_month == 1:
        return "calendar years"
    return f"{_MONTHS[start_month - 1]} to {_MONTHS[(start_month - 2) % 12]}"


def year_labels(index: pd.DatetimeIndex, start_month: int) -> np.ndarray:
    """Each day's hydrological year, named by the calendar year it ends in (water year 2020 is Oct 2019 to Sep 2020)."""
    years: np.ndarray = np.asarray(index.year, dtype=int)
    if start_month == 1:
        return years
    return years + (np.asarray(index.month) >= start_month).astype(int)


def annual_series(series: pd.Series, *, stat: str = "max", start_month: int = 10,
                  min_days: int = MIN_DAYS) -> pd.Series:
    """One value per complete year: the maximum daily flow (``stat="max"``) or the minimum 7-day mean
    (``stat="min7"``), over years starting in ``start_month``.

    The record is first made daily (the mean of each day, as the single-station study does), and a year is
    complete when at least ``min_days`` of its days have a value. Returns a Series indexed by the year's label
    (the calendar year it ends in). With ``start_month=1`` and ``min_days=292`` it is exactly
    :func:`aquascope.explore._annual_max` (the calendar-year rule of the single-station study).
    """
    s = series.dropna()
    if not len(s):
        return pd.Series(dtype=float)
    daily = s.resample("D").mean()
    labels = year_labels(daily.index, start_month)
    counts = daily.notna().groupby(labels).sum()
    if stat == "max":
        values = daily.groupby(labels).max()
    elif stat == "min7":
        # A 7-day window counts for the year its last day falls in, and needs all seven days.
        values = daily.rolling(7, min_periods=7).mean().groupby(labels).min()
    else:
        raise ValueError("stat is max or min7")
    keep = counts[counts >= int(min_days)].index
    out = values.reindex(keep).dropna()
    out.index = out.index.astype(int)
    out.index.name = "year"
    return out.astype(float)


def sen_slope_years(years: Iterable[float], values: Iterable[float]) -> float:
    """Sen's slope against the actual years: the median of every pairwise slope (y_j - y_i) / (t_j - t_i).

    A missing year stays a gap in time, so the slope is per year even when a year is not complete. With no
    year missing it equals :func:`aquascope.analysis.trends.sens_slope` on the values alone.
    """
    t: np.ndarray = np.asarray(list(years), dtype=float)
    y: np.ndarray = np.asarray(list(values), dtype=float)
    i, j = np.triu_indices(len(y), k=1)
    dt = t[j] - t[i]
    ok = dt != 0
    if not ok.any():
        return float("nan")
    return float(np.median((y[j] - y[i])[ok] / dt[ok]))


# ── one gauge ───────────────────────────────────────────────────────────────


def annual_trend(annual: pd.Series, *, min_years: int = MIN_YEARS, alpha: float = ALPHA_GLOBAL,
                 start_year: int | None = None, end_year: int | None = None) -> dict[str, Any]:
    """Mann-Kendall and Sen's slope of an annual series (index: year): the answer for one gauge.

    Returns ``status`` ("painted" or "too_short"), ``value`` (Sen's slope in % of the series mean per decade),
    ``p``, ``tau``, ``slope_per_year`` (in the series' unit), ``mean``, ``n_years``, ``first_year``,
    ``last_year``, ``direction`` ("up", "down" or "none" at ``alpha``) and a ``note``.
    """
    a = annual.dropna()
    if start_year is not None:
        a = a[a.index >= int(start_year)]
    if end_year is not None:
        a = a[a.index <= int(end_year)]
    n = int(len(a))
    out: dict[str, Any] = {
        "status": "too_short", "value": None, "p": None, "tau": None, "slope_per_year": None, "mean": None,
        "n_years": n, "first_year": int(a.index.min()) if n else None, "last_year": int(a.index.max()) if n else None,
        "missing_years": None, "direction": None, "note": "",
    }
    if n:
        out["missing_years"] = int(a.index.max() - a.index.min() + 1 - n)
    if n < int(min_years):
        out["note"] = f"{n} complete years; the question needs {int(min_years)}"
        return out
    from aquascope.analysis.trends import mann_kendall

    mk = mann_kendall(a.to_numpy(dtype=float), alpha=alpha)
    slope = sen_slope_years(a.index, a.to_numpy(dtype=float))
    mean = float(a.mean())
    p = float(mk.p_value)
    out.update(status="painted", p=p, tau=_num(mk.tau), slope_per_year=_num(slope), mean=_num(mean),
               direction=("up" if mk.tau > 0 else "down") if p < alpha else "none")
    if mean > 0 and math.isfinite(slope):
        out["value"] = _num(100.0 * 10.0 * slope / mean, 3)
    else:
        out["note"] = "the mean is zero, so the change has no per-cent size"
    return out


def flood_trend(series: pd.Series, *, latitude: float | None = None, year_start_month: int | None = None,
                min_years: int = MIN_YEARS, min_days: int = MIN_DAYS, start_year: int | None = None,
                end_year: int | None = None, alpha: float = ALPHA_GLOBAL) -> dict[str, Any]:
    """The trend in annual maximum daily flow at one gauge (see :func:`annual_trend` for the keys)."""
    m = start_month_for("flood_trend", latitude, year_start_month)
    out = annual_trend(annual_series(series, stat="max", start_month=m, min_days=min_days), min_years=min_years,
                       alpha=alpha, start_year=start_year, end_year=end_year)
    out["year_start_month"] = m
    return out


def low_flow_trend(series: pd.Series, *, latitude: float | None = None, year_start_month: int | None = None,
                   min_years: int = MIN_YEARS, min_days: int = MIN_DAYS, start_year: int | None = None,
                   end_year: int | None = None, alpha: float = ALPHA_GLOBAL) -> dict[str, Any]:
    """The trend in the annual minimum 7-day mean flow at one gauge (see :func:`annual_trend`)."""
    m = start_month_for("low_flow_trend", latitude, year_start_month)
    out = annual_trend(annual_series(series, stat="min7", start_month=m, min_days=min_days), min_years=min_years,
                       alpha=alpha, start_year=start_year, end_year=end_year)
    out["year_start_month"] = m
    return out


def baseflow_index(signature: dict[str, Any], *, min_years: int = MIN_YEARS, **_: Any) -> dict[str, Any]:
    """The baseflow index from a gauge's ``signatures.parquet`` row (Lyne-Hollick, built by the Archive)."""
    years = signature.get("record_years")
    years = float(years) if years is not None and not (isinstance(years, float) and math.isnan(years)) else None
    bfi = _num(signature.get("bfi"), 3)
    out: dict[str, Any] = {"status": "too_short", "value": None, "p": None, "n_years": None,
                           "first_year": None, "last_year": None, "direction": None, "note": ""}
    for k, col in (("first_year", "record_start"), ("last_year", "record_end")):
        v = signature.get(col)
        if v:
            out[k] = int(str(v)[:4])
    data_years = signature.get("data_years")
    try:
        out["n_years"] = _num(float(data_years), 3) if data_years is not None else None
    except (TypeError, ValueError):
        out["n_years"] = None
    if years is None or years < min_years:
        out["note"] = f"{years or 0:.1f} years in the Archive; the question needs {int(min_years)}"
        return out
    if bfi is None:
        out["note"] = str(signature.get("notes") or "no baseflow index in signatures.parquet")
        return out
    out.update(status="painted", value=bfi)
    return out


def _flood_compute(series: pd.Series, **kw: Any) -> dict[str, Any]:
    return flood_trend(series, **kw)


def _low_compute(series: pd.Series, **kw: Any) -> dict[str, Any]:
    return low_flow_trend(series, **kw)


QUESTIONS: dict[str, Question] = {
    "flood_trend": Question(
        id="flood_trend", label="Trend in annual floods", short="Flood trend", kind="trend", needs="daily",
        unit="% per decade", breaks=(-10.0, -5.0, -2.0, 2.0, 5.0, 10.0), diverging=True,
        describe="Sen's slope of the annual maximum daily flow, in per cent of its mean per decade, over complete "
                 "hydrological years; Mann-Kendall p-value.",
        year_start=(10, 4), references=("mann1945", "kendall1975", "sen1968", "bh1995", "wilks2016"),
        compute=_flood_compute),
    "low_flow_trend": Question(
        id="low_flow_trend", label="Trend in low flows", short="Low-flow trend", kind="trend", needs="daily",
        unit="% per decade", breaks=(-10.0, -5.0, -2.0, 2.0, 5.0, 10.0), diverging=True,
        describe="Sen's slope of the annual minimum 7-day mean flow, in per cent of its mean per decade, over "
                 "complete climate years; Mann-Kendall p-value.",
        year_start=(4, 10), references=("mann1945", "kendall1975", "sen1968", "bh1995", "wilks2016"),
        compute=_low_compute),
    "baseflow_index": Question(
        id="baseflow_index", label="Baseflow index", short="Baseflow index", kind="value", needs="signatures",
        unit="", breaks=(0.2, 0.35, 0.5, 0.65, 0.8), diverging=False,
        describe="The share of the flow that is baseflow, from the Lyne-Hollick filter (alpha 0.925, three passes) "
                 "the Archive runs on every mirrored record.",
        references=("lyne1979",), compute=baseflow_index),
}

#: The questions planned next (#657): they plug into :data:`QUESTIONS` as plain functions of one gauge.
PLANNED = ("q100_per_km2", "geoglows_kge")


def questions() -> list[dict[str, Any]]:
    """The registry as JSON, in picker order."""
    return [q.public() for q in QUESTIONS.values()]


def _question(question: str) -> Question:
    if question not in QUESTIONS:
        raise ValueError(f"unknown question {question!r}; the questions are {', '.join(QUESTIONS)}")
    return QUESTIONS[question]


def class_of(question: str, value: float | None) -> int | None:
    """The legend class (0 to len(breaks)) a value falls in, or None for no value."""
    if value is None:
        return None
    return int(np.searchsorted(np.asarray(_question(question).breaks), float(value), side="right"))


# ── which gauges ────────────────────────────────────────────────────────────


def _in_box(lat: float, lon: float, bbox: list[float] | tuple[float, ...]) -> bool:
    west, south, east, north = (float(x) for x in bbox)
    if not south <= lat <= north:
        return False
    return west <= lon <= east if west <= east else (lon >= west or lon <= east)


def _sig_rows(signatures: Any) -> dict[str, dict[str, Any]]:
    if signatures is None:
        return {}
    if isinstance(signatures, pd.DataFrame):
        recs = signatures.to_dict("records")
    elif isinstance(signatures, dict):
        return {str(k): dict(v) for k, v in signatures.items()}
    else:
        recs = list(signatures)
    out = {}
    for r in recs:
        if r.get("variable") not in (None, "discharge"):
            continue
        out[_key(str(r["source"]), str(r["station_id"]))] = dict(r)
    return out


def _years(sig: dict[str, Any] | None) -> float:
    if not sig:
        return -1.0
    v: Any = sig.get("record_years")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if math.isnan(f) else f


def plan(stations: list[dict[str, Any]], signatures: Any, *, bbox: list[float] | None = None,
         min_years: float = MIN_YEARS, max_gauges: int = MAX_GAUGES) -> dict[str, Any]:
    """Which gauges a painting reads, and why the others are not painted.

    ``stations`` are catalog rows (``source``, ``station_id``, optional ``site_id``, ``name``,
    ``latitude``/``lat``, ``longitude``/``lon``, ``variables``); only those that record discharge (or say
    nothing) and sit in ``bbox`` count. ``signatures`` (the Archive's ``signatures.parquet``, a DataFrame or
    rows) says which the Archive mirrors and for how long. One gauge per site (the longest Archive record). A
    gauge with no Archive record is ``no_archive``, one shorter than ``min_years`` is ``too_short``, and past
    ``max_gauges`` (longest records first) ``over_cap``. The Explorer's ``planGauges`` (paint-core.js) follows
    the same rules on the same fixture.

    Returns ``{"read": [rows to read], "rows": [every gauge with its status], "counts": {status: n}}``.
    """
    sig = _sig_rows(signatures)
    by_site: dict[tuple[str, str], dict[str, Any]] = {}
    for s in stations:
        lat = s.get("latitude", s.get("lat"))
        lon = s.get("longitude", s.get("lon"))
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)
        if bbox is not None and not _in_box(lat, lon, bbox):
            continue
        variables = s.get("variables")
        if variables is not None and len(variables) and "discharge" not in list(variables):
            continue
        key = _key(str(s["source"]), str(s["station_id"]))
        row: dict[str, Any] = {"key": key, "source": str(s["source"]), "station_id": str(s["station_id"]),
               "name": s.get("name") or "", "latitude": lat, "longitude": lon,
               "record_years": _num(_years(sig.get(key)), 4) if key in sig else None}
        site = (row["source"], str(s.get("site_id") or s["station_id"]))
        have = by_site.get(site)
        if have is None:
            by_site[site] = row
        else:
            y_new, y_old = _years(sig.get(key)), _years(sig.get(have["key"]))
            if y_new > y_old or (y_new == y_old and row["station_id"] < have["station_id"]):
                by_site[site] = row
    rows = sorted(by_site.values(), key=lambda r: (-_years(sig.get(r["key"])), r["key"]))
    read: list[dict[str, Any]] = []
    for r in rows:
        y = _years(sig.get(r["key"]))
        if r["key"] not in sig:
            r.update(status="no_archive", note="not in the AquaScope Archive")
        elif y < float(min_years):
            r.update(status="too_short", note=f"{y:.1f} years in the Archive; the question needs {min_years:g}")
        elif len(read) >= int(max_gauges):
            r.update(status="over_cap", note=f"past the cap of {int(max_gauges)} gauges a view; zoom in")
        else:
            r.update(status="queued", note="")
            read.append(r)
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("queued", "too_short", "no_archive", "over_cap")}
    return {"read": read, "rows": rows, "counts": counts}


# ── answering ───────────────────────────────────────────────────────────────


def _params(question: str, params: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(params or {})
    out: dict[str, Any] = {"min_years": int(p.get("min_years") or MIN_YEARS)}
    if _question(question).needs == "daily":
        out["min_days"] = int(p.get("min_days") or MIN_DAYS)
        for k in ("start_year", "end_year", "year_start_month"):
            if p.get(k) not in (None, ""):
                out[k] = int(p[k])
        out["alpha"] = float(p.get("alpha_global") or ALPHA_GLOBAL)
    return out


def answer_gauge(question: str, *, series: pd.Series | None = None, signature: dict[str, Any] | None = None,
                 latitude: float | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """One gauge's answer: ``series`` for a daily question, its ``signature`` row for the others."""
    q = _question(question)
    kw = _params(question, params)
    assert q.compute is not None
    if q.needs == "daily":
        if series is None or not len(series.dropna()):
            return {"status": "failed", "value": None, "p": None, "n_years": 0, "note": "the record is empty"}
        return q.compute(series, latitude=latitude, **kw)
    return q.compute(signature or {}, **kw)


def gauge_rows(question: str, items: list[dict[str, Any]],
               params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Answer a batch of gauges. Each item carries ``key``, ``source``, ``station_id``, ``name``, ``latitude``,
    ``longitude`` and one of ``series`` (a pandas Series), ``csv_gz`` (the Archive's file, as bytes) or
    ``signature`` (a ``signatures.parquet`` row); ``fetch_error`` says why the page could not download it. A gauge
    that cannot be read is ``failed``, never an error."""
    from aquascope.archive.observations import read_csv_gz

    out = []
    for it in items:
        base = {k: it.get(k) for k in ("key", "source", "station_id", "name", "latitude", "longitude",
                                       "record_years")}
        if not base.get("key"):
            base["key"] = _key(str(it.get("source")), str(it.get("station_id")))
        if it.get("fetch_error"):
            out.append({**base, "status": "failed", "value": None, "p": None,
                        "note": str(it["fetch_error"])[:160]})
            continue
        try:
            series = it.get("series")
            if series is None and it.get("csv_gz") is not None:
                series = read_csv_gz(bytes(it["csv_gz"]))
            ans = answer_gauge(question, series=series, signature=it.get("signature"),
                               latitude=it.get("latitude"), params=params)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the painting
            msg = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
            ans = {"status": "failed", "value": None, "p": None, "note": msg}
        out.append({**base, **ans})
    return out


# ── the region ──────────────────────────────────────────────────────────────


def benjamini_hochberg(pvalues: list[float], q: float) -> list[bool]:
    """Which p-values the Benjamini-Hochberg procedure rejects at false discovery rate ``q``."""
    from aquascope.area_study import benjamini_hochberg as _bh

    return list(_bh(pvalues, q=q))


def field_significance(rows: list[dict[str, Any]], *, alpha_global: float = ALPHA_GLOBAL,
                       correlated: bool = True) -> dict[str, Any]:
    """Is the pattern as a whole more than chance? The false discovery rate over the painted gauges' p-values.

    ``alpha_FDR`` is ``2 * alpha_global`` when ``correlated`` (Wilks 2016, for spatially correlated tests), else
    ``alpha_global``. Returns the counts tested, locally significant up and down (p < alpha_global), the gauges
    the procedure keeps (``significant``, by key), the largest p-value it keeps (``p_threshold``), whether the
    field is significant, a one-line verdict and the caveat.
    """
    tested = [r for r in rows if r.get("status") == "painted" and r.get("p") is not None]
    n = len(tested)
    a_fdr = (2.0 if correlated else 1.0) * float(alpha_global)
    local_up = sum(1 for r in tested if r["p"] < alpha_global and (r.get("tau") or 0) > 0)
    local_down = sum(1 for r in tested if r["p"] < alpha_global and (r.get("tau") or 0) < 0)
    out: dict[str, Any] = {
        "n_tested": n, "alpha_global": float(alpha_global), "alpha_fdr": _num(a_fdr), "correlated": bool(correlated),
        "local": {"up": local_up, "down": local_down, "expected_by_chance": _num(alpha_global * n, 3)},
        "significant": [], "n_significant": 0, "up": 0, "down": 0, "p_threshold": None,
        "field_significant": False,
        "caveat": "Gauges on one river, or in one weather system, share floods, so their tests are not independent. "
                  "The false discovery rate at twice the global level keeps the field test near its nominal level "
                  "for moderate to strong spatial correlation (Wilks, 2016); it does not make each gauge's test "
                  "stronger. A trend says nothing about its cause (climate, land use, dams or a new rating).",
    }
    if n == 0:
        out["verdict"] = "No gauge in view has enough complete years to test."
        return out
    rejected = benjamini_hochberg([float(r["p"]) for r in tested], q=a_fdr)
    sig = [r for r, keep in zip(tested, rejected) if keep]
    out.update(significant=[r["key"] for r in sig], n_significant=len(sig),
               up=sum(1 for r in sig if (r.get("tau") or 0) > 0), down=sum(1 for r in sig if (r.get("tau") or 0) < 0),
               p_threshold=_num(max((float(r["p"]) for r in sig), default=None)) if sig else None,
               field_significant=bool(sig))
    head = (f"{local_up} up and {local_down} down of {n} gauges at p < {alpha_global:g} "
            f"(about {alpha_global * n:.0f} by chance)")
    if sig:
        lean = "rising" if out["up"] > out["down"] else "falling" if out["down"] > out["up"] else "mixed"
        out["verdict"] = (f"{head}; {len(sig)} stay significant at a false discovery rate of {a_fdr:g} "
                          f"({out['up']} up, {out['down']} down), so the field is significant at "
                          f"{alpha_global:g}: the {lean} pattern is more than chance.")
    else:
        out["verdict"] = (f"{head}; none survive a false discovery rate of {a_fdr:g}, so the field is not "
                          f"significant at {alpha_global:g}: the pattern could be chance.")
    return out


def _mark(rows: list[dict[str, Any]], field: dict[str, Any] | None, alpha_global: float) -> None:
    keep = set((field or {}).get("significant") or [])
    for r in rows:
        if r.get("status") == "painted" and r.get("p") is not None:
            r["fdr_significant"] = r["key"] in keep
            r["locally_significant"] = bool(r["p"] < alpha_global)
        else:
            r["fdr_significant"] = None
            r["locally_significant"] = None


def _licences(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    from aquascope.registry import SOURCES

    out = []
    for src in sorted({r["source"] for r in rows if r.get("status") == "painted"}):
        info = SOURCES.get(src)
        out.append({"source": src, "label": getattr(info, "label", src) if info else src,
                    "license": getattr(info, "license", "unknown") if info else "unknown",
                    "attribution": getattr(info, "attribution", "") if info else ""})
    return out


TABLE_COLUMNS = [
    "source", "station_id", "name", "latitude", "longitude", "status", "value", "unit", "p", "fdr_significant",
    "locally_significant", "tau", "slope_per_year", "mean", "n_years", "first_year", "last_year", "missing_years",
    "year_start_month", "record_years", "note",
]


def summarize(question: str, rows: list[dict[str, Any]], *, params: dict[str, Any] | None = None,
              bbox: list[float] | None = None) -> dict[str, Any]:
    """The painting as JSON: the gauges (with ``fdr_significant`` marked), field significance, the counts by
    status, a headline, the legend classes, the licences of the sources painted, GeoJSON, the table and the
    methods paragraph with its references."""
    q = _question(question)
    p = dict(params or {})
    alpha_global = float(p.get("alpha_global") or ALPHA_GLOBAL)
    correlated = p.get("correlated", True) is not False
    rows = [dict(r) for r in rows]
    for r in rows:
        r["unit"] = q.unit
    field = field_significance(rows, alpha_global=alpha_global, correlated=correlated) if q.kind == "trend" else None
    _mark(rows, field, alpha_global)
    counts = {k: sum(1 for r in rows if r.get("status") == k) for k in STATUSES}
    painted = [r for r in rows if r.get("status") == "painted" and r.get("value") is not None]
    values = [float(r["value"]) for r in painted]
    stats = {"n": len(values), "median": _num(float(np.median(values)), 3) if values else None,
             "min": _num(min(values), 3) if values else None, "max": _num(max(values), 3) if values else None}
    if q.kind == "trend":
        headline = (f"{q.label}: {counts['painted']} gauges painted"
                    + (f", median {stats['median']:+g} {q.unit}" if stats["median"] is not None else "")
                    + f". {field['verdict'] if field else ''}")
    else:
        headline = (f"{q.label}: {counts['painted']} gauges painted"
                    + (f", median {stats['median']:g}" if stats["median"] is not None else "") + ".")
    unread = counts["no_archive"] + counts["over_cap"] + counts["failed"]
    if unread or counts["too_short"]:
        headline += (f" Not painted: {counts['too_short']} too short, {counts['no_archive']} not in the Archive"
                     + (f", {counts['over_cap']} over the cap" if counts["over_cap"] else "")
                     + (f", {counts['failed']} unreadable" if counts["failed"] else "") + ".")
    features = []
    for r in rows:
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        props = {k: v for k, v in r.items() if k not in ("latitude", "longitude")}
        props["cls"] = class_of(question, r.get("value")) if r.get("status") == "painted" else None
        features.append({"type": "Feature", "geometry": {"type": "Point",
                                                          "coordinates": [r["longitude"], r["latitude"]]},
                         "properties": props})
    result: dict[str, Any] = {
        "question": q.public(), "params": {**_params(question, p), "alpha_global": alpha_global,
                                           "correlated": correlated},
        "bbox": [float(x) for x in bbox] if bbox is not None else None,
        "counts": counts, "stats": stats, "field": field, "headline": headline.strip(),
        "rows": rows, "licences": _licences(rows),
        "geojson": {"type": "FeatureCollection", "features": features},
        "table": {"columns": TABLE_COLUMNS, "rows": [[r.get(c) for c in TABLE_COLUMNS] for r in rows]},
    }
    result["methods"] = methods_paragraph(result)
    result["references"] = [REFERENCES[k] for k in (*q.references, "aquascope")]
    return result


def _box_words(bbox: list[float] | None) -> str:
    if not bbox:
        return "the gauges given"
    w, s, e, n = bbox

    def lat(v: float) -> str:
        return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"

    def lon(v: float) -> str:
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"

    return f"the box {lat(s)} to {lat(n)}, {lon(w)} to {lon(e)}"


def methods_paragraph(result: dict[str, Any]) -> str:
    """A methods paragraph for the painting, with author-year citations (the references are in ``references``)."""
    from aquascope import __version__

    q = result["question"]
    prm = result.get("params") or {}
    counts = result.get("counts") or {}
    n = counts.get("painted", 0)
    where = _box_words(result.get("bbox"))
    lic = "; ".join(f"{x['label']} ({x['license']})" for x in result.get("licences") or []) or "none painted"
    min_years = prm.get("min_years", MIN_YEARS)
    if q["kind"] == "trend":
        north, south = q["year_start"]["north"], q["year_start"]["south"]
        lats = [r.get("latitude") for r in result.get("rows") or [] if r.get("status") == "painted"]
        has_n = any(x is not None and x >= 0 for x in lats)
        has_s = any(x is not None and x < 0 for x in lats)
        if prm.get("year_start_month"):
            years = year_words(int(prm["year_start_month"]))
        elif has_n and has_s:
            years = f"{year_words(north)} north of the equator and {year_words(south)} south of it"
        else:
            years = year_words(south if has_s else north)
        series = "annual maximum daily mean discharge" if q["id"] == "flood_trend" else \
            "annual minimum 7-day mean discharge"
        window = ""
        if prm.get("start_year") or prm.get("end_year"):
            window = f", limited to the years {prm.get('start_year') or 'from the start'} to " \
                     f"{prm.get('end_year') or 'the latest'}"
        f = result.get("field") or {}
        a_g = f.get("alpha_global", ALPHA_GLOBAL)
        a_f = f.get("alpha_fdr", 2 * ALPHA_GLOBAL)
        rule = (f"at alpha_FDR = 2 alpha_global = {a_f:g}, which Wilks (2016) recommends for spatially correlated "
                "tests") if f.get("correlated", True) else f"at alpha_FDR = alpha_global = {a_f:g}"
        outcome = (f"{f.get('n_significant', 0)} remained significant "
                   f"({f.get('up', 0)} increasing, {f.get('down', 0)} decreasing), so the field is "
                   f"{'' if f.get('field_significant') else 'not '}significant at alpha_global = {a_g:g}")
        text = (
            f"We tested for monotonic trends in the {series} at every gauge in {where} that the AquaScope Archive "
            f"mirrors with at least {min_years} complete years of record{window}. Years run {years}, and a year is "
            f"complete when at least {prm.get('min_days', MIN_DAYS)} of its days have a daily mean. The trend was "
            "tested with the Mann-Kendall test (Mann, 1945; Kendall, 1975) and its size estimated with Sen's (1968) "
            "slope against the year, expressed in per cent of the series mean per decade. Field significance was "
            f"assessed with the Benjamini-Hochberg false discovery rate procedure (Benjamini and Hochberg, 1995) "
            f"{rule}. Of the {f.get('n_tested', 0)} gauges tested, {outcome}. "
        )
    else:
        text = (
            f"We mapped the baseflow index at every gauge in {where} that the AquaScope Archive mirrors with at "
            f"least {min_years} years of record ({n} gauges). The index is the share of the total flow left by the "
            "Lyne and Hollick (1979) digital filter (alpha 0.925, three passes) on the daily record, as the Archive "
            "computes it for signatures.parquet. "
        )
    skipped = []
    if counts.get("too_short"):
        skipped.append(f"{counts['too_short']} with shorter records")
    if counts.get("no_archive"):
        skipped.append(f"{counts['no_archive']} not mirrored by the Archive")
    if counts.get("over_cap"):
        skipped.append(f"{counts['over_cap']} beyond the cap of gauges read per view")
    if counts.get("failed"):
        skipped.append(f"{counts['failed']} whose records could not be read")
    if skipped:
        text += f"Gauges in the area left out: {', '.join(skipped)}. "
    text += (f"Daily discharge from the AquaScope Archive: {lic}. Computed with AquaScope {__version__} "
             "(https://doi.org/10.5281/zenodo.21903143).")
    return text


# ── the whole painting, for the CLI and the MCP tool ────────────────────────


def _archive_reader(source: str, station_id: str) -> pd.Series | None:
    from aquascope.archive.observations import fetch_archived_series

    return fetch_archived_series(source, station_id, "discharge")


def paint(
    question: str = "flood_trend",
    *,
    bbox: list[float] | tuple[float, float, float, float] | None = None,
    stations: list[dict[str, Any]] | None = None,
    signatures: Any = None,
    params: dict[str, Any] | None = None,
    max_gauges: int = MAX_GAUGES,
    reader: Callable[[str, str], pd.Series | None] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Paint one question over every gauge in ``bbox`` (west, south, east, north) or in ``stations``.

    ``stations`` defaults to the published catalog and ``signatures`` to the published ``signatures.parquet``.
    Records are read from the AquaScope Archive only (``reader(source, station_id)`` replaces the reader in
    tests), never live from an agency. ``params``: ``min_years`` (default :data:`MIN_YEARS`), ``min_days``,
    ``start_year``, ``end_year``, ``year_start_month``, ``alpha_global`` and ``correlated``. ``on_row`` gets
    each gauge's row as it is answered. Returns :func:`summarize`'s result.
    """
    q = _question(question)
    p = dict(params or {})
    if bbox is None and stations is None:
        raise ValueError("give a bbox [west, south, east, north] or stations")
    if bbox is not None:
        bbox = [float(x) for x in bbox]
        if len(bbox) != 4:
            raise ValueError("bbox is [west, south, east, north]")
    if stations is None:
        from aquascope.archive.catalog import load_stations

        stations = load_stations()
    if signatures is None:
        from aquascope.archive.signatures import load_signatures

        signatures = load_signatures()
    sig = _sig_rows(signatures)
    planned = plan(stations, sig, bbox=bbox, min_years=float(p.get("min_years") or MIN_YEARS),
                   max_gauges=max_gauges)
    reader = reader or _archive_reader
    done: dict[str, dict[str, Any]] = {}
    for r in planned["read"]:
        item = {k: r[k] for k in ("key", "source", "station_id", "name", "latitude", "longitude", "record_years")}
        if q.needs == "daily":
            try:
                series = reader(r["source"], r["station_id"])
            except Exception as exc:  # noqa: BLE001 - one unreadable record is a failed gauge, not a failed map
                logger.info("archive read failed for %s: %s", r["key"], exc)
                series = None
            item["series"] = series if series is not None else pd.Series(dtype=float)
        else:
            item["signature"] = sig.get(r["key"], {})
        row = gauge_rows(question, [item], p)[0]
        done[r["key"]] = row
        if on_row:
            on_row(row)
    rows = [done.get(r["key"], r) for r in planned["rows"]]
    return summarize(question, rows, params=p, bbox=bbox)


def to_csv(result: dict[str, Any]) -> str:
    """The per-gauge table as CSV."""
    table = result.get("table") or {}
    buf = io.StringIO()
    pd.DataFrame(table.get("rows") or [], columns=table.get("columns") or TABLE_COLUMNS).to_csv(buf, index=False)
    return buf.getvalue()


def methods_text(result: dict[str, Any]) -> str:
    """The methods paragraph followed by its references, as plain text (the Explorer's "Methods" download)."""
    q = result.get("question") or {}
    lines = [f"Paint the globe: {q.get('label', '')}", "", result.get("methods") or methods_paragraph(result), "",
             "References", ""]
    lines += [f"- {r}" for r in result.get("references") or []]
    if result.get("field"):
        lines += ["", "Caveat", "", result["field"]["caveat"]]
    return "\n".join(lines) + "\n"


__all__ = [
    "ALPHA_GLOBAL", "MAX_GAUGES", "MIN_DAYS", "MIN_YEARS", "PLANNED", "QUESTIONS", "REFERENCES", "STATUSES",
    "TABLE_COLUMNS", "Question", "annual_series", "annual_trend", "answer_gauge", "baseflow_index", "class_of",
    "field_significance", "flood_trend", "gauge_rows", "low_flow_trend", "methods_paragraph", "methods_text", "paint",
    "plan", "questions", "sen_slope_years", "start_month_for", "summarize", "to_csv", "year_labels", "year_words",
]
