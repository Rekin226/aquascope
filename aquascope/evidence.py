"""The evidence ladder: every global model scored against a gauge's own record, graded, and said in words (#518).

A gauge is the measurement; a global model is a guess about the same river. :func:`model_skill` puts them side
by side over the years they share and scores each model the way hydrologists do:

* KGE (Kling-Gupta efficiency) and its three parts: correlation ``r``, variability ratio ``alpha`` and bias ratio
  ``beta``; NSE; percent bias (PBIAS);
* the relative error of the 2-, 10- and 100-year flows, each read from its own annual-maxima fit (GEV by
  L-moments) over the same years, so the flood a model would design for is compared with the gauge's;
* a grade, A to D, from KGE, demoted one letter when the 100-year flow is off by more than half (see
  :data:`GRADING`);
* one plain sentence that says where the models disagree ("GEOGLOWS's 100-year flow is 38 % above the gauge's").

The models, and where each is scored:

* **GEOGLOWS v2** (CC BY 4.0): the river reach near the gauge whose upstream area matches the gauge's catchment
  (:func:`geoglows_site`), its daily record from the GEOGLOWS API. Scored live, in the browser too.
* **GloFAS v4** via Open-Meteo (CC BY 4.0): the 0.05 degree cell around the gauge whose mean flow is closest to
  the gauge's (:func:`aquascope.explore.snap_glofas_cell`). Scored live. Picking the cell by mean flow flatters
  its bias, so read its ``r`` and ``alpha`` before its ``beta``.
* **NWM v3.0 retrospective** (US only, public domain) and **Google GRRR** (the Flood Hub model's reanalysis,
  CC BY 4.0): read in CI only (the NWM store touches hundreds of chunks per reach, and Google's bucket sends no
  CORS headers), scored there by :mod:`aquascope.archive.skill`, and published with the Archive under
  ``skill/model_skill.parquet``. :func:`model_skill` merges those rows for the gauge.

Everything here is plain Python over numpy and pandas, so the Explorer's worker runs it unchanged.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GRADING",
    "MODELS",
    "disagreement",
    "geoglows_site",
    "grade",
    "lean_on",
    "model_skill",
    "published_rows",
    "score",
]

REPO_ID = "Rekin226/aquascope-gauges"
SKILL_FOLDER = "skill"
SKILL_TABLE = "model_skill.parquet"
SKILL_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{SKILL_FOLDER}/{SKILL_TABLE}"

#: The models on the ladder: a label, the licence and attribution the page shows, and where each is scored.
MODELS: dict[str, dict[str, Any]] = {
    "geoglows": {
        "label": "GEOGLOWS v2", "licence": "CC BY 4.0", "where": "live",
        "attribution": "GEOGLOWS v2, the GEOGloWS ECMWF Streamflow Service (CC BY 4.0)",
        "site": "river reach",
    },
    "glofas": {
        "label": "GloFAS v4", "licence": "CC BY 4.0", "where": "live",
        "attribution": "GloFAS (Copernicus Emergency Management Service) via Open-Meteo.com (CC BY 4.0)",
        "site": "0.05 degree grid cell",
    },
    "nwm": {
        "label": "NWM v3.0", "licence": "public domain (NOAA)", "where": "published",
        "attribution": "NOAA National Water Model v3.0 retrospective (AWS Open Data, public domain)",
        "site": "NHDPlus reach (US only)",
    },
    "grrr": {
        "label": "Google GRRR", "licence": "CC BY 4.0", "where": "published",
        "attribution": "Google Research, Flood Hub hydrologic reanalysis (GRRR, model 8583a5c2), CC BY 4.0",
        "site": "HydroBASINS level-12 outlet",
    },
}

#: The grade, from KGE (Gupta et al. 2009), and the flood check that can lower it. Knoben et al. (2019) showed
#: that KGE = 1 - sqrt(2), about -0.41, is what the mean flow itself scores, so a model at or below it does no
#: better than a flat line at the gauge's mean: grade D. The A and B bands (0.75 and 0.5) are AquaScope's choice
#: of round thresholds, not a published standard; they are fixed here so a grade means the same at every gauge.
GRADING: dict[str, Any] = {
    "metric": "KGE (2009) over the daily overlap",
    "bands": [
        {"grade": "A", "min_kge": 0.75, "words": "tracks the gauge closely"},
        {"grade": "B", "min_kge": 0.5, "words": "usable with care"},
        {"grade": "C", "min_kge": -0.41, "words": "better than the gauge's mean flow, but not by much"},
        {"grade": "D", "min_kge": None, "words": "no better than the gauge's mean flow"},
    ],
    "flood_demotion_pct": 50.0,
    "flood_demotion": "one letter lower (to D at most) when the 100-year flow (the 10-year when there is no "
                      "100-year comparison) differs from the gauge's by more than 50 %",
    "min_overlap_days": 3 * 365,
    "min_flood_years": 10,
    "area_ratio_ok": [0.5, 2.0],
    "not_graded": "fewer than three years of shared daily data, or a model site whose upstream area is off the "
                  "gauge's by more than a factor of two",
    "references": [
        "Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error "
        "and NSE performance criteria. J. Hydrol. 377, 80-91.",
        "Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? "
        "Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores. Hydrol. Earth Syst. Sci. 23, 4323-4331.",
        "Hosking, J. R. M. (1990). L-moments. J. R. Stat. Soc. B 52, 105-124.",
    ],
}

RETURN_PERIODS = (2, 10, 100)
DEFAULT_YEARS = 30
GEOGLOWS_SEARCH_M = 2000.0
GEOGLOWS_CANDIDATES = 5
GLOFAS_START = "1984-01-01"

METHOD = {
    "name": "Model skill at the gauge (the evidence ladder)",
    "text": "Each global model's daily discharge set against the gauge's over the days both have: KGE with its "
    "correlation, variability and bias ratios, NSE and percent bias; the 2-, 10- and 100-year flows of each from its "
    "own GEV (L-moments) fit to the same years' annual maxima, and their relative error. Grade A (KGE >= 0.75), "
    "B (>= 0.5), C (> -0.41, the mean-flow benchmark), else D; one letter lower when the 100-year flow is off by "
    "more than 50 %.",
    "citation": "Gupta, H. V. et al. (2009). J. Hydrol. 377, 80-91; Knoben, W. J. M. et al. (2019). Hydrol. Earth "
    "Syst. Sci. 23, 4323-4331.",
}


# ── small helpers ────────────────────────────────────────────────────────────


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _daily(s: Any) -> Any:
    """A clean daily-mean series: a DatetimeIndex without time zone, finite non-negative values."""
    import pandas as pd

    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(pd.Series(s.values), errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(pd.to_datetime(s.index)))
    if s.index.tz is not None:
        s.index = s.index.tz_convert(None)
    s = s[(s >= 0) & s.notna()]
    if not len(s):
        return s
    return s.resample("D").mean().dropna()


def _is_cms(unit: str) -> bool:
    """True for the spellings of cubic metres per second the collectors use (m3/s, m³/s, cms, ...)."""
    u = str(unit or "").lower().replace(" ", "")
    return u in {"m3/s", "m³/s", "m^3/s", "m3s-1", "m3.s-1", "cms", "cumecs"}


def _label(model: str) -> str:
    return str(MODELS.get(model, {}).get("label") or model)


# ── scoring one model ───────────────────────────────────────────────────────


def grade(kge: float | None, flood_error_pct: float | None = None) -> dict[str, Any]:
    """The letter for a KGE, lowered one step when the flood error is beyond :data:`GRADING`'s limit.

    Returns ``{"grade", "why"}``; ``grade`` is None when there is no KGE to grade.
    """
    if kge is None or not math.isfinite(float(kge)):
        return {"grade": None, "why": "not graded: no KGE"}
    letter = "D"
    for band in GRADING["bands"]:
        if band["min_kge"] is None or float(kge) >= band["min_kge"]:
            letter = band["grade"]
            break
    if letter == "C" and float(kge) <= -0.41:
        letter = "D"
    why = f"KGE {float(kge):.2f}"
    limit = GRADING["flood_demotion_pct"]
    if flood_error_pct is not None and math.isfinite(float(flood_error_pct)) and abs(float(flood_error_pct)) > limit:
        lowered = {"A": "B", "B": "C", "C": "D", "D": "D"}[letter]
        if lowered != letter:
            why += f", lowered from {letter} because the flood flow is off by {abs(float(flood_error_pct)):.0f} %"
        letter = lowered
    return {"grade": letter, "why": why}


def _annual_max(s: Any) -> Any:
    from aquascope.explore import _annual_max

    return _annual_max(s)


def flood_errors(obs: Any, sim: Any, return_periods: tuple[int, ...] = RETURN_PERIODS) -> dict[str, Any]:
    """The T-year flows of the gauge and the model, each from its own GEV (L-moments) fit to the annual maxima of
    the same well-covered years, and the model's relative error at each T (in %)."""
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments

    am_o, am_s = _annual_max(obs), _annual_max(sim)
    years = sorted(set(am_o.index.year) & set(am_s.index.year))
    out: dict[str, Any] = {"n_years": len(years), "return_periods": list(return_periods),
                           "gauge": {}, "model": {}, "error_pct": {}}
    if len(years) < GRADING["min_flood_years"]:
        out["note"] = (f"flood flows need {GRADING['min_flood_years']} shared years of annual maxima; "
                       f"there are {len(years)}")
        return out
    o = am_o[am_o.index.year.isin(years)].to_numpy(dtype=float)
    m = am_s[am_s.index.year.isin(years)].to_numpy(dtype=float)
    try:
        fo = fit_gev_lmoments(o, return_periods=list(return_periods))
        fm = fit_gev_lmoments(m, return_periods=list(return_periods))
    except Exception as exc:  # noqa: BLE001 - a degenerate sample (all zeros) has no fit
        out["note"] = f"no flood fit: {exc}"
        return out
    for t in return_periods:
        qo, qm = float(fo.return_periods[t]), float(fm.return_periods[t])
        key = f"{t}"
        out["gauge"][key] = _num(qo)
        out["model"][key] = _num(qm)
        out["error_pct"][key] = _num(100.0 * (qm - qo) / qo, 1) if qo > 0 and math.isfinite(qm) else None
    return out


def score(obs: Any, sim: Any, *, model: str, site: dict[str, Any] | None = None) -> dict[str, Any]:
    """One row of the skill table: ``model`` against the gauge over the days both have.

    ``obs`` and ``sim`` are pandas Series of daily discharge (m3/s) with a DatetimeIndex. ``site`` describes where
    the model was read (``site_id``, ``match``, ``area_ratio``, ...) and travels with the row. A site whose
    ``area_ratio`` is outside :data:`GRADING`'s band is scored but not graded: a model of a different river is not
    evidence about this one.
    """
    import numpy as np

    from aquascope.analysis.metrics import kge, nse, pbias

    o, m = _daily(obs), _daily(sim)
    both = o.index.intersection(m.index)
    o, m = o.reindex(both), m.reindex(both)
    row: dict[str, Any] = {"model": model, "label": _label(model), "n_days": int(len(both)),
                           "start": both.min().strftime("%Y-%m-%d") if len(both) else None,
                           "end": both.max().strftime("%Y-%m-%d") if len(both) else None,
                           "licence": MODELS.get(model, {}).get("licence"),
                           **{k: v for k, v in (site or {}).items() if k not in ("model",)}}
    if len(both) < 2:
        row.update(grade=None, why="no shared days with the gauge")
        return row
    ov, mv = o.to_numpy(dtype=float), m.to_numpy(dtype=float)
    mo, mm, so, sm = float(ov.mean()), float(mv.mean()), float(ov.std()), float(mv.std())
    r = float(np.corrcoef(ov, mv)[0, 1]) if so > 0 and sm > 0 else float("nan")
    row.update(kge=_num(kge(ov, mv)), r=_num(r), alpha=_num(sm / so if so > 0 else float("nan")),
               beta=_num(mm / mo if mo > 0 else float("nan")), nse=_num(nse(ov, mv)), pbias=_num(pbias(ov, mv), 2),
               mean_gauge=_num(mo), mean_model=_num(mm))
    fl = flood_errors(o, m)
    row["flood_years"] = fl["n_years"]
    for t in RETURN_PERIODS:
        row[f"q{t}_gauge"] = fl["gauge"].get(f"{t}")
        row[f"q{t}_model"] = fl["model"].get(f"{t}")
        row[f"q{t}_error_pct"] = fl["error_pct"].get(f"{t}")
    if fl.get("note"):
        row["flood_note"] = fl["note"]
    lo, hi = GRADING["area_ratio_ok"]
    ratio = row.get("area_ratio")
    if len(both) < GRADING["min_overlap_days"]:
        row.update(grade=None, why=f"not graded: {len(both)} shared days, fewer than three years")
    elif ratio is not None and not (lo <= float(ratio) <= hi):
        row.update(grade=None, why=f"not graded: the model's upstream area is {float(ratio):.2f} times the "
                                   "gauge's, so it is another river")
    else:
        fe = row.get("q100_error_pct") if row.get("q100_error_pct") is not None else row.get("q10_error_pct")
        row.update(grade(row.get("kge"), fe))
    row["sentence"] = disagreement(row)
    return row


# ── words ────────────────────────────────────────────────────────────────────


def _possessive(label: str) -> str:
    return f"{label}'" if label.endswith("s") else f"{label}'s"


def disagreement(row: dict[str, Any]) -> str:
    """One sentence on how a model differs from the gauge: the 100-year flow when there is one (else the 10- or
    2-year), else the mean flow."""
    who = _possessive(str(row.get("label") or _label(str(row.get("model")))))
    for t in (100, 10, 2):
        e = row.get(f"q{t}_error_pct")
        if e is not None:
            if abs(e) < 5:
                return f"{who} {t}-year flow is within {abs(e):.0f} % of the gauge's."
            return f"{who} {t}-year flow is {abs(e):.0f} % {'above' if e > 0 else 'below'} the gauge's."
    pb = row.get("pbias")
    if pb is not None:
        if abs(pb) < 5:
            return f"{who} mean flow is within {abs(pb):.0f} % of the gauge's."
        return f"{who} mean flow is {abs(pb):.0f} % {'above' if pb > 0 else 'below'} the gauge's."
    return f"{who} record does not overlap the gauge's."


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The best graded model and the sentence that says it, plus the largest flood disagreement."""
    graded = [r for r in rows if r.get("grade") and r.get("kge") is not None]
    out: dict[str, Any] = {"best": None, "best_grade": None, "sentence": None, "disagreement": None}
    if graded:
        best = max(graded, key=lambda r: float(r["kge"]))
        out.update(best=best["model"], best_grade=best["grade"], best_kge=best["kge"])
        if len(graded) == 1:
            sentence = (f"{best['label']} is the only model graded here (KGE {float(best['kge']):.2f}, "
                        f"grade {best['grade']}).")
        else:
            sentence = f"{best['label']} fits this gauge best (KGE {float(best['kge']):.2f}, grade {best['grade']})."
    else:
        sentence = "No model could be graded at this gauge."
    worst = None
    for t in (100, 10, 2):
        cands = [r for r in rows if r.get(f"q{t}_error_pct") is not None]
        if cands:
            worst = max(cands, key=lambda r: abs(float(r[f"q{t}_error_pct"])))
            break
    if worst is not None:
        out["disagreement"] = disagreement(worst)
        sentence = f"{sentence} {out['disagreement']}"
    out["sentence"] = sentence
    return out


# ── where each model is read ────────────────────────────────────────────────


def geoglows_site(lat: float, lon: float, area_km2: float | None = None, *,
                  max_distance_m: float = GEOGLOWS_SEARCH_M, candidates: int = GEOGLOWS_CANDIDATES) -> dict[str, Any]:
    """The GEOGLOWS v2 reach that stands for the gauge.

    With the gauge's catchment area, every reach within ``max_distance_m`` (up to ``candidates``) has its upstream
    area summed and the one whose area is closest to the gauge's (on a log scale) wins
    (:func:`aquascope.rivers.match_by_area`, which a gauge's snap uses too): a main-stem gauge then gets the main
    stem, not the tributary that happens to pass nearer. Without an area the nearest reach is taken and the match
    is recorded as ``nearest``.

    Returns ``{"site_id", "distance_m", "model_area_km2", "area_ratio", "match", "n_candidates"}`` or
    ``{"error": ...}``.
    """
    from aquascope import rivers

    near = rivers.reaches_near(lat, lon, max_distance_m=max_distance_m, limit=candidates)
    if not near:
        return {"error": f"no GEOGLOWS reach within {max_distance_m:,.0f} m of the gauge"}
    if not area_km2 or not float(area_km2) > 0:
        r = near[0]
        return {"site_id": str(r["river_id"]), "distance_m": r["distance_m"], "model_area_km2": None,
                "area_ratio": None, "match": "nearest", "n_candidates": len(near)}
    matched = rivers.match_by_area(near, area_km2)
    if matched is None:
        r = near[0]
        return {"site_id": str(r["river_id"]), "distance_m": r["distance_m"], "model_area_km2": None,
                "area_ratio": None, "match": "nearest", "n_candidates": len(near)}
    r = matched["reach"]
    return {"site_id": str(r["river_id"]), "distance_m": r["distance_m"],
            "model_area_km2": matched["upstream_area_km2"], "area_ratio": matched["area_ratio"], "match": "area",
            "n_candidates": len(near)}


def geoglows_series(river_id: Any) -> Any:
    """The reach's simulated daily discharge since 1940 (GEOGLOWS API, the same read as the River tab)."""
    from aquascope import rivers

    store: dict[str, Any] = {}
    res = rivers.reach_record(river_id, store=store)
    if res.get("error"):
        raise RuntimeError(res["error"])
    return store.get("series")


def glofas_site(lat: float, lon: float, gauge_mean: float, area_km2: float | None = None) -> dict[str, Any]:
    """The GloFAS cell around the gauge whose mean flow is closest to the gauge's (a 3 x 3 window, one probe year:
    light on Open-Meteo's free quota, which counts a long request as many calls)."""
    from aquascope.explore import snap_glofas_cell

    cell = snap_glofas_cell(lat, lon, gauge_mean, window=1, probe_years=1, area_km2=area_km2)
    if cell.get("lat") is None:
        return {"error": cell.get("why") or "no GloFAS cell"}
    return {"site_id": f"{cell['lat']:.3f},{cell['lon']:.3f}", "site_lat": cell["lat"], "site_lon": cell["lon"],
            "distance_m": _num(float(cell.get("offset_km") or 0.0) * 1000.0, 0), "area_ratio": None,
            "match": "mean_flow", "mean_flow_ratio": cell.get("ratio")}


def glofas_series(lat: float, lon: float, start: str, end: str) -> Any:
    """GloFAS daily discharge for one cell (Open-Meteo flood API)."""
    import pandas as pd

    from aquascope.registry import build_collector

    flood = build_collector("openmeteo", mode="flood")
    raw = flood.fetch_raw(latitude=lat, longitude=lon, start_date=start, end_date=end, daily=["river_discharge"])
    daily = (raw or {}).get("daily") or {}
    return pd.Series(pd.to_numeric(pd.Series(daily.get("river_discharge") or []), errors="coerce").to_numpy(float),
                     index=pd.to_datetime(daily.get("time") or []))


# ── the published table (NWM and GRRR, and the map's colouring) ─────────────


def _published_frame(refresh: bool = False) -> Any:
    """The published skill table as a DataFrame (needs pyarrow), or None when it is not published or not readable."""
    import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    from aquascope.archive.catalog import _download, cache_dir

    try:
        path = _download(SKILL_URL, cache_dir() / "model_skill.parquet", refresh)
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - not published yet, or offline: the live models still score
        logger.info("published skill table unavailable: %s", exc)
        return None


def published_rows(source: str, station_id: str, *, frame: Any = None) -> list[dict[str, Any]]:
    """The published rows for one gauge (every model CI scored there), as plain dicts."""
    df = _published_frame() if frame is None else frame
    if df is None or not len(df):
        return []
    sel = df[(df["source"] == source) & (df["station_id"].astype(str) == str(station_id))]
    return [_clean_row(r) for r in sel.to_dict("records")]


def _clean_row(r: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in r.items():
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


# ── the ladder for one gauge ────────────────────────────────────────────────


def _station_row(source: str, station_id: str) -> dict[str, Any] | None:
    from aquascope.archive.catalog import load_stations

    try:
        rows = load_stations()
    except Exception as exc:  # noqa: BLE001 - the caller may pass lat/lon itself
        logger.info("catalog unavailable: %s", exc)
        return None
    return next((r for r in rows if r["source"] == source and str(r["station_id"]) == str(station_id)), None)


def _catchment_area(source: str, station_id: str) -> float | None:
    """The gauge's catchment area from the Archive's station catchments (the agency's figure, else BasinATLAS's
    upstream area); None without pyarrow or the table."""
    try:
        import pyarrow  # noqa: F401

        from aquascope.archive.similar import load_station_catchments

        df = load_station_catchments()
    except Exception as exc:  # noqa: BLE001 - the area only sharpens the reach choice
        logger.info("station catchments unavailable: %s", exc)
        return None
    hit = df[(df["source"] == source) & (df["station_id"].astype(str) == str(station_id))]
    if not len(hit):
        return None
    v = _num(hit.iloc[0].get("area_km2"), 1)
    return v if v and v > 0 else None


def _thin(s: Any, cap: int = 4000) -> dict[str, list[Any]]:
    """A plotting copy: at most ``cap`` points (every k-th day), dates as ISO strings."""
    if s is None or not len(s):
        return {"t": [], "v": []}
    step = max(1, int(math.ceil(len(s) / cap)))
    t = s.iloc[::step]
    return {"t": [d.strftime("%Y-%m-%d") for d in t.index], "v": [_num(v, 3) for v in t.to_numpy(dtype=float)]}


def model_skill(
    source: str | None = None,
    station_id: str | None = None,
    *,
    series: Any = None,
    lat: float | None = None,
    lon: float | None = None,
    area_km2: float | None = None,
    models: list[str] | tuple[str, ...] | None = None,
    published: list[dict[str, Any]] | None = None,
    years: int | None = DEFAULT_YEARS,
    include_series: bool = False,
) -> dict[str, Any]:
    """Score every global model at a gauge: the evidence ladder for one station.

    Give a station (``source`` and ``station_id``: its record, position and catchment area come from the
    Archive), or an observed daily discharge ``series`` (m3/s) with ``lat``/``lon`` (and ``area_km2`` when known).
    ``years`` keeps the last N years of the gauge record (default 30; None for all of it); every model is scored
    over the part of that window it covers. ``models`` limits the ladder (default all four). GEOGLOWS and GloFAS
    are read live; NWM and GRRR come from ``published`` (rows of ``skill/model_skill.parquet``, as the Explorer
    passes them) or from the published table when pyarrow can read it.

    Returns ``{"models": [rows], "best", "best_grade", "sentence", "disagreement", "grading", ...}``; each row
    has KGE, r, alpha, beta, NSE, PBIAS, the 2-, 10- and 100-year flows of gauge and model with the error in %,
    the grade and why, the model site and how it was matched, and a one-line disagreement sentence.
    ``include_series`` adds the plotting copies of the observed and modelled records under ``series``.
    """
    import pandas as pd

    wanted = [m for m in (models or list(MODELS)) if m in MODELS]
    notes: list[str] = []
    row = None
    if source and station_id and (lat is None or lon is None):
        row = _station_row(source, station_id)
        if row is not None:
            lat = row["latitude"] if lat is None else lat
            lon = row["longitude"] if lon is None else lon
            if area_km2 is None:
                a = (row.get("extra") or {}).get("catchment_area_km2")
                try:
                    area_km2 = float(a) if a is not None and float(a) > 0 else None
                except (TypeError, ValueError):
                    area_km2 = None
    if area_km2 is None and source and station_id:
        area_km2 = _catchment_area(source, station_id)
    if series is None:
        if not (source and station_id):
            raise ValueError("give a station (source and station_id) or an observed series with lat and lon")
        from aquascope.explore import fetch_series

        fetched = fetch_series(source, station_id, variable="discharge")
        series = fetched.get("series")
        unit = str(fetched.get("unit") or "")
        if fetched.get("variable") != "discharge":
            series = None
        elif unit and not _is_cms(unit):
            # every model speaks m3/s; a record in another unit would score as a huge bias, so it is not scored
            series = None
            notes.append(f"The gauge's discharge is in {unit}, not m3/s, so the models are not scored against it.")
    obs = _daily(series)
    if lat is None or lon is None:
        raise ValueError("the gauge's position is unknown: pass lat and lon")
    base: dict[str, Any] = {
        "source": source, "station_id": station_id, "lat": _num(lat, 5), "lon": _num(lon, 5),
        "area_km2": _num(area_km2, 1), "grading": GRADING, "methods": [METHOD],
        "attribution": "; ".join(MODELS[m]["attribution"] for m in wanted),
        "computed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if len(obs) < 2:
        return {**base, "models": [], "error": "no daily discharge record in m3/s at this gauge to score models "
                "against", **summarize([]), "notes": notes}
    if years:
        obs = obs[obs.index >= obs.index.max() - pd.Timedelta(days=int(float(years) * 365.25))]
    base.update(obs_start=obs.index.min().strftime("%Y-%m-%d"), obs_end=obs.index.max().strftime("%Y-%m-%d"),
                obs_days=int(len(obs)))
    if area_km2 is None:
        notes.append("No catchment area for this gauge, so the GEOGLOWS reach is the nearest one, not the one "
                     "matched by upstream area.")
    rows: list[dict[str, Any]] = []
    plot: dict[str, Any] = {"observed": _thin(obs)} if include_series else {}
    start, end = obs.index.min().strftime("%Y-%m-%d"), obs.index.max().strftime("%Y-%m-%d")

    if "geoglows" in wanted:
        try:
            site = geoglows_site(float(lat), float(lon), area_km2)
            if site.get("error"):
                rows.append({"model": "geoglows", "label": _label("geoglows"), "grade": None, "why": site["error"]})
            else:
                sim = geoglows_series(site["site_id"])
                sim = _daily(sim)
                sim = sim[(sim.index >= obs.index.min()) & (sim.index <= obs.index.max())]
                rows.append(score(obs, sim, model="geoglows", site=site))
                if include_series:
                    plot["geoglows"] = _thin(sim)
        except Exception as exc:  # noqa: BLE001 - one model failing leaves the rest of the ladder standing
            rows.append({"model": "geoglows", "label": _label("geoglows"), "grade": None,
                         "why": f"GEOGLOWS unavailable: {exc}"})
    if "glofas" in wanted:
        g_start = max(start, GLOFAS_START)
        if g_start > end:
            rows.append({"model": "glofas", "label": _label("glofas"), "grade": None,
                         "why": "the gauge record ends before GloFAS begins (1984)"})
        else:
            try:
                site = glofas_site(float(lat), float(lon), float(obs.mean()), area_km2)
                if site.get("error"):
                    rows.append({"model": "glofas", "label": _label("glofas"), "grade": None, "why": site["error"]})
                else:
                    sim = _daily(glofas_series(site["site_lat"], site["site_lon"], g_start, end))
                    row_g = score(obs, sim, model="glofas", site={k: v for k, v in site.items()
                                                                  if k not in ("site_lat", "site_lon")})
                    row_g["note"] = "cell picked by mean flow, which flatters its bias: read r and alpha first"
                    rows.append(row_g)
                    if include_series:
                        plot["glofas"] = _thin(sim)
            except Exception as exc:  # noqa: BLE001
                rows.append({"model": "glofas", "label": _label("glofas"), "grade": None,
                             "why": f"GloFAS unavailable: {exc}"})
    pub = published
    if pub is None and source and station_id and any(m in wanted for m in ("nwm", "grrr")):
        pub = published_rows(source, station_id)
    for m in ("nwm", "grrr"):
        if m not in wanted:
            continue
        hit = next((dict(r) for r in (pub or []) if r.get("model") == m), None)
        if hit is not None:
            hit["label"] = _label(m)
            hit["from_table"] = True
            hit.setdefault("sentence", disagreement(hit))
            rows.append(hit)
    if not any(r.get("model") in ("nwm", "grrr") for r in rows) and any(m in wanted for m in ("nwm", "grrr")):
        notes.append("NWM and Google GRRR are scored by the monthly skill run and published with the Archive; "
                     "this gauge has no published row yet.")
    out = {**base, "models": rows, **summarize(rows), "notes": notes}
    if include_series:
        out["series"] = plot
    return out


# ── which model to lean on near a site (the Studio's input) ─────────────────


def _haversine_km(lat1: float, lon1: float, lat2: Any, lon2: Any) -> Any:
    import numpy as np

    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def lean_on(lat: float, lon: float, *, radius_km: float = 150.0, k: int = 8, frame: Any = None) -> dict[str, Any]:
    """Which global model to lean on near a site, from the published skill at the nearest graded gauges.

    For an ungauged site the question is not which model is best in general but which one tracked the gauges
    around it. This takes up to ``k`` gauges within ``radius_km`` that the monthly skill run graded, the median
    KGE of each model over them, and names the model with the highest median (at least two gauges). Returns
    ``{"model", "label", "median_kge", "n_gauges", "by_model", "sentence"}``, or ``{"model": None, "sentence"}``
    when there is no published skill nearby.
    """
    import numpy as np

    df = _published_frame() if frame is None else frame
    if df is None or not len(df) or "lat" not in df.columns:
        return {"model": None, "sentence": "No published model skill is available yet to say which model to lean "
                                           "on here."}
    g = df[df["grade"].notna()].copy()
    if not len(g):
        return {"model": None, "sentence": "No graded gauge in the published skill table."}
    g["km"] = _haversine_km(float(lat), float(lon), g["lat"].to_numpy(float), g["lon"].to_numpy(float))
    g = g[g["km"] <= float(radius_km)]
    gauges = g.drop_duplicates(["source", "station_id"]).nsmallest(int(k), "km")[["source", "station_id"]]
    g = g.merge(gauges, on=["source", "station_id"])
    if not len(g):
        return {"model": None, "radius_km": radius_km,
                "sentence": f"No gauge within {radius_km:,.0f} km has a graded model, so there is no local evidence "
                            "for one model over another."}
    by: dict[str, dict[str, Any]] = {}
    for m, part in g.groupby("model"):
        by[str(m)] = {"label": _label(str(m)), "median_kge": _num(float(np.median(part["kge"].astype(float))), 3),
                      "n_gauges": int(part[["source", "station_id"]].drop_duplicates().shape[0]),
                      "max_km": _num(float(part["km"].max()), 0)}
    ok = {m: v for m, v in by.items() if v["n_gauges"] >= 2 and v["median_kge"] is not None}
    if not ok:
        return {"model": None, "by_model": by, "radius_km": radius_km,
                "sentence": "Fewer than two graded gauges per model nearby, so no model is preferred here."}
    best = max(ok, key=lambda m: ok[m]["median_kge"])
    b = ok[best]
    others = [f"{v['label']} {v['median_kge']:.2f} at {v['n_gauges']}" for m, v in ok.items() if m != best]
    sentence = (f"Near this site, {b['label']} tracked the gauges best: median KGE {b['median_kge']:.2f} at "
                f"{b['n_gauges']} gauges within {b['max_km']:,.0f} km"
                + (f" (against {'; '.join(others)})" if others else "")
                + f". Lean on {b['label']}'s numbers here and read the others as a second opinion.")
    return {"model": best, "label": b["label"], "median_kge": b["median_kge"], "n_gauges": b["n_gauges"],
            "by_model": by, "radius_km": radius_km, "sentence": sentence}
