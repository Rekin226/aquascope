"""The forecast track record at a gauge or reach, read from the published verification (#785).

:mod:`aquascope.archive.verify` scores the Archive's issued forecasts against what the gauge then recorded and
publishes ``verify/forecast_skill.parquet`` (every gauge pooled, per model, variant, lead day and month) and
``verify/forecast_skill_gauges.parquet`` (each gauge's own record over the newest 30 days, per model, variant and
lead window). This module is the one place that reads those rows back and says them:

* :func:`track_lines`: the river card's lines ("GEOGLOWS, corrected: within a factor of 1.8 (84 pairs, 18 Oct to
  16 Nov)"); the Explorer's ``track-record-core.js`` says the same words and a test holds them together;
* :func:`rank_models`: the models at a gauge ranked by their verified factor, where enough pairs exist (the
  evidence ladder);
* :func:`forecast_skill`: the scorecard for a station or a point, which the MCP tool of the same name returns.

Nothing here computes a score: every number comes from the published tables. Plain Python over pandas, so the
Explorer's worker can import it; reading the tables needs pyarrow and is skipped without it.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

from aquascope.archive.verify import (
    ALL_MONTHS,
    DEFAULT_REPO,
    FOLDER,
    GAUGE_TABLE,
    MIN_PAIRS,
    TABLE,
    TRACK_DAYS,
    TRACK_LEADS,
    TRUST_BANDS,
    UNKNOWN,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LABELS",
    "forecast_skill",
    "gauge_rows",
    "rank_models",
    "track_head",
    "track_line",
    "track_lines",
]

#: How each model is named on the card.
LABELS = {"geoglows": "GEOGLOWS", "glofas": "GloFAS"}
MODEL_ORDER = ("geoglows", "glofas")
GAUGES_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO}/resolve/main/{FOLDER}/{GAUGE_TABLE}"
TABLE_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO}/resolve/main/{FOLDER}/{TABLE}"
#: How far a lat/lon may be from a forecast gauge for :func:`forecast_skill` to answer with that gauge.
NEAR_KM = 25.0
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── saying a row ────────────────────────────────────────────────────────────


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _one(x: float) -> str:
    """One decimal, rounded half up as the Explorer's ``oneDecimal`` does, so the two never disagree on a tie."""
    return f"{math.floor(float(x) * 10 + 0.5) / 10:.1f}"


def _short_day(iso: Any) -> str:
    """'18 Oct' for '2026-10-18'; '' when it is not a date."""
    try:
        d = date.fromisoformat(str(iso)[:10])
    except ValueError:
        return ""
    return f"{d.day} {_MONTHS[d.month - 1]}"


def _label(row: dict[str, Any]) -> str:
    name = LABELS.get(str(row.get("model")), str(row.get("model") or "model"))
    return f"{name}, corrected" if row.get("variant") == "corrected" else name


def _leads(lo: Any, hi: Any) -> str:
    lo, hi = int(lo), int(hi)
    if lo == hi:
        return "the same day" if lo == 0 else f"{lo} day{'' if lo == 1 else 's'} ahead"
    return f"{lo} to {hi} days ahead"


def track_head(window_days: int = TRACK_DAYS, leads: tuple[int, int] = TRACK_LEADS) -> str:
    """The line above the card's rows: 'Forecast track record here, last 30 days, 1 to 3 days ahead:'."""
    return f"Forecast track record here, last {int(window_days)} days, {_leads(*leads)}:"


def track_line(row: dict[str, Any], *, min_pairs: int = MIN_PAIRS) -> str:
    """One model's line: 'GEOGLOWS, corrected: within a factor of 1.8 (84 pairs, 18 Oct to 16 Nov).', or
    'GloFAS: not enough pairs yet (12 of 30).' when ``enough_pairs`` is false."""
    n = int(_num(row.get("n_pairs")) or 0)
    factor = _num(row.get("factor"))
    if not row.get("enough_pairs") or factor is None:
        return f"{_label(row)}: not enough pairs yet ({n} of {int(min_pairs)})."
    span = f", {_short_day(row.get('first_valid'))} to {_short_day(row.get('last_valid'))}" \
        if row.get("first_valid") and row.get("last_valid") else ""
    return f"{_label(row)}: within a factor of {_one(factor)} ({n} pair{'' if n == 1 else 's'}{span})."


def _in_window(row: dict[str, Any], leads: tuple[int, int]) -> bool:
    return _num(row.get("lead_min")) == leads[0] and _num(row.get("lead_max")) == leads[1]


def pick(rows: list[dict[str, Any]], *, variant: str = "corrected",
         leads: tuple[int, int] = TRACK_LEADS) -> list[dict[str, Any]]:
    """One row per model for the card, in model order: the ``leads`` window, the asked ``variant`` when the model
    has it, else raw."""
    out = []
    for m in MODEL_ORDER:
        mine = [r for r in rows or [] if r.get("model") == m and _in_window(r, leads)]
        hit = next((r for r in mine if r.get("variant") == variant), None) \
            or next((r for r in mine if r.get("variant") == "raw"), None)
        if hit:
            out.append(hit)
    return out


def track_lines(rows: list[dict[str, Any]], *, variant: str = "corrected", leads: tuple[int, int] = TRACK_LEADS,
                min_pairs: int = MIN_PAIRS) -> dict[str, Any] | None:
    """The card's block for one gauge or reach: ``{"head", "lines": [{"model", "variant", "text", "trust"}]}``,
    or None when the published table has nothing for it (the card then shows nothing)."""
    chosen = pick(rows, variant=variant, leads=leads)
    if not chosen:
        return None
    days = int(_num(chosen[0].get("window_days")) or TRACK_DAYS)
    return {"head": track_head(days, leads),
            "lines": [{"model": r.get("model"), "variant": r.get("variant"),
                       "text": track_line(r, min_pairs=min_pairs),
                       "trust": str(r.get("trust") or UNKNOWN) if r.get("enough_pairs") else UNKNOWN}
                      for r in chosen]}


# ── ranking the models at a gauge (the evidence ladder) ─────────────────────


def rank_models(rows: list[dict[str, Any]], *, leads: tuple[int, int] = TRACK_LEADS) -> dict[str, Any]:
    """The models at one gauge ranked by verified skill: every model and variant with enough pairs in the
    ``leads`` window, smallest factor first. Returns ``{"ranked": [...], "not_enough": [...], "sentence",
    "window_days", "leads"}``; ``ranked`` is empty (and the sentence says so) where no model has enough pairs."""
    window = [r for r in rows or [] if _in_window(r, leads)]
    ok = sorted((r for r in window if r.get("enough_pairs") and _num(r.get("factor")) is not None),
                key=lambda r: (float(r["factor"]), MODEL_ORDER.index(r["model"]) if r.get("model") in MODEL_ORDER
                               else 9, r.get("variant") != "corrected"))
    thin = [r for r in window if r not in ok]

    def brief(r: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
        out = {"model": r.get("model"), "variant": r.get("variant"), "label": _label(r),
               "factor": _num(r.get("factor")), "n_pairs": int(_num(r.get("n_pairs")) or 0),
               "first_valid": r.get("first_valid"), "last_valid": r.get("last_valid"),
               "trust": str(r.get("trust") or UNKNOWN) if r.get("enough_pairs") else UNKNOWN}
        if rank is not None:
            out["rank"] = rank
        return out

    ranked = [brief(r, i + 1) for i, r in enumerate(ok)]
    days = int(_num((window[0] if window else {}).get("window_days")) or TRACK_DAYS)
    when = f"over the last {days} days, {_leads(*leads)}"
    if not ranked:
        sentence = (f"No forecast has enough verified pairs here yet ({MIN_PAIRS} needed {when})."
                    if window else "No issued forecast has been verified at this gauge.")
    else:
        b = ranked[0]
        sentence = (f"Verified against this gauge {when}, {b['label']} did best: within a factor of "
                    f"{_one(b['factor'])} ({b['n_pairs']} pairs)")
        rest = [f"{r['label']} {_one(r['factor'])}" for r in ranked[1:]]
        sentence += f", then {'; '.join(rest)}." if rest else "."
    return {"window_days": days, "leads": list(leads), "ranked": ranked, "not_enough": [brief(r) for r in thin],
            "sentence": sentence}


# ── reading the published tables ────────────────────────────────────────────


def _clean(r: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in r.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        elif v is not None and type(v).__name__ in ("NAType", "NaTType"):
            v = None
        out[k] = v
    return out


def _frame(url: str, name: str, refresh: bool = False) -> Any:
    """A published table as a DataFrame, or None (not published yet, offline, or no pyarrow)."""
    import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    from aquascope.archive.catalog import _download, cache_dir

    try:
        return pd.read_parquet(_download(url, cache_dir() / name, refresh))
    except Exception as exc:  # noqa: BLE001 - verify/ is not published yet, or the Hub is out of reach
        logger.info("%s unavailable: %s", name, exc)
        return None


def gauge_rows(source: str | None = None, station_id: str | None = None, *, river_id: int | None = None,
               frame: Any = None) -> list[dict[str, Any]]:
    """A gauge's published track-record rows (by ``source`` and ``station_id``, or the gauges on ``river_id``),
    as plain dicts; [] when the table is not published or has nothing there."""
    df = _frame(GAUGES_URL, "forecast_skill_gauges.parquet") if frame is None else frame
    if df is None or not len(df):
        return []
    if river_id is not None:
        sel = df[df["river_id"].astype("Float64") == float(river_id)]
    else:
        sel = df[df["station_id"].astype(str) == str(station_id)]
        if source:
            sel = sel[sel["source"] == source]
    return [_clean(r) for r in sel.to_dict("records")]


def _haversine_km(lat1: float, lon1: float, lat2: Any, lon2: Any) -> Any:
    import numpy as np

    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def _lead_window(lead_days: Any) -> tuple[int, int] | None:
    """``lead_days`` as a window: 2 -> (2, 2), [1, 3] or '1-3' -> (1, 3), None -> None."""
    if lead_days is None or lead_days == "":
        return None
    if isinstance(lead_days, (int, float)):
        return int(lead_days), int(lead_days)
    if isinstance(lead_days, str):
        parts = [p for p in lead_days.replace("to", "-").replace(",", "-").split("-") if p.strip()]
    else:
        parts = list(lead_days)
    nums = [int(float(p)) for p in parts]
    if not nums or len(nums) > 2:
        raise ValueError("lead_days is a day (2) or a window ([1, 3] or '1-3')")
    return (nums[0], nums[-1]) if nums[0] <= nums[-1] else (nums[-1], nums[0])


def forecast_skill(station_id: str | None = None, *, source: str | None = None, lat: float | None = None,
                   lon: float | None = None, model: str | None = None, lead_days: Any = None,
                   variant: str | None = None, radius_km: float = NEAR_KM, gauges: Any = None,
                   network: Any = None) -> dict[str, Any]:
    """The verified forecast scorecard at a gauge: how close each model's issued forecasts came to what the gauge
    then recorded, over the newest 30 days.

    Give ``station_id`` (with ``source`` when the id is not unique), or ``lat``/``lon`` for the nearest forecast
    gauge within ``radius_km``. ``model`` (geoglows, glofas), ``variant`` (raw, corrected) and ``lead_days``
    (a day, or a window like ``[1, 3]``; default every published window) narrow the rows. Returns the gauge, its
    ``rows`` (pair count, dates, factor, median log10 error, CRPS, exceedance counts, trust), the models
    ranked by verified skill at 1 to 3 days (:func:`rank_models`), the matching every-gauge rows from
    ``verify/forecast_skill.parquet`` under ``network`` for context, and the trust bands. ``gauges`` and
    ``network`` take the tables directly (tests, or a caller that has read them).
    """
    window = _lead_window(lead_days)
    df = _frame(GAUGES_URL, "forecast_skill_gauges.parquet") if gauges is None else gauges
    base: dict[str, Any] = {"min_pairs": MIN_PAIRS, "window_days": TRACK_DAYS, "trust_bands": list(TRUST_BANDS),
                            "file": f"{FOLDER}/{GAUGE_TABLE}"}
    if df is None or not len(df):
        return {**base, "error": "The forecast verification is not published yet (verify/ in the Archive)."}
    gauge: dict[str, Any] = {}
    if station_id is not None:
        sel = df[df["station_id"].astype(str) == str(station_id)]
        if source:
            sel = sel[sel["source"] == source]
        keys = sel[["source", "station_id"]].drop_duplicates()
        if len(keys) > 1:
            return {**base, "error": f"station_id {station_id} is at several sources; pass source",
                    "sources": sorted(keys["source"].astype(str))}
        if not len(keys):
            return {**base, "error": f"No verified forecasts for {source + '/' if source else ''}{station_id} yet: "
                                     "it is not one of the Archive's forecast gauges, or has no pairs yet."}
        gauge = {"source": str(keys.iloc[0]["source"]), "station_id": str(keys.iloc[0]["station_id"])}
    elif lat is not None and lon is not None:
        if "lat" not in df or df["lat"].isna().all():
            return {**base, "error": "The published table has no gauge positions; pass station_id."}
        pos = df.dropna(subset=["lat", "lon"]).drop_duplicates(["source", "station_id"]).copy()
        pos["km"] = _haversine_km(float(lat), float(lon), pos["lat"].to_numpy(float), pos["lon"].to_numpy(float))
        near = pos.nsmallest(1, "km")
        if not len(near) or float(near.iloc[0]["km"]) > float(radius_km):
            return {**base, "error": f"No forecast gauge with a verified record within {radius_km:g} km of "
                                     f"{float(lat):.3f}, {float(lon):.3f}."}
        g = near.iloc[0]
        gauge = {"source": str(g["source"]), "station_id": str(g["station_id"]),
                 "distance_km": round(float(g["km"]), 2)}
    else:
        raise ValueError("give station_id, or lat and lon")
    every = gauge_rows(gauge["source"], gauge["station_id"], frame=df)
    if every:
        first = every[0]
        for k in ("river_id", "lat", "lon"):
            if first.get(k) is not None:
                gauge[k] = first[k]
    rows = [r for r in every if (model is None or r.get("model") == model)
            and (variant is None or r.get("variant") == variant)
            and (window is None or _in_window(r, window))]
    ranking = rank_models([r for r in every if model is None or r.get("model") == model])
    out = {**base, "gauge": gauge, "rows": rows, "ranking": ranking, "sentence": ranking["sentence"]}
    net = _frame(TABLE_URL, "forecast_skill.parquet") if network is None else network
    if net is not None and len(net):
        n = net[net["month"] == ALL_MONTHS]
        if model:
            n = n[n["model"] == model]
        if variant:
            n = n[n["variant"] == variant]
        lo, hi = window or TRACK_LEADS
        n = n[(n["lead_day"] >= lo) & (n["lead_day"] <= hi)]
        out["network"] = [_clean(r) for r in n.to_dict("records")]
    return out
