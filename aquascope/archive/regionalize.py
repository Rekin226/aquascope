"""Prediction in ungauged basins from the Archive: donor signatures, transfer, and leave-one-out skill (#53).

The donor search (:mod:`aquascope.archive.similar`) says which gauged
catchments look like an ungauged one. This module says what flow regime to
expect there, and how much to trust it:

1. **Signatures of every gauged donor** (:func:`compute_station_signatures`,
   run weekly after the bundles): from the archived daily discharge and the
   catchment area, per station: mean flow, median, Q95 (low flow) and Q05
   (high flow, both as exceedance percentiles), mean annual maximum, all in
   mm/d; runoff ratio against BasinATLAS precipitation; baseflow index, FDC
   slope, high- and low-flow frequency, zero-flow fraction, seasonality,
   flashiness (from :func:`aquascope.hydrology.signatures.compute_signatures`).
   Published as ``basins/station_signatures.parquet``.
2. **Transfer** (:func:`regionalize`): either *similarity-weighted averaging*
   over the k nearest donors (weights 1/(distance + 0.05); geometric mean for
   the mm/d magnitudes) or *ridge regression* of each signature on the
   standardised catchment attributes fitted over every donor, with the
   residual spread as the uncertainty. These are the two classical
   regionalisation routes (Bloeschl et al. 2013; Oudin et al. 2008).
3. **Leave-one-out skill** (:func:`loo_skill`): every donor predicted from
   the others; NSE / R² and median absolute relative error per signature and
   method, published as ``basins/regionalization_skill.json`` so the tool can
   say how good it is instead of hoping.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquascope.archive.similar import FEATURES, _transform, similar_basins

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
SIGNATURES_NAME = "station_signatures.parquet"
SKILL_NAME = "regionalization_skill.json"

# signature -> (transform for transfer, unit, label). "log" = positive magnitude, averaged geometrically.
SIGNATURES: dict[str, tuple[str | None, str, str]] = {
    "q_mean_mm": ("log", "mm/d", "mean daily flow"),
    "q_median_mm": ("log", "mm/d", "median daily flow"),
    "q95_mm": ("log", "mm/d", "low flow: exceeded 95 % of days"),
    "q05_mm": ("log", "mm/d", "high flow: exceeded 5 % of days"),
    "q_annual_max_mm": ("log", "mm/d", "mean annual daily maximum"),
    "runoff_ratio": (None, "-", "mean flow / BasinATLAS precipitation"),
    "baseflow_index": (None, "-", "baseflow / total flow"),
    "fdc_slope": (None, "-", "slope of the flow-duration curve (log space, 33-66 %)"),
    "high_flow_frequency": (None, "days/yr", "days above 3 x median per year"),
    "low_flow_frequency": (None, "days/yr", "days below 0.2 x median per year"),
    "zero_flow_fraction": (None, "-", "fraction of zero-flow days"),
    "seasonality_index": (None, "-", "Markham seasonality of monthly flow"),
    "flashiness_index": (None, "-", "Richards-Baker flashiness"),
}
# physical bounds the estimate and its band are clipped to (None = open)
BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "runoff_ratio": (0.0, None), "baseflow_index": (0.0, 1.0), "fdc_slope": (0.0, None),
    "high_flow_frequency": (0.0, 366.0), "low_flow_frequency": (0.0, 366.0), "zero_flow_fraction": (0.0, 1.0),
    "seasonality_index": (0.0, 1.0), "flashiness_index": (0.0, None),
}
MIN_YEARS = 10.0
RIDGE_LAMBDA = 1.0
DONOR_COUNT_FLOWS = {"q_mean_mm": "Mean daily flow", "q95_mm": "Q95 (low flow)", "q05_mm": "Q05 (high flow)"}


def signatures_url(repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/basins/{SIGNATURES_NAME}"


def skill_url(repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/basins/{SKILL_NAME}"


# ── 1. donor signatures ─────────────────────────────────────────────────────


def station_signature(q: pd.Series, area_km2: float, precipitation_mm_yr: float | None = None, *,
                      min_years: float = MIN_YEARS) -> dict[str, Any] | None:
    """Signatures for one station from its daily discharge (m3/s) and catchment area (km2); None if too short."""
    from aquascope.hydrology.signatures import compute_signatures

    q = q.dropna()
    q = q[q >= 0]
    if q.empty or not area_km2 or area_km2 <= 0:
        return None
    daily = q.resample("D").mean().dropna()
    years = (daily.index.max() - daily.index.min()).days / 365.25 if len(daily) > 1 else 0.0
    if years < min_years or len(daily) < 365 * 5:
        return None
    mm = daily * 86.4 / float(area_km2)
    try:
        rep = compute_signatures(daily)
    except Exception as exc:  # noqa: BLE001 - one odd record must not sink the table
        logger.info("signatures failed: %s", exc)
        return None
    am = mm.resample("YS").max()
    counts = mm.resample("YS").count()
    am = am[counts >= 292].dropna()
    out = {
        "n_days": int(len(daily)), "n_years": round(years, 1),
        "start": daily.index.min().date().isoformat(), "end": daily.index.max().date().isoformat(),
        "q_mean_mm": float(mm.mean()), "q_median_mm": float(mm.median()),
        "q95_mm": float(np.percentile(mm.to_numpy(), 5)), "q05_mm": float(np.percentile(mm.to_numpy(), 95)),
        "q_annual_max_mm": float(am.mean()) if len(am) else float("nan"),
        "runoff_ratio": float(mm.mean() * 365.25 / precipitation_mm_yr) if precipitation_mm_yr else float("nan"),
        "baseflow_index": float(rep.baseflow_index), "fdc_slope": float(rep.fdc_slope),
        "high_flow_frequency": float(rep.high_flow_frequency), "low_flow_frequency": float(rep.low_flow_frequency),
        "zero_flow_fraction": float(rep.zero_flow_fraction), "seasonality_index": float(rep.seasonality_index),
        "flashiness_index": float(rep.flashiness_index),
    }
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in out.items()}


def compute_station_signatures(
    bundles: dict[str, pd.DataFrame],
    catchments: pd.DataFrame,
    *,
    min_years: float = MIN_YEARS,
) -> pd.DataFrame:
    """Signatures for every station in the discharge bundles that has a catchment row (area) and enough record."""
    by_key = (catchments.drop_duplicates(["source", "station_id"]).set_index(["source", "station_id"])
              if not catchments.empty else None)
    rows: list[dict[str, Any]] = []
    for source, df in bundles.items():
        if df.empty:
            continue
        for sid, sub in df.groupby("station_id"):
            if by_key is None or (source, sid) not in by_key.index:
                continue
            row = by_key.loc[(source, sid)]
            area = float(row.get("area_km2") or 0.0)
            precip = row.get("precipitation_mm_yr")
            precip = None if precip is None or (isinstance(precip, float) and math.isnan(precip)) else float(precip)
            s = pd.Series(sub["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(sub["date"])))
            sig = station_signature(s, area, precip, min_years=min_years)
            if sig is None:
                continue
            rows.append({"source": source, "station_id": sid, "area_km2": area, "area_source": row.get("area_source"),
                         "hybas_id": int(row.get("hybas_id") or 0), **sig})
    out = pd.DataFrame(rows)
    logger.info("station signatures: %d stations", len(out))
    return out


def load_station_signatures(*, repo_id: str = DEFAULT_REPO_ID, refresh: bool = False,
                            path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        from aquascope.archive.catalog import _download, cache_dir

        path = _download(signatures_url(repo_id), cache_dir() / f"{repo_id.replace('/', '__')}__{SIGNATURES_NAME}",
                         refresh)
    return pd.read_parquet(path)


def load_skill(*, repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any] | None:
    try:
        from aquascope.archive.catalog import _download, cache_dir

        p = _download(skill_url(repo_id), cache_dir() / f"{repo_id.replace('/', '__')}__{SKILL_NAME}", False)
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - skill numbers are nice to have
        return None


# ── 2. transfer ─────────────────────────────────────────────────────────────


def _design_matrix(table: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    cols, stats = [], {}
    for f in features:
        col, how, _w = FEATURES[f]
        x = _transform(table[col].to_numpy(dtype=float), how)
        mu, sd = float(np.nanmean(x)), float(np.nanstd(x)) or 1.0
        stats[f] = (mu, sd)
        cols.append(np.nan_to_num((x - mu) / sd, nan=0.0))
    return np.column_stack(cols) if cols else np.zeros((len(table), 0)), stats


def _target_vector(tvals: dict[str, float], features: list[str], stats: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array([(_transform(tvals[f], FEATURES[f][1]) - stats[f][0]) / stats[f][1] for f in features])


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> tuple[np.ndarray, float]:
    """Ridge with intercept; returns (coef incl. intercept, residual std)."""
    xa = np.column_stack([np.ones(len(x)), x])
    penalty = lam * np.eye(xa.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xa.T @ xa + penalty, xa.T @ y)
    resid = y - xa @ beta
    dof = max(len(y) - xa.shape[1], 1)
    return beta, float(np.sqrt(np.sum(resid ** 2) / dof))


def _prep_y(values: np.ndarray, how: str | None) -> np.ndarray:
    return np.log(np.clip(values, 1e-3, None)) if how == "log" else values


def _unprep(v: float, how: str | None, name: str | None = None) -> float:
    out = float(np.exp(v)) if how == "log" else float(v)
    lo, hi = BOUNDS.get(name or "", (None, None))
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum((weights / weights.sum()) * values))


def _donor_count_comparison(name: str, values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Compare complete prefixes of one similarity ranking with the same transfer as the primary estimate."""
    how, unit, _ = SIGNATURES[name]
    estimates: dict[int, float] = {}
    counts = {}
    for k in (5, 10):
        counts[k] = int(np.isfinite(values[:k]).sum())
        if counts[k] == k:
            mean = _weighted_mean(_prep_y(values[:k], how), weights[:k])
            estimates[k] = _unprep(mean, how, name)
    five, ten = estimates.get(5), estimates.get(10)
    return {
        "label": DONOR_COUNT_FLOWS[name], "unit": unit, "n5": counts[5], "n10": counts[10],
        "k5": round(five, 4) if five is not None else None,
        "k10": round(ten, 4) if ten is not None else None,
        "change_pct": round(100 * (ten - five) / five, 2) if five and ten is not None else None,
    }


def regionalize(
    target: dict[str, Any],
    *,
    k: int = 10,
    method: str = "similarity",
    signatures: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
    catalog: list[dict[str, Any]] | None = None,
    features: list[str] | None = None,
    exclude: tuple[str, str] | None = None,
    compare_donor_counts: bool = False,
) -> dict[str, Any]:
    """Estimate the flow signatures of an ungauged catchment from gauged donors.

    ``target`` as in :func:`aquascope.archive.similar.similar_basins` (guide keys or a
    ``describe_catchment`` result, plus latitude/longitude). ``method``: ``similarity``
    (inverse-distance-weighted mean over the ``k`` most similar donors with signatures),
    ``regression`` (ridge on standardised attributes over all donors), or ``both``.
    Returns per-signature estimates with an uncertainty band, the donors used, and the citation.
    With ``compare_donor_counts``, compare the mean, Q95 and Q05 from the first five and ten donors
    in the same similarity ranking when the site and the donor table have a common feature.
    The primary transfer still uses the requested ``k``.
    """
    from aquascope.archive.catalog import load_stations
    from aquascope.archive.similar import _target_values, load_station_catchments

    if method not in ("similarity", "regression", "both"):
        raise ValueError("method must be similarity, regression or both")
    sig = signatures if signatures is not None else load_station_signatures()
    tab = table if table is not None else load_station_catchments()
    cat = catalog if catalog is not None else load_stations()
    if sig.empty:
        return {"error": "no donor signatures available yet"}
    sig = sig.set_index(["source", "station_id"]) if not isinstance(sig.index, pd.MultiIndex) else sig
    donors_tab = tab.merge(sig.reset_index()[["source", "station_id"]], on=["source", "station_id"], how="inner")
    if exclude:
        donors_tab = donors_tab[~((donors_tab["source"] == exclude[0]) & (donors_tab["station_id"] == exclude[1]))]
    if donors_tab.empty:
        return {"error": "no donors with both a catchment row and signatures"}
    tvals = _target_values(target)
    feats = [f for f in (features or list(FEATURES)) if f in FEATURES and tvals.get(f) is not None
             and FEATURES[f][0] in donors_tab.columns]
    out: dict[str, Any] = {"method": method, "features_used": feats, "n_donors_available": int(len(donors_tab)),
                           "estimates": {}}
    if method in ("similarity", "both"):
        compare_donor_counts = compare_donor_counts and bool(feats)
        primary_k = max(1, int(k))
        ranked = similar_basins(target, k=max(primary_k, 10) if compare_donor_counts else primary_k,
                                method="similarity", require_discharge=True, table=donors_tab,
                                catalog=cat, features=feats)
        donors = ranked["stations"]
        weights = np.array([1.0 / (d["score"] + 0.05) for d in donors])
        est: dict[str, Any] = {}
        comparison = {}
        comparison_donors: list[dict[str, Any]] = []
        if compare_donor_counts:
            comparison_donors = [{"source": d["source"], "station_id": d["station_id"], "score": d["score"],
                                  "signatures": {}} for d in donors[:10]]
        for name, (how, unit, label) in SIGNATURES.items():
            vals = np.array([sig.at[(d["source"], d["station_id"]), name] for d in donors], dtype=float)
            if compare_donor_counts and name in DONOR_COUNT_FLOWS:
                comparison[name] = _donor_count_comparison(name, vals, weights)
                for donor, value in zip(comparison_donors, vals):
                    donor["signatures"][name] = float(value) if np.isfinite(value) else None
            vals = vals[:primary_k]
            ok = np.isfinite(vals)
            if ok.sum() == 0:
                continue
            y = _prep_y(vals[ok], how)
            ww = weights[:primary_k][ok]
            mean = _weighted_mean(y, ww)
            sd = float(np.sqrt(_weighted_mean((y - mean) ** 2, ww)))
            est[name] = {
                "value": round(_unprep(mean, how, name), 4), "unit": unit, "label": label,
                "low": round(_unprep(mean - sd, how, name), 4), "high": round(_unprep(mean + sd, how, name), 4),
                "donor_min": round(float(np.min(vals[ok])), 4), "donor_max": round(float(np.max(vals[ok])), 4),
                "n_donors": int(ok.sum()),
            }
        if compare_donor_counts:
            out["donor_count_sensitivity"] = {"estimates": comparison, "donors": comparison_donors}
        primary = donors[:primary_k]
        out["similarity"] = {"estimates": est, "donors": primary, "k": len(primary)}
        out["estimates"] = est
    if method in ("regression", "both"):
        x, stats = _design_matrix(donors_tab, feats)
        xt = _target_vector(tvals, feats, stats)
        est_r: dict[str, Any] = {}
        joined = donors_tab.merge(sig.reset_index(), on=["source", "station_id"], how="left", suffixes=("", "_sig"))
        for name, (how, unit, label) in SIGNATURES.items():
            if name not in joined.columns:
                continue
            y_raw = joined[name].to_numpy(dtype=float)
            ok = np.isfinite(y_raw)
            if ok.sum() < max(20, x.shape[1] + 5):
                continue
            beta, rsd = _fit_ridge(x[ok], _prep_y(y_raw[ok], how))
            pred = float(beta[0] + xt @ beta[1:])
            est_r[name] = {"value": round(_unprep(pred, how, name), 4), "unit": unit, "label": label,
                           "low": round(_unprep(pred - rsd, how, name), 4),
                           "high": round(_unprep(pred + rsd, how, name), 4),
                           "n_donors": int(ok.sum()), "residual_sd": round(rsd, 4)}
        out["regression"] = {"estimates": est_r, "n_fit": int(len(donors_tab))}
        if method == "regression":
            out["estimates"] = est_r
    out.update({
        "license": "per-source (streamflow), CC-BY-4.0 (BasinATLAS)",
        "methods": [{
            "name": "Regionalisation of flow signatures from similar gauged basins",
            "text": "Donor gauges ranked by physical similarity (standardised BasinATLAS catchment attributes); "
                    "signatures transferred as an inverse-distance-weighted average over the k nearest donors "
                    "(geometric mean for magnitudes) and/or by ridge regression on the attributes over all donors; "
                    "the band is one weighted standard deviation of the donors (or the regression residual).",
            "citation": "Bloeschl, G. et al. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University "
                        "Press; Oudin, L. et al. (2008). Water Resour. Res. 44, W03413; Addor, N. et al. (2018). "
                        "A ranking of hydrological signatures based on their predictability in space. Water Resour. "
                        "Res. 54, 8792-8812.",
        }],
    })
    return out


def regionalize_point(lat: float, lon: float, *, k: int = 10, method: str = "similarity",
                      desc: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    """Describe the catchment of a point (BasinATLAS) and regionalise the flow signatures onto it.

    ``desc`` is a ``describe_catchment`` result already in hand (the browser's, built from the sub-basin row
    it read itself); with ``table``, ``signatures``, ``catalog`` and ``skill`` in ``kw`` nothing here touches a
    file, which is how the Explorer runs this step inside Pyodide.
    """
    skill = kw.pop("skill") if "skill" in kw else load_skill()
    if desc is None:
        from aquascope.archive.basins import describe_catchment

        desc = describe_catchment(lat, lon, upstream=False)
    if desc.get("error"):
        return {"latitude": lat, "longitude": lon, "error": desc["error"]}
    target = {"latitude": lat, "longitude": lon, "attributes": desc.get("attributes", {}),
              "sub_basin": desc.get("sub_basin")}
    kw.setdefault("compare_donor_counts", True)
    res = regionalize(target, k=k, method=method, **kw)
    res.update({"latitude": lat, "longitude": lon, "sub_basin": desc.get("sub_basin"),
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if skill:
        res["skill"] = {"note": "leave-one-out over the donors, published with the archive",
                        "by_signature": skill.get("methods", {}).get(method if method != "both" else "similarity", {})}
    return res


# ── 3. leave-one-out skill ──────────────────────────────────────────────────


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    denom = float(np.sum((obs - obs.mean()) ** 2))
    return float(1 - np.sum((sim - obs) ** 2) / denom) if denom > 0 else float("nan")


def loo_skill(
    signatures: pd.DataFrame,
    table: pd.DataFrame,
    catalog: list[dict[str, Any]],
    *,
    k: int = 10,
    methods: tuple[str, ...] = ("similarity", "regression"),
    max_stations: int | None = None,
) -> dict[str, Any]:
    """Leave-one-out: predict every donor from the others; NSE (log space for magnitudes), R2 and median APE."""
    sig = signatures.copy()
    if isinstance(sig.index, pd.MultiIndex):
        sig = sig.reset_index()
    keys = list(zip(sig["source"], sig["station_id"]))
    if max_stations and len(keys) > max_stations:
        pick = np.unique(np.linspace(0, len(keys) - 1, int(max_stations)).round().astype(int))
        keys = [keys[i] for i in pick]  # an even stride across sources, not the first N of one agency
    tab_idx = table.drop_duplicates(["source", "station_id"]).set_index(["source", "station_id"])
    sig_idx = sig.drop_duplicates(["source", "station_id"]).set_index(["source", "station_id"])
    meta = {(r["source"], r["station_id"]): r for r in catalog}
    preds: dict[str, dict[str, list[tuple[float, float]]]] = {m: {s: [] for s in SIGNATURES} for m in methods}
    n_done = 0
    for src, sid in keys:
        if (src, sid) not in tab_idx.index:
            continue
        row = tab_idx.loc[(src, sid)]
        st = meta.get((src, sid), {})
        target = {**{c: row.get(c) for c in tab_idx.columns}, "latitude": st.get("latitude"),
                  "longitude": st.get("longitude")}
        for m in methods:
            try:
                res = regionalize(target, k=k, method=m, signatures=sig, table=table, catalog=catalog,
                                  exclude=(src, sid))
            except Exception as exc:  # noqa: BLE001
                logger.info("loo failed for %s/%s (%s): %s", src, sid, m, exc)
                continue
            est = res.get("estimates", {})
            for s in SIGNATURES:
                obs = sig_idx.at[(src, sid), s] if s in sig_idx.columns else None
                if obs is None or (isinstance(obs, float) and math.isnan(obs)) or s not in est:
                    continue
                preds[m][s].append((float(obs), float(est[s]["value"])))
        n_done += 1
    out: dict[str, Any] = {"n_stations": n_done, "n_signature_stations": int(len(sig)), "k": k,
                           "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "methods": {}}
    for m in methods:
        out["methods"][m] = {}
        for s, pairs in preds[m].items():
            if len(pairs) < 10:
                continue
            o = np.array([p[0] for p in pairs])
            p_ = np.array([p[1] for p in pairs])
            how = SIGNATURES[s][0]
            oo, pp = (_prep_y(o, how), _prep_y(p_, how)) if how == "log" else (o, p_)
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs(p_ - o) / np.where(np.abs(o) > 1e-9, np.abs(o), np.nan)
            r = float(np.corrcoef(oo, pp)[0, 1]) if np.std(oo) > 0 and np.std(pp) > 0 else float("nan")
            nse = _nse(oo, pp)
            med = float(np.nanmedian(ape)) if np.isfinite(ape).any() else float("nan")
            out["methods"][m][s] = {"n": len(pairs), "nse": None if math.isnan(nse) else round(nse, 3),
                                    "r2": round(r * r, 3) if not math.isnan(r) else None,
                                    "median_ape": None if math.isnan(med) else round(med, 3),
                                    "space": "log" if how == "log" else "linear"}
    return out


__all__ = ["BOUNDS", "SIGNATURES", "compute_station_signatures", "load_skill", "load_station_signatures", "loo_skill",
           "regionalize", "regionalize_point", "station_signature", "signatures_url", "skill_url"]
