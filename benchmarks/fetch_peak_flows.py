"""Fetch USGS annual peak flows and compute reference quantiles.

Downloads peak-flow records for the 10 CAMELS benchmark catchments via
``dataretrieval``, caches them as CSVs, and fits three extreme-value
distributions to produce reference quantiles at [2, 5, 10, 25, 50, 100]-yr
return periods:

- GEV-MLE      — ``scipy.stats.genextreme``
- GEV-LMoments — ``lmoments3`` (Hosking 1997)
- LP3          — ``scipy.stats.pearson3`` on log10 peaks

The reference quantiles are computed independently of AquaScope and serve
as a cross-check for the benchmark harness.

Usage::

    pip install "aquascope[benchmarks]"
    python benchmarks/fetch_peak_flows.py

Outputs (inside ``data/camels_benchmark/``)::

    peaks/<gauge_id>_peaks.csv   — cached annual peak series
    ffa_reference.json           — reference quantiles + provenance

Notes
-----
- ``dataretrieval`` / ``lmoments3`` are NOT runtime dependencies of
  AquaScope. They are only needed for this generation script.
- Results depend on the USGS NWIS snapshot at fetch time.
  Re-running may produce slightly different records as USGS updates.
- USGS annual peaks are reported in cfs; they are converted to m³/s
  here to match the rest of the benchmark data.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from benchmarks import _paths
from benchmarks.results_models import RETURN_PERIODS

BASE = pathlib.Path(__file__).resolve().parent
BENCHMARK_DIR = _paths.BENCHMARK_DIR
DAILY_CATCHMENTS_FILE = _paths.DAILY_CATCHMENTS_FILE
PEAKS_DIR = _paths.PEAKS_DIR
FFA_REFERENCE_FILE = _paths.FFA_REFERENCE_FILE

FT3S_TO_M3S = 0.028316846592

# USGS Peak Discharge Qualification codes whose value is NOT a usable
# annual-max observation (USGS Water-Data Report "Discharge Codes"):
#   3 = discharge affected by dam failure
#   4 = discharge less than indicated value (censored)
#   8 = discharge greater than indicated value (censored)
# Codes 1, 2, 5, 6, 7, 9 (and multi-code combinations) represent real
# peak magnitudes and are kept; only their occurrence is tallied.
DROPPED_PEAK_CODES = {"3", "4", "8"}


# ---------------------------------------------------------------------------
# Fetching from NWIS via dataretrieval
# ---------------------------------------------------------------------------


def _peak_codes(value: object) -> set[str]:
    """Normalize one ``peak_cd`` cell to a set of qualifier codes.

    NWIS returns ``peak_cd`` as a mixed-type column (strings like ``"2,O"``,
    floats like ``6.0``) with comma-separated multi-codes.
    """
    if value is None:
        return set()
    if isinstance(value, float) and not np.isfinite(value):
        return set()

    codes = set()
    for part in str(value).strip().split(","):
        part = part.strip().upper()
        if part.endswith(".0") and part[:-2].isdigit():
            part = part[:-2]
        if part:
            codes.add(part)

    return codes


def _clean_peaks(peaks_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Screen a raw NWIS peaks response to usable annual-max observations.

    Drops rows that are not true annual maxima — USGS qualifier codes
    ``{3, 4, 8}`` (dam failure / censored) — plus null, sentinel, and
    non-positive peak values that would affect ``log10``/GEV/L-moments.
    Usable but imperfect records (codes 5, 6, 7, 9, multi-codes) are kept.

    Returns ``(cleaned_series, counts)``, where ``counts`` records how many
    rows were downloaded, kept and removed for each reason (qualifier code
    indicating poor data or invalid value), as well as a breakdown of each
    qualifier seen, showcasing the overall quality of collected data.
    """
    if "peak_cd" in peaks_df.columns:
        codes_seen: Counter = Counter()
        for code in peaks_df["peak_cd"].map(_peak_codes):
            codes_seen.update(code)
        has_bad_qualifier = peaks_df["peak_cd"].map(lambda value: bool(_peak_codes(value) & DROPPED_PEAK_CODES))
        rows_removed_qualifier = int(has_bad_qualifier.sum())
    else:
        codes_seen = Counter()
        has_bad_qualifier = pd.Series(False, index=peaks_df.index)
        rows_removed_qualifier = 0

    values = pd.to_numeric(peaks_df["peak_va"], errors="coerce").astype(np.float64)
    usable_value = values.notna() & values.gt(0)
    rows_removed_bad_value = int((~usable_value).sum())

    kept = peaks_df.loc[~has_bad_qualifier & usable_value]
    counts = {
        "rows_downloaded": len(peaks_df),
        "rows_removed_qualifier": rows_removed_qualifier,
        "rows_removed_bad_value": rows_removed_bad_value,
        "rows_kept": len(kept),
        "qualifiers_seen": dict(sorted(codes_seen.items())),
    }
    return kept["peak_va"].astype(np.float64), counts


def _prepare_annual_series(peaks: pd.Series) -> tuple[pd.Series, dict]:
    """Enforce one clean observation per USGS water year.

    The NWIS peaks service normally returns one row per water year, but that
    is an assumption, not a guarantee: rows can arrive out of order, share a
    water year, or lack a usable date. This sorts by date, drops undated
    rows, and keeps the largest value per water year (USGS convention:
    Oct 1 of the previous calendar year through Sep 30).

    Returns ``(annual_series, counts)`` where ``counts`` tallies undated and
    duplicate rows plus the resulting number of water years.
    """
    frame = peaks.rename("peak_va").to_frame()
    frame = frame.sort_index()

    missing_dates = frame.index.isna()
    counts = {
        "rows_removed_missing_date": int(missing_dates.sum()),
        "rows_removed_duplicate": 0,
        "water_years": 0,
    }
    frame = frame.loc[~missing_dates]

    if not frame.empty:
        # Water year ends in the calendar year for Oct 1–Dec 31 peaks.
        water_year = frame.index.year.where(frame.index.month < 10, frame.index.year + 1)
        frame = frame.assign(water_year=water_year)
        rows_before = len(frame)
        keep_labels = frame.groupby("water_year")["peak_va"].idxmax()
        annual = frame.loc[keep_labels, "peak_va"].sort_index()
        counts["rows_removed_duplicate"] = rows_before - len(annual)
        counts["water_years"] = len(annual)
    else:
        annual = frame["peak_va"]

    return annual, counts


def _fetch_peaks(gauge_id: str) -> pd.Series:
    """Fetch USGS annual peak flow series for one gauge.

    Returns a Series of peak_va (m³/s) indexed by datetime, with a
    ``clean_counts`` provenance dict attached via ``Series.attrs``.
    Raises on fetch failure or insufficient data.
    """
    try:
        from dataretrieval import nwis

        peaks_df = nwis.get_record(sites=gauge_id, service="peaks")
        if peaks_df is None or peaks_df.empty or "peak_va" not in peaks_df.columns:
            raise ValueError(f"no peak_va data returned for {gauge_id}")

        peak_va, clean_counts = _clean_peaks(peaks_df)
        # USGS peak discharges are reported in cubic feet per second (cfs).
        # Convert to m³/s to match the rest of the benchmark data.
        peak_va = peak_va * FT3S_TO_M3S
        peak_va, annual_counts = _prepare_annual_series(peak_va)

        if len(peak_va) < 10:
            raise ValueError(
                f"Fetch failed: only {len(peak_va)} annual peaks for {gauge_id} collected; need at least 10"
            )

        clean_counts.update(annual_counts)
        peak_va.attrs["clean_counts"] = clean_counts
        return peak_va

    except Exception as exc:
        raise RuntimeError(f"Failed to fetch peaks for {gauge_id}") from exc


# ---------------------------------------------------------------------------
# Apply scipy distribution fits
# ---------------------------------------------------------------------------


def _fit_gev_mle(peak_va: np.ndarray) -> tuple[dict[int, float], bool]:
    """Fit GEV via scipy MLE and return quantiles.

    Returns ``(quantiles, stable)``. ``stable`` is True when the MLE shape
    falls within the plausible bound (``abs(shape) <= 0.5`` and ``scale > 0``).
    When not stable, a :class:`RuntimeWarning` is emitted — the MLE is unreliable
    on short/heavy-tailed records.
    """
    from scipy.stats import genextreme

    shape, loc, scale = genextreme.fit(peak_va)

    stable = bool(abs(shape) <= 0.5 and scale > 0)
    if not stable:
        warnings.warn(
            "GEV-MLE fit produced an unstable/extreme shape "
            f"(shape={shape:.3f}, scale={scale:.3f}); an independent "
            "L-moments fit is better for this gauge",
            RuntimeWarning,
            stacklevel=2,
        )

    quantiles = {}
    for rp in RETURN_PERIODS:
        prob = 1 - 1.0 / rp
        quantiles[rp] = round(float(genextreme.ppf(prob, shape, loc=loc, scale=scale)), 2)

    return quantiles, stable


def _fit_gev_lmoments(peak_va: np.ndarray) -> dict[int, float]:
    """Fit GEV via L-moments (§Hosking 1997) and return quantiles.

    Uses the :mod:`lmoments3` package — the de-facto standard Python L-moment
    implementation — so the reference is fully independent of AquaScope.
    """
    from lmoments3 import distr

    data = np.asarray(peak_va, dtype=np.float64)
    params = distr.gev.lmom_fit(data)
    gev = distr.gev

    quantiles = {}
    for rp in RETURN_PERIODS:
        # isf() takes the survival probability; for a return period of rp
        # years, the exceedance probability per year is 1/rp.
        quantiles[rp] = round(float(gev.isf(1.0 / rp, **params)), 2)
    return quantiles


def _fit_lp3(peak_va: np.ndarray) -> dict[int, float]:
    """Fit a basic Log-Pearson Type III and return quantiles.

    Uses log10-transformed annual maxima with an unadjusted moment skew.
    This is NOT a full Bulletin 17C fit: regional-skew weighting,
    low-outlier treatment, and EMA (Bulletin 17B/17C, Bulletin 14B) are
    not implemented, so it is a cross-check baseline, not the federal
    standard method.
    """
    from scipy.stats import pearson3

    log_peaks = np.log10(np.asarray(peak_va, dtype=np.float64))
    mean = float(np.mean(log_peaks))
    std = float(np.std(log_peaks, ddof=1))
    skew = float(np.mean(((log_peaks - mean) / std) ** 3) if std > 0 else 0.0)

    quantiles = {}
    for rp in RETURN_PERIODS:
        prob = 1 - 1.0 / rp
        log_val = pearson3.ppf(prob, skew, loc=mean, scale=std)
        quantiles[rp] = round(float(10**log_val), 2)
    return quantiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_artifacts(peak_frames: dict[str, pd.DataFrame], reference: dict) -> None:
    """Write cached peaks + reference JSON, replacing outputs atomically.

    Every artifact is first written to a ``*.tmp`` sibling in its final
    directory; only after all temps are on disk are the final paths swapped
    in via ``Path.replace``. A failure anywhere during compute — which happens
    before this function is called — leaves the previous artifacts untouched,
    so a rerun never exposes a partially refreshed cache.
    """
    PEAKS_DIR.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[pathlib.Path, pathlib.Path]] = []
    tmp_json = FFA_REFERENCE_FILE.with_suffix(FFA_REFERENCE_FILE.suffix + ".tmp")
    try:
        for gauge_id, frame in peak_frames.items():
            csv_path = PEAKS_DIR / f"{gauge_id}_peaks.csv"
            tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
            staged.append((tmp_path, csv_path))
            frame.to_csv(tmp_path, index=False)

        staged.append((tmp_json, FFA_REFERENCE_FILE))
        with open(tmp_json, "w") as f:
            json.dump(reference, f, indent=2)

        for tmp_path, final_path in staged:
            tmp_path.replace(final_path)
    finally:
        for tmp_path, _ in staged:
            if tmp_path.exists():
                tmp_path.unlink()


def main() -> None:
    """Fetch peak flows and compute reference quantiles for all gauges."""
    with open(DAILY_CATCHMENTS_FILE) as f:
        catchments = json.load(f)

    reference: dict = {
        "metadata": {
            "generated": datetime.now().strftime("%Y-%m-%d"),
            "source": "USGS NWIS peak-flow service",
            "return_periods": RETURN_PERIODS,
            "note": (
                "Reference quantiles computed independently of AquaScope: GEV-MLE and "
                "LP3 via scipy, GEV-LMoments via lmoments3 (Hosking 1997), on USGS "
                "annual peak series. Rows carrying USGS qualifier codes 3/4/8 (dam "
                "failure / censored), plus non-positive, sentinel, undated, or "
                "duplicate-water-year records, are excluded; see each catchment's "
                "data_quality for row counts. Serves as a cross-check, not ground truth."
            ),
        },
        "catchments": {},
    }

    peak_frames: dict[str, pd.DataFrame] = {}

    print(f"Fetching USGS annual peak flows for {len(catchments)} catchments ...\n")

    for c in catchments:
        gauge_id = c["gauge_id"]
        name = c["name"]
        print(f"--- {gauge_id}: {name} ---")

        peak_va = _fetch_peaks(gauge_id)

        # Hold the cache series in memory until the whole run has succeeded
        # so a mid-run failure cannot leave a partially refreshed cache.
        peaks_df = peak_va.reset_index()
        peaks_df.columns = ["date", "peak_va"]
        peak_frames[gauge_id] = peaks_df

        n_years = len(peak_va)
        year_min = peak_va.index.min().year if hasattr(peak_va.index.min(), "year") else "?"
        year_max = peak_va.index.max().year if hasattr(peak_va.index.max(), "year") else "?"
        print(f"  {n_years} annual peaks ({year_min}–{year_max})")

        arr = peak_va.values.astype(np.float64)

        # Fit all three distributions
        gev_mle, mle_stable = _fit_gev_mle(arr)
        gev_lmom = _fit_gev_lmoments(arr)
        lp3 = _fit_lp3(arr)

        print(f"  GEV-MLE   100-yr: {gev_mle[100]:.1f} m³/s ({'stable' if mle_stable else 'unstable'})")
        print(f"  GEV-LMom  100-yr: {gev_lmom[100]:.1f} m³/s")
        print(f"  LP3       100-yr: {lp3[100]:.1f} m³/s")

        entry = {
            "name": name,
            "num_records": n_years,
            "start_year": year_min,
            "end_year": year_max,
            "mle_stable": mle_stable,
            "scipy_gev": {str(k): v for k, v in gev_mle.items()},
            "scipy_gev_lmoments": {str(k): v for k, v in gev_lmom.items()},
            "scipy_lp3": {str(k): v for k, v in lp3.items()},
        }
        clean_counts = peak_va.attrs.get("clean_counts")
        if clean_counts is not None:
            entry["data_quality"] = clean_counts
        reference["catchments"][gauge_id] = entry
        print()

    # All fits succeeded — atomically publish the complete result set.
    _write_artifacts(peak_frames, reference)

    n_ok = len(reference["catchments"])
    print(f"Done. {n_ok}/{len(catchments)} gauges processed.")
    print(f"  Peaks cached in:  {PEAKS_DIR}")
    print(f"  Reference file:   {FFA_REFERENCE_FILE}")


if __name__ == "__main__":
    main()
