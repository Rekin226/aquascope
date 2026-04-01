"""Validate AquaScope hydrological computations against CAMELS benchmarks.

Compares computed signatures against published CAMELS catchment attributes
for 10 diverse US catchments.  This serves as a regression test and
credibility check for the hydrology module.

Usage::

    python examples/validation/validate_camels.py

References
----------
- Addor et al. (2017), doi:10.5194/hess-21-5293-2017
- Newman et al. (2015), doi:10.5194/hess-19-209-2015
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from aquascope.hydrology.baseflow import lyne_hollick
from aquascope.hydrology.flow_duration import flow_duration_curve
from aquascope.hydrology.signatures import compute_signatures

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "camels_benchmark"
CATCHMENTS_FILE = BENCHMARK_DIR / "catchments.json"
OUTPUT_CSV = pathlib.Path(__file__).resolve().parent / "validation_results.csv"

TOLERANCE_PCT = 0.25  # ±25 % relative tolerance for synthetic data
BFI_ABS_TOL = 0.15  # absolute tolerance for baseflow index


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_catchments() -> list[dict]:
    """Load benchmark catchment metadata."""
    with open(CATCHMENTS_FILE) as f:
        return json.load(f)


def _load_timeseries(gauge_id: str) -> tuple[pd.Series, pd.Series]:
    """Load synthetic discharge and precipitation for *gauge_id*.

    Returns
    -------
    discharge:
        Daily discharge (m³/s) with DatetimeIndex.
    precipitation:
        Daily precipitation (mm) with DatetimeIndex.
    """
    csv_path = BENCHMARK_DIR / f"{gauge_id}.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date")
    discharge = df["discharge_cms"]
    precipitation = df["precipitation_mm"]
    return discharge, precipitation


def compute_fdc_slope(discharge: pd.Series) -> float:
    """Compute FDC slope between 33rd and 66th exceedance percentiles.

    Parameters
    ----------
    discharge:
        Daily discharge series.

    Returns
    -------
    Slope of the log-transformed FDC between 33 % and 66 % exceedance.
    Negative for typical catchments (uses natural log).
    """
    fdc = flow_duration_curve(discharge, percentiles=[33, 66])
    q33_exc = fdc.percentiles[33]  # higher flow (exceeded 33 % of time)
    q66_exc = fdc.percentiles[66]  # lower flow  (exceeded 66 % of time)
    if q33_exc > 0 and q66_exc > 0:
        return (np.log(q66_exc) - np.log(q33_exc)) / (0.66 - 0.33)
    return float("nan")


def _pct_error(computed: float, published: float) -> float:
    """Return relative error as a fraction (e.g. 0.15 = 15 %)."""
    if published == 0:
        return 0.0 if computed == 0 else float("inf")
    return abs(computed - published) / abs(published)


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------
# Main validation
# ------------------------------------------------------------------

def validate() -> list[dict]:
    """Run validation for every benchmark catchment.

    Returns
    -------
    List of per-catchment result dictionaries.
    """
    catchments = _load_catchments()
    results: list[dict] = []

    header = (
        f"{'Gauge':>10s}  {'Metric':<22s}  {'Published':>10s}  "
        f"{'Computed':>10s}  {'Error%':>7s}  {'Status':>6s}"
    )
    print("=" * len(header))
    print("CAMELS Benchmark Validation — AquaScope hydrology module")
    print("=" * len(header))
    print()

    for c in catchments:
        gauge = c["gauge_id"]
        name = c["name"]
        print(f"--- {gauge}: {name} ({c['climate']}) ---")

        discharge, precipitation = _load_timeseries(gauge)
        sig = compute_signatures(discharge, precipitation=precipitation, area_km2=c["area_km2"])
        bf = lyne_hollick(discharge)
        fdc_slope = compute_fdc_slope(discharge)

        checks = [
            ("q_mean", sig.mean_flow, c["published_q_mean"], TOLERANCE_PCT),
            ("q5", sig.q5, c["published_q5"], TOLERANCE_PCT),
            ("q95", sig.q95, c["published_q95"], TOLERANCE_PCT),
            ("baseflow_index (sig)", sig.baseflow_index, c["published_baseflow_index"], BFI_ABS_TOL),
            ("baseflow_index (LH)", bf.bfi, c["published_baseflow_index"], BFI_ABS_TOL),
            ("fdc_slope", fdc_slope, c["published_fdc_slope"], TOLERANCE_PCT),
            ("peak_month", float(sig.peak_month), float(c["published_peak_month"]), 2.0),
        ]

        row: dict = {"gauge_id": gauge, "name": name}
        for metric, computed, published, tol in checks:
            if metric == "peak_month":
                # Allow ±2 months (circular)
                diff = abs(computed - published)
                diff = min(diff, 12 - diff)
                ok = diff <= tol
                err_str = f"{diff:.0f}mo"
            elif metric.startswith("baseflow_index"):
                ok = abs(computed - published) <= tol
                err_str = f"{abs(computed - published):.3f}"
            else:
                err = _pct_error(computed, published)
                ok = err <= tol
                err_str = f"{err * 100:.1f}%"

            print(f"  {metric:<22s}  pub={published:>10.3f}  "
                  f"comp={computed:>10.3f}  err={err_str:>7s}  {_pass_fail(ok)}")
            row[f"{metric}_published"] = published
            row[f"{metric}_computed"] = computed
            row[f"{metric}_pass"] = ok

        results.append(row)
        print()

    return results


def _write_csv(results: list[dict]) -> None:
    """Write results to a summary CSV."""
    if not results:
        return
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {OUTPUT_CSV}")


def main() -> None:
    """Entry point."""
    results = validate()
    _write_csv(results)

    total = sum(
        1 for r in results for k, v in r.items() if k.endswith("_pass") and v
    )
    total_checks = sum(
        1 for r in results for k in r if k.endswith("_pass")
    )
    failed = total_checks - total
    print(f"\nSummary: {total}/{total_checks} checks passed, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
