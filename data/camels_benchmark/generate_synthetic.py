#!/usr/bin/env python
"""Generate synthetic daily streamflow for CAMELS benchmark catchments.

Creates reproducible daily discharge and precipitation time series for 10
USGS catchments, calibrated to approximate published CAMELS statistics.
These series serve as regression-test inputs, **not** real observations.

Usage::

    python data/camels_benchmark/generate_synthetic.py

References
----------
Addor et al. (2017), doi:10.5194/hess-21-5293-2017
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CATCHMENTS_FILE = HERE / "catchments.json"

N_YEARS = 10
START_DATE = "2000-01-01"


def _generate_one(
    catchment: dict,
    rng: np.random.Generator,
    *,
    n_years: int = N_YEARS,
) -> pd.DataFrame:
    """Generate synthetic daily discharge and precipitation for one catchment.

    Parameters
    ----------
    catchment:
        Dictionary with published attributes from *catchments.json*.
    rng:
        NumPy random generator for reproducibility.
    n_years:
        Number of years to generate (default 10).

    Returns
    -------
    DataFrame with columns ``date``, ``discharge_cms``, ``precipitation_mm``.
    """
    dates = pd.date_range(START_DATE, periods=n_years * 365, freq="D")
    n = len(dates)

    q_mean = catchment["published_q_mean"]
    q5 = catchment["published_q5"]
    q95 = catchment["published_q95"]
    peak_month = catchment["published_peak_month"]
    bfi = catchment["published_baseflow_index"]
    rr = catchment["published_runoff_ratio"]

    # --- derive log-normal spread from target percentiles ----------------
    q5_safe = max(q5, q_mean * 0.01)
    q95_safe = max(q95, q_mean * 1.5)
    sigma = (np.log(q95_safe) - np.log(q5_safe)) / 3.29  # ≈ 2 * 1.645
    sigma = max(sigma, 0.2)

    # --- AR(1) persistence proportional to baseflow fraction -------------
    ar = min(0.98, 0.82 + 0.15 * bfi)

    # --- seasonal component peaking at target month ----------------------
    doy = dates.dayofyear.values.astype(float)
    peak_doy = (peak_month - 0.5) * 30.44
    seasonal_amp = 0.4 * sigma
    seasonal = seasonal_amp * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)

    # --- correlated noise (AR-1) ----------------------------------------
    z = np.zeros(n)
    innovations = rng.standard_normal(n)
    noise_std = sigma * np.sqrt(1 - ar**2)
    for i in range(1, n):
        z[i] = ar * z[i - 1] + noise_std * innovations[i]

    # --- log-normal discharge -------------------------------------------
    mu = np.log(max(q_mean, 0.001))
    log_q = mu + seasonal + z
    q = np.exp(log_q)

    # --- rescale to exact target mean -----------------------------------
    q *= q_mean / q.mean()
    q = np.maximum(q, 0.0)

    # --- synthetic precipitation ----------------------------------------
    wet_frac = max(0.15, min(0.55, rr + 0.15))
    wet_mask = rng.random(n) < wet_frac
    precip = np.zeros(n)
    n_wet = int(wet_mask.sum())
    if n_wet > 0:
        precip[wet_mask] = rng.exponential(5.0, n_wet)
    if rr > 0 and precip.sum() > 0:
        precip *= (q.sum() / rr) / precip.sum()

    return pd.DataFrame({
        "date": dates,
        "discharge_cms": np.round(q, 4),
        "precipitation_mm": np.round(precip, 2),
    })


def main() -> None:
    """Generate CSV files for all benchmark catchments."""
    rng = np.random.default_rng(42)

    with open(CATCHMENTS_FILE) as f:
        catchments = json.load(f)

    print(f"Generating synthetic data for {len(catchments)} catchments …\n")

    for c in catchments:
        gauge = c["gauge_id"]
        df = _generate_one(c, rng)
        out = HERE / f"{gauge}.csv"
        df.to_csv(out, index=False)

        q = df["discharge_cms"]
        print(
            f"  {gauge} ({c['name'][:35]:35s}) — "
            f"mean={q.mean():.2f}, "
            f"q5={np.percentile(q, 5):.2f}, "
            f"q95={np.percentile(q, 95):.2f}"
        )

    print(f"\n✓ Generated {len(catchments)} CSV files in {HERE}")


if __name__ == "__main__":
    main()
