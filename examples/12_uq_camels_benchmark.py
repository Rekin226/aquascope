"""Multi-basin uncertainty-quantification benchmark for GR4J (#78, epic #71).

Runs GR4J with quantile prediction intervals across the bundled CAMELS
benchmark basins and reports, per basin and in aggregate:

  * PICP  — coverage of the central 90% interval (target ~0.90)
  * CRPS  — continuous ranked probability score (lower is better)

and draws a reliability diagram (observed vs nominal coverage). This is the
multi-basin demonstration that makes the GR4J UQ citable.

Note: the bundled CAMELS files are *synthetic* regression data with no PET
column, so PET is approximated by a seasonal climatology and discharge is
converted from m3/s to mm/day using the published catchment area. This is a
methods demonstration, not a scientific result.

Requires: pip install 'aquascope[ml,viz]'   (statsmodels not needed)
Run:      python examples/12_uq_camels_benchmark.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from aquascope.analysis.metrics import crps_from_quantiles, picp
from aquascope.models.rainfall_runoff import predict_quantiles

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "camels_benchmark"
CRPS_GRID = [round(q, 2) for q in np.arange(0.05, 1.0, 0.05)]
RELIABILITY_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_basin(gauge_id: str, area_km2: float, max_days: int | None = None):
    """Load a basin as (precip, pet_proxy, discharge_mm_per_day)."""
    df = pd.read_csv(BENCHMARK_DIR / f"{gauge_id}.csv", parse_dates=["date"]).set_index("date")
    if max_days:
        df = df.iloc[:max_days]
    precip = df["precipitation_mm"]
    # m3/s -> mm/day over the catchment: Q * 86400 s / (area_km2 * 1e6 m2) * 1000 mm/m
    q_mm = df["discharge_cms"] * 86.4 / area_km2
    # Seasonal PET climatology proxy (no PET in the synthetic data).
    doy = df.index.dayofyear.to_numpy()
    pet = pd.Series(np.clip(2.5 + 2.0 * np.sin(2 * np.pi * (doy - 110) / 365.0), 0.5, None),
                    index=df.index)
    return precip, pet, q_mm


def run_benchmark(basin_ids=None, *, max_days=None, maxiter=20, warmup_days=365):
    """Run the GR4J UQ benchmark; returns a per-basin results list."""
    catchments = {c["gauge_id"]: c for c in json.loads((BENCHMARK_DIR / "catchments.json").read_text())}
    ids = basin_ids or list(catchments)
    quantiles = sorted(set(CRPS_GRID) | {0.05, 0.5, 0.95} | set(RELIABILITY_LEVELS))
    results = []
    for gid in ids:
        area = catchments[gid]["area_km2"]
        precip, pet, q_mm = load_basin(gid, area, max_days=max_days)
        res = predict_quantiles(
            precip, pet, q_mm, quantiles=quantiles,
            method="residual", warmup_days=warmup_days, maxiter=maxiter, heteroscedastic=True,
        )
        ev = slice(warmup_days, None)
        obs = q_mm.values[ev]
        cov90 = picp(obs, res.quantiles[0.05].values[ev], res.quantiles[0.95].values[ev])
        crps = crps_from_quantiles(obs, {q: res.quantiles[q].values[ev] for q in CRPS_GRID})
        observed_cov = {
            q: float(np.nanmean(obs <= res.quantiles[q].values[ev])) for q in RELIABILITY_LEVELS
        }
        results.append({"gauge_id": gid, "name": catchments[gid]["name"],
                        "picp90": cov90, "crps": crps, "reliability": observed_cov})
    return results


def main() -> None:
    results = run_benchmark()
    print(f"{'gauge_id':<10} {'PICP(90%)':>10} {'CRPS':>8}  name")
    for r in results:
        print(f"{r['gauge_id']:<10} {r['picp90']:>10.3f} {r['crps']:>8.3f}  {r['name']}")
    mean_picp = float(np.mean([r["picp90"] for r in results]))
    mean_crps = float(np.mean([r["crps"] for r in results]))
    print(f"\nAggregate: mean PICP(90%) = {mean_picp:.3f} (target 0.90), mean CRPS = {mean_crps:.3f}")

    # Reliability diagram: observed vs nominal coverage, pooled across basins.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = {q: float(np.mean([r["reliability"][q] for r in results])) for q in RELIABILITY_LEVELS}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(RELIABILITY_LEVELS, [pooled[q] for q in RELIABILITY_LEVELS], "o-", label="GR4J residual UQ")
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("observed coverage")
    ax.set_title(f"GR4J UQ reliability ({len(results)} CAMELS basins)")
    ax.legend()
    out = pathlib.Path(__file__).resolve().parent / "uq_reliability.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Reliability diagram saved to {out}")


if __name__ == "__main__":
    main()
