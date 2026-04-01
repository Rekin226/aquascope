#!/usr/bin/env python3
"""Example 04 — Hydrological analysis toolkit.

Demonstrates AquaScope's hydrology module with synthetic discharge data:

1. Flow Duration Curve (FDC) with percentile extraction
2. Baseflow separation (Lyne-Hollick & Eckhardt)
3. Recession analysis
4. Flood frequency analysis (GEV)
5. Low-flow statistics (7Q10)
"""

import numpy as np
import pandas as pd

# ── Generate synthetic daily discharge (10 years) ────────────────────────

print("▸ Generating 10 years of synthetic discharge …\n")
rng = np.random.default_rng(42)
days = 365 * 10
dates = pd.date_range("2014-01-01", periods=days, freq="D")

t = np.arange(days) / 365.0
seasonal = 25 + 15 * np.sin(2 * np.pi * t)  # spring peak
noise = rng.exponential(5, days)
storms = rng.choice([0, 0, 0, 0, 1], days) * rng.exponential(40, days)
discharge = pd.Series(np.maximum(seasonal + noise + storms, 0.5), index=dates, name="Q")

print(f"  Range: {discharge.min():.1f} – {discharge.max():.1f} m³/s")
print(f"  Mean:  {discharge.mean():.1f} m³/s\n")

# ── 1. Flow Duration Curve ───────────────────────────────────────────────

from aquascope.hydrology import flow_duration_curve

fdc = flow_duration_curve(discharge)
print("▸ Flow Duration Curve Percentiles:")
for pct in [5, 10, 50, 90, 95]:
    print(f"    Q{pct:>2d} = {fdc.percentiles[pct]:.2f} m³/s")

# ── 2. Baseflow separation ──────────────────────────────────────────────

from aquascope.hydrology import eckhardt, lyne_hollick

bf_lh = lyne_hollick(discharge)
bf_ek = eckhardt(discharge)
print(f"\n▸ Baseflow Index:")
print(f"    Lyne-Hollick: BFI = {bf_lh.bfi:.3f}")
print(f"    Eckhardt:     BFI = {bf_ek.bfi:.3f}")

# ── 3. Recession analysis ───────────────────────────────────────────────

from aquascope.hydrology import recession_analysis

rec = recession_analysis(discharge)
print(f"\n▸ Recession Analysis:")
print(f"    Segments found:     {len(rec.segments)}")
print(f"    Recession constant: {rec.recession_constant:.1f} days")
print(f"    Half-life:          {rec.half_life_days:.1f} days")
print(f"    R²:                 {rec.r_squared:.4f}")

# ── 4. Flood frequency analysis ─────────────────────────────────────────

from aquascope.hydrology import fit_gev

ffa = fit_gev(discharge)
print(f"\n▸ Flood Frequency Analysis (GEV):")
for rp in [2, 5, 10, 25, 50, 100]:
    val = ffa.return_periods.get(rp, 0)
    ci = ffa.confidence_intervals.get(rp)
    ci_str = f"  90% CI: [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
    print(f"    {rp:>5d}-yr flood: {val:.1f} m³/s{ci_str}")

# ── 5. Low-flow statistics ──────────────────────────────────────────────

from aquascope.hydrology import low_flow_stat

q7_10 = low_flow_stat(discharge, n_day=7, return_period=10)
q30_5 = low_flow_stat(discharge, n_day=30, return_period=5)
print(f"\n▸ Low-Flow Statistics:")
print(f"    7Q10 = {q7_10:.2f} m³/s")
print(f"   30Q5  = {q30_5:.2f} m³/s")

# ── 6. Visualise ─────────────────────────────────────────────────────────

print("\n▸ Generating plots …")
try:
    from aquascope.viz import plot_fdc, plot_return_periods
    from aquascope.viz.hydro import plot_hydrograph

    plot_fdc(discharge, save_path="data/fdc.png")
    print("  ✓ data/fdc.png")

    plot_return_periods(ffa.return_periods, observed_max=float(discharge.max()), save_path="data/return_periods.png")
    print("  ✓ data/return_periods.png")

    plot_hydrograph(bf_lh.df.rename(columns={"total": "discharge"}), save_path="data/hydrograph.png")
    print("  ✓ data/hydrograph.png")
except ImportError:
    print("  Install viz deps: pip install aquascope[viz]")

print("\nDone!")
