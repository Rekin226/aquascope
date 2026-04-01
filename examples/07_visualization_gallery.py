#!/usr/bin/env python3
"""Example 07 — Visualisation gallery.

Demonstrates all AquaScope visualization capabilities using synthetic
data.  Run this script to generate a gallery of plots in ``data/gallery/``.

Generated plots:
1. Time-series plot
2. Multi-parameter overlay
3. Forecast with confidence bands
4. Observed vs predicted scatter
5. Box plot by station
6. Correlation heatmap
7. Flow duration curve
8. Hydrograph with baseflow
9. SPI drought timeline
10. Return period plot
11. Station scatter map
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/gallery")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("Install viz deps: pip install aquascope[viz]")
    raise SystemExit(1)

from aquascope.viz import (
    plot_boxplot,
    plot_fdc,
    plot_forecast,
    plot_heatmap,
    plot_multi_param,
    plot_observed_vs_predicted,
    plot_return_periods,
    plot_spi_timeline,
    plot_station_scatter,
    plot_timeseries,
)
from aquascope.viz.hydro import plot_hydrograph
from aquascope.viz.timeseries import plot_residuals

print("▸ Generating AquaScope visualisation gallery …\n")

# ── 1. Time-series ───────────────────────────────────────────────────────

dates = pd.date_range("2020-01-01", periods=365, freq="D")
ts = pd.DataFrame({"value": np.sin(np.linspace(0, 4 * np.pi, 365)) * 10 + 50}, index=dates)
plot_timeseries(ts, title="Daily Water Level", ylabel="Level (m)", save_path=str(OUT / "01_timeseries.png"))
print("  ✓ 01_timeseries.png")

# ── 2. Multi-parameter ──────────────────────────────────────────────────

ts["pH"] = rng.normal(7.2, 0.3, 365)
ts["DO"] = rng.normal(8.5, 1.0, 365)
plot_multi_param(ts, columns=["value", "pH", "DO"], title="Multi-Parameter", save_path=str(OUT / "02_multi_param.png"))
print("  ✓ 02_multi_param.png")

# ── 3. Forecast ──────────────────────────────────────────────────────────

forecast = pd.DataFrame({
    "yhat": np.linspace(50, 55, 30) + rng.normal(0, 2, 30),
    "yhat_lower": np.linspace(45, 50, 30),
    "yhat_upper": np.linspace(55, 60, 30),
}, index=pd.date_range("2021-01-01", periods=30, freq="D"))
plot_forecast(observed=ts[["value"]], forecast=forecast, title="30-Day Forecast",
              save_path=str(OUT / "03_forecast.png"))
print("  ✓ 03_forecast.png")

# ── 4. Observed vs Predicted ─────────────────────────────────────────────

obs = pd.Series(rng.normal(50, 10, 200))
pred = obs * 0.95 + rng.normal(0, 3, 200)
plot_observed_vs_predicted(obs, pred, metrics={"NSE": 0.91, "KGE": 0.88, "RMSE": 3.2},
                           save_path=str(OUT / "04_obs_vs_pred.png"))
print("  ✓ 04_obs_vs_pred.png")

# ── 5. Box plot ──────────────────────────────────────────────────────────

box_df = pd.DataFrame({
    "value": np.concatenate([rng.normal(m, 1, 50) for m in [7.0, 7.5, 6.8, 7.2]]),
    "station_name": np.repeat(["Taipei", "Taichung", "Kaohsiung", "Hsinchu"], 50),
})
plot_boxplot(box_df, title="pH by Station", ylabel="pH", save_path=str(OUT / "05_boxplot.png"))
print("  ✓ 05_boxplot.png")

# ── 6. Heatmap ──────────────────────────────────────────────────────────

heat_df = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=["pH", "DO", "BOD5", "COD", "Turbidity"])
heat_df["DO"] = heat_df["pH"] * -0.6 + rng.normal(0, 0.5, 100)  # add correlation
plot_heatmap(heat_df, save_path=str(OUT / "06_heatmap.png"))
print("  ✓ 06_heatmap.png")

# ── 7. Flow Duration Curve ──────────────────────────────────────────────

q = pd.Series(rng.exponential(20, 3650) + 5, index=pd.date_range("2015-01-01", periods=3650, freq="D"))
plot_fdc(q, save_path=str(OUT / "07_fdc.png"))
print("  ✓ 07_fdc.png")

# ── 8. Hydrograph ───────────────────────────────────────────────────────

hydro_df = pd.DataFrame({
    "discharge": q.values[:365],
    "baseflow": q.values[:365] * 0.6,
}, index=q.index[:365])
plot_hydrograph(hydro_df, save_path=str(OUT / "08_hydrograph.png"))
print("  ✓ 08_hydrograph.png")

# ── 9. SPI Timeline ─────────────────────────────────────────────────────

spi_dates = pd.date_range("2015-01-01", periods=120, freq="MS")
spi_df = pd.DataFrame({"spi_3": rng.normal(0, 1.2, 120)}, index=spi_dates)
plot_spi_timeline(spi_df, save_path=str(OUT / "09_spi_timeline.png"))
print("  ✓ 09_spi_timeline.png")

# ── 10. Return Periods ──────────────────────────────────────────────────

rp = {2: 50.0, 5: 80.0, 10: 105.0, 25: 135.0, 50: 158.0, 100: 182.0}
plot_return_periods(rp, observed_max=120.0, save_path=str(OUT / "10_return_periods.png"))
print("  ✓ 10_return_periods.png")

# ── 11. Station Scatter ─────────────────────────────────────────────────

stations = pd.DataFrame({
    "station_name": ["Taipei", "Taichung", "Kaohsiung", "Hsinchu", "Tainan"],
    "latitude": [25.03, 24.15, 22.63, 24.80, 22.99],
    "longitude": [121.57, 120.68, 120.30, 120.97, 120.21],
    "pH": [7.2, 6.8, 7.5, 7.0, 7.3],
})
plot_station_scatter(stations, value_col="pH", title="Station pH Values",
                     save_path=str(OUT / "11_station_scatter.png"))
print("  ✓ 11_station_scatter.png")

# ── 12. Residuals ───────────────────────────────────────────────────────

plot_residuals(obs, pred, save_path=str(OUT / "12_residuals.png"))
print("  ✓ 12_residuals.png")

print(f"\n✓ Gallery complete!  {len(os.listdir(OUT))} plots saved to {OUT}/")
