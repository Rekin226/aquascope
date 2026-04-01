#!/usr/bin/env python3
"""Example 05 — Flood forecasting workflow.

End-to-end demonstration of the FloodChallenge:

1. Fetch historical discharge from Open-Meteo (or use synthetic data)
2. Fit a forecasting model
3. Generate flood risk assessment
4. Visualise results
"""

import numpy as np
import pandas as pd

# ── Generate realistic flood-prone discharge ─────────────────────────────

print("▸ Preparing discharge data …\n")
rng = np.random.default_rng(42)
days = 365 * 8
dates = pd.date_range("2016-01-01", periods=days, freq="D")

base = 30 + 20 * np.sin(2 * np.pi * np.arange(days) / 365)
noise = rng.exponential(8, days)
# Add flood events
flood_events = rng.choice([0] * 20 + [1], days) * rng.exponential(80, days)
discharge = pd.Series(np.maximum(base + noise + flood_events, 1.0), index=dates, name="value")

print(f"  Period:   {dates[0].date()} to {dates[-1].date()}")
print(f"  Mean Q:   {discharge.mean():.1f} m³/s")
print(f"  Max Q:    {discharge.max():.1f} m³/s")

# ── Run FloodChallenge ───────────────────────────────────────────────────

print("\n▸ Running flood challenge …")
from aquascope.challenges.flood import FloodChallenge

fc = FloodChallenge()
fc.load_dataframe(discharge.to_frame())

# Fit model and forecast 14 days
fc.fit(model_name="arima")
forecast = fc.forecast(days=14)
print(f"  Forecast peak: {forecast['yhat'].max():.1f} m³/s")

# Risk assessment
risk = fc.assess_risk(forecast)
print(f"\n▸ Risk Assessment:")
print(f"    Risk level:    {risk['risk_level']}")
print(f"    Description:   {risk['description']}")
print(f"    Peak forecast: {risk['peak_forecast']:.1f}")
print(f"    75th pctile:   {risk['threshold_75']:.1f}")
print(f"    95th pctile:   {risk['threshold_95']:.1f}")

if risk.get("return_periods"):
    print(f"\n▸ Return Periods:")
    for rp, val in sorted(risk["return_periods"].items()):
        print(f"    {rp}-year: {val:.1f} m³/s")

# ── Visualise ────────────────────────────────────────────────────────────

print("\n▸ Generating forecast plot …")
try:
    from aquascope.viz import plot_forecast

    # Show last 90 days of observed + 14-day forecast
    recent = discharge.iloc[-90:].to_frame()
    plot_forecast(
        observed=recent,
        forecast=forecast,
        title="14-Day Flood Forecast",
        ylabel="Discharge (m³/s)",
        save_path="data/flood_forecast.png",
    )
    print("  ✓ data/flood_forecast.png")
except ImportError:
    print("  Install viz deps: pip install aquascope[viz]")

print("\nDone!  CLI equivalent: aquascope solve 'Forecast flooding for my river'")
