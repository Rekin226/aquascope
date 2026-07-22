"""Demo dataset builders for the AquaScope dashboard.

Every generator is deterministic (fixed seed) and cached with
``st.cache_data`` so page switches never re-simulate the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def water_quality() -> pd.DataFrame:
    """180 days of synthetic water-quality readings (pH, DO, turbidity, nitrate).

    Includes a ``discharge`` column and station coordinates so every
    dashboard page has something to work with.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    params = ["ph", "dissolved_oxygen", "turbidity", "nitrate"]

    t = np.linspace(0, 4 * np.pi, len(dates))

    rows = []
    for i, date in enumerate(dates):
        discharge = max(0.1, 5.0 + 3.0 * np.sin(t[i]) + rng.normal(0, 0.5))
        for param in params:
            base = {
                "ph": 7.2 + 0.5 * np.sin(t[i]),
                "dissolved_oxygen": 7.0 + 2.0 * np.cos(t[i]),
                "turbidity": 3.5 + 2.0 * np.abs(np.sin(t[i])),
                "nitrate": 30.0 + 20.0 * np.sin(t[i] + 1),
            }[param]
            noise = rng.normal(0, {"ph": 0.3, "dissolved_oxygen": 0.8, "turbidity": 0.5, "nitrate": 5.0}[param])
            rows.append(
                {
                    "sample_datetime": date,
                    "station_id": "DEMO-001",
                    "station_name": "Tamsui River Demo Station",
                    "parameter": param,
                    "value": round(float(base + noise), 3),
                    "discharge": round(float(discharge), 3),
                    "latitude": 25.17,
                    "longitude": 121.44,
                    "source": "demo",
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def streamflow_40y(n_years: int = 40, seed: int = 7) -> pd.DataFrame:
    """Multi-decade daily-discharge record for frequency & signature analysis.

    Long enough (≥3 annual maxima, ≥365 days) for extreme-value fitting and
    hydrological signatures. Returned as a DataFrame with ``sample_datetime``
    and ``discharge`` columns so it drops straight into the shared workspace.
    """
    rng = np.random.default_rng(seed)
    end_year = 2023
    idx = pd.date_range(f"{end_year - n_years + 1}-01-01", f"{end_year}-12-31", freq="D")
    doy = idx.dayofyear.to_numpy()

    # Seasonal baseflow (wet summer monsoon peak ~ day 200)
    seasonal = 6.0 + 4.0 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    baseflow = np.clip(seasonal, 1.0, None)

    # Stochastic storm quickflow, amplified in the wet season
    wet = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * (doy - 100) / 365.25), 0, 1)
    storms = rng.gamma(shape=1.3, scale=3.0, size=len(idx)) * wet
    flood_mask = rng.random(len(idx)) < 0.01
    storms = storms + flood_mask * rng.gamma(shape=2.0, scale=12.0, size=len(idx)) * wet

    flow = np.clip(baseflow + storms, 0.1, None)
    return pd.DataFrame(
        {
            "sample_datetime": idx,
            "station_id": "DEMO-FLOW",
            "station_name": "Demo River Gauge",
            "discharge": np.round(flow, 3),
            "latitude": 24.8,
            "longitude": 121.2,
            "source": "demo_streamflow",
        }
    )


@st.cache_data(show_spinner=False)
def weather_season(days: int = 150, start: str = "2023-03-01", seed: int = 11):
    """Growing-season daily weather DataFrame + precipitation for FAO-56.

    Returns ``(weather_df, precip_series)`` with the columns required by
    :func:`aquascope.agri.eto.penman_monteith_series`.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    warming = 6.0 * np.sin(np.pi * t / days)  # warms toward mid-season

    t_min = 14.0 + warming + rng.normal(0, 1.0, days)
    t_max = 26.0 + warming + rng.normal(0, 1.2, days)
    rh_max = np.clip(82.0 + rng.normal(0, 4.0, days), 50, 100)
    rh_min = np.clip(45.0 + rng.normal(0, 5.0, days), 15, rh_max - 5)
    wind = np.clip(2.0 + rng.normal(0, 0.4, days), 0.5, None)
    solar = np.clip(20.0 + 6.0 * np.sin(np.pi * t / days) + rng.normal(0, 1.5, days), 5, None)

    weather = pd.DataFrame(
        {
            "t_min": np.round(t_min, 2),
            "t_max": np.round(t_max, 2),
            "rh_min": np.round(rh_min, 1),
            "rh_max": np.round(rh_max, 1),
            "wind_speed": np.round(wind, 2),
            "solar_radiation": np.round(solar, 2),
        },
        index=idx,
    )
    rain_prob = 0.12 + 0.18 * np.clip(np.sin(np.pi * t / days), 0, 1)
    precip = np.where(rng.random(days) < rain_prob, rng.gamma(2.0, 6.0, days), 0.0)
    precip_series = pd.Series(np.round(precip, 2), index=idx, name="precipitation")
    return weather, precip_series
