"""Agricultural Water page — FAO-56 ET₀, crop coefficients, irrigation scheduling."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _demo

logger = logging.getLogger(__name__)

_AGRI_CROPS = [
    "maize", "wheat_winter", "rice_paddy", "soybean", "potato", "tomato",
    "cotton", "sugarcane", "barley", "onion", "cabbage", "sunflower",
    "citrus", "grape",
]


def render() -> None:
    st.title("🌾 Agricultural Water")
    st.markdown(
        "FAO-56 **Penman-Monteith** reference ET₀, crop water requirements, and "
        "irrigation scheduling — with **single (Kc)** or **dual (Kcb + Ke)** "
        "crop-coefficient methods."
    )

    st.subheader("1 · Weather & site")
    weather, precip = _demo.weather_season()
    st.caption(
        f"Using a demo growing-season weather record ({len(weather)} days from "
        f"{weather.index[0].date()}). Swap in real Open-Meteo data via the Collect page in production."
    )
    c1, c2 = st.columns(2)
    latitude = c1.number_input("Latitude (°)", -90.0, 90.0, 25.0, 0.5)
    elevation = c2.number_input("Elevation (m)", -100.0, 5000.0, 10.0, 10.0)

    st.subheader("2 · Crop & scheduling")
    c3, c4, c5 = st.columns(3)
    crop = c3.selectbox("Crop", _AGRI_CROPS, index=0)
    planting = c4.date_input("Planting date", value=weather.index[0].date())
    efficiency = c5.slider("Irrigation efficiency", 0.4, 1.0, 0.7, 0.05)

    method_label = st.radio(
        "Crop-coefficient method",
        ["Single (Kc)", "Dual (Kcb + Ke)"],
        horizontal=True,
        help="Dual splits ETc into basal transpiration (Kcb) and soil evaporation (Ke).",
    )
    method = "dual" if method_label.startswith("Dual") else "single"

    kc_max, few, kr = 1.20, 1.0, 1.0
    if method == "dual":
        d1, d2, d3 = st.columns(3)
        kc_max = d1.slider("Kc_max (after wetting)", 1.0, 1.4, 1.20, 0.05)
        few = d2.slider("Exposed-wetted fraction (few)", 0.1, 1.0, 1.0, 0.05)
        kr = d3.slider("Evaporation reduction (Kr)", 0.1, 1.0, 1.0, 0.05)

    if not st.button("💧 Compute irrigation schedule", type="primary"):
        return

    try:
        from aquascope.agri import irrigation_schedule
        from aquascope.agri.eto import penman_monteith_series

        with st.spinner("Computing ET₀ (Penman-Monteith) and scheduling…"):
            eto = penman_monteith_series(weather, latitude=latitude, elevation=elevation)
            sched = irrigation_schedule(
                eto, precip, crop, planting,
                efficiency=efficiency, method=method,
                kc_max=kc_max, few=few, kr=kr,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Agricultural water computation failed: {exc}")
        logger.exception("Agri water error")
        return

    st.subheader("Season summary")
    total_etc = float(sched["etc"].sum())
    total_net = float(sched["net_irrigation"].sum())
    total_gross = float(sched["gross_irrigation"].sum())
    total_rain = float(sched["effective_rain"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season ET₀", f"{float(eto.mean()):.2f} mm/d")
    m2.metric("Crop ET (ETc)", f"{total_etc:.0f} mm")
    m3.metric("Net irrigation", f"{total_net:.0f} mm")
    m4.metric("Gross irrigation", f"{total_gross:.0f} mm")
    st.caption(
        f"Effective rainfall over season: {total_rain:.0f} mm · "
        f"Season length: {len(sched)} days · Method: {method_label}"
    )

    dates = pd.to_datetime(sched["date"])

    tab_demand, tab_kc, tab_sched = st.tabs(["💧 Water demand", "📈 Crop coefficients", "🗓️ Schedule table"])

    with tab_demand:
        st.plotly_chart(_charts.water_demand(dates, sched, crop))
        st.plotly_chart(_charts.cumulative_irrigation(dates, sched))

    with tab_kc:
        st.plotly_chart(_charts.kc_curves(dates, sched, method, crop))
        st.caption("Stages: initial → development → mid-season → late-season (FAO-56).")

    with tab_sched:
        st.dataframe(sched, width="stretch")
        st.download_button(
            "⬇️ Download schedule (CSV)",
            data=sched.to_csv(index=False),
            file_name=f"aquascope_irrigation_{crop}_{method}.csv",
            mime="text/csv",
        )
