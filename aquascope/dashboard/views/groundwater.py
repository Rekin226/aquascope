"""Groundwater page — SGI drought index, recharge estimation, aquifer tools."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _state

logger = logging.getLogger(__name__)

_LEVEL_HINTS = ("level", "gwl", "water_level", "head", "wtr_level")


def render() -> None:
    st.title("💧 Groundwater")
    st.markdown("Groundwater drought indices, recharge estimation, and aquifer hydraulics.")

    df = _state.require_data("Groundwater needs a well-level series — try the 40-yr streamflow demo as a stand-in.")
    if df is None:
        return

    prof = _state.profile(df)

    analysis = st.selectbox(
        "Analysis",
        ["Groundwater Drought (SGI)", "Recharge Estimation", "Aquifer Test Calculator"],
    )
    st.divider()

    try:
        if analysis == "Groundwater Drought (SGI)":
            _sgi_drought(df, prof)
        elif analysis == "Recharge Estimation":
            _recharge(df, prof)
        elif analysis == "Aquifer Test Calculator":
            _aquifer_test()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Groundwater analysis failed: {exc}")
        logger.exception("Groundwater error")


def _level_col(prof: _state.DataProfile, key: str) -> str | None:
    """Pick a water-level column, preferring names that look like levels."""
    cols = prof.numeric_cols
    if not cols:
        st.warning("No numeric columns found — this dataset has no water-level data to work with.")
        return None
    hinted = next((c for c in cols if any(h in c.lower() for h in _LEVEL_HINTS)), None)
    default = hinted or (prof.discharge_col if prof.discharge_col in cols else cols[0])
    return st.selectbox("Water-level column", cols, index=cols.index(default), key=key)


def _sgi_drought(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.groundwater.drought import drought_events, standardised_groundwater_index

    col = _level_col(prof, "sgi_col")
    if not col:
        return

    levels = _state.datetime_indexed(df, col, prof)
    if not isinstance(levels.index, pd.DatetimeIndex) or levels.empty:
        st.warning(
            "SGI needs a water-level series on a datetime index. "
            "This dataset doesn't have one — try the 40-yr streamflow demo below."
        )
        if st.button("Load 40-yr streamflow demo", key="sgi_demo"):
            _state.load_demo("streamflow")
            st.rerun()
        return

    min_per_month = st.slider("Minimum observations per calendar month", 3, 12, 5)

    try:
        sgi = standardised_groundwater_index(levels, min_per_month=min_per_month)
    except ValueError as exc:
        st.warning(f"Couldn't compute SGI: {exc}")
        return

    threshold = st.slider("Drought threshold (SGI ≤)", -3.0, 0.0, -1.0, 0.1)
    events = drought_events(sgi, threshold=threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current SGI", f"{sgi.dropna().iloc[-1]:.2f}" if not sgi.dropna().empty else "N/A")
    c2.metric("Drought events", str(len(events)))
    c3.metric("Worst SGI on record", f"{sgi.min():.2f}" if not sgi.dropna().empty else "N/A")

    st.plotly_chart(_charts.spi_bars(sgi.index, sgi, title="SGI Timeline — Groundwater Drought"))

    if events:
        event_rows = [
            {
                "Start": e.start,
                "End": e.end,
                "Duration (months)": e.duration,
                "Severity": round(e.severity, 2),
                "Peak SGI": round(e.peak, 2),
            }
            for e in events
        ]
        with st.expander("Drought events", expanded=True):
            st.dataframe(pd.DataFrame(event_rows), width="stretch")


def _recharge(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.groundwater.recharge import water_table_fluctuation

    col = _level_col(prof, "recharge_col")
    if not col:
        return

    levels = _state.datetime_indexed(df, col, prof)
    if not isinstance(levels.index, pd.DatetimeIndex) or len(levels) < 2:
        st.warning(
            "Water Table Fluctuation needs at least 2 dated water-level readings. "
            "This dataset doesn't qualify — try the 40-yr streamflow demo below."
        )
        if st.button("Load 40-yr streamflow demo", key="recharge_demo"):
            _state.load_demo("streamflow")
            st.rerun()
        return

    specific_yield = st.slider("Specific yield (Sy)", 0.01, 0.5, 0.15, 0.01)

    try:
        result = water_table_fluctuation(levels, specific_yield=specific_yield)
    except ValueError as exc:
        st.warning(f"Couldn't estimate recharge: {exc}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Estimated recharge", f"{result.value_mm_per_year:.1f} mm/yr")
    c2.metric("Total water-level rise", f"{result.metadata.get('total_rise_m', 0):.2f} m")

    st.caption(
        f"Method: {result.method} · Specific yield = {specific_yield:.3f} · "
        f"Period ≈ {result.metadata.get('period_years', 0):.1f} years"
    )

    st.plotly_chart(_charts.timeseries(levels.to_frame(name="level").reset_index(names="date"), x="date", y="level",
                                       title="Water-level record", ylab="Level (m)"))


def _aquifer_test() -> None:
    from aquascope.groundwater.aquifer import theis_drawdown

    st.markdown("**Theis (1935) — transient drawdown in a confined aquifer.**")

    c1, c2, c3, c4 = st.columns(4)
    T = c1.number_input("Transmissivity T (m²/day)", min_value=0.01, value=500.0, step=10.0)  # noqa: N806
    S = c2.number_input("Storativity S", min_value=1e-6, value=0.001, step=0.0001, format="%.5f")  # noqa: N806
    Q = c3.number_input("Pumping rate Q (m³/day)", min_value=0.01, value=1000.0, step=50.0)  # noqa: N806
    r = c4.number_input("Distance from well r (m)", min_value=0.01, value=100.0, step=10.0)

    t_days = st.slider("Time since pumping began (days)", 0.01, 100.0, 10.0, 0.01)

    try:
        s = theis_drawdown(T, S, Q, r, t_days)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't compute drawdown: {exc}")
        return

    st.metric("Drawdown", f"{float(s):.4f} m")
    st.caption("Theis solution assumes a confined, homogeneous, isotropic aquifer of infinite extent.")