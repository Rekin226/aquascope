"""Hydrology Lab page — FDC, baseflow, recession, flood frequency, signatures."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _state

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("🌊 Hydrology Lab")
    st.markdown("Interactive hydrological analysis with live parameter controls.")

    df = _state.require_data("Hydrology needs a discharge series — the 40-yr streamflow demo is ideal.")
    if df is None:
        return

    prof = _state.profile(df)

    analysis = st.selectbox(
        "Analysis",
        ["Flow Duration Curve", "Baseflow Separation", "Recession Analysis", "Flood Frequency", "Flow Signatures"],
    )
    st.divider()

    try:
        if analysis == "Flow Duration Curve":
            _fdc(df, prof)
        elif analysis == "Baseflow Separation":
            _baseflow(df, prof)
        elif analysis == "Recession Analysis":
            _recession(df, prof)
        elif analysis == "Flood Frequency":
            _flood_freq(df, prof)
        elif analysis == "Flow Signatures":
            _signatures(df, prof)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Hydrology analysis failed: {exc}")
        logger.exception("Hydrology error")


def _discharge_col(prof: _state.DataProfile, key: str) -> str | None:
    cols = prof.numeric_cols
    if not cols:
        st.warning("No numeric columns found.")
        return None
    default = prof.discharge_col if prof.discharge_col in cols else cols[0]
    return st.selectbox("Discharge column", cols, index=cols.index(default), key=key)


def _fdc(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.hydrology import flow_duration_curve

    col = _discharge_col(prof, "fdc_col")
    if not col:
        return
    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    result = flow_duration_curve(q)
    q50 = result.percentiles.get(50, float("nan"))
    q95 = result.percentiles.get(95, float("nan"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Q50 (median flow)", f"{q50:.3f}")
    c2.metric("Q95 (low flow)", f"{q95:.3f}")
    c3.metric("Days of record", f"{len(q):,}")

    st.plotly_chart(_charts.fdc(q))


def _baseflow(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.hydrology import eckhardt, lyne_hollick, ukih

    col = _discharge_col(prof, "bf_col")
    if not col:
        return
    method = st.radio("Method", ["Lyne-Hollick", "Eckhardt", "UKIH"], horizontal=True)

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    if method == "Lyne-Hollick":
        c1, c2 = st.columns(2)
        alpha = c1.slider("Filter parameter (α)", 0.90, 0.99, 0.925, 0.005)
        passes = c2.slider("Number of passes", 1, 5, 3)
        result = lyne_hollick(q, alpha=alpha, n_passes=passes)
    elif method == "Eckhardt":
        c1, c2 = st.columns(2)
        alpha = c1.slider("Filter parameter (α)", 0.90, 0.99, 0.925, 0.005)
        bfi_max = c2.slider("BFI_max", 0.1, 1.0, 0.8, 0.05)
        result = eckhardt(q, alpha=alpha, bfi_max=bfi_max)
    else:
        block_size = st.slider("Block size (days)", 3, 10, 5)
        st.caption("UKIH picks smoothed minima as turning points and interpolates baseflow between them.")
        result = ukih(q, block_size=block_size)

    st.metric("Baseflow Index (BFI)", f"{result.bfi:.3f}")

    dates = _state.datetime_indexed(df, col, prof).index if prof.has_time else result.df.index
    st.plotly_chart(
        _charts.hydrograph(
            dates[: len(result.df)], result.df["total"], result.df["baseflow"], title=f"Baseflow separation — {method}"
        )
    )


def _recession(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.hydrology import recession_analysis

    col = _discharge_col(prof, "rec_col")
    if not col:
        return
    min_length = st.slider("Minimum recession length (days)", 3, 30, 5)

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    result = recession_analysis(q, min_length=min_length)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recession constant (K)", f"{result.recession_constant:.4f}" if result.recession_constant else "N/A")
    c2.metric("Segments found", str(len(result.segments)))
    c3.metric("R²", f"{result.r_squared:.3f}" if result.r_squared else "N/A")

    if result.segments:
        seg_data = [
            {
                "Start": s.start,
                "End": s.end,
                "Duration (days)": (s.end - s.start).days if hasattr(s.end, "days") else len(s.discharge),
            }
            for s in result.segments
        ]
        st.dataframe(pd.DataFrame(seg_data), width="stretch")


def _flood_freq(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.hydrology import fit_gev

    col = _discharge_col(prof, "ff_col")
    if not col:
        return
    q = _state.datetime_indexed(df, col, prof)
    if q.empty:
        st.warning("Selected column has no data.")
        return

    with st.spinner("Fitting GEV to annual maxima…"):
        result = fit_gev(q)

    shape, loc, scale = result.params if len(result.params) == 3 else (0.0, 0.0, 1.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Shape (ξ)", f"{shape:.4f}")
    c2.metric("Location (μ)", f"{loc:.2f}")
    c3.metric("Scale (σ)", f"{scale:.2f}")

    if result.return_periods:
        periods = list(result.return_periods.keys())
        levels = list(result.return_periods.values())
        st.plotly_chart(_charts.return_curve(periods, levels, title="GEV return levels"))
        table = pd.DataFrame({"Return period (yr)": periods, "Discharge": [round(v, 2) for v in levels]})
        with st.expander("Return-level table"):
            st.dataframe(table, width="stretch")

    st.caption("Need Log-Pearson III / Gumbel or bootstrap confidence bounds? Use the **Extreme Events** page.")


def _signatures(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    from aquascope.hydrology import compute_signatures

    col = _discharge_col(prof, "sig_col")
    if not col:
        return
    q = _state.datetime_indexed(df, col, prof)

    if not isinstance(q.index, pd.DatetimeIndex) or len(q) < 365:
        st.warning(
            f"Flow signatures need ≥365 daily values on a datetime index — got {len(q):,}. "
            "Load the 40-year streamflow demo below."
        )
        if st.button("Load 40-yr streamflow demo", key="sig_demo"):
            _state.load_demo("streamflow")
            st.rerun()
        return

    with st.spinner("Computing hydrological signatures…"):
        report = compute_signatures(q)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean flow", f"{report.mean_flow:.2f}")
    c2.metric("Baseflow Index", f"{report.baseflow_index:.3f}")
    c3.metric("Flashiness (R-B)", f"{report.flashiness_index:.3f}")
    c4.metric("Q5/Q95 ratio", f"{report.q5_q95_ratio:.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Q5 (high flow)", f"{report.q5:.2f}")
    c6.metric("Q95 (low flow)", f"{report.q95:.2f}")
    c7.metric("Peak month", str(report.peak_month))
    c8.metric("Seasonality", f"{report.seasonality_index:.3f}")

    sig_rows = [
        ("Mean flow", report.mean_flow),
        ("Median flow", report.median_flow),
        ("Coefficient of variation", report.cv),
        ("IQR", report.iqr),
        ("High-flow frequency (/yr)", report.high_flow_frequency),
        ("High-flow duration (days)", report.high_flow_duration),
        ("Low-flow frequency (/yr)", report.low_flow_frequency),
        ("Low-flow duration (days)", report.low_flow_duration),
        ("Zero-flow fraction", report.zero_flow_fraction),
        ("Rising-limb density", report.rising_limb_density),
        ("Mean recession constant", report.mean_recession_constant),
    ]
    sig_df = pd.DataFrame([{"Signature": n, "Value": round(v, 4) if v is not None else None} for n, v in sig_rows])
    with st.expander("All signatures", expanded=True):
        st.dataframe(sig_df, width="stretch")
