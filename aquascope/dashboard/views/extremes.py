"""Extreme Events page — GEV / LP3 / Gumbel frequency analysis with bootstrap CIs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _demo, _state

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("🌀 Extreme Events")
    st.markdown(
        "Block-maxima frequency analysis for floods and droughts: fit **GEV**, "
        "**Log-Pearson III**, or **Gumbel** to annual maxima and estimate design "
        "return levels with bootstrap confidence bounds."
    )

    src = st.radio("Data source", ["Demo streamflow (40 yrs)", "Workspace dataset"], horizontal=True)

    series = None
    if src == "Demo streamflow (40 yrs)":
        demo = _demo.streamflow_40y()
        series = pd.Series(
            demo["discharge"].to_numpy(), index=pd.DatetimeIndex(demo["sample_datetime"]), name="discharge"
        )
        st.caption(f"Synthetic daily discharge: {len(series):,} days, {series.index.year.nunique()} years.")
    else:
        df = _state.require_data("Extreme-value analysis needs a long discharge record.")
        if df is None:
            return
        prof = _state.profile(df)
        cols = prof.numeric_cols
        if not cols:
            st.warning("No numeric columns in the workspace dataset.")
            return
        default = prof.discharge_col if prof.discharge_col in cols else cols[0]
        col = st.selectbox("Value column", cols, index=cols.index(default))
        series = _state.datetime_indexed(df, col, prof)

    st.divider()

    c1, c2, c3 = st.columns(3)
    dist_label = c1.selectbox("Distribution", ["GEV", "Log-Pearson III", "Gumbel"])
    dist = {"GEV": "gev", "Log-Pearson III": "lp3", "Gumbel": "gumbel"}[dist_label]
    conf = c2.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    n_boot = c3.select_slider("Bootstrap samples", options=[100, 200, 300, 500, 1000], value=300)

    rp_options = [2, 5, 10, 25, 50, 100, 200, 500]
    return_periods = st.multiselect("Return periods (years)", rp_options, default=[2, 5, 10, 25, 50, 100])
    if not return_periods:
        st.warning("Select at least one return period.")
        return

    if not st.button("📐 Run frequency analysis", type="primary"):
        return

    try:
        from aquascope.analysis.extreme_events import estimate_return_periods, fit_distribution

        if isinstance(series.index, pd.DatetimeIndex):
            n_years = series.resample("YE").max().dropna().shape[0]
        else:
            n_years = series.dropna().shape[0]
        if n_years < 3:
            st.error(
                f"Need at least 3 annual maxima — found {n_years}. "
                "Use the demo streamflow or a longer record."
            )
            return

        with st.spinner("Fitting distribution and bootstrapping return levels…"):
            fit = fit_distribution(series, distribution=dist)
            result = estimate_return_periods(
                series,
                distribution=dist,
                return_periods=tuple(float(t) for t in sorted(return_periods)),
                confidence_level=conf,
                n_bootstrap=int(n_boot),
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Frequency analysis failed: {exc}")
        logger.exception("Extreme events error")
        return

    st.subheader("Goodness of fit")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distribution", dist_label)
    m2.metric("AIC", f"{fit.aic:.1f}")
    m3.metric("KS p-value", f"{fit.ks_pvalue:.3f}")
    m4.metric("Annual maxima (n)", str(fit.n_samples))
    st.caption("Fitted parameters: " + ", ".join(f"{k} = {v:.4g}" for k, v in fit.parameters.items()))

    # Empirical annual maxima via Weibull plotting position
    if isinstance(series.index, pd.DatetimeIndex):
        amax = series.resample("YE").max().dropna().to_numpy()
    else:
        amax = series.dropna().to_numpy()
    amax_sorted = np.sort(amax)
    n = amax_sorted.size
    ranks = np.arange(1, n + 1)
    emp_t = (n + 1) / (n + 1 - ranks)

    st.plotly_chart(
        _charts.return_curve(
            result.return_periods,
            result.return_levels,
            lower=result.lower_bound,
            upper=result.upper_bound,
            emp_periods=emp_t,
            emp_values=amax_sorted,
            conf=conf,
            title=f"{dist_label} return-level curve",
        )
    )

    table = pd.DataFrame(
        {
            "Return period (yr)": result.return_periods,
            "Return level": [round(x, 2) for x in result.return_levels],
            f"Lower ({int(conf * 100)}%)": [round(x, 2) for x in result.lower_bound],
            f"Upper ({int(conf * 100)}%)": [round(x, 2) for x in result.upper_bound],
        }
    )
    st.dataframe(table, width="stretch")
    st.download_button(
        "⬇️ Download return levels (CSV)",
        data=table.to_csv(index=False),
        file_name=f"aquascope_return_levels_{dist}.csv",
        mime="text/csv",
    )
