"""Quality Alerts page — WHO guideline screening with status flags."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _insights, _state

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("⚠️ Water Quality Alerts")
    st.markdown("Screen water-quality parameters against WHO guideline thresholds.")

    df = _state.require_data("Alerts need long-format quality data (`parameter` + `value` columns).")
    if df is None:
        return

    prof = _state.profile(df)

    with st.expander("📋 WHO guideline reference", expanded=False):
        ref_rows = [
            {"Parameter": param, "Min": str(lo), "Max": "∞" if hi == float("inf") else str(hi), "Unit": unit}
            for param, (lo, hi, unit) in _insights.WHO_GUIDELINES.items()
        ]
        st.dataframe(pd.DataFrame(ref_rows), width="stretch")

    if not prof.has_params:
        st.warning(
            "Expected `parameter` and `value` columns (long-format water-quality data). "
            "The water-quality demo dataset has the right shape."
        )
        return

    rows = _insights.who_exceedances(df, prof)
    if not rows:
        st.info(
            "No WHO-monitored parameters found in this dataset. Monitored: "
            + ", ".join(_insights.WHO_GUIDELINES.keys())
        )
        return

    alerts = [r for r in rows if r["status_plain"] == "Alert"]
    warnings = [r for r in rows if r["status_plain"] == "Warning"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Parameters checked", len(rows))
    c2.metric("🔴 Alerts (>10%)", len(alerts))
    c3.metric("🟡 Warnings (>0%)", len(warnings))

    if alerts:
        st.error("🔴 **Alert:** " + ", ".join(a["parameter"] for a in alerts) + " exceed the 10% threshold.")
    if warnings:
        st.warning("🟡 **Warning:** " + ", ".join(w["parameter"] for w in warnings) + " show some exceedances.")
    if not alerts and not warnings:
        st.success("✅ All monitored parameters within WHO guidelines.")

    st.plotly_chart(_charts.exceedance_bar(rows))

    table = pd.DataFrame(
        [
            {
                "Parameter": r["parameter"],
                "Guideline": r["rule"],
                "Samples": r["n"],
                "Exceedances": r["n_exceed"],
                "Exceedance %": r["pct"],
                "Status": {"OK": "🟢 OK", "Warning": "🟡 Warning", "Alert": "🔴 Alert"}[r["status_plain"]],
            }
            for r in rows
        ]
    )
    st.dataframe(table, width="stretch")

    st.divider()
    st.subheader("Deep-dive: full challenge analysis")
    st.caption("Runs `WaterQualityChallenge` for per-parameter statistics against WHO guidelines.")

    default_site = df[prof.station_col].iloc[0] if prof.station_col else "SITE-001"
    site_id = st.text_input("Site ID", value=str(default_site))

    if st.button("Run full challenge analysis", key="btn_wq_challenge"):
        with st.spinner("Running water-quality challenge analysis…"):
            try:
                from aquascope.challenges import WaterQualityChallenge

                wq = WaterQualityChallenge(site_id=site_id)

                param_dfs: dict = {}
                for param_name in df[prof.param_col].unique():
                    subset = df[df[prof.param_col] == param_name].copy()
                    if prof.datetime_col:
                        subset.index = pd.to_datetime(subset[prof.datetime_col], errors="coerce")
                    param_dfs[param_name] = subset[[prof.value_col]].rename(columns={prof.value_col: "value"})

                wq.load_dataframes(param_dfs)
                exceedances = wq.check_who_guidelines()

                if exceedances.empty:
                    st.info("No WHO guideline data available for the loaded parameters.")
                else:
                    exceeded = exceedances[exceedances["status"] == "EXCEEDANCE"]
                    if not exceeded.empty:
                        st.warning(f"⚠️ Found exceedances in **{len(exceeded)}** parameter(s)")
                    else:
                        st.success("✅ All parameters within WHO guidelines")
                    st.dataframe(
                        exceedances[
                            ["variable", "n_measurements", "mean", "n_exceedances",
                             "pct_exceedances", "guideline_low", "guideline_high", "status"]
                        ],
                        width="stretch",
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Challenge analysis failed: {exc}")
                logger.exception("Water quality challenge error")
