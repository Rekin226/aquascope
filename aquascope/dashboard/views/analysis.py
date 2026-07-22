"""Analyze & Clean page — EDA, quality assessment, and preprocessing."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _insights, _state

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("🔬 Analyze & Clean")
    st.markdown("Automated EDA, quality scoring, and one-click preprocessing for the workspace dataset.")

    df = _state.require_data("Analysis needs a dataset in the workspace.")
    if df is None:
        return

    _insights.render_panel(df, key_prefix="analysis")

    tab_eda, tab_quality, tab_prep = st.tabs(["📊 EDA report", "🔍 Quality assessment", "🧹 Preprocess"])

    with tab_eda:
        _render_eda(df)
    with tab_quality:
        _render_quality(df)
    with tab_prep:
        _render_preprocess(df)


def _render_eda(df: pd.DataFrame) -> None:
    try:
        from aquascope.analysis.eda import generate_eda_report

        with st.spinner("Generating EDA report…"):
            report = generate_eda_report(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{report.n_records:,}")
        c2.metric("Stations", report.n_stations)
        c3.metric("Parameters", report.n_parameters)
        c4.metric("Completeness", f"{report.completeness_pct:.1f}%")

        if report.parameters:
            st.markdown("**Parameter statistics**")
            param_rows = [
                {
                    "Parameter": p.name,
                    "Count": p.count,
                    "Mean": round(p.mean, 3) if p.mean is not None else None,
                    "Std": round(p.std, 3) if p.std is not None else None,
                    "Min": p.min,
                    "Max": p.max,
                    "Outliers": p.outlier_count,
                }
                for p in report.parameters
            ]
            st.dataframe(pd.DataFrame(param_rows), width="stretch")

        prof = _state.profile(df)
        if prof.value_col:
            from aquascope.dashboard import _charts

            idx = prof.numeric_cols.index(prof.value_col) if prof.value_col in prof.numeric_cols else 0
            col = st.selectbox("Distribution of", prof.numeric_cols, index=idx)
            st.plotly_chart(_charts.histogram(df[col].dropna(), title=f"Distribution — {col}", xlab=col))

    except Exception as exc:  # noqa: BLE001
        st.error(f"EDA failed: {exc}")
        logger.exception("EDA error")


def _render_quality(df: pd.DataFrame) -> None:
    try:
        from aquascope.analysis.quality import assess_quality

        with st.spinner("Running quality assessment…"):
            report = assess_quality(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Completeness", f"{report.completeness_pct:.1f}%")
        c2.metric("Duplicates", report.n_duplicates)
        c3.metric("Recommended steps", len(report.recommended_steps))

        if report.recommended_steps:
            st.markdown("**Recommended preprocessing** — pre-selected in the Preprocess tab.")
            for step in report.recommended_steps:
                st.markdown(f"- `{step}`")
            st.session_state["_recommended_steps"] = list(report.recommended_steps)
        else:
            st.success("✅ No preprocessing needed — the dataset looks clean.")

    except Exception as exc:  # noqa: BLE001
        st.error(f"Quality assessment failed: {exc}")
        logger.exception("Quality assessment error")


def _render_preprocess(df: pd.DataFrame) -> None:
    available = ["remove_duplicates", "fill_missing", "remove_outliers", "normalize", "resample_daily"]
    default = [s for s in st.session_state.get("_recommended_steps", []) if s in available]
    selected = st.multiselect(
        "Steps to apply (in order)",
        available,
        default=default,
        help="Run Quality assessment first to pre-select the recommended steps.",
    )

    if st.button("🧹 Apply preprocessing", type="primary", disabled=not selected):
        try:
            from aquascope.analysis.quality import preprocess

            with st.spinner("Preprocessing…"):
                cleaned = preprocess(df, steps=selected)
            _state.set_data(
                cleaned, st.session_state.get(_state.SOURCE_KEY, "session"), f"{_state.source_label()} (cleaned)"
            )
            st.success(f"✅ Preprocessed: {len(cleaned):,} records (was {len(df):,}). Workspace updated.")
            st.dataframe(cleaned.head(20), width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Preprocessing failed: {exc}")
            logger.exception("Preprocessing error")
