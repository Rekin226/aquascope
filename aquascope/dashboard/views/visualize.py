"""Visualize page — interactive Plotly charts and station maps."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from aquascope.dashboard import _charts, _insights, _state

logger = logging.getLogger(__name__)

_PLOTS = {
    "timeseries": "📈 Time series",
    "boxplot": "📦 Box plot",
    "heatmap": "🗺️ Correlation heatmap",
    "histogram": "📊 Distribution",
    "who_exceedances": "⚠️ WHO exceedances",
    "station_map": "📍 Station map",
    "fdc": "🌊 Flow duration curve",
    "hydrograph": "💧 Hydrograph",
    "spi_timeline": "🏜️ SPI timeline",
    "return_periods": "🔁 Return periods",
}


def render() -> None:
    st.title("📈 Visualize")
    st.markdown("Fully interactive charts — hover for values, drag to zoom, click legends to toggle series.")

    df = _state.require_data("Visualization needs a dataset in the workspace.")
    if df is None:
        return

    prof = _state.profile(df)

    plot_key = st.selectbox("Chart", list(_PLOTS.keys()), format_func=lambda k: _PLOTS[k])

    try:
        if plot_key == "timeseries":
            _timeseries(df, prof)
        elif plot_key == "boxplot":
            _boxplot(df, prof)
        elif plot_key == "heatmap":
            st.plotly_chart(_charts.corr_heatmap(df, "Correlation matrix"))
        elif plot_key == "histogram":
            col = _num_col(df, prof)
            if col:
                st.plotly_chart(_charts.histogram(df[col].dropna(), f"Distribution — {col}", col))
        elif plot_key == "who_exceedances":
            _who(df, prof)
        elif plot_key == "station_map":
            _map(df, prof)
        elif plot_key == "fdc":
            col = _num_col(df, prof, prefer_discharge=True)
            if col:
                st.plotly_chart(_charts.fdc(df[col].dropna()))
        elif plot_key == "hydrograph":
            _hydrograph(df, prof)
        elif plot_key == "spi_timeline":
            _spi(df, prof)
        elif plot_key == "return_periods":
            _return_periods(df, prof)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Plot failed: {exc}")
        logger.exception("Visualization error")


def _num_col(df: pd.DataFrame, prof: _state.DataProfile, prefer_discharge: bool = False) -> str | None:
    cols = prof.numeric_cols
    if not cols:
        st.warning("No numeric columns found.")
        return None
    default = None
    if prefer_discharge and prof.discharge_col:
        default = prof.discharge_col
    elif prof.value_col in cols:
        default = prof.value_col
    return st.selectbox("Column", cols, index=cols.index(default) if default in cols else 0)


def _timeseries(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    if not prof.has_time:
        st.warning("No datetime column detected — a time series needs one.")
        return
    col = _num_col(df, prof)
    if not col:
        return
    plot_df = df.copy()
    plot_df[prof.datetime_col] = pd.to_datetime(plot_df[prof.datetime_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[prof.datetime_col, col]).sort_values(prof.datetime_col)

    color = None
    if prof.has_params and col == prof.value_col and df[prof.param_col].nunique() > 1:
        n_params = int(df[prof.param_col].nunique())
        if n_params <= 8:
            color = prof.param_col
        else:
            keep = st.multiselect(
                "Parameters (max 8 shown)", sorted(df[prof.param_col].astype(str).unique()),
                default=sorted(df[prof.param_col].astype(str).unique())[:4],
                max_selections=8,
            )
            plot_df = plot_df[plot_df[prof.param_col].astype(str).isin(keep)]
            color = prof.param_col

    st.plotly_chart(_charts.timeseries(plot_df, prof.datetime_col, col, color=color, title=f"{col} over time"))


def _boxplot(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    num_cols = prof.numeric_cols
    cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() <= 30]
    if not num_cols or not cat_cols:
        st.warning("Need one numeric column and one categorical column (≤30 groups) for a box plot.")
        return
    c1, c2 = st.columns(2)
    value = c1.selectbox("Value", num_cols, index=num_cols.index(prof.value_col) if prof.value_col in num_cols else 0)
    default_grp = prof.param_col if prof.param_col in cat_cols else cat_cols[0]
    group = c2.selectbox("Group by", cat_cols, index=cat_cols.index(default_grp))
    st.plotly_chart(_charts.box(df, value, group, f"{value} by {group}"))


def _who(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    rows = _insights.who_exceedances(df, prof)
    if not rows:
        st.info("No WHO-monitored parameters found (needs long-format `parameter` + `value` columns).")
        return
    st.plotly_chart(_charts.exceedance_bar(rows))
    st.caption("🟢 OK · 🟡 Warning (any exceedance) · 🔴 Alert (>10% of samples)")


def _map(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    if not prof.has_geo:
        st.warning("No latitude/longitude columns detected.")
        return
    st.plotly_chart(
        _charts.station_map(df, prof.lat_col, prof.lon_col, hover=prof.station_col)
    )
    n = df.dropna(subset=[prof.lat_col, prof.lon_col]).drop_duplicates(
        subset=[c for c in (prof.lat_col, prof.lon_col, prof.station_col) if c]
    ).shape[0]
    st.caption(f"{n} unique station location(s). Scroll to zoom, drag to pan.")


def _hydrograph(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    col = _num_col(df, prof, prefer_discharge=True)
    if not col:
        return
    series = _state.datetime_indexed(df, col, prof)
    st.plotly_chart(_charts.hydrograph(series.index, series, title=f"Hydrograph — {col}"))


def _spi(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    spi_cols = [c for c in prof.numeric_cols if "spi" in c.lower()] or prof.numeric_cols
    col = st.selectbox("SPI column", spi_cols)
    series = _state.datetime_indexed(df, col, prof)
    st.plotly_chart(_charts.spi_bars(series.index, series))
    if "spi" not in col.lower():
        st.caption("Tip: this chart is designed for SPI values — the selected column is plotted as-is.")


def _return_periods(df: pd.DataFrame, prof: _state.DataProfile) -> None:
    col = _num_col(df, prof, prefer_discharge=True)
    if not col:
        return
    series = _state.datetime_indexed(df, col, prof)
    from aquascope.hydrology import fit_gev

    with st.spinner("Fitting GEV distribution…"):
        result = fit_gev(series)
    periods = list(result.return_periods.keys())
    levels = list(result.return_periods.values())
    st.plotly_chart(
        _charts.return_curve(periods, levels, title=f"GEV return levels — {col}")
    )
    st.caption("For LP3/Gumbel fits and bootstrap confidence bounds, use the **Extreme Events** page.")
