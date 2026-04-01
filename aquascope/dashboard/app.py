"""
AquaScope Interactive Dashboard — Streamlit application.

Launch with::

    streamlit run aquascope/dashboard/app.py
    # or via CLI:
    aquascope dashboard
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_SOURCES: list[tuple[str, str]] = [
    ("taiwan_moenv", "Taiwan MOENV (Water Quality)"),
    ("taiwan_wra_level", "Taiwan WRA (Water Level)"),
    ("taiwan_wra_reservoir", "Taiwan WRA (Reservoir)"),
    ("usgs", "USGS (US Geological Survey)"),
    ("sdg6", "UN SDG 6 Indicators"),
    ("gemstat", "GEMStat (Global Water Quality)"),
    ("taiwan_civil_iot", "Taiwan Civil IoT"),
    ("wqp", "WQP (Water Quality Portal)"),
    ("openmeteo", "Open-Meteo (Weather/Hydro)"),
    ("copernicus", "Copernicus Climate Data"),
]

_PLOT_TYPES: list[tuple[str, str]] = [
    ("timeseries", "📈 Time Series"),
    ("boxplot", "📦 Box Plot"),
    ("heatmap", "🗺️ Heatmap"),
    ("who_exceedances", "⚠️ WHO Exceedances"),
    ("station_map", "📍 Station Map"),
    ("fdc", "🌊 Flow Duration Curve"),
    ("hydrograph", "💧 Hydrograph"),
    ("spi_timeline", "🏜️ SPI Timeline"),
    ("return_periods", "🔁 Return Periods"),
]

WHO_GUIDELINES: dict[str, tuple[float, float, str]] = {
    "ph": (6.5, 8.5, "pH units"),
    "dissolved_oxygen": (5.0, float("inf"), "mg/L"),
    "turbidity": (0, 5.0, "NTU"),
    "nitrate": (0, 50.0, "mg/L"),
    "e_coli": (0, 0, "CFU/100mL"),
    "arsenic": (0, 0.01, "mg/L"),
    "lead": (0, 0.01, "mg/L"),
    "mercury": (0, 0.001, "mg/L"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_streamlit():
    """Import and return streamlit, raising a helpful error if missing."""
    try:
        import streamlit as st

        return st
    except ImportError as exc:
        msg = (
            "Streamlit is required for the dashboard. "
            "Install it with:  pip install aquascope[dashboard]"
        )
        raise ImportError(msg) from exc


def _samples_to_dataframe(records: list) -> pd.DataFrame:
    """Convert a list of Pydantic schema objects to a DataFrame."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.model_dump() for r in records])


def _load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame."""
    import pandas as pd

    content = uploaded_file.getvalue().decode("utf-8")
    return pd.read_csv(StringIO(content))


def _load_json(uploaded_file) -> pd.DataFrame:
    """Read an uploaded JSON file into a DataFrame."""
    import pandas as pd

    content = uploaded_file.getvalue().decode("utf-8")
    return pd.read_json(StringIO(content))


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------


def page_home() -> None:
    """Render the Home overview page."""
    st = _require_streamlit()

    st.title("🌊 AquaScope Dashboard")
    st.markdown(
        """
        **AquaScope** is an open-source water data aggregation toolkit with
        AI-powered research methodology recommendations.

        ### Features
        - 📊 **10 data collectors** — Taiwan MOENV, USGS, GEMStat, WQP, SDG6, Open-Meteo, Copernicus & more
        - 🔬 **Automated EDA & quality assessment** on collected data
        - 🤖 **AI recommender** — rule-based + optional LLM-enhanced methodology suggestions
        - 🌊 **Hydrology toolkit** — flow duration curves, baseflow separation, recession analysis, flood frequency
        - 📈 **Publication-quality visualisations** with matplotlib/seaborn/folium
        - ⚠️ **Water quality alerts** against WHO/EPA/EU thresholds
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Sources", "10")
    col2.metric("Plot Types", str(len(_PLOT_TYPES)))
    col3.metric("AI Methodologies", "26")

    st.divider()
    st.subheader("Quick Start")
    st.markdown(
        """
        1. Go to **📊 Data Collection** to fetch water data from any source.
        2. Explore your data with **🔬 Analysis** (EDA + quality checks).
        3. Create charts in **📈 Visualization**.
        4. Run hydrological models in **🌊 Hydrology**.
        5. Get AI methodology suggestions in **🤖 AI Recommender**.
        """
    )

    if "collected_data" in st.session_state and st.session_state["collected_data"] is not None:
        df = st.session_state["collected_data"]
        st.success(f"✅ Active dataset: **{len(df)} records**, {df.columns.tolist()}")


# ---------------------------------------------------------------------------
# Page: Data Collection
# ---------------------------------------------------------------------------


def page_data_collection() -> None:
    """Render the Data Collection page."""
    st = _require_streamlit()

    st.title("📊 Data Collection")
    st.markdown("Fetch water data from any of AquaScope's supported sources.")

    source_key = st.selectbox(
        "Data Source",
        options=[k for k, _ in _DATA_SOURCES],
        format_func=lambda k: dict(_DATA_SOURCES)[k],
    )

    st.subheader("Parameters")
    col1, col2 = st.columns(2)

    api_key = col1.text_input("API Key (if required)", type="password")
    output_fmt = col2.selectbox("Output format", ["json", "csv"])

    # Source-specific parameters
    kwargs: dict = {}
    if source_key in ("openmeteo", "copernicus"):
        c1, c2 = st.columns(2)
        kwargs["latitude"] = c1.number_input("Latitude", value=25.0, min_value=-90.0, max_value=90.0)
        kwargs["longitude"] = c2.number_input("Longitude", value=121.5, min_value=-180.0, max_value=180.0)
        c3, c4 = st.columns(2)
        kwargs["start_date"] = str(c3.date_input("Start Date"))
        kwargs["end_date"] = str(c4.date_input("End Date"))
        if source_key == "openmeteo":
            kwargs["mode"] = st.selectbox("Mode", ["weather", "forecast", "flood"])

    elif source_key == "usgs":
        kwargs["days"] = st.slider("Days of data", 1, 365, 30)

    elif source_key == "sdg6":
        kwargs["countries"] = st.text_input("Country codes (comma-separated ISO3)", "TWN")

    elif source_key == "wqp":
        kwargs["state"] = st.text_input("US State code (e.g. US:06)", "US:06")

    if api_key:
        kwargs["api_key"] = api_key

    if st.button("🚀 Collect Data", type="primary"):
        with st.spinner(f"Collecting from {dict(_DATA_SOURCES).get(source_key, source_key)}…"):
            try:
                from aquascope import collect

                records = collect(source_key, **kwargs)
                df = _samples_to_dataframe(records)

                st.session_state["collected_data"] = df
                st.session_state["collected_source"] = source_key
                st.success(f"✅ Collected **{len(df)} records**")

                st.subheader("Data Preview")
                st.dataframe(df.head(100), use_container_width=True)

                st.download_button(
                    "⬇️ Download data",
                    data=df.to_csv(index=False) if output_fmt == "csv" else df.to_json(orient="records", indent=2),
                    file_name=f"aquascope_{source_key}.{output_fmt}",
                    mime="text/csv" if output_fmt == "csv" else "application/json",
                )

            except Exception as exc:
                st.error(f"Collection failed: {exc}")
                logger.exception("Data collection error")


# ---------------------------------------------------------------------------
# Page: Analysis
# ---------------------------------------------------------------------------


def page_analysis() -> None:
    """Render the Analysis page (EDA + Quality Assessment)."""
    st = _require_streamlit()

    st.title("🔬 Analysis")
    st.markdown("Run exploratory data analysis and quality assessment on your water data.")

    # Data source selection
    data_source = st.radio(
        "Data source",
        ["Use collected data (session)", "Upload CSV", "Upload JSON"],
        horizontal=True,
    )

    df: pd.DataFrame | None = None

    if data_source == "Use collected data (session)":
        df = st.session_state.get("collected_data")
        if df is None:
            st.info("No data in session. Collect data first or upload a file.")
            return
        st.success(f"Using session data: {len(df)} records")
    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = _load_csv(uploaded)
    elif data_source == "Upload JSON":
        uploaded = st.file_uploader("Upload JSON file", type=["json"])
        if uploaded:
            df = _load_json(uploaded)

    if df is None or df.empty:
        st.warning("Please provide data to analyse.")
        return

    st.dataframe(df.head(20), use_container_width=True)
    st.divider()

    tab_eda, tab_quality = st.tabs(["📊 EDA Report", "🔍 Quality Assessment"])

    # ── EDA ──────────────────────────────────────────────────────────
    with tab_eda:
        if st.button("Run EDA", key="btn_eda"):
            with st.spinner("Generating EDA report…"):
                try:
                    from aquascope.analysis.eda import generate_eda_report, print_eda_report

                    report = generate_eda_report(df)
                    text = print_eda_report(report)

                    st.text(text)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Records", report.n_records)
                    col2.metric("Stations", report.n_stations)
                    col3.metric("Parameters", report.n_parameters)
                    col4.metric("Completeness", f"{report.completeness_pct:.1f}%")

                    if report.parameters:
                        st.subheader("Parameter Statistics")
                        import pandas as pd

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
                        st.dataframe(pd.DataFrame(param_rows), use_container_width=True)

                except Exception as exc:
                    st.error(f"EDA failed: {exc}")
                    logger.exception("EDA error")

    # ── Quality ──────────────────────────────────────────────────────
    with tab_quality:
        if st.button("Assess Quality", key="btn_quality"):
            with st.spinner("Running quality assessment…"):
                try:
                    from aquascope.analysis.quality import assess_quality, print_quality_report

                    report = assess_quality(df)
                    text = print_quality_report(report)

                    st.text(text)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Completeness", f"{report.completeness_pct:.1f}%")
                    col2.metric("Duplicates", report.n_duplicates)
                    col3.metric("Recommendations", len(report.recommended_steps))

                    if report.recommended_steps:
                        st.subheader("Recommended Steps")
                        for step in report.recommended_steps:
                            st.markdown(f"- `{step}`")

                    # Preprocessing
                    st.divider()
                    st.subheader("Preprocess Data")
                    available_steps = [
                        "remove_duplicates", "fill_missing", "remove_outliers", "normalize", "resample_daily",
                    ]
                    selected_steps = st.multiselect("Steps to apply", available_steps, default=report.recommended_steps)

                    if st.button("Apply Preprocessing", key="btn_preprocess"):
                        with st.spinner("Preprocessing…"):
                            from aquascope.analysis.quality import preprocess

                            cleaned = preprocess(df, steps=selected_steps)
                            st.session_state["collected_data"] = cleaned
                            st.success(f"✅ Preprocessed: {len(cleaned)} records (was {len(df)})")
                            st.dataframe(cleaned.head(20), use_container_width=True)

                except Exception as exc:
                    st.error(f"Quality assessment failed: {exc}")
                    logger.exception("Quality assessment error")


# ---------------------------------------------------------------------------
# Page: Visualization
# ---------------------------------------------------------------------------


def page_visualization() -> None:
    """Render the Visualization page."""
    st = _require_streamlit()
    import matplotlib

    matplotlib.use("Agg")

    st.title("📈 Visualization")
    st.markdown("Create publication-quality plots from your water data.")

    df = st.session_state.get("collected_data")
    if df is None:
        st.info("No data in session. Collect or upload data first (go to **📊 Data Collection** or **🔬 Analysis**).")
        return

    plot_key = st.selectbox(
        "Plot type",
        options=[k for k, _ in _PLOT_TYPES],
        format_func=lambda k: dict(_PLOT_TYPES)[k],
    )

    title = st.text_input("Plot title (optional)")

    try:
        if plot_key == "timeseries":
            _render_timeseries(st, df, title)
        elif plot_key == "boxplot":
            _render_boxplot(st, df, title)
        elif plot_key == "heatmap":
            _render_heatmap(st, df, title)
        elif plot_key == "who_exceedances":
            _render_who_exceedances(st, df, title)
        elif plot_key == "station_map":
            _render_station_map(st, df)
        elif plot_key == "fdc":
            _render_fdc(st, df, title)
        elif plot_key == "hydrograph":
            _render_hydrograph(st, df, title)
        elif plot_key == "spi_timeline":
            _render_spi_timeline(st, df, title)
        elif plot_key == "return_periods":
            _render_return_periods(st, df, title)
    except Exception as exc:
        st.error(f"Plot failed: {exc}")
        logger.exception("Visualization error")


def _render_timeseries(st, df: pd.DataFrame, title: str) -> None:
    """Render a time-series plot."""
    from aquascope.viz import plot_timeseries

    columns = [c for c in df.columns if c not in ("source", "station_id", "station_name", "remark")]
    value_col = st.selectbox("Value column", columns, index=columns.index("value") if "value" in columns else 0)

    fig = plot_timeseries(df, value_col=value_col, title=title or "Time Series")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_boxplot(st, df: pd.DataFrame, title: str) -> None:
    """Render a box plot."""
    from aquascope.viz import plot_boxplot

    fig = plot_boxplot(df, title=title or "Box Plot")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_heatmap(st, df: pd.DataFrame, title: str) -> None:
    """Render a correlation heatmap."""
    from aquascope.viz import plot_heatmap

    fig = plot_heatmap(df, title=title or "Heatmap")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_who_exceedances(st, df: pd.DataFrame, title: str) -> None:
    """Render WHO exceedance chart."""
    from aquascope.viz import plot_who_exceedances

    fig = plot_who_exceedances(df, title=title or "WHO Guideline Exceedances")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_station_map(st, df: pd.DataFrame) -> None:
    """Render a station map using folium or fallback to st.map."""
    try:
        from aquascope.viz import plot_station_map

        m = plot_station_map(df)

        try:
            from streamlit_folium import st_folium

            st_folium(m, width=700, height=500)
        except ImportError:
            import streamlit.components.v1 as components

            html = m._repr_html_()
            components.html(html, height=500)
    except ImportError:
        st.warning("Folium is required for interactive maps. Install with: `pip install aquascope[viz]`")
        if "latitude" in df.columns and "longitude" in df.columns:
            st.map(df[["latitude", "longitude"]].dropna())
    except Exception as exc:
        st.error(f"Map rendering failed: {exc}")


def _render_fdc(st, df: pd.DataFrame, title: str) -> None:
    """Render a flow duration curve."""
    from aquascope.viz import plot_fdc

    fig = plot_fdc(df, title=title or "Flow Duration Curve")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_hydrograph(st, df: pd.DataFrame, title: str) -> None:
    """Render a hydrograph."""
    from aquascope.viz import plot_hydrograph

    fig = plot_hydrograph(df, title=title or "Hydrograph")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_spi_timeline(st, df: pd.DataFrame, title: str) -> None:
    """Render an SPI timeline."""
    from aquascope.viz import plot_spi_timeline

    fig = plot_spi_timeline(df, title=title or "SPI Timeline")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_return_periods(st, df: pd.DataFrame, title: str) -> None:
    """Render return period plot."""
    from aquascope.viz import plot_return_periods

    fig = plot_return_periods(df, title=title or "Return Periods")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: Hydrology
# ---------------------------------------------------------------------------


def page_hydrology() -> None:
    """Render the Hydrology analysis page."""
    st = _require_streamlit()

    st.title("🌊 Hydrology")
    st.markdown("Run hydrological analyses with interactive parameter controls.")

    df = st.session_state.get("collected_data")
    if df is None:
        st.info("No data in session. Collect discharge data first or upload via **🔬 Analysis**.")
        return

    analysis = st.selectbox(
        "Analysis type",
        ["Flow Duration Curve", "Baseflow Separation", "Recession Analysis", "Flood Frequency"],
    )

    st.divider()

    try:
        if analysis == "Flow Duration Curve":
            _hydro_fdc(st, df)
        elif analysis == "Baseflow Separation":
            _hydro_baseflow(st, df)
        elif analysis == "Recession Analysis":
            _hydro_recession(st, df)
        elif analysis == "Flood Frequency":
            _hydro_flood_freq(st, df)
    except Exception as exc:
        st.error(f"Hydrology analysis failed: {exc}")
        logger.exception("Hydrology error")


def _hydro_fdc(st, df: pd.DataFrame) -> None:
    """Flow duration curve analysis."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    from aquascope.hydrology import flow_duration_curve

    columns = list(df.select_dtypes(include="number").columns)
    col = st.selectbox("Discharge column", columns)

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    result = flow_duration_curve(q)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("Q50 (median)", f"{result.q50:.3f}")
    col2.metric("Q95 (low-flow)", f"{result.q95:.3f}")

    from aquascope.viz import plot_fdc

    fig = plot_fdc(result)
    st.pyplot(fig)
    plt.close(fig)


def _hydro_baseflow(st, df: pd.DataFrame) -> None:
    """Baseflow separation analysis."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    from aquascope.hydrology import eckhardt, lyne_hollick

    columns = list(df.select_dtypes(include="number").columns)
    col = st.selectbox("Discharge column", columns, key="bf_col")

    method = st.radio("Method", ["Lyne-Hollick", "Eckhardt"], horizontal=True)

    alpha = st.slider("Filter parameter (α)", 0.90, 0.99, 0.925, 0.005)

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    if method == "Lyne-Hollick":
        passes = st.slider("Number of passes", 1, 5, 3)
        result = lyne_hollick(q, alpha=alpha, passes=passes)
    else:
        bfi_max = st.slider("BFI_max", 0.1, 1.0, 0.8, 0.05)
        result = eckhardt(q, alpha=alpha, bfi_max=bfi_max)

    st.subheader("Results")
    st.metric("Baseflow Index (BFI)", f"{result.bfi:.3f}")

    from aquascope.viz import plot_hydrograph

    fig = plot_hydrograph(result)
    st.pyplot(fig)
    plt.close(fig)


def _hydro_recession(st, df: pd.DataFrame) -> None:
    """Recession analysis."""
    from aquascope.hydrology import recession_analysis

    columns = list(df.select_dtypes(include="number").columns)
    col = st.selectbox("Discharge column", columns, key="rec_col")

    min_length = st.slider("Minimum recession length (days)", 3, 30, 5)

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    result = recession_analysis(q, min_length=min_length)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("Recession constant (K)", f"{result.k:.4f}")
    col2.metric("Segments found", str(result.n_segments))

    if result.segments:
        import pandas as pd

        seg_data = [
            {"Start": s.start, "End": s.end, "Duration": s.duration, "K": round(s.k, 4)}
            for s in result.segments
        ]
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)


def _hydro_flood_freq(st, df: pd.DataFrame) -> None:
    """Flood frequency analysis."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    from aquascope.hydrology import fit_gev

    columns = list(df.select_dtypes(include="number").columns)
    col = st.selectbox("Annual max discharge column", columns, key="ff_col")

    q = df[col].dropna()
    if q.empty:
        st.warning("Selected column has no data.")
        return

    result = fit_gev(q)

    st.subheader("GEV Distribution Fit")
    col1, col2, col3 = st.columns(3)
    col1.metric("Shape (ξ)", f"{result.shape:.4f}")
    col2.metric("Location (μ)", f"{result.loc:.2f}")
    col3.metric("Scale (σ)", f"{result.scale:.2f}")

    if hasattr(result, "return_levels") and result.return_levels:
        import pandas as pd

        rl_data = [{"Return Period (yr)": rp, "Discharge": round(val, 2)} for rp, val in result.return_levels.items()]
        st.dataframe(pd.DataFrame(rl_data), use_container_width=True)

    from aquascope.viz import plot_return_periods

    fig = plot_return_periods(result)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: AI Recommender
# ---------------------------------------------------------------------------


def page_ai_recommender() -> None:
    """Render the AI Recommender page."""
    st = _require_streamlit()

    st.title("🤖 AI Methodology Recommender")
    st.markdown("Get research methodology recommendations based on your dataset characteristics.")

    tab_manual, tab_auto = st.tabs(["Manual Profile", "Auto-detect from Data"])

    with tab_manual:
        goal = st.text_area("Research goal", placeholder="e.g. Trend analysis of dissolved oxygen in Tamsui River")
        parameters = st.text_input("Parameters (comma-separated)", placeholder="DO, BOD5, COD, pH, NH3-N")
        scope = st.text_input("Geographic scope", value="Taiwan")
        keywords = st.text_input("Keywords (comma-separated)", placeholder="trend, seasonal, water quality")

        col1, col2, col3 = st.columns(3)
        n_records = col1.number_input("Number of records", 0, 1_000_000, 0)
        n_stations = col2.number_input("Number of stations", 0, 10_000, 0)
        years = col3.number_input("Time span (years)", 0.0, 100.0, 0.0)

        top_k = st.slider("Number of recommendations", 1, 20, 5)

        if st.button("🔍 Get Recommendations", key="btn_rec_manual"):
            _run_recommendations(st, goal, parameters, scope, keywords, n_records, n_stations, years, top_k)

    with tab_auto:
        df = st.session_state.get("collected_data")
        if df is None:
            st.info("No data in session. Collect data first, then come back here for auto-recommendations.")
            return

        st.success(f"Dataset loaded: {len(df)} records")
        goal_auto = st.text_area("Research goal (optional)", key="goal_auto")
        top_k_auto = st.slider("Number of recommendations", 1, 20, 5, key="topk_auto")

        if st.button("🔍 Auto-recommend from Data", key="btn_rec_auto"):
            with st.spinner("Profiling dataset and generating recommendations…"):
                try:
                    from aquascope.ai_engine.recommender import recommend
                    from aquascope.analysis.eda import profile_dataset

                    profile = profile_dataset(df)
                    if goal_auto:
                        profile.research_goal = goal_auto

                    recs = recommend(profile, top_k=top_k_auto)
                    _display_recommendations(st, recs)
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")
                    logger.exception("AI recommender error")


def _run_recommendations(
    st, goal: str, parameters: str, scope: str, keywords: str,
    n_records: int, n_stations: int, years: float, top_k: int,
) -> None:
    """Run the rule-based recommender with manual profile inputs."""
    with st.spinner("Generating recommendations…"):
        try:
            from aquascope.ai_engine.recommender import DatasetProfile, recommend

            profile = DatasetProfile(
                parameters=[p.strip() for p in parameters.split(",") if p.strip()],
                n_records=n_records,
                n_stations=n_stations,
                time_span_years=years,
                geographic_scope=scope,
                research_goal=goal,
                keywords=[k.strip() for k in keywords.split(",") if k.strip()],
            )

            recs = recommend(profile, top_k=top_k)
            _display_recommendations(st, recs)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            logger.exception("AI recommender error")


def _display_recommendations(st, recs: list) -> None:
    """Display recommendation results."""
    if not recs:
        st.warning("No recommendations found. Try broadening your parameters or goal.")
        return

    st.subheader(f"Top {len(recs)} Recommendations")

    for i, rec in enumerate(recs, 1):
        score_color = "🟢" if rec.score >= 60 else "🟡" if rec.score >= 30 else "🔴"
        with st.expander(f"{score_color} #{i} — {rec.methodology.name} (score: {rec.score:.0f})", expanded=(i <= 3)):
            st.markdown(f"**Category:** {rec.methodology.category}")
            st.markdown(f"**Description:** {rec.methodology.description}")
            if rec.rationale:
                st.markdown(f"**Rationale:** {rec.rationale}")
            if rec.methodology.required_parameters:
                st.markdown(f"**Required parameters:** {', '.join(rec.methodology.required_parameters)}")
            if rec.methodology.tags:
                st.markdown(f"**Tags:** {', '.join(rec.methodology.tags)}")
            st.progress(rec.score / 100)


# ---------------------------------------------------------------------------
# Page: Water Quality Alerts
# ---------------------------------------------------------------------------


def page_water_quality_alerts() -> None:
    """Render the Water Quality Alerts page."""
    st = _require_streamlit()

    st.title("⚠️ Water Quality Alerts")
    st.markdown("Check water quality parameters against WHO/EPA/EU WFD guideline thresholds.")

    df = st.session_state.get("collected_data")
    if df is None:
        st.info("No data in session. Collect water quality data first.")
        return

    # Show reference thresholds
    with st.expander("📋 WHO Guideline Reference", expanded=False):
        import pandas as pd

        ref_rows = [
            {"Parameter": param, "Min": lo, "Max": hi if hi != float("inf") else "∞", "Unit": unit}
            for param, (lo, hi, unit) in WHO_GUIDELINES.items()
        ]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True)

    # Check if the data has the right structure
    if "parameter" not in df.columns or "value" not in df.columns:
        st.warning(
            "Expected columns `parameter` and `value` in the dataset. "
            "This page works best with normalised water quality data."
        )
        return

    if st.button("🔎 Check Exceedances", type="primary"):
        with st.spinner("Checking against WHO guidelines…"):
            _check_exceedances(st, df)

    st.divider()
    st.subheader("Challenge-Based Analysis")
    st.markdown("Use the `WaterQualityChallenge` class for deeper analysis.")

    site_id = st.text_input("Site ID", value=df["station_id"].iloc[0] if "station_id" in df.columns else "SITE-001")

    if st.button("Run Full Challenge Analysis", key="btn_wq_challenge"):
        with st.spinner("Running water quality challenge analysis…"):
            try:
                from aquascope.challenges import WaterQualityChallenge

                wq = WaterQualityChallenge(site_id=site_id)

                # Build parameter DataFrames from the normalised data
                param_dfs: dict = {}
                for param_name in df["parameter"].unique():
                    subset = df[df["parameter"] == param_name].copy()
                    for dt_col in ("sample_datetime", "reading_datetime"):
                        if dt_col in subset.columns:
                            import pandas as pd

                            subset.index = pd.to_datetime(subset[dt_col])
                            break
                    param_dfs[param_name] = subset[["value"]]

                wq.load_dataframes(param_dfs)
                exceedances = wq.check_who_guidelines()

                if exceedances:
                    st.warning(f"⚠️ Found exceedances in **{len(exceedances)}** parameters")
                    for param, info in exceedances.items():
                        st.markdown(f"- **{param}**: {info}")
                else:
                    st.success("✅ All parameters within WHO guidelines")
            except Exception as exc:
                st.error(f"Challenge analysis failed: {exc}")
                logger.exception("Water quality challenge error")


def _check_exceedances(st, df) -> None:
    """Check dataframe values against WHO guidelines and display results."""
    import pandas as pd

    results = []
    parameters = df["parameter"].str.lower().unique()

    for param in parameters:
        if param not in WHO_GUIDELINES:
            continue

        lo, hi, unit = WHO_GUIDELINES[param]
        subset = df[df["parameter"].str.lower() == param]["value"].dropna()

        if subset.empty:
            continue

        n_total = len(subset)
        if hi == float("inf"):
            n_exceed = int((subset < lo).sum())
            rule = f"≥ {lo} {unit}"
        elif lo == 0:
            n_exceed = int((subset > hi).sum())
            rule = f"≤ {hi} {unit}"
        else:
            n_exceed = int(((subset < lo) | (subset > hi)).sum())
            rule = f"{lo}–{hi} {unit}"

        pct = (n_exceed / n_total * 100) if n_total > 0 else 0
        status = "🔴 ALERT" if pct > 10 else "🟡 Warning" if pct > 0 else "🟢 OK"

        results.append({
            "Parameter": param,
            "Guideline": rule,
            "Samples": n_total,
            "Exceedances": n_exceed,
            "Exceedance %": round(pct, 1),
            "Status": status,
        })

    if results:
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        alerts = [r for r in results if "ALERT" in r["Status"]]
        warnings = [r for r in results if "Warning" in r["Status"]]

        if alerts:
            names = ", ".join(a["Parameter"] for a in alerts)
            st.error(f"🔴 **{len(alerts)} parameter(s) exceed 10% threshold:** {names}")
        if warnings:
            names = ", ".join(w["Parameter"] for w in warnings)
            st.warning(f"🟡 **{len(warnings)} parameter(s) have some exceedances:** {names}")
        if not alerts and not warnings:
            st.success("✅ All monitored parameters within WHO guidelines")
    else:
        st.info("No WHO-monitored parameters found in the dataset. Monitored: " + ", ".join(WHO_GUIDELINES.keys()))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

_PAGES: dict[str, tuple[str, callable]] = {
    "home": ("🏠 Home", page_home),
    "collection": ("📊 Data Collection", page_data_collection),
    "analysis": ("🔬 Analysis", page_analysis),
    "visualization": ("📈 Visualization", page_visualization),
    "hydrology": ("🌊 Hydrology", page_hydrology),
    "ai_recommender": ("🤖 AI Recommender", page_ai_recommender),
    "alerts": ("⚠️ Water Quality Alerts", page_water_quality_alerts),
}


def main() -> None:
    """Streamlit app entry point — sets up sidebar navigation and renders the selected page."""
    st = _require_streamlit()

    st.set_page_config(
        page_title="AquaScope Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("🌊 AquaScope")
    st.sidebar.markdown("---")

    page_labels = [label for label, _ in _PAGES.values()]

    selected_label = st.sidebar.radio("Navigate", page_labels, index=0)

    # Map label back to key
    label_to_key = {label: key for key, (label, _) in _PAGES.items()}
    selected_key = label_to_key[selected_label]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"AquaScope v0.2.0 | Dashboard v{__version__}")

    # Session state info in sidebar
    if "collected_data" in st.session_state and st.session_state["collected_data"] is not None:
        n = len(st.session_state["collected_data"])
        src = st.session_state.get("collected_source", "unknown")
        st.sidebar.success(f"📦 {n} records ({src})")

    # Render selected page
    _, page_func = _PAGES[selected_key]
    page_func()


# Module-level version for the dashboard __init__
__version__ = "0.1.0"

if __name__ == "__main__":
    main()
