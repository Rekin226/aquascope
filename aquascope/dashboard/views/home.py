"""Home page — hero, live workspace insights, and guided entry points."""

from __future__ import annotations

import streamlit as st

from aquascope.dashboard import _insights, _state


def render() -> None:
    from aquascope import __version__

    st.markdown(
        f"""
        <div class="aq-hero">
          <div class="aq-hero-title">🌊 AquaScope</div>
          <div class="aq-hero-sub">
            Open-source water intelligence — 21 live data sources, a full hydrology &amp;
            agricultural-water toolkit, and AI-assisted research methodology, in one workspace.
          </div>
          <div class="aq-hero-badge">v{__version__} · MIT
            · <a href="https://github.com/Rekin226/aquascope" target="_blank">GitHub</a>
            · <a href="https://pypi.org/project/aquascope/" target="_blank">PyPI</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = _state.get_data()
    if df is not None:
        st.markdown(f"**Active workspace:** {_state.source_label()}")
        _insights.render_panel(df, key_prefix="home")
    else:
        st.markdown("")
        with st.container(border=True):
            st.markdown("##### 🚀 Start in one click")
            st.caption(
                "Load a demo dataset to explore every page instantly, or pull real data "
                "from 21 live sources — USGS, GRDC, Open-Meteo, Taiwan WRA, and more."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✨ Demo: water quality", width="stretch", key="home_wq"):
                    _state.load_demo("water_quality")
                    st.rerun()
            with c2:
                if st.button("✨ Demo: 40-yr streamflow", width="stretch", key="home_sf"):
                    _state.load_demo("streamflow")
                    st.rerun()
            with c3:
                if st.button("🌐 Collect real data →", width="stretch", key="home_collect"):
                    _state.goto("collect")

    st.markdown("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live data sources", "21")
    m2.metric("Interactive chart types", "12+")
    m3.metric("Hydrology & agri models", "20+")
    m4.metric("AI methodologies", "26")

    st.markdown("")
    st.subheader("What's inside")

    cards = [
        ("📡 Collect", "collect",
         "21 live sources across Taiwan, the US, Europe, Japan, Korea, India, France, "
         "and global archives, plus CSV/JSON upload."),
        ("🔬 Analyze & clean", "analysis",
         "Automated EDA, data-quality scoring, and one-click preprocessing (dedupe, fill, outliers, resampling)."),
        ("📈 Visualize", "visualize",
         "Interactive time series, box plots, correlation heatmaps, station maps, FDCs, hydrographs, and more."),
        ("🌊 Hydrology lab", "hydrology",
         "Flow-duration curves, baseflow separation (Lyne-Hollick / Eckhardt / UKIH), "
         "recession analysis, and 20+ flow signatures."),
        ("🌀 Extreme events", "extremes",
         "GEV / Log-Pearson III / Gumbel fits on annual maxima with bootstrap "
         "confidence bounds and design return levels."),
        ("🌾 Agricultural water", "agri",
         "FAO-56 Penman-Monteith ET₀, single & dual (Kcb + Ke) crop coefficients, and full irrigation scheduling."),
        ("🤖 AI recommender", "ai",
         "Rule-based + optional LLM-enhanced research-methodology recommendations from your dataset's profile."),
        ("⚠️ Quality alerts", "alerts",
         "WHO / EPA / EU guideline screening with per-parameter exceedance rates and status flags."),
    ]

    for row_start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (title, key, desc) in zip(cols, cards[row_start : row_start + 4]):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
                if st.button("Open →", key=f"card_{key}", width="stretch"):
                    _state.goto(key)

    st.caption(
        "Every page runs on real `aquascope` library functions — what you see here is "
        "exactly what `pip install aquascope` gives you in Python."
    )
