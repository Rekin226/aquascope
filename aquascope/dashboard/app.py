"""
AquaScope Interactive Dashboard — Streamlit application entry point.

Launch with::

    streamlit run aquascope/dashboard/app.py
    # or via CLI:
    aquascope dashboard

The app is organised as a multipage workspace (``st.navigation``): one shared
dataset flows through every page, a smart-insights layer profiles whatever is
loaded, and all charts are interactive Plotly figures on a single visual system.
"""

from __future__ import annotations

__version__ = "1.0.0"

_CSS = """
<style>
/* Hide the Streamlit Deploy button — not relevant for end users */
[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"] { display: none !important; }

/* Hero banner on the Home page */
.aq-hero {
    background: linear-gradient(120deg, #0d3f7a 0%, #1565C0 55%, #1e88a8 100%);
    border-radius: 0.75rem;
    padding: 2.2rem 2.4rem 2rem 2.4rem;
    margin-bottom: 1.2rem;
    color: #ffffff;
}
.aq-hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 0.45rem;
}
.aq-hero-sub {
    font-size: 1.05rem;
    opacity: 0.92;
    max-width: 46rem;
    margin-bottom: 0.8rem;
}
.aq-hero-badge { font-size: 0.85rem; opacity: 0.85; }
.aq-hero-badge a { color: #cde2fb; text-decoration: none; }
.aq-hero-badge a:hover { text-decoration: underline; }

/* Slightly tighter sidebar */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
"""


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


def main() -> None:
    """Streamlit app entry point."""
    st = _require_streamlit()

    st.set_page_config(
        page_title="AquaScope Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get help": "https://rekin226.github.io/aquascope/",
            "Report a bug": "https://github.com/Rekin226/aquascope/issues",
            "About": "AquaScope — open-source water data aggregation & analysis toolkit.",
        },
    )

    from aquascope.dashboard import _charts, _state
    from aquascope.dashboard.views import (
        agri,
        ai,
        alerts,
        analysis,
        collect,
        extremes,
        home,
        hydrology,
        visualize,
    )

    _charts.register_template()
    st.markdown(_CSS, unsafe_allow_html=True)

    pages = {
        "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
        "collect": st.Page(collect.render, title="Collect Data", icon=":material/cloud_download:", url_path="collect"),
        "analysis": st.Page(analysis.render, title="Analyze & Clean", icon=":material/science:", url_path="analysis"),
        "visualize": st.Page(visualize.render, title="Visualize", icon=":material/monitoring:", url_path="visualize"),
        "hydrology": st.Page(hydrology.render, title="Hydrology Lab", icon=":material/water:", url_path="hydrology"),
        "extremes": st.Page(extremes.render, title="Extreme Events", icon=":material/cyclone:", url_path="extremes"),
        "agri": st.Page(agri.render, title="Agricultural Water", icon=":material/agriculture:", url_path="agri"),
        "ai": st.Page(ai.render, title="AI Recommender", icon=":material/smart_toy:", url_path="ai"),
        "alerts": st.Page(alerts.render, title="Quality Alerts", icon=":material/warning:", url_path="alerts"),
    }
    _state.register_pages(pages)

    nav = st.navigation(
        {
            "": [pages["home"]],
            "Data": [pages["collect"], pages["analysis"]],
            "Explore": [pages["visualize"]],
            "Water science": [pages["hydrology"], pages["extremes"], pages["agri"]],
            "Intelligence": [pages["ai"], pages["alerts"]],
        }
    )

    _render_sidebar_workspace(st)
    nav.run()


def _render_sidebar_workspace(st) -> None:
    """Persistent workspace card at the bottom of the sidebar."""
    from aquascope import __version__ as aquascope_version
    from aquascope.dashboard import _state

    with st.sidebar:
        st.divider()
        df = _state.get_data()
        if df is not None:
            with st.container(border=True):
                st.markdown("**📦 Workspace**")
                st.caption(f"{_state.source_label()} · {len(df):,} rows × {len(df.columns)} cols")
                c1, c2 = st.columns(2)
                c1.download_button(
                    "⬇️ CSV",
                    data=df.to_csv(index=False),
                    file_name="aquascope_workspace.csv",
                    mime="text/csv",
                    width="stretch",
                    key="ws_download",
                )
                if c2.button("🗑️ Clear", width="stretch", key="ws_clear"):
                    _state.clear_data()
                    st.rerun()
        else:
            st.caption("📦 Workspace: empty")
            if st.button("✨ Load demo data", width="stretch", key="ws_demo"):
                _state.load_demo("water_quality")
                st.rerun()

        st.caption(f"AquaScope v{aquascope_version} · dashboard v{__version__}")


if __name__ == "__main__":
    main()
