"""Shared workspace state, smart column detection, and navigation helpers.

The dashboard keeps ONE active dataset in ``st.session_state`` (the
"workspace"). Every page reads from it through :func:`get_data` /
:func:`require_data`, and :func:`profile` auto-detects the columns that
drive each analysis so users almost never have to pick them manually.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from aquascope import workbench

DATA_KEY = "collected_data"
SOURCE_KEY = "collected_source"
LABEL_KEY = "collected_label"

# Filled by app.py at startup: page-key -> st.Page object
_PAGES: dict[str, object] = {}

_DATETIME_CANDIDATES = (
    "sample_datetime",
    "reading_datetime",
    "observation_datetime",
    "forecast_datetime",
    "date",
    "datetime",
    "timestamp",
    "time",
)

_DISCHARGE_HINTS = ("discharge", "flow", "streamflow", "q_cms")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def register_pages(pages: dict[str, object]) -> None:
    """Store the st.Page registry so any view can jump between pages."""
    _PAGES.clear()
    _PAGES.update(pages)


def goto(key: str) -> None:
    """Programmatically switch to another dashboard page."""
    page = _PAGES.get(key)
    if page is not None:
        st.switch_page(page)


# ---------------------------------------------------------------------------
# Workspace dataset
# ---------------------------------------------------------------------------


def get_data() -> pd.DataFrame | None:
    """Return the active workspace DataFrame, or None."""
    df = st.session_state.get(DATA_KEY)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def set_data(df: pd.DataFrame, source: str, label: str | None = None) -> None:
    """Replace the active workspace dataset."""
    st.session_state[DATA_KEY] = df
    st.session_state[SOURCE_KEY] = source
    st.session_state[LABEL_KEY] = label or source


def clear_data() -> None:
    """Drop the active workspace dataset."""
    for key in (DATA_KEY, SOURCE_KEY, LABEL_KEY):
        st.session_state.pop(key, None)


def source_label() -> str:
    return str(st.session_state.get(LABEL_KEY, st.session_state.get(SOURCE_KEY, "—")))


def load_demo(kind: str = "water_quality") -> None:
    """Load one of the built-in demo datasets into the workspace."""
    from aquascope.dashboard import _demo

    if kind == "streamflow":
        set_data(_demo.streamflow_40y(), "demo_streamflow", "Demo: 40-yr daily streamflow")
    else:
        set_data(_demo.water_quality(), "demo", "Demo: water quality (180 d)")


# ---------------------------------------------------------------------------
# Smart column detection
# ---------------------------------------------------------------------------
# The detection rules and the analyses behind every page now live in
# aquascope.workbench, so the browser, the MCP server and the CLI run exactly
# what these pages run. These names are re-exported for the views.

DataProfile = workbench.DataProfile
profile = workbench.profile
datetime_indexed = workbench.datetime_indexed


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def require_data(hint: str = "") -> pd.DataFrame | None:
    """Return the workspace dataset, or render a friendly empty state.

    Views call this at the top and simply ``return`` when it yields None.
    """
    df = get_data()
    if df is not None:
        return df

    default_hint = "Collect data, upload a file, or start with a demo dataset."
    st.info("**No dataset in the workspace yet.** " + (hint or default_hint))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✨ Demo: water quality", width="stretch", key=f"es_wq_{hint[:12]}"):
            load_demo("water_quality")
            st.rerun()
    with c2:
        if st.button("✨ Demo: 40-yr streamflow", width="stretch", key=f"es_sf_{hint[:12]}"):
            load_demo("streamflow")
            st.rerun()
    with c3:
        if st.button("🌐 Collect real data →", width="stretch", key=f"es_col_{hint[:12]}"):
            goto("collect")
    return None
