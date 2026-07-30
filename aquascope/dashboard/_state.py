"""Shared workspace state, smart column detection, and navigation helpers.

The dashboard keeps ONE active dataset in ``st.session_state`` (the
"workspace"). Every page reads from it through :func:`get_data` /
:func:`require_data`, and :func:`profile` auto-detects the columns that
drive each analysis so users almost never have to pick them manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

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


@dataclass
class DataProfile:
    """Auto-detected structure of the workspace dataset."""

    n_records: int = 0
    datetime_col: str | None = None
    station_col: str | None = None
    param_col: str | None = None
    value_col: str | None = None
    discharge_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    numeric_cols: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    n_stations: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    span_years: float = 0.0
    completeness_pct: float = 100.0

    @property
    def has_time(self) -> bool:
        return self.datetime_col is not None

    @property
    def has_geo(self) -> bool:
        return self.lat_col is not None and self.lon_col is not None

    @property
    def has_params(self) -> bool:
        return self.param_col is not None and self.value_col is not None


def _first_match(columns, candidates) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def profile(df: pd.DataFrame) -> DataProfile:
    """Detect the meaningful columns in *df* — the dashboard's smart layer."""
    prof = DataProfile(n_records=len(df))
    cols = list(df.columns)
    prof.numeric_cols = list(df.select_dtypes(include="number").columns)

    # datetime: named candidate, else any datetime64 column
    prof.datetime_col = _first_match(cols, _DATETIME_CANDIDATES)
    if prof.datetime_col is None:
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                prof.datetime_col = c
                break

    prof.station_col = _first_match(cols, ("station_name", "station_id", "site_id", "site_name", "station"))
    prof.param_col = _first_match(cols, ("parameter", "variable", "characteristic_name"))
    prof.value_col = _first_match(cols, ("value", "result_value", "reading_value"))
    if prof.value_col is None and prof.numeric_cols:
        geo_names = ("latitude", "longitude", "lat", "lon", "elevation")
        non_geo = [c for c in prof.numeric_cols if c.lower() not in geo_names]
        prof.value_col = non_geo[0] if non_geo else prof.numeric_cols[0]

    for c in prof.numeric_cols:
        if any(h in c.lower() for h in _DISCHARGE_HINTS):
            prof.discharge_col = c
            break

    prof.lat_col = _first_match(cols, ("latitude", "lat"))
    prof.lon_col = _first_match(cols, ("longitude", "lon", "lng"))

    if prof.param_col:
        try:
            prof.parameters = sorted(df[prof.param_col].dropna().astype(str).unique().tolist())[:50]
        except Exception:  # noqa: BLE001
            prof.parameters = []
    if prof.station_col:
        try:
            prof.n_stations = int(df[prof.station_col].nunique())
        except Exception:  # noqa: BLE001
            prof.n_stations = 0

    if prof.datetime_col is not None:
        try:
            dt = pd.to_datetime(df[prof.datetime_col], errors="coerce", utc=True).dropna()
            if not dt.empty:
                prof.date_min = dt.min()
                prof.date_max = dt.max()
                prof.span_years = float((prof.date_max - prof.date_min).days / 365.25)
        except Exception:  # noqa: BLE001
            pass

    if len(df) and len(df.columns):
        prof.completeness_pct = float(100.0 * (1 - df.isna().sum().sum() / (len(df) * len(df.columns))))

    return prof


def datetime_indexed(df: pd.DataFrame, col: str, prof: DataProfile | None = None) -> pd.Series:
    """Return ``df[col]`` as a Series with a DatetimeIndex when possible."""
    series = df[col].dropna()
    dt_col = (prof.datetime_col if prof else None) or _first_match(df.columns, _DATETIME_CANDIDATES)
    if dt_col is not None:
        try:
            idx = pd.to_datetime(df.loc[series.index, dt_col], errors="coerce")
            series = series[idx.notna().to_numpy()]
            series.index = pd.DatetimeIndex(idx.dropna())
        except Exception:  # noqa: BLE001 — fall back to integer index
            pass
    return series


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
