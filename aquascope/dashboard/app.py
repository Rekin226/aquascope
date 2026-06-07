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
    ("aquastat", "AQUASTAT (FAO Global Water)"),
    ("eu_wfd", "EU WFD (Water Framework Directive)"),
    ("japan_mlit", "Japan MLIT (Water Info)"),
    ("korea_wamis", "Korea WAMIS (Water Resources)"),
    ("wapor", "WaPOR (FAO Evapotranspiration)"),
]

# Sources that require a free user-provided API key to return any data.
# Map: source_key -> (display_label, signup_url)
_API_KEY_SOURCES: dict[str, tuple[str, str]] = {
    "taiwan_moenv": ("Taiwan MOENV", "https://data.moenv.gov.tw/en/apikey"),
    "copernicus": ("Copernicus CDS", "https://cds.climate.copernicus.eu/how-to-api"),
}

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

_WORKFLOW_STEPS = ["📊 Collect", "🔬 Analyze", "📈 Visualize", "🌊 Hydrology"]

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


def _load_demo_data() -> pd.DataFrame:
    """Build a compact demo water-quality dataset that works across all dashboard pages."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    params = ["ph", "dissolved_oxygen", "turbidity", "nitrate"]

    # Seasonal signal so time-series plots look interesting
    t = np.linspace(0, 4 * np.pi, len(dates))

    rows = []
    for i, date in enumerate(dates):
        discharge = max(0.1, 5.0 + 3.0 * np.sin(t[i]) + rng.normal(0, 0.5))
        for param in params:
            base = {
                "ph": 7.2 + 0.5 * np.sin(t[i]),
                "dissolved_oxygen": 7.0 + 2.0 * np.cos(t[i]),
                "turbidity": 3.5 + 2.0 * np.abs(np.sin(t[i])),
                "nitrate": 30.0 + 20.0 * np.sin(t[i] + 1),
            }[param]
            noise = rng.normal(0, {"ph": 0.3, "dissolved_oxygen": 0.8, "turbidity": 0.5, "nitrate": 5.0}[param])
            rows.append({
                "sample_datetime": date,
                "station_id": "DEMO-001",
                "station_name": "Tamsui River Demo Station",
                "parameter": param,
                "value": round(float(base + noise), 3),
                "discharge": round(float(discharge), 3),
                "latitude": 25.17,
                "longitude": 121.44,
                "source": "demo",
            })

    return pd.DataFrame(rows)


def _show_workflow_step(st, current: int) -> None:
    """Render a compact workflow progress indicator."""
    steps_display = "  →  ".join(
        f"**{s}**" if i == current else s
        for i, s in enumerate(_WORKFLOW_STEPS)
    )
    st.caption(f"Workflow: {steps_display}")
    st.progress(current / (len(_WORKFLOW_STEPS) - 1))


def _demo_data_cta(st) -> bool:
    """Show a 'Load demo dataset' button. Returns True if demo data was loaded."""
    if st.button("Load demo dataset", key=f"demo_{id(st)}", use_container_width=False):
        st.session_state["collected_data"] = _load_demo_data()
        st.session_state["collected_source"] = "demo"
        st.rerun()
    return False


def _inject_global_css(st) -> None:
    """Inject CSS tweaks: hide Deploy button, style nav, fix sidebar H1."""
    st.markdown(
        """
        <style>
        /* P5: Hide the Streamlit Deploy button — not relevant for end users */
        [data-testid="stDeployButton"],
        [data-testid="stAppDeployButton"] { display: none !important; }

        /* Sidebar nav: tighten spacing, remove radio dot visual noise */
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.1rem;
        }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.45rem 0.6rem;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: background 0.15s;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(0, 0, 0, 0.06);
        }
        /* Hide the radio circle — the active highlight is enough */
        section[data-testid="stSidebar"] .stRadio [role="radio"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------


def page_home() -> None:
    """Render the Home overview page."""
    st = _require_streamlit()

    # P6: Use st.title here (H1). Sidebar uses st.sidebar.markdown("## …") = H2.
    st.title("🌊 AquaScope Dashboard")
    st.markdown(
        """
        **AquaScope** is an open-source water data aggregation toolkit with
        AI-powered research methodology recommendations.

        ### Features
        - 📊 **15 data collectors** — USGS, GEMStat, AQUASTAT, EU WFD, Japan MLIT, Korea WAMIS, WaPOR, Open-Meteo & more
        - 🔬 **Automated EDA & quality assessment** on collected data
        - 🤖 **AI recommender** — rule-based + optional LLM-enhanced methodology suggestions
        - 🌊 **Hydrology toolkit** — flow duration curves, baseflow separation, recession analysis, flood frequency
        - 📈 **Publication-quality visualisations** with matplotlib/seaborn/folium
        - ⚠️ **Water quality alerts** against WHO/EPA/EU thresholds
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Sources", "15")
    col2.metric("Plot Types", str(len(_PLOT_TYPES)))
    col3.metric("AI Methodologies", "26")

    if "collected_data" in st.session_state and st.session_state["collected_data"] is not None:
        df = st.session_state["collected_data"]
        st.success(f"✅ Active dataset: **{len(df)} records**, {df.columns.tolist()}")

    st.divider()

    # P7: Interactive workflow cards with navigation buttons
    st.subheader("Quick Start")
    st.caption("Follow these steps to go from raw data to publication-ready analysis.")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Step 1 — Collect**")
        st.caption("Fetch from 15 real water data sources, or load a demo dataset to explore the app instantly.")
        if st.button("📊 Data Collection →", key="qs_collect", use_container_width=True):
            st.session_state["_nav_pending"] = "📊 Data Collection"
            st.rerun()

    with c2:
        st.markdown("**Step 2 — Analyze**")
        st.caption("Run EDA and quality assessment. Detect outliers, nulls, and get preprocessing recommendations.")
        if st.button("🔬 Analysis →", key="qs_analysis", use_container_width=True):
            st.session_state["_nav_pending"] = "🔬 Analysis"
            st.rerun()

    with c3:
        st.markdown("**Step 3 — Visualize**")
        st.caption("Create time-series, boxplots, station maps, flow duration curves, and more.")
        if st.button("📈 Visualization →", key="qs_viz", use_container_width=True):
            st.session_state["_nav_pending"] = "📈 Visualization"
            st.rerun()

    with c4:
        st.markdown("**Step 4 — Insights**")
        st.caption("Run hydrology models, get AI methodology suggestions, and check WHO quality alerts.")
        if st.button("🤖 AI Recommender →", key="qs_ai", use_container_width=True):
            st.session_state["_nav_pending"] = "🤖 AI Recommender"
            st.rerun()

    st.divider()
    st.caption("No data yet? Click **📊 Data Collection** above and choose a source, or use **Load demo dataset** on any analysis page.")


# ---------------------------------------------------------------------------
# Page: Data Collection
# ---------------------------------------------------------------------------


def page_data_collection() -> None:
    """Render the Data Collection page."""
    st = _require_streamlit()

    st.title("📊 Data Collection")
    st.markdown("Fetch water data from any of AquaScope's supported sources.")

    def _label(key: str) -> str:
        base = dict(_DATA_SOURCES)[key]
        return f"🔑 {base}" if key in _API_KEY_SOURCES else base

    source_key = st.selectbox(
        "Data Source",
        options=[k for k, _ in _DATA_SOURCES],
        format_func=_label,
    )

    if source_key in _API_KEY_SOURCES:
        provider, signup_url = _API_KEY_SOURCES[source_key]
        st.info(
            f"🔑 **{provider} requires a free API key.** "
            f"Get one at [{signup_url}]({signup_url}) and paste it below — "
            f"without it, the API will reject the request."
        )

    st.subheader("Parameters")
    col1, col2 = st.columns(2)

    api_key = col1.text_input("API Key (if required)", type="password")
    output_fmt = col2.selectbox("Output format", ["json", "csv"])

    # Source-specific parameters
    kwargs: dict = {}
    if source_key == "taiwan_wra_level":
        st.info("Real-time snapshot — returns current readings from all river stations. No historical date range is available via this API.")

    elif source_key == "taiwan_wra_reservoir":
        st.info("Daily snapshot — returns the most recent day's reservoir data. No historical date range is available via this API.")

    elif source_key == "taiwan_moenv":
        kwargs["limit"] = st.slider("Records to fetch (most recent first)", 100, 5_000, 500, step=100)

    elif source_key == "taiwan_civil_iot":
        st.caption("SensorThings API — filter by date range to narrow observations.")
        _c1, _c2 = st.columns(2)
        _sd = _c1.date_input("Start Date", value=None, key="ciot_start")
        _ed = _c2.date_input("End Date", value=None, key="ciot_end")
        if _sd:
            kwargs["start_date"] = str(_sd)
        if _ed:
            kwargs["end_date"] = str(_ed)

    elif source_key in ("openmeteo", "copernicus"):
        c1, c2 = st.columns(2)
        kwargs["latitude"] = c1.number_input("Latitude", value=25.0, min_value=-90.0, max_value=90.0)
        kwargs["longitude"] = c2.number_input("Longitude", value=121.5, min_value=-180.0, max_value=180.0)
        c3, c4 = st.columns(2)
        kwargs["start_date"] = str(c3.date_input("Start Date"))
        kwargs["end_date"] = str(c4.date_input("End Date"))
        if source_key == "openmeteo":
            kwargs["mode"] = st.selectbox("Mode", ["weather", "forecast", "flood"])

    elif source_key == "usgs":
        kwargs["days"] = st.slider("Days of data", 1, 30, 3)
        st.caption(
            "USGS covers thousands of US stations — use a region filter to keep response times under 30 s."
        )

        _USGS_REGIONS = {
            "No filter (all US — slow)": None,
            "Northeast US": "-80,37,-66,48",
            "Southeast US": "-92,24,-80,37",
            "Midwest US": "-104,36,-80,48",
            "Pacific Northwest": "-125,42,-104,50",
            "Southwest US": "-125,32,-104,42",
            "Custom bbox": "__custom__",
        }
        region_label = st.selectbox("Region filter", list(_USGS_REGIONS.keys()), index=1)
        bbox_val = _USGS_REGIONS[region_label]
        if bbox_val == "__custom__":
            bbox_val = st.text_input(
                "Bounding box (minLon,minLat,maxLon,maxLat)",
                placeholder="-80,37,-66,48",
            ) or None
        if bbox_val:
            kwargs["bbox"] = bbox_val

        kwargs["max_items"] = st.slider("Max records", 100, 10_000, 2_000, step=100)

    elif source_key == "sdg6":
        _SDG6_COUNTRIES = {
            "Taiwan": "TWN", "China": "CHN", "Japan": "JPN", "South Korea": "KOR",
            "India": "IND", "Indonesia": "IDN", "Philippines": "PHL",
            "Vietnam": "VNM", "Thailand": "THA", "Malaysia": "MYS",
            "Singapore": "SGP", "Bangladesh": "BGD", "Pakistan": "PAK",
            "Nepal": "NPL", "Sri Lanka": "LKA",
            "United States": "USA", "Canada": "CAN", "Mexico": "MEX",
            "Brazil": "BRA", "Argentina": "ARG", "Chile": "CHL",
            "United Kingdom": "GBR", "France": "FRA", "Germany": "DEU",
            "Italy": "ITA", "Spain": "ESP", "Portugal": "PRT",
            "Netherlands": "NLD", "Belgium": "BEL", "Switzerland": "CHE",
            "Austria": "AUT", "Sweden": "SWE", "Norway": "NOR",
            "Finland": "FIN", "Denmark": "DNK", "Poland": "POL",
            "Russia": "RUS", "Ukraine": "UKR", "Turkey": "TUR",
            "Australia": "AUS", "New Zealand": "NZL",
            "South Africa": "ZAF", "Egypt": "EGY", "Nigeria": "NGA",
            "Kenya": "KEN", "Ethiopia": "ETH", "Morocco": "MAR",
            "Saudi Arabia": "SAU", "UAE": "ARE", "Israel": "ISR",
            "Iran": "IRN", "Iraq": "IRQ",
        }
        _selected_names = st.multiselect(
            "Countries",
            list(_SDG6_COUNTRIES.keys()),
            default=["Germany"],
            help="Taiwan is not included in UN SDG data. Try Germany, United States, India, etc.",
        )
        kwargs["country_codes"] = ",".join(_SDG6_COUNTRIES[n] for n in _selected_names) if _selected_names else None
        _SDG6_LABELS = {
            "6.1.1": "Safely managed drinking water",
            "6.2.1": "Safely managed sanitation",
            "6.3.1": "Safely treated wastewater",
            "6.3.2": "Good ambient water quality",
            "6.4.1": "Water-use efficiency",
            "6.4.2": "Water stress",
            "6.5.1": "IWRM implementation",
            "6.5.2": "Transboundary cooperation",
            "6.6.1": "Water-related ecosystems",
        }
        kwargs["indicator_codes"] = [
            st.selectbox(
                "Indicator",
                list(_SDG6_LABELS.keys()),
                index=5,
                format_func=lambda c: f"{c} ({_SDG6_LABELS[c]})",
            )
        ]

    elif source_key == "gemstat":
        st.info(
            "📦 GEMStat data is a ~200 MB archive hosted on Zenodo. "
            "The first collection downloads and caches it locally — this takes 1–3 minutes. "
            "Subsequent runs load from the local cache instantly."
        )
        st.caption("Taiwan is not in GEMStat — use Taiwan MOENV or WRA for Taiwan data.")
        _GEMSTAT_COUNTRIES = [
            "Argentina", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
            "Canada", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland",
            "France", "Germany", "Greece", "Hungary", "Iceland", "India", "Ireland",
            "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
            "Macedonia (the former Yugoslav Republic of)", "Malta", "Mexico",
            "Montenegro", "Netherlands (-the )", "Norway", "Poland", "Portugal",
            "Romania", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
            "Switzerland", "Turkey", "United States of America (the)", "Uruguay",
        ]
        kwargs["country"] = st.selectbox("Country", _GEMSTAT_COUNTRIES, index=_GEMSTAT_COUNTRIES.index("Germany"))
        _gc1, _gc2 = st.columns(2)
        _gsd = _gc1.date_input("Start Date (optional)", value=None, key="gemstat_start")
        _ged = _gc2.date_input("End Date (optional)", value=None, key="gemstat_end")
        if _gsd:
            kwargs["start_date"] = str(_gsd)
        if _ged:
            kwargs["end_date"] = str(_ged)
        kwargs["max_records"] = st.slider("Max records", 500, 20_000, 5_000, step=500)

    elif source_key == "wqp":
        _WQP_STATES = {
            "Alabama": "US:01", "Alaska": "US:02", "Arizona": "US:04",
            "Arkansas": "US:05", "California": "US:06", "Colorado": "US:08",
            "Connecticut": "US:09", "Delaware": "US:10", "Florida": "US:12",
            "Georgia": "US:13", "Hawaii": "US:15", "Idaho": "US:16",
            "Illinois": "US:17", "Indiana": "US:18", "Iowa": "US:19",
            "Kansas": "US:20", "Kentucky": "US:21", "Louisiana": "US:22",
            "Maine": "US:23", "Maryland": "US:24", "Massachusetts": "US:25",
            "Michigan": "US:26", "Minnesota": "US:27", "Mississippi": "US:28",
            "Missouri": "US:29", "Montana": "US:30", "Nebraska": "US:31",
            "Nevada": "US:32", "New Hampshire": "US:33", "New Jersey": "US:34",
            "New Mexico": "US:35", "New York": "US:36", "North Carolina": "US:37",
            "North Dakota": "US:38", "Ohio": "US:39", "Oklahoma": "US:40",
            "Oregon": "US:41", "Pennsylvania": "US:42", "Rhode Island": "US:44",
            "South Carolina": "US:45", "South Dakota": "US:46", "Tennessee": "US:47",
            "Texas": "US:48", "Utah": "US:49", "Vermont": "US:50",
            "Virginia": "US:51", "Washington": "US:53", "West Virginia": "US:54",
            "Wisconsin": "US:55", "Wyoming": "US:56",
        }
        _wqp_state_name = st.selectbox(
            "State",
            list(_WQP_STATES.keys()),
            index=list(_WQP_STATES.keys()).index("California"),
        )
        kwargs["state_code"] = _WQP_STATES[_wqp_state_name]

    elif source_key == "aquastat":
        st.caption("FAO AQUASTAT global water resources and agricultural water use indicators.")
        _AQUASTAT_COUNTRIES = {
            "Global (all countries)": "all", "Egypt": "EGY", "India": "IND",
            "United States": "USA", "Brazil": "BRA", "China": "CHN",
            "France": "FRA", "Germany": "DEU", "Nigeria": "NGA",
            "Australia": "AUS", "Mexico": "MEX", "Spain": "ESP",
        }
        _ac = st.selectbox("Country", list(_AQUASTAT_COUNTRIES.keys()))
        kwargs["country_code"] = _AQUASTAT_COUNTRIES[_ac]
        _yc1, _yc2 = st.columns(2)
        kwargs["start_year"] = int(_yc1.number_input("Start year", 1960, 2023, 2000, step=1))
        kwargs["end_year"] = int(_yc2.number_input("End year", 1960, 2023, 2023, step=1))

    elif source_key == "eu_wfd":
        st.caption("EU Water Framework Directive monitoring via the EEA DiscoData API.")
        _EU_COUNTRIES = {
            "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT",
            "Netherlands": "NL", "Poland": "PL", "Austria": "AT",
            "Belgium": "BE", "Sweden": "SE", "Finland": "FI",
        }
        _ec = st.selectbox("Country", list(_EU_COUNTRIES.keys()))
        kwargs["country"] = _EU_COUNTRIES[_ec]
        kwargs["water_body_type"] = st.selectbox("Water body type", ["river", "lake", "groundwater"])
        if st.checkbox("Filter by year"):
            kwargs["year"] = int(st.number_input("Year", 2000, 2023, 2018, step=1))

    elif source_key == "japan_mlit":
        st.caption("Japan MLIT Water Information System.")
        kwargs["prefecture"] = st.selectbox(
            "Prefecture",
            ["Tokyo", "Osaka", "Kyoto", "Aichi", "Niigata", "Hokkaido", "Fukuoka"],
        )
        kwargs["parameter"] = st.selectbox(
            "Parameter", ["water_level", "discharge", "water_quality", "rainfall"]
        )
        _jc1, _jc2 = st.columns(2)
        _jsd = _jc1.date_input("Start Date (optional)", value=None, key="mlit_start")
        _jed = _jc2.date_input("End Date (optional)", value=None, key="mlit_end")
        if _jsd:
            kwargs["start_date"] = str(_jsd)
        if _jed:
            kwargs["end_date"] = str(_jed)

    elif source_key == "korea_wamis":
        st.caption("Korea WAMIS (Water Resources Management Information System).")
        kwargs["basin"] = st.selectbox(
            "Basin", ["Han", "Nakdong", "Geum", "Yeongsan", "Seomjin"]
        )
        kwargs["parameter"] = st.selectbox(
            "Parameter", ["water_level", "discharge", "water_quality", "dam_storage"]
        )
        _kc1, _kc2 = st.columns(2)
        _ksd = _kc1.date_input("Start Date (optional)", value=None, key="wamis_start")
        _ked = _kc2.date_input("End Date (optional)", value=None, key="wamis_end")
        if _ksd:
            kwargs["start_date"] = str(_ksd)
        if _ked:
            kwargs["end_date"] = str(_ked)

    elif source_key == "wapor":
        st.caption("FAO WaPOR remote-sensing evapotranspiration and productivity data.")
        kwargs["variable"] = st.selectbox(
            "Variable",
            ["RET", "AETI", "NPP"],
            format_func=lambda v: {
                "RET": "RET — reference evapotranspiration",
                "AETI": "AETI — actual ET & interception",
                "NPP": "NPP — net primary production",
            }[v],
        )
        _wc1, _wc2 = st.columns(2)
        kwargs["start_date"] = str(_wc1.date_input("Start Date", key="wapor_start"))
        kwargs["end_date"] = str(_wc2.date_input("End Date", key="wapor_end"))
        _bbox_str = st.text_input(
            "Bounding box (west,south,east,north)", placeholder="31.0,29.0,32.0,30.5"
        )
        if _bbox_str.strip():
            try:
                kwargs["bbox"] = tuple(float(x) for x in _bbox_str.split(","))
            except ValueError:
                st.warning("Bounding box must be four comma-separated numbers.")

    if api_key:
        kwargs["api_key"] = api_key

    # P3: Use type="primary" with blue theme from config.toml (no longer red)
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
                st.dataframe(df.head(100), width="stretch")

                st.download_button(
                    "⬇️ Download data",
                    data=df.to_csv(index=False) if output_fmt == "csv" else df.to_json(orient="records", indent=2, date_format="iso"),
                    file_name=f"aquascope_{source_key}.{output_fmt}",
                    mime="text/csv" if output_fmt == "csv" else "application/json",
                )

            except Exception as exc:
                st.error(f"Collection failed: {exc}")
                logger.exception("Data collection error")

    st.divider()
    st.caption("Don't have API credentials yet? Use **Load demo dataset** on the Analysis page to explore the dashboard with sample water quality data.")


# ---------------------------------------------------------------------------
# Page: Analysis
# ---------------------------------------------------------------------------


def page_analysis() -> None:
    """Render the Analysis page (EDA + Quality Assessment)."""
    st = _require_streamlit()

    st.title("🔬 Analysis")
    st.markdown("Run exploratory data analysis and quality assessment on your water data.")

    # P4: Workflow progress
    _show_workflow_step(st, 1)
    st.markdown("")

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
            # P1: Demo data CTA in empty state
            st.info("No data in session. Collect data first, upload a file, or load the demo dataset.")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption("The demo dataset contains 180 days of synthetic water quality readings (pH, DO, turbidity, nitrate) with seasonal patterns — enough to explore all dashboard features.")
            with col2:
                if st.button("Load demo dataset", use_container_width=True, key="demo_analysis"):
                    st.session_state["collected_data"] = _load_demo_data()
                    st.session_state["collected_source"] = "demo"
                    st.rerun()
            return
        src_label = st.session_state.get("collected_source", "session")
        st.success(f"Using session data: {len(df)} records ({src_label})")
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

    st.dataframe(df.head(20), width="stretch")
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
                        st.dataframe(pd.DataFrame(param_rows), width="stretch")

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
                            st.dataframe(cleaned.head(20), width="stretch")

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

    # P4: Workflow progress
    _show_workflow_step(st, 2)
    st.markdown("")

    df = st.session_state.get("collected_data")
    if df is None:
        # P1: Demo data CTA in empty state
        st.info("No data in session. Collect or upload data first, or load the demo dataset below.")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("The demo dataset includes time-series, multi-parameter, and station map data — perfect for exploring all 9 plot types.")
        with col2:
            if st.button("Load demo dataset", use_container_width=True, key="demo_viz"):
                st.session_state["collected_data"] = _load_demo_data()
                st.session_state["collected_source"] = "demo"
                st.rerun()
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

    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(exclude="number").columns)

    default_val = "value" if "value" in num_cols else (num_cols[0] if num_cols else None)
    default_grp = next((c for c in ("parameter", "station_name", "station_id") if c in cat_cols), cat_cols[0] if cat_cols else None)

    if not num_cols or not cat_cols:
        st.warning("Need at least one numeric column and one categorical column for a box plot.")
        return

    value_col = st.selectbox("Value column", num_cols, index=num_cols.index(default_val) if default_val else 0, key="bp_val")
    group_col = st.selectbox("Group column", cat_cols, index=cat_cols.index(default_grp) if default_grp else 0, key="bp_grp")

    fig = plot_boxplot(df, value_col=value_col, group_col=group_col, title=title or "Box Plot")
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
    import pandas as pd

    from aquascope.viz import plot_who_exceedances

    if "parameter" not in df.columns or "value" not in df.columns:
        st.warning("WHO Exceedances requires `parameter` and `value` columns.")
        return

    rows = []
    for param, (lo, hi, _unit) in WHO_GUIDELINES.items():
        subset = df[df["parameter"].str.lower() == param]["value"].dropna()
        if subset.empty:
            continue
        n = len(subset)
        if hi == float("inf"):
            n_exceed = int((subset < lo).sum())
        elif lo == 0:
            n_exceed = int((subset > hi).sum())
        else:
            n_exceed = int(((subset < lo) | (subset > hi)).sum())
        pct = n_exceed / n * 100
        rows.append({"variable": param, "pct_exceedances": round(pct, 1), "status": "FAIL" if n_exceed > 0 else "PASS"})

    if not rows:
        st.info("No WHO-monitored parameters found in dataset.")
        return

    who_df = pd.DataFrame(rows)
    fig = plot_who_exceedances(who_df, title=title or "WHO Guideline Exceedances")
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

    num_cols = list(df.select_dtypes(include="number").columns)
    if not num_cols:
        st.warning("No numeric columns found for Flow Duration Curve.")
        return
    default = "discharge" if "discharge" in num_cols else num_cols[0]
    col = st.selectbox("Discharge column", num_cols, index=num_cols.index(default), key="fdc_col")

    fig = plot_fdc(df[col].dropna(), title=title or "Flow Duration Curve")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_hydrograph(st, df: pd.DataFrame, title: str) -> None:
    """Render a hydrograph."""
    import pandas as pd

    from aquascope.viz import plot_hydrograph

    num_cols = list(df.select_dtypes(include="number").columns)
    if not num_cols:
        st.warning("No numeric columns found for Hydrograph.")
        return
    default = "discharge" if "discharge" in num_cols else num_cols[0]
    total_col = st.selectbox("Discharge column", num_cols, index=num_cols.index(default), key="hg_col")

    plot_df = df.copy()
    for dt_col in ("sample_datetime", "reading_datetime", "date", "datetime"):
        if dt_col in plot_df.columns:
            plot_df.index = pd.to_datetime(plot_df[dt_col])
            break

    fig = plot_hydrograph(plot_df, total_col=total_col, baseflow_col=None, title=title or "Hydrograph")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_spi_timeline(st, df: pd.DataFrame, title: str) -> None:
    """Render an SPI timeline."""
    import pandas as pd

    from aquascope.viz import plot_spi_timeline

    num_cols = list(df.select_dtypes(include="number").columns)
    if not num_cols:
        st.warning("No numeric columns found for SPI Timeline.")
        return
    spi_candidates = [c for c in num_cols if "spi" in c.lower()] or num_cols
    spi_col = st.selectbox("SPI column", spi_candidates, key="spi_col")

    plot_df = df[[spi_col]].copy()
    for dt_col in ("sample_datetime", "reading_datetime", "date", "datetime"):
        if dt_col in df.columns:
            plot_df.index = pd.to_datetime(df[dt_col])
            break

    fig = plot_spi_timeline(plot_df, spi_col=spi_col, title=title or "SPI Timeline")
    st.pyplot(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render_return_periods(st, df: pd.DataFrame, title: str) -> None:
    """Render return period plot — fits GEV to the selected column first."""
    import pandas as pd

    from aquascope.hydrology import fit_gev
    from aquascope.viz import plot_return_periods

    num_cols = list(df.select_dtypes(include="number").columns)
    if not num_cols:
        st.warning("No numeric columns found for Return Periods.")
        return
    default = "discharge" if "discharge" in num_cols else num_cols[0]
    col = st.selectbox("Discharge column", num_cols, index=num_cols.index(default), key="rp_col")

    q = df[col].dropna()
    # fit_gev needs a DatetimeIndex to extract annual maxima
    for dt_col in ("sample_datetime", "reading_datetime", "date", "datetime"):
        if dt_col in df.columns:
            q.index = pd.to_datetime(df.loc[q.index, dt_col])
            break

    with st.spinner("Fitting GEV distribution…"):
        result = fit_gev(q)

    fig = plot_return_periods(result.return_periods, observed_max=float(q.max()), title=title or "Return Periods")
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

    # P4: Workflow progress
    _show_workflow_step(st, 3)
    st.markdown("")

    df = st.session_state.get("collected_data")
    if df is None:
        # P1: Demo data CTA in empty state
        st.info("No data in session. Collect discharge data first or load the demo dataset.")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("The demo dataset includes a `discharge` column with seasonal flow patterns — ready for FDC, baseflow separation, and flood frequency analysis.")
        with col2:
            if st.button("Load demo dataset", use_container_width=True, key="demo_hydro"):
                st.session_state["collected_data"] = _load_demo_data()
                st.session_state["collected_source"] = "demo"
                st.rerun()
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
    import pandas as pd

    q50 = result.percentiles.get(50, float("nan"))
    q95 = result.percentiles.get(95, float("nan"))
    col1, col2 = st.columns(2)
    col1.metric("Q50 (median)", f"{q50:.3f}")
    col2.metric("Q95 (low-flow)", f"{q95:.3f}")

    from aquascope.viz import plot_fdc

    fig = plot_fdc(pd.Series(result.discharge))
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
        result = lyne_hollick(q, alpha=alpha, n_passes=passes)
    else:
        bfi_max = st.slider("BFI_max", 0.1, 1.0, 0.8, 0.05)
        result = eckhardt(q, alpha=alpha, bfi_max=bfi_max)

    st.subheader("Results")
    st.metric("Baseflow Index (BFI)", f"{result.bfi:.3f}")

    from aquascope.viz import plot_hydrograph

    fig = plot_hydrograph(result.df, total_col="total", baseflow_col="baseflow")
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Recession constant (K)", f"{result.recession_constant:.4f}" if result.recession_constant else "N/A")
    col2.metric("Segments found", str(len(result.segments)))
    col3.metric("R²", f"{result.r_squared:.3f}" if result.r_squared else "N/A")

    if result.segments:
        import pandas as pd

        seg_data = [
            {
                "Start": s.start,
                "End": s.end,
                "Duration (days)": (s.end - s.start).days if hasattr(s.end, "days") else len(s.discharge),
            }
            for s in result.segments
        ]
        st.dataframe(pd.DataFrame(seg_data), width="stretch")


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
    shape, loc, scale = result.params if len(result.params) == 3 else (0.0, 0.0, 1.0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Shape (ξ)", f"{shape:.4f}")
    col2.metric("Location (μ)", f"{loc:.2f}")
    col3.metric("Scale (σ)", f"{scale:.2f}")

    if result.return_periods:
        import pandas as pd

        rl_data = [{"Return Period (yr)": rp, "Discharge": round(val, 2)} for rp, val in result.return_periods.items()]
        st.dataframe(pd.DataFrame(rl_data), width="stretch")

    from aquascope.viz import plot_return_periods

    fig = plot_return_periods(result.return_periods, observed_max=float(q.max()))
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page: AI Recommender
# ---------------------------------------------------------------------------

_PROVIDER_LABELS = {
    "rule_based": "Rule-based (free, no key needed)",
    "huggingface": "HuggingFace Inference API (free)",
    "groq": "Groq (free tier — fast open-source models)",
    "openai": "OpenAI",
    "ollama": "Ollama (local)",
}

_PROVIDER_KEY_LINKS = {
    "huggingface": ("Get free HF token", "https://huggingface.co/settings/tokens"),
    "groq": ("Get free Groq key", "https://console.groq.com/keys"),
    "openai": ("OpenAI API keys", "https://platform.openai.com/api-keys"),
    "ollama": None,
}


def _render_llm_config(st) -> dict | None:
    """Render LLM provider config UI; returns config dict or None for rule-based."""
    from aquascope.ai_engine.recommender import PROVIDER_BASE_URLS, PROVIDER_MODELS

    with st.expander("⚙️ LLM Enhancement (optional)", expanded=False):
        st.caption(
            "Augment rule-based scoring with a language model for richer rationales. "
            "HuggingFace and Groq both offer free tiers — no credit card required."
        )
        provider = st.selectbox(
            "Provider",
            list(_PROVIDER_LABELS.keys()),
            format_func=lambda k: _PROVIDER_LABELS[k],
            key="llm_provider",
        )
        if provider == "rule_based":
            return None

        default_models = PROVIDER_MODELS.get(provider, [])
        model = st.selectbox(
            "Model",
            default_models + ["(custom)"],
            key="llm_model_select",
        )
        if model == "(custom)":
            model = st.text_input("Custom model name", key="llm_model_custom")

        link_info = _PROVIDER_KEY_LINKS.get(provider)
        if link_info:
            label, url = link_info
            st.caption(f"[{label}]({url})")

        if provider == "ollama":
            base_url = st.text_input(
                "Ollama base URL",
                value=PROVIDER_BASE_URLS["ollama"],
                key="llm_base_url",
            )
            api_key = None
        else:
            api_key = st.text_input("API Key", type="password", key="llm_api_key")
            base_url = PROVIDER_BASE_URLS.get(provider)

        if not model:
            st.warning("Select or enter a model name.")
            return None

        return {"provider": provider, "model": model, "api_key": api_key or None, "base_url": base_url}


def page_ai_recommender() -> None:
    """Render the AI Recommender page."""
    st = _require_streamlit()

    st.title("🤖 AI Methodology Recommender")
    st.markdown("Get research methodology recommendations based on your dataset characteristics.")

    llm_config = _render_llm_config(st)

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
            _run_recommendations(st, goal, parameters, scope, keywords, n_records, n_stations, years, top_k, llm_config)

    with tab_auto:
        df = st.session_state.get("collected_data")
        if df is None:
            st.info("No data in session. Load demo data or collect data first, then return here for auto-recommendations.")
            if st.button("Load demo dataset", key="demo_ai", use_container_width=False):
                st.session_state["collected_data"] = _load_demo_data()
                st.session_state["collected_source"] = "demo"
                st.rerun()
            return

        st.success(f"Dataset loaded: {len(df)} records")
        goal_auto = st.text_area("Research goal (optional)", key="goal_auto")
        top_k_auto = st.slider("Number of recommendations", 1, 20, 5, key="topk_auto")

        if st.button("🔍 Auto-recommend from Data", key="btn_rec_auto"):
            with st.spinner("Profiling dataset and generating recommendations…"):
                try:
                    from aquascope.ai_engine.recommender import recommend, recommend_with_llm
                    from aquascope.analysis.eda import profile_dataset

                    profile = profile_dataset(df)
                    if goal_auto:
                        profile.research_goal = goal_auto

                    if llm_config:
                        recs = recommend_with_llm(
                            profile, top_k=top_k_auto,
                            model=llm_config["model"],
                            api_key=llm_config["api_key"],
                            base_url=llm_config["base_url"],
                        )
                    else:
                        recs = recommend(profile, top_k=top_k_auto)
                    _display_recommendations(st, recs)
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")
                    logger.exception("AI recommender error")


def _run_recommendations(
    st, goal: str, parameters: str, scope: str, keywords: str,
    n_records: int, n_stations: int, years: float, top_k: int,
    llm_config: dict | None = None,
) -> None:
    """Run the recommender (rule-based, or LLM-enhanced if llm_config provided)."""
    with st.spinner("Generating recommendations…"):
        try:
            from aquascope.ai_engine.recommender import DatasetProfile, recommend, recommend_with_llm

            profile = DatasetProfile(
                parameters=[p.strip() for p in parameters.split(",") if p.strip()],
                n_records=n_records,
                n_stations=n_stations,
                time_span_years=years,
                geographic_scope=scope,
                research_goal=goal,
                keywords=[k.strip() for k in keywords.split(",") if k.strip()],
            )

            if llm_config:
                recs = recommend_with_llm(
                    profile, top_k=top_k,
                    model=llm_config["model"],
                    api_key=llm_config["api_key"],
                    base_url=llm_config["base_url"],
                )
            else:
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
            if rec.methodology.applicable_parameters:
                st.markdown(f"**Applicable parameters:** {', '.join(rec.methodology.applicable_parameters)}")
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
        # P1: Demo data CTA in empty state
        st.info("No data in session. Collect water quality data first or load the demo dataset.")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("The demo dataset includes pH, dissolved oxygen, turbidity, and nitrate — all checked against WHO guidelines, with intentional exceedances for illustration.")
        with col2:
            if st.button("Load demo dataset", use_container_width=True, key="demo_alerts"):
                st.session_state["collected_data"] = _load_demo_data()
                st.session_state["collected_source"] = "demo"
                st.rerun()
        return

    # Show reference thresholds
    with st.expander("📋 WHO Guideline Reference", expanded=False):
        import pandas as pd

        ref_rows = [
            {"Parameter": param, "Min": str(lo), "Max": "∞" if hi == float("inf") else str(hi), "Unit": unit}
            for param, (lo, hi, unit) in WHO_GUIDELINES.items()
        ]
        st.dataframe(pd.DataFrame(ref_rows), width="stretch")

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

                if exceedances.empty:
                    st.info("No WHO guideline data available for the loaded parameters.")
                else:
                    exceeded = exceedances[exceedances["status"] == "EXCEEDANCE"]
                    if not exceeded.empty:
                        st.warning(f"⚠️ Found exceedances in **{len(exceeded)}** parameters")
                    else:
                        st.success("✅ All parameters within WHO guidelines")
                    st.dataframe(exceedances[["variable", "n_measurements", "mean", "n_exceedances", "pct_exceedances", "guideline_low", "guideline_high", "status"]])
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
        st.dataframe(result_df, width="stretch")

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
        # P8: "auto" collapses sidebar on mobile automatically
        initial_sidebar_state="auto",
    )

    # P5 + P3: Inject global CSS — hides Deploy button, styles nav radio as link list
    _inject_global_css(st)

    page_labels = [label for label, _ in _PAGES.values()]

    # P5: Sidebar uses H2 markdown (not st.sidebar.title which renders as H1)
    # so the main page title remains the sole H1. (P6)
    st.sidebar.markdown("## 🌊 AquaScope")
    st.sidebar.markdown("---")

    # P2 + P5: "Navigate" label hidden via label_visibility; CSS removes radio dots
    # so it looks like a clean link list.
    # Apply pending navigation BEFORE the radio widget is instantiated — writing to a
    # widget-bound key after instantiation raises StreamlitAPIException.
    if "_nav_pending" in st.session_state:
        st.session_state["current_page"] = st.session_state.pop("_nav_pending")
    elif "current_page" not in st.session_state:
        st.session_state["current_page"] = page_labels[0]

    selected_label = st.sidebar.radio(
        "Navigate",
        page_labels,
        key="current_page",
        label_visibility="collapsed",
    )

    # Map label back to page key
    label_to_key = {label: key for key, (label, _) in _PAGES.items()}
    selected_key = label_to_key[selected_label]

    from aquascope import __version__ as _aquascope_version

    st.sidebar.markdown("---")
    st.sidebar.caption(f"AquaScope v{_aquascope_version} | Dashboard v{__version__}")

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
