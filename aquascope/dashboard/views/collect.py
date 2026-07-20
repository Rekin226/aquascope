"""Collect Data page — every AquaScope source, file upload, and demo data."""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import streamlit as st

from aquascope.dashboard import _insights, _state

logger = logging.getLogger(__name__)

# key -> (label, region, one-line description)
SOURCES: dict[str, tuple[str, str, str]] = {
    "usgs": ("USGS Water Services", "United States",
             "Real-time discharge, gauge height, temperature from thousands of US gauges"),
    "grdc": ("GRDC river discharge", "Global", "In-situ gauges (Zenodo subset) + RSEG satellite discharge estimates"),
    "openmeteo": ("Open-Meteo", "Global", "Weather history, forecasts and GloFAS flood discharge for any coordinate"),
    "sdg6": ("UN SDG 6 indicators", "Global", "Country-level water & sanitation indicators (water stress, IWRM, …)"),
    "gemstat": ("GEMStat water quality", "Global",
                "UNEP global surface & groundwater quality archive (~200 MB, cached locally)"),
    "aquastat": ("FAO AQUASTAT", "Global", "National water resources and agricultural water-use statistics"),
    "wapor": ("FAO WaPOR", "Africa & Near East", "Remote-sensing evapotranspiration and biomass productivity rasters"),
    "copernicus": ("Copernicus CDS", "Global", "ERA5 / GloFAS climate reanalysis (requires free CDS key)"),
    "wqp": ("Water Quality Portal", "United States", "EPA/USGS harmonised water-quality samples by state"),
    "hubeau_hydrometrie": ("Hub'Eau hydrométrie", "France",
                           "Real-time water level & discharge from French national gauges"),
    "eu_wfd": ("EU Water Framework Directive", "Europe",
               "EEA DiscoData ecological/chemical status of European water bodies"),
    "taiwan_moenv": ("Taiwan MOENV", "Taiwan",
                     "River water-quality monitoring (requires free MOENV key)"),
    "taiwan_wra_level": ("Taiwan WRA water level", "Taiwan",
                         "Real-time river stage snapshot across all WRA stations"),
    "taiwan_wra_reservoir": ("Taiwan WRA reservoirs", "Taiwan", "Daily reservoir storage and operations"),
    "taiwan_wra_fhy": ("Taiwan WRA FHY real-time", "Taiwan",
                       "Real-time water level / rainfall / discharge (FHY portal)"),
    "taiwan_wra_iot": ("Taiwan WRA IoT", "Taiwan", "Real-time groundwater level and rainfall accumulation"),
    "taiwan_datagov": ("Taiwan data.gov.tw", "Taiwan", "Open-government real-time river & groundwater levels"),
    "taiwan_civil_iot": ("Taiwan Civil IoT", "Taiwan", "SensorThings water observations (flood sensors etc.)"),
    "japan_mlit": ("Japan MLIT", "Japan", "Water level, discharge, quality and rainfall by prefecture"),
    "korea_wamis": ("Korea WAMIS", "South Korea", "Water level, discharge, quality and dam storage by basin"),
    "india_wris": ("India WRIS", "India", "River water level by state / district / agency"),
}

_API_KEY_SOURCES: dict[str, tuple[str, str]] = {
    "taiwan_moenv": ("Taiwan MOENV", "https://data.moenv.gov.tw/en/apikey"),
    "copernicus": ("Copernicus CDS", "https://cds.climate.copernicus.eu/how-to-api"),
}

_REGION_ORDER = ["Global", "United States", "Europe", "France", "Taiwan", "Japan",
                 "South Korea", "India", "Africa & Near East"]


def render() -> None:
    st.title("📡 Collect Data")
    st.markdown(
        f"Pull water data from **{len(SOURCES)} live sources**, upload your own file, "
        "or start instantly with a demo dataset. Whatever you load becomes the shared "
        "**workspace dataset** used by every analysis page."
    )

    tab_api, tab_upload, tab_demo = st.tabs(["🌐 Live sources", "📁 Upload file", "✨ Demo data"])

    with tab_api:
        _render_api_tab()
    with tab_upload:
        _render_upload_tab()
    with tab_demo:
        _render_demo_tab()


# ---------------------------------------------------------------------------
# Live sources
# ---------------------------------------------------------------------------


def _render_api_tab() -> None:
    import sys

    if sys.platform == "emscripten":
        st.info(
            "🧪 **In-browser demo** — requests go straight from your browser to each "
            "data provider, so only CORS-friendly APIs respond. Verified working here: "
            "**Open-Meteo, USGS, Hub'Eau, UN SDG 6, FAO AQUASTAT, Taiwan WRA**. "
            "Sources that block cross-origin requests, and large-archive sources "
            "(GEMStat, GRDC in-situ), need a local install: "
            "`pip install \"aquascope[dashboard]\"`."
        )

    c_region, c_source = st.columns([1, 2])
    regions = ["All regions"] + [r for r in _REGION_ORDER if any(v[1] == r for v in SOURCES.values())]
    region = c_region.selectbox("Region", regions)

    keys = [k for k, v in SOURCES.items() if region == "All regions" or v[1] == region]

    def _label(key: str) -> str:
        label, reg, _ = SOURCES[key]
        prefix = "🔑 " if key in _API_KEY_SOURCES else ""
        return f"{prefix}{label} · {reg}"

    source_key = c_source.selectbox("Data source", keys, format_func=_label)
    st.caption(SOURCES[source_key][2])

    if source_key in _API_KEY_SOURCES:
        provider, signup_url = _API_KEY_SOURCES[source_key]
        st.info(
            f"🔑 **{provider} requires a free API key.** Get one at "
            f"[{signup_url}]({signup_url}) and paste it below."
        )
        api_key = st.text_input("API key", type="password", key=f"key_{source_key}")
    else:
        api_key = ""

    ctor_kwargs: dict = {}
    fetch_kwargs: dict = {}
    _source_form(source_key, ctor_kwargs, fetch_kwargs)

    if st.button("🚀 Collect data", type="primary", key="collect_btn"):
        label = SOURCES[source_key][0]
        with st.spinner(f"Collecting from {label}…"):
            try:
                records = _run_collector(source_key, api_key, ctor_kwargs, fetch_kwargs)
                if not records:
                    st.warning("The source returned no records for these parameters.")
                    return
                df = _records_to_df(records)
                _state.set_data(df, source_key, label)
                st.success(f"✅ Collected **{len(df):,} records** from {label} — saved to the workspace.")
            except Exception as exc:  # noqa: BLE001 — surface any API failure to the user
                st.error(f"Collection failed: {exc}")
                logger.exception("Data collection error")
                return

        _insights.render_panel(df, key_prefix="collect")
        with st.expander("Preview (first 100 rows)", expanded=False):
            st.dataframe(df.head(100), width="stretch")
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name=f"aquascope_{source_key}.csv",
            mime="text/csv",
        )


def _source_form(source_key: str, ctor: dict, fetch: dict) -> None:  # noqa: C901, PLR0915 — one branch per source
    """Render source-specific parameter widgets, filling ctor/fetch kwargs."""
    if source_key == "taiwan_wra_level":
        st.info("Real-time snapshot — returns current readings from all river stations.")

    elif source_key == "taiwan_wra_reservoir":
        st.info("Daily snapshot — returns the most recent day's reservoir data.")

    elif source_key == "taiwan_wra_fhy":
        ctor["data_type"] = st.selectbox(
            "Data type", ["water", "rainfall", "flow"],
            format_func=lambda v: {"water": "Water level", "rainfall": "Rainfall", "flow": "River discharge"}[v],
        )

    elif source_key == "taiwan_wra_iot":
        ctor["data_type"] = st.selectbox(
            "Data type", ["groundwater", "rainfall"],
            format_func=lambda v: {"groundwater": "Groundwater level", "rainfall": "Rainfall accumulation"}[v],
        )

    elif source_key == "taiwan_datagov":
        ctor["dataset_id"] = st.selectbox(
            "Dataset", ["25768", "161082"],
            format_func=lambda v: {"25768": "River water level (real-time)",
                                   "161082": "Groundwater level (real-time)"}[v],
        )
        fetch["limit"] = st.slider("Max records", 100, 5_000, 1_000, step=100)

    elif source_key == "taiwan_moenv":
        fetch["limit"] = st.slider("Records to fetch (most recent first)", 100, 5_000, 500, step=100)

    elif source_key == "taiwan_civil_iot":
        st.caption("SensorThings API — filter by date range to narrow observations.")
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start date", value=None, key="ciot_start")
        ed = c2.date_input("End date", value=None, key="ciot_end")
        if sd:
            fetch["start_date"] = str(sd)
        if ed:
            fetch["end_date"] = str(ed)

    elif source_key == "grdc":
        mode = st.radio(
            "Source type", ["in_situ", "satellite"], horizontal=True,
            format_func=lambda v: {"in_situ": "In-situ gauges (Zenodo)", "satellite": "Satellite RSEG (DaRUS)"}[v],
        )
        fetch["source_type"] = mode
        st.caption("First run downloads and caches the archive locally — allow a few minutes.")

    elif source_key == "hubeau_hydrometrie":
        c1, c2 = st.columns(2)
        grandeur = c1.selectbox(
            "Quantity", ["both", "Q", "H"],
            format_func=lambda v: {"both": "Discharge + water level", "Q": "Discharge (Q)", "H": "Water level (H)"}[v],
        )
        if grandeur != "both":
            fetch["grandeur_hydro"] = grandeur
        station = c2.text_input("Station code (optional)", placeholder="K002000101")
        if station.strip():
            fetch["code_station"] = station.strip()
        fetch["max_items"] = st.slider("Max records", 500, 20_000, 5_000, step=500)

    elif source_key == "india_wris":
        c1, c2 = st.columns(2)
        fetch["state_name"] = c1.text_input("State", value="Assam")
        fetch["district_name"] = c2.text_input("District", value="Kamrup")
        fetch["agency_name"] = st.text_input("Agency", value="CWC")
        c3, c4 = st.columns(2)
        fetch["startdate"] = str(c3.date_input("Start date", key="wris_start"))
        fetch["enddate"] = str(c4.date_input("End date", key="wris_end"))

    elif source_key in ("openmeteo", "copernicus"):
        c1, c2 = st.columns(2)
        fetch["latitude"] = c1.number_input("Latitude", value=25.0, min_value=-90.0, max_value=90.0)
        fetch["longitude"] = c2.number_input("Longitude", value=121.5, min_value=-180.0, max_value=180.0)
        c3, c4 = st.columns(2)
        fetch["start_date"] = str(c3.date_input("Start date"))
        fetch["end_date"] = str(c4.date_input("End date"))
        if source_key == "openmeteo":
            ctor["mode"] = st.selectbox(
                "Mode", ["weather", "forecast", "flood"],
                format_func=lambda v: {"weather": "Weather history", "forecast": "Forecast",
                                       "flood": "GloFAS flood discharge"}[v],
            )

    elif source_key == "usgs":
        fetch["days"] = st.slider("Days of data", 1, 30, 3)
        st.caption("USGS covers thousands of US stations — use a region filter to keep responses fast.")
        regions = {
            "Northeast US": "-80,37,-66,48",
            "Southeast US": "-92,24,-80,37",
            "Midwest US": "-104,36,-80,48",
            "Pacific Northwest": "-125,42,-104,50",
            "Southwest US": "-125,32,-104,42",
            "No filter (all US — slow)": None,
            "Custom bbox": "__custom__",
        }
        region_label = st.selectbox("Region filter", list(regions.keys()), index=0)
        bbox_val = regions[region_label]
        if bbox_val == "__custom__":
            bbox_val = st.text_input("Bounding box (minLon,minLat,maxLon,maxLat)", placeholder="-80,37,-66,48") or None
        if bbox_val:
            fetch["bbox"] = bbox_val
        fetch["max_items"] = st.slider("Max records", 100, 10_000, 2_000, step=100)

    elif source_key == "sdg6":
        countries = {
            "Germany": "DEU", "United States": "USA", "India": "IND", "Japan": "JPN",
            "France": "FRA", "United Kingdom": "GBR", "Brazil": "BRA", "China": "CHN",
            "Australia": "AUS", "South Korea": "KOR", "Indonesia": "IDN", "Philippines": "PHL",
            "Vietnam": "VNM", "Thailand": "THA", "Mexico": "MEX", "Spain": "ESP",
            "Italy": "ITA", "Netherlands": "NLD", "Sweden": "SWE", "Turkey": "TUR",
            "Egypt": "EGY", "Nigeria": "NGA", "Kenya": "KEN", "South Africa": "ZAF",
            "Saudi Arabia": "SAU", "Israel": "ISR", "Iran": "IRN", "Pakistan": "PAK",
            "Bangladesh": "BGD", "Singapore": "SGP",
        }
        selected = st.multiselect(
            "Countries", list(countries.keys()), default=["Germany"],
            help="Taiwan is not included in UN SDG data.",
        )
        fetch["country_codes"] = ",".join(countries[n] for n in selected) if selected else None
        labels = {
            "6.1.1": "Safely managed drinking water", "6.2.1": "Safely managed sanitation",
            "6.3.1": "Safely treated wastewater", "6.3.2": "Good ambient water quality",
            "6.4.1": "Water-use efficiency", "6.4.2": "Water stress",
            "6.5.1": "IWRM implementation", "6.5.2": "Transboundary cooperation",
            "6.6.1": "Water-related ecosystems",
        }
        fetch["indicator_codes"] = [
            st.selectbox("Indicator", list(labels.keys()), index=5, format_func=lambda c: f"{c} ({labels[c]})")
        ]

    elif source_key == "gemstat":
        st.info(
            "📦 GEMStat is a ~200 MB Zenodo archive. The first collection downloads and "
            "caches it locally (1–3 min); later runs are instant."
        )
        countries = [
            "Argentina", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
            "Canada", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland",
            "France", "Germany", "Greece", "Hungary", "Iceland", "India", "Ireland",
            "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
            "Macedonia (the former Yugoslav Republic of)", "Malta", "Mexico",
            "Montenegro", "Netherlands (-the )", "Norway", "Poland", "Portugal",
            "Romania", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
            "Switzerland", "Turkey", "United States of America (the)", "Uruguay",
        ]
        fetch["country"] = st.selectbox("Country", countries, index=countries.index("Germany"))
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start date (optional)", value=None, key="gemstat_start")
        ed = c2.date_input("End date (optional)", value=None, key="gemstat_end")
        if sd:
            fetch["start_date"] = str(sd)
        if ed:
            fetch["end_date"] = str(ed)
        fetch["max_records"] = st.slider("Max records", 500, 20_000, 5_000, step=500)

    elif source_key == "wqp":
        states = {
            "Alabama": "US:01", "Alaska": "US:02", "Arizona": "US:04", "Arkansas": "US:05",
            "California": "US:06", "Colorado": "US:08", "Connecticut": "US:09", "Delaware": "US:10",
            "Florida": "US:12", "Georgia": "US:13", "Hawaii": "US:15", "Idaho": "US:16",
            "Illinois": "US:17", "Indiana": "US:18", "Iowa": "US:19", "Kansas": "US:20",
            "Kentucky": "US:21", "Louisiana": "US:22", "Maine": "US:23", "Maryland": "US:24",
            "Massachusetts": "US:25", "Michigan": "US:26", "Minnesota": "US:27", "Mississippi": "US:28",
            "Missouri": "US:29", "Montana": "US:30", "Nebraska": "US:31", "Nevada": "US:32",
            "New Hampshire": "US:33", "New Jersey": "US:34", "New Mexico": "US:35", "New York": "US:36",
            "North Carolina": "US:37", "North Dakota": "US:38", "Ohio": "US:39", "Oklahoma": "US:40",
            "Oregon": "US:41", "Pennsylvania": "US:42", "Rhode Island": "US:44", "South Carolina": "US:45",
            "South Dakota": "US:46", "Tennessee": "US:47", "Texas": "US:48", "Utah": "US:49",
            "Vermont": "US:50", "Virginia": "US:51", "Washington": "US:53", "West Virginia": "US:54",
            "Wisconsin": "US:55", "Wyoming": "US:56",
        }
        name = st.selectbox("State", list(states.keys()), index=list(states.keys()).index("California"))
        fetch["state_code"] = states[name]

    elif source_key == "aquastat":
        countries = {
            "Global (all countries)": "all", "Egypt": "EGY", "India": "IND", "United States": "USA",
            "Brazil": "BRA", "China": "CHN", "France": "FRA", "Germany": "DEU",
            "Nigeria": "NGA", "Australia": "AUS", "Mexico": "MEX", "Spain": "ESP",
        }
        c = st.selectbox("Country", list(countries.keys()))
        fetch["country_code"] = countries[c]
        c1, c2 = st.columns(2)
        fetch["start_year"] = int(c1.number_input("Start year", 1960, 2023, 2000, step=1))
        fetch["end_year"] = int(c2.number_input("End year", 1960, 2023, 2023, step=1))

    elif source_key == "eu_wfd":
        countries = {
            "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT", "Netherlands": "NL",
            "Poland": "PL", "Austria": "AT", "Belgium": "BE", "Sweden": "SE", "Finland": "FI",
        }
        c = st.selectbox("Country", list(countries.keys()))
        fetch["country"] = countries[c]
        fetch["water_body_type"] = st.selectbox("Water body type", ["river", "lake", "groundwater"])
        if st.checkbox("Filter by year"):
            fetch["year"] = int(st.number_input("Year", 2000, 2023, 2018, step=1))

    elif source_key == "japan_mlit":
        fetch["prefecture"] = st.selectbox(
            "Prefecture", ["Tokyo", "Osaka", "Kyoto", "Aichi", "Niigata", "Hokkaido", "Fukuoka"]
        )
        fetch["parameter"] = st.selectbox("Parameter", ["water_level", "discharge", "water_quality", "rainfall"])
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start date (optional)", value=None, key="mlit_start")
        ed = c2.date_input("End date (optional)", value=None, key="mlit_end")
        if sd:
            fetch["start_date"] = str(sd)
        if ed:
            fetch["end_date"] = str(ed)

    elif source_key == "korea_wamis":
        fetch["basin"] = st.selectbox("Basin", ["Han", "Nakdong", "Geum", "Yeongsan", "Seomjin"])
        fetch["parameter"] = st.selectbox("Parameter", ["water_level", "discharge", "water_quality", "dam_storage"])
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start date (optional)", value=None, key="wamis_start")
        ed = c2.date_input("End date (optional)", value=None, key="wamis_end")
        if sd:
            fetch["start_date"] = str(sd)
        if ed:
            fetch["end_date"] = str(ed)

    elif source_key == "wapor":
        fetch["variable"] = st.selectbox(
            "Variable", ["RET", "AETI", "NPP"],
            format_func=lambda v: {
                "RET": "RET — reference evapotranspiration",
                "AETI": "AETI — actual ET & interception",
                "NPP": "NPP — net primary production",
            }[v],
        )
        c1, c2 = st.columns(2)
        fetch["start_date"] = str(c1.date_input("Start date", key="wapor_start"))
        fetch["end_date"] = str(c2.date_input("End date", key="wapor_end"))
        bbox_str = st.text_input("Bounding box (west,south,east,north)", placeholder="31.0,29.0,32.0,30.5")
        if bbox_str.strip():
            try:
                fetch["bbox"] = tuple(float(x) for x in bbox_str.split(","))
            except ValueError:
                st.warning("Bounding box must be four comma-separated numbers.")


def _records_to_df(records: list) -> pd.DataFrame:
    """Convert schema objects to a DataFrame, flattening nested locations.

    Many AquaScope schemas carry coordinates as a nested ``location`` object;
    flattening to ``latitude``/``longitude`` lets the station map and the
    smart-insights layer pick them up automatically.
    """
    df = pd.DataFrame([r.model_dump() for r in records])
    if "location" in df.columns and "latitude" not in df.columns:
        loc = df["location"].apply(lambda v: v if isinstance(v, dict) else {})
        lat = loc.apply(lambda v: v.get("latitude"))
        lon = loc.apply(lambda v: v.get("longitude"))
        if lat.notna().any():
            df["latitude"] = lat
            df["longitude"] = lon
            df = df.drop(columns=["location"])
    return df


def _run_collector(source_key: str, api_key: str, ctor: dict, fetch: dict):
    """Instantiate the right collector and fetch — covers all 21 sources."""
    from aquascope import collectors as c

    factories = {
        "usgs": lambda: c.USGSCollector(api_key=api_key or "DEMO_KEY"),
        "grdc": lambda: c.GRDCCollector(),
        "openmeteo": lambda: c.OpenMeteoCollector(mode=ctor.get("mode", "weather")),
        "sdg6": lambda: c.SDG6Collector(),
        "gemstat": lambda: c.GEMStatCollector(),
        "aquastat": lambda: c.AquastatCollector(),
        "wapor": lambda: c.WaPORCollector(),
        "copernicus": lambda: c.CopernicusCollector(),
        "wqp": lambda: c.WQPCollector(),
        "hubeau_hydrometrie": lambda: c.HubeauHydrometrieCollector(),
        "eu_wfd": lambda: c.EUWFDCollector(),
        "taiwan_moenv": lambda: c.TaiwanMOENVCollector(api_key=api_key or ""),
        "taiwan_wra_level": lambda: c.TaiwanWRAWaterLevelCollector(),
        "taiwan_wra_reservoir": lambda: c.TaiwanWRAReservoirCollector(),
        "taiwan_wra_fhy": lambda: c.TaiwanWRAFhyCollector(data_type=ctor.get("data_type", "water")),
        "taiwan_wra_iot": lambda: c.TaiwanWRAIoTCollector(data_type=ctor.get("data_type", "groundwater")),
        "taiwan_datagov": lambda: c.TaiwanDataGovCollector(dataset_id=ctor.get("dataset_id", "25768")),
        "taiwan_civil_iot": lambda: c.TaiwanCivilIoTCollector(),
        "japan_mlit": lambda: c.JapanMLITCollector(),
        "korea_wamis": lambda: c.KoreaWAMISCollector(),
        "india_wris": lambda: c.IndiaWRISCollector(),
    }
    collector = factories[source_key]()
    kwargs = dict(fetch)
    if api_key and source_key == "copernicus":
        kwargs["api_key"] = api_key
    return collector.collect(**kwargs)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _render_upload_tab() -> None:
    st.markdown("Upload a CSV or JSON file — column types, dates, and coordinates are auto-detected.")
    uploaded = st.file_uploader("CSV or JSON", type=["csv", "json"], label_visibility="collapsed")
    if uploaded is None:
        return
    try:
        content = uploaded.getvalue().decode("utf-8")
        if uploaded.name.lower().endswith(".json"):
            df = pd.read_json(StringIO(content))
        else:
            df = pd.read_csv(StringIO(content))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse file: {exc}")
        return

    if df.empty:
        st.warning("The file parsed but contains no rows.")
        return

    if st.button(f"Use **{uploaded.name}** as workspace dataset ({len(df):,} rows)", type="primary"):
        _state.set_data(df, "upload", f"Upload: {uploaded.name}")
        st.rerun()

    _insights.render_panel(df, key_prefix="upload")
    with st.expander("Preview (first 100 rows)", expanded=True):
        st.dataframe(df.head(100), width="stretch")


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def _render_demo_tab() -> None:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**💧 Water quality (180 days)**")
        st.caption(
            "pH, dissolved oxygen, turbidity and nitrate at one station, with a "
            "discharge column and coordinates. Includes intentional WHO exceedances. "
            "Exercises every dashboard page."
        )
        if st.button("Load water-quality demo", width="stretch", key="demo_tab_wq"):
            _state.load_demo("water_quality")
            st.rerun()
    with c2, st.container(border=True):
        st.markdown("**🌊 Daily streamflow (40 years)**")
        st.caption(
            "Four decades of synthetic daily discharge with monsoon seasonality and "
            "flood pulses — ideal for flood frequency, baseflow separation, and "
            "flow signatures."
        )
        if st.button("Load streamflow demo", width="stretch", key="demo_tab_sf"):
            _state.load_demo("streamflow")
            st.rerun()

    df = _state.get_data()
    if df is not None and str(st.session_state.get(_state.SOURCE_KEY, "")).startswith("demo"):
        st.success(f"Demo dataset active — {len(df):,} records in the workspace.")
        _insights.render_panel(df, key_prefix="demo")
