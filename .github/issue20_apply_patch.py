from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one exact anchor, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "aquascope/schemas/water_data.py",
    '    IRELAND_OPW = "ireland_opw"\n\n    PEGELONLINE = "pegelonline"',
    '    IRELAND_OPW = "ireland_opw"\n'
    '    SOUTH_AFRICA_DWS = "south_africa_dws"\n\n'
    '    PEGELONLINE = "pegelonline"',
)

replace_once(
    "aquascope/collectors/__init__.py",
    "from aquascope.collectors.sdg6 import SDG6Collector\n",
    "from aquascope.collectors.sdg6 import SDG6Collector\n"
    "from aquascope.collectors.south_africa_dws import SouthAfricaDWSCollector\n",
)
replace_once(
    "aquascope/collectors/__init__.py",
    '    "SDG6Collector",\n',
    '    "SDG6Collector",\n    "SouthAfricaDWSCollector",\n',
)

registry_entry = '''    # ── Africa ──────────────────────────────────────────────────────────
    "south_africa_dws": _s(
        key="south_africa_dws", label="South Africa DWS Verified Hydrology", region="South Africa",
        description="Verified daily river discharge and point water level from South African gauges",
        agency="South African Department of Water and Sanitation", country="ZAF",
        homepage="https://www.dws.gov.za/Hydrology/",
        variables=("discharge", "water_level"),
        output_model="StreamflowReading | WaterLevelReading",
        license="unknown", redistributable=False,
        attribution="South African Department of Water and Sanitation (reuse terms not yet verified)",
    ),
'''
replace_once(
    "aquascope/registry.py",
    "    # ── Europe ──────────────────────────────────────────────────────────\n",
    registry_entry
    + "    # ── Europe ──────────────────────────────────────────────────────────\n",
)
replace_once(
    "aquascope/registry.py",
    '        "sdg6": lambda: c.SDG6Collector(),\n',
    '        "sdg6": lambda: c.SDG6Collector(),\n'
    '        "south_africa_dws": lambda: c.SouthAfricaDWSCollector(),\n',
)

cli_block = '''    if source == "south_africa_dws":
        if not args.station:
            logger.error("South Africa DWS requires --station with a DWS gauge code, e.g. C1H001.")
            sys.exit(1)
        kwargs["station_id"] = args.station
        kwargs["variable"] = args.variable or "discharge"
        if args.days is not None:
            kwargs["days"] = args.days
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
'''
replace_once(
    "aquascope/cli.py",
    '    if source == "ireland_opw" and args.max_stations:\n',
    cli_block + '    if source == "ireland_opw" and args.max_stations:\n',
)
replace_once(
    "aquascope/cli.py",
    '"Number of days (USGS/UKEA/PEGELONLINE/BOM; PEGELONLINE max: 31)"',
    '"Number of days (USGS/UKEA/PEGELONLINE/BOM/South Africa DWS; '
    'PEGELONLINE max: 31)"',
)
replace_once(
    "aquascope/cli.py",
    'p_collect.add_argument("--variable", default=None, '
    'help="Variable code for the selected collector (WaPOR)")',
    'p_collect.add_argument(\n'
    '        "--variable",\n'
    '        default=None,\n'
    '        help="Variable code (WaPOR), or discharge/water_level '
    '(South Africa DWS)",\n'
    '    )',
)
replace_once(
    "aquascope/cli.py",
    'p_collect.add_argument("--start-date", default=None, '
    'help="Start date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM)")',
    'p_collect.add_argument(\n'
    '        "--start-date",\n'
    '        default=None,\n'
    '        help="Start date YYYY-MM-DD '
    '(openmeteo/copernicus/UKEA/BOM/South Africa DWS)",\n'
    '    )',
)
replace_once(
    "aquascope/cli.py",
    'p_collect.add_argument("--end-date", default=None, '
    'help="End date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM)")',
    'p_collect.add_argument(\n'
    '        "--end-date",\n'
    '        default=None,\n'
    '        help="End date YYYY-MM-DD '
    '(openmeteo/copernicus/UKEA/BOM/South Africa DWS)",\n'
    '    )',
)
replace_once(
    "aquascope/cli.py",
    '        "--station", default=None, '
    'help="Station UUID/SUID (PEGELONLINE/UKEA), or AWRC station number (BOM)"\n',
    '        "--station",\n'
    '        default=None,\n'
    '        help="Station UUID/SUID (PEGELONLINE/UKEA), AWRC number (BOM), '
    'or DWS gauge code",\n',
)

replace_once(
    "aquascope/dashboard/views/collect.py",
    '    "bom": {"station_id": "AWRC station number"},\n',
    '    "bom": {"station_id": "AWRC station number"},\n'
    '    "south_africa_dws": {"station_id": "DWS station code"},\n',
)
replace_once(
    "aquascope/dashboard/views/collect.py",
    '    "India",\n    "Africa & Near East",\n',
    '    "India",\n    "South Africa",\n    "Africa & Near East",\n',
)
dashboard_block = '''    elif source_key == "south_africa_dws":
        st.caption(
            "DWS Verified Hydrology — daily mean discharge or point water level. "
            "Leave both dates blank for the most recent 30 days."
        )
        station = st.text_input("DWS station code", placeholder="e.g. C1H001")
        if station.strip():
            fetch["station_id"] = station.strip()
        fetch["variable"] = st.selectbox(
            "Variable",
            ["discharge", "water_level"],
            format_func=lambda value: {
                "discharge": "Daily mean discharge (m³/s)",
                "water_level": "Point water level (m)",
            }[value],
        )
        c1, c2 = st.columns(2)
        sd = c1.date_input("Start date (optional)", value=None, key="dws_start")
        ed = c2.date_input("End date (optional)", value=None, key="dws_end")
        if sd:
            fetch["start_date"] = str(sd)
        if ed:
            fetch["end_date"] = str(ed)

'''
replace_once(
    "aquascope/dashboard/views/collect.py",
    '    elif source_key == "noaa_nwps":\n',
    dashboard_block + '    elif source_key == "noaa_nwps":\n',
)

replace_once(
    "tests/test_cli/test_dashboard_required_fields.py",
    'def test_bom_without_a_station_is_caught():\n'
    '    """BOMCollector.fetch_raw raises the same way; same form shape, '
    'same fix."""\n'
    '    assert missing_required_fields("bom", '
    '{"parameter_type": "Water Course Discharge"}) == [\n'
    '        "AWRC station number"\n'
    '    ]\n\n\n',
    'def test_bom_without_a_station_is_caught():\n'
    '    """BOMCollector.fetch_raw raises the same way; same form shape, '
    'same fix."""\n'
    '    assert missing_required_fields("bom", '
    '{"parameter_type": "Water Course Discharge"}) == [\n'
    '        "AWRC station number"\n'
    '    ]\n\n\n'
    'def test_dws_without_a_station_is_caught():\n'
    '    assert missing_required_fields("south_africa_dws", '
    '{"variable": "discharge"}) == [\n'
    '        "DWS station code"\n'
    '    ]\n\n\n',
)

replace_once(
    "docs/data_sources.md",
    "AquaScope ships **29 collectors**",
    "AquaScope ships **30 collectors**",
)
replace_once(
    "docs/data_sources.md",
    "| [India WRIS](https://indiawris.gov.in) | India | River water level | REST | ✅ |\n",
    "| [India WRIS](https://indiawris.gov.in) | India | River water level | REST | ✅ |\n"
    "| [South Africa DWS](https://www.dws.gov.za/Hydrology/) | South Africa | "
    "Verified river discharge, water level | HTML / text | ✅ |\n",
)
replace_once(
    "docs/data_sources.md",
    "| Japan MLIT / Korea WAMIS | No | Open access |\n",
    "| Japan MLIT / Korea WAMIS | No | Open access |\n"
    "| South Africa DWS | No | Open access; provider backend availability varies |\n",
)
replace_once(
    "docs/data_sources.md",
    "## PEGELONLINE (Germany)\n",
    Path(".github/issue20_dws_section.md").read_text(encoding="utf-8")
    + "## PEGELONLINE (Germany)\n",
)

readme_replacements = [
    (
        "AquaScope unifies **29 global water-data sources**",
        "AquaScope unifies **30 global water-data sources**",
    ),
    ("| 29 unified data collectors |", "| 30 unified data collectors |"),
    (
        "### 3. Collect data from any of the 29 sources",
        "### 3. Collect data from any of the 30 sources",
    ),
    (
        "29 data collectors spanning four regions",
        "30 data collectors spanning five regions",
    ),
    (
        "| [Data sources](docs/data_sources.md) | All 29 sources, endpoints, "
        "API-key requirements |",
        "| [Data sources](docs/data_sources.md) | All 30 sources, endpoints, "
        "API-key requirements |",
    ),
    ("India WRIS, GRDC", "India WRIS, South Africa DWS, GRDC"),
]
for old, new in readme_replacements:
    replace_once("README.md", old, new)
replace_once(
    "README.md",
    "- 🌏 **Asia-Pacific** — Taiwan MOENV / WRA / Civil IoT / DataGov, "
    "Japan MLIT, Korea WAMIS, India WRIS\n"
    "- 🌐 **Global** — GEMStat (170 countries), UN SDG 6, OpenMeteo, "
    "FAO AQUASTAT, FAO WaPOR, GRDC (river discharge)\n",
    "- 🌏 **Asia-Pacific** — Taiwan MOENV / WRA / Civil IoT / DataGov, "
    "Japan MLIT, Korea WAMIS, India WRIS\n"
    "- 🌍 **Africa** — South Africa DWS Verified Hydrology "
    "(river discharge + water level)\n"
    "- 🌐 **Global** — GEMStat (170 countries), UN SDG 6, OpenMeteo, "
    "FAO AQUASTAT, FAO WaPOR, GRDC (river discharge)\n",
)

replace_once(
    "docs/index.md",
    "AquaScope unifies **29 global water-data sources**",
    "AquaScope unifies **30 global water-data sources**",
)
replace_once(
    "docs/index.md",
    "| 29 unified data collectors                   |",
    "| 30 unified data collectors                   |",
)
replace_once(
    "docs/index.md",
    "Taiwan MOENV / WRA, Japan MLIT, Korea WAMIS, OpenMeteo, and UN SDG 6",
    "Taiwan MOENV / WRA, Japan MLIT, Korea WAMIS, South Africa DWS, OpenMeteo, and UN SDG 6",
)

replace_once(
    "ROADMAP.md",
    "- [x] 29 data source collectors (Taiwan ×8, USA ×3, Global ×5, FAO ×2, EU, France, Germany, Ireland, UK, Japan, Korea, India, Chile, Brazil, Australia)",
    "- [x] 30 data source collectors (Taiwan ×8, USA ×3, Global ×5, FAO ×2, EU, France, Germany, Ireland, UK, Japan, Korea, India, South Africa, Chile, Brazil, Australia)",
)

replace_once(
    "CHANGELOG.md",
    "## [Unreleased]\n\n### Added\n",
    "## [Unreleased]\n\n### Added\n"
    "- **South Africa DWS Verified Hydrology collector** (#20). Adds verified daily mean river discharge and point water-level collection through the deterministic `HyData.aspx` interface, normalised to `StreamflowReading` and `WaterLevelReading`. The collector validates the response body because DWS can return a Kisters `ScriptServerODBC` backend failure with HTTP 200, and fails closed rather than representing that page as hydrological data.\n",
)

print("Deterministic patch applied.")
