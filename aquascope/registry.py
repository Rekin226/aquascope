"""Single source of truth for AquaScope's data-source registry.

Metadata (key, label, region, description, API-key requirement) is a plain
dict with no collector imports, so it's cheap to use for CLI --source
choices and dashboard labels at import time. The actual collector-class
mapping lives inside build_collector() and imports aquascope.collectors
lazily, matching the existing lazy-import discipline in cli.py and
dashboard/views/collect.py — a source's classes are only imported when
someone actually collects from it.

Previously duplicated as collector_map in cli.py and _FACTORIES in
dashboard/views/collect.py (26 entries each, issue #58 review). One entry
added here is visible to every consumer (CLI, dashboard, and the planned
REST API).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceMeta:
    key: str
    label: str
    region: str
    description: str
    requires_api_key: bool = False
    api_key_signup_url: str | None = None


SOURCES: dict[str, SourceMeta] = {
    "usgs": SourceMeta("usgs", "USGS Water Services", "United States",
        "Real-time discharge, gauge height, temperature from thousands of US gauges"),
    "grdc": SourceMeta("grdc", "GRDC river discharge", "Global",
        "In-situ gauges (Zenodo subset) + RSEG satellite discharge estimates"),
    "openmeteo": SourceMeta("openmeteo", "Open-Meteo", "Global",
        "Weather history, forecasts and GloFAS flood discharge for any coordinate"),
    "sdg6": SourceMeta("sdg6", "UN SDG 6 indicators", "Global",
        "Country-level water & sanitation indicators (water stress, IWRM, …)"),
    "gemstat": SourceMeta("gemstat", "GEMStat water quality", "Global",
        "UNEP global surface & groundwater quality archive (~200 MB, cached locally)"),
    "aquastat": SourceMeta("aquastat", "FAO AQUASTAT", "Global",
        "National water resources and agricultural water-use statistics"),
    "wapor": SourceMeta("wapor", "FAO WaPOR", "Africa & Near East",
        "Remote-sensing evapotranspiration and biomass productivity rasters"),
    "copernicus": SourceMeta("copernicus", "Copernicus CDS", "Global",
        "ERA5 / GloFAS climate reanalysis (requires free CDS key)",
        requires_api_key=True, api_key_signup_url="https://cds.climate.copernicus.eu/how-to-api"),
    "wqp": SourceMeta("wqp", "Water Quality Portal", "United States",
        "EPA/USGS harmonised water-quality samples by state"),
    "hubeau_hydrometrie": SourceMeta("hubeau_hydrometrie", "Hub'Eau hydrométrie", "France",
        "Real-time water level & discharge from French national gauges"),
    "eu_wfd": SourceMeta("eu_wfd", "EU Water Framework Directive", "Europe",
        "EEA DiscoData ecological/chemical status of European water bodies"),
    "taiwan_moenv": SourceMeta("taiwan_moenv", "Taiwan MOENV", "Taiwan",
        "River water-quality monitoring (requires free MOENV key)",
        requires_api_key=True, api_key_signup_url="https://data.moenv.gov.tw/en/apikey"),
    "taiwan_wra_level": SourceMeta("taiwan_wra_level", "Taiwan WRA water level", "Taiwan",
        "Real-time river stage snapshot across all WRA stations"),
    "taiwan_wra_reservoir": SourceMeta("taiwan_wra_reservoir", "Taiwan WRA reservoirs", "Taiwan",
        "Daily reservoir storage and operations"),
    "taiwan_wra_fhy": SourceMeta("taiwan_wra_fhy", "Taiwan WRA FHY real-time", "Taiwan",
        "Real-time water level / rainfall / discharge (FHY portal)"),
    "taiwan_wra_iot": SourceMeta("taiwan_wra_iot", "Taiwan WRA IoT", "Taiwan",
        "Real-time groundwater level and rainfall accumulation"),
    "taiwan_datagov": SourceMeta("taiwan_datagov", "Taiwan data.gov.tw", "Taiwan",
        "Open-government real-time river & groundwater levels"),
    "taiwan_civil_iot": SourceMeta("taiwan_civil_iot", "Taiwan Civil IoT", "Taiwan",
        "SensorThings water observations (flood sensors etc.)"),
    "japan_mlit": SourceMeta("japan_mlit", "Japan MLIT", "Japan",
        "Water level, discharge, quality and rainfall by prefecture"),
    "korea_wamis": SourceMeta("korea_wamis", "Korea WAMIS", "South Korea",
        "Water level, discharge, quality and dam storage by basin"),
    "india_wris": SourceMeta("india_wris", "India WRIS", "India",
        "River water level by state / district / agency"),
    "noaa_nwps": SourceMeta("noaa_nwps", "NOAA NWPS", "United States",
        "River stage and discharge forecasts across US stream gauges"),
    "ireland_opw": SourceMeta("ireland_opw", "Ireland OPW", "Ireland",
        "Real-time river & lake water levels from waterlevel.ie"),
    "pegelonline": SourceMeta("pegelonline", "PEGELONLINE", "Germany",
        "Real-time river stage and discharge from German federal waterways (WSV)"),
    "camels_cl": SourceMeta("camels_cl", "CAMELS-CL", "Chile",
        "Daily observed streamflow & catchment attributes for 516 Chilean catchments"),
    "camels_br": SourceMeta("camels_br", "CAMELS-BR", "Brazil",
        "Daily observed streamflow & catchment attributes for Brazilian catchments"),
    "uk_ea": SourceMeta("uk_ea", "UK Environment Agency", "United Kingdom",
        "Real-time river level, flow, rainfall, and groundwater observations from UK EA telemetry"),
}


def source_keys() -> list[str]:
    """Sorted list of every valid source key — the single choices/validation list."""
    return sorted(SOURCES.keys())


def build_collector(source_key: str, api_key: str | None = None, **ctor_kwargs):
    """Instantiate the collector for source_key.

    Imports aquascope.collectors lazily (only when actually collecting),
    matching the existing lazy-import pattern elsewhere in this codebase.
    ``ctor_kwargs`` carries source-specific constructor parameters
    (e.g. mode= for openmeteo, data_type= for the Taiwan WRA FHY/IoT
    sources) — callers only pass what that source accepts.
    """
    from aquascope import collectors as c

    factories = {
        "usgs": lambda: c.USGSCollector(api_key=api_key or "DEMO_KEY"),
        "grdc": lambda: c.GRDCCollector(),
        "openmeteo": lambda: c.OpenMeteoCollector(mode=ctor_kwargs.get("mode", "weather")),
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
        "taiwan_wra_fhy": lambda: c.TaiwanWRAFhyCollector(data_type=ctor_kwargs.get("data_type", "water")),
        "taiwan_wra_iot": lambda: c.TaiwanWRAIoTCollector(data_type=ctor_kwargs.get("data_type", "groundwater")),
        "taiwan_datagov": lambda: c.TaiwanDataGovCollector(dataset_id=ctor_kwargs.get("dataset_id", "25768")),
        "taiwan_civil_iot": lambda: c.TaiwanCivilIoTCollector(),
        "japan_mlit": lambda: c.JapanMLITCollector(),
        "korea_wamis": lambda: c.KoreaWAMISCollector(),
        "india_wris": lambda: c.IndiaWRISCollector(),
        "noaa_nwps": lambda: c.NOAANWPSCollector(),
        "ireland_opw": lambda: c.IrelandOPWCollector(),
        "pegelonline": lambda: c.PegelonlineCollector(),
        "camels_cl": lambda: c.CAMELSCLCollector(),
        "camels_br": lambda: c.CAMELSBRCollector(),
        "uk_ea": lambda: c.UKEACollector(),
    }
    if source_key not in factories:
        raise ValueError(f"Unknown source {source_key!r}. Available: {source_keys()}")
    return factories[source_key]()
