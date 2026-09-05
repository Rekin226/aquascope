"""Single source of truth for AquaScope's data-source registry.

``SOURCES`` describes every collector once: what it is (label, agency,
country), what it measures (``variables``), how it can be queried
(``supports_bbox``, ``supports_station_lookup``), what it emits
(``output_model``), and on what terms (``license``, ``redistributable``,
``attribution``). It is a plain dict with no collector imports, so it is cheap
to use for CLI ``--source`` choices, dashboard labels, MCP tool descriptions
and the harvest job at import time. The actual collector-class mapping lives
inside :func:`build_collector` and imports ``aquascope.collectors`` lazily.

Two rules keep this honest:

* ``redistributable`` defaults to ``False``. It is only ``True`` when someone
  has read the source's terms and recorded them in ``license``. The archive
  (#188) mirrors observations only from redistributable sources; everything
  else is catalog-only with a link back to the agency.
* ``supports_station_lookup`` must match whether the collector class
  overrides :meth:`aquascope.collectors.base.BaseCollector.stations`. A test
  guards that, together with "every collector class is registered".

Previously duplicated as ``collector_map`` in ``cli.py`` and ``_FACTORIES``
in ``dashboard/views/collect.py`` (issue #58 review, extracted in #163) and
extended with the catalog and license fields in #187.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from aquascope.schemas.station import VARIABLES, Station

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCES",
    "SourceMeta",
    "StationCatalog",
    "build_collector",
    "find_stations",
    "redistributable_sources",
    "source_keys",
    "station_catalogs",
    "station_sources",
]


@dataclass(frozen=True)
class SourceMeta:
    """Everything a consumer needs to know about a source without importing it."""

    key: str
    label: str
    region: str
    description: str
    requires_api_key: bool = False
    api_key_signup_url: str | None = None
    # v2 (#187): provenance, coverage and terms
    agency: str = ""
    country: str = ""  # ISO 3166-1 alpha-3, or "GLOBAL" / "EU"
    homepage: str | None = None
    variables: tuple[str, ...] = ()
    supports_bbox: bool = False
    supports_station_lookup: bool = False
    output_model: str = ""
    license: str = "unknown"
    redistributable: bool = False
    attribution: str = ""

    def __post_init__(self) -> None:
        unknown = [v for v in self.variables if v not in VARIABLES]
        if unknown:
            raise ValueError(f"{self.key}: unknown variable(s) {unknown}; allowed: {list(VARIABLES)}")


def _s(**kwargs) -> SourceMeta:
    return SourceMeta(**kwargs)


SOURCES: dict[str, SourceMeta] = {
    # ── Americas ────────────────────────────────────────────────────────
    "usgs": _s(
        key="usgs", label="USGS Water Services", region="United States",
        description="Daily and instantaneous discharge, gauge height, temperature and water quality from US gauges",
        agency="U.S. Geological Survey", country="USA",
        homepage="https://waterdata.usgs.gov/",
        variables=("discharge", "water_level", "water_quality"),
        supports_bbox=True, supports_station_lookup=True,
        output_model="StreamflowReading | WaterLevelReading | WaterQualitySample",
        license="US-PD", redistributable=True,
        attribution="U.S. Geological Survey, National Water Information System (public domain)",
    ),
    "noaa_nwps": _s(
        key="noaa_nwps", label="NOAA NWPS", region="United States",
        description="River stage and discharge observations and forecasts across US stream gauges",
        agency="NOAA National Water Prediction Service", country="USA",
        homepage="https://water.noaa.gov/",
        variables=("discharge", "water_level"),
        supports_bbox=True,
        output_model="StreamflowReading",
        license="US-PD", redistributable=True,
        attribution="NOAA National Water Prediction Service (public domain)",
    ),
    "wqp": _s(
        key="wqp", label="Water Quality Portal", region="United States",
        description="EPA/USGS harmonised water-quality samples by state",
        agency="EPA / USGS Water Quality Portal", country="USA",
        homepage="https://www.waterqualitydata.us/",
        variables=("water_quality",),
        output_model="WaterQualitySample",
        license="US-PD", redistributable=True,
        attribution="Water Quality Portal (EPA, USGS, NWQMC); check per-organization terms for non-federal data",
    ),
    "camels_cl": _s(
        key="camels_cl", label="CAMELS-CL", region="Chile",
        description="Daily observed streamflow and catchment attributes for 516 Chilean catchments",
        agency="CR2, Universidad de Chile", country="CHL",
        homepage="https://camels.cr2.cl/",
        variables=("discharge",),
        output_model="StreamflowReading",
        license="CC-BY-4.0", redistributable=True,
        attribution="Alvarez-Garreton et al. (2018), CAMELS-CL, CR2",
    ),
    "camels_br": _s(
        key="camels_br", label="CAMELS-BR", region="Brazil",
        description="Daily observed streamflow and catchment attributes for Brazilian catchments",
        agency="CAMELS-BR (Chagas et al.), Zenodo", country="BRA",
        homepage="https://zenodo.org/records/3964745",
        variables=("discharge",),
        output_model="StreamflowReading",
        license="CC-BY-4.0", redistributable=True,
        attribution="Chagas et al. (2020), CAMELS-BR",
    ),
    # ── Europe ──────────────────────────────────────────────────────────
    "uk_ea": _s(
        key="uk_ea", label="Environment Agency (England)", region="United Kingdom",
        description="River level, flow, rainfall and groundwater telemetry from the EA Hydrology API (England)",
        agency="Environment Agency", country="GBR",
        homepage="https://environment.data.gov.uk/hydrology/",
        variables=("discharge", "water_level", "precipitation", "groundwater_level"),
        supports_bbox=True, supports_station_lookup=True,
        output_model="StreamflowReading | WaterLevelReading | WaterQualitySample",
        license="OGL-UK-3.0", redistributable=True,
        attribution="Environment Agency, Open Government Licence v3.0",
    ),
    "hubeau_hydrometrie": _s(
        key="hubeau_hydrometrie", label="Hub'Eau hydrométrie", region="France",
        description="Water level and discharge from French national gauges (Hub'Eau v2)",
        agency="Eaufrance / SCHAPI (Hub'Eau)", country="FRA",
        homepage="https://hubeau.eaufrance.fr/page/api-hydrometrie",
        variables=("discharge", "water_level"),
        supports_bbox=True, supports_station_lookup=True,
        output_model="StreamflowReading | WaterLevelReading",
        license="etalab-2.0", redistributable=True,
        attribution="Hub'Eau, Eaufrance, Licence Ouverte / Open Licence 2.0",
    ),
    "pegelonline": _s(
        key="pegelonline", label="PEGELONLINE", region="Germany",
        description="Water level and discharge from German federal waterways (WSV), last 31 days",
        agency="Wasserstraßen- und Schifffahrtsverwaltung des Bundes", country="DEU",
        homepage="https://www.pegelonline.wsv.de/",
        variables=("water_level", "discharge"),
        supports_station_lookup=True,
        output_model="WaterLevelReading | StreamflowReading",
        license="DL-DE-BY-2.0", redistributable=True,
        attribution="Wasserstraßen- und Schifffahrtsverwaltung des Bundes (WSV), PEGELONLINE, dl-de/by-2-0",
    ),
    "ireland_opw": _s(
        key="ireland_opw", label="Ireland OPW", region="Ireland",
        description="River and lake water levels from waterlevel.ie (15-minute)",
        agency="Office of Public Works", country="IRL",
        homepage="https://waterlevel.ie/",
        variables=("water_level",),
        supports_station_lookup=True,
        output_model="WaterLevelReading",
        license="CC-BY-4.0", redistributable=True,
        attribution="Office of Public Works, waterlevel.ie (CC BY 4.0)",
    ),
    "eu_wfd": _s(
        key="eu_wfd", label="EU Water Framework Directive", region="Europe",
        description="EEA DiscoData ecological and chemical status of European water bodies",
        agency="European Environment Agency", country="EU",
        homepage="https://discodata.eea.europa.eu/",
        variables=("water_quality",),
        output_model="WaterQualitySample",
        license="EEA-standard-reuse", redistributable=True,
        attribution="European Environment Agency (EEA), standard re-use policy",
    ),
    # ── Asia ────────────────────────────────────────────────────────────
    "taiwan_wra_level": _s(
        key="taiwan_wra_level", label="Taiwan WRA water level", region="Taiwan",
        description="Real-time river stage snapshot across all WRA stations",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://opendata.wra.gov.tw/",
        variables=("water_level",),
        output_model="WaterLevelReading",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Water Resources Agency, Ministry of Economic Affairs (Open Government Data License 1.0)",
    ),
    "taiwan_wra_reservoir": _s(
        key="taiwan_wra_reservoir", label="Taiwan WRA reservoirs", region="Taiwan",
        description="Daily reservoir storage and operations",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://opendata.wra.gov.tw/",
        variables=("reservoir_storage",),
        output_model="ReservoirStatus",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Water Resources Agency, Ministry of Economic Affairs (Open Government Data License 1.0)",
    ),
    "taiwan_wra_groundwater": _s(
        key="taiwan_wra_groundwater", label="Taiwan WRA groundwater (annual)", region="Taiwan",
        description="Annual groundwater levels for the WRA well network, with well metadata",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://opendata.wra.gov.tw/",
        variables=("groundwater_level",),
        output_model="GroundwaterLevel",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Water Resources Agency, Ministry of Economic Affairs (Open Government Data License 1.0)",
    ),
    "taiwan_wra_groundwater_daily": _s(
        key="taiwan_wra_groundwater_daily", label="Taiwan WRA groundwater (daily)", region="Taiwan",
        description="Daily groundwater levels per zone from the WRA HydroInfo (gweb) portal",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://gweb.wra.gov.tw/",
        variables=("groundwater_level",),
        output_model="GroundwaterLevel",
        license="unknown", redistributable=False,
        attribution="Water Resources Agency, Ministry of Economic Affairs",
    ),
    "taiwan_wra_fhy": _s(
        key="taiwan_wra_fhy", label="Taiwan WRA FHY real-time", region="Taiwan",
        description="Real-time water level, rainfall and discharge (FHY portal)",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://fhy.wra.gov.tw/",
        variables=("water_level", "precipitation", "discharge"),
        output_model="WaterQualitySample",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Water Resources Agency, Ministry of Economic Affairs (Open Government Data License 1.0)",
    ),
    "taiwan_wra_iot": _s(
        key="taiwan_wra_iot", label="Taiwan WRA IoT", region="Taiwan",
        description="Real-time groundwater level (v2 API; rainfall requires paid membership, not supported)",
        agency="Water Resources Agency, MOEA", country="TWN",
        homepage="https://iot.wra.gov.tw/",
        variables=("groundwater_level",),
        output_model="WaterQualitySample",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Water Resources Agency, Ministry of Economic Affairs (Open Government Data License 1.0)",
    ),
    "taiwan_datagov": _s(
        key="taiwan_datagov", label="Taiwan data.gov.tw", region="Taiwan",
        description="Open-government real-time river and groundwater levels",
        agency="data.gov.tw (WRA datasets)", country="TWN",
        homepage="https://data.gov.tw/",
        variables=("water_level", "groundwater_level"),
        output_model="WaterLevelReading",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="data.gov.tw, Open Government Data License 1.0",
    ),
    "taiwan_civil_iot": _s(
        key="taiwan_civil_iot", label="Taiwan Civil IoT", region="Taiwan",
        description="SensorThings water observations (flood sensors, level, flow, rain)",
        agency="Civil IoT Taiwan", country="TWN",
        homepage="https://ci.taiwan.gov.tw/",
        variables=("water_level", "discharge", "precipitation"),
        output_model="WaterQualitySample",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Civil IoT Taiwan, Open Government Data License 1.0",
    ),
    "taiwan_cwa": _s(
        key="taiwan_cwa", label="Taiwan CWA climate stations", region="Taiwan",
        description="Daily station climate observations (rainfall, temperature, humidity, wind, evaporation) via CODIS",
        agency="Central Weather Administration", country="TWN",
        homepage="https://codis.cwa.gov.tw/",
        variables=("climate", "precipitation"),
        supports_station_lookup=True,
        output_model="ClimateReading",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Central Weather Administration, Open Government Data License 1.0",
    ),
    "taiwan_moenv": _s(
        key="taiwan_moenv", label="Taiwan MOENV", region="Taiwan",
        description="River water-quality monitoring (requires free MOENV key)",
        requires_api_key=True, api_key_signup_url="https://data.moenv.gov.tw/en/apikey",
        agency="Ministry of Environment", country="TWN",
        homepage="https://data.moenv.gov.tw/",
        variables=("water_quality",),
        output_model="WaterQualitySample",
        license="OGDL-Taiwan-1.0", redistributable=True,
        attribution="Ministry of Environment, Open Government Data License 1.0",
    ),
    "japan_mlit": _s(
        key="japan_mlit", label="Japan MLIT", region="Japan",
        description="Water level, discharge, quality and rainfall by prefecture",
        agency="Ministry of Land, Infrastructure, Transport and Tourism", country="JPN",
        homepage="http://www1.river.go.jp/",
        variables=("water_level", "discharge", "water_quality", "precipitation"),
        output_model="WaterQualitySample",
        license="Public Data License (Version 1.0)", redistributable=True,
        attribution="Ministry of Land, Infrastructure, Transport and Tourism",
    ),
    "korea_wamis": _s(
        key="korea_wamis", label="Korea WAMIS", region="South Korea",
        description="Water level, discharge, quality and dam storage by basin",
        agency="WAMIS (K-water / Ministry of Environment)", country="KOR",
        homepage="http://www.wamis.go.kr/",
        variables=("water_level", "discharge", "water_quality", "reservoir_storage"),
        output_model="WaterQualitySample",
        license="unknown", redistributable=False,
        attribution="WAMIS (terms not yet verified)",
    ),
    "india_wris": _s(
        key="india_wris", label="India WRIS", region="India",
        description="River water level by state / district / agency",
        agency="India-WRIS, Ministry of Jal Shakti", country="IND",
        homepage="https://indiawris.gov.in/",
        variables=("water_level",),
        output_model="WaterLevelReading",
        license="unknown", redistributable=False,
        attribution="India-WRIS (terms not yet verified)",
    ),
    # ── Oceania ─────────────────────────────────────────────────────────
    "bom": _s(
        key="bom", label="BOM Water Data Online", region="Australia",
        description="Streamflow, water level, storage and groundwater level from ~8,000 Australian gauging stations",
        agency="Bureau of Meteorology", country="AUS",
        homepage="http://www.bom.gov.au/waterdata/",
        variables=("discharge", "water_level", "reservoir_storage", "groundwater_level", "precipitation",
                   "water_quality"),
        output_model="StreamflowReading | WaterLevelReading | WaterQualitySample",
        license="CC-BY-3.0-AU", redistributable=True,
        attribution="Bureau of Meteorology, © Commonwealth of Australia",
        supports_station_lookup=True,
    ),
    # ── Global ──────────────────────────────────────────────────────────
    "grdc": _s(
        key="grdc", label="GRDC river discharge", region="Global",
        description="In-situ gauges (Zenodo subset) and RSEG satellite discharge estimates",
        agency="Global Runoff Data Centre (BfG)", country="GLOBAL",
        homepage="https://grdc.bafg.de/",
        variables=("discharge",),
        output_model="StreamflowReading",
        license="GRDC Policy Guidelines", redistributable=False,
        attribution="The Global Runoff Data Centre, 56068 Koblenz, Germany",
    ),
    "openmeteo": _s(
        key="openmeteo", label="Open-Meteo", region="Global",
        description="Weather history, forecasts and GloFAS river discharge for any coordinate",
        agency="Open-Meteo", country="GLOBAL",
        homepage="https://open-meteo.com/",
        variables=("precipitation", "climate", "discharge"),
        supports_bbox=False,
        output_model="WaterQualitySample",
        license="CC-BY-4.0", redistributable=True,
        attribution="Open-Meteo.com (CC BY 4.0); ERA5 and GloFAS via Copernicus",
    ),
    "copernicus": _s(
        key="copernicus", label="Copernicus CDS", region="Global",
        description="ERA5 / GloFAS climate reanalysis (requires free CDS key)",
        requires_api_key=True, api_key_signup_url="https://cds.climate.copernicus.eu/how-to-api",
        agency="ECMWF / Copernicus Climate Change Service", country="GLOBAL",
        homepage="https://cds.climate.copernicus.eu/",
        variables=("climate", "discharge"),
        output_model="WaterQualitySample",
        license="Copernicus-licence", redistributable=True,
        attribution="Copernicus Climate Change Service (C3S), ECMWF",
    ),
    "gemstat": _s(
        key="gemstat", label="GEMStat water quality", region="Global",
        description="UNEP global surface and groundwater quality archive (~200 MB, cached locally)",
        agency="UNEP GEMS/Water", country="GLOBAL",
        homepage="https://gemstat.org/",
        variables=("water_quality",),
        output_model="WaterQualitySample",
        license="unknown", redistributable=False,
        attribution="UNEP GEMS/Water Data Centre (terms not yet verified)",
    ),
    "sdg6": _s(
        key="sdg6", label="UN SDG 6 indicators", region="Global",
        description="Country-level water and sanitation indicators (water stress, IWRM, …)",
        agency="UN Statistics Division", country="GLOBAL",
        homepage="https://unstats.un.org/sdgs/",
        variables=("indicator",),
        output_model="SDG6Indicator",
        license="UNdata Terms of Use", redistributable=True,
        attribution="United Nations Statistics Division, SDG Global Database (UNdata)",
    ),
    "aquastat": _s(
        key="aquastat", label="FAO AQUASTAT", region="Global",
        description="National water resources and agricultural water-use statistics",
        agency="FAO", country="GLOBAL",
        homepage="https://www.fao.org/aquastat/",
        variables=("indicator",),
        output_model="AquastatRecord",
        license="CC-BY-4.0", redistributable=True,
        attribution="FAO AQUASTAT (© FAO, CC BY 4.0)",
    ),
    "wapor": _s(
        key="wapor", label="FAO WaPOR", region="Africa & Near East",
        description="Remote-sensing evapotranspiration and biomass productivity rasters",
        agency="FAO WaPOR", country="GLOBAL",
        homepage="https://data.apps.fao.org/wapor/",
        variables=("evapotranspiration",),
        supports_bbox=True,
        output_model="WaPORObservation",
        license="CC-BY-4.0", redistributable=True,
        attribution="FAO WaPOR (CC BY 4.0)",
    ),
}


def source_keys() -> list[str]:
    """Sorted list of every valid source key: the single choices/validation list."""
    return sorted(SOURCES.keys())


def station_sources(variable: str | None = None) -> list[str]:
    """Keys of sources that expose a station catalog (optionally measuring ``variable``)."""
    keys = [k for k, m in SOURCES.items() if m.supports_station_lookup]
    if variable:
        keys = [k for k in keys if variable in SOURCES[k].variables]
    return sorted(keys)


def redistributable_sources() -> list[str]:
    """Keys of sources whose terms allow mirroring their observations."""
    return sorted(k for k, m in SOURCES.items() if m.redistributable)


def build_collector(source_key: str, api_key: str | None = None, **ctor_kwargs):
    """Instantiate the collector for ``source_key``.

    Imports ``aquascope.collectors`` lazily (only when actually collecting).
    ``ctor_kwargs`` carries source-specific constructor parameters (``mode=``
    for openmeteo, ``data_type=`` for the Taiwan WRA FHY/IoT sources); callers
    only pass what that source accepts. ``api_key`` is handed to the collector
    untouched so each one resolves its own env-var and demo-key fallbacks.
    """
    from aquascope import collectors as c

    factories = {
        "usgs": lambda: c.USGSCollector(api_key=api_key),
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
        "taiwan_cwa": lambda: c.TaiwanCWACollector(),
        "taiwan_moenv": lambda: c.TaiwanMOENVCollector(api_key=api_key or ""),
        "taiwan_wra_level": lambda: c.TaiwanWRAWaterLevelCollector(),
        "taiwan_wra_reservoir": lambda: c.TaiwanWRAReservoirCollector(),
        "taiwan_wra_groundwater": lambda: c.TaiwanWRAGroundwaterCollector(),
        "taiwan_wra_groundwater_daily": lambda: c.TaiwanWRAGroundwaterDailyCollector(
            zones=ctor_kwargs.get("zones"), aggregate=ctor_kwargs.get("aggregate", "monthly")
        ),
        "taiwan_wra_fhy": lambda: c.TaiwanWRAFhyCollector(data_type=ctor_kwargs.get("data_type", "water")),
        "taiwan_wra_iot": lambda: c.TaiwanWRAIoTCollector(data_type=ctor_kwargs.get("data_type", "groundwater")),
        "taiwan_datagov": lambda: c.TaiwanDataGovCollector(
            dataset_id=ctor_kwargs.get("dataset_id", "73c4c3de-4045-4765-abeb-89f9f9cd5ff0")
        ),
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
        "bom": lambda: c.BOMCollector(),
    }
    if source_key not in factories:
        raise ValueError(f"Unknown source {source_key!r}. Available: {source_keys()}")
    return factories[source_key]()


@dataclass
class StationCatalog:
    """One source's station lookup result, kept even when it failed."""

    source: str
    stations: list[Station] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def station_catalogs(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    variable: str | None = None,
    sources: list[str] | None = None,
    max_items: int | None = None,
    max_workers: int = 4,
    api_key: str | None = None,
) -> dict[str, StationCatalog]:
    """Ask every station-capable source for its catalog, in parallel.

    Returns one :class:`StationCatalog` per source, including the ones that
    failed (``error`` set, ``stations`` empty), so a health report can tell
    "endpoint failed" from "no stations here".
    """
    keys = sources or station_sources(variable)
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown source(s) {unknown}. Available: {source_keys()}")
    keys = [k for k in keys if SOURCES[k].supports_station_lookup]
    if variable:
        keys = [k for k in keys if variable in SOURCES[k].variables]

    def one(key: str) -> StationCatalog:
        t0 = time.perf_counter()
        try:
            collector = build_collector(key, api_key=api_key)
            found = collector.stations(bbox=bbox, variable=variable, max_items=max_items)
            return StationCatalog(source=key, stations=list(found), seconds=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the others
            logger.warning("[%s] station lookup failed: %s", key, exc)
            return StationCatalog(source=key, error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)

    if not keys:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(keys)))) as pool:
        results = list(pool.map(one, keys))
    return {r.source: r for r in results}


def find_stations(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    variable: str | None = None,
    sources: list[str] | None = None,
    max_items: int | None = None,
    max_workers: int = 4,
    api_key: str | None = None,
) -> list[Station]:
    """Stations from every catalog-capable source, flattened.

    ``bbox`` is ``(west, south, east, north)`` in WGS84 degrees. Failed sources
    are logged and skipped; use :func:`station_catalogs` when you need the
    per-source outcome.
    """
    catalogs = station_catalogs(
        bbox=bbox, variable=variable, sources=sources, max_items=max_items, max_workers=max_workers, api_key=api_key
    )
    out: list[Station] = []
    for key in sorted(catalogs):
        out.extend(catalogs[key].stations)
    return out
