# Data Sources

AquaScope ships **20 collectors** that normalise water data into typed Pydantic records. One API call per source, one schema across the toolkit.

Most sources emit point observations and share the unified `water_data` schema (`WaterQualitySample`, `WaterLevelReading`, `ReservoirStatus`). Three aggregate/gridded sources use purpose-built record types that match their data shape: **FAO AQUASTAT** returns country-level `AquastatRecord`, **UN SDG 6** returns `SDG6Indicator`, and **FAO WaPOR** returns gridded `WaPORObservation`.

To request a new source, open an [issue](https://github.com/Rekin226/aquascope/issues/new/choose) using the *New Data Source Request* template, or join the [Discussion](https://github.com/Rekin226/aquascope/discussions) thread on data-source priorities.

---

## Sources

| Source | Region | Data Types | API | Status |
| :--- | :--- | :--- | :--- | :---: |
| [Taiwan MOENV](https://data.moenv.gov.tw) | Taiwan | River / tap water quality, RPI | REST | ✅ |
| [Taiwan WRA](https://opendata.wra.gov.tw) | Taiwan | Water levels, reservoir status | REST | ✅ |
| [Taiwan Civil IoT](https://sta.ci.taiwan.gov.tw) | Taiwan | Real-time sensors (level, flow, rain) | SensorThings | ✅ |
| [Taiwan WRA FHY](https://fhy.wra.gov.tw) | Taiwan | Real-time water level, rainfall, discharge | REST | ✅ |
| [Taiwan WRA IoT](https://iot.wra.gov.tw) | Taiwan | Groundwater level, rainfall accumulation | REST | ✅ |
| [Taiwan data.gov.tw](https://data.gov.tw) | Taiwan | Real-time river + groundwater level | REST | ✅ |
| [Taiwan WRA Groundwater](https://opendata.wra.gov.tw) | Taiwan | Annual groundwater levels + well metadata (992 wells, 1992–) | REST | ✅ |
| [USGS](https://api.waterdata.usgs.gov) | USA | Streamflow, water quality, gage height | OGC | ✅ |
| [Water Quality Portal](https://waterqualitydata.us) | USA | Integrated WQ from 400+ agencies | REST / CSV | ✅ |
| [GEMStat](https://gemstat.org) | Global | Freshwater quality (170+ countries) | Zenodo | ✅ |
| [UN SDG 6](https://sdg6data.org) | Global | SDG 6 indicators (6.1.1 – 6.6.1) | REST | ✅ |
| [OpenMeteo](https://open-meteo.com) | Global | Weather (temp, precip, wind, solar) | REST | ✅ |
| [Copernicus](https://cds.climate.copernicus.eu) | Global | ERA5 reanalysis, climate projections | CDS API | ✅ |
| [FAO AQUASTAT](https://www.fao.org/aquastat) | Global | Country-level water withdrawal, irrigation | FAOSTAT API | ✅ |
| [FAO WaPOR](https://www.fao.org/in-action/remote-sensing-for-water-productivity) | Global | Satellite ET, biomass, water productivity | REST | ✅ |
| [EU WFD](https://www.eea.europa.eu) | Europe | Water Framework Directive status | REST | ✅ |
| [Hub'Eau](https://hubeau.eaufrance.fr/api/v2/hydrometrie) | France | River water level, discharge | REST | ✅ |
| [Japan MLIT](https://www.mlit.go.jp) | Japan | Hydrometeorology, river observations | REST | ✅ |
| [Korea WAMIS](https://www.wamis.go.kr) | Korea | Hydrology, dam operations | REST | ✅ |
| [India WRIS](https://indiawris.gov.in) | India | River water level | REST | ✅ |
| [UK Environment Agency](https://environment.data.gov.uk) | UK | ? | REST | ✅ |

---

## API Keys — what you need before collecting

| Source | Key required? | How to get one |
| :--- | :---: | :--- |
| Taiwan MOENV | Recommended | [Register](https://data.moenv.gov.tw/en/apikey) — free |
| Taiwan WRA / Civil IoT | No | Open access |
| USGS | Optional | [Request](https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys) — free |
| Water Quality Portal | No | Open access |
| GEMStat | No | Open access via Zenodo |
| UN SDG 6 | No | Open access |
| OpenMeteo | No | Open access |
| Copernicus CDS | **Yes** | [Register](https://cds.climate.copernicus.eu/user/register) — free |
| FAO AQUASTAT / WaPOR | No | Open access |
| EU WFD | No | Open access |
| Hub'Eau | No | Open access |
| Japan MLIT / Korea WAMIS | No | Open access |
| UK Environment Agency | No | Open access |

---

## Adding a new source

Want to add your country's water data? See the contributor guide: [adding a data source](guides/adding_data_source.md).
