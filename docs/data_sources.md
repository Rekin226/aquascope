# Data Sources

AquaScope ships **29 collectors** that normalise water data into typed Pydantic records. One API call per source, one schema across the toolkit.

Most sources emit point observations and share the unified `water_data` schema (`WaterQualitySample`, `WaterLevelReading`, `ReservoirStatus`). Three aggregate/gridded sources use purpose-built record types that match their data shape: **FAO AQUASTAT** returns country-level `AquastatRecord`, **UN SDG 6** returns `SDG6Indicator`, and **FAO WaPOR** returns gridded `WaPORObservation`.

To request a new source, open an [issue](https://github.com/Rekin226/aquascope/issues/new/choose) using the *New Data Source Request* template, or join the [Discussion](https://github.com/Rekin226/aquascope/discussions) thread on data-source priorities.

---

## Sources

| Source | Region | Data Types | API | Status |
| :--- | :--- | :--- | :--- | :---: |
| [Taiwan CWA CODIS](https://codis.cwa.gov.tw) | Taiwan | Daily climate: rainfall, temperature, humidity, radiation, wind | REST | ✅ |
| [Taiwan MOENV](https://data.moenv.gov.tw) | Taiwan | River / tap water quality, RPI | REST | ✅ |
| [Taiwan WRA](https://opendata.wra.gov.tw) | Taiwan | Water levels, reservoir status | REST | ✅ |
| [Taiwan Civil IoT](https://sta.ci.taiwan.gov.tw) | Taiwan | Real-time sensors (level, flow, rain) | SensorThings | ✅ |
| [Taiwan WRA FHY](https://fhy.wra.gov.tw) | Taiwan | Real-time water level, rainfall, discharge | REST | ✅ |
| [Taiwan WRA IoT](https://iot.wra.gov.tw) | Taiwan | Groundwater level, rainfall accumulation | REST | ✅ |
| [Taiwan data.gov.tw](https://data.gov.tw) | Taiwan | Real-time river + groundwater level | REST | ✅ |
| [Taiwan WRA Groundwater](https://opendata.wra.gov.tw) | Taiwan | Annual groundwater levels + well metadata (992 wells, 1992–) | REST | ✅ |
| [USGS](https://api.waterdata.usgs.gov) | USA | Streamflow, water quality, gage height | OGC | ✅ |
| [NOAA NWPS](https://api.water.noaa.gov/nwps/v1/) | USA | Streamflow observations, location metadata | REST | ✅ |
| [Water Quality Portal](https://waterqualitydata.us) | USA | Integrated WQ from 400+ agencies | REST / CSV | ✅ |
| [GEMStat](https://gemstat.org) | Global | Freshwater quality (170+ countries) | Zenodo | ✅ |
| [UN SDG 6](https://sdg6data.org) | Global | SDG 6 indicators (6.1.1 – 6.6.1) | REST | ✅ |
| [OpenMeteo](https://open-meteo.com) | Global | Weather (temp, precip, wind, solar) | REST | ✅ |
| [Copernicus](https://cds.climate.copernicus.eu) | Global | ERA5 reanalysis, climate projections | CDS API | ✅ |
| [FAO AQUASTAT](https://www.fao.org/aquastat) | Global | Country-level water withdrawal, irrigation | FAOSTAT API | ✅ |
| [FAO WaPOR](https://www.fao.org/in-action/remote-sensing-for-water-productivity) | Global | Satellite ET, biomass, water productivity | REST | ✅ |
| [EU WFD](https://www.eea.europa.eu) | Europe | Water Framework Directive status | REST | ✅ |
| [Hub'Eau](https://hubeau.eaufrance.fr/api/v2/hydrometrie) | France | River water level, discharge | REST | ✅ |
| [PEGELONLINE](https://www.pegelonline.wsv.de/webservice/dokuRestapi) | Germany | River water level, discharge | REST | ✅ |
| [Japan MLIT](https://www.mlit.go.jp) | Japan | Hydrometeorology, river observations | REST | ✅ |
| [Korea WAMIS](https://www.wamis.go.kr) | Korea | Hydrology, dam operations | REST | ✅ |
| [India WRIS](https://indiawris.gov.in) | India | River water level | REST | ✅ |
| [GRDC](https://zenodo.org/records/19126732) | Global | River discharge (in-situ gauges + RSEG satellite) | Zenodo / Dataverse | ✅ |
| [CAMELS-CL](https://www.cr2.cl/camels-cl/) | Chile | Daily observed streamflow, catchment attributes | ZIP / CSV | ✅ |
| [CAMELS-BR](https://doi.org/10.5281/zenodo.3709337) | Brazil | Daily observed streamflow, catchment attributes | ZIP / CSV | ✅ |
| [Ireland OPW](https://waterlevel.ie) | Ireland | River / lake water level (15-min resolution) | GeoJSON / CSV | ✅ |
| [Environment Agency (England)](https://environment.data.gov.uk) | England (UK) | River and groundwater levels, river flow, rainfall data | REST | ✅ |
| [BOM Water Data Online](http://www.bom.gov.au/waterdata/) | Australia | Streamflow, water level, storage, groundwater level | KISTERS WISKI (KiWIS) | ✅ |

---

## API Keys — what you need before collecting

| Source | Key required? | How to get one |
| :--- | :---: | :--- |
| Taiwan MOENV | Recommended | [Register](https://data.moenv.gov.tw/en/apikey) — free |
| Taiwan WRA / Civil IoT | No | Open access |
| USGS | Optional | [Request](https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys) — free |
| NOAA NWPS | No | Open access |
| Water Quality Portal | No | Open access |
| GEMStat | No | Open access via Zenodo |
| UN SDG 6 | No | Open access |
| OpenMeteo | No | Open access |
| Copernicus CDS | **Yes** | [Register](https://cds.climate.copernicus.eu/user/register) — free |
| FAO AQUASTAT / WaPOR | No | Open access |
| EU WFD | No | Open access |
| Hub'Eau | No | Open access |
| PEGELONLINE | No | Open access |
| Japan MLIT / Korea WAMIS | No | Open access |
| CAMELS-CL | No | Open access |
| CAMELS-BR | No | Open access via Zenodo |
| Ireland OPW | No | Open access via waterlevel.ie |
| Environment Agency (England) | No | Open access |
| BOM Water Data Online | No | Open access |

---

## Adding a new source

Want to add your country's water data? See the contributor guide: [adding a data source](guides/adding_data_source.md).

## PEGELONLINE (Germany)

- **Source type:** `pegelonline`
- **Coverage:** German federal waterways — water level (`W`) and discharge (`Q`)
- **Collector:** `aquascope.collectors.pegelonline.PegelonlineCollector`

PEGELONLINE stations should be addressed by UUID. The upstream service keeps
raw measurements for only the most recent **31 days**, so the collector rejects
larger relative windows instead of returning a misleading partial result.

**Usage:**
```python
from aquascope.collectors import PegelonlineCollector

collector = PegelonlineCollector()
readings = collector.collect(
    station_id="593647aa-9fea-43ec-a7d6-6476a76ae868",  # Bonn
    days=7,
)
```

From the CLI:
```bash
# Both water level and discharge (when the station publishes both)
aquascope collect --source pegelonline \
  --station 593647aa-9fea-43ec-a7d6-6476a76ae868 --days 7

# Discharge only
aquascope collect --source pegelonline \
  --station 593647aa-9fea-43ec-a7d6-6476a76ae868 --timeseries Q
```

## BOM Water Data Online (Australia)

- **Source type:** `bom`
- **Coverage:** Australia — streamflow, water level, storage, and groundwater level from ~8,000 gauging stations
- **Collector:** `aquascope.collectors.bom.BOMCollector`

BOM stations are addressed by AWRC station number. The collector resolves
the underlying KISTERS WISKI `ts_id` for a station/parameter combination
(defaulting to the quality-checked merged daily mean time series) before
fetching values. Discharge, quality, and rainfall parameters normalise to
`WaterQualitySample`; level-type parameters (`Water Course Level`,
`Storage Level`, `Ground Water Level`) normalise to `WaterLevelReading`.

**Usage:**
```python
from aquascope.collectors import BOMCollector

collector = BOMCollector()

# Discharge -- confirmed working with the default time-series name.
samples = collector.collect(
    station_id="410001",  # Murrumbidgee River at Wagga Wagga
    parameter_type="Water Course Discharge",
    days=30,
)

# Water level -- confirmed working. Note: not every station maintains a
# discharge series (see below), but level is broadly reliable.
levels = collector.collect(
    station_id="409001",  # Murray River at Albury (Union Bridge)
    parameter_type="Water Course Level",
    days=30,
)
```

From the CLI:
```bash
aquascope collect --source bom --station 410001 --days 30
aquascope collect --source bom --station 409001 --parameter-type "Water Course Level" --days 30
```

**Known data-availability quirks (verified against the live API):**

- **Not every station has a populated discharge series.** BOM publishes a
  `Water Course Discharge` parameter entry for most stations, but on
  regulated rivers with locks/weirs the merged daily-mean series can be
  entirely `None`, and even the raw continuous series can hold a constant
  placeholder value (observed: `0.0` with quality code `210`) rather than
  real readings. This collector already filters out `None` values, so a
  call against such a station simply returns an empty list rather than
  fabricated zeros — but it's worth checking `Water Course Level` for that
  station instead, since level is usually maintained even when discharge
  isn't.
- **Daily-mean naming varies by station.** Some stations only populate
  `DMQaQc.Merged.DailyMean.24HR` (calendar day), others also/instead
  populate `DMQaQc.Merged.DailyMean.09HR` (the 9am-to-9am Australian
  hydrological day). The collector defaults to `24HR`; pass
  `ts_name="DMQaQc.Merged.DailyMean.09HR"` if a station's `24HR` series
  comes back empty.

## GRDC (Global Runoff Data Centre)

**Source type:** `grdc`
**Coverage:** Global river discharge — in-situ gauge stations + satellite discharge estimates
**Collector:** `aquascope.collectors.grdc.GRDCCollector`

Two `source_type` values, both required for downstream filtering (e.g. Prediction
in Ungauged Basins, #53):

- `in_situ` — curated gauge-station subset published on Zenodo
  ([record 19126732](https://zenodo.org/records/19126732)).
  **License: CC BY-NC 4.0 — non-commercial use only, attribution required.**
- `satellite` — RSEG remote-sensing discharge extension, published on DaRUS
  ([doi:10.18419/darus-3558](https://doi.org/10.18419/darus-3558)).

The classic GRDC portal (grdc.bafg.de) requires an email request-form with ~24h
turnaround and is not used by this collector — both sources above are directly
downloadable.

SAEM (a second satellite extension, [doi:10.18419/darus-4475](https://doi.org/10.18419/darus-4475))
is not yet integrated — planned as a follow-up.

**Usage:**
```python
from aquascope.collectors import GRDCCollector

collector = GRDCCollector()
in_situ = collector.collect(source_type="in_situ")
satellite = collector.collect(source_type="satellite")
```

From the CLI:
```bash
aquascope collect --source grdc                    # in-situ gauges (default)
aquascope collect --source grdc --mode satellite   # RSEG satellite estimates
```

## CAMELS-CL (Chile)

- **Source type:** `camels_cl`
- **Coverage:** Chile — daily observed streamflow and catchment attributes for 516 catchments
- **Collector:** `aquascope.collectors.camels_cl.CAMELSCLCollector`

**Usage:**
```python
from aquascope.collectors import CAMELSCLCollector

collector = CAMELSCLCollector()
readings = collector.collect(station_ids=["1001001"])
```

## CAMELS-BR (Brazil)

- **Source type:** `camels_br`
- **Coverage:** Brazil — daily observed streamflow and catchment attributes for 897 catchments
- **Collector:** `aquascope.collectors.camels_br.CAMELSBRCollector`

**Usage:**
```python
from aquascope.collectors import CAMELSBRCollector

collector = CAMELSBRCollector()
readings = collector.collect(station_ids=["10500000"])
```

From the CLI:
```bash
aquascope collect --source camels_br --station-ids 10500000
```
