# FAO Agriculture Integration Plan

This guide defines the next FAO integrations that fit AquaScope best and turns
them into a concrete implementation plan for this repository.

The intent is to keep AquaScope focused on water intelligence across scales:
country water use, basin hydrology, and field-scale irrigation demand.

---

## Current Baseline

The repository already contains the core building blocks for an FAO-aligned
agriculture vertical:

- `aquascope.collectors.aquastat.AquastatCollector`
- `aquascope.collectors.wapor.WaPORCollector`
- `aquascope.agri` FAO-56 evapotranspiration, crop water demand, and soil water
  balance tools
- `aquascope.schemas.agriculture` for AQUASTAT and ET-oriented records

What is still missing is productization:

- full CLI exposure for FAO sources
- a workflow that connects FAO data to the agriculture module end to end
- a clear schema strategy for WaPOR outputs beyond reference ET
- examples and contributor guidance for agricultural workflows

---

## Recommended FAO Targets

The next integrations should stay within three FAO-aligned targets. Two are
already partially present in the codebase and should be completed first.

### 1. AQUASTAT via FAOSTAT API

**Status:** already scaffolded in `AquastatCollector`

**Endpoint family to support now**

- `GET /en/data/AQUASTAT`

**Parameters already aligned with the current collector**

- `area` — ISO3 country code or `all`
- `element` — AQUASTAT variable IDs
- `year` — year or year range
- `output_type=objects`

**Priority AQUASTAT variables**

- `4263` — Total water withdrawal
- `4253` — Agricultural water withdrawal
- `4254` — Industrial water withdrawal
- `4255` — Municipal water withdrawal
- `4192` — Total renewable water resources
- `4312` — Total area equipped for irrigation

**What AquaScope should produce**

- existing `AquastatRecord` objects for raw normalized ingest
- country-year time series for water use benchmarking
- irrigation intensity indicators such as agricultural withdrawal per irrigated
  area
- cross-country comparison tables ready for `recommend`, `eda`, and reporting

**Why this is next**

AQUASTAT gives AquaScope a national and policy layer that complements the
existing hydrology and field-scale agriculture modules.

### 2. WaPOR v3 API for evapotranspiration and productivity

**Status:** partially scaffolded in `WaPORCollector`

**Verified access pattern already used in the repo**

- `GET /catalog/workspaces/WAPOR-3/cubes/{cube_code}`

**Key query inputs already reflected in the collector**

- `startDate`
- `endDate`
- `bbox`

**Priority cube codes**

- `RET` — Reference evapotranspiration
- `AETI` — Actual evapotranspiration and interception
- `NPP` — Net primary production

**What AquaScope should produce**

- existing `ETReference` records for `RET`
- generic WaPOR observations for cube summaries and time series
- area-of-interest ET summaries suitable for irrigation planning
- biomass or productivity indicators for later water-productivity workflows

**Why this is next**

WaPOR is the field and district layer that bridges climate inputs to crop water
planning. It is the strongest FAO complement to the existing FAO-56 code.

### 3. FAO-56 workflow orchestration on top of FAO data

**Status:** already implemented at the function level in `aquascope.agri`

**No new external endpoint required**

This step is not about adding another FAO API first. It is about operationally
combining:

- WaPOR ET inputs
- Open-Meteo weather and precipitation
- AquaScope FAO-56 crop coefficient logic
- soil water balance and irrigation scheduling

**What AquaScope should produce**

- crop water requirement tables by day and growth stage
- irrigation schedules with net and gross water demand
- auto-irrigation recommendations for a field or district
- explainable outputs for researchers and extension teams

---

## Not Recommended Yet

Do not expand into broad FAOSTAT agriculture domains before the above is
stable. In particular, avoid pulling in many crop, trade, or livestock domains
until AquaScope has a crisp water-agriculture product story.

The right order is:

1. finish AQUASTAT and WaPOR user flows
2. connect them to the FAO-56 module
3. only then add broader FAO productivity or food-system comparisons

---

## Proposed Schema Plan

The existing schema layer is a good start. The next additions should stay small
and explicit.

### Keep as-is

- `AquastatRecord`
- `ETReference`
- `CropWaterRequirement`
- `IrrigationDemand`
- `SoilWaterStatus`

### Add next

#### `WaPORObservation`

Use for cube summaries and AOI time-series values when the output is not only
reference ET.

Recommended fields:

- `source: str = "WAPOR"`
- `cube_code: str`
- `cube_label: str | None`
- `date: date | None`
- `start_date: date | None`
- `end_date: date | None`
- `bbox: tuple[float, float, float, float] | None`
- `value: float`
- `unit: str | None`
- `statistic: str | None`
- `aoi_id: str | None`
- `level: str | None`

#### `CropWaterPlan`

Use as the aggregate result of a full irrigation-planning workflow.

Recommended fields:

- `crop: str`
- `location_name: str | None`
- `start_date: date`
- `end_date: date`
- `total_eto_mm: float`
- `total_etc_mm: float`
- `total_effective_rain_mm: float`
- `total_net_irrigation_mm: float`
- `total_gross_irrigation_mm: float`
- `days_triggered: int`
- `method: str`

#### `WaterProductivityRecord`

Reserve for the first combined AQUASTAT + WaPOR productivity workflow.

Recommended fields:

- `location_id: str`
- `year: int | None`
- `period_label: str | None`
- `aeti_mm: float | None`
- `npp_value: float | None`
- `agricultural_withdrawal_m3: float | None`
- `irrigated_area_ha: float | None`
- `productivity_per_water: float | None`

---

## Proposed CLI Plan

The current `collect` command should expose the FAO collectors first. After
that, AquaScope should add a dedicated agriculture command group.

### Phase 1: expose current FAO collectors

```bash
aquascope collect --source aquastat \
  --country EGY \
  --variables 4263,4253,4312 \
  --start-year 2010 \
  --end-year 2023

aquascope collect --source wapor \
  --bbox 30.50,29.80,31.10,30.20 \
  --variable RET \
  --start-date 2024-04-01 \
  --end-date 2024-06-30
```

### Phase 2: add agriculture workflow commands

```bash
aquascope agri eto --weather-file weather.csv --method penman_monteith

aquascope agri demand \
  --crop maize \
  --eto-file eto.csv \
  --precip-file precip.csv \
  --planting-date 2026-04-01

aquascope agri balance \
  --crop maize \
  --eto-file eto.csv \
  --precip-file precip.csv \
  --soil-fc 0.30 \
  --soil-wp 0.15 \
  --root-depth 1.0

aquascope agri benchmark \
  --aquastat-file data/raw/aquastat_20260401.json \
  --metric agricultural_withdrawal_per_irrigated_area
```

### Phase 3: add end-to-end FAO planning command

```bash
aquascope agri plan \
  --crop maize \
  --lat 29.95 \
  --lon 31.25 \
  --planting-date 2026-04-01 \
  --source wapor+openmeteo \
  --soil-fc 0.30 \
  --soil-wp 0.15 \
  --root-depth 1.0
```

This command should orchestrate data retrieval, ET estimation, crop water
requirement calculation, soil-water balance, and irrigation guidance.

---

## Prioritized Implementation Plan

### Phase 0. Surface What Already Exists

**Goal:** make the current FAO work reachable and trustworthy.

Repository changes:

- expose `aquastat` and `wapor` in `aquascope/cli.py`
- add source entries in `list-sources`
- add smoke tests for CLI source parity
- add a FAO example script under `examples/`
- update `README.md` and docs links

Expected output:

- users can collect FAO data without importing collectors manually
- docs match the package surface

### Phase 1. Productize AQUASTAT Country Benchmarking

**Goal:** support country-scale water and irrigation benchmarking.

Touch points:

- `aquascope/collectors/aquastat.py`
- `aquascope/schemas/agriculture.py`
- `aquascope/analysis/eda.py`
- `aquascope/reporting/`
- `tests/test_agri/`

Deliverables:

- reliable `AquastatCollector.collect()` workflow
- helper analysis functions for country-year indicators
- benchmark-ready report tables

Primary output objects:

- `AquastatRecord`
- derived benchmark DataFrames

### Phase 2. Productize WaPOR AOI Time-Series Support

**Goal:** move WaPOR from basic cube access to usable ET time series.

Touch points:

- `aquascope/collectors/wapor.py`
- `aquascope/schemas/agriculture.py`
- `aquascope/utils/storage.py`
- `tests/test_agri/`

Deliverables:

- support for AOI-oriented time-series outputs
- better normalization for `RET`, `AETI`, and `NPP`
- optional metadata discovery helpers for cube definitions

Primary output objects:

- `ETReference`
- `WaPORObservation`

### Phase 3. Add a Flagship Irrigation Planning Workflow

**Goal:** make agriculture a real end-user workflow instead of a set of low-level
functions.

Recommended new module:

- `aquascope/agri/planner.py`

Recommended public function:

```python
plan_irrigation(
    crop: str,
    planting_date: date,
    eto_series: pd.Series,
    precip_series: pd.Series,
    soil: SoilProperties,
    efficiency: float = 0.7,
) -> CropWaterPlan
```

Deliverables:

- workflow wrapper around `crop_water_requirement`, `irrigation_schedule`, and
  `SoilWaterBalance`
- summarized plan outputs for reporting and dashboards
- simple explainable inputs and outputs for non-programmer users

Primary output objects:

- `CropWaterPlan`
- `IrrigationDemand`
- `SoilWaterStatus`

### Phase 4. Add Water Productivity Benchmarking

**Goal:** connect national water policy and field-scale productivity.

Inputs:

- AQUASTAT agricultural withdrawal
- AQUASTAT irrigated area
- WaPOR `AETI`
- WaPOR `NPP`

Recommended new module:

- `aquascope/agri/productivity.py`

Primary output objects:

- `WaterProductivityRecord`

This phase is where AquaScope becomes distinct from a generic hydrology toolkit.

---

## Example Workflows

The examples below are the recommended user journeys to build next.

### Workflow A. Country Irrigation Benchmarking

**Question:** which countries are using the most agricultural water relative to
their irrigated area?

Proposed command flow:

```bash
aquascope collect --source aquastat \
  --country all \
  --variables 4253,4312 \
  --start-year 2015 \
  --end-year 2023 \
  --format json

aquascope agri benchmark \
  --aquastat-file data/raw/aquastat_latest.json \
  --metric agricultural_withdrawal_per_irrigated_area
```

Expected outputs:

- ranked country table
- trend lines by country
- report section for policy comparison

### Workflow B. Field or District Irrigation Demand

**Question:** how much irrigation water is needed for maize during the current
season in a target area?

Proposed command flow:

```bash
aquascope collect --source wapor \
  --bbox 30.50,29.80,31.10,30.20 \
  --variable RET \
  --start-date 2026-04-01 \
  --end-date 2026-07-31

aquascope collect --source openmeteo \
  --mode weather \
  --lat 29.95 \
  --lon 31.25 \
  --start-date 2026-04-01 \
  --end-date 2026-07-31

aquascope agri plan \
  --crop maize \
  --planting-date 2026-04-01 \
  --eto-file data/raw/wapor_ret_latest.json \
  --precip-file data/raw/openmeteo_latest.json \
  --soil-fc 0.30 \
  --soil-wp 0.15 \
  --root-depth 1.0
```

Expected outputs:

- daily ET and ETc table
- irrigation trigger dates
- total net and gross demand for the season

### Workflow C. Water Productivity Comparison

**Question:** where is agricultural water converted to biomass most efficiently?

Proposed command flow:

```bash
aquascope collect --source aquastat \
  --country EGY \
  --variables 4253,4312 \
  --start-year 2018 \
  --end-year 2023

aquascope collect --source wapor \
  --bbox 30.50,29.80,31.10,30.20 \
  --variable AETI \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

aquascope collect --source wapor \
  --bbox 30.50,29.80,31.10,30.20 \
  --variable NPP \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

aquascope agri productivity \
  --aquastat-file data/raw/aquastat_latest.json \
  --aeti-file data/raw/wapor_aeti_latest.json \
  --npp-file data/raw/wapor_npp_latest.json
```

Expected outputs:

- biomass-per-water indicators
- region or country benchmark charts
- combined policy plus biophysical productivity report

---

## Immediate Recommendation

If only one agriculture feature is implemented next, it should be this:

1. expose `aquastat` and `wapor` in the CLI
2. add `aquascope agri plan`
3. use WaPOR `RET` plus Open-Meteo precipitation as the first supported
   irrigation-planning workflow

That is the shortest path from existing code to a distinctive user-facing
feature.