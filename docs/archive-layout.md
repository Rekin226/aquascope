# Archive layout: the contract

The Archive is the public Hugging Face dataset
[`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges). People read it from
DuckDB, R, QGIS and pandas without installing AquaScope, so its paths and columns are a public interface, the
same as a Python function. This page lists every published prefix, its files and columns, units, cadence,
licence and the workflow that writes it. [Stability](stability.md) says what may change; [The Archive](archive.md)
explains how to use it.

Every file is at `https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/<path>`, or
`hf://datasets/Rekin226/aquascope-gauges/<path>` in DuckDB and pandas.

## The promise

From 1.0, within a major version:

- **Paths stay.** A published path is not renamed or removed. A new layout is added beside the old one, and the
  old one keeps being written for the deprecation period (see [Stability](stability.md#deprecation)).
- **Columns stay.** A column is not renamed, removed, retyped or given a new unit. New columns may be added
  at the end, so read columns by name, not by position.
- **Keys stay.** `(source, station_id)` identifies a gauge everywhere, `river_id` a GEOGLOWS v2 reach,
  `hybas_id` a BasinATLAS sub-basin. Dates are ISO 8601 (`YYYY-MM-DD`, months `YYYY-MM`), in UTC.
- **Units are fixed per column** and given below. Discharge is m3/s, levels are metres, precipitation is mm.
- **Cadence is a target, not a guarantee.** The schedules below are what CI runs. A run can fail;
  `health.json`, `obs/refresh_status.json` and each prefix's `manifest.json` say when a file was last made.
- **Values change.** Rows are added and revised as agencies revise their records and as new days arrive. A
  method fix can change a derived number; the CHANGELOG says so.

What is not covered: the dataset card (`README.md`), the contents of the `manifest.json` and `index.json` files
beyond the keys named here, row order, Parquet row-group sizes and compression, and anything under a prefix this
page marks experimental.

Each scheduled job writes only under its own prefix (the harvest also writes the catalogue files at the root),
plus its own record under `health/jobs/`.
The forecast, warnings, climatology, skill and bulletin jobs build a smoke run without publishing it.

## Status on 2026-10-12

Checked against the live dataset (file listing, Parquet schemas and JSON manifests). Every prefix below is
published except `health/`, whose writer was merged on 2026-10-12 and publishes from each job's next run;
`scout/`, `waterbalance/` and the daily `flowstatus/` files first appeared on 2026-10-11.

| Prefix | Status | Stability |
| --- | --- | --- |
| root catalogue (`stations.parquet`, ...) | live | stable |
| `obs/` | live | stable |
| `basins/` | live | stable |
| `skill/` | live | experimental |
| `bulletins/` | live | experimental |
| `forecasts/` (`issued/`, `status/`, `reaches.parquet`) | live | experimental |
| `forecasts/warnings/` | live | experimental |
| `feeds/` | live | experimental |
| `flowstatus/climatology/` | live | experimental |
| `flowstatus/` (daily files) | live | experimental |
| `scout/` | live | experimental |
| `waterbalance/` | live | experimental |
| `context/` | live | experimental |
| `health/` | from the next scheduled runs | experimental |

The experimental prefixes were first published in October 2026 by the 0.26.0 and 0.27.0 jobs. They follow the
same rules for adding columns, but a path or column there may change in a minor release, with a CHANGELOG
entry, until it is declared stable.

## Root: the station catalogue

Written by [`harvest.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/harvest.yml),
weekly (Mondays 03:17 UTC), with `aquascope harvest stations`.

| File | What |
| --- | --- |
| `stations.parquet` | GeoParquet 1.0, one row per station from every station-capable source |
| `stations.geojson` | the same rows as GeoJSON points |
| `health.json` | the last run, per source: `source`, `ok`, `n_stations`, `seconds`, `error`, `license`, `redistributable`, `agency`; plus `run_at`, `aquascope_version`, `n_stations`, `n_ok`, `n_failed`, `files` |
| `signatures.parquet` | flow signatures of every mirrored discharge station |
| `README.md` | the dataset card (not part of the contract) |

`stations.parquet` columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | string | AquaScope source key (`usgs`, `uk_ea`, ...; see [Data sources](data_sources.md)) |
| `station_id` | string | the station's id in the catalogue (join key) |
| `site_id` | string | the physical site within a source; defaults to `station_id` |
| `name` | string | station name as the agency gives it |
| `latitude`, `longitude` | double | WGS84 degrees |
| `variables` | list of string | what the station measures (`discharge`, `water_level`, `groundwater_level`, `precipitation`, ...) |
| `period_start`, `period_end` | date | the record's span where the agency says it, else empty |
| `url` | string | deep link to the agency's page |
| `river`, `country`, `agency` | string | where known |
| `license` | string | the source's licence identifier |
| `redistributable` | bool | true when the source's terms allow mirroring observations |
| `extra` | string | source-specific fields as JSON |
| `geometry` | binary | WKB point (GeoParquet `geo` metadata) |

`signatures.parquet` columns: `source`, `station_id`, `variable`, `latitude`, `longitude`, `unit`,
`record_start`, `record_end` (dates as strings), `record_years`, `n_days`, `data_years`, `completeness`,
`q_mean`, `q5`, `q50`, `q95` (m3/s; `q5` is exceeded 5 % of days), `bfi` (Lyne-Hollick baseflow index),
`n_amax_years`, `amax_mk_p`, `amax_sen_slope` (m3/s per year), `amax_trend` (`rising`, `falling` or `none`),
`q100` (m3/s, GEV by L-moments), `amax_doy_mean`, `amax_doy_r`, `notes` (why a value is empty).

Licence: each row carries its source's licence in `license`; observations-derived values follow the source.
Attribution per source is in `aquascope.registry.SOURCES[key].attribution`.

## `obs/`: daily observations

Written by `harvest.yml`, weekly, incrementally (`python -m aquascope.archive.refresh`, then
`aquascope harvest bundles`). Only sources whose registry entry has `redistributable=True` are mirrored.

| Path | Columns |
| --- | --- |
| `obs/<variable>/<source>/<station_id>.csv.gz` | `date,value`: one row per day |
| `obs/<variable>/<source>.parquet` | `station_id` (string), `date` (date32), `value` (double) |
| `obs/manifest.json` | keys `sources`, `bundles`, `bundle_status`, `updated_at`, `version`; per `"<source>/<variable>"`, each station's period, count and unit |
| `obs/last_run.json` | the last harvest per source and variable: `attempted`, `harvested`, `empty`, `failed`, `seconds`, `errors`, `budget_exhausted`, `stopped` |
| `obs/refresh_status.json` | per source and variable: `status`, `station_budget`, `time_budget_seconds`, `exit_code`, start and end times |

`<variable>` is one of `discharge` (m3/s), `water_level` (m), `groundwater_level` (m) and `precipitation`
(mm per day). `<source>` is the source key; `<station_id>` matches `stations.parquet`. Values are daily, in
those units, as the agency publishes them after AquaScope's unit conversion; quality flags are not carried.

Licence: the source's own (the `license` column of `stations.parquet`, and the table in [The Archive](archive.md)).

## `basins/`: BasinATLAS catchments

The BasinATLAS files are written by [`basins.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/basins.yml)
(manual; BasinATLAS v1.0 does not change). The three station tables are rebuilt weekly by `harvest.yml`.

| File | What | Writer |
| --- | --- | --- |
| `basins/lev12_topology.parquet` | `hybas_id`, `next_down`, `next_sink`, `main_bas` (int64), `sub_area`, `up_area` (km2), `pfaf_id`, `endo`, `coast`, `order`, `lat`, `lon` (a representative point) | basins.yml |
| `basins/lev12_attributes.parquet` | every BasinATLAS attribute per level-12 sub-basin, keyed by `hybas_id` | basins.yml |
| `basins/lev12.fgb` | simplified level-12 polygons, FlatGeobuf with a spatial index | basins.yml |
| `basins/lev12.pmtiles`, `basins/lev06.pmtiles` | vector tiles for maps | basins.yml |
| `basins/build.json` | `built_at`, `n_basins`, `files`, `attribution`, `license` | basins.yml |
| `basins/station_catchments.parquet` | each station's sub-basin: `source`, `station_id`, `hybas_id`, `up_area`, `sub_area`, `area_km2`, `area_source`, `attribute_scope`, then catchment attributes with their unit in the name (`elevation_m`, `slope_deg`, `precipitation_mm_yr`, `pet_mm_yr`, `aet_mm_yr`, `aridity_index`, `temperature_c`, `snow_cover_pct`, `runoff_mm_yr`, `discharge_m3s`, land cover, soil, population and regulation fields) | harvest.yml |
| `basins/station_signatures.parquet` | donor signatures for regionalisation, in mm per day over the catchment: `source`, `station_id`, `area_km2`, `area_source`, `hybas_id`, `n_days`, `n_years`, `start`, `end`, `q_mean_mm`, `q_median_mm`, `q95_mm`, `q05_mm`, `q_annual_max_mm`, `runoff_ratio`, `baseflow_index`, `fdc_slope`, `high_flow_frequency`, `low_flow_frequency`, `zero_flow_fraction`, `seasonality_index`, `flashiness_index` | harvest.yml |
| `basins/regionalization_skill.json` | leave-one-out skill of the regionalisation methods: `n_stations`, `n_signature_stations`, `k`, `run_at`, `methods` | harvest.yml |

Licence: CC BY 4.0, BasinATLAS v1.0 (Linke et al. 2019, https://doi.org/10.1038/s41597-019-0300-6). The station
tables also carry each source's terms for the gauge data they are computed from.

## `skill/`: the evidence ladder (experimental)

Written by [`model-skill.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/model-skill.yml),
monthly (the 3rd, 05:41 UTC).

| File | What |
| --- | --- |
| `skill/model_skill.parquet` | one row per gauge and model: `source`, `station_id`, `lat`, `lon`, `area_km2`, `model`, `label`, `grade` (A to D), `why`, `is_best`, `kge`, `r`, `alpha`, `beta`, `nse`, `pbias`, `mean_gauge`, `mean_model` (m3/s), `n_days`, `start`, `end`, `flood_years`, `q2_*`, `q10_*`, `q100_*` (gauge and model, m3/s, and `_error_pct`), `site_id`, `match`, `distance_m`, `model_area_km2`, `area_ratio`, `sentence`, `licence`, `computed_at` |
| `skill/manifest.json` | the run: `built_at`, `n_rows`, `n_gauges`, `rows_by_model`, `dropped`, `failures`, `models`, `grading` |

Licence: model output under each model's terms (GEOGLOWS v2 and Google GRRR CC BY 4.0, NWM public domain), scored
against gauge data under each source's terms.

## `forecasts/`: issued forecasts and today's status (experimental)

Written by [`forecast-archive.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/forecast-archive.yml),
daily (06:41 UTC), with `python -m aquascope.archive.forecasts`.

| File | What |
| --- | --- |
| `forecasts/status/latest.parquet` | each live gauge's flow today against normal: `source`, `station_id`, `value_date`, `value` (m3/s), `percentile`, `class` (`much_below`, `below`, `normal`, `above`, `much_above`), `n_years` |
| `forecasts/status/<YYYY-MM-DD>.parquet` | the same snapshot, kept by date |
| `forecasts/status/latest.json` | `made`, `date`, `sources`, `n`, `counts`, `classes`, `method` |
| `forecasts/issued/<YYYY-MM-DD>.parquet` | one row per gauge, model and valid day: `issue_date`, `source`, `station_id`, `river_id`, `model`, `init_date`, `valid_date`, `lead_day`, `mean`, `median`, `p25`, `p75`, `min`, `max`, `high_res` (m3/s, raw) and the same with `_c` (corrected to the gauge), `kge_raw`, `kge_corrected`, `reach_mean_ratio`, `gauge_q2` to `gauge_q100` (m3/s), `by`, `generated` |
| `forecasts/reaches.parquet` | each gauge's reach: `source`, `station_id`, `river_id`, `distance_m`, `lat`, `lon`, `glofas_lat`, `glofas_lon`, `snapped_at` |
| `forecasts/manifest.json` | `issues` (date, file, counts, what was dropped and why), `columns`, `status`, `licence` |

Licence: GEOGLOWS v2 output CC BY 4.0; GloFAS through Open-Meteo CC BY 4.0. Model output, labelled so.

### `forecasts/warnings/`: floods ahead (experimental)

Written by [`flood-warnings.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/flood-warnings.yml),
daily (14:23 UTC), with `python -m aquascope.archive.warnings`. Model output, not an official warning.

| File | What |
| --- | --- |
| `forecasts/warnings/latest.parquet` | one row per reach expected to reach its 2-year flow in 15 days: `river_id`, `lat`, `lon`, `strahler_order`, `area_km2`, `peak_cms`, `peak_date`, `rp` (years), `share` (of 51 members), `q2` to `q100` (m3/s), `daily` (one character per day, `0` to `6`), `gauges` |
| `forecasts/warnings/<YYYY-MM-DD>.parquet` | the same table, kept by the forecast's start date (the last 60 days, then the first of each month; see [retention](archive.md#retention-of-the-daily-files)) |
| `forecasts/warnings/latest.geojson` | the same reaches as slim points for maps |
| `forecasts/warnings/manifest.json` | `issue_date`, `valid_from`, `valid_to`, `counts`, `thresholds`, `method`, `smoke` |

Licence: CC BY-NC-SA 4.0, because the return-period thresholds come from GEOGLOWS's retrospective return periods,
which their metadata licenses that way. The forecast itself is CC BY 4.0.

## `feeds/`: per-gauge Atom feeds (experimental)

Written by `forecast-archive.yml`, daily, after `forecasts/` (`python -m aquascope.archive.feeds`).

| File | What |
| --- | --- |
| `feeds/<source>/<station_id>.xml` | an Atom 1.0 feed per live gauge (the id with anything but `A-Z a-z 0-9 . _ -` as `_`) |
| `feeds/index.json` | `updated`, `date`, `base`, `n`, `rights`, `feeds` (gauge to path) |
| `feeds/state.parquet` | the entries kept per gauge: `source`, `station_id`, `kind`, `cls`, `level`, `entry_id`, `updated`, `link`, `title`, `summary` |

Licence: the gauge's agency for observations; GEOGLOWS v2 CC BY 4.0 for forecast entries.

## `flowstatus/`: flow against normal (experimental)

| Path | What | Writer, cadence |
| --- | --- | --- |
| `flowstatus/climatology/month-01.parquet` to `month-12.parquet` | `river_id` (int64), `p10`, `p25`, `p75`, `p90` (float32, m3/s): the percentiles of daily flow in that calendar month, 1950 to 2019 | [`flow-climatology.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/flow-climatology.yml), manual, once |
| `flowstatus/climatology/reaches.parquet` | `river_id`, `lat`, `lon`, `strahler_order` | flow-climatology.yml |
| `flowstatus/climatology/manifest.json` | `source`, `base_period`, `percentiles`, `min_strahler_order`, `n`, `files`, `licence` | flow-climatology.yml |
| `flowstatus/latest.parquet` | `river_id` (int64), `flow_cms` (float32, m3/s), `class` (uint8, 1 much below to 5 much above) | flood-warnings.yml, daily |
| `flowstatus/latest.bin.gz`, `flowstatus/<YYYY-MM-DD>.bin.gz` | the compact class file the Explorer draws from (format in `aquascope.flowstatus`); dated files (the last 60 days, then the first of each month; see [retention](archive.md#retention-of-the-daily-files)) | flood-warnings.yml, daily |
| `flowstatus/manifest.json` | the day's run: `issue_date`, `n`, `counts`, `classes`, `percentiles`, `base_period`, `files`, `licence` | flood-warnings.yml, daily |

Licence: CC BY 4.0 (GEOGLOWS v2 forecast and flow duration curves).

## `scout/`: what stands out today (experimental)

Written by `flood-warnings.yml`, daily, after the Floods ahead issue (`python -m aquascope.map_scout`).

| File | What |
| --- | --- |
| `scout/latest.json` | `made`, `date`, `method`, `inputs`, `findings`, `picks`, `counts`, `sources`: the world's findings, each of one kind (`status`, `floods_ahead`, `floods_past`, `gauges_today`, `models_disagree`), with its place, numbers, reason and source |
| `scout/<YYYY-MM-DD>.json` | the same, kept by date (the last 60 days, then the first of each month; see [retention](archive.md#retention-of-the-daily-files)) |

Licence: each finding names its layer, under that layer's licence.

## `waterbalance/`: subbasins by monthly water balance (experimental)

Written by [`water-balance.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/water-balance.yml),
monthly (the 9th, 05:41 UTC), with `python -m aquascope.waterbalance`. Modelled, not measured.

| File | What |
| --- | --- |
| `waterbalance/monthly.parquet` | one row per HydroBASINS level-4 subbasin and month since 1990: `hybas_id` (int64), `month` (`YYYY-MM`), `p_mm`, `et_mm`, `q_mm`, `closure_mm` (mm in the month), `q_out_cms`, `q_in_cms` (m3/s), `era5t` (bool), `p_anom`, `et_anom`, `q_anom`, `closure_anom` (mm against 1991-2020) |
| `waterbalance/normals.parquet` | `hybas_id`, `month_of_year`, `p_mm`, `et_mm`, `q_mm`, `closure_mm`: 1991-2020 means |
| `waterbalance/basins.parquet` | `hybas_id`, `k`, `area_km2`, `lat`, `lon`, `outlets`, `down_id`, `up_ids`, `linked`, `runoff_ok`, `runoff_check`, `era5_cells` |
| `waterbalance/basins.geojson` | simplified outlines |
| `waterbalance/normals.json`, `waterbalance/years/<YYYY>.json` | the Explorer's files |
| `waterbalance/manifest.json` | months covered, method, licences, cost |

Licence: CC BY 4.0 (ERA5 from Copernicus Climate Change Service / ECMWF, GEOGLOWS v2, BasinATLAS).

## `bulletins/`: the monthly state of the rivers (experimental)

Written by [`bulletin.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/bulletin.yml), monthly
(the 3rd, 08:17 UTC), with `python -m aquascope.bulletin`.

| File | What |
| --- | --- |
| `bulletins/<YYYY-MM>/status.parquet` | one row per gauge: `source`, `station_id`, `month`, `value` (monthly mean flow, m3/s), `n_days`, `percentile`, `class`, `n_years`, `median` (m3/s), `ratio`, `record`, `country`, `basin_id` |
| `bulletins/<YYYY-MM>/bulletin.json` | the bulletin's numbers, per country and basin |
| `bulletins/<YYYY-MM>/bulletin.html`, `bulletin.md`, `map.png` | the bulletin for reading and printing |
| `bulletins/index.json` | `latest`, `months` (newest first: `month`, `label`, `made`, `classed`, `considered`, `counts`, `headline`, `sources`) |

Licence: computed from the mirrored observations, under each source's terms.

## `health/`: scheduled job health (experimental)

Written by every scheduled workflow's last step, whether the job passed or not (`aquascope.archive.health_jobs`,
#593). A smoke run's record is never published.

| File | What |
| --- | --- |
| `health/jobs/<job>.json` | one job: `job`, `label`, `workflow`, `cadence`, `cadence_hours`, `stale_after_hours`, `layers`, `writes`, `updated_at`, `last_run`, `last_success`, `last_failure` (each with `status`, `started_at`, `finished_at`, `duration_s`, `event`, `run_url`, `counts`, `data_date`, `reasons`, `notes`), `consecutive_failures`, `history` |
| `health/jobs/index.json` | `about`, `updated_at`, `jobs`: every job in `JOBS` with its expected cadence and newest state, which the Explorer reads |

Licence: none of its own; the files hold run metadata only, no observations.

## `context/`: place-context mirrors (experimental)

Written by [`mirror-context.yml`](https://github.com/Rekin226/aquascope/blob/main/.github/workflows/mirror-context.yml),
manual (run again when a source releases a new version). Each dataset is a cell-sorted Parquet plus one small
gzipped CSV per 2-degree cell (`n00_e000` names the cell by its south-west corner).

| Path | What | Licence |
| --- | --- | --- |
| `context/dams/gdw_barriers.parquet`, `context/dams/cells/<cell>.csv.gz` | Global Dam Watch v1.0 barriers: `gdw_id`, `name`, `reservoir`, `river`, `country`, `year`, `height_m`, `capacity_mcm`, `area_km2`, `main_use`, `dor_pc`, `catchment_km2`, `lat`, `lon`, `grand_id` (`cell` in the Parquet) | CC BY 4.0 |
| `context/ghcn/prcp_stations.csv.gz` | NOAA GHCN-Daily stations that record precipitation: `id`, `lat`, `lon`, `elev_m`, `name`, `first_year`, `last_year` | CC0 |
| `context/floods/groundsource.parquet`, `context/floods/groundsource/cells/<cell>.csv.gz` | Groundsource flood events from news: `uuid`, `start_date`, `end_date`, `lat`, `lon`, `west`, `south`, `east`, `north`, `area_km2` | CC BY 4.0 |
| `context/floods/microsoft.parquet`, `context/floods/microsoft/cells/<cell>.csv.gz` | Microsoft AI for Good Sentinel-1 flood detections per 0.05 degree cell and month: `lat`, `lon`, `year`, `month`, `n` | MIT |
| `context/floods/monthly/grid.parquet` | both flood sources per half-degree cell and month: `month`, `row`, `col`, `lat`, `lon`, `news`, `radar` | CC BY 4.0 (news), MIT (radar) |
| `context/floods/monthly/index.json`, `context/floods/monthly/months/<YYYY-MM>.json.gz` | the months covered, and one compact file per month for the Explorer | as above |
| `context/manifest.json` | per dataset: `rows`, `cells`, `folder`, `licence`, `attribution`, `source`, `built` | |

## Changing this page

A pull request that adds a prefix, file or column updates this page in the same change;
`tests/test_public_api.py` checks that every prefix the code publishes is listed here. A pull request that would
rename or remove a path or column follows the deprecation rules in [Stability](stability.md).
