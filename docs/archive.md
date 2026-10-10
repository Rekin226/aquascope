# The Archive: the world's public gauges as one open dataset

AquaScope's collectors already reach national agencies on four continents. The
Archive turns that reach into a data asset anyone can use without Python: a
scheduled harvest writes what the sources answer into cloud-native files and
publishes them to a public Hugging Face dataset,
[`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges).

Phase 0 is the **station catalog**. Phase 1 mirrors **daily observations**
for the sources whose terms allow it, filling up week by week. Phase 2 adds
**more variables** (water level, rainfall, groundwater level) and **one
Parquet bundle per variable and source** for whole-source reads. See
[#188](https://github.com/Rekin226/aquascope/issues/188) for the plan.

## What is in it

| file | contents |
| --- | --- |
| `stations.parquet` | GeoParquet 1.0 (WKB point geometry, WGS84). One row per station: `source`, `station_id`, `name`, `latitude`, `longitude`, `variables`, `period_start`, `period_end`, `url`, `river`, `country`, `agency`, `license`, `redistributable`, `extra`. |
| `stations.geojson` | the same rows as GeoJSON, for tools and browsers that don't read parquet |
| `health.json` | per-source status of the last run: station count, seconds, error message if the endpoint failed |
| `README.md` | the dataset card, regenerated on every run, with the per-source licence table |
| `obs/<variable>/<source>/<station_id>.csv.gz` | daily values for one station (`date,value`; discharge m3/s, water and groundwater level m, precipitation mm/day), only for redistributable sources |
| `obs/<variable>/<source>.parquet` | the folder above as one Parquet bundle: `station_id, date, value`, sorted, snappy; `station_id` joins to `stations.parquet` |
| `obs/manifest.json` | every harvested station with period, count, unit, measure note and harvest time, keyed by `source/variable`, plus every bundle; `obs/last_run.json` the last run's per-source tallies |

Sources with a station catalog today: USGS, the Environment Agency (England), Hub'Eau
(France), PEGELONLINE (Germany), Ireland OPW, Taiwan CWA. Every source that
gains a `stations()` implementation appears on the next run.

## Query it in place

DuckDB reads the parquet over HTTPS (Hugging Face serves range requests):

```sql
INSTALL httpfs; LOAD httpfs;
SELECT source, count(*) AS n
FROM 'https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet'
GROUP BY source ORDER BY n DESC;
```

Python, no aquascope needed:

```python
import pandas as pd
stations = pd.read_parquet("hf://datasets/Rekin226/aquascope-gauges/stations.parquet")
```

GeoPandas / QGIS open the parquet directly as a point layer.

## Observations, incrementally

Each weekly run harvests a budget of stations per source and variable and
re-harvests a station once it is older than 30 days, so the archive grows
without ever hammering an agency. What is mirrored:

| source | variables | where the daily value comes from |
| --- | --- | --- |
| `usgs` | discharge, water_level | USGS Water Data API (OGC v1) daily means, statistic 00003, parameters 00060 and 00065 (gage height converted from feet to metres) |
| `uk_ea` | discharge, water_level, precipitation, groundwater_level | Hydrology API daily mean flow; daily max level where no daily mean is published; daily rainfall totals; borehole levels in metres above Ordnance Datum (manual dips or logger) |
| `hubeau_hydrometrie` | discharge | obs_elab QmnJ, the elaborated daily mean discharge |
| `taiwan_cwa` | precipitation | CODIS daily rainfall |
| `poland_imgw` | discharge, water_level | IMGW-PIB daily archive (hydrological-year zips, one per month to 2022 and one per year after; the current year is published once it ends). The archive host sends no CORS headers, so this mirror is what the Explorer serves (#391) |
| `greece_openhi` | discharge, water_level, precipitation | 15-minute telemetry folded to daily means, rainfall to daily totals. The browser cannot call OpenHi directly (its API allows cross-origin requests from localhost only), so this mirror is what the Explorer serves (#408) |

Read one station:

```python
import pandas as pd
s = pd.read_csv("hf://datasets/Rekin226/aquascope-gauges/obs/discharge/usgs/USGS-01646500.csv.gz")
```

a whole source and variable in one go (the bundle):

```python
df = pd.read_parquet("hf://datasets/Rekin226/aquascope-gauges/obs/groundwater_level/uk_ea.parquet")
# or, with aquascope: from aquascope.archive import load_observations; df = load_observations("uk_ea", "groundwater_level")
```

or with DuckDB, joined to the catalog:

```sql
SELECT s.name, o.date, o.value
FROM 'hf://datasets/Rekin226/aquascope-gauges/obs/discharge/uk_ea.parquet' o
JOIN 'hf://datasets/Rekin226/aquascope-gauges/stations.parquet' s USING (station_id)
WHERE s.name ILIKE '%thames%';
```

The Explorer and `aquascope.explore.fetch_series` read a station's file first
(one HTTPS GET, no agency load) and only fall back to the agency when the
archive has no file yet. `fetch_series(..., variable="water_level")` (and the
`variable` argument of the MCP `analyze_station` / `get_timeseries` tools)
picks one variable at stations that have several.

Run it yourself: `aquascope harvest obs --out archive --source uk_ea --variable groundwater_level --max-stations 50`
(add `--sync-from Rekin226/aquascope-gauges` for an incremental run and
`--publish` to upload), then `aquascope harvest bundles --out archive` to roll
the folders into Parquet.

## Catchments: BasinATLAS in the Archive (`basins/`)

Gauges are points; hydrology happens in catchments. The Archive carries the
level-12 sub-basins of HydroATLAS v1.0 / BasinATLAS (Linke et al. 2019,
CC BY 4.0: about a million polygons of ~130 km² with routing and some 280
attributes each) so any point on land can be placed in its catchment and the
catchment described without a GIS:

| file | contents |
| --- | --- |
| `basins/lev12.fgb` | simplified sub-basin polygons as FlatGeobuf with a spatial index: a point-in-polygon lookup over HTTPS reads a few kilobytes |
| `basins/lev12_topology.parquet` | `hybas_id, next_down, next_sink, main_bas, sub_area, up_area, pfaf_id, endo, coast, order, lat, lon` |
| `basins/lev12_attributes.parquet` | every BasinATLAS attribute per sub-basin, including the upstream-aggregated `*_u*` fields, sorted by `hybas_id` |
| `basins/lev12.pmtiles`, `basins/lev06.pmtiles` | vector tiles for the Explorer |

```bash
pip install "aquascope[basins]"
aquascope basins at 48.85 2.35            # the Seine at Paris: sub-basin, upstream area, climate, land cover, soils, dams
aquascope basins at 25.04 121.56 --local  # only the level-12 sub-basin containing the point
aquascope basins upstream 2120018800      # every level-12 sub-basin draining to this one
```

```python
from aquascope.archive import basins
res = basins.describe_catchment(48.85, 2.35)      # dict: sub_basin, upstream, attributes, licence, methods
topo = basins.Topology(basins.load_topology())     # upstream_ids / downstream_ids over the whole graph
```

The MCP tool `describe_catchment(lat, lon)` and the analyst expose the same
function, and the Explorer shows the card and highlights the upstream
sub-basins for any station or clicked point.

### Similar gauged basins (prediction in ungauged basins, the practical half)

The weekly harvest also spatially joins every catalog station to its
sub-basin and publishes `basins/station_catchments.parquet` (station, sub-basin,
upstream area and the catchment attributes above). On that table,

```bash
aquascope basins similar 25.04 121.56 --k 8            # gauges whose catchments most resemble this point's
aquascope basins similar --station usgs/USGS-01646500  # ... or a station's own catchment (itself excluded)
aquascope basins similar 48.85 2.35 --method proximity --source hubeau_hydrometrie
```

ranks the gauged stations by weighted Euclidean distance in standardised
BasinATLAS attribute space (log area, elevation, slope, precipitation,
aridity, temperature, snow, forest, cropland, urban, clay, sand, population
density, regulation), by great-circle distance, or both (`combined`, the
default), and prints the per-feature deltas. That is the donor-selection
step of regionalisation (Bloeschl et al. 2013; Oudin et al. 2008); the MCP
tool `similar_basins` and the analyst use it ("find gauges like this
ungauged site, then analyse the best donors"), and the Explorer lists them
under the catchment card. `aquascope.archive.similar` is the module. Built by
`.github/workflows/basins.yml` (`aquascope basins build` plus `ogr2ogr` and
`tippecanoe`); it downloads BasinATLAS from figshare, so it runs on demand,
not weekly. Why BasinATLAS and not HydroBASINS: the HydroSHEDS core licence
forbids stand-alone redistribution, HydroATLAS is CC BY 4.0.

### Estimated flow regime (prediction in ungauged basins, the predictive half)

Donors are only useful if something is transferred from them. Every week,
after the bundles, the harvest computes the flow signatures of every gauged
station with ten or more years of archived discharge and a catchment row
(`basins/station_signatures.parquet`: mean, median, Q95 and Q05 daily flow and
mean annual maximum in mm/d over the catchment area, runoff ratio against
BasinATLAS precipitation, baseflow index, flow-duration-curve slope, high- and
low-flow frequency, zero-flow fraction, seasonality, flashiness), then predicts
every donor from the others and publishes the skill
(`basins/regionalization_skill.json`: NSE, R2 and median absolute relative
error per signature and method, leave-one-out over an even sample of the
donors). With that in place,

```bash
aquascope basins regionalize 52.29 -3.51            # an ungauged point in mid Wales
aquascope basins regionalize 46.85 1.9 --method both --json
```

describes the point's catchment, ranks the donors by physical similarity and
returns each signature as an estimate with a band: `similarity` (default) is
the inverse-distance-weighted mean over the k = 10 most similar donors
(geometric mean for the mm/d magnitudes, band = one weighted standard
deviation), `regression` a ridge fit of each signature on the standardised
attributes over all donors (band = residual spread), `both` returns the two.
The leave-one-out skill of every number comes back with it, so the answer
reads "mean flow 4.1 mm/d (1.7 to 10.2), leave-one-out NSE 0.48, median error
23 %" rather than a bare number. The same estimates are the MCP tool
`regionalize_signatures`, an analyst tool, and the "Estimated flow regime"
table under the similar-basins list in the Explorer (computed in the browser
from the same two files). `aquascope.archive.regionalize` is the module
(`station_signature`, `compute_station_signatures`, `regionalize`,
`regionalize_point`, `loo_skill`); `aquascope basins signatures` and
`aquascope basins loo` are the workflow steps. Bloeschl et al. (2013), Oudin
et al. (2008), Addor et al. (2018) for the method; the skill numbers are the
honest part, and they will move as the archive fills up (795 donors across
three agencies at the first run; NSE in log space 0.3 to 0.5 for the flow
magnitudes, lower for the shape signatures). This closes the loop opened
in #53.

## Place context mirrors (`context/`)

The flood history, dams and rain-gauge index of the [place-context layers](explorer.md#context-of-a-place)
(#520) live under `context/`. Only datasets whose licence allows redistribution are mirrored.

| path | what | licence |
| --- | --- | --- |
| `context/floods/groundsource.parquet` | Google Groundsource flood events from news: dates, the centre and box of the affected area | CC BY 4.0 (Zenodo 18647054) |
| `context/floods/microsoft.parquet` | Microsoft AI for Good Sentinel-1 floods, 2014-2024, as filtered monthly detection counts on a 0.05 degree grid (the dataset card's own false-positive filters) | MIT |
| `context/dams/gdw_barriers.parquet` | Global Dam Watch v1.0 barriers: name, river, year, height, storage, main use | CC BY 4.0 (figshare 25988293) |
| `context/ghcn/prcp_stations.csv.gz` | the NOAA GHCN-Daily stations that record precipitation, with their first and last year | CC0 |
| `context/manifest.json` | what is published, row counts, licences, and the cells present | |

Each Parquet is sorted by 2-degree cell, so DuckDB (in the browser too) or a range reader fetches a box
without the whole file, and each dataset also has one small gzipped CSV per cell (`.../cells/n50_e006.csv.gz`)
for the Explorer's Python worker, which has no Parquet reader. The manual `mirror-context` workflow builds and
publishes them; each step is also `python -m aquascope.archive.context_mirror <step>`. Global Water Watch is
not mirrored: its data licence is not confirmed.

## Issued forecasts and today's status (`forecasts/`)

Once a day the `forecast-archive` workflow (#517) looks at the Archive's discharge gauges with a live record: a
mirrored series of 10 years or more whose last value, after asking the agency for its newest days, is at most 3
days old. It writes only under `forecasts/`; the catalogue and the observations are never touched.

| path | what | licence |
| --- | --- | --- |
| `forecasts/status/latest.parquet` | each live gauge's flow today against normal: `source`, `station_id`, `value_date`, `value`, `percentile`, `class`, `n_years` (the Explorer's "Today vs normal" colouring) | derived from the mirrored observations |
| `forecasts/status/latest.json` | when the snapshot was made, the sources it covers, the count per class | |
| `forecasts/status/<date>.parquet` | the same snapshot, kept by date | |
| `forecasts/issued/<date>.parquet` | for up to 250 of those gauges with a snapped GEOGLOWS reach: the GEOGLOWS and GloFAS forecasts as issued that day, one row per gauge, model and valid day, the ensemble statistics raw and (GEOGLOWS) corrected to the gauge, the GEOGLOWS run's start date (`init_date`) and the lead day from it, the correction's hindcast KGE, the reach-to-gauge mean-flow ratio, and the gauge's own 2- to 100-year flows (`gauge_q2` to `gauge_q100`) | GEOGLOWS v2 and Open-Meteo (GloFAS v4) output, both CC BY 4.0 |
| `forecasts/reaches.parquet` | each gauge's GEOGLOWS `river_id`, the snap distance and the GloFAS cell used; IDs only, no geometry | |
| `forecasts/manifest.json` | every issue date, how many gauges, and how many were dropped and why (the daily cap, the time budget, no fresh value, no river reach) | |

The point of keeping what was issued is forecast skill at each lead time, measured later against what the gauge
then recorded; the skill the Explorer shows today is the correction's skill on the simulation. Every step is also
`python -m aquascope.archive.forecasts run|publish --out build`. A run with `max_items` set is a smoke run and
never publishes.

### Floods ahead (`forecasts/warnings/`)

Once a day the `flood-warnings` workflow (#546) reads the whole GEOGLOWS v2 global forecast
(`s3://geoglows-v2-forecasts/YYYYMMDD00.zarr`: 51 ensemble members, 15 days, 6.8 million reaches) and, for every reach
of Strahler order 5 and up and every reach an Archive gauge sits on, compares the ensemble-mean daily flow with the
reach's own 2-, 5-, 10-, 25-, 50- and 100-year flows (GEOGLOWS's retrospective return periods: a Gumbel fit to the
annual maxima of daily flow since 1940). It writes only under `forecasts/warnings/`.

| path | what | licence |
| --- | --- | --- |
| `forecasts/warnings/latest.parquet` | every reach expected to reach its 2-year flow in the 15 days: `river_id`, `lat`, `lon`, `strahler_order`, `area_km2`, `peak_cms`, `peak_date`, `rp` (the class, in years), `share` (members that agree), `q2` to `q100`, `daily` (one class per day) and `gauges` | CC BY-NC-SA 4.0 (the forecast is CC BY 4.0; the return periods' own metadata says CC BY-NC-SA 4.0) |
| `forecasts/warnings/latest.geojson` | the same reaches as slim points for the Explorer, at most 10,000, highest class first | as above |
| `forecasts/warnings/<date>.parquet` | the same table, kept by the forecast's start date | as above |
| `forecasts/warnings/manifest.json` | the issue date, counts by class, reaches checked, the method, the thresholds, what it is not, the licences and the issue history | |

Reaches whose 2-year flow is under 5 m3/s (mostly dry desert channels) are not classed, unless a gauge is on them. The cost: the forecast's
chunks hold 686 reaches each, all members and steps together, about 16 MB compressed; the bigger rivers sit together,
so order 5 and up is 3,401 of the 9,970 chunks (about 54 GB a day, streamed and dropped, nothing kept on disk). A
time budget stops reading at 4.5 hours and counts what it skipped. Model output, not an official warning. Every step
is also `python -m aquascope.archive.warnings run|publish --out build`; `--max-chunks N` is a smoke run that reads N
chunks spread over the globe, is marked so, and is never published.

## Monthly bulletins (`bulletins/`)

On the 3rd of every month the `bulletin` workflow (#523) writes last month's state of
the rivers: each gauge's monthly mean against the same month in its other years, in the
HydroSOS classes, rolled up per country and per river basin. It writes only under
`bulletins/`; the catalogue and the observations are never touched.

| path | what | licence |
| --- | --- | --- |
| `bulletins/<YYYY-MM>/bulletin.html`, `bulletin.md`, `map.png` | the bulletin, print-ready, and its map | derived from the mirrored observations; sources listed in each bulletin |
| `bulletins/<YYYY-MM>/bulletin.json` | every number, the per-gauge list included | |
| `bulletins/<YYYY-MM>/status.parquet` | one row per classed gauge: `source`, `station_id`, `month`, `value`, `n_days`, `percentile`, `class`, `n_years`, `median`, `ratio`, `record`, `country`, `basin_id` | |
| `bulletins/index.json` | every month published, newest first | |

Share-alike sources are left out of it. See [the bulletin](bulletin.md) for the method.

## Per-gauge feeds (`feeds/`)

After the forecasts, the same daily workflow (#521) keeps an Atom feed for every gauge in the status snapshot,
so a feed reader can follow a river. It reads the files the forecast step just wrote and writes only under
`feeds/`.

| path | what |
| --- | --- |
| `feeds/<source>/<station_id>.xml` | an Atom 1.0 feed: an entry when the gauge's class against normal changes, and one when the GEOGLOWS forecast corrected to the gauge passes the gauge's own 2-year flow (or a rarer one) in the next 15 days, at most once in 3 days unless a rarer flow is passed; the newest 20 entries. Characters other than letters, digits, `.`, `_` and `-` in the station id become `_` |
| `feeds/index.json` | every gauge with a feed and its path, when the feeds were made, how many entries were new |
| `feeds/state.parquet` | the entries kept per gauge, which the next run continues |

Forecast entries are model output (GEOGLOWS v2, CC BY 4.0) and say they are not flood warnings. Every step is
also `python -m aquascope.archive.feeds run|publish --out build`.

## Caravan-format export

`aquascope caravan export --source uk_ea --out caravan_gb` turns the archive
(catalog + discharge bundle + BasinATLAS + Open-Meteo forcing) into a Caravan
sub-dataset: per-gauge daily forcing and mm/d streamflow, climate indices and
HydroATLAS-style attributes. See [caravan.md](caravan.md).

## Terms

The station catalog is factual metadata (where a gauge is, what it measures)
and every row links back to the agency page. Observations will only be
mirrored for sources whose licence permits it: the registry entry's
`redistributable` flag is the gate, and it is `False` until someone has read
the terms and recorded the licence id. The current state per source is in the
dataset card and in `aquascope list-sources`.

## When a source breaks: report, then try to repair

Two workflows keep the collectors honest without a human watching every
Monday:

- **Report** (`.github/scripts/harvest_issues.py`, in the harvest run): one
  `collector-health` issue per failing source with the error, a deterministic
  diagnosis (404, 429, TLS, timeout, 5xx, format), a reproduce command; the
  issue is commented while the source keeps failing and closed when it
  recovers.
- **Repair** (`.github/workflows/repair.yml` after each scheduled harvest,
  `aquascope.maintenance.repair`): for failures that can have a code cause
  (404, changed format, unclassified; never 429 / TLS / timeouts), the job
  gathers evidence (the collector's source and tests, its registry entry, the
  last commits touching it, live probes of the URLs it uses: status, content
  type, first bytes), asks a model for either `no_fix` or a minimal unified
  diff limited to the collector and its tests, applies it in the working
  tree, and verifies it: `ruff`, the collector's own tests, a live smoke call.
  Green means a branch `repair/<source>-<date>` and a pull request labelled
  `collector-health` + `automated-repair` for the maintainer to review; red
  means the patch is reverted and the health issue gets a comment with the
  model's reasoning and the rejected diff. Nothing merges by itself.

The repair job needs a model key as a repository secret (`GROQ_API_KEY`,
`OPENAI_API_KEY`, or `AQUASCOPE_LLM_API_KEY` with `AQUASCOPE_LLM_BASE_URL` /
`AQUASCOPE_LLM_MODEL` as repository variables); without one it exits quietly.
`REPAIR_TOKEN` (a fine-grained PAT with contents + pull-requests write) makes
the opened PRs run CI like any other; with the default `GITHUB_TOKEN` they
need a nudge (empty commit or close/reopen). Locally:
`python .github/scripts/harvest_repair.py archive/health.json --dry-run`.

## Run it yourself

```bash
pip install "aquascope[archive]"
aquascope harvest stations --out archive            # local files only
aquascope harvest stations --out archive --publish you/your-dataset   # needs HF_TOKEN
```

The scheduled run lives in `.github/workflows/harvest.yml` (Mondays 03:17 UTC,
or on demand from the Actions tab). It never fails because one agency is down;
`health.json` and the job summary say which one did.

## Versioned snapshots and citation

The software DOI and the dataset are separate objects. No archive DOI has been
registered by this implementation. Until one exists, cite the agencies plus the
immutable Hugging Face dataset commit, downloaded files and their content hashes.
Do not cite the software concept DOI as if it identified a data snapshot.

The proposed cadence is a **quarterly reviewed archive snapshot**; weekly harvests
continue as operational updates. Prepare a deposit from a directory downloaded
at one exact Hugging Face commit:

```bash
python -m aquascope.archive.snapshot archive \
  --revision REPLACE_WITH_40_CHARACTER_HUB_COMMIT \
  --date 2026-09-23 --out archive-deposit
```

The command packages the GeoParquet catalog, health, manifest and declared
per-source/variable observation bundles. Its metadata lists file hashes,
source-specific licences and agency credits, and separates mirrored observations
from catalog-only sources. It validates paths and refuses unapproved mirrored
sources. It does not upload, reserve a DOI or claim publication.

Before deposition, verify that the supplied directory actually came from the
stated commit, inspect its metadata and retain each source's terms. The maintainer
must create the dataset deposit under their Zenodo account. After publication,
record the real dataset concept and version DOIs in CITATION.cff, the Explorer
citation dialog and the dataset card, and test the links. This final registration
and wiring remains the acceptance gate of [issue 444](https://github.com/Rekin226/aquascope/issues/444).
