# Use the Archive without aquascope: R, QGIS, DuckDB, Julia

Everything the harvest publishes is plain open files on a public Hugging
Face dataset that serves HTTP range requests, so any tool that reads
Parquet, GeoJSON, FlatGeobuf or PMTiles can use it in place. No account, no
key. Base URL:

```
https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/
```

| file | what | tools |
| --- | --- | --- |
| `stations.parquet` | the station catalog, GeoParquet 1.0 (WKB point geometry) | pandas, DuckDB, R arrow, GeoPandas, QGIS, GDAL |
| `stations.geojson` | the same as GeoJSON | anything |
| `obs/<variable>/<source>/<station_id>.csv.gz` | one station's daily record (`date,value`, SI) | anything |
| `obs/<variable>/<source>.parquet` | a whole source and variable (`station_id, date, value`) | pandas, DuckDB, R arrow |
| `basins/lev12.fgb`, `basins/lev12_topology.parquet`, `basins/lev12_attributes.parquet`, `basins/station_catchments.parquet` | BasinATLAS sub-basins (CC BY 4.0), routing, attributes, station catchments | GDAL, GeoPandas, DuckDB, QGIS |
| `basins/lev12.pmtiles`, `basins/lev06.pmtiles` | sub-basin outlines as vector tiles | MapLibre, QGIS 3.32+ |

## R

```r
library(arrow)
base <- "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/"
stations <- read_parquet(paste0(base, "stations.parquet"))          # 45k rows, WKB geometry column
flow <- read_parquet(paste0(base, "obs/discharge/uk_ea.parquet"))   # station_id, date, value (m3/s)
thames <- subset(stations, grepl("thames", name, ignore.case = TRUE) & source == "uk_ea")
head(merge(flow, thames[, c("station_id", "name")], by = "station_id"))
```

or with DuckDB from R (`library(duckdb)`), the same SQL as below. `sf::st_read()`
opens `stations.parquet` and `basins/lev12.fgb` as spatial layers through GDAL:
`st_read("/vsicurl/https://.../stations.parquet")`.

## DuckDB (CLI, Python, R, anywhere)

```sql
INSTALL httpfs; LOAD httpfs;
SET VARIABLE base = 'https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/';
-- stations per source
SELECT source, count(*) FROM read_parquet(getvariable('base') || 'stations.parquet') GROUP BY 1 ORDER BY 2 DESC;
-- a whole source's daily discharge joined to names
SELECT s.name, o.date, o.value
FROM read_parquet(getvariable('base') || 'obs/discharge/hubeau_hydrometrie.parquet') o
JOIN read_parquet(getvariable('base') || 'stations.parquet') s USING (station_id)
WHERE s.name ILIKE '%seine%' AND o.date >= DATE '2020-01-01';
-- every gauge with its catchment's mean elevation and rainfall (BasinATLAS)
SELECT source, station_id, elevation_m, precipitation_mm_yr, up_area
FROM read_parquet(getvariable('base') || 'basins/station_catchments.parquet') LIMIT 10;
```

Older DuckDB versions: write the URL out instead of `getvariable`.

## QGIS

### AquaScope Gauges plugin (QGIS 3.34 LTR+)

This repository includes a searchable QGIS plugin in
`integrations/qgis/aquascope_gauges/`.

Features:

- Search the station catalog by name, river, agency, source, or station ID.
- Optionally filter results to the current map extent.
- Display up to 100 matching stations.
- Inspect station metadata and open agency or AquaScope Explorer links.
- View the catalog as a map layer styled by data source.

To install manually:

1. Copy `integrations/qgis/aquascope_gauges/` into the active QGIS
   profile's Python `plugins` directory. Keep the directory name
   `aquascope_gauges`.
2. Restart QGIS.
3. Open Plugins → Manage and Install Plugins.
4. Enable AquaScope Gauges.

The plugin uses GDAL `/vsicurl/` to access the public catalog over HTTPS.
It attempts Parquet when the GDAL Parquet driver is available and falls
back to GeoJSON when necessary. As a rough guide, the published
Parquet catalog is approximately 5 MB and GeoJSON approximately 25 MB;
actual transfer depends on GDAL range requests, caching, and server behavior.

The plugin does not intentionally create a separate local copy of the
complete catalog. Actual network transfer depends on GDAL, the data
format, and server support.

See [`integrations/qgis/README.md`](https://github.com/Rekin226/aquascope/blob/main/integrations/qgis/README.md)
for installation, usage, and troubleshooting details.

- **Stations**: Layer → Add Layer → Add Vector Layer → Protocol HTTP(S), URI
  `https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet`
  (QGIS 3.28+ with GDAL's Parquet driver; or use `stations.geojson`). Or drag
  [`integrations/qgis/aquascope_gauges.qlr`](https://github.com/Rekin226/aquascope/blob/main/integrations/qgis/aquascope_gauges.qlr)
  into the map: the catalog styled by source, plus the sub-basin outlines.
- **Sub-basins**: Layer → Add Layer → Add Vector Tile Layer → New generic
  connection, URL `https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/basins/lev12.pmtiles`
  (QGIS 3.32+ reads PMTiles over HTTP), or the FlatGeobuf as a vector layer
  through `/vsicurl/…/basins/lev12.fgb` (spatially indexed, so panning only
  fetches what is on screen).
- **A station's record**: the agency link is in the `url` attribute; the daily
  series is `…/obs/<variable>/<source>/<station_id>.csv.gz` (Add Delimited
  Text Layer accepts an HTTP URL). The Explorer at
  https://rekin226-aquascope-explorer.static.hf.space/#s=<source>/<station_id>
  computes flood frequency and trend for it in the browser.

## Julia

```julia
using Parquet2, DataFrames, HTTP
base = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/"
stations = DataFrame(Parquet2.Dataset(HTTP.get(base * "stations.parquet").body))
```

## Terms

The catalog is factual metadata with a link back to each agency; observations
are mirrored only for sources whose licence allows it (the `license` and
`redistributable` columns say which); BasinATLAS is CC BY 4.0 (Linke et al.
2019). The dataset card lists the attribution for each source.
