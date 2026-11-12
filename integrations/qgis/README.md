# AquaScope in QGIS

AquaScope provides two ways to explore its hydrological station catalog in
QGIS:

1. **AquaScope Gauges plugin** — a searchable station catalog with station
   details and links to agency pages and AquaScope Explorer.
2. **`aquascope_gauges.qlr`** — a ready-to-use layer definition for users
   who prefer not to install a plugin.

Both options access the public AquaScope archive hosted on Hugging Face.

## AquaScope Gauges plugin

The plugin supports QGIS 3.34 LTR and later.

### Features

- Search stations by station name, river, agency, source, or station ID.
- Optionally limit search results to the current map extent.
- Display up to 100 matching stations.
- View station metadata, including station ID, source, agency, river,
  country, and observation period.
- Open the original agency station page when an HTTPS URL is available.
- Open the selected station in AquaScope Explorer.
- Display the station catalog on the QGIS map with source-based styling.
- Discover archived daily observation variables for the selected station.
- Retrieve daily observations and display them in a table.
- Plot daily observations in a time-series view.

### Installation

The installable plugin package is located in `integrations/qgis/aquascope_gauges/`.

For manual installation:

1. Locate your active QGIS profile's Python plugins directory.
2. Copy the `integrations/qgis/aquascope_gauges/` directory into that
   directory. Keep its name as `aquascope_gauges`.
3. Confirm that all six required files are directly inside the
   `aquascope_gauges` directory: `metadata.txt`, `__init__.py`,
   `plugin.py`, `helpers.py`, `observations.py`, and `plot.py`.
4. Restart QGIS.
5. Open **Plugins → Manage and Install Plugins**.
6. Find and enable **AquaScope Gauges**.

The plugin should then be available from the AquaScope Gauges plugin menu.

### Using the plugin

1. Open **Plugins → AquaScope Gauges**.
2. Enter a station name, river, agency, source, or station ID.
3. Optionally enable **Limit results to current map extent**.
4. Select **Search stations**.
5. Select a result to view its station details.
6. Choose an available variable under **Observation variable**.
7. Select **Fetch observations**.
8. Inspect the records in **Table** or **Time Series**.
9. Use the agency or AquaScope Explorer links when available.

The plugin displays up to 100 matching results per search.

### Remote data access

The plugin accesses the catalog using HTTPS and GDAL's `/vsicurl/`
virtual file system.

- When the installed GDAL build supports the Parquet driver, the plugin
  attempts to open `stations.parquet`.
- If Parquet is unavailable or cannot be opened, it attempts the
  `stations.geojson` fallback.
- Parquet access can use HTTP range requests, depending on GDAL and
  server support.
- As a rough guide, the published Parquet catalog is approximately 5 MB
  and the GeoJSON catalog approximately 25 MB. Actual transfer can vary
  with GDAL range requests, caching, and server behavior.

The plugin does not intentionally download and store a separate local
copy of the complete catalog. Actual network transfer depends on the
format, GDAL implementation, server behavior, and QGIS version.

### Troubleshooting

**The station catalog cannot be opened**

- Check your internet connection.
- Confirm that Hugging Face is reachable over HTTPS.
- Check that your QGIS installation includes GDAL's required vector
  drivers.
- Check the QGIS message bar for the format that was opened.

**Parquet is unavailable**

The plugin attempts to use GeoJSON as a fallback. The amount of data
transferred may be larger.

**No stations appear in the search**

- Try a shorter search term.
- Clear the map extent filter.
- Check that the station catalog layer is valid and visible in the
  QGIS project.

**Agency link is unavailable**

Some catalog records may not contain an agency URL. The plugin only
displays the agency link when a valid HTTPS URL is available.

## Zero-install option: Layer Definition File

No plugin installation is required to use the existing
`aquascope_gauges.qlr` file.

- Drag `aquascope_gauges.qlr` into QGIS 3.28+, or use
  **Layer → Add Layer → Add from Layer Definition File**.
- It adds the world station catalog, styled by source, with map tips
  containing station information and the agency link.
- It also adds BasinATLAS level-12 sub-basins from the indexed
  FlatGeobuf dataset. The sub-basins become visible when zoomed in.

The layer definition reads the remote archive rather than requiring a
separate local catalog download.

## BasinATLAS vector tiles

To use vector tiles for the sub-basins:

1. Open **Layer → Add Vector Tile Layer → New Generic Connection**.
2. Use the following URL:

`https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/basins/lev12.pmtiles`

This workflow requires QGIS 3.32+.

## Daily station observations

The AquaScope Gauges plugin supports retrieving archived daily observations
directly from the selected station.

### Fetching observations

1. Open **Plugins → AquaScope Gauges**.
2. Search for a station and select it.
3. Review its metadata in the station details panel.
4. Choose an available variable from **Observation variable**.
5. Select **Fetch observations**.
6. Open the **Table** tab to inspect daily dates and values.
7. Open the **Time Series** tab to view the plotted observations.

Available variables depend on the station's archive source. The plugin
only offers variables supported by that source. If no archived variables
are available, the selector displays an explanatory message.

### Archive format and units

Daily records are stored using this archive path:

`obs/<variable>/<source>/<station_id>.csv.gz`

The compressed CSV contains `date` and `value` columns. The plugin
retrieves the individual station archive over HTTPS and parses it using
Python's standard library; it does not require PyArrow or Matplotlib.

Published values use SI units where applicable:

| Variable | Unit |
| --- | --- |
| Discharge | m³/s |
| Water level | m |
| Groundwater level | m |
| Precipitation | mm |

Missing archives, invalid responses, and empty records are reported in
the plugin rather than silently displayed as valid observations.

### Additional station exploration

The AquaScope Explorer provides additional station-level exploration:

`https://rekin226-aquascope-explorer.static.hf.space/#s=<source>/<station_id>`

It includes flood-frequency and trend information where available.

## Data and licensing

- Catalog rows link back to their source agency.
- Mirrored observations follow the source licence recorded in the
  `license` column.
- BasinATLAS is licensed under CC BY 4.0 (Linke et al., 2019).

For additional reader and integration information, see
[`docs/readers.md`](../../docs/readers.md).
