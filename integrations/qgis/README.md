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

### Installation

The plugin is located in `integrations/qgis/`.

For manual installation:

1. Locate your active QGIS profile's Python plugins directory.
2. Copy the `integrations/qgis/` directory into that directory.
3. Rename the copied directory to `aquascope_gauges` if necessary.
4. Confirm that `metadata.txt`, `__init__.py`, and `plugin.py` are directly
   inside the `aquascope_gauges` directory.
5. Restart QGIS.
6. Open **Plugins → Manage and Install Plugins**.
7. Find and enable **AquaScope Gauges**.

The plugin should then be available from the AquaScope Gauges plugin menu.

### Using the plugin

1. Open **Plugins → AquaScope Gauges**.
2. Enter a station name, river, agency, source, or station ID.
3. Optionally enable **Limit results to current map extent**.
4. Select **Search stations**.
5. Select a result to view its station details.
6. Use the agency or AquaScope Explorer links when available.

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
- GeoJSON fallback may transfer substantially more data than Parquet.

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

## Station observation records

A station's daily record is stored at:

`…/obs/<variable>/<source>/<station_id>.csv.gz`

QGIS users can use **Add Delimited Text Layer** with an appropriate
HTTP URL when supported by their QGIS/GDAL environment.

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
