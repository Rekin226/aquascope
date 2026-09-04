// Explorer configuration. Everything is a public URL; nothing here is secret.
export const CONFIG = {
  // The Archive (#188): station catalog published by `aquascope harvest stations`.
  stationsParquet: "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet",
  stationsGeoJSON: "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.geojson",
  // Catchments: BasinATLAS (HydroATLAS v1.0, CC BY 4.0) level-12 sub-basins published by basins.yml.
  basinsBase: "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/basins/",
  // DuckDB-WASM reads the parquet in place (HF serves range requests with CORS).
  duckdbModule: "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.32.0/+esm",
  // Pyodide runtime for the analysis worker.
  pyodideIndexURL: "https://cdn.jsdelivr.net/pyodide/v0.28.2/full/",
  // Written by the deploy step next to index.html: {"wheel": "aquascope-X.Y.Z-py3-none-any.whl"}.
  wheelsJson: "./wheels.json",
  // Cap on how far back to ask each agency, in years. null asks for the full
  // record, from the catalog's first date for the station (#270).
  years: null,
  // Cache-busting token, replaced at deploy time (git sha).
  build: "__BUILD__",
};
