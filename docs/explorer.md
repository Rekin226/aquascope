# The Explorer: click any gauge on Earth, nothing to install

**Live:** [rekin226-aquascope-explorer.static.hf.space](https://rekin226-aquascope-explorer.static.hf.space/)
(a copy also runs under [/explorer/app/](https://rekin226.github.io/aquascope/explorer/app/) on this docs site).

A static page, no server. It reads the station catalog from the
[Archive](archive.md) with DuckDB-WASM, shows every station on a MapLibre map,
and when you click one it fetches the observed record from the agency and runs
aquascope in your browser (Pyodide) to compute:

- the hydrograph and annual maxima,
- flood frequency: GEV by L-moments and Log-Pearson III with analytical 90 %
  confidence limits, plus an optional bootstrap GEV band (1,000 refits, on
  demand),
- the flow-duration curve with Q95 / Q10,
- a Mann-Kendall trend with Sen's slope on the annual means,
- and a "Methods and citations" panel naming exactly what was computed and the
  references, so the numbers are defensible in a report.

Every result has a permalink (`#s=<source>/<station_id>`), a CSV download, and
a link to the agency page. Data licence and attribution are shown per source.

Click anywhere that is not a gauge and you get the **hydrology of that point**
(`#p=<lat>,<lon>`): ERA5 rainfall and temperature, FAO-56 reference
evapotranspiration and the aridity class, a monthly climate chart, GloFAS
modelled discharge with an indicative return-period table (clearly labelled
as a model, not a gauge), and the nearest gauges to click next. All from
Open-Meteo, keyless.

Stations already mirrored in the [Archive](archive.md) load from it (one small
file) instead of from the agency, so they are faster and do not add load
upstream.

## My data: your own record, the same analyses

**My data** (top left) is the other half of the app. Drop a CSV or an Excel
export, paste a table, or send the gauge you are looking at. `aquascope.ingest`
works out which column is the date and which is the value, converts to SI and
shows a QA report (gaps, duplicates, flat runs, outliers, units) before anything
is computed. Then the [workbench](https://github.com/Rekin226/aquascope/blob/main/aquascope/workbench.py)
analyses appear by tab:

| tab | analyses |
| --- | --- |
| Quality | exploratory summary, quality report, preprocessing, insights, the WHO drinking-water screen |
| Hydrology | flow-duration curve, baseflow separation (Lyne-Hollick, Eckhardt, UKIH), recession, flow signatures, FAO-56 ET0 and irrigation scheduling |
| Extremes | GEV flood frequency, return periods (GEV / LP3 / Gumbel with confidence limits and the empirical points) |
| Groundwater | standardised groundwater index with drought events, water-table-fluctuation recharge, Theis drawdown |

Each has its own parameters, a chart, a JSON download, and its methods and
citations, accumulated for whatever you actually ran. **Nothing is uploaded**:
the file is read in your tab and the analysis runs there.

These are the same functions the Streamlit dashboard used to own, so
`aquascope dashboard`, the MCP server, `aquascope ask` and this page all run one
implementation rather than four that drift. The public Streamlit deployments
have been retired in favour of this.

## The shell

The map is the page. It fills the window, and the layer rail, the inspector and
the Analyst float over it as cards, so the map is the first thing you see and
opening a panel never takes width from it. The rail (the stacked-squares button
at the top left) is closed on arrival and groups its controls under **Sources**,
**Basemap**, **Relief**, **Overlays** and **Credits**; the chevron at the top
right folds the inspector away when you want the whole map. On a phone the map
keeps the screen and the inspector is a bottom sheet.

The world view is a **globe**, because the coverage is worldwide and thin, and
Mercator spends its pixels on the empty high latitudes. MapLibre eases the globe
back to Mercator by about zoom 5, so everything below the world view behaves
like an ordinary map. The button under the layers button switches projection,
and `gl=0` in a link pins a flat map.

## Map layers

The rail is a layer stack, and every layer in it is keyless and free to use.
Nothing here needs an account, and each carries its attribution and licence in
the rail, in the map credits and in an info panel.

**Basemaps**: OpenFreeMap light, dark and streets (OpenStreetMap data, ODbL);
Sentinel-2 cloudless 2016 and 2025 from EOX; EOX Terrain Light; NASA GIBS VIIRS
true colour for a chosen day; USGS imagery over the United States.

**Relief**: 3D terrain and hillshade from the AWS Terrain Tiles DEM. Hillshade
is on by default; the light basemap on its own is near-white land on pale water,
which reads as an empty page rather than as a map.

**Overlays**, each with an opacity slider and its own colour scale: GPM IMERG
precipitation rate, SMAP root-zone soil moisture, MODIS snow cover, MODIS land
surface temperature, GRACE water storage anomaly, and ESA WorldCover land cover.
The time-driven ones share a single date control, so you can walk a flood or a
snowmelt day by day.

**The gauges themselves** carry their agency as a shape as well as a colour: a
circle for USGS, a triangle for the Environment Agency, a square for Hub'Eau, a
diamond for PEGELONLINE, a pentagon for OPW and a cross for CWA. Colour alone
does not survive colour-vision deficiency (Hub'Eau's red and the Environment
Agency's green are ΔE 4.2 apart under deuteranopia, and they are the two largest
European sources), so identity carries two channels. They can also be coloured
by record length or by how recently they last reported, with a legend, and the
shape goes on saying the agency underneath; a density heat map shows where the
world is actually measured. **Select an area** drags a box and hands back
the gauges inside it as CSV.

The whole state (basemap, overlays, opacity, date, terrain, globe, colouring)
lives in the URL, so a view is a link.

Google Maps and Google Earth tiles are deliberately absent: their terms forbid
this use. Esri's legacy imagery answers without a token but Esri's own
documentation requires one, so it is out too.

## Catchments

For every USGS gauge, and for any point clicked in the United States, the
Explorer draws the upstream drainage basin on the map and shows its area:
the USGS Network Linked Data Index (NLDI) traces NHDPlus V2 catchments for
NWIS sites (`/nwissite/USGS-<id>/basin`) and, for a point, for the nearest
flowline (`/hydrolocation` then `/comid/<id>/basin`). Public domain, fetched
straight from `api.water.usgs.gov` (CORS), nothing stored on our side, and the
method and source are added to the citations panel.

Everywhere else on land, **BasinATLAS** (HydroATLAS v1.0, CC BY 4.0, from the
Archive's `basins/` files) takes over: the level-12 sub-basin containing the
station or point is found in the browser (FlatGeobuf range reads), its
upstream sub-basins are walked in DuckDB-WASM and highlighted on the map
(toggle "Basins" in the legend for the outlines), and a card shows the
upstream area and BasinATLAS's own upstream-aggregated attributes: elevation,
slope, precipitation, PET, aridity, temperature, snow, natural discharge,
forest / cropland / urban / glacier / lake / karst extent, soil texture,
population density, regulation by dams, human footprint. Citation in the
methods panel.

Under the catchment card, **Similar gauged basins** lists the gauges whose
catchments look most like the point's or the station's (standardised
BasinATLAS attributes combined with distance, from
`basins/station_catchments.parquet`), the donor list one needs at an
ungauged site; click one to open it. Below it, **Estimated flow regime** is
what those donors suggest for the point: mean, low (Q95) and high (Q05) daily
flow and mean annual maximum in mm/d, baseflow index, seasonality and
flashiness, each transferred as a similarity-weighted mean over the ten
closest donors with a band, and the archive's published leave-one-out skill
(NSE, median error) next to every number (`basins/station_signatures.parquet`
and `regionalization_skill.json`, computed weekly by the harvest; see
[archive.md](archive.md#estimated-flow-regime-prediction-in-ungauged-basins-the-predictive-half)).
Not a measurement, and it says so.

Why not HydroBASINS itself: the HydroSHEDS core licence forbids distributing
the data "as a stand-alone product" and requires an end-user licence, so it
cannot be hosted on the free-tier archive; HydroATLAS is CC BY 4.0, which is
why BasinATLAS is what we mirror. MERIT-Basins is CC BY-NC.

## Ask ✨: the Analyst in the page

The **Ask** button (top right) opens the [Analyst](analyst.md) inside the
Explorer. Type a question ("What is the 100-year flood of the Thames at
Kingston, and how sure can we be?"), pick a provider (Groq and Hugging Face
have free tiers; Anthropic, OpenAI, Mistral, OpenRouter, or any
OpenAI-compatible endpoint), paste your key, and the same `aquascope.ai_engine.analyst.ask`
that runs behind `aquascope ask` runs in the browser worker: the model picks
the tools (`find_stations` over the catalog already loaded in your tab,
`analyze_station`, `flood_frequency`, `get_timeseries`, `anywhere`),
aquascope executes them, and the answer ends with a **Data** and a **Methods
and citations** section assembled from the tool results. Every station the
tools touched becomes a chip that opens it on the map; the report can be
copied or downloaded as Markdown.

Your key travels from your tab straight to the provider you chose: the page
has no server, and the request is made by the browser worker (a plain
`urllib` client, `aquascope.ai_engine.llm_transport`, that also lets
`aquascope ask` run without the `openai` package). The key is kept in the tab
unless you tick "remember", which stores it in your browser's local storage.
The model has to support tool calling; the provider defaults do.

### Three ways to use it, and only one needs a key

**Worked examples.** The drawer opens on questions that were already answered
and recorded: the question, every tool call with its arguments, the answer, and
the checks. The prose is a recording, and the panel says so. **Run the tools
again** re-runs the deterministic half in your own browser, live, with no key,
and shows the fresh numbers beside the recorded ones. The traces are produced by
`python -m aquascope.showcase` and published weekly by
`.github/workflows/showcase.yml`, so they track the current archive.

**On your device.** If your browser has Chrome's built-in Prompt API, Ask uses
it directly. Otherwise you can choose to download a small open model (about 2 GB,
WebLLM over WebGPU, cached by the browser after the first time). Neither is
reliable at native tool calling at that size, so this tier uses a smaller loop:
the model picks one tool at a time from five by replying with JSON, and after a
few steps it writes the answer. It is labelled "on your device, reduced tool
set" rather than presented as the full Analyst.

**Your own key**, as described above, for the full tool loop.

### For an assistant already in your browser

Where the browser supports [WebMCP](https://github.com/webmachinelearning/webmcp)
(`navigator.modelContext`), the Explorer registers `find_stations`,
`analyze_station`, `anywhere`, `describe_catchment` and `show_on_map` as tools,
so an assistant in the same browser can query every gauge in the archive with
aquascope installed nowhere. It is entirely feature-detected: where the API is
absent, which is most browsers today, nothing changes. For assistants outside
the browser, `aquascope mcp` is the same tools over [MCP](mcp.md), including
`station_view`, which returns an inline hydrograph view for clients that
support the MCP Apps extension.

## What works today (Phase 0 of [#189](https://github.com/Rekin226/aquascope/issues/189))

| source | record you get | analyses |
| --- | --- | --- |
| USGS | daily mean discharge (or gage height), full record requested (from the catalog's first date) | all of the above |
| Environment Agency (England) | daily mean flow (falls back to level, rainfall, groundwater), full record requested | all of the above |
| Hub'Eau (France) | daily mean discharge (obs_elab `QmnJ`, multi-decade where computed), else last 30 days real-time | all of the above when the daily series exists |
| PEGELONLINE (Germany) | last 31 days of W / Q | hydrograph |
| Ireland OPW | last month of 15-minute levels | hydrograph |
| Taiwan CWA | daily rainfall, last 10 years (one request per year at the source, a few seconds each) | hydrograph, annual maxima, trend |

Flood frequency needs at least 10 complete years of daily flow; the page says
so when a record is shorter.

## Rainfall-runoff model in the page (GR4J)

Discharge records in m3/s with a catchment area get a **Rainfall-runoff model**
card. "Calibrate GR4J" fetches daily precipitation and FAO-56 ET0 at the gauge
from Open-Meteo (the ERA5-Land/ERA5 blend the Caravan exporter and HydroGym
use), converts the record to mm/d over the station's catchment area (agency,
else BasinATLAS), and calibrates the four GR4J parameters by differential
evolution (population 20, 40 generations, KGE on the first 65 % of the record
after a one-year warm-up) in the page: `explorer/gr4j.js` is a line-for-line
port of `aquascope.models.rainfall_runoff.GR4J`, checked against it to
round-off in the test suite, and 40 years run in about 4 ms per simulation, so
820 simulations take two seconds. You get X1 to X4, KGE / NSE / log-NSE /
PBIAS on the calibration and the validation periods, and the last six years
of observed against simulated flow. Point forcing, not catchment-averaged, so
wet mountainous catchments under-run and snow basins do badly; the numbers say
so rather than hide it. The Python model got the same treatment (the daily
loop is nine times faster, same numbers to 1e-14), which is what makes
`aquascope gym` usable.

## How it is built

`explorer/` in the repository, no build step:

- `index.html`, `style.css` and ES modules under `src/` (still no bundler):
  `map.js`, `layers.js` and `layer-ui.js` (MapLibre, the basemap and overlay
  registry), `catalog.js` (DuckDB-WASM over the archive's GeoParquet, GeoJSON
  fallback), `search.js`, `shell.js` and `url.js` (the map-first shell and
  URL-as-state), the `panel-*.js` inspectors, `charts.js` (Plotly), `ask.js`
  with `showcase.js` and `local-model.js`, and `webmcp.js`. `app.js` wires them
  together.
- `worker.js`: a Web Worker that loads Pyodide, numpy / scipy / pandas, and the
  aquascope wheel, then calls `aquascope.explore`.
- `aquascope.explore` (in the package): the Python half, the same
  `(source, station) -> answer` entry point the CLI and the MCP server use.
  It runs unchanged in CPython, which is how it is tested (`tests/test_explore.py`).
- `build.py`: assembles the site (wheel + `wheels.json` + cache-busting token).
  `.github/workflows/explorer.yml` publishes it to the Hugging Face static Space
  on every push to `main`; `docs.yml` adds it under `/explorer/app/` here.

First analysis in a session loads about 15 MB of Python once; the catalog
itself arrives in a few seconds. Sources that don't allow browser fetches
(CORS) will come through the Archive as it grows.

## Run it locally

```bash
pip install build
python explorer/build.py --out dist-explorer
cd dist-explorer && python -m http.server 8000   # then open http://localhost:8000/
```
