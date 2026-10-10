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

The full record is fetched by default; the **Period** control on the record card
cuts it to the last 40 or 20 years. Every result has a permalink
(`#s=<source>/<station_id>`, plus `&yr=40` or `&yr=20` for a shorter period), a
CSV download, an **Export for…** menu that writes the record as inputs for HEC-HMS, HEC-RAS,
HEC-SSP, SWMM, MODFLOW 6, Delft-FEWS or Raven ([engineering exports](engineering_exports.md)),
and a link to the agency page. Data licence and attribution are shown per source.

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
surface temperature, GRACE water storage anomaly, ESA WorldCover land cover, and
JRC Global Surface Water (how often each 30 m pixel was water from 1984 to 2024).
The time-driven ones follow one date, set in the time bar (below).

## Time on the map

The **time bar** sits at the bottom of the map whenever a dated layer is on (the
clock button under the projection button opens it at any time). Every dated
layer follows its one date: VIIRS true colour, IMERG rain, SMAP soil moisture,
MODIS snow and land temperature, GRACE water storage.

- **‹ ›** step back and forward, by a day, a week or a month.
- **Play** walks a range and loops. Without a range it plays the twelve steps up
  to the date.
- **Click a day on a chart** (a hydrograph, a GR4J run, a comparison) and the
  map jumps to that day, with a short "Map set to" note. If no dated layer is
  on, rain comes on so the jump shows. Annual-maximum markers do not move the
  map, because they are drawn at 1 July rather than on the day of the peak.
- Behind **⋯**: the step, the range, **Compare** and **Make a GIF**.
  **Compare** lays a second map over the first, cut by a handle you drag: the
  right side shows another date, or one other dated layer. The gauges stay on
  the left, where they can still be clicked. **Make a GIF** plays the range
  (up to 40 frames, 640 px wide), waits for each day's tiles, stamps the date
  and the NASA credit on every frame and downloads the file. It is all made in
  the browser, with [gifenc](https://github.com/mattdesl/gifenc) (MIT).

Each layer knows its first and last day (GRACE in GIBS stops in July 2022, SMAP
starts in March 2015), and the bar says so when the date is outside one. GRACE
also knows its missing months (the GRACE to GRACE-FO gap) and its images that
start mid-month, so each month asks for the image GIBS really has.
`aquascope layers list --live` shows the exact intervals and gaps from GIBS, and
`aquascope layers frames LAYER --start --end --step` the dates and tile URLs of
a time-lapse (the MCP tools `dated_layers` and `layer_frames` are the same
functions).

The date, the step, the range and the compare date go in the link
(`d=2024-05-01&ts=week&r=2024-01-01..2024-06-30&cmp=2023-05-01`), so a
time-lapse view can be shared. Other features can follow the same date: it is
`state.date`, changed only through `setTime()` in `core.js`, with `onTime()` to
subscribe.

**The gauges themselves** carry their agency as a shape as well as a colour: a
circle for USGS, a triangle for the Environment Agency, a square for Hub'Eau, a
diamond for PEGELONLINE, a pentagon for OPW and a cross for CWA. Colour alone
does not survive colour-vision deficiency (Hub'Eau's red and the Environment
Agency's green are ΔE 4.2 apart under deuteranopia, and they are the two largest
European sources), so identity carries two channels. They can also be coloured
by record length or by how recently they last reported, with a legend, and the
shape goes on saying the agency underneath; a density heat map shows where the
world is actually measured. **Select an area** drags a box and hands back
the gauges inside it as CSV; its **Context** button lists what the box holds
(see below) and puts its flood events on the map.

The whole state (basemap, overlays, opacity, date, terrain, globe, colouring)
lives in the URL, so a view is a link.

## Floods past

![July 2019 replayed: radar sees the monsoon floods along the Ganges and the Brahmaputra](img/globe/floods-past-frame.png)

Where floods actually happened, on the globe, from the moment the page opens.
**Floods past** is on by default and shows two things, kept apart:

- **Reported in the news** (orange dots): flood events Google's
  [Groundsource](https://doi.org/10.5281/zenodo.18647054) extracted from news
  articles, from 2000 (CC BY 4.0). Each counts once, in the month it began, at
  the centre of the area it affected. A report says a flood happened; it does
  not measure it, and places with more news coverage have more reports.
- **Seen by radar** (violet squares): 20 m Sentinel-1 pixels the
  [Microsoft AI for Good Lab](https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset)
  classified as flood water, after the dataset's own false-alarm filters,
  October 2014 to September 2024 only (MIT).

Both are summed per half-degree cell (about 55 km) and month. At the world view
they are two heat maps, radar over news; from zoom 4.5 each cell is a circle
(news) and a shaded square (radar). Sizes and shades grow with the logarithm of
the count, so one very reported city does not drown out a region.

The layer follows the time bar, like every dated layer:

- with no range set it shows the **twelve months up to the map date**. The page
  opens after the last month on record, so it shows the latest twelve and says
  "latest on record";
- **playing**, or stepping by month, shows **one month at a time**, so a flood
  season replays. The ▶ in the legend sets this up for the months on screen and
  plays them;
- a **range** set behind ⋯ (and not playing) shows all its months together, up
  to 60.

**Click a cell** (from zoom 4.5) for its counts, its news events with their
dates and areas, and the months radar saw flooding there; **Open this place**
takes it to the point panel. The legend's ⓘ explains the two sources, ⌃ folds
it to one line (how it starts on a phone), and × or the rail's **Floods past**
row turns it off (`fp=0` in the link).

The data is a few small files in the Archive under `context/floods/monthly/`
(`index.json`, one gzipped JSON per month, and `grid.parquet` with every month),
rolled up from the Groundsource and Microsoft mirrors by the `mirror-context`
workflow (`python -m aquascope.archive.context_mirror floods-monthly`; the
workflow's `floods_monthly_only` input rebuilds just this from the published
mirrors). Until it is published the layer stays off the map and the rail says
so. The counts and the event lists come from
`aquascope.context.floods_past.flood_events_month`, which is also
`aquascope context --floods-past [--month YYYY-MM | --from --to] [--bbox]` and
the MCP tool `flood_events_month`.

Google Maps and Google Earth tiles are deliberately absent: their terms forbid
this use. Esri's legacy imagery answers without a token but Esri's own
documentation requires one, so it is out too.

## Context of a place

Click a point, or open a gauge, and open the **Context** tab: one line per
layer at that place, each read when the tab is opened, with the sources and
licences at the foot and one small chart (flood events per year, or the rain
gauge's yearly totals). The layers are read side by side in light workers (see
[Speed](#speed)) and each line appears as it lands.

| line | what it says | data (licence) |
| --- | --- | --- |
| Flood history | flood events in the news within 25 km, and the months Sentinel-1 radar saw flooding, 2014 to 2024 | Google Groundsource (CC BY 4.0); Microsoft AI for Good flood dataset (MIT) |
| Surface water | how often this 30 m pixel was water from 1984 to 2024, and the change since 1984-1999 | JRC Global Surface Water v1.5 (Copernicus, free and without restriction) |
| Flood depth | modelled river flood depth at the 10 to 500-year floods | JRC CEMS-GloFAS hazard maps v2.1.2, via a Source Cooperative COG mirror (CC BY 4.0) |
| Dams | dams within 50 km, nearest first, with their storage | Global Dam Watch v1.0 (CC BY 4.0) |
| Rain gauge | the nearest GHCN-Daily station with precipitation, its span and mean yearly total | NOAA NCEI GHCN-Daily (CC0) |
| Evaporation | actual evapotranspiration and interception for the latest year, 300 m | FAO WaPOR v3 |
| Soil | topsoil texture and plant-available water in the top metre, 1 km | ISRIC SoilGrids 2.0 (CC BY 4.0) |

The same lines come from `aquascope context LAT LON` and the MCP tool
`place_context`. Rasters are read a pixel at a time with HTTP range requests by
a small pure-Python Cloud-Optimized GeoTIFF reader (`aquascope.utils.cog`), so
nothing needs GDAL. Flood events and dams come from the Archive's `context/`
mirror, built by the `mirror-context` workflow; until it is published those
lines say so. Global Water Watch reservoir series are not used: their licence
is not confirmed.

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

A click on a hillside is not a river. Every point click is first snapped to the
river network (see **Rivers** below), and when no stream runs within 1 km the
card says so, rather than quoting the upstream area of the level-12 sub-basin
the point sits in: that area belongs to the river at the bottom of the slope,
which used to be reported as the point's own catchment.

Why not HydroBASINS itself: the HydroSHEDS core licence forbids distributing
the data "as a stand-alone product" and requires an end-user licence, so it
cannot be hosted on the free-tier archive; HydroATLAS is CC BY 4.0, which is
why BasinATLAS is what we mirror. MERIT-Basins is CC BY-NC.

## Rivers

A click lands on a river, not just a coordinate. The point is snapped to a
reach of the GEOGLOWS v2 river network (about 6.8 million reaches, TDX-Hydro
geometry) within 1 km: of the reaches in reach, the main channel (the highest
stream order, the nearer on a tie), since a click beside a big river is often
nearer a small stream than the river's mapped centreline. The marker moves onto
the river, and a line under the title says how far it moved, which reach it is,
and when a smaller stream was nearer (on the Jamuna: "Snapped 959 m to the main
channel (order 8); a smaller stream is 925 m away."). With no stream within 1 km it says that
instead and offers the nearest mapped reach, and when a river at least two
orders bigger lies a little further off (a braided river's water can be
kilometres from its centreline) it offers that too. A gauge takes the nearest
line, since it sits on its own river; where the reaches near it differ in size
and its catchment area is known, the reach whose upstream area matches it
(the evidence ladder's rule).

The **River** tab shows that reach's simulated daily discharge from 1940 to the
latest weekly update, analysed the way a gauge is: the hydrograph with the
annual maxima, the return-period table (GEV by L-moments and Log-Pearson III
with 90 % intervals, downloadable as CSV), the flow-duration curve and the
monthly regime with its 10th to 90th percentile band. It is a model (ERA5 runoff
routed down the network), labelled modelled everywhere, and a gauge on the same
river outranks it. The record comes from the GEOGLOWS REST API, under CC BY 4.0.

**Trace to the sea** follows the reach downstream to its outlet with the
model's own routing tables, draws the path on the map, and lists the gauges
within 2 km of it in the order the water reaches them, with the length and the
area that drains to the starting reach. Dams within 2 km of the path show as
squares on the line and in a list with their storage, main use and the km where
the water meets them (Global Dam Watch v1.0, CC BY 4.0, from the Archive's
mirror). One line names the countries the river crosses (Natural Earth 1:50m,
public domain) and another says whether dams upstream regulate the starting
reach. Until the dam mirror is published the card says so and the rest of the
trace stands. The **Rivers (GEOGLOWS)** layer in the
rail draws the whole network by stream order, read in place from the 2.4 GB
`streams.pmtiles` in the GEOGLOWS bucket. The network geometry is CC BY-SA 4.0:
shown here, never republished.

The same functions are `aquascope river snap|record|area|trace|dams` and the MCP
tools `snap_to_river`, `reach_record`, `upstream_area`, `trace_downstream` and
`upstream_dams`.

## Evidence: the models against the gauge

On a gauge with three or more years of daily discharge, the **Evidence** tab
sets the record beside the global models on the same river: the gauge drawn
bold, GEOGLOWS v2 and GloFAS thin, on one plot. A table scores each model (KGE
with r, alpha and beta, NSE, bias, and the error at the 2-, 10- and 100-year
flows) and grades it A to D, and one sentence says which fits best and where
they disagree. GEOGLOWS and GloFAS are computed in the page for that gauge (about
half a minute); NWM v3 (US) and Google GRRR come from the table the monthly CI
run publishes. The best grade also shows as a small badge next to the gauge's
dates, and **Best model skill** in the rail's gauge colouring paints every gauge
by it, with a legend. Until the first monthly run publishes the table, that
colouring is grey and the legend says why. How the grades work:
[evidence.md](evidence.md).
## Now and next

The **Now** tab on a gauge says, in one sentence, where today's flow sits against
normal for the date: its percentile against the values 7 days either side of the
same date in every other year of the record, and one of the five classes the USGS
National Water Dashboard and WMO HydroSOS use (much below normal, below, normal,
above, much above). It needs 10 years in that window and says so when there are
fewer. An Archive copy is first topped up with the agency's newest days.

Under it, the next 15 days: the GEOGLOWS v2 ensemble for the gauge's river reach
(the middle half and the full range shaded, the mean as a line), GloFAS v4 through
Open-Meteo as a dotted line, the last 30 observed days and the return-period lines.
On a discharge gauge the GEOGLOWS forecast is corrected to the gauge's own record
by flow-duration quantile mapping (one curve per calendar month), and a line under
the plot gives the skill of that correction, fitted on the first 60 % of the years
the model and the gauge share and scored on the rest ("Corrected forecast: KGE 0.47
on the 1992-2026 hindcast, raw 0.21."), then the bias and the days above the
gauge's 2-year flow it caught, raw against corrected. When the reach's simulated
mean flow is more than twice or under half the gauge's, a line says the gauge may
be on another river than that reach. That is the skill of the simulation, not of
the forecast at each lead time: the daily `forecast-archive` workflow keeps every
forecast as issued so that skill can be measured as it builds up
([details](archive.md#issued-forecasts-and-todays-status-forecasts)).

On a clicked point the tab shows the reach's simulated status (against its own
86 years) and the raw forecast. The map date moves a dotted marker across the plot.

The forecast arrives in two steps: the GEOGLOWS ensemble and its sentence first,
then (with a line saying what is still coming) GloFAS, the thresholds and the
status, which need the reach's simulated record since 1940, and on a gauge the
correction.

### Speed

Python in the browser reads one URL at a time, so a worker answers one call
after another. Besides the main worker, the Explorer starts up to three light
workers (one on a phone) without pandas or scipy, which boot in seconds and take
the calls that only read the network: the river snap of a click, the quick
forecast and the Context layers, quickest first. They run side by side and
beside the main worker, and a browser that cannot start them sends those calls
to the main worker as before. Forecasts and Context lines already read are kept
for the session.

**Today vs normal** in the gauge colouring of the layers panel colours the gauges
from the daily status snapshot, with a legend that names the sources it covers and
when it was made; gauges without a fresh record are grey. Until the first snapshot
is published the gauges keep their agency colours and the legend says so.

Both forecasts are model output under CC BY 4.0 (GEOGLOWS v2; Open-Meteo, free for
non-commercial use). The same functions are `aquascope now` and the MCP tools
`flow_status`, `flow_forecast` and `correct_to_gauge`.

## The monthly bulletin

**Bulletin** in the Tools menu opens last month's state of the rivers in a reader:
the document the monthly workflow wrote (every Archive gauge's monthly mean against
the same month in its other years, by country and river basin, with the new records
and a map), with Print or save as PDF and the Markdown beside it. **Last month's
status** in the gauge colouring colours the gauges by their class in that bulletin;
gauges it did not class are light grey. Before the first bulletin is published, both
say so and the gauges keep their agency colours. The numbers come from
`aquascope.bulletin` ([details](bulletin.md)); the page only shows them.

## Watch: since you were here

**☆ Watch** on a gauge, on a clicked point's river reach and on a drawn area keeps it
in a watch list in this browser (no account; the page still works when storage is
blocked, it just forgets on reload). On a watched gauge one line asks where to flag
the forecast: the 2-year flow by default, the 5- to 100-year flow from the gauge's own
record, or a value.

The next time the Explorer opens without a link to something else, a **Since you
were here** panel checks each watched place in turn and says, in one line each, what
changed since the last visit: new days of data and the latest value, today's class
against normal and the one before, the forecast peak in the next 15 days against the
threshold (from the daily forecast archive, else GEOGLOWS asked there and then;
modelled), and flood events in the news nearby that started since. An area says how
many of its gauges are above normal today. **Dismiss** closes it; **Watched** in the
Tools menu opens it again, and each name jumps to the place.

Every line comes from `aquascope.watch.watch_digest` in the worker, the same function
as `aquascope watch ID... --since DATE` and the MCP tool `watch_digest`. A gauge with a
live record also has **Follow (Atom)** in its ··· menu: a feed of its status changes
and forecast alerts, written daily ([feeds](archive.md#per-gauge-feeds-feeds)).

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

## Study: a complete study at a place

The drawer's second mode. **Study this place** on a gauge or on a point (or
the Study button at the top right) opens it with the site set, and the
[Studio](studio.md) crew runs in the same Pyodide worker as everything else:
you say the problem, the Consultant asks at most three questions (chips
answer them, **Just go** takes the defaults), the Scout lists what is in
reach, the Methodologist shows the plan as a card you **Approve**, **Edit**
(the arguments inline, revalidated) or **Decline**, the Analysts run it with
a gate on every step while the timeline and the figures appear, and the
Author's report lands with the key numbers, the figures, what is not
established, **Download bundle** and links for the Word, Excel, Markdown,
notebook and `study.yaml` files. The input stays open for a follow-up: a
question is answered from the workspace, a change is planned, run and
re-authored.

Keyless by default, which is a complete study; the key Ask holds is offered
on one line for the prose. An on-device model that is already there (Chrome's
built-in model, or the small model Ask loaded) joins the keyless crew: it
reads the first sentence into a brief, writes the plan at review (the
engine's validator checks it before the card shows it; the playbook's plan
stands when it fails, and the card says why) and, after the run, the prose
(four calls at most; the Critic's checks drop a sentence with a number the
results do not carry, and the line under the answer says how many). The
card says who wrote what, and nothing is ever downloaded for it.

Drop a CSV or an XLSX (turned into CSV in the worker), or attach the table
open in My data, and the Scout lists it with its QA next to the gauges.
**Stop** terminates the worker and boots it again; the plan is kept and the
next Approve rebuilds the study. A study is saved in the browser after every
reply (IndexedDB, the last five): opening Study at the same place offers
**Resume the last study**, and a `workspace.json` from a bundle dropped on
the board resumes too. The figures and the documents are made in the worker
and stay there until you ask for one; the plotting and document libraries
load once, before the first run, and the Study modules themselves load on
first use, never on a visit that runs no study. See
[studio.md](studio.md#in-the-explorer).

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
  registry), `timeline.js`, `time-ui.js`, `compare-map.js` and `gif.js` (the map
  date, the time bar, swipe compare and the GIF), `catalog.js` (DuckDB-WASM over the archive's GeoParquet, GeoJSON
  fallback), `search.js`, `shell.js` and `url.js` (the map-first shell and
  URL-as-state), the `panel-*.js` inspectors, `charts.js` (Plotly), `ask.js`
  with `showcase.js` and `local-model.js`, `studio.js` with `intake.js`,
  `studio-device.js` and `study-store.js` (Study, loaded on first use), and
  `webmcp.js`. `app.js` wires them together.
- `worker.js`: a Web Worker that loads Pyodide, numpy / scipy / pandas, and the
  aquascope wheel, then calls `aquascope.explore`; the `studio` message drives
  `aquascope.studio.Studio` for Study, loading matplotlib and the document
  libraries only when a study runs. Its ops: `start`, `say` (with a
  `proposed` brief), `approve` (with a `plan`), `follow_up`, `narrate`,
  `context` (a role's compact context with its system prompt), `check_plan`,
  `prompts`, `file`, `export` and `table` (an XLSX to CSV), each guarded on
  the engine having the method.
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
