# Changelog

All notable changes to AquaScope are documented here.

## [Unreleased]

### Added
- The showcase records incrementally. Eight questions cost roughly a free tier's entire daily token budget, so a run that exhausts it used to lose everything and start again next time. A run now skips examples that are already recorded and still fresh (`--refresh-after`, 30 days), republishes them alongside whatever it manages this time, and the job seeds itself from the open recordings branch. A half-finished set fills in over successive runs instead of resetting. (#233)
- **A social preview card, and the keyboard and screen-reader pass** (#231 follow-ups): a link to the Explorer previewed as a blank rectangle, because the page promised `twitter:card=summary_large_image` and declared no image. `explorer/make_og_image.py` draws one from the catalog the site actually serves (every gauge as a dot, which is a world map made only of places someone measures water), and `build.py` now copies binary assets so it ships. On the keyboard: opening the Ask drawer, the modal or the mobile rail moves focus into it, Escape closes it, and closing returns focus to the control that opened it; the modal traps Tab, which is what `aria-modal="true"` had been promising. `explorer/src/a11y.js` also adds one polite live region, so a card that reports "failed, retry" says it to a screen reader and not only on screen.
- **The Analyst reaches level 3** (#234): `run_python` lets it write a short snippet against aquascope, the workbench, pandas and numpy with the data on screen in scope, for the questions no fixed tool covers (imports are allow-listed and the usual escape hatches refused; the sandbox module is explicit that the platform, not the check, is the boundary). Deterministic checks now run over the answer and the results behind it (every number appears in a tool result, a return level carries its interval, a significance claim matches the p-value, units and the record are named), and unmet ones are printed under the answer as "what this answer does not establish". Every answer also comes with the steps that produced it: `aquascope ask --study study.yaml`, then `aquascope run study.yaml` re-runs them with no model in the loop and writes a report, a results file and a manifest that hashes each step so a drifting re-run is visible. That is #54's declarative runner, and `aquascope run` now takes either a study file or the old `--method`/`--file` pipeline.
- **Ask without a key** (#233): the Ask panel used to open on a credentials form, so most visitors never saw what the Analyst does. It now opens on three tiers. **Worked examples**: eight real questions (a flood curve at Kingston, Q95 on the Seine and the Loire, a Potomac trend, Taipei against London, boreholes near Cambridge, an ungauged Portuguese river, a catchment description, a flood against its donors) recorded once a week by CI with the maintainer's key and replayed in the page, with every tool call, the Data and Methods sections and the checks; "run the tools again" re-runs the deterministic half live in the reader's browser, so only the prose is a recording and the page says so. **On your device**: where the browser has Chrome's built-in Prompt API, or the reader chooses to download a small WebLLM model over WebGPU, Ask runs with no key and no server on a reduced tool set, and labels itself that way rather than pretending to be the full Analyst. **Your own key**, unchanged, for the full loop. `python -m aquascope.showcase` records the traces; `.github/workflows/showcase.yml` publishes them, and says so politely and stops when no key is configured.
- **Tools for an agent in the browser, and a view inside the chat** (#236): where the browser supports WebMCP (`navigator.modelContext`), the Explorer registers `find_stations`, `analyze_station`, `anywhere`, `describe_catchment` and `show_on_map`, so an assistant in the same browser can query the world's gauges without aquascope being installed anywhere; it is entirely feature-detected, so nothing changes where the API is absent. The MCP server gains `station_view`, which returns the ordinary `analyze_station` result plus a small HTML view (a dependency-free inline hydrograph, the headline numbers and the attribution) for clients that support the MCP Apps extension.
- **Places in the search box** (#231 follow-up): "Blois" or "Taipei" now find the town as well as the gauge, through Photon (keyless, and its terms allow autocomplete, which OSM's own Nominatim does not). Picking a place opens the climate, catchment and nearest gauges for that spot.
- **The Workbench in the Explorer** (#235): a "My data" mode where you drop a CSV or an Excel export (or paste a table, or send the gauge you are looking at) and the analyses run on it in your browser. `aquascope.ingest` works out which column is the date and which is the value, converts to SI and shows a QA report; the page then offers the workbench analyses by tab (Quality, Hydrology, Extremes, Groundwater) with their own parameter controls, charts, JSON download and accumulated methods and citations. Nothing is uploaded: the file is read in the tab.
- **`aquascope.workbench`**: the analyses behind the dashboard's pages as plain functions that take a DataFrame and return a JSON-serialisable dict, so the same code runs in the browser (Pyodide), in the MCP server, in the Analyst's tool loop, in the CLI and behind Streamlit. Sixteen analyses: exploratory summary, quality report, preprocessing, insights, the WHO drinking-water screen, flow-duration curve, baseflow separation (Lyne-Hollick, Eckhardt, UKIH), recession, GEV flood frequency, flow signatures, return periods (GEV/LP3/Gumbel with confidence limits and the empirical points), FAO-56 reference ET0 and irrigation scheduling, the standardised groundwater index with drought events, water-table-fluctuation recharge, and Theis drawdown. Each result carries its methods and citations. The three column-detection rules the dashboard had grown are one resolver (`pick_column`, with `prefer="discharge" | "level" | "value"`), and the dashboard now calls into the module instead of keeping its own copy. New MCP and Analyst tools `list_analyses` and `analyse_table(csv, analysis, params)` put all of it in reach of an assistant. (#235)
- **Layers v1 in the Explorer** (#232): the map now has a layer stack instead of one hard-coded basemap. Eight basemaps (OpenFreeMap light, dark and streets; Sentinel-2 cloudless 2016 and 2025; EOX Terrain Light; NASA GIBS VIIRS true colour for a chosen day; USGS imagery for the United States), 3D terrain, hillshade and a globe toggle from the AWS Terrain Tiles DEM, and six environmental overlays with opacity sliders and their own colour scales: GPM IMERG precipitation rate, SMAP root-zone soil moisture, MODIS snow cover, MODIS land surface temperature, GRACE water storage anomaly and ESA WorldCover land cover. The time-driven layers share one date control, so you can walk a flood or a snowmelt day by day. The gauges themselves can be coloured by agency, by record length or by how recently they last reported, with a legend, and a density heat map shows where the world is actually measured. "Select an area" drags a box and gives the gauges inside it as CSV. Every layer carries its attribution and licence in the rail, in the map credits and in an info panel; nothing needs a key, and the whole state (basemap, overlays, opacity, date, terrain, globe, colouring) is in the URL. Google Maps and Earth tiles are deliberately absent (their terms forbid this use), as is Esri's legacy imagery (it answers without a token, but Esri's docs require one).
- **The Explorer becomes an app**: a three-pane shell (sources and overlays on the left, the map, a tabbed inspector on the right) with the Analyst in a drawer beside the inspector instead of replacing it, so the gauge you are reading stays on screen while it works, and the question carries what you are looking at (station or point, map view, hidden sources) as a visible context line you can switch off. The station page is now Overview · Floods · Flows · Model · Catchment · Similar · Methods rather than one long scroll, and a tab that cannot be filled says why ("GR4J needs four or more years of daily discharge in m³/s") instead of quietly disappearing. Every card reports loading, "not available here" or "failed, retry" in place of a silent `console.info`, the Pyodide download has a progress bar wherever you are in the page, and GR4J, the bootstrap CI and Ask can be stopped. **The URL is the state**: selection, active tab, map view, source filter and the basins overlay all live in the hash, Back and Forward work, and "Copy link" gives the canonical Space URL rather than the iframe it happens to run in. Search is accent-folded, ranked and keyboard-navigable ("rhone" finds "Le Rhône à Anthon"), and a hit whose source is filtered out un-hides it instead of doing nothing. Every figure keeps its PNG button, every table has a CSV button, GR4J exports its simulation, and a **Cite** button finally shows the BibTeX and the concept DOI next to the methods used on the page. On a phone the sources and overlays slide over the map instead of vanishing, and the panel takes the rest. `explorer/app.js` is split into ES modules under `explorer/src/` (still no bundler, still a static site) and `explorer/build.py` copies the tree. (#231)
- Crop coefficients for **sorghum**, **groundnut** and **sugar beet** in `aquascope.agri.crop_water`: single-crop `KC_TABLE`, basal `KCB_TABLE` and `DEFAULT_STAGE_LENGTHS` entries, taken from FAO-56 Tables 12 and 17.

### Changed
- USGS gage height observations (`00065`) now normalise to `WaterLevelReading` in metres with `unit="m"` rather than falling through to generic `WaterQualitySample` in feet, preserving significant figures and aligning with all other hydrometric collectors. (#240)
- The public Streamlit deployments are retired: the stlite Space is now a redirect to the Explorer (it had been serving a v0.8.1 wheel), and the Docker Space files are removed (Hugging Face has not hosted them on a free account since 2026). `aquascope dashboard` stays in the package for local use and now runs the same `aquascope.workbench` code as everything else. (#235)
- **One LLM provider registry** (`aquascope/ai_engine/providers.py`): the analyst, the dashboard recommender and the Explorer read the same list instead of keeping three copies that drifted (which is why one retired Groq model broke all of them at once). The page gets it as `explorer/providers.json`, regenerated by `python -m aquascope.ai_engine.providers` and checked by a test; each provider now also carries its free-tier note and signup link, shown under the picker.
- `CITATION.cff` lists the v0.12.0 version DOI `10.5281/zenodo.21995649` (concept DOI unchanged).

### Fixed
- A second pass over the Analyst's checks, after reading all eight recorded answers. Four more ways a correct answer was marked down: a station named only in the search result that found it counted as unnamed, because the analysis payload carries the id and no name; `mm yr-1` and `km2` were read as claims of -1 and 2; a coordinate the question gave and the tool accepted was called invented; and "no significant trend in low flow" was called a contradiction of a test that ran on annual means. Identifiers are now found at any depth, unit exponents are not numbers, a tool's arguments count as grounding, conventional thresholds like 0.05 are not measurements, and a significance claim is only weighed against the series the test actually ran on. The coordinate handling is stricter than before, not looser: "68.6 W" is checked as -68.6, so a hemisphere error is caught where previously nothing was. (#233, #234)
- A question was lost whenever the model's own tool call would not parse as JSON. Groq rejects the whole request with `400 tool_use_failed` when that happens, and it happened twice running on one showcase question: the model wrote raw Python where the JSON arguments object belongs. The model wrote it, so it is asked to write it again, with a short note saying the arguments must be one JSON object and how to escape code inside it. Two attempts, then the failure surfaces: a model that cannot format a call will not learn on the third. (#233, #234)
- The context budget from the previous fix under-read its own requests: it measured the text of each message and ignored the `tool_calls` the model writes, which for a `run_python` call is a whole snippet, so a conversation budgeted at 6,000 tokens arrived as 9,300 and was refused. The whole message payload is measured now. And rather than guess a window we cannot see, a `413` is treated as the provider saying so: the budget halves and the request goes again, down to a floor where a 413 must mean something else. (#233, #234)
- Half the showcase questions failed with `413 Request too large ... Limit 8000, Requested 12073`. Not a rate limit, so retrying could never help: the accumulated tool results were larger than the whole per-minute window. A per-result cap of 14,000 characters is no use when three results together exceed the budget, so the loop now fits the whole conversation to `MAX_CONTEXT_CHARS` before each request, trimming the oldest tool results first and leaving the newest, the question and the system prompt untouched. Each trim says how much it removed. (#233, #234)
- The Analyst's checks failed on correct answers, which is worse than having no checks, because the unmet ones are printed to the reader as "what this answer does not establish". The first live recording scored 2 of 5 on an answer that was right in every particular. The cause was typography: a good model writes `m³ s⁻¹`, a non-breaking hyphen inside a station id, and `14 555` grouped with a narrow space, and the checks compared all of it against plain ASCII. The answer is now folded (NFKC, dashes to ASCII, grouped digits joined) before anything is compared, a unit is recognised however it is spelled and is no longer read as a claim of its own digits, and dates and derived percentages are not treated as numbers needing a source. A fabricated number, a missing interval, a mismatched significance claim, a missing unit and an unnamed record are all still caught, each with a test. (#233, #234)
- The showcase job installed `.[dev]`, but `describe_catchment` and `regionalize_signatures` read BasinATLAS through pyogrio, so two questions answered over a failed tool. It installs `.[dev,basins]` now, and an entry whose tools all failed is no longer published: a model will answer regardless, and an answer with nothing behind it is the last thing to show as a worked example. (#233)
- `showcase.yml` broke three times around the same seam, and the workflow tests now cover it properly. A pull-request body written inline in a `run: |` block put its paragraphs at column 0, which ends the block scalar, and GitHub answered 422 on dispatch; PyYAML had not caught it, because a paragraph containing a colon parses as a valid top-level key. Rewriting it as a heredoc then failed on the runner, because YAML strips the block indent and the terminator, sitting inside an `else`, arrived indented, so bash reported "here-document delimited by end-of-file". The body is built with `printf` now, which has no column requirements at all. Three guards: every workflow parses, its top-level keys are only the eight GitHub defines, and every `run:` script passes `bash -n` with GitHub's `${{ }}` expressions stubbed out. Each was checked against the broken form before being kept. (#233)
- A free tier's per-minute limit ended a run instead of pausing it. The LLM transport now retries 429 and 5xx, waiting exactly as long as the provider asked ("Please try again in 14.5725s"), backing off 1, 2, 4, 8 seconds when it says nothing, capped at 30 and bounded at four retries; 401 and 4xx are not retried, because they will not improve with time. The showcase recorder also pauses between questions (`--pause`, 25 s by default). The first live recording got 1 of 8: one tool-calling question spends most of a minute's token budget on Groq's free tier, and nothing waited for the window to refill. (#233)
- `showcase.yml` pushed its recordings straight to `main`, which branch protection rejects. It opens a pull request on `showcase/recordings` instead, labelled `no-changelog`. That is also the better shape: the recorded prose is written by a model and published under the project's name, so it should be read before it goes live. (#233)
- `aquascope.showcase.diagnose` reported a rejected key (401) as a permissions problem and pointed at Hugging Face's Inference Providers setting, which is the fix for 403 and sends you looking in the wrong place. The two are separate now: 401 says the value itself is wrong, names the shape of a Groq key and shows how to pipe it in so a prompt cannot truncate it. (#233)
- Clicking **Ask** in the first moments after the Explorer loaded did nothing at all, and said nothing about why: `initAsk()` opened with `await loadProviders()` (a fetch of `providers.json`) and only attached the button's click listener afterwards, so the app's headline control was inert until that response came back. Everything that does not need the provider list is now wired before the first await, the list fills in last, and running before it arrives says so instead of throwing on an undefined provider. (#231)
- The docs site never showed the Explorer's documentation page: `docs.yml` copied the app itself into `site/explorer/`, on top of the `explorer.md` that mkdocs had just rendered there, so the nav entry "The Explorer (click any gauge)" opened the app and the page was unreachable. The copy now lives under `/explorer/app/`. (#231)
- `python -m aquascope.showcase` reported nothing but "recorded 0/8" and a wall of identical tracebacks when the key it found was not allowed to call a model (a Hugging Face write token scoped to repositories cannot call Inference Providers). It now says which of those it is and what to set instead, and the workflow warns when HF_TOKEN is the only key available. (#233)
- `.github/workflows/showcase.yml` was not valid YAML: the multi-line git commit message sat at column 0, which ends the `run: |` block scalar and takes the rest of the file with it. GitHub does not fail loudly on this, it simply registers no triggers, so the weekly recording never fired and `workflow run showcase.yml` answered "this workflow has no workflow_dispatch trigger". The message is now two `-m` flags, and a new test parses every workflow file and asserts each one still declares triggers and jobs. (#233)
- The Explorer's worker fetched the aquascope wheel by a URL whose filename does not change between deploys, so a returning browser could run a cached, older wheel against a newer page (which is how a missing module first showed up). The worker now fetches the wheel with `cache: "reload"` and installs it from Pyodide's filesystem.
- `aquascope.models` no longer imports the scikit-learn ensembles at package import time (PEP 562 lazy attributes), so `from aquascope.models.rainfall_runoff import GR4J`, `aquascope.gym` and `pip install "aquascope[gym]"` work on a bare install; the ensembles still import on first use with the `ml` extra.
- Groq retired `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on 2026-08-16, so `aquascope ask --provider groq`, the Explorer's Ask panel and the dashboard recommender failed with a model-not-found error on the default. The Groq defaults are now `openai/gpt-oss-120b` (analyst, Explorer) and `openai/gpt-oss-120b` / `openai/gpt-oss-20b` (recommender picker); an explicit `--model` still wins.
- Explorer: the "Monthly climate" chart on the click-anywhere card drew eight bars instead of twelve. Plotly merges categorical x values with the same label, and the one-letter month labels repeated (M, J, A), so May, June, July and August were folded into March, January and April. Months are now labelled Jan..Dec.
- USGS keyed path: `stateCd` / `countyCd` are translated to the ANSI codes the OGC API expects (`MD` to `24`, `24033` to `033` with the implied state `24` if `state_code` is not in the parameter list). Comma-separated `stateCd` / `countyCd` / `huc` values trigger a warning since the OGC API takes single values for these fields - we choose to pass the first value of the comma-separated list as the parameter in these instances rather than returning nothing. (#160)
- Explorer: `fmt(x, digits)` ignored `digits` below 10, so catchment areas and populations printed three decimals ("303.412 km²"); the trend sentence read "a increasing trend"; and a slow network was reported as "WebGL is off" (WebGL is now tested directly, and a slow style load says so). (#231)

## [0.12.0] - 2026-08-18

The "on top of the archive" release. With the world's public gauges, their
observations and their BasinATLAS catchments in one place, aquascope now does
what a library alone could not: it says what flow to expect where there is no
gauge (and how much to trust that), finds the gauged basins that look like
yours, exports Caravan-format datasets, calibrates a rainfall-runoff model in
the browser in seconds, evaluates hydrologic agents on real basins, repairs
its own collectors, and reads from R, QGIS and DuckDB without any of it
installed. Same day as 0.11.0 on purpose: 0.11.0 was the platform, 0.12.0 is
the first things built on it.

### Added
- **GR4J in the Explorer, and a 9x faster GR4J** (`explorer/gr4j.js`, "Rainfall-runoff model" card): discharge stations with a catchment area get a "Calibrate GR4J" button; the page fetches Open-Meteo precipitation and FAO-56 ET0 at the gauge, converts the record to mm/d over the station's area, calibrates X1..X4 by differential evolution (KGE, first 65 % of the record after a one-year warm-up) in the browser in about two seconds for 40 years, and shows the parameters, KGE/NSE/log-NSE/PBIAS on calibration and validation, and observed vs simulated flow. The JS model is checked against the Python one to round-off (`tests/test_models/test_gr4j_js.py`, runs when node is present). `GR4J.simulate` in Python now uses a plain-float production loop, one convolution per unit hydrograph and a plain-float routing loop instead of two `np.roll` per day: same numbers to 1e-14, about nine times faster (12 years: 30 ms -> 3 ms), which is what makes calibration and HydroGym episodes quick. (#189 Phase 3, first item)
- **HydroGym Phase 0** (`aquascope.gym`, `pip install aquascope[gym]`, `aquascope gym basins|run|leaderboard`, `docs/gym.md`, `notebooks/08_hydrogym_phase0.ipynb`): a gym-style evaluation environment for hydrologic agents. `CalibrationEnv` wraps GR4J calibration on one basin as an episode (action = X1..X4 or the unit cube, reward = NSE / KGE / log-NSE on the calibration period after warm-up, NSE/KGE/log-NSE/PBIAS on calibration and validation in `info`, a 16-number observation plus the raw daily frame), passes gymnasium's `check_env` and works without gymnasium; `synthetic_basin` (GR4J truth + noise, offline) and `load_basin` (any Archive station with a catchment area: discharge bundle in mm/d over the agency or BasinATLAS area, Open-Meteo precipitation and FAO-56 ET0 at the gauge, cached), `suggest_basins` from the signatures table (long, perennial, low-snow records); baselines `random_search`, `nelder_mead` (env-only) and `differential_evolution` (free simulator, each generation one step) with `run_leaderboard`. (#175, Phase 0)
- **Estimated flow regime for ungauged points** (`aquascope.archive.regionalize`, `aquascope basins regionalize LAT LON`, MCP + analyst tool `regionalize_signatures`, Explorer "Estimated flow regime" table): the weekly harvest now computes the flow signatures of every gauged station with 10+ years of archived discharge and a catchment area (`basins/station_signatures.parquet`: mean, median, Q95, Q05 and mean annual maximum daily flow in mm/d, runoff ratio, baseflow index, FDC slope, high/low-flow frequency, zero-flow fraction, seasonality, flashiness) and predicts every donor from the others (`basins/regionalization_skill.json`: NSE, R2, median absolute relative error per signature and method, leave-one-out). Any point then gets each signature transferred from the k most similar donors (inverse-distance weights, geometric mean for magnitudes, band = one weighted standard deviation) or by ridge regression on the standardised catchment attributes over all donors, with the leave-one-out skill next to every number. `aquascope basins signatures` and `aquascope basins loo` are the workflow steps. Bloeschl et al. 2013, Oudin et al. 2008, Addor et al. 2018. Closes the predictive half of #53.
- **Catchment areas in the station catalog**: USGS (`drainage_area` from monitoring-locations), UK EA (`catchmentArea`) and Hub'Eau (`surface_bv`, one referentiel/sites join) `stations()` now carry `extra.catchment_area_km2`. `basins/station_catchments.parquet` gains `area_km2` / `area_source` (agency area, else BasinATLAS `up_area`) and `attribute_scope`: a gauge that drains only a corner of its level-12 sub-basin (agency area under half the sub-basin's upstream area) is described by the sub-basin's own attributes rather than the big river's; the similar-basins search uses `area_km2` and a station's own row as the target; the Caravan exporter reads the catalog area before calling the agency. `harvest.yml` gets a `skip_obs` input for catalog-only reruns.
- **Readers without aquascope** (`docs/readers.md`, `integrations/qgis/aquascope_gauges.qlr`): how to use the Archive from R (arrow, sf), DuckDB, QGIS (GeoParquet, FlatGeobuf and PMTiles over HTTP; a drag-and-drop layer definition styled by source with agency links) and Julia.
- **Self-healing harvest** (`aquascope.maintenance.repair`, `.github/scripts/harvest_repair.py`, `.github/workflows/repair.yml`): after each scheduled harvest, failing sources with a possible code cause (404, changed format, unclassified) get an automated repair attempt: evidence (collector source and tests, registry entry, recent commits, live probes of the URLs it uses), one model call that must answer `no_fix` or a minimal unified diff limited to the collector and its tests, then `git apply --check`, `ruff`, the collector's tests and a live smoke call; a verified patch becomes a `repair/<source>-<date>` branch and a pull request labelled `collector-health` + `automated-repair` (never merged automatically), anything else a comment on the health issue with the reasoning and the rejected diff. Runs only when an LLM key secret is set. (direction review, "self-healing collectors")
- **Similar gauged basins** (`aquascope.archive.similar`, `aquascope basins similar LAT LON | --station SOURCE/ID`, MCP `similar_basins`, Explorer): the weekly harvest joins every catalog station to its BasinATLAS sub-basin and publishes `basins/station_catchments.parquet`; on it, any point or station gets the gauged basins whose catchments resemble it most (weighted distance in standardised BasinATLAS attribute space, great-circle distance, or both), with per-feature deltas and the citation (Bloeschl et al. 2013, Oudin et al. 2008). The donor-selection half of prediction in ungauged basins (#53); the analyst is told to use it before analysing donors. `load_stations(path=...)` reads a local `stations.parquet`.
- **Caravan-format export from the Archive** (`aquascope caravan export|validate`, `aquascope.archive.caravan`): per-gauge daily forcing (Open-Meteo's ERA5-Land + ERA5 blend at the gauge point) and area-normalised streamflow in mm/d from the discharge bundles, Caravan's climate indices (a port of `caravan_utils.calculate_climate_indices`, FAO-PM variants), HydroATLAS-style attributes from BasinATLAS, `attributes_other` with area and provenance, licences and a `provenance.json`; USGS, UK EA and Hub'Eau, areas from the agency (`drainage_area`, `catchmentArea`, `surface_bv`) else BasinATLAS. The Open-Meteo collector accepts `models=` for the `/archive` endpoint. (#100)

### Changed
- README, docs and `CITATION.cff` carry the Zenodo concept DOI `10.5281/zenodo.21903143` (v0.11.0: `10.5281/zenodo.21989509`).

## [0.11.0] - 2026-08-18

The platform release. aquascope grows from a Python library into an open
archive of the world's public gauges, a zero-install Explorer on top of it,
and an analyst that answers questions from it with citations. Everything runs
on free tiers (GitHub Actions, Hugging Face datasets and static Spaces,
Open-Meteo, the agencies' own APIs).

### Added

**The Archive** (`aquascope.archive`, `aquascope harvest`, weekly
`harvest.yml`, dataset [`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges)) (#188)

- Station catalogs: `BaseCollector.stations()`, `aquascope.find_stations()` and `aquascope stations --bbox w,s,e,n --variable discharge --format geojson|csv|json` for USGS, UK EA, Hub'Eau, PEGELONLINE, Ireland OPW and Taiwan CWA. The harvest publishes `stations.parquet` (GeoParquet 1.0), `stations.geojson`, `health.json` and a regenerated dataset card; 45,919 stations on the first run. (#187)
- Daily observations, budgeted and incremental: `obs/<variable>/<source>/<station_id>.csv.gz` for USGS discharge and gage height (feet converted to metres), UK EA flow, level, rainfall and borehole groundwater levels (mAOD), Hub'Eau daily mean discharge (`obs_elab`) and Taiwan CWA rainfall, only for sources whose registry entry says `redistributable=True`. `obs/manifest.json` (version 2, keyed `source/variable`) carries the cursor; a station is refreshed after 30 days; failures never sink a run.
- Parquet bundles: `aquascope harvest bundles` rolls each folder into `obs/<variable>/<source>.parquet` (`station_id, date, value`, joinable to `stations.parquet`); `aquascope.archive.load_observations(source, variable)` reads one from the Hub.
- Catchments: the level-12 sub-basins of BasinATLAS (HydroATLAS v1.0, CC BY 4.0) published under `basins/` (indexed FlatGeobuf for point lookups, topology and attribute parquet, PMTiles) by `basins.yml`; `aquascope basins at LAT LON | upstream | build`; `describe_catchment(lat, lon)` gives the sub-basin, the upstream area and the catchment's climate, land cover, soils, population and regulation, reading only the needed parquet row groups over HTTP. New `basins` extra (pyogrio, geopandas, shapely). HydroBASINS itself was ruled out on licence grounds. (#189, #100)
- Harvest self-reporting (`.github/scripts/harvest_issues.py`): one `collector-health` issue per failing source with a deterministic diagnosis (404 / 429 / TLS / timeout / 5xx / format), commented while it keeps failing, closed when the source recovers.
- `aquascope.explore.fetch_series` reads a station's archive file first and falls back to the agency, so the Explorer and the MCP tools stop re-hitting agencies for mirrored stations.

**The Explorer** (`explorer/`, live at https://rekin226-aquascope-explorer.static.hf.space/ and under `/explorer/` on the docs site) (#189)

- A static page with no server: a MapLibre map of every station in the Archive (DuckDB-WASM reads the GeoParquet in place, GeoJSON fallback), search, permalinks. Click a station and the observed record is fetched through aquascope's own collectors in a Pyodide worker: hydrograph, annual maxima, GEV (L-moments) and Log-Pearson III return levels with 90 % confidence limits, an on-demand bootstrap GEV band, the flow-duration curve, a Mann-Kendall trend, a CSV download and a methods-and-citations panel.
- Click anywhere that is not a gauge (`aquascope.explore.anywhere`, `#p=<lat>,<lon>`): ERA5 rainfall and temperature, FAO-56 ET0 and aridity class, GloFAS modelled discharge with an indicative return-period table, and the nearest gauges, from Open-Meteo, keyless.
- Catchments: NLDI drainage basins drawn for every USGS gauge and any US point (public domain, nothing stored); BasinATLAS everywhere else on land, with the sub-basin found in the browser (FlatGeobuf range reads), the upstream sub-basins walked in DuckDB-WASM and highlighted on the map, and an attribute card ("Basins" toggle in the legend).
- Ask ✨: `aquascope.ai_engine.analyst.ask` runs inside the browser worker; bring your own key (Groq, Hugging Face, OpenAI, Mistral, OpenRouter or a custom OpenAI-compatible endpoint) and it goes from the tab straight to the provider. Tool calls stream into a log, the report renders with its Data and Methods sections, stations touched become chips that open on the map.
- `aquascope.explore` (`analyze_station`, `fetch_series`, `analyze_series`, `flood_ci`, `to_csv`, `anywhere`) is the shared `(source, station_id) -> answer` entry point behind the Explorer, the CLI and the MCP server. Deployed by `explorer.yml`.

**The Analyst and the MCP server**

- MCP server (`aquascope mcp`, `aquascope.mcp_server`, new `mcp` extra) for Claude Desktop / Claude Code / Cursor: `list_sources`, `find_stations` (the published catalog, no agency call, nearest-first or bbox), `get_timeseries`, `analyze_station`, `flood_frequency`, `describe_catchment`, `describe_methods`, `archive_health`, plus two resources; response sizes are bounded by design; works with the Python SDK 1.x and 2.x. (#113)
- `aquascope ask` (`aquascope.ai_engine.analyst`): a tool loop over the same functions for any OpenAI-compatible endpoint with tool calling; the Markdown report ends with **Data** and **Methods and citations** sections assembled from tool results, never from the model. `aquascope.ai_engine.llm_transport.UrllibChatClient`, a dependency-free client that also runs under pyodide-http, is used when the `openai` package is missing, so `ask` needs no extra.
- `aquascope ingest` (`aquascope.ingest`): any CSV/TSV/Excel/JSON export, a heuristic (or LLM-proposed, heuristic-validated) column mapping, a deterministic apply (sentinels dropped before unit conversion, duplicates, timezone), a QA report (coverage, gaps, spikes, negatives, per-year coverage), a clean `date,value` CSV in SI units plus `.qa.json` / `.qa.md`, and `analyze_series` on the result.
- Dashboard: deployment-supplied free AI (`hosted_llm_config()` reads `HF_TOKEN` or `AQUASCOPE_LLM_API_KEY` + `_BASE_URL` / `_MODEL` from the environment or `st.secrets`) offered as the default provider when present; `RecommendationResult` and `recommend_with_llm_detailed()` report which engine answered (`llm` / `rule_based`), the provider and model, and a readable error when the LLM was asked for but not used; a troubleshooting section for the recommender.

**Collectors and the registry**

- Shared source registry with coverage and terms (`aquascope/registry.py`): every collector described once with `agency`, `country`, `variables` (the `aquascope.schemas.station.VARIABLES` vocabulary), `supports_bbox`, `supports_station_lookup`, `output_model`, `license`, `redistributable` and `attribution`; the CLI, the dashboard Collect page, `aquascope.collect()` and the two Taiwan WRA groundwater collectors read from it; `tests/test_registry.py` guards drift. Registry extraction by @laishettikarthik-tech (#163), extension per #187.
- BOM (Australia) collector (`collectors/bom.py`): streamflow, water level, storage and groundwater level from the Bureau of Meteorology's Water Data Online (KiWIS), about 8,000 stations, no key. Thanks @adjenk (#181, closes #4)
- Taiwan CWA climate collector (`collectors/taiwan_cwa.py`): daily rainfall, temperature, humidity, radiation, wind and pan evaporation from the keyless CODIS archive, history back to 1960, in the new `ClimateReading` schema; the observed-forcing layer for CAMELS-TW. (#177, contributes to #100)
- Hub'Eau elaborated series (`fetch_raw(elaborated="QmnJ")`): the `/obs_elab` endpoint (multi-decade daily and monthly means and extremes) normalises into `StreamflowReading` / `WaterLevelReading`, so French stations get flood frequency.
- USGS: station names without a key (NWIS site service fallback when the OGC walk is throttled); the per-record catchment-area lookup is memoized.
- `relax_strict_tls` on `CachedHTTPClient` for Taiwan government certificate chains that Python 3.13+ rejects by default, keeping full chain and hostname verification (#169). Flow-duration-curve slope signature `aquascope.hydrology.fdc_slope` and `SignatureReport.fdc_slope` (#45). CI CHANGELOG enforcement with a `no-changelog` opt-out label (#144).

### Fixed

- USGS OGC pagination could spin forever: `next` links were re-fetched with `params={}`, which made httpx drop the cursor, so page one came back from the disk cache in a silent loop; next links are now fetched as-is and a repeated cursor stops the walk.
- USGS keyed path filters: with an API key, `station_id` / `parameter` / `stateCd` / `countyCd` / `huc` now map onto the OGC daily collection's filter parameters instead of being ignored (a per-station fetch used to crawl the national collection). (closes #160)
- USGS discharge normalises into `StreamflowReading` (ft³/s to m³/s, miles² to km², significant figures kept, catchment area fetched when missing). (#155, contributes to #97 and #104)
- USGS CLI `--days` and kwarg handling, with tests for parameter acceptance and API-key selection. (#159)
- Hub'Eau: water level and discharge normalise into `WaterLevelReading` and `StreamflowReading` (L/s to m³/s without a rounding error, catchment area from one batched `referentiel/sites` lookup); station-filtered fetches send `code_entite`, the only station parameter Hub'Eau v2 accepts (`code_station` was silently ignored and returned the whole network); pagination no longer strips the filters off the `next` link. (#164, closes #97 and #104)
- The AI recommender no longer degrades silently: a missing `openai` package, a bad key, a dead endpoint or an unparseable reply used to look like a successful call; the dashboard and CLI now say which engine produced the list and why. The Hugging Face provider pointed at a retired host (now `router.huggingface.co/v1`); the LLM output contract asked for an array while forcing an object (now `{"recommendations": [...]}` with tolerant parsing); every shipped Hugging Face model and Groq's `mixtral-8x7b-32768` were unserved and are replaced with models verified against the live catalogues.

## [0.10.0] - 2026-08-12

A correctness release first and a feature release second.

Two sign errors in the GEV L-moment estimator had been shipping since v0.4.0,
quietly returning flood quantiles off by up to a factor of two. They are fixed,
and the Potomac worked example that had been publishing a 500-year estimate 54%
below the FEMA reference is regenerated and now validated against that reference
on both estimators. If you have used `flood_analysis(method="gev_lmoments")` or
`regional_frequency_analysis`, please recheck your numbers.

Alongside that: Brazil and the UK join the collector list, bringing it to 27,
every registered collector is now reachable from the dashboard with a CI guard
to keep it that way, groundwater analysis gets its own dashboard page, and the
CLI gained shell completion.

Most of this release came from the community again. Thanks to @taran-dev4u,
@JamesBoardman27, @laishettikarthik-tech, and @adjenk.

### Added
- **Shell completion** (`aquascope completion bash|zsh|fish`): emits an argcomplete script for the chosen shell, bringing the CLI to 20 commands. Thanks @adjenk (#157, closes #28).
- **Every collector reachable from the dashboard** (`dashboard/views/collect.py`): the Collect page went from 21 sources to all 27, adding NOAA NWPS, Ireland OPW, PEGELONLINE, CAMELS-CL, UK EA, and CAMELS-BR with per-source parameter forms and region entries. Home-page counts now derive from `len(SOURCES)` instead of a hardcoded number, and a drift guard (`tests/test_cli/test_dashboard_sources.py`) fails CI if a registered collector is missing from the page, so a new source cannot merge without wiring the UI. Thanks @taran-dev4u (#145, closes #143).
- **UK Environment Agency collector** (`collectors/uk_ea.py`): real-time river level, flow, rainfall, and groundwater telemetry from the EA Hydrology API, at 15-minute or daily resolution across the England station network. Emitted as `StreamflowReading` and `WaterLevelReading`, with station, WISKI ID, bounding-box and observed-property filters. No API key required. Thanks @JamesBoardman27 (#133).
- **Groundwater dashboard page** (`dashboard/views/groundwater.py`): SGI standardised groundwater index, drought event detection, recharge estimation, and aquifer tools as a workspace page. Thanks @laishettikarthik-tech (#139, closes #125).
- **CAMELS-BR collector** (`collectors/camels_br.py`): daily observed streamflow for Brazilian catchments from the CAMELS-BR large-sample dataset on Zenodo, joined with catchment attributes (gauge name, coordinates, area). Emitted as `StreamflowReading` with `catchment_area_km2` set, so `runoff_mm_day` comes for free. AquaScope's 27th source. Thanks @taran-dev4u (#140, closes #124).
- **Flow Duration Curve (FDC) slope signature** (`aquascope.hydrology.fdc_slope`): added log-space percentile slope signature function and `fdc_slope` field on `SignatureReport`. Thanks @taran-dev4u (#148, closes #45).

### Fixed
- **`flood_analysis(method="gev_lmoments")` returned incorrect return levels** in every release from v0.4.0 through v0.9.0. Two sign errors in the GEV L-moment estimator (the shape passed to scipy, and the location term) made fitted quantiles off by roughly 0.5x to 2x depending on the tail, always understating heavy-tailed floods. Parameter recovery is now within 0.5% of truth on large synthetic samples. `fit_gev` also seeds its ML fit from L-moments and constrains the shape to a plausible range, so 40-year records no longer produce absurd return levels (previously a shape of -6.2 and a 100-year flood of 3.3e11). **If you used `gev_lmoments`, recheck your results.** Thanks @taran-dev4u (#154, closes #119).
- **`regional_frequency_analysis` growth curve carried the same two sign errors**, so regional return levels were low and got worse at longer return periods (on a homogeneous synthetic region the 100-year growth factor was 41% below the analytical value). Corrected alongside the estimator above. A residual uniform bias remains under investigation in #156.
- **Potomac flood-frequency example corrected** (`docs/examples/potomac_flood_frequency.md`): the published GEV column came from the buggy estimator and read 54% below the FEMA reference at the 500-year level. Regenerated against the live USGS record, the GEV estimates now sit within 10% of the FEMA Flood Insurance Study at every return period. The sample size label and a stale `bootstrap_ci` argument in the reproduction snippet were fixed too.
- **Runoff ratio date index alignment** (`aquascope.hydrology.runoff_ratio`): extracted standalone `runoff_ratio` function that strictly aligns precipitation and discharge dates via inner index intersection prior to computing total volume ratios. Thanks @taran-dev4u (#148).

## [0.9.0] - 2026-08-04

Four new countries, a rebuilt dashboard, and a live in-browser demo. AquaScope
now reaches 25 data sources, and the Streamlit app went from a single-file
monolith to a multipage workspace that runs client-side in a browser with the
collectors still working.

Most of this release came from the community: seven external contributors wrote
all four new collectors and most of the new analysis and plotting features.
Thank you.

### Added
- **NOAA NWPS collector** (`collectors/noaa_nwps.py`): the US National Water
  Prediction Service, covering streamflow forecasts, stream observations, crest
  history, flood impacts, low-water history, flood-category levels, and gauge
  metadata. No API key required. Available as
  `aquascope collect --source noaa_nwps`. Thanks @AB1775 (#138).
- **Ireland OPW collector** (`collectors/ireland_opw.py`): real-time river and
  lake water level from waterlevel.ie at 15-minute resolution, across hundreds
  of Office of Public Works stations. Station geometry comes from the OPW
  GeoJSON index and series from the per-station CSV exports, with column names
  resolved defensively since the endpoint serves more than one CSV shape.
  Station refs outside the 1-41000 republication range are filtered out per
  OPW's terms. No API key required. Thanks @laishettikarthik-tech
  (#130, closes #123).
- **Germany PEGELONLINE collector** (`collectors/pegelonline.py`): recent water
  level and discharge observations from federal waterway gauges, emitted as
  `WaterLevelReading` and `StreamflowReading`. Supports station UUIDs, W/Q
  selection, CLI collection, and the upstream 31-day history limit. No API key
  required. Thanks @taran-dev4u (#131).
- **CAMELS-CL collector** (`collectors/camels_cl.py`): daily observed streamflow
  for 516 Chilean catchments from the CR2 large-sample dataset, joined with
  catchment attributes (area, coordinates). AquaScope's first South American
  source. Thanks @adjenk (#116).
- **Area-normalized streamflow** (`hydrology.streamflow`): `stage_to_runoff`
  chains a rating curve straight through to mm/day, and
  `discharge_cms_to_runoff_mm_day` converts a single discharge value given a
  catchment area. `StreamflowReading` gained `catchment_area_km2`, and both
  catchment area and runoff now route into the xarray export. Thanks
  @laishettikarthik-tech (#103, closes #98).
- **Double-mass plot** (`viz.diagnostics.double_mass_plot`): the classic
  cumulative-versus-cumulative consistency check for spotting station shifts or
  instrument changes in paired records. Thanks @aobaruwa (#107).
- **Optional Plotly backend for `plot_hydrograph`** (`viz/hydro.py`): pass
  `backend="plotly"` for an interactive figure instead of matplotlib. The
  default stays matplotlib, so existing code is unaffected. Thanks @adjenk
  (#134, closes #25).
- **Canonical SPI drought classification** (`climate.indices.drought_class`):
  McKee et al. category labels with explicit boundary and missing-value
  handling. Thanks @taran-dev4u (#132).
- **Dashboard 2.0** (`aquascope/dashboard/`): the Streamlit dashboard was rebuilt
  from a 2,100-line monolith into a multipage workspace app (`st.navigation`)
  with a shared dataset flowing through every page.
  - **Smart insights layer** (`_insights.py`): whatever dataset lands in the
    workspace is auto-profiled (datetime/discharge/parameter/coordinate columns
    detected), quality-scored, quietly screened against WHO guidelines, and
    answered with one-click "suggested next steps" that navigate to the right
    analysis page.
  - **21 collectors in the UI** (`views/collect.py`): the Collect page went from
    15 sources to 21, adding GRDC, Hub'Eau, India WRIS, Taiwan data.gov.tw, and
    Taiwan WRA IoT and FHY, with per-source parameter forms, a region filter,
    CSV/JSON upload, and two demo datasets. Nested `location` objects are
    flattened to `latitude`/`longitude` so maps work out of the box. The four
    sources added later in this release (NOAA NWPS, Ireland OPW, PEGELONLINE,
    CAMELS-CL) are reachable from the Python API and the CLI but are not wired
    into the Collect page yet.
  - **Interactive Plotly charts everywhere** (`_charts.py`): time series, box
    plots, correlation heatmaps, histograms, FDCs (with direct Q50/Q95 labels),
    hydrographs with baseflow fill, SPI timelines, return-level curves with
    bootstrap CI bands, WHO exceedance bars, MapLibre station maps, and the
    FAO-56 demand/Kc/cumulative-irrigation suite, replacing static matplotlib
    PNGs. One shared template: a fixed colorblind-validated categorical order,
    a single-hue blue ramp for magnitude, a blue↔red diverging scale for
    polarity, and reserved status colors for alerts.
  - **Persistent workspace sidebar**: active dataset card with row/column
    counts, one-click CSV download, clear, and demo loading from any page.
  - `plotly>=5.18` added to the `dashboard` extra; `streamlit` floor raised
    to 1.36 (first release with `st.navigation`).
- **Hosted-demo deployment scaffolding**: root `streamlit_app.py` +
  `requirements.txt` for Streamlit Community Cloud one-click deploys, and a
  ready-to-push Hugging Face Space under `deploy/huggingface-space/`
  (closes the "hosted Streamlit demo" roadmap gap, #34).
- **Colorblind-safe palette** (`viz/styles.py`): `apply_aqua_style(palette="colorblind")`
  switches the `axes.prop_cycle` to the Okabe-Ito / Color Universal Design
  8-colour palette. The default behaviour is unchanged. Thanks @sairajkasam
  (#110).
- **Baseflow edge-case tests**: Lyne-Hollick and Eckhardt are now pinned on
  their invariants (baseflow bounded by total flow, BFI in [0, 1], steady-state
  behaviour, filter-parameter boundaries, index preservation with NaNs, and the
  empty-series case). Thanks @widjajs (#112).

### Changed
- `SPIModel` now delegates SPI calculation and classification to
  `climate.indices` instead of maintaining its own divergent gamma fit. SPI
  values and drought classes are consistent across the two entry points; if you
  compared them before, expect small numerical differences from `SPIModel`.
  Thanks @taran-dev4u (#132).
- `.streamlit/config.toml` now pins the dashboard's categorical chart colors
  (`chartCategoricalColors`) so native Streamlit charts match the Plotly theme,
  and disables usage-stats gathering.
- `apply_aqua_style()` now explicitly sets `axes.prop_cycle` to `SERIES_COLOURS`
  on every call (previously the cycle was not set, inheriting Matplotlib's
  default). Visually identical for existing code.
- Documentation: README examples, source counts, and ROADMAP issue links were
  audited and corrected against the actual code, with CI drift guards so the
  counts cannot silently go stale again (#128; CLI command count by
  @navaneethsankar07 in #129). The contributors board is now maintained by the
  all-contributors bot.

### Fixed
- **USGS keyless queries work again.** The collector falls back to the legacy
  REST API when a query runs without an API key, instead of failing. Thanks
  @taran-dev4u (#137).
- **CAMELS-CL records are no longer silently dropped** when catchment attributes
  are missing. NaN from a missed `catchment_attributes.csv` join is truthy, so
  it reached Pydantic validation and the resulting error discarded the whole
  record. NaN now maps to `None` for gauge name, coordinates, and area (#118).
- **Live collectors now work in the in-browser (WASM) demo.** `CachedHTTPClient`
  detects Pyodide/Emscripten and routes requests through `urllib` (patched to
  browser XHR by pyodide-http) instead of httpx's socket transport, which hangs
  in WebAssembly. Verified in-browser: Open-Meteo, USGS, Hub'Eau, UN SDG 6,
  AQUASTAT, Taiwan WRA. CORS-blocked sources fail fast with a clear hint, and
  the Collect page shows a browser-demo banner listing what works.

## [0.8.1] - 2026-07-17

Two new river-discharge sources, France (Hub'Eau) and GRDC, and a repair to how
the CLI advertises and reaches its collectors.

### Added
- **France (Hub'Eau)** (`collectors/france_hubeau.py`): `HubeauHydrometrieCollector`
  collects real-time river water level and discharge from Hub'Eau's Hydrométrie
  API, France's national open hydrometry service. No API key required. Available
  as `aquascope collect --source hubeau_hydrometrie`. Readings are emitted as
  `WaterQualitySample`, following the pattern already established in `usgs.py`.
- **GRDC river discharge** (`collectors/grdc.py`): `GRDCCollector` reaches global
  river discharge without the classic GRDC portal's email request-form gate. Two
  modes: `in_situ` (the curated gauge-station subset published on Zenodo) and
  `satellite` (the RSEG remote-sensing discharge extension published on DaRUS).
  Each reading is tagged with `source_type`, so downstream work such as
  Prediction in Ungauged Basins can filter gauge from satellite. Note that the
  in-situ subset is licensed CC BY-NC 4.0: non-commercial use only, attribution
  required. Available as `aquascope collect --source grdc`, with `--mode`
  selecting `in_situ` (default) or `satellite`.
- **`StreamflowReading` schema** (`schemas/water_data.py`): the canonical record
  for river discharge, carrying `discharge_cms`, `source_type`, and an optional
  `uncertainty_cms` for satellite products. `records_to_xarray()` converts it to
  a `discharge` data variable alongside the existing sample and water-level
  record types.

### Fixed
- **`aquascope list-sources` rendered 8 of 22 sources as blank placeholders.**
  The info table is keyed by `DataSource` value, but the Hub'Eau entry was keyed
  `hubeau_hydrometrie` while its enum value is `france_hubeau`, so the lookup
  never matched. GRDC, EU WFD, Japan MLIT, Korea WAMIS, and India WRIS had no
  entry at all. All now render their region, data types, and endpoint.
- **`GRDCCollector` was unreachable from the CLI.** It was registered in
  `collectors/__init__.py` and documented, but had no `collect` entry, leaving it
  importable from Python only.
- **`japan_mlit` and `korea_wamis` were unreachable from the CLI.** Both were
  already in the collector map but missing from the `--source` choices, so
  argparse rejected them as invalid.

### Notes
- River discharge is currently emitted under two schemas: GRDC uses the new
  `StreamflowReading`, while Hub'Eau and USGS use `WaterQualitySample`.
  Consolidating on `StreamflowReading` is planned and will be a breaking change
  for those collectors when it lands.
- `GRACE` and `USGS_GW` are declared in `DataSource` but have no collector yet,
  and still list without metadata.
- India WRIS is not exposed through `aquascope collect`: its collector requires
  arguments the command does not pass. It remains available from Python.

## [0.8.0] - 2026-07-10

Groundwater drought: daily Taiwan groundwater data, SGI and SPI drought
indices, and an end-to-end drought-propagation case study.

### Added
- **Worked example** (`examples/13_groundwater_drought_sgi.py`): an end-to-end
  case study, data to result, characterising Taiwan's 2020-2021 groundwater
  drought. AquaScope collects daily groundwater (`TaiwanWRAGroundwaterDailyCollector`)
  and ERA5 rainfall (`OpenMeteoCollector`) for representative aquifers, computes
  SGI and SPI, and reports each aquifer's drought-propagation timescale (7-23
  months) and 2021 drought severity (SGI down to ~-2, severe).
- **Drought indices** (`groundwater/drought.py`, `climate/indices.py`):
  `standardised_groundwater_index()` (SGI, Bloomfield & Marchant 2013: per
  calendar-month non-parametric normal-scores transform) and
  `standardized_precipitation_index()` (SPI, McKee 1993: gamma fit with zero
  handling and configurable accumulation scale), plus `drought_events()` to
  extract runs below a threshold from any standardised index. These make
  groundwater-meteorological drought-propagation analysis (SGI vs SPI lag) a
  first-class AquaScope workflow.
- **Daily Taiwan groundwater** (`collectors/taiwan_wra.py`):
  `TaiwanWRAGroundwaterDailyCollector` reaches the sub-annual (daily)
  groundwater-level series from the WRA gweb HydroInfo portal, which the
  open-data API does not expose (it tops out at annual statistics). Per-well
  records span roughly 2005-2025 (Zhuoshui/Choushui fan back to the late
  1990s). Supports zone aliases, date clipping, and `aggregate="monthly"`
  (the input to a Standardised Groundwater Index) or `"daily"`. Rate-limits
  and caches every request. This unlocks monthly SGI and SPI/SPEI
  drought-propagation analysis on AquaScope-collected data.
- **Well coordinates for daily groundwater**: `TaiwanWRAGroundwaterDailyCollector`
  gains `with_metadata=True` (default), joining the open-data 井況 well-status
  dataset to populate each reading's `location` (TWD97 → WGS84) and
  `well_depth_m`. The gweb station id matches the suffix of the open-data
  `wellidentifier` after `GW` (with a well-name fallback), making the daily
  series spatially complete.

### Changed
- `CachedHTTPClient.post_json()` now sends a JSON body and shares the retry,
  rate-limit, and body-keyed disk-cache behaviour of `get_json()` (previously
  a thin wrapper with no body, retries, or caching).

### Fixed
- `TaiwanWRAGroundwaterDailyCollector` now drops the gweb missing-data sentinel
  (`-9998`) and de-duplicates window overlaps (the portal can return data past
  the requested `endDate`, so the same well-day appeared twice with different
  values; the later window now wins). Previously these leaked into the series.

## [0.7.0] — 2026-06-26

Interoperability and uncertainty: AquaScope now composes with the scientific-Python
geo stack and reports calibrated uncertainty on model output.

### Added
- **xarray / GeoPandas interop** (`io/interop.py`): `records_to_xarray()` converts
  time-series records to an `xarray.Dataset` (dims `(time, station_id)`, per-parameter
  variables, lat/lon coords); `records_to_geodataframe()` converts point records to a
  `geopandas.GeoDataFrame` (Point geometry, EPSG:4326). Every collector also accepts
  `collect(as_xarray=...)` / `collect(as_geodataframe=...)`. New `[interop]` extra (#70).
- **GR4J quantile prediction intervals** (`models/rainfall_runoff.py`): `predict_quantiles()`
  produces calibrated uncertainty bands via a residual or parameter-ensemble method
  (heteroscedastic option); the deterministic `simulate()` path is unchanged (#77).
- **Probabilistic metrics** (`analysis/metrics.py`): `pinball_loss`, `picp`, `mpiw`,
  `crps_ensemble`, and `crps_from_quantiles` for scoring interval and ensemble
  forecasts (#76).
- **Multi-basin UQ benchmark** (`examples/12_uq_camels_benchmark.py`): GR4J quantile UQ
  across the bundled CAMELS basins with per-basin and aggregate PICP/CRPS and a
  reliability diagram. On the bundled basins the residual method reaches ~0.90
  central-interval coverage against the 0.90 nominal target (#78).
- **Documentation**: an uncertainty-quantification guide and an updated xarray/GeoPandas
  integration guide.

### Changed
- `require()` gains a `group` override so optional-dependency errors point at the
  correct extra (e.g. `[interop]`).
- JOSS paper (`paper.md`) condensed and updated to v0.7.0 with the interop and UQ work.

## [0.6.0] — 2026-06-26

### Added
- **GR4J rainfall-runoff model** (`models/rainfall_runoff.py`): conceptual daily rainfall-runoff model with auto-calibration against NSE / KGE / log-NSE objectives (#52). This is the keystone modelling feature that turns AquaScope from a data + statistics toolkit into a simulation tool.
- **Shared model-evaluation metrics** (`analysis/metrics.py`): NSE, KGE, PBIAS, RMSE, and R² in one reusable module for scoring model predictions (#60).
- **GeoJSON export** for the `collect` command's `--format` option (#64).
- **Extreme-events module** (`analysis/extreme_events.py`): frequency analysis for hydrological extremes (annual maxima/minima series, return-period estimation), with type annotations on all public functions.
- **FAO-56 dual crop coefficient** (`agri/crop_water.py`): new Kcb + Ke mode separates basal transpiration from soil evaporation for more accurate crop water demand, alongside the existing single-Kc mode (#22, #49).
- **UKIH smoothed-minima baseflow separation** (`hydrology/baseflow.py`): adds the UK Institute of Hydrology block method (`ukih`) to the existing Eckhardt and Lyne-Hollick filters, exported via the public hydrology API (#43, #48).
- **India WRIS collector** (`collectors/india_wris.py`): river water-level data from India's Water Resources Information System (#15).
- **Dashboard data sources**: AQUASTAT, EU Water Framework Directive, Japan MLIT, Korea WAMIS, and WaPOR are now selectable in the Streamlit Data Collection page (#14).
- **Dashboard analytical pages**: the Streamlit app gains an **Extreme Events** page (block-maxima frequency analysis with return-level curves and bootstrap confidence bands), an **Agricultural Water** page (FAO-56 ET0 plus the single-Kc / dual Kcb+Ke irrigation workflow), and a **Flow Signatures** analysis plus UKIH baseflow option on the Hydrology page. All new pages ship offline demo-data fallbacks so they work without API keys.
- **`penman_monteith_series`** is now re-exported from `aquascope.agri` for daily ET0 over a weather DataFrame.
- **Edge-case tests** for `SoilWaterBalance` auto-irrigation (#35) and for the new modules.

### Fixed
- **Irrigation efficiency leak** (`agri/water_balance.py`): efficiency losses no longer leak into deep percolation, which previously inflated the groundwater-recharge term (#38, #39).
- **Collector HTTP robustness**: the WQP, Japan MLIT, and Korea WAMIS collectors now route every request through the shared `CachedHTTPClient`, so they get retries, rate-limiting, and disk caching like the other collectors. The Japan MLIT and Korea WAMIS collectors previously called a non-existent `client.get()`, which was swallowed by a broad `except` and made them return empty results on every call. A new `CachedHTTPClient.get_text()` method backs the WQP CSV path.
- **Taiwan WRA water level** (`collectors/taiwan_wra.py`): readings now carry station coordinates when the feed provides them (TWD97/WGS84 lat-lon keys) instead of always setting `location=None`.
- **USGS API key** (`collectors/usgs.py`): the collector no longer hard-defaults to the rate-limited `DEMO_KEY`. It reads `api_key=...` or the `USGS_API_KEY` environment variable, and warns before falling back to `DEMO_KEY`.

### Changed
- **Governance and contributor onboarding**: added `MAINTAINERS.md` with area owners, `.github/CODEOWNERS`, all-contributors recognition in `CONTRIBUTORS.md`, a contributor ladder in `CONTRIBUTING.md`, and a "Major features" plus "Good first issues" section in `ROADMAP.md`.
- **Test coverage gate** raised from 60% to 70% (`pyproject.toml`); added the first tests for `utils/http_client.py`, plus config tests for the USGS and Taiwan WRA collectors.
- **Documentation accuracy**: data-source count synced to 19 across README, docs, dashboard, and citation metadata; clarified that aggregate/gridded sources (AQUASTAT, SDG 6, WaPOR) use purpose-built record types rather than the unified `water_data` schema.
- **Software citation**: added `CITATION.cff`; bumped version to 0.6.0.

## [0.5.0] — 2026-06-05

### Added
- **Multi-provider LLM support** (`ai_engine/recommender.py`) — AI recommender now supports **HuggingFace Inference API** (free), **Groq** (free tier), **Ollama** (local), and OpenAI. `PROVIDER_BASE_URLS` and `PROVIDER_MODELS` constants are exported for dashboard consumption. JSON-object response mode enabled only where supported.
- **Dashboard LLM provider picker** — new `_render_llm_config()` UI lets users switch providers with free-tier links (HF + Groq) directly in the Streamlit dashboard.
- **USGS region filter** (`collectors/usgs.py`) — new `bbox` and `max_items` parameters cap paginated requests to a geographic bounding box and total record count. Dashboard exposes 5 preset US region filters (Northeast, Southeast, Midwest, Pacific Northwest, Southwest) plus custom bbox input.
- **SDG6 country picker** — dashboard Data Collection page replaces free-text ISO3 input with a 50+ country dropdown for the UN SDG 6 source.
- **WQP state picker** — US Water Quality Portal source now has a full US state dropdown in the dashboard.
- **Taiwan Civil IoT date filtering** (`collectors/taiwan_civil_iot.py`) — `start_date` / `end_date` parameters build OData `phenomenonTime` filter clauses automatically.
- **Dashboard source hints** — Taiwan WRA (level + reservoir) sources now show informational banners clarifying snapshot-only APIs; Taiwan MOENV exposes a record-count slider.

### Fixed
- **GEMStat collector** (`collectors/gemstat.py`) — completely rewritten: now downloads, caches, and parses the GEMStat Zenodo ZIP archive (~200 MB, cached to `data/cache/` after first call). Supports `country`, `parameters`, `start_date`, and `end_date` filtering. Previously returned only file metadata.
- **Taiwan WRA Reservoir collector** (`collectors/taiwan_wra.py`) — field names updated to match current API response format (lowercase keys: `reservoirname`, `dwl`, `inflow`, `outflow`, `capacity`, `nwlmax`). Storage percentage now computed from `capacity / nwlmax`.
- **Dashboard navigation** (`dashboard/app.py`) — fixed `StreamlitAPIException` caused by writing to a widget-bound `current_page` key after the radio widget was already instantiated. Navigation now uses a `_nav_pending` staging key applied before widget creation.
- **Viz backend guard** (`viz/styles.py`) — `_save_or_show()` no longer calls `plt.show()` when using the non-interactive Agg backend (eliminates `FigureCanvasAgg is non-interactive` warnings in CI and headless environments).

## [0.4.0] — 2026-04-01

### Added
- **Groundwater module** (`aquascope/groundwater/`) — GRACE satellite data integration, well monitoring, recharge estimation, and aquifer hydraulics analysis
- **Climate projections module** (`aquascope/climate/`) — CMIP6 scenario analysis, statistical downscaling, Palmer Drought Severity Index (PDSI), and climate impact assessment
- **JOSS paper** — Added `paper.md` and `paper.bib` for Journal of Open Source Software submission
- **EU Water Framework Directive collector** (in progress) — European water body status and compliance data
- **Japan MLIT collector** (in progress) — Japanese river and water quality monitoring data
- **Korea WAMIS collector** (in progress) — Korean water resources management information
- **15 data source collectors** total across global water monitoring networks
- **New CLI commands**: `groundwater`, `climate` for the new modules
- **New convenience API functions** in `aquascope.api` for streamlined programmatic access
- **Agricultural water module** (`aquascope/agri/`) — crop water demand, ET₀ calculation, water balance, productivity benchmarking, and irrigation planning
- **Alerts module** (`aquascope/alerts/`) — threshold-based monitoring, anomaly checking, and notification system
- **Advanced analysis** — changepoint detection, copula modelling
- **Hydrological modelling** (`aquascope/hydrology/`) — rainfall-runoff, routing, flood frequency, baseflow separation, CAMELS benchmarking
- **AI agent and planner** — multi-step research planning and autonomous execution
- **685+ tests** across all modules

### Changed
- Bumped version to 0.4.0
- Expanded optional dependency groups: `forecast`, `copernicus`, `scientific`, `dashboard`, `spatial`
- Added Python 3.13 classifier
- GitHub Actions publish workflow for PyPI releases via trusted publishing

## [0.2.0] — 2026-03-12

### Added
- **Analysis module** — Automated EDA (`aquascope eda`) with per-parameter statistics, outlier detection (IQR), correlation matrix, and completeness scoring
- **Data quality pipeline** — Assessment + preprocessing (`aquascope quality --fix`) with duplicate removal, imputation, outlier filtering, normalization, and daily resampling
- **7 model pipelines** — Auto-execute research methodologies via `aquascope run`:
  - Mann-Kendall trend analysis
  - Taiwan River Pollution Index (RPI)
  - PCA + K-Means clustering
  - Random Forest classification
  - XGBoost regression
  - ARIMA time-series forecasting
  - Pearson correlation analysis
- **3 new data collectors**:
  - GEMStat (UNEP global freshwater quality via Zenodo)
  - Taiwan Civil IoT (real-time SensorThings API)
  - US Water Quality Portal (USGS + EPA + 400 agencies)
- **13 new research methodologies** in the knowledge base (26 total), including: ARIMA forecasting, Transformer-based prediction, SWMM/QUAL2K process models, kriging spatial interpolation, isotope hydrology, paired watershed design, and more
- **5 new CLI commands**: `eda`, `quality`, `run`, `list-methods`, `list-sources`
- **Documentation guides**: Architecture, Adding a Data Source, Adding a Methodology, Running Pipelines
- **Jupyter quickstart tutorial** (`notebooks/01_quickstart_tutorial.ipynb`)
- **Comprehensive test suite** — 69 tests covering analysis, pipelines, collectors, AI engine

### Changed
- Bumped version to 0.2.0 (Beta status)
- `pandas` and `numpy` are now core dependencies (not optional)
- Updated `collect` CLI to support all 8 data sources
- Expanded `pyproject.toml` with `viz`, `ml` optional dependency groups

## [0.1.0] — 2026-03-10

### Added
- Initial release
- 5 data collectors: Taiwan MOENV, Taiwan WRA (level + reservoir), USGS, UN SDG 6
- Unified Pydantic schemas for water data
- AI methodology recommender with 13 built-in methodologies
- Rule-based scoring + optional LLM enhancement
- CLI with `collect` and `recommend` commands
- HTTP client with caching and rate limiting
- 12 tests, ruff lint, GitHub Actions CI/CD
- Contributing guide, MIT license
