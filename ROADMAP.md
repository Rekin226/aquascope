# Roadmap

The roadmap reflects what's shipped, what's in-flight, and what's planned. Open items are reordered each release based on community demand in [Discussions](https://github.com/Rekin226/aquascope/discussions/categories/ideas).

## Where this is going (August 2026 direction review)

AquaScope is becoming **the open, continuously updated, citable record of the world's public water gauges, plus the zero-install place to look at them**; the Python library is the harvester and method engine underneath. Four layers, all on free infrastructure:

1. **The Archive** ([`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges)): every station catalog as GeoParquet, daily observations mirrored per station for sources whose terms allow it, harvested weekly by CI, DuckDB-readable in place. [#188](https://github.com/Rekin226/aquascope/issues/188)
2. **The Explorer** ([live](https://rekin226-aquascope-explorer.static.hf.space/)): click any gauge on Earth (or anywhere at all) and get the record, flood frequency with confidence limits, flow duration, trend and citations, computed in your browser. [#189](https://github.com/Rekin226/aquascope/issues/189)
3. **The Analyst**: the same tools for assistants (`aquascope mcp`, [#113](https://github.com/Rekin226/aquascope/issues/113)) and for people who ask in plain language (`aquascope ask`), plus `aquascope ingest` for any agency export.
4. **Ecosystem**: the GeoLibre plugin (`integrations/geolibre`), QGIS/R readers of the archive, a data paper.

### Next: one platform (August 2026 direction pass)

The four layers exist. They are also four separate things to learn: the Explorer is a map with a scrolling panel, the Streamlit dashboard is a different app with different methods and no gauges, and the Analyst answers two-tool questions well and complex ones not at all. The next arc merges them into **one zero-install app, GIS-grade, with the Python engine running in a browser worker**: the map is one mode, a workbench (your own data, plus the dashboard's methods) is another, and the Analyst is a drawer that keeps context across both. One engine, three faces: every capability stays a plain Python function in the package, and the web app, the MCP server and the CLI are thin faces over it.

- [ ] The app's shell, information architecture and UX: tabbed inspector, Ask beside the station, URL-as-state, search, mobile, export, "cite this" ([#231](https://github.com/Rekin226/aquascope/issues/231))
- [ ] Layers v1: free keyless basemaps, Sentinel imagery, 3D terrain and globe, NASA GIBS climate rasters with a date slider, BasinATLAS choropleths, gauge heat maps, measure and draw ([#232](https://github.com/Rekin226/aquascope/issues/232))
- [ ] Try the AI without a key: showcase replays, a model in the browser, an optional community demo pool, one provider registry ([#233](https://github.com/Rekin226/aquascope/issues/233))
- [ ] The Analyst, level 3: a Python sandbox tool in the page, plan then execute then verify, a streamed trace, and every answer as a report plus a re-runnable `study.yaml` ([#234](https://github.com/Rekin226/aquascope/issues/234), with [#54](https://github.com/Rekin226/aquascope/issues/54))
- [ ] Workbench: the dashboard's ten pages as panels over the same engine (`aquascope/workbench.py`), upload and ingest, a Data mode; Streamlit stays as the local UI ([#235](https://github.com/Rekin226/aquascope/issues/235))
- [ ] WebMCP tools in the page and MCP Apps views for `aquascope mcp` ([#236](https://github.com/Rekin226/aquascope/issues/236))
- [ ] HydroGym as a public, verifiable hydrology-agent benchmark on real basins, and the agent measured on it ([#175](https://github.com/Rekin226/aquascope/issues/175))

## Shipped

- [x] 29 data source collectors (Taiwan ×8, USA ×3, Global ×5, FAO ×2, EU, France, Germany, Ireland, UK, Japan, Korea, India, Chile, Brazil, Australia)
- [x] Rule-based + LLM methodology recommender (26 methods, OpenAI / Groq / HuggingFace / Ollama)
- [x] 26 auto-executable analysis pipelines
- [x] GR4J conceptual rainfall-runoff model + auto-calibration (NSE / KGE / log-NSE)
- [x] Model-evaluation metrics (NSE, KGE, PBIAS, RMSE, R²)
- [x] Bulletin 17C flood frequency with EMA
- [x] FAO-56 Penman-Monteith + crop water requirements (single Kc + dual Kcb/Ke modes)
- [x] Baseflow separation (Eckhardt, Lyne-Hollick, UKIH smoothed-minima)
- [x] Extreme-events frequency analysis (annual maxima/minima, return periods)
- [x] Bayesian UQ, copulas, ensembles, transfer learning
- [x] Spatial hydrology (DEM, watershed, Strahler)
- [x] Scientific I/O (WaterML, HEC, SWMM, NetCDF, HDF5)
- [x] Interactive Streamlit dashboard
- [x] 1,000+ tests with CAMELS benchmark validation
- [x] Theory guide with equations and DOI citations
- [x] EU Water Framework Directive collector
- [x] Japan MLIT / Korea WAMIS collectors
- [x] Groundwater module (GRACE, well databases, recharge, aquifer hydraulics)
- [x] Climate projection workflows (CMIP6, downscaling, PDSI, scenario analysis)
- [x] JOSS paper drafted (`paper.md` + `paper.bib`)
- [x] PyPI release (sdist + wheel + GitHub Actions publish workflow)
- [x] Shared source registry with licence and station-catalog metadata; `find_stations()` and `aquascope stations` over six catalogs (#187)
- [x] The Archive, Phase 0 + 1 + 2: 45,919-station GeoParquet catalog, per-station daily observations (discharge, water level, rainfall, groundwater level) and one Parquet bundle per variable and source on Hugging Face, weekly harvest with collector-health issues (#188)
- [x] The Explorer: static MapLibre + DuckDB-WASM + Pyodide page with click-any-gauge analysis and click-anywhere climate cards (#189)
- [x] MCP server (`aquascope mcp`), the Analyst (`aquascope ask`) and `aquascope ingest` (#113)
- [x] GeoLibre plugin: AquaScope Gauges (`integrations/geolibre`)
- [x] Explorer Phase 2: Ask ✨ (the Analyst in the page), NLDI catchments, BasinATLAS catchments everywhere, similar gauged basins (#189, #213, #218)
- [x] Caravan-format export from the archive (`aquascope caravan export`, #217) and the similar-basins donor search (#53, the practical half)
- [x] Self-healing harvest: automated repair proposals as reviewable PRs (`repair.yml`)

## In progress

- [x] Hosted demo (try without installing): [live on Hugging Face](https://huggingface.co/spaces/Rekin226/aquascope-dashboard), runs fully in-browser via stlite/WebAssembly. Being folded into the one app; the dashboard Space ships an older wheel and will redirect there ([#235](https://github.com/Rekin226/aquascope/issues/235))
- [ ] Tutorial notebooks on Binder / Colab

## Planned

- [ ] Additional data sources — vote at [Discussions → Ideas](https://github.com/Rekin226/aquascope/discussions/categories/ideas)
- [ ] Multi-language documentation (中文, Français, 日本語)
- [ ] ReadTheDocs hosting
- [ ] NumFOCUS Sponsored Project application

## Major features — leveling up

Ambitious, high-impact work that takes AquaScope to the next level. These are [`major feature`](https://github.com/Rekin226/aquascope/labels/major%20feature) · `help wanted` — larger than a weekend, mentorship available. Comment on the issue to discuss scope before starting.

- [ ] Archive Phase 3: reservoir storage and water-quality variables, more agencies with a `stations()` catalog (Australia BOM, Taiwan WRA), a data paper ([#188](https://github.com/Rekin226/aquascope/issues/188))
- [ ] Explorer Phase 3: GR4J calibrated in the page shipped (JS port + differential evolution, ~2 s for 40 years); still open here: agency catchment boundaries under open licences (UK NRFA), sub-daily where terms allow ([#189](https://github.com/Rekin226/aquascope/issues/189)). The shell and UX rebuild is [#231](https://github.com/Rekin226/aquascope/issues/231), the map layers [#232](https://github.com/Rekin226/aquascope/issues/232)
- [ ] CAMELS-TW: `aquascope caravan export` is ready; the Taiwan daily discharge collector ([#211](https://github.com/Rekin226/aquascope/issues/211)) is the missing leg ([#100](https://github.com/Rekin226/aquascope/issues/100), [#99](https://github.com/Rekin226/aquascope/issues/99))
- [x] Prediction in Ungauged Basins: flow signatures regionalised over the similar-basins donors (similarity-weighted transfer + ridge regression, leave-one-out skill published with the archive; `aquascope basins regionalize`, MCP `regionalize_signatures`, Explorer table) ([#53](https://github.com/Rekin226/aquascope/issues/53)); parameter regionalisation (GR4J parameters from donors) is the follow-up
- [x] HydroGym Phase 0: `aquascope.gym.CalibrationEnv` (gymnasium API, GR4J calibration on any Archive basin or a synthetic one, NSE/KGE/log-NSE reward with validation metrics), three baselines, a leaderboard, `aquascope gym`; Phases 1 and 2 stay open and are now scoped as a public, verifiable hydrology-agent benchmark on real basins (task suite with unsolvable tasks, held-out splits, cost accounting, frontier and small models, leaderboard) ([#175](https://github.com/Rekin226/aquascope/issues/175))
- [ ] Declarative, reproducible study runner `aquascope run study.yaml` with provenance ([#54](https://github.com/Rekin226/aquascope/issues/54))
- [ ] Plugin architecture — third-party collectors & methodologies via entry points ([#55](https://github.com/Rekin226/aquascope/issues/55))
- [ ] Large-sample CAMELS benchmark — automated accuracy report ([#56](https://github.com/Rekin226/aquascope/issues/56))

## Good first issues — up for grabs

Newcomers welcome. Just comment to claim one, then follow the [contributor ladder](CONTRIBUTING.md) (`good first issue` → `good second issue` → area owner).

- [ ] Edge-case tests for the FAO-56 ETo functions ([#40](https://github.com/Rekin226/aquascope/issues/40))
- [ ] New collector: Germany PEGELONLINE ([#122](https://github.com/Rekin226/aquascope/issues/122))
- [ ] New collector: Ireland OPW waterlevel.ie ([#123](https://github.com/Rekin226/aquascope/issues/123))
- [ ] Type annotations for `climate/indices.py` ([#32](https://github.com/Rekin226/aquascope/issues/32))

_(#41 colorblind palette and #42 baseflow edge-case tests — ✅ shipped.)_

**Ready for more?** The [`good second issue`](https://github.com/Rekin226/aquascope/labels/good%20second%20issue) tier: Mann-Kendall trend test + Sen's slope ([#44](https://github.com/Rekin226/aquascope/issues/44)), flow-duration-curve slope + runoff ratio ([#45](https://github.com/Rekin226/aquascope/issues/45)), Dashboard Groundwater page ([#125](https://github.com/Rekin226/aquascope/issues/125)), SPI unification ([#127](https://github.com/Rekin226/aquascope/issues/127)), and the CAMELS-BR collector ([#124](https://github.com/Rekin226/aquascope/issues/124)).

**Prefer hunting bugs?** Two real, reproducible ones are open: GEV flood-frequency confidence intervals blow up on typical record lengths ([#119](https://github.com/Rekin226/aquascope/issues/119)) and the keyless USGS quickstart path fails ([#120](https://github.com/Rekin226/aquascope/issues/120)).

## How to influence the roadmap

- 👍 **Vote** on existing requests in [Discussions → Ideas](https://github.com/Rekin226/aquascope/discussions/categories/ideas)
- 💡 **Propose** something new with the *Ideas* discussion category
- 🐛 **File** bugs and edge cases in [Issues](https://github.com/Rekin226/aquascope/issues/new/choose)
- 🤝 **Contribute** — see [CONTRIBUTING.md](CONTRIBUTING.md)
