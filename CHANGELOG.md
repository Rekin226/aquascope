# Changelog

All notable changes to AquaScope are documented here.

## [0.6.0] — 2026-06-26

### Added
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
