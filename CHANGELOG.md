# Changelog

All notable changes to AquaScope are documented here.

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
