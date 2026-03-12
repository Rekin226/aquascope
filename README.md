# AquaScope

[![CI](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](CHANGELOG.md)

**Open-source water data aggregation toolkit with AI-powered research methodology recommendations.**

AquaScope collects water-quality, hydrology, and environmental data from Taiwan's government open APIs and global sources, normalises it into a unified schema, and uses an AI engine to recommend and auto-execute suitable research methodologies for water-related studies.

Built to help researchers worldwide access, analyse, and make informed methodological decisions about water data — from MBBR pilot evaluations in Burkina Faso to long-term river quality trends in Taiwan.

---

## Features

- **8 data source collectors** — Taiwan MOENV, WRA, Civil IoT; USGS; GEMStat; Water Quality Portal; UN SDG 6
- **Unified data schemas** — All sources normalised into consistent Pydantic models
- **AI methodology recommender** — Rule-based scoring engine with optional LLM enhancement (OpenAI, Anthropic, local Ollama)
- **26 built-in research methodologies** — Statistical analysis, ML forecasting, process engineering, remote sensing, hydrological modelling, policy analysis
- **7 auto-executable pipelines** — Trend analysis, WQI, PCA clustering, Random Forest, XGBoost, ARIMA, correlation
- **Automated EDA** — Profile any water dataset with one command
- **Data quality pipeline** — Detect and fix duplicates, missing values, outliers, temporal gaps
- **CLI and Python API** — Use from the command line or import as a library
- **Extensible** — Add new data sources and methodologies via pull request
- **CI/CD ready** — GitHub Actions for linting, testing, and type checking

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLI / Python API                     │
├───────────┬───────────┬──────────────┬──────────────────┤
│ Collectors│ Analysis  │  Pipelines   │    AI Engine      │
│ (8 APIs)  │ EDA+QA    │ (7 methods)  │ (26 methodologies)│
├───────────┴───────────┴──────────────┴──────────────────┤
│              Unified Schemas (Pydantic)                   │
├─────────────────────────────────────────────────────────┤
│              Utilities (HTTP, Storage)                    │
└─────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | Region | Data Types | API | Status |
|--------|--------|------------|-----|--------|
| [Taiwan MOENV](https://data.moenv.gov.tw) | Taiwan | River/tap water quality, RPI | REST | ✅ |
| [Taiwan WRA](https://opendata.wra.gov.tw) | Taiwan | Water levels, reservoir status | REST | ✅ |
| [Taiwan Civil IoT](https://sta.ci.taiwan.gov.tw) | Taiwan | Real-time sensors (level, flow, rain) | SensorThings | ✅ New |
| [USGS](https://api.waterdata.usgs.gov) | USA | Streamflow, water quality, gage height | OGC | ✅ |
| [Water Quality Portal](https://waterqualitydata.us) | USA | Integrated WQ from 400+ agencies | REST/CSV | ✅ New |
| [GEMStat](https://gemstat.org) | Global | Freshwater quality (170+ countries) | Zenodo | ✅ New |
| [UN SDG 6](https://sdg6data.org) | Global | SDG 6 indicators (6.1.1 – 6.6.1) | REST | ✅ |

**Want to add your country's water data?** See our [guide to adding data sources](docs/guides/adding_data_source.md) — contributions from every region are welcome.

## Quick Start

### Installation

```bash
# From source (recommended for contributors)
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all]"

# Or minimal install (collectors + recommender only)
pip install -e .
```

### CLI Usage

```bash
# Collect Taiwan river water quality data
aquascope collect --source taiwan_moenv --api-key YOUR_MOENV_KEY

# Collect USGS data for the last 7 days
aquascope collect --source usgs --days 7

# Collect from the US Water Quality Portal
aquascope collect --source wqp --state US:06

# Run EDA on collected data
aquascope eda --file data/raw/taiwan_moenv_20260312.json

# Assess data quality and auto-fix
aquascope quality --file data/raw/taiwan_moenv_20260312.json --fix

# Get methodology recommendations
aquascope recommend \
  --parameters DO,BOD5,COD,NH3-N,SS \
  --goal "long-term water quality trend analysis" \
  --years 10 --n-stations 20 --scope "Taiwan"

# Execute a methodology pipeline
aquascope run --method trend_analysis \
  --file data/raw/taiwan_moenv_20260312.json \
  --output results/trends.json

# List all methodologies and available pipelines
aquascope list-methods

# List all data sources
aquascope list-sources
```

### Python API

```python
import pandas as pd
from aquascope.collectors import TaiwanMOENVCollector, USGSCollector
from aquascope.analysis.eda import generate_eda_report, profile_dataset
from aquascope.analysis.quality import assess_quality, preprocess
from aquascope.ai_engine.recommender import recommend
from aquascope.pipelines.model_builder import run_pipeline

# 1. Collect data
collector = TaiwanMOENVCollector(api_key="YOUR_KEY")
records = collector.collect()
df = pd.DataFrame([r.model_dump() for r in records])

# 2. Quality check & preprocess
quality = assess_quality(df)
df_clean = preprocess(df, steps=quality.recommended_steps)

# 3. Auto-profile and recommend
profile = profile_dataset(df_clean)
profile.research_goal = "Trend analysis and pollution source identification"
recs = recommend(profile, top_k=5)

# 4. Execute the top recommendation
result = run_pipeline(recs[0].methodology.id, df_clean)
print(result.summary)
print(result.metrics)
```

## Built-in Research Methodologies (26)

| Category | Methodologies | Pipelines |
|----------|--------------|-----------|
| Statistical | Mann-Kendall Trend, WQI/RPI, PCA + Clustering, Correlation Analysis, Bayesian Inference, Copula Dependence | ✅ 4 |
| Machine Learning | LSTM Forecasting, Random Forest, XGBoost Regression, Transformer Prediction, Autoencoder Anomaly Detection | ✅ 2 |
| Time-Series | ARIMA/SARIMA Forecasting | ✅ 1 |
| Process Engineering | MBBR Pilot, MBR Fouling, A2O Nutrient Removal, SWMM Urban Drainage, QUAL2K River Modelling | — |
| Spatial Analysis | Satellite Eutrophication, GIS Watershed, Kriging Interpolation | — |
| Hydrological | SWAT Modelling, Isotope Hydrology, Paired Watershed Design | — |
| Policy | SDG 6 Benchmarking, IWRM Assessment | — |

## Analysis Pipelines

AquaScope can auto-build and execute 7 research methodologies:

```bash
aquascope run --method <id> --file <data.json>
```

| Pipeline | What It Does | Key Output |
|----------|-------------|------------|
| `trend_analysis` | Mann-Kendall trend test per station/parameter | Trend direction, p-value, slope |
| `wqi_calculation` | Taiwan River Pollution Index (RPI) | RPI score + pollution category |
| `pca_clustering` | PCA dimensionality reduction + K-Means | Variance explained, cluster labels |
| `random_forest_classification` | Water quality classification | Accuracy, feature importance |
| `xgboost_regression` | Predict target parameter | R², RMSE, feature importance |
| `arima_forecast` | ARIMA time-series forecasting | AIC/BIC, forecast values |
| `correlation_analysis` | Pearson correlation between parameters | Correlation matrix, significant pairs |

## Project Structure

```
aquascope/
├── aquascope/
│   ├── collectors/          # 8 data collection modules
│   │   ├── base.py          # Abstract base collector
│   │   ├── taiwan_moenv.py  # Taiwan Ministry of Environment
│   │   ├── taiwan_wra.py    # Taiwan Water Resources Agency
│   │   ├── taiwan_civil_iot.py  # Taiwan Civil IoT SensorThings
│   │   ├── usgs.py          # USGS Water Data OGC API
│   │   ├── sdg6.py          # UN SDG 6 indicators
│   │   ├── gemstat.py       # GEMStat global freshwater
│   │   └── wqp.py           # US Water Quality Portal
│   ├── ai_engine/           # AI methodology recommender
│   │   ├── knowledge_base.py  # 26 methodology catalogue
│   │   └── recommender.py     # Scoring engine + LLM mode
│   ├── analysis/            # Data analysis modules
│   │   ├── eda.py           # Exploratory Data Analysis
│   │   └── quality.py       # Quality assessment + preprocessing
│   ├── pipelines/           # Auto-executable methodology pipelines
│   │   └── model_builder.py # 7 pipeline implementations
│   ├── schemas/             # Pydantic data models
│   ├── utils/               # HTTP client, storage helpers
│   └── cli.py               # 7-command CLI interface
├── tests/                   # 69 tests
├── notebooks/               # Jupyter tutorials
├── docs/guides/             # Contributor documentation
├── .github/                 # CI workflows + issue templates
├── pyproject.toml           # Project configuration
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Contribution guide
└── LICENSE                  # MIT License
```

## Contributing

We actively welcome contributions from the global water research community! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**High-impact contributions:**

- **New data source collectors** — Add APIs from your country (Japan, Korea, EU, India, Brazil, any region!)
- **New research methodologies** — Expand the AI recommender's knowledge base
- **New pipelines** — Make more methodologies auto-executable
- **Jupyter notebooks** — Tutorials and analysis examples
- **Translations** — Help make AquaScope accessible in more languages

### Guides for Contributors

- [Architecture Overview](docs/guides/architecture.md)
- [Adding a Data Source](docs/guides/adding_data_source.md)
- [Adding a Methodology](docs/guides/adding_methodology.md)
- [Running Pipelines](docs/guides/running_pipelines.md)

## Getting API Keys

| Source | Key Required? | How to Get |
|--------|:---:|------------|
| Taiwan MOENV | Recommended | [Register here](https://data.moenv.gov.tw/en/apikey) (free) |
| Taiwan WRA | No | Open access |
| Taiwan Civil IoT | No | Open access (SensorThings) |
| USGS | Optional | [Request here](https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys) (free, higher rate limits) |
| Water Quality Portal | No | Open access |
| GEMStat | No | Open access via Zenodo |
| UN SDG 6 | No | Open access |

## Roadmap

- [x] Taiwan MOENV, WRA, Civil IoT collectors
- [x] USGS, SDG 6, GEMStat, WQP collectors
- [x] Rule-based + LLM methodology recommender (26 methods)
- [x] 7 auto-executable analysis pipelines
- [x] EDA and data quality modules
- [x] CLI (7 commands) + Python API
- [x] CI/CD with GitHub Actions
- [x] Comprehensive documentation & Jupyter tutorial
- [ ] EU Water Framework Directive collector
- [ ] Japan MLIT / Korea WAMIS collectors
- [ ] Interactive dashboard (Streamlit / Panel)
- [ ] ODE/PDE numerical solvers (contaminant transport, groundwater flow)
- [ ] Research paper template generator
- [ ] Multi-language documentation (中文, Français, 日本語)

## Citation

If you use AquaScope in your research, please cite:

```bibtex
@software{aquascope2026,
  title     = {AquaScope: Open-Source Water Data Aggregation and AI Research Methodology Recommender},
  author    = {AquaScope Contributors},
  year      = {2026},
  url       = {https://github.com/Rekin226/aquascope},
  version   = {0.2.0},
  license   = {MIT}
}
```

## License

MIT License — see [LICENSE](LICENSE).
