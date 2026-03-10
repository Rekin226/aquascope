# AquaScope

**Open-source water data aggregation toolkit with AI-powered research methodology recommendations.**

AquaScope collects water-quality, hydrology, and environmental data from Taiwan's government open APIs and global sources, normalises it into a unified schema, and uses an AI engine to recommend suitable research methodologies for water-related studies.

Built to help researchers worldwide access, analyse, and make informed methodological decisions about water data — from MBBR pilot evaluations in Burkina Faso to long-term river quality trends in Taiwan.

---

## Features

- **Multi-source data collection** — Taiwan MOENV, Taiwan WRA, USGS, UN SDG 6, with a plug-in architecture for adding more
- **Unified data schemas** — All sources normalised into consistent Pydantic models
- **AI methodology recommender** — Rule-based scoring engine with optional LLM enhancement (OpenAI, Anthropic, local Ollama)
- **13+ built-in research methodologies** — Statistical analysis, ML forecasting, process engineering (MBBR, MBR, A2O), remote sensing, hydrological modelling, policy analysis
- **CLI and Python API** — Use from the command line or import as a library
- **Extensible** — Add new data sources and methodologies via pull request
- **CI/CD ready** — GitHub Actions for linting, testing, and type checking

## Data Sources

| Source | Region | Data Types | Update Freq |
|--------|--------|------------|-------------|
| [Taiwan MOENV](https://data.moenv.gov.tw) | Taiwan | River/tap water quality, RPI | Monthly |
| [Taiwan WRA](https://opendata.wra.gov.tw) | Taiwan | Water levels, reservoir status, groundwater | Real-time / Daily |
| [USGS OGC API](https://api.waterdata.usgs.gov) | USA | Streamflow, water quality, gage height | Continuous / Daily |
| [UN SDG 6](https://sdg6data.org/en/api) | Global | SDG 6 indicators (6.1.1 – 6.6.1) | Annual |

**Planned:** GEMStat, EU Water Framework Directive, Japan MLIT, Korea WAMIS, WHO WASH — contributions welcome!

## Quick Start

### Installation

```bash
# From source (recommended for contributors)
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

### CLI Usage

```bash
# Collect Taiwan river water quality data
aquascope collect --source taiwan_moenv --api-key YOUR_MOENV_KEY

# Collect USGS data for the last 7 days
aquascope collect --source usgs --days 7

# Collect SDG 6 data for specific countries
aquascope collect --source sdg6 --countries TWN,BFA,USA

# Get methodology recommendations
aquascope recommend \
  --parameters DO,BOD5,COD,NH3-N,SS \
  --goal "long-term water quality trend analysis" \
  --years 10 --n-stations 20 --scope "Taiwan"

# LLM-enhanced recommendations (requires OpenAI key)
aquascope recommend \
  --parameters DO,BOD5,COD \
  --goal "MBBR pilot study" \
  --use-llm --llm-api-key sk-...

# Recommend from a collected data file
aquascope recommend --from-file data/raw/taiwan_moenv_20260310.json
```

### Python API

```python
from aquascope.collectors import TaiwanMOENVCollector, USGSCollector
from aquascope.ai_engine import recommend, DatasetProfile

# Collect data
moenv = TaiwanMOENVCollector(api_key="YOUR_KEY")
samples = moenv.collect()

# Build a dataset profile and get recommendations
profile = DatasetProfile(
    parameters=["DO", "BOD5", "COD", "NH3-N", "SS"],
    n_records=len(samples),
    n_stations=15,
    time_span_years=5.0,
    geographic_scope="Taiwan — Tamsui River",
    research_goal="Trend analysis and pollution source identification",
    keywords=["trend", "multivariate", "monitoring"],
)

recommendations = recommend(profile, top_k=5)
for rec in recommendations:
    print(f"{rec.methodology.name}: {rec.score} — {rec.rationale}")
```

## Built-in Research Methodologies

| Category | Methodologies |
|----------|--------------|
| Statistical | Mann-Kendall Trend Analysis, WQI Computation, PCA + Cluster Analysis |
| Machine Learning | LSTM Forecasting, Random Forest Classification, XGBoost Regression |
| Process Engineering | MBBR Pilot Evaluation, MBR Fouling Study, A2O Nutrient Removal |
| Remote Sensing | Satellite Eutrophication Assessment, GIS Watershed Analysis |
| Hydrological Modelling | SWAT Modelling |
| Policy Analysis | SDG 6 Cross-Country Benchmarking |

## Project Structure

```
aquascope/
├── aquascope/
│   ├── collectors/          # Data collection modules
│   │   ├── base.py          # Abstract base collector
│   │   ├── taiwan_moenv.py  # Taiwan Ministry of Environment
│   │   ├── taiwan_wra.py    # Taiwan Water Resources Agency
│   │   ├── usgs.py          # USGS Water Data OGC API
│   │   └── sdg6.py          # UN SDG 6 indicators
│   ├── ai_engine/           # AI methodology recommender
│   │   ├── knowledge_base.py  # Methodology catalogue
│   │   └── recommender.py     # Scoring engine + LLM mode
│   ├── schemas/             # Pydantic data models
│   │   └── water_data.py    # Unified water data schemas
│   ├── utils/               # Shared utilities
│   │   ├── http_client.py   # HTTP client with cache + retry
│   │   └── storage.py       # Data persistence helpers
│   └── cli.py               # Command-line interface
├── tests/                   # Test suite
├── examples/                # Usage examples
├── notebooks/               # Jupyter notebooks
├── data/                    # Local data storage (gitignored)
├── docs/                    # Documentation
├── .github/                 # CI workflows + issue templates
├── pyproject.toml           # Project configuration
├── CONTRIBUTING.md          # Contribution guide
├── LICENSE                  # MIT License
└── README.md
```

## Contributing

We actively welcome contributions from the global water research community! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**High-impact contributions:**

- **New data source collectors** — Add APIs from your country (Japan, Korea, EU, India, Brazil, any country!)
- **New research methodologies** — Expand the AI recommender's knowledge base
- **Jupyter notebooks** — Tutorials and analysis examples in any language
- **Translations** — Help make AquaScope accessible in more languages

## Getting API Keys

| Source | Key Required? | How to Get |
|--------|:---:|------------|
| Taiwan MOENV | Recommended | [Register here](https://data.moenv.gov.tw/en/apikey) (free) |
| Taiwan WRA | No | Open access |
| USGS | Optional | [Request here](https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys) (free, higher rate limits) |
| UN SDG 6 | No | Open access |

## Roadmap

- [x] Taiwan MOENV river water quality collector
- [x] Taiwan WRA water level + reservoir collectors
- [x] USGS OGC API collector
- [x] UN SDG 6 indicator collector
- [x] Rule-based methodology recommender
- [x] LLM-enhanced recommendations (OpenAI / Ollama)
- [x] CLI interface
- [x] CI/CD with GitHub Actions
- [ ] GEMStat (global freshwater quality) collector
- [ ] EU Water Framework Directive collector
- [ ] Interactive dashboard (Streamlit / Panel)
- [ ] Automated data quality assessment
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
  license   = {MIT}
}
```

## License

MIT License — see [LICENSE](LICENSE).
