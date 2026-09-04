<div align="center">

<img src="docs/assets/logo.svg" alt="AquaScope logo" width="160"/>

# AquaScope

**Open-source Python toolkit for water data, hydrology, and agricultural water management — with an AI engine that recommends and auto-executes research methodologies.**

[![CI](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml)
[![Pyodide](https://github.com/Rekin226/aquascope/actions/workflows/pyodide-smoke.yml/badge.svg)](https://github.com/Rekin226/aquascope/actions/workflows/pyodide-smoke.yml)
[![PyPI version](https://img.shields.io/pypi/v/aquascope.svg?color=blue&cacheSeconds=300&v=2)](https://pypi.org/project/aquascope/)
[![Python](https://img.shields.io/pypi/pyversions/aquascope.svg?color=informational&cacheSeconds=300&v=2)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21903143.svg)](https://doi.org/10.5281/zenodo.21903143)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-1000%2B%20passing-brightgreen.svg)](#)
[![Live Explorer Demo – Runs in Your Browser](https://img.shields.io/badge/%F0%9F%8C%8A%20Live%20Demo-AquaScope%20Explorer-blue)](https://rekin226-aquascope-explorer.static.hf.space/)

[![GitHub stars](https://img.shields.io/github/stars/Rekin226/aquascope?style=social)](https://github.com/Rekin226/aquascope/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Rekin226/aquascope?style=social)](https://github.com/Rekin226/aquascope/network/members)

[**🌊 Live Explorer Demo - Runs in Your Browser, No Install Required**](https://rekin226-aquascope-explorer.static.hf.space/) ·
[**Install**](#-install) ·
[**Examples**](#-examples) ·
[**CLI**](#-cli) ·
[**Features**](docs/features.md) ·
[**Docs**](#-documentation) ·
[**Roadmap**](ROADMAP.md) ·
[**Discussions**](https://github.com/Rekin226/aquascope/discussions)

[![Support on Ko-fi](https://img.shields.io/badge/Support%20AquaScope-Ko--fi-FF5E5B?logo=kofi&logoColor=white)](https://ko-fi.com/getaquascope) if AquaScope helps your research.

🌐 Read this in: [Français](docs/i18n/README.fr.md)

</div>

---

AquaScope unifies **29 global water-data sources** behind one Python schema, then layers a full scientific computing stack on top — from **Bulletin 17C flood frequency** to **FAO-56 crop water requirements** — wrapped in an AI engine that scores **26 research methodologies** against your dataset and auto-executes **26 analysis pipelines**. Validated against the CAMELS benchmark with 1,000+ tests.

---

## 🌍 Try it without installing anything

**[AquaScope Explorer](https://rekin226-aquascope-explorer.static.hf.space/)**: every public gauge we can reach on one map
(45,919 stations from USGS, UK EA, Hub'Eau, PEGELONLINE, Ireland OPW and Taiwan CWA). Click one and get the observed record,
flood frequency with confidence limits, flow duration and trend, computed in your browser by aquascope on Pyodide.
The catalog behind it is an open GeoParquet dataset, [`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges), harvested weekly.
Press **Ask ✨** to type a question in plain language (bring your own key, Groq and Hugging Face are free): the model picks the
tools, aquascope runs them in your browser, and the answer ends with the data used and the methods with citations.
Not a Python user? The same files open in [R, QGIS, DuckDB and Julia](docs/readers.md) in place; `integrations/qgis/` has a
drag-and-drop layer definition.

Prefer an assistant? `pip install "aquascope[mcp]"` then `claude mcp add aquascope -- aquascope mcp` gives Claude (or any
MCP client) `find_stations`, `get_timeseries`, `analyze_station` and `flood_frequency` over the same catalog and methods
([docs](docs/mcp.md)).

## ✨ What you can do

- 🌊 **Pull water data** from USGS, FAO AQUASTAT, FAO WaPOR, GEMStat, EU WFD, Copernicus ERA5, France Hub'Eau, Taiwan MOENV/WRA/Civil IoT/DataGov, Japan MLIT, Korea WAMIS, India WRIS, GRDC, CAMELS-CL, OpenMeteo, UN SDG 6, US Water Quality Portal — **one unified Python API**.
- 📈 **Run hydrological analyses** — Bulletin 17C flood frequency (GEV / LP3 / Gumbel / non-stationary GEV / EMA), baseflow separation, rating curves, 22 hydrological signatures.
- 🌾 **Plan agricultural water** — FAO-56 Penman-Monteith ET₀, crop water requirements for 23 crops, irrigation scheduling, soil water balance with auto-irrigation.
- 🤖 **Ask the AI engine** — describe your goal in plain English and get a recommended methodology, scored against your dataset profile and auto-executed. LLM enhancement via OpenAI, Groq (free), HuggingFace (free), or local Ollama.
- 📊 **Visualise + report** — 16 plot types, Q-Q / P-P diagnostics, Markdown / HTML reports with embedded figures, threshold alerts (WHO / EPA / EU WFD).
- 🗺️ **Spatial hydrology** — DEM processing, D8 flow direction, watershed delineation, Strahler ordering.

For the full capability list see [docs/features.md](docs/features.md).

## 📊 Why AquaScope

| | AquaScope | HEC-SSP | R `lmom` | Standalone collectors |
| :--- | :---: | :---: | :---: | :---: |
| Bulletin 17C FFA + EMA | ✅ | ✅ | partial | — |
| Non-stationary GEV | ✅ | — | partial | — |
| Baseflow separation (Lyne-Hollick, Eckhardt) | ✅ | — | — | — |
| FAO-56 Penman-Monteith ET₀ + crop water | ✅ | — | — | — |
| 29 unified data collectors | ✅ | — | — | per-source |
| AI methodology recommender (OpenAI / Groq / HF / Ollama) | ✅ | — | — | — |
| Interactive Streamlit dashboard | ✅ | — | — | — |
| Free, MIT, Python-native | ✅ | partial | ✅ | varies |

---

## ⚡ Install

```bash
pip install aquascope              # core — collectors + hydrology
pip install "aquascope[all]"       # everything — ML, viz, spatial, dashboard
```

Feature-group extras:

```bash
pip install "aquascope[ml]"           # sklearn, xgboost, statsmodels
pip install "aquascope[viz]"          # matplotlib, seaborn, folium
pip install "aquascope[scientific]"   # xarray, netcdf4, h5py
pip install "aquascope[interop]"      # xarray + geopandas (collect as_xarray / as_geodataframe)
pip install "aquascope[spatial]"      # rasterio, geopandas, shapely
pip install "aquascope[dashboard]"    # streamlit
pip install "aquascope[forecast]"     # prophet, torch (for LSTM)
```

For development:

```bash
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
pip install -e ".[all,dev]"
```

---

## 🚀 Examples

### 1. Flood frequency analysis (Bulletin 17C)

```python
from aquascope.api import flood_analysis

result = flood_analysis(daily_discharge, method="gev", return_periods=[10, 50, 100])
print(result.return_periods)
# {10: 1840.2, 50: 2530.7, 100: 2870.4}
print(result.confidence_intervals)
# {10: (1690.4, 2010.6), 50: (2280.1, 2820.9), 100: (2540.6, 3260.5)}
```

Switch `method` to `"lp3"`, `"gumbel"`, `"gev_lmoments"`, or `"gpd"`. Non-stationary GEV (`fit_nonstationary_gev`) and Bulletin 17C EMA for censored records (`expected_moments_algorithm`) are available in `aquascope.hydrology.flood_frequency`.

### 2. Baseflow separation + hydrological signatures

```python
from aquascope.api import baseflow_analysis, compute_all_signatures

bf  = baseflow_analysis(daily_discharge, method="eckhardt")   # or "lyne_hollick"
sig = compute_all_signatures(daily_discharge)

print(bf.bfi)                  # baseflow index, e.g. 0.42
print(sig.q5, sig.q95)         # high-flow / low-flow exceedances
print(sig.flashiness_index)    # Richards-Baker flashiness index
```

22 signatures across magnitude, variability, timing, recession, and flashiness — see [docs/features.md](docs/features.md#hydrological-analysis).

### 3. Collect data from any of the 29 sources

```python
from aquascope import find_stations
from aquascope.collectors import USGSCollector, AquastatCollector, WaPORCollector

# Which gauges measure discharge around Greater London? (USGS, UK EA, Hub'Eau,
# PEGELONLINE, Ireland OPW and Taiwan CWA expose station catalogs; more coming)
gauges = find_stations(bbox=(-0.5, 51.3, 0.3, 51.7), variable="discharge")
print(gauges[0].name, gauges[0].url)

usgs = USGSCollector()   # pass api_key=... for reliable access
flow = usgs.collect(days=7, bbox="-77.6,38.7,-76.9,39.1")   # Potomac basin, last week

aquastat = AquastatCollector()
egy_water = aquastat.collect(country_code="EGY", variable_ids=[4263, 4253, 4312])

wapor = WaPORCollector()
et = wapor.collect(
    bbox=(30.5, 29.8, 31.1, 30.2),
    variable="RET",
    start_date="2026-04-01",
    end_date="2026-07-31",
)
```

Every collector returns records in the **same Pydantic schema**, so downstream analyses don't care where the data came from. See [docs/data_sources.md](docs/data_sources.md) for the full list.

### 4. FAO-56 crop water requirements + soil water balance

```python
from datetime import date
from aquascope.agri import (
    penman_monteith_daily,
    crop_water_requirement,
    SoilWaterBalance,
)
from aquascope.agri.water_balance import SoilProperties

# Reference ET (FAO-56 Penman-Monteith) — Cairo, July
eto = penman_monteith_daily(
    t_min=18.0, t_max=32.0, rh_min=40, rh_max=80,
    u2=2.0, rs=22.0, latitude=30.0, elevation=70, doy=180,
)

# Crop water requirement for maize from planting through harvest — eto_series is
# a daily ET₀ pd.Series (build one with penman_monteith_series on a weather DataFrame)
cwr = crop_water_requirement(eto_series, crop="maize", planting_date=date(2026, 4, 1))

# Soil water balance with auto-irrigation triggers — returns a daily DataFrame
soil    = SoilProperties(field_capacity=0.30, wilting_point=0.15, root_depth=1.0)
balance = SoilWaterBalance(soil).auto_irrigate(
    cwr["etc"], precip_series, efficiency=0.7,
)
print(balance["irrigation_mm"].sum())             # total irrigation applied (mm)
print(int(balance["irrigation_trigger"].sum()))   # number of deficit days
```

Notebook tutorial: [agricultural water demand and irrigation scheduling](notebooks/07_agricultural_water_demand.ipynb).

### 5. AI methodology recommender

```python
from aquascope.ai_engine import DatasetProfile, recommend

# Describe your dataset and goal — get ranked, scored methodologies
profile = DatasetProfile(
    parameters=["DO", "BOD5", "COD"],
    n_records=4_500,
    time_span_years=6.0,
    research_goal="detect long-term pollution trends with seasonality",
)
recs = recommend(profile)

for r in recs[:3]:
    print(f"{r.score:5.1f}  {r.methodology.id:<18}  {r.rationale[:46]}…")
#  55.9  trend_analysis      Your dataset includes bod5, cod, do which are…
#  54.6  lstm_forecasting    Your dataset includes bod5, cod, do which are…
#  54.6  arima_forecast      Your dataset includes bod5, cod, do which are…
```

Then auto-execute the top result with `run_pipeline(recs[0].methodology.id, df)`.

### 6. Change-point detection + copula dependence

```python
from aquascope.api import detect_changepoints, fit_copula

cps  = detect_changepoints(annual_runoff, method="pettitt")
cop  = fit_copula(rainfall, runoff, family="auto")    # AIC-selects Gaussian/Clayton/Gumbel/Frank
cp   = cps.changepoints[0]
print(cp.timestamp, cp.p_value)
print(cop.family, cop.parameter, cop.aic)
```

### 7. Bayesian regression with uncertainty quantification

```python
from aquascope.api import bayesian_regression

# Annual rainfall → runoff with full posterior + convergence diagnostics
posterior = bayesian_regression(X=annual_precip, y=annual_runoff)

print(posterior.posterior_mean)
# {'beta_0': 12.4, 'beta_1': 0.82, 'sigma2': 41.6}

print(posterior.credible_intervals["beta_1"])
# (0.78, 0.86)        ← 95% credible interval on slope

print(posterior.r_hat)
# {'beta_0': 1.00, 'beta_1': 1.00, 'sigma2': 1.00}    ← Gelman–Rubin, converged

print(posterior.dic, posterior.effective_sample_size["beta_1"])
# 124.7  9842.0       ← model fit + effective sample size
```

Switch to MCMC with `degree>1` for polynomial models, or pass `prior_precision` for informative priors. Conjugate linear, polynomial, and Metropolis-Hastings backends are all available.

---

## 💻 CLI

AquaScope ships a 27-command CLI (`agri`, `basins`, `caravan`, `gym` and `playbooks` carry subcommands) for the most common workflows:

```bash
# Find stations, then collect data
aquascope stations --bbox -0.5,51.3,0.3,51.7 --variable discharge --format geojson
aquascope harvest stations --out archive          # the open gauge catalog (GeoParquet)
aquascope basins at 48.85 2.35                    # the catchment of any point: area, climate, land cover, soils, dams (BasinATLAS)
aquascope basins similar 25.04 121.56             # gauged basins whose catchments look most like this point's (ungauged-site donors)
aquascope basins regionalize 52.29 -3.51          # estimated flow regime of an ungauged point from those donors, with the leave-one-out skill
aquascope assess 51.415 -0.308 --problem flood_risk   # what can be answered here: gauges in reach, catchment, which methods the record supports
aquascope caravan export --source uk_ea --out caravan_gb   # a Caravan-format large-sample dataset from the archive
aquascope gym run --basin uk_ea/013054a3-670e-49ee-afda-e0865a449197   # HydroGym: calibrate GR4J on a real basin as a gym episode
aquascope mcp                                     # serve the same tools to Claude / Cursor over MCP
aquascope ask "100-year flood of the Seine at Paris?"   # the analyst: tools + a cited Markdown report
aquascope ingest agency_export.csv --unit cfs     # any CSV/Excel -> clean daily series + QA report
aquascope collect --source usgs --days 365
aquascope collect --source wapor --bbox 30.5,29.8,31.1,30.2 --variable RET --start-date 2026-04-01

# Hydrological analysis
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv --method eckhardt

# Agriculture planning
aquascope agri plan --crop maize --planting-date 2026-04-01 --lat 30.0 --lon 31.25

# AI recommendation + natural-language problem solving
aquascope recommend --parameters DO,BOD5,COD --goal "pollution trend detection"
aquascope solve "Design flow for a road crossing, 100-year return period" --lat 51.415 --lon -0.308

# Interactive Streamlit dashboard — multipage workspace with 21 live sources,
# smart auto-insights, and fully interactive Plotly charts
aquascope dashboard

# Shell tab-completion
eval "$(aquascope completion bash)"   # add this to ~/.bashrc (or .zshrc / config.fish)
```

Run `aquascope --help` for the full command list.

---

## 🌍 Data sources at a glance

29 data collectors spanning four regions (highlights below, full list in the [docs](docs/data_sources.md)):

- 🌎 **Americas** — USGS (streamflow + WQ), NOAA NWPS (US streamflow), Water Quality Portal (400+ agencies), CAMELS-CL (Chile), CAMELS-BR (Brazil)
- 🌍 **Europe** — EU Water Framework Directive, Copernicus ERA5, France Hub'Eau, Germany PEGELONLINE, England's Environment Agency
- 🌏 **Asia-Pacific** — Taiwan MOENV / WRA / Civil IoT / DataGov, Japan MLIT, Korea WAMIS, India WRIS
- 🌐 **Global** — GEMStat (170 countries), UN SDG 6, OpenMeteo, FAO AQUASTAT, FAO WaPOR, GRDC (river discharge)

Full details, endpoints, and API-key requirements: [docs/data_sources.md](docs/data_sources.md). Want to add your country's water service? See [adding a data source](docs/guides/adding_data_source.md).

---

## 🧪 Scientifically validated

- **1,000+ tests** — covering every collector, hydrology method, and pipeline (spatial and ARIMA tests require the optional `[all]` / `[ml]` extras)
- **CAMELS benchmark** — a 10-catchment validation subset of the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels) ships with the repo at `data/camels_benchmark/` and runs as part of CI
- **Every method cited** — equations, decision trees, and DOI references for all 26 methodologies live in the [theory guide](docs/theory.md)
- **JOSS paper in preparation** — see [`paper.md`](paper.md) and [`paper.bib`](paper.bib)

---

## 📚 Documentation

| Resource | What it covers |
| :--- | :--- |
| [Features](docs/features.md) | Full capability list — hydrology, agriculture, ML, spatial, I/O |
| [Data sources](docs/data_sources.md) | All 29 sources, endpoints, API-key requirements |
| [Theory guide](docs/theory.md) | Equations, DOI citations, decision trees for every method |
| [Methodology matrix](docs/methodology_matrix.md) | When to use which method |
| [Architecture](docs/guides/architecture.md) | How AquaScope is structured internally |
| [FAQ](docs/faq.md) · [Troubleshooting](docs/troubleshooting.md) | Common questions and fixes |
| [Use cases](docs/use_cases.md) | Real-world applications and case studies |
| [HydroGym](docs/gym.md) | A gym-style calibration environment over real basins, with baselines and a leaderboard |
| [Integration guides](docs/integration_guides/) | xarray, QGIS, R interoperability |
| [Contributing](CONTRIBUTING.md) | How to add a data source, methodology, or test |

---

## 🤝 Contributing

We welcome contributions from the global water and agriculture research community. Highest-impact contributions right now:

- **New data source collectors** — your country / region
- **New research methodologies** — expand the AI recommender
- **New crop coefficients** — extend the FAO Kc table
- **Jupyter tutorials** and validation studies — compare against HEC-SSP, R packages, etc.

### 📌 Where to start

📍 **[Data sources wanted — help us map every country's water data 🌍](https://github.com/Rekin226/aquascope/issues/11)** — our pinned meta-issue. Want your country in AquaScope? Start here.

New contributor? These [`good first issue`](https://github.com/Rekin226/aquascope/labels/good%20first%20issue)s are scoped with clear acceptance criteria — just comment to claim one:

| Area | Open issues |
| :--- | :--- |
| 🌍 **New data collectors** | [Brazil](https://github.com/Rekin226/aquascope/issues/17) · [Canada](https://github.com/Rekin226/aquascope/issues/18) · [South Africa](https://github.com/Rekin226/aquascope/issues/20) · [Australia](https://github.com/Rekin226/aquascope/issues/4) |
| 🌾 **Agriculture** | [Kc for millet/cassava/chickpea](https://github.com/Rekin226/aquascope/issues/21) · [Kc for sorghum/groundnut/sugar beet](https://github.com/Rekin226/aquascope/issues/5) |
| 📈 **Methodologies** | [SPEI drought index](https://github.com/Rekin226/aquascope/issues/23) · [Budyko framework](https://github.com/Rekin226/aquascope/issues/24) |
| 📊 **Visualization** | [interactive Plotly hydrograph](https://github.com/Rekin226/aquascope/issues/25) · [double-mass curve](https://github.com/Rekin226/aquascope/issues/26) |
| 💻 **CLI** | [`--output` to JSON/CSV](https://github.com/Rekin226/aquascope/issues/27) · [shell completion](https://github.com/Rekin226/aquascope/issues/28) |
| 📚 **Docs & tutorials** | [Colab/Binder badges](https://github.com/Rekin226/aquascope/issues/29) · [groundwater notebook](https://github.com/Rekin226/aquascope/issues/30) · [agri irrigation notebook](https://github.com/Rekin226/aquascope/issues/6) · [translate the docs (zh/fr/ja)](https://github.com/Rekin226/aquascope/issues/31) |
| 🧪 **Code quality & tests** | [type annotations](https://github.com/Rekin226/aquascope/issues/32) |

Browse the [full issue list](https://github.com/Rekin226/aquascope/issues) or vote on what to build next in [Discussions → Ideas](https://github.com/Rekin226/aquascope/discussions/categories/ideas).

See [CONTRIBUTING.md](CONTRIBUTING.md), the [adding a data source](docs/guides/adding_data_source.md) guide, and the [adding a methodology](docs/guides/adding_methodology.md) guide.

### 🪜 The contributor ladder

We want contributors to grow, not vanish after one PR. There's a clear path: start with a [`good first issue`](https://github.com/Rekin226/aquascope/labels/good%20first%20issue), then graduate to a [`good second issue`](https://github.com/Rekin226/aquascope/labels/good%20second%20issue) (a bigger self-contained piece that builds on what you learned), and after a few PRs in one area we'll invite you to help triage and review. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for details.

## 🙌 Contributors

Thanks to these wonderful people who make AquaScope possible ([emoji key](CONTRIBUTORS.md#contribution-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Rekin226"><img src="https://github.com/Rekin226.png?s=100" width="100px;" alt="Abdoul Rachid Ouedraogo"/><br /><sub><b>Abdoul Rachid Ouedraogo</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Rekin226" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Rekin226" title="Documentation">📖</a> <a href="#maintenance-Rekin226" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/vaishnavidesai09"><img src="https://github.com/vaishnavidesai09.png?s=100" width="100px;" alt="Vaishnavi Desai"/><br /><sub><b>Vaishnavi Desai</b></sub></a><br /><a href="#plugin-vaishnavidesai09" title="Plugin/utility libraries">🔌</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Karthick03219"><img src="https://github.com/Karthick03219.png?s=100" width="100px;" alt="Karthick"/><br /><sub><b>Karthick</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Karthick03219" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/sagiB74"><img src="https://github.com/sagiB74.png?s=100" width="100px;" alt="sagiB74"/><br /><sub><b>sagiB74</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=sagiB74" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/laishettikarthik-tech"><img src="https://github.com/laishettikarthik-tech.png?s=100" width="100px;" alt="Karthik Laishetti"/><br /><sub><b>Karthik Laishetti</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=laishettikarthik-tech" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/issues?q=author%3Alaishettikarthik-tech" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/adjenk"><img src="https://github.com/adjenk.png?s=100" width="100px;" alt="Adam Jenkins"/><br /><sub><b>Adam Jenkins</b></sub></a><br /><a href="#plugin-adjenk" title="Plugin/utility libraries">🔌</a> <a href="https://github.com/Rekin226/aquascope/commits?author=adjenk" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=adjenk" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/widjajs"><img src="https://github.com/widjajs.png?s=100" width="100px;" alt="Steven Widjaja"/><br /><sub><b>Steven Widjaja</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=widjajs" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/sairajkasam"><img src="https://github.com/sairajkasam.png?s=100" width="100px;" alt="Sai Raj Kasam"/><br /><sub><b>Sai Raj Kasam</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=sairajkasam" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/safiashaik04"><img src="https://github.com/safiashaik04.png?s=100" width="100px;" alt="safiashaik04"/><br /><sub><b>safiashaik04</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=safiashaik04" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/navaneethsankar07"><img src="https://github.com/navaneethsankar07.png?s=100" width="100px;" alt="Navaneeth Sankar"/><br /><sub><b>Navaneeth Sankar</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=navaneethsankar07" title="Documentation">📖</a> <a href="https://github.com/Rekin226/aquascope/commits?author=navaneethsankar07" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/taran-dev4u"><img src="https://avatars.githubusercontent.com/u/78680216?v=4?s=100" width="100px;" alt="Taran"/><br /><sub><b>Taran</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=taran-dev4u" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=taran-dev4u" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aobaruwa"><img src="https://avatars.githubusercontent.com/u/28014016?v=4?s=100" width="100px;" alt="Ahmed Baruwa"/><br /><sub><b>Ahmed Baruwa</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=aobaruwa" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=aobaruwa" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/AB1775"><img src="https://avatars.githubusercontent.com/u/66264218?v=4?s=100" width="100px;" alt="Anthony"/><br /><sub><b>Anthony</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=AB1775" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=AB1775" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/JamesBoardman27"><img src="https://avatars.githubusercontent.com/u/77696811?v=4?s=100" width="100px;" alt="James Boardman"/><br /><sub><b>James Boardman</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Documentation">📖</a> <a href="https://github.com/Rekin226/aquascope/issues?q=author%3AJamesBoardman27" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/khyahahati"><img src="https://avatars.githubusercontent.com/u/132439126?v=4?s=100" width="100px;" alt="Khyati Tiwari"/><br /><sub><b>Khyati Tiwari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=khyahahati" title="Code">💻</a> <a href="#data-khyahahati" title="Data">🔣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Prakshal0809"><img src="https://avatars.githubusercontent.com/u/116380035?v=4?s=100" width="100px;" alt="PRAKSHAL BHAVINKUMAR BHANDARI"/><br /><sub><b>PRAKSHAL BHAVINKUMAR BHANDARI</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Prakshal0809" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Osheun"><img src="https://avatars.githubusercontent.com/u/138526540?v=4?s=100" width="100px;" alt="Osheun"/><br /><sub><b>Osheun</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://dipakchaudhari.me"><img src="https://avatars.githubusercontent.com/u/111210939?v=4?s=100" width="100px;" alt="Dipak Chaudhari"/><br /><sub><b>Dipak Chaudhari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=dchaudhari7177" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=dchaudhari7177" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Sanchar127"><img src="https://avatars.githubusercontent.com/u/143952019?v=4?s=100" width="100px;" alt="Sanchar127"/><br /><sub><b>Sanchar127</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Sanchar127" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Sanchar127" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="http://harikp.com"><img src="https://avatars.githubusercontent.com/u/64578610?v=4?s=100" width="100px;" alt="hari"/><br /><sub><b>hari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Mr-Neutr0n" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/leatke"><img src="https://avatars.githubusercontent.com/u/147705788?v=4?s=100" width="100px;" alt="leatke"/><br /><sub><b>leatke</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=leatke" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/taliapulsifer"><img src="https://avatars.githubusercontent.com/u/70988138?v=4?s=100" width="100px;" alt="Talia Pulsifer"/><br /><sub><b>Talia Pulsifer</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=taliapulsifer" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=taliapulsifer" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

Your first merged PR puts you on this board, every kind of contribution counts. See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## 📜 Citation

If you use AquaScope in your research, please cite:

```bibtex
@software{aquascope2026,
  title   = {AquaScope: Open-Source Water Data Aggregation, Hydrological Analysis, and Agricultural Water Management Toolkit},
  author  = {Ouédraogo, Abdoul Rachid},
  year    = {2026},
  url     = {https://github.com/Rekin226/aquascope},
  version = {0.14.0},
  doi     = {10.5281/zenodo.21903143},
  license = {MIT}
}
```

Machine-readable metadata lives in [CITATION.cff](CITATION.cff); GitHub's "Cite this
repository" button renders it in APA and BibTeX. Every tagged release is archived on
Zenodo; `10.5281/zenodo.21903143` is the concept DOI that always resolves to the latest
version (v0.13.0 is [10.5281/zenodo.22152064](https://doi.org/10.5281/zenodo.22152064)).

## 📄 License

MIT — see [LICENSE](LICENSE).
