<div align="center">

<img src="docs/assets/logo.svg" alt="AquaScope logo" width="160"/>

# AquaScope Hydrology

**Explore river records and run reproducible water analyses in your browser.**

Find a gauge, inspect its usable record, and export data, figures and methods.
No installation; core Explorer workflows need no API key.

[![CI](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/aquascope/actions/workflows/ci.yml)
[![Pyodide](https://github.com/Rekin226/aquascope/actions/workflows/pyodide-smoke.yml/badge.svg)](https://github.com/Rekin226/aquascope/actions/workflows/pyodide-smoke.yml)
[![PyPI version](https://img.shields.io/pypi/v/aquascope.svg?color=blue&cacheSeconds=300&v=2)](https://pypi.org/project/aquascope/)
[![Python](https://img.shields.io/pypi/pyversions/aquascope.svg?color=informational&cacheSeconds=300&v=2)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21903143.svg)](https://doi.org/10.5281/zenodo.21903143)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-3200%2B%20passing-brightgreen.svg)](#)
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

AquaScope unifies **37 global water-data sources** behind one Python schema, then layers a full scientific computing stack on top — from **flood-frequency methods** to **FAO-56 crop water requirements** — wrapped in an AI engine that scores **27 research methodologies** against your dataset and auto-executes **26 analysis pipelines**. Regression checks include the CAMELS benchmark with 3,200+ tests across the project.
The daily benchmark inputs are synthetic; flood benchmarks also use observed USGS annual peaks.
See [validation scope](docs/validation_scope.md) for comparators and limitations.

---

## 🌍 Try it without installing anything

**[Open AquaScope Explorer](https://rekin226-aquascope-explorer.static.hf.space/)**.
Start with one task:

- **Click the map:** a gauge, a river or any place answers in a small card right there: what it is, today
  against normal, the last year (or the next 15 days on a river), one number, and Details for the full panel.
- **Find river data:** search a gauge, inspect its actual available period and units, then download CSV,
  or the inputs for HEC-HMS, HEC-RAS, HEC-SSP, SWMM, MODFLOW 6, Delft-FEWS or Raven (**Export for…**).
- **Explore a worked analysis:** open a recorded study, read its limits and reproduce its plan at another gauge.
- **Analyse my table:** use a sample CSV, check the inferred columns, then replace it with your own data.
- **Watch a river:** press ☆ Watch on a gauge, a reach or a drawn area. The next visit opens on what changed
  since: new data, today's status, a forecast above your threshold, new flood events nearby. A gauge with a live
  record also has an Atom feed. No account; the list stays in your browser.
- **Follow a river:** the river network is on the globe from the start, the big rivers first and the small
  streams as you zoom, with a moving dash showing which way the water goes. Click anywhere and the point snaps to
  its river (or says no stream is near), and the map lights up everything that drains to that reach and its way to
  the sea. The River tab shows 86 years of simulated daily flow for that reach, its return periods and flow-duration curve, and
  traces it to the sea past the gauges and dams on the way, naming the countries it crosses. Simulated, and
  labelled so.
- **Now and next:** a gauge's **Now** tab says where today's flow sits against normal for the date, and plots the
  next 15 days from GEOGLOWS and GloFAS with the return-period lines, corrected to the gauge's own record with the
  correction's skill beside it. The map can colour the gauges by today against normal. Model forecasts, labelled so.
- **See floods ahead:** the globe opens with the river reaches the GEOGLOWS forecast expects to reach their 2-year
  flow in the next 15 days glowing in their return-period class, the gauges on them pulsing. Play the 15 days with
  the time bar, click a reach for its peak and day. A model forecast, not an official warning.
- **Read the month:** **Bulletin** (in Tools) opens last month's state of the rivers: every Archive gauge against
  the same month in its other years, by country and river basin, with the new records. The map can colour the
  gauges by last month's status.
- **Read a place:** click anywhere and open **Context** for what flooded there before, how often the ground has
  been water since 1984, modelled flood depth, dams, soil, evaporation and the nearest rain gauge, each with its source.

Catalog coverage varies by agency and variable. A station on the map is not a guarantee
of accessible observations or a sufficiently long record. Explorer fetches the full record
by default (or the last 40 or 20 years, your choice) and shows the period it actually
analyzes; modelled discharge is distinguished from gauge observations.

**The state of the world's rivers, on the globe:** the map opens on every river basin
coloured from much below to much above normal for the newest month (GEOGLOWS v2's
monthly HydroSOS map), with the gauges coloured by today against normal where they
report. Press play to watch droughts and floods move across continents, month by
month back to 1990 ([details](docs/explorer.md#world-river-status)).

**Time on the map:** one date drives the river status and the NASA satellite, rain, soil
moisture, snow and water storage layers. Play a range, click a day on a hydrograph to see
the map on that day, swipe-compare two dates, and save the range as a GIF. The date is in
the link ([details](docs/explorer.md#time-on-the-map)).

**Study** guides you from a question through a plan you approve to a report and export
bundle: Word, Excel, figures, notebook, findings and study YAML. Core studies run without
an API key. Optional model setup is available when you choose to use it. Beyond the design
flow, a study can test whether the flood is changing, calibrate GR4J at the gauge and run
"what if" rainfall and warming scenarios, or carry seven CMIP6 models through it to 2050
([advanced studies](docs/advanced_studies.md)).

The [open archive](https://huggingface.co/datasets/Rekin226/aquascope-gauges) supplies
station catalogs and mirrored observations; inspect observation refresh status separately
from catalog health. Exports also work with [R, QGIS, DuckDB and Julia](docs/readers.md).
See the [executable Python quickstart](docs/getting_started.md) and
[validation scope](docs/validation_scope.md) before using a flood result in a decision.

Prefer an assistant? `pip install "aquascope[mcp]"` then `claude mcp add aquascope -- aquascope mcp` gives Claude (or any
MCP client) `find_stations`, `get_timeseries`, `analyze_station` and `flood_frequency` over the same catalog and methods
([docs](docs/mcp.md)).

## 🧑‍🔬 Run a study on your machine

```bash
pip install "aquascope[studio,basins]"   # the Word report, workbook, figures, notebook and the catchment step
aquascope studio
```

That is all. The Studio asks where (a gauge name like `Fish River Fort Kent`, a station id like `USGS-01013500`,
or `lat, lon`) and what you want to know ("Is flooding here getting worse?"). Then it asks only what the study still
needs, one pick-list question at a time with the reason (a trend question: which period; a design question: which
return period), shows the plan, and runs only when you approve it (`e` edits a step, e.g. `s3.return_period=200`). The bundle lands in
`./studio-<id>/`: a technical report and a short memo (`report.docx`, `memo.docx`, and `report.html`, which prints
to PDF), `workbook.xlsx`, `study.ipynb`, `figures/` (300 dpi PNG and SVG), `findings.json` and `study.yaml`, which
re-runs the whole study with `aquascope run study.yaml`. `--style style.yaml` puts your organisation, project and
names on the cover. Then `aquascope studio ./studio-<id>/` opens it on the Study Desk to revise it: leave out a suspect flood, change the return
period or the distribution, see how far the answer moves, add review comments and sign it off, with every change
recorded as a revision. After the report, ask a follow-up (another gauge, a
trend, the flow duration curve) and the bundle is updated.

It needs no key: the playbooks plan and the templates write. If you have one, the Studio asks: paste it
(hidden), it is checked with one short request, and the model writes the brief, the plan and the prose, with a $1
spend ceiling. No key yet? Groq's free tier works, and the Studio links you to it. It can remember the key for next
time (`~/.config/aquascope/keys.json`, readable only by you), and a key already in your environment
(`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, ...) is offered instead. The numbers are the same either way: they come from the tools, and a
sentence whose number is in no result is dropped. One line, no questions:

```bash
aquascope studio "Is flooding getting worse?" --at USGS-01013500 --provider anthropic --max-usd 1 --yes
```

On macOS with Homebrew or python.org Python, `pip install` may refuse ("externally managed environment"); make a
virtual environment first: `python3 -m venv .venv && source .venv/bin/activate`. More in [docs/studio.md](docs/studio.md).

## ✨ What you can do

- 🌊 **Pull water data** from USGS, NOAA NWPS, Colorado DWR/CDSS, US Water Quality Portal, England's Environment Agency, France Hub'Eau, Germany PEGELONLINE, Ireland OPW, Greece Hydroscope and OpenHi.net, Poland IMGW-PIB, EU WFD, Taiwan MOENV/WRA/CWA/Civil IoT/DataGov, Japan MLIT, Korea WAMIS, India WRIS, South Africa DWS, Australia BOM, Brazil ANA Hidroweb, CAMELS-CL and CAMELS-BR, GRDC, GEMStat, Copernicus ERA5, OpenMeteo, FAO AQUASTAT, FAO WaPOR and UN SDG 6 — **one unified Python API**.
- 🏞️ **Treat rivers as objects**: snap any point to its GEOGLOWS v2 river reach (about 6.8 million worldwide), read that reach's simulated daily flow since 1940 with return periods, flow-duration curve and monthly regime (labelled modelled), and trace it downstream to the sea with the gauges and dams it passes and the countries it crosses, and see whether dams upstream regulate it (Global Dam Watch), or list every reach that drains to it and the reaches to its outlet. `aquascope river snap|record|area|trace|dams|upstream|downstream`, the MCP tools, the Explorer's River tab, and the Studio's ungauged studies all use the same functions.
- 🪜 **Grade the global models at a gauge**: GEOGLOWS v2, GloFAS, NWM v3 (US) and Google's Flood Hub reanalysis set against the gauge's own record, with KGE and its parts, bias, the error at the 2-, 10- and 100-year flows, a grade from A to D and one sentence on where they disagree. `aquascope evidence skill`, the MCP tool `model_skill`, the Explorer's Evidence tab and "Best model skill" map colouring, and a monthly CI table in the Archive; the Studio uses it to say which model to lean on near an ungauged site ([how the grades work](docs/evidence.md)).
- 🔭 **Now and next**: where a gauge's flow sits today against normal for the date (the USGS and WMO HydroSOS classes, at least 10 years behind it), and the next 15 days from GEOGLOWS and GloFAS with return-period thresholds, corrected to the gauge's record by flow-duration quantile mapping with the hindcast skill of that correction. A daily workflow archives the forecasts as issued, so real forecast skill builds up, and another classes the whole global forecast against each reach's return periods every day: the reaches expected to reach their 2-year flow in 15 days (`aquascope warnings`, the MCP tool `flood_warnings`, the Explorer's Floods ahead layer). `aquascope now`, the MCP tools, the Explorer's Now tab and its "Today vs normal" map colouring use the same functions.
- 🗓️ **A monthly state of the rivers**: on the 3rd of each month a workflow places every Archive gauge's monthly mean flow against the same month in its other years (25 days a month, 10 years, HydroSOS classes), rolls it up per country and per BasinATLAS river basin, names the new monthly records and the gauges furthest from normal, and writes a print-ready HTML and Markdown bulletin with a map into the Archive. The summary is written by rules, not a model. `aquascope bulletin`, the MCP tool `status_bulletin` and the Explorer's Bulletin reader and "Last month's status" colouring use the same functions ([docs](docs/bulletin.md)).
- ⭐ **Watch a river**: star gauges, reaches and areas, and get what changed since you last looked: new data and the latest value, today's status class against the one then, the 15-day forecast against a threshold you set (a value or a return period) and new flood events nearby. `aquascope watch ID... --since DATE`, the MCP tool `watch_digest` and the Explorer's "Since you were here" panel use the same function, and a daily workflow writes an Atom feed per live gauge.
- 📈 **Run hydrological analyses** — flood frequency (GEV / LP3 / Gumbel / non-stationary GEV, with separate EMA routines), baseflow separation, rating curves, 22 hydrological signatures.
- 🌾 **Plan agricultural water** — FAO-56 Penman-Monteith ET₀, crop water requirements for 26 crops (olive, grape, citrus and winter wheat resolved by variety and canopy), irrigation scheduling, soil water balance with auto-irrigation.
- 🤖 **Ask the AI engine** — describe your goal in plain English and get a recommended methodology, scored against your dataset profile and auto-executed. LLM enhancement via OpenAI, Groq (free), HuggingFace (free), or local Ollama.
- 🧑‍🔬 **Hand a study to the crew** — `aquascope studio "PROBLEM" --lat --lon`: a Consultant, a Scout, a Methodologist, Analysts, an Interpreter, a Critic and an Author over one workspace, the plan shown before it runs, every step gated (a failed gate fails its step, not the study), every answer graded (established, indicative, screening, not established) with findings that point at the result they rest on, a request for the data that would unlock a question instead of a decline, and the bundle at the end: a technical report and a memo (Word and print-ready HTML) that lead with the answer and its grade, the Excel workbook, publication figures, the notebook, findings.json and study.yaml. Also in the Explorer and over MCP. The [advanced studies](docs/advanced_studies.md) go past the design flow: change points and a nonstationary flood fit, GR4J with a snow store validated on years it never saw, "what if" scenarios, and CMIP6 change factors through the calibrated model.
- 🧭 **Read the context of any place**: `aquascope context LAT LON` (also over MCP and in the Explorer): flood events in the news (Groundsource) and Sentinel-1 radar floods 2014-2024, surface water since 1984 (JRC), modelled flood depth at the 10 to 500-year floods (JRC GloFAS), dams (Global Dam Watch), soil texture and available water (SoilGrids), actual ET (FAO WaPOR) and the nearest NOAA GHCN-Daily rain gauge. Keyless, each line with its licence; rasters are read pixel by pixel from Cloud-Optimized GeoTIFFs in pure Python. The news and radar floods are also rolled into a monthly half-degree grid that the Explorer's globe shows from the start and replays with the time bar (`aquascope context --floods-past`, MCP `flood_events_month`).
- 🛠️ **Hand a record to the engineering tools**: `aquascope export --to hec-ssp` (or `hec-hms`, `hec-ras`, `dss`, `swmm`, `modflow6`, `fews`, `raven`) writes ready inputs from any gauge or CSV, real `.dss` through HEC's own `hecdss`; the same files come from the Explorer, the MCP server and every Studio bundle. Our Bulletin 17C is checked against the published Bulletin 17C examples, honestly: it matches on a plain record and does not yet on one with low outliers ([engineering exports](docs/engineering_exports.md)).
- 📊 **Visualise + report** — 17 plot types, Q-Q / P-P diagnostics, Markdown / HTML reports with embedded figures, threshold alerts (WHO / EPA / EU WFD).
- 🗺️ **Spatial hydrology** — DEM processing, D8 flow direction, watershed delineation, Strahler ordering.

For the full capability list see [docs/features.md](docs/features.md).

## 📊 Why AquaScope

| | AquaScope | HEC-SSP | R `lmom` | Standalone collectors |
| :--- | :---: | :---: | :---: | :---: |
| LP3 / EMA routines (workflow validation required) | ✅ | ✅ | partial | — |
| Non-stationary GEV | ✅ | — | partial | — |
| CMIP6 change factors through a calibrated GR4J, in one study | ✅ | — | — | — |
| Baseflow separation (Lyne-Hollick, Eckhardt) | ✅ | — | — | — |
| FAO-56 Penman-Monteith ET₀ + crop water | ✅ | — | — | — |
| 37 unified data collectors | ✅ | — | — | per-source |
| AI methodology recommender (OpenAI / Groq / HF / Ollama) | ✅ | — | — | — |
| Interactive Streamlit dashboard | ✅ | — | — | — |
| Free, MIT, Python-native | ✅ | partial | ✅ | varies |

---

## ⚡ Install

```bash
pip install aquascope              # core — collectors + hydrology
pip install "aquascope[all]"       # everything — ML, viz, spatial, dashboard
uv tool install "aquascope[all]"   # or as a command-line tool in its own environment
```

To upgrade later, `aquascope update` finds the newest release and upgrades it the way it
was installed (uv, pipx, or the environment's pip); `aquascope update --check` only looks,
and `aquascope --version` says what you have.

Feature-group extras:

```bash
pip install "aquascope[ml]"           # sklearn, xgboost, statsmodels
pip install "aquascope[viz]"          # matplotlib, seaborn, folium
pip install "aquascope[scientific]"   # xarray, netcdf4, h5py
pip install "aquascope[interop]"      # xarray + geopandas (collect as_xarray / as_geodataframe)
pip install "aquascope[spatial]"      # rasterio, geopandas, shapely
pip install "aquascope[dashboard]"    # streamlit
pip install "aquascope[forecast]"     # prophet, torch (for LSTM)
pip install "aquascope[studio]"       # matplotlib, openpyxl, python-docx (the Studio's Word, Excel, figures)
pip install "aquascope[basins]"       # pyogrio, geopandas (BasinATLAS catchments, similar basins)
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

### 3. Collect data from any of the 37 sources

```python
from aquascope import find_stations
from aquascope.collectors import USGSCollector, AquastatCollector, WaPORCollector

# Which gauges measure discharge around Greater London? (USGS, UK EA, Hub'Eau,
# PEGELONLINE, Ireland OPW, Greece (Hydroscope + OpenHi.net) and Taiwan CWA
# expose station catalogs; more coming)
gauges = find_stations(bbox=(-0.5, 51.3, 0.3, 51.7), variable="discharge")
print(gauges[0].name, gauges[0].url)

usgs = USGSCollector()   # keyless; a free api_key=... raises the rate limit
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

AquaScope ships a 43-command CLI (`agri`, `basins`, `caravan`, `eval`, `evidence`, `gym`, `layers`, `playbooks` and `river` carry subcommands) for the most common workflows:

```bash
# Find stations, then collect data
aquascope stations --bbox -0.5,51.3,0.3,51.7 --variable discharge --format geojson
aquascope harvest stations --out archive          # the open gauge catalog (GeoParquet)
aquascope basins at 48.85 2.35                    # the catchment of any point: area, climate, land cover, soils, dams (BasinATLAS)
aquascope basins similar 25.04 121.56             # gauged basins whose catchments look most like this point's (ungauged-site donors)
aquascope basins regionalize 52.29 -3.51          # estimated flow regime of an ungauged point from those donors, with the leave-one-out skill
aquascope river snap 46.948 7.452                 # the river reach at a point (GEOGLOWS v2), or "no stream within 1 km"
aquascope river record --at 46.948 7.452          # that reach's simulated daily flow since 1940: return periods, FDC (modelled)
aquascope river trace --at 46.948 7.452           # follow it to the sea: length, path, the gauges, dams and countries it passes
aquascope river dams --at 46.948 7.452            # the dams upstream of that reach and the degree of regulation
aquascope now --station usgs/USGS-01350000        # today against normal, and the 15-day forecast corrected to the gauge
aquascope bulletin 2026-09 --out bulletin         # last month's state of the rivers: HydroSOS classes, HTML and Markdown
aquascope warnings --bbox -10 35 30 60            # floods ahead: reaches expected to reach their 2-year flow in 15 days
aquascope watch usgs/USGS-01350000 river:230260670 --since 2026-10-01   # what changed since then: data, status, forecast, floods
aquascope assess 51.415 -0.308 --problem flood_risk   # what can be answered here: gauges in reach, catchment, which methods the record supports
aquascope context 51.86 5.95                      # flood history, surface water, flood depth, dams, rain gauge, ET and soil at a place
aquascope caravan export --source uk_ea --out caravan_gb   # a Caravan-format large-sample dataset from the archive
aquascope export --to hec-ssp --station usgs/01134500   # inputs for HEC-HMS/RAS/SSP, SWMM, MODFLOW 6, Delft-FEWS or Raven
aquascope gym run --basin uk_ea/013054a3-670e-49ee-afda-e0865a449197   # HydroGym: calibrate GR4J on a real basin as a gym episode
aquascope layers frames precip --start 2024-05-01 --end 2024-05-20   # a time-lapse of a dated map layer: dates and tile URLs
aquascope map "trace the Nile to the sea" --resolve   # plain words to the Explorer's map actions (keyless rules)
aquascope mcp                                     # serve the same tools to Claude / Cursor over MCP
aquascope ask "100-year flood of the Seine at Paris?"   # the analyst: tools + a cited Markdown report
aquascope ingest agency_export.csv --unit cfs     # any CSV/Excel -> clean daily series + QA report
aquascope collect --source usgs --days 365
aquascope collect --source wapor --bbox 30.5,29.8,31.1,30.2 --variable RET --start-date 2026-04-01

# Hydrological analysis
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv --method eckhardt -o baseflow.json

# Agriculture planning
aquascope agri plan --crop maize --planting-date 2026-04-01 --lat 30.0 --lon 31.25 -o plan.csv

# AI recommendation + natural-language problem solving
aquascope recommend --parameters DO,BOD5,COD --goal "pollution trend detection" -o recommendations.json
aquascope solve "Design flow for a road crossing, 100-year return period" --lat 51.415 --lon -0.308
aquascope studio                                     # the crew: asks where and what, then brief, plan, run, bundle
aquascope studio "Design flow for a road crossing, 100-year, and how sure can we be" --at "Thames Kingston" --out kingston/
aquascope studio kingston/ --exclude-years 2014 --sign checked="A. Name"  # the Study Desk: revise, review, sign
aquascope eval score kingston/                       # how the crew did: gates, Critic, report quality, time, cost
aquascope eval stats studies/ --by model             # many studies at once: grades, gate failures, cost per study
aquascope area-study --bbox=-0.9,51.2,0.3,51.8       # a flood study over every gauge in a box: Q100, flood trends, a regional curve

# Interactive Streamlit dashboard — multipage workspace with 37 live sources,
# smart auto-insights, and fully interactive Plotly charts
aquascope dashboard

# Shell tab-completion
eval "$(aquascope completion bash)"   # add this to ~/.bashrc (or .zshrc / config.fish)
```

Run `aquascope --help` for the full command list.

---

## 🌍 Data sources at a glance

37 data collectors spanning five regions (highlights below, full list in the [docs](docs/data_sources.md)):

- 🌎 **Americas** — USGS (streamflow + WQ), NOAA NWPS (US streamflow), Colorado DWR/CDSS, Water Quality Portal (400+ agencies), CAMELS-CL (Chile), CAMELS-BR and ANA Hidroweb (Brazil)
- 🌍 **Europe** — EU Water Framework Directive, Copernicus ERA5, France Hub'Eau, Germany PEGELONLINE, England's Environment Agency, Ireland OPW, Greece Hydroscope (national archive) and OpenHi.net (live telemetry), Poland IMGW-PIB (daily archive from 1951 and the live network)
- 🌍 **Africa** — South Africa DWS (verified discharge and water level)
- 🌏 **Asia-Pacific** — Taiwan MOENV / WRA / CWA / Civil IoT / DataGov, Japan MLIT, Korea WAMIS, India WRIS, Australia BOM
- 🌐 **Global** — GEMStat (170 countries), UN SDG 6, OpenMeteo, FAO AQUASTAT, FAO WaPOR, GRDC (river discharge)

Full details, endpoints, and API-key requirements: [docs/data_sources.md](docs/data_sources.md). Want to add your country's water service? See [adding a data source](docs/guides/adding_data_source.md).

---

## 🧪 Scientifically validated

- **3,200+ tests** covering every collector, hydrology method, and pipeline (spatial and ARIMA tests require the optional `[all]` / `[ml]` extras)
- **CAMELS benchmark** — a 10-catchment validation subset of the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels) ships with the repo at `data/camels_benchmark/` and runs as part of CI
- **Every method cited** — equations, decision trees, and DOI references for all 27 methodologies live in the [theory guide](docs/theory.md)
- **JOSS paper in preparation** — see [`paper.md`](paper.md) and [`paper.bib`](paper.bib)

---

## 📚 Documentation

| Resource | What it covers |
| :--- | :--- |
| [Features](docs/features.md) | Full capability list — hydrology, agriculture, ML, spatial, I/O |
| [Data sources](docs/data_sources.md) | All 37 sources, endpoints, API-key requirements |
| [Theory guide](docs/theory.md) | Equations, DOI citations, decision trees for every method |
| [Methodology matrix](docs/methodology_matrix.md) | When to use which method |
| [Architecture](docs/guides/architecture.md) | How AquaScope is structured internally |
| [FAQ](docs/faq.md) · [Troubleshooting](docs/troubleshooting.md) | Common questions and fixes |
| [Use cases](docs/use_cases.md) | Real-world applications and case studies |
| [Advanced studies](docs/advanced_studies.md) | Is the flood changing, climate change to 2050, "what if" the rain drops: what runs and what the numbers cannot say |
| [HydroGym](docs/gym.md) | A gym-style calibration environment over real basins, with baselines and a leaderboard |
| [Evaluating studies](docs/evaluation.md) | `aquascope eval`: a study's scorecard and trace, and stats across many runs, read from the bundles |
| [HydroGym benchmark](docs/hydrogym.md) | Hydrology agents scored on real sites: task outcomes, plan quality against expert plans, and the report the user receives |
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
| 💻 **CLI** | `--output` to JSON/CSV · [shell completion](https://github.com/Rekin226/aquascope/issues/28) |
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
      <td align="center" valign="top" width="20%"><a href="https://github.com/JamesBoardman27"><img src="https://avatars.githubusercontent.com/u/77696811?v=4?s=100" width="100px;" alt="James Boardman"/><br /><sub><b>James Boardman</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=JamesBoardman27" title="Documentation">📖</a> <a href="https://github.com/Rekin226/aquascope/issues?q=author%3AJamesBoardman27" title="Bug reports">🐛</a> <a href="#infra-JamesBoardman27" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-JamesBoardman27" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/khyahahati"><img src="https://avatars.githubusercontent.com/u/132439126?v=4?s=100" width="100px;" alt="Khyati Tiwari"/><br /><sub><b>Khyati Tiwari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=khyahahati" title="Code">💻</a> <a href="#data-khyahahati" title="Data">🔣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Prakshal0809"><img src="https://avatars.githubusercontent.com/u/116380035?v=4?s=100" width="100px;" alt="PRAKSHAL BHAVINKUMAR BHANDARI"/><br /><sub><b>PRAKSHAL BHAVINKUMAR BHANDARI</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Prakshal0809" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Osheun"><img src="https://avatars.githubusercontent.com/u/138526540?v=4?s=100" width="100px;" alt="Osheun"/><br /><sub><b>Osheun</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Osheun" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://dipakchaudhari.me"><img src="https://avatars.githubusercontent.com/u/111210939?v=4?s=100" width="100px;" alt="Dipak Chaudhari"/><br /><sub><b>Dipak Chaudhari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=dchaudhari7177" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=dchaudhari7177" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Sanchar127"><img src="https://avatars.githubusercontent.com/u/143952019?v=4?s=100" width="100px;" alt="Sanchar127"/><br /><sub><b>Sanchar127</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Sanchar127" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Sanchar127" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="http://harikp.com"><img src="https://avatars.githubusercontent.com/u/64578610?v=4?s=100" width="100px;" alt="hari"/><br /><sub><b>hari</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Mr-Neutr0n" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Mr-Neutr0n" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/leatke"><img src="https://avatars.githubusercontent.com/u/147705788?v=4?s=100" width="100px;" alt="leatke"/><br /><sub><b>leatke</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=leatke" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=leatke" title="Documentation">📖</a> <a href="#research-leatke" title="Research">🔬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/taliapulsifer"><img src="https://avatars.githubusercontent.com/u/70988138?v=4?s=100" width="100px;" alt="Talia Pulsifer"/><br /><sub><b>Talia Pulsifer</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=taliapulsifer" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=taliapulsifer" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/mikemikimike"><img src="https://avatars.githubusercontent.com/u/186855910?v=4?s=100" width="100px;" alt="mikemikimike"/><br /><sub><b>mikemikimike</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=mikemikimike" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=mikemikimike" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jddrtn"><img src="https://avatars.githubusercontent.com/u/202679891?v=4?s=100" width="100px;" alt="Jade"/><br /><sub><b>Jade</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=jddrtn" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=jddrtn" title="Tests">⚠️</a> <a href="#data-jddrtn" title="Data">🔣</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/be-student"><img src="https://avatars.githubusercontent.com/u/80899085?v=4?s=100" width="100px;" alt="송은우"/><br /><sub><b>송은우</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=be-student" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=be-student" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=be-student" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/mohanasrujana"><img src="https://avatars.githubusercontent.com/u/59142214?v=4?s=100" width="100px;" alt="Satya Srujana Pilli"/><br /><sub><b>Satya Srujana Pilli</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=mohanasrujana" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=mohanasrujana" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/itsnevu"><img src="https://avatars.githubusercontent.com/u/129736887?v=4?s=100" width="100px;" alt="0x"/><br /><sub><b>0x</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=itsnevu" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://travelsafepilot.com"><img src="https://avatars.githubusercontent.com/u/241781992?v=4?s=100" width="100px;" alt="bazinga0027-gif"/><br /><sub><b>bazinga0027-gif</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=bazinga0027-gif" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=bazinga0027-gif" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=bazinga0027-gif" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/HarshRajSinghania"><img src="https://avatars.githubusercontent.com/u/40535627?v=4?s=100" width="100px;" alt="Harsh Raj Singhania"/><br /><sub><b>Harsh Raj Singhania</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=HarshRajSinghania" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=HarshRajSinghania" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://bharani-kudala.me/"><img src="https://avatars.githubusercontent.com/u/121555407?v=4?s=100" width="100px;" alt="Kudala Bharani Kumar Reddy"/><br /><sub><b>Kudala Bharani Kumar Reddy</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=kudala-bharani" title="Tests">⚠️</a> <a href="https://github.com/Rekin226/aquascope/commits?author=kudala-bharani" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Berserker-GM"><img src="https://avatars.githubusercontent.com/u/229895835?v=4?s=100" width="100px;" alt="Berserker-GM"/><br /><sub><b>Berserker-GM</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Berserker-GM" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=Berserker-GM" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="https://galabavamsi.github.io/portfolio/"><img src="https://avatars.githubusercontent.com/u/51828882?v=4?s=100" width="100px;" alt="GALABA VAMSI"/><br /><sub><b>GALABA VAMSI</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=Galabavamsi" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/didemkastan"><img src="https://avatars.githubusercontent.com/u/273810938?v=4?s=100" width="100px;" alt="Didem KAŞTAN"/><br /><sub><b>Didem KAŞTAN</b></sub></a><br /><a href="https://github.com/Rekin226/aquascope/commits?author=didemkastan" title="Code">💻</a> <a href="https://github.com/Rekin226/aquascope/commits?author=didemkastan" title="Tests">⚠️</a></td>
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
  version = {0.26.0},
  doi     = {10.5281/zenodo.23246553},
  license = {MIT}
}
```

Machine-readable metadata lives in [CITATION.cff](CITATION.cff); GitHub's "Cite this
repository" button renders it in APA and BibTeX. Every tagged release is archived on
Zenodo; `10.5281/zenodo.21903143` is the concept DOI that always resolves to the latest
version (v0.26.0 is [10.5281/zenodo.23246553](https://doi.org/10.5281/zenodo.23246553)).

## 📄 License

MIT — see [LICENSE](LICENSE).
