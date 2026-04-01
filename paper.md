---
title: 'AquaScope: An Open-Source Python Toolkit for Unified Water Data Aggregation, Hydrological Analysis, and AI-Powered Research Methodology Recommendations'
tags:
  - Python
  - hydrology
  - water quality
  - open data
  - artificial intelligence
  - flood frequency analysis
  - FAO-56
  - evapotranspiration
authors:
  - name: AquaScope Contributors
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researchers
    index: 1
date: 2 July 2025
bibliography: paper.bib
---

# Summary

AquaScope is an open-source Python toolkit (v0.3.0, MIT license) that unifies water
data collection from 12 global sources, comprehensive hydrological and statistical
analysis, agricultural water management, and AI-powered research methodology
recommendations into a single, coherent package. It addresses a persistent challenge
in water resources research: the fragmentation of data access, analytical methods, and
tooling across disparate software ecosystems. AquaScope normalises heterogeneous data
into unified Pydantic schemas, provides over 40 analytical methods spanning flood
frequency analysis, baseflow separation, extreme value theory, and FAO-56
evapotranspiration, and includes a knowledge-base-driven AI engine that recommends
appropriate research methodologies based on dataset characteristics. The toolkit is
available at <https://github.com/Rekin226/aquascope>.

# Statement of Need

Water resources research requires practitioners to navigate a fragmented landscape of
data sources, analytical techniques, and software tools. Streamflow records from the
U.S. Geological Survey (USGS) National Water Information System [@USGS_NWIS], water
quality observations from the Water Quality Portal (WQP), satellite-derived
evapotranspiration from FAO WaPOR, climate reanalysis from Copernicus ERA5, and
sustainability indicators from the UN SDG 6 database each expose different APIs, data
formats, and access patterns. Researchers typically write bespoke scripts for each
source, introducing inconsistencies and impeding reproducibility.

Beyond data access, the analytical methods themselves span multiple domains.
Flood frequency analysis following Bulletin 17C [@England2019] requires L-moment
estimation [@Hosking1997] and the Expected Moments Algorithm (EMA) for censored data.
Agricultural water management demands FAO-56 Penman-Monteith reference
evapotranspiration calculations [@Allen1998]. Change-point detection involves choosing
among PELT [@Killick2012], CUSUM, and Pettitt [@Pettitt1979] algorithms depending on
data characteristics. Multivariate dependence modelling requires copula families
[@Nelsen2006] with appropriate selection criteria. Each of these methods is often
available only in specialised packages with incompatible interfaces.

Existing open-source tools address subsets of these needs. Packages such as
`hydrostats` provide statistical metrics for streamflow comparison, `HydroTools`
offers access to select U.S. hydrological data services, and `pySTEPS` focuses on
probabilistic precipitation nowcasting. However, no single toolkit integrates
multi-source data collection, a comprehensive hydrological analysis suite, agricultural
water management, advanced statistical and machine learning methods, and intelligent
methodology guidance.

AquaScope fills this gap by providing an end-to-end workflow—from raw data ingestion
through analysis to methodology recommendation—in a single, well-tested Python package
with a unified API. It targets hydrologists, environmental engineers, agricultural
scientists, and water resources researchers who need to combine data from multiple
sources and apply appropriate analytical methods without assembling a patchwork of
incompatible tools.

# Key Features

## Data Aggregation from 12 Global Sources

AquaScope implements collectors for 12 water data sources, each subclassing a common
`BaseCollector` and normalising responses into shared Pydantic schemas
(`WaterQualitySample`, `WaterLevelReading`, `ReservoirStatus`, `SDG6Indicator`). The
supported sources are: USGS NWIS [@USGS_NWIS], Taiwan MOENV, Taiwan WRA, Taiwan Civil
IoT (SensorThings API), UN SDG 6, GEMStat (covering 170+ countries), the Water
Quality Portal (aggregating 400+ U.S. agencies), Open-Meteo, Copernicus CDS (ERA5
reanalysis and climate projections), FAO AQUASTAT (country-level water withdrawal),
and FAO WaPOR (satellite evapotranspiration). An `httpx`-based HTTP client
[@HTTPX2024] provides caching, automatic retries with exponential backoff, and rate
limiting across all collectors.

## Hydrological Analysis Suite

The hydrology module implements flood frequency analysis using Generalised Extreme
Value (GEV), Log-Pearson Type III (LP3), and Gumbel distributions, with L-moment
parameter estimation [@Hosking1997] and Bulletin 17C procedures including EMA for
censored data [@England2019]. Non-stationary GEV models with time-varying parameters
[@Coles2001] and regional frequency analysis with discordancy and heterogeneity
measures are also supported. Baseflow separation uses the Lyne–Hollick [@Lyne1979]
and Eckhardt [@Eckhardt2005] recursive digital filters. Additional capabilities
include flow duration curves with Weibull plotting positions, recession analysis,
22 hydrological signatures (magnitude, variability, timing, flashiness), and
power-law rating curve fitting with shift detection. Diagnostic tools include Q-Q
and P-P plots, return level analysis, and leave-one-out cross-validation with
coverage probability assessment.

## Agricultural Water Management

AquaScope implements the complete FAO-56 Penman-Monteith reference evapotranspiration
(ET₀) methodology [@Allen1998], including all intermediate calculations (psychrometric
constant, saturation vapour pressure, net radiation). A Hargreaves temperature-only
alternative is provided for data-sparse contexts. Crop water requirements are computed
for 20 crops using FAO-56 crop coefficients ($K_c$) across initial, mid-season, and
late-season growth stages. Irrigation scheduling accounts for effective rainfall, net
and gross irrigation demand, and application efficiency. A daily soil water balance
model tracks root-zone depletion with automatic irrigation triggers.

## Advanced Statistical and Machine Learning Methods

The statistical analysis capabilities include Bayesian uncertainty quantification via
conjugate linear regression and Markov chain Monte Carlo sampling [@Gelman2013],
copula dependence modelling with four families (Gaussian, Clayton, Gumbel, Frank) and
AIC-based selection [@Nelsen2006], change-point detection using PELT [@Killick2012],
CUSUM, Pettitt [@Pettitt1979], and binary segmentation algorithms, and extreme value
analysis with L-moment estimation and non-stationary GEV models [@Coles2001].
Trend detection uses the Mann-Kendall test [@Mann1945; @Kendall1975] and the
Standardised Precipitation Index [@McKee1993] for drought characterisation.

Machine learning models include ARIMA/SARIMA, Prophet, Random Forest, XGBoost, and
LSTM neural networks. Ensemble methods support weighted averaging, stacking, and
adaptive strategies. Transfer learning enables model reuse across catchments.

## AI-Powered Methodology Recommendations

A knowledge base of 26 research methodologies—spanning statistical analysis, machine
learning, time-series forecasting, process engineering, spatial analysis, hydrological
modelling, and policy assessment—drives an AI recommendation engine. Given a dataset
profile (computed automatically from the data), the engine scores each methodology
using rule-based heuristics matched against data characteristics (sample size,
parameter types, temporal extent, spatial coverage). An optional large language model
(LLM) integration provides natural-language reasoning about methodology suitability.
A HydroAgent interface enables conversational problem-solving for users who prefer
natural language interaction over command-line operation.

## Scientific I/O and Interoperability

AquaScope reads and writes OGC WaterML 2.0 [@WaterML2012] for standards-compliant
data exchange, HEC-DSS and HEC-RAS formats for U.S. Army Corps of Engineers
interoperability, EPA SWMM input files for urban stormwater modelling, and NetCDF and
HDF5 for multidimensional scientific data. GeoJSON export supports spatial
visualisation and GIS workflows.

## Spatial Hydrology

The spatial module provides DEM processing, D8 flow direction and accumulation
routing, automated watershed delineation, and Strahler stream ordering. These tools
enable catchment-scale analysis directly from elevation data without external GIS
software.

## Visualisation and Reporting

Sixteen plot functions cover time-series, box plots, heatmaps, spatial maps (via
Folium), flow duration curves, and hydrographs. Diagnostic panels provide Q-Q, P-P,
and return level plots for distribution fitting validation. An interactive Streamlit
dashboard with seven pages enables exploratory analysis without writing code.
Automated Markdown and HTML report generation with embedded figures supports
reproducible documentation.

# Design and Architecture

AquaScope follows a pipeline architecture:

```
Collectors (fetch + normalise) → Pydantic Schemas → Analysis → AI Recommender → Pipelines
```

**Collectors** implement `fetch_raw()` and `normalise()` methods, converting source-
specific JSON, XML, or CSV responses into unified Pydantic models. **Schemas** use
Pydantic v2 [@Pydantic2024] for runtime validation with full type annotations,
ensuring data integrity at ingestion boundaries. **Analysis modules** accept pandas
[@McKinney2010] DataFrames and NumPy [@Harris2020] arrays, using SciPy [@Virtanen2020]
for statistical computations. Internal result structures (`DatasetProfile`,
`Recommendation`, `PipelineResult`, `QualityReport`) use Python dataclasses rather
than Pydantic models, separating validation concerns from computation. The **AI
engine** scores methodologies against dataset profiles and returns ranked
recommendations. **Pipelines** are registered in a central registry and executed
by method identifier, returning standardised `PipelineResult` objects.

Lazy imports throughout the package ensure that only required dependencies are loaded,
allowing users to install minimal subsets (e.g., `pip install aquascope[viz]` for
visualisation only). A command-line interface exposes 14 commands (`collect`,
`recommend`, `eda`, `quality`, `run`, `hydro`, `agri`, `forecast`, `plot`, `alerts`,
`solve`, `dashboard`, `list-methods`, `list-sources`) for scriptable workflows.

The test suite comprises 539 tests across 47 test modules, including validation
against the CAMELS large-sample hydrology dataset [@Addor2017] for hydrological
signature computation. Continuous integration runs tests on Python 3.10–3.12 with
lint (Ruff) and type checking (mypy).

# Comparison with Existing Tools

| Feature                       | AquaScope | hydrostats | HydroTools | pySTEPS |
|-------------------------------|:---------:|:----------:|:----------:|:-------:|
| Multi-source data collection  | 12        | —          | 3          | —       |
| Unified data schemas          | ✓         | —          | —          | —       |
| Flood frequency (Bulletin 17C)| ✓         | —          | —          | —       |
| FAO-56 ET₀ and crop Kc       | ✓         | —          | —          | —       |
| Copula analysis               | ✓         | —          | —          | —       |
| Change-point detection        | ✓         | —          | —          | —       |
| ML/ensemble forecasting       | ✓         | —          | —          | ✓       |
| AI methodology recommendations| ✓         | —          | —          | —       |
| Scientific I/O (WaterML, HEC) | ✓         | —          | partial    | —       |
| Interactive dashboard         | ✓         | —          | —          | —       |

AquaScope is unique in combining data aggregation from 12 global sources with a
comprehensive analytical toolkit and an AI-driven methodology recommendation engine
in a single package. While individual features may overlap with specialised tools,
no existing package provides this integrated workflow from data collection through
analysis to methodology guidance.

# Acknowledgements

AquaScope builds upon the scientific Python ecosystem, particularly pandas
[@McKinney2010], NumPy [@Harris2020], SciPy [@Virtanen2020], and Pydantic
[@Pydantic2024]. The authors thank the maintainers of these foundational libraries,
as well as the data providers—USGS, Taiwan MOENV, FAO, UN Environment Programme, and
the Copernicus Climate Data Store—whose open data policies make integrated water
resources research possible. The groundwater analysis capabilities draw on the
foundational work of @Theis1935 and @CooperJacob1946.

# References
