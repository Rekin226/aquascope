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
  - name: Ouédraogo Abdoul Rachid
    orcid: 0000-0002-4616-4153
    affiliation: 1
affiliations:
  - name: National Central University, Taiwan
    index: 1
date: 26 June 2026
bibliography: paper.bib
---

# Summary

AquaScope is an open-source Python toolkit (v0.7.0, MIT license) that unifies water
data collection from 20 global sources, comprehensive hydrological and statistical
analysis, agricultural water management, and AI-powered research methodology
recommendations into a single, coherent package. It addresses a persistent challenge
in water resources research: the fragmentation of data access, analytical methods, and
tooling across disparate software ecosystems. AquaScope normalises heterogeneous data
into unified Pydantic schemas, provides over 40 analytical methods spanning conceptual
rainfall-runoff modelling (GR4J with NSE/KGE/log-NSE auto-calibration), flood
frequency analysis, baseflow separation, extreme value theory, and FAO-56
evapotranspiration, and includes a knowledge-base-driven AI engine that recommends
appropriate research methodologies based on dataset characteristics. Unlike existing
unified data clients, which focus on United States services, AquaScope's collectors
span East and South Asia, Europe, and global FAO/UN sources. The toolkit is
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

Beyond data access, the analytical methods themselves — flood frequency analysis,
evapotranspiration, change-point detection, copula dependence modelling, and others —
span multiple domains and are often available only in specialised packages with
incompatible interfaces.

Existing open-source tools address subsets of these needs. The HyRiver suite
(`pygeohydro`, `pynhd`, `py3dep`) provides excellent, well-maintained access to United
States hydrological services and returns analysis-ready `xarray` and `geopandas`
objects; `dataretrieval` wraps USGS and multi-agency U.S. services; `hydrostats`
provides streamflow comparison metrics; and `pySTEPS` focuses on probabilistic
precipitation nowcasting. Two gaps remain. First, the mature unified-access clients are
**United States-centric**: a researcher working in Taiwan, Japan, Korea, India, the
European Union, or with global FAO/UN datasets must still assemble bespoke clients.
Second, no single toolkit couples multi-source data collection with a comprehensive
hydrological analysis suite, agricultural water management, advanced statistical and
machine-learning methods, and intelligent methodology guidance.

AquaScope addresses both. Its 20 collectors span East and South Asia (Taiwan MOENV and
WRA networks, Japan MLIT, Korea WAMIS, India WRIS), Europe (EU Water Framework
Directive), the United States (USGS, Water Quality Portal), and global providers
(GEMStat, Open-Meteo, Copernicus, UN SDG 6, FAO AQUASTAT and WaPOR). On top of this it
provides an end-to-end workflow, from raw data ingestion through analysis to methodology
recommendation, in a single, well-tested Python package with a unified API. It targets
hydrologists, environmental engineers, agricultural scientists, and water resources
researchers, particularly those working outside the United States, who need to combine
data from multiple sources and apply appropriate analytical methods without assembling a
patchwork of incompatible tools.

# Key Features

**Data aggregation.** AquaScope implements collectors for 19 water data sources, each
subclassing a common `BaseCollector` and normalising responses into shared Pydantic
schemas. Coverage spans Asia (Taiwan MOENV and WRA networks, Taiwan Civil IoT via the
OGC SensorThings API, Japan MLIT, Korea WAMIS, India WRIS), Europe (EU Water Framework
Directive), the United States (USGS NWIS [@USGS_NWIS]; the Water Quality Portal,
aggregating 400+ agencies), and global providers (GEMStat, UN SDG 6, Open-Meteo,
Copernicus ERA5, FAO AQUASTAT and WaPOR). A shared `httpx`-based HTTP client
[@HTTPX2024] provides caching, retries with exponential back-off, and rate limiting.

**Analysis.** The hydrology module provides the GR4J conceptual rainfall-runoff model
[@Perrin2003] with auto-calibration against Nash–Sutcliffe Efficiency [@Nash1970],
Kling–Gupta Efficiency [@Gupta2009], and log-NSE, and adds calibrated quantile
prediction intervals so model output is uncertainty-aware. Flood frequency analysis
follows Bulletin 17C [@England2019] with L-moment estimation [@Hosking1997] and EMA for
censored data; non-stationary GEV [@Coles2001] and regional frequency analysis are
supported. Baseflow separation offers the Lyne–Hollick [@Lyne1979], Eckhardt
[@Eckhardt2005], and UKIH methods, alongside flow-duration curves, recession analysis,
and 22 hydrological signatures. Agricultural water management implements the full FAO-56
Penman-Monteith ET₀ methodology [@Allen1998] with single- and dual-coefficient crop
water demand, irrigation scheduling, and a daily soil-water balance. Statistical methods
include Bayesian uncertainty quantification [@Gelman2013], copula dependence modelling
[@Nelsen2006], change-point detection (PELT [@Killick2012], Pettitt [@Pettitt1979]),
Mann-Kendall trend testing [@Mann1945; @Kendall1975], and machine-learning forecasters
(ARIMA, Prophet, Random Forest, XGBoost, LSTM) with ensembles and transfer learning.
A knowledge base of 26 methodologies drives a recommendation engine that scores methods
against an automatically computed dataset profile, with optional LLM-based reasoning.

**Interoperability.** Every collector's records convert to `xarray.Dataset` and
`geopandas.GeoDataFrame`, so AquaScope data feeds the wider Pangeo and machine-learning
ecosystem rather than remaining a closed schema. The toolkit additionally reads and
writes OGC WaterML 2.0 [@WaterML2012], HEC-DSS/HEC-RAS, EPA SWMM, NetCDF, HDF5, and
GeoJSON. A spatial module delineates watersheds from digital elevation models, and an
interactive Streamlit dashboard with dedicated hydrology, extreme-events, and
agricultural-water labs supports code-free exploration.

# Design and Architecture

AquaScope follows a pipeline architecture — collectors fetch and normalise raw
responses into Pydantic v2 [@Pydantic2024] schemas, which feed analysis modules built
on pandas [@McKinney2010], NumPy [@Harris2020], and SciPy [@Virtanen2020], an AI
recommender, and registered pipelines. Lazy imports let users install minimal subsets
(e.g. `pip install aquascope[interop]`), and a command-line interface exposes the main
workflows for scripting. The package ships over 820 tests, including validation against
the CAMELS large-sample hydrology dataset [@Addor2017], with continuous integration on
Python 3.10–3.12, linting (Ruff), and type checking (mypy).

# Comparison with Existing Tools

| Feature                        | AquaScope | HyRiver | dataretrieval | hydrostats | pySTEPS |
|--------------------------------|:---------:|:-------:|:-------------:|:----------:|:-------:|
| Multi-source data collection   | 19        | U.S.    | U.S.          | —          | —       |
| Non-U.S. / global coverage     | ✓         | —       | —             | —          | —       |
| Unified data schemas           | ✓         | ✓       | —             | —          | —       |
| Conceptual rainfall-runoff (GR4J)| ✓       | —       | —             | —          | —       |
| Flood frequency (Bulletin 17C) | ✓         | —       | —             | —          | —       |
| FAO-56 ET₀ and crop Kc        | ✓         | —       | —             | —          | —       |
| Copula / change-point analysis | ✓         | —       | —             | —          | —       |
| ML/ensemble forecasting        | ✓         | —       | —             | —          | ✓       |
| AI methodology recommendations | ✓         | —       | —             | —          | —       |
| Scientific I/O (WaterML, HEC)  | ✓         | partial | —             | —          | —       |
| Interactive dashboard          | ✓         | —       | —             | —          | —       |

The HyRiver suite is the closest comparator for data access and is more mature for
United States services; AquaScope interoperates with the same `xarray`/`geopandas`
objects while extending coverage well beyond the United States and adding a
comprehensive analytical toolkit and an AI-driven methodology recommender in a single
package. No existing package provides this integrated, geographically broad workflow
from data collection through analysis to methodology guidance.

# Acknowledgements

AquaScope builds upon the scientific Python ecosystem, particularly pandas
[@McKinney2010], NumPy [@Harris2020], SciPy [@Virtanen2020], and Pydantic
[@Pydantic2024]. The author thanks the maintainers of these foundational libraries,
as well as the data providers—USGS, Taiwan MOENV, FAO, UN Environment Programme, and
the Copernicus Climate Data Store—whose open data policies make integrated water
resources research possible. The groundwater analysis capabilities draw on the
foundational work of @Theis1935 and @CooperJacob1946.

# References
