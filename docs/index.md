# AquaScope Hydrology

**Explore river records and run reproducible water analyses in your browser.**

Find a gauge, inspect its usable record, and export data, figures and methods.
No installation; core Explorer workflows need no API key.

[![PyPI version](https://img.shields.io/pypi/v/aquascope.svg?color=blue)](https://pypi.org/project/aquascope/)
[![Python](https://img.shields.io/pypi/pyversions/aquascope.svg?color=informational)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Rekin226/aquascope/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-3200%2B%20passing-brightgreen.svg)](https://github.com/Rekin226/aquascope/actions)
[![GitHub stars](https://img.shields.io/github/stars/Rekin226/aquascope?style=social)](https://github.com/Rekin226/aquascope/stargazers)

AquaScope unifies **37 global water-data sources** behind one Python schema. On top of that it layers a full scientific computing stack, from **flood-frequency methods** to **FAO-56 crop water requirements**, wrapped in an AI engine that scores **27 research methodologies** against your dataset and auto-executes **26 analysis pipelines**. Regression checks include the CAMELS benchmark with 3,200+ tests across the project.
The daily benchmark inputs are synthetic; flood benchmarks also use observed USGS annual peaks.
See [validation scope](validation_scope.md) for comparators and limitations.

---

## Install

```bash
pip install aquascope              # core: collectors + hydrology
pip install "aquascope[all]"       # everything: ML, viz, spatial, dashboard
```

[Full install options →](getting_started.md#run-a-reproducible-python-example)

---

## What you can do

- 🌊 **Pull water data** from USGS, NOAA NWPS, Colorado DWR/CDSS, the US Water Quality Portal, England's Environment Agency, France Hub'Eau, Germany PEGELONLINE, Ireland OPW, Greece Hydroscope, Poland IMGW-PIB, EU WFD, Taiwan MOENV / WRA / CWA, Japan MLIT, Korea WAMIS, India WRIS, South Africa DWS, Australia BOM, Brazil ANA Hidroweb, CAMELS-CL and CAMELS-BR, GRDC, GEMStat, Copernicus ERA5, OpenMeteo, FAO AQUASTAT, FAO WaPOR and UN SDG 6, with one unified Python API.
- 📈 **Run hydrological analyses**: flood frequency (GEV / LP3 / Gumbel / non-stationary GEV, with separate EMA routines), baseflow separation, rating curves, and 22 hydrological signatures.
- 🌾 **Plan agricultural water**: FAO-56 Penman–Monteith ET₀, crop water requirements for 26 crops (olive, grape, citrus and winter wheat resolved by variety and canopy), irrigation scheduling, and soil water balance with auto-irrigation.
- 🤖 **Ask the AI engine**: describe your goal in plain English, get a recommended methodology scored against your dataset, and auto-execute it.
- 🧑‍🔬 **Run a study**: from a question to a gated plan and a report bundle, keyless. The [advanced studies](advanced_studies.md) test whether the flood is changing, calibrate GR4J for "what if" scenarios and carry CMIP6 models through it to 2050.
- 🧭 **Read the context of any place**: `aquascope context LAT LON` (also over MCP and in the Explorer's Context tab): flood events in the news and Sentinel-1 radar floods 2014-2024, surface water since 1984, modelled flood depth, dams, soil, actual ET and the nearest rain gauge, each with its licence. See [the Explorer docs](explorer.md#context-of-a-place).
- 📊 **Visualise and report**: 17 plot types, Q-Q / P-P diagnostics, Markdown / HTML reports with embedded figures, threshold alerts (WHO / EPA / EU WFD).
- 🗺️ **Spatial hydrology**: DEM processing, D8 flow direction, watershed delineation, Strahler ordering.

[Full feature list →](features.md)

---

## Why AquaScope

|                                              | AquaScope | HEC-SSP | R `lmom` | Standalone collectors |
| :------------------------------------------- | :-------: | :-----: | :------: | :-------------------: |
| Bulletin 17C FFA + EMA                       |    ✅     |   ✅    | partial  |          no           |
| Non-stationary GEV                           |    ✅     |   no    | partial  |          no           |
| CMIP6 change factors through a calibrated GR4J |  ✅     |   no    |    no    |          no           |
| Baseflow separation (Lyne-Hollick, Eckhardt) |    ✅     |   no    |    no    |          no           |
| FAO-56 Penman–Monteith ET₀ + crop water      |    ✅     |   no    |    no    |          no           |
| 37 unified data collectors                   |    ✅     |   no    |    no    |       per-source       |
| AI methodology recommender                   |    ✅     |   no    |    no    |          no           |
| Interactive Streamlit dashboard              |    ✅     |   no    |    no    |          no           |
| Free, MIT, Python-native                     |    ✅     | partial |    ✅    |        varies         |

---

## Start here

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting_started.md)**
  Install AquaScope, pull a USGS station, and run a Bulletin 17C flood-frequency analysis in ten minutes.

- :material-book-open-variant: **[User guide](features.md)**
  Full feature catalog: hydrology, agriculture, ML, spatial, I/O, AI recommender.

- :material-flask-empty-outline: **[Examples](examples/potomac_flood_frequency.md)**
  Real-world case studies with verified results and published-source validation.

- :material-code-braces: **[API reference](api.md)**
  Every public function, class, and method, auto-generated from the source.

- :material-notebook-outline: **[Agricultural water tutorial](https://github.com/Rekin226/aquascope/blob/main/notebooks/07_agricultural_water_demand.ipynb)**
  End-to-end FAO-56 notebook with Open-Meteo fallback and irrigation scheduling.

</div>

---

## Validation and reproducibility

- **3,200+ tests** across every collector, hydrology method, and pipeline.
- **CAMELS benchmark**: synthetic daily regression fixtures and observed annual peaks for ten catchments, bundled at `data/camels_benchmark/`, run on every CI build.
- **Every method cited**: equations, decision trees, and DOI references for all 27 methodologies live in the [theory guide](theory.md).
- **JOSS paper in preparation**: see [`paper.md`](https://github.com/Rekin226/aquascope/blob/main/paper.md) and [`paper.bib`](https://github.com/Rekin226/aquascope/blob/main/paper.bib).

---

## Cite AquaScope

```bibtex
@software{aquascope2026,
  title   = {AquaScope: Open-Source Water Data Aggregation, Hydrological Analysis,
             and Agricultural Water Management Toolkit},
  author  = {AquaScope Contributors},
  year    = {2026},
  url     = {https://github.com/Rekin226/aquascope},
  version = {0.27.0},
  doi     = {10.5281/zenodo.23301359},
  license = {MIT}
}
```

[GitHub](https://github.com/Rekin226/aquascope) · [PyPI](https://pypi.org/project/aquascope/) · [Discussions](https://github.com/Rekin226/aquascope/discussions) · [Ko-fi](https://ko-fi.com/getaquascope)
