# AquaScope

**Open-source Python toolkit for water data, hydrology, and agricultural water management, with an AI engine that recommends and auto-executes research methodologies.**

[![PyPI version](https://img.shields.io/pypi/v/aquascope.svg?color=blue)](https://pypi.org/project/aquascope/)
[![Python](https://img.shields.io/pypi/pyversions/aquascope.svg?color=informational)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Rekin226/aquascope/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-820%2B%20passing-brightgreen.svg)](https://github.com/Rekin226/aquascope/actions)
[![GitHub stars](https://img.shields.io/github/stars/Rekin226/aquascope?style=social)](https://github.com/Rekin226/aquascope/stargazers)

AquaScope unifies **19 global water-data APIs** behind one Python schema. On top of that it layers a full scientific computing stack, from **Bulletin 17C flood frequency** to **FAO-56 crop water requirements**, wrapped in an AI engine that scores **26 research methodologies** against your dataset and auto-executes **7 analysis pipelines**. Validated against the CAMELS benchmark with 820+ tests.

---

## Install

```bash
pip install aquascope              # core: collectors + hydrology
pip install "aquascope[all]"       # everything: ML, viz, spatial, dashboard
```

[Full install options →](getting_started.md#install)

---

## What you can do

- 🌊 **Pull water data** from USGS, FAO AQUASTAT, FAO WaPOR, GEMStat, EU WFD, Copernicus ERA5, Taiwan MOENV / WRA, Japan MLIT, Korea WAMIS, OpenMeteo, and UN SDG 6, with one unified Python API.
- 📈 **Run hydrological analyses**: Bulletin 17C flood frequency (GEV / LP3 / Gumbel / non-stationary GEV / EMA), baseflow separation, rating curves, and 22 hydrological signatures.
- 🌾 **Plan agricultural water**: FAO-56 Penman–Monteith ET₀, crop water requirements for 20 crops, irrigation scheduling, and soil water balance with auto-irrigation.
- 🤖 **Ask the AI engine**: describe your goal in plain English, get a recommended methodology scored against your dataset, and auto-execute it.
- 📊 **Visualise and report**: 16 plot types, Q-Q / P-P diagnostics, Markdown / HTML reports with embedded figures, threshold alerts (WHO / EPA / EU WFD).
- 🗺️ **Spatial hydrology**: DEM processing, D8 flow direction, watershed delineation, Strahler ordering.

[Full feature list →](features.md)

---

## Why AquaScope

|                                              | AquaScope | HEC-SSP | R `lmom` | Standalone collectors |
| :------------------------------------------- | :-------: | :-----: | :------: | :-------------------: |
| Bulletin 17C FFA + EMA                       |    ✅     |   ✅    | partial  |          no           |
| Non-stationary GEV                           |    ✅     |   no    | partial  |          no           |
| Baseflow separation (Lyne-Hollick, Eckhardt) |    ✅     |   no    |    no    |          no           |
| FAO-56 Penman–Monteith ET₀ + crop water      |    ✅     |   no    |    no    |          no           |
| 19 unified data collectors                   |    ✅     |   no    |    no    |       per-source       |
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

</div>

---

## Scientifically validated

- **820+ tests** across every collector, hydrology method, and pipeline.
- **CAMELS benchmark**: a 10-catchment validation subset bundled at `data/camels_benchmark/`, run on every CI build.
- **Every method cited**: equations, decision trees, and DOI references for all 26 methodologies live in the [theory guide](theory.md).
- **JOSS paper in submission**: see [`paper.md`](https://github.com/Rekin226/aquascope/blob/main/paper.md) and [`paper.bib`](https://github.com/Rekin226/aquascope/blob/main/paper.bib).

---

## Cite AquaScope

```bibtex
@software{aquascope2026,
  title   = {AquaScope: Open-Source Water Data Aggregation, Hydrological Analysis,
             and Agricultural Water Management Toolkit},
  author  = {AquaScope Contributors},
  year    = {2026},
  url     = {https://github.com/Rekin226/aquascope},
  version = {0.4.0},
  license = {MIT}
}
```

[GitHub](https://github.com/Rekin226/aquascope) · [PyPI](https://pypi.org/project/aquascope/) · [Discussions](https://github.com/Rekin226/aquascope/discussions) · [Ko-fi](https://ko-fi.com/getaquascope)
