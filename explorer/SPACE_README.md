---
title: AquaScope Explorer
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: static
pinned: true
license: mit
short_description: Click any public water gauge on Earth, nothing to install
---

# 🌊 AquaScope Explorer

Every public water gauge [AquaScope](https://github.com/Rekin226/aquascope) can
reach, on one map. Click a station and get the observed record straight from
the agency, plus flood frequency (GEV, Log-Pearson III with confidence limits),
flow duration and trend, computed **in your browser** by aquascope running on
Pyodide. Nothing to install, no server, no account.

- Station catalog: [`Rekin226/aquascope-gauges`](https://huggingface.co/datasets/Rekin226/aquascope-gauges)
  (GeoParquet, harvested weekly), read in place with DuckDB-WASM. The catalog
  holds **75,892 stations** and the observation mirror holds daily records for
  **6,231 of them** (2026-11-02 harvest).
- Sources with a catalog today (11): USGS, the Environment Agency (England), Hub'Eau
  (France), PEGELONLINE (Germany), Ireland OPW, Taiwan CWA, the Australian Bureau of
  Meteorology, Brazil ANA, Greece Hydroscope, Greece OpenHi and Poland IMGW.
- Full daily records: USGS, UK EA, Hub'Eau (obs_elab), Poland IMGW (discharge and
  water level) and Greece OpenHi (discharge, water level, rainfall). Real-time
  feeds (last month): PEGELONLINE, OPW. Daily rainfall: Taiwan CWA.
- Click anywhere else for the hydrology of that point (ERA5, FAO-56 ET0,
  GloFAS) from Open-Meteo.
- **Ask ✨**: a plain-language question, answered from real gauges by
  aquascope's tools running in your browser, with a Data and Methods section.
  Bring your own key (Groq and Hugging Face have free tiers); it goes from
  your tab straight to the provider.

Source and issues: https://github.com/Rekin226/aquascope (Explorer epic #189, Archive epic #188).
