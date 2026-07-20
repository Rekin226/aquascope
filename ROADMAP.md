# Roadmap

The roadmap reflects what's shipped, what's in-flight, and what's planned. Open items are reordered each release based on community demand in [Discussions](https://github.com/Rekin226/aquascope/discussions/categories/ideas).

## Shipped

- [x] 19 data source collectors (Taiwan ×7, USA ×2, Global ×4, FAO ×2, EU, Japan, Korea, India)
- [x] Rule-based + LLM methodology recommender (26 methods, OpenAI / Groq / HuggingFace / Ollama)
- [x] 7 auto-executable analysis pipelines
- [x] GR4J conceptual rainfall-runoff model + auto-calibration (NSE / KGE / log-NSE)
- [x] Model-evaluation metrics (NSE, KGE, PBIAS, RMSE, R²)
- [x] Bulletin 17C flood frequency with EMA
- [x] FAO-56 Penman-Monteith + crop water requirements (single Kc + dual Kcb/Ke modes)
- [x] Baseflow separation (Eckhardt, Lyne-Hollick, UKIH smoothed-minima)
- [x] Extreme-events frequency analysis (annual maxima/minima, return periods)
- [x] Bayesian UQ, copulas, ensembles, transfer learning
- [x] Spatial hydrology (DEM, watershed, Strahler)
- [x] Scientific I/O (WaterML, HEC, SWMM, NetCDF, HDF5)
- [x] Interactive Streamlit dashboard
- [x] 820+ tests with CAMELS benchmark validation
- [x] Theory guide with equations and DOI citations
- [x] EU Water Framework Directive collector
- [x] Japan MLIT / Korea WAMIS collectors
- [x] Groundwater module (GRACE, well databases, recharge, aquifer hydraulics)
- [x] Climate projection workflows (CMIP6, downscaling, PDSI, scenario analysis)
- [x] JOSS paper submission (`paper.md` + `paper.bib`)
- [x] PyPI release (sdist + wheel + GitHub Actions publish workflow)

## In progress

- [x] Hosted demo (try without installing) — [live on Hugging Face](https://huggingface.co/spaces/Rekin226/aquascope-dashboard), runs fully in-browser via stlite/WebAssembly
- [ ] Tutorial notebooks on Binder / Colab

## Planned

- [ ] Additional data sources — vote at [Discussions → Ideas](https://github.com/Rekin226/aquascope/discussions/categories/ideas)
- [ ] Multi-language documentation (中文, Français, 日本語)
- [ ] ReadTheDocs hosting
- [ ] NumFOCUS Sponsored Project application

## Major features — leveling up

Ambitious, high-impact work that takes AquaScope to the next level. These are [`major feature`](https://github.com/Rekin226/aquascope/labels/major%20feature) · `help wanted` — larger than a weekend, mentorship available. Comment on the issue to discuss scope before starting.

- [ ] Prediction in Ungauged Basins — regionalize signatures/parameters ([#53](https://github.com/Rekin226/aquascope/issues/53)) — *now unblocked by the shipped GR4J keystone*
- [ ] Declarative, reproducible study runner `aquascope run study.yaml` with provenance ([#54](https://github.com/Rekin226/aquascope/issues/54))
- [ ] Plugin architecture — third-party collectors & methodologies via entry points ([#55](https://github.com/Rekin226/aquascope/issues/55))
- [ ] Large-sample CAMELS benchmark — automated accuracy report ([#56](https://github.com/Rekin226/aquascope/issues/56))

## Good first issues — up for grabs

Newcomers welcome. Just comment to claim one, then follow the [contributor ladder](CONTRIBUTING.md) (`good first issue` → `good second issue` → area owner).

- [ ] Edge-case tests for the FAO-56 ETo functions ([#40](https://github.com/Rekin226/aquascope/issues/40))
- [ ] Colorblind-safe plot palette in `viz/styles.py` ([#41](https://github.com/Rekin226/aquascope/issues/41))
- [ ] Edge-case tests for baseflow separation ([#42](https://github.com/Rekin226/aquascope/issues/42))

**Ready for more?** The [`good second issue`](https://github.com/Rekin226/aquascope/labels/good%20second%20issue) tier: Mann-Kendall trend test + Sen's slope ([#44](https://github.com/Rekin226/aquascope/issues/44)) and flow-duration-curve slope + runoff ratio ([#45](https://github.com/Rekin226/aquascope/issues/45)). _(UKIH baseflow #43 — ✅ shipped in #48.)_

## How to influence the roadmap

- 👍 **Vote** on existing requests in [Discussions → Ideas](https://github.com/Rekin226/aquascope/discussions/categories/ideas)
- 💡 **Propose** something new with the *Ideas* discussion category
- 🐛 **File** bugs and edge cases in [Issues](https://github.com/Rekin226/aquascope/issues/new/choose)
- 🤝 **Contribute** — see [CONTRIBUTING.md](CONTRIBUTING.md)
