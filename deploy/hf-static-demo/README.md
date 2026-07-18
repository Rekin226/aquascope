---
title: AquaScope Dashboard
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Water data & hydrology toolkit — live in-browser demo
---

# 🌊 AquaScope Dashboard — live demo

The full [AquaScope](https://github.com/Rekin226/aquascope) Streamlit dashboard
running **entirely in your browser** via [stlite](https://github.com/whitphx/stlite)
(Pyodide/WebAssembly) — no server, no install.

First load downloads Python plus the scientific stack (~30–90 s), then
everything is instant: demo datasets, smart insights, the hydrology lab,
extreme-value analysis, FAO-56 irrigation scheduling, and the AI methodology
recommender.

Note: because this demo has no backend, live data collectors may be blocked by
the source APIs' CORS policies. For full functionality install locally:

```bash
pip install "aquascope[dashboard]"
aquascope dashboard
```

- **Source:** https://github.com/Rekin226/aquascope
- **PyPI:** https://pypi.org/project/aquascope/
- **Docs:** https://rekin226.github.io/aquascope/
