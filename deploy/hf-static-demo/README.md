---
title: AquaScope Dashboard
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: "Moved: the analyses are in the AquaScope Explorer"
---

# This Space has moved

The dashboard's analyses live in the **[AquaScope Explorer](https://huggingface.co/spaces/Rekin226/aquascope-explorer)**
now, under **My data**: drop a CSV or an Excel export and the same tools run on it in your browser, next to the
world's public gauges on a map.

Why the move: the analyses were reachable only from a Streamlit server, so the browser could not run them, the MCP
server could not offer them and the Analyst could not use them. They are now `aquascope.workbench`, plain functions
that return JSON, and every surface calls the same code (#235).

The Streamlit dashboard is still in the package for local use, where a full Python environment is available:

```bash
pip install "aquascope[dashboard]"
aquascope dashboard
```

This Space is a redirect. To update it, edit `deploy/hf-static-demo/` in the repo and upload the folder with
`huggingface_hub.HfApi().upload_folder(...)`.
