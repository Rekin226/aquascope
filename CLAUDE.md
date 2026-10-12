# AquaScope, for Claude

Read this before changing anything. It is the short version of how this repo is
put together and what CI will fail you on. `CONTRIBUTING.md` is the
contributor-facing version with more hand-holding.

## What this is

A Python library for water data and hydrology, and a platform built on top of it:

1. **The Archive** (`aquascope/archive/`): every station catalog as GeoParquet
   plus daily observations, published to the Hugging Face dataset
   `Rekin226/aquascope-gauges` and rebuilt weekly by CI.
2. **The Explorer** (`explorer/`): a static web app (MapLibre + DuckDB-WASM +
   Pyodide) that runs the library in the browser. No server, no install.
3. **The Analyst**: `aquascope ask`, `aquascope mcp`, `aquascope ingest`.
4. **Ecosystem**: the GeoLibre plugin in `integrations/`, docs, the data paper.

## The one rule: one engine, three faces

Every capability is a plain Python function `(inputs) -> dict` in the package.
The web app, the MCP server and the CLI are **thin faces over the same
functions**. `aquascope/explore.py`, `aquascope/workbench.py` and
`aquascope/mcp_server.py` are where those functions live.

So: implement in the package first, then expose. Logic written inside
`explorer/*.js`, inside a Streamlit page, or inside a CLI handler is going the
wrong way and will be asked to move. This is the rule that stops the platform
forking again; it already did once, when the Explorer and the Streamlit
dashboard ended up sharing nothing but the package name.

## Layout

```
aquascope/
  collectors/      one file per data source, subclassing BaseCollector
  schemas/         Pydantic records: water_data.py, groundwater.py, climate.py,
                   agriculture.py, station.py
  registry.py      SOURCES: the single source of truth about every collector
  archive/         the harvester behind the Hugging Face dataset
  explore.py       station/point analysis used by the Explorer and the Analyst
  workbench.py     the analyses shared by every face (the `TOOLS` table)
  mcp_server.py    `aquascope mcp`
  ai_engine/       the Analyst tool loop (analyst.py) and the recommender
  hydrology/ groundwater/ agri/ climate/ spatial/ analysis/ models/
  io/interop.py    records to xarray/pandas
  ingest.py study.py gym/ maintenance/ pipelines/ reporting/ alerts/
  dashboard/       Streamlit, local only (the public deployments were retired)
  cli.py           every verb
explorer/          the static web app (see below)
integrations/      the GeoLibre plugin
tests/             mirrors the package layout
docs/              MkDocs site; docs/data_sources.md holds the sources table
```

## Setup and the three local checks

```bash
pip install -e ".[dev,all]"

ruff check aquascope/ tests/          # ruff is pinned to 0.15.20 in [dev]
pytest
mypy <changed file> --follow-imports=skip --ignore-missing-imports
```

mypy is informational in CI (it cannot fail the build) and there is a known
backlog outside your change. ruff and pytest are not informational.

`pre-commit install` is set up (trailing whitespace, end-of-file, YAML, ruff,
ruff-format, mypy) and clears most lint problems before they reach CI.

## What CI enforces

`ci.yml` runs four jobs, and two more workflows gate a PR:

- **explorer-contracts**: `node --test explorer/tests/*.test.mjs` on Node 24,
  the Explorer's JavaScript unit tests.
- **lint**: ruff over `aquascope/` and `tests/`, then informational mypy.
- **test**: the matrix is **Python 3.10, 3.11 and 3.12**.
- **contributors-board**: fails when a commit author is missing from the README
  all-contributors board. It runs on `main` too, so an uncredited merge turns
  main red until the credit lands.
- **CHANGELOG Guard** (`changelog.yml`): every PR needs a `CHANGELOG.md` entry
  under `## [Unreleased]`, or the `no-changelog` label. The check re-runs when
  the label changes.
- First PRs from a new contributor wait on a maintainer to approve the workflow
  run. "No checks reported" means pending approval, not passing.

Three tests act as structural guards and fail for reasons that look unrelated to
your change:

- `tests/test_registry.py`: every collector class is registered in `SOURCES`, and
  `supports_station_lookup` matches whether the class really overrides
  `stations()`.
- `tests/test_docs_counts.py`: the data-sources table in `docs/data_sources.md`
  is the canonical source count; every hand-written count in the README, docs and
  CLI help must agree with it.
- `tests/test_ci/test_workflows_parse.py`: every workflow file is valid YAML. A
  workflow GitHub cannot parse fails silently, registering no triggers at all.

## Adding a data collector

The most common contribution. Done means all of:

1. `<Source>Collector` in `aquascope/collectors/<source>.py`, subclassing
   `BaseCollector` with `fetch_raw()` and `normalise()`.
2. `normalise()` returns real schema objects, never dicts.
3. A new value in the `DataSource` enum and registration in
   `collectors/__init__.py`.
4. **A `SOURCES` entry in `aquascope/registry.py`.** This is the step people
   miss. It carries `label`, `region`, `variables`, `output_model`,
   `supports_bbox`, `supports_station_lookup` and the licence fields, and the
   CLI choices, dashboard labels, MCP tool descriptions and the harvest all read
   from it. `redistributable` stays `False` until someone has actually read the
   source's terms and recorded them in `license`: the archive only mirrors
   observations from redistributable sources.
5. Mocked-HTTP tests in `tests/test_collectors/test_<source>.py`, including
   pagination if the API paginates.
6. A row in the `docs/data_sources.md` table, plus any prose counts that shift.

HTTP goes through `CachedHTTPClient` (`aquascope/utils/http_client.py`) with
`base_url`, a `RateLimiter` and `cache_ttl_seconds`. `get_json(path, params)`
joins onto `base_url` but passes an absolute URL through unchanged, so
cursor-style pagination works.

## Which schema holds what

- **`StreamflowReading`** is the canonical discharge record: `discharge_cms`,
  `reading_datetime`, a required `source_type` (`in_situ` or `satellite`),
  optional `uncertainty_cms` and `catchment_area_km2` (which enables the
  `runoff_mm_day` property). `grdc.py` is the clean reference; `usgs.py` shows
  discharge and chemistry side by side.
- **`WaterLevelReading`**: stage or level only, in metres.
- **`WaterQualitySample`**: free-form `parameter` + `value` + `unit`. Do not use
  it for discharge in new code, even though older collectors did.
- Groundwater, climate and agriculture have their own schema modules.

`io/interop.py` routes each type into its own xarray variable (`discharge`,
`water_level`, generic parameters), so the choice is not cosmetic. Note that
`StreamflowReading` and `WaterLevelReading` share `reading_datetime`, so that
dispatch discriminates on `discharge_cms` first; keep that ordering if you touch
`records_to_xarray`.

## Recurring traps

- **Python 3.10 cannot parse a trailing `Z`** in `datetime.fromisoformat`. Use
  `datetime.fromisoformat(ts.replace("Z", "+00:00"))`. Inside a broad `except`,
  this silently drops rows and only the 3.10 job goes red.
- **Broad excepts in `normalise()`** hide real parse errors. Narrow them, or log
  a skipped-row count so data loss is visible.
- **ruff** selects `E,F,I,N,W,UP` at line length 120, but
  `[tool.ruff.lint.per-file-ignores]` waives `E501` for collectors and most
  science modules and `N803`/`N806` where the maths wants capitals. Read that
  table before assuming a rule applies.
- **Heavy imports must stay lazy.** Base install is httpx, pydantic, pandas,
  numpy, scipy. scikit-learn, geopandas, plotting and the rest live behind
  extras, and anything the browser loads has to import on a bare install.

## The Explorer (`explorer/`)

A static app with **no bundler**: `index.html` plus ES modules in
`explorer/src/*.js`, a Pyodide worker in `worker.js`, and `build.py` to assemble
a deployable directory.

- **`build.py` copies an explicit list of globs** (`TEXT_GLOBS`,
  `BINARY_GLOBS`). A new asset type or subdirectory that is not in that list is
  present locally and missing in production. This has already happened once, with
  the recorded showcase traces.
- **Cache busting**: every `__BUILD__` placeholder is replaced with the git short
  sha at build time. A literal `?v=__BUILD__` reaching the browser means the
  replacement did not run.
- **The worker loads** micropip, numpy, scipy, pandas, pydantic and httpx, then
  `pyodide-http` and the aquascope wheel. Any package function the app calls must
  work with only those.
- **Local preview** (ES modules need a real server, `file://` will not do):

  ```bash
  pip install build                      # build.py shells out to `python -m build`
  python explorer/build.py --out /tmp/dist-explorer
  python -m http.server -d /tmp/dist-explorer 8000
  ```

  Pass `--wheel dist/aquascope-X.Y.Z-py3-none-any.whl` to reuse an existing
  wheel and skip the build. Building one clears `dist/aquascope-*.whl` first.

- **`tests/test_explorer/test_explorer_assets.py`** stands in for a build step:
  it checks that element ids the modules reach for exist in the page, that
  imports resolve, and that no `__BUILD__` token would survive a deploy. Run it
  after touching anything in `explorer/`.
- **Deploy is automatic**: `explorer.yml` pushes to the Hugging Face Space on
  every push to `main` touching `explorer/**`, `aquascope/**` or
  `pyproject.toml`.

## Weekly automation

- `harvest.yml` (Mondays 03:17 UTC) rebuilds the archive, publishes to Hugging
  Face, and opens, updates or closes one `collector-health` issue per failing
  source. Scope is the sources that implement `stations()` (see
  `station_sources()`), not every collector. The issue step only runs on the
  scheduled event, so a `workflow_dispatch` run never files anything.
- `repair.yml` runs after it, tries a model-written patch per failing source,
  verifies it with lint, the collector's tests and a live smoke call, and opens a
  PR labelled `automated-repair` when it passes. Nothing merges itself, and these
  patches deserve more scrutiny than a human's, not less: a model will happily
  hard-code a URL or widen an `except` until the tests pass.
- `showcase.yml` re-records the Explorer's worked AI examples.
- The honest health headline is the archive's `health.json`, not the source count.
- Every scheduled job writes `health/jobs/<job>.json` and keeps one `job-health` issue from a last step with
  `if: always()` (`aquascope/archive/health_jobs.py`, run through `.github/scripts/job_health.py`). A new scheduled
  workflow adds a row to `JOBS` and that step, or `tests/test_archive/test_health_jobs.py` fails.

## Pull requests

Branch off `main` (never commit to `main` directly), write tests, add the
CHANGELOG entry, keep commits atomic, and reference the issue in the description.
Releases are their own pipeline: version in both `pyproject.toml` and
`aquascope/__init__.py`, CHANGELOG, CITATION.cff, the BibTeX in `README.md` and
`docs/index.md`, then a GitHub Release that triggers the PyPI publish, then the
new Zenodo version DOI back into `CITATION.cff`.
