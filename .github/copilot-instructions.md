# Copilot Instructions for AquaScope

AquaScope is a Python library for water data and hydrology, and a platform built on it: the Archive (station catalogs and daily observations published to the Hugging Face dataset `Rekin226/aquascope-gauges`), the Explorer (a static web app in `explorer/` that runs the library in the browser), and the Analyst (`aquascope ask`, `aquascope mcp`, `aquascope ingest`). `CLAUDE.md` at the repo root is the fuller version of this file, and `CONTRIBUTING.md` is the contributor guide.

## The one rule: one engine, three faces

Every capability is a plain Python function in the package. The Explorer, the MCP server and the CLI are thin faces over the same functions (`aquascope/explore.py`, `aquascope/workbench.py`, `aquascope/mcp_server.py`). Implement in the package first, then expose it; logic written inside `explorer/*.js`, a Streamlit page or a CLI handler will be asked to move.

## Build, Test, and Lint

```bash
# Install with all optional dependencies
pip install -e ".[dev,all]"

# Run all tests (in parallel, as CI does)
pytest -n auto

# Run a single test file
pytest tests/test_collectors/test_grdc.py

# Lint (ruff is pinned in the dev extra)
ruff check aquascope/ tests/

# Type check the file you changed (informational in CI; there is a known backlog)
mypy <changed file> --follow-imports=skip --ignore-missing-imports
```

CI runs ruff and an informational mypy pass on Python 3.12, the test suite on 3.10, 3.11 and 3.12, the Explorer's JavaScript tests (`node --test explorer/tests/*.test.mjs`), a contributors-board check, and a CHANGELOG guard that needs an entry under `## [Unreleased]` or the `no-changelog` label.

## Architecture

- **`collectors/`**: one module per data source, each subclassing `BaseCollector` with `fetch_raw()` and `normalise()`.
- **`registry.py`**: `SOURCES`, the single source of truth about every collector (labels, regions, variables, licence fields). The CLI, dashboard, MCP tool descriptions and the harvest all read it.
- **`schemas/`**: Pydantic records in `water_data.py`, `groundwater.py`, `climate.py`, `agriculture.py` and `station.py`. `StreamflowReading` is the canonical discharge record.
- **`ai_engine/`**: `knowledge_base.py` defines the `METHODOLOGIES`; `recommender.py` scores them against a `DatasetProfile`; `analyst.py` is the Analyst tool loop.
- **`pipelines/model_builder.py`**: executable pipelines registered in `PIPELINE_REGISTRY`, each returning a `PipelineResult`.
- **`utils/http_client.py`**: `CachedHTTPClient` with retries, rate limiting and caching.
- **`cli.py`**: every verb, with lazy imports inside the command functions.

## Key Conventions

- **Python 3.10+**: `X | None` unions and `list[str]` generics. Python 3.10 cannot parse a trailing `Z` in `datetime.fromisoformat`, so use `ts.replace("Z", "+00:00")`.
- **`from __future__ import annotations`** at the top of every module.
- **Pydantic for data schemas only** (`schemas/`); internal structures use `@dataclass`.
- **Google-style docstrings** and full type hints on public functions.
- **One logger per module**: `logger = logging.getLogger(__name__)`.
- **Ruff** selects `E, F, I, N, W, UP` at line length 120; `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` waives some rules per file.
- **Heavy imports stay lazy**: the base install is httpx, pydantic, pandas, numpy and scipy, and anything the browser loads must import on that alone.
- **Tests mirror the package layout** (`tests/test_<module>/`).

## Adding a New Data Collector

1. Create `aquascope/collectors/<source>.py`: subclass `BaseCollector`, implement `fetch_raw()` and `normalise()`, and return schema objects, never dicts.
2. Add a value to the `DataSource` enum in `schemas/water_data.py` and register the class in `collectors/__init__.py`.
3. Add a `SOURCES` entry in `aquascope/registry.py`. Keep `redistributable=False` until the source's terms have been read and recorded in `license`.
4. Add mocked-HTTP tests in `tests/test_collectors/test_<source>.py`.
5. Add a row to the table in `docs/data_sources.md`. `tests/test_docs_counts.py` checks that table and every stated source count against the registry.

## Adding a New Methodology / Pipeline

- **Methodology only**: add a `ResearchMethodology` instance to `METHODOLOGIES` in `ai_engine/knowledge_base.py`.
- **Executable pipeline**: add a `run_<method_id>(df, config=None) -> PipelineResult` function in `pipelines/model_builder.py` and register it in `PIPELINE_REGISTRY`.
