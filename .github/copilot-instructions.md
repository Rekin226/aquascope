# Copilot Instructions for AquaScope

AquaScope is a water data aggregation toolkit with AI-powered research methodology recommendations. It collects water-quality and hydrology data from 8 global sources, normalises them into unified Pydantic schemas, and uses an AI engine to recommend and auto-execute research methodologies.

## Build, Test, and Lint

```bash
# Install with all optional dependencies (ML, viz, LLM, dev tools)
pip install -e ".[dev,all]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_collectors/test_taiwan_moenv.py

# Run a single test by name
pytest -k "test_normalise_produces_correct_records"

# Lint
ruff check aquascope/ tests/

# Type check
mypy aquascope/ --ignore-missing-imports
```

CI runs lint and type-check on Python 3.12; tests on 3.10, 3.11, and 3.12.

## Architecture

The data flow is a pipeline:

```
Collectors (fetch + normalise) → Pydantic Schemas → Analysis (EDA, Quality) → AI Recommender → Pipelines (execute)
```

- **`collectors/`** — 8 source modules, each subclassing `BaseCollector`. Implement `fetch_raw()` and `normalise()` to return unified schema objects.
- **`schemas/water_data.py`** — Pydantic models (`WaterQualitySample`, `WaterLevelReading`, `ReservoirStatus`, `SDG6Indicator`) plus a `DataSource` enum.
- **`analysis/`** — `eda.py` for dataset profiling, `quality.py` for assessment and preprocessing.
- **`ai_engine/`** — `knowledge_base.py` defines 26 `ResearchMethodology` entries; `recommender.py` scores them against a `DatasetProfile` (rule-based + optional LLM).
- **`pipelines/model_builder.py`** — 7 executable pipeline functions registered in `_PIPELINE_REGISTRY`, each returning a `PipelineResult`.
- **`utils/`** — `http_client.py` (CachedHTTPClient with retries and rate limiting), `storage.py` (save to JSON/CSV).
- **`cli.py`** — 7 argparse commands: `collect`, `recommend`, `eda`, `quality`, `run`, `list-methods`, `list-sources`. Uses lazy imports — modules are imported inside command functions.

## Key Conventions

- **Python 3.10+** — uses `X | None` union syntax and `list[str]` generics.
- **`from __future__ import annotations`** at the top of every module.
- **Pydantic for data schemas only** (`schemas/`); internal structures use `@dataclass` (`DatasetProfile`, `Recommendation`, `PipelineResult`, `QualityReport`, `EDAReport`).
- **Google-style docstrings** with `Parameters`, `Returns`, `Raises` sections.
- **Full type hints** on all public functions.
- **One logger per module**: `logger = logging.getLogger(__name__)`.
- **Ruff** enforces style — line length is 120; rules: E, F, I, N, W, UP.
- **Tests mirror the package structure** (`tests/test_<module>/`). No `conftest.py`; tests are self-contained using `setup_method` and inline helper functions.

## Adding a New Data Collector

1. Create `aquascope/collectors/<source>.py` — subclass `BaseCollector`, implement `fetch_raw()` and `normalise()`.
2. Add an entry to the `DataSource` enum in `schemas/water_data.py`.
3. Export the class from `aquascope/collectors/__init__.py`.
4. Register it in `cli.py`'s `cmd_collect` function.
5. Add tests in `tests/test_collectors/`.

## Adding a New Methodology / Pipeline

- **Methodology only**: Add a `ResearchMethodology` instance to `METHODOLOGIES` in `ai_engine/knowledge_base.py`.
- **Executable pipeline**: Add a `run_<method_id>(df, config=None) -> PipelineResult` function in `pipelines/model_builder.py` and register it in `_PIPELINE_REGISTRY`.
