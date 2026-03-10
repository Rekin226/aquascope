# Contributing to AquaScope

Thank you for your interest in contributing to AquaScope! This project aims to be a community-driven resource for water researchers worldwide. Whether you are a hydrologist, environmental engineer, data scientist, or student — your contributions are welcome.

## Ways to Contribute

### 1. Add a New Data Source Collector

We want to cover water APIs from every country. To add a new collector:

1. Create a new file in `aquascope/collectors/` (e.g., `japan_mlit.py`).
2. Subclass `BaseCollector` and implement `fetch_raw()` and `normalise()`.
3. Map raw API fields to our unified schemas in `aquascope/schemas/water_data.py`.
4. Add tests in `tests/test_collectors/`.
5. Document the data source in your module docstring (API URL, required keys, datasets).

```python
from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import WaterQualitySample, DataSource

class JapanMLITCollector(BaseCollector):
    name = "japan_mlit"

    def fetch_raw(self, **kwargs):
        return self.client.get_json("https://api.example.jp/water/v1/quality")

    def normalise(self, raw):
        # Convert raw records into WaterQualitySample instances
        ...
```

### 2. Add a Research Methodology

To expand the AI recommender's knowledge base:

1. Open `aquascope/ai_engine/knowledge_base.py`.
2. Add a new `ResearchMethodology` entry to the `METHODOLOGIES` list.
3. Include: description, applicable parameters, data requirements, scale, complexity, references, and tags.
4. Add a test in `tests/test_ai_engine/` to verify your methodology is findable.

Or simply open an issue using the **New Research Methodology** template.

### 3. Improve the AI Recommender

The scoring algorithm in `aquascope/ai_engine/recommender.py` can be improved:

- Better heuristics for data sufficiency scoring
- Additional scoring dimensions (e.g., data frequency, parameter correlations)
- LLM prompt improvements for the enhanced mode

### 4. Add Notebooks / Tutorials

Example Jupyter notebooks in `notebooks/` help new users get started. Contributions in English, French, Chinese, or any language are appreciated.

### 5. Fix Bugs / Improve Docs

Bug fixes and documentation improvements are always welcome.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Rekin226/aquascope.git
cd aquascope

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run linter
ruff check aquascope/ tests/

# Type checking
mypy aquascope/ --ignore-missing-imports
```

## Pull Request Guidelines

1. **Fork** the repository and create a feature branch.
2. Write tests for new functionality.
3. Ensure `pytest`, `ruff check`, and `mypy` pass.
4. Keep commits atomic and write clear commit messages.
5. Update documentation if you change public APIs.
6. Reference related issues in your PR description.

## Code Style

- Follow PEP 8; enforced by `ruff`.
- Use type hints for all public functions.
- Write docstrings (Google or NumPy style).
- Keep modules focused — one collector per file, one concept per module.

## Reporting Issues

- **Bugs**: Use the Bug Report template.
- **New data sources**: Use the New Data Source template.
- **New methodologies**: Use the New Research Methodology template.

## Code of Conduct

Be respectful, inclusive, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
