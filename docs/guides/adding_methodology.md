# Adding a New Research Methodology

AquaScope's AI recommender draws from a curated knowledge base of research methodologies. Adding a new one involves two steps: knowledge base entry and (optionally) a pipeline implementation.

## Step 1: Add to the Knowledge Base

Edit `aquascope/ai_engine/knowledge_base.py` and append a new `ResearchMethodology` to the `METHODOLOGIES` list:

```python
ResearchMethodology(
    id="your_method_id",                     # Unique snake_case identifier
    name="Your Method Name",                  # Human-readable name
    category="statistical",                   # One of: statistical, machine_learning,
                                             #   process_engineering, remote_sensing,
                                             #   hydrological_modelling, policy_analysis
    description="A brief description of what this method does and when to use it.",
    applicable_parameters=[                   # Water quality parameters this applies to
        "DO", "BOD5", "COD", "pH", "SS",
    ],
    data_requirements=[                       # What data is needed
        "time-series > 2 years",
        "multiple stations",
    ],
    typical_scale="regional",                 # lab / pilot / field / regional / global
    complexity="medium",                      # low / medium / high
    references=[                              # Key academic references
        "Author et al. (2020). Title. Journal, 1(2), 3-4. DOI: ...",
    ],
    tags=[                                    # Search tags for matching
        "keyword1", "keyword2", "keyword3",
    ],
),
```

### Field Guidelines

| Field | Purpose | Tips |
|-------|---------|------|
| `id` | Used by pipelines and CLI | Must be unique, use snake_case |
| `applicable_parameters` | Helps the recommender match datasets | Use standard abbreviations (DO, BOD5, COD, etc.) |
| `data_requirements` | Shown in recommendations | Be specific about what the method needs |
| `tags` | Used for keyword matching in recommendations | Include synonyms, related terms |
| `complexity` | Helps researchers choose appropriate methods | "low" = basic stats, "high" = advanced ML/numerical |

## Step 2: (Optional) Add a Pipeline Implementation

If you want users to be able to auto-run your methodology, add a pipeline:

### 2a. Create the pipeline function

Add to `aquascope/pipelines/model_builder.py`:

```python
def run_your_method(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Your method description."""
    config = config or {}
    # ... implementation ...

    return PipelineResult(
        method_id="your_method_id",
        method_name="Your Method Name",
        summary="Human-readable summary of results.",
        metrics={"key_metric": value},
        details={"raw_results": {...}},
    )
```

### 2b. Register in the pipeline dispatcher

Add to `PIPELINE_REGISTRY`:

```python
PIPELINE_REGISTRY: dict[str, callable] = {
    # ... existing pipelines ...
    "your_method_id": run_your_method,
}
```

## Step 3: Write Tests

Add tests in `tests/test_pipelines/test_model_builder.py` or a new test file:

```python
def test_your_method():
    df = _make_sample_data()
    result = run_your_method(df)
    assert isinstance(result, PipelineResult)
    assert result.method_id == "your_method_id"
```

## Step 4: Verify

```bash
# Run tests
pytest tests/ -v

# Check lint
ruff check .

# Verify the methodology appears
aquascope list-methods
```

## Examples of Good Methodologies to Add

- **Numerical methods** — Finite element analysis for groundwater flow, PDE-based contaminant transport
- **Forecasting** — Prophet time-series, Transformer-based water quality prediction
- **Process models** — QUAL2K, WASP, MIKE, HEC-RAS
- **Machine learning** — GNN for sensor networks, autoencoders for anomaly detection
- **Field methods** — Isotope tracing, sediment analysis, bioassays
