# Running Analysis Pipelines

AquaScope v0.2.0 includes an end-to-end workflow: collect data → assess quality → run EDA → get AI recommendations → auto-execute the recommended methodology.

## Available Pipelines

| Pipeline ID | Method Name | Category | Key Output |
|-------------|-------------|----------|------------|
| `trend_analysis` | Mann-Kendall Trend Test | Statistical | Trend direction, significance, slope per station |
| `wqi_calculation` | Taiwan RPI (River Pollution Index) | Statistical | RPI score + pollution category per station-date |
| `pca_clustering` | PCA + K-Means | ML | Variance explained, cluster assignments, loadings |
| `random_forest_classification` | Random Forest | ML | Accuracy, feature importance, classification report |
| `xgboost_regression` | XGBoost Regression | ML | R², RMSE, feature importance |
| `arima_forecast` | ARIMA Time-Series Forecast | Statistical | AIC/BIC, forecast values, RMSE |
| `correlation_analysis` | Pearson Correlation | Statistical | Correlation matrix, significant pairs |

## CLI Workflow

### 1. Collect Data

```bash
aquascope collect --source taiwan_moenv --api-key YOUR_KEY --format json
```

### 2. Assess Quality

```bash
aquascope quality --file data/raw/taiwan_moenv_20260312.json
# Optionally auto-fix issues:
aquascope quality --file data/raw/taiwan_moenv_20260312.json --fix
```

### 3. Run EDA

```bash
aquascope eda --file data/raw/taiwan_moenv_20260312.json
# Include AI recommendations:
aquascope eda --file data/raw/taiwan_moenv_20260312.json --recommend
```

### 4. Get Recommendations

```bash
aquascope recommend \
  --parameters DO,BOD5,COD,NH3-N,SS \
  --goal "trend analysis over 10 years" \
  --years 10 --n-stations 20
```

### 5. Execute a Pipeline

```bash
# Run the recommended methodology
aquascope run --method trend_analysis \
  --file data/raw/taiwan_moenv_20260312.json \
  --output results/trend_results.json

# Run with custom config
aquascope run --method pca_clustering \
  --file data/raw/taiwan_moenv_20260312.json \
  --config '{"n_clusters": 4, "n_components": 3}'
```

## Python API Workflow

```python
import pandas as pd
from aquascope.collectors import TaiwanMOENVCollector
from aquascope.analysis.eda import generate_eda_report, profile_dataset
from aquascope.analysis.quality import assess_quality, preprocess
from aquascope.ai_engine.recommender import recommend
from aquascope.pipelines.model_builder import run_pipeline

# 1. Collect
collector = TaiwanMOENVCollector(api_key="YOUR_KEY")
records = collector.collect()

# 2. Convert to DataFrame
df = pd.DataFrame([r.model_dump() for r in records])

# 3. Quality check & preprocess
quality = assess_quality(df)
print(f"Completeness: {quality.completeness_pct}%")
df_clean = preprocess(df, steps=quality.recommended_steps)

# 4. EDA + Auto-recommend
profile = profile_dataset(df_clean)
recs = recommend(profile, top_k=5)
print(f"Top recommendation: {recs[0].methodology.name}")

# 5. Execute the top recommendation
result = run_pipeline(recs[0].methodology.id, df_clean)
print(result.summary)
print(result.metrics)
```

## CCME WQI alongside PCA and correlation

CCME WQI is available through `aquascope.workbench.run("ccme_wqi", ...)`.
It accepts the `parameter` and `value` columns from WaterQualitySample
tables, with additional metadata columns preserved in the input table.

The example below uses a CSV containing `station_id`, `sample_datetime`,
`parameter`, `value`, and `unit`. It selects one station and reporting year,
then passes the same measurements to CCME and the existing correlation
and PCA pipelines.

Replace the station, dates, and illustrative guideline limits with values
appropriate to your study. The concentration limits below assume mg/L;
the pH limits use the pH scale.

```python
import pandas as pd

from aquascope import workbench
from aquascope.pipelines.model_builder import run_pipeline

df = pd.read_csv(
    "water_quality.csv",
    parse_dates=["sample_datetime"],
    dtype={"station_id": str},
)

# Illustrative limits; choose appropriate guidelines for your study.
guidelines = {
    "DO": {"min": 5.0},
    "pH": {"min": 6.5, "max": 9.0},
    "TP": {"max": 0.05},
    "Pb": {"max": 0.004},
}

selected = df.loc[
    (df["station_id"] == "STATION-1")
    & (df["sample_datetime"] >= "2024-01-01")
    & (df["sample_datetime"] < "2025-01-01")
    & df["parameter"].isin(guidelines)
].copy()

wqi = workbench.run("ccme_wqi", selected, guidelines=guidelines)
print(wqi["score"], wqi["category"])

correlation = run_pipeline("correlation_analysis", selected)
print(correlation.details["correlation_matrix"])

pca = run_pipeline(
    "pca_clustering",
    selected,
    config={"n_components": 2, "n_clusters": 3},
)
print(pca.metrics)
```

PCA requires the project's existing `ml` extra. Its current implementation
requires at least ten complete sampling dates and two parameters.
Use data suitable for each analysis: CCME excludes missing measurements
individually, whereas these PCA and correlation pipelines use complete
rows after pivoting the selected parameters.

Pass measurements in their original physical units to CCME. The PCA
pipeline performs its own standardization internally.

The workbench result includes the score, category, F1/F2/F3 factors,
supplied guidelines, and method citation. The Analyst can call this same
calculation through its existing `analyse_table` tool.

This implementation supports finite guideline limits with minimums
greater than or equal to zero and maximums greater than zero. Nonpositive
measurements that fail a minimum guideline raise an error. These are
implementation limits, not universal water-quality standards.

The existing `wqi_calculation` pipeline continues to compute Taiwan RPI.
Use the workbench call above for CCME.

Reference: [CCME Water Quality Index User's Manual, 2017 Update](https://ccme.ca/en/res/wqimanualen.pdf).

## Pipeline Configuration

Each pipeline accepts an optional `config` dict:

### trend_analysis
```python
{"alpha": 0.05, "parameters": ["DO", "BOD5"]}
```

### pca_clustering
```python
{"n_clusters": 3, "n_components": 2}
```

### random_forest_classification
```python
{"target": "category"}
```

### xgboost_regression
```python
{"target_parameter": "DO"}
```

### arima_forecast
```python
{"target_parameter": "DO", "order": [1, 1, 1], "forecast_steps": 12}
```

## Adding New Pipelines

See [Adding a Methodology](adding_methodology.md) for instructions on contributing new pipeline implementations.
