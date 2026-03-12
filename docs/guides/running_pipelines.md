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
