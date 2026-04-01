# Use Cases Gallery

Real-world scenarios demonstrating how AquaScope supports water
research, monitoring, and decision-making across diverse contexts.

---

## 1. Taiwan River Basin Water Quality Monitoring

**Problem.** Taiwan's Environmental Protection Administration monitors
over 300 river stations monthly. Researchers need a reproducible
workflow to fetch the latest data, assess quality, detect long-term
trends, and identify pollution hotspots — without manually downloading
spreadsheets from multiple portals.

**AquaScope Workflow**

```bash
# Collect the latest river water quality data
aquascope collect --source taiwan_moenv --api-key $MOENV_KEY

# Profile the dataset
aquascope eda --file data/raw/taiwan_moenv_latest.json

# Auto-fix quality issues (duplicates, outliers)
aquascope quality --file data/raw/taiwan_moenv_latest.json --fix

# Detect long-term trends with Mann-Kendall
aquascope run --method trend_analysis \
  --file data/clean/taiwan_moenv_latest.json

# Compute River Pollution Index scores
aquascope run --method wqi_calculation \
  --file data/clean/taiwan_moenv_latest.json
```

**Expected Outputs**

- EDA report: 15 parameters profiled, missing-value heatmap, distributions.
- Trend results: statistically significant decreasing BOD at 40 % of
  stations (p < 0.05), indicating improving water quality.
- RPI scores and pollution category per station per month.

---

## 2. US Groundwater Contamination Detection

**Problem.** A county health department suspects PFAS contamination in
shallow groundwater wells near an industrial site. They need to pull
multi-agency data from the Water Quality Portal, flag anomalous
readings, and classify contamination severity.

**AquaScope Workflow**

```python
from aquascope.collectors import WQPCollector
from aquascope.analysis.quality import assess_quality, preprocess
from aquascope.analysis.eda import profile_dataset
from aquascope.ai_engine.recommender import recommend
from aquascope.pipelines.model_builder import run_pipeline
import pandas as pd

# Collect WQP data for the target county
collector = WQPCollector()
records = collector.collect(state="US:36", county="US:36:001")
df = pd.DataFrame([r.model_dump() for r in records])

# Quality check and preprocess
report = assess_quality(df)
df_clean = preprocess(df, steps=report.recommended_steps)

# Profile and get AI recommendations
profile = profile_dataset(df_clean)
profile.research_goal = "Detect PFAS contamination anomalies"
recs = recommend(profile, top_k=3)

# Run the top-recommended pipeline
result = run_pipeline(recs[0].methodology.id, df_clean)
print(result.summary)
```

**Expected Outputs**

- Quality report highlighting 12 % missing values and 3 duplicate records.
- AI recommender suggests: PCA + Clustering, Random Forest classification,
  Correlation Analysis.
- PCA clusters reveal a distinct group of 8 wells with elevated PFOS/PFOA.

---

## 3. European River Ecological Status Assessment

**Problem.** An EU-funded research consortium needs to assess ecological
status across 170+ countries using GEMStat freshwater data, aligned with
the Water Framework Directive's "good ecological status" benchmarks.

**AquaScope Workflow**

```bash
# Collect GEMStat global freshwater quality data
aquascope collect --source gemstat

# Run EDA to understand parameter coverage
aquascope eda --file data/raw/gemstat_latest.json

# Cluster stations by water quality signature
aquascope run --method pca_clustering \
  --file data/raw/gemstat_latest.json

# Correlate parameters to identify pollution drivers
aquascope run --method correlation_analysis \
  --file data/raw/gemstat_latest.json
```

**Expected Outputs**

- Dataset profile: 50+ parameters across 12,000 stations, 42 countries.
- PCA reveals 3 principal components explaining 78 % of variance
  (organic pollution, nutrient loading, heavy metals).
- Correlation matrix identifies strong BOD–COD and TN–TP co-occurrence,
  pointing to agricultural runoff as a dominant driver.

---

## 4. Global Drought Early Warning System

**Problem.** A humanitarian organisation monitors drought indicators
across sub-Saharan Africa. They need to combine UN SDG 6 water-stress
indicators with USGS streamflow data and Taiwan's reservoir levels to
build a multi-source drought severity index.

**AquaScope Workflow**

```python
from aquascope.collectors import SDG6Collector, USGSCollector
from aquascope.collectors import TaiwanWRACollector
from aquascope.analysis.eda import profile_dataset
from aquascope.ai_engine.recommender import recommend
import pandas as pd

# Collect from three sources
sdg6 = SDG6Collector().collect()
usgs = USGSCollector().collect(site="09380000", days=730)
wra = TaiwanWRACollector().collect()

# Combine into a unified dataframe
frames = []
for records in [sdg6, usgs, wra]:
    frames.append(pd.DataFrame([r.model_dump() for r in records]))
df_all = pd.concat(frames, ignore_index=True)

# Profile and recommend
profile = profile_dataset(df_all)
profile.research_goal = "Drought severity index construction"
recs = recommend(profile, top_k=5)
for r in recs:
    print(f"  {r.rank}. {r.methodology.name} (score: {r.score:.2f})")
```

**Expected Outputs**

- Merged dataset spanning 3 sources, 2 years, 4 parameter categories.
- Top recommendations: ARIMA forecasting (for trend projection),
  Correlation Analysis (cross-source relationships), PCA + Clustering
  (drought severity grouping).
- ARIMA forecast projects reservoir storage 6 months ahead with
  RMSE < 5 %.

---

## 5. Flood Risk Assessment for Urban Planning

**Problem.** A municipal planning department needs historical flood data
to inform zoning decisions. They want to combine USGS gage-height records
with Taiwan Civil IoT real-time sensor data to characterise flood
frequency and identify high-risk zones.

**AquaScope Workflow**

```bash
# Collect USGS gage height data
aquascope collect --source usgs --days 3650

# Collect Taiwan Civil IoT real-time water level sensors
aquascope collect --source taiwan_civil_iot

# Profile both datasets
aquascope eda --file data/raw/usgs_latest.json
aquascope eda --file data/raw/civil_iot_latest.json

# Run XGBoost regression to predict flood stage
aquascope run --method xgboost_regression \
  --file data/raw/usgs_latest.json \
  --config '{"target": "gage_height"}'

# Trend analysis on peak annual flows
aquascope run --method trend_analysis \
  --file data/raw/usgs_latest.json
```

**Expected Outputs**

- 10-year USGS record: 3,650 daily observations across 5 stations.
- Civil IoT real-time snapshot: 200+ sensors, 15-minute intervals.
- XGBoost model predicts gage height with R² = 0.91, highlighting
  upstream precipitation and antecedent soil moisture as top features.
- Mann-Kendall trend test shows statistically significant increasing
  peak flows at 3 of 5 stations, supporting flood risk upgrades in
  the zoning plan.
