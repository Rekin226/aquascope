# Methodology Decision Matrix

Use this guide to select the right AquaScope methodology for your research question.

## Quick Decision Tree

```
What's your goal?
│
├─ Understand my data → EDA (aquascope eda)
│   ├─ Find trends? → Mann-Kendall / Sen's Slope
│   ├─ Find outliers? → IQR / Isolation Forest
│   └─ Assess quality? → aquascope quality
│
├─ Predict future values → Forecasting
│   ├─ Simple/fast? → ARIMA
│   ├─ Seasonal data? → Prophet
│   ├─ Complex patterns? → Random Forest / XGBoost
│   └─ Long sequences? → LSTM
│
├─ Assess flood risk → FloodChallenge
│   ├─ Return periods? → GEV / LP3 (aquascope hydro --analysis flood-freq)
│   ├─ Forecast floods? → aquascope solve "forecast flooding..."
│   └─ Flow statistics? → FDC (aquascope hydro --analysis fdc)
│
├─ Monitor drought → DroughtChallenge
│   ├─ SPI computation → aquascope solve "assess drought..."
│   └─ Water balance → DroughtChallenge.water_balance()
│
├─ Assess water quality → WaterQualityChallenge
│   ├─ WHO compliance? → check_who_guidelines()
│   ├─ Detect anomalies? → detect_anomalies()
│   └─ Trend analysis? → trend_analysis()
│
└─ Classify/compare → Statistical methods
    ├─ Compare stations? → ANOVA / Kruskal-Wallis
    ├─ Cluster stations? → K-means / Hierarchical
    └─ Water Quality Index? → WQI pipeline
```

## Detailed Matrix

### By Data Characteristics

| Your Data | Records | Temporal | Spatial | Best Methods |
|-----------|---------|----------|---------|-------------|
| Short time-series (<1 yr) | <500 | ✅ | ❌ | ARIMA, basic stats |
| Long time-series (>3 yr) | >1000 | ✅ | ❌ | Prophet, LSTM, Mann-Kendall, GEV |
| Multi-station snapshot | >100 | ❌ | ✅ | Kriging, IDW, clustering |
| Multi-station time-series | >5000 | ✅ | ✅ | Full suite — EDA → recommend |
| Single parameter | any | any | any | Direct analysis or forecast |
| Multi-parameter | any | any | any | Correlation, PCA, WQI |

### By Research Goal

| Goal | Method ID | Pipeline | Min Data |
|------|-----------|----------|----------|
| Trend detection | `mann_kendall` | `trend_analysis` | 30 records |
| Seasonal decomposition | `stl_decomposition` | `seasonal_decomposition` | 2 years |
| Spatial interpolation | `idw_kriging` | `spatial_interpolation` | 10 stations |
| Water quality index | `wqi` | `water_quality_index` | 5 parameters |
| Clustering | `clustering` | `cluster_analysis` | 50 records |
| Time-series forecast | `arima` / `prophet` | `forecast` CLI | 365 records |
| Anomaly detection | `isolation_forest` | Quality challenge | 100 records |
| Flood frequency | `gev` / `lp3` | `hydro --analysis flood-freq` | 5 years |
| Low-flow analysis | `7q10` / `30q5` | `hydro --analysis low-flow` | 3 years |
| Baseflow separation | `lyne_hollick` / `eckhardt` | `hydro --analysis baseflow` | 1 year |
| Flow duration curve | `fdc` | `hydro --analysis fdc` | 1 year |

### Model Selection for Forecasting

| Scenario | Best Model | Why |
|----------|-----------|-----|
| Quick forecast, small data | ARIMA | Fast, interpretable, handles trends |
| Seasonal patterns (rain, temp) | Prophet | Built-in seasonality + holidays |
| Non-linear relationships | Random Forest | Handles interactions, robust |
| High-accuracy needed | XGBoost | State-of-art for tabular data |
| Long-term dependencies | LSTM | Captures temporal patterns |
| Anomaly detection | Isolation Forest | Unsupervised, handles multivariate |
| Drought monitoring | SPI Model | Standard WMO approach |

## CLI Quick Reference

```bash
# Let AI choose for you
aquascope recommend --from-file data.json --goal "your research question"

# Or use the NL agent
aquascope solve "your research question in plain English"

# List all methods
aquascope list-methods

# Run a specific pipeline
aquascope run --method trend_analysis --file data.json
aquascope forecast --model prophet --file timeseries.csv --days 30
aquascope hydro --analysis fdc --file discharge.csv
```
