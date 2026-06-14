# Getting started

A 10-minute walkthrough: install AquaScope, pull a real river-discharge record from USGS, run a Bulletin 17C flood-frequency analysis, and read the results.

By the end you'll have a return-level table and a Q-Q diagnostic plot for a real US gauge. Same workflow FEMA, USACE, and state DOTs use.

---

## Install

AquaScope is on PyPI. The **core install** gives you the collectors and the hydrology engine:

```bash
pip install aquascope
```

For visualization (matplotlib + seaborn + folium), add the `viz` extra:

```bash
pip install "aquascope[viz]"
```

To get everything (ML, forecasting, spatial, Streamlit dashboard, scientific I/O), use `[all]`:

```bash
pip install "aquascope[all]"
```

Available extras: `ml`, `forecast`, `copernicus`, `viz`, `llm`, `scientific`, `dashboard`, `spatial`, `all`, `dev`. Full details in the [features guide](features.md).

Python 3.10 or newer.

---

## 1. Pull data from USGS

The simplest collector. No API key required.

```python
from aquascope.collectors import USGSCollector

usgs = USGSCollector()
records = usgs.collect(
    station_id="01646500",   # Potomac River near Washington, DC
    parameter="00060",       # daily mean discharge (cfs)
    days=365,
)

print(len(records), "records")
print(records[0])
```

Every collector returns records in the **same Pydantic schema**, so downstream analyses don't care where the data came from. Switch `USGSCollector` to `WaporCollector`, `AquastatCollector`, or any of the [15 sources](data_sources.md). The rest of your code stays the same.

---

## 2. Run a flood-frequency analysis

Fit a generalized extreme value (GEV) distribution to the annual peak series and get 10 / 50 / 100-year return levels:

```python
from aquascope.api import flood_analysis

result = flood_analysis(
    daily_discharge,                      # pandas Series or DataFrame
    method="gev",
    return_periods=[10, 50, 100],
)

print(result.return_levels)
#   return_period  return_level  lower_ci  upper_ci
# 0           10        1840.2     1690.4    2010.6
# 1           50        2530.7     2280.1    2820.9
# 2          100        2870.4     2540.6    3260.5
```

Swap `method` for `"lp3"` (Bulletin 17C standard), `"gumbel"`, `"gpd"`, or `"ns_gev"` for non-stationary analysis. Pass `censored=True` for EMA on records with peak-over-threshold gaps.

---

## 3. Look at the diagnostics

Q-Q and P-P plots come straight from `result.params` and `result.annual_max`:

```python
from aquascope.viz import qq_diagnostic_plot

fig = qq_diagnostic_plot(result)
fig.savefig("qq_diagnostic.png", dpi=150)
```

The Q-Q plot is the reviewer's first check: does the fitted distribution actually match the upper tail? On real records (1936 St. Patrick's Day flood, anyone?) the diagnostic catches outliers that MLE estimators silently fold into the fit.

---

## 4. Compute hydrological signatures

22 signatures across magnitude, variability, timing, recession, and flashiness, in one function call:

```python
from aquascope.api import baseflow_analysis, compute_all_signatures

bf  = baseflow_analysis(daily_discharge, method="eckhardt")   # or "lyne_hollick"
sig = compute_all_signatures(daily_discharge)

print(bf.bfi)                  # baseflow index, e.g. 0.42
print(sig["Q5"], sig["Q95"])   # high-flow / low-flow exceedances
print(sig["flashiness"])       # Richards–Baker flashiness index
```

---

## 5. Ask the AI engine

Describe your dataset and goal in plain English. The recommender scores 26 methodologies and ranks them by fit:

```python
from aquascope.ai_engine import recommend

recs = recommend(
    parameters=["DO", "BOD5", "COD"],
    n_records=4_500,
    temporal=True,
    spatial=False,
    goal="detect long-term pollution trends with seasonality",
)

for r in recs[:3]:
    print(f"{r.score:.2f}  {r.method_id:<20}  {r.rationale}")
# 0.92  mann_kendall          Strong fit: temporal data, >30 records, trend goal
# 0.87  stl_decomposition     Seasonal patterns + multi-year data
# 0.81  prophet               Forecasting-capable, handles seasonality natively
```

Then auto-execute the top result with `run_pipeline(recs[0].method_id, df)`.

---

## Where to go next

You've now done end-to-end: collect, analyze, diagnose, recommend. The natural next stops:

- **[Potomac flood-frequency example](examples/potomac_flood_frequency.md)**: full case study with real numbers and validation against FEMA-published flood estimates.
- **[Features](features.md)**: the complete capability catalog.
- **[Methodology matrix](methodology_matrix.md)**: when to use which method, decision-tree style.
- **[Theory guide](theory.md)**: equations, citations, and decision trees for every method (668 lines, the closest thing AquaScope has to a textbook).
- **[Data sources](data_sources.md)**: all 15 collectors with endpoint details and API-key requirements.
- **[Integration guides](integration_guides/xarray_integration.md)**: interop with xarray, QGIS, R.

Stuck? Check the [FAQ](faq.md), [troubleshooting](troubleshooting.md), or open a [discussion](https://github.com/Rekin226/aquascope/discussions).
