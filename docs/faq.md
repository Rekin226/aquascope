# Frequently Asked Questions

## General

### What is AquaScope?
AquaScope is an open-source water data aggregation toolkit with AI-powered research methodology recommendations. It collects water-quality and hydrology data from 10 global sources, normalises them into unified schemas, and uses an AI engine to recommend and auto-execute research methodologies.

### Who is AquaScope for?
- **Hydrologists** — flow analysis, flood forecasting, baseflow separation
- **Environmental scientists** — water quality monitoring, trend analysis
- **Civil engineers** — flood frequency analysis, low-flow statistics
- **Researchers** — data collection, methodology selection, reproducible workflows
- **Students** — learning hydrology and water science with real data

### What Python versions are supported?
Python 3.10 and above. We test on 3.10, 3.11, and 3.12.

### What data sources are available?
| Source | Region | Data Type | Auth Required |
|--------|--------|-----------|---------------|
| `taiwan_moenv` | Taiwan | Water quality | No |
| `taiwan_wra_level` | Taiwan | Water levels | No |
| `taiwan_wra_reservoir` | Taiwan | Reservoir status | No |
| `usgs` | USA | River discharge | No |
| `sdg6` | Global | SDG 6 indicators | No |
| `gemstat` | Global | Water quality | No |
| `taiwan_civil_iot` | Taiwan | IoT sensors | No |
| `wqp` | USA | Water quality | No |
| `openmeteo` | Global | Weather/forecast | No |
| `copernicus` | Global | GloFAS discharge | API key |

## Installation

### How do I install all features?
```bash
pip install aquascope[all]
```

### What are the optional dependency groups?
- `ml` — scikit-learn, XGBoost, statsmodels (for ML models)
- `forecast` — Prophet, PyTorch (for time-series forecasting)
- `viz` — matplotlib, seaborn, folium (for visualisation)
- `scientific` — xarray, netCDF4, h5py (for NetCDF/HDF5 export)
- `copernicus` — cdsapi, cfgrib, xarray (for Copernicus data)
- `llm` — openai (for LLM-enhanced recommendations)
- `dev` — pytest, ruff, mypy (for development)
- `all` — everything above

### XGBoost fails on macOS — how do I fix it?
XGBoost requires `libomp` on macOS:
```bash
brew install libomp
```

## Usage

### How do I collect data from the CLI?
```bash
aquascope collect --source usgs --days 7 --format json
aquascope collect --source openmeteo --lat 25.03 --lon 121.57 --mode weather
```

### How do I get AI recommendations?
```bash
# From a data file
aquascope recommend --from-file data/raw/water_data.json --goal "trend analysis"

# Or from Python
from aquascope import recommend
recs = recommend(file="data.json", goal="flood risk")
```

### How do I run hydrological analysis?
```bash
aquascope hydro --analysis fdc --file discharge.csv
aquascope hydro --analysis baseflow --file discharge.csv
aquascope hydro --analysis flood-freq --file discharge.csv
aquascope hydro --analysis low-flow --file discharge.csv --n-day 7 --return-period 10
```

### How do I use the natural language agent?
```bash
aquascope solve "Forecast flooding at latitude 25.03, longitude 121.57 for 14 days"
```

```python
from aquascope.ai_engine.agent import HydroAgent
agent = HydroAgent()
result = agent.solve("Assess drought conditions in Taipei")
print(agent.explain(result))
```

### How do I create visualisations?
```python
from aquascope.viz import plot_timeseries, plot_fdc, plot_forecast
plot_timeseries(df, title="Daily Discharge", save_path="output.png")
```

```bash
aquascope plot --type timeseries --file data.csv --output plot.png
```

## Data & Formats

### What output formats are supported?
- **JSON** — default, human-readable
- **CSV** — tabular, spreadsheet-compatible
- **NetCDF** — scientific standard (requires `aquascope[scientific]`)
- **HDF5** — large dataset storage (requires `aquascope[scientific]`)
- **GeoJSON** — GIS-compatible spatial data

### How do I export to NetCDF?
```python
from aquascope.utils.storage import export_netcdf
export_netcdf(dataframe, "output.nc", variable_name="discharge")
```

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues.
