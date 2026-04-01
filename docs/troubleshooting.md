# Troubleshooting Guide

## Installation Issues

### `ModuleNotFoundError: No module named 'sklearn'`
Install the ML dependencies:
```bash
pip install aquascope[ml]
```

### `ModuleNotFoundError: No module named 'prophet'`
Install the forecast dependencies:
```bash
pip install aquascope[forecast]
```

### `ModuleNotFoundError: No module named 'matplotlib'`
Install the visualisation dependencies:
```bash
pip install aquascope[viz]
```

### XGBoost `XGBoostError` on macOS
XGBoost requires `libomp` (OpenMP runtime) on macOS:
```bash
brew install libomp
pip install xgboost --force-reinstall
```

### `ImportError: No module named 'xarray'`
Install scientific format dependencies:
```bash
pip install aquascope[scientific]
```

## Data Collection Issues

### `ConnectionError` / `TimeoutError` when collecting data
- Check your internet connection
- Some APIs may be temporarily down; try again later
- Use `--days` to limit the request size for USGS
- Open-Meteo has rate limits; the built-in client handles retries automatically

### Empty results from a collector
- **taiwan_moenv**: Data updates monthly; ensure the API is accessible
- **usgs**: Specify valid site numbers or use default (major gauges)
- **openmeteo**: Ensure `--lat`/`--lon` are valid coordinates
- **copernicus**: Requires a valid CDS API key in `~/.cdsapirc`

### `ValueError: Unknown source`
Check available sources:
```bash
aquascope list-sources
```

## Analysis Issues

### `ValueError: Need ≥5 years of data`
Flood frequency analysis (GEV/LP3) requires at least 5 years of annual maxima. Collect more data or use a shorter analysis method.

### `ValueError: Need ≥3 complete years`
Low-flow statistics (7Q10, 30Q5) need at least 3 complete water years. Ensure your data covers multiple years.

### EDA report shows 0 parameters
Your data file may not have the expected schema. Ensure it contains records with `parameter` and `value` fields (for water quality) or a numeric column (for time-series).

### Poor model performance (low NSE/KGE)
- **Insufficient data**: Most models need 1+ years of daily data
- **Wrong model**: Use `aquascope recommend` to find the best method for your data
- **Non-stationary data**: Try differencing or use Prophet (handles trends/seasonality)
- **Outliers**: Run `aquascope quality --file data.json --fix` first

## Visualisation Issues

### Plots don't display (no window appears)
Matplotlib needs an interactive backend. In scripts, use `save_path` instead:
```python
plot_timeseries(df, save_path="output.png")  # saves to file
```

In Jupyter notebooks, add:
```python
%matplotlib inline
```

### Folium maps not rendering in Jupyter
Ensure folium is installed (`pip install folium`) and the notebook trusts HTML output. Try:
```python
from IPython.display import display
m = plot_station_map(stations)
display(m)
```

## CLI Issues

### `aquascope: command not found`
Ensure AquaScope is installed in your active environment:
```bash
pip install -e .
# or
python -m aquascope --help
```

### `error: unrecognized arguments`
Check the command's help:
```bash
aquascope <command> --help
```

## Getting Help

1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/your-org/aquascope/issues)
3. Open a new issue with:
   - Python version (`python --version`)
   - AquaScope version (`python -c "import aquascope; print(aquascope.__version__)"`)
   - Full error traceback
   - Minimal reproducing example
