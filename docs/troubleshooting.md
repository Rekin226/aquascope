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

## AI Recommender Issues

### How do I try the AI recommender for free?
Three options, cheapest first:

1. **Do nothing.** The rule-based scorer needs no key, no account, and no
   network. It is the default and it scores all 26 methodologies.
2. **Free HuggingFace token.** Create one at
   <https://huggingface.co/settings/tokens> (read access, no credit card),
   then paste it under **⚙️ LLM enhancement → HuggingFace**. Models known to
   be served are listed in the Model dropdown.
3. **Ollama, fully offline.** `ollama serve`, then select **Ollama (local)**.
   No account and no data leaves your machine.

There is no anonymous hosted inference: HuggingFace returns 401 without a
token. If you are running a public deployment for others, see
[the Explorer](https://rekin226-aquascope-explorer.static.hf.space/), which runs the same analyses in your browser
for how to supply one token server-side so visitors need no account.

### "Free hosted AI unavailable"
The deployment's shared free quota is exhausted or its token is invalid. You
still get rule-based results. Add your own free key under **⚙️ LLM
enhancement** to bypass the shared quota.

### "LLM unavailable — showing rule-based results"
The recommender always returns results: if the language model cannot be reached
it falls back to the built-in rule-based scorer and tells you why. The message
after the banner is the actual cause:

| Message | Fix |
|---|---|
| `The 'openai' package is not installed` | `pip install "aquascope[llm]"` |
| `rejected the API key (authentication failed)` | Check the key, and that it belongs to the selected provider |
| `does not serve the model '<name>'` | Pick a different model; provider catalogues change over time |
| `rate limit or quota exceeded` | Wait, or switch provider |
| `Could not reach <provider>` | Check network access, or that Ollama is running (`ollama serve`) |
| `replied, but the model did not return JSON` | Use a stronger instruction-following model |

The rule-based results are still valid, they just come from the heuristic
scorer rather than a model.

### Recommendations look generic
Rationales like "This methodology is generally applicable to …" come from the
rule-based scorer. If you selected an LLM provider and still see these, check
for the fallback banner above: the model was not used.

### The recommender hangs for a long time
Local Ollama models have a 120 s timeout. A large model on CPU can use all of
it. Pick a smaller model (for example `llama3.2`) or run `ollama serve` with GPU
acceleration.

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
