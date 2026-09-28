# Get a useful result with AquaScope Hydrology

[Open Explorer](https://rekin226-aquascope-explorer.static.hf.space/) to find a
river record, inspect the actual available period and missingness, then download
CSV data or start a study. Core workflows need no installation or API key.
Coverage and available variables differ by agency. A catalog pin alone does not
prove that observations are accessible or suitable for an analysis.

## Run a reproducible Python example

This example uses the committed **observed USGS annual-peak snapshot** for Fish
River near Fort Kent, Maine (01013500), in cubic metres per second. It needs no
network once installed. These are annual peaks, not the synthetic daily series
in the adjacent benchmark folder, and not the daily-mean maxima used by Explorer.
Consequently its return levels need not equal an Explorer analysis.

```bash
git clone https://github.com/Rekin226/aquascope.git
cd aquascope
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[viz]"
```

Run this block from the repository root. It creates a return-level table, a
Q–Q diagnostic figure, and provenance recording the precise input file hash.

```python
import hashlib
import json
from pathlib import Path

import pandas as pd

from aquascope import __version__
from aquascope.api import flood_analysis
from aquascope.viz import qq_plot

source = Path("data/camels_benchmark/peaks/01013500_peaks.csv")
frame = pd.read_csv(source)
# Already annual peaks: keep the numeric index to avoid reaggregating by calendar year.
peaks = pd.Series(frame["peak_va"].to_numpy(), name="discharge_m3s").dropna()
result = flood_analysis(peaks, method="gev_lmoments", return_periods=[10, 50, 100])
table = pd.DataFrame([
    {"return_period_years": period, "discharge_m3s": value}
    for period, value in result.return_periods.items()
])
output = Path("quickstart-output")
output.mkdir(exist_ok=True)
table.to_csv(output / "return_levels.csv", index=False)
figure = qq_plot(result.annual_max, "gev", result.params)
figure.savefig(output / "qq_diagnostic.png", dpi=150)
provenance = {
    "software_version": __version__, "station": "USGS-01013500",
    "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "input": str(source), "start": frame["date"].min(), "end": frame["date"].max(),
    "variable": "annual peak discharge", "unit": "m3/s", "estimator": "gev_lmoments",
    "n_peaks": len(peaks), "interval": None,
}
(output / "provenance.json").write_text(json.dumps(provenance, indent=2))
print(table.to_string(index=False))
```

The L-moments fit does not provide an interval in this example. Do not attach an
interval from a separate GEV maximum-likelihood fit. Supported convenience
methods are `gev`, `gev_lmoments`, `lp3`, `gumbel`, and `gpd`; each has its own
input assumptions. Nonstationary and EMA routines have separate APIs.

A GEV fit is not a complete Bulletin 17C procedure. A regulatory flood study
requires appropriate annual peaks, record screening, historical/censored-data
handling and domain review. Neither this example nor a fit to daily mean maxima
certifies a design flood. See the [validation scope](validation_scope.md).

## Retrieve a current record separately

Live agency access can fail or return a different period. Inspect the returned
period and units before fitting a model; one year of daily data cannot support
a multiyear flood-frequency analysis. The dedicated live smoke workflow checks
retrieval independently of the fixture-backed tutorial.

```bash
python -m aquascope.cli --help
```

In Python, use `aquascope.explore.analyze_station("usgs", "USGS-01013500")`
for the same record analysis as Explorer. Its `start`, `end`, `n`, `fetch_note`,
`data_snapshot`, and method eligibility describe the data actually analyzed.
Use `store={}` as a keyword argument to retain the full Series for CSV export.

## Take an Explorer result into your tools

- **Python:** download the CSV and read it with `pandas.read_csv`; preserve units,
  timestamps and provenance. The study bundle includes a replay notebook.
- **R:** use `read.csv("record.csv")` for a downloaded record; see [readers](readers.md)
  for archive Parquet and spatial workflows.
- **QGIS:** add a downloaded station GeoParquet as a vector layer. A discharge CSV
  is a time series, not a spatial layer unless you also supply station coordinates.
  See [QGIS integration](integration_guides/qgis_integration.md).
- **Repeat a study:** retain the entire bundle, including `workspace.json`, input
  tables, `findings.json`, and the notebook. Use [Save complete study](adoption/sharing.md)
  for reopening in Explorer. Reopening a saved study preserves its
  result; rerunning against a live agency can produce new data.

Browse [data sources](data_sources.md), [methods](methodology_matrix.md), and
[troubleshooting](troubleshooting.md) when selecting your next workflow.
