# Snowmelt and daily runoff on the Durance

At Embrun in the French Alps, precipitation can fall as rain in the valley while snow accumulates upstream.
This example divides the catchment into five equal-area elevation bands, routes their rain and melt through GR4J,
and checks simulated runoff against a year of discharge withheld from calibration. All water fluxes below are
expressed in millimetres over the relevant area.

## Station and inputs

Use the daily record for station X0310010 (La Durance à Embrun) distributed in airGR 1.7.9. It includes
precipitation, mean temperature and potential evapotranspiration from SAFRAN, and observed discharge from EDF
via Banque Hydro. With the R package installed, export the forcing and observed runoff once:

```r
data("X0310010", package = "airGR")
write.csv(data.frame(date = format(BasinObs$DatesR, "%Y-%m-%d", tz = "UTC"),
                     BasinObs[c("P", "T", "E", "Qmm")]), "durance.csv", row.names = FALSE)
BasinInfo$HypsoData[1 + c(10, 30, 50, 70, 90)]
```

The five medians are 1386, 1869, 2170, 2406 and 2697 m. Use the catchment median of 2170 m as the forcing
reference elevation. The example assumes a constant temperature lapse of 6.5 °C/km. When measured gradients
are available, set `lapse_rate` to a value appropriate for the site. The airGR package also offers a seasonal
lapse-rate preprocessing option. [airGR observation data](https://cran.r-universe.dev/airGR/doc/page/BasinObs)
and [station metadata](https://www.hydro.eaufrance.fr/sitehydro/X0310010/fiche).

## Fit and evaluate

```python
import pandas as pd

from aquascope.analysis.metrics import nse
from aquascope.models.cemaneige import CemaNeige
from aquascope.models.rainfall_runoff import GR4J, calibrate

d = pd.read_csv("durance.csv", parse_dates=["date"], index_col="date").loc["1999":"2002"]
bands = dict(band_elevations=[1386, 1869, 2170, 2406, 2697], forcing_elevation=2170)
train = d.loc[:"2001"]
snow_train = CemaNeige(train["P"], train["T"], **bands)
fit = calibrate(snow_train, train["E"], train["Qmm"], objective="nse",
                warmup_days=365, seed=42, maxiter=2)

# Carry the training snowfall climatology forward. Run from 1999 so the snow and
# GR4J stores entering the evaluation year contain the preceding years' water.
snow = CemaNeige(d["P"], d["T"], mean_annual_snowfall=snow_train.mean_annual_snowfall, **bands)
params = fit.params
daily = snow.simulate(ctg=params["CTG"], kf=params["Kf"])
model = GR4J(*(params[key] for key in ("X1", "X2", "X3", "X4")))
q = model.simulate(daily.liquid, d["E"]).streamflow
print("2002 NSE", nse(d["Qmm"].loc["2002"].to_numpy(), q.loc["2002"].to_numpy()))
```

Using SciPy 1.15.3, the 1999–2001 training record gave a basin mean annual snowfall of 505.45 mm.
After the 1999 warm-up, calibration NSE was 0.937. The independent 2002 NSE was 0.773. The small
optimization budget keeps the example quick. Scores can vary with the optimizer and its version.

The snow model starts with empty stores. Each day, snowfall accumulates in each band and its thermal state
determines whether melting can begin. The area-average of rain plus melt supplies GR4J. The daily basin balance
can be checked directly:

```python
storage = daily.snowpack.mean(axis=1)
change = storage - storage.shift(fill_value=0)
assert (d["P"] - daily.liquid - change).abs().max() < 1e-9
```

The classic two-parameter formulation uses CTG for thermal inertia and Kf for the melt rate.
The unit tests compare daily output to the non-hysteretic snow routine from airGR 1.7.9. A separate check
passed identical band forcing from the 1,461 days above to the two snow routines (CTG 0.5, Kf 4,
mean annual snowfall 505.4471 mm). The largest difference in basin rain plus melt was below
0.000001 mm/day.
[Valéry, Andréassian & Perrin (2014)](https://doi.org/10.1016/j.jhydrol.2014.04.058)
and [INRAE model description](https://webgr.inrae.fr/eng/tools/hydrological-models/snow-model).
