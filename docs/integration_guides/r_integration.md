# R Integration Guide

Export AquaScope data to CSV or NetCDF and analyse it in R using
tidyverse, hydroTSM, and other hydrological packages.

## Prerequisites

```bash
# Python side
pip install "aquascope[scientific]"
```

```r
# R side
install.packages(c("tidyverse", "hydroTSM", "ncdf4", "lubridate"))
```

## Export to CSV from AquaScope

```python
import pandas as pd
from aquascope.collectors import USGSCollector

collector = USGSCollector()
records = collector.collect(site="09380000", days=365)
df = pd.DataFrame([r.model_dump() for r in records])
df.to_csv("usgs_streamflow.csv", index=False)
print(f"Exported {len(df)} records to usgs_streamflow.csv")
```

## Load CSV in R (tidyverse)

```r
library(tidyverse)
library(lubridate)

df <- read_csv("usgs_streamflow.csv") %>%
  mutate(timestamp = ymd_hms(timestamp))

# Summary statistics per station
df %>%
  group_by(station_id) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    sd_value   = sd(value, na.rm = TRUE),
    n          = n()
  )

# Time-series plot
df %>%
  ggplot(aes(x = timestamp, y = value, colour = station_id)) +
  geom_line() +
  labs(title = "USGS Streamflow", y = "Discharge (cfs)") +
  theme_minimal()
```

## Export to NetCDF and Load in R

```python
import xarray as xr

ds = xr.Dataset.from_dataframe(
    df.set_index(["station_id", "timestamp"])
)
ds.to_netcdf("usgs_streamflow.nc")
```

```r
library(ncdf4)

nc <- nc_open("usgs_streamflow.nc")
print(nc)

values <- ncvar_get(nc, "value")
times  <- ncvar_get(nc, "timestamp")
nc_close(nc)
```

## Hydrological Analysis with hydroTSM

```r
library(hydroTSM)

# Convert to zoo time-series
flow_ts <- zoo::read.zoo(
  df %>% select(timestamp, value),
  FUN = identity
)

# Monthly summary
monthlyfunction(flow_ts, FUN = mean, na.rm = TRUE)

# Flow duration curve
fdc(flow_ts, main = "Flow Duration Curve — USGS 09380000")

# Seasonal analysis
smry(flow_ts)
```

## Trend Analysis with the Kendall Package

```r
install.packages("Kendall")
library(Kendall)

monthly_means <- df %>%
  mutate(month = floor_date(timestamp, "month")) %>%
  group_by(month) %>%
  summarise(mean_val = mean(value, na.rm = TRUE))

result <- MannKendall(monthly_means$mean_val)
print(result)
```

## Tips

- Use `arrow::read_parquet()` in R for faster I/O on large
  AquaScope exports (save with `df.to_parquet()` in Python).
- The `dataRetrieval` R package can complement AquaScope's USGS
  collector for station metadata lookups.
- For spatial analysis in R, export GeoJSON from AquaScope and
  load with `sf::st_read("stations.geojson")`.
