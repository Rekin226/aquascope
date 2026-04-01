# xarray Integration Guide

Export AquaScope water-quality and hydrology data to
[xarray](https://docs.xarray.dev/) Datasets for climate and
spatiotemporal analysis.

## Prerequisites

```bash
pip install "aquascope[scientific]"   # installs xarray, netcdf4, h5py
```

## Quick Export

```python
import pandas as pd
import xarray as xr
from aquascope.collectors import TaiwanMOENVCollector

# 1. Collect data
collector = TaiwanMOENVCollector(api_key="YOUR_KEY")
records = collector.collect()
df = pd.DataFrame([r.model_dump() for r in records])

# 2. Convert to xarray Dataset
ds = xr.Dataset.from_dataframe(
    df.set_index(["station_id", "timestamp"])
)
print(ds)
```

## Adding Coordinates and Attributes

```python
# Attach CF-compliant metadata for interoperability
ds.attrs["title"] = "AquaScope Taiwan River Water Quality"
ds.attrs["Conventions"] = "CF-1.8"
ds.attrs["source"] = "Taiwan MOENV via AquaScope"

# Rename columns to CF standard names where applicable
ds = ds.rename({"water_temperature": "sea_water_temperature"})
ds["sea_water_temperature"].attrs["units"] = "degree_Celsius"
```

## Saving to NetCDF

```python
ds.to_netcdf("taiwan_wq.nc", engine="netcdf4")

# Reload and verify
ds2 = xr.open_dataset("taiwan_wq.nc")
print(ds2)
```

## Time-Series Resampling

```python
# Monthly mean per station
monthly = ds.resample(timestamp="ME").mean()
monthly["DO"].plot(col="station_id", col_wrap=4)
```

## Merging Multiple Sources

```python
from aquascope.collectors import USGSCollector

usgs = USGSCollector()
usgs_records = usgs.collect(site="09380000", days=365)
df_usgs = pd.DataFrame([r.model_dump() for r in usgs_records])
ds_usgs = xr.Dataset.from_dataframe(
    df_usgs.set_index(["station_id", "timestamp"])
)

# Merge along a new "source" dimension
combined = xr.concat(
    [ds.expand_dims("source"), ds_usgs.expand_dims("source")],
    dim="source",
)
combined["source"] = ["taiwan_moenv", "usgs"]
```

## Climate Analysis Example

```python
# Compute seasonal anomalies
climatology = ds.groupby("timestamp.season").mean("timestamp")
anomaly = ds.groupby("timestamp.season") - climatology
anomaly["DO"].plot(col="season")
```

## Tips

- Use `engine="h5netcdf"` for HDF5-backed files when working with
  large datasets.
- Set `chunks={"timestamp": 100}` when opening to enable Dask-backed
  lazy loading for out-of-core computation.
- AquaScope's `scientific` extra includes everything you need for
  NetCDF and HDF5 round-trips.
