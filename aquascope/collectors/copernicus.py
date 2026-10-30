"""
Copernicus Early Warning Data Store (EWDS) collector — GloFAS river discharge.

GloFAS is published on the Copernicus Early Warning Data Store
(https://ewds.climate.copernicus.eu/), not on the Climate Data Store, so this
collector always sends its requests to the EWDS API.  The datasets are
distributed under the CEMS-FLOODS licence; accept it on the dataset page.

Requirements
------------
- ``cdsapi`` Python package: ``pip install aquascope[copernicus]``
- A key in ``.cdsapirc`` or ``CDSAPI_KEY`` (``CDSAPI_RC`` points at another
  file).  Register free at https://ewds.climate.copernicus.eu/ and accept the
  CEMS-FLOODS licence for the GloFAS dataset.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterQualitySample

logger = logging.getLogger(__name__)


class CopernicusCollector(BaseCollector):
    """Fetch GloFAS river-discharge data from the Copernicus EWDS.

    Parameters
    ----------
    dataset : str
        EWDS dataset ID.  Default is the GloFAS historical dataset.

    Notes
    -----
    ``variable``, ``product_type`` and ``system_version`` are dataset-specific:
    the historical dataset accepts ``version_4_0`` and ``version_5_0``, while
    the forecast and seasonal reforecast datasets have their own versions and
    variable names.

    Example
    -------
    >>> collector = CopernicusCollector()
    >>> records = collector.collect(
    ...     latitude=48.85, longitude=2.35,
    ...     year="2023", month=["01", "02", "03"],
    ... )
    """

    name = "copernicus"

    #: GloFAS lives on the Early Warning Data Store, not the Climate Data Store.
    EWDS_URL = "https://ewds.climate.copernicus.eu/api"

    #: Every day the dataset form offers.  The EWDS rejects a request without
    #: ``day``, so the whole month is the default.
    ALL_DAYS = tuple(f"{d:02d}" for d in range(1, 32))

    GLOFAS_HISTORICAL = "cems-glofas-historical"
    GLOFAS_FORECAST = "cems-glofas-forecast"
    GLOFAS_REFORECAST = "cems-glofas-seasonal-reforecast"

    def __init__(self, dataset: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset or self.GLOFAS_HISTORICAL

    def fetch_raw(
        self,
        *,
        latitude: float,
        longitude: float,
        year: str | list[str] = "2023",
        month: str | list[str] = "01",
        day: str | list[str] | None = None,
        variable: str = "average_river_discharge_in_the_last_24_hours",
        product_type: str = "consolidated",
        system_version: str = "version_5_0",
        timespan: str = "time_mean",
        data_format: str = "grib2",
        download_format: str = "unarchived",
        api_key: str | None = None,
    ) -> list[dict]:
        """Fetch one GloFAS grid cell from the EWDS and return parsed records.

        Parameters
        ----------
        latitude, longitude : float
            Site coordinates.
        year, month, day : str | list[str]
            Temporal selection.  ``day`` defaults to every day the form
            offers (``"01"`` to ``"31"``) because the EWDS requires it; pass
            it explicitly for a month with fewer days.
        variable : str
            EWDS variable name.  Defaults to the historical daily-average
            river discharge.
        product_type, system_version : str
            EWDS dataset options.  The default system version is
            ``version_5_0``; pass ``version_4_0`` for the older system.
        timespan : str
            EWDS timespan, also required.  ``"time_mean"`` (the default) is
            what the average discharge and runoff variables need; the
            snow-depth and soil-wetness variables are ``"instantaneous"``.
        data_format : str
            ``"grib2"`` (default, parsed with cfgrib) or ``"netcdf"``.
        download_format : str
            EWDS download packaging.  Keep ``"unarchived"`` so the downloaded
            file can be parsed directly.
        api_key : str | None
            EWDS key.  Falls back to ``CDSAPI_KEY`` and then ``.cdsapirc``.
        """
        if data_format not in ("grib2", "netcdf"):
            raise ValueError("data_format must be 'grib2' or 'netcdf'")
        if timespan not in ("time_mean", "instantaneous"):
            raise ValueError("timespan must be 'time_mean' or 'instantaneous'")
        if download_format != "unarchived":
            raise ValueError(
                "download_format must be 'unarchived'; the collector parses the downloaded file directly"
            )

        try:
            import cdsapi
        except ImportError:
            raise ImportError("Install cdsapi: pip install aquascope[copernicus]")

        if day is None:
            day = list(self.ALL_DAYS)
        elif isinstance(day, str):
            day = [day]

        request: dict = {
            "variable": variable,
            "product_type": product_type,
            "system_version": system_version,
            "timespan": timespan,
            "hydrological_model": "lisflood",
            "year": year,
            "month": month,
            "day": day,
            "data_format": data_format,
            "download_format": download_format,
            "area": [latitude + 0.5, longitude - 0.5, latitude - 0.5, longitude + 0.5],
        }

        key = api_key or os.environ.get("CDSAPI_KEY") or None
        client = cdsapi.Client(url=self.EWDS_URL, key=key)
        suffix = ".grib2" if data_format == "grib2" else ".nc"
        with tempfile.TemporaryDirectory() as tmpdir:
            target = str(Path(tmpdir) / f"download{suffix}")
            client.retrieve(self.dataset, request, target)
            return self._parse_download(target, latitude, longitude, data_format)

    def _parse_download(self, path: str, lat: float, lon: float, data_format: str) -> list[dict]:
        """Parse a GRIB2 or NetCDF file into a list of dicts."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("Install xarray: pip install aquascope[copernicus]")

        if data_format == "netcdf":
            ds = xr.open_dataset(path)
        else:
            try:
                ds = xr.open_dataset(path, engine="cfgrib")
            except ImportError:
                raise ImportError("Install cfgrib: pip install aquascope[copernicus]")

        records: list[dict] = []
        for var_name in ds.data_vars:
            da = ds[var_name]
            # Find nearest grid point
            if "latitude" in da.dims:
                da = da.sel(latitude=lat, longitude=lon, method="nearest")
            for i, time_val in enumerate(da.time.values if "time" in da.dims else [da.time.values]):
                val = float(da.values[i]) if "time" in da.dims else float(da.values)
                dt = str(time_val)[:19]
                records.append({
                    "datetime": dt,
                    "variable": str(var_name),
                    "value": val,
                    "latitude": lat,
                    "longitude": lon,
                })
        return records

    def normalise(self, raw: list[dict]) -> Sequence[BaseModel]:
        """Convert parsed EWDS records into ``WaterQualitySample`` objects."""
        records: list[WaterQualitySample] = []
        for r in raw:
            try:
                records.append(WaterQualitySample(
                    source=DataSource.COPERNICUS,
                    station_id=f"copernicus_{r['latitude']}_{r['longitude']}",
                    station_name=f"Copernicus ({r['latitude']}, {r['longitude']})",
                    location=GeoLocation(latitude=r["latitude"], longitude=r["longitude"]),
                    sample_datetime=datetime.fromisoformat(r["datetime"]),
                    parameter=r["variable"],
                    value=r["value"],
                    unit="m³/s",
                ))
            except Exception as e:
                logger.warning("Skipping Copernicus record: %s", e)

        logger.info("Normalised %d records from Copernicus EWDS", len(records))
        return records
