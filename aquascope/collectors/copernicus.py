"""
Copernicus Climate Data Store (CDS) collector — GloFAS river-discharge forecasts.

Provides access to Copernicus Climate Change Service datasets via the CDS API.
Currently focuses on GloFAS (Global Flood Awareness System) river discharge data.

Requirements
------------
- ``cdsapi`` Python package: ``pip install cdsapi``
- A ``.cdsapirc`` file or ``CDSAPI_URL`` / ``CDSAPI_KEY`` environment variables.
  Register free at https://cds.climate.copernicus.eu/
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterQualitySample

logger = logging.getLogger(__name__)


class CopernicusCollector(BaseCollector):
    """Fetch GloFAS river-discharge data from Copernicus CDS.

    Parameters
    ----------
    dataset : str
        CDS dataset ID.  Default is the GloFAS historical dataset.

    Example
    -------
    >>> collector = CopernicusCollector()
    >>> records = collector.collect(
    ...     latitude=48.85, longitude=2.35,
    ...     year="2023", month=["01", "02", "03"],
    ... )
    """

    name = "copernicus"

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
        variable: str = "river_discharge_in_the_last_24_hours",
        product_type: str = "consolidated",
        system_version: str = "version_4_0",
    ) -> list[dict]:
        """Fetch data via CDS API and return parsed records.

        Parameters
        ----------
        latitude, longitude : float
            Site coordinates.
        year, month, day : str | list[str]
            Temporal selection.
        variable : str
            CDS variable name.
        product_type, system_version : str
            CDS-specific dataset options.
        """
        try:
            import cdsapi
        except ImportError:
            raise ImportError("Install cdsapi: pip install cdsapi")

        request: dict = {
            "variable": variable,
            "product_type": product_type,
            "system_version": system_version,
            "hydrological_model": "lisflood",
            "year": year,
            "month": month,
            "format": "grib",
            "area": [latitude + 0.5, longitude - 0.5, latitude - 0.5, longitude + 0.5],
        }
        if day:
            request["day"] = day

        c = cdsapi.Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = str(Path(tmpdir) / "download.grib")
            c.retrieve(self.dataset, request, target)
            return self._parse_grib(target, latitude, longitude)

    def _parse_grib(self, path: str, lat: float, lon: float) -> list[dict]:
        """Parse a GRIB file into a list of dicts."""
        try:
            import xarray as xr

            ds = xr.open_dataset(path, engine="cfgrib")
        except ImportError:
            raise ImportError("Install cfgrib + xarray: pip install cfgrib xarray")

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
        """Convert parsed GRIB records into ``WaterQualitySample`` objects."""
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

        logger.info("Normalised %d records from Copernicus CDS", len(records))
        return records
