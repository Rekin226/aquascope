"""Convert AquaScope records to scientific-Python geo objects.

Bridges the unified Pydantic schema to ``xarray.Dataset`` (gridded/time-series)
and ``geopandas.GeoDataFrame`` (point/station), so AquaScope data can feed the
wider Pangeo / NeuralHydrology ecosystem instead of being a closed schema.

Both helpers lazily import their backends and are available via the optional
``interop`` extra: ``pip install 'aquascope[interop]'``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from aquascope.utils.imports import require

if TYPE_CHECKING:  # pragma: no cover
    import geopandas
    import xarray

logger = logging.getLogger(__name__)


def _source_label(records: list[Any]) -> str:
    src = getattr(records[0], "source", None)
    return getattr(src, "value", str(src)) if src is not None else "unknown"


def _attach_station_coords(ds: xarray.Dataset, loc_map: dict[str, Any]) -> xarray.Dataset:
    """Attach lat/lon coords along the ``station_id`` dimension.

    Stations without a location get NaN coordinates rather than being dropped.
    """
    sids = [str(s) for s in ds["station_id"].values]
    lat = [getattr(loc_map.get(s), "latitude", float("nan")) for s in sids]
    lon = [getattr(loc_map.get(s), "longitude", float("nan")) for s in sids]
    return ds.assign_coords(
        lat=("station_id", lat),
        lon=("station_id", lon),
    )


def records_to_xarray(records: Sequence[Any]) -> xarray.Dataset:
    """Convert time-series records to an ``xarray.Dataset``.

    Supports ``WaterQualitySample`` (one data variable per parameter),
    ``StreamflowReading`` (a single ``discharge`` variable), and
    ``WaterLevelReading`` (a single ``water_level`` variable). The result has
    dimensions ``(time, station_id)`` with ``lat``/``lon`` station coordinates
    and per-variable ``units`` attributes.

    An empty input yields an empty ``Dataset``.
    """
    xr = require("xarray", feature="xarray export", group="interop")
    import pandas as pd

    records = list(records)
    if not records:
        return xr.Dataset()

    first = records[0]

    # Discriminate on the timestamp/value field. StreamflowReading and
    # WaterLevelReading both use reading_datetime, so discharge_cms must be
    # checked first (ReservoirStatus also has water_level, so that alone
    # isn't a safe discriminator either).
    if hasattr(first, "sample_datetime"):
        units: dict[str, str] = {}
        loc_map: dict[str, Any] = {}
        rows = []
        for r in records:
            rows.append(
                {
                    "time": r.sample_datetime,
                    "station_id": r.station_id,
                    "parameter": r.parameter,
                    "value": r.value,
                }
            )
            if r.parameter not in units and r.unit:
                units[r.parameter] = r.unit
            loc_map.setdefault(r.station_id, r.location)
        wide = pd.DataFrame(rows).pivot_table(
            index=["time", "station_id"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        ds = wide.to_xarray()
        for param in ds.data_vars:
            if units.get(str(param)):
                ds[param].attrs["units"] = units[str(param)]

    elif hasattr(first, "discharge_cms"):
        loc_map = {}
        unit: str | None = None
        rows = []
        for r in records:
            rows.append(
                {
                    "time": r.reading_datetime,
                    "station_id": r.station_id,
                    "discharge": r.discharge_cms,
                }
            )
            loc_map.setdefault(r.station_id, r.location)
            unit = unit or r.unit
        wide = pd.DataFrame(rows).groupby(["time", "station_id"])["discharge"].mean().to_frame()
        ds = wide.to_xarray()
        if unit:
            ds["discharge"].attrs["units"] = unit

    elif hasattr(first, "reading_datetime"):
        loc_map = {}
        unit: str | None = None
        rows = []
        for r in records:
            rows.append(
                {
                    "time": r.reading_datetime,
                    "station_id": r.station_id,
                    "water_level": r.water_level,
                }
            )
            loc_map.setdefault(r.station_id, r.location)
            unit = unit or r.unit
        wide = pd.DataFrame(rows).groupby(["time", "station_id"])["water_level"].mean().to_frame()
        ds = wide.to_xarray()
        if unit:
            ds["water_level"].attrs["units"] = unit

    else:
        raise TypeError(
            "records_to_xarray supports WaterQualitySample, WaterLevelReading, "
            f"and StreamflowReading records; got {type(first).__name__}."
        )

    ds = _attach_station_coords(ds, loc_map)
    ds.attrs["source"] = _source_label(records)
    return ds


def records_to_geodataframe(records: Sequence[Any]) -> geopandas.GeoDataFrame:
    """Convert point/station records to a ``geopandas.GeoDataFrame``.

    One row per record, with ``Point`` geometry in EPSG:4326. Records without
    a location keep a null geometry (the count is logged) rather than being
    dropped. The nested ``location`` field is flattened to ``latitude`` /
    ``longitude`` columns.
    """
    gpd = require("geopandas", feature="GeoDataFrame export", group="interop")
    require("shapely", feature="GeoDataFrame export", group="interop")
    from shapely.geometry import Point

    records = list(records)
    if not records:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    rows = []
    geoms = []
    n_missing = 0
    for r in records:
        data = r.model_dump()
        loc = data.pop("location", None)
        src = data.get("source")
        if src is not None:
            data["source"] = getattr(src, "value", src)
        if loc:
            data["latitude"] = loc["latitude"]
            data["longitude"] = loc["longitude"]
            geoms.append(Point(loc["longitude"], loc["latitude"]))
        else:
            data["latitude"] = None
            data["longitude"] = None
            geoms.append(None)
            n_missing += 1
        rows.append(data)

    if n_missing:
        logger.info(
            "%d of %d records have no location; null geometry assigned.",
            n_missing,
            len(records),
        )

    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
