"""
Helpers for persisting collected data to CSV / Parquet / JSON.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from aquascope.utils.imports import require

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def save_records(
    records: Sequence[BaseModel],
    dest_dir: str | Path = "data/raw",
    prefix: str = "water_data",
    fmt: str = "json",
) -> Path:
    """
    Persist a list of Pydantic model instances.

    Parameters
    ----------
    records : list[BaseModel]
        Data records (WaterQualitySample, WaterLevelReading, etc.).
    dest_dir : str | Path
        Directory to write to.
    prefix : str
        File name prefix.
    fmt : str
        ``"json"`` or ``"csv"``.

    Returns
    -------
    Path  — the file that was written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.{fmt}"
    filepath = dest / filename

    dicts = [r.model_dump(mode="json") for r in records]

    if fmt == "json":
        filepath.write_text(json.dumps(dicts, ensure_ascii=False, indent=2, default=str))
    elif fmt == "csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV export.  pip install pandas")  # noqa: B904
        df = pd.DataFrame(dicts)
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")

    logger.info("Saved %d records → %s", len(records), filepath)
    return filepath


def export_netcdf(
    df: pd.DataFrame,
    path: str | Path,
    *,
    variable_name: str = "value",
    attrs: dict | None = None,
) -> Path:
    """Export a DataFrame to CF-convention NetCDF via xarray.

    A ``DatetimeIndex`` is mapped to a ``time`` dimension.  If ``latitude``
    and ``longitude`` columns exist they are promoted to coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.  A ``DatetimeIndex`` becomes the *time* dimension.
    path : str | Path
        Destination file path (should end in ``.nc``).
    variable_name : str
        Name for the primary data variable in the NetCDF dataset.
    attrs : dict | None
        Extra global attributes merged into the file metadata.

    Returns
    -------
    Path — the file that was written.

    Raises
    ------
    ImportError
        If *xarray* or *netCDF4* is not installed.
    """
    xr = require("xarray", feature="NetCDF export")

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset({variable_name: xr.DataArray(df)})

    # Promote lat/lon columns to coordinates when present.
    for col in ("latitude", "longitude"):
        if col in df.columns:
            ds = ds.assign_coords({col: ("index", df[col].values)})

    # If the index is datetime, rename to 'time' for CF compliance.
    import pandas as _pd

    if hasattr(df.index, "dtype") and _pd.api.types.is_datetime64_any_dtype(df.index):
        ds = ds.rename({"index": "time"})

    global_attrs: dict[str, str] = {
        "title": "AquaScope water data export",
        "source": "AquaScope",
        "Conventions": "CF-1.8",
    }
    if attrs:
        global_attrs.update(attrs)
    ds.attrs = global_attrs

    ds.to_netcdf(filepath)
    logger.info("Exported NetCDF → %s", filepath)
    return filepath


def export_geojson(records: Sequence[BaseModel], path: str | Path) -> Path:
    """Export Pydantic records with a ``location`` attribute to GeoJSON.

    Each record that contains a ``location`` field (expected to have
    ``latitude`` and ``longitude`` attributes) becomes a GeoJSON *Feature*.
    Records without a ``location`` are silently skipped after a warning.

    Parameters
    ----------
    records : Sequence[BaseModel]
        Pydantic model instances.  Those with a ``location`` attribute
        carrying ``latitude`` / ``longitude`` are included.
    path : str | Path
        Destination file path (should end in ``.geojson`` or ``.json``).

    Returns
    -------
    Path — the file that was written.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    features: list[dict] = []
    skipped = 0
    for rec in records:
        loc = getattr(rec, "location", None)
        if loc is None or not hasattr(loc, "latitude") or not hasattr(loc, "longitude"):
            skipped += 1
            continue

        props = rec.model_dump(mode="json")
        props.pop("location", None)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [loc.longitude, loc.latitude],
            },
            "properties": props,
        }
        features.append(feature)

    if skipped:
        logger.warning("Skipped %d records without a valid location attribute", skipped)

    collection = {"type": "FeatureCollection", "features": features}
    filepath.write_text(json.dumps(collection, ensure_ascii=False, indent=2, default=str))
    logger.info("Exported GeoJSON (%d features) → %s", len(features), filepath)
    return filepath


def export_hdf5(df: pd.DataFrame, path: str | Path, *, key: str = "data") -> Path:
    """Export a DataFrame to HDF5 via ``pandas.to_hdf``.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    path : str | Path
        Destination file path (should end in ``.h5`` or ``.hdf5``).
    key : str
        HDF5 group key under which the table is stored.

    Returns
    -------
    Path — the file that was written.

    Raises
    ------
    ImportError
        If *tables* (PyTables) is not installed.
    """
    require("tables", feature="HDF5 export")

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_hdf(filepath, key=key, mode="w")
    logger.info("Exported HDF5 → %s", filepath)
    return filepath
