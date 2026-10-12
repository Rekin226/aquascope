"""AquaScope I/O module — interoperability with hydrological data formats.

Provides read/write support for:
- **WaterML 2.0** (OGC/ISO standard XML for hydrological time series)
- **HEC-DSS / HEC-RAS** (US Army Corps of Engineers modelling formats)
- **EPA SWMM** (Storm Water Management Model: input writers, and readers for a model and its binary results)
- **Model results on a mesh** (HEC-RAS 2D plan HDF5 and UGRID NetCDF, in :mod:`aquascope.io.mesh`)

The names below are imported on first use, so ``import aquascope.io.mesh`` (which the Explorer's mesh worker
runs in the browser) does not pull in pandas.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "HECDSSRecord": "aquascope.io.hec",
    "dataframe_to_hec_format": "aquascope.io.hec",
    "write_hec_dss_csv": "aquascope.io.hec",
    "write_hec_ras_flow": "aquascope.io.hec",
    "records_to_geodataframe": "aquascope.io.interop",
    "records_to_xarray": "aquascope.io.interop",
    "write_swmm_rainfall": "aquascope.io.swmm",
    "write_swmm_timeseries": "aquascope.io.swmm",
    "read_swmm_inp": "aquascope.io.swmm",
    "read_swmm_out": "aquascope.io.swmm",
    "swmm_summary": "aquascope.io.swmm",
    "WaterMLTimeSeries": "aquascope.io.waterml",
    "dataframe_to_waterml": "aquascope.io.waterml",
    "read_waterml": "aquascope.io.waterml",
    "waterml_to_dataframe": "aquascope.io.waterml",
    "write_waterml": "aquascope.io.waterml",
    "read_mesh": "aquascope.io.mesh",
    "read_ras_plan": "aquascope.io.mesh",
    "read_ugrid": "aquascope.io.mesh",
    "mesh_summary": "aquascope.io.mesh",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'aquascope.io' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
