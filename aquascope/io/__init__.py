"""AquaScope I/O module — interoperability with hydrological data formats.

Provides read/write support for:
- **WaterML 2.0** (OGC/ISO standard XML for hydrological time series)
- **HEC-DSS / HEC-RAS** (US Army Corps of Engineers modelling formats)
- **EPA SWMM** (Storm Water Management Model input formats)
"""

from __future__ import annotations

from aquascope.io.hec import (
    HECDSSRecord,
    dataframe_to_hec_format,
    write_hec_dss_csv,
    write_hec_ras_flow,
)
from aquascope.io.swmm import write_swmm_rainfall, write_swmm_timeseries
from aquascope.io.waterml import (
    WaterMLTimeSeries,
    dataframe_to_waterml,
    read_waterml,
    waterml_to_dataframe,
    write_waterml,
)

__all__ = [
    "HECDSSRecord",
    "WaterMLTimeSeries",
    "dataframe_to_hec_format",
    "dataframe_to_waterml",
    "read_waterml",
    "waterml_to_dataframe",
    "write_hec_dss_csv",
    "write_hec_ras_flow",
    "write_swmm_rainfall",
    "write_swmm_timeseries",
    "write_waterml",
]
