"""Watershed delineation and spatial hydrology analysis."""

from aquascope.spatial.catchment_stats import CatchmentStats, compute_catchment_stats, stations_to_catchments
from aquascope.spatial.dem import DEMData, compute_slope, fill_sinks, load_dem
from aquascope.spatial.flow import extract_streams, flow_accumulation, flow_direction_d8
from aquascope.spatial.watershed import Watershed, delineate_watershed, snap_pour_point, strahler_order

__all__ = [
    "CatchmentStats",
    "DEMData",
    "Watershed",
    "compute_catchment_stats",
    "compute_slope",
    "delineate_watershed",
    "extract_streams",
    "fill_sinks",
    "flow_accumulation",
    "flow_direction_d8",
    "load_dem",
    "snap_pour_point",
    "stations_to_catchments",
    "strahler_order",
]
