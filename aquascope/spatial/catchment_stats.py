"""
Station-level catchment morphometric analysis.

Computes physical descriptors (area, slope, stream density, etc.) for
delineated watersheds and provides batch processing of monitoring
stations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from aquascope.spatial.dem import DEMData, compute_slope
from aquascope.spatial.watershed import Watershed, delineate_watershed, snap_pour_point

logger = logging.getLogger(__name__)


@dataclass
class CatchmentStats:
    """Morphometric statistics for a delineated catchment.

    Attributes
    ----------
    area_km2 : float
        Catchment area in square kilometres.
    mean_elevation : float
        Mean elevation of cells within the catchment.
    mean_slope : float
        Mean slope (degrees) of cells within the catchment.
    stream_density : float
        Stream density in km of channel per km² of catchment.
    elongation_ratio : float
        Ratio of the diameter of a circle with the same area to the
        maximum catchment length.
    """

    area_km2: float
    mean_elevation: float
    mean_slope: float
    stream_density: float
    elongation_ratio: float


# ── Public API ───────────────────────────────────────────────────────


def compute_catchment_stats(
    watershed: Watershed,
    dem: DEMData,
    streams: np.ndarray,
) -> CatchmentStats:
    """Compute morphometric statistics for a delineated watershed.

    Parameters
    ----------
    watershed : Watershed
        Delineated catchment (boolean mask + metadata).
    dem : DEMData
        DEM covering the same extent.
    streams : np.ndarray
        Boolean mask of stream-network cells.

    Returns
    -------
    CatchmentStats
        Computed descriptors.
    """
    mask = watershed.mask
    elev = dem.elevation[mask]

    mean_elevation = float(np.mean(elev)) if elev.size > 0 else 0.0

    slope_grid = compute_slope(dem)
    mean_slope = float(np.mean(slope_grid[mask])) if mask.any() else 0.0

    # Stream density (km of stream per km²)
    stream_in_ws = streams & mask
    n_stream_cells = int(stream_in_ws.sum())
    cell_res_m = abs(dem.transform[0]) if dem.transform is not None else 1.0
    stream_length_km = n_stream_cells * cell_res_m / 1000.0
    area_km2 = watershed.area_km2 if watershed.area_km2 > 0 else 1.0
    stream_density = stream_length_km / area_km2

    # Elongation ratio: diameter of equal-area circle / max length
    elongation_ratio = _elongation_ratio(mask, cell_res_m)

    return CatchmentStats(
        area_km2=watershed.area_km2,
        mean_elevation=round(mean_elevation, 2),
        mean_slope=round(mean_slope, 2),
        stream_density=round(stream_density, 4),
        elongation_ratio=round(elongation_ratio, 4),
    )


def stations_to_catchments(
    stations: list[dict],
    dem: DEMData,
    flow_dir: np.ndarray,
    accumulation: np.ndarray,
    snap_distance: int = 5,
) -> list[tuple[dict, Watershed]]:
    """Delineate catchments for a list of monitoring stations.

    Each station dict must contain ``"latitude"`` and ``"longitude"``
    keys.  The geographic coordinates are converted to row/col indices,
    snapped to the nearest high-accumulation cell, then a watershed is
    delineated.

    Parameters
    ----------
    stations : list[dict]
        Station records with at least ``latitude`` and ``longitude``.
    dem : DEMData
        DEM for the study area.
    flow_dir : np.ndarray
        D8 flow direction grid.
    accumulation : np.ndarray
        Flow accumulation grid.
    snap_distance : int
        Search radius (in cells) for snapping to stream.

    Returns
    -------
    list[tuple[dict, Watershed]]
        Pairs of ``(station_dict, Watershed)``.
    """
    results: list[tuple[dict, Watershed]] = []

    for station in stations:
        lat = station.get("latitude")
        lon = station.get("longitude")
        if lat is None or lon is None:
            logger.warning("Station %s missing coordinates — skipped.", station.get("station_id", "?"))
            continue

        row, col = _lonlat_to_rowcol(lon, lat, dem)
        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            logger.warning("Station %s falls outside DEM — skipped.", station.get("station_id", "?"))
            continue

        snapped = snap_pour_point(accumulation, (row, col), snap_distance)
        ws = delineate_watershed(flow_dir, snapped, dem=dem)
        results.append((station, ws))
        logger.info(
            "Station %s → catchment %.2f km²",
            station.get("station_id", "?"),
            ws.area_km2,
        )

    return results


# ── Helpers ──────────────────────────────────────────────────────────


def _lonlat_to_rowcol(lon: float, lat: float, dem: DEMData) -> tuple[int, int]:
    """Convert geographic coordinates to pixel row/col using the DEM transform."""
    if dem.transform is not None:
        inv = ~dem.transform
        col_f, row_f = inv * (lon, lat)
        return int(round(row_f)), int(round(col_f))
    return int(round(lat)), int(round(lon))


def _elongation_ratio(mask: np.ndarray, cell_res_m: float) -> float:
    """Compute elongation ratio from a catchment mask."""
    n_cells = int(mask.sum())
    if n_cells == 0:
        return 0.0

    area_m2 = n_cells * cell_res_m * cell_res_m
    diameter = 2.0 * math.sqrt(area_m2 / math.pi)

    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 1.0

    # Max length: furthest pair of catchment cells (approximate via bbox diagonal)
    row_range = (coords[:, 0].max() - coords[:, 0].min()) * cell_res_m
    col_range = (coords[:, 1].max() - coords[:, 1].min()) * cell_res_m
    max_length = math.sqrt(row_range ** 2 + col_range ** 2)

    return diameter / max_length if max_length > 0 else 1.0
