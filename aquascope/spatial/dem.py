"""
DEM (Digital Elevation Model) processing utilities.

Provides loading, sink filling, and slope computation for
raster elevation data stored as GeoTIFF files.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquascope.utils.imports import require

logger = logging.getLogger(__name__)


@dataclass
class DEMData:
    """Container for a loaded DEM raster.

    Attributes
    ----------
    elevation : np.ndarray
        2-D array of elevation values.
    transform : Any
        Rasterio ``Affine`` transform mapping pixel to CRS coordinates.
    crs : Any
        Coordinate reference system (rasterio CRS object).
    nodata : float
        Value used to represent missing data in the elevation grid.
    shape : tuple[int, int]
        ``(rows, cols)`` of the elevation array.
    """

    elevation: np.ndarray
    transform: Any  # rasterio Affine
    crs: Any
    nodata: float
    shape: tuple[int, int]


# ── Public API ───────────────────────────────────────────────────────


def load_dem(path: str | Path) -> DEMData:
    """Load a DEM from a GeoTIFF file using *rasterio*.

    Parameters
    ----------
    path : str | Path
        Path to a GeoTIFF file containing a single-band elevation raster.

    Returns
    -------
    DEMData
        Populated DEM container.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file cannot be read by rasterio.
    """
    rasterio = require("rasterio", feature="DEM processing")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")

    logger.info("Loading DEM from %s", path)
    with rasterio.open(path) as src:
        elevation = src.read(1).astype(np.float64)
        nodata = float(src.nodata) if src.nodata is not None else -9999.0
        return DEMData(
            elevation=elevation,
            transform=src.transform,
            crs=src.crs,
            nodata=nodata,
            shape=elevation.shape,
        )


def fill_sinks(dem: DEMData) -> DEMData:
    """Fill sinks (depressions) in a DEM using the priority-flood algorithm.

    Processes cells from the grid boundary inward via a min-heap.  Any
    interior cell whose elevation is lower than its already-processed
    neighbour is raised to match that neighbour, effectively removing
    local depressions.

    Parameters
    ----------
    dem : DEMData
        Input DEM (not modified in-place).

    Returns
    -------
    DEMData
        New ``DEMData`` with sinks filled.
    """
    logger.info("Filling sinks on %s DEM", dem.shape)
    elev = dem.elevation.copy()
    rows, cols = elev.shape
    filled = np.full_like(elev, np.inf)
    visited = np.zeros((rows, cols), dtype=bool)

    # Seed the heap with boundary cells
    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                filled[r, c] = elev[r, c]
                visited[r, c] = True
                heapq.heappush(heap, (elev[r, c], r, c))

    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                raised = max(elev[nr, nc], h)
                filled[nr, nc] = raised
                heapq.heappush(heap, (raised, nr, nc))

    return DEMData(
        elevation=filled,
        transform=dem.transform,
        crs=dem.crs,
        nodata=dem.nodata,
        shape=dem.shape,
    )


def compute_slope(dem: DEMData) -> np.ndarray:
    """Compute slope in degrees from a DEM using a 3×3 gradient window.

    Uses ``numpy.gradient`` to estimate ∂z/∂x and ∂z/∂y, then converts
    the magnitude to degrees.  Cell sizes are derived from the DEM
    transform when available; otherwise a unit cell size is assumed.

    Parameters
    ----------
    dem : DEMData
        Input DEM.

    Returns
    -------
    np.ndarray
        Slope in degrees with the same shape as the DEM.
    """
    elev = dem.elevation
    cellsize_x = abs(dem.transform[0]) if dem.transform is not None else 1.0
    cellsize_y = abs(dem.transform[4]) if dem.transform is not None else 1.0

    dz_dy, dz_dx = np.gradient(elev, cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return np.degrees(slope_rad)
