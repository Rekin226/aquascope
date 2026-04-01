"""
Flow direction and accumulation analysis.

Implements the D8 (deterministic eight-neighbour) flow-routing model
commonly used in terrain analysis and watershed hydrology.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from aquascope.spatial.dem import DEMData

logger = logging.getLogger(__name__)

# D8 direction encoding (ArcGIS convention):
#   32  64  128
#   16   X    1
#    8   4    2
#
# Index order matches _NEIGHBOURS below.
_D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

# (row_offset, col_offset) for E, SE, S, SW, W, NW, N, NE
_NEIGHBOURS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

# Distance weights: 1.0 for cardinal, sqrt(2) for diagonal
_DISTANCES = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])


# ── Public API ───────────────────────────────────────────────────────


def flow_direction_d8(dem: DEMData) -> np.ndarray:
    """Compute D8 flow direction for every cell in *dem*.

    For each cell the steepest downhill neighbour among its eight
    neighbours is selected.  The returned grid is encoded using the
    ArcGIS convention: ``1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N,
    128=NE``.  Flat cells default to the first valid (non-uphill)
    neighbour; cells with no downhill neighbour are assigned ``0``.

    Parameters
    ----------
    dem : DEMData
        Input DEM.

    Returns
    -------
    np.ndarray
        ``uint8`` grid of D8 direction codes with the same shape as
        the DEM.
    """
    elev = dem.elevation
    rows, cols = elev.shape
    fdir = np.zeros((rows, cols), dtype=np.uint8)

    logger.info("Computing D8 flow directions for %s grid", (rows, cols))

    for r in range(rows):
        for c in range(cols):
            max_drop = 0.0
            best_code: int = 0
            first_valid: int = 0
            for i, (dr, dc) in enumerate(_NEIGHBOURS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = (elev[r, c] - elev[nr, nc]) / _DISTANCES[i]
                    if drop > max_drop:
                        max_drop = drop
                        best_code = int(_D8_CODES[i])
                    if first_valid == 0 and elev[nr, nc] <= elev[r, c]:
                        first_valid = int(_D8_CODES[i])

            # Flat handling: use first non-uphill neighbour
            fdir[r, c] = best_code if best_code != 0 else first_valid

    return fdir


def flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """Compute flow accumulation from a D8 flow direction grid.

    Each cell's accumulation value equals the number of upstream cells
    whose flow path passes through it.  A topological-sort approach is
    used: cells with zero in-degree (no upstream contributors) are
    processed first, propagating counts downstream.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid (``uint8``).

    Returns
    -------
    np.ndarray
        ``int64`` accumulation grid (same shape as *flow_dir*).
    """
    rows, cols = flow_dir.shape
    accum = np.ones((rows, cols), dtype=np.int64)
    in_degree = np.zeros((rows, cols), dtype=np.int64)

    logger.info("Computing flow accumulation for %s grid", (rows, cols))

    # Build in-degree map
    for r in range(rows):
        for c in range(cols):
            code = flow_dir[r, c]
            if code == 0:
                continue
            idx = _code_to_index(code)
            if idx is None:
                continue
            dr, dc = _NEIGHBOURS[idx]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                in_degree[nr, nc] += 1

    # Seed queue with zero-in-degree cells
    queue: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if in_degree[r, c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        code = flow_dir[r, c]
        if code == 0:
            continue
        idx = _code_to_index(code)
        if idx is None:
            continue
        dr, dc = _NEIGHBOURS[idx]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr, nc] += accum[r, c]
            in_degree[nr, nc] -= 1
            if in_degree[nr, nc] == 0:
                queue.append((nr, nc))

    return accum


def extract_streams(accumulation: np.ndarray, threshold: int = 100) -> np.ndarray:
    """Extract a stream network where accumulation ≥ *threshold*.

    Parameters
    ----------
    accumulation : np.ndarray
        Flow accumulation grid.
    threshold : int
        Minimum number of upstream cells for a cell to be classified
        as part of the stream network.

    Returns
    -------
    np.ndarray
        Boolean mask (``True`` = stream cell).
    """
    return accumulation >= threshold


# ── Helpers ──────────────────────────────────────────────────────────


def _code_to_index(code: int) -> int | None:
    """Map a D8 direction code to the index into ``_NEIGHBOURS``."""
    code_map = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7}
    return code_map.get(code)
