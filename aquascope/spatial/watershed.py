"""
Watershed delineation and stream-order computation.

Provides catchment boundary extraction from D8 flow direction grids
and Strahler stream ordering for channel networks.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from aquascope.spatial.dem import DEMData
from aquascope.spatial.flow import _NEIGHBOURS, _code_to_index

logger = logging.getLogger(__name__)


@dataclass
class Watershed:
    """Delineated catchment area.

    Attributes
    ----------
    mask : np.ndarray
        Boolean mask where ``True`` marks cells inside the catchment.
    area_km2 : float
        Catchment area in square kilometres.
    pour_point : tuple[int, int]
        ``(row, col)`` of the outlet cell.
    boundary : list[tuple[float, float]]
        ``(lon, lat)`` coordinates tracing the catchment boundary.
    stream_cells : np.ndarray
        Boolean mask of stream network cells within the catchment.
    """

    mask: np.ndarray
    area_km2: float
    pour_point: tuple[int, int]
    boundary: list[tuple[float, float]] = field(default_factory=list)
    stream_cells: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


# ── Public API ───────────────────────────────────────────────────────


def delineate_watershed(
    flow_dir: np.ndarray,
    pour_point: tuple[int, int],
    dem: DEMData | None = None,
) -> Watershed:
    """Delineate the catchment draining to *pour_point*.

    Uses a BFS flood-fill that traces upstream: starting at the outlet
    it finds every cell whose D8 flow path leads to *pour_point*.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid (``uint8``).
    pour_point : tuple[int, int]
        ``(row, col)`` of the outlet.
    dem : DEMData | None
        Optional DEM used to compute area and boundary coordinates.

    Returns
    -------
    Watershed
        Delineated catchment object.
    """
    rows, cols = flow_dir.shape
    mask = np.zeros((rows, cols), dtype=bool)
    pr, pc = pour_point

    if not (0 <= pr < rows and 0 <= pc < cols):
        raise ValueError(f"Pour point ({pr}, {pc}) is outside the grid ({rows}×{cols}).")

    logger.info("Delineating watershed from pour point (%d, %d)", pr, pc)

    # BFS upstream: find all cells that flow INTO current cell
    mask[pr, pc] = True
    queue: deque[tuple[int, int]] = deque()
    queue.append((pr, pc))

    # Reverse-lookup: for each neighbour direction i, the opposite
    # direction code is what a neighbour must have to flow into us.
    reverse_map = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}

    while queue:
        r, c = queue.popleft()
        for i, (dr, dc) in enumerate(_NEIGHBOURS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                # Does neighbour (nr, nc) flow into (r, c)?
                rev_idx = reverse_map[i]
                expected_code = [1, 2, 4, 8, 16, 32, 64, 128][rev_idx]
                if flow_dir[nr, nc] == expected_code:
                    mask[nr, nc] = True
                    queue.append((nr, nc))

    # Compute area
    area_km2 = _compute_area(mask, dem)

    # Extract boundary coordinates
    boundary = _extract_boundary(mask, dem)

    return Watershed(
        mask=mask,
        area_km2=area_km2,
        pour_point=pour_point,
        boundary=boundary,
        stream_cells=np.zeros_like(mask),
    )


def snap_pour_point(
    accumulation: np.ndarray,
    point: tuple[int, int],
    snap_distance: int = 5,
) -> tuple[int, int]:
    """Snap a pour point to the nearest high-accumulation cell.

    Searches within a square window of radius *snap_distance* around
    *point* and returns the ``(row, col)`` of the cell with the
    highest flow accumulation.

    Parameters
    ----------
    accumulation : np.ndarray
        Flow accumulation grid.
    point : tuple[int, int]
        ``(row, col)`` of the initial pour point.
    snap_distance : int
        Search radius in cells.

    Returns
    -------
    tuple[int, int]
        Snapped ``(row, col)``.
    """
    rows, cols = accumulation.shape
    r, c = point
    best = (r, c)
    best_acc = -1

    r_min = max(0, r - snap_distance)
    r_max = min(rows, r + snap_distance + 1)
    c_min = max(0, c - snap_distance)
    c_max = min(cols, c + snap_distance + 1)

    for rr in range(r_min, r_max):
        for cc in range(c_min, c_max):
            if accumulation[rr, cc] > best_acc:
                best_acc = accumulation[rr, cc]
                best = (rr, cc)

    logger.debug("Snapped pour point (%d, %d) → (%d, %d)  acc=%d", r, c, best[0], best[1], best_acc)
    return best


def strahler_order(flow_dir: np.ndarray, streams: np.ndarray) -> np.ndarray:
    """Compute Strahler stream order for a channel network.

    Rules:

    * Headwater segments (no upstream tributaries) receive order 1.
    * When two streams of the same order *n* merge the result is
      *n + 1*.
    * When streams of different orders merge the higher order is kept.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid (``uint8``).
    streams : np.ndarray
        Boolean mask of stream cells.

    Returns
    -------
    np.ndarray
        ``int32`` grid of Strahler orders (non-stream cells are ``0``).
    """
    rows, cols = flow_dir.shape
    order = np.zeros((rows, cols), dtype=np.int32)

    # Build in-degree map (stream cells only)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            if not streams[r, c]:
                continue
            code = flow_dir[r, c]
            if code == 0:
                continue
            idx = _code_to_index(code)
            if idx is None:
                continue
            dr, dc = _NEIGHBOURS[idx]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and streams[nr, nc]:
                in_degree[nr, nc] += 1

    # Seed with headwater cells (in-degree 0 within stream network)
    queue: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if streams[r, c] and in_degree[r, c] == 0:
                order[r, c] = 1
                queue.append((r, c))

    # Track tributary orders flowing into each cell
    incoming_orders: dict[tuple[int, int], list[int]] = {}

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
        if not (0 <= nr < rows and 0 <= nc < cols and streams[nr, nc]):
            continue

        key = (nr, nc)
        if key not in incoming_orders:
            incoming_orders[key] = []
        incoming_orders[key].append(order[r, c])

        in_degree[nr, nc] -= 1
        if in_degree[nr, nc] == 0:
            orders = incoming_orders[key]
            max_ord = max(orders)
            count_max = orders.count(max_ord)
            order[nr, nc] = max_ord + 1 if count_max >= 2 else max_ord
            queue.append((nr, nc))

    return order


# ── Helpers ──────────────────────────────────────────────────────────


def _compute_area(mask: np.ndarray, dem: DEMData | None) -> float:
    """Compute catchment area in km² from a boolean mask."""
    n_cells = int(mask.sum())
    if dem is not None and dem.transform is not None:
        cell_x = abs(dem.transform[0])
        cell_y = abs(dem.transform[4])
        cell_area_m2 = cell_x * cell_y
        return n_cells * cell_area_m2 / 1e6
    return float(n_cells)


def _extract_boundary(mask: np.ndarray, dem: DEMData | None) -> list[tuple[float, float]]:
    """Extract (lon, lat) boundary coordinates from mask edge cells."""
    rows, cols = mask.shape
    boundary: list[tuple[float, float]] = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            is_edge = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not mask[nr, nc]:
                    is_edge = True
                    break
            if is_edge:
                if dem is not None and dem.transform is not None:
                    x, y = dem.transform * (c + 0.5, r + 0.5)
                    boundary.append((x, y))
                else:
                    boundary.append((float(c), float(r)))

    return boundary
