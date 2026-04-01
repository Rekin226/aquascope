"""Tests for the spatial hydrology module using synthetic DEMs."""

import os

import numpy as np
import pytest

from aquascope.spatial.catchment_stats import CatchmentStats, compute_catchment_stats, stations_to_catchments
from aquascope.spatial.dem import DEMData, compute_slope, fill_sinks, load_dem
from aquascope.spatial.flow import extract_streams, flow_accumulation, flow_direction_d8
from aquascope.spatial.watershed import Watershed, delineate_watershed, snap_pour_point, strahler_order

# ── Helpers ──────────────────────────────────────────────────────────


def _make_dem(elevation: np.ndarray, cellsize: float = 30.0) -> DEMData:
    """Build a synthetic DEMData from a numpy array."""
    from rasterio.transform import from_bounds

    rows, cols = elevation.shape
    transform = from_bounds(0.0, 0.0, cols * cellsize, rows * cellsize, cols, rows)
    return DEMData(
        elevation=elevation.astype(np.float64),
        transform=transform,
        crs=None,
        nodata=-9999.0,
        shape=elevation.shape,
    )


def _sloped_dem(rows: int = 5, cols: int = 5) -> np.ndarray:
    """Return an elevation grid that slopes south and east (NW=high, SE=low)."""
    r = np.arange(rows).reshape(-1, 1)
    c = np.arange(cols).reshape(1, -1)
    return (10.0 - r - c).astype(np.float64)


def _v_valley_dem(rows: int = 7, cols: int = 7) -> np.ndarray:
    """Return a V-shaped valley with the channel running down the centre column."""
    elev = np.zeros((rows, cols), dtype=np.float64)
    centre = cols // 2
    for r in range(rows):
        for c in range(cols):
            elev[r, c] = abs(c - centre) * 10.0 + (rows - 1 - r) * 1.0
    return elev


def _write_geotiff(path: str, elevation: np.ndarray, cellsize: float = 30.0) -> None:
    """Write a single-band GeoTIFF for testing load_dem."""
    import rasterio
    from rasterio.transform import from_bounds

    rows, cols = elevation.shape
    transform = from_bounds(0.0, 0.0, cols * cellsize, rows * cellsize, cols, rows)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float64",
        crs="EPSG:32650",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation.astype(np.float64), 1)


# ── DEM Tests ────────────────────────────────────────────────────────


class TestLoadDEM:
    def test_load_dem_reads_geotiff(self):
        path = "test_dem_load.tif"
        try:
            elev = _sloped_dem(5, 5)
            _write_geotiff(path, elev)
            dem = load_dem(path)
            assert dem.shape == (5, 5)
            np.testing.assert_array_almost_equal(dem.elevation, elev)
            assert dem.nodata == -9999.0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_dem_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dem("nonexistent.tif")


class TestFillSinks:
    def test_fill_raises_depressed_cell(self):
        elev = np.array([
            [9, 9, 9, 9, 9],
            [9, 8, 8, 8, 9],
            [9, 8, 2, 8, 9],
            [9, 8, 8, 8, 9],
            [9, 9, 9, 9, 5],
        ], dtype=np.float64)
        dem = _make_dem(elev)
        filled = fill_sinks(dem)
        # The sink at (2,2)=2 should be raised to at least its lowest exit
        assert filled.elevation[2, 2] >= 5.0

    def test_fill_preserves_monotone_surface(self):
        elev = _sloped_dem(5, 5)
        dem = _make_dem(elev)
        filled = fill_sinks(dem)
        np.testing.assert_array_almost_equal(filled.elevation, elev)


class TestComputeSlope:
    def test_flat_surface_zero_slope(self):
        elev = np.ones((5, 5), dtype=np.float64) * 100.0
        dem = _make_dem(elev)
        slope = compute_slope(dem)
        assert slope.shape == (5, 5)
        np.testing.assert_array_almost_equal(slope, 0.0)

    def test_sloped_surface_positive(self):
        elev = _sloped_dem(5, 5)
        dem = _make_dem(elev)
        slope = compute_slope(dem)
        # Interior cells should have positive slope
        assert np.all(slope[1:-1, 1:-1] > 0.0)


# ── Flow Tests ───────────────────────────────────────────────────────


class TestFlowDirectionD8:
    def test_simple_slope_flows_southeast(self):
        """On a NW-high / SE-low grid, interior cells should flow SE (code 2)."""
        elev = _sloped_dem(5, 5)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        # Interior cell (1,1): steepest drop is SE (diagonal = 2/sqrt2 ≈ 1.41)
        assert fdir[1, 1] == 2  # SE

    def test_all_cells_have_direction(self):
        elev = _sloped_dem(5, 5)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        # Bottom-right corner might be 0 (no downhill) but all others should be > 0
        assert fdir[0, 0] > 0
        assert fdir[2, 2] > 0


class TestFlowAccumulation:
    def test_v_valley_centre_highest(self):
        """In a V-shaped valley the centre-bottom cell should have the highest accumulation."""
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        accum = flow_accumulation(fdir)
        centre_col = 3
        bottom_row = 6
        # Centre-bottom should be the maximum (or close to it)
        assert accum[bottom_row, centre_col] == accum.max()

    def test_headwater_cells_have_one(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        accum = flow_accumulation(fdir)
        # Top-left corner should be a headwater (accumulation = 1)
        assert accum[0, 0] == 1


class TestExtractStreams:
    def test_threshold_filtering(self):
        accum = np.array([
            [1, 1, 1],
            [1, 50, 1],
            [1, 100, 200],
        ], dtype=np.int64)
        streams = extract_streams(accum, threshold=100)
        assert streams[2, 1] is np.True_
        assert streams[2, 2] is np.True_
        assert streams[1, 1] is np.False_

    def test_high_threshold_empty(self):
        accum = np.ones((5, 5), dtype=np.int64)
        streams = extract_streams(accum, threshold=10)
        assert not streams.any()


# ── Watershed Tests ──────────────────────────────────────────────────


class TestDelineateWatershed:
    def test_traces_upstream_cells(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        ws = delineate_watershed(fdir, pour_point=(6, 3), dem=dem)
        assert isinstance(ws, Watershed)
        # Pour point itself should be in the mask
        assert ws.mask[6, 3]
        # At least some upstream cells should be included
        assert ws.mask.sum() > 1

    def test_pour_point_outside_raises(self):
        elev = _sloped_dem(5, 5)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        with pytest.raises(ValueError):
            delineate_watershed(fdir, pour_point=(10, 10))

    def test_area_positive(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        ws = delineate_watershed(fdir, pour_point=(6, 3), dem=dem)
        assert ws.area_km2 > 0


class TestSnapPourPoint:
    def test_snaps_to_high_accumulation(self):
        accum = np.ones((11, 11), dtype=np.int64)
        # Place a high-accumulation cell at (5, 5)
        accum[5, 5] = 500
        snapped = snap_pour_point(accum, (4, 4), snap_distance=3)
        assert snapped == (5, 5)

    def test_no_better_cell_stays(self):
        accum = np.ones((5, 5), dtype=np.int64)
        snapped = snap_pour_point(accum, (2, 2), snap_distance=2)
        # All cells equal — should pick one (first encountered: top-left of window)
        assert 0 <= snapped[0] < 5
        assert 0 <= snapped[1] < 5


# ── Strahler Order Tests ────────────────────────────────────────────


class TestStrahlerOrder:
    def test_y_junction(self):
        """Two order-1 tributaries merging should produce order 2."""
        # Layout (5x5):
        #   Streams at (0,0)->(1,1), (0,4)->(1,3), both flow to (2,2)->(3,2)->(4,2)
        streams = np.zeros((5, 5), dtype=bool)
        fdir = np.zeros((5, 5), dtype=np.uint8)

        # Left tributary: (0,0) → (1,1) → (2,2)
        streams[0, 0] = True
        fdir[0, 0] = 2   # SE
        streams[1, 1] = True
        fdir[1, 1] = 2   # SE

        # Right tributary: (0,4) → (1,3) → (2,2)
        streams[0, 4] = True
        fdir[0, 4] = 8   # SW
        streams[1, 3] = True
        fdir[1, 3] = 8   # SW

        # Main stem: (2,2) → (3,2) → (4,2)
        streams[2, 2] = True
        fdir[2, 2] = 4   # S
        streams[3, 2] = True
        fdir[3, 2] = 4   # S
        streams[4, 2] = True
        fdir[4, 2] = 0   # outlet

        order = strahler_order(fdir, streams)

        # Headwaters
        assert order[0, 0] == 1
        assert order[0, 4] == 1

        # After Y-junction where two order-1 streams merge
        assert order[2, 2] == 2

    def test_non_stream_cells_zero(self):
        streams = np.zeros((5, 5), dtype=bool)
        fdir = np.zeros((5, 5), dtype=np.uint8)
        order = strahler_order(fdir, streams)
        assert np.all(order == 0)


# ── Catchment Stats Tests ───────────────────────────────────────────


class TestComputeCatchmentStats:
    def test_returns_catchment_stats(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        ws = delineate_watershed(fdir, pour_point=(6, 3), dem=dem)
        accum = flow_accumulation(fdir)
        streams = extract_streams(accum, threshold=3)
        stats = compute_catchment_stats(ws, dem, streams)
        assert isinstance(stats, CatchmentStats)
        assert stats.area_km2 > 0
        assert stats.mean_elevation >= 0
        assert stats.mean_slope >= 0
        assert stats.elongation_ratio > 0

    def test_stream_density_positive(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        accum = flow_accumulation(fdir)
        streams = extract_streams(accum, threshold=3)
        ws = delineate_watershed(fdir, pour_point=(6, 3), dem=dem)
        stats = compute_catchment_stats(ws, dem, streams)
        assert stats.stream_density >= 0


class TestStationsToCatchments:
    def test_batch_delineation(self):
        elev = _v_valley_dem(11, 11)
        dem = _make_dem(elev, cellsize=30.0)
        fdir = flow_direction_d8(dem)
        accum = flow_accumulation(fdir)

        # Place station at geographic coords that map near centre-bottom
        # With from_bounds(0,0,330,330,11,11) cell (10,5) maps to ~(150+15, 0+15)=(165,15)
        # Inverse: lon=165 → col≈5, lat=15 → row≈10
        stations = [{"station_id": "S1", "latitude": 15.0, "longitude": 165.0}]
        results = stations_to_catchments(stations, dem, fdir, accum, snap_distance=3)
        assert len(results) == 1
        station, ws = results[0]
        assert station["station_id"] == "S1"
        assert ws.mask.sum() > 0

    def test_skips_missing_coords(self):
        elev = _v_valley_dem(7, 7)
        dem = _make_dem(elev)
        fdir = flow_direction_d8(dem)
        accum = flow_accumulation(fdir)
        stations = [{"station_id": "S1"}]  # no lat/lon
        results = stations_to_catchments(stations, dem, fdir, accum)
        assert len(results) == 0
