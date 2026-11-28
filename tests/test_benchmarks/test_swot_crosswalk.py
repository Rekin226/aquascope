"""Offline regression cases for the one-VPU reach crosswalk."""

import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely", minversion="2")

from benchmarks import swot_crosswalk as sw  # noqa: E402


def line(y, length=5000):
    return shapely.LineString([(0, y), (length, y)])


def frames(geometries, river_areas, targets):
    streams = gpd.GeoDataFrame(
        {"LINKNO": list(range(1, len(geometries) + 1)), "DSContArea": river_areas, "strmOrder": 5},
        geometry=geometries, crs=sw.NZTM)
    sword = gpd.GeoDataFrame(targets, columns=["reach_id", "facc", "type", "geometry"], geometry="geometry",
                             crs=sw.NZTM).assign(river_name="River")
    return streams, sword


def test_index_units_multipart_sampling_and_close_alternatives():
    multipart = shapely.MultiLineString([[(0, 0), (100, 0)], [(100, 2000), (5000, 2000)]])
    streams, sword = frames([multipart, line(5000)], [100e6, 90e6], [
        (10, 100, 1, line(0)), (11, 100, 1, line(2000)),
        (12, 90, 1, line(5010)), (13, 90, 1, line(5080))])
    links, candidates = sw.crosswalk(streams, sword.sample(frac=1, random_state=4))

    assert links.status.tolist() == ["matched", "ambiguous"]
    assert links.sword_id.iloc[0] == 11 and pd.isna(links.sword_id.iloc[1])
    assert links.geoglows_km2.tolist() == [100, 90]
    assert candidates.query("river_id == 2").sword_id.tolist() == [12, 13]
    assert candidates.query("sword_id == 11").area_ratio.item() == 1


def test_area_coverage_and_special_feature_rejections():
    streams, sword = frames([line(y) for y in (0, 2000, 4000, 6000, 8000)], [100e6] * 5, [
        (10, 10, 1, line(0)),                    # Same course, tenfold difference in area
        (11, -9999, 1, line(2000)),                # SWORD fill value
        (12, 100, 1, line(4000, length=100)),      # Endpoint contact with a short segment
        (13, 100, 6, line(6000)),                  # Ghost reach
        (14, 50, 1, line(8500)),                   # Exactly on both inclusive thresholds
    ])
    links, _ = sw.crosswalk(streams, sword)
    assert links.status.tolist() == ["area_failed", "area_missing", "geometry_failed", "no_nearby_river", "matched"]
    assert links.sword_id.isna().tolist() == [True, True, True, True, False]


def test_distinct_geoglows_segments_can_share_a_sword_reach():
    streams, sword = frames([line(0), line(200)], [95e6, 100e6], [(10, 100, 1, line(100))])
    links, _ = sw.crosswalk(streams, sword)
    assert links.sword_id.tolist() == [10, 10]
