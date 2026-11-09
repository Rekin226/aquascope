"""A cross-section and a hydrograph through a flood model's results (#621), on the review's synthetic grid.

The grid is 10 x 8 cells of 1 km on a flat bed at 10 m; the west half (columns 0 to 4) floods over eight days,
deepest on day 4, and each column a little deeper than the one west of it (tests/test_mesh_review.py).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")

from aquascope import flood_section as F  # noqa: E402, N812
from aquascope import mesh_review as R  # noqa: E402, N812
from aquascope.io import mesh as M  # noqa: E402, N812
from tests.test_mesh_review import BED, NX, SHAPE, STEPS, cell, clean_depths, make_run  # noqa: E402


@pytest.fixture
def mesh_of(tmp_path):
    opened = []

    def open_(depth=None, **kw):
        m = M.read_mesh(make_run(tmp_path / f"run{len(opened)}.p01.hdf", depth, **kw))
        opened.append(m)
        return m

    yield open_
    for m in opened:
        m.close()


def centre(m, i):
    return [float(m.lon[i]), float(m.lat[i])]


def depth_at(row, col, step):
    return SHAPE[step] * (1 + 0.1 * col + 0.01 * row) if col < 5 else 0.0


def test_the_index_finds_the_same_cells_as_the_review(mesh_of):
    m = mesh_of()
    w, s, e, n = m.bbox()
    rng = np.random.default_rng(3)
    lon = rng.uniform(w - 0.01, e + 0.01, 400)
    lat = rng.uniform(s - 0.01, n + 0.01, 400)
    got = F.cells_along(m, lon, lat)
    assert (got == R.cells_at(m, lon, lat)).all()
    assert (got >= 0).any() and (got < 0).any()
    assert F.cell_at(m, *centre(m, cell(2, 3))) == cell(2, 3)


def test_a_section_reads_the_ground_and_the_water_along_the_line(mesh_of):
    m = mesh_of()
    line = [centre(m, cell(3, 0)), centre(m, cell(3, NX - 1))]
    s = F.section(m, line)
    d = np.array(s["distance_m"])
    assert d[0] == 0 and d[-1] == pytest.approx(9000, rel=0.01)   # nine cell widths, centre to centre
    assert s["facts"]["on_mesh_pct"] == 100.0
    assert set(s["cell"]) == {cell(3, c) for c in range(NX)}
    assert s["ground"] == [BED] * len(d)
    assert "Cells Minimum Elevation" in s["ground_is"]
    assert len(s["levels"]) == STEPS
    # at the peak (day 4): the water stands on the west half, deeper eastward, and the east half is dry
    peak = s["levels"][3]
    for k, c in enumerate(s["cell"]):
        col = c % NX
        if col < 5:
            assert peak[k] == pytest.approx(BED + depth_at(3, col, 3), abs=1e-3)
        else:
            assert peak[k] is None
    # the highest water of the run is the peak's, cell by cell
    for k, c in enumerate(s["cell"]):
        col = c % NX
        want = BED + max(depth_at(3, col, t) for t in range(STEPS)) if col < 5 else None
        assert s["max_level"][k] == (pytest.approx(want, abs=1e-3) if want is not None else None)
    f = s["facts"]
    assert f["deepest"] == pytest.approx(depth_at(3, 4, 3), abs=1e-3) and f["deepest_step"] == 3
    assert f["widest_step"] in range(1, STEPS)
    assert f["widest_wet_m"] == pytest.approx(4500, rel=0.05)   # from column 0's centre to the column 4/5 edge
    json_safe = all(v is None or math.isfinite(v) for row in s["levels"] for v in row)
    assert json_safe


def test_a_section_partly_off_the_mesh_says_so(mesh_of):
    m = mesh_of()
    w, s, e, n = m.bbox()
    line = [[w - (e - w) * 0.5, (s + n) / 2], centre(m, cell(4, 2))]
    out = F.section(m, line)
    assert -1 in out["cell"]
    assert out["facts"]["on_mesh_pct"] < 100
    assert any("off the mesh" in note for note in out["notes"])
    assert all(v is None for v, c in zip(out["levels"][3], out["cell"]) if c < 0)


def test_a_section_needs_a_line_across_the_mesh(mesh_of):
    m = mesh_of()
    with pytest.raises(M.MeshError, match="two or more"):
        F.section(m, [centre(m, 0)])
    with pytest.raises(M.MeshError, match="same point"):
        F.section(m, [centre(m, 0), centre(m, 0)])
    w, s, e, n = m.bbox()
    with pytest.raises(M.MeshError, match="does not cross"):
        F.section(m, [[e + 1, n + 1], [e + 2, n + 2]])


def test_a_section_without_bed_levels_gives_the_water_surface_only(mesh_of):
    m = mesh_of(bed=False)
    s = F.section(m, [centre(m, cell(3, 0)), centre(m, cell(3, NX - 1))])
    assert s["ground"] is None and s["ground_is"] == ""
    assert any("water surface only" in note for note in s["notes"])
    assert s["levels"][3][0] == pytest.approx(BED + depth_at(3, 0, 3), abs=1e-3)


def test_the_hydrograph_marks_arrival_peak_and_time_under_water(mesh_of):
    m = mesh_of()
    h = F.hydrograph(m, cell(3, 1), threshold=1.0)
    depth = [depth_at(3, 1, t) for t in range(STEPS)]
    for got, want in zip(h["depth"], depth):
        assert (got is None) if want <= M.DRY_DEPTH else got == pytest.approx(want, abs=1e-3)
    assert h["ground"] == BED
    assert h["level"][3] == pytest.approx(BED + depth[3], abs=1e-3)
    wet = [t for t in range(STEPS) if depth[t] > 1.0]
    assert wet == [2, 3, 4]
    assert h["arrival"]["step"] == 2 and h["arrival"]["hours"] == 48
    assert h["peak"]["step"] == 3 and h["peak"]["depth"] == pytest.approx(depth[3], abs=1e-3)
    assert h["recession"]["step"] == 5
    assert h["duration_h"] == 72 and h["still_wet"] is False
    assert h["source"] == F.HEC_MAPS


def test_the_hydrograph_with_the_dry_depth_runs_to_the_end_of_the_run(mesh_of):
    m = mesh_of()
    h = F.hydrograph(m, cell(3, 1))
    assert h["threshold"] == M.DRY_DEPTH
    assert h["arrival"]["step"] == 1
    assert h["still_wet"] is True and h["recession"] is None
    assert h["duration_h"] == 24 * (STEPS - 2)
    assert "at least" in h["note"]


def test_the_hydrograph_by_place_and_of_a_dry_cell(mesh_of):
    m = mesh_of()
    lon, lat = centre(m, cell(5, 2))
    assert F.hydrograph(m, lon=lon, lat=lat)["cell"] == cell(5, 2)
    dry = F.hydrograph(m, cell(3, 8))
    assert dry["wet"] is False and dry["arrival"] is None and "never got deeper" in dry["note"]
    with pytest.raises(M.MeshError, match="outside"):
        F.hydrograph(m, lon=lon + 5, lat=lat)
    with pytest.raises(M.MeshError, match="Say which cell"):
        F.hydrograph(m)


def test_a_second_rise_keeps_one_duration_like_hec_ras(mesh_of):
    depth = clean_depths()
    c = cell(2, 2)
    depth[:, c] = [0.0, 0.8, 0.0, 0.0, 1.5, 0.6, 0.0, 0.0]
    m = mesh_of(depth)
    h = F.hydrograph(m, c, threshold=0.5)
    assert h["arrival"]["step"] == 1 and h["recession"]["step"] == 6
    assert h["duration_h"] == 120 and h["dry_spells"] == 2
    assert h["peak"]["step"] == 4
    assert "rose again" in h["note"]

