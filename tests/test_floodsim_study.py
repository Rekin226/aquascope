"""Offline study contract and reference comparison tests."""

import base64
import zlib

import numpy as np
import pytest

from aquascope import floodsim as fs


@pytest.fixture
def simulated(monkeypatch):
    window = {"x0": 0, "y0": 0, "nx": 2, "ny": 2, "zoom": 10}
    grid = {"window": window}
    depth = np.array([[0.0, 1.0], [2.0, 0.0]])
    monkeypatch.setattr(fs, "flood_reach", lambda **kwargs: {
        "grid": grid, "run": {"h_max": depth, "t_s": 3600, "steady_at_s": 3600},
        "summary": {"river_id": 123456789, "wet_km2": 0.2, "max_depth_m": 2.0},
    })
    monkeypatch.setattr(fs, "depth_over_terrain", lambda g, h: h)
    return depth


def test_comparison_and_payload(simulated):
    out = fs.study_flood_inundation(river_id=123456789, reference_fetch=lambda g, rp: [[0, 1], [0, 1]])
    assert out["check"]["ok"] is True
    assert out["check"]["status"] == "compared"
    assert out["check"]["csi"] == pytest.approx(1 / 3, abs=0.001)
    np.testing.assert_allclose(fs.depth_from_payload(out["depth"]), simulated)
    tif = fs.to_geotiff_bytes({"window": out["depth"]["window"]}, simulated)
    assert tif[:4] in (b"II*\x00", b"MM\x00*")


@pytest.mark.parametrize("reference,status", [
    (lambda g, rp: np.full((2, 2), np.nan), "no_coverage"),
    (lambda g, rp: (_ for _ in ()).throw(OSError("offline")), "unavailable"),
])
def test_unavailable_comparison_preserves_depth(simulated, reference, status):
    out = fs.study_flood_inundation(river_id=123456789, reference_fetch=reference)
    assert out["check"]["ok"] and out["check"]["status"] == status
    assert out["check"]["csi"] is None
    np.testing.assert_array_equal(fs.depth_from_payload(out["depth"]), simulated)


def test_a_specified_flow_has_no_reference_and_a_short_return_period_uses_the_published_map(simulated):
    out = fs.study_flood_inundation(river_id=123456789, flow_cms=10,
                                    reference_fetch=lambda g, rp: [[0, 1], [0, 1]])
    assert out["check"]["status"] == "no_reference"
    assert out["check"]["csi"] is None and out["check"]["return_period"] is None
    out = fs.study_flood_inundation(river_id=123456789, return_period=25,
                                    reference_fetch=lambda g, rp: [[0, 1], [0, 1]])
    assert out["check"]["status"] == "compared"
    assert out["check"]["return_period"] == 20, "the nearest published map at or below 25 years"
    assert out["check"]["csi"] == pytest.approx(1 / 3, abs=0.001)


@pytest.mark.parametrize("kwargs", [
    {"hours": 0}, {"flow_cms": 0}, {"coarse": "false"}, {"lat": 1},
    {"return_period": 100, "flow_cms": 10}, {"flow_cms": float("nan")},
])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        fs.validate_flood_scenario({"river_id": 123456789, **kwargs})


def test_bounded_payload_decode():
    pack = {"window": {"nx": 1, "ny": 1}, "dtype": "float32", "encoding": "zlib+base64",
            "data": base64.b64encode(zlib.compress(b"x" * 100000)).decode()}
    with pytest.raises(ValueError):
        fs.depth_from_payload(pack)
    pack["window"]["nx"] = 4000001
    with pytest.raises(ValueError):
        fs.depth_from_payload(pack)


def test_incomplete_run_is_not_ok(monkeypatch, simulated):
    monkeypatch.setattr(fs, "flood_reach", lambda **kwargs: {
        "grid": {"window": {"x0": 0, "y0": 0, "nx": 2, "ny": 2, "zoom": 10}},
        "run": {"h_max": simulated, "t_s": 1, "steady_at_s": None},
        "summary": {"river_id": 123456789, "wet_km2": 0.2, "max_depth_m": 2.0},
    })
    out = fs.study_flood_inundation(river_id=123456789, reference_fetch=lambda g, rp: np.zeros((2, 2)))
    assert out["check"]["ok"] is False
    assert "did not complete" in out["check"]["summary"]
    assert "not necessarily final" in out["check"]["summary"]


def test_jrc_on_grid_reads_a_run_grid_through_its_window(monkeypatch):
    from aquascope.utils import cog as cogmod

    def missing(url):
        raise cogmod.COGNotFound(url)

    monkeypatch.setattr(cogmod, "open_cog", missing)
    win = {"x0": 0, "y0": 0, "zoom": 2, "nx": 4, "ny": 3, "size": 4, "bounds": [-180.0, -85.0, 180.0, 85.0]}
    grid = {"window": win, "nx": 4, "ny": 3, "dx": 1000.0}
    out = fs.jrc_on_grid(grid, 100)
    assert out.shape == (3, 4)
    assert np.isnan(out).all()
