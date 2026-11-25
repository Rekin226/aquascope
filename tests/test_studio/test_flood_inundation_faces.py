"""The three Study faces export the same offline inundation raster."""

import base64
import io
import json
import socket
import struct
import sys
import zipfile

import numpy as np
import pytest

from aquascope import cli, floodsim, mcp_server
from tests.test_explorer.test_studio_worker import _face


@pytest.mark.parametrize("face", ["cli", "mcp", "explorer"])
def test_inundation_faces_export_depth_and_reference_check(face, tmp_path, monkeypatch):
    """Exercise real Study and export; replace only the simulation and external reference."""
    def offline(*args, **kwargs):
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket, "create_connection", offline)
    window = {"x0": 0, "y0": 0, "nx": 2, "ny": 2, "zoom": 10}
    depth = np.array([[0.0, 1.0], [2.0, 0.0]])
    calls = []

    def simulated(**kwargs):
        calls.append(kwargs)
        return {"grid": {"window": window, "z": np.zeros((2, 2)), "z_burnt": np.zeros((2, 2))},
                "run": {"h_max": depth, "t_s": 3600, "steady_at_s": 3600},
                "summary": {"river_id": 123456789, "wet_km2": 0.2, "max_depth_m": 2.0}}

    monkeypatch.setattr(floodsim, "flood_reach", simulated)
    monkeypatch.setattr(floodsim, "jrc_on_grid", lambda *args, **kwargs: np.array([[0.0, 1.0], [0.0, 1.0]]))
    problem = "Map the flood inundation at this reach."
    scenario = {"river_id": 123456789, "return_period": 100, "hours": 1, "coarse": True}
    intake = {"flood_inundation": scenario}
    if face == "cli":
        monkeypatch.setattr(sys, "argv", ["aquascope", "studio", problem, "--lat", "47", "--lon", "8.2",
                                          "--intake", "flood_inundation=" + json.dumps(scenario), "--yes", "--quiet",
                                          "--out", str(tmp_path)])
        cli.main()
        workspace = json.loads((tmp_path / "workspace.json").read_text(encoding="utf-8"))
        tif = (tmp_path / "rasters" / "s1_depth.tif").read_bytes()
    elif face == "mcp":
        opened = mcp_server.studio_start(problem, 47, 8.2, intake=intake)
        assert "error" not in opened, opened
        result = mcp_server.studio_approve(opened["workspace"])
        assert "error" not in result, result
        workspace = result["workspace"]
        exported = mcp_server.studio_export(workspace, str(tmp_path))
        assert "error" not in exported, exported
        tif = (tmp_path / "rasters" / "s1_depth.tif").read_bytes()
    else:
        call = _face()["studio_call"]
        opened = call({"op": "start", "lat": 47, "lon": 8.2, "text": problem, "intake": intake})
        assert "error" not in opened, opened
        result = call({"op": "approve", "workspace": opened["workspace"]})
        assert "error" not in result, result
        workspace = result["workspace"]
        exported = call({"op": "export", "workspace": workspace})
        assert exported["media_type"] == "application/zip"
        raw = base64.b64decode(exported["data"], validate=True)
        assert len(raw) == exported["size"]
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            tif = archive.read("rasters/s1_depth.tif")

    assert workspace["run"]["ok"]
    step, = workspace["run"]["results"]
    payload = step["result"]
    assert calls and calls[0]["river_id"] == scenario["river_id"]
    assert calls[0]["return_period"] == 100 and calls[0]["hours"] == 1 and calls[0]["coarse"] is True
    assert payload["depth"]["encoding"] == "zlib+base64"
    assert payload["depth"]["dtype"] == "float32"
    np.testing.assert_array_equal(floodsim.depth_from_payload(payload["depth"]), depth)
    check = payload["check"]
    assert check["ok"] and check["status"] == "compared"
    assert check["csi"] == pytest.approx(1 / 3, abs=0.001)

    # Read TIFF IFD and uncompressed pixel bytes without requiring rasterio.
    assert tif[:4] == b"II*\x00"
    ifd = struct.unpack_from("<I", tif, 4)[0]
    count = struct.unpack_from("<H", tif, ifd)[0]
    tags = {struct.unpack_from("<H", tif, ifd + 2 + i * 12)[0]:
            struct.unpack_from("<HHII", tif, ifd + 2 + i * 12)[1:] for i in range(count)}
    assert tags[256][2] == tags[257][2] == 2
    assert tags[258][2] == 32 and tags[259][2] == 1 and tags[277][2] == 1 and tags[339][2] == 3
    offset = tags[273][2]
    assert tags[279][2] == 16
    np.testing.assert_array_equal(np.frombuffer(tif, dtype="<f4", count=4, offset=offset).reshape(2, 2),
                                  [[-9999, 1], [2, -9999]])
    assert b"-9999.0\x00" in tif
    geokeys = tags[34735]
    assert struct.unpack_from("<16H", tif, geokeys[2])[-1] == 3857
