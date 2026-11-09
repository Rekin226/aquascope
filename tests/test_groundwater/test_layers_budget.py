"""MODFLOW layers, cross-sections, hydrographs (#625) and the water budget from a .cbc (#626)."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from aquascope.groundwater import budget as gb
from aquascope.groundwater import layers as gv
from aquascope.groundwater.surface import read_modflow_heads

sys.path.insert(0, str(Path(__file__).parent))
import synthetic_models as sm  # noqa: E402


@pytest.fixture(scope="module")
def layered():
    files = sm.layered_files()
    res = read_modflow_heads(files["synthetic.hds"], files["synthetic.dis"], sidecars=[files["synthetic.nam"]])
    return files, res, gv.model_layers(res, files["synthetic.dis"])


# ── the budget reader ───────────────────────────────────────────────────────


@pytest.mark.parametrize("double", [False, True])
def test_compact_budget_gives_each_term_in_and_out(layered, double):
    files, res, _ = layered
    model = sm.layered_model()
    cbc, truth = sm.layered_budget(model, double=double)
    wb = gb.water_budget(cbc, heads=res)
    assert wb["precision"] == ("double" if double else "single") and wb["compact"]
    assert wb["unit"] == "m³/day"
    assert wb["internal"] == ["FLOW RIGHT FACE", "FLOW FRONT FACE", "FLOW LOWER FACE"]
    assert wb["dates"][0] == res["dates"][0] and len(wb["times"]) == sm.MONTHS
    by = {t["text"]: t for t in wb["terms"]}
    assert set(by) == {"STORAGE", "CONSTANT HEAD", "WELLS", "RECHARGE", "RIVER LEAKAGE"}
    assert by["RIVER LEAKAGE"]["label"] == "Rivers" and by["CONSTANT HEAD"]["label"] == "Constant heads"
    rel = 1e-5 if not double else 1e-9
    for m, want in enumerate(truth):
        for text, (vin, vout) in want.items():
            assert by[text]["in"][m] == pytest.approx(vin, rel=rel, abs=1e-3)
            assert by[text]["out"][m] == pytest.approx(vout, rel=rel, abs=1e-3)
    # the synthetic budget balances, so the discrepancy MODFLOW would print is nil
    assert max(abs(d) for d in wb["discrepancy_pct"]) < 1e-3
    # the well pumps 2,500 m3/day for 14 months, then eases off to nothing over 8
    assert by["WELLS"]["out"][0] == pytest.approx(2500.0) and by["WELLS"]["out"][-1] == pytest.approx(0.0, abs=1e-6)
    at = gb.budget_at(wb, 13)
    assert at["time_index"] == 13 and {r["label"] for r in at["terms"]} >= {"Wells", "Storage"}
    assert at["total_in"] == pytest.approx(at["total_out"], rel=1e-5)


def test_reader_matches_flopy(tmp_path, layered):
    flopy = pytest.importorskip("flopy")
    model = sm.layered_model()
    cbc, _ = sm.layered_budget(model)
    p = tmp_path / "m.cbc"
    p.write_bytes(cbc)
    cb = flopy.utils.CellBudgetFile(str(p))
    ours = gb.read_budget(cbc)["records"]
    assert len(ours) == len(cb.recordarray)
    for k in (0, 7, 13, 23):
        for text in ("STORAGE", "WELLS", "RECHARGE", "RIVER LEAKAGE", "CONSTANT HEAD"):
            theirs = cb.get_data(kstpkper=(0, k), text=text)[0]
            mine = [r for r in ours if r["text"] == text and r["kper"] == k + 1][0]
            if isinstance(theirs, np.recarray) or getattr(theirs, "dtype", None) is not None and theirs.dtype.names:
                assert np.allclose(np.sort(theirs["q"]), np.sort(mine["values"]))
                assert np.array_equal(np.sort(theirs["node"]), np.sort(mine["nodes"]))
            else:
                vals = np.asarray(theirs, float)
                if text == "RECHARGE":   # IMETH 3: FloPy returns (layer array, values)
                    vals = np.asarray(theirs[1], float) if isinstance(theirs, list) else vals
                assert np.isclose(vals.sum(), np.asarray(mine["values"], float).sum(), rtol=1e-5)
    assert [float(t) for t in cb.get_times()][:3] == pytest.approx([r["totim"] for r in ours[::8]][:3])


def test_full_layout_has_no_times_until_the_heads_give_them(layered):
    _, res, _ = layered
    n = sm.NROW * sm.NCOL * sm.NLAY
    sto = np.zeros(n)
    sto[:10] = 5.0
    sto[10:14] = -2.0
    ch = np.zeros(n)
    ch[0] = 30.0
    data = b""
    for m in (1, 2):
        data += sm.cbc_record(1, m, "STORAGE", sm.NCOL, sm.NROW, sm.NLAY, values=sto * m, double=True)
        data += sm.cbc_record(1, m, "CONSTANT HEAD", sm.NCOL, sm.NROW, sm.NLAY, values=-ch, double=True)
    plain = gb.water_budget(data)
    assert not plain["compact"] and plain["times"] == [None, None] and plain["dates"] is None
    assert any("full layout" in n for n in plain["notes"])
    assert plain["unit"] == "L³/T"
    by = {t["text"]: t for t in plain["terms"]}
    assert by["STORAGE"]["in"] == [50.0, 100.0] and by["STORAGE"]["out"] == [8.0, 16.0]
    assert by["CONSTANT HEAD"]["out"] == [30.0, 30.0]
    dated = gb.water_budget(data, heads=res)
    assert dated["times"] == [res["times"][0], res["times"][1]] and dated["dates"][1] == res["dates"][1]
    assert dated["unit"] == "m³/day"


def test_modflow6_layout_adds_up_packages_and_leaves_out_flows_between_cells():
    ncell = 6
    kw = {"double": True, "delt": 1.0, "pertim": 1.0, "totim": 1.0}
    data = sm.cbc_record(1, 1, "FLOW-JA-FACE", 20, 1, 1, imeth=1, values=np.linspace(-1, 1, 20), **kw)
    data += sm.cbc_record(1, 1, "DATA-SPDIS", ncell, 1, 1, imeth=6, nodes=list(range(1, 7)), values=[9.0] * 6,
                          aux=("QX", "QY", "QZ"), names=("GWF", "GWF", "GWF", "NPF"), **kw)
    data += sm.cbc_record(1, 1, "STO-SS", ncell, 1, 1, imeth=1, values=[0.5, -0.25, 0, 0, 0, 0], **kw)
    data += sm.cbc_record(1, 1, "WEL", ncell, 1, 1, imeth=6, nodes=[2], values=[-100.0], **kw)
    data += sm.cbc_record(1, 1, "WEL", ncell, 1, 1, imeth=6, nodes=[5], values=[-50.0],
                          names=("GWF", "WEL-2", "GWF", "WEL-2"), **kw)
    data += sm.cbc_record(1, 1, "RCH", ncell, 1, 1, imeth=6, nodes=list(range(1, 7)), values=[25.0] * 6,
                          aux=("MULT",), names=("GWF", "RCHA", "GWF", "RCHA"), **kw)
    data += sm.cbc_record(1, 1, "CHD", ncell, 1, 1, imeth=6, nodes=[1], values=[-0.25], **kw)
    wb = gb.water_budget(data, time_units="days", length_units="meters")
    by = {t["text"]: t for t in wb["terms"]}
    assert set(by) == {"STO-SS", "WEL", "RCH", "CHD"}
    assert by["WEL"]["out"] == [150.0] and by["WEL"]["label"] == "Wells"
    assert by["RCH"]["in"] == [150.0] and by["STO-SS"]["in"] == [0.5] and by["STO-SS"]["out"] == [0.25]
    assert wb["internal"] == ["FLOW-JA-FACE", "DATA-SPDIS"]
    assert wb["total_in"] == [150.5] and wb["total_out"] == [150.5] and wb["discrepancy_pct"] == [0.0]
    assert wb["unit"] == "m³/day"


def test_imeth_4_and_5_and_a_start_date():
    kw = {"delt": 10.0, "pertim": 10.0}
    data = b""
    for k, totim in enumerate((10.0, 20.0)):
        data += sm.cbc_record(1, k + 1, "RECHARGE", 3, 2, 2, imeth=4, values=[1.0] * 6, totim=totim, **kw)
        data += sm.cbc_record(1, k + 1, "DRAINS", 3, 2, 2, imeth=5, nodes=[3, 4], values=[-2.0, -1.0],
                              aux=("IFACE",), totim=totim, **kw)
    wb = gb.water_budget(data, start_date="2024-01-01")
    assert wb["times"] == [10.0, 20.0] and wb["dates"] == ["2024-01-11", "2024-01-21"]
    by = {t["label"]: t for t in wb["terms"]}
    assert by["Recharge"]["in"] == [6.0, 6.0] and by["Drains"]["out"] == [3.0, 3.0]
    assert wb["discrepancy_pct"][0] == pytest.approx(100 * (6 - 3) / 4.5)


def test_not_a_budget_file_says_so():
    with pytest.raises(ValueError, match="empty"):
        gb.read_budget(b"")
    with pytest.raises(ValueError, match="not a MODFLOW cell-by-cell budget file"):
        gb.read_budget(b"\x00" * 200)
    cbc, _ = sm.layered_budget(sm.layered_model())
    with pytest.raises(ValueError):
        gb.read_budget(cbc[:-10])
    head = sm.hds_bytes([(1, 1, 1.0, 1, np.zeros((2, 2)))])
    with pytest.raises(ValueError):
        gb.read_budget(head)   # a head file is not a budget file
    assert gb.term_label("RIVER LEAKAGE") == "Rivers" and gb.term_label("SOMETHING ELSE") == "Something else"


def test_budget_at_checks_its_index(layered):
    _, res, _ = layered
    wb = gb.water_budget(sm.layered_budget(sm.layered_model())[0], heads=res)
    assert gb.budget_at(wb, -1)["time_index"] == sm.MONTHS - 1
    with pytest.raises(ValueError):
        gb.budget_at(wb, sm.MONTHS)
    json.dumps(wb)


# ── the layers ──────────────────────────────────────────────────────────────


def test_layer_elevations_from_a_mf2005_dis_match_the_model(layered):
    files, res, lay = layered
    model = sm.layered_model()
    el = gv.read_layer_elevations(files["synthetic.dis"])
    assert el["nlay"] == 3 and el["ncpl"] == sm.NROW * sm.NCOL
    assert np.allclose(el["top"], model["top"].ravel(), atol=1e-3)
    for k in range(3):
        assert np.allclose(el["botm"][k], model["botm"][k].ravel(), atol=1e-3)
    assert lay["layers"] == [1, 2, 3] and lay["base"] == -40.0 and lay["length_units"] == "meters"
    assert lay["exaggeration"] in (2, 5, 10) and "Vertical exaggeration" in lay["note"]
    s = gv.layers_summary(lay)
    assert [r["layer"] for r in s["per_layer"]] == [1, 2, 3] and "top" not in s
    assert s["per_layer"][1]["mean_thickness"] == pytest.approx(9.0 - 0.0008 * 2000.0, abs=0.01)


def test_mf2005_confining_beds_are_skipped():
    text = ("    2    1    2    1    4    2\n 1 0\nCONSTANT 10.0\nCONSTANT 10.0\nCONSTANT 50.0\n"
            "CONSTANT 40.0\nCONSTANT 35.0\nCONSTANT 0.0\n")
    el = gv.read_layer_elevations(text.encode())
    assert el["top"].tolist() == [50.0, 50.0]
    assert el["botm"].tolist() == [[40.0, 40.0], [0.0, 0.0]]   # 35 is the confining bed's bottom, not a layer
    # ...but it is where layer 2 starts: the bed between 40 and 35 is a gap, not part of layer 2
    assert el["tops"].tolist() == [[50.0, 50.0], [35.0, 35.0]]
    assert gv.layer_top(el, 1).tolist() == [35.0, 35.0]


def test_mf6_dis_with_layered_botm_and_idomain():
    text = """BEGIN DIMENSIONS
  NLAY 2
  NROW 1
  NCOL 3
END DIMENSIONS
BEGIN GRIDDATA
  DELR
    CONSTANT 10.0
  DELC
    CONSTANT 10.0
  TOP
    INTERNAL FACTOR 1.0
    12.0 11.0 10.0
  BOTM LAYERED
    CONSTANT 5.0
    INTERNAL
    -1.0 -2.0 -3.0
  IDOMAIN
    INTERNAL
    1 1 0 1 1 1
END GRIDDATA
"""
    el = gv.read_layer_elevations(text.encode())
    assert el["top"].tolist() == [12.0, 11.0, 10.0]
    assert el["botm"].tolist() == [[5.0, 5.0, 5.0], [-1.0, -2.0, -3.0]]
    assert el["idomain"].tolist() == [[1, 1, 0], [1, 1, 1]]


def test_binary_grid_file_layers():
    nlay, nrow, ncol = 2, 1, 2
    defs = [("NCELLS", "INTEGER", []), ("NLAY", "INTEGER", []), ("NROW", "INTEGER", []), ("NCOL", "INTEGER", []),
            ("DELR", "DOUBLE", [ncol]), ("DELC", "DOUBLE", [nrow]), ("TOP", "DOUBLE", [nrow * ncol]),
            ("BOTM", "DOUBLE", [nlay * nrow * ncol]), ("IDOMAIN", "INTEGER", [nlay * nrow * ncol])]
    head = b"".join(f"{t:<49}\n".encode() for t in ("GRID DIS", "VERSION 1", f"NTXT {len(defs)}", "LENTXT 100"))
    for name, typ, dims in defs:
        head += f"{name} {typ} NDIM {len(dims)} {' '.join(map(str, dims))}".ljust(99).encode() + b"\n"
    data = struct.pack("<iiii", nlay * nrow * ncol, nlay, nrow, ncol)
    data += np.array([10.0, 10.0]).tobytes() + np.array([10.0]).tobytes() + np.array([20.0, 21.0]).tobytes()
    data += np.array([5.0, 6.0, -5.0, -4.0]).tobytes() + np.array([1, 1, 1, -1], "<i4").tobytes()
    el = gv.read_layer_elevations(head + data)
    assert el["top"].tolist() == [20.0, 21.0] and el["botm"].tolist() == [[5.0, 6.0], [-5.0, -4.0]]
    assert el["idomain"].tolist() == [[1, 1], [1, -1]]


def test_nice_exaggeration():
    assert [gv.nice_exaggeration(v) for v in (0.3, 1, 1.9, 4.9, 7, 19, 230)] == [1, 1, 1, 2, 5, 10, 200]


def test_a_grid_from_another_model_is_refused(layered):
    _, res, _ = layered
    other = "    1    2    2    1    4    2\n 0\nCONSTANT 1.0\nCONSTANT 1.0\nCONSTANT 5.0\nCONSTANT 0.0\n"
    with pytest.raises(ValueError, match="the same model"):
        gv.model_layers(res, other.encode())


# ── the cross-section ───────────────────────────────────────────────────────


def _row_line(res, row, pad=0.002):
    """A west to east line through the middle of a row (1-based), a little past the model's edges."""
    rings = np.asarray(res["rings"]).reshape(-1, 4, 2)
    ncol = res["grid"]["ncol"]
    first, last = rings[(row - 1) * ncol], rings[(row - 1) * ncol + ncol - 1]
    west = first[[0, 3]].mean(axis=0)
    east = last[[1, 2]].mean(axis=0)
    return [[west[0] - pad, west[1]], [east[0] + pad, east[1]]]


def test_section_through_the_well_row(layered):
    _, res, lay = layered
    sec = gv.cross_section(res, lay, _row_line(res, 15))
    segs = sec["segments"]
    assert [s["row"] for s in segs] == [15] * sm.NCOL
    assert [s["col"] for s in segs] == list(range(1, sm.NCOL + 1))
    widths = [s["to_m"] - s["from_m"] for s in segs]
    assert np.allclose(widths, 100.0, atol=2.0)          # 100 m cells, the edges found to half a metre
    assert sec["length_m"] > 4000 and "outside the model" in sec["notes"][0]
    k = 22   # the well's column
    cell = segs[k]["cell"]
    heads = np.asarray(res["heads"])
    assert sec["heads"][13, 2, k] == pytest.approx(heads[13, 2, cell])
    assert sec["top"][0, k] == pytest.approx(lay["top"][cell]) and sec["bottom"][2, k] == pytest.approx(-40.0)
    assert sec["water_table"][13, k] == pytest.approx(heads[13, 0, cell])
    # the cone: the chalk's head recovers most at the well once the pumping stops (month 14 to month 24)
    assert int(np.nanargmax(sec["heads"][23, 2] - sec["heads"][13, 2])) == k
    out = gv.section_summary(sec, 13)
    json.dumps(out)
    assert out["time_index"] == 13 and len(out["heads_at_time"]) == 3


def test_a_head_above_the_top_is_said_to_be_no_water_table(layered):
    _, res, lay = layered
    line = _row_line(res, 15)
    assert not any("potentiometric" in n for n in gv.cross_section(res, lay, line)["notes"])
    confined = {**res, "heads": np.asarray(res["heads"], float) + 100.0}   # every head far above the ground
    assert any("potentiometric" in n for n in gv.cross_section(confined, lay, line)["notes"])


def test_section_with_a_bend_and_through_inactive_cells(layered):
    _, res, lay = layered
    a, b = _row_line(res, 2)
    c = _row_line(res, 28)[1]
    sec = gv.cross_section(res, lay, [a, b, c])
    assert len(sec["vertices_m"]) == 3 and sec["vertices_m"][1] < sec["length_m"]
    first = sec["segments"][0]
    assert first["row"] == 2 and first["col"] <= 3                # rows 1 to 5 lose their first columns
    assert np.isnan(sec["water_table"][0, 0]) and np.isnan(sec["heads"][0, 0, 0])   # inactive: no head


def test_a_line_off_the_model_and_bad_lines(layered):
    _, res, lay = layered
    far = gv.cross_section(res, lay, [[0.0, 0.0], [0.01, 0.0]])
    assert far["error"] == "The line does not cross the model."
    for bad in ([[2.3, 48.8]], [[2.3, 48.8], [2.3, 48.8]], [[2.3, "x"], [2.4, 48.8]]):
        with pytest.raises(ValueError):
            gv.cross_section(res, lay, bad)


# ── a cell's heads over time ────────────────────────────────────────────────


def test_hydrograph_at_the_well(layered):
    _, res, lay = layered
    c = (sm.WELL[0]) * sm.NCOL + sm.WELL[1]
    h = gv.cell_hydrograph(res, c, lay)
    assert (h["row"], h["col"]) == (15, 23) and [s["layer"] for s in h["series"]] == [1, 2, 3]
    s3 = h["series"][2]
    assert len(s3["head"]) == sm.MONTHS and s3["top"] == pytest.approx(lay["botm"][1][c], abs=1e-3)
    assert s3["bottom"] == -40.0
    lowest = int(np.argmin(s3["head"]))
    assert lowest == 11 and s3["min_at"] == res["dates"][lowest]   # the dry season's low, with the well on
    assert s3["change"] == pytest.approx(s3["last"] - s3["first"])
    assert "layer 3" in h["summary"]
    json.dumps(h)


def test_hydrograph_of_an_inactive_cell_and_a_bad_index(layered):
    _, res, _ = layered
    h = gv.cell_hydrograph(res, 0)
    assert all(s["min"] is None for s in h["series"]) and "No head" in h["summary"]
    assert all(v is None for v in h["series"][0]["head"])
    with pytest.raises(ValueError):
        gv.cell_hydrograph(res, sm.NROW * sm.NCOL)


# ── the faces ───────────────────────────────────────────────────────────────


def _write(tmp_path):
    files = sm.layered_files()
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return {k.split(".")[1]: str(tmp_path / k) for k in files}


def test_cli_modflow_views(tmp_path, monkeypatch, capsys):
    from aquascope import cli

    p = _write(tmp_path)
    _, res, _ = None, read_modflow_heads(p["hds"], p["dis"], sidecars=[p["nam"]]), None
    a, b = _row_line(res, 15)
    argv = ["aquascope", "modflow", p["hds"], "--grid", p["dis"], "--sidecar", p["nam"], "--time", "13", "--layers",
            "--section", f"{a[0]},{a[1]}", f"{b[0]},{b[1]}", "--cell", "15", "23", "--budget", p["cbc"]]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    text = capsys.readouterr().out
    assert "layers: 3" in text and "layer 2: mean top" in text
    assert "section: A 4." in text and "40 cells and 3 layer(s)" in text
    assert "cell row 15, column 23: Head over 24 saved time(s)" in text
    assert "water budget at 2025-03-02T03:00 (m³/day)" in text and "Wells" in text and "discrepancy" in text
    monkeypatch.setattr(sys, "argv", argv + ["--json"])
    cli.main()
    out = json.loads(capsys.readouterr().out)
    assert out["layers"] == [1, 2, 3]                      # the head file's layers, unchanged
    assert out["layer_geometry"]["exaggeration"] >= 1 and out["hydrograph"]["col"] == 23
    assert out["budget_at"]["time_index"] == 13 and len(out["section"]["segments"]) == 40


def test_cli_section_needs_the_grid(tmp_path, monkeypatch):
    from aquascope import cli

    p = _write(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aquascope", "modflow", p["hds"], "--cell-size", "100", "--origin", "448000",
                                      "5411000", "--crs", "EPSG:32631", "--section", "2.30,48.86", "2.33,48.86"])
    with pytest.raises(SystemExit):
        cli.main()


def test_mcp_modflow_heads_views(tmp_path):
    from aquascope import mcp_server as m

    p = _write(tmp_path)
    res = read_modflow_heads(p["hds"], p["dis"], sidecars=[p["nam"]])
    out = m.modflow_heads(p["hds"], p["dis"], [p["nam"]], time_index=13, budget_path=p["cbc"],
                          section=_row_line(res, 15), cell_row=15, cell_col=23)
    assert "views_error" not in out
    assert out["layer_geometry"]["layers"] == [1, 2, 3] and out["layers"] == [1, 2, 3]
    assert len(out["section"]["segments"]) == 40 and out["hydrograph"]["row"] == 15
    assert out["budget_at"]["unit"] == "m³/day"
    bad = m.modflow_heads(p["hds"], p["dis"], [p["nam"]], cell_row=99, cell_col=1)
    assert "cell_row" in bad["views_error"]
    json.dumps(out)


def test_the_views_are_experimental():
    from aquascope.experimental import is_experimental

    assert is_experimental("aquascope.groundwater.layers") and is_experimental("aquascope.groundwater.budget")
    assert not math.isnan(gv.STACK_SHARE)
