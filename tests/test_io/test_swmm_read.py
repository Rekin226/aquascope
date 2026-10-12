"""Reading SWMM models and results (#560), on small synthetic files written here.

The .out writer below follows the layout SWMM writes in src/solver/output.c; the reader was checked against
swmm-toolkit's own output reader on EPA's examples (docs/assessments/swmm.md), and these tests pin that layout.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from aquascope.io.swmm import (
    MAGIC,
    SwmmError,
    read_swmm_inp,
    read_swmm_out,
    swmm_summary,
)

INP = """[TITLE]
;;Project Title/Notes
Tiny network

[OPTIONS]
FLOW_UNITS           CMS
START_DATE           06/01/2024

[JUNCTIONS]
;;Name  Elev  MaxDepth  InitDepth  SurDepth  Aponded
J1      10.0  2.0       0          0         0
J2      9.0   2.5       0          0         0

[OUTFALLS]
;;Name  Elev  Type
O1      8.0   FREE

[CONDUITS]
;;Name  From  To  Length  Roughness  InOffset  OutOffset
C1      J1    J2  120.5   0.013      0         0
C2      J2    O1  80      0.013      0         0

[PUMPS]
P1      O1    J1  Curve1  ON  0  0

[MAP]
DIMENSIONS 0 0 1 1
Units      Degrees

[COORDINATES]
;;Node  X-Coord  Y-Coord
J1      10.000   50.000
J2      10.010   50.000
O1      10.020   49.990

[VERTICES]
;;Link  X-Coord  Y-Coord
C2      10.015   49.995
"""

START_DAYS = 45444.0  # 2024-06-01 in SWMM's days since 1899-12-30
STEP = 600


def _name_block(names: list[str]) -> bytes:
    return b"".join(struct.pack("<i", len(n)) + n.encode() for n in names)


def write_out(path: Path, *, nodes=("J1", "J2", "O1"), links=("C1", "C2", "P1"), periods=4,
              n_sys=15, node_flood=None, link_flow=None, version=52004, flow_code=3, truncate=False) -> Path:
    """A SWMM results file with one subcatchment, the nodes and links given and no pollutants. Node flooding is
    node_flood[period][node]; link flow is link_flow[period][link]; link capacity is |flow| / 10."""
    subs = ["S1"]
    nn, nl = len(nodes), len(links)
    n_sv, n_nv, n_lv = 8, 6, 5
    head = struct.pack("<7i", MAGIC, version, flow_code, len(subs), nn, nl, 0)
    id_pos = len(head)
    ids = _name_block(subs) + _name_block(list(nodes)) + _name_block(list(links))
    prop_pos = id_pos + len(ids)
    node_types = [0 if n.startswith("J") else 1 for n in nodes]
    link_types = [1 if lk.startswith("P") else 0 for lk in links]
    props = struct.pack("<2i", 1, 1) + struct.pack(f"<{len(subs)}f", *([1.0] * len(subs)))
    props += struct.pack("<4i", 3, 0, 2, 3) + b"".join(struct.pack("<3f", t, 9.0, 2.0) for t in node_types)
    props += struct.pack("<6i", 5, 0, 4, 4, 3, 5) + b"".join(struct.pack("<5f", t, 0, 0, 1.0, 100.0)
                                                             for t in link_types)
    variables = b"".join(struct.pack(f"<{k + 1}i", k, *range(k)) for k in (n_sv, n_nv, n_lv, n_sys))
    variables += struct.pack("<d", START_DAYS) + struct.pack("<i", STEP)
    out_pos = prop_pos + len(props) + len(variables)
    body = b""
    for p in range(periods):
        vals = [0.0] * (len(subs) * n_sv)
        for j in range(nn):
            depth = 0.1 * (p + 1) + j
            flood = node_flood[p][j] if node_flood else 0.0
            vals += [depth, 9.0 + depth, 0.0, 0.0, 1.0, flood]
        for j in range(nl):
            q = link_flow[p][j] if link_flow else 0.0
            vals += [q, 0.5, 1.0, 0.0, abs(q) / 10.0]
        vals += [0.0] * n_sys
        body += struct.pack("<d", START_DAYS + (p + 1) * STEP / 86400) + struct.pack(f"<{len(vals)}f", *vals)
    tail = struct.pack("<6i", id_pos, prop_pos, out_pos, periods, 0, MAGIC)
    data = head + ids + props + variables + body + tail
    if truncate:
        data = data[: len(data) - 12]
    path.write_bytes(data)
    return path


FLOOD = [[0, 0, 0], [0, 2.0, 0], [0, 3.0, 0], [0, 0, 0]]
FLOW = [[1, 2, 0], [4, -9, 0], [3, 5, 1], [1, 1, 0]]


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "tiny.inp"
    inp.write_text(INP)
    out = write_out(tmp_path / "tiny.out", node_flood=FLOOD, link_flow=FLOW)
    return inp, out


def test_read_out_header_and_values(files):
    _inp, out = files
    with read_swmm_out(out) as o:
        assert o.version_label == "5.2.4" and o.flow_units == "CMS" and o.units["depth"] == "m"
        assert o.nodes == ["J1", "J2", "O1"] and o.links == ["C1", "C2", "P1"] and o.subcatchments == ["S1"]
        assert o.node_types == ["junction", "junction", "outfall"] and o.link_types == ["conduit", "conduit", "pump"]
        assert o.n_periods == 4 and o.report_step_s == STEP
        assert o.times() == ["2024-06-01T00:10:00", "2024-06-01T00:20:00", "2024-06-01T00:30:00",
                             "2024-06-01T00:40:00"]
        np.testing.assert_allclose(o.values("node", "flooding"), np.array(FLOOD, dtype="f4"))
        np.testing.assert_allclose(o.values("link", "flow", ["C2"])[:, 0], [2, -9, 5, 1])
        np.testing.assert_allclose(o.values("node", "depth", ["O1"])[:, 0], [2.1, 2.2, 2.3, 2.4], rtol=1e-6)


def test_blocks_give_the_same_values(files, monkeypatch):
    _inp, out = files
    monkeypatch.setattr("aquascope.io.swmm.BLOCK_VALUES", 1)  # one period per block
    with read_swmm_out(out) as o:
        np.testing.assert_allclose(o.values("link", "flow"), np.array(FLOW, dtype="f4"))
        pk = o.peaks("link", "flow", absolute=True)
        assert list(pk["max"]) == [4, -9, 1] and list(pk["period"]) == [1, 1, 2]


def test_swmm_51_system_variable_count(tmp_path):
    """SWMM 5.1 writes 14 system variables, 5.2 writes 15; the reader takes the count from the file."""
    out = write_out(tmp_path / "old.out", n_sys=14, version=51015, node_flood=FLOOD, link_flow=FLOW)
    with read_swmm_out(out) as o:
        assert o.version_label == "5.1.15" and o.n_system_vars == 14
        np.testing.assert_allclose(o.values("node", "flooding")[:, 1], [0, 2, 3, 0])


def test_bad_files(tmp_path):
    bad = tmp_path / "x.out"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(SwmmError, match="not a SWMM results file"):
        read_swmm_out(bad)
    cut = write_out(tmp_path / "cut.out", truncate=True)
    with pytest.raises(SwmmError):
        read_swmm_out(cut)
    with pytest.raises(SwmmError, match="No such file"):
        read_swmm_out(tmp_path / "missing.out")
    with read_swmm_out(write_out(tmp_path / "ok.out")) as o:
        with pytest.raises(SwmmError, match="No node variable"):
            o.values("node", "speed")
        with pytest.raises(SwmmError, match="No link named"):
            o.values("link", "flow", ["nope"])


def test_read_inp_network(files):
    pytest.importorskip("pyproj")
    inp, _out = files
    net = read_swmm_inp(inp)
    assert net.flow_units == "CMS" and net.map_units == "degrees" and net.title == "Tiny network"
    assert [(n["id"], n["type"]) for n in net.nodes] == [("J1", "junction"), ("J2", "junction"), ("O1", "outfall")]
    assert net.nodes[0]["max_depth"] == 2.0 and net.nodes[2]["max_depth"] is None
    c1, c2, p1 = net.links
    assert (c1["from"], c1["to"], c1["length"]) == ("J1", "J2", 120.5) and p1["type"] == "pump"
    assert c2["xy"] == [[10.01, 50.0], [10.015, 49.995], [10.02, 49.99]]
    assert net.crs is not None and net.nodes[0]["lon"] == pytest.approx(10.0) and net.nodes[0]["lat"] == 50.0
    assert net.bbox() == pytest.approx([10.0, 49.99, 10.02, 50.0])


def test_read_inp_without_a_coordinate_system(tmp_path):
    inp = tmp_path / "plain.inp"
    inp.write_text(INP.replace("Units      Degrees", "Units      None"))
    net = read_swmm_inp(inp)
    assert net.crs is None and "lon" not in net.nodes[0] and net.bbox() is None
    assert any("crs" in n for n in net.notes)
    other = tmp_path / "other.txt"
    other.write_text("hello\n")
    with pytest.raises(SwmmError, match="SWMM"):
        read_swmm_inp(other)


def test_summary_numbers(files):
    pytest.importorskip("pyproj")
    inp, out = files
    res = swmm_summary(inp, out, top=2, element="C2")
    assert res["ok"] and res["network"]["node_types"] == {"junction": 2, "outfall": 1}
    assert res["results"]["periods"] == 4 and res["results"]["start"] == "2024-06-01T00:10:00"
    fl = res["flooding"]
    assert fl["nodes_flooded"] == 1 and fl["unit"] == "CMS"
    row = fl["nodes"][0]
    assert row["id"] == "J2" and row["peak"] == 3.0 and row["peak_time"] == "2024-06-01T00:30:00"
    assert row["hours_flooded"] == pytest.approx(2 * STEP / 3600, abs=0.01)
    assert row["volume_m3"] == pytest.approx((2 + 3) * STEP)  # CMS x seconds
    assert row["lat"] == 50.0
    flows = res["flows"]
    assert [r["id"] for r in flows["links"]] == ["C2", "C1"] and flows["links"][0]["peak"] == -9.0
    assert flows["links"][0]["peak_capacity"] == 0.9 and flows["conduits_full"] == 0
    assert res["series"]["kind"] == "link" and res["series"]["flow"] == [2.0, -9.0, 5.0, 1.0]


def test_summary_errors(tmp_path):
    assert swmm_summary()["ok"] is False
    res = swmm_summary(out=tmp_path / "missing.out")
    assert res["ok"] is False and "No such file" in res["error"]


def test_cli_and_mcp(files, monkeypatch, capsys):
    pytest.importorskip("pyproj")
    inp, out = files
    from aquascope.cli import main

    monkeypatch.setattr("sys.argv", ["aquascope", "swmm", str(inp), str(out)])
    main()
    text = capsys.readouterr().out
    assert "Nodes flooded: 1 of 3" in text and "J2 (junction): peak 3.0 CMS" in text
    monkeypatch.setattr("sys.argv", ["aquascope", "swmm", str(out), "--json"])
    main()
    assert json.loads(capsys.readouterr().out)["flows"]["links"][0]["id"] == "C2"

    from aquascope import mcp_server

    res = mcp_server.swmm_summary(inp=str(inp), out=str(out), top=1)
    assert res["ok"] and len(res["flooding"]["nodes"]) == 1


def test_long_series_is_thinned(tmp_path, monkeypatch):
    monkeypatch.setattr("aquascope.io.swmm.SERIES_POINTS", 2)
    out = write_out(tmp_path / "long.out", periods=4, node_flood=FLOOD, link_flow=FLOW)
    res = swmm_summary(out=out, element="J2")
    s = res["series"]
    assert s["every"] == 2 and s["flooding"] == [0.0, 3.0] and s["times"] == ["2024-06-01T00:10:00",
                                                                             "2024-06-01T00:30:00"]


def test_damaged_files_give_a_plain_error(tmp_path):
    """Coordinates that are not numbers are skipped and counted; a results file whose property tables have the
    wrong shape is a SwmmError, so the MCP tool answers ok false instead of raising."""
    inp = tmp_path / "odd.inp"
    inp.write_text(INP.replace("J2      10.010   50.000", "J2      abc      50.000"))
    net = read_swmm_inp(inp)
    assert net.nodes[1]["x"] is None and any("not numbers" in n for n in net.notes)
    # a header that says each node has zero properties: the type column is missing
    head = struct.pack("<7i", MAGIC, 52004, 0, 0, 1, 0, 0)
    ids = _name_block(["J1"])
    props = struct.pack("<i", 0) + struct.pack("<i", 0) + struct.pack("<i", 0)
    tail = struct.pack("<6i", len(head), len(head) + len(ids), len(head) + len(ids) + len(props), 0, 0, MAGIC)
    bad = tmp_path / "props.out"
    bad.write_bytes(head + ids + props + b"\x00" * 16 + tail)
    res = swmm_summary(out=bad)
    assert res["ok"] is False and "not a complete SWMM results file" in res["error"]


def test_link_place_is_the_middle_of_its_line(files):
    pytest.importorskip("pyproj")
    inp, out = files
    res = swmm_summary(inp, out, top=3)
    c1 = next(r for r in res["flows"]["links"] if r["id"] == "C1")
    assert c1["lon"] == pytest.approx(10.005) and c1["lat"] == pytest.approx(50.0)
