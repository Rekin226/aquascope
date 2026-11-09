"""Synthetic MODFLOW files for the layer, section, hydrograph and budget tests (#625, #626). Everything is made up.

A copy of the test-model generator the maintainer's checks use, extended with a cell-by-cell budget writer in the
three layouts MODFLOW writes (full, compact, MODFLOW 6) and a three-layer model with sloping layers and a budget.
Run it as a script to write that model to a folder:

    python tests/test_groundwater/synthetic_models.py OUT_DIR

which writes OUT_DIR/modflow-layered/ (synthetic.nam, .dis, .hds, .cbc).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np


def _text(s: str) -> bytes:
    return f"{s:>16}".encode("ascii")


def hds_bytes(frames, *, double: bool = False, text: str = "HEAD") -> bytes:
    """A binary head file: frames of (kstp, kper, totim, layer, 2D array)."""
    f, dt = ("d", "<f8") if double else ("f", "<f4")
    out = b""
    for kstp, kper, totim, lay, arr in frames:
        nrow, ncol = arr.shape
        out += struct.pack(f"<ii{f}{f}", kstp, kper, totim, totim) + _text(text)
        out += struct.pack("<iii", ncol, nrow, lay) + np.asarray(arr, dt).tobytes()
    return out


def cbc_record(kstp, kper, text, ncol, nrow, nlay, *, imeth=None, values=None, nodes=None, layer=None,
               delt=1.0, pertim=1.0, totim=1.0, double=False, aux=(), names=None, nodes2=None) -> bytes:
    """One budget record. ``imeth=None`` writes the full (non-compact) layout; 1 to 6 the compact ones."""
    f, dt = ("d", "<f8") if double else ("f", "<f4")
    if imeth is None:
        return (struct.pack("<ii", kstp, kper) + _text(text) + struct.pack("<iii", ncol, nrow, nlay)
                + np.asarray(values, dt).tobytes())
    out = struct.pack("<ii", kstp, kper) + _text(text) + struct.pack("<iii", ncol, nrow, -abs(nlay))
    out += struct.pack(f"<i{f}{f}{f}", imeth, delt, pertim, totim)
    if imeth in (1, 4):
        return out + np.asarray(values, dt).tobytes()
    if imeth == 3:
        return out + np.asarray(layer, "<i4").tobytes() + np.asarray(values, dt).tobytes()
    if imeth == 2:
        rec = np.zeros(len(nodes), [("n", "<i4"), ("q", dt)])
        rec["n"], rec["q"] = nodes, values
        return out + struct.pack("<i", len(nodes)) + rec.tobytes()
    # 5 and 6: a list with auxiliary values (6 after the model and package names, with two ids per row)
    if imeth == 6:
        out += b"".join(_text(n) for n in (names or ("GWF", "WEL-1", "GWF", "WEL-1")))
    out += struct.pack("<i", 1 + len(aux)) + b"".join(_text(a) for a in aux)
    fields = [("n", "<i4")] + ([("n2", "<i4")] if imeth == 6 else []) + [("q", dt)]
    fields += [(f"a{k}", dt) for k in range(len(aux))]
    rec = np.zeros(len(nodes), fields)
    rec["n"], rec["q"] = nodes, values
    if imeth == 6:
        rec["n2"] = nodes if nodes2 is None else nodes2
    for k in range(len(aux)):
        rec[f"a{k}"] = 1.0
    return out + struct.pack("<i", len(nodes)) + rec.tobytes()


# ── a three-layer model with sloping layers, a pumped confined aquifer and its budget ──────────────────────

NROW, NCOL, NLAY, CELL = 30, 40, 3, 100.0
XUL, YUL = 448_000.0, 5_414_000.0          # FloPy's xul/yul (the upper-left corner), EPSG:32631, near Paris
MONTHS = 24
DAYS = 30.4375
WELL = (14, 22)                            # row 15, column 23 (0-based here), screened in layer 3
Q_WELL = 2500.0                            # m3/day while it pumps


def layered_model():
    r, c = np.mgrid[0:NROW, 0:NCOL]
    x = (c + 0.5) * CELL
    y = (NROW - r - 0.5) * CELL
    top = 76.0 - 0.003 * x + 4.0 * np.sin(x / 700.0) * np.cos(y / 900.0)       # rolling ground falling east
    b1 = 38.0 - 0.002 * x + 1.5 * np.sin(y / 600.0)                              # sand above a clay
    b2 = b1 - (9.0 - 0.0008 * x)                                                 # the clay thins to the east
    b3 = np.full_like(top, -40.0)                                                # the chalk's base
    inactive = (r + c < 5) | ((r > 25) & (c > 35))
    dist = np.hypot((c - WELL[1]) * CELL, (r - WELL[0]) * CELL) + 50.0
    cone = np.log(1500.0 / np.minimum(dist, 1500.0)) / np.log(30.0)
    heads, pumping = [], []
    for m in range(MONTHS):
        p = 1.0 if m < 14 else max(0.0, 1.0 - (m - 13) / 8.0)
        pumping.append(p)
        season = 0.6 * np.sin(2 * np.pi * (m - 2) / 12.0)
        h1 = 64.0 - 0.0045 * x + 0.001 * y + season - p * 0.8 * cone
        h3 = h1 - 1.2 - p * 6.5 * cone
        h2 = (h1 + h3) / 2.0
        heads.append([np.where(inactive, 1.0e30, h) for h in (h1, h2, h3)])
    return {"top": top, "botm": [b1, b2, b3], "heads": heads, "pumping": pumping, "inactive": inactive,
            "cone": cone}


def layered_budget(model, *, double: bool = False) -> tuple[bytes, list[dict]]:
    """The model's compact budget file (as MODFLOW-2005 writes with COMPACT BUDGET) and the terms it holds, per
    month: {name: (in, out)} in m3/day."""
    act = ~model["inactive"]
    ncell = NROW * NCOL
    area = CELL * CELL
    node = lambda lay, r, c: lay * ncell + r * NCOL + c + 1  # noqa: E731
    out, truth = b"", []
    for m in range(MONTHS):
        kw = {"delt": DAYS, "pertim": DAYS, "totim": DAYS * (m + 1), "double": double}
        k = (1, m + 1)
        rch_rate = 0.00025 * (1.0 + 0.7 * np.cos(2 * np.pi * m / 12.0))          # wetter winters
        rch = np.where(act, rch_rate * area, 0.0)
        p = model["pumping"][m]
        well = -Q_WELL * p
        river_nodes = [node(0, r, NCOL - 1) for r in range(NROW) if act[r, NCOL - 1]]
        river_q = -(2600.0 - 900.0 * p) / len(river_nodes)                      # the gaining river loses flow
        ch_rows = [r for r in range(NROW) if act[r, 0]]
        ch_nodes = [node(lay, r, 0) for lay in (0, 2) for r in ch_rows]
        ch_q = (300.0 + 700.0 * p) / len(ch_nodes)
        tin = float(rch.sum()) + ch_q * len(ch_nodes)
        tout = -well - river_q * len(river_nodes)
        s = tout - tin                                                           # > 0: water released from storage
        w = np.where(act, 0.2 + model["cone"], 0.0)
        w3 = np.zeros((NLAY, NROW, NCOL))
        w3[0], w3[2] = 0.3 * w, 0.7 * w
        sto = s * w3 / w3.sum()
        season = 0.08 * float(rch.sum()) * np.sin(2 * np.pi * m / 12.0)   # stored in some cells, released in others
        sto[0] += np.where(act, season / act.sum(), 0.0)
        sto[2] -= np.where(act, season / act.sum(), 0.0)
        frf = np.zeros((NLAY, NROW, NCOL))
        frf[:, :, :-1] = 2.0                                                     # flows between cells: not budget terms
        out += cbc_record(*k, "STORAGE", NCOL, NROW, NLAY, imeth=1, values=sto.ravel(), **kw)
        out += cbc_record(*k, "CONSTANT HEAD", NCOL, NROW, NLAY, imeth=2, nodes=ch_nodes,
                          values=[ch_q] * len(ch_nodes), **kw)
        out += cbc_record(*k, "FLOW RIGHT FACE", NCOL, NROW, NLAY, imeth=1, values=frf.ravel(), **kw)
        out += cbc_record(*k, "FLOW FRONT FACE", NCOL, NROW, NLAY, imeth=1, values=-frf.ravel(), **kw)
        out += cbc_record(*k, "FLOW LOWER FACE", NCOL, NROW, NLAY, imeth=1, values=frf.ravel(), **kw)
        out += cbc_record(*k, "WELLS", NCOL, NROW, NLAY, imeth=2, nodes=[node(2, *WELL)], values=[well], **kw)
        out += cbc_record(*k, "RECHARGE", NCOL, NROW, NLAY, imeth=3, layer=np.ones(NROW * NCOL, int),
                          values=rch.ravel(), **kw)
        out += cbc_record(*k, "RIVER LEAKAGE", NCOL, NROW, NLAY, imeth=2, nodes=river_nodes,
                          values=[river_q] * len(river_nodes), **kw)
        pos, neg = float(sto[sto > 0].sum()), float(-sto[sto < 0].sum())
        truth.append({"STORAGE": (pos, neg), "CONSTANT HEAD": (ch_q * len(ch_nodes), 0.0), "WELLS": (0.0, -well),
                      "RECHARGE": (float(rch.sum()), 0.0), "RIVER LEAKAGE": (0.0, -river_q * len(river_nodes))})
    return out, truth


def _array_text(a: np.ndarray, name: str) -> str:
    rows = "\n".join(" " + " ".join(f"{v:.3f}" for v in row) for row in np.asarray(a).reshape(NROW, NCOL))
    return f"INTERNAL 1.0 (FREE) -1 #{name}\n{rows}\n"


def layered_files() -> dict[str, bytes]:
    model = layered_model()
    georef = f"#xul:{XUL:.0f}; yul:{YUL:.0f}; rotation:0; epsg:32631; start_datetime:1/1/2024\n"
    dis = ("# Discretization file, SYNTHETIC three-layer test model (not real)\n" + georef
           + f"{NLAY:5d}{NROW:5d}{NCOL:5d}{MONTHS:5d}    4    2\n 0 0 0\n"
           + f"INTERNAL 1.0 (FREE) -1 #delr\n {' '.join(['100.0'] * NCOL)}\n"
           + f"INTERNAL 1.0 (FREE) -1 #delc\n {' '.join(['100.0'] * NROW)}\n"
           + _array_text(model["top"], "top")
           + "".join(_array_text(b, f"botm layer {k + 1}") for k, b in enumerate(model["botm"]))
           + "".join(f" {DAYS} 1 1.0 TR\n" for _ in range(MONTHS)))
    frames = [(1, m + 1, DAYS * (m + 1), lay + 1, model["heads"][m][lay]) for m in range(MONTHS) for lay in range(NLAY)]
    nam = ("# Name file, SYNTHETIC three-layer test model (not real)\n" + georef
           + "LIST 2 synthetic.list\nBAS6 3 synthetic.bas\nDIS 11 synthetic.dis\nOC 14 synthetic.oc\n"
           + "DATA(BINARY) 50 synthetic.cbc REPLACE\nDATA(BINARY) 51 synthetic.hds REPLACE\n")
    cbc, _ = layered_budget(model)
    return {"synthetic.nam": nam.encode(), "synthetic.dis": dis.encode(), "synthetic.hds": hds_bytes(frames),
            "synthetic.cbc": cbc}


README = """modflow-layered/ (synthetic.nam, synthetic.dis, synthetic.hds, synthetic.cbc)
  A MODFLOW-2005-style model with three sloping layers (a sand, a clay that thins to the east, a chalk),
  30 x 40 cells of 100 m near Paris (UTM 31N), 24 monthly heads from January 2024, and a well pumping
  2,500 m3/day from the chalk at row 15, column 23 for 14 months, then easing off over 8. The .cbc is a
  compact budget (storage, constant heads, wells, recharge, river leakage, and flows between cells).
  Made by tests/test_groundwater/synthetic_models.py in the aquascope repository. Everything is made up.
  Test: drop all four files. Expect 3D layers, a cross-section through the cone, three hydrograph lines
  at the well's cell, and a budget where storage gives up water in the drier months while the well
  pumps, and refills after.
"""


if __name__ == "__main__":
    out = Path(sys.argv[1]) / "modflow-layered"
    out.mkdir(parents=True, exist_ok=False)   # never over existing files
    for name, data in layered_files().items():
        (out / name).write_bytes(data)
    (out / "README.txt").write_text(README)
    print("written to", out)
