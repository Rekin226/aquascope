"""The water budget of a MODFLOW run, from its cell-by-cell budget file (#626). Experimental.

MODFLOW writes the flow terms of every cell, for every saved time step, to a binary *cell-by-cell budget file*
(``.cbc``, also ``.bud`` or ``.cbb``). :func:`read_budget` reads one with numpy only (so it runs in the browser's
Pyodide), in each of the layouts MODFLOW writes:

- the **full** layout (MODFLOW-2005 without ``COMPACT BUDGET``): a header (KSTP, KPER, TEXT, NCOL, NROW, NLAY) and the
  whole 3D array;
- the **compact** layout (``COMPACT BUDGET`` in MODFLOW-2005's Output Control, and always in MODFLOW 6): the same header
  with NLAY negative, a second header (IMETH, DELT, PERTIM, TOTIM), then the values as IMETH says: 1 a full array, 2 a
  list of (cell, value), 3 a layer array and a 2D array, 4 a 2D array for layer 1, 5 a list with auxiliary values, and
  6 (MODFLOW 6) a list of (id1, id2, values) after the model and package names;

in single or double precision (found from the file, not assumed). The layouts are those of the MODFLOW-2005
documentation (Harbaugh, 2005, USGS Techniques and Methods 6-A16, https://doi.org/10.3133/tm6A16; its Output Control
input page, where COMPACT BUDGET is set, https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/oc.htm), and IMETH 6 as FloPy's
``CellBudgetFile`` reads MODFLOW 6's files (https://flopy.readthedocs.io/en/latest/source/flopy.utils.binaryfile.html);
a test checks the reader against FloPy where FloPy is installed.

:func:`water_budget` turns the records into the budget a modeller reads: per saved time, for each boundary and storage
term, the volume rate **in** (the sum of its positive cell values: water entering the groundwater system) and **out**
(the sum of its negative values), and the percent discrepancy, ``100 * (IN - OUT) / ((IN + OUT) / 2)``, as MODFLOW's
own volumetric budget reports it (Harbaugh, 2005). Flows between cells (``FLOW RIGHT FACE``, ``FLOW-JA-FACE`` and the
others) and MODFLOW 6's ``DATA-`` records are internal and left out of the totals. Rates are in the model's length
cubed per its time unit.
"""

from __future__ import annotations

import math
import re
import struct
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

#: Records that are flows between cells, or data, not a source or sink of the groundwater system.
INTERNAL = re.compile(r"^(FLOW (RIGHT|FRONT|LOWER) FACE|FLOW-JA-FACE.*|DATA-.*|FLOW JA FACE)$")
_TEXT_OK = re.compile(rb"^[\x20-\x7e]{16}$")

#: What each budget term is called on the chart, by the record's text (MODFLOW-2005 and MODFLOW 6 names).
LABELS = {
    "STORAGE": "Storage", "STO-SS": "Storage (specific storage)", "STO-SY": "Storage (specific yield)",
    "CONSTANT HEAD": "Constant heads", "CHD": "Constant heads",
    "RECHARGE": "Recharge", "RCH": "Recharge", "RCHA": "Recharge", "UZF RECHARGE": "Recharge (UZF)",
    "WELLS": "Wells", "WEL": "Wells", "MNW2": "Wells (MNW2)", "MAW": "Wells (MAW)",
    "RIVER LEAKAGE": "Rivers", "RIV": "Rivers", "STREAM LEAKAGE": "Streams", "SFR": "Streams",
    "DRAINS": "Drains", "DRN": "Drains", "HEAD DEP BOUNDS": "General heads", "GHB": "General heads",
    "ET": "Evapotranspiration", "EVT": "Evapotranspiration", "EVTA": "Evapotranspiration",
    "ET SEGMENTS": "Evapotranspiration", "GW ET": "Evapotranspiration (UZF)", "LAKE SEEPAGE": "Lakes",
    "LAK": "Lakes", "UZF-GWRCH": "Recharge (UZF)", "UZF-GWET": "Evapotranspiration (UZF)",
    "SURFACE LEAKAGE": "Surface leakage (UZF)", "UZF-GWD": "Surface leakage (UZF)",
}


def _as_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    return Path(data).read_bytes()


def term_label(text: str) -> str:
    """The chart's name for a budget record's text: "Recharge" for RECHARGE or RCH, else the text as written."""
    t = str(text).strip().upper()
    if t in LABELS:
        return LABELS[t]
    return t.capitalize() if t else "Unnamed"


def _parse(buf: bytes, prec: str) -> list[dict[str, Any]]:
    """Every record of the file in one precision; raises ValueError when the bytes do not fit it."""
    fsize = 4 if prec == "single" else 8
    fch = "f" if prec == "single" else "d"
    rdt = np.dtype("<f4" if prec == "single" else "<f8")
    pos, end = 0, len(buf)
    out: list[dict[str, Any]] = []

    def need(n: int) -> None:
        if pos + n > end:
            raise ValueError(f"the file ends inside a record at byte {pos}")

    def ints(n: int) -> np.ndarray:
        nonlocal pos
        need(4 * n)
        a = np.frombuffer(buf, "<i4", n, pos)
        pos += 4 * n
        return a

    def reals(n: int) -> np.ndarray:
        nonlocal pos
        need(fsize * n)
        a = np.frombuffer(buf, rdt, n, pos)
        pos += fsize * n
        return a

    while pos < end:
        need(36)
        kstp, kper = struct.unpack_from("<ii", buf, pos)
        text = buf[pos + 8:pos + 24]
        ncol, nrow, nlay = struct.unpack_from("<iii", buf, pos + 24)
        if not _TEXT_OK.match(text) or not text.strip() or not (0 < ncol < 10**9 and 0 < nrow < 10**9 and nlay != 0):
            raise ValueError(f"unreadable record header at byte {pos}")
        if not (0 <= kstp < 10**8 and 0 <= kper < 10**8):
            raise ValueError(f"unreadable time step at byte {pos}")
        pos += 36
        rec: dict[str, Any] = {"kstp": kstp, "kper": kper, "text": text.decode("ascii").strip(), "ncol": ncol,
                               "nrow": nrow, "nlay": abs(nlay), "imeth": 0, "delt": None, "pertim": None,
                               "totim": None}
        n2d, n3d = ncol * nrow, ncol * nrow * abs(nlay)
        if nlay > 0:  # the full layout: the 3D array and nothing else
            rec["values"] = reals(n3d)
            out.append(rec)
            continue
        need(4 + 3 * fsize)
        imeth = struct.unpack_from("<i", buf, pos)[0]
        delt, pertim, totim = struct.unpack_from(f"<{fch}{fch}{fch}", buf, pos + 4)
        pos += 4 + 3 * fsize
        if not all(math.isfinite(v) for v in (delt, pertim, totim)) or not 0 <= imeth <= 6:
            raise ValueError(f"unreadable compact header at byte {pos}")
        rec.update(imeth=imeth, delt=float(delt), pertim=float(pertim), totim=float(totim))
        if imeth in (0, 1):
            rec["values"] = reals(n3d)
        elif imeth == 2:
            nlist = int(ints(1)[0])
            if nlist < 0:
                raise ValueError("a negative list length")
            need(nlist * (4 + fsize))
            dt = np.dtype([("node", "<i4"), ("q", rdt)])
            arr = np.frombuffer(buf, dt, nlist, pos)
            pos += nlist * dt.itemsize
            rec["values"], rec["nodes"] = arr["q"], arr["node"]
        elif imeth == 3:
            rec["layer"] = ints(n2d)
            rec["values"] = reals(n2d)
        elif imeth == 4:
            rec["values"] = reals(n2d)
        else:  # 5 (MODFLOW-2005 list with auxiliary values) and 6 (MODFLOW 6)
            if imeth == 6:
                need(64)
                rec["names"] = [buf[pos + 16 * k:pos + 16 * (k + 1)].decode("ascii", "replace").strip() for k in range(4)]
                pos += 64
            nauxp1 = int(ints(1)[0])
            if not 1 <= nauxp1 < 1000:
                raise ValueError("an unreadable number of list values")
            need(16 * (nauxp1 - 1))
            pos += 16 * (nauxp1 - 1)
            nlist = int(ints(1)[0])
            if nlist < 0:
                raise ValueError("a negative list length")
            ids = [("node", "<i4")] + ([("node2", "<i4")] if imeth == 6 else [])
            dt = np.dtype(ids + [("q", rdt)] + [(f"aux{k}", rdt) for k in range(nauxp1 - 1)])
            need(nlist * dt.itemsize)
            arr = np.frombuffer(buf, dt, nlist, pos)
            pos += nlist * dt.itemsize
            rec["values"], rec["nodes"] = arr["q"], arr["node"]
        out.append(rec)
    return out


def budget_precision(buf: bytes) -> str:
    """'single' or 'double': the one in which the whole file reads, record after record, to its last byte."""
    errors = []
    for prec in ("single", "double"):
        try:
            if _parse(buf, prec):
                return prec
        except ValueError as exc:
            errors.append(f"{prec}: {exc}")
    raise ValueError("not a MODFLOW cell-by-cell budget file (" + "; ".join(errors) + ")")


def read_budget(data: Any) -> dict[str, Any]:
    """Read a MODFLOW cell-by-cell budget file (bytes or a path): MODFLOW-2005 full or compact, or MODFLOW 6.

    Returns {precision, compact, records}: each record is {kstp, kper, text, ncol, nrow, nlay, imeth, delt, pertim,
    totim (None in the full layout), values} with ``nodes`` (1-based cells) for the list layouts and ``layer`` for
    IMETH 3."""
    buf = _as_bytes(data)
    if not buf:
        raise ValueError("an empty budget file")
    prec = budget_precision(buf)
    records = _parse(buf, prec)
    return {"precision": prec, "compact": any(r["imeth"] > 0 or r["totim"] is not None for r in records),
            "records": records}


def _key(r: dict[str, Any]) -> tuple[int, int]:
    return (int(r["kstp"]), int(r["kper"]))


def water_budget(data: Any, *, heads: dict[str, Any] | None = None, start_date: Any = None,
                 time_units: str | None = None, length_units: str | None = None) -> dict[str, Any]:
    """The water budget over time from a cell-by-cell budget file (bytes, a path, or :func:`read_budget`'s result).

    ``heads``: the result of :func:`aquascope.groundwater.surface.read_modflow_heads` for the same run, when there
    is one; its times, dates and units then label the budget (and give times to a full-layout file, which has none).
    Otherwise ``start_date`` (YYYY-MM-DD) and ``time_units`` date the totals, and ``length_units`` names the volume.

    Returns a dict: ``times``, ``dates`` (or None), ``kstpkper``, ``terms`` (each {text, label, in, out}, lists of
    rates per time), ``total_in``, ``total_out``, ``discrepancy_pct``, ``unit`` (for example "m³/day"),
    ``internal`` (the records left out, flows between cells), ``precision``, ``compact``, ``notes`` and a ``summary``."""
    parsed = data if isinstance(data, dict) and "records" in data else read_budget(data)
    recs = parsed["records"]
    if not recs:
        raise ValueError("no records in the budget file")
    keys: list[tuple[int, int]] = []
    seen: dict[tuple[int, int], int] = {}
    totim: dict[tuple[int, int], float] = {}
    for r in recs:
        k = _key(r)
        if k not in seen:
            seen[k] = len(keys)
            keys.append(k)
        if r.get("totim") is not None:
            totim.setdefault(k, float(r["totim"]))
    terms: dict[str, dict[str, Any]] = {}
    internal: list[str] = []
    for r in recs:
        text = r["text"].upper()
        if INTERNAL.match(text):
            if text not in internal:
                internal.append(text)
            continue
        # MODFLOW 6 names each package's record by its type; two packages of one type add up under one term.
        t = terms.setdefault(text, {"text": text, "label": term_label(text), "in": [0.0] * len(keys),
                                    "out": [0.0] * len(keys)})
        v = np.asarray(r["values"], float)
        v = v[np.isfinite(v)]
        i = seen[_key(r)]
        t["in"][i] += float(v[v > 0].sum())
        t["out"][i] += float(-v[v < 0].sum())
    notes: list[str] = []
    times: list[float | None] = [totim.get(k) for k in keys]
    dates = None
    units_t = (time_units or "").lower() or None
    units_l = (length_units or "").lower() or None
    if heads is not None and not heads.get("error"):
        hk = {tuple(map(int, kk)): j for j, kk in enumerate(heads.get("kstpkper") or [])}
        htimes = heads.get("times") or []
        hdates = heads.get("dates")
        times = [times[i] if times[i] is not None else (htimes[hk[k]] if k in hk else None) for i, k in enumerate(keys)]
        if hdates:
            dates = [hdates[hk[k]] if k in hk else None for k in keys]
        units_t = units_t or heads.get("time_units")
        units_l = units_l or (heads.get("grid") or {}).get("length_units")
        missing = sum(1 for k in keys if k not in hk)
        if missing:
            notes.append(f"{missing} of the budget's {len(keys)} saved times are not in the head file.")
    elif start_date and all(t is not None for t in times):
        from aquascope.groundwater.surface import TIME_UNITS_DAYS, _parse_start

        try:
            d0 = _parse_start(start_date)
            per = TIME_UNITS_DAYS.get(units_t or "days", 1.0)
            dates = [(d0 + timedelta(days=float(t) * per)).date().isoformat() for t in times]  # type: ignore[arg-type]
        except ValueError:
            notes.append(f"Could not read the start date {start_date!r}; times are model times.")
    if any(t is None for t in times):
        notes.append("The file is in the full layout, which does not store times: the budget is labelled by stress "
                     "period and time step.")
    order = sorted(terms.values(), key=lambda t: -(sum(t["in"]) + sum(t["out"])))
    tin = [sum(t["in"][i] for t in order) for i in range(len(keys))]
    tout = [sum(t["out"][i] for t in order) for i in range(len(keys))]
    disc = [round(100.0 * (a - b) / ((a + b) / 2.0), 4) if (a + b) > 0 else None for a, b in zip(tin, tout)]
    for t in order:
        t["in"] = [round(x, 6) for x in t["in"]]
        t["out"] = [round(x, 6) for x in t["out"]]
    vol = {"meters": "m³", "metres": "m³", "m": "m³", "feet": "ft³", "ft": "ft³", "centimeters": "cm³"}.get(units_l or "")
    per = {"days": "day", "day": "day", "d": "day", "seconds": "s", "s": "s", "hours": "hour", "minutes": "minute",
           "years": "year"}.get(units_t or "")
    unit = f"{vol or 'L³'}/{per or 'T'}"
    if not vol or not per:
        notes.append("The files do not name the length or time unit: rates are in the model's own units (L³/T).")
    worst = max((abs(d) for d in disc if d is not None), default=None)
    names = ", ".join(t["label"] for t in order) or "none"
    return {
        "kind": "modflow_budget", "precision": parsed["precision"], "compact": parsed["compact"],
        "times": times, "dates": dates, "kstpkper": [list(k) for k in keys], "terms": order,
        "total_in": [round(x, 6) for x in tin], "total_out": [round(x, 6) for x in tout],
        "discrepancy_pct": disc, "unit": unit, "internal": internal, "notes": notes,
        "method": "In: the sum of a term's positive cell values (into the groundwater system); out: of its negative "
                  "values. Discrepancy 100 x (in - out) / ((in + out) / 2), as MODFLOW's budget reports it "
                  "(Harbaugh, 2005, USGS TM 6-A16).",
        "summary": (f"Water budget for {len(keys)} saved time(s): {names}."
                    + (f" Largest discrepancy {worst:.2f}%." if worst is not None else "")),
    }


def budget_at(res: dict[str, Any], time_index: int = -1) -> dict[str, Any]:
    """One saved time of :func:`water_budget`'s result: each term's in, out and net, and the totals."""
    n = len(res["kstpkper"])
    t = int(time_index)
    if not -n <= t < n:
        raise ValueError(f"time_index out of range (0 to {n - 1})")
    t %= n
    rows = [{"label": x["label"], "in": x["in"][t], "out": x["out"][t], "net": round(x["in"][t] - x["out"][t], 6)}
            for x in res["terms"]]
    return {"time_index": t, "time": res["times"][t], "date": (res.get("dates") or [None] * n)[t],
            "kstpkper": res["kstpkper"][t], "terms": rows, "total_in": res["total_in"][t],
            "total_out": res["total_out"][t], "discrepancy_pct": res["discrepancy_pct"][t], "unit": res["unit"]}


__all__ = ["INTERNAL", "LABELS", "budget_at", "budget_precision", "read_budget", "term_label", "water_budget"]
