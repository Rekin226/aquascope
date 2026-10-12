"""EPA SWMM files: writers for input sections, and readers for a model and its results (#560).

Writers for the US EPA Storm Water Management Model (SWMM):

- **SWMM timeseries**: ``[TIMESERIES]`` section entries.
- **SWMM rainfall**: ``.dat`` rain-gauge data files.

Readers, in plain Python and numpy so they run in the browser (Pyodide) as well:

- :func:`read_swmm_inp` reads the drainage network of a ``.inp`` file: nodes (junctions, outfalls, storage,
  dividers) and links (conduits, pumps, orifices, weirs, outlets) with their map coordinates, and their longitude
  and latitude when the coordinate system is known.
- :func:`read_swmm_out` reads the binary results file (``.out``) of SWMM 5.0 to 5.2: the model's report times and
  every node, link and subcatchment variable, one variable at a time, in blocks, so a large file is never held in
  memory whole.
- :func:`swmm_summary` is the small answer for the MCP server and the CLI (``aquascope swmm FILE``): the network,
  the report period, the nodes that flood and the links that carry the most flow. Every number in it is computed
  here from the files.

SWMM map coordinates are in the project's own units ([MAP] Units: none, feet, metres or degrees). A model in
degrees lands on the globe as it is; any other needs ``crs=`` (an EPSG code or WKT), as the mesh readers do.

References
----------
- SWMM User's Manual: https://www.epa.gov/water-research/storm-water-management-model-swmm
- The binary results file layout is the one SWMM writes in ``src/solver/output.c`` (public domain,
  https://github.com/USEPA/Stormwater-Management-Model).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ── SWMM Timeseries ─────────────────────────────────────────────────────────
def write_swmm_timeseries(
    df: pd.DataFrame,
    name: str,
    path: str | Path,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> None:
    """Write an EPA SWMM timeseries input section.

    Output format::

        [TIMESERIES]
        ;;Name          Date        Time        Value
        name            MM/DD/YYYY  HH:MM       value

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing timestamps and values.
    name : str
        Timeseries identifier used by SWMM.
    path : str | Path
        Destination file path.
    time_col : str
        Column with timestamps.
    value_col : str
        Column with numeric values.
    """
    import pandas as pd

    path = Path(path)
    lines: list[str] = [
        "[TIMESERIES]",
        ";;Name          Date        Time        Value",
    ]

    for _, row in df.iterrows():
        ts = pd.Timestamp(row[time_col])
        date_str = ts.strftime("%m/%d/%Y")
        time_str = ts.strftime("%H:%M")
        value = float(row[value_col])
        lines.append(f"{name:<16s}{date_str:<12s}{time_str:<12s}{value:.4f}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("write_swmm_timeseries: wrote %d rows to %s", len(df), path)


# ── SWMM Rainfall .dat ──────────────────────────────────────────────────────
def write_swmm_rainfall(
    df: pd.DataFrame,
    gauge_name: str,
    path: str | Path,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> None:
    """Write an EPA SWMM rainfall data file (``.dat`` format).

    Output format::

        StationID Year Month Day Hour Minute Value

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing timestamps and rainfall values.
    gauge_name : str
        Rain-gauge / station identifier.
    path : str | Path
        Destination file path.
    time_col : str
        Column with timestamps.
    value_col : str
        Column with rainfall values.
    """
    import pandas as pd

    path = Path(path)
    lines: list[str] = []

    for _, row in df.iterrows():
        ts = pd.Timestamp(row[time_col])
        value = float(row[value_col])
        lines.append(
            f"{gauge_name} {ts.year:4d} {ts.month:2d} {ts.day:2d} "
            f"{ts.hour:2d} {ts.minute:2d} {value:.4f}"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("write_swmm_rainfall: wrote %d rows to %s", len(df), path)


# ── Reading a model and its results (#560) ──────────────────────────────────

MAGIC = 516114522
MAX_FILE_BYTES = 2 * 1024**3
# Values read per block when scanning the results (about 32 MB of float32).
BLOCK_VALUES = 8_000_000
# The most values per variable an element's time series returns.
SERIES_POINTS = 2000

FLOW_UNITS = ("CFS", "GPM", "MGD", "CMS", "LPS", "MLD")
# Cubic metres per second in one of each flow unit: exact from 1 ft = 0.3048 m and 1 US gallon = 3.785411784 L.
TO_M3S = {"CFS": 0.028316846592, "GPM": 3.785411784e-3 / 60, "MGD": 3785.411784 / 86400,
          "CMS": 1.0, "LPS": 1e-3, "MLD": 1000.0 / 86400}
NODE_TYPES = ("junction", "outfall", "storage", "divider")
LINK_TYPES = ("conduit", "pump", "orifice", "weir", "outlet")
# The first variables of each object, in file order; pollutant concentrations follow them.
SUBCATCH_VARS = ("rainfall", "snow_depth", "evaporation", "infiltration", "runoff", "groundwater_flow",
                 "groundwater_elevation", "soil_moisture")
NODE_VARS = ("depth", "head", "volume", "lateral_inflow", "total_inflow", "flooding")
LINK_VARS = ("flow", "depth", "velocity", "volume", "capacity")

_NODE_SECTIONS = {"JUNCTIONS": "junction", "OUTFALLS": "outfall", "STORAGE": "storage", "DIVIDERS": "divider"}
_LINK_SECTIONS = {"CONDUITS": "conduit", "PUMPS": "pump", "ORIFICES": "orifice", "WEIRS": "weir",
                  "OUTLETS": "outlet"}
_EPOCH = datetime(1899, 12, 30)


class SwmmError(ValueError):
    """A SWMM file the readers cannot use, with a message meant for the person who gave it."""


def _check_size(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        raise SwmmError(f"No such file: {p}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SwmmError(f"{p.name} is {size / 1024**3:.1f} GB; the SWMM reader reads files up to "
                        f"{MAX_FILE_BYTES // 1024**3} GB.")
    return int(size)


def unit_system(flow_units: str) -> str:
    """'US' for CFS, GPM and MGD; 'SI' for CMS, LPS and MLD (SWMM's own rule for every other quantity)."""
    return "US" if flow_units.upper() in ("CFS", "GPM", "MGD") else "SI"


def units_of(flow_units: str) -> dict[str, str]:
    """The unit of each result variable for a model in these flow units."""
    us = unit_system(flow_units) == "US"
    length = "ft" if us else "m"
    return {"flow": flow_units.upper(), "flooding": flow_units.upper(), "lateral_inflow": flow_units.upper(),
            "total_inflow": flow_units.upper(), "runoff": flow_units.upper(), "depth": length, "head": length,
            "volume": "ft3" if us else "m3", "velocity": "ft/s" if us else "m/s", "capacity": "fraction full",
            "rainfall": "in/h" if us else "mm/h"}


def _version_label(code: int) -> str:
    """52004 -> '5.2.4', 51015 -> '5.1.15': SWMM writes major*10000 + minor*1000 + patch."""
    return f"{code // 10000}.{(code // 1000) % 10}.{code % 1000}"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _from_days(days: float) -> datetime:
    """SWMM dates are days since 1899-12-30; rounded to the second."""
    return _EPOCH + timedelta(seconds=round(float(days) * 86400.0))


# ── the network (.inp) ──────────────────────────────────────────────────────

@dataclass
class SwmmNetwork:
    """The drainage network of a SWMM input file: nodes and links with map coordinates (x, y) and, once placed,
    longitude and latitude."""

    name: str
    title: str
    options: dict[str, str]
    map_units: str | None
    nodes: list[dict[str, Any]]
    links: list[dict[str, Any]]
    subcatchments: int
    crs: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def flow_units(self) -> str:
        return self.options.get("FLOW_UNITS", "CFS").upper()

    @property
    def placed(self) -> bool:
        return self.crs is not None

    def bbox(self) -> list[float] | None:
        """[west, south, east, north] in degrees over the placed nodes and link vertices, or None."""
        lons = [n["lon"] for n in self.nodes if n.get("lon") is not None]
        lats = [n["lat"] for n in self.nodes if n.get("lat") is not None]
        for lk in self.links:
            for lon, lat in lk.get("lonlat") or []:
                lons.append(lon)
                lats.append(lat)
        if not lons:
            return None
        return [min(lons), min(lats), max(lons), max(lats)]


def _sections(text: str) -> dict[str, list[list[str]]]:
    """Each [SECTION] as rows of whitespace-split fields, comments (;) and blank lines dropped."""
    out: dict[str, list[list[str]]] = {}
    current: list[list[str]] | None = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            current = out.setdefault(line[1:line.index("]")].strip().upper(), [])
            continue
        if current is not None:
            current.append(line.split())
    return out


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _place(x: np.ndarray, y: np.ndarray, crs_in: Any, origin: str) -> tuple[np.ndarray, np.ndarray, dict]:
    from aquascope.io.mesh import MeshError, _crs_label, _transformer

    try:
        src, tr = _transformer(crs_in)
    except MeshError as exc:
        raise SwmmError(str(exc)) from exc
    lon, lat = tr.transform(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    lon, lat = np.asarray(lon, dtype="f8"), np.asarray(lat, dtype="f8")
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any() or np.nanmax(np.abs(lat[ok])) > 90 or np.nanmax(np.abs(lon[ok])) > 180:
        raise SwmmError("The network does not land on the globe in the coordinate system given. Check the "
                        "projection.")
    return lon, lat, {"label": _crs_label(src), "from": origin}


def read_swmm_inp(path: str | Path, *, crs: str | int | None = None) -> SwmmNetwork:
    """Read the drainage network of a SWMM ``.inp`` file.

    Nodes come from [JUNCTIONS], [OUTFALLS], [STORAGE] and [DIVIDERS] with their invert elevation and, where the
    section has it, their full depth; links from [CONDUITS], [PUMPS], [ORIFICES], [WEIRS] and [OUTLETS] with
    their end nodes and, for conduits, their length. Coordinates come from [COORDINATES] and [VERTICES], in the
    project's map units. With ``crs`` (an EPSG code or WKT), or a [MAP] in degrees, each node also gets ``lon``
    and ``lat`` and each link a ``lonlat`` line from its upstream node through its vertices to its downstream
    node.
    """
    _check_size(path)
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SwmmError(f"{p.name} could not be read ({exc}).") from exc
    sec = _sections(text)
    if not any(s in sec for s in (*_NODE_SECTIONS, *_LINK_SECTIONS)):
        raise SwmmError(f"{p.name} has no SWMM node or link sections; is it a SWMM .inp file?")

    title = " ".join(" ".join(r) for r in sec.get("TITLE", [])[:1])
    options = {r[0].upper(): " ".join(r[1:]) for r in sec.get("OPTIONS", []) if r}
    map_units = None
    for r in sec.get("MAP", []):
        if r[0].upper() == "UNITS" and len(r) > 1:
            map_units = r[1].lower()

    # A row whose coordinates are not numbers is skipped and counted, not a crash.
    coords: dict[str, tuple[float, float]] = {}
    vertices: dict[str, list[tuple[float, float]]] = {}
    unreadable = 0
    for section, into in (("COORDINATES", None), ("VERTICES", vertices)):
        for r in sec.get(section, []):
            if len(r) < 3:
                continue
            x, y = _float(r[1]), _float(r[2])
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                unreadable += 1
            elif into is None:
                coords[r[0]] = (x, y)
            else:
                into.setdefault(r[0], []).append((x, y))

    nodes: list[dict[str, Any]] = []
    for name, kind in _NODE_SECTIONS.items():
        for r in sec.get(name, []):
            # junctions and storage give the full depth second; outfalls and dividers have none in that place.
            depth = _float(r[2]) if kind in ("junction", "storage") and len(r) > 2 else None
            xy = coords.get(r[0])
            nodes.append({"id": r[0], "type": kind, "invert": _float(r[1]) if len(r) > 1 else None,
                          "max_depth": depth, "x": xy[0] if xy else None, "y": xy[1] if xy else None})
    links: list[dict[str, Any]] = []
    for name, kind in _LINK_SECTIONS.items():
        for r in sec.get(name, []):
            if len(r) < 3:
                continue
            length = _float(r[3]) if kind == "conduit" and len(r) > 3 else None
            line = None
            if r[1] in coords and r[2] in coords:
                line = [list(coords[r[1]]), *(list(v) for v in vertices.get(r[0], [])), list(coords[r[2]])]
            links.append({"id": r[0], "type": kind, "from": r[1], "to": r[2], "length": length, "xy": line})

    net = SwmmNetwork(name=p.name, title=title, options=options, map_units=map_units, nodes=nodes, links=links,
                      subcatchments=len(sec.get("SUBCATCHMENTS", [])))
    if unreadable:
        net.notes.append(f"Coordinate rows that are not numbers, skipped: {unreadable}.")
    missing = sum(1 for n in nodes if n["x"] is None)
    if missing:
        net.notes.append(f"Nodes without coordinates in [COORDINATES]: {missing} of {len(nodes)}.")

    given = None
    if crs is not None and str(crs).strip():
        from aquascope.io.mesh import _given_crs

        given = (_given_crs(crs), "given")
    elif map_units == "degrees":
        given = ("EPSG:4326", "map units in degrees")
    if given is None:
        net.notes.append("The file states no coordinate system; pass crs (an EPSG code) to place it on the globe.")
        return net

    xs = [n["x"] for n in nodes if n["x"] is not None]
    ys = [n["y"] for n in nodes if n["x"] is not None]
    for lk in links:
        for x, y in lk["xy"] or []:
            xs.append(x)
            ys.append(y)
    if not xs:
        return net
    lon, lat, info = _place(np.array(xs), np.array(ys), given[0], given[1])
    net.crs = info
    k = 0
    for n in nodes:
        if n["x"] is not None:
            if np.isfinite(lon[k]) and np.isfinite(lat[k]):
                n["lon"], n["lat"] = round(float(lon[k]), 7), round(float(lat[k]), 7)
            k += 1
    for lk in links:
        if lk["xy"]:
            m = len(lk["xy"])
            pts = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon[k:k + m], lat[k:k + m])
                   if np.isfinite(a) and np.isfinite(b)]
            if len(pts) >= 2:
                lk["lonlat"] = pts
            k += m
    return net


# ── the results (.out) ──────────────────────────────────────────────────────

class SwmmOutput:
    """A SWMM binary results file, opened lazily: the header, names and properties at once, the values one
    variable at a time (:meth:`values`, :meth:`peaks`). Use it as a context manager or call :meth:`close`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.size = _check_size(path)
        if self.size < 14 * 4:
            raise SwmmError(f"{self.path.name} is too short to be a SWMM results file.")
        self._fh = open(self.path, "rb")  # noqa: SIM115 - closed by close()
        try:
            self._read_header()
        except SwmmError:
            self.close()
            raise
        except (struct.error, UnicodeDecodeError, ValueError, IndexError, OSError) as exc:
            # A header that points past the end, or property tables of the wrong shape: a damaged file.
            self.close()
            raise SwmmError(f"{self.path.name} is not a complete SWMM results file ({exc}).") from exc

    def __enter__(self) -> SwmmOutput:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def _ints(self, n: int) -> tuple[int, ...]:
        return struct.unpack(f"<{n}i", self._fh.read(4 * n))

    def _read_header(self) -> None:
        fh = self._fh
        magic, version, flow, ns, nn, nl, npol = self._ints(7)
        if magic != MAGIC:
            raise SwmmError(f"{self.path.name} is not a SWMM results file (its first bytes are not SWMM's).")
        fh.seek(-6 * 4, 2)
        id_pos, prop_pos, out_pos, periods, error, magic2 = self._ints(6)
        if magic2 != MAGIC:
            raise SwmmError(f"{self.path.name} ends early: SWMM did not finish writing it (the run stopped?).")
        if not 0 <= flow < len(FLOW_UNITS):
            raise SwmmError(f"{self.path.name} names an unknown flow unit code ({flow}).")
        self.version = int(version)
        self.flow_units = FLOW_UNITS[flow]
        self.error_code = int(error)
        self.n_periods = int(periods)

        fh.seek(id_pos)

        def names(n: int) -> list[str]:
            out = []
            for _ in range(n):
                (k,) = self._ints(1)
                out.append(fh.read(k).decode("utf-8", "replace"))
            return out

        self.subcatchments = names(ns)
        self.nodes = names(nn)
        self.links = names(nl)
        self.pollutants = names(npol)

        fh.seek(prop_pos)
        (k,) = self._ints(1)
        self._ints(k)
        fh.read(4 * k * ns)
        (k,) = self._ints(1)
        self._ints(k)
        node_props = np.frombuffer(fh.read(4 * k * nn), dtype="<f4").reshape(nn, k) if nn else np.zeros((0, k))
        (k,) = self._ints(1)
        self._ints(k)
        link_props = np.frombuffer(fh.read(4 * k * nl), dtype="<f4").reshape(nl, k) if nl else np.zeros((0, k))
        self.node_types = [NODE_TYPES[int(t)] if 0 <= int(t) < len(NODE_TYPES) else "node" for t in node_props[:, 0]]
        self.link_types = [LINK_TYPES[int(t)] if 0 <= int(t) < len(LINK_TYPES) else "link" for t in link_props[:, 0]]

        counts = []
        for _ in range(4):
            (k,) = self._ints(1)
            self._ints(k)
            counts.append(k)
        self.n_subcatch_vars, self.n_node_vars, self.n_link_vars, self.n_system_vars = counts
        (start,) = struct.unpack("<d", fh.read(8))
        (step,) = self._ints(1)
        self.start = _from_days(start)
        self.report_step_s = int(step)
        self._out_pos = out_pos
        self._n_values = ns * counts[0] + nn * counts[1] + nl * counts[2] + counts[3]
        self._period_bytes = 8 + 4 * self._n_values
        if out_pos + self.n_periods * self._period_bytes > self.size:
            raise SwmmError(f"{self.path.name} holds fewer results than its header says; is it complete?")

    # ── metadata ──
    @property
    def version_label(self) -> str:
        return _version_label(self.version)

    @property
    def units(self) -> dict[str, str]:
        return units_of(self.flow_units)

    def time_at(self, period: int) -> str:
        """The time of one report period: the first is one report step after the report start, as SWMM writes."""
        return _iso(self.start + timedelta(seconds=self.report_step_s * (int(period) + 1)))

    def times(self) -> list[str]:
        """Every report time."""
        return [self.time_at(k) for k in range(self.n_periods)]

    def _column(self, kind: str, var: str | int) -> tuple[int, int, int]:
        """(offset of the first object, stride between objects, object count) for one variable."""
        if kind == "node":
            names, nv, base, n = NODE_VARS, self.n_node_vars, len(self.subcatchments) * self.n_subcatch_vars, \
                len(self.nodes)
        elif kind == "link":
            names, nv, n = LINK_VARS, self.n_link_vars, len(self.links)
            base = len(self.subcatchments) * self.n_subcatch_vars + len(self.nodes) * self.n_node_vars
        elif kind == "subcatchment":
            names, nv, base, n = SUBCATCH_VARS, self.n_subcatch_vars, 0, len(self.subcatchments)
        else:
            raise SwmmError(f"Unknown object kind {kind!r}: node, link or subcatchment.")
        idx = var if isinstance(var, int) else (names.index(var) if var in names else -1)
        if not 0 <= idx < nv:
            raise SwmmError(f"No {kind} variable {var!r}; the {kind} variables are {', '.join(names)}.")
        return base + idx, nv, n

    def _blocks(self, cols: np.ndarray):
        """Yield (first period, values[periods_in_block, len(cols)]) over the whole file."""
        per_block = max(1, BLOCK_VALUES // max(1, self._n_values))
        rec = np.dtype([("t", "<f8"), ("v", "<f4", (self._n_values,))])
        for first in range(0, self.n_periods, per_block):
            count = min(per_block, self.n_periods - first)
            self._fh.seek(self._out_pos + first * self._period_bytes)
            buf = self._fh.read(count * self._period_bytes)
            block = np.frombuffer(buf, dtype=rec, count=count)
            yield first, block["v"][:, cols]

    def _cols(self, kind: str, var: str | int, ids: list[str] | None) -> tuple[np.ndarray, list[str]]:
        start, stride, n = self._column(kind, var)
        names = {"node": self.nodes, "link": self.links, "subcatchment": self.subcatchments}[kind]
        if ids is None:
            pick = list(range(n))
        else:
            where = {name: i for i, name in enumerate(names)}
            missing = [i for i in ids if i not in where]
            if missing:
                raise SwmmError(f"No {kind} named {', '.join(map(str, missing[:5]))} in {self.path.name}.")
            pick = [where[i] for i in ids]
        return start + stride * np.asarray(pick, dtype=np.intp), [names[i] for i in pick]

    def values(self, kind: str, var: str | int, ids: list[str] | None = None) -> np.ndarray:
        """One variable over time: float32 array [periods, objects] for the objects named (all by default)."""
        cols, _names = self._cols(kind, var, ids)
        out = np.empty((self.n_periods, len(cols)), dtype="f4")
        for first, block in self._blocks(cols):
            out[first:first + len(block)] = block
        return out

    def peaks(self, kind: str, var: str | int, *, absolute: bool = False) -> dict[str, np.ndarray]:
        """Per object, scanning the file once: the largest value (largest magnitude with ``absolute``, sign
        kept), the period it happens in, and how many periods are above zero."""
        cols, _names = self._cols(kind, var, None)
        n = len(cols)
        best = np.full(n, -np.inf)
        signed = np.zeros(n)
        at = np.full(n, -1, dtype=np.intp)
        positive = np.zeros(n, dtype=np.intp)
        total = np.zeros(n)
        for first, block in self._blocks(cols):
            b = block.astype("f8")
            mag = np.abs(b) if absolute else b
            if not len(b):
                continue
            i = np.argmax(mag, axis=0)
            m = mag[i, np.arange(n)]
            better = m > best
            best[better] = m[better]
            signed[better] = b[i, np.arange(n)][better]
            at[better] = first + i[better]
            positive += (b > 0).sum(axis=0)
            total += np.where(b > 0, b, 0.0).sum(axis=0)
        return {"max": signed if n else best, "period": at, "positive_periods": positive, "positive_sum": total}


def read_swmm_out(path: str | Path) -> SwmmOutput:
    """Open a SWMM binary results file (``.out``) of SWMM 5.0 to 5.2."""
    return SwmmOutput(path)


# ── the summary for the MCP server and the CLI ──────────────────────────────

def _num(x: float | None, nd: int = 3) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def _where(net: SwmmNetwork | None, kind: str) -> dict[str, dict[str, Any]]:
    if net is None:
        return {}
    items = net.nodes if kind == "node" else net.links
    out = {}
    for it in items:
        if kind == "node" and it.get("lon") is not None:
            out[it["id"]] = {"lon": it["lon"], "lat": it["lat"]}
        elif kind == "link" and it.get("lonlat"):
            # the middle of the line: its middle vertex, or halfway along its middle segment
            pts = it["lonlat"]
            a, b = pts[(len(pts) - 1) // 2], pts[len(pts) // 2]
            out[it["id"]] = {"lon": round((a[0] + b[0]) / 2, 7), "lat": round((a[1] + b[1]) / 2, 7)}
    return out


def _flooding(o: SwmmOutput, net: SwmmNetwork | None, top: int) -> dict[str, Any]:
    pk = o.peaks("node", "flooding")
    step_h = o.report_step_s / 3600.0
    to_m3 = TO_M3S[o.flow_units] * o.report_step_s
    where = _where(net, "node")
    flooded = [i for i in range(len(o.nodes)) if pk["positive_periods"][i] > 0]
    flooded.sort(key=lambda i: -pk["max"][i])
    rows = []
    for i in flooded[:top]:
        row = {"id": o.nodes[i], "type": o.node_types[i], "peak": _num(pk["max"][i]),
               "peak_time": o.time_at(pk["period"][i]) if pk["period"][i] >= 0 else None,
               "hours_flooded": _num(pk["positive_periods"][i] * step_h, 2),
               "volume_m3": _num(pk["positive_sum"][i] * to_m3, 1)}
        row.update(where.get(o.nodes[i], {}))
        rows.append(row)
    return {"unit": o.flow_units, "nodes_flooded": len(flooded), "nodes": rows,
            "total_volume_m3": _num(sum(pk["positive_sum"][i] for i in flooded) * to_m3, 1)}


def _flows(o: SwmmOutput, net: SwmmNetwork | None, top: int) -> dict[str, Any]:
    pk = o.peaks("link", "flow", absolute=True)
    cap = o.peaks("link", "capacity")
    where = _where(net, "link")
    order = sorted(range(len(o.links)), key=lambda i: -abs(pk["max"][i]))
    rows = []
    for i in order[:top]:
        row = {"id": o.links[i], "type": o.link_types[i], "peak": _num(pk["max"][i]),
               "peak_time": o.time_at(pk["period"][i]) if pk["period"][i] >= 0 else None,
               "peak_capacity": _num(cap["max"][i], 3)}
        row.update(where.get(o.links[i], {}))
        rows.append(row)
    full = int(sum(1 for i in range(len(o.links)) if o.link_types[i] == "conduit" and cap["max"][i] >= 0.999))
    return {"unit": o.flow_units, "links": rows, "conduits_full": full}


def _series(o: SwmmOutput, element: str) -> dict[str, Any]:
    if element in o.nodes:
        kind, variables = "node", ("depth", "total_inflow", "flooding")
    elif element in o.links:
        kind, variables = "link", ("flow", "depth", "velocity")
    else:
        raise SwmmError(f"No node or link named {element!r} in {o.path.name}.")
    # A long run is thinned to at most SERIES_POINTS values per variable (every n-th report step, said so).
    every = max(1, -(-o.n_periods // SERIES_POINTS))
    out: dict[str, Any] = {"id": element, "kind": kind, "every": every,
                           "times": [o.time_at(k) for k in range(0, o.n_periods, every)], "units": {}}
    for v in variables:
        vals = o.values(kind, v, [element])[::every, 0]
        out[v] = [_num(x, 4) for x in vals]
        out["units"][v] = o.units[v]
    return out


def swmm_summary(inp: str | Path | None = None, out: str | Path | None = None, *, crs: str | int | None = None,
                 top: int = 5, element: str | None = None) -> dict[str, Any]:
    """What a SWMM model and its results hold, without their arrays: the network (nodes and links by type, the
    coordinate system and extent when placed), the report period, the nodes that flood (peak rate, its time,
    hours and volume from the report steps) and the links with the largest flows (peak, its time, and the
    largest fraction full). ``element`` adds one node's or link's time series (at most SERIES_POINTS values per
    variable; ``every`` says which report steps were kept). Every number is computed from the
    files; ok is false with a plain error for a file the readers cannot use."""
    if inp is None and out is None:
        return {"ok": False, "error": "Give a SWMM .inp file, its .out results file, or both."}
    top = max(1, min(int(top), 50))
    res: dict[str, Any] = {"ok": True, "notes": []}
    net = None
    try:
        if inp is not None:
            net = read_swmm_inp(inp, crs=crs)
            kinds: dict[str, int] = {}
            for n in net.nodes:
                kinds[n["type"]] = kinds.get(n["type"], 0) + 1
            lkinds: dict[str, int] = {}
            for lk in net.links:
                lkinds[lk["type"]] = lkinds.get(lk["type"], 0) + 1
            res["network"] = {"file": net.name, "title": net.title, "flow_units": net.flow_units,
                              "map_units": net.map_units, "nodes": len(net.nodes), "node_types": kinds,
                              "links": len(net.links), "link_types": lkinds, "subcatchments": net.subcatchments,
                              "crs": net.crs, "bbox": [round(v, 6) for v in net.bbox()] if net.bbox() else None}
            res["notes"].extend(net.notes)
        if out is not None:
            with read_swmm_out(out) as o:
                res["results"] = {"file": o.path.name, "swmm_version": o.version_label, "flow_units": o.flow_units,
                                  "units": o.units, "subcatchments": len(o.subcatchments), "nodes": len(o.nodes),
                                  "links": len(o.links), "pollutants": o.pollutants, "periods": o.n_periods,
                                  "report_step_s": o.report_step_s,
                                  "start": o.time_at(0) if o.n_periods else None,
                                  "end": o.time_at(o.n_periods - 1) if o.n_periods else None}
                if o.error_code:
                    res["notes"].append(f"SWMM reported error code {o.error_code} for this run.")
                if o.n_periods:
                    res["flooding"] = _flooding(o, net, top)
                    res["flows"] = _flows(o, net, top)
                    res["notes"].append("Hours flooded and flood volumes are summed over the report steps; the "
                                        "SWMM report (.rpt) sums over every routing step and is the reference.")
                if net is not None:
                    unknown = len(set(o.nodes) - {n["id"] for n in net.nodes})
                    if unknown:
                        res["notes"].append(f"{unknown} result nodes are not in {net.name}; are they the same "
                                            "model?")
                if element:
                    res["series"] = _series(o, element)
    except SwmmError as exc:
        return {"ok": False, "error": str(exc)}
    return res
