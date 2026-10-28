"""Flood it (#658): a 2D screening flood from a river reach, with the local-inertial scheme. Experimental (a spike).

The scheme is the local-inertial approximation of the shallow water equations used by LISFLOOD-FP (Bates,
Horritt and Fewtrell, 2010, J. Hydrol. 387, 33-45, doi:10.1016/j.jhydrol.2010.03.027), in the q-centred form of
de Almeida, Bates, Freer and Souvignet (2012, Water Resour. Res. 48, W05528, doi:10.1029/2011WR011570). On a
regular grid of cells of side ``dx``, depth ``h`` and bed ``z`` at cell centres and unit discharge ``q`` (m2/s) on
the faces, one step is:

* the flow depth at a face: ``hf = max(eta_l, eta_r) - max(z_l, z_r)`` with ``eta = z + h``;
* the face discharge (Manning friction, semi-implicit)::

      q_new = [theta q + (1 - theta)/2 (q_left + q_right) - g hf dt (eta_r - eta_l)/dx]
              / [1 + g dt n^2 |q| / hf^(7/3)]

  where ``q_left`` and ``q_right`` are the old discharges at the two neighbouring faces in the same direction
  (``theta = 1`` gives Bates et al. 2010); a face with ``hf`` at or below :data:`H_DRY` carries nothing;
* the depth: ``h_new = h + dt/dx (q_in - q_out)`` summed over the four faces, plus any inflow ``Q dt / dx^2``;
* the time step: ``dt = alpha dx / sqrt(g h_max)`` (the CFL condition of Bates et al. 2010, eq. 12).

The forms were checked against the equations quoted in Shaw et al. (2021, GMD 14, 3577, eqs. 18-19, the
LISFLOOD-FP 8.0 paper) and Cozzolino et al. (2019, Adv. Water Resour. 133, 103422, eqs. 46-48), which also give
the coefficients used here: ``alpha = 0.7`` (de Almeida et al. 2012) and ``theta = 0.7`` (de Almeida and Bates
2013).

Wet and dry: a face carries water only where ``hf > H_DRY``. Depth never goes negative: when a cell's outgoing
faces would take more than it holds in a step, they are scaled down so that it empties exactly (a flux limiter
that keeps the volume exact, rather than clipping negative depths, which would create water). The edges of the
grid are a free outfall: a ghost cell beyond the edge has the edge cell's bed and no water, and the edge faces only
let water out. The volume that leaves is counted, so the budget closes: final = initial + inflow - outflow.

What the scheme is not good at: supercritical flow. Without the advection term it under-estimates fast fronts
(de Almeida and Bates 2013; Cozzolino et al. 2019 show it fails for supercritical flow), so the Ritter dam break
below is a check with a stated tolerance, not an exact match. Floods on flood plains are slow, which is what the
scheme is for.

Terrain: the Terrarium tiles the Explorer's hillshade uses (``elevation-tiles-prod``, Mapzen/Tilezen joerd, a
mix of SRTM, GMTED, ETOPO1 and national lidar; decode ``h = R*256 + G + B/256 - 32768``). No bathymetry: the river
is burnt in (:func:`burn_channel`). This is a screening flood: terrain errors of metres, no levees unless the
terrain shows them or one is drawn (:func:`apply_levee`, :func:`apply_breach`, #663), no rain.

numpy and the standard library only (the PNG tiles are decoded with zlib), so it runs in Pyodide too.
"""

from __future__ import annotations

import math
import struct
import time
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "ALPHA", "CAP_HOURS", "G", "H_DRY", "N_CHANNEL", "N_FLOODPLAIN", "RETURN_PERIODS", "STEADY_TOL", "THETA",
    "TERRARIUM", "WALL_RULE", "WEIR_C", "annual_maxima", "apply_breach", "apply_levee", "burn_channel",
    "compare_exposure",
    "compare_runs", "design_flow", "edit_grid", "edits_from", "levee_profile", "read_line", "supercover_cells",
    "weir_flow",
    "cell_velocity", "channel_depth", "compare_depths", "corridor",
    "decode_png_rgb", "edge_codes", "flood_reach", "flow_curve", "flow_for", "flow_words", "forecast_series",
    "frame_times",
    "depth_over_terrain", "jrc_on_grid", "routed_series", "to_geotiff", "to_ugrid", "wall_contact", "wall_words",
    "matched_flow", "pixel_size_m", "plan_window", "prepare_grid", "reach_curve", "return_period_for", "ritter",
    "run_grid",
    "run_summary", "simulate", "stream_reaches", "terrain_grid", "terrain_window", "terrarium_elevation", "window",
]

G = 9.81
#: The CFL coefficient (de Almeida et al. 2012, as used by Cozzolino et al. 2019).
ALPHA = 0.7
#: The q-centred weight (de Almeida and Bates 2013, as used by Cozzolino et al. 2019); 1 is Bates et al. 2010.
THETA = 0.7
#: A face whose flow depth is at or below this (m) carries no water.
H_DRY = 1e-3
#: Manning's n for the burnt channel and the flood plain: screening defaults, not calibrated. Chow (1959) gives
#: about 0.025-0.04 for clean natural channels and 0.03-0.08 for pasture to light brush; JRC's maps are made with
#: their own values.
N_CHANNEL = 0.035
N_FLOODPLAIN = 0.06
#: The slope used to size the channel is kept in this range (the terrain's slope over a few km is noisy).
SLOPE_RANGE = (1e-4, 1e-2)

TERRARIUM = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
TERRAIN_CREDIT = ("Terrain: Terrarium tiles (Mapzen/Tilezen joerd on AWS Open Data: SRTM, GMTED2010, ETOPO1, "
                  "national lidar and others; see the joerd attribution list)")
EARTH_CIRCUMFERENCE_M = 40075016.686


# ── the solver ──────────────────────────────────────────────────────────────

#: Edge codes (:func:`edge_codes`): a wall, or a free outfall (a ghost cell with the edge cell's bed and no water).
#: A positive code is a normal-depth outfall with that bed slope: ``q = h^(5/3) S^(1/2) / n`` leaves the edge cell.
WALL = 0.0
FREE = math.nan


def edge_codes(ny: int, nx: int, closed: bool | str = False, edges: dict[str, Any] | None = None
               ) -> dict[str, np.ndarray]:
    """The code of every edge face, by side: ``"n"`` and ``"s"`` (``nx`` faces, west to east), ``"w"`` and ``"e"``
    (``ny`` faces, north to south). :data:`FREE` (NaN) is a free outfall, :data:`WALL` (0) a wall, a positive number
    a normal-depth outfall with that slope. ``closed=True`` walls every side and a string walls some (``"ns"``);
    ``edges`` gives sides (or single faces) explicitly and wins."""
    out = {"n": np.full(nx, FREE), "s": np.full(nx, FREE), "w": np.full(ny, FREE), "e": np.full(ny, FREE)}
    walls = "nsew" if closed is True else (closed or "")
    for side in walls:
        out[side][:] = WALL
    for side, val in (edges or {}).items():
        out[side][:] = np.broadcast_to(np.asarray(val, dtype="f8"), out[side].shape)
    return out


def _apply_edge(q: np.ndarray, h: np.ndarray, n: np.ndarray, code: np.ndarray, sign: float) -> np.ndarray:
    """Edge faces after their code: a free outfall keeps the ghost-cell discharge (outflow only), a wall carries
    nothing, a normal-depth outfall carries ``sign * h^(5/3) sqrt(S) / n`` from the edge cell."""
    s = np.where(code > 0, code, 0.0)
    nd = sign * np.where(h > H_DRY, np.maximum(h, 0.0) ** (5.0 / 3.0) * np.sqrt(s) / np.maximum(n, 1e-6), 0.0)
    return np.where(np.isnan(code), q, np.where(code > 0, nd, 0.0))


def _faces(z: np.ndarray, eta: np.ndarray, n: np.ndarray, q: np.ndarray, dt: float, dx: float, theta: float,
           axis: int) -> np.ndarray:
    """New discharges on every face along ``axis`` (1: x, between columns; 0: y, between rows), edges included.

    ``q`` has one more entry than the cells along ``axis``. Ghost cells beyond the edges have the edge cell's
    bed and no water, so the edge faces drain the grid; they are clipped to outflow only.
    """
    zl = np.concatenate([np.take(z, [0], axis=axis), z], axis=axis)
    zr = np.concatenate([z, np.take(z, [-1], axis=axis)], axis=axis)
    el = np.concatenate([np.take(z, [0], axis=axis), eta], axis=axis)
    er = np.concatenate([eta, np.take(z, [-1], axis=axis)], axis=axis)
    nl = np.concatenate([np.take(n, [0], axis=axis), n], axis=axis)
    nr = np.concatenate([n, np.take(n, [-1], axis=axis)], axis=axis)
    hf = np.maximum(el, er) - np.maximum(zl, zr)
    # The q-centred average: the old discharge here and at the two faces either side (the edge faces use their own).
    lo = np.concatenate([np.take(q, [0], axis=axis), np.take(q, range(q.shape[axis] - 1), axis=axis)], axis=axis)
    hi = np.concatenate([np.take(q, range(1, q.shape[axis]), axis=axis), np.take(q, [-1], axis=axis)], axis=axis)
    qc = theta * q + 0.5 * (1.0 - theta) * (lo + hi)
    nf = 0.5 * (nl + nr)
    wet = hf > H_DRY
    hs = np.where(wet, hf, 1.0)
    out = (qc - G * hs * dt * (er - el) / dx) / (1.0 + G * dt * nf * nf * np.abs(q) / hs ** (7.0 / 3.0))
    out = np.where(wet, out, 0.0)
    # Edge faces only let water out of the grid.
    first = [slice(None)] * 2
    last = [slice(None)] * 2
    first[axis] = 0
    last[axis] = -1
    out[tuple(first)] = np.minimum(out[tuple(first)], 0.0)
    out[tuple(last)] = np.maximum(out[tuple(last)], 0.0)
    return out


def _limit(h: np.ndarray, qx: np.ndarray, qy: np.ndarray, dt: float, dx: float) -> None:
    """Scale each cell's outgoing faces so that the cell cannot go below zero in this step (in place)."""
    out = (np.maximum(qx[:, 1:], 0) + np.maximum(-qx[:, :-1], 0)
           + np.maximum(qy[1:, :], 0) + np.maximum(-qy[:-1, :], 0)) * dt / dx
    f = np.where(out > h, h / np.where(out > 0, out, 1.0), 1.0)
    # Each face takes the factor of the cell its water leaves (its donor).
    fx_l = np.concatenate([f[:, :1], f], axis=1)     # the cell left of each x face (the edge: the edge cell)
    fx_r = np.concatenate([f, f[:, -1:]], axis=1)
    qx *= np.where(qx > 0, fx_l, fx_r)
    fy_t = np.concatenate([f[:1, :], f], axis=0)
    fy_b = np.concatenate([f, f[-1:, :]], axis=0)
    qy *= np.where(qy > 0, fy_t, fy_b)


def flow_at(q: Any, t: float, table_dt_s: float) -> float:
    """A source's flow at time ``t`` (s): a number, or a table of flows every ``table_dt_s`` from ``t = 0``, read
    linearly between its entries and held at the ends."""
    if np.ndim(q) == 0:
        return float(q)
    tab = np.asarray(q, dtype="f8")
    k = t / table_dt_s
    if k <= 0:
        return float(tab[0])
    if k >= len(tab) - 1:
        return float(tab[-1])
    i = int(math.floor(k))
    return float(tab[i] + (tab[i + 1] - tab[i]) * (k - i))


def cell_velocity(h: np.ndarray, qx: np.ndarray, qy: np.ndarray, dry: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """The eastward and northward velocity (m/s) at the cell centres: the mean unit discharge of a cell's two faces
    in each direction over its depth (rows run north to south, so a positive y discharge flows south). NaN where the
    cell is shallower than ``dry``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        u = 0.5 * (qx[:, :-1] + qx[:, 1:]) / h
        v = -0.5 * (qy[:-1, :] + qy[1:, :]) / h
    wet = h > dry
    return np.where(wet, u, np.nan), np.where(wet, v, np.nan)


def simulate(z: Any, dx: float, *, duration_s: float, h0: Any = None, n: Any = N_FLOODPLAIN,
             sources: list[tuple[int, int, Any]] | None = None, frames_every_s: Any = None,
             theta: float = THETA, alpha: float = ALPHA, dt_max: float = 10.0, closed: bool | str = False,
             edges: dict[str, Any] | None = None, table_dt_s: float = 3600.0, steady_tol: float | None = None,
             steady_every_s: float = 3600.0, steady_min_s: float = 3600.0, wet_m: float = 0.05,
             frames_uv: bool = False, max_steps: int = 5_000_000) -> dict[str, Any]:
    """Run the local-inertial scheme on bed ``z`` (m, rows north to south) with cells of side ``dx`` (m).

    ``sources`` are ``(row, col, flow)`` inflows: a flow in m3/s, or a table of flows every ``table_dt_s`` seconds
    from the start (:func:`flow_at`). The edges are a free outfall unless ``closed`` walls them (``True``, or a
    string of sides such as ``"ns"``) or ``edges`` codes them (:func:`edge_codes`: walls, free or normal-depth
    outfalls, side by side or face by face). ``frames_every_s`` is a step (s) or a list of times for the depth
    frames (and with ``frames_uv`` the velocities, :func:`cell_velocity`).

    ``steady_tol`` stops the run at a near steady state: at every ``steady_every_s`` of model time from
    ``steady_min_s`` on, when both the wet area (deeper than ``wet_m``) and the volume on the grid changed by less
    than ``steady_tol`` (a share) since the last check. ``duration_s`` is then the cap. ``steady_at_s`` says when
    (None when the cap came first), with the area and volume at each check in ``checks``.

    Returns the final depth ``h``, the deepest depth of the run ``h_max``, the time each cell first got wetter
    than 0.1 m (``arrival_s``, NaN if never), the frames with their ``times_s``, the step count, the wall time and
    the volume budget.
    """
    z = np.asarray(z, dtype="f8")
    if z.ndim != 2:
        raise ValueError("z must be a 2D grid")
    ny, nx = z.shape
    h = np.zeros_like(z) if h0 is None else np.array(h0, dtype="f8")
    nn = np.broadcast_to(np.asarray(n, dtype="f8"), z.shape)
    qx = np.zeros((ny, nx + 1))
    qy = np.zeros((ny + 1, nx))
    src = [(int(r), int(c), q if np.ndim(q) else float(q)) for r, c, q in (sources or [])]
    codes = edge_codes(ny, nx, closed, edges)
    h_max = h.copy()
    arrival = np.full(z.shape, np.nan)
    v0 = float(h.sum()) * dx * dx
    v_in = v_out = 0.0
    frames: list[np.ndarray] = []
    uv: list[tuple[np.ndarray, np.ndarray]] = []
    times: list[float] = []
    if frames_every_s is None:
        frame_at: list[float] = []
    elif np.ndim(frames_every_s):
        frame_at = sorted(float(x) for x in frames_every_s)
    else:
        step_s = float(frames_every_s)
        frame_at = [k * step_s for k in range(int(duration_s // step_s) + 1)]
    fi = 0
    check_every = float(steady_every_s)
    next_check = max(check_every, steady_min_s) if steady_tol else math.inf
    checks: list[dict[str, float]] = []
    steady_at = None
    last = None
    t = 0.0
    steps = 0
    started = time.perf_counter()

    def area_volume() -> tuple[float, float]:
        return float((h > wet_m).sum()) * dx * dx, float(h.sum()) * dx * dx

    def take() -> None:
        frames.append(h.astype("f4"))
        if frames_uv:
            u, v = cell_velocity(h, qx, qy)
            uv.append((u.astype("f4"), v.astype("f4")))
        times.append(t)

    if steady_tol:
        last = area_volume()
    while t < duration_s - 1e-9 and steps < max_steps:
        while fi < len(frame_at) and t >= frame_at[fi] - 1e-9:
            take()
            fi += 1
        hm = float(h.max())
        dt = min(dt_max, duration_s - t, alpha * dx / math.sqrt(G * hm) if hm > H_DRY else dt_max)
        if fi < len(frame_at):
            dt = min(dt, max(frame_at[fi] - t, 1e-6))
        if next_check < math.inf:
            dt = min(dt, max(next_check - t, 1e-6))
        eta = z + h
        qx = _faces(z, eta, nn, qx, dt, dx, theta, axis=1)
        qy = _faces(z, eta, nn, qy, dt, dx, theta, axis=0)
        qx[:, 0] = _apply_edge(qx[:, 0], h[:, 0], nn[:, 0], codes["w"], -1.0)
        qx[:, -1] = _apply_edge(qx[:, -1], h[:, -1], nn[:, -1], codes["e"], 1.0)
        qy[0, :] = _apply_edge(qy[0, :], h[0, :], nn[0, :], codes["n"], -1.0)
        qy[-1, :] = _apply_edge(qy[-1, :], h[-1, :], nn[-1, :], codes["s"], 1.0)
        _limit(h, qx, qy, dt, dx)
        v_out += float((qx[:, -1].sum() - qx[:, 0].sum() + qy[-1, :].sum() - qy[0, :].sum()) * dx * dt)
        h = h + dt / dx * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :])
        h = np.maximum(h, 0.0)   # only round-off can go below zero after the limiter
        for r, c, q in src:
            qq = flow_at(q, t, table_dt_s)
            h[r, c] += qq * dt / (dx * dx)
            v_in += qq * dt
        t += dt
        steps += 1
        np.maximum(h_max, h, out=h_max)
        newly = np.isnan(arrival) & (h > 0.1)
        arrival[newly] = t
        if t >= next_check - 1e-9:
            a, v = area_volume()
            da = abs(a - last[0]) / max(a, dx * dx)
            dv = abs(v - last[1]) / max(v, 1e-9)
            checks.append({"t_s": t, "wet_m2": a, "volume_m3": v, "d_area": da, "d_volume": dv})
            last = (a, v)
            next_check += check_every
            if da < steady_tol and dv < steady_tol:
                steady_at = t
                break
    while fi < len(frame_at) and frame_at[fi] <= t + 1e-9:
        if not times or times[-1] < t - 1e-9:
            take()
        fi += 1
    if frame_at and (not times or times[-1] < t - 1e-9):
        take()
    wall = time.perf_counter() - started
    v1 = float(h.sum()) * dx * dx
    budget_err = v1 - (v0 + v_in - v_out)
    scale = max(v0 + v_in, 1e-12)
    return {
        "h": h, "h_max": h_max, "arrival_s": arrival, "frames": frames, "uv": uv, "times_s": times, "t_s": t,
        "steps": steps, "wall_s": wall, "steps_per_s": steps / wall if wall > 0 else None,
        "steady_at_s": steady_at, "checks": checks, "qx": qx, "qy": qy,
        "volume": {"initial_m3": v0, "inflow_m3": v_in, "outflow_m3": v_out, "final_m3": v1,
                   "error_m3": budget_err, "error_rel": budget_err / scale},
    }


# ── the analytic check ──────────────────────────────────────────────────────


def ritter(x: Any, t: float, h0: float, x0: float = 0.0) -> np.ndarray:
    """Ritter's (1892) dam break on a dry, flat, frictionless bed: the depth at ``x`` a time ``t`` after a dam at
    ``x0`` holding ``h0`` upstream (x < x0) vanishes. ``c0 = sqrt(g h0)``; the depth is ``h0`` for
    ``x - x0 <= -c0 t``, ``(2 c0 - (x - x0)/t)^2 / (9 g)`` up to the front at ``x - x0 = 2 c0 t``, and zero beyond."""
    x = np.asarray(x, dtype="f8")
    c0 = math.sqrt(G * h0)
    xi = (x - x0) / t
    h = (2 * c0 - xi) ** 2 / (9 * G)
    return np.where(xi <= -c0, h0, np.where(xi >= 2 * c0, 0.0, h))


# ── the terrain ─────────────────────────────────────────────────────────────


def decode_png_rgb(data: bytes) -> np.ndarray:
    """An 8-bit RGB or RGBA, non-interlaced PNG (what the Terrarium tiles are) as a (rows, cols, 3) uint8 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    i = 8
    idat = b""
    width = height = channels = 0
    while i < len(data):
        (length,) = struct.unpack(">I", data[i:i + 4])
        kind = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or colour not in (2, 6) or interlace:
                raise ValueError("only 8-bit RGB(A) non-interlaced PNGs are read")
            channels = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        i += 12 + length
    raw = np.frombuffer(zlib.decompress(idat), dtype="u1")
    stride = width * channels
    rows = raw.reshape(height, stride + 1)
    out = np.zeros((height, stride), dtype="i4")
    prev = np.zeros(stride, dtype="i4")
    for r in range(height):
        ftype = rows[r, 0]
        line = rows[r, 1:].astype("i4")
        if ftype == 0:
            cur = line
        elif ftype == 2:
            cur = (line + prev) & 255
        else:
            cur = np.zeros(stride, dtype="i4")
            for c in range(stride):
                a = cur[c - channels] if c >= channels else 0
                b = prev[c]
                if ftype == 1:
                    p = a
                elif ftype == 3:
                    p = (a + b) >> 1
                else:   # 4, Paeth
                    cc = prev[c - channels] if c >= channels else 0
                    pa, pb, pc = abs(b - cc), abs(a - cc), abs(a + b - 2 * cc)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else cc)
                cur[c] = (line[c] + p) & 255
        out[r] = cur
        prev = cur
    return out.reshape(height, width, channels)[:, :, :3].astype("u1")


def terrarium_elevation(rgb: np.ndarray) -> np.ndarray:
    """Terrarium encoding to metres: ``R*256 + G + B/256 - 32768``."""
    rgb = np.asarray(rgb, dtype="f8")
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def pixel_size_m(lat: float, zoom: int) -> float:
    """Ground size (m) of a 256-pixel Web Mercator tile's pixel at ``lat`` and ``zoom``."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (256 * 2 ** zoom)


def _world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    s = 256 * 2 ** zoom
    x = (lon + 180.0) / 360.0 * s
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * s
    return x, y


def _px_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    s = 256 * 2 ** zoom
    lon = x / s * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / s))))
    return lon, lat


def _http_get(url: str) -> bytes:
    import httpx

    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


def terrain_window(x0: int, y0: int, nx: int, ny: int, zoom: int,
                   fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """The Terrarium terrain of a window of ``nx`` x ``ny`` pixels at ``zoom`` whose top-left world pixel is
    (``x0``, ``y0``): ``z`` (m, rows north to south), ``dx`` (m, the pixel's ground size at the window's middle
    latitude), the window itself and its ``bounds`` [W, S, E, N]."""
    get = fetch or _http_get
    tx0, ty0 = x0 // 256, y0 // 256
    tx1, ty1 = (x0 + nx - 1) // 256, (y0 + ny - 1) // 256
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256))
    n = 2 ** zoom
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            rgb = decode_png_rgb(get(TERRARIUM.format(z=zoom, x=tx % n, y=ty)))
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = \
                terrarium_elevation(rgb)
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    z = mosaic[oy:oy + ny, ox:ox + nx]
    return {"z": z, **window(x0, y0, nx, ny, zoom)}


def window(x0: int, y0: int, nx: int, ny: int, zoom: int) -> dict[str, Any]:
    """A grid's geometry without its terrain: the world-pixel origin, the size, the cell size at the middle
    latitude and the bounds [W, S, E, N]."""
    w, n_ = _px_lonlat(x0, y0, zoom)
    e, s = _px_lonlat(x0 + nx, y0 + ny, zoom)
    mid = _px_lonlat(x0 + nx / 2, y0 + ny / 2, zoom)[1]
    return {"dx": pixel_size_m(mid, zoom), "x0": int(x0), "y0": int(y0), "zoom": int(zoom), "nx": int(nx),
            "ny": int(ny), "size": int(max(nx, ny)), "bounds": [w, s, e, n_]}


def terrain_grid(lon: float, lat: float, *, zoom: int = 12, size: int = 512,
                 fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """A ``size`` x ``size`` window of Terrarium terrain centred on (lon, lat), at ``zoom``'s pixel size."""
    cx, cy = _world_px(lon, lat, zoom)
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    g = terrain_window(x0, y0, size, size, zoom, fetch=fetch)
    g["dx"] = pixel_size_m(lat, zoom)
    return g


def cell_of(grid: dict[str, Any], lon: float, lat: float) -> tuple[int, int]:
    """The (row, col) of the grid cell holding (lon, lat)."""
    x, y = _world_px(lon, lat, grid["zoom"])
    return int(math.floor(y - grid["y0"])), int(math.floor(x - grid["x0"]))


def cell_lonlat(grid: dict[str, Any], row: Any, col: Any) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre longitudes and latitudes for arrays of rows and columns."""
    s = 256 * 2 ** grid["zoom"]
    x = grid["x0"] + np.asarray(col, dtype="f8") + 0.5
    y = grid["y0"] + np.asarray(row, dtype="f8") + 0.5
    lon = x / s * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / s))))
    return lon, lat


# ── the channel ─────────────────────────────────────────────────────────────


def channel_depth(q_bankfull: float, width_m: float, slope: float, n: float = N_CHANNEL) -> float:
    """The depth (m) at which a wide rectangular channel of ``width_m`` carries ``q_bankfull`` (m3/s) at normal
    depth by Manning's equation, ``Q = (1/n) W d^(5/3) S^(1/2)``, so ``d = (n Q / (W S^(1/2)))^(3/5)``. The slope
    is kept within :data:`SLOPE_RANGE`."""
    s = min(max(float(slope), SLOPE_RANGE[0]), SLOPE_RANGE[1])
    return (n * float(q_bankfull) / (float(width_m) * math.sqrt(s))) ** 0.6


def _line_cells(grid: dict[str, Any], line: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """The cells a (lon, lat) polyline passes through, in order, without repeats (sampled every half cell)."""
    zoom = grid["zoom"]
    pts = [_world_px(lo, la, zoom) for lo, la in line]
    out: list[tuple[int, int]] = []
    nx, ny = grid.get("nx", grid["size"]), grid.get("ny", grid["size"])
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        k = max(1, int(math.ceil(2 * math.hypot(bx - ax, by - ay))))
        for i in range(k + 1):
            x = ax + (bx - ax) * i / k - grid["x0"]
            y = ay + (by - ay) * i / k - grid["y0"]
            r, c = int(math.floor(y)), int(math.floor(x))
            if 0 <= r < ny and 0 <= c < nx and (not out or out[-1] != (r, c)):
                if (r, c) not in out[-3:]:
                    out.append((r, c))
    return out


#: The most a channel is cut below the terrain beyond its own depth (m). Where the drawn line runs over ground
#: higher than that above the river upstream, it is off the valley there, and is not trenched through.
MAX_CUT_M = 10.0


def burn_channel(z: np.ndarray, cells: list[tuple[int, int]], depth_m: float, into: np.ndarray | None = None,
                 max_cut_m: float | None = None) -> np.ndarray:
    """``z`` with a channel ``depth_m`` deep along ``cells`` (ordered upstream to downstream): a copy, or written
    into ``into`` (where another channel may already be burnt; the deeper bed wins).

    The bank line is the terrain ``z`` along the cells made non-increasing downstream (a running minimum), so the
    burnt bed has no pits or bumps from the terrain's noise; the bed is that line minus ``depth_m``, but never
    more than ``max_cut_m`` below the terrain plus the depth (None: no limit)."""
    out = np.array(z, dtype="f8") if into is None else into
    if not cells:
        return out
    bank = np.minimum.accumulate(np.array([z[r, c] for r, c in cells], dtype="f8"))
    for (r, c), b in zip(cells, bank):
        bed = b - depth_m
        if max_cut_m is not None:
            bed = max(bed, z[r, c] - depth_m - max_cut_m)
        out[r, c] = min(out[r, c], bed)
    return out


def _orient(z: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cells ordered so that the higher end comes first (upstream)."""
    if len(cells) > 1 and z[cells[0]] < z[cells[-1]]:
        return cells[::-1]
    return cells


# ── levees and breaches (#663) ──────────────────────────────────────────────

#: The broad-crested weir coefficient in SI units for critical depth over the crest: ``Q = C b H^1.5`` with
#: ``C = (2/3)^(3/2) g^(1/2)`` (about 1.705 m^0.5/s), the ideal weir of the open-channel texts (Henderson 1966,
#: *Open Channel Flow*).
WEIR_C = (2.0 / 3.0) ** 1.5 * math.sqrt(G)
#: The freeboard (m) a suggested crest keeps above the highest water along a drawn line.
FREEBOARD_M = 0.5


def _cell_xy(win: dict[str, Any], line: Any) -> list[tuple[float, float]]:
    """A (lon, lat) polyline in the window's cell coordinates: x along the columns, y down the rows."""
    pts = []
    for p in line:
        lon, lat = float(p[0]), float(p[1])
        if not (math.isfinite(lon) and math.isfinite(lat)) or abs(lat) >= 85.06:
            raise ValueError("a line's points must be finite longitudes and latitudes inside Web Mercator")
        x, y = _world_px(lon, lat, win["zoom"])
        pts.append((x - win["x0"], y - win["y0"]))
    if len(pts) < 2:
        raise ValueError("a line needs at least two points")
    return pts


def _supercover(ax: float, ay: float, bx: float, by: float) -> list[tuple[int, int]]:
    """Every cell (row, col) the segment from (ax, ay) to (bx, by) touches, in order: Amanatides and Woo's (1987)
    traversal, taking both side cells where the segment passes exactly through a cell corner."""
    c, r = math.floor(ax), math.floor(ay)
    c1, r1 = math.floor(bx), math.floor(by)
    dx, dy = bx - ax, by - ay
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    tx = ((c + (sx > 0)) - ax) / dx if dx else math.inf
    ty = ((r + (sy > 0)) - ay) / dy if dy else math.inf
    ddx = abs(1.0 / dx) if dx else math.inf
    ddy = abs(1.0 / dy) if dy else math.inf
    out = [(r, c)]
    for _ in range(abs(c1 - c) + abs(r1 - r) + 2):
        if (r, c) == (r1, c1) or min(tx, ty) > 1.0:
            break
        if abs(tx - ty) < 1e-12:
            out += [(r, c + sx), (r + sy, c)]
            c, r, tx, ty = c + sx, r + sy, tx + ddx, ty + ddy
        elif tx < ty:
            c, tx = c + sx, tx + ddx
        else:
            r, ty = r + sy, ty + ddy
        out.append((r, c))
    return out


def supercover_cells(grid: dict[str, Any], line: Any) -> list[tuple[int, int]]:
    """Every grid cell a (lon, lat) polyline touches, in order along it and without repeats: a supercover
    rasterisation, so consecutive cells share a face and a line of raised cells has no gap at a corner that water
    could leak through. ``grid`` is a :func:`prepare_grid` answer or a :func:`window`. Cells off the grid are left
    out."""
    win = grid.get("window", grid)
    pts = _cell_xy(win, line)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        for rc in _supercover(ax, ay, bx, by):
            if rc not in seen and 0 <= rc[0] < win["ny"] and 0 <= rc[1] < win["nx"]:
                seen.add(rc)
                out.append(rc)
    return out


def _natural(grid: dict[str, Any]) -> dict[str, np.ndarray]:
    nat = grid.get("natural")
    return nat if nat else {"z": np.asarray(grid["z"], dtype="f8"), "z_burnt": np.asarray(grid["z_burnt"], dtype="f8")}


def _line_m(win: dict[str, Any], line: Any) -> float:
    pts = [(float(p[0]), float(p[1])) for p in line]
    return sum(_haversine_km(a[1], a[0], b[1], b[0]) for a, b in zip(pts, pts[1:])) * 1000.0


def apply_levee(grid: dict[str, Any], line: Any, crest: float) -> dict[str, Any]:
    """A levee along ``line`` (a list of [lon, lat]): every cell the line crosses (:func:`supercover_cells`) is
    raised to ``crest``.

    ``crest`` is an **elevation** in metres on the terrain's own datum (the Terrarium tiles: metres above mean sea
    level), not a height above the ground: a levee's crest is level along it while the ground under it is not. A
    cell already higher than the crest is left as it is. A levee drawn across a river dams it: the cells of a burnt
    channel under the line are raised too, and counted in ``channel_cells``.

    Returns the new terrain the run floods (``terrain``, the grid's ``z_burnt`` with the levee), the new terrain
    the map and the depths are drawn from (``ground``), the edited cells (``cells``, [row, col] in order along the
    line) and what changed. The grid itself is not changed; the same line and crest always give the same cells."""
    crest = float(crest)
    if not math.isfinite(crest):
        raise ValueError("a levee's crest must be a number (m above mean sea level)")
    zb = np.array(grid["z_burnt"], dtype="f8")
    zg = np.array(grid["z"], dtype="f8")
    cells = supercover_cells(grid, line)
    if not cells:
        raise ValueError("the levee's line does not cross the grid")
    rr = np.array([r for r, _ in cells], dtype=np.intp)
    cc = np.array([c for _, c in cells], dtype=np.intp)
    under = zg[rr, cc].copy()
    channel = int(np.count_nonzero(zb[rr, cc] < zg[rr, cc] - 1e-9))
    raised = zb[rr, cc] < crest
    zb[rr, cc] = np.maximum(zb[rr, cc], crest)
    zg[rr, cc] = np.maximum(zg[rr, cc], crest)
    return {"kind": "levee", "terrain": zb, "ground": zg, "cells": [[int(r), int(c)] for r, c in cells],
            "crest_m": round(crest, 3), "raised": int(raised.sum()), "channel_cells": channel,
            "ground_m": [round(float(under.min()), 2), round(float(under.max()), 2)],
            "height_m": [round(float(max(crest - under.max(), 0.0)), 2),
                         round(float(max(crest - under.min(), 0.0)), 2)],
            "length_m": round(_line_m(grid.get("window", grid), line), 1)}


def apply_breach(grid: dict[str, Any], line: Any, invert: float | None = None) -> dict[str, Any]:
    """A breach along ``line``: every cell the line crosses (:func:`supercover_cells`) is lowered to ``invert``,
    an elevation in metres on the terrain's datum, as a levee's crest. A cell already lower is left as it is.
    Without an ``invert`` each cell goes back to its natural ground (the grid's terrain before any levee: a gap in
    a drawn levee). Draw a breach along the levee over the stretch that fails; its width is the cells it crosses.

    Returns ``terrain``, ``ground`` and ``cells`` as :func:`apply_levee` does, the cells lowered and the gap's
    width along the line."""
    zb = np.array(grid["z_burnt"], dtype="f8")
    zg = np.array(grid["z"], dtype="f8")
    nat = _natural(grid)
    cells = supercover_cells(grid, line)
    if not cells:
        raise ValueError("the breach's line does not cross the grid")
    rr = np.array([r for r, _ in cells], dtype=np.intp)
    cc = np.array([c for _, c in cells], dtype=np.intp)
    if invert is None:
        tb, tg = nat["z_burnt"][rr, cc], nat["z"][rr, cc]
    else:
        if not math.isfinite(float(invert)):
            raise ValueError("a breach's invert must be a number (m above mean sea level)")
        tb = tg = np.full(rr.size, float(invert))
    lowered = zb[rr, cc] > tb + 1e-9
    zb[rr, cc] = np.minimum(zb[rr, cc], tb)
    zg[rr, cc] = np.minimum(zg[rr, cc], tg)
    return {"kind": "breach", "terrain": zb, "ground": zg, "cells": [[int(r), int(c)] for r, c in cells],
            "invert_m": None if invert is None else round(float(invert), 3), "lowered": int(lowered.sum()),
            "floor_m": [round(float(zg[rr, cc].min()), 2), round(float(zg[rr, cc].max()), 2)],
            "length_m": round(_line_m(grid.get("window", grid), line), 1)}


def read_line(geo: Any) -> list[list[list[float]]]:
    """Lines from GeoJSON (a LineString, a MultiLineString, a Feature or a FeatureCollection of them, as a dict or
    text) or a plain list of [lon, lat]: a list of lines, each a list of [lon, lat]."""
    import json

    if isinstance(geo, (str, bytes)):
        geo = json.loads(geo)
    if isinstance(geo, list):
        if geo and isinstance(geo[0], (list, tuple)) and geo[0] and isinstance(geo[0][0], (int, float)):
            return [[[float(p[0]), float(p[1])] for p in geo]]
        return [ln for part in geo for ln in read_line(part)]
    if not isinstance(geo, dict):
        raise ValueError("a line must be GeoJSON or a list of [lon, lat]")
    kind = geo.get("type")
    if kind == "FeatureCollection":
        return [ln for f in geo.get("features") or [] for ln in read_line(f)]
    if kind == "Feature":
        return read_line(geo.get("geometry") or {})
    if kind == "LineString":
        return [[[float(p[0]), float(p[1])] for p in geo["coordinates"]]]
    if kind == "MultiLineString":
        return [[[float(p[0]), float(p[1])] for p in ln] for ln in geo["coordinates"]]
    raise ValueError(f"a levee or a breach is a line, not a {kind}")


def edits_from(levees: Any = (), crests: Any = (), breaches: Any = (), inverts: Any = ()) -> list[dict[str, Any]]:
    """The edits of :func:`edit_grid` from the CLI's and the MCP tool's arguments: ``levees`` and ``breaches`` are
    each a list of lines (GeoJSON, as text or a dict, or a list of [lon, lat]: :func:`read_line`, so one entry may
    hold several lines); ``crests`` is one crest for every levee or one per entry (m above mean sea level), and
    ``inverts`` likewise for the breaches (none: the natural ground). Levees first, then breaches."""
    def many(x: Any) -> list[Any]:
        if x is None:
            return []
        if isinstance(x, (str, dict)) or (isinstance(x, list) and x and isinstance(x[0], (list, tuple))
                                          and x[0] and isinstance(x[0][0], (int, float))):
            return [x]
        return list(x)

    def pick(vals: list[Any], k: int, what: str) -> Any:
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        if k >= len(vals):
            raise ValueError(f"give one {what} for all, or one for each")
        return vals[k]

    lv, br = many(levees), many(breaches)
    cr = [float(c) for c in many(crests) if c is not None] if not isinstance(crests, (int, float)) else [float(crests)]
    iv = [float(c) for c in many(inverts) if c is not None] if not isinstance(inverts, (int, float)) \
        else [float(inverts)]
    out: list[dict[str, Any]] = []
    for k, geo in enumerate(lv):
        crest = pick(cr, k, "crest")
        if crest is None:
            raise ValueError("a levee needs a crest: an elevation in m above mean sea level")
        out += [{"kind": "levee", "line": ln, "crest": crest} for ln in read_line(geo)]
    for k, geo in enumerate(br):
        out += [{"kind": "breach", "line": ln, "invert": pick(iv, k, "invert")} for ln in read_line(geo)]
    return out


def edit_grid(grid: dict[str, Any], edits: list[dict[str, Any]], *, skip_outside: bool = False) -> dict[str, Any]:
    """A :func:`prepare_grid` answer with levees and breaches applied in the order given: each edit is
    ``{"kind": "levee", "line": [[lon, lat], ...], "crest": m}`` or ``{"kind": "breach", "line": [...],
    "invert": m}`` (the invert may be left out: the natural ground). The answer is a new grid (the arrays copied,
    the input unchanged) with ``natural`` (the terrain before any edit) and ``edits`` (each edit's cells and
    numbers). The same edits always give the same grid. With ``skip_outside`` an edit whose line misses the grid
    is listed with ``skipped`` instead of stopping (the Explorer keeps drawn lines when another river is
    flooded)."""
    nat = _natural(grid)
    g = {**grid, "natural": nat, "z": np.array(nat["z"]), "z_burnt": np.array(nat["z_burnt"]), "edits": []}
    for k, e in enumerate(edits or []):
        kind = e.get("kind")
        if kind not in ("levee", "breach"):
            raise ValueError(f"edit {k + 1}: kind must be levee or breach, not {kind!r}")
        if kind == "levee" and e.get("crest") is None:
            raise ValueError(f"edit {k + 1}: a levee needs a crest (m above mean sea level)")
        if skip_outside and not supercover_cells(g, e["line"]):
            g["edits"].append({"kind": kind, "skipped": "the line does not cross this grid", "cells": [],
                               **({"id": e["id"]} if "id" in e else {})})
            continue
        out = (apply_levee(g, e["line"], float(e["crest"])) if kind == "levee"
               else apply_breach(g, e["line"], e.get("invert")))
        g["z_burnt"], g["z"] = out.pop("terrain"), out.pop("ground")
        g["edits"].append({**out, "line": [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in e["line"]],
                           **({"id": e["id"]} if "id" in e else {})})
    return g


def levee_profile(grid: dict[str, Any], line: Any, h: Any = None, *, wet_m: float = 0.05,
                  freeboard_m: float = FREEBOARD_M) -> dict[str, Any]:
    """The ground and the flood level along a drawn line, for setting a levee's crest: for each cell the line
    crosses (in order), the distance along the line, the natural ground, the terrain now (with any levee) and,
    given a run's depth ``h`` (over the burnt bed, as :func:`run_grid` returns it), the water level where it is
    wetter than ``wet_m`` (None where dry). ``suggested_crest_m`` is the highest water along the line plus
    ``freeboard_m`` (:data:`FREEBOARD_M` by default), rounded up to 10 cm; with no water on the line, the highest
    ground plus 1 m."""
    win = grid.get("window", grid)
    cells = supercover_cells(grid, line)
    if not cells:
        raise ValueError("the line does not cross the grid")
    nat = _natural(grid)
    rr = np.array([r for r, _ in cells], dtype=np.intp)
    cc = np.array([c for _, c in cells], dtype=np.intp)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rr), np.diff(cc)))]) * float(grid.get("dx", win["dx"]))
    ground = nat["z"][rr, cc]
    now = np.asarray(grid["z"], dtype="f8")[rr, cc]
    water: list[float | None] = [None] * len(cells)
    if h is not None:
        hh = np.asarray(h, dtype="f8").reshape(np.shape(grid["z"]))[rr, cc]
        lvl = np.asarray(grid["z_burnt"], dtype="f8")[rr, cc] + hh
        water = [round(float(v), 2) if d > wet_m else None for v, d in zip(lvl, hh)]
    wet = [w for w in water if w is not None]
    top = max(wet) + float(freeboard_m) if wet else float(ground.max()) + 1.0
    return {"cells": [[int(r), int(c)] for r, c in cells], "distance_m": [round(float(d), 1) for d in dist],
            "ground_m": [round(float(v), 2) for v in ground], "terrain_m": [round(float(v), 2) for v in now],
            "water_m": water, "highest_water_m": round(max(wet), 2) if wet else None,
            "lowest_ground_m": round(float(ground.min()), 2), "highest_ground_m": round(float(ground.max()), 2),
            "suggested_crest_m": math.ceil(round(top * 10.0, 6)) / 10.0, "freeboard_m": float(freeboard_m),
            "length_m": round(_line_m(win, line), 1)}


def weir_flow(width_m: float, head_m: float, c: float = WEIR_C) -> float:
    """A broad-crested weir's flow (m3/s): ``Q = C b H^1.5``, ``b`` the crest's width and ``H`` the upstream water
    level above the crest (the approach velocity head left out)."""
    return float(c) * float(width_m) * max(float(head_m), 0.0) ** 1.5


def compare_runs(g0: dict[str, Any], h0: Any, g1: dict[str, Any], h1: Any, *, wet_m: float = 0.05
                 ) -> dict[str, Any]:
    """Before and after on the same grid (the largest depths of two runs, over the burnt bed as :func:`run_grid`
    gives them): the wet area of each, the cells kept dry (wet before, dry after), the cells newly wet, and the
    change in depth where either is wet."""
    d0 = depth_over_terrain(g0, h0)
    d1 = depth_over_terrain(g1, h1)
    w0, w1 = d0 > wet_m, d1 > wet_m
    a = float(g1["dx"]) ** 2 / 1e6
    either = w0 | w1
    diff = (d1 - d0)[either]
    return {"wet_km2_before": round(float(w0.sum()) * a, 3), "wet_km2_after": round(float(w1.sum()) * a, 3),
            "wet_km2_change": round(float(w1.sum() - w0.sum()) * a, 3),
            "kept_dry_km2": round(float((w0 & ~w1).sum()) * a, 3),
            "newly_wet_km2": round(float((w1 & ~w0).sum()) * a, 3),
            "max_depth_m_before": round(float(d0.max()), 2), "max_depth_m_after": round(float(d1.max()), 2),
            "depth_change_m": ({"min": round(float(diff.min()), 2), "max": round(float(diff.max()), 2),
                                "median": round(float(np.median(diff)), 2)} if diff.size else None)}


# ── comparing with a reference map ──────────────────────────────────────────


def compare_depths(model: Any, reference: Any, *, wet_m: float = 0.05, mask: Any = None) -> dict[str, Any]:
    """Agreement between two depth grids on the same cells: the cells wet in both, in one only, the critical
    success index ``hits / (hits + misses + false alarms)`` (the share of the cells wet in either that are wet in
    both), the hit rate (share of the reference's wet cells the model wets), and the depth difference (model -
    reference) where both are wet."""
    m = np.asarray(model, dtype="f8")
    ref = np.asarray(reference, dtype="f8")
    ok = np.isfinite(ref) & np.isfinite(m)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    mw = (m > wet_m) & ok
    rw = (ref > wet_m) & ok
    hits = int((mw & rw).sum())
    misses = int((~mw & rw).sum())
    false = int((mw & ~rw).sum())
    either = hits + misses + false
    d = (m - ref)[mw & rw]
    return {
        "cells": int(ok.sum()), "wet_model": int(mw.sum()), "wet_reference": int(rw.sum()), "wet_both": hits,
        "reference_only": misses, "model_only": false,
        "csi": round(hits / either, 3) if either else None,
        "hit_rate": round(hits / (hits + misses), 3) if hits + misses else None,
        "false_alarm_ratio": round(false / (hits + false), 3) if hits + false else None,
        "depth_diff_m": None if not d.size else {
            "mean": round(float(d.mean()), 2), "median": round(float(np.median(d)), 2),
            "mean_abs": round(float(np.abs(d).mean()), 2), "p10": round(float(np.percentile(d, 10)), 2),
            "p90": round(float(np.percentile(d, 90)), 2)},
        "wet_m": wet_m,
    }


def jrc_on_grid(grid: dict[str, Any], return_period: int = 100) -> np.ndarray:
    """The JRC CEMS-GloFAS flood depth (m, 3 arc-seconds) of ``return_period`` at every grid cell's centre (nearest
    pixel; NaN outside the map, 0 where JRC has no flood)."""
    from aquascope.flood_depth import NODATA, depth_url, tile_for
    from aquascope.utils.cog import COGNotFound, open_cog

    nx, ny = grid.get("nx", grid["size"]), grid.get("ny", grid["size"])
    rr, cc = np.mgrid[0:ny, 0:nx]
    lon, lat = cell_lonlat(grid, rr, cc)
    out = np.full((ny, nx), np.nan)
    names = {tile_for(float(la), float(lo)) for la, lo in ((lat.min(), lon.min()), (lat.min(), lon.max()),
                                                           (lat.max(), lon.min()), (lat.max(), lon.max()))}
    for name in sorted(x for x in names if x):
        try:
            cog = open_cog(depth_url(name, return_period))
        except COGNotFound:
            continue
        img = cog.images[0]
        x0, dxp, _, y0, _, dyp = img.transform
        col = np.floor((lon - x0) / dxp).astype(int)
        row = np.floor((lat - y0) / dyp).astype(int)
        inside = (col >= 0) & (col < img.width) & (row >= 0) & (row < img.height)
        if not inside.any():
            continue
        c0, c1 = int(col[inside].min()), int(col[inside].max())
        r0, r1 = int(row[inside].min()), int(row[inside].max())
        win = np.asarray(cog.read_window(c0, r0, c1 + 1, r1 + 1), dtype="f8")
        vals = win[row[inside] - r0, col[inside] - c0]
        vals = np.where((vals == NODATA) | ~np.isfinite(vals), 0.0, vals)
        out[inside] = vals
    return out


# ── the flows: GEOGLOWS's annual maxima, Log-Pearson III ───────────────────

#: GEOGLOWS v2's annual maxima of its retrospective simulation, one value a year since 1940 for every reach: the
#: same values as the yearly maxima of the API's daily record (checked on reach 230567275: the 85 years agree).
MAXIMA_ZARR = "https://geoglows-v2.s3.us-west-2.amazonaws.com/retrospective/maximums.zarr"
MAXIMA_VARIABLE = "daily"
MAXIMA_LICENCE = "CC BY-NC-SA 4.0 (the store's own licence attribute)"
#: The return periods fitted, and those a person can pick.
FIT_RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 200, 500)
RETURN_PERIODS = (2, 5, 10, 25, 50, 100)
FLOW_METHOD = ("Log-Pearson III (aquascope.hydrology.flood_frequency.fit_lp3, as the reach card's thresholds) on "
               "GEOGLOWS v2's annual maxima of simulated daily flow since 1940 (retrospective/maximums.zarr)")


class _Maxima:
    """The annual-maxima store, a reach column at a time. River ids are sorted in the store, so a reach is found
    by a binary search over the id chunks (about five reads), then one chunk of 500 reaches holds its years."""

    def __init__(self, fetch: Callable[[str], bytes | None] | None = None, base: str = MAXIMA_ZARR):
        from aquascope.archive import warnings as fw

        self.fw = fw
        self.base = base
        self.fetch = fetch or _http_get
        self.meta = fw._meta(self.fetch, base)
        self.ids_meta = self.meta["river_id/.zarray"]
        self.q_meta = self.meta[f"{MAXIMA_VARIABLE}/.zarray"]
        self.n_ids = int(self.ids_meta["shape"][0])
        self.id_chunk = int(self.ids_meta["chunks"][0])
        self._ids: dict[int, np.ndarray] = {}
        self._cols: dict[int, np.ndarray] = {}
        t = fw._read_1d(self.fetch, base, "time", self.meta).astype("i8")
        units = (self.meta.get("time/.zattrs") or {}).get("units", "")
        if not units.startswith("days since "):
            raise RuntimeError(f"unexpected time units {units!r} in {base}")
        epoch = np.datetime64(units.replace("days since ", "")[:10])
        self.years = [int(str(epoch + np.timedelta64(int(d), "D"))[:4]) for d in t]

    def _chunk_ids(self, i: int) -> np.ndarray:
        if i not in self._ids:
            data = self.fetch(f"{self.base}/river_id/{i}")
            if data is None:
                raise RuntimeError(f"{self.base}/river_id/{i} is missing")
            a = self.fw.decode_chunk(data, self.ids_meta).reshape(-1).astype("i8")
            self._ids[i] = a[: max(0, min(self.id_chunk, self.n_ids - i * self.id_chunk))]
        return self._ids[i]

    def column(self, rid: int) -> int | None:
        lo, hi = 0, (self.n_ids - 1) // self.id_chunk
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if int(self._chunk_ids(mid)[0]) <= rid:
                lo = mid
            else:
                hi = mid - 1
        ids = self._chunk_ids(lo)
        k = int(np.searchsorted(ids, rid))
        if k < len(ids) and int(ids[k]) == rid:
            return lo * self.id_chunk + k
        return None

    def maxima(self, rids: list[int]) -> dict[int, np.ndarray]:
        tc, cc = self.q_meta["chunks"]
        if tc != self.q_meta["shape"][0]:
            raise RuntimeError("the maxima store splits its years across chunks; this reader wants them whole")
        out: dict[int, np.ndarray] = {}
        for rid in dict.fromkeys(int(r) for r in rids):
            col = self.column(rid)
            if col is None:
                continue
            ch = col // cc
            if ch not in self._cols:
                data = self.fetch(f"{self.base}/{MAXIMA_VARIABLE}/0.{ch}")
                if data is None:
                    raise RuntimeError(f"{self.base}/{MAXIMA_VARIABLE}/0.{ch} is missing")
                self._cols[ch] = self.fw.decode_chunk(data, self.q_meta)
            out[rid] = np.asarray(self._cols[ch][:, col % cc], dtype="f8")
        return out


_MAXIMA: list[_Maxima] = []


def annual_maxima(rids: list[int], *, fetch: Callable[[str], bytes | None] | None = None
                  ) -> tuple[list[int], dict[int, np.ndarray]]:
    """The years and each reach's annual maxima of simulated daily flow (m3/s) from :data:`MAXIMA_ZARR`. Reading
    the store needs a Blosc decoder (``pip install numcodecs``; the Explorer's Python has it)."""
    if fetch is not None:
        store = _Maxima(fetch)
    else:
        if not _MAXIMA:
            _MAXIMA.append(_Maxima())
        store = _MAXIMA[0]
    return store.years, store.maxima(rids)


def flow_curve(years: list[int], maxima: Any) -> dict[str, Any]:
    """A reach's T-year flows from its annual maxima: Log-Pearson III, GEV by L-moments if that fails (the same
    two fits, in the same order, as the reach card's thresholds), never falling as T grows."""
    import pandas as pd

    from aquascope.hydrology.flood_frequency import fit_gev_lmoments, fit_lp3

    am = np.asarray(maxima, dtype="f8")
    ok = np.isfinite(am) & (am >= 0)
    s = pd.Series(am[ok], index=pd.to_datetime([f"{y}-01-01" for y, k in zip(years, ok) if k]))
    if len(s) < 10 or not (s > 0).any():
        return {"error": f"{len(s)} years of record; a fit needs 10"}
    try:
        fit = fit_lp3(s, return_periods=list(FIT_RETURN_PERIODS), ci_level=0.90)
        method = "Log-Pearson III"
    except Exception:  # noqa: BLE001 - the other fit the package reports
        fit = fit_gev_lmoments(s, return_periods=list(FIT_RETURN_PERIODS))
        method = "GEV by L-moments"
    q = [float(fit.return_periods[t]) for t in FIT_RETURN_PERIODS]
    q = [float(v) for v in np.maximum.accumulate(np.maximum(q, 0.0))]
    return {"T": list(FIT_RETURN_PERIODS), "q": [round(v, 3) for v in q], "method": method, "n_years": int(len(s)),
            "first_year": int(s.index.min().year), "last_year": int(s.index.max().year)}


def flow_for(curve: dict[str, Any], return_period: float) -> float:
    """The flow of ``return_period`` years on a curve: read linearly in log(T) between the fitted points, held at
    the ends (2 and 500 years)."""
    lt = np.log(np.asarray(curve["T"], dtype="f8"))
    t = min(max(float(return_period), curve["T"][0]), curve["T"][-1])
    return float(np.interp(math.log(t), lt, curve["q"]))


def return_period_for(curve: dict[str, Any], q: float) -> float | None:
    """The return period (years) of flow ``q`` on a curve, between 2 and 500 years; None outside that range."""
    qs = np.asarray(curve["q"], dtype="f8")
    if not (qs[0] <= q <= qs[-1]) or qs[-1] <= qs[0]:
        return None
    k = int(np.searchsorted(qs, q, side="right")) - 1
    k = min(max(k, 0), len(qs) - 2)
    lo, hi = qs[k], qs[k + 1]
    f = 0.0 if hi <= lo else (q - lo) / (hi - lo)
    lt = math.log(curve["T"][k]) + f * (math.log(curve["T"][k + 1]) - math.log(curve["T"][k]))
    return float(math.exp(lt))


def flow_words(curve: dict[str, Any]) -> str:
    """How every tool names a curve's method and source, the same words in Flood it, Route it, the reservoir and
    Replay (the Explorer's ``design-flow-core.js`` writes the same from the same fields)."""
    return (f"{curve['method']} on GEOGLOWS v2's annual maxima of simulated daily flow, "
            f"{curve['first_year']} to {curve['last_year']}")


def reach_curve(river_id: int, *, fetch: Callable[[str], bytes | None] | None = None) -> dict[str, Any]:
    """One reach's T-year flow curve: :func:`flow_curve` on its :func:`annual_maxima`; ``{"error": ...}`` when the
    store has no record for it or the fit cannot be made."""
    rid = int(river_id)
    years, am = annual_maxima([rid], fetch=fetch)
    if rid not in am:
        return {"error": f"reach {rid} is not in GEOGLOWS's annual maxima store"}
    return flow_curve(years, am[rid])


def design_flow(river_id: int, return_period: float, *,
                fetch: Callable[[str], bytes | None] | None = None) -> dict[str, Any]:
    """A reach's ``return_period``-year flow, the one design flow every tool uses (Flood it, Route it, Draw a
    reservoir, Replay): Log-Pearson III on GEOGLOWS v2's annual maxima (:func:`reach_curve`), read on the curve
    with :func:`flow_for`. ``return_period`` is 2 to 500 years, the range the curve is fitted over.

    Returns ``flow_cms``, ``return_period``, the curve's ``method``, ``n_years``, ``first_year`` and
    ``last_year``, ``words`` (:func:`flow_words`), the store (``source``) and its ``licence``. Raises
    ``ValueError`` for a return period outside the range and ``RuntimeError`` when the reach has no curve."""
    t = float(return_period)
    lo, hi = FIT_RETURN_PERIODS[0], FIT_RETURN_PERIODS[-1]
    if not (lo <= t <= hi):
        raise ValueError(f"return_period must be between {lo} and {hi} years (the range the flows are fitted over)")
    curve = reach_curve(river_id, fetch=fetch)
    if curve.get("error"):
        raise RuntimeError(f"no flood frequency curve for reach {int(river_id)}: {curve['error']}")
    q = flow_for(curve, t)
    return {"river_id": int(river_id), "return_period": t, "flow_cms": round(q, 3), "method": curve["method"],
            "n_years": curve["n_years"], "first_year": curve["first_year"], "last_year": curve["last_year"],
            "words": flow_words(curve), "source": MAXIMA_ZARR, "licence": MAXIMA_LICENCE}


def matched_flow(click: dict[str, Any], q: float, other: dict[str, Any]) -> float:
    """The flow at another reach that matches ``q`` at the clicked reach: the other reach's flow at the same return
    period; below the clicked reach's 2-year flow (or above its 500-year flow) the same share of the other reach's
    2-year (or 500-year) flow."""
    q2c, q500c = click["q"][0], click["q"][-1]
    if q <= q2c:
        return float(other["q"][0] * (q / q2c if q2c > 0 else 0.0))
    if q >= q500c:
        return float(other["q"][-1] * q / q500c)
    return flow_for(other, return_period_for(click, q) or 2.0)


# ── the river network in the grid ───────────────────────────────────────────

#: River followed up and down from the click: reaches whose catchment centre is within this many km.
CORRIDOR_KM = 6.0
#: Ground this high above the nearest river is outside the valley the grid keeps (m).
VALLEY_M = 10.0
#: The rivers that make the valley: those with a 100-year flow at least this share of the clicked reach's.
VALLEY_SHARE = 0.1
#: The search margin around the river for the valley (km), and the cells kept around it.
VALLEY_PAD_KM = 3.0
VALLEY_PAD_CELLS = 6
#: The largest grid: a side, and the cells; a larger valley runs at the next zoom out (cells twice as large).
MAX_SIDE = 1024
MAX_CELLS = 524_288
STREAM_ZOOM = 12


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 12742.0 * math.asin(min(1.0, math.sqrt(a)))


def stream_reaches(bounds: list[float], zoom: int = STREAM_ZOOM, cap: int = 64) -> dict[int, dict[str, Any]]:
    """Every GEOGLOWS reach drawn in ``bounds`` [W, S, E, N]: ``river_id -> {"order", "lines"}`` with the lines in
    (lon, lat), from the stream tiles at ``zoom`` (12: every reach is in the archive there)."""
    from aquascope import rivers

    w, s, e, n = bounds
    x0, y0 = rivers._world(w, n, zoom)
    x1, y1 = rivers._world(e, s, zoom)
    tiles = [(tx, ty) for tx in range(int(x0 // 4096), int(x1 // 4096) + 1)
             for ty in range(int(y0 // 4096), int(y1 // 4096) + 1)]
    if len(tiles) > cap:
        raise ValueError(f"{len(tiles)} stream tiles for this grid; the most read is {cap}")
    out: dict[int, dict[str, Any]] = {}
    for tx, ty in tiles:
        box = (tx * 4096.0, ty * 4096.0, (tx + 1) * 4096.0, (ty + 1) * 4096.0)
        for rid, item in rivers._tile_reaches(zoom, tx, ty).items():
            r = out.setdefault(int(rid), {"order": int(item.get("order") or 0), "lines": []})
            for line in item["lines"]:
                # Each tile carries its lines a little past its edges; cut to the tile itself, so the pieces of
                # one reach meet end to end at the tile edges instead of overlapping.
                for part in _clip_line(line, box):
                    r["lines"].append([rivers._lonlat(px, py, zoom) for px, py in part])
    return out


def _clip_line(line: list[tuple[float, float]], box: tuple[float, float, float, float]
               ) -> list[list[tuple[float, float]]]:
    """A polyline cut to a box (x0, y0, x1, y1): the parts inside, each a polyline (Liang-Barsky per segment)."""
    x0, y0, x1, y1 = box
    parts: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        dx, dy = bx - ax, by - ay
        t0, t1 = 0.0, 1.0
        ok = True
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0:
                    ok = False
                    break
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if not ok or t0 > t1:
            if len(cur) >= 2:
                parts.append(cur)
            cur = []
            continue
        a = (ax + t0 * dx, ay + t0 * dy)
        b = (ax + t1 * dx, ay + t1 * dy)
        if cur and cur[-1] == a:
            cur.append(b)
        else:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [a, b]
        if t1 < 1.0:
            parts.append(cur)
            cur = []
    if len(cur) >= 2:
        parts.append(cur)
    return parts


def _accumulated(net: Any, i: int) -> dict[int, float]:
    """Each reach upstream of index ``i`` (its own included) with the catchment area draining to it (m2)."""
    ups = net.upstream(i)
    acc: dict[int, float] = {}
    for j in reversed(ups):
        acc[j] = float(net.area_m2[j]) + sum(acc.get(c, 0.0) for c in net.children(j))
    return acc


def _line_km(lines: list[list[tuple[float, float]]]) -> float:
    return sum(_haversine_km(a[1], a[0], b[1], b[0]) for ln in lines for a, b in zip(ln, ln[1:]))


def corridor(net: Any, river_id: int, reaches: dict[int, dict[str, Any]], km: float = CORRIDOR_KM) -> list[int]:
    """The main stem through ``river_id``, upstream to downstream: up the branch with the largest catchment at each
    confluence and down the network, until ``km`` of drawn river (``reaches``' lines) each way. Reaches the stream
    tiles do not draw (short links) are walked through."""
    i = net.index(int(river_id))
    if i is None:
        raise LookupError(f"reach {river_id} is not in this processing unit")
    acc = _accumulated(net, i)

    def length(j: int) -> float:
        item = reaches.get(int(net.ids[j]))
        return _line_km(item["lines"]) if item else 0.0

    up: list[int] = []
    j, gone = i, 0.0
    while len(up) < 400 and gone < km:
        kids = net.children(j)
        if not kids:
            break
        j = max(kids, key=lambda c: acc.get(c, 0.0))
        up.append(j)
        gone += length(j)
    down: list[int] = []
    j, gone = i, 0.0
    while len(down) < 400 and gone < km:
        d = net.index(int(net.ds[j])) if int(net.ds[j]) > 0 else None
        if d is None:
            break
        down.append(d)
        gone += length(d)
        j = d
    return [int(net.ids[k]) for k in [*reversed(up), i, *down]]


def _links(net: Any, rid: int, drawn: set[int], known: set[int], depth: int = 6) -> tuple[list[int], list[int], int]:
    """A drawn reach's neighbours: the drawn reaches above it, the reaches above it that the tiles draw outside the
    grid (rivers entering it), and the drawn reach below it. Reaches the tiles do not draw at all (short links,
    not in ``known``) are walked through, at most ``depth`` of them."""
    k = net.index(rid)
    ups: list[int] = []
    outside: list[int] = []
    stack = [(c, 0) for c in net.children(k)]
    while stack:
        c, d = stack.pop()
        cid = int(net.ids[c])
        if cid in drawn:
            ups.append(cid)
        elif cid in known:
            outside.append(cid)
        elif d < depth:
            stack.extend((g, d + 1) for g in net.children(c))
    ds, j = -1, k
    for _ in range(depth + 1):
        nxt = int(net.ds[j])
        if nxt <= 0:
            break
        if nxt in drawn:
            ds = nxt
            break
        j = net.index(nxt)
        if j is None:
            break
    return ups, outside, ds


def plan_window(river_id: int | None = None, lat: float | None = None, lon: float | None = None, *,
                km: float = CORRIDOR_KM, zoom: int = 12, net: Any = None,
                reaches: dict[int, dict[str, Any]] | None = None,
                curves: dict[int, dict[str, Any]] | None = None,
                maxima_fetch: Callable[[str], bytes | None] | None = None) -> dict[str, Any]:
    """The first half of a run: the window the grid is cut from, the reaches in it and their flows.

    The window follows the river: the main stem (:func:`corridor`) from the click, its drawn lines' extent padded
    by :data:`VALLEY_PAD_KM` for the valley (cut down to the valley by :func:`prepare_grid`), at most
    :data:`MAX_SIDE` cells a side at ``zoom``. Every reach drawn in it comes with its Strahler order, its lines,
    the drawn reaches above it (``children``) and below it (``ds``) in GEOGLOWS's routing configuration
    (:func:`_links`), and its flow curve (:func:`flow_curve`). ``net``, ``reaches`` and ``curves`` skip the
    downloads."""
    from aquascope import rivers

    if river_id is None:
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon to snap to the nearest reach")
        snap = rivers.snap_to_river(lat, lon)
        if not snap.get("snapped"):
            raise LookupError(snap.get("message") or "no river near this point")
        river_id = int(snap["river_id"])
    rid = int(river_id)
    if net is None:
        net = rivers._network_for(rid, lat, lon)
    i = net.index(rid)
    if i is None:
        raise LookupError(f"reach {rid} is not in GEOGLOWS's network")
    if lat is None or lon is None:
        lat, lon = float(net.lat[i]), float(net.lon[i])
    kx = 111.32 * max(0.05, math.cos(math.radians(lat)))
    reaches_given = reaches
    if reaches is None:
        pad = km + VALLEY_PAD_KM
        reaches = stream_reaches([lon - pad / kx, lat - pad / 111.32, lon + pad / kx, lat + pad / 111.32])
    stem = corridor(net, rid, reaches, km)
    # The stem's lines within reach of the click (a long reach's line can run far past it).
    pts = [p for r in stem if r in reaches for ln in reaches[r]["lines"] for p in ln
           if math.hypot((p[0] - lon) * kx, (p[1] - lat) * 111.32) <= km * 1.25] or [(lon, lat)]
    w = min(p[0] for p in pts) - VALLEY_PAD_KM / kx
    e = max(p[0] for p in pts) + VALLEY_PAD_KM / kx
    s = min(p[1] for p in pts) - VALLEY_PAD_KM / 111.32
    n = max(p[1] for p in pts) + VALLEY_PAD_KM / 111.32
    xa, ya = _world_px(w, n, zoom)
    xb, yb = _world_px(e, s, zoom)
    cx, cy = _world_px(lon, lat, zoom)
    x0, y0 = int(math.floor(xa)), int(math.floor(ya))
    nx, ny = int(math.ceil(xb)) - x0, int(math.ceil(yb)) - y0
    if nx > MAX_SIDE:
        x0, nx = int(round(min(max(cx - MAX_SIDE / 2, xa), xb - MAX_SIDE))), MAX_SIDE
    if ny > MAX_SIDE:
        y0, ny = int(round(min(max(cy - MAX_SIDE / 2, ya), yb - MAX_SIDE))), MAX_SIDE
    win = window(x0, y0, nx, ny, zoom)
    # Every reach drawn near the window: a river entering it is drawn just outside.
    bw, bs, be, bn = win["bounds"]
    m = 2.0
    near = stream_reaches([bw - m / kx, bs - m / 111.32, be + m / kx, bn + m / 111.32]) if reaches_given is None \
        else reaches
    known = {int(r) for r in near} | {int(r) for r in reaches}
    lines_of = {**{int(r): v for r, v in reaches.items()}, **{int(r): v for r, v in near.items()}}
    drawn = {r for r, item in lines_of.items() if net.index(r) is not None
             and any(_line_cells(win, ln) for ln in item["lines"])}
    if rid not in drawn:
        raise LookupError(f"reach {rid} has no line in the stream tiles here")
    inside: dict[int, dict[str, Any]] = {}
    for r in drawn:
        ups, outside, ds = _links(net, r, drawn, known)
        inside[r] = {"order": int(lines_of[r]["order"]), "lines": lines_of[r]["lines"], "ds": ds, "children": ups,
                     "outside": outside}
    wanted = sorted(set(inside) | {o for v in inside.values() for o in v["outside"]})
    if curves is None:
        years, am = annual_maxima(wanted, fetch=maxima_fetch)
        curves = {r: flow_curve(years, v) for r, v in am.items()}
    for r in inside:
        inside[r]["curve"] = curves.get(r) or {"error": "no simulated record for this reach"}
        inside[r]["outside_curves"] = {str(o): curves.get(o) or {"error": "no simulated record"}
                                       for o in inside[r]["outside"]}
    if "q" not in inside[rid]["curve"]:
        raise LookupError(f"reach {rid} has no flood flows: {inside[rid]['curve'].get('error')}")
    return {"river_id": rid, "lat": float(lat), "lon": float(lon), "vpu": getattr(net, "vpu", None),
            "stem": stem, "window": win, "reaches": {str(k): v for k, v in inside.items()}}


# ── the grid: valley, channels, inflows and edges ──────────────────────────


def _reach_cells(win: dict[str, Any], lines: list[list[tuple[float, float]]], tol_deg: float = 5e-4
                 ) -> list[tuple[int, int]]:
    """A reach's cells in order along its line. The stream tiles cut a reach's line into pieces, one a tile; the
    pieces are put back end to end: two ends closer than ``tol_deg`` join, the walk starts at an end that joins
    nothing and follows the joins, so a long winding reach is never folded back on itself (pieces it cannot reach
    follow, nearest first). Upstream or downstream is decided later."""
    pieces = [list(ln) for ln in lines if len(ln) >= 2]
    if not pieces:
        return []

    def gap(p: tuple[float, float], q: tuple[float, float]) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def partner(i: int, end: int) -> tuple[int, int] | None:
        pt = pieces[i][-1 if end else 0]
        best = None
        for j in range(len(pieces)):
            if j == i:
                continue
            for e in (0, 1):
                d = gap(pt, pieces[j][-1 if e else 0])
                if d <= tol_deg and (best is None or d < best[0]):
                    best = (d, j, e)
        return None if best is None else (best[1], best[2])

    free = [(i, e) for i in range(len(pieces)) for e in (0, 1) if partner(i, e) is None]
    i, e = free[0] if free else (0, 0)
    seen: set[int] = set()
    ordered = []
    while i is not None and i not in seen:
        seen.add(i)
        piece = pieces[i] if e == 0 else pieces[i][::-1]   # walk in from end e
        ordered.append(piece)
        nxt = partner(i, 1 - e)
        i, e = (nxt if nxt and nxt[0] not in seen else (None, 0))
    left = [k for k in range(len(pieces)) if k not in seen]
    while left:
        end = ordered[-1][-1]
        j = min(left, key=lambda k: min(gap(end, pieces[k][0]), gap(end, pieces[k][-1])))
        ordered.append(pieces[j] if gap(end, pieces[j][0]) <= gap(end, pieces[j][-1]) else pieces[j][::-1])
        left.remove(j)
    cells: list[tuple[int, int]] = []
    for piece in ordered:
        for rc in _line_cells(win, piece):
            if not cells or rc != cells[-1]:
                cells.append(rc)
    return list(dict.fromkeys(cells))


#: How far (cells) a river line is moved onto the lowest ground beside it: GEOGLOWS's lines come from another,
#: coarser terrain model, and in a winding valley they cross the spurs (a channel burnt there would be a trench).
SNAP_CELLS = 2


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """The 8-connected cells from ``a`` to ``b``, ``b`` included, ``a`` not."""
    (r0, c0), (r1, c1) = a, b
    n = max(abs(r1 - r0), abs(c1 - c0))
    return [(int(round(r0 + (r1 - r0) * k / n)), int(round(c0 + (c1 - c0) * k / n))) for k in range(1, n + 1)]


def snap_chain(chain: list[tuple[int, int]], z: np.ndarray, radius: int = SNAP_CELLS) -> list[tuple[int, int]]:
    """A river's ordered cells moved onto the lowest terrain within ``radius`` cells of each, joined up again
    (8-connected) and without repeats: where the drawn line runs along the valley floor it stays put, where it
    cuts across a spur it moves down beside it."""
    if radius <= 0 or not chain:
        return list(chain)
    ny, nx = z.shape
    moved = []
    for r, c in chain:
        r0, r1, c0, c1 = max(0, r - radius), min(ny, r + radius + 1), max(0, c - radius), min(nx, c + radius + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        # The lowest cell, the nearest of the lowest when several tie (a flat floor keeps the line where it is).
        k = int(np.argmin(z[r0:r1, c0:c1] + 1e-6 * ((yy - r) ** 2 + (xx - c) ** 2)))
        moved.append((r0 + k // (c1 - c0), c0 + k % (c1 - c0)))
    out = [moved[0]]
    for cell in moved[1:]:
        if cell == out[-1]:
            continue
        for step in _bresenham(out[-1], cell):
            if step not in out[-3:]:
                out.append(step)
    return list(dict.fromkeys(out))


def four_connected(cells: list[tuple[int, int]], z: np.ndarray) -> list[tuple[int, int]]:
    """Ordered cells with a cell added at every diagonal step (the lower of the two that close it): water moves
    between cells only through their faces, so a channel that steps diagonally would be dammed at each step by
    whatever ground stands at the corner."""
    if not cells:
        return []
    out = [cells[0]]
    for r, c in cells[1:]:
        pr, pc = out[-1]
        if abs(r - pr) == 1 and abs(c - pc) == 1:
            a, b = (r, pc), (pr, c)
            out.append(a if z[a] <= z[b] else b)
        out.append((r, c))
    return list(dict.fromkeys(out))


def _nearest(cells: list[tuple[int, int]], others: list[tuple[int, int]]) -> float:
    if not others:
        return math.inf
    o = np.asarray(others, dtype="f8")
    return float(min(np.min((o[:, 0] - r) ** 2 + (o[:, 1] - c) ** 2) for r, c in cells))


def _orient_chain(chain: list[tuple[int, int]], z: np.ndarray, ds_cells: list[tuple[int, int]],
                  up_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The chain upstream first: its end nearer the reach below (or nearer the reaches above) decides; with
    neither in the grid, the higher end is upstream."""
    if len(chain) < 2:
        return chain
    a, b = chain[:1], chain[-1:]
    if ds_cells:
        return chain if _nearest(b, ds_cells) <= _nearest(a, ds_cells) else chain[::-1]
    if up_cells:
        return chain if _nearest(a, up_cells) <= _nearest(b, up_cells) else chain[::-1]
    k = max(1, min(5, len(chain) // 2))
    za = float(np.mean([z[p] for p in chain[:k]]))
    zb = float(np.mean([z[p] for p in chain[-k:]]))
    return chain if za >= zb else chain[::-1]


def chain_slope(z: np.ndarray, cells: list[tuple[int, int]], dx: float) -> float:
    """A river's slope along ordered cells: the least-squares fall of its bank line (made non-increasing downstream,
    as :func:`burn_channel` makes it) against the distance along the cells, kept within :data:`SLOPE_RANGE`."""
    if len(cells) < 3:
        return SLOPE_RANGE[0]
    bank = np.minimum.accumulate(np.array([z[c] for c in cells], dtype="f8"))
    steps = np.hypot(np.diff([c[0] for c in cells]), np.diff([c[1] for c in cells]))
    s = np.concatenate([[0.0], np.cumsum(steps)]) * dx
    if s[-1] <= 0:
        return SLOPE_RANGE[0]
    fall = -float(np.polyfit(s, bank, 1)[0])
    return min(max(fall, SLOPE_RANGE[0]), SLOPE_RANGE[1])


def _valley_box(z: np.ndarray, channel: np.ndarray, valley_m: float = VALLEY_M,
                pad: int = VALLEY_PAD_CELLS) -> tuple[int, int, int, int]:
    """Rows and columns [r0, r1) x [c0, c1) of the valley: the cells no higher than ``valley_m`` above the nearest
    river cell (a height above the nearest drainage, by straight-line distance), padded by ``pad`` cells."""
    from scipy import ndimage

    if not channel.any():
        return 0, z.shape[0], 0, z.shape[1]
    _, (ri, ci) = ndimage.distance_transform_edt(~channel, return_indices=True)
    hand = z - z[ri, ci]
    rows, cols = np.nonzero(hand <= valley_m)
    r0, r1 = max(0, int(rows.min()) - pad), min(z.shape[0], int(rows.max()) + 1 + pad)
    c0, c1 = max(0, int(cols.min()) - pad), min(z.shape[1], int(cols.max()) + 1 + pad)
    return r0, r1, c0, c1


def coarsen(z: Any) -> np.ndarray:
    """Halve a grid: the mean of 2 x 2 blocks (an odd last row or column is dropped)."""
    a = np.asarray(z, dtype="f8")
    ny, nx = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:ny, :nx]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def choice_label(choice: dict[str, Any]) -> str:
    """The flow choice in words: "the 100-year flow", "500 m³/s", or the hydrograph's own label."""
    if "return_period" in choice:
        return f"the {float(choice['return_period']):g}-year flow"
    if "flow_cms" in choice:
        return f"{float(choice['flow_cms']):,.0f} m³/s"
    return str(choice.get("label") or "a hydrograph")


def prepare_grid(plan: dict[str, Any], z: Any, choice: dict[str, Any] | None = None, *, coarse: bool = False,
                 valley_m: float = VALLEY_M, max_cells: int = MAX_CELLS) -> dict[str, Any]:
    """The second half: the grid a run floods, from :func:`plan_window`'s answer and the window's terrain ``z``.

    * **The valley.** The window is cut to the cells at most ``valley_m`` above the nearest river, plus a margin.
      A valley still larger than ``max_cells`` (or ``coarse``, the fallback without a GPU) runs on cells twice as
      large.
    * **The channels.** Every reach is burnt in along its line, one cell wide, to the depth that carries its own
      2-year flow at bankfull (:func:`channel_depth`, with the network's slope), its bank made non-increasing
      downstream. Manning's n is :data:`N_CHANNEL` there and :data:`N_FLOODPLAIN` elsewhere.
    * **The inflows.** Every reach whose reaches upstream are all outside the grid (a river entering it, or a
      headwater starting in it) gets its own flow at its first cell. The flow at the clicked reach is chosen by
      ``choice``: ``{"return_period": T}``, ``{"flow_cms": q}``, or ``{"series": {"t_s": [...], "q": [...]},
      "label": ...}`` (a hydrograph at the clicked reach). Each inflow gets its own reach's flow at the same
      return period (:func:`matched_flow`), hour by hour for a hydrograph.
    * **The edges.** Walls, but a normal-depth outfall (with the network's slope) across the valley where a river
      leaves the grid, and a free outfall where an edge cell is below sea level.
    """
    choice = choice or {"return_period": 100}
    win0 = plan["window"]
    z0 = np.asarray(z, dtype="f8").reshape(win0["ny"], win0["nx"])
    reaches = {int(k): v for k, v in plan["reaches"].items()}
    click = int(plan["river_id"])
    click_curve = reaches[click]["curve"]
    # The valley is the big rivers' (a 100-year flow at least VALLEY_SHARE of the clicked reach's): the small streams
    # in the hills around would otherwise make every slope a valley.
    q100 = flow_for(click_curve, 100)
    chan0 = np.zeros(z0.shape, dtype=bool)
    for r, v in reaches.items():
        cv = v["curve"]
        if r != click and ("q" not in cv or flow_for(cv, 100) < VALLEY_SHARE * q100):
            continue
        for ln in v["lines"]:
            for rc in _line_cells(win0, ln):
                chan0[rc] = True
    r0, r1, c0, c1 = _valley_box(z0, chan0, valley_m)
    win = window(win0["x0"] + c0, win0["y0"] + r0, c1 - c0, r1 - r0, win0["zoom"])
    zc = z0[r0:r1, c0:c1]
    if coarse or zc.size > max_cells:
        zc = coarsen(zc)
        win = window(win["x0"] // 2, win["y0"] // 2, zc.shape[1], zc.shape[0], win["zoom"] - 1)
    ny, nx = zc.shape
    dx = win["dx"]
    raw = {r: _reach_cells(win, v["lines"]) for r, v in reaches.items()}
    raw = {r: cells for r, cells in raw.items() if cells}
    order: dict[int, list[tuple[int, int]]] = {}
    for r, cells in raw.items():
        up = [c for k in reaches[r]["children"] for c in raw.get(k, [])]
        order[r] = snap_chain(_orient_chain(cells, zc, raw.get(reaches[r]["ds"], []), up), zc)
    # Each reach's channel runs on into the reach below it: from its last cell to the nearest cell of the next, so a
    # confluence or a gap between two lines never leaves a bank of ground across the river.
    for r, cells in order.items():
        below = order.get(reaches[r]["ds"])
        if not below or not cells:
            continue
        last = cells[-1]
        tgt = min(below, key=lambda c: (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2)
        if max(abs(tgt[0] - last[0]), abs(tgt[1] - last[1])) > 1:
            order[r] = cells + _bresenham(last, tgt)[:-1]
        order[r] = four_connected(order[r] + [tgt], zc)[:-1] if tgt not in order[r] else order[r]
    order = {r: four_connected(cells, zc) for r, cells in order.items()}
    stem = [r for r in plan.get("stem", []) if r in order] or [click]
    slope = chain_slope(zc, [c for r in stem for c in order[r]], dx)
    slopes = {r: slope if r in stem or len(cells) < 20 else chain_slope(zc, cells, dx) for r, cells in order.items()}
    zb = np.array(zc)
    n = np.full(zc.shape, N_FLOODPLAIN)
    channels: dict[int, float] = {}

    def q2_of(r: int) -> float:
        cv = reaches[r]["curve"]
        return float(cv["q"][0]) if "q" in cv else 0.0

    for r in sorted(order, key=q2_of):   # bigger rivers last: where two lines share a cell, the bigger bed wins
        if q2_of(r) <= 0:
            continue
        depth = channel_depth(q2_of(r), dx, slopes[r])
        burn_channel(zc, order[r], depth, into=zb, max_cut_m=MAX_CUT_M)
        for rc in order[r]:
            n[rc] = N_CHANNEL
        channels[r] = round(depth, 2)
    # The inflows. A reach whose line crosses the edge at its upstream end brings its own flow in there; a reach that
    # starts inside the grid takes the flow of each reach above it that is outside the grid, or its own flow when
    # nothing is above it at all (a headwater).
    sources: list[dict[str, Any]] = []
    for r, cells in order.items():
        if q2_of(r) <= 0:
            continue
        r_, c_ = cells[0]
        at_edge = min(r_, ny - 1 - r_, c_, nx - 1 - c_) <= 2
        rec = reaches[r]
        inner = [k for k in rec["children"] if k in order]
        outer = [int(k) for k in rec.get("outside", [])] + [k for k in rec["children"] if k not in order]
        base = {"cell": [int(r_), int(c_)], "into": r, "order": rec["order"]}
        if at_edge and not inner:
            sources.append({**base, "river_id": r, "enters": True, "curve": rec["curve"]})
        elif not inner and not outer:
            sources.append({**base, "river_id": r, "enters": False, "curve": rec["curve"]})
        else:
            for k in outer:
                cv = rec.get("outside_curves", {}).get(str(k)) or reaches.get(k, {}).get("curve") or {}
                if "q" in cv:
                    sources.append({**base, "river_id": k, "enters": True, "curve": cv})
    table_dt = 3600.0
    hours = None
    if "series" in choice:
        ser = choice["series"]
        t = np.asarray(ser["t_s"], dtype="f8")
        q = np.asarray(ser["q"], dtype="f8")
        ok = np.isfinite(t) & np.isfinite(q)
        t, q = t[ok] - t[ok][0], np.maximum(q[ok], 0.0)
        grid_t = np.arange(0.0, t[-1] + table_dt / 2, table_dt)
        q_click = np.interp(grid_t, t, q)
        for s in sources:
            s["flow"] = [round(matched_flow(click_curve, float(v), s["curve"]), 3) for v in q_click]
            s["peak"] = max(s["flow"])
        click_q = float(q_click.max())
        hours = float(grid_t[-1] / 3600.0)
    else:
        click_q = (flow_for(click_curve, float(choice["return_period"])) if "return_period" in choice
                   else float(choice["flow_cms"]))
        for s in sources:
            s["flow"] = round(matched_flow(click_curve, click_q, s["curve"]), 3)
            s["peak"] = s["flow"]
    t_eq = return_period_for(click_curve, click_q)
    codes = edge_codes(ny, nx, closed=True)
    exits: list[dict[str, Any]] = []
    for r, cells in order.items():
        if reaches[r]["ds"] in order:
            continue
        rr_, cc_ = cells[-1]
        dist = {"n": rr_, "s": ny - 1 - rr_, "w": cc_, "e": nx - 1 - cc_}
        side = min(dist, key=lambda k: dist[k])
        if dist[side] > 3:
            continue
        line = {"n": zc[0, :], "s": zc[-1, :], "w": zc[:, 0], "e": zc[:, -1]}[side]
        k = cc_ if side in "ns" else rr_
        bank = float(np.min(line[max(0, k - 2):k + 3]))
        lo = hi = k
        while lo > 0 and line[lo - 1] <= bank + valley_m:
            lo -= 1
        while hi < len(line) - 1 and line[hi + 1] <= bank + valley_m:
            hi += 1
        codes[side][lo:hi + 1] = slope
        exits.append({"river_id": int(r), "side": side, "from": int(lo), "to": int(hi)})
    for side, line in (("n", zc[0, :]), ("s", zc[-1, :]), ("w", zc[:, 0]), ("e", zc[:, -1])):
        codes[side][line < 0] = FREE
    return {
        "window": win, "nx": int(nx), "ny": int(ny), "dx": float(dx), "z": zc, "z_burnt": zb, "n": n,
        "slope": round(slope, 6), "channels": channels, "sources": sources, "edges": codes, "exits": exits,
        "table_dt_s": table_dt, "hours": hours,
        "click": {"river_id": click, "flow_cms": round(click_q, 3),
                  "return_period": None if t_eq is None else round(t_eq, 1), "curve": click_curve},
        "choice": choice, "label": choice_label(choice), "coarse": win["zoom"] < win0["zoom"],
        "crop": [int(r0), int(r1), int(c0), int(c1)],
    }


# ── running it ──────────────────────────────────────────────────────────────

#: The near-steady-state rule: the wet area and the volume each change by less than this share in an hour of
#: model time. The cap is the longest run (hours): where the rule has not been met by then, the card says so.
STEADY_TOL = 0.01
STEADY_EVERY_S = 3600.0
CAP_HOURS = 72.0
#: Frames: every 15 minutes for the first three hours, then hourly.
FRAME_EARLY_S, FRAME_EARLY_UNTIL_S, FRAME_LATE_S = 900.0, 3 * 3600.0, 3600.0


def frame_times(hours: float) -> list[float]:
    """The model times of the frames a run keeps, up to ``hours``."""
    end = hours * 3600.0
    out = [k * FRAME_EARLY_S for k in range(int(min(end, FRAME_EARLY_UNTIL_S) // FRAME_EARLY_S) + 1)]
    t = FRAME_EARLY_UNTIL_S + FRAME_LATE_S
    while t <= end + 1e-6:
        out.append(t)
        t += FRAME_LATE_S
    return out


def run_grid(g: dict[str, Any], *, cap_hours: float = CAP_HOURS, steady_tol: float | None = STEADY_TOL,
             frames: bool = True) -> dict[str, Any]:
    """Run :func:`simulate` on a :func:`prepare_grid` answer: to the near steady state (or the cap) for a constant
    flow, to the hydrograph's end for a hydrograph."""
    hours = g["hours"] if g["hours"] else cap_hours
    srcs = [(s["cell"][0], s["cell"][1], s["flow"]) for s in g["sources"]]
    return simulate(g["z_burnt"], g["dx"], duration_s=hours * 3600.0, n=g["n"], sources=srcs, edges=g["edges"],
                    table_dt_s=g["table_dt_s"], steady_tol=None if g["hours"] else steady_tol,
                    steady_every_s=STEADY_EVERY_S, frames_every_s=frame_times(hours) if frames else None,
                    frames_uv=frames)


def depth_over_terrain(g: dict[str, Any], h: Any) -> np.ndarray:
    """The water's depth above the terrain the map is drawn from: the depth over the burnt bed less the channel
    cut, never below zero. Off the channels the two are the same; in a channel this is the water over its banks."""
    return np.maximum(np.asarray(g["z_burnt"]) + np.asarray(h) - np.asarray(g["z"]), 0.0)


#: When the flood is held by a wall of the grid (:func:`wall_contact`): water deeper than ``wet_m`` behind a walled
#: edge face, further than ``near`` cells from an inflow and ``gap`` faces from an open edge, at ``min_faces`` faces
#: or more. The Explorer's ``floodsim-core.js`` (``WALL_RULE``, ``wallContact``) uses the same numbers.
WALL_RULE = {"wet_m": 0.1, "near": 40, "gap": 3, "min_faces": 3}


def wall_contact(depth: Any, edges: dict[str, Any], sources: list[Any], **rule: Any) -> dict[str, Any]:
    """Where a run's flood is held by a wall of its grid: the walled edge faces (code 0) with water deeper than the
    rule's ``wet_m`` behind them, away from the rivers that enter (``near`` cells, Chebyshev) and from the open edges
    where the river leaves (``gap`` faces). There the valley goes on past the grid, and the wall holds water that
    would spread, so the flood near it is deeper and narrower than it would be. ``depth`` is above the terrain
    (rows north to south); ``sources`` are ``(row, col)`` cells or dicts with ``"cell"``. Returns ``faces``,
    ``sides`` (faces per side) and ``held`` (``min_faces`` or more)."""
    r_ = {**WALL_RULE, **rule}
    d = np.asarray(depth, dtype="f8")
    ny, nx = d.shape
    cells = [tuple(s["cell"]) if isinstance(s, dict) else tuple(s[:2]) for s in sources]
    near, gap = int(r_["near"]), int(r_["gap"])

    def far(r: int, c: int) -> bool:
        return all(max(abs(sr - r), abs(sc - c)) > near for sr, sc in cells)

    sides = {}
    for k, n_faces, cell in (("n", nx, lambda i: (0, i)), ("s", nx, lambda i: (ny - 1, i)),
                             ("w", ny, lambda i: (i, 0)), ("e", ny, lambda i: (i, nx - 1))):
        code = np.broadcast_to(np.asarray(edges.get(k, FREE) if edges else FREE, dtype="f8"), (n_faces,))
        count = 0
        for i in range(n_faces):
            if code[i] != 0:
                continue
            lo, hi = max(0, i - gap), min(n_faces, i + gap + 1)
            if np.any(code[lo:hi] != 0):    # NaN (free) and a slope (normal depth) are open
                continue
            r, c = cell(i)
            if d[r, c] > float(r_["wet_m"]) and far(r, c):
                count += 1
        sides[k] = count
    faces = sum(sides.values())
    return {"faces": faces, "sides": sides, "held": faces >= int(r_["min_faces"])}


def wall_words(contact: dict[str, Any], dx: float) -> str:
    """:func:`wall_contact`'s warning in plain words, or "" when the flood is not held by a wall: what it means and
    what to do (the Explorer's ``wallWords`` writes the same)."""
    if not contact or not contact.get("held"):
        return ""
    names = {"n": "north", "s": "south", "w": "west", "e": "east"}
    order = sorted(((k, v) for k, v in contact["sides"].items() if v > 0), key=lambda kv: -kv[1])
    where = [names[k] for k, _ in order]
    km = contact["faces"] * float(dx) / 1000.0
    along = f"{km:.1f} km" if km >= 1 else f"{round(km * 1000)} m"
    sides = (f"{', '.join(where[:-1])} and {where[-1]} edges" if len(where) > 1 else f"{where[0]} edge")
    return (f"The water reaches the grid's {sides} along {along}. The valley goes on past it, and the edge holds the "
            "water like a wall, so the flood there is deeper and narrower than it would be. Flood a reach nearer that "
            "edge to see it.")


def run_summary(g: dict[str, Any], res: dict[str, Any], wet_m: float = 0.05) -> dict[str, Any]:
    """The numbers a card or a JSON file gives for a run (depths above the terrain, :func:`depth_over_terrain`)."""
    hm = depth_over_terrain(g, res["h_max"])
    wet = hm > wet_m
    dx = g["dx"]
    held = wall_contact(hm, g["edges"], g["sources"])
    return {
        "river_id": g["click"]["river_id"], "flow": g["label"], "flow_cms": g["click"]["flow_cms"],
        "return_period": g["click"]["return_period"], "hours_run": round(res["t_s"] / 3600.0, 2),
        "steady_at_h": None if res["steady_at_s"] is None else round(res["steady_at_s"] / 3600.0, 2),
        "steady_rule": (f"wet area and volume each change by under {STEADY_TOL:.0%} in an hour of model time"
                        if not g["hours"] else None),
        "wet_km2": round(float(wet.sum()) * dx * dx / 1e6, 3), "max_depth_m": round(float(hm.max()), 2),
        "cells": int(g["nx"] * g["ny"]), "nx": g["nx"], "ny": g["ny"], "cell_m": round(dx, 2),
        "zoom": g["window"]["zoom"], "bounds": [round(v, 6) for v in g["window"]["bounds"]],
        "sources": [{"river_id": s["river_id"], "cell": s["cell"], "enters_grid": s["enters"],
                     "peak_cms": s["peak"]} for s in g["sources"]],
        "exits": g["exits"], "channel_slope": g["slope"], "steps": res["steps"],
        "wall_s": round(res["wall_s"], 2), "volume": {k: round(v, 3) for k, v in res["volume"].items()},
        "water_at_walled_edge": held["held"], "wall_contact": held,
        **({"edits": [{k: v for k, v in e.items() if k != "cells"} | {"cells": len(e["cells"])}
                      for e in g["edits"]]} if g.get("edits") else {}),
        "label": LABEL, "method": SCHEME, "flows_from": FLOW_METHOD, "licences": LICENCES,
    }


LABEL = ("screening flood: no bathymetry, terrain errors of metres, a one-cell channel, every river at the same "
         "return period at once")
SCHEME = ("Local-inertial shallow water (Bates, Horritt and Fewtrell 2010; q-centred, de Almeida et al. 2012) on "
          "Terrarium terrain, the GEOGLOWS v2 rivers burnt in")
LICENCES = {
    "terrain": "Terrarium tiles on AWS Open Data (Mapzen/Tilezen joerd; SRTM, GMTED2010, ETOPO1 and others)",
    "rivers": "GEOGLOWS v2 river network (TDX-Hydro, NGA), CC BY-SA 4.0",
    "routing": "GEOGLOWS v2 routing configuration, CC BY 4.0",
    "flows": f"GEOGLOWS v2 retrospective annual maxima, {MAXIMA_LICENCE}",
}


def flood_reach(river_id: int | None = None, lat: float | None = None, lon: float | None = None, *,
                return_period: float | None = None, flow_cms: float | None = None,
                series: dict[str, Any] | None = None, hours: float | None = None, coarse: bool = False,
                terrain_fetch: Callable[[str], bytes] | None = None, frames: bool = True,
                edits: list[dict[str, Any]] | None = None, compare: bool = False,
                **plan_kwargs: Any) -> dict[str, Any]:
    """A screening flood from a river reach (``river_id``, or the reach nearest ``lat``/``lon``), the whole way:
    :func:`plan_window`, the terrain (:func:`terrain_window`), :func:`prepare_grid` and :func:`run_grid`.

    The flow is ``return_period`` (years; 100 when nothing is given), ``flow_cms`` (m3/s at the clicked reach) or
    ``series`` (a hydrograph at the clicked reach, ``{"t_s": [...], "q": [...], "label": ...}``). ``hours`` caps
    the run (:data:`CAP_HOURS` by default; a hydrograph runs to its end). ``edits`` are levees and breaches
    (:func:`edit_grid`) applied to the grid before the run; with ``compare`` the grid is also run without them,
    and the answer has ``before`` (its grid, run and summary) and the summary a ``change`` (:func:`compare_runs`).
    Returns ``{"plan", "grid", "run", "summary"}``."""
    if series is not None:
        choice: dict[str, Any] = {"series": series, "label": series.get("label") or "a hydrograph"}
    elif flow_cms is not None:
        choice = {"flow_cms": float(flow_cms)}
    else:
        choice = {"return_period": float(return_period or 100)}
    plan = plan_window(river_id, lat, lon, **plan_kwargs)
    w = plan["window"]
    terr = terrain_window(w["x0"], w["y0"], w["nx"], w["ny"], w["zoom"], fetch=terrain_fetch)
    g = prepare_grid(plan, terr["z"], choice, coarse=coarse)
    out: dict[str, Any] = {"plan": plan}
    if edits:
        edited = edit_grid(g, edits)
        if compare:
            r0 = run_grid(g, cap_hours=float(hours or CAP_HOURS), frames=frames)
            out["before"] = {"grid": g, "run": r0, "summary": run_summary(g, r0)}
        g = edited
    res = run_grid(g, cap_hours=float(hours or CAP_HOURS), frames=frames)
    s = run_summary(g, res)
    if "before" in out:
        b = out["before"]["summary"]
        s["before"] = {k: b[k] for k in ("wet_km2", "max_depth_m", "steady_at_h", "hours_run")}
        s["change"] = compare_runs(out["before"]["grid"], out["before"]["run"]["h_max"], g, res["h_max"])
    return {**out, "grid": g, "run": res, "summary": s}


def _join(segments: list[tuple[float, float, float, float]]) -> list[list[tuple[float, float]]]:
    """(lon1, lat1, lon2, lat2) segments chained into polylines where one ends where the next starts."""
    lines: list[list[tuple[float, float]]] = []
    for x1, y1, x2, y2 in segments:
        if lines and lines[-1][-1] == (x1, y1):
            lines[-1].append((x2, y2))
        else:
            lines.append([(x1, y1), (x2, y2)])
    return lines


# ── hydrographs: the forecast, and a wave routed down the river ───────────


def forecast_series(river_id: int, *, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """GEOGLOWS's 15-day forecast at a reach as a hydrograph for :func:`prepare_grid`: the ensemble mean
    (``flow_avg``, hourly then 3-hourly) from its first time, with its peak and when it comes."""
    from datetime import datetime

    from aquascope import rivers

    stats = stats if stats is not None else rivers.forecast_stats(int(river_id))
    if stats.get("error"):
        raise LookupError(stats["error"])
    times, q = [], []
    for t, v in zip(stats.get("datetime") or [], stats.get("flow_avg") or []):
        if v is None:
            continue
        times.append(datetime.fromisoformat(str(t).replace("Z", "+00:00")).replace(tzinfo=None))
        q.append(float(v))
    if len(q) < 2:
        raise LookupError("GEOGLOWS returned no forecast for this reach")
    t0 = times[0]
    k = int(np.argmax(q))
    return {"t_s": [(t - t0).total_seconds() for t in times], "q": q, "start": t0.isoformat(timespec="minutes"),
            "peak_cms": round(q[k], 3), "peak_at": times[k].isoformat(timespec="minutes"),
            "label": f"GEOGLOWS's 15-day forecast from {t0:%d %b %H:%M} UTC (ensemble mean)",
            "source": "GEOGLOWS v2 forecast (ECMWF ensemble), CC BY 4.0"}


def routed_series(plan: dict[str, Any], return_period: float = 100, *, duration_h: float = 24.0,
                  upstream_km: float = 30.0, net: Any = None, curves: dict[int, dict[str, Any]] | None = None,
                  fetch_maxima: Callable[[str], bytes | None] | None = None) -> dict[str, Any]:
    """The wave Route it would send (#656), as a hydrograph at the clicked reach: a triangle peaking at the
    ``return_period``-year flow of the main river ``upstream_km`` of drawn river above the click (aquascope.routing's
    ``hydrograph``), rising for a third of ``duration_h``, routed down to the clicked reach by Muskingum with
    GEOGLOWS's own k and x (aquascope.routing's ``route_network``)."""
    from aquascope import rivers, routing

    rid = int(plan["river_id"])
    net = net or rivers._network_for(rid, plan.get("lat"), plan.get("lon"))
    i = net.index(rid)
    reaches = {int(k): v for k, v in plan["reaches"].items()}
    acc = _accumulated(net, i)
    path = [i]
    gone = 0.0
    while gone < upstream_km and len(path) < 400:
        kids = net.children(path[-1])
        if not kids:
            break
        j = max(kids, key=lambda c: acc.get(c, 0.0))
        path.append(j)
        item = reaches.get(int(net.ids[j]))
        gone += _line_km(item["lines"]) if item else 2.0
    top = path[-1]
    top_id = int(net.ids[top])
    if curves is None or top_id not in curves:
        years, am = annual_maxima([top_id], fetch=fetch_maxima)
        curves = {**(curves or {}), top_id: flow_curve(years, am.get(top_id, []))}
    if "q" not in curves[top_id]:
        raise LookupError(f"reach {top_id} has no flood flows")
    peak = flow_for(curves[top_id], return_period)
    k, x, ds_index = routing._params(net)
    order = list(reversed(path))
    hours = int(math.ceil(duration_h * 3))
    q_in = routing.hydrograph(peak, duration_h, hours, shape="triangle")
    out = routing.route_network(order, ds_index, k, x, hours, inflow={top: q_in})
    q = [float(v) for v in out[i]]
    return {"t_s": [h * float(routing.DT_S) for h in range(len(q))], "q": q, "peak_cms": round(max(q), 3),
            "entered_cms": round(peak, 3), "from_river_id": top_id, "km": round(gone, 1),
            "label": (f"Route it's {return_period:g}-year wave: {peak:,.0f} m³/s entering {gone:,.0f} km upstream "
                      f"over {duration_h:g} h, routed down"),
            "source": routing.METHOD["name"]}


# ── writing a run ───────────────────────────────────────────────────────────


def kept_cells(h_max: np.ndarray, ring: int = 3, dry: float = 0.01) -> np.ndarray:
    """The cells a UGRID file keeps (flat indices): wetter than ``dry`` at some time, plus ``ring`` cells around."""
    from scipy import ndimage

    wet = np.asarray(h_max) > dry
    keep = ndimage.binary_dilation(wet, structure=np.ones((2 * ring + 1, 2 * ring + 1), dtype=bool)) if wet.any() \
        else wet
    return np.flatnonzero(keep)


def to_ugrid(path: str, g: dict[str, Any], res: dict[str, Any], *, start: str | None = None,
             title: str = "Flood it") -> dict[str, Any]:
    """Write a run as a UGRID 2D mesh in NetCDF-3 classic (the same file the Explorer hands its flood views): the
    wet cells and a ring of dry ones as square faces in longitude and latitude, the bed (the terrain with the
    channels burnt in), and the water level and velocity at every frame. ``aquascope mesh`` and the Explorer's
    model viewer read it like any model result. ``start`` is the date of the run's time zero (today by default)."""
    from datetime import date

    from scipy.io import netcdf_file

    win = g["window"]
    nx, zoom = win["nx"], win["zoom"]
    cells = kept_cells(res["h_max"])
    rows, cols = np.divmod(cells, nx)
    corner = {}

    def node(r: int, c: int) -> int:
        key = r * (nx + 1) + c
        if key not in corner:
            corner[key] = len(corner)
        return corner[key]

    conn = np.array([[node(r + 1, c), node(r + 1, c + 1), node(r, c + 1), node(r, c)] for r, c in zip(rows, cols)],
                    dtype="i4").reshape(-1, 4)
    keys = np.fromiter(corner.keys(), dtype="i8", count=len(corner))
    nr, nc = np.divmod(keys, nx + 1)
    s = 256 * 2 ** zoom
    node_x = (win["x0"] + nc) / s * 360.0 - 180.0
    node_y = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (win["y0"] + nr) / s))))
    fx, fy = cell_lonlat({**win}, rows, cols)
    # The bed a reader sees is the terrain (the channels' cut is the model's, not the ground's): the level is the
    # water over the burnt bed, never below the terrain.
    bed = np.asarray(g["z"], dtype="f8").reshape(-1)[cells]
    burnt = np.asarray(g["z_burnt"], dtype="f8").reshape(-1)[cells]
    frames = res["frames"]
    uv = res.get("uv") or [(None, None)] * len(frames)
    s1 = np.stack([np.maximum(bed, burnt + np.asarray(f, dtype="f8").reshape(-1)[cells]) for f in frames]
                  ).astype("f4")
    ux = np.stack([np.nan_to_num(np.asarray(u, dtype="f8").reshape(-1)[cells]) if u is not None
                   else np.zeros(cells.size) for u, _ in uv]).astype("f4")
    uy = np.stack([np.nan_to_num(np.asarray(v, dtype="f8").reshape(-1)[cells]) if v is not None
                   else np.zeros(cells.size) for _, v in uv]).astype("f4")
    t0 = start or date.today().isoformat()
    with netcdf_file(path, "w", version=1) as nc:
        nc.title = title
        nc.source = "aquascope Flood it (local-inertial screening flood)"
        nc.Conventions = "CF-1.8 UGRID-1.0"
        nc.comment = LABEL
        nc.createDimension("nmesh2d_node", int(keys.size))
        nc.createDimension("nmesh2d_face", int(cells.size))
        nc.createDimension("max_nmesh2d_face_nodes", 4)
        nc.createDimension("time", len(frames))
        m = nc.createVariable("mesh2d", "i4", ())
        m.cf_role = "mesh_topology"
        m.topology_dimension = 2
        m.node_coordinates = "mesh2d_node_x mesh2d_node_y"
        m.face_node_connectivity = "mesh2d_face_nodes"
        m.face_coordinates = "mesh2d_face_x mesh2d_face_y"
        for name, dim, data, units, std in (("mesh2d_node_x", "nmesh2d_node", node_x, "degrees_east", "longitude"),
                                            ("mesh2d_node_y", "nmesh2d_node", node_y, "degrees_north", "latitude"),
                                            ("mesh2d_face_x", "nmesh2d_face", fx, "degrees_east", "longitude"),
                                            ("mesh2d_face_y", "nmesh2d_face", fy, "degrees_north", "latitude")):
            v = nc.createVariable(name, "f8", (dim,))
            v[:] = data
            v.units = units
            v.standard_name = std
        v = nc.createVariable("mesh2d_face_nodes", "i4", ("nmesh2d_face", "max_nmesh2d_face_nodes"))
        v[:] = conn
        v.cf_role = "face_node_connectivity"
        v.start_index = 0
        v = nc.createVariable("time", "f8", ("time",))
        v[:] = np.asarray(res["times_s"], dtype="f8")
        v.units = f"seconds since {t0} 00:00:00"
        v.standard_name = "time"
        for name, dims, data, units, long in (
                ("mesh2d_flowelem_bl", ("nmesh2d_face",), bed.astype("f4"), "m", "bed level (the terrain)"),
                ("mesh2d_s1", ("time", "nmesh2d_face"), s1, "m", "water level (the bed where dry)"),
                ("mesh2d_ucx", ("time", "nmesh2d_face"), ux, "m s-1", "eastward velocity"),
                ("mesh2d_ucy", ("time", "nmesh2d_face"), uy, "m s-1", "northward velocity")):
            v = nc.createVariable(name, "f4", dims)
            v[:] = data
            v.mesh = "mesh2d"
            v.location = "face"
            v.units = units
            v.long_name = long
            if name == "mesh2d_flowelem_bl":
                v.standard_name = "altitude"
    return {"path": path, "faces": int(cells.size), "steps": len(frames)}


def compare_exposure(g: dict[str, Any], before: Any, after: Any, read: dict[str, Any], *,
                     threshold_m: float | None = None) -> dict[str, Any]:
    """What gets flooded before and after a levee or a breach: the buildings, the road length and the people under
    two depth grids on the same grid (depths above the terrain, :func:`depth_over_terrain`), counted by
    :func:`aquascope.flood_exposure.exposure_payload` (the flood views' own count) from what
    :func:`aquascope.flood_exposure.read_sources` read. The two runs go in as two time steps of one UGRID file, so
    every building and stretch of road says whether it was flooded before, after, or both.

    Returns ``before``, ``after`` and ``change`` for each count, the buildings and road pieces whose state changed
    (``kept_dry``: flooded before, dry after; ``newly_flooded``: dry before, flooded after) with their outlines,
    the threshold and each source's licence and credit."""
    import os
    import tempfile

    from aquascope.flood_exposure import exposure_payload
    from aquascope.io.mesh import read_mesh

    d0 = np.maximum(np.asarray(before, dtype="f8").reshape(g["ny"], g["nx"]), 0.0)
    d1 = np.maximum(np.asarray(after, dtype="f8").reshape(g["ny"], g["nx"]), 0.0)
    zero = {"before": 0, "after": 0, "change": 0}
    if not ((d0 > 0.01).any() or (d1 > 0.01).any()):
        return {"buildings": dict(zero), "roads_km": dict(zero), "people": dict(zero), "kept_dry": [],
                "newly_flooded": [], "roads_changed": [], "sources": {}, "errors": dict(read.get("errors", {}))}
    flat = {**g, "z_burnt": g["z"]}
    run = {"h_max": np.maximum(d0, d1), "frames": [d0, d1], "times_s": [0.0, 3600.0]}
    fd, path = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    try:
        to_ugrid(path, flat, run, start="2000-01-01", title="Flood it, before and after")
        m = read_mesh(path)
        pay = exposure_payload(m, read, threshold_m=threshold_m)
        m.close()
    finally:
        os.unlink(path)
    out: dict[str, Any] = {"threshold_m": pay["threshold_m"], "sources": pay["sources"],
                           "errors": pay.get("errors", {}), "limits": pay.get("limits", {})}
    for key, src, field in (("buildings", "buildings", "per_step"), ("roads_km", "roads", "per_step_km"),
                            ("people", "population", "per_step")):
        if src in pay:
            b, a = pay[src][field][0], pay[src][field][1]
            out[key] = {"before": b, "after": a, "change": round(a - b, 3)}
    feats = pay.get("buildings", {}).get("features", [])
    out["kept_dry"] = [{"id": f["id"], "ring": f["ring"]} for f in feats if f["wet"] == "10"]
    out["newly_flooded"] = [{"id": f["id"], "ring": f["ring"]} for f in feats if f["wet"] == "01"]
    out["still_flooded"] = sum(1 for f in feats if f["wet"] == "11")
    out["roads_changed"] = [{"coords": p["coords"], "change": "kept_dry" if p["wet"] == "10" else "newly_flooded",
                             "length_m": p["length_m"]}
                            for p in pay.get("roads", {}).get("pieces", []) if p["wet"] in ("10", "01")]
    return out


def to_geotiff(path: str, g: dict[str, Any], depth: Any, *, nodata: float = -9999.0) -> dict[str, Any]:
    """Write a depth grid (m, rows north to south) as a single-band float32 GeoTIFF in Web Mercator (EPSG:3857),
    the grid's own projection: dry cells (under 1 cm) get ``nodata``."""
    win = g["window"]
    a = np.asarray(depth, dtype="f4").reshape(win["ny"], win["nx"])
    a = np.where(a > 0.01, a, np.float32(nodata)).astype("<f4")
    s = 256 * 2 ** win["zoom"]
    px = 2 * math.pi * 6378137.0 / s
    x0 = -math.pi * 6378137.0 + win["x0"] * px
    y0 = math.pi * 6378137.0 - win["y0"] * px
    h, w = a.shape
    data = a.tobytes()
    keys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857]
    nod = (repr(float(nodata)) + "\0").encode()
    entries = []   # (tag, type, count, value bytes)
    entries.append((256, 4, 1, struct.pack("<I", w)))
    entries.append((257, 4, 1, struct.pack("<I", h)))
    entries.append((258, 3, 1, struct.pack("<HH", 32, 0)))
    entries.append((259, 3, 1, struct.pack("<HH", 1, 0)))
    entries.append((262, 3, 1, struct.pack("<HH", 1, 0)))
    entries.append((273, 4, 1, b"DATA"))
    entries.append((277, 3, 1, struct.pack("<HH", 1, 0)))
    entries.append((278, 4, 1, struct.pack("<I", h)))
    entries.append((279, 4, 1, struct.pack("<I", len(data))))
    entries.append((284, 3, 1, struct.pack("<HH", 1, 0)))
    entries.append((339, 3, 1, struct.pack("<HH", 3, 0)))
    entries.append((33550, 12, 3, struct.pack("<ddd", px, px, 0.0)))
    entries.append((33922, 12, 6, struct.pack("<dddddd", 0.0, 0.0, 0.0, x0, y0, 0.0)))
    entries.append((34735, 3, len(keys), struct.pack(f"<{len(keys)}H", *keys)))
    entries.append((42113, 2, len(nod), nod))
    n = len(entries)
    ifd_size = 2 + 12 * n + 4
    extra_at = 8 + ifd_size
    extra = b""
    fields = []
    for tag, typ, count, val in entries:
        if val == b"DATA":
            fields.append((tag, typ, count, None))
            continue
        if len(val) <= 4:
            fields.append((tag, typ, count, val.ljust(4, b"\0")))
        else:
            off = extra_at + len(extra)
            extra += val + (b"\0" if len(val) % 2 else b"")
            fields.append((tag, typ, count, struct.pack("<I", off)))
    data_at = extra_at + len(extra)
    out = b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", n)
    for tag, typ, count, val in fields:
        out += struct.pack("<HHI", tag, typ, count) + (val if val is not None else struct.pack("<I", data_at))
    out += struct.pack("<I", 0) + extra + data
    with open(path, "wb") as f:
        f.write(out)
    return {"path": path, "width": w, "height": h, "crs": "EPSG:3857", "pixel_m": round(px, 4)}
