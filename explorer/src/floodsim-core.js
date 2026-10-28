// Flood it (#658), the pure part: no DOM, no map, so node can test it and the fallback worker can import it.
//
// The local-inertial scheme of Bates, Horritt and Fewtrell (2010) in the q-centred form of de Almeida et al.
// (2012), the same steps as aquascope/floodsim.py's simulate() (a node test runs both on one grid and compares
// them): the face flow depth, the semi-implicit Manning friction, the edge faces (a wall, a free outfall or a
// normal-depth outfall), a flux limiter that keeps depths non-negative and the volume exact, the inflows (a flow or
// a table of flows), and the CFL step dt = alpha dx / sqrt(g h_max). FLOODSIM_WGSL is the same scheme for the GPU
// (floodsim-gpu.js), in f32.
//
// The grid itself (the valley, the channels burnt in, the inflows and the edges) is the package's:
// aquascope.floodsim.plan_window and prepare_grid, run in the worker. Here too: the Terrarium decode, the
// near-steady-state rule, the cell velocities, a run's frames kept compactly, and the UGRID NetCDF file that hands
// a run to the flood views (#619 to #622).

export const G = 9.81;
export const ALPHA = 0.7;
export const THETA = 0.7;
export const H_DRY = 1e-3;
export const N_FLOODPLAIN = 0.06;
export const WET_M = 0.05;
export const STEADY_TOL = 0.01;
export const STEADY_EVERY_S = 3600;
export const TERRARIUM = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png";
export const EARTH_CIRCUMFERENCE_M = 40075016.686;
export const LABEL = "screening flood: no bathymetry, terrain errors of metres, every river at the same return period";
export const FLOODSIM_CREDIT = "Local-inertial scheme (Bates et al. 2010; de Almeida et al. 2012) run on your device; " +
  "terrain: Terrarium tiles (Mapzen/Tilezen joerd: SRTM, GMTED2010, ETOPO1 and others); rivers: GEOGLOWS v2 " +
  "(TDX-Hydro, CC BY-SA 4.0); flows: GEOGLOWS v2 retrospective annual maxima (CC BY-NC-SA 4.0), Log-Pearson III. " +
  "A screening flood, not a design flood.";

// ── the run's words ─────────────────────────────────────────────────────────

/** Model time in words: "45 min", "27 h", "3 h 20 min". */
export function hm(s) {
  const m = Math.round(Number(s) / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ""}`;
}

/** A run's label with its article: "the 100-year flow", but "954 m³/s" as it is. */
export function theLabel(label) {
  const l = String(label || "");
  return /^[\d,.]+ m³\/s/.test(l) ? l : `the ${l}`;
}

/**
 * The flow line of a run: what was chosen and, unless the words already say it, the flow at the reach once.
 * run: { words, choice, g: { click: { flow_cms, return_period }, sources } }; q formats a flow in m³/s.
 */
export function flowLine(run, q) {
  const c = run.g.click;
  const said = run.choice.flow_cms || run.choice.forecast === "peak" || String(run.words).includes(q(c.flow_cms));
  const about = c.return_period && !run.choice.return_period ? `, about the ${Math.round(c.return_period)}-year flow,` : "";
  return { words: run.words, rest: `${about} at this reach${said ? "" : ` (${q(c.flow_cms)})`}. ` +
    `Every river in view at the same return period: ${run.g.sources.length} inflows.` };
}

/**
 * The time bar's label for frame time t (s). A steady run holds one flow until the flood stops growing, so its
 * clock is the hours since the water was turned on, not a time of day: "5 h into a steady run", and its last frame
 * "steady after 27 h". A hydrograph's run counts from the hydrograph's start: "+5 h".
 */
export function clockLabel(t, { steady = false, steadyAt = null, last = false } = {}) {
  if (!steady) return `+${hm(t)}`;
  if (last && steadyAt) return `steady after ${hm(steadyAt)}`;
  return `${hm(t)} into a steady run`;
}

/** The time bar's note for a run (short: on a phone it sits under the sheet; the run's flow is on the card). */
export function clockSummary({ steady = true } = {}) {
  return steady ? "Flood it: hours since the water was turned on." : "Flood it: hours from the hydrograph's start.";
}

/** The legend's line over the depth ramp. */
export function depthTag(label, t, { steady = false } = {}) {
  if (t === null || t === undefined) return `Depth of ${theLabel(label)}, a screening flood`;
  return steady ? `Depth after ${hm(t)} of a steady run at ${theLabel(label)}, a screening flood`
    : `Depth at +${hm(t)} of ${theLabel(label)}, a screening flood`;
}

// ── terrain ─────────────────────────────────────────────────────────────────

/** Terrarium RGB(A) bytes -> metres: R*256 + G + B/256 - 32768. `rgba` has `channels` bytes a pixel. */
export function terrariumElevation(rgba, channels = 4) {
  const n = Math.floor(rgba.length / channels);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const k = i * channels;
    out[i] = rgba[k] * 256 + rgba[k + 1] + rgba[k + 2] / 256 - 32768;
  }
  return out;
}

export const pixelSizeM = (lat, zoom) => EARTH_CIRCUMFERENCE_M * Math.cos((lat * Math.PI) / 180) / (256 * 2 ** zoom);

export function worldPx(lon, lat, zoom) {
  const s = 256 * 2 ** zoom;
  const r = (lat * Math.PI) / 180;
  return [((lon + 180) / 360) * s, ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * s];
}

export function pxLonLat(x, y, zoom) {
  const s = 256 * 2 ** zoom;
  return [(x / s) * 360 - 180, (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / s))) * 180) / Math.PI];
}

/** The Terrarium tiles a window { x0, y0, nx, ny, zoom } (world pixels) needs, and where it sits in their mosaic. */
export function windowTiles(win) {
  const { x0, y0, nx, ny, zoom } = win;
  const tx0 = Math.floor(x0 / 256), ty0 = Math.floor(y0 / 256);
  const tx1 = Math.floor((x0 + nx - 1) / 256), ty1 = Math.floor((y0 + ny - 1) / 256);
  const n = 2 ** zoom;
  const tiles = [];
  for (let ty = ty0; ty <= ty1; ty++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      const x = ((tx % n) + n) % n;
      tiles.push({ x, y: ty, col: tx - tx0, row: ty - ty0,
        url: TERRARIUM.replace("{z}", zoom).replace("{x}", x).replace("{y}", ty) });
    }
  }
  return { tiles, cols: tx1 - tx0 + 1, rows: ty1 - ty0 + 1, ox: x0 - tx0 * 256, oy: y0 - ty0 * 256 };
}

/** A window's corners for an image source: [[W, N], [E, N], [E, S], [W, S]]. */
export function corners(win) {
  const [w, n] = pxLonLat(win.x0, win.y0, win.zoom);
  const [e, s] = pxLonLat(win.x0 + win.nx, win.y0 + win.ny, win.zoom);
  return [[w, n], [e, n], [e, s], [w, s]];
}

// ── the solver (CPU) ────────────────────────────────────────────────────────

/** A source's flow at time t (s): a number, or a table of flows every tableDtS from 0, linear, held at the ends. */
export function flowAt(q, t, tableDtS) {
  if (typeof q === "number") return q;
  if (q.length === 1) return q[0];
  const k = t / tableDtS;
  if (k <= 0) return q[0];
  if (k >= q.length - 1) return q[q.length - 1];
  const i = Math.floor(k);
  return q[i] + (q[i + 1] - q[i]) * (k - i);
}

/** Edge codes by side (n, s: nx faces; w, e: ny faces): NaN (or null) free, 0 wall, > 0 normal depth at that slope. */
export function edgeCodes(nx, ny, edges) {
  const side = (k, len) => {
    const v = edges && edges[k];
    const out = new Float64Array(len).fill(NaN);
    if (v === undefined || v === null) return out;
    if (typeof v === "number") return out.fill(v);
    for (let i = 0; i < len; i++) out[i] = v[i] === null || v[i] === undefined ? NaN : v[i];
    return out;
  };
  return { n: side("n", nx), s: side("s", nx), w: side("w", ny), e: side("e", ny) };
}

/**
 * A run on the CPU (the fallback, and the twin the node test checks against Python).
 * grid: { z (array ny*nx, rows north to south), nx, ny, dx, n (array or number), h0, edges }
 * opts: { sources: [[row, col, flow or table]], tableDtS, theta, alpha, dtMax }
 */
export function createCpuSim(grid, opts = {}) {
  const { nx, ny, dx } = grid;
  const z = Float64Array.from(grid.z);
  const N = nx * ny;
  const n = typeof grid.n === "number" || grid.n === undefined
    ? new Float64Array(N).fill(grid.n === undefined ? N_FLOODPLAIN : grid.n) : Float64Array.from(grid.n);
  const codes = edgeCodes(nx, ny, grid.edges);
  const theta = opts.theta ?? THETA, alpha = opts.alpha ?? ALPHA, dtMax = opts.dtMax ?? 10;
  const tableDtS = opts.tableDtS ?? 3600;
  const h = grid.h0 ? Float64Array.from(grid.h0) : new Float64Array(N);
  let qx = new Float64Array(ny * (nx + 1)), qy = new Float64Array((ny + 1) * nx);
  let qxn = new Float64Array(qx.length), qyn = new Float64Array(qy.length);
  const f = new Float64Array(N);
  const hMax = Float64Array.from(h);
  const sources = (opts.sources || []).map(([r, c, q]) => [r * nx + c, typeof q === "number" ? q : Float64Array.from(q)]);
  const st = { t: 0, steps: 0, vIn: 0, vOut: 0, v0: h.reduce((a, b) => a + b, 0) * dx * dx };

  function face(zl, zr, el, er, nl, nr, qOld, qLo, qHi, dt) {
    const hf = Math.max(el, er) - Math.max(zl, zr);
    if (!(hf > H_DRY)) return 0;
    const qc = theta * qOld + 0.5 * (1 - theta) * (qLo + qHi);
    const nf = 0.5 * (nl + nr);
    return (qc - (G * hf * dt * (er - el)) / dx) / (1 + (G * dt * nf * nf * Math.abs(qOld)) / hf ** (7 / 3));
  }

  // An edge face after its code: free keeps q (already outflow only), a wall carries nothing, normal depth carries
  // sign h^(5/3) sqrt(S) / n from the edge cell.
  function edge(q, code, i, sign) {
    if (Number.isNaN(code)) return q;
    if (!(code > 0)) return 0;
    return h[i] > H_DRY ? (sign * h[i] ** (5 / 3) * Math.sqrt(code)) / Math.max(n[i], 1e-6) : 0;
  }

  function step(dt) {
    for (let r = 0; r < ny; r++) {
      const row = r * nx, fr = r * (nx + 1);
      for (let k = 0; k <= nx; k++) {
        const il = row + Math.max(k - 1, 0), ir = row + Math.min(k, nx - 1);
        const zl = z[il], zr = z[ir];
        const el = k === 0 ? zl : zl + h[il];
        const er = k === nx ? zr : zr + h[ir];
        const qo = qx[fr + k];
        const lo = k === 0 ? qo : qx[fr + k - 1], hi = k === nx ? qo : qx[fr + k + 1];
        let q = face(zl, zr, el, er, n[il], n[ir], qo, lo, hi, dt);
        if (k === 0) q = edge(Math.min(q, 0), codes.w[r], ir, -1);
        else if (k === nx) q = edge(Math.max(q, 0), codes.e[r], il, 1);
        qxn[fr + k] = q;
      }
    }
    for (let k = 0; k <= ny; k++) {
      for (let c = 0; c < nx; c++) {
        const it = Math.max(k - 1, 0) * nx + c, ib = Math.min(k, ny - 1) * nx + c;
        const zt = z[it], zb = z[ib];
        const et = k === 0 ? zt : zt + h[it];
        const eb = k === ny ? zb : zb + h[ib];
        const qo = qy[k * nx + c];
        const lo = k === 0 ? qo : qy[(k - 1) * nx + c], hi = k === ny ? qo : qy[(k + 1) * nx + c];
        let q = face(zt, zb, et, eb, n[it], n[ib], qo, lo, hi, dt);
        if (k === 0) q = edge(Math.min(q, 0), codes.n[c], ib, -1);
        else if (k === ny) q = edge(Math.max(q, 0), codes.s[c], it, 1);
        qyn[k * nx + c] = q;
      }
    }
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const i = r * nx + c, fr = r * (nx + 1);
        const out = (Math.max(qxn[fr + c + 1], 0) + Math.max(-qxn[fr + c], 0)
          + Math.max(qyn[(r + 1) * nx + c], 0) + Math.max(-qyn[r * nx + c], 0)) * dt / dx;
        f[i] = out > h[i] ? h[i] / out : 1;
      }
    }
    for (let r = 0; r < ny; r++) {
      const fr = r * (nx + 1);
      for (let k = 0; k <= nx; k++) {
        const q = qxn[fr + k];
        qxn[fr + k] = q * (q > 0 ? f[r * nx + Math.max(k - 1, 0)] : f[r * nx + Math.min(k, nx - 1)]);
      }
    }
    for (let k = 0; k <= ny; k++) {
      for (let c = 0; c < nx; c++) {
        const q = qyn[k * nx + c];
        qyn[k * nx + c] = q * (q > 0 ? f[Math.max(k - 1, 0) * nx + c] : f[Math.min(k, ny - 1) * nx + c]);
      }
    }
    let out = 0;
    for (let r = 0; r < ny; r++) out += qxn[r * (nx + 1) + nx] - qxn[r * (nx + 1)];
    for (let c = 0; c < nx; c++) out += qyn[ny * nx + c] - qyn[c];
    st.vOut += out * dx * dt;
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const i = r * nx + c, fr = r * (nx + 1);
        const v = h[i] + (dt / dx) * (qxn[fr + c] - qxn[fr + c + 1] + qyn[r * nx + c] - qyn[(r + 1) * nx + c]);
        h[i] = v > 0 ? v : 0;
      }
    }
    for (const [i, q] of sources) {
      const qq = flowAt(q, st.t, tableDtS);
      h[i] += (qq * dt) / (dx * dx);
      st.vIn += qq * dt;
    }
    for (let i = 0; i < N; i++) if (h[i] > hMax[i]) hMax[i] = h[i];
    [qx, qxn] = [qxn, qx];
    [qy, qyn] = [qyn, qy];
  }

  function dtNow(remaining) {
    let hm = 0;
    for (let i = 0; i < N; i++) if (h[i] > hm) hm = h[i];
    return Math.min(dtMax, remaining, hm > H_DRY ? (alpha * dx) / Math.sqrt(G * hm) : dtMax);
  }

  /** Advance to time `until` (s), or by at most `maxSteps`; returns the state. */
  function advance(until, maxSteps = Infinity) {
    let k = 0;
    while (st.t < until - 1e-9 && k < maxSteps) {
      const dt = dtNow(until - st.t);
      step(dt);
      st.t += dt;
      st.steps++;
      k++;
    }
    return st;
  }

  function volume() {
    let v = 0;
    for (let i = 0; i < N; i++) v += h[i];
    const final = v * dx * dx;
    return { initial_m3: st.v0, inflow_m3: st.vIn, outflow_m3: st.vOut, final_m3: final,
      error_m3: final - (st.v0 + st.vIn - st.vOut) };
  }

  return { h, hMax, state: st, advance, volume, nx, ny, dx, faces: () => ({ qx, qy }) };
}

// ── a run's clock: frames, checks and the steady state ─────────────────────

/** The model times of a run's frames up to `hours`: every 15 minutes for 3 hours, then hourly (as frame_times). */
export function frameTimes(hours) {
  const end = hours * 3600, out = [];
  for (let t = 0; t <= Math.min(end, 3 * 3600) + 1e-6; t += 900) out.push(t);
  for (let t = 4 * 3600; t <= end + 1e-6; t += 3600) out.push(t);
  return out;
}

/** The times a run stops at: its frames and, for a steady run, the hourly checks; sorted, without repeats. */
export function stopTimes(hours, steady) {
  const set = new Set(frameTimes(hours).map((t) => Math.round(t)));
  if (steady) for (let t = STEADY_EVERY_S; t <= hours * 3600 + 1e-6; t += STEADY_EVERY_S) set.add(Math.round(t));
  return [...set].sort((a, b) => a - b);
}

/** The wet area (m2, deeper than WET_M) and the volume (m3) of a depth grid. */
export function areaVolume(h, dx, wet = WET_M) {
  let cells = 0, sum = 0;
  for (let i = 0; i < h.length; i++) { const v = h[i]; if (v > wet) cells++; sum += v; }
  return { wet_m2: cells * dx * dx, volume_m3: sum * dx * dx };
}

/** The near-steady-state rule: the wet area and the volume each changed by less than tol since the last check. */
export function steadyCheck(prev, cur, dx, tol = STEADY_TOL) {
  const dArea = Math.abs(cur.wet_m2 - prev.wet_m2) / Math.max(cur.wet_m2, dx * dx);
  const dVolume = Math.abs(cur.volume_m3 - prev.volume_m3) / Math.max(cur.volume_m3, 1e-9);
  return { d_area: dArea, d_volume: dVolume, steady: dArea < tol && dVolume < tol };
}

/** Velocities at the cell centres (east u, north v) from the face discharges, NaN shallower than `dry`. */
export function cellVelocity(h, qx, qy, nx, ny, dry = 0.01) {
  const u = new Float32Array(nx * ny), v = new Float32Array(nx * ny);
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      const i = r * nx + c;
      if (!(h[i] > dry)) { u[i] = NaN; v[i] = NaN; continue; }
      u[i] = (0.5 * (qx[r * (nx + 1) + c] + qx[r * (nx + 1) + c + 1])) / h[i];
      v[i] = (-0.5 * (qy[r * nx + c] + qy[(r + 1) * nx + c])) / h[i];
    }
  }
  return { u, v };
}

/** A frame kept compactly: the cells wetter than `dry`, their depth and velocity. */
export function packFrame(t, h, u, v, dry = 0.01) {
  let k = 0;
  for (let i = 0; i < h.length; i++) if (h[i] > dry) k++;
  const idx = new Uint32Array(k), d = new Float32Array(k), uu = new Float32Array(k), vv = new Float32Array(k);
  k = 0;
  for (let i = 0; i < h.length; i++) {
    if (!(h[i] > dry)) continue;
    idx[k] = i; d[k] = h[i]; uu[k] = u ? u[i] : NaN; vv[k] = v ? v[i] : NaN; k++;
  }
  return { t, idx, h: d, u: uu, v: vv };
}

/** A packed frame's depths on the whole grid. */
export function unpackDepth(frame, cells) {
  const out = new Float32Array(cells);
  for (let k = 0; k < frame.idx.length; k++) out[frame.idx[k]] = frame.h[k];
  return out;
}

/** Sources that share a cell, merged (their flows or tables added): the GPU adds one inflow a cell. */
export function mergeSources(sources) {
  const at = new Map();
  for (const [r, c, q] of sources) {
    const key = `${r},${c}`;
    const tab = typeof q === "number" ? [q] : Array.from(q);
    if (!at.has(key)) { at.set(key, [r, c, tab]); continue; }
    const old = at.get(key)[2];
    const len = Math.max(old.length, tab.length);
    const sum = [];
    for (let k = 0; k < len; k++) sum.push((old[Math.min(k, old.length - 1)] || 0) + (tab[Math.min(k, tab.length - 1)] || 0));
    at.get(key)[2] = sum;
  }
  return [...at.values()];
}

/**
 * The run's loop, the same for the GPU and the worker: advance to each stop (stopTimes), keep a frame where one is
 * due (onFrame gets it packed, with the full depth grid), and, for a steady run, stop at the first hourly check
 * that meets the near-steady-state rule. engine: { advance(t), depth(), faces() } (sync or async).
 * Returns { t, steadyAt (s or null), checks, frames, hMax? }.
 */
export async function runLoop(engine, { nx, ny, dx, hours, steady = true, onFrame = null, onCheck = null, keep = true, cancelled = () => false }) {
  const frameSet = new Set(frameTimes(hours).map((t) => Math.round(t)));
  const stops = stopTimes(hours, steady);
  const frames = [];
  const checks = [];
  let prev = { wet_m2: 0, volume_m3: 0 };
  let steadyAt = null;
  let t = 0;
  const take = async (time) => {
    const h = await engine.depth();
    let u = null, v = null;
    if (keep) { const { qx, qy } = await engine.faces(); ({ u, v } = cellVelocity(h, qx, qy, nx, ny)); }
    const f = keep ? packFrame(time, h, u, v) : { t: time };
    frames.push(f);
    if (onFrame) await onFrame(f, h);
    return h;
  };
  await take(0);
  for (const stop of stops) {
    if (stop === 0) continue;
    if (cancelled()) break;
    await engine.advance(stop);
    t = stop;
    let h = null;
    if (frameSet.has(stop)) h = await take(stop);
    if (steady && stop % STEADY_EVERY_S === 0) {
      if (!h) h = await engine.depth();
      const cur = areaVolume(h, dx);
      const c = { t_s: stop, ...cur, ...steadyCheck(prev, cur, dx) };
      checks.push(c);
      if (onCheck) onCheck(c);
      prev = cur;
      if (c.steady && stop >= STEADY_EVERY_S) {
        if (!frameSet.has(stop)) await take(stop);
        steadyAt = stop;
        break;
      }
    }
  }
  return { t, steadyAt, checks, frames };
}

// ── the summary and the drawing ─────────────────────────────────────────────

/** The water above the terrain (aquascope.floodsim.depth_over_terrain): bed + h - ground, never below 0. */
export function depthOverTerrain(h, bed, ground) {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) { const v = bed[i] + h[i] - ground[i]; out[i] = v > 0 ? v : 0; }
  return out;
}

/** The wet area (km2) and the deepest depth of a depth grid. */
export function summary(h, dx, wet = WET_M) {
  let cells = 0, deepest = 0;
  for (const v of h) { if (v > wet) cells++; if (v > deepest) deepest = v; }
  return { wet_km2: (cells * dx * dx) / 1e6, max_m: deepest, wet_cells: cells };
}

/** The rows and columns of the wet cells' box, or null when nothing is wet. */
export function wetBox(h, nx, ny, wet = WET_M) {
  let r0 = ny, r1 = -1, c0 = nx, c1 = -1;
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      if (!(h[r * nx + c] > wet)) continue;
      if (r < r0) r0 = r; if (r > r1) r1 = r; if (c < c0) c0 = c; if (c > c1) c1 = c;
    }
  }
  return r1 < 0 ? null : { r0, r1, c0, c1 };
}

/** The wall rule's numbers, the same as aquascope.floodsim.WALL_RULE: deeper than wetM, further than `near` cells from
 * an inflow and `gap` faces from an open edge, at `minFaces` wall faces at least. */
export const WALL_RULE = { wetM: 0.1, near: 40, gap: 3, minFaces: 3 };

/**
 * Where the flood is held by a wall of the grid (aquascope.floodsim.wall_contact, the same rule): the walled edge
 * faces with water deeper than `wetM` behind them, away from the rivers that enter (`near` cells) and from the open
 * edges where the river leaves (`gap` faces). There the valley goes on past the grid, and the wall holds water that
 * would spread. Returns { faces, sides: { n, s, w, e } (faces per side), held } with held true at `minFaces` or more.
 * h: depth above the terrain (ny*nx, rows north to south); sources: [[row, col, ...]].
 */
export function wallContact(h, nx, ny, edges, sources = [], rule = WALL_RULE) {
  const { wetM, near, gap, minFaces } = { ...WALL_RULE, ...rule };
  const codes = edgeCodes(nx, ny, edges);
  const far = (r, c) => sources.every(([sr, sc]) => Math.max(Math.abs(sr - r), Math.abs(sc - c)) > near);
  const sides = { n: 0, s: 0, w: 0, e: 0 };
  const side = (k, len, cell) => {
    const code = codes[k];
    for (let i = 0; i < len; i++) {
      if (code[i] !== 0) continue;
      let open = false;
      for (let d = -gap; d <= gap && !open; d++) open = i + d >= 0 && i + d < len && code[i + d] !== 0;
      if (open) continue;
      const [r, c] = cell(i);
      if (h[r * nx + c] > wetM && far(r, c)) sides[k]++;
    }
  };
  side("n", nx, (i) => [0, i]);
  side("s", nx, (i) => [ny - 1, i]);
  side("w", ny, (i) => [i, 0]);
  side("e", ny, (i) => [i, nx - 1]);
  const faces = sides.n + sides.s + sides.w + sides.e;
  return { faces, sides, held: faces >= minFaces };
}

/** Whether the flood is held by a wall of the grid (wallContact's `held`). */
export function wetAtWall(h, nx, ny, edges, sources = [], rule = WALL_RULE) {
  return wallContact(h, nx, ny, edges, sources, rule).held;
}

const SIDE_WORDS = { n: "north", s: "south", w: "west", e: "east" };

/**
 * The warning in plain words, or "" when the flood is not held by a wall: what it means and what to do.
 * contact: wallContact's answer; dx: the cell's size in m.
 */
export function wallWords(contact, dx) {
  if (!contact || !contact.held) return "";
  const where = Object.entries(contact.sides).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]).map(([k]) => SIDE_WORDS[k]);
  const km = (contact.faces * dx) / 1000;
  const along = km >= 1 ? `${km.toFixed(1)} km` : `${Math.round(km * 1000)} m`;
  const sides = where.length > 1 ? `${where.slice(0, -1).join(", ")} and ${where.at(-1)} edges` : `${where[0]} edge`;
  return `The water reaches the grid's ${sides} along ${along}. The valley goes on past it, and the edge holds the ` +
    "water like a wall, so the flood there is deeper and narrower than it would be. Flood a reach nearer that edge to see it.";
}

// ── the hand-off to the flood views: a UGRID NetCDF-3 file ─────────────────
// The run's wet cells (and a ring of dry ones, so a section shows the banks) as the faces of a UGRID 2D mesh in
// longitude and latitude, with the bed level, the water level and the velocity at every frame. aquascope.io.mesh
// reads it like any model result, so the flood views (smooth depth, hazard, arrival, sections, exposure) show a
// Flood it run exactly as they show a HEC-RAS run. aquascope.floodsim.to_ugrid writes the same file in Python.

/** The cells a file keeps: wetter than `dry` at some time, plus `ring` cells around them. */
export function keptCells(hMax, nx, ny, ring = 3, dry = 0.01) {
  const keep = new Uint8Array(nx * ny);
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      if (!(hMax[r * nx + c] > dry)) continue;
      for (let dr = -ring; dr <= ring; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= ny) continue;
        for (let dc = -ring; dc <= ring; dc++) {
          const cc = c + dc;
          if (cc >= 0 && cc < nx) keep[rr * nx + cc] = 1;
        }
      }
    }
  }
  const out = [];
  for (let i = 0; i < keep.length; i++) if (keep[i]) out.push(i);
  return Uint32Array.from(out);
}

// NetCDF-3 classic (CDF-1): big-endian, a header of dimensions, attributes and variables, then the data.
const NC = { BYTE: 1, CHAR: 2, SHORT: 3, INT: 4, FLOAT: 5, DOUBLE: 6, DIM: 10, VAR: 11, ATT: 12 };

class Bytes {
  constructor() { this.parts = []; this.len = 0; }
  push(u8) { this.parts.push(u8); this.len += u8.length; }
  int(v) { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v); this.push(b); }
  name(s) {
    const enc = new TextEncoder().encode(s);
    this.int(enc.length);
    this.push(enc);
    this.pad(enc.length);
  }
  pad(n) { const p = (4 - (n % 4)) % 4; if (p) this.push(new Uint8Array(p)); }
  concat() {
    const out = new Uint8Array(this.len);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  }
}

function attList(b, atts) {
  const keys = Object.keys(atts || {});
  if (!keys.length) { b.int(0); b.int(0); return; }
  b.int(NC.ATT); b.int(keys.length);
  for (const k of keys) {
    const v = atts[k];
    b.name(k);
    if (typeof v === "string") {
      const enc = new TextEncoder().encode(v);
      b.int(NC.CHAR); b.int(enc.length); b.push(enc); b.pad(enc.length);
    } else {
      const arr = Array.isArray(v) ? v : [v];
      const isInt = arr.every((x) => Number.isInteger(x));
      b.int(isInt ? NC.INT : NC.DOUBLE); b.int(arr.length);
      const data = new Uint8Array(arr.length * (isInt ? 4 : 8));
      const dv = new DataView(data.buffer);
      arr.forEach((x, i) => (isInt ? dv.setInt32(i * 4, x) : dv.setFloat64(i * 8, x)));
      b.push(data);
    }
  }
}

/** A NetCDF-3 classic file from { dims: [[name, size]], atts, vars: [{ name, dims, type, atts, data }] }. */
export function netcdf3(spec) {
  const size = { [NC.INT]: 4, [NC.FLOAT]: 4, [NC.DOUBLE]: 8 };
  const dimIndex = new Map(spec.dims.map(([k], i) => [k, i]));
  const dimSize = new Map(spec.dims);
  const vsize = spec.vars.map((v) => {
    const n = v.dims.reduce((a, d) => a * dimSize.get(d), 1) * size[v.type];
    return n + ((4 - (n % 4)) % 4);
  });
  const header = (offsets) => {
    const b = new Bytes();
    b.push(new Uint8Array([0x43, 0x44, 0x46, 0x01]));
    b.int(0);                                   // no record dimension
    b.int(NC.DIM); b.int(spec.dims.length);
    for (const [k, n] of spec.dims) { b.name(k); b.int(n); }
    attList(b, spec.atts);
    b.int(NC.VAR); b.int(spec.vars.length);
    spec.vars.forEach((v, i) => {
      b.name(v.name);
      b.int(v.dims.length);
      for (const d of v.dims) b.int(dimIndex.get(d));
      attList(b, v.atts);
      b.int(v.type); b.int(vsize[i]); b.int(offsets[i]);
    });
    return b;
  };
  const first = header(spec.vars.map(() => 0)).len;
  const offsets = [];
  let o = first;
  for (const s of vsize) { offsets.push(o); o += s; }
  const b = header(offsets);
  spec.vars.forEach((v, i) => {
    const data = new Uint8Array(vsize[i]);
    const dv = new DataView(data.buffer);
    const src = v.data;
    for (let k = 0; k < src.length; k++) {
      if (v.type === NC.INT) dv.setInt32(k * 4, src[k]);
      else if (v.type === NC.FLOAT) dv.setFloat32(k * 4, src[k]);
      else dv.setFloat64(k * 8, src[k]);
    }
    b.push(data);
  });
  return b.concat();
}

export const NC_TYPES = NC;

/**
 * The UGRID file of a run. run: { win { x0, y0, nx, ny, zoom }, bed (the burnt bed, ny*nx), ground (the terrain),
 * hMax, frames: [packFrame], start (ISO date of time 0), title }. The file's bed is the terrain and its level the
 * water over the burnt bed, never below the terrain, as aquascope.floodsim.to_ugrid. Returns { bytes, cells }.
 */
export function ugridFile(run) {
  const { win } = run;
  const { nx, ny, zoom } = win;
  const cells = keptCells(run.hMax, nx, ny);
  const nf = cells.length;
  const node = new Map();
  const nodeXY = [];
  const conn = new Int32Array(nf * 4);
  const fx = new Float64Array(nf), fy = new Float64Array(nf), bed = new Float32Array(nf);
  const nodeOf = (r, c) => {
    const key = r * (nx + 1) + c;
    if (!node.has(key)) { node.set(key, nodeXY.length); nodeXY.push(pxLonLat(win.x0 + c, win.y0 + r, zoom)); }
    return node.get(key);
  };
  const slot = new Int32Array(nx * ny).fill(-1);
  for (let k = 0; k < nf; k++) {
    const i = cells[k], r = Math.floor(i / nx), c = i % nx;
    slot[i] = k;
    // Counter-clockwise from the south-west corner (rows run north to south).
    conn[4 * k] = nodeOf(r + 1, c); conn[4 * k + 1] = nodeOf(r + 1, c + 1);
    conn[4 * k + 2] = nodeOf(r, c + 1); conn[4 * k + 3] = nodeOf(r, c);
    const [lo, la] = pxLonLat(win.x0 + c + 0.5, win.y0 + r + 0.5, zoom);
    fx[k] = lo; fy[k] = la; bed[k] = run.ground ? run.ground[i] : run.bed[i];
  }
  const burnt = new Float32Array(nf);
  for (let k = 0; k < nf; k++) burnt[k] = run.bed[cells[k]];
  const nt = run.frames.length;
  const s1 = new Float32Array(nt * nf), ux = new Float32Array(nt * nf), uy = new Float32Array(nt * nf);
  run.frames.forEach((f, j) => {
    const base = j * nf;
    for (let k = 0; k < nf; k++) { s1[base + k] = bed[k]; ux[base + k] = 0; uy[base + k] = 0; }
    for (let k = 0; k < f.idx.length; k++) {
      const m = slot[f.idx[k]];
      if (m < 0) continue;
      s1[base + m] = Math.max(bed[m], burnt[m] + f.h[k]);
      ux[base + m] = Number.isFinite(f.u[k]) ? f.u[k] : 0;
      uy[base + m] = Number.isFinite(f.v[k]) ? f.v[k] : 0;
    }
  });
  const bytes = netcdf3({
    dims: [["nmesh2d_node", nodeXY.length], ["nmesh2d_face", nf], ["max_nmesh2d_face_nodes", 4], ["time", nt]],
    atts: { title: run.title || "Flood it", source: "aquascope Flood it (local-inertial screening flood)",
      Conventions: "CF-1.8 UGRID-1.0", comment: LABEL },
    vars: [
      { name: "mesh2d", dims: [], type: NC.INT, data: [0], atts: { cf_role: "mesh_topology", topology_dimension: 2,
        node_coordinates: "mesh2d_node_x mesh2d_node_y", face_node_connectivity: "mesh2d_face_nodes",
        face_coordinates: "mesh2d_face_x mesh2d_face_y" } },
      { name: "mesh2d_node_x", dims: ["nmesh2d_node"], type: NC.DOUBLE, data: nodeXY.map((p) => p[0]),
        atts: { units: "degrees_east", standard_name: "longitude" } },
      { name: "mesh2d_node_y", dims: ["nmesh2d_node"], type: NC.DOUBLE, data: nodeXY.map((p) => p[1]),
        atts: { units: "degrees_north", standard_name: "latitude" } },
      { name: "mesh2d_face_x", dims: ["nmesh2d_face"], type: NC.DOUBLE, data: fx, atts: { units: "degrees_east", standard_name: "longitude" } },
      { name: "mesh2d_face_y", dims: ["nmesh2d_face"], type: NC.DOUBLE, data: fy, atts: { units: "degrees_north", standard_name: "latitude" } },
      { name: "mesh2d_face_nodes", dims: ["nmesh2d_face", "max_nmesh2d_face_nodes"], type: NC.INT, data: conn,
        atts: { cf_role: "face_node_connectivity", start_index: 0 } },
      { name: "time", dims: ["time"], type: NC.DOUBLE, data: run.frames.map((f) => f.t),
        atts: { units: `seconds since ${run.start || "2000-01-01"} 00:00:00`, standard_name: "time" } },
      { name: "mesh2d_flowelem_bl", dims: ["nmesh2d_face"], type: NC.FLOAT, data: bed,
        atts: { mesh: "mesh2d", location: "face", units: "m", standard_name: "altitude", long_name: "bed level (the terrain)" } },
      { name: "mesh2d_s1", dims: ["time", "nmesh2d_face"], type: NC.FLOAT, data: s1,
        atts: { mesh: "mesh2d", location: "face", units: "m", long_name: "water level (the bed where dry)" } },
      { name: "mesh2d_ucx", dims: ["time", "nmesh2d_face"], type: NC.FLOAT, data: ux,
        atts: { mesh: "mesh2d", location: "face", units: "m s-1", long_name: "eastward velocity" } },
      { name: "mesh2d_ucy", dims: ["time", "nmesh2d_face"], type: NC.FLOAT, data: uy,
        atts: { mesh: "mesh2d", location: "face", units: "m s-1", long_name: "northward velocity" } },
    ],
  });
  return { bytes, cells: nf };
}

// ── the GPU scheme (WGSL) ───────────────────────────────────────────────────
// Seven storage buffers (the default limit is eight a stage), f32: zn (the bed, Manning's n, the edge codes in the
// order n, s, w, e with -1 for a free outfall, the inflow cells, and their flow tables), h and hmax (cells), lim
// (the limiter, cells), qO and qN (the x faces, ny*(nx+1), then the y faces, (ny+1)*nx: the old and the new
// discharges, swapped every step through two bind groups), and st: { dt, t, tEnd, hmaxBits (an atomic u32, the
// largest depth's bits: positive floats order like their bits), n (the steps taken) }. Per step: stepDt (one
// thread), faces (both directions), limit, scale, depth, inflow (one thread an inflow). Ghost cells, the edge codes,
// the limiter and the inflows as in createCpuSim.

export const FLOODSIM_WGSL = /* wgsl */ `
struct P { nx: u32, ny: u32, nsrc: u32, nt: u32, dx: f32, theta: f32, alpha: f32, dtMax: f32, g: f32, hDry: f32, tabDt: f32, pad: f32 };
struct S { dt: f32, t: f32, tEnd: f32, hmaxBits: atomic<u32>, n: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> st: S;
@group(0) @binding(2) var<storage, read> zn: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
@group(0) @binding(4) var<storage, read_write> hmax: array<f32>;
@group(0) @binding(5) var<storage, read_write> lim: array<f32>;
@group(0) @binding(6) var<storage, read> qO: array<f32>;
@group(0) @binding(7) var<storage, read_write> qN: array<f32>;

fn ncells() -> u32 { return p.nx * p.ny; }
fn nqx() -> u32 { return (p.nx + 1u) * p.ny; }
fn oEdge() -> u32 { return 2u * ncells(); }
fn oSrc() -> u32 { return 2u * ncells() + 2u * p.nx + 2u * p.ny; }
fn oTab() -> u32 { return oSrc() + p.nsrc; }

@compute @workgroup_size(1)
fn stepDt() {
  let hm = bitcast<f32>(atomicLoad(&st.hmaxBits));
  var dt = p.dtMax;
  if (hm > p.hDry) { dt = min(dt, p.alpha * p.dx / sqrt(p.g * hm)); }
  dt = max(min(dt, st.tEnd - st.t), 0.0);
  st.dt = dt;
  st.t = st.t + dt;
  if (dt > 0.0) { st.n = st.n + 1u; }
  atomicStore(&st.hmaxBits, 0u);
}

fn face(zl: f32, zr: f32, el: f32, er: f32, nl: f32, nr: f32, qo: f32, lo: f32, hi: f32, dt: f32) -> f32 {
  let hf = max(el, er) - max(zl, zr);
  if (!(hf > p.hDry)) { return 0.0; }
  let qc = p.theta * qo + 0.5 * (1.0 - p.theta) * (lo + hi);
  let nf = 0.5 * (nl + nr);
  return (qc - p.g * hf * dt * (er - el) / p.dx) / (1.0 + p.g * dt * nf * nf * abs(qo) / pow(hf, 7.0 / 3.0));
}

// An edge face after its code (-1 free: q as it is; 0 wall; > 0 normal depth at that slope, from cell i).
fn edge(q: f32, code: f32, i: u32, sgn: f32) -> f32 {
  if (code < 0.0) { return q; }
  if (!(code > 0.0)) { return 0.0; }
  let d = h[i];
  if (!(d > p.hDry)) { return 0.0; }
  return sgn * pow(d, 5.0 / 3.0) * sqrt(code) / max(zn[ncells() + i], 1e-6);
}

@compute @workgroup_size(16, 16)
fn faces(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = p.nx; let ny = p.ny; let dt = st.dt; let N = ncells(); let X = nqx(); let E = oEdge();
  let k = id.x; let r = id.y;
  if (r < ny && k <= nx) {            // x face k (0..nx) of row r
    let il = r * nx + select(k - 1u, 0u, k == 0u);
    let ir = r * nx + min(k, nx - 1u);
    let zl = zn[il]; let zr = zn[ir];
    let el = select(zl + h[il], zl, k == 0u);
    let er = select(zr + h[ir], zr, k == nx);
    let fi = r * (nx + 1u) + k;
    let qo = qO[fi];
    let lo = select(qO[max(fi, 1u) - 1u], qo, k == 0u);
    let hi = select(qO[min(fi + 1u, X - 1u)], qo, k == nx);
    var q = face(zl, zr, el, er, zn[N + il], zn[N + ir], qo, lo, hi, dt);
    if (k == 0u) { q = edge(min(q, 0.0), zn[E + 2u * nx + r], ir, -1.0); }
    else if (k == nx) { q = edge(max(q, 0.0), zn[E + 2u * nx + ny + r], il, 1.0); }
    qN[fi] = q;
  }
  let c = k; let kk = r;
  if (c < nx && kk <= ny) {           // y face kk (0..ny) of column c
    let it = select(kk - 1u, 0u, kk == 0u) * nx + c;
    let ib = min(kk, ny - 1u) * nx + c;
    let zt = zn[it]; let zb = zn[ib];
    let et = select(zt + h[it], zt, kk == 0u);
    let eb = select(zb + h[ib], zb, kk == ny);
    let fi = kk * nx + c;
    let qo = qO[X + fi];
    let lo = select(qO[X + max(fi, nx) - nx], qo, kk == 0u);
    let hi = select(qO[X + min(fi + nx, (ny + 1u) * nx - 1u)], qo, kk == ny);
    var q = face(zt, zb, et, eb, zn[N + it], zn[N + ib], qo, lo, hi, dt);
    if (kk == 0u) { q = edge(min(q, 0.0), zn[E + c], ib, -1.0); }
    else if (kk == ny) { q = edge(max(q, 0.0), zn[E + nx + c], it, 1.0); }
    qN[X + fi] = q;
  }
}

@compute @workgroup_size(16, 16)
fn limit(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x; let r = id.y; let nx = p.nx; let X = nqx();
  if (c >= nx || r >= p.ny) { return; }
  let i = r * nx + c; let fr = r * (nx + 1u);
  let o = (max(qN[fr + c + 1u], 0.0) + max(-qN[fr + c], 0.0) + max(qN[X + (r + 1u) * nx + c], 0.0) + max(-qN[X + r * nx + c], 0.0)) * st.dt / p.dx;
  lim[i] = select(1.0, h[i] / o, o > h[i]);
}

@compute @workgroup_size(16, 16)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
  let nx = p.nx; let ny = p.ny; let k = id.x; let r = id.y; let X = nqx();
  if (r < ny && k <= nx) {
    let fi = r * (nx + 1u) + k;
    let q = qN[fi];
    let donor = select(r * nx + min(k, nx - 1u), r * nx + select(k - 1u, 0u, k == 0u), q > 0.0);
    qN[fi] = q * lim[donor];
  }
  let c = k; let kk = r;
  if (c < nx && kk <= ny) {
    let fi = X + kk * nx + c;
    let q = qN[fi];
    let donor = select(min(kk, ny - 1u) * nx + c, select(kk - 1u, 0u, kk == 0u) * nx + c, q > 0.0);
    qN[fi] = q * lim[donor];
  }
}

var<workgroup> wmax: atomic<u32>;

@compute @workgroup_size(16, 16)
fn depth(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wmax, 0u); }
  workgroupBarrier();
  let c = id.x; let r = id.y; let nx = p.nx; let X = nqx();
  if (c < nx && r < p.ny) {
    let i = r * nx + c; let fr = r * (nx + 1u); let dt = st.dt;
    var v = h[i] + dt / p.dx * (qN[fr + c] - qN[fr + c + 1u] + qN[X + r * nx + c] - qN[X + (r + 1u) * nx + c]);
    v = max(v, 0.0);
    h[i] = v;
    hmax[i] = max(hmax[i], v);
    atomicMax(&wmax, bitcast<u32>(v));
  }
  workgroupBarrier();
  if (li == 0u) { atomicMax(&st.hmaxBits, atomicLoad(&wmax)); }
}

// The inflows, one thread each, after the depths: the flow at the start of the step, read from the source's table.
@compute @workgroup_size(64)
fn inflow(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= p.nsrc) { return; }
  let i = u32(zn[oSrc() + k]);
  let base = oTab() + k * p.nt;
  let t0 = (st.t - st.dt) / p.tabDt;
  var q = zn[base];
  if (p.nt > 1u) {
    let last = f32(p.nt - 1u);
    let x = clamp(t0, 0.0, last);
    let j = min(u32(floor(x)), p.nt - 2u);
    q = zn[base + j] + (zn[base + j + 1u] - zn[base + j]) * (x - f32(j));
  }
  let v = h[i] + q * st.dt / (p.dx * p.dx);
  h[i] = v;
  hmax[i] = max(hmax[i], v);
  atomicMax(&st.hmaxBits, bitcast<u32>(v));
}
`;
