// Your model results on the globe (#555), the pure part: no DOM, no map, so node can test it.
//
// The numbers come from the package (aquascope.io.mesh, run in mesh-worker.js): the cells, the depth or
// water surface per cell, the velocities. This file decides how they are drawn and said: the blue ramp and
// its scale, which file is accepted, where a click lands (which cell), which cells carry an arrow at the
// current zoom, how a particle moves, and the words of the legend and the card.

export const MAX_FILE_BYTES = 2 * 1024 ** 3;   // the same limit as aquascope.io.mesh.MAX_FILE_BYTES
export const MAX_CELLS = 250000;               // aquascope.io.mesh.MAX_CELLS
export const ACCEPT = ".hdf,.h5,.hdf5,.nc,.nc4,.cdf";
const EXT = /\.(hdf|h5|hdf5|nc|nc4|cdf)$/i;
const GEOMETRY = /\.g\d\d\.hdf$/i;   // a HEC-RAS geometry file

// Light to deep blue, the flood-depth family (flood-depth-core.js), so a user's depth reads like the JRC one.
export const STOPS = [
  { t: 0, c: [214, 236, 250], a: 0.62 },
  { t: 0.2, c: [150, 200, 240], a: 0.74 },
  { t: 0.45, c: [88, 156, 222], a: 0.84 },
  { t: 0.7, c: [43, 108, 196], a: 0.9 },
  { t: 1, c: [20, 52, 140], a: 0.95 },
];

/** Whether a dropped file is one the viewer reads; returns null when it is, else the reason in words. */
export function checkFile(file) {
  if (!file) return "No file.";
  const name = String(file.name || "");
  if (!EXT.test(name)) {
    return `${name || "This file"} is not a model result the viewer reads. Drop a HEC-RAS plan file (.p01.hdf) or a ` +
      "UGRID NetCDF file (.nc).";
  }
  if (GEOMETRY.test(name)) {
    return `${name} is a HEC-RAS geometry file, with no results in it. Drop the plan's result file (.p01.hdf), ` +
      "or a stripped plan file (.p01.timeseries.hdf) together with this one.";
  }
  if (Number(file.size) > MAX_FILE_BYTES) {
    return `${name} is ${(Number(file.size) / 1024 ** 3).toFixed(1)} GB; the viewer reads files up to 2 GB.`;
  }
  if (!(Number(file.size) > 0)) return `${name} is empty.`;
  return null;
}

/**
 * The files of one drop or pick: the model result, and a HEC-RAS geometry file (.g01.hdf) dropped with it for
 * a stripped plan file that holds the results but not the mesh. Returns { file, geometry } (geometry null when
 * none came), or { why } in words when the drop holds no result the viewer reads.
 */
export function pickModelFiles(files) {
  const list = [...(files || [])].filter(Boolean);
  if (!list.length) return { why: "No file." };
  const geometry = list.find((f) => GEOMETRY.test(String(f.name || ""))) || null;
  const rest = list.filter((f) => f !== geometry);
  if (!rest.length) return { why: checkFile(geometry) };
  const file = rest.find((f) => checkFile(f) === null);
  if (!file) return { why: checkFile(rest[0]) };
  if (geometry && Number(geometry.size) > MAX_FILE_BYTES) return { why: checkFile({ name: geometry.name, size: geometry.size }) };
  return { file, geometry };
}

// aquascope.io.mesh's error for a plan file without its mesh (a stripped .p01.timeseries.hdf): its opening words
// (a Python test keeps the two equal) and the part it names as missing.
export const NO_GEOMETRY = "holds model results but not the mesh they sit on";

/**
 * The reader's error, as the page says it. A plan file without its mesh is told what to drop next: the full
 * plan file, or the same file with its geometry file. The 2D area whose mesh is missing is named in words,
 * not as an HDF5 path. Every other error is the package's own words.
 */
export function friendlyError(message, name = "") {
  const msg = String(message || "");
  if (!msg.includes(NO_GEOMETRY)) return msg;
  const missing = (/: (.+?) is missing\./.exec(msg) || [])[1] || "";
  const area = (/^Geometry\/2D Flow Areas\/([^/]+)/.exec(missing) || [])[1];
  const what = area && area !== "Attributes" ? ` (no mesh for the 2D area ${area})` : "";
  return `${name || "This file"} has the model results but not the mesh${what}. Drop the full plan file ` +
    "(.p01.hdf), or this file together with its geometry file (.g01.hdf).";
}

/** "BaldEagle.p18.hdf" -> "BaldEagle.p18.hdf"; long names keep their ends: "Grevelingen-FM…_map.nc". */
export function shortName(name, max = 28) {
  const s = String(name || "");
  if (s.length <= max) return s;
  const keep = max - 1;
  return `${s.slice(0, Math.ceil(keep * 0.6))}…${s.slice(s.length - Math.floor(keep * 0.4))}`;
}

// ── the scale ───────────────────────────────────────────────────────────────

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/**
 * The colour scale for a mesh: depth from 0 on a square-root scale up to its 99th percentile (one deep pool
 * does not wash the rest out), water surface linear between its 1st and 99th percentiles.
 */
export function scaleFor(values, variable = "depth") {
  const v = [];
  for (const x of values || []) if (Number.isFinite(x)) v.push(x);
  v.sort((a, b) => a - b);
  if (!v.length) return { lo: 0, hi: 1, sqrt: variable === "depth" };
  if (variable === "depth") {
    const hi = niceCeil(Math.max(quantile(v, 0.99), 1e-6));
    return { lo: 0, hi, sqrt: true };
  }
  let lo = quantile(v, 0.01), hi = quantile(v, 0.99);
  if (!(hi > lo)) { lo = v[0]; hi = v[v.length - 1] + (v[v.length - 1] === v[0] ? 1 : 0); }
  return { lo: niceFloor(lo), hi: niceCeil(hi), sqrt: false };
}

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(Math.abs(x) || 1));
  const m = Math.abs(x) / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
export function niceCeil(x) {
  if (!(x > 0)) return x <= 0 ? Math.ceil(x) : 1;
  const step = niceStep(x) / 10;
  return Math.round(Math.ceil(x / step) * step * 1e6) / 1e6;
}
export function niceFloor(x) {
  if (x === 0) return 0;
  const step = niceStep(x) / 10;
  return Math.round(Math.floor(x / step) * step * 1e6) / 1e6;
}

/** Where a value sits on the scale, 0 to 1 (null for a dry or missing cell). */
export function position(value, scale) {
  if (!Number.isFinite(value)) return null;
  const { lo, hi, sqrt } = scale;
  let t = (value - lo) / ((hi - lo) || 1);
  t = Math.min(1, Math.max(0, t));
  return sqrt ? Math.sqrt(t) : t;
}

/** [r, g, b, a] for a value, or null where there is nothing to draw. */
export function colorAt(value, scale) {
  const t = position(value, scale);
  if (t === null) return null;
  for (let k = 0; k < STOPS.length - 1; k++) {
    const a = STOPS[k], b = STOPS[k + 1];
    if (t <= b.t) {
      const f = (t - a.t) / (b.t - a.t);
      return [0, 1, 2].map((i) => Math.round(a.c[i] + (b.c[i] - a.c[i]) * f)).concat(Math.round((a.a + (b.a - a.a) * f) * 100) / 100);
    }
  }
  const last = STOPS[STOPS.length - 1];
  return [...last.c, last.a];
}

/** The scale's colours in N bins (one path per bin when the mesh is painted): bin k covers t in [k/N, (k+1)/N). */
export function bins(n = 32) {
  const out = [];
  for (let k = 0; k < n; k++) {
    const t = (k + 0.5) / n;
    let c = null;
    for (let j = 0; j < STOPS.length - 1; j++) {
      const a = STOPS[j], b = STOPS[j + 1];
      if (t <= b.t) {
        const f = (t - a.t) / (b.t - a.t);
        c = [0, 1, 2].map((i) => Math.round(a.c[i] + (b.c[i] - a.c[i]) * f)).concat(Math.round((a.a + (b.a - a.a) * f) * 100) / 100);
        break;
      }
    }
    out.push(`rgba(${c[0]},${c[1]},${c[2]},${c[3]})`);
  }
  return out;
}

export function binOf(value, scale, n = 32) {
  const t = position(value, scale);
  return t === null ? -1 : Math.min(n - 1, Math.floor(t * n));
}

export function rampCss() {
  return `linear-gradient(90deg, ${STOPS.map((s) => `rgba(${s.c.join(",")},${s.a}) ${Math.round(s.t * 100)}%`).join(", ")})`;
}

/** The legend's ticks: [{ at: percent, label }], at 0, the middle and the top of the scale. */
export function ticks(scale) {
  const vals = scale.sqrt ? [scale.lo, scale.hi / 4, scale.hi] : [scale.lo, (scale.lo + scale.hi) / 2, scale.hi];
  return vals.map((v) => ({ at: Math.round((position(v, scale) ?? 0) * 100), label: fmt(v) }));
}

// ── numbers as words ────────────────────────────────────────────────────────

export function fmt(x, digits = null) {
  if (!Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  const d = digits ?? (a >= 100 ? 0 : a >= 10 ? 1 : 2);
  return Number(x.toFixed(d)).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "1999-01-02T14:00:00" -> "2 Jan 1999 14:00" (the model's own clock; no time zone is assumed). */
export function modelTime(iso, { year = true } = {}) {
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?/.exec(String(iso || ""));
  if (!m) return String(iso || "");
  const day = `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}${year ? ` ${m[1]}` : ""}`;
  return m[4] ? `${day} ${m[4]}:${m[5]}` : day;
}

/** "37 steps, 1 Jan 1999 12:00 to 4 Jan 12:00" (the year once when both ends share it) */
export function rangeWords(times) {
  if (!times || !times.length) return "no time steps";
  if (times.length === 1) return `one step, ${modelTime(times[0])}`;
  const a = times[0], b = times[times.length - 1];
  const same = String(a).slice(0, 4) === String(b).slice(0, 4);
  return `${times.length} steps, ${modelTime(a)} to ${modelTime(b, { year: !same })}`;
}

export function variableWords(variable) {
  return variable === "depth" ? "depth" : "water surface";
}

/** A compass point for a velocity (u east, v north): "north-east". */
export function heading(u, v) {
  if (!Number.isFinite(u) || !Number.isFinite(v) || (u === 0 && v === 0)) return "";
  const names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
  const bearing = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;
  return names[Math.round(bearing / 45) % 8];
}

/** The bearing an arrow points to, degrees clockwise from north. */
export function bearing(u, v) {
  return (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;
}

// ── where a click lands ─────────────────────────────────────────────────────

/**
 * A grid over the mesh for finding the cell under a point: each bucket lists the cells whose outline box
 * touches it. mesh: { ring_offsets, ring_lon, ring_lat } (typed arrays).
 */
export function buildIndex(mesh, target = 4) {
  const off = mesh.ring_offsets, xs = mesh.ring_lon, ys = mesh.ring_lat;
  const n = off.length - 1;
  let w = Infinity, s = Infinity, e = -Infinity, nn = -Infinity;
  const boxes = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (let k = off[i]; k < off[i + 1]; k++) {
      if (xs[k] < a) a = xs[k];
      if (xs[k] > c) c = xs[k];
      if (ys[k] < b) b = ys[k];
      if (ys[k] > d) d = ys[k];
    }
    boxes.set([a, b, c, d], i * 4);
    if (a < w) w = a;
    if (b < s) s = b;
    if (c > e) e = c;
    if (d > nn) nn = d;
  }
  const side = Math.max(1, Math.round(Math.sqrt(n / target)));
  const gx = side, gy = side;
  const dx = (e - w) / gx || 1, dy = (nn - s) / gy || 1;
  const buckets = Array.from({ length: gx * gy }, () => []);
  for (let i = 0; i < n; i++) {
    const x0 = Math.max(0, Math.floor((boxes[i * 4] - w) / dx)), x1 = Math.min(gx - 1, Math.floor((boxes[i * 4 + 2] - w) / dx));
    const y0 = Math.max(0, Math.floor((boxes[i * 4 + 1] - s) / dy)), y1 = Math.min(gy - 1, Math.floor((boxes[i * 4 + 3] - s) / dy));
    for (let gxi = x0; gxi <= x1; gxi++) for (let gyi = y0; gyi <= y1; gyi++) buckets[gyi * gx + gxi].push(i);
  }
  return { bbox: [w, s, e, nn], gx, gy, dx, dy, buckets, boxes, mesh };
}

function inRing(x, y, xs, ys, a, b) {
  let inside = false;
  for (let i = a, j = b - 1; i < b; j = i++) {
    const xi = xs[i], yi = ys[i], xj = xs[j], yj = ys[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** The cell under (lon, lat), or -1. */
export function cellAt(index, lon, lat) {
  const [w, s, e, n] = index.bbox;
  if (!(lon >= w && lon <= e && lat >= s && lat <= n)) return -1;
  const gxi = Math.min(index.gx - 1, Math.floor((lon - w) / index.dx));
  const gyi = Math.min(index.gy - 1, Math.floor((lat - s) / index.dy));
  const { ring_offsets: off, ring_lon: xs, ring_lat: ys } = index.mesh;
  for (const i of index.buckets[gyi * index.gx + gxi]) {
    const b = index.boxes;
    if (lon < b[i * 4] || lon > b[i * 4 + 2] || lat < b[i * 4 + 1] || lat > b[i * 4 + 3]) continue;
    if (inRing(lon, lat, xs, ys, off[i], off[i + 1])) return i;
  }
  return -1;
}

// ── arrows and particles ────────────────────────────────────────────────────

/**
 * Which cells carry an arrow: at most one per `spacing` pixels on screen, the one nearest each grid point.
 * pts: [{ i, x, y }] the wet, moving cells in view with their screen positions.
 */
export function thinArrows(pts, spacing = 34) {
  const best = new Map();
  for (const p of pts) {
    const kx = Math.floor(p.x / spacing), ky = Math.floor(p.y / spacing);
    const cx = (kx + 0.5) * spacing, cy = (ky + 0.5) * spacing;
    const d = (p.x - cx) ** 2 + (p.y - cy) ** 2;
    const key = `${kx},${ky}`;
    const had = best.get(key);
    if (!had || d < had.d) best.set(key, { i: p.i, d });
  }
  return [...best.values()].map((b) => b.i);
}

/** An arrow's size from its speed: 0.45 at rest to 1 at the scale's top speed. */
export function arrowSize(speed, top) {
  if (!(speed > 0) || !(top > 0)) return 0.5;
  return Math.round((0.5 + 0.5 * Math.min(1, Math.sqrt(speed / top))) * 100) / 100;
}

/**
 * Move a particle one frame: (lon, lat) by velocity (u east, v north, in the file's units per second) over
 * `seconds` of exaggerated time. Returns [lon, lat]. Metres or feet per degree come from `metresPerUnit`.
 */
export function advect(lon, lat, u, v, seconds, metresPerUnit = 1) {
  const R = 6371008.8;
  const dn = v * seconds * metresPerUnit, de = u * seconds * metresPerUnit;
  const dLat = (dn / R) * 180 / Math.PI;
  const dLon = (de / (R * Math.cos(lat * Math.PI / 180))) * 180 / Math.PI;
  return [lon + dLon, lat + dLat];
}

/** Metres in one of the file's length units. */
export function metresPer(unit) {
  const u = String(unit || "").toLowerCase();
  if (u.startsWith("ft") || u.startsWith("feet") || u.startsWith("foot")) return 0.3048;
  return 1;
}

/**
 * How much faster than real time particles move so the fastest water crosses about `pxPerSecond` screen pixels
 * a second at this zoom: metresPerPixel from the map, top speed in the file's units.
 */
export function particleTimeScale(topSpeed, metresPerPixel, unitMetres = 1, pxPerSecond = 60) {
  if (!(topSpeed > 0) || !(metresPerPixel > 0)) return 0;
  return (pxPerSecond * metresPerPixel) / (topSpeed * unitMetres);
}

// ── the canvas the mesh is painted on ───────────────────────────────────────

const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (Math.max(-85, Math.min(85, lat)) * Math.PI) / 360));

/**
 * The picture's frame: the mesh box in Web Mercator, at a size where a typical cell is a few pixels across,
 * capped at `max` pixels on the long side. Returns { width, height, x(lon), y(lat), corners }.
 */
export function canvasFrame(bbox, nCells, max = 2048) {
  const [w, s, e, n] = bbox;
  const pad = Math.max((e - w), (n - s)) * 0.002;
  const W = w - pad, E = e + pad, S = s - pad, N = n + pad;
  const mx = (E - W) * Math.PI / 180, my = mercY(N) - mercY(S);
  const aspect = mx / (my || 1e-9);
  const perSide = Math.sqrt(Math.max(1, nCells));
  const want = Math.min(max, Math.max(1024, Math.round(perSide * 32)));
  const width = aspect >= 1 ? want : Math.max(16, Math.round(want * aspect));
  const height = aspect >= 1 ? Math.max(16, Math.round(want / aspect)) : want;
  const yN = mercY(N), ySpan = my || 1e-9;
  return {
    width, height,
    x: (lon) => ((lon - W) / (E - W)) * width,
    y: (lat) => ((yN - mercY(lat)) / ySpan) * height,
    corners: [[W, N], [E, N], [E, S], [W, S]],
  };
}

/**
 * Which cell covers each pixel of the picture: an Int32Array of width x height, -1 where no cell is. Pixel
 * centres are tested (scanline fill, even-odd), so neighbouring cells share no seam and leave no gap. Done
 * once per mesh: each time step then only looks colours up (paintPixels), which is what keeps playing smooth
 * on a mesh of a quarter of a million cells.
 * px, py: the outline points in the picture's pixels (ring_offsets as for the mesh).
 */
export function rasterizeIds(off, px, py, width, height) {
  const ids = new Int32Array(width * height).fill(-1);
  const xs = [];
  for (let i = 0; i < off.length - 1; i++) {
    const a = off[i], b = off[i + 1];
    if (b - a < 3) continue;
    let y0 = Infinity, y1 = -Infinity;
    for (let k = a; k < b; k++) { if (py[k] < y0) y0 = py[k]; if (py[k] > y1) y1 = py[k]; }
    const r0 = Math.max(0, Math.ceil(y0 - 0.5)), r1 = Math.min(height - 1, Math.floor(y1 - 0.5));
    for (let r = r0; r <= r1; r++) {
      const yc = r + 0.5;
      xs.length = 0;
      for (let k = a, j = b - 1; k < b; j = k++) {
        const ya = py[j], yb = py[k];
        if ((ya > yc) !== (yb > yc)) xs.push(px[j] + ((yc - ya) * (px[k] - px[j])) / (yb - ya));
      }
      xs.sort((m, n) => m - n);
      for (let q = 0; q + 1 < xs.length; q += 2) {
        const c0 = Math.max(0, Math.ceil(xs[q] - 0.5)), c1 = Math.min(width - 1, Math.floor(xs[q + 1] - 0.5));
        const row = r * width;
        for (let c = c0; c <= c1; c++) ids[row + c] = i;
      }
    }
  }
  return ids;
}

/** The scale's bin colours as packed RGBA (little-endian Uint32, as an ImageData's buffer reads them). */
export function packedBins(n = 32) {
  return bins(n).map((css) => {
    const [r, g, b, a] = css.slice(5, -1).split(",").map(Number);
    return ((Math.round(a * 255) << 24) | (b << 16) | (g << 8) | r) >>> 0;
  });
}

/** Fill a picture: each pixel takes the colour of its cell's bin (0, transparent, for none or dry). */
export function paintPixels(out, ids, cellColour) {
  for (let p = 0; p < ids.length; p++) {
    const i = ids[p];
    out[p] = i < 0 ? 0 : cellColour[i];
  }
  return out;
}

// ── the card ────────────────────────────────────────────────────────────────

/**
 * What the card says about one cell. meta: the mesh's meta; at: { value, u, v } now (or null for the largest
 * view); largest: { value, speed } over the run; series: { value: [...] } or null.
 */
export function cellFacts(meta, cell, { at = null, largest = {}, when = null } = {}) {
  const variable = meta.variable;
  const unit = (meta.units && meta.units[variable]) || "";
  const vunit = (meta.units && meta.units.velocity) || "";
  const area = (meta.areas || []).find((a) => cell >= a.start && cell < a.start + a.cells);
  const words = variableWords(variable);
  const title = area && meta.areas.length > 1 ? `${area.name}, cell ${cell - area.start + 1}` : `Cell ${cell + 1}`;
  let status;
  if (at) {
    status = Number.isFinite(at.value)
      ? `${cap(words)} ${fmt(at.value)}${unit ? ` ${unit}` : ""} at ${modelTime(when)}`
      : `Dry at ${modelTime(when)}`;
  } else {
    status = Number.isFinite(largest.value)
      ? `Largest ${words} ${fmt(largest.value)}${unit ? ` ${unit}` : ""} over the run`
      : "Dry all through the run";
  }
  let note = "";
  if (at && Number.isFinite(at.u) && Number.isFinite(at.v)) {
    const sp = Math.hypot(at.u, at.v);
    note = sp >= 0.005 ? `Velocity ${fmt(sp)}${vunit ? ` ${vunit}` : ""}, towards the ${heading(at.u, at.v)}.`
      : "Still water at this step.";
  } else if (Number.isFinite(largest.speed)) {
    note = `Largest speed ${fmt(largest.speed)}${vunit ? ` ${vunit}` : ""}.`;
  }
  const figure = Number.isFinite(largest.value)
    ? { value: fmt(largest.value), unit, label: `largest ${words}` } : null;
  return { title, status, note, figure, unit };
}

function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

/** "HEC-RAS 6.5 February 2024" -> "HEC-RAS 6.5"; other model names pass through, shortened. */
export function modelShort(model) {
  const s = String(model || "");
  const m = /^(HEC-RAS\s+[\d.]+)/i.exec(s);
  return m ? m[1] : shortName(s, 32);
}

/** The legend line under the ramp. */
export function legendLine(meta, { step = null } = {}) {
  const words = variableWords(meta.variable);
  const cells = `${Number(meta.cells).toLocaleString("en-GB")} cells`;
  if (step === null || !meta.times || !meta.times.length) return `Largest ${words} in each cell over the run · ${cells}`;
  return `${cap(words)} at ${modelTime(meta.times[step])} · step ${step + 1} of ${meta.times.length}`;
}
