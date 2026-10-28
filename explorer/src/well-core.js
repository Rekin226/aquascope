// Drop a well (#660), the pure part: the words, the colours, the contour features, the fill's pixels, the sliders'
// log scales, the wells on the map as an undoable value, and the MODFLOW cell under a well. No DOM, no map: node
// imports this (explorer/tests/well-core.test.mjs). Every drawdown comes from aquascope.sketch.drawdown in the
// worker; this module only decides how it looks and what it says.

// The defaults are the package's (aquascope.sketch.drawdown.DEFAULTS; tests/test_sketch_drawdown.py keeps them
// equal): a municipal well in a confined sand aquifer, read after 30 days.
export const DEFAULTS = { rate: 1000, T: 500, S: 1e-4, days: 30 };
// The sliders' spans (log scales). The typical ranges (Heath 1983) sit inside them and the panel says so.
export const SPANS = { rate: [10, 100000], T: [1, 100000], S: [1e-6, 1e-2], days: [1, 3650] };
export const TYPICAL = { T: [1, 100000], S: [1e-5, 1e-3] };
export const MAX_WELLS = 10;

// The cone's colours by contour level (0.05 m outermost): light teal to deep blue, the drawdown ramp of the MODFLOW
// views read the same way (darker is deeper).
export const LEVEL_COLORS = [
  [0.05, "#8fd3c8"], [0.1, "#5fbfc0"], [0.2, "#3aa3bd"], [0.5, "#2a83b5"], [1, "#2563a6"], [2, "#284693"],
  [5, "#2b2d7c"], [10, "#2a1a60"], [20, "#230f48"], [50, "#1a0833"], [100, "#12041f"],
];

export function levelColor(level) {
  let c = LEVEL_COLORS[0][1];
  for (const [lv, col] of LEVEL_COLORS) if (level >= lv - 1e-9) c = col;
  return c;
}

// ── words ───────────────────────────────────────────────────────────────────

const group = (n) => Math.round(n).toLocaleString("en-US");

/** A pumping rate in m³/day with its L/s: "1,000 m³/day (11.6 L/s)". */
export function rateWords(m3day) {
  const ls = m3day / 86.4;
  const l = ls >= 100 ? group(ls) : ls >= 10 ? ls.toFixed(1) : ls.toPrecision(2);
  return `${group(m3day)} m³/day (${l} L/s)`;
}

/** A drawdown, m: two decimals under 10 m, one under 100. */
export function metres(m) {
  if (m === null || m === undefined || !Number.isFinite(m)) return "n/a";
  return m < 10 ? m.toFixed(2) : m < 100 ? m.toFixed(1) : group(m);
}

/** A distance: "850 m", "21.8 km". */
export function distWords(m) {
  if (!Number.isFinite(m)) return "n/a";
  return m >= 1000 ? `${(m / 1000).toFixed(m >= 100000 ? 0 : 1)} km` : `${group(m)} m`;
}

/** Days as words: "30 days", "1 year", "6 hours". */
export function daysWords(d) {
  if (d < 1) return `${Math.round(d * 24)} hours`;
  if (d >= 365 && Math.abs(d / 365 - Math.round(d / 365)) < 0.02) {
    const y = Math.round(d / 365);
    return `${y} year${y === 1 ? "" : "s"}`;
  }
  const n = d < 10 ? Number(d.toPrecision(2)) : Math.round(d);
  return `${n} day${n === 1 ? "" : "s"}`;
}

/** A storativity as written: "1e-4", "0.0051". */
export function sWords(s) {
  if (s >= 0.001) return String(Number(s.toPrecision(2)));
  const e = Math.floor(Math.log10(s));
  const m = Number((s / 10 ** e).toPrecision(2));
  return m === 1 ? `1e${e}` : `${m}e${e}`;
}

// ── the sliders: log scales from 0 to 1000 ──────────────────────────────────

export const SLIDER_MAX = 1000;

export function toSlider(value, [lo, hi]) {
  const v = Math.min(hi, Math.max(lo, value));
  return Math.round((Math.log(v / lo) / Math.log(hi / lo)) * SLIDER_MAX);
}

/** A slider position back to a value, rounded to two significant figures (so 500, 1e-4, 30 come out clean). */
export function fromSlider(pos, [lo, hi]) {
  const v = lo * (hi / lo) ** (Math.min(SLIDER_MAX, Math.max(0, pos)) / SLIDER_MAX);
  return Number(v.toPrecision(2));
}

// ── the wells on the map: one undoable value ────────────────────────────────

/**
 * The value the action log keeps for the wells (slot "wells"): { wells: [{ id, lon, lat, rate }], T, S, days }.
 * Checked and copied; throws on a bad one. Ids are kept (they name the markers) and given to wells without one.
 */
export function wellsValue(v) {
  const num = (x, what) => {
    const n = Number(x);
    if (x === null || x === undefined || x === "" || typeof x === "boolean" || !Number.isFinite(n)) {
      throw new Error(`${what} must be a number`);
    }
    return n;
  };
  const wells = ((v && v.wells) || []).map((w) => ({
    id: w.id, lon: num(w.lon, "a well's lon"), lat: num(w.lat, "a well's lat"),
    rate: num(w.rate ?? DEFAULTS.rate, "a well's rate"),
  }));
  if (wells.length > MAX_WELLS) throw new Error(`at most ${MAX_WELLS} wells`);
  for (const w of wells) {
    if (Math.abs(w.lat) > 89 || Math.abs(w.lon) > 180) throw new Error("a well must be on the map");
    if (!(w.rate > 0)) throw new Error("a well's rate must be above 0");
  }
  let next = Math.max(0, ...wells.map((w) => Number(w.id) || 0));
  for (const w of wells) if (!w.id) w.id = ++next;
  return {
    wells,
    T: num(v && v.T !== undefined ? v.T : DEFAULTS.T, "the transmissivity"),
    S: num(v && v.S !== undefined ? v.S : DEFAULTS.S, "the storativity"),
    days: num(v && v.days !== undefined ? v.days : DEFAULTS.days, "the days"),
  };
}

/** The value with one well added (it gets the next id). */
export function addWell(value, { lon, lat, rate = DEFAULTS.rate }) {
  const v = wellsValue(value || { wells: [] });
  const id = Math.max(0, ...v.wells.map((w) => w.id)) + 1;
  return wellsValue({ ...v, wells: [...v.wells, { id, lon, lat, rate }] });
}

/** The value with one well changed ({ lon, lat } or { rate }) or removed (patch null). */
export function editWell(value, id, patch) {
  const v = wellsValue(value);
  const wells = patch === null ? v.wells.filter((w) => w.id !== id)
    : v.wells.map((w) => (w.id === id ? { ...w, ...patch } : w));
  return wellsValue({ ...v, wells });
}

/** The arguments of the worker's "map" op (aquascope.sketch.drawdown.drawdown_map) for a value. */
export function mapArgs(v, { frames = false } = {}) {
  return {
    wells: v.wells.map((w) => ({ lon: w.lon, lat: w.lat, rate_m3_day: w.rate, name: `Well ${w.id}` })),
    days: v.days, transmissivity: v.T, storativity: v.S, frames,
  };
}

/** The log's line for a change from `before` to `after`. */
export function logLabel(before, after) {
  const nb = before ? before.wells.length : 0, na = after ? after.wells.length : 0;
  if (!after || !na) return nb ? "Take the wells off the map" : "No wells";
  if (na > nb) {
    const w = after.wells[after.wells.length - 1];
    return `Drop well ${w.id}, ${rateWords(w.rate)}`;
  }
  if (na < nb) return `Remove a well (${na} left)`;
  for (const w of after.wells) {
    const o = before.wells.find((x) => x.id === w.id);
    if (!o) continue;
    if (o.rate !== w.rate) return `Well ${w.id} pumps ${rateWords(w.rate)}`;
    if (o.lon !== w.lon || o.lat !== w.lat) return `Move well ${w.id}`;
  }
  if (before.T !== after.T || before.S !== after.S) return `Aquifer: T ${after.T} m²/day, S ${sWords(after.S)}`;
  if (before.days !== after.days) return `Read the cone after ${daysWords(after.days)}`;
  return "Wells";
}

// ── what goes on the map ────────────────────────────────────────────────────

/** The contours ([{level, label, lines}], from the package) as line features with their colour. */
export function contourFeatures(contours) {
  const features = [];
  for (const c of contours || []) {
    features.push({
      type: "Feature",
      properties: { level: c.level, label: c.label, color: levelColor(c.level), edge: c.level <= 0.05 + 1e-9 ? 1 : 0 },
      geometry: { type: "MultiLineString", coordinates: c.lines },
    });
  }
  return { type: "FeatureCollection", features };
}

/**
 * The fill's pixels from the package's coarse grid (north row first): the colour of the level band each cell is in,
 * more opaque for deeper water, clear outside the 5 cm line. Returns { px: Uint8ClampedArray, width, height }.
 */
export function fillPixels(image) {
  const w = image.nx, h = image.ny;
  const px = new Uint8ClampedArray(w * h * 4);
  for (let k = 0; k < w * h; k++) {
    const s = image.values[k];
    if (!(s >= 0.05)) continue;
    const c = levelColor(s);
    const t = Math.min(1, Math.log10(s / 0.05) / Math.log10(200));   // 5 cm to 10 m
    px[4 * k] = parseInt(c.slice(1, 3), 16);
    px[4 * k + 1] = parseInt(c.slice(3, 5), 16);
    px[4 * k + 2] = parseInt(c.slice(5, 7), 16);
    px[4 * k + 3] = Math.round(70 + 110 * t);
  }
  return { px, width: w, height: h };
}

/** The image source's corners from a bbox [w, s, e, n]. */
export const corners = ([w, s, e, n]) => [[w, n], [e, n], [e, s], [w, s]];

/** The time bar's label for a frame: "after 3 days". */
export const frameLabel = (days) => `after ${daysWords(days)}`;

/** The panel's one-line answer for a result. */
export function headline(res) {
  const ws = res.wells || [];
  if (!ws.length) return "";
  const deep = ws.reduce((a, b) => (b.at_well_m > a.at_well_m ? b : a));
  const who = ws.length === 1 ? "the well" : `${deep.name}, the deepest of ${ws.length},`;
  return `After ${daysWords(res.days)}, ${who} draws the water down ${metres(deep.at_well_m)} m at its face; ` +
    `the 5 cm line reaches ${distWords(res.reach_m)} out.`;
}

// ── a MODFLOW model under the well, for context ─────────────────────────────

function inRing(ring, lon, lat) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * The MODFLOW cell under a point, from the model's cell rings (flat [lon, lat, ...] with offsets per cell): its
 * index and size across, east-west and north-south, in metres (the ring's extent on a local plane), or null.
 */
export function modflowCellAt(rings, offsets, lon, lat) {
  if (!rings || !offsets) return null;
  for (let c = 0; c + 1 < offsets.length; c++) {
    const ring = [];
    for (let k = offsets[c]; k < offsets[c + 1]; k++) ring.push([rings[2 * k], rings[2 * k + 1]]);
    if (ring.length < 3 || !inRing(ring, lon, lat)) continue;
    const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
    const m = 6371008.8 * Math.PI / 180;
    const k = Math.cos((lat * Math.PI) / 180);
    return { cell: c, dx_m: (Math.max(...xs) - Math.min(...xs)) * m * k, dy_m: (Math.max(...ys) - Math.min(...ys)) * m };
  }
  return null;
}

// Peaceman (1978): a finite-difference model's well cell holds the head found at an equivalent radius of about
// 0.2 times a square cell's side from the well (Peaceman, D.W., 1978, Interpretation of well-block pressures in
// numerical reservoir simulation, SPE Journal 18(3), 183-194).
export const PEACEMAN = 0.2;

/** Where the model's well cell reads the cone: Peaceman's equivalent radius for the cell, in metres. */
export const peacemanRadius = (cell) => PEACEMAN * Math.sqrt(cell.dx_m * cell.dy_m);

/**
 * The MODFLOW context for a well in a model's cell: its cell, its size and layer, and (when the code has computed
 * it) the Theis drawdown at Peaceman's radius, what the model's well cell would read instead of the well's face.
 */
export function modflowWords(ctx, cell, atCell = null) {
  if (!ctx || !cell) return "";
  const size = `${distWords(cell.dx_m)} by ${distWords(cell.dy_m)}`;
  const layers = ctx.layers && ctx.layers.length > 1 ? ` of ${ctx.layers.length}` : "";
  const head = `Your MODFLOW model here: cell ${cell.cell + 1}, ${size}, layer ${ctx.layer}${layers}.`;
  if (atCell === null || atCell === undefined) return head;
  return `${head} Its well cell would read about ${metres(atCell)} m, the drawdown ${distWords(peacemanRadius(cell))} ` +
    "from the well (Peaceman 1978: 0.2 times the cell's side), not the well's own.";
}
