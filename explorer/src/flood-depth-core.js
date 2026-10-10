// Flood depth where floods are forecast (#554), the pure part: no DOM, no map,
// so node can test it and the decoding worker can import it.
//
// When Floods ahead (#546) says a reach will pass its 10-, 25-, 50- or
// 100-year flow, the JRC CEMS-GloFAS flood depth map of the nearest return
// period at or below that class is drawn around the reach: 10, 20, 50 or 100
// years (JRC has no 2- or 5-year maps). aquascope/flood_depth.py is the
// package side (the CLI and the MCP tool); this file keeps the same tile list,
// radius rule and colour ramp (a Python test keeps them equal), cuts the map
// into half-degree cells, and paints each cell from the depth windows the
// worker reads.

import { classOn } from "./floods-ahead-core.js?v=__BUILD__";

export const DEPTH_BASE = "https://data.source.coop/nlebovits/jrc-glofas";
export const DEPTH_RETURN_PERIODS = [10, 20, 50, 75, 100, 200, 500];
export const DEPTH_LABEL = "may flood in the next 15 days, model estimate";
// geotiff.js (MIT), pinned: the same build the world river status uses.
export const GEOTIFF_MODULE = "https://cdn.jsdelivr.net/npm/geotiff@3.0.5/+esm";
export const MIN_DEPTH_M = 0.05;
// 90 m pixels: below this zoom a flood plain is a few screen pixels, so nothing is read.
export const DEPTH_MINZOOM = 7;
// The map is cut into cells of half a degree (600 x 600 pixels of the map), each its own picture.
export const CELL_DEG = 0.5;
export const CELL_PX = 600;
// At most this many cells on the map at once, the nearest to the middle of the view first.
export const MAX_CELLS = 16;
// The outer share of each circle fades out, so the edge of "around the reach" reads as soft, not as a cut.
export const FADE = 0.3;

export const DEPTH_CREDIT = {
  label: "Flood depth",
  attribution: 'JRC CEMS-GloFAS global river flood hazard maps v2.1.2, &copy; European Union ' +
    '(<a href="https://data.jrc.ec.europa.eu/collection/id-0054">JRC</a>), read from the ' +
    '<a href="https://source.coop/nlebovits/jrc-glofas">Source Cooperative</a> mirror',
  licence: "CC BY 4.0",
};

// Depth (m) -> colour and opacity, light to deep blue, with the JRC classes' break points (1, 3 and 10 m) as
// stops. The same as aquascope.flood_depth.RAMP.
export const RAMP = [
  { depth_m: 0.05, hex: "#bfe0f7", alpha: 0.55 },
  { depth_m: 1.0, hex: "#6aaee8", alpha: 0.74 },
  { depth_m: 3.0, hex: "#2f7ccc", alpha: 0.86 },
  { depth_m: 10.0, hex: "#173f99", alpha: 0.93 },
];

// Every tile of the mirror (the same 271 for each return period), as aquascope.flood_depth.DEPTH_TILES.
export const DEPTH_TILES = [
  "ID1_N70_W180", "ID2_N80_W170", "ID3_N70_W170", "ID4_N60_W170", "ID5_N80_W160", "ID6_N70_W160",
  "ID7_N60_W160", "ID8_N80_W150", "ID9_N70_W150", "ID10_N60_W150", "ID11_N80_W140", "ID12_N70_W140",
  "ID13_N60_W140", "ID14_N80_W130", "ID15_N70_W130", "ID16_N60_W130", "ID17_N50_W130", "ID18_N40_W130",
  "ID19_N80_W120", "ID20_N70_W120", "ID21_N60_W120", "ID22_N50_W120", "ID23_N40_W120", "ID24_N30_W120",
  "ID25_N80_W110", "ID26_N70_W110", "ID27_N60_W110", "ID28_N50_W110", "ID29_N40_W110", "ID30_N30_W110",
  "ID31_N20_W110", "ID32_N90_W100", "ID33_N80_W100", "ID34_N70_W100", "ID35_N60_W100", "ID36_N50_W100",
  "ID37_N40_W100", "ID38_N30_W100", "ID39_N20_W100", "ID40_N90_W90", "ID41_N80_W90", "ID42_N70_W90",
  "ID43_N60_W90", "ID44_N50_W90", "ID45_N40_W90", "ID46_N30_W90", "ID47_N20_W90", "ID48_N10_W90",
  "ID49_N0_W90", "ID50_N90_W80", "ID51_N80_W80", "ID52_N70_W80", "ID53_N60_W80", "ID54_N50_W80",
  "ID55_N40_W80", "ID56_N30_W80", "ID57_N20_W80", "ID58_N10_W80", "ID59_N0_W80", "ID60_S10_W80",
  "ID61_S20_W80", "ID62_S30_W80", "ID63_S40_W80", "ID64_S50_W80", "ID65_N90_W70", "ID66_N80_W70",
  "ID67_N70_W70", "ID68_N60_W70", "ID69_N50_W70", "ID70_N20_W70", "ID71_N10_W70", "ID72_N0_W70",
  "ID73_S10_W70", "ID74_S20_W70", "ID75_S30_W70", "ID76_S40_W70", "ID77_S50_W70", "ID78_N80_W60",
  "ID79_N70_W60", "ID80_N60_W60", "ID81_N50_W60", "ID82_N10_W60", "ID83_N0_W60", "ID84_S10_W60",
  "ID85_S20_W60", "ID86_S30_W60", "ID87_N80_W50", "ID88_N70_W50", "ID89_N60_W50", "ID90_N10_W50",
  "ID91_N0_W50", "ID92_S10_W50", "ID93_S20_W50", "ID94_N80_W40", "ID95_N70_W40", "ID96_N0_W40",
  "ID97_S10_W40", "ID98_N80_W30", "ID99_N70_W30", "ID100_N80_W20", "ID101_N70_W20", "ID102_N60_W20",
  "ID103_N30_W20", "ID104_N20_W20", "ID105_N10_W20", "ID106_N60_W10", "ID107_N50_W10", "ID108_N40_W10",
  "ID109_N30_W10", "ID110_N20_W10", "ID111_N10_W10", "ID112_N70_W0", "ID113_N60_W0", "ID114_N50_W0",
  "ID115_N40_W0", "ID116_N30_W0", "ID117_N20_W0", "ID118_N10_W0", "ID119_N0_W0", "ID120_N70_E10",
  "ID121_N60_E10", "ID122_N50_E10", "ID123_N40_E10", "ID124_N30_E10", "ID125_N20_E10", "ID126_N10_E10",
  "ID127_N0_E10", "ID128_S10_E10", "ID129_S20_E10", "ID130_S30_E10", "ID131_N80_E20", "ID132_N70_E20",
  "ID133_N60_E20", "ID134_N50_E20", "ID135_N40_E20", "ID136_N30_E20", "ID137_N20_E20", "ID138_N10_E20",
  "ID139_N0_E20", "ID140_S10_E20", "ID141_S20_E20", "ID142_S30_E20", "ID143_N80_E30", "ID144_N70_E30",
  "ID145_N60_E30", "ID146_N50_E30", "ID147_N40_E30", "ID148_N30_E30", "ID149_N20_E30", "ID150_N10_E30",
  "ID151_N0_E30", "ID152_S10_E30", "ID153_S20_E30", "ID154_S30_E30", "ID155_N70_E40", "ID156_N60_E40",
  "ID157_N50_E40", "ID158_N40_E40", "ID159_N30_E40", "ID160_N20_E40", "ID161_N10_E40", "ID162_N0_E40",
  "ID163_S10_E40", "ID164_S20_E40", "ID165_N70_E50", "ID166_N60_E50", "ID167_N50_E50", "ID168_N40_E50",
  "ID169_N30_E50", "ID170_N20_E50", "ID171_N10_E50", "ID172_S10_E50", "ID173_N80_E60", "ID174_N70_E60",
  "ID175_N60_E60", "ID176_N50_E60", "ID177_N40_E60", "ID178_N30_E60", "ID179_N80_E70", "ID180_N70_E70",
  "ID181_N60_E70", "ID182_N50_E70", "ID183_N40_E70", "ID184_N30_E70", "ID185_N20_E70", "ID186_N10_E70",
  "ID187_N80_E80", "ID188_N70_E80", "ID189_N60_E80", "ID190_N50_E80", "ID191_N40_E80", "ID192_N30_E80",
  "ID193_N20_E80", "ID194_N10_E80", "ID195_N80_E90", "ID196_N70_E90", "ID197_N60_E90", "ID198_N50_E90",
  "ID199_N40_E90", "ID200_N30_E90", "ID201_N20_E90", "ID202_N10_E90", "ID203_N0_E90", "ID204_N80_E100",
  "ID205_N70_E100", "ID206_N60_E100", "ID207_N50_E100", "ID208_N40_E100", "ID209_N30_E100", "ID210_N20_E100",
  "ID211_N10_E100", "ID212_N0_E100", "ID213_N80_E110", "ID214_N70_E110", "ID215_N60_E110", "ID216_N50_E110",
  "ID217_N40_E110", "ID218_N30_E110", "ID219_N20_E110", "ID220_N10_E110", "ID221_N0_E110", "ID222_S10_E110",
  "ID223_S20_E110", "ID224_S30_E110", "ID225_N80_E120", "ID226_N70_E120", "ID227_N60_E120", "ID228_N50_E120",
  "ID229_N40_E120", "ID230_N30_E120", "ID231_N20_E120", "ID232_N10_E120", "ID233_N0_E120", "ID234_S10_E120",
  "ID235_S20_E120", "ID236_S30_E120", "ID237_N80_E130", "ID238_N70_E130", "ID239_N60_E130", "ID240_N50_E130",
  "ID241_N40_E130", "ID242_N0_E130", "ID243_S10_E130", "ID244_S20_E130", "ID245_S30_E130", "ID246_N80_E140",
  "ID247_N70_E140", "ID248_N60_E140", "ID249_N50_E140", "ID250_N40_E140", "ID251_N0_E140", "ID252_S10_E140",
  "ID253_S20_E140", "ID254_S30_E140", "ID255_S40_E140", "ID256_N80_E150", "ID257_N70_E150", "ID258_N60_E150",
  "ID259_N0_E150", "ID260_S10_E150", "ID261_S20_E150", "ID262_S30_E150", "ID263_N80_E160", "ID264_N70_E160",
  "ID265_N60_E160", "ID266_S40_E160", "ID267_N80_E170", "ID268_N70_E170", "ID269_N60_E170", "ID270_S30_E170",
  "ID271_S40_E170",
];

const TILE_INDEX = new Map(DEPTH_TILES.map((n) => {
  const [, la, lo] = n.split("_");
  const top = Number(la.slice(1)) * (la[0] === "N" ? 1 : -1);
  const left = Number(lo.slice(1)) * (lo[0] === "E" ? 1 : -1);
  return [`${top},${left}`, n];
}));

/** The 10-degree tile holding a point, or null over the sea and outside 60 S to 80 N. */
export function tileFor(lat, lon) {
  return TILE_INDEX.get(`${Math.ceil(lat / 10) * 10},${Math.floor(lon / 10) * 10}`) || null;
}

export const depthUrl = (name, rp) => `${DEPTH_BASE}/depth-rp${rp}/${name}/${name}_RP${rp}_depth.tif`;

/** The depth map for a forecast class: the largest return period not above it, or null below 10 years. */
export function depthReturnPeriod(cls) {
  const c = Number(cls) || 0;
  let out = null;
  for (const rp of DEPTH_RETURN_PERIODS) if (rp <= c) out = rp;
  return out;
}

/** How far around a reach the depth is drawn, km: 1.5 per Strahler order above 3, 2.5 to 10 (aquascope.flood_depth). */
export function reachRadiusKm(order) {
  const o = Number.isFinite(Number(order)) && order !== null && order !== "" ? Math.trunc(Number(order)) : 5;
  return Math.min(10, Math.max(2.5, 1.5 * (o - 3)));
}

const KM_PER_DEG = 111.32;
const cosd = (lat) => Math.max(0.05, Math.cos((lat * Math.PI) / 180));

/** The reaches with a depth map on forecast day `i` (-1: the 15-day peak), deepest class first. */
export function activeReaches(features, i) {
  const out = [];
  for (const f of features || []) {
    const p = f.properties || {};
    const cls = classOn(p, i);
    const rp = depthReturnPeriod(cls);
    if (!rp) continue;
    const [lon, lat] = f.geometry.coordinates;
    out.push({ id: Number(p.river_id), lon, lat, order: p.order, cls, rp, r: reachRadiusKm(p.order), props: p });
  }
  return out.sort((a, b) => b.rp - a.rp || b.cls - a.cls);
}

/** A reach's circle as [west, south, east, north]. */
export function diskBox(lon, lat, rKm) {
  const dlat = rKm / KM_PER_DEG, dlon = rKm / (KM_PER_DEG * cosd(lat));
  return [lon - dlon, lat - dlat, lon + dlon, lat + dlat];
}

export const cellKey = (ix, iy) => `${ix}:${iy}`;

/** [west, south, east, north] of a cell. */
export function cellBox(key) {
  const [ix, iy] = key.split(":").map(Number);
  return [ix * CELL_DEG, iy * CELL_DEG, (ix + 1) * CELL_DEG, (iy + 1) * CELL_DEG];
}

/** The cells the circles touch: key -> { key, box, reaches } (a reach sits in every cell its circle reaches). */
export function cellsFor(reaches) {
  const cells = new Map();
  for (const r of reaches || []) {
    const [w, s, e, n] = diskBox(r.lon, r.lat, r.r);
    for (let ix = Math.floor(w / CELL_DEG); ix <= Math.floor(e / CELL_DEG); ix++) {
      for (let iy = Math.floor(s / CELL_DEG); iy <= Math.floor(n / CELL_DEG); iy++) {
        const key = cellKey(ix, iy);
        if (!cells.has(key)) cells.set(key, { key, box: cellBox(key), reaches: [] });
        cells.get(key).reaches.push(r);
      }
    }
  }
  return cells;
}

/** What a cell needs read: one window per return period, each from the one tile the cell sits in. */
export function cellParts(cell) {
  const [w, s, e, n] = cell.box;
  const name = tileFor((s + n) / 2, (w + e) / 2);
  if (!name) return [];
  const rps = [...new Set(cell.reaches.map((r) => r.rp))].sort((a, b) => a - b);
  return rps.map((rp) => ({ rp, url: depthUrl(name, rp) }));
}

/** A cell's "signature": which reaches it draws and from which maps. The same signature is the same picture. */
export const cellSignature = (cell) => cell.reaches.map((r) => `${r.id}@${r.rp}`).sort().join(",");

/** The nearest cells to the middle of the view, at most `max`, among those inside `bounds` [w, s, e, n]. */
export function cellsInView(cells, bounds, center, max = MAX_CELLS) {
  const [w, s, e, n] = bounds;
  const [cx, cy] = center;
  const out = [];
  for (const c of cells.values()) {
    const [cw, cs, ce, cn] = c.box;
    if (ce < w || cw > e || cn < s || cs > n) continue;
    const dx = ((cw + ce) / 2 - cx) * cosd(cy), dy = (cs + cn) / 2 - cy;
    out.push([dx * dx + dy * dy, c]);
  }
  return out.sort((a, b) => a[0] - b[0]).slice(0, max).map((x) => x[1]);
}

// ── the colours ─────────────────────────────────────────────────────────────

const STOPS = RAMP.map((s) => ({ d: s.depth_m, c: [1, 3, 5].map((i) => parseInt(s.hex.slice(i, i + 2), 16)), a: s.alpha }));

/** [r, g, b, alpha 0-1] for a depth, or null for dry (aquascope.flood_depth.ramp_color). */
export function rampColor(depth) {
  if (!(depth >= MIN_DEPTH_M)) return null;
  const last = STOPS[STOPS.length - 1];
  if (depth >= last.d) return [...last.c, last.a];
  for (let k = 0; k < STOPS.length - 1; k++) {
    const a = STOPS[k], b = STOPS[k + 1];
    if (depth <= b.d) {
      const t = Math.max(0, (depth - a.d) / (b.d - a.d));
      return [0, 1, 2].map((i) => Math.round(a.c[i] + (b.c[i] - a.c[i]) * t)).concat(Math.round((a.a + (b.a - a.a) * t) * 1000) / 1000);
    }
  }
  return null;
}

/** The legend's gradient, as a CSS linear-gradient over 0 to 10 m on a square-root scale (where the ramp's stops sit). */
export function rampCss() {
  const at = (d) => `${Math.round(Math.sqrt(d / 10) * 100)}%`;
  return `linear-gradient(90deg, ${STOPS.map((s) => `rgba(${s.c.join(",")},${s.a}) ${at(s.d)}`).join(", ")})`;
}

// ── the picture of one cell ─────────────────────────────────────────────────

// Web Mercator, so a cell's picture lines up on the map at every latitude: row y of h lies at this latitude.
const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const latOfMercY = (y) => (Math.atan(Math.sinh(y)) * 180) / Math.PI;
export function rowLats(s, n, h) {
  const top = mercY(n), bottom = mercY(s);
  const out = new Float64Array(h);
  for (let y = 0; y < h; y++) out[y] = latOfMercY(top + ((y + 0.5) / h) * (bottom - top));
  return out;
}

/**
 * Paint one cell: inside each reach's circle the depth of that reach's map, the deepest map where circles
 * overlap, faded towards the rim. `grids` maps a return period to { data, x0, y0, dx, dy, width, height }:
 * the window the worker read (x0, y0 its top-left corner, dy negative).
 * Returns { rgba, depthCm, rpAt, wet, max }: the picture, the depth (cm) and the map (return period) per pixel.
 */
export function paintCell(cell, grids, w = CELL_PX, h = CELL_PX) {
  const [cw, cs, ce, cn] = cell.box;
  const lats = rowLats(cs, cn, h);
  const rpAt = new Uint16Array(w * h);
  const fade = new Float32Array(w * h);
  const pxDeg = (ce - cw) / w;
  for (const r of cell.reaches) {
    if (!grids.has(r.rp)) continue;
    const [bw, bs, be, bn] = diskBox(r.lon, r.lat, r.r);
    const x0 = Math.max(0, Math.floor((bw - cw) / pxDeg)), x1 = Math.min(w - 1, Math.ceil((be - cw) / pxDeg));
    if (x1 < x0) continue;
    const kx = KM_PER_DEG * cosd(r.lat);
    for (let y = 0; y < h; y++) {
      const lat = lats[y];
      if (lat < bs || lat > bn) continue;
      const dyKm = (lat - r.lat) * KM_PER_DEG;
      for (let x = x0; x <= x1; x++) {
        const dxKm = (cw + (x + 0.5) * pxDeg - r.lon) * kx;
        const d = Math.sqrt(dxKm * dxKm + dyKm * dyKm);
        if (d > r.r) continue;
        const f = Math.min(1, (r.r - d) / (FADE * r.r));
        const i = y * w + x;
        if (r.rp > rpAt[i]) rpAt[i] = r.rp;
        if (f > fade[i]) fade[i] = f;
      }
    }
  }
  const rgba = new Uint8ClampedArray(w * h * 4);
  const depthCm = new Uint16Array(w * h);
  let wet = 0, max = 0;
  for (let y = 0; y < h; y++) {
    const lat = lats[y];
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const rp = rpAt[i];
      if (!rp) continue;
      const g = grids.get(rp);
      const col = Math.floor((cw + (x + 0.5) * pxDeg - g.x0) / g.dx);
      const row = Math.floor((lat - g.y0) / g.dy);
      if (col < 0 || row < 0 || col >= g.width || row >= g.height) continue;
      const v = g.data[row * g.width + col];
      const c = rampColor(v);
      if (!c) continue;
      wet++;
      if (v > max) max = v;
      depthCm[i] = Math.min(65535, Math.round(v * 100));
      const o = i * 4;
      rgba[o] = c[0]; rgba[o + 1] = c[1]; rgba[o + 2] = c[2];
      rgba[o + 3] = Math.round(255 * c[3] * fade[i]);
    }
  }
  return { rgba, depthCm, rpAt, wet, max };
}

/** The depth (m) and map (return period) a painted cell holds at a point, or null outside it or where dry. */
export function probeCell(cell, painted, lon, lat, w = CELL_PX, h = CELL_PX) {
  const [cw, cs, ce, cn] = cell.box;
  if (lon < cw || lon >= ce || lat < cs || lat >= cn || !painted) return null;
  const x = Math.min(w - 1, Math.floor(((lon - cw) / (ce - cw)) * w));
  const top = mercY(cn), bottom = mercY(cs);
  const y = Math.min(h - 1, Math.max(0, Math.floor(((mercY(lat) - top) / (bottom - top)) * h)));
  const i = y * w + x;
  const cm = painted.depthCm[i];
  return cm ? { depth: cm / 100, rp: painted.rpAt[i] } : null;
}

/** The reach a point belongs to: among the circles holding it, the one of map `rp` nearest to it. */
export function reachAt(reaches, lon, lat, rp) {
  let best = null, bestD = Infinity;
  for (const r of reaches || []) {
    if (rp && r.rp !== rp) continue;
    const d = Math.hypot((lon - r.lon) * KM_PER_DEG * cosd(r.lat), (lat - r.lat) * KM_PER_DEG);
    if (d <= r.r && d < bestD) { best = r; bestD = d; }
  }
  return best;
}

// ── words ───────────────────────────────────────────────────────────────────

const fmtDepth = (m) => (m >= 10 ? Math.round(m).toString() : m.toFixed(1));

/** The legend's one line for the view. */
export function depthLegendLine({ n, inView, zoom }) {
  if (!n) return "No river is forecast to pass its 10-year flow on this day.";
  const of = `${n.toLocaleString("en-GB")} reach${n === 1 ? "" : "es"} forecast to pass the 10-year flow`;
  if (zoom < DEPTH_MINZOOM) return `${of}. Zoom in on one to see the depth.`;
  if (!inView) return `None in view, of ${of}.`;
  return `${inView.toLocaleString("en-GB")} in view, of ${of}.`;
}

/** The card's words for a click on the depth. */
export function depthFacts(reach, depth, rp, { day = -1, when = "" } = {}) {
  const cls = reach.cls;
  const map = rp === cls ? `the ${rp}-year depth map` : `the ${rp}-year depth map, the nearest at or below it`;
  return {
    title: `About ${fmtDepth(depth)} m deep`,
    status: `Reach ${reach.id} is forecast to pass its ${cls}-year flow${when ? ` ${when}` : ""}; this is ${map}.`,
    figure: { value: fmtDepth(depth), unit: "m", label: `${rp}-year depth here` },
    note: day >= 0 ? "" : "Shown for the 15-day peak; move the time bar into the forecast to step through the days.",
  };
}
