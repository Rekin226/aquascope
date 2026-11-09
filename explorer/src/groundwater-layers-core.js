// Groundwater views (#625, #626), the pure part: no DOM, no map, so node can test it.
//
// Every number drawn here comes from the package (aquascope.groundwater.layers and .budget, in the worker): the
// layers' tops and bottoms, the cross-section, a cell's heads over time, the water budget. This module turns them
// into what is drawn: the 3D prisms, the section, the hydrograph and the budget chart, all as plain SVG strings.

import { rgb, valueColor } from "./groundwater-core.js?v=__BUILD__";

/** One colour per layer for lines (layer 1 first): a calm categorical set that reads in light and dark. */
export const LAYER_LINES = ["#1d6fb8", "#c4622d", "#3f8f3a", "#7a52a8", "#b8336a", "#2b8a8a", "#8a6d1f", "#4a5d73"];
/** The section's layer bands, earth tones from light to deeper, repeating. */
export const LAYER_FILLS = ["#f3ead5", "#ddd0b3", "#ece2cb", "#d6c7a6", "#efe6d0", "#dacbae"];
/** The same layers below the water table: pale blues, one shade per layer, so saturated ground reads as water. */
export const SATURATED_FILLS = ["#d3e5f4", "#bcd5ec", "#cfe2f3", "#b6d0e9", "#d7e7f5", "#c0d8ee"];
export const WATER = "#1d6fb8";
export const layerLine = (i) => LAYER_LINES[i % LAYER_LINES.length];

/** Each budget term's colour, by its label (the package's names); unknown terms take a grey from the end. */
const TERM_COLORS = {
  Recharge: "#4c9a4a", "Recharge (UZF)": "#78b36a", Storage: "#9a8466", "Storage (specific storage)": "#9a8466",
  "Storage (specific yield)": "#c2ab86", "Constant heads": "#7a52a8", Wells: "#c4473a", "Wells (MNW2)": "#c4473a",
  "Wells (MAW)": "#c4473a", Rivers: "#1d6fb8", Streams: "#4f9ad6", Drains: "#d08a2e", "General heads": "#2b8a8a",
  Evapotranspiration: "#b8a23a", "Evapotranspiration (UZF)": "#d1bd5a", Lakes: "#3b7fa8", "Surface leakage (UZF)": "#8fb3cc",
};
const SPARE = ["#6b7b8c", "#8f6b8c", "#6b8c7a", "#8c7a6b"];
export function termColor(label, i = 0) {
  return TERM_COLORS[label] || SPARE[i % SPARE.length];
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fin = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const r1 = (v) => Math.round(v * 10) / 10;

/** Round numbers for an axis: about n ticks between lo and hi. */
export function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || 10 * p;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step || 0);   // no "-0" on an axis
  return out;
}

/** "350 m", "4.2 km". */
export function distWords(m) {
  if (!fin(m)) return "";
  return Math.abs(m) >= 1000 ? `${(m / 1000).toFixed(Math.abs(m) >= 10000 ? 0 : 1)} km` : `${Math.round(m)} m`;
}

/** A rate as a person reads it: "2,500", "12.4", "0.031". */
export function rateWords(v) {
  if (!fin(v)) return "–";
  const a = Math.abs(v);
  if (a >= 100) return Math.round(v).toLocaleString("en-GB");
  if (a >= 1) return v.toFixed(1);
  if (a === 0) return "0";
  return v.toPrecision(2);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** A model date as an axis reads it: "Mar 2024"; with day: "15 Mar 2024". A model time without a date: "t 30". */
export function dateWords(d, { day = false } = {}) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(d || ""));
  if (!m) return fin(d) ? `t ${Number(d).toPrecision(4).replace(/\.?0+$/, "")}` : "";
  return `${day ? `${Number(m[3])} ` : ""}${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
}

/** The vertical exaggeration choices offered next to the 3D stack, around the package's suggestion. */
export function exaggerationChoices(suggested) {
  const all = [1, 2, 5, 10, 20, 50, 100, 200, 500];
  const k = Math.max(0, all.indexOf(Number(suggested)));
  return all.slice(Math.max(0, k - 2), k + 3);
}

// ── 3D: the layers as prisms ────────────────────────────────────────────────

/**
 * The layers as fill-extrusion features: one prism per cell and shown layer, from the layer's bottom to its top,
 * measured up from the model's base, times the exaggeration, in metres (what MapLibre extrudes). Coloured by the
 * head at time t on the run's range; a cell with no head now but one at another time (dry) is grey; a cell with no
 * head at any time (inactive) is left out. gap (metres) pulls the layers apart, each raised that much above the one
 * under it, so each layer's top shows.
 *
 * geo: { top (cells), botm (layers x cells, flat), tops (optional, layers x cells: each layer's own top), nlay, base,
 * metres_per_unit }; model: { rings, offsets, heads,
 * shape (time x layer x cell), layers (the head file's, 1-based) }.
 */
export function prismFeatures(geo, model, t, { exaggeration = 1, shown = null, range = [0, 1], wet = null, gap = 0 } = {}) {
  const { rings, offsets, heads, shape } = model;
  const [, nl, nc] = shape;
  const mper = Number(geo.metres_per_unit) || 1;
  const ve = Number(exaggeration) || 1;
  const features = [];
  const hpos = new Map((model.layers || []).map((lay, i) => [lay, i]));
  for (let i = 0; i < geo.nlay; i++) {
    if (shown && !shown.has(i)) continue;
    const hl = hpos.get(i + 1);
    const lift = gap * (geo.nlay - 1 - i);   // pulled apart: each layer raised above the one under it
    for (let c = 0; c < nc; c++) {
      const top = geo.tops ? geo.tops[i * nc + c] : i === 0 ? geo.top[c] : geo.botm[(i - 1) * nc + c];   // tops: LAYCBD
      const bot = geo.botm[i * nc + c];
      if (!Number.isFinite(top) || !Number.isFinite(bot) || top <= bot) continue;
      const h = hl === undefined ? NaN : heads[(t * nl + hl) * nc + c];
      const everWet = hl !== undefined && (wet ? wet[hl * nc + c] : Number.isFinite(h));
      if (!everWet) continue;
      const ring = [];
      for (let k = offsets[c]; k < offsets[c + 1]; k++) ring.push([rings[2 * k], rings[2 * k + 1]]);
      ring.push(ring[0]);
      features.push({
        type: "Feature", id: i * nc + c,
        properties: { c, l: i, base: r1((bot - geo.base) * mper * ve + lift), height: r1((top - geo.base) * mper * ve + lift),
          color: Number.isFinite(h) ? rgb(valueColor(h, range, "level")) : "#b9c0c7" },
        geometry: { type: "Polygon", coordinates: [ring] },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

/** Which layer-cells ever have a head (time x layer x cell heads): a Uint8Array (layer x cell). */
export function everWet(heads, shape) {
  const [nt, nl, nc] = shape;
  const out = new Uint8Array(nl * nc);
  for (let t = 0; t < nt; t++) {
    const off = t * nl * nc;
    for (let k = 0; k < nl * nc; k++) if (!out[k] && Number.isFinite(heads[off + k])) out[k] = 1;
  }
  return out;
}

// ── the cross-section ───────────────────────────────────────────────────────

/** A value of the section's arrays: top/bottom are layer x segment, heads time x layer x segment, water table time x
 * segment (flat Float32Arrays, as the worker sends them). */
export function sectionValue(sec, what, t, i, s) {
  const n = sec.segments.length;
  const nl = sec.layers.length;
  let v;
  if (what === "top") v = sec.top[i * n + s];
  else if (what === "bottom") v = sec.bottom[i * n + s];
  else if (what === "head") v = sec.heads[(t * nl + i) * n + s];
  else v = sec.water_table[t * n + s];
  return Number.isFinite(v) ? v : null;
}

/** Runs of contiguous segments (a part of the line outside the model breaks the run). */
function runs(segs) {
  const out = [];
  let cur = [];
  segs.forEach((s, k) => {
    if (cur.length && s.from_m - segs[k - 1].to_m > 1) { out.push(cur); cur = []; }
    cur.push(k);
  });
  if (cur.length) out.push(cur);
  return out;
}

/** The section's vertical range over every time, so stepping through time keeps the axes still. */
export function sectionRange(sec) {
  let lo = Infinity, hi = -Infinity;
  const take = (v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  sec.top.forEach(take); sec.bottom.forEach(take); sec.water_table.forEach(take);
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

/** A vertical range around the heads alone (every layer, every time), for a close look at the water table and the
 * cone; the layers are cut at its edges. */
export function headRange(sec) {
  let lo = Infinity, hi = -Infinity;
  for (const v of sec.heads) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!(hi >= lo)) return sectionRange(sec);
  const pad = Math.max(0.5, (hi - lo) * 0.25);
  return [lo - pad, hi + pad];
}

/**
 * The section as SVG: the layers as bands between their tops and bottoms (stepped, one value per cell, as the model
 * holds them), the saturated ground tinted below the water table, the water table (the head of the uppermost layer
 * with one) as a bold line with its symbol, and the heads of the deeper layers as dashed lines. Returns { svg, ve,
 * x0, x1 } (ve: the drawing's own vertical exaggeration; x0 and x1 the plot's pixel edges, for the hover).
 */
export function sectionSvg(sec, t, { width = 720, height = 220, unit = "m", title = "" } = {}) {
  const segs = sec.segments;
  const nl = sec.layers.length;
  const L = 50, R = 74, T = 18, B = 26;
  const pw = width - L - R, ph = height - T - B;
  const [ylo, yhi] = sec.yRange || sectionRange(sec);
  const X = (m) => L + (m / sec.length_m) * pw;
  const Y = (v) => T + (1 - (v - ylo) / (yhi - ylo)) * ph;
  const f = (v) => v.toFixed(1);
  const parts = [];
  const groups = runs(segs);
  // the layer bands
  for (let i = 0; i < nl; i++) {
    for (const g of groups) {
      const up = [], dn = [];
      for (const s of g) {
        const tp = sectionValue(sec, "top", t, i, s), bt = sectionValue(sec, "bottom", t, i, s);
        if (tp === null || bt === null) continue;
        up.push(`${f(X(segs[s].from_m))},${f(Y(tp))}`, `${f(X(segs[s].to_m))},${f(Y(tp))}`);
        dn.unshift(`${f(X(segs[s].to_m))},${f(Y(bt))}`, `${f(X(segs[s].from_m))},${f(Y(bt))}`);
      }
      if (up.length) parts.push(`<polygon class="gwl-band" points="${[...up, ...dn].join(" ")}" fill="${LAYER_FILLS[i % LAYER_FILLS.length]}"/>`);
    }
  }
  // the saturated ground, layer by layer: below the water table, in every layer down to the deepest with a head
  // (a layer whose head stands above its top is full)
  for (const g of groups) {
    for (const s of g) {
      const wt = sectionValue(sec, "wt", t, 0, s);
      if (wt === null) continue;
      let last = -1;
      for (let i = 0; i < nl; i++) if (sectionValue(sec, "head", t, i, s) !== null) last = i;
      for (let i = 0; i <= last; i++) {
        const tp = sectionValue(sec, "top", t, i, s), bt = sectionValue(sec, "bottom", t, i, s);
        if (tp === null || bt === null) continue;
        const top = Math.min(wt, tp);
        if (top <= bt) continue;
        parts.push(`<rect class="gwl-sat" x="${f(X(segs[s].from_m))}" y="${f(Y(top))}" width="${f(Math.max(0.5, X(segs[s].to_m) - X(segs[s].from_m)))}" height="${f(Y(bt) - Y(top))}" fill="${SATURATED_FILLS[i % SATURATED_FILLS.length]}"/>`);
      }
    }
  }
  // cells with no head in any layer now (inactive, or dry through): hatched, so a gap reads as a gap
  let noHead = false;
  for (const g of groups) {
    for (const s of g) {
      let any = false;
      for (let i = 0; i < nl && !any; i++) any = sectionValue(sec, "head", t, i, s) !== null;
      if (any) continue;
      const tp = sectionValue(sec, "top", t, 0, s);
      let bt = null;
      for (let i = nl - 1; i >= 0 && bt === null; i--) bt = sectionValue(sec, "bottom", t, i, s);
      if (tp === null || bt === null) continue;
      noHead = true;
      parts.push(`<rect class="gwl-nohead" x="${f(X(segs[s].from_m))}" y="${f(Y(tp))}" width="${f(Math.max(0.5, X(segs[s].to_m) - X(segs[s].from_m)))}" height="${f(Y(bt) - Y(tp))}" fill="url(#gwl-hatch)"/>`);
    }
  }
  // layer outlines, drawn over the tint
  for (let i = 0; i < nl; i++) {
    for (const g of groups) {
      const pts = [];
      for (const s of g) {
        const bt = sectionValue(sec, "bottom", t, i, s);
        if (bt !== null) pts.push(`${f(X(segs[s].from_m))},${f(Y(bt))}`, `${f(X(segs[s].to_m))},${f(Y(bt))}`);
      }
      if (pts.length) parts.push(`<polyline class="gwl-edge" points="${pts.join(" ")}"/>`);
      if (i === 0) {
        const tops = [];
        for (const s of g) {
          const tp = sectionValue(sec, "top", t, 0, s);
          if (tp !== null) tops.push(`${f(X(segs[s].from_m))},${f(Y(tp))}`, `${f(X(segs[s].to_m))},${f(Y(tp))}`);
        }
        if (tops.length) parts.push(`<polyline class="gwl-ground" points="${tops.join(" ")}"/>`);
      }
    }
  }
  // layer names, inside each band at its widest point near the start
  for (let i = 0; i < nl; i++) {
    let best = null;
    for (let s = 0; s < segs.length; s++) {
      const tp = sectionValue(sec, "top", t, i, s), bt = sectionValue(sec, "bottom", t, i, s);
      if (tp === null || bt === null) continue;
      if (!best || (Y(bt) - Y(tp)) > best.h + 6) best = { s, h: Y(bt) - Y(tp), y: (Y(bt) + Y(tp)) / 2 };
      if (best.h > 22 && s > segs.length / 6) break;
    }
    if (best && best.h >= 11) {
      parts.push(`<text class="gwl-lname" x="${f(Math.max(L + 4, X(segs[best.s].from_m) + 4))}" y="${f(best.y + 3.5)}">Layer ${sec.layers[i]}</text>`);
    }
  }
  // the heads: the water table bold, the deeper layers' heads dashed (each through its cells' middles)
  const line = (vals) => {
    const out = [];
    for (const g of groups) {
      const pts = [];
      for (const s of g) {
        const v = vals(s);
        if (v === null) { if (pts.length > 1) out.push(pts.join(" ")); pts.length = 0; continue; }
        pts.push(`${f(X((segs[s].from_m + segs[s].to_m) / 2))},${f(Y(v))}`);
      }
      if (pts.length) out.push(pts.join(" "));
    }
    return out;
  };
  let wtLayer = -1;
  for (let i = 0; i < nl && wtLayer < 0; i++) {
    for (let s = 0; s < segs.length; s++) if (sectionValue(sec, "head", t, i, s) !== null) { wtLayer = i; break; }
  }
  const ends = [];
  for (let i = wtLayer + 1; i < nl; i++) {
    const ls = line((s) => sectionValue(sec, "head", t, i, s));
    for (const p of ls) parts.push(`<polyline class="gwl-head" stroke="${layerLine(i)}" points="${p}"/>`);
    const last = [...segs.keys()].reverse().find((s) => sectionValue(sec, "head", t, i, s) !== null);
    if (last !== undefined) ends.push({ y: Y(sectionValue(sec, "head", t, i, last)), text: `Layer ${sec.layers[i]} head`, color: layerLine(i) });
  }
  const wl = line((s) => sectionValue(sec, "wt", t, 0, s));
  for (const p of wl) parts.push(`<polyline class="gwl-wt" points="${p}"/>`);
  const lastWt = [...segs.keys()].reverse().find((s) => sectionValue(sec, "wt", t, 0, s) !== null);
  if (lastWt !== undefined) ends.push({ y: Y(sectionValue(sec, "wt", t, 0, lastWt)), text: "Water table", color: WATER, bold: true });
  // the water table symbol, an open triangle standing on the line
  const sym = segs.findIndex((s, k) => k >= segs.length * 0.18 && sectionValue(sec, "wt", t, 0, k) !== null);
  if (sym >= 0) {
    const cx = X((segs[sym].from_m + segs[sym].to_m) / 2), cy = Y(sectionValue(sec, "wt", t, 0, sym));
    parts.push(`<path class="gwl-sym" d="M${f(cx - 5)},${f(cy - 9)} L${f(cx + 5)},${f(cy - 9)} L${f(cx)},${f(cy - 1.5)} Z"/>`);
  }
  // end labels, nudged apart
  ends.sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 11) ends[k].y = ends[k - 1].y + 11;
  for (const e of ends) {
    parts.push(`<text class="gwl-end${e.bold ? " b" : ""}" x="${f(L + pw + 5)}" y="${f(Math.max(T + 6, Math.min(T + ph, e.y)) + 3.5)}" fill="${e.color}">${esc(e.text)}</text>`);
  }
  // axes
  const axis = [];
  for (const v of niceTicks(ylo, yhi, 4)) {
    axis.push(`<line class="gwl-grid" x1="${L}" x2="${L + pw}" y1="${f(Y(v))}" y2="${f(Y(v))}"/>`,
      `<text class="gwl-tick" x="${L - 5}" y="${f(Y(v) + 3.5)}" text-anchor="end">${esc(String(v))}</text>`);
  }
  for (const m of niceTicks(0, sec.length_m, 5)) {
    axis.push(`<text class="gwl-tick" x="${f(X(m))}" y="${height - 8}" text-anchor="middle">${esc(distWords(m))}</text>`);
  }
  axis.push(`<text class="gwl-unit" x="6" y="${T - 6}">${esc(unit || "elevation")}</text>`,
    `<text class="gwl-ab" x="${L}" y="${T - 5}">A</text>`, `<text class="gwl-ab" x="${L + pw}" y="${T - 5}" text-anchor="end">A′</text>`);
  for (const v of (sec.vertices_m || []).slice(1, -1)) {
    axis.push(`<line class="gwl-bend" x1="${f(X(v))}" x2="${f(X(v))}" y1="${T}" y2="${T + ph}"/>`);
  }
  const mper = Number(sec.metres_per_unit) || 1;
  const ve = (ph / ((yhi - ylo) * mper)) / (pw / sec.length_m);   // vertical pixels per metre over horizontal
  const svg = `<svg class="gwl-svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="none" role="img" aria-label="${esc(title || "Cross-section")}">` +
    `<defs><clipPath id="gwl-clip"><rect x="${L}" y="${T}" width="${pw}" height="${ph}"/></clipPath>` +
    '<pattern id="gwl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
    '<rect width="6" height="6" class="gwl-hatch-bg"/><line x1="0" y1="0" x2="0" y2="6" class="gwl-hatch-ln"/></pattern></defs>' +
    `${axis.join("")}<g clip-path="url(#gwl-clip)">${parts.filter((x) => !x.startsWith("<text class=\"gwl-end")).join("")}</g>` +
    `${parts.filter((x) => x.startsWith("<text class=\"gwl-end")).join("")}<line class="gwl-cursor" x1="0" x2="0" y1="${T}" y2="${T + ph}" visibility="hidden"/></svg>`;
  return { svg, ve, x0: L, x1: L + pw, top: T, bottom: T + ph, noHead };
}

/** The drawing's vertical exaggeration as the panel states it: "x12" (rounded to a sensible figure). */
export function veWords(ve) {
  if (!(ve > 0)) return "";
  if (ve < 1.5) return `x${ve.toFixed(1)}`;
  if (ve < 20) return `x${Math.round(ve)}`;
  return `x${Math.round(ve / 5) * 5}`;
}

/** What the hover shows at a distance along the line: the cell, and each layer's top, bottom and head. */
export function sectionAt(sec, t, m) {
  const s = sec.segments.findIndex((g) => m >= g.from_m && m <= g.to_m);
  if (s < 0) return null;
  const g = sec.segments[s];
  return {
    s, at_m: m, cell: g.cell, row: g.row, col: g.col,
    water_table: sectionValue(sec, "wt", t, 0, s),
    layers: sec.layers.map((lay, i) => ({ layer: lay, top: sectionValue(sec, "top", t, i, s),
      bottom: sectionValue(sec, "bottom", t, i, s), head: sectionValue(sec, "head", t, i, s) })),
  };
}

// ── a cell's heads over time ────────────────────────────────────────────────

/** The hydrograph as SVG: one line per layer (the package's series), the current time marked. Returns { svg, key }. */
export function hydrographSvg(h, t, { width = 300, height = 128, unit = "m" } = {}) {
  const series = (h.series || []).filter((s) => s.min !== null && s.min !== undefined);
  if (!series.length) return null;
  const n = (h.times || []).length;
  const L = 40, R = 8, T = 8, B = 20;
  const pw = width - L - R, ph = height - T - B;
  let lo = Math.min(...series.map((s) => s.min)), hi = Math.max(...series.map((s) => s.max));
  if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const X = (k) => L + (n > 1 ? k / (n - 1) : 0.5) * pw;
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * ph;
  const f = (v) => v.toFixed(1);
  const parts = [];
  for (const v of niceTicks(lo, hi, 3)) {
    parts.push(`<line class="gwl-grid" x1="${L}" x2="${L + pw}" y1="${f(Y(v))}" y2="${f(Y(v))}"/>`,
      `<text class="gwl-tick" x="${L - 4}" y="${f(Y(v) + 3.5)}" text-anchor="end">${esc(String(v))}</text>`);
  }
  const lab = (k) => dateWords((h.dates && h.dates[k]) || h.times[k]);
  parts.push(`<text class="gwl-tick" x="${L}" y="${height - 5}">${esc(lab(0))}</text>`,
    `<text class="gwl-tick" x="${L + pw}" y="${height - 5}" text-anchor="end">${esc(lab(n - 1))}</text>`);
  if (t >= 0 && t < n) parts.push(`<line class="gwl-now" x1="${f(X(t))}" x2="${f(X(t))}" y1="${T}" y2="${T + ph}"/>`);
  (h.series || []).forEach((s, i) => {
    if (s.min === null || s.min === undefined) return;
    const pts = [];
    s.head.forEach((v, k) => { if (v !== null) pts.push(`${f(X(k))},${f(Y(v))}`); });
    parts.push(`<polyline class="gwl-hline" stroke="${layerLine(s.layer - 1)}" points="${pts.join(" ")}"/>`);
    const v = s.head[t];
    if (v !== null && v !== undefined) parts.push(`<circle cx="${f(X(t))}" cy="${f(Y(v))}" r="2.6" fill="${layerLine(s.layer - 1)}"/>`);
    void i;
  });
  const svg = `<svg class="gwl-svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">` +
    `<text class="gwl-unit" x="2" y="${T + 2}">${esc(unit)}</text>${parts.join("")}</svg>`;
  const key = series.map((s) => `<span class="gwl-key"><i style="background:${layerLine(s.layer - 1)}"></i>Layer ${s.layer}</span>`).join("");
  return { svg, key };
}

/** The hydrograph's words for the card: each layer's head now, and how far it moved over the run. */
export function hydrographWords(h, t, unit = "m") {
  const u = unit ? ` ${unit}` : "";
  return (h.series || []).filter((s) => s.min !== null).map((s) => {
    const now = s.head[t];
    return `Layer ${s.layer}: ${now === null || now === undefined ? "no head now" : `${now.toFixed(2)}${u}`}` +
      ` (lowest ${s.min.toFixed(2)}${u}, ${dateWords(s.min_at, { day: true }) || s.min_at})`;
  });
}

// ── the water budget ────────────────────────────────────────────────────────

/** The budget's vertical range: the largest total in and out over the run. */
function budgetMax(b) {
  return Math.max(1e-12, ...b.total_in.map(Number).filter(Number.isFinite), ...b.total_out.map(Number).filter(Number.isFinite));
}

/**
 * The water budget over time as SVG: per saved time, the terms flowing in stacked above the zero line and those
 * flowing out stacked below it, each term in its colour; the current time outlined. Returns { svg, key, x0, x1 }.
 */
export function budgetSvg(b, t, { width = 720, height = 200, title = "" } = {}) {
  const n = b.kstpkper.length;
  const L = 56, R = 10, T = 12, B = 22;
  const pw = width - L - R, ph = height - T - B;
  const mx = budgetMax(b) * 1.06;
  const Y = (v) => T + (1 - (v + mx) / (2 * mx)) * ph;
  const bw = pw / n;
  const gap = bw > 6 ? Math.min(3, bw * 0.18) : 0;
  const f = (v) => v.toFixed(2);
  const parts = [];
  for (const v of niceTicks(-mx, mx, 4)) {
    parts.push(`<line class="gwl-grid" x1="${L}" x2="${L + pw}" y1="${f(Y(v))}" y2="${f(Y(v))}"/>`,
      `<text class="gwl-tick" x="${L - 5}" y="${f(Y(v) + 3.5)}" text-anchor="end">${esc(rateWords(Math.abs(v)))}</text>`);
  }
  if (t >= 0 && t < n) parts.push(`<rect class="gwl-col" x="${f(L + t * bw)}" y="${T}" width="${f(bw)}" height="${ph}"/>`);
  for (let k = 0; k < n; k++) {
    let up = 0, dn = 0;
    const x = L + k * bw + gap / 2, w = Math.max(0.6, bw - gap);
    b.terms.forEach((term, i) => {
      const vin = Number(term.in[k]) || 0, vout = Number(term.out[k]) || 0;
      const c = termColor(term.label, i);
      if (vin > 0) { parts.push(`<rect x="${f(x)}" y="${f(Y(up + vin))}" width="${f(w)}" height="${f(Y(up) - Y(up + vin))}" fill="${c}"/>`); up += vin; }
      if (vout > 0) { parts.push(`<rect x="${f(x)}" y="${f(Y(-dn))}" width="${f(w)}" height="${f(Y(-dn - vout) - Y(-dn))}" fill="${c}"/>`); dn += vout; }
    });
  }
  parts.push(`<line class="gwl-zero" x1="${L}" x2="${L + pw}" y1="${f(Y(0))}" y2="${f(Y(0))}"/>`,
    `<text class="gwl-side" x="${L - 5}" y="${f(Y(0) - 14)}" text-anchor="end">in ↑</text>`,
    `<text class="gwl-side" x="${L - 5}" y="${f(Y(0) + 22)}" text-anchor="end">out ↓</text>`);
  const lab = (k) => dateWords((b.dates && b.dates[k]) || b.times[k]) || `period ${b.kstpkper[k][1]}, step ${b.kstpkper[k][0]}`;
  const ticks = n <= 1 ? [0] : [0, Math.round((n - 1) / 2), n - 1];
  ticks.forEach((k, j) => parts.push(`<text class="gwl-tick" x="${f(L + (k + 0.5) * bw)}" y="${height - 6}" text-anchor="${j === 0 ? "start" : j === ticks.length - 1 ? "end" : "middle"}">${esc(lab(k))}</text>`));
  const svg = `<svg class="gwl-svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="none" role="img" aria-label="${esc(title || "Water budget")}">` +
    `<text class="gwl-unit" x="4" y="${T}">${esc(b.unit)}</text>${parts.join("")}</svg>`;
  const key = b.terms.map((term, i) => `<span class="gwl-key"><i style="background:${termColor(term.label, i)}"></i>${esc(term.label)}</span>`).join("");
  return { svg, key, x0: L, x1: L + pw, n };
}

/** One saved time of the budget as table rows: each term's in, out and net, then the totals and the discrepancy. */
export function budgetRows(b, t) {
  const rows = b.terms.map((term, i) => ({ label: term.label, color: termColor(term.label, i), in: Number(term.in[t]) || 0,
    out: Number(term.out[t]) || 0, net: (Number(term.in[t]) || 0) - (Number(term.out[t]) || 0) }));
  return { rows, total_in: b.total_in[t], total_out: b.total_out[t], discrepancy: b.discrepancy_pct[t],
    when: dateWords((b.dates && b.dates[t]) || b.times[t], { day: true }) || `period ${b.kstpkper[t][1]}, step ${b.kstpkper[t][0]}` };
}

/** The budget's saved time for a heads time step: the one with the same stress period and time step, else none. */
export function budgetStep(b, kstpkper) {
  if (!b || !kstpkper) return -1;
  return b.kstpkper.findIndex((k) => k[0] === kstpkper[0] && k[1] === kstpkper[1]);
}
