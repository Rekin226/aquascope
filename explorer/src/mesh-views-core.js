// The flood views of a model result (#619, #620), the pure part: no DOM, no map, so node can test it.
//
// The numbers are the package's (aquascope.mesh_views, run in mesh-worker.js): the smooth depth on a grid, the
// contour lines, the hazard class, the arrival time and the largest depth and speed of every cell. This file
// decides how they are drawn and said: each lens's colours and key, how a grid becomes pixels, and the words of
// the card for the clicked cell.

import { STOPS, fmt, modelTime } from "./mesh-core.js?v=__BUILD__";

/** The lenses, in the order the legend offers them. `needs` is the package's name in meta.available. */
export const LENSES = [
  { id: "depth", label: "Depth", title: "The depth at the step on show, or the largest over the run" },
  { id: "hazard", label: "Hazard", title: "Flood hazard H1 to H6 from depth and speed (AIDR Guideline 7-3)" },
  { id: "arrival", label: "Arrival", title: "When each place first floods" },
  { id: "max_depth", label: "Max depth", title: "The largest depth over the run" },
  { id: "max_speed", label: "Max speed", title: "The largest speed over the run" },
];

/** The lenses a mesh offers: those the package computed, in the legend's order. */
export function lensesFor(meta) {
  const have = new Set((meta && meta.available) || ["depth"]);
  return LENSES.filter((l) => have.has(l.id));
}

// H1 (light) to H6 (dark), warm: an ordered ramp, with each class named in the key and the card.
export const HAZARD_COLOURS = ["#f3df70", "#f6ac52", "#e97237", "#b93a2c", "#7f2446", "#3d1a45"];
// Arrival: one hue, dark for the first places to flood, light for the last.
export const ARRIVAL_STOPS = [
  { t: 0, c: [52, 28, 104], a: 0.9 }, { t: 0.35, c: [104, 76, 160], a: 0.86 },
  { t: 0.7, c: [158, 136, 200], a: 0.8 }, { t: 1, c: [218, 208, 236], a: 0.74 },
];
// Speed: one hue, light (slow) to dark (fast), crimson, apart from the depth blues and the map's own tans.
export const SPEED_STOPS = [
  { t: 0, c: [249, 212, 227], a: 0.78 }, { t: 0.3, c: [236, 128, 170], a: 0.84 },
  { t: 0.65, c: [184, 30, 100], a: 0.9 }, { t: 1, c: [96, 10, 52], a: 0.95 },
];

// The key's short line per hazard class; the card and the key's tooltip give Table 1's full words.
export const HAZARD_SHORT = ["Generally safe", "Unsafe for small vehicles", "Unsafe for vehicles, children, the elderly",
  "Unsafe for vehicles and people", "Also building damage", "Also building failure"];

const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));

/** Packed RGBA (little-endian, as an ImageData buffer reads it). */
export function pack(r, g, b, a) { return ((Math.round(a * 255) << 24) | (b << 16) | (g << 8) | r) >>> 0; }

function at(stops, t) {
  for (let k = 0; k < stops.length - 1; k++) {
    const a = stops[k], b = stops[k + 1];
    if (t <= b.t) {
      const f = (t - a.t) / ((b.t - a.t) || 1);
      return [0, 1, 2].map((i) => Math.round(a.c[i] + (b.c[i] - a.c[i]) * f)).concat(a.a + (b.a - a.a) * f);
    }
  }
  const last = stops[stops.length - 1];
  return [...last.c, last.a];
}

/** A ramp as N packed colours: bin k covers t in [k/N, (k+1)/N). */
export function packedRamp(stops, n = 64) {
  const out = new Uint32Array(n);
  for (let k = 0; k < n; k++) { const [r, g, b, a] = at(stops, (k + 0.5) / n); out[k] = pack(r, g, b, a); }
  return out;
}

export function rampCssOf(stops) {
  return `linear-gradient(90deg, ${stops.map((s) => `rgba(${s.c.join(",")},${s.a}) ${Math.round(s.t * 100)}%`).join(", ")})`;
}

/** The packed colour of each hazard class, index 0 (dry) transparent. */
export function hazardPacked(alpha = 0.88) {
  return [0, ...HAZARD_COLOURS.map((h) => { const [r, g, b] = hex(h); return pack(r, g, b, alpha); })];
}

/** The scale a lens draws on: { lo, hi, sqrt }, as mesh-core's scaleFor. */
export function lensScale(lens, meta, depthScale, speedMax) {
  if (lens === "arrival") {
    const hi = meta && meta.arrival && Number.isFinite(meta.arrival.last_h) ? meta.arrival.last_h : 1;
    return { lo: 0, hi: hi > 0 ? hi : 1, sqrt: false };
  }
  if (lens === "max_speed") return { lo: 0, hi: speedMax > 0 ? niceUp(speedMax) : 1, sqrt: false };
  return depthScale;
}

function niceUp(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  const m = x / p;
  return (m <= 1 ? 1 : m <= 1.5 ? 1.5 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 3 ? 3 : m <= 5 ? 5 : m <= 7.5 ? 7.5 : 10) * p;
}

/** The ramp of a lens (the depth lenses keep mesh-core's blues). */
export function lensStops(lens) {
  return lens === "arrival" ? ARRIVAL_STOPS : lens === "max_speed" ? SPEED_STOPS : STOPS;
}

/**
 * Colour a smooth grid (Float32Array, NaN where nothing is drawn) into packed pixels with a ramp: `out` is the
 * image's Uint32Array view, `colours` packedRamp's output.
 */
export function paintGrid(out, values, scale, colours) {
  const n = colours.length, { lo, hi, sqrt } = scale, span = (hi - lo) || 1;
  for (let p = 0; p < values.length; p++) {
    const v = values[p];
    if (!Number.isFinite(v)) { out[p] = 0; continue; }
    let t = (v - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    if (sqrt) t = Math.sqrt(t);
    out[p] = colours[Math.min(n - 1, Math.floor(t * n))];
  }
  return out;
}

/** The key's ticks for a lens's ramp: [{ at: percent, label }]. */
export function lensTicks(lens, scale) {
  if (lens === "arrival") {
    const hi = scale.hi;
    return [0, hi / 2, hi].map((h) => ({ at: Math.round((h / (hi || 1)) * 100), label: hoursWords(h, { short: true }) }));
  }
  const vals = scale.sqrt ? [scale.lo, scale.hi / 4, scale.hi] : [scale.lo, (scale.lo + scale.hi) / 2, scale.hi];
  return vals.map((v) => {
    let t = (v - scale.lo) / ((scale.hi - scale.lo) || 1);
    if (scale.sqrt) t = Math.sqrt(Math.max(0, t));
    return { at: Math.round(t * 100), label: fmt(v) };
  });
}

/** 0 -> "at the start", 2.5 -> "2 h 30 min", 30 -> "1 day 6 h"; short: "0 h", "2.5 h", "30 h". */
export function hoursWords(h, { short = false } = {}) {
  if (!Number.isFinite(h)) return "–";
  if (short) return `${fmt(h, h < 10 && h % 1 ? 1 : 0)} h`;
  if (h <= 0) return "at the start";
  const mins = Math.round(h * 60);
  const d = Math.floor(mins / 1440), hh = Math.floor((mins % 1440) / 60), mm = mins % 60;
  const parts = [];
  if (d) parts.push(`${d} day${d === 1 ? "" : "s"}`);
  if (hh) parts.push(`${hh} h`);
  if (mm && !d) parts.push(`${mm} min`);
  return parts.join(" ") || "under a minute";
}

/** Square metres in words: "4,800 m²", "18.5 ha", "2.3 km²". */
export function areaWords(m2) {
  if (!Number.isFinite(m2) || m2 <= 0) return "none";
  if (m2 < 10000) return `${Math.round(m2).toLocaleString("en-GB")} m²`;
  if (m2 < 1e6) return `${fmt(m2 / 1e4)} ha`;
  return `${fmt(m2 / 1e6)} km²`;
}

/**
 * What the card says about one cell under a lens: { status, figure, note, color } or null for the depth lens
 * (whose card is mesh-core's cellFacts). views: { meta (the package's lenses meta), arrays (per cell) };
 * times: the model's ISO times; units: the mesh's units.
 */
export function lensFacts(lens, views, cell, times, units = {}) {
  if (!views || lens === "depth") return null;
  const { meta, arrays } = views;
  const lenUnit = units.depth || units.water_surface || "";
  const vUnit = units.velocity || "";
  const when = (k) => (Number.isInteger(k) && k >= 0 && times && times[k] ? modelTime(times[k]) : null);
  if (lens === "hazard" && arrays.hazard) {
    const h = arrays.hazard[cell];
    if (!h) return { status: "Dry all through the run: no hazard", figure: null, note: "", color: null };
    const cls = meta.hazard.classes[h - 1];
    const k = arrays.hazard_step ? arrays.hazard_step[cell] : -1;
    const t = when(k);
    const dv = arrays.hazard_dv[cell], d = arrays.hazard_depth[cell], v = arrays.hazard_speed[cell];
    const how = meta.hazard.from === "step"
      ? `Worst${t ? ` at ${t}` : ""}, with ${fmt(d)} m of water moving at ${fmt(v)} m/s (depth × velocity ${fmt(dv)} m²/s).`
      : `From the largest depth ${fmt(d)} m and the largest speed ${fmt(v)} m/s (depth × velocity ${fmt(dv)} m²/s).`;
    return {
      status: `${cls.code}: ${cls.words}`, color: HAZARD_COLOURS[h - 1],
      figure: { value: cls.code, unit: "", label: "highest hazard over the run" },
      note: `${how} ${limitWords(cls)} ${meta.hazard.source.short}.`,
    };
  }
  if (lens === "arrival" && arrays.arrival_step) {
    const k = arrays.arrival_step[cell];
    const rule = meta.arrival.depth_m;
    if (k < 0) return { status: `Never ${fmt(rule)} m deep in this run`, figure: null, note: `Arrival is ${meta.arrival.rule}.`, color: null };
    const hh = arrays.arrival_h[cell];
    return {
      status: k === 0 ? `Wet from the start of the run (${when(0)})` : `Floods at ${when(k)}, ${hoursWords(hh)} after the start`,
      figure: { value: k === 0 ? "0" : fmt(hh, hh < 10 ? 1 : 0), unit: "h", label: "arrival after the start" },
      note: `Arrival is ${meta.arrival.rule}, read at the model's output steps.`, color: null,
    };
  }
  if (lens === "max_depth") {
    const v = arrays.max_depth[cell];
    const ws = meta.variable === "water_surface";
    if (!Number.isFinite(v)) return { status: "Dry all through the run", figure: null, note: "", color: null };
    const t = when(arrays.max_depth_step ? arrays.max_depth_step[cell] : -1);
    const what = ws ? "water surface" : "depth";
    return { status: `Largest ${what} ${fmt(v)}${lenUnit ? ` ${lenUnit}` : ""} over the run`,
      figure: { value: fmt(v), unit: lenUnit, label: `largest ${what}` },
      note: t ? `The ${ws ? "highest" : "deepest"} of the output steps is ${t}.` : "", color: null };
  }
  if (lens === "max_speed" && arrays.max_speed) {
    const v = arrays.max_speed[cell];
    if (!Number.isFinite(v)) return { status: "Dry all through the run", figure: null, note: "", color: null };
    const t = when(arrays.max_speed_step ? arrays.max_speed_step[cell] : -1);
    return { status: `Largest speed ${fmt(v)}${vUnit ? ` ${vUnit}` : ""} over the run`,
      figure: { value: fmt(v), unit: vUnit, label: "largest speed" },
      note: t ? `The fastest of the output steps is ${t}.` : "", color: null };
  }
  return null;
}

/** "H4 means D × V up to 1 m²/s, D up to 2 m and V up to 2 m/s." */
export function limitWords(cls) {
  if (!cls) return "";
  if (cls.dv === null || cls.dv === undefined) return `${cls.code} is past an H5 limit: depth × velocity over 4 m²/s, depth over 4 m or speed over 4 m/s.`;
  return `${cls.code} means depth × velocity up to ${fmt(cls.dv)} m²/s, depth up to ${fmt(cls.depth)} m and speed up to ${fmt(cls.velocity)} m/s.`;
}

/** The key's line under a lens. */
export function lensLine(lens, meta, { step = null, times = [] } = {}) {
  if (lens === "hazard") return `Highest flood hazard in each cell over the run, ${meta.hazard.rule === "the highest class reached at any output step" ? "from depth and speed at every step" : "from the largest depth and speed"}`;
  if (lens === "arrival") return `When each place first has ${fmt(meta.arrival.depth_m)} m of water, after the start of the run`;
  if (lens === "max_depth") return meta.variable === "water_surface" ? "Highest water surface in each place over the run" : "Largest depth in each place over the run";
  if (lens === "max_speed") return "Largest speed in each place over the run";
  return step === null || !times.length ? "Largest depth over the run" : `Depth at ${modelTime(times[step])}`;
}
