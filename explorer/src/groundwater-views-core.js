// What a hydrogeologist reads off MODFLOW heads (#623, #624), the pure part: no DOM, no map, so node can test it.
//
// The numbers (contours, arrows, drawdown, wells, pathlines, capture zones) come from Python,
// aquascope.groundwater.views.model_views, in the worker. Here they become map features, a smooth head surface
// (cell-centre values interpolated between centres, the way a contoured water table reads, never across an
// inactive cell), the legend's lines and the well card.

import { rampColor, rangeT, surfaceAlpha, HEAD_RAMP, CHANGE_RAMP } from "./groundwater-core.js?v=__BUILD__";

/** The three ways to look at a model: the head with its contours and arrows, the drawdown cone, where wells draw. */
export const VIEW_MODES = [
  { id: "head", label: "Head", show: ["head", "arrows"] },
  { id: "drawdown", label: "Drawdown", show: ["drawdown"] },
  { id: "capture", label: "Capture", show: ["head", "capture"] },
];
export const modeOf = (id) => VIEW_MODES.find((m) => m.id === id) || VIEW_MODES[0];

/** Colours for wells and their capture zones: distinct, muted, readable on the blue head ramp. */
export const WELL_COLORS = ["#b8336a", "#d9701e", "#6b4fa1", "#2a7f62", "#8c6d1f", "#3d5a80"];
export const wellColor = (n) => WELL_COLORS[(Math.max(1, n) - 1) % WELL_COLORS.length];
const wellNumber = (id) => Number(String(id || "").replace(/^\D+/, "")) || 1;

/** Pixels per cell for the smooth surface: about 1,600 on the long side, 2 to 16 per cell. */
export function smoothScale(nrow, ncol) {
  return Math.max(2, Math.min(16, Math.floor(1600 / Math.max(nrow, ncol, 1))));
}

const HEAD_ALPHA = 215;
const UNDER_ZONES_ALPHA = 150;

/**
 * The smooth surface as RGBA pixels (north row first, `scale` pixels per cell): bilinear between cell centres,
 * missing neighbours dropped from the weights, and transparent over an inactive or dry cell.
 * `mode`: "head" (the head ramp), "capture" (the same, lighter, under the zones) or "drawdown" (values are the
 * change since the reference, the diverging ramp: brown fell, blue-green rose, fading toward no change).
 */
export function paintSmooth(values, nrow, ncol, range, mode, scale) {
  const w = ncol * scale, h = nrow * scale;
  const px = new Uint8ClampedArray(w * h * 4);
  const ramp = mode === "drawdown" ? CHANGE_RAMP : HEAD_RAMP;
  const at = (i, j) => values[i * ncol + j];
  for (let y = 0; y < h; y++) {
    const own = Math.floor(y / scale);
    const fi = (y + 0.5) / scale - 0.5;
    const i0 = Math.max(0, Math.min(nrow - 2, Math.floor(fi)));
    const i1 = Math.min(i0 + 1, nrow - 1);
    const a = Math.max(0, Math.min(1, fi - i0));
    for (let x = 0; x < w; x++) {
      const ownJ = Math.floor(x / scale);
      if (!Number.isFinite(at(own, ownJ))) continue;
      const fj = (x + 0.5) / scale - 0.5;
      const j0 = Math.max(0, Math.min(ncol - 2, Math.floor(fj)));
      const j1 = Math.min(j0 + 1, ncol - 1);
      const b = Math.max(0, Math.min(1, fj - j0));
      let tot = 0, ws = 0;
      for (const [ii, jj, wt] of [[i0, j0, (1 - a) * (1 - b)], [i0, j1, (1 - a) * b], [i1, j0, a * (1 - b)], [i1, j1, a * b]]) {
        const v = at(ii, jj);
        if (Number.isFinite(v) && wt > 0) { tot += v * wt; ws += wt; }
      }
      const v = ws > 1e-9 ? tot / ws : at(own, ownJ);
      const t = rangeT(v, range);
      const c = rampColor(ramp, t);
      const k = (y * w + x) * 4;
      px[k] = c[0]; px[k + 1] = c[1]; px[k + 2] = c[2];
      px[k + 3] = mode === "drawdown" ? surfaceAlpha(t, "change") : mode === "capture" ? UNDER_ZONES_ALPHA : HEAD_ALPHA;
    }
  }
  return { px, width: w, height: h };
}

/** The change since the reference step, per cell (negative where the water table fell): the drawdown, signed for
 * the diverging ramp. NaN where either step has no head. */
export function changeSince(frame, refFrame) {
  const out = new Float32Array(frame.length);
  for (let k = 0; k < frame.length; k++) out[k] = frame[k] - refFrame[k];
  return out;
}

/** A level as a label: as many decimals as the interval needs, the unit on index lines. */
export function levelLabel(level, step, unit = "") {
  const d = !step || Number.isInteger(step) ? 0 : Math.min(3, (String(Number(step.toPrecision(6))).split(".")[1] || "").length);
  const s = Number(level).toFixed(d);
  return unit ? `${s} ${unit}` : s;
}

/** Contour levels as line features: `index` every fifth interval (drawn heavier, like a map's index contours). */
export function contourFeatures(block, unit = "", { drawdown = false } = {}) {
  if (!block || !block.levels) return [];
  const step = block.step || 1;
  return block.levels.map((lv) => {
    const n = Math.round(lv.level / step);
    // drawdown levels are the fall (positive down), labelled the way a cone is read; a rise reads "+0.5 m"
    const rise = drawdown && lv.level < 0;
    return {
      type: "Feature",
      properties: { level: lv.level, index: n % 5 === 0, rise,
        label: rise ? `+${levelLabel(-lv.level, step, unit)}` : levelLabel(lv.level, step, unit) },
      geometry: { type: "MultiLineString", coordinates: lv.lines },
    };
  });
}

/** Flow arrows as points with their bearing and size. */
export function arrowFeatures(flow) {
  return ((flow && flow.arrows) || []).map((a) => ({
    type: "Feature", properties: { b: a.bearing, s: Math.round((0.5 + 0.45 * a.size) * 100) / 100 },
    geometry: { type: "Point", coordinates: [a.lon, a.lat] },
  }));
}

/** Wells as points: pumping now, in this layer or not. */
export function wellFeatures(wells) {
  return (wells || []).map((w) => ({
    type: "Feature",
    properties: { id: w.id, label: w.id, pumping: Boolean(w.pumping), here: w.in_layer !== false, color: wellColor(wellNumber(w.id)) },
    geometry: { type: "Point", coordinates: [w.lon, w.lat] },
  }));
}

/** Capture zones as polygons and their pathlines as lines, in each well's colour. */
export function captureFeatures(capture) {
  const zones = [], paths = [];
  for (const z of capture || []) {
    const color = wellColor(wellNumber(z.well));
    for (const ring of z.rings || []) {
      if (ring.length < 4) continue;
      const closed = ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1] ? ring : [...ring, ring[0]];
      zones.push({ type: "Feature", properties: { well: z.well, color, km2: z.area_km2 }, geometry: { type: "Polygon", coordinates: [closed] } });
    }
    for (const p of z.paths || []) {
      if (p.length > 1) paths.push({ type: "Feature", properties: { well: z.well, color }, geometry: { type: "LineString", coordinates: p } });
    }
  }
  return { zones, paths };
}

const fc = (features) => ({ type: "FeatureCollection", features });
export const EMPTY_FC = fc([]);

/** Every vector layer's data for a views result in a mode. */
export function viewFeatures(v, mode, unit = "") {
  if (!v || v.error) return { contours: EMPTY_FC, arrows: EMPTY_FC, wells: EMPTY_FC, zones: EMPTY_FC, paths: EMPTY_FC };
  const cap = mode === "capture" ? captureFeatures(v.capture) : { zones: [], paths: [] };
  const lines = mode === "drawdown" ? contourFeatures(v.drawdown, unit, { drawdown: true })
    : contourFeatures(v.head, unit);
  return {
    contours: fc(lines),
    arrows: fc(mode === "head" ? arrowFeatures(v.flow) : []),
    wells: fc(wellFeatures(v.wells)),
    zones: fc(cap.zones),
    paths: fc(cap.paths),
  };
}

const fmt = (x, d = 2) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? "" : Number(x).toFixed(d));
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** The typical hydraulic gradient as a reader says it: "about 1 in 220". */
export function gradientWords(g) {
  if (!Number.isFinite(g) || g <= 0) return "";
  const one = 1 / g;
  const r = one >= 1000 ? Math.round(one / 100) * 100 : one >= 100 ? Math.round(one / 10) * 10 : Math.round(one);
  return `about 1 in ${r.toLocaleString("en-GB")}`;
}

/** The rate a well pumps, in the model's units when it names them. */
export function rateWords(q, unit, timeUnits) {
  if (q === null || q === undefined || !Number.isFinite(q)) return "";
  const t = { days: "d", day: "d", seconds: "s", hours: "h", minutes: "min", years: "yr" }[String(timeUnits || "").toLowerCase()];
  const u = unit && t ? ` ${unit}³/${t}` : " (model units)";
  const v = Math.abs(q) >= 100 ? Math.round(Math.abs(q)).toLocaleString("en-GB") : fmt(Math.abs(q), 1);
  return q < 0 ? `pumping ${v}${u}` : q > 0 ? `injecting ${v}${u}` : "not pumping";
}

/** The legend's lines for a mode: short, what you see and how to read it. */
export function viewLines(v, mode, { unit = "", stepLabel = (t) => `step ${t + 1}` } = {}) {
  if (!v) return ["Drawing…"];
  if (v.error) return [v.error];
  const u = unit ? ` ${unit}` : "";
  const out = [];
  const pumping = (v.wells || []).filter((w) => w.pumping);
  if (mode === "head") {
    if (v.head && v.head.step) out.push(`Contours every ${levelLabel(v.head.step, v.head.step)}${u}, heavier every ${levelLabel(5 * v.head.step, v.head.step)}${u}.`);
    if (v.flow && v.flow.towards) out.push(`Arrows: groundwater flow, down the head gradient: mostly to the ${v.flow.towards}, ${gradientWords(v.flow.gradient)}.`);
  } else if (mode === "drawdown") {
    const d = v.drawdown || {};
    // a reference later than the step on show (the wells idle at the end of the run) is read "against", not "since"
    const later = Number.isInteger(v.time_index) && Number.isInteger(d.reference) && d.reference > v.time_index;
    out.push(`${later ? `Change against ${stepLabel(d.reference)}, a later step` : `Change since ${stepLabel(d.reference)}`}: ` +
      `brown where the water table is lower now, blue-green where it is higher` +
      `${d.step ? `; rings every ${levelLabel(d.step, d.step)}${u}` : ""}.`);
    const deep = (v.wells || []).filter((w) => Number.isFinite(w.drawdown)).sort((a, b) => b.drawdown - a.drawdown)[0];
    if (deep && deep.drawdown > 0) out.push(`Deepest at ${deep.id}: ${fmt(deep.drawdown)}${u} down.`);
  } else if (mode === "capture") {
    if (!pumping.length) out.push("No well pumps in this layer at this step, so nothing is captured.");
    else {
      out.push("Lines: where the water reaching each well comes from, this step's heads held still. Shaded: its capture zone.");
      out.push(pumping.map((w) => `${w.id} ${Number.isFinite(w.capture_km2) ? `${fmt(w.capture_km2, w.capture_km2 < 1 ? 2 : 1)} km²` : ""}`).join(" · "));
    }
  }
  if ((v.wells || []).length) {
    out.push(v.wells_from === "wel" ? `${plural(v.wells.length, "well")} from the model's .wel file, ${pumping.length} pumping now.`
      : `${plural(v.wells.length, "well")} found in the heads (no .wel file): where the drawdown peaks.`);
  }
  return out;
}

/** The map card for one well. */
export function wellCardOf(v, w, { unit = "", timeUnits = "days", when = "" } = {}) {
  const u = unit ? ` ${unit}` : "";
  const rate = rateWords(w.q, unit, timeUnits);
  const status = w.from === "wel"
    ? (w.in_layer ? (rate ? `${rate[0].toUpperCase()}${rate.slice(1)} at this step.` : "In this layer.")
      : `Screened in layer ${w.layer}: shown here, but it does not draw water from this layer.`)
    : (w.pumping ? "Found in the heads: water flows into this cell at this step." : "Found in the heads; not drawing water at this step.");
  return {
    what: w.from === "wel" ? "Well, from the model's .wel file" : "Well, found in the heads",
    title: `Well ${w.id}`,
    sub: `row ${w.model_row ?? w.row}, column ${w.model_col ?? w.col}, layer ${w.layer}${when ? ` · ${when}` : ""}`,
    status: { text: status, color: null },
    figure: Number.isFinite(w.drawdown) ? { value: fmt(w.drawdown), unit, label: "drawdown" }
      : Number.isFinite(w.head) ? { value: fmt(w.head), unit, label: "head" } : null,
    note: [Number.isFinite(w.head) ? `Head ${fmt(w.head)}${u}.` : "",
      Number.isFinite(w.capture_km2) ? `Capture zone about ${fmt(w.capture_km2, w.capture_km2 < 1 ? 2 : 1)} km² at this step.` : "",
      w.from === "found" ? v.wells_how : ""].filter(Boolean).join(" "),
  };
}
