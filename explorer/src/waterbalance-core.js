// Water balance on the map (#557), the pure part: no DOM, so node can test it.
//
// The numbers are made in Python (aquascope/waterbalance.py, in CI) and published under waterbalance/ in the
// Archive: one file of outlines, one per year of every variable and anomaly, the 1991-2020 normals. This file
// only decides the colours, the words and the card's bars, from those numbers as they are.

export const WB_BASE = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/waterbalance/";
export const WB_CREDIT = {
  label: "Water balance",
  attribution: 'ERA5 (Copernicus Climate Change Service, ECMWF) and <a href="https://www.geoglows.org">GEOGLOWS v2</a>; ' +
    "HydroBASINS level-4 subbasins (BasinATLAS, Linke et al. 2019)",
  licence: "CC BY 4.0",
};
export const WB_SHORT_CREDIT = "ERA5 (C3S/ECMWF) and GEOGLOWS v2, CC BY 4.0. Modelled.";

export const VARIABLES = [
  { id: "p", label: "Rain", word: "rain" },
  { id: "et", label: "Evaporation", word: "evaporation" },
  { id: "q", label: "Runoff", word: "runoff" },
  { id: "closure", label: "Closure", word: "closure error" },
];
export const MODES = [{ id: "amount", label: "Amount" }, { id: "anom", label: "vs normal" }];
export const NO_VALUE = "#c9ced3";

// mm per month. Sequential for the amounts, diverging for what has a sign. Runoff and the anomalies go brown to
// teal like the river status (colour-blind safe); the closure error is orange to purple, so it never reads as
// "wetter" or "drier": it is what the three products leave unexplained.
const RAMPS = {
  p: [[0, "#f7fbff"], [10, "#deebf7"], [25, "#c6dbef"], [50, "#9ecae1"], [100, "#6baed6"], [200, "#3182bd"],
    [400, "#08519c"]],
  et: [[0, "#ffffe5"], [10, "#f7fcb9"], [25, "#d9f0a3"], [50, "#addd8e"], [75, "#78c679"], [100, "#41ab5d"],
    [150, "#238443"], [200, "#005a32"]],
  q: [[-50, "#a6611a"], [-5, "#dfc27d"], [0, "#f5f5f5"], [5, "#c7eae5"], [20, "#80cdc1"], [50, "#35978f"],
    [100, "#01665e"], [200, "#003c30"]],
  closure: [[-100, "#b35806"], [-50, "#f1a340"], [-15, "#fee0b6"], [0, "#f7f7f7"], [15, "#d8daeb"], [50, "#998ec3"],
    [100, "#542788"]],
};
const ANOM_COLORS = ["#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5", "#c7eae5", "#5ab4ac", "#01665e"];
const ANOM_SPAN = { p: [10, 40, 100], et: [5, 15, 40], q: [5, 20, 50], closure: [10, 40, 100] };

// Why a subbasin's runoff is withheld though its reaches exist (the outline's "c", aquascope.waterbalance
// RUNOFF_CHECKS): fixed words only.
export const RUNOFF_WITHHELD = {
  outlet: "the reach paired with its outlet carries less than half of what flows in, so it is not the river leaving it.",
  through: "the river brings in more than a metre of water a month over it, so its own runoff is lost in the difference of two large flows.",
  ratio: "over the record its outlet carries more water than falls on it as rain, so the paired reach drains more than this subbasin.",
  upstream: "its inflow comes from a subbasin whose outlet reach fails these checks.",
};

/** The card's line on a subbasin without runoff, from the outline's properties, or null where it has runoff. */
export function runoffNote(props = {}) {
  if (String(props.r) !== "0") return null;
  const why = RUNOFF_WITHHELD[props.c];
  return why ? `Runoff withheld: ${why}` : "No runoff here: GEOGLOWS has no outlet reach for this subbasin or for one upstream of it.";
}

export const variableOf = (id) => VARIABLES.find((v) => v.id === id) || VARIABLES[0];

/** The colour stops, [[mm, colour], ...] ascending, for a variable shown as an amount or against its normal. */
export function rampStops(variable, mode = "amount") {
  const v = variableOf(variable).id;
  if (mode !== "anom") return RAMPS[v].map((s) => s.slice());
  const [a, b, c] = ANOM_SPAN[v];
  return [-c, -b, -a, 0, a, b, c].map((x, i) => [x, ANOM_COLORS[i]]);
}

/** The colour of one value on a ramp (linear between stops, clamped at the ends), as "#rrggbb". */
export function colorFor(value, stops) {
  if (!Number.isFinite(value) || !stops || !stops.length) return NO_VALUE;
  if (value <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [x1, c1] = stops[i];
    if (value <= x1) {
      const [x0, c0] = stops[i - 1];
      const f = (value - x0) / (x1 - x0 || 1);
      const a = hexRgb(c0), b = hexRgb(c1);
      return `#${a.map((v, j) => Math.round(v + (b[j] - v) * f).toString(16).padStart(2, "0")).join("")}`;
    }
  }
  return stops[stops.length - 1][1];
}

export function hexRgb(hex) {
  const h = String(hex).replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}

/** The MapLibre fill-color for the feature-state value "v": the ramp where it is a number, grey where not. */
export function fillColorExpression(variable, mode) {
  const stops = rampStops(variable, mode).flat();
  return ["case", ["==", ["typeof", ["feature-state", "v"]], "number"],
    ["interpolate", ["linear"], ["feature-state", "v"], ...stops], NO_VALUE];
}

/** The key of the year file's array for a variable and mode: "p_mm", "q_anom", ... */
export const fieldOf = (variable, mode) => `${variableOf(variable).id}_${mode === "anom" ? "anom" : "mm"}`;

/** Where ``month`` (YYYY-MM) is in a year file, or -1. */
export function monthIndex(doc, month) {
  return doc && Array.isArray(doc.months) ? doc.months.indexOf(month) : -1;
}

/** One map frame: the value of every subbasin (index k) for the month, null where it has none. */
export function frameValues(doc, month, variable, mode) {
  const i = monthIndex(doc, month);
  const arr = i >= 0 ? (doc[fieldOf(variable, mode)] || [])[i] : null;
  return Array.isArray(arr) ? arr : null;
}

/** The month the map date asks for, when the published table has it: "2026-09", or null. */
export function wbMonthFor(date, manifest) {
  const ym = String(date || "").slice(0, 7);
  if (!/^\d{4}-\d{2}$/.test(ym) || !manifest || !manifest.first_month || !manifest.last_month) return null;
  if (ym < manifest.first_month || ym > manifest.last_month) return null;
  return (manifest.years || []).includes(ym.slice(0, 4)) ? ym : null;
}

/** The layer as the time bar sees it (layers.js datedLayersOn). */
export function wbDatedLayer(manifest) {
  return {
    id: "water-balance", label: "Water balance", time: true, monthly: true,
    since: `${(manifest && manifest.first_month) || "1990-01"}-01`,
    until: manifest && manifest.last_month ? `${manifest.last_month}-01` : null,
    attribution: WB_CREDIT.attribution, licence: WB_CREDIT.licence, credit: "ERA5, GEOGLOWS, CC BY 4.0",
  };
}

const SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
export const monthLabel = (ym) => (/^\d{4}-\d{2}$/.test(ym || "") ? `${SHORT[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}` : "");
export const longMonth = (ym) => (/^\d{4}-\d{2}$/.test(ym || "") ? `${LONG[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}` : "");

/** The legend row's few words: "Rain, Sep 2026", "Runoff vs normal, Sep 2026", or why there is no map. */
export function rowSummary({ manifest, month, variable, mode, loading = false, failed = false, date = null }) {
  if (failed && !manifest) return "could not read";
  if (!manifest) return "loading";
  const what = `${variableOf(variable).label}${mode === "anom" ? " vs normal" : ""}`;
  if (month) return `${what}, ${monthLabel(month)}${loading ? ", loading" : ""}${failed ? ", could not read" : ""}`;
  const ym = String(date || "").slice(0, 7);
  return `${monthLabel(manifest.first_month)} to ${monthLabel(manifest.last_month)} only` +
    (ym && /^\d{4}-\d{2}$/.test(ym) ? `; none for ${monthLabel(ym)}` : "");
}

/** The ramp's three labels under the key: low end, middle, high end, in mm. */
export function rampEnds(variable, mode) {
  const s = rampStops(variable, mode);
  const fmt = (x) => (mode === "anom" || x < 0 ? `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x)}` : String(x));
  const mid = s[Math.floor(s.length / 2)][0];
  return [fmt(s[0][0]), fmt(mid), `${fmt(s[s.length - 1][0])} mm`];
}

/** The CSS gradient of a ramp, evenly spaced by stop (how the key draws it). */
export function rampCss(variable, mode) {
  const s = rampStops(variable, mode);
  return `linear-gradient(90deg, ${s.map((x, i) => `${x[1]} ${Math.round((i / (s.length - 1)) * 100)}%`).join(", ")})`;
}

/** What the variable means, in one line under the key. */
export function variableNote(variable, mode, { normals = true } = {}) {
  const v = variableOf(variable).id;
  if (mode === "anom" && !normals) {
    return "No 1991-2020 normal yet: the monthly job is still filling in those years.";
  }
  if (mode === "anom") return "Against the subbasin's 1991-2020 mean for the month.";
  return {
    p: "Rain over each subbasin in the month (ERA5).",
    et: "Water evaporated from each subbasin in the month (ERA5).",
    q: "Runoff made in each subbasin: outlet flow minus inflow (GEOGLOWS). Brown: the river loses water.",
    closure: "P − ET − Q as it comes out: storage change and the products' errors. Small over whole years.",
  }[v];
}

const fmt0 = (x) => `${Math.round(x).toLocaleString("en-US")}`;
const signed = (x) => `${x >= 0 ? "+" : "−"}${fmt0(Math.abs(x))}`;

/**
 * One subbasin's month in words: the same sentence as aquascope.waterbalance.basin_sentence, every number from
 * the file. `row`: { month, p_mm, et_mm, q_mm, closure_mm, p_anom }.
 */
export function basinSentence(row) {
  const parts = [];
  if (Number.isFinite(row.p_mm)) parts.push(`${fmt0(row.p_mm)} mm of rain`);
  if (Number.isFinite(row.et_mm)) parts.push(`${fmt0(row.et_mm)} mm evaporated`);
  if (Number.isFinite(row.q_mm)) {
    parts.push(row.q_mm >= 0 ? `${fmt0(row.q_mm)} mm ran off` : `the river lost ${fmt0(Math.abs(row.q_mm))} mm crossing it`);
  }
  if (!parts.length) return `${longMonth(row.month)}: no water balance for this subbasin.`;
  let text = `${longMonth(row.month)}: ${parts.join(", ")}`;
  if (Number.isFinite(row.closure_mm)) text += `; P − ET − Q leaves ${signed(row.closure_mm)} mm`;
  else if (!Number.isFinite(row.q_mm)) text += "; no runoff, so no closure";
  if (Number.isFinite(row.p_anom)) {
    text += `. Rain was ${fmt0(Math.abs(row.p_anom))} mm ${row.p_anom >= 0 ? "above" : "below"} the 1991-2020 normal`;
  }
  return `${text}.`;
}

/** A subbasin's row of a year file: { month, p_mm, ..., closure_anom, era5t } for month index i. */
export function rowOf(doc, i, k) {
  const out = { month: doc.months[i], era5t: Boolean(doc.era5t && doc.era5t[i]) };
  for (const v of ["p", "et", "q", "closure"]) {
    for (const m of ["mm", "anom"]) {
      const a = doc[`${v}_${m}`] && doc[`${v}_${m}`][i];
      const x = a ? a[k] : null;
      out[`${v}_${m}`] = Number.isFinite(x) ? x : null;
    }
  }
  return out;
}

const BAR = { p: "#4a8fc7", et: "#7fb069", q: "#1f8a80", loss: "#a6611a", tick: "#1f2933" };

/**
 * The card's bars: the year's months, rain beside evaporation with runoff stacked on it (the gap between the
 * two is the closure error), the normal rain as a short dark tick, the month on the map shaded. An SVG string.
 */
export function balanceBars(rows, { month = null, normalP = null, w = 264, h = 100 } = {}) {
  const top = 14, base = h - 16;   // the scale's label sits above the tallest bar, not on it
  const vals = [];
  rows.forEach((r, i) => {
    if (Number.isFinite(r.p_mm)) vals.push(r.p_mm);
    const out = (Number.isFinite(r.et_mm) ? r.et_mm : 0) + (Number.isFinite(r.q_mm) ? Math.max(0, r.q_mm) : 0);
    vals.push(out);
    const np = normalP && normalP[+String(r.month).slice(5, 7) - 1];   // the calendar month's normal
    if (Number.isFinite(np)) vals.push(np);
  });
  const lows = rows.map((r) => (Number.isFinite(r.q_mm) ? Math.min(0, r.q_mm) : 0));
  const hi = Math.max(1, ...vals);
  const lo = Math.min(0, ...lows);
  const span = hi - lo;
  const zero = top + ((hi - 0) / span) * (base - top);
  const y = (v) => top + ((hi - v) / span) * (base - top);
  const slot = w / 12, bw = Math.max(3, slot * 0.32);
  const parts = [`<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">`];
  rows.forEach((r, i) => {
    const m = +String(r.month).slice(5, 7) - 1;
    const x0 = m * slot;
    if (r.month === month) parts.push(`<rect x="${x0.toFixed(1)}" y="0" width="${slot.toFixed(1)}" height="${h - 12}" fill="rgba(31,41,51,0.07)" rx="2"/>`);
    const xp = x0 + slot / 2 - bw - 0.5, xo = x0 + slot / 2 + 0.5;
    if (Number.isFinite(r.p_mm) && r.p_mm > 0) {
      parts.push(`<rect x="${xp.toFixed(1)}" y="${y(r.p_mm).toFixed(1)}" width="${bw.toFixed(1)}" height="${(zero - y(r.p_mm)).toFixed(1)}" fill="${BAR.p}"/>`);
    }
    let stack = 0;
    if (Number.isFinite(r.et_mm) && r.et_mm > 0) {
      parts.push(`<rect x="${xo.toFixed(1)}" y="${y(r.et_mm).toFixed(1)}" width="${bw.toFixed(1)}" height="${(zero - y(r.et_mm)).toFixed(1)}" fill="${BAR.et}"/>`);
      stack = r.et_mm;
    }
    if (Number.isFinite(r.q_mm) && r.q_mm > 0) {
      parts.push(`<rect x="${xo.toFixed(1)}" y="${y(stack + r.q_mm).toFixed(1)}" width="${bw.toFixed(1)}" height="${(y(stack) - y(stack + r.q_mm)).toFixed(1)}" fill="${BAR.q}"/>`);
    } else if (Number.isFinite(r.q_mm) && r.q_mm < 0) {
      parts.push(`<rect x="${xo.toFixed(1)}" y="${zero.toFixed(1)}" width="${bw.toFixed(1)}" height="${(y(r.q_mm) - zero).toFixed(1)}" fill="${BAR.loss}"/>`);
    }
    const np = normalP && normalP[m];
    if (Number.isFinite(np)) {
      parts.push(`<line x1="${(xp - 1).toFixed(1)}" x2="${(xp + bw + 1).toFixed(1)}" y1="${y(np).toFixed(1)}" y2="${y(np).toFixed(1)}" stroke="${BAR.tick}" stroke-width="1.4"/>`);
    }
  });
  parts.push(`<line x1="0" x2="${w}" y1="${zero.toFixed(1)}" y2="${zero.toFixed(1)}" stroke="#9aa5b1" stroke-width="0.8"/>`);
  for (let m = 0; m < 12; m++) {
    const ym = rows.length ? `${String(rows[0].month).slice(0, 4)}-${String(m + 1).padStart(2, "0")}` : "";
    parts.push(`<text x="${(m * slot + slot / 2).toFixed(1)}" y="${h - 3}" text-anchor="middle" font-size="9" ` +
      `fill="${ym === month ? "#1f2933" : "#7b8794"}"${ym === month ? ' font-weight="600"' : ""}>${LONG[m][0]}</text>`);
  }
  parts.push(`<text x="2" y="${top - 5}" font-size="8.5" fill="#7b8794">${fmt0(hi)} mm</text>`);
  parts.push("</svg>");
  return parts.join("");
}

/** The bars' key, as HTML (fixed words only). */
export function barsKey({ normal = false, loss = false } = {}) {
  const sw = (c, t) => `<span class="wb-k"><i style="background:${c}"></i>${t}</span>`;
  return sw(BAR.p, "rain") + sw(BAR.et, "evaporation") + sw(BAR.q, "runoff") + (loss ? sw(BAR.loss, "river loses") : "") +
    (normal ? '<span class="wb-k"><i class="wb-tick"></i>normal rain</span>' : "");
}

/**
 * The map card for subbasin k in `month`: { title, sub, status, chart, note }.
 * `props`: the outline's properties { id, a (km²), r (runoff possible: 1 or 0) }.
 */
export function basinCard(doc, k, month, props = {}, { normals = null, variable = "p", mode = "amount" } = {}) {
  const i = monthIndex(doc, month);
  const area = Number(props.a);
  const title = "Level-4 subbasin";
  const sub = `HydroBASINS ${props.id || ""}${Number.isFinite(area) && area > 0 ? ` · ${fmt0(area)} km²` : ""}`;
  if (i < 0) return { title, sub, status: { text: `No water balance for ${longMonth(month)}.`, muted: true }, chart: null, note: null };
  const row = rowOf(doc, i, k);
  const v = fieldOf(variable, mode);
  const color = colorFor(row[v], rampStops(variable, mode));
  const rows = doc.months.map((_, j) => rowOf(doc, j, k));
  const normalP = normals && Array.isArray(normals.p) ? normals.p.map((mrow) => (mrow ? mrow[k] : null)) : null;
  const hasNormal = Boolean(normalP && normalP.some(Number.isFinite));
  const notes = [];
  const rn = runoffNote(props);
  if (rn) notes.push(rn);
  if (row.era5t) notes.push("ERA5T, preliminary: recomputed once ERA5 is final (about three months on).");
  const full = rows.length === 12 && rows.every((r) => Number.isFinite(r.closure_mm));
  if (full) {
    const sum = rows.reduce((s, r) => s + r.closure_mm, 0);
    notes.push(`Over ${doc.year}, P − ET − Q adds up to ${signed(sum)} mm.`);
  }
  return {
    title, sub,
    status: { text: basinSentence(row), color },
    chart: {
      svg: balanceBars(rows, { month, normalP: hasNormal ? normalP : null }),
      alt: `Rain, evaporation and runoff in each month of ${doc.year}`,
      key: barsKey({ normal: hasNormal, loss: rows.some((r) => Number.isFinite(r.q_mm) && r.q_mm < 0) }),
      label: `${doc.year}, mm per month`,
    },
    note: notes.join(" ") || null,
  };
}

/** The hash: wb=1 turns it on, wbv the variable, wbm=anom against normal. null when the hash says nothing. */
export function readWbParam(hash) {
  const q = new URLSearchParams(String(hash || "").replace(/^#/, ""));
  if (!q.has("wb")) return null;
  const v = q.get("wbv");
  return {
    on: q.get("wb") === "1",
    variable: VARIABLES.some((x) => x.id === v) ? v : "p",
    mode: q.get("wbm") === "anom" ? "anom" : "amount",
  };
}

/** Whether any subbasin has a normal (the anomaly view needs one). */
export function hasNormals(normals) {
  return Boolean(normals && Array.isArray(normals.p) && normals.p.some((m) => Array.isArray(m) && m.some(Number.isFinite)));
}
