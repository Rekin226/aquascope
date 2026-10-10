// The FEWS view (#556), the pure part: no DOM, no map, so node can test it.
// Forecast points coloured by the threshold their forecast reaches, and one
// click giving the ensemble plume, the way Delft-FEWS and the GloFAS/EFAS
// viewers show them. Every number comes from aquascope.nownext (plume() for a
// river reach, forecast_points() for the Archive's forecast gauges); this
// module only decides the colours, the words and the drawing.
//
// The colours. The classes are the Floods ahead ones (floods-ahead-core.js), so
// a reach and a gauge passing the 5-year flow wear the same orange. "Below the
// 2-year flow" is a quiet slate, and on the map it is a hollow ring rather than
// a solid mark, so it is told apart by shape as well as by colour. Checked with
// colourDistances() below (Machado, Oliveira and Fernandes 2009, full
// severity): every pair of neighbouring classes stays at least 10 CIE76 units
// apart under simulated protanopia, deuteranopia and tritanopia, and the slate
// at least 20 from every class. Lightness falls with the class for normal
// vision; under protanopia only the 25- and 50-year pair swap lightness, and
// they stay far apart in hue.

import { FLOOD_CLASSES, addDays, shortDay } from "./floods-ahead-core.js?v=__BUILD__";

export const BELOW = { rp: 0, label: "below the 2-year flow", color: "#7d8fa1" };
export const FEWS_CLASSES = [BELOW, ...FLOOD_CLASSES];
// Days of a gauge's record drawn before the forecast starts (aquascope.nownext.PLUME_OBS_DAYS).
export const OBS_DAYS = 21;

export const fewsColor = (rp) => (FEWS_CLASSES.find((c) => c.rp === Number(rp || 0)) || BELOW).color;
export const classWords = (rp) => (Number(rp) ? `the ${Number(rp)}-year flow` : BELOW.label);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));
const ISO = /^(\d{4})-(\d{2})-(\d{2})/;
const dayNo = (iso) => {
  const m = ISO.exec(String(iso || ""));
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3]) / 86400000 : null;
};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const dm = (iso) => {
  const m = ISO.exec(String(iso || ""));
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}` : "";
};

// A flow as the card prints it: the precision follows the magnitude (nownext._fmt_q).
export function flowText(x) {
  if (!finite(x)) return "";
  const n = Number(x);
  const a = Math.abs(n);
  if (a >= 100) return Math.round(n).toLocaleString("en-GB");
  if (a >= 10) return n.toFixed(1);
  if (a === 0) return "0";
  return String(Number(n.toPrecision(2)));
}

// ── colour vision ───────────────────────────────────────────────────────────

// Machado, Oliveira and Fernandes (2009), severity 1, applied to linear RGB.
const CVD = {
  protanopia: [0.152286, 1.052583, -0.204868, 0.114503, 0.786281, 0.099216, -0.003882, -0.048116, 1.051998],
  deuteranopia: [0.367322, 0.860646, -0.227968, 0.280085, 0.672501, 0.047413, -0.011820, 0.042940, 0.968881],
  tritanopia: [1.255528, -0.076749, -0.178779, -0.078411, 0.930809, 0.147602, 0.004733, 0.691367, 0.303900],
};
const linear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const rgb = (hex) => [1, 3, 5].map((i) => linear(parseInt(String(hex).slice(i, i + 2), 16) / 255));

/** CIELAB (D65) of a hex colour, as seen with full `kind` colour blindness (or normal vision). */
export function labOf(hex, kind = null) {
  let [r, g, b] = rgb(hex);
  const m = kind && CVD[kind];
  if (m) {
    const c = [r, g, b];
    [r, g, b] = [0, 1, 2].map((i) => Math.min(1, Math.max(0, m[i * 3] * c[0] + m[i * 3 + 1] * c[1] + m[i * 3 + 2] * c[2])));
  }
  const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
  const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
}

/** Colour difference (CIE76) between neighbouring classes, for each kind of vision. */
export function colourDistances(colors = FLOOD_CLASSES.map((c) => c.color)) {
  const out = {};
  for (const kind of [null, ...Object.keys(CVD)]) {
    const lab = colors.map((c) => labOf(c, kind));
    out[kind || "normal"] = lab.slice(1).map((l, i) => Math.hypot(l[0] - lab[i][0], l[1] - lab[i][1], l[2] - lab[i][2]));
  }
  return out;
}

// ── words ───────────────────────────────────────────────────────────────────

/** The class line over a forecast gauge's plume: the class its forecast reaches, or that it has none. */
export function plumeHead(p) {
  if (!p) return "";
  if (!p.thresholds) return "Forecast: no return-period flows here to class against.";
  if (!Number(p.rp)) return `Forecast: ${BELOW.label} on every day.`;
  const first = p.first_date ? `, from ${shortDay(p.first_date)}` : "";
  return `Forecast: ${classWords(p.rp)} at its highest${first}.`;
}

/** The quiet label over a plume: how far it reaches, what it is, and the run it comes from. */
export function plumeLabel(p, { corrected = false } = {}) {
  if (!p) return "";
  const days = (p.date || []).length || 15;
  const what = corrected ? "corrected to this gauge" : "modelled";
  const run = p.issued ? `, run of ${dm(p.issued)}` : "";
  return `Next ${days} days, ${what}${run}`;
}

/** The plume as a sentence for a screen reader. */
export function plumeAlt(p) {
  if (!p || !(p.date || []).length) return "No forecast to draw.";
  const n = p.date.length;
  const peak = finite(p.peak) ? `, the ensemble mean peaking at ${flowText(p.peak)} m³/s on ${dm(p.peak_date)}` : "";
  const lines = p.thresholds ? `; dashed lines at the ${p.thresholds.return_periods.join(", ")}-year flows` : "";
  const obs = p.observed && p.observed.t && p.observed.t.length ? "; the gauge's own record before it" : "";
  return `Forecast flow for the ${n} days from ${dm(p.date[0])}: median, middle half and full range of the ensemble${peak}${lines}${obs}.`;
}

/** The key under the chart. */
// The card's one line on the members (aquascope.nownext.plume's share): the 2-year flow, the reach's class and the
// highest flow any member reaches. The package's members_line names every return period; the card keeps it short.
export function membersShort(p) {
  const n = Number(p && p.n_members);
  const share = p && p.share;
  if (!share || !n) return "";
  const count = (t) => (finite(share[String(t)]) ? Math.round(Number(share[String(t)]) * n) : null);
  const k2 = count(2);
  if (k2 === null) return "";
  if (!k2) return `None of the ${n} members reaches the 2-year flow.`;
  const years = FLOOD_CLASSES.map((c) => c.rp).filter((t) => count(t));
  const picks = [2];
  const rp = Number(p.rp) || 0;
  if (rp > 2 && count(rp)) picks.push(rp);
  const top = years.at(-1);
  if (top > picks.at(-1)) picks.push(top);
  const first = k2 === n ? `All ${n} members reach` : `${k2} of the ${n} members ${k2 === 1 ? "reaches" : "reach"}`;
  const rest = picks.slice(1).map((t) => `${count(t) === n ? "all" : count(t)} the ${t}-year`);
  const tail = rest.length ? (rest.length === 1 ? ` and ${rest[0]}` : `, ${rest[0]} and ${rest[1]}`) : "";
  return `${first} the 2-year flow${tail}.`;
}

export function plumeKey(p) {
  const bits = [
    `<span><i class="pl-k-med"></i>median</span>`,
    `<span><i class="pl-k-mid"></i>middle half</span>`,
    `<span><i class="pl-k-all"></i>full range</span>`,
  ];
  if (p && (p.class_daily || []).some((c) => Number(c))) {
    const strip = FLOOD_CLASSES.slice(0, 3).map((c) => `<b style="background:${c.color}"></b>`).join("");
    bits.push(`<span><i class="pl-k-cls">${strip}</i>class by day</span>`);
  }
  if (p && p.observed && p.observed.t && p.observed.t.length) bits.push(`<span><i class="pl-k-obs"></i>observed</span>`);
  return bits.join("");
}

// ── the drawing ─────────────────────────────────────────────────────────────

/**
 * The top of the flow axis and the return-period lines to draw. The axis follows the bulk of the forecast (the
 * middle half, the median and the gauge's record), up to `reach` times its top: a lone wet member can
 * run ten times higher than the rest, and would flatten everything else, so the full range is cut there and its
 * real top named (`clipped`). Every line under the bulk's top is drawn, and the next one above when it falls
 * inside the axis; a line further up is named as `above` instead.
 */
export function plumeScale(p, { reach = 3 } = {}) {
  const bulkVals = [];
  for (const k of ["p75", "median"]) for (const v of p[k] || []) if (finite(v)) bulkVals.push(Number(v));
  for (const v of (p.observed && p.observed.v) || []) if (finite(v)) bulkVals.push(Number(v));
  const tops = (p.max || []).filter(finite).map(Number);
  const bulk = bulkVals.length ? Math.max(...bulkVals) : 0;
  const top = Math.max(bulk, ...tops, 0);
  const ceiling = bulk > 0 ? Math.min(top, bulk * reach) : top;
  const lines = [];
  let above = null;
  const t = p.thresholds;
  if (t && t.q) {
    t.return_periods.forEach((rp, i) => {
      const q = Number(t.q[i]);
      if (!finite(q)) return;
      if (q <= bulk) lines.push({ rp: Number(rp), q });
      else if (!above) above = { rp: Number(rp), q };
    });
  }
  if (above && bulk > 0 && above.q <= bulk * reach) {
    lines.push(above);
    above = null;
  }
  const ymax = Math.max(ceiling, ...lines.map((l) => l.q)) * 1.08 || 1;
  return { ymax, lines, above, clipped: top > ymax ? top : null };
}

const axisText = (v) => (v >= 10 ? Math.round(v).toLocaleString("en-GB") : String(Number(v.toPrecision(2))));

/** "1,000" style axis numbers on a round step. */
function niceTop(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/**
 * The plume as an SVG string, `w` by `h` pixels: the full range and the middle half as bands, the median as a line,
 * the return-period flows as dashed lines in their class colours (labelled in a gutter on the right), the gauge's
 * own record before the run as a line with dots, a mark at the run's start, a strip under the days in each day's
 * class colour (the ensemble mean, the Floods ahead rule), and the map's date when it falls inside.
 */
export function plumeSvg(p, { w = 300, h = 132, mapDate = null } = {}) {
  const dates = p.date || [];
  if (dates.length < 2) return "";
  const start = dayNo(dates[0]);
  const end = dayNo(dates[dates.length - 1]);
  const obs = (p.observed && p.observed.t) ? p.observed : { t: [], v: [] };
  const obsStart = obs.t.length ? Math.min(dayNo(obs.t[0]), start) : start;
  const gutter = 34;
  const L = 2, T = 14, B = 20;
  const R = w - gutter;
  const H = h - T - B;
  const scale = plumeScale(p);
  const ymax = niceTop(scale.ymax);
  const x = (d) => L + ((d - obsStart) / Math.max(1, end - obsStart)) * (R - L);
  const y = (v) => T + H * (1 - Math.min(ymax, Math.max(0, Number(v))) / ymax);
  const r = (k) => Math.round(k * 10) / 10;
  const day = (i) => start + i;
  const band = (lo, hi) => {
    const top = [];
    const bot = [];
    for (let i = 0; i < dates.length; i++) {
      if (!finite(lo[i]) || !finite(hi[i])) continue;
      top.push(`${r(x(day(i)))},${r(y(hi[i]))}`);
      bot.unshift(`${r(x(day(i)))},${r(y(lo[i]))}`);
    }
    return top.length > 1 ? `M${top.join("L")}L${bot.join("L")}Z` : "";
  };
  const line = (vals, at) => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (!finite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${r(x(at(i)))},${r(y(v))}`;
      pen = true;
    });
    return d;
  };
  const parts = [];
  // Light grid: the axis top and the half way line, so a flow can be read off.
  parts.push(`<line class="pl-grid" x1="${L}" x2="${R}" y1="${r(y(ymax))}" y2="${r(y(ymax))}"/>`);
  parts.push(`<line class="pl-grid" x1="${L}" x2="${R}" y1="${r(y(ymax / 2))}" y2="${r(y(ymax / 2))}"/>`);
  parts.push(`<line class="pl-base" x1="${L}" x2="${R}" y1="${r(y(0))}" y2="${r(y(0))}"/>`);
  parts.push(`<text class="pl-ax" x="${L}" y="${r(y(ymax) - 3)}">${esc(axisText(ymax))} m³/s</text>`);
  // The map's date, when it is one of the drawn days.
  const md = dayNo(mapDate);
  if (md !== null && md >= obsStart && md <= end) {
    parts.push(`<rect class="pl-today" x="${r(x(md) - 3)}" y="${T}" width="6" height="${r(H)}" rx="2"/>`);
  }
  parts.push(`<path class="pl-all" d="${band(p.min || [], p.max || [])}"/>`);
  parts.push(`<path class="pl-mid" d="${band(p.p25 || [], p.p75 || [])}"/>`);
  // Return-period flows: dashed, in the class colour, under the median so the forecast stays on top.
  const labels = [];
  for (const l of scale.lines) {
    const yy = r(y(l.q));
    parts.push(`<line class="pl-thr" x1="${L}" x2="${R}" y1="${yy}" y2="${yy}" style="stroke:${fewsColor(l.rp)}"/>`);
    labels.push({ y: yy, rp: l.rp });
  }
  // Labels in the gutter, nudged apart when two lines sit close.
  labels.sort((a, b) => b.y - a.y);
  let last = Infinity;
  for (const lab of labels) {
    const yy = Math.min(lab.y + 3.5, last - 10);
    last = yy;
    parts.push(`<rect class="pl-thr-sw" x="${R + 4}" y="${r(yy - 6)}" width="3" height="7" rx="1" style="fill:${fewsColor(lab.rp)}"/>`);
    parts.push(`<text class="pl-thr-t" x="${R + 9}" y="${r(yy)}">${lab.rp}-yr</text>`);
  }
  if (scale.above) {
    parts.push(`<text class="pl-thr-t pl-above" x="${R + 4}" y="${T + 9}">${scale.above.rp}-yr</text>`);
    parts.push(`<text class="pl-thr-t pl-above" x="${R + 4}" y="${T + 19}">↑ ${esc(flowText(scale.above.q))}</text>`);
  }
  if (scale.clipped) {
    parts.push(`<text class="pl-ax" x="${R - 2}" y="${r(y(ymax) - 3)}" text-anchor="end">range runs to ${esc(flowText(scale.clipped))} ↑</text>`);
  }
  parts.push(`<path class="pl-med" d="${line(p.median || [], day)}"/>`);
  // The gauge's own record: a line with a dot a day.
  if (obs.t.length) {
    const od = obs.t.map(dayNo);
    parts.push(`<path class="pl-obs" d="${line(obs.v, (i) => od[i])}"/>`);
    obs.v.forEach((v, i) => {
      if (finite(v)) parts.push(`<circle class="pl-obs-dot" cx="${r(x(od[i]))}" cy="${r(y(v))}" r="1.7"/>`);
    });
  }
  // The run's start: a dotted line from the axis up, its day under it.
  const x0 = r(x(start));
  parts.push(`<line class="pl-run" x1="${x0}" x2="${x0}" y1="${T + 10}" y2="${r(y(0))}"/>`);
  // Each day's class, a strip under the days.
  const cls = p.class_daily || [];
  const cw = (R - L) / Math.max(1, end - obsStart);
  cls.forEach((c, i) => {
    if (!Number(c)) return;
    parts.push(`<rect class="pl-cls" x="${r(x(day(i)) - cw / 2 + 0.5)}" y="${r(y(0) + 2)}" width="${r(Math.max(1, cw - 1))}" height="3" style="fill:${fewsColor(c)}"/>`);
  });
  // Days along the bottom: the first drawn, the run's start (bold), a week on when there is room, and the last.
  const ticks = [];
  const early = obsStart < start - 3;
  if (early) ticks.push({ d: obsStart, a: "start" });
  ticks.push({ d: start, a: early ? "middle" : "start", run: true });
  if (!early && end - start > 9) ticks.push({ d: start + 7, a: "middle" });
  ticks.push({ d: end, a: "end" });
  for (const t of ticks) {
    const iso = new Date(t.d * 86400000).toISOString().slice(0, 10);
    parts.push(`<text class="pl-ax${t.run ? " pl-ax-run" : ""}" x="${r(x(t.d))}" y="${h - 4}" text-anchor="${t.a}">${esc(dm(iso))}</text>`);
  }
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">${parts.join("")}</svg>`;
}

/** The forecast gauges as map features (positions from the catalogue), classed ones only, highest class on top. */
export function pointsGeoJSON(points, positions, { day = -1 } = {}) {
  const feats = [];
  for (const pt of points || []) {
    if (!pt.classed) continue;
    const at = positions(pt.key);
    if (!at || !finite(at.lat) || !finite(at.lon)) continue;
    const c = day >= 0 ? Number((pt.class_daily || [])[day] || 0) : Number(pt.rp || 0);
    feats.push({ type: "Feature", geometry: { type: "Point", coordinates: [Number(at.lon), Number(at.lat)] },
      properties: { key: pt.key, c } });
  }
  feats.sort((a, b) => a.properties.c - b.properties.c);
  return { type: "FeatureCollection", features: feats };
}

/** "10 Oct" for an ISO day. */
export const dayLabel = dm;

/** Which day of a forecast the map's date is, or -1 outside it. */
export function dayOf(dates, mapDate) {
  const a = dayNo((dates || [])[0]);
  const b = dayNo(mapDate);
  if (a === null || b === null) return -1;
  const i = b - a;
  return i >= 0 && i < (dates || []).length ? i : -1;
}

/** The legend's one line about the forecast gauges. */
export function pointsLine(res) {
  if (!res || !res.points) return "";
  const classed = res.points.filter((p) => p.classed);
  if (!classed.length) return "";
  const up = classed.filter((p) => Number(p.rp) > 0).length;
  const when = res.issue_date ? `, ${shortDay(res.issue_date)}` : "";
  const n = `${classed.length.toLocaleString("en-GB")} forecast gauge${classed.length === 1 ? "" : "s"}`;
  if (!up) return `${n}${when}: none expected to reach the 2-year flow.`;
  return `${n}${when}: ${up} expected to reach the 2-year flow or more.`;
}

/** Gauge record days to draw before a forecast that starts on `first` ({t, v} in, {t, v} out). */
// A record that stops more than STALE_DAYS before the run is left out: a stub weeks earlier squeezes the plume
// and says little. recordEnd() gives the day it stops, for the card to say so.
export const STALE_DAYS = 7;
export function obsBefore(series, first, days = OBS_DAYS) {
  const t = (series && series.t) || [];
  const v = (series && series.v) || [];
  const f = dayNo(first);
  if (f === null || !t.length) return null;
  const out = { t: [], v: [] };
  let last = null;
  for (let i = 0; i < t.length; i++) {
    const d = dayNo(t[i]);
    if (d === null || d < f - days || d > f + 30 || !finite(v[i])) continue;
    out.t.push(String(t[i]).slice(0, 10));
    out.v.push(Number(v[i]));
    last = Math.max(last ?? d, d);
  }
  return out.t.length && last >= f - STALE_DAYS ? out : null;
}
/** The last day of a record before ``first`` (ISO), or null. */
export function recordEnd(series, first) {
  const t = (series && series.t) || [];
  const v = (series && series.v) || [];
  const f = dayNo(first);
  let best = null;
  for (let i = 0; i < t.length; i++) {
    const d = dayNo(t[i]);
    if (d !== null && d < f && finite(v[i]) && (best === null || d > dayNo(best))) best = String(t[i]).slice(0, 10);
  }
  return best;
}

export { addDays };
