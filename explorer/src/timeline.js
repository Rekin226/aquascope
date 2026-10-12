// Time on the map (#522): the pure part. Dates, steps, ranges, which layer can
// show which day, and how the time state goes in and out of the URL.
//
// No DOM, no map, no window: node imports this directly (explorer/tests/
// timeline.test.mjs). The map date itself lives in core.js (state.date and the
// onTime bus); time-ui.js is the control that drives it.
//
// Dates are plain UTC days, "YYYY-MM-DD", because that is what the GIBS tiles
// take and what a hydrograph's x axis carries. The same stepping rules live in
// aquascope/map_time.py, which the CLI and the MCP server use.

import { imageFor } from "./layers.js?v=__BUILD__";

export const STEPS = ["day", "week", "month"];
export const DEFAULT_STEP = "day";
// How many steps a range covers when the reader presses play without setting one.
export const DEFAULT_SPAN = 12;
// The most frames a play loop or a GIF will walk (a GIF caps lower, see GIF_FRAMES).
export const MAX_FRAMES = 120;
export const GIF_FRAMES = 40;

const ISO = /^(\d{4})-(\d{2})-(\d{2})$/;

export function isIsoDate(s) {
  const m = ISO.exec(String(s || ""));
  if (!m) return false;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return d.getUTCFullYear() === +m[1] && d.getUTCMonth() === +m[2] - 1 && d.getUTCDate() === +m[3];
}

export const todayIso = (now = new Date()) => now.toISOString().slice(0, 10);

const toDate = (iso) => {
  const m = ISO.exec(iso);
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
};
const iso = (d) => d.toISOString().slice(0, 10);
const daysInMonth = (y, m0) => new Date(Date.UTC(y, m0 + 1, 0)).getUTCDate();

/**
 * Move a date by n steps. A month step keeps the day of the month where it can,
 * so 31 January plus one month is 28 (or 29) February, not 3 March.
 */
export function addStep(date, step = DEFAULT_STEP, n = 1) {
  if (!isIsoDate(date)) return date;
  const d = toDate(date);
  if (step === "week") d.setUTCDate(d.getUTCDate() + 7 * n);
  else if (step === "month") {
    const total = d.getUTCFullYear() * 12 + d.getUTCMonth() + n;
    const y = Math.floor(total / 12), m0 = total - y * 12;
    return iso(new Date(Date.UTC(y, m0, Math.min(d.getUTCDate(), daysInMonth(y, m0)))));
  } else d.setUTCDate(d.getUTCDate() + n);
  return iso(d);
}

export function clampDate(date, min, max) {
  if (!isIsoDate(date)) return date;
  if (min && date < min) return min;
  if (max && date > max) return max;
  return date;
}

/** Every date from `from` to `to` (both included) at `step`, at most `cap` of them. */
export function frameDates(from, to, step = DEFAULT_STEP, cap = MAX_FRAMES) {
  if (!isIsoDate(from) || !isIsoDate(to)) return { dates: [], truncated: false };
  let a = from, b = to;
  if (b < a) [a, b] = [b, a];
  const dates = [];
  for (let i = 0; ; i++) {
    const d = addStep(a, step, i);   // from the start each time, so month steps do not drift
    if (d > b) return { dates, truncated: false };
    if (dates.length >= cap) return { dates, truncated: true };
    dates.push(d);
  }
}

/** The range a play or a GIF walks when the reader has not set one: DEFAULT_SPAN steps ending at the date. */
export function defaultRange(date, step = DEFAULT_STEP, today = todayIso()) {
  const to = clampDate(date, null, today);
  return { from: addStep(to, step, -(DEFAULT_SPAN - 1)), to };
}

export function normaliseRange(range) {
  if (!range || !isIsoDate(range.from) || !isIsoDate(range.to)) return null;
  return range.from <= range.to ? { from: range.from, to: range.to } : { from: range.to, to: range.from };
}

/** The frame after `date` inside a list of frames, wrapping round to the first. */
export function nextFrame(dates, date) {
  if (!dates.length) return null;
  const i = dates.indexOf(date);
  if (i < 0) return dates.find((d) => d > date) || dates[0];
  return dates[(i + 1) % dates.length];
}

// ── what a layer can show ───────────────────────────────────────────────────

/**
 * A dated layer's first and last day; an open end is today. A monthly product's
 * last image covers its whole month, so its last day is the end of that month.
 */
export function layerSpan(layer, today = todayIso()) {
  const since = (layer && layer.since) || null;
  if (!layer || !layer.until) return { since, until: today };
  if (!layer.monthly) return { since, until: layer.until };
  const d = toDate(layer.until);
  return { since, until: iso(new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), daysInMonth(d.getUTCFullYear(), d.getUTCMonth())))) };
}

/** Whether a date falls inside a dated layer's first and last day. */
function inSpan(layer, date, today) {
  const { since, until } = layerSpan(layer, today);
  return (!since || date >= since) && date <= until;
}

/**
 * Whether a dated layer has data for a date. A layer that lists its images
 * (`periods`, GRACE) knows its gaps too; for the others only the span is known.
 */
export function layerCovers(layer, date, today = todayIso()) {
  if (!layer || !layer.time || !isIsoDate(date)) return true;
  if (!inSpan(layer, date, today)) return false;
  return !layer.periods || Boolean(imageFor(layer, date));
}

/** The dated layers in `layers` that cannot show `date`, for the "no data for this day" note. */
export function layersMissing(layers, date, today = todayIso()) {
  return (layers || []).filter((l) => l && l.time && !layerCovers(l, date, today));
}

/** "14 Jul 2021": a fixed format, so the note reads the same in every locale and in tests. */
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function shortDate(date) {
  if (!isIsoDate(date)) return String(date || "");
  const [y, m, d] = date.split("-");
  return `${+d} ${MONTHS[+m - 1]} ${y}`;
}

/** The note for a layer that cannot show a date: when it runs, or that this month is a gap. */
export function missingNote(layer, date, today = todayIso()) {
  if (isIsoDate(date) && inSpan(layer, date, today)) {
    return `${layer.label}: no image for ${shortDate(date).replace(/^\d+ /, "")}.`;
  }
  return `${layer.label}: ${spanLabel(layer)} only.`;
}

/** One line that says when a layer runs, for the out-of-range note. Monthly products in months. */
export function spanLabel(layer) {
  const fmt = (d) => (layer.monthly ? shortDate(d).replace(/^\d+ /, "") : shortDate(d));
  const since = layer.since ? fmt(layer.since) : "the start";
  return layer.until ? `${since} to ${fmt(layer.until)}` : `${since} onwards`;
}

// ── a date out of a chart click ─────────────────────────────────────────────

/**
 * The day a Plotly point stands for, or null when the x value is not a date
 * (a month name, a return period, a year as a number). Plotly hands back
 * "2021-07-14", "2021-07-14 06:00", a pandas isoformat with an offset or a
 * Date, depending on the trace.
 */
export function chartDate(x) {
  if (x instanceof Date) return Number.isFinite(x.getTime()) ? iso(x) : null;
  const m = /^(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/
    .exec(String(x ?? "").trim());
  return m && isIsoDate(m[1]) ? m[1] : null;
}

// ── the URL ─────────────────────────────────────────────────────────────────
//
// d=2024-05-01            the map date
// ts=week                 the play step (left out for a day)
// r=2024-01-01..2024-06-30 the range a play or a GIF walks
// cmp=2023-05-01          swipe compare on, the right side's date
// cl=precip               the right side's layer (left out: the same layers as the left)

/** Read the time part of a parsed hash (a URLSearchParams). Anything malformed is ignored, not guessed. */
export function readTimeParams(q) {
  const out = {};
  const d = q.get("d");
  if (isIsoDate(d)) out.date = d;
  const ts = q.get("ts");
  if (STEPS.includes(ts)) out.step = ts;
  const r = String(q.get("r") || "").split("..");
  const range = r.length === 2 ? normaliseRange({ from: r[0], to: r[1] }) : null;
  if (range) out.range = range;
  const cmp = q.get("cmp");
  if (isIsoDate(cmp)) {
    out.compare = { date: cmp, layer: null };
    const cl = q.get("cl");
    if (cl && /^[a-z0-9-]+$/.test(cl)) out.compare.layer = cl;
  }
  return out;
}

/** Write the time state into a URLSearchParams. `dated` says whether a dated layer is on. */
export function writeTimeParams(q, { date, step, range, compare } = {}, { dated = false } = {}) {
  if (isIsoDate(date) && (dated || range || compare)) q.set("d", date);
  if (step && step !== DEFAULT_STEP && STEPS.includes(step)) q.set("ts", step);
  const r = normaliseRange(range);
  if (r) q.set("r", `${r.from}..${r.to}`);
  if (compare && isIsoDate(compare.date)) {
    q.set("cmp", compare.date);
    if (compare.layer) q.set("cl", compare.layer);
  }
  return q;
}

// ── the GIF frame ───────────────────────────────────────────────────────────

/** The GIF frame size for a map canvas: no wider than `maxWidth`, never upscaled, even numbers. */
export function gifSize(width, height, maxWidth = 640) {
  const w = Math.max(2, Math.round(width || 0)), h = Math.max(2, Math.round(height || 0));
  const scale = Math.min(1, maxWidth / w);
  const even = (n) => Math.max(2, Math.floor(n * scale / 2) * 2);
  return { width: even(w), height: even(h) };
}

// ── annual maxima on the hydrograph ─────────────────────────────────────────

/**
 * Where to draw a record's annual maxima: on the day of each peak when the analysis gave the days
 * (aquascope.explore.analyze_series, "annual_max.date"), so clicking one moves the map to that flood;
 * at 1 July of each year otherwise (an older cached result), and then the markers do not set the map date.
 */
export function annualMaxPoints(am) {
  const years = (am && Array.isArray(am.year)) ? am.year : [];
  const dates = (am && Array.isArray(am.date)) ? am.date : [];
  const onDay = years.length > 0 && dates.length === years.length && dates.every((d) => isIsoDate(d));
  return { x: onDay ? dates.slice() : years.map((y) => `${y}-07-01`), onDay };
}
