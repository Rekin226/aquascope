// Plotly defaults and the export helpers. Every figure keeps its PNG button
// (the mode bar used to be off entirely, so no chart on the page could be
// saved), and every table on the page can be downloaded as CSV.
//
// The layout is built from the stylesheet's own tokens rather than written out
// here, so a chart cannot drift from the page around it. That also fixes dark
// mode: Plotly's defaults are #444 text on #EBF0F8 gridlines, which on a dark
// card is grey-on-grey text under glaring white rules.

import { downloadBlob, setTime, toCsv } from "./core.js?v=__BUILD__";
import { chartDate, todayIso } from "./timeline.js?v=__BUILD__";

const token = (name, fallback) => {
  if (typeof getComputedStyle !== "function") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
};

const prefersDark = () =>
  Boolean(globalThis.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);

// ── colour helpers ──────────────────────────────────────────────────────────
// Enough colour maths for one job: keep a series line legible on whichever
// surface it lands on. Not a colour library.

function parseHex(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const toHex = (rgb) => `#${rgb.map((c) => Math.round(Math.max(0, Math.min(255, c))).toString(16).padStart(2, "0")).join("")}`;

function luminance(rgb) {
  const f = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
}

function contrast(a, b) {
  const la = luminance(a), lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

const mix = (a, b, t) => a.map((c, i) => c + (b[i] - c) * t);

/**
 * A series colour that clears the 3:1 mark-contrast floor on the current card.
 *
 * The six agency colours are the map's identity system: the dot, the rail
 * swatch, the search hit and the chart line are deliberately the same colour,
 * so they cannot be re-picked per theme without the chart and the map
 * disagreeing. They were chosen against a white page, though, and on the dark
 * card the darkest of them (OPW's purple) is 1.96:1. Rather than keep a second
 * palette that could drift from the first, lift the colour towards the page's
 * own ink until it clears the floor. Light mode is left exactly as it is.
 */
export function seriesColor(hex) {
  const rgb = parseHex(hex);
  if (!rgb || !prefersDark()) return hex;
  const surface = parseHex(token("--bg", "#0e151c")) || [14, 21, 28];
  const ink = parseHex(token("--ink", "#e7eef5")) || [231, 238, 245];
  if (contrast(rgb, surface) >= 3) return hex;
  for (let t = 0.1; t <= 0.9; t += 0.1) {
    const lifted = mix(rgb, ink, t);
    if (contrast(lifted, surface) >= 3) return toHex(lifted);
  }
  return toHex(ink);
}

/**
 * The colour for a mark that emphasises a series rather than naming a second
 * one: the annual maxima on a hydrograph, a modelled line beside an observed
 * record. It has to sit beside any of the six agency colours, and a fixed hue
 * cannot: red markers on the UK's green line collapse to ΔE 5.5 under
 * protanopia, and on a French gauge the old red simulated line (#e53935) and
 * the red observed line (#c62828) are ΔE 7.1 apart with *normal* vision. Ink
 * separates by lightness instead, which every hue clears in both modes.
 */
export const emphasisColor = () => token("--ink-2", prefersDark() ? "#c2cfdb" : "#33475a");

// The card the plot is drawn on, for rings that punch a mark out of a line.
export const surfaceColor = () => token("--bg", prefersDark() ? "#0e151c" : "#ffffff");

// ── layout ──────────────────────────────────────────────────────────────────

export function plotLayout() {
  const muted = token("--muted", "#64798a");
  const ink = token("--ink", "#0f1c26");
  const line = token("--line", "#e3eaf1");
  const surface = token("--surface-solid", "#ffffff");
  const axis = {
    gridcolor: line,
    zerolinecolor: line,
    linecolor: line,
    tickfont: { color: muted },
    title: { font: { color: muted } },
    automargin: true,
  };
  return {
    margin: { l: 48, r: 12, t: 8, b: 36 },
    height: 240,
    font: { family: "system-ui, -apple-system, sans-serif", size: 11, color: muted },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { ...axis },
    yaxis: { ...axis },
    legend: { font: { color: ink }, bgcolor: "rgba(0,0,0,0)", borderwidth: 0 },
    hoverlabel: { bgcolor: surface, bordercolor: line, font: { color: ink } },
    colorway: [token("--blue", "#0b6bb8"), muted],
  };
}

// Kept as a getter so nothing captures one theme's values at import time.
export const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: "hover",
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"],
  toImageButtonOptions: { format: "png", scale: 2, filename: "aquascope-figure" },
};

// A call site that passes `xaxis: { type: "log" }` means "and also log", not
// "instead of the theme", so the axis objects merge one level deep.
const NESTED = ["xaxis", "yaxis", "xaxis2", "yaxis2", "legend", "margin", "hoverlabel", "font"];

function mergeLayout(base, extra) {
  const out = { ...base, ...extra };
  for (const key of NESTED) {
    if (extra && extra[key] && base[key]) out[key] = { ...base[key], ...extra[key] };
  }
  return out;
}

// ── peak to map (#522) ──────────────────────────────────────────────────────
// Clicking a day on any dated chart (a hydrograph, a model run, a comparison)
// moves the map date there, so the rain, soil moisture and satellite layers show
// that day. A trace whose x is not the real day (annual maxima are drawn at 1
// July) opts out with meta: { mapDate: false }.

const mapDateTrace = (t) => Boolean(t && !(t.meta && t.meta.mapDate === false) &&
  Array.isArray(t.x) && t.x.length && chartDate(t.x[0]));

const HINT = '<br><span style="font-size:10px;opacity:.75">click: map to this day</span>';

/** The day a Plotly click event points at, or null. Exported for the tests' sake. */
export function clickedDay(points) {
  for (const pt of points || []) {
    const t = pt && (pt.data || pt.fullData);
    if (t && t.meta && t.meta.mapDate === false) continue;
    const day = chartDate(pt && pt.x);
    if (day) return day;
  }
  return null;
}

function wireMapDate(id, traces) {
  const el = typeof document !== "undefined" && document.getElementById(id);
  if (!el || el.__aqMapDate || !(traces || []).some(mapDateTrace) || typeof el.on !== "function") return;
  el.__aqMapDate = true;
  el.on("plotly_click", (ev) => {
    const day = clickedDay(ev && ev.points);
    if (day && day <= todayIso()) setTime({ date: day }, { source: "chart" });
  });
}

// Series colours are lifted for the current theme here rather than at each call
// site, so a new chart cannot forget to do it.
function themeTraces(traces) {
  return (traces || []).map((t) => {
    if (!t || typeof t !== "object") return t;
    const next = { ...t };
    if (mapDateTrace(t) && typeof next.hovertemplate === "string" && next.hovertemplate.includes("<extra>")) {
      next.hovertemplate = next.hovertemplate.replace("<extra>", `${HINT}<extra>`);
    }
    if (next.line && typeof next.line.color === "string") {
      next.line = { ...next.line, color: seriesColor(next.line.color) };
    }
    if (next.marker && typeof next.marker.color === "string") {
      next.marker = { ...next.marker, color: seriesColor(next.marker.color) };
    }
    return next;
  });
}

// Which figures are on the page, so a theme change can re-draw them.
const drawn = new Map();

// A figure that can name its own PNG download.
export function plot(id, traces, layout, filename) {
  const config = filename
    ? { ...PLOT_CONFIG, toImageButtonOptions: { ...PLOT_CONFIG.toImageButtonOptions, filename } }
    : PLOT_CONFIG;
  drawn.set(id, { traces, layout, filename });
  const out = Plotly.react(id, themeTraces(traces), mergeLayout(plotLayout(), layout), config);
  // Plotly adds .on() to the element as it draws; wire the click once that is done.
  Promise.resolve(out).then(() => wireMapDate(id, traces)).catch(() => {});
  return out;
}

/** Re-draw every figure on the page against the current theme. */
export function retheme() {
  for (const [id, spec] of drawn) {
    if (!document.getElementById(id)) { drawn.delete(id); continue; }
    try { plot(id, spec.traces, spec.layout, spec.filename); } catch { drawn.delete(id); }
  }
}

// The page follows the system theme, and a chart drawn before the reader
// switched should not stay in the old one.
if (globalThis.matchMedia) {
  const q = matchMedia("(prefers-color-scheme: dark)");
  const onChange = () => retheme();
  if (q.addEventListener) q.addEventListener("change", onChange);
  else if (q.addListener) q.addListener(onChange);
}

// ── tables ──────────────────────────────────────────────────────────────────

// Read a rendered <table> back out as CSV so the download always matches what
// the page shows (superscripts and CI columns included).
export function tableToCsv(table) {
  const rows = [...table.querySelectorAll("tr")].map((tr) =>
    [...tr.querySelectorAll("th,td")].map((td) => td.textContent.replace(/\s+/g, " ").trim()));
  return rows.length ? toCsv(rows[0], rows.slice(1)) : "";
}

export function downloadTable(table, filename) {
  const csv = tableToCsv(table);
  if (csv) downloadBlob(filename, csv, "text/csv");
}

// Adds a small "CSV" button next to a table, once.
export function addTableDownload(container, table, filename) {
  if (!container || !table || container.querySelector(".table-dl")) return;
  const b = document.createElement("button");
  b.className = "btn tiny table-dl";
  b.type = "button";
  b.textContent = "CSV";
  b.title = "Download this table";
  b.addEventListener("click", () => downloadTable(table, filename));
  container.appendChild(b);
}
