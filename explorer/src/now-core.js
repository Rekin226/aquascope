// Now and next (#517), the pure part: no DOM, no map, so node can test it.
// Every number (the percentile, the forecast, the correction, its skill) comes
// from Python (aquascope.nownext); this module only decides how it is drawn:
// the class colours, which threshold lines to show, the forecast traces and
// the time-bar marker.

// The five classes of the USGS National Water Dashboard and WMO HydroSOS, in
// the same order and with the same ids as aquascope.nownext.STATUS_CLASSES.
// Brown to teal (colour-blind safe), with a neutral slate for normal.
export const STATUS_CLASSES = [
  { id: "much_below", label: "much below normal", color: "#a6611a" },
  { id: "below", label: "below normal", color: "#dfc27d" },
  { id: "normal", label: "normal", color: "#7b8794" },
  { id: "above", label: "above normal", color: "#80cdc1" },
  { id: "much_above", label: "much above normal", color: "#018571" },
];
export const NO_STATUS_COLOR = "#d5dbe0";
export const FORECAST_CREDIT = "GEOGLOWS v2 (GEOGloWS ECMWF Streamflow Service) and GloFAS v4 via Open-Meteo, both CC BY 4.0";

export const statusClass = (id) => STATUS_CLASSES.find((c) => c.id === id) || null;

// The colour a gauge gets under "Today vs normal". Before any snapshot has
// loaded (statusMap null) a gauge keeps its agency colour; once one has, a
// gauge without a fresh status takes `missing`: light grey by default, its
// agency colour on the map's default view (#544), where most gauges have none.
export function nowColor(statusMap, key, fallback, missing = NO_STATUS_COLOR) {
  if (!statusMap) return fallback;
  const s = statusMap.get(key);
  const c = s && statusClass(s.cls);
  return c ? c.color : missing;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function whenText(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?/.exec(String(iso || ""));
  if (!m) return "";
  const day = `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}`;
  return m[4] ? `${day}, ${m[4]}:${m[5]} UTC` : day;
}

// The legend's sentence: when the snapshot was made and which sources it covers,
// or, before the first one exists, what will happen.
export function snapshotLine(meta, labelOf = (s) => s) {
  if (!meta || meta.missing) {
    return "No status snapshot yet: the daily forecast job makes one. Until then the gauges keep their agency colours.";
  }
  const sources = (meta.sources || []).map(labelOf);
  const list = sources.length ? sources.join(", ") : "no source";
  return `Daily snapshot of ${whenText(meta.made || meta.date)}: ${meta.n || 0} gauges with a fresh record, from ${list}.`;
}

// How far above everything else on the plot (the ensemble's top included) the
// lowest line may sit and still be drawn. Past this the y axis stretches to the
// line and presses the forecast flat against zero (the Potomac in a dry
// October: a 62 m³/s forecast, an ensemble top of 374, under a 2,876 m³/s
// 2-year flow); the sentence above the plot still names the threshold.
export const FAR_THRESHOLD = 4;

// The return-period lines worth drawing: those within reach of the forecast
// (up to 1.5 times its highest value) and the lowest one, so the nearest
// threshold is on the plot, unless it is more than FAR_THRESHOLD times the
// forecast; at most `max` of them.
export function thresholdsToShow(thr, ymax, max = 3) {
  if (!thr || !Array.isArray(thr.q) || !Array.isArray(thr.return_periods)) return [];
  const all = thr.return_periods.map((T, i) => ({ T, q: thr.q[i] }))
    .filter((t) => Number.isFinite(t.q)).sort((a, b) => a.q - b.q);
  if (!all.length) return [];
  const known = Number.isFinite(ymax) && ymax > 0;
  const top = known ? ymax * 1.5 : -Infinity;
  const far = known ? ymax * FAR_THRESHOLD : Infinity;
  const near = all.filter((t, i) => (i === 0 && t.q <= far) || t.q <= top);
  return near.slice(0, max);
}

// The days a forecast plot covers: the recent observed days and the forecast days.
export function plotRange(fc, recent) {
  const days = [];
  for (const part of [fc && fc.geoglows, fc && fc.glofas, recent]) {
    const d = part && (part.date || part.t);
    if (Array.isArray(d)) days.push(...d.filter(Boolean));
  }
  if (!days.length) return null;
  days.sort();
  return [days[0], days[days.length - 1]];
}

// A vertical line at the map date when it falls inside the plot, else nothing.
export function markerShapes(date, range, color = "#33475a") {
  if (!date || !range || date < range[0] || date > range[1]) return [];
  return [{ type: "line", xref: "x", yref: "paper", x0: date, x1: date, y0: 0, y1: 1,
    line: { color, width: 1.5, dash: "dot" } }];
}

const BLUE = "#1e88e5";
const GLOFAS = "#ef6c00";
const THRESHOLD = "#c62828";

function band(dates, lo, hi, fillcolor, name) {
  return [
    { x: dates, y: hi, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: dates, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor, name, hoverinfo: "skip",
      showlegend: false },
  ];
}

const values = (part, k) => (part && Array.isArray(part[k]) ? part[k] : null);

// The forecast as Plotly traces. With a correction, the GEOGLOWS bands and mean
// are the corrected ones and the raw mean is a thin dashed line; the thresholds
// are then the gauge's own, else the reach's (both chosen in Python).
export function forecastTraces(fc, { recent = null, ink = "#33475a" } = {}) {
  const traces = [];
  if (!fc) return traces;
  const g = fc.geoglows && !fc.geoglows.error ? fc.geoglows : null;
  const corr = fc.correction && fc.correction.forecast ? fc.correction.forecast : null;
  const main = corr || g;
  if (main && Array.isArray(main.date) && main.date.length) {
    const d = main.date;
    if (values(main, "min") && values(main, "max")) traces.push(...band(d, main.min, main.max, "rgba(30,136,229,0.10)", "ensemble range"));
    if (values(main, "p25") && values(main, "p75")) traces.push(...band(d, main.p25, main.p75, "rgba(30,136,229,0.25)", "middle half of the ensemble"));
    if (values(main, "mean")) {
      traces.push({ x: d, y: main.mean, type: "scatter", mode: "lines", name: corr ? "GEOGLOWS, corrected to the gauge" : "GEOGLOWS ensemble mean",
        line: { color: BLUE, width: 2.2 }, hovertemplate: "%{x}<br>%{y:.3~f} m³/s<extra></extra>" });
    }
    if (corr && g && values(g, "mean")) {
      traces.push({ x: g.date, y: g.mean, type: "scatter", mode: "lines", name: "GEOGLOWS raw",
        line: { color: BLUE, width: 1, dash: "dash" }, hovertemplate: "%{x}<br>raw %{y:.3~f} m³/s<extra></extra>" });
    }
  }
  const gl = fc.glofas && !fc.glofas.error ? fc.glofas : null;
  if (gl && values(gl, "mean")) {
    traces.push({ x: gl.date, y: gl.mean, type: "scatter", mode: "lines", name: "GloFAS mean",
      line: { color: GLOFAS, width: 1.6, dash: "dot" }, hovertemplate: "%{x}<br>GloFAS %{y:.3~f} m³/s<extra></extra>" });
  }
  if (recent && Array.isArray(recent.t) && recent.t.length) {
    traces.push({ x: recent.t, y: recent.v, type: "scatter", mode: "lines", name: "observed",
      line: { color: ink, width: 1.8 }, hovertemplate: "%{x}<br>observed %{y:.3~f} m³/s<extra></extra>" });
  }
  const range = plotRange(fc, recent);
  const ys = traces.flatMap((t) => (t.y || []).filter(Number.isFinite));
  const ymax = ys.length ? Math.max(...ys) : null;
  const thr = corr ? fc.gauge_thresholds : fc.thresholds;
  if (range) {
    for (const t of thresholdsToShow(thr, ymax)) {
      traces.push({ x: range, y: [t.q, t.q], type: "scatter", mode: "lines+text", showlegend: false,
        text: ["", `${t.T}-yr`], textposition: "top left", textfont: { color: THRESHOLD, size: 10 },
        line: { color: THRESHOLD, width: 1, dash: "dash" }, meta: { mapDate: false },
        hovertemplate: `${t.T}-year flow: %{y:.3~f} m³/s<extra></extra>` });
    }
  }
  return traces;
}

// The few sentences under the plot that say how far to trust the correction:
// its KGE, then bias and the days above the 2-year flow, and a warning when the
// reach's mean flow is far from the gauge's (all worded in Python).
export function skillText(fc) {
  const c = fc && fc.correction;
  if (!c) return "";
  const reach = fc.reach_check && fc.reach_check.note ? ` ${fc.reach_check.note}` : "";
  if (c.error) return `Not corrected to the gauge: ${c.error}`;
  const s = c.skill || {};
  const bits = [c.skill_line, c.skill_detail, s.note].filter(Boolean).join(" ");
  return bits + reach;
}
