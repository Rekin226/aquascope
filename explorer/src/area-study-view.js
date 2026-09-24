// Study this area: the pure presentation helpers (colours, legend, sorting,
// progress wording). No DOM and no map, so they run under node in the tests.
// The numbers themselves come from aquascope.area_study in the worker.

export const TREND_COLORS = { up: "#c62828", down: "#1565c0", none: "#78909c", untested: "#ffffff" };
const RAMP = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"];
export const NO_VALUE_COLOR = "#ffffff";

export const COLOR_BY = [
  { id: "trend", label: "Trend in annual maxima" },
  { id: "q100_per_km2", label: "Q100 per km²" },
  { id: "q100", label: "Q100" },
  { id: "record_years", label: "Record length" },
];

// The table's columns: [field, header, kind]. Kind drives formatting and sorting.
export const TABLE = [
  ["name", "Gauge", "text"],
  ["status", "Status", "text"],
  ["record_years", "Years", "num"],
  ["mean", "Mean", "num"],
  ["q100", "Q100", "num"],
  ["q100_per_km2", "Q100/km²", "num"],
  ["regional_q100", "Regional Q100", "num"],
  ["trend", "Trend", "text"],
  ["trend_p", "p", "p"],
  ["fdr_significant", "After FDR", "bool"],
];

const isNum = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

// Quantile class breaks over the values present (at most four breaks, five classes).
export function classBreaks(values) {
  const xs = values.filter(isNum).map(Number).sort((a, b) => a - b);
  if (xs.length < 2) return [];
  const out = [];
  for (const q of [0.2, 0.4, 0.6, 0.8]) {
    const v = xs[Math.min(xs.length - 1, Math.floor(q * xs.length))];
    if (!out.length || v > out[out.length - 1]) out.push(v);
  }
  return out;
}

// A MapLibre paint expression colouring the study's pins by `mode`.
export function pinColor(mode, features) {
  if (mode === "trend") {
    return ["match", ["coalesce", ["get", "trend"], "untested"],
      "up", TREND_COLORS.up, "down", TREND_COLORS.down, "none", TREND_COLORS.none, TREND_COLORS.untested];
  }
  const breaks = classBreaks(features.map((f) => f.properties[mode]));
  if (!breaks.length) return ["case", ["==", ["typeof", ["get", mode]], "number"], RAMP[2], NO_VALUE_COLOR];
  const step = ["step", ["to-number", ["get", mode]], RAMP[0]];
  breaks.forEach((b, i) => step.push(b, RAMP[Math.min(i + 1, RAMP.length - 1)]));
  return ["case", ["==", ["typeof", ["get", mode]], "number"], step, NO_VALUE_COLOR];
}

// Legend rows [{color, label}] matching pinColor.
export function legend(mode, features, fmt = (x) => String(x)) {
  if (mode === "trend") {
    const n = (t) => features.filter((f) => (f.properties.trend || "untested") === t).length;
    return [
      { color: TREND_COLORS.up, label: `Up (${n("up")})` },
      { color: TREND_COLORS.down, label: `Down (${n("down")})` },
      { color: TREND_COLORS.none, label: `No trend (${n("none")})` },
      { color: TREND_COLORS.untested, label: `Not tested (${n("untested")})` },
    ];
  }
  const breaks = classBreaks(features.map((f) => f.properties[mode]));
  const missing = features.filter((f) => !isNum(f.properties[mode])).length;
  const rows = [];
  if (breaks.length) {
    rows.push({ color: RAMP[0], label: `below ${fmt(breaks[0])}` });
    for (let i = 0; i < breaks.length; i++) {
      const hi = breaks[i + 1];
      rows.push({ color: RAMP[Math.min(i + 1, RAMP.length - 1)], label: hi === undefined ? `${fmt(breaks[i])} and up` : `${fmt(breaks[i])} to ${fmt(hi)}` });
    }
  } else if (features.length > missing) {
    rows.push({ color: RAMP[2], label: "value" });
  }
  if (missing) rows.push({ color: NO_VALUE_COLOR, label: `no value (${missing})` });
  return rows;
}

// Sort a copy of the rows by one field; empty values always last.
export function sortRows(rows, field, dir = 1) {
  const kind = (TABLE.find((c) => c[0] === field) || [])[2] || "text";
  const key = (r) => {
    const v = r[field];
    if (v === null || v === undefined || v === "") return null;
    if (kind === "num" || kind === "p") return Number(v);
    if (kind === "bool") return v ? 1 : 0;
    return String(v).toLowerCase();
  };
  return [...rows].sort((a, b) => {
    const x = key(a), y = key(b);
    if (x === null && y === null) return 0;
    if (x === null) return 1;
    if (y === null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });
}

// One line for the progress bar from an engine event.
export function progressText(e) {
  if (!e) return "";
  if (e.phase === "inventory") return `${e.total} gauges to study`;
  if (e.phase === "fetch") {
    return e.done >= e.total ? "Records read" : `Reading records ${e.done + 1} of ${e.total}${e.site ? ` (${e.site})` : ""}`;
  }
  if (e.phase === "analyse") return `Analysing ${e.done + 1} of ${e.total}`;
  if (e.phase === "regional") return e.done >= e.total ? "Done" : "Regional methods";
  return "";
}

// The share of the whole run an event stands for (fetching is most of it).
export function progressShare(e) {
  if (!e || !e.total) return 0;
  const f = Math.min(1, e.done / e.total);
  if (e.phase === "inventory") return 0.02;
  if (e.phase === "fetch") return 0.02 + 0.78 * f;
  if (e.phase === "analyse") return 0.8 + 0.15 * f;
  if (e.phase === "regional") return 0.95 + 0.05 * f;
  return 0;
}
