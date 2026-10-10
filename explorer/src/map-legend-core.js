// "On the map" (#543 design pass), the pure part: no DOM, so node can test it.
//
// One legend for every layer on the globe, instead of a card per layer. Each
// layer module registers a row (map-legend.js); this file decides the order of
// the rows, how a row reads when it is on, off or has nothing to show, and how
// the folded legend names itself.

// Top to bottom in the same order as the layers on the map (#543): what is
// drawn on top is listed first.
export const ROW_ORDER = ["river-lit", "gauges", "forecast-points", "floods-ahead", "flood-depth", "rivers", "floods-past", "status"];

export function rowRank(id) {
  const i = ROW_ORDER.indexOf(id);
  return i < 0 ? ROW_ORDER.length : i;
}

/** The rows in map order; rows the order does not know go last, in the order they came. */
export function sortRows(rows) {
  return [...(rows || [])].map((r, i) => [r, i])
    .sort((a, b) => rowRank(a[0].id) - rowRank(b[0].id) || a[1] - b[1]).map(([r]) => r);
}

/**
 * How a row reads: "on" (drawn), "off" (the reader turned it off) or "empty"
 * (on, but its source has nothing to draw: one muted line, no key to open).
 */
export function rowState({ on, empty } = {}) {
  if (!on) return "off";
  return empty ? "empty" : "on";
}

/** The layers being drawn, for the folded chip: "On the map · 4". */
export function chipLabel(states) {
  const n = (states || []).filter((s) => s === "on").length;
  return n ? `On the map · ${n}` : "On the map";
}

/** The legend starts folded to its chip where the map is small (a phone, 520 px and under). */
export function startsFolded(width) {
  return Number(width) > 0 && Number(width) <= 520;
}

/** Whether a row starts open: the river status's colours are the first thing to read, on a wide screen. */
export function startsOpen(id, width) {
  return id === "status" && !startsFolded(width);
}
