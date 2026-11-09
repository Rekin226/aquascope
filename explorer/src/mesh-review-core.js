// Review my model results (#564), the pure part: no DOM, no map, so node can test it
// (explorer/tests/mesh-review-core.test.mjs).
//
// The checks and every number in them are aquascope.mesh_review's, run in the mesh worker beside the dropped
// file. This module turns a finding into the pin map-actions.js draws, picks the cells to flag from the pins
// still on the map (an undone pin takes its flags with it), and says in a few words who worded the review.

// The checks as aquascope.mesh_review.CHECKS labels them (a test keeps the two lists equal).
export const CHECK_LABELS = {
  negative: "Negative depths",
  mass_balance: "Volume balance",
  isolated: "Isolated wet cells",
  flicker: "Wet and dry flicker",
  tiny: "Very shallow water",
  floods_radar: "Floods seen by radar",
  floods_news: "Floods in the news",
  gauges: "Gauges in the mesh",
  deepest: "Deepest water",
};

// The three severities, most severe first: the pin, the cell flags and the card use the same colours.
export const SEVERITY = {
  problem: { colour: "#c62828", words: "Problem" },
  warning: { colour: "#d97706", words: "Worth a look" },
  // Slate, not blue: the model's depth is painted in blues.
  info: { colour: "#3b4a5c", words: "For information" },
};
const RANK = { problem: 0, warning: 1, info: 2 };

// How many cells one pin may flag on the map (the package caps a finding at 5,000).
export const MAX_FLAG_CELLS = 5000;

/** The pin map-actions.js's addPin takes, from one finding: its reason, its facts, its source and its cells. */
export function reviewPin(f, { mesh = "" } = {}) {
  const facts = (f.facts || []).slice(0, 5).map((x) => ({
    label: x.label, value: x.shown, ...(x.unit ? { unit: x.unit } : {}), ...(x.when ? { when: x.when } : {}),
  }));
  return {
    lat: f.lat, lon: f.lon, title: f.title, text: f.reason, facts, source: f.source, rank: f.rank,
    kind: "review", severity: SEVERITY[f.severity] ? f.severity : "info", check: f.label || CHECK_LABELS[f.check] || "",
    cells: (f.cells || []).slice(0, MAX_FLAG_CELLS), mesh,
  };
}

/**
 * The cells to flag on the map: for each cell of the mesh `mesh`, the most severe of the live review pins that
 * flag it. `pins` are map-actions' pins (log.pins()), so an undone pin flags nothing. Returns Map(cell -> severity).
 */
export function flaggedCells(pins, mesh) {
  const out = new Map();
  for (const p of pins || []) {
    if (!p || p.kind !== "review" || p.mesh !== mesh) continue;
    for (const c of p.cells || []) {
      const was = out.get(c);
      if (was === undefined || RANK[p.severity] < RANK[was]) out.set(c, p.severity);
    }
  }
  return out;
}

/**
 * Who picked the review's pins, for the line under the bar: [who, how]. The words are always the package's. `asked`
 * names a model that was asked but whose choice is on no pin (refused, or no answer): said, not "no model" (#590).
 */
export function reviewWho(by, model = "", asked = "") {
  if (by === "device") return [model || "On-device model", "picked on this device; checks and words by code"];
  if (by === "key") return [model || "Your model", "picked with your key; checks and words by code"];
  if (asked) return ["Rules", `${asked} asked, its choice not used; checks and words by code`];
  return ["Rules", "checks by code, no model"];
}

/**
 * The line itself: the file reviewed (#590), the package's summary (its counts are the rules' counts), and how
 * many pins. `picked` says a model chose and ordered them; `leftOut` is how many findings it did not pin.
 */
export function reviewLine(res, n, name = "", { picked = false, leftOut = 0 } = {}) {
  const summary = (res && res.summary) || "";
  const head = name ? `${name}: ` : "";
  if (!n) return `${head}${summary} No pin to drop.`.trim();
  const order = picked ? "in the model's order" : "most severe first";
  const left = leftOut ? ` ${leftOut} finding${leftOut === 1 ? "" : "s"} not pinned (the model left ${leftOut === 1 ? "it" : "them"} out; the checks list has every one).` : "";
  return `${head}${summary} ${n} pin${n === 1 ? "" : "s"}, ${order}.${left}`.trim();
}

const overlaps = (a, b) => Boolean(a && b) && a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1];

/**
 * Which model result the review is about (#590): the one on screen, and among several the one shown last. The
 * review reads HEC-RAS 2D and UGRID meshes; a MODFLOW model on screen is not one it covers, and an older mesh
 * off screen is not reviewed in its place. `mesh` and `modflow` are { name, bbox, openedAt, visible } or null,
 * `view` the map's [w, s, e, n]. Returns { review: true, name } or { review: false, line }.
 */
export function reviewTarget({ mesh = null, modflow = null, view = null } = {}) {
  const seen = (m) => Boolean(m) && m.visible !== false && (!view || overlaps(m.bbox, view));
  const meshSeen = seen(mesh), flowSeen = seen(modflow);
  const newer = (a, b) => (Number(a && a.openedAt) || 0) >= (Number(b && b.openedAt) || 0);
  if (mesh && (meshSeen && (!flowSeen || newer(mesh, modflow)))) return { review: true, name: mesh.name };
  if (modflow && flowSeen) {
    const elsewhere = mesh ? ` ${mesh.name} is loaded but not in view: move to it, or use “Review these results” on its legend row.` : "";
    return { review: false, line: `${modflow.name} (MODFLOW heads) is the model on screen: the review reads HEC-RAS 2D and UGRID meshes, not MODFLOW heads.${elsewhere}` };
  }
  if (mesh) return { review: false, line: `${mesh.name} is not in view: move to it, or use “Review these results” on its legend row.` };
  return { review: false, line: "No model result on the map to review. Drop a HEC-RAS plan or a UGRID NetCDF file on the map first." };
}

/** One short row per check, for the legend: its label, what happened, and the package's line. */
export function checkRows(checks) {
  const mark = { flagged: "!", ok: "✓", skipped: "–" };
  return (checks || []).map((c) => ({
    check: c.check, label: c.label || CHECK_LABELS[c.check] || c.check, status: c.status,
    mark: mark[c.status] || "·", line: c.line || "",
  }));
}

/**
 * The catalogue's gauges inside a box ([w, s, e, n]), as the review's gauge check takes them: at most `max`,
 * nearest the box's centre first (the mesh worker then keeps those standing in a cell).
 */
export function gaugesInBox(stations, bbox, max = 400) {
  const [w, s, e, n] = bbox;
  const cx = (w + e) / 2, cy = (s + n) / 2;
  const rows = [];
  for (const r of stations || []) {
    if (!Number.isFinite(r.lat) || !Number.isFinite(r.lon)) continue;
    if (r.lon < w || r.lon > e || r.lat < s || r.lat > n) continue;
    rows.push({ source: r.source, station_id: r.station_id, name: r.name || r.station_id, lat: r.lat, lon: r.lon });
  }
  rows.sort((a, b) => Math.hypot(a.lon - cx, a.lat - cy) - Math.hypot(b.lon - cx, b.lat - cy));
  return rows.slice(0, max);
}

/** The model's run as months and days for the gauge records: { start, end } (ISO), or null without times. */
export function runSpan(times) {
  if (!times || !times.length) return null;
  return { start: String(times[0]).slice(0, 10), end: String(times[times.length - 1]).slice(0, 10) };
}
