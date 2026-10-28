// Levees and breaches on Flood it's grid (#663), the pure part: the drawn lines as the package's edits, their map
// features, the words for a run with them, the crest control's profile drawing and the before-and-after rows. No
// DOM, no map: node imports this (explorer/tests/floodsim-sketch-core.test.mjs). The numbers (which cells a line
// raises, the suggested crest, what gets flooded) come from aquascope.floodsim in the worker; this only draws.

const r2 = (v) => Math.round(v * 100) / 100;
const fmt1 = (v) => (Math.abs(v) >= 100 ? String(Math.round(v)) : (Math.round(v * 10) / 10).toFixed(1));

/** A drawn line, checked: { id, kind: "levee" | "breach", line: [[lon, lat], ...], crest?, invert? }. */
export function checkSketch(s) {
  if (!s || !["levee", "breach"].includes(s.kind)) throw new Error("a sketch is a levee or a breach");
  const line = (s.line || []).map((p) => [Number(p[0]), Number(p[1])]);
  if (line.length < 2 || line.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(y) > 85 || Math.abs(x) > 180)) {
    throw new Error("a line needs two or more [lon, lat] points");
  }
  const out = { id: Number(s.id), kind: s.kind, line: line.map(([x, y]) => [Math.round(x * 1e6) / 1e6, Math.round(y * 1e6) / 1e6]) };
  if (s.kind === "levee") {
    if (!Number.isFinite(Number(s.crest))) throw new Error("a levee needs a crest (m above mean sea level)");
    out.crest = r2(Number(s.crest));
  } else if (s.invert !== undefined && s.invert !== null && s.invert !== "") {
    out.invert = r2(Number(s.invert));
  }
  return out;
}

/** The sketches as aquascope.floodsim.edit_grid's edits: levees first, then breaches, each in the order drawn. */
export function sketchEdits(sketches) {
  const list = [...(sketches || [])].sort((a, b) => a.id - b.id);
  const edit = (s) => ({ id: s.id, kind: s.kind, line: s.line, ...(s.kind === "levee" ? { crest: s.crest } : { invert: s.invert ?? null }) });
  return [...list.filter((s) => s.kind === "levee").map(edit), ...list.filter((s) => s.kind === "breach").map(edit)];
}

/** A key that is the same for the same edits (a run with them can be told from one without). */
export const editsKey = (edits) => JSON.stringify((edits || []).map((e) => [e.kind, e.line, e.crest ?? null, e.invert ?? null]));

/** The words a run's label gets for its edits: "with a levee at 14.2 m", "with 2 levees and a breach". */
export function sketchWords(edits) {
  const lv = (edits || []).filter((e) => e.kind === "levee"), br = (edits || []).filter((e) => e.kind === "breach");
  if (!lv.length && !br.length) return "";
  const parts = [];
  if (lv.length === 1) parts.push(`a levee at ${fmt1(lv[0].crest)} m`);
  else if (lv.length) parts.push(`${lv.length} levees`);
  if (br.length) parts.push(br.length === 1 ? "a breach" : `${br.length} breaches`);
  return `with ${parts.join(" and ")}`;
}

/** The lines on the map: each sketch (its kind, whether it is selected) and the line being drawn. */
export function sketchFeatures(sketches, { selected = null, draft = null } = {}) {
  const feats = (sketches || []).map((s) => ({
    type: "Feature", id: s.id,
    properties: { id: s.id, kind: s.kind, selected: s.id === selected ? 1 : 0,
      label: s.kind === "levee" ? `${fmt1(s.crest)} m` : "breach" },
    geometry: { type: "LineString", coordinates: s.line },
  }));
  if (draft && draft.length) {
    feats.push({ type: "Feature", properties: { id: -1, kind: "draft", selected: 1, label: "" },
      geometry: draft.length > 1 ? { type: "LineString", coordinates: draft } : { type: "Point", coordinates: draft[0] } });
  }
  return { type: "FeatureCollection", features: feats };
}

/**
 * The crest control's drawing: the natural ground (filled), the flood level where the line is wet, the terrain now
 * (a levee already there) and the crest, along the line, as SVG path strings in a w x h box. `profile` is
 * aquascope.floodsim.levee_profile's answer; `crest` the level to draw (null: none).
 */
export function profilePaths(profile, crest = null, { w = 260, h = 84, pad = 4 } = {}) {
  const d = profile.distance_m, g = profile.ground_m, t = profile.terrain_m || g, wat = profile.water_m || [];
  const vals = [...g, ...t, ...wat.filter((v) => v !== null)];
  if (crest !== null && Number.isFinite(crest)) vals.push(crest);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1) { const mid = (hi + lo) / 2; lo = mid - 0.5; hi = mid + 0.5; }
  const span = hi - lo;
  lo -= span * 0.08; hi += span * 0.12;
  const L = d.at(-1) || 1;
  const x = (v) => r2(pad + ((w - 2 * pad) * v) / L);
  const y = (v) => r2(h - pad - ((h - 2 * pad) * (v - lo)) / (hi - lo));
  const line = (vs) => vs.map((v, i) => `${i ? "L" : "M"}${x(d[i])} ${y(v)}`).join("");
  const ground = `${line(g)}L${x(L)} ${h}L${x(0)} ${h}Z`;
  const now = t.some((v, i) => Math.abs(v - g[i]) > 0.005) ? line(t) : "";
  let water = "", open = false;
  wat.forEach((v, i) => {
    if (v === null) { open = false; return; }
    water += `${open ? "L" : "M"}${x(d[i])} ${y(v)}`;
    open = true;
  });
  const crestPath = crest !== null && Number.isFinite(crest) ? `M${x(0)} ${y(crest)}L${x(L)} ${y(crest)}` : "";
  return { ground, now, water, crest: crestPath, lo: r2(lo), hi: r2(hi), y };
}

/** The crest's height above the ground along the line: [least, most] in m (0 where the ground is higher). */
export function crestHeights(profile, crest) {
  const hs = profile.ground_m.map((g) => Math.max(0, crest - g));
  return [r2(Math.min(...hs)), r2(Math.max(...hs))];
}

/** The rows of the before-and-after table: buildings, roads (km) and people, each with before, after and change. */
export function exposureChangeRows(c) {
  const rows = [];
  const add = (key, label, unit, dp) => {
    if (!c || !c[key]) return;
    const { before, after, change } = c[key];
    const f = (v) => (dp ? (Math.round(v * 10 ** dp) / 10 ** dp).toFixed(dp) : Math.round(v).toLocaleString("en-GB"));
    rows.push({ key, label, unit, before: f(before), after: f(after),
      change: `${change > 0 ? "+" : change < 0 ? "−" : "±"}${f(Math.abs(change))}`, sign: Math.sign(change) });
  };
  add("buildings", "Buildings", "", 0);
  add("roads_km", "Roads", "km", 1);
  add("people", "People", "", 0);
  return rows;
}

/** The map features of the change: buildings kept dry and newly flooded, and the stretches of road that changed. */
export function exposureChangeFeatures(c) {
  const feats = [];
  for (const [key, list] of [["kept_dry", c && c.kept_dry], ["newly_flooded", c && c.newly_flooded]]) {
    for (const b of list || []) {
      const ring = b.ring.length && (b.ring[0][0] !== b.ring.at(-1)[0] || b.ring[0][1] !== b.ring.at(-1)[1]) ? [...b.ring, b.ring[0]] : b.ring;
      feats.push({ type: "Feature", properties: { change: key }, geometry: { type: "Polygon", coordinates: [ring] } });
    }
  }
  for (const p of (c && c.roads_changed) || []) {
    feats.push({ type: "Feature", properties: { change: p.change }, geometry: { type: "LineString", coordinates: p.coords } });
  }
  return { type: "FeatureCollection", features: feats };
}

/**
 * The cells where two runs on the same grid differ (wet in one only, or deeper by more than `tol` m), as a box
 * { r0, r1, c0, c1 } padded by `pad` cells and kept on the grid; null when they do not differ. Before and after is
 * counted inside it, so the open sources are read only where the levee or the breach changed the flood.
 */
export function changeBox(a, b, nx, ny, { wet = 0.05, tol = 0.05, pad = 12 } = {}) {
  let r0 = ny, r1 = -1, c0 = nx, c1 = -1;
  for (let i = 0; i < a.length; i++) {
    const wa = a[i] > wet, wb = b[i] > wet;
    if (wa === wb && (!wa || Math.abs(a[i] - b[i]) <= tol)) continue;
    const r = Math.floor(i / nx), c = i - r * nx;
    if (r < r0) r0 = r; if (r > r1) r1 = r;
    if (c < c0) c0 = c; if (c > c1) c1 = c;
  }
  if (r1 < 0) return null;
  return { r0: Math.max(0, r0 - pad), r1: Math.min(ny - 1, r1 + pad), c0: Math.max(0, c0 - pad), c1: Math.min(nx - 1, c1 + pad) };
}
