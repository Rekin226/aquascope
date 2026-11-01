// Change land use (#661), the pure part: the uses and their words, the inputs checked, the area and the outlet on
// the map, the cover bar, the facts and the log's lines. No DOM, no map: node imports this
// (explorer/tests/landuse-core.test.mjs). Every curve number, depth and peak comes from aquascope.sketch.landuse in
// the worker; this module only says them.

// The uses, as aquascope.sketch.landuse.TR55 labels them (tests/test_sketch_landuse.py keeps the two equal).
export const USES = [
  ["residential_1_8", "Residential, 1/8 acre lots or town houses (65% impervious)"],
  ["residential_1_4", "Residential, 1/4 acre lots (38% impervious)"],
  ["residential_1_2", "Residential, 1/2 acre lots (25% impervious)"],
  ["residential_1_acre", "Residential, 1 acre lots (20% impervious)"],
  ["commercial", "Commercial and business (85% impervious)"],
  ["industrial", "Industrial (72% impervious)"],
  ["paved", "Paved parking lots, roofs, driveways"],
  ["newly_graded", "Newly graded areas (no vegetation)"],
  ["open_space_good", "Open space, lawns and parks, good condition (grass cover > 75%)"],
  ["row_crops", "Row crops, straight row, good condition"],
  ["pasture_good", "Pasture, grassland or range, good condition"],
  ["meadow", "Meadow, continuous grass mowed for hay"],
  ["woods_good", "Woods, good condition"],
  ["impervious", "Your own share of impervious area"],
];
// WorldCover's legend (aquascope.sketch.landuse.WORLDCOVER), for the cover bar.
export const COVER_COLORS = {
  10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0",
  80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
};
// The drawn area's colour by what it becomes: built (red-brown), farmed (violet), green (green).
export const USE_TONE = { built: "#b5452a", farmed: "#8e44ad", green: "#2e7d32" };
const TONE = {
  residential_1_8: "built", residential_1_4: "built", residential_1_2: "built", residential_1_acre: "built",
  commercial: "built", industrial: "built", paved: "built", newly_graded: "built", impervious: "built",
  row_crops: "farmed", open_space_good: "green", pasture_good: "green", meadow: "green", woods_good: "green",
};
export const DISTRIBUTIONS = ["I", "IA", "II", "III"];
// What the panel starts with: the package's placeholder design storm (aquascope.sketch.landuse.DEFAULT_RAIN_MM),
// for the reader to replace with the local 24-hour design depth.
export const START = { use: "residential_1_4", rain_mm: 100, distribution: "II", soil_group: "", impervious_pct: 50 };

export const useLabel = (key) => (USES.find(([k]) => k === key) || [key, key])[1];
export const useShort = (key) => useLabel(key).split(" (")[0];
export const useTone = (key) => USE_TONE[TONE[key] || "built"];

const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));
const r5 = (v) => Math.round(Number(v) * 1e5) / 1e5;

/** A drawn outline, checked: [[lon, lat], ...] of 3 to 200 points (the ring is not closed). Throws on a bad one. */
export function checkRing(points) {
  let ring = (points || []).map((p) => [Number(p[0]), Number(p[1])]);
  if (ring.length > 3 && ring[0][0] === ring.at(-1)[0] && ring[0][1] === ring.at(-1)[1]) ring = ring.slice(0, -1);
  if (ring.length < 3 || ring.length > 200 || ring.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y) ||
    Math.abs(x) > 180 || Math.abs(y) > 85)) {
    throw new Error("a land-use change needs an area of 3 to 200 [lon, lat] points");
  }
  return ring.map(([x, y]) => [r5(x), r5(y)]);
}

/** The worker's arguments for landuse_change, checked. */
export function landuseArgs({ polygon, use, rain_mm, distribution, soil_group, impervious_pct }) {
  const out = { polygon: checkRing(polygon) };
  if (!USES.some(([k]) => k === use)) throw new Error("pick what the area becomes");
  out.use = use;
  if (!finite(rain_mm) || !(Number(rain_mm) >= 1 && Number(rain_mm) <= 2000)) {
    throw new Error("the design storm must be from 1 to 2000 mm");
  }
  out.rain_mm = Number(rain_mm);
  out.distribution = DISTRIBUTIONS.includes(distribution) ? distribution : "II";
  if (soil_group) {
    if (!["A", "B", "C", "D"].includes(soil_group)) throw new Error("the soil group is A, B, C or D");
    out.soil_group = soil_group;
  }
  if (use === "impervious") {
    if (!finite(impervious_pct) || Number(impervious_pct) < 0 || Number(impervious_pct) > 100) {
      throw new Error("the impervious share is 0 to 100%");
    }
    out.impervious_pct = Number(impervious_pct);
  }
  return out;
}

// ── words ───────────────────────────────────────────────────────────────────

const sig = (x, d = 2) => (Math.abs(x) >= 100 ? Math.round(x).toLocaleString("en-US") : Number(x.toFixed(d)).toString());

/** A flow: "9.84", "0.42", "1,250". */
export const flow = (q) => (q === null || q === undefined ? "n/a" : sig(q, Math.abs(q) < 1 ? 3 : 2));

/** A signed change: "+12%", "-3%". */
export const pctWords = (p) => (p === null || p === undefined ? "" : `${p > 0 ? "+" : ""}${Math.round(p)}%`);

/** The map label on the area: what it becomes and the change in its peak. */
export function areaLabel(res) {
  if (!res) return "";
  return `${useShort(res.use)}\n${pctWords(res.change.peak_pct)} peak`;
}

/** The panel's one line. */
export function headline(res) {
  const b = res.before, a = res.after;
  return `${res.area_km2 >= 1 ? res.area_km2.toFixed(2) : res.area_km2.toFixed(3)} km² as ${useShort(res.use).toLowerCase()}: ` +
    `runoff ${Math.round(b.q_mm)} to ${Math.round(a.q_mm)} mm from ${res.rain_mm} mm of rain, the area's peak ` +
    `${flow(b.peak_cms)} to ${flow(a.peak_cms)} m³/s (${pctWords(res.change.peak_pct)}).`;
}

/** The before and after table's rows: [label, before, after]. */
export function compareRows(res) {
  const b = res.before, a = res.after;
  return [
    ["Curve number", b.cn.toFixed(0), a.cn.toFixed(0)],
    ["Runoff, mm", b.q_mm.toFixed(0), a.q_mm.toFixed(0)],
    ["Time of concentration, h", b.tc_h.toFixed(2), a.tc_h.toFixed(2)],
    ["Peak, m³/s", flow(b.peak_cms), flow(a.peak_cms)],
  ];
}

/** The cover under the area now as a bar: [{ color, share, label }], the biggest first. */
export function coverBar(rows) {
  return (rows || []).filter((r) => r.share > 0).sort((x, y) => y.share - x.share)
    .map((r) => ({ color: COVER_COLORS[r.code] || "#999999", share: r.share, label: r.label, cn: r.cn }));
}

/** The soil groups as words: "C 80%, D 20%". */
export const soilWords = (soils) => Object.entries(soils || {}).map(([g, f]) => `${g} ${Math.round(f * 100)}%`).join(", ");

/** The log's line for a change. */
export function logLabel(value) {
  if (!value || !value.result) return value ? "Draw an area for a land-use change" : "Take the land-use change off";
  const r = value.result;
  return `${useShort(r.use)} on ${r.area_km2.toFixed(r.area_km2 >= 1 ? 1 : 2)} km²: peak ${pctWords(r.change.peak_pct)}`;
}

/** Route it's arguments for the added peak (a triangle carrying the added volume), or null when it is not larger. */
export const routeArgs = (res) => (res && res.route ? { peak_cms: res.route.peak_cms, duration_h: res.route.duration_h,
  shape: "triangle" } : null);

// ── on the map ──────────────────────────────────────────────────────────────

/** The area (filled in what it becomes, labelled) and the way to its reach. */
export function areaFeatures(value, { live = false } = {}) {
  const fc = { type: "FeatureCollection", features: [] };
  if (!value || !value.polygon) return fc;
  const res = value.result;
  const ring = [...value.polygon, value.polygon[0]];
  fc.features.push({ type: "Feature", properties: { kind: "area", color: res ? useTone(res.use) : "#1f4e8c",
    label: res ? areaLabel(res) : "", live: live ? 1 : 0 }, geometry: { type: "Polygon", coordinates: [ring] } });
  if (res && res.reach && res.reach.from) {
    fc.features.push({ type: "Feature", properties: { kind: "path" },
      geometry: { type: "LineString", coordinates: [res.reach.from, [res.reach.lon, res.reach.lat]] } });
    fc.features.push({ type: "Feature", properties: { kind: "outlet", label: `${flow(res.after.peak_cms)} m³/s` },
      geometry: { type: "Point", coordinates: [res.reach.lon, res.reach.lat] } });
  }
  return fc;
}

/** The camera's box round an area: [[w, s], [e, n]]. */
export function ringBounds(ring, res = null) {
  const pts = [...(ring || [])];
  if (res && res.reach) pts.push([res.reach.lon, res.reach.lat]);
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
}
