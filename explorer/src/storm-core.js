// Draw a storm (#670), the pure part: no DOM, no map, so node can test it. The storm's runoff and its wave come
// from Python (aquascope.routing.route_storm: rain depth times a runoff coefficient over every unit catchment under
// the drawn area, routed down GEOGLOWS's own network with Muskingum). This module checks the inputs, draws the
// area and the hydrographs, and says the numbers the package wrote.

import { flowWords, hoursWords } from "./route-core.js?v=__BUILD__";

/** What the panel starts with: the package's own defaults for the hours and the runoff coefficient. */
export const STORM_START = { rain_mm: 50, duration_h: 6, runoff_coefficient: 0.3 };
const LIMITS = { rain_mm: [0, 2000], duration_h: [0, 720], runoff_coefficient: [0, 1] };

const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));
const r5 = (v) => Math.round(Number(v) * 1e5) / 1e5;

/** A drawn outline, checked: [[lon, lat], ...] of 3 to 200 points (the ring is not closed). Throws on a bad one. */
export function checkRing(points) {
  let ring = (points || []).map((p) => [Number(p[0]), Number(p[1])]);
  if (ring.length > 3 && ring[0][0] === ring.at(-1)[0] && ring[0][1] === ring.at(-1)[1]) ring = ring.slice(0, -1);
  if (ring.length < 3 || ring.length > 200 || ring.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 180 || Math.abs(y) > 85)) {
    throw new Error("a storm needs an area of 3 to 200 [lon, lat] points");
  }
  return ring.map(([x, y]) => [r5(x), r5(y)]);
}

/** The worker's arguments for route_storm, checked: { polygon, rain_mm, duration_h, runoff_coefficient }. */
export function stormArgs({ polygon, rain_mm, duration_h, runoff_coefficient }) {
  const out = { polygon: checkRing(polygon) };
  for (const [key, v] of [["rain_mm", rain_mm], ["duration_h", duration_h], ["runoff_coefficient", runoff_coefficient]]) {
    const [lo, hi] = LIMITS[key];
    if (!finite(v) || !(Number(v) > lo && Number(v) <= hi)) {
      throw new Error({ rain_mm: "the rain must be above 0 and at most 2000 mm",
        duration_h: "the hours must be above 0 and at most 720",
        runoff_coefficient: "the runoff coefficient must be above 0 and at most 1" }[key]);
    }
    out[key] = Number(v);
  }
  return out;
}

/** "50 mm in 6 h": the label the area carries on the map. */
export function stormLabel(s) {
  if (!s || !finite(s.rain_mm)) return "";
  return `${Number(s.rain_mm)} mm in ${hoursWords(s.duration_h)}`;
}

/**
 * The area on the map: the outline (closed), and its label once at the middle of its corners (a polygon's own label
 * would be placed again in every tile it crosses).
 */
export function stormFeatures(storm, { live = false } = {}) {
  if (!storm || !storm.polygon || storm.polygon.length < 3) return { type: "FeatureCollection", features: [] };
  const ring = [...storm.polygon, storm.polygon[0]];
  const n = storm.polygon.length;
  const mid = [storm.polygon.reduce((a, p) => a + p[0], 0) / n, storm.polygon.reduce((a, p) => a + p[1], 0) / n];
  return { type: "FeatureCollection", features: [
    { type: "Feature", properties: { live: live ? 1 : 0 }, geometry: { type: "Polygon", coordinates: [ring] } },
    { type: "Feature", properties: { label: stormLabel(storm) }, geometry: { type: "Point", coordinates: mid } },
  ] };
}

/** The [[w, s], [e, n]] box of an outline, or null. */
export function ringBounds(ring) {
  if (!ring || !ring.length) return null;
  const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
  return [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
}

/**
 * The card's chart: each hydrograph the package returned (`res.hydrographs`: the reach with the biggest peak and
 * the main outlet, hourly m³/s) as an SVG path in a w x h box, on one scale, with the hours across.
 */
export function hydroChart(hydros, { w = 300, h = 96, pad = { l: 34, r: 8, t: 8, b: 18 } } = {}) {
  const list = (hydros || []).filter((d) => d && Array.isArray(d.cms) && d.cms.length > 1);
  if (!list.length) return null;
  const top = Math.max(...list.flatMap((d) => d.cms)) || 1;
  const n = Math.max(...list.map((d) => d.cms.length)) - 1;
  const dt = Number(list[0].dt_h) || 1;
  const x = (i) => Math.round((pad.l + ((w - pad.l - pad.r) * i) / n) * 10) / 10;
  const y = (v) => Math.round((h - pad.b - ((h - pad.t - pad.b) * v) / top) * 10) / 10;
  const paths = list.map((d) => ({
    role: d.role, river_id: d.river_id, peak_cms: d.peak_cms,
    d: d.cms.map((v, i) => `${i ? "L" : "M"}${x(i)} ${y(v)}`).join(""),
  }));
  const hours = n * dt;
  const step = [6, 12, 24, 48, 96, 168].find((s) => hours / s <= 6) || 168;
  const ticks = [];
  for (let t = 0; t <= hours; t += step) ticks.push({ x: x(t / dt), label: hoursWords(t) });
  return { paths, top, hours, ticks, y0: y(0), yTop: y(top), w, h, pad };
}

/** The result's first line, from the package's numbers. */
export function stormHeadline(res) {
  if (!res || !res.ok || !res.storm) return "";
  const s = res.storm, b = res.biggest || {};
  const when = b.peak_h === null || b.peak_h === undefined ? "still rising at the end" : `${hoursWords(b.peak_h)} after the rain starts`;
  return `${Number(s.rain_mm)} mm in ${hoursWords(s.duration_h)} over ${Math.round(s.area_km2).toLocaleString("en-US")} km²: ` +
    `the rivers below carry up to ${flowWords(b.peak_cms)} m³/s more, ${when}.`;
}

/** The card's numbers: [{ label, value }], each written by the package. */
export function stormFacts(res) {
  if (!res || !res.ok || !res.storm) return [];
  const s = res.storm;
  const rows = [
    { label: "Runs off", value: `${Number(s.runoff_coefficient)} of the rain` },
    { label: "Under the storm", value: `${s.n_reaches.toLocaleString("en-US")} reaches, ${Math.round(s.area_km2).toLocaleString("en-US")} km²` },
    { label: "Runoff", value: `${volumeWords(s.runoff_m3)}` },
  ];
  const outlet = (res.hydrographs || []).find((d) => d.role === "outlet");
  if (outlet) rows.push({ label: "At the main outlet", value: `${flowWords(outlet.peak_cms)} m³/s, after ${hoursWords(outlet.peak_h)}` });
  if (finite(s.out_pct)) rows.push({ label: "Left by the outlets", value: `${Number(s.out_pct)} % of the runoff in ${hoursWords(res.hours)}` });
  return rows;
}

/** "19.3 million m³", "840,000 m³". */
export function volumeWords(m3) {
  if (!finite(m3)) return "n/a";
  const v = Number(m3);
  if (v >= 1e9) return `${(v / 1e9).toFixed(2).replace(/\.?0+$/, "")} km³`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1).replace(/\.0$/, "")} million m³`;
  return `${Math.round(v).toLocaleString("en-US")} m³`;
}

/** The action log's line for the area drawn or changed. */
export function areaLogLabel(storm, { moved = false } = {}) {
  return `${moved ? "Move" : "Draw"} the storm area (${storm.polygon.length} points)`;
}
