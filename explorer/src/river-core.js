// Rivers as objects (#516), the pure part: no DOM, no map, so node can test it.
// What a river reach is, its record and its path to the sea all come from
// Python (aquascope.rivers); this module only decides how they are drawn and
// said: the stream-order styling, the snap sentence, and the growing line of
// the trace animation.

// The GEOGLOWS v2 global stream network, read in place by range requests
// (2.4 GB, CORS open). TDX-Hydro geometry, CC BY-SA 4.0: shown, never republished.
export const STREAMS_PMTILES = "https://geoglows-v2.s3.us-west-2.amazonaws.com/hydrography-global/streams.pmtiles";
export const RIVERS_CREDIT = {
  label: "Rivers",
  attribution: 'GEOGLOWS v2 stream network from <a href="https://registry.opendata.aws/geoglows-v2/">TDX-Hydro</a> (NGA)',
  licence: "CC BY-SA 4.0, shown and not republished",
};
// The same credit, short, for the map's attribution line (the network is on by default since #545).
export const RIVERS_ATTRIBUTION =
  'Rivers: <a href="https://registry.opendata.aws/geoglows-v2/">GEOGLOWS v2, TDX-Hydro (NGA)</a>, CC BY-SA 4.0';
export const RECORD_CREDIT = "GEOGLOWS v2 retrospective simulation (GEOGloWS ECMWF Streamflow Service), CC BY 4.0";

// The network on the globe (#545). The tiles hold orders 6 and up at every
// zoom, 4 and up from zoom 6 and all from zoom 8; the style fades a river in
// by its order as you zoom, so the globe shows the great rivers (order 8 and
// up, order 7 faintly), a continent its main tributaries and a valley every stream. Each table
// is zoom -> [[Strahler order, value], ...]; MapLibre interpolates between.
const WIDTH = {
  1: [[6, 0], [7, 0.6], [8, 1.4], [9, 2.3], [10, 3.1], [12, 4]],
  3: [[6, 0.4], [7, 1], [8, 1.6], [9, 2.4], [10, 3.2], [12, 4.2]],
  5: [[4, 0.3], [6, 1], [8, 1.8], [10, 3.2]],
  8: [[1, 0.4], [4, 0.9], [6, 1.7], [10, 4.2]],
  12: [[1, 1], [6, 3], [10, 6.5]],
};
const OPACITY = {
  1: [[6, 0], [7, 0.4], [8, 0.85], [9, 0.95], [10, 1]],
  3: [[6, 0], [7, 0.6], [8, 0.85], [10, 0.95]],
  5: [[4, 0], [5, 0.4], [6, 0.7], [8, 0.85], [10, 0.95]],
  8: [[1, 0.35], [3, 0.55], [6, 0.85], [10, 0.95]],
  11: [[1, 0.6], [4, 0.85], [10, 0.95]],
};
// The moving glints: only on the lines wide enough to carry them.
// A light glint walking down a blue line (#543 design pass): you see the water move without a dark dash
// marching over every river; on the globe only the great rivers carry it.
const FLOW_OPACITY = {
  1: [[8, 0], [9, 0.45], [10, 0.6]],
  3: [[7, 0], [8, 0.5], [10, 0.6]],
  5: [[5, 0], [6, 0.5], [10, 0.62]],
  8: [[2, 0], [3, 0.5], [10, 0.62]],
};

const ORDER = ["coalesce", ["get", "strahlerOrder"], 1];
const byOrder = (stops, f = (v) => v) => ["interpolate", ["linear"], ORDER, ...stops.flatMap(([o, v]) => [o, f(v)])];
const byZoom = (table, f) => ["interpolate", ["linear"], ["zoom"],
  ...Object.entries(table).flatMap(([z, stops]) => [Number(z), f(stops)])];

// Line width by Strahler order, growing with zoom: big rivers read at a
// globe's scale, the headwater streams only once you are close.
export function riverWidth() { return byZoom(WIDTH, (stops) => byOrder(stops)); }
// `scale` dims the plain network while a click's network is lit, so the answer stands out.
export function riverOpacity(scale = 1) { return byZoom(OPACITY, (stops) => byOrder(stops, (v) => Number((v * scale).toFixed(3)))); }
export function flowOpacity() { return byZoom(FLOW_OPACITY, (stops) => byOrder(stops)); }

// The network around a clicked reach is drawn by feature-state on riverId:
// hl 1 = drains here (upstream), 2 = on the way to the sea, 3 = the reach itself.
export const HL = { up: 1, down: 2, here: 3 };
const hl = ["coalesce", ["feature-state", "hl"], 0];

// Wider than the plain line, and never thinner than a hair, so a lit
// tributary still shows at a zoom where the plain network hides it.
export function highlightWidth(extra = 0) {
  return byZoom(WIDTH, (stops) => ["match", hl,
    HL.here, byOrder(stops, (v) => Math.max(2.6, v + 2.2) + extra),
    HL.down, byOrder(stops, (v) => Math.max(1.8, v + 1.4) + extra),
    HL.up, byOrder(stops, (v) => Math.max(0.7, v + 0.5) + extra),
    0]);
}
export function highlightColor(theme) {
  return ["match", hl, HL.here, theme.down, HL.down, theme.down, theme.up];
}
export function highlightOpacity(on = 1) { return ["case", [">", hl, 0], on, 0]; }

// Colours that hold on every basemap and for every kind of colour vision: the
// network a calm blue, what drains to a click a stronger blue, its way to the
// sea orange (blue against orange is the pair no colour blindness merges),
// each lit line on a casing of the basemap's own background.
export const RIVER_THEMES = {
  light: { line: "#3478bd", flow: "#e6f3ff", up: "#0f4f9c", down: "#d95f02", casing: "#ffffff" },
  dark: { line: "#4f9de0", flow: "#e2f1ff", up: "#8fd0ff", down: "#ff9d42", casing: "#0b141d" },
  imagery: { line: "#6bb9f2", flow: "#ffffff", up: "#a8dcff", down: "#ff9d42", casing: "#0b141d" },
};
export function riverTheme(basemap) {
  if (basemap === "dark") return RIVER_THEMES.dark;
  if (basemap === "satellite" || basemap === "satellite-recent") return RIVER_THEMES.imagery;
  return RIVER_THEMES.light;
}

// The flow animation is a dash that walks along each line. MapLibre has no
// dash offset, so the phase is baked into the dash array: FLOW_STEPS arrays
// per period, reused (each new array costs a row in MapLibre's dash atlas, so
// the phase is never continuous). TDX-Hydro draws a reach from its downstream
// end, so the dash walks towards the start of the line: the way the water goes.
export const FLOW_DASH = 1.2;     // dash length, in line widths
export const FLOW_PERIOD = 9;     // dash plus gap
export const FLOW_STEPS = 24;
export const FLOW_FPS = 16;
export function flowDash(step, { dash = FLOW_DASH, period = FLOW_PERIOD, steps = FLOW_STEPS } = {}) {
  const k = ((Math.round(step) % steps) + steps) % steps;
  // Downstream is towards the line's start, so the dash's offset shrinks as time goes on.
  const s = Number((((steps - k) % steps) * (period / steps)).toFixed(4));
  if (s + dash <= period) return [0, s, dash, Number((period - s - dash).toFixed(4))];
  const head = Number((s + dash - period).toFixed(4));
  return [head, Number((period - dash).toFixed(4)), Number((period - s).toFixed(4)), 0];
}

// The states to set for a lit network: the reach, its way to the sea, what
// drains to it, each id once with the strongest role.
export function networkStates(reachId, up = [], down = []) {
  const out = new Map();
  for (const id of up || []) out.set(Number(id), HL.up);
  for (const id of down || []) out.set(Number(id), HL.down);
  if (reachId !== null && reachId !== undefined) out.set(Number(reachId), HL.here);
  return out;
}

// The few words under the lit network: how much drains here and how far the
// water goes. Every number is Python's (aquascope.rivers.upstream_ids and
// downstream_ids); this only says it.
export function networkSummary(net) {
  if (!net) return { up: "", down: "", cut: "" };
  const u = net.upstream || {}, d = net.downstream || {};
  const n = Number(u.n_upstream);
  const area = Number(u.upstream_area_km2);
  const areaText = Number.isFinite(area)
    ? area >= 1e6 ? `${Number((area / 1e6).toFixed(2))} million km²` : `${Math.round(area).toLocaleString("en-US")} km²`
    : "";
  let up = Number.isFinite(n) ? `${n.toLocaleString("en-US")} reach${n === 1 ? "" : "es"}` : "";
  if (areaText) up += `${up ? ", " : ""}${areaText}`;
  // A big basin is lit by its largest reaches only; the key says where the cut fell.
  const cut = u.truncated && Number.isFinite(Number(u.min_area_km2))
    ? `lit: reaches draining over ${Math.round(Number(u.min_area_km2)).toLocaleString("en-US")} km²` : "";
  const m = Number(d.n_ids);
  const down = Number.isFinite(m) ? `${m.toLocaleString("en-US")} reach${m === 1 ? "" : "es"}${d.truncated ? " and on" : ""}` : "";
  return { up, down, cut };
}

function distanceText(m) {
  const n = Number(m);
  if (!Number.isFinite(n)) return "";
  return n < 1000 ? `${Math.round(n)} m` : `${Number((n / 1000).toFixed(1))} km`;
}

// The bigger river beyond the tolerance the snap named (aquascope.rivers: a braided river's water can be
// kilometres from its mapped centreline), as a clause; the page offers it with a button.
function largerText(snap) {
  const l = snap && snap.larger;
  return l ? ` A larger river (order ${l.strahler_order}) is ${distanceText(l.distance_m)} away.` : "";
}

// One plain sentence for the snap, whichever way it went. Python chose the reach (the main channel within
// the tolerance, or for a gauge the reach whose upstream area matches its catchment); this says which.
export function snapLine(snap, { gauge = false } = {}) {
  if (!snap) return "";
  if (snap.snapped) {
    const order = snap.strahler_order ? `, stream order ${snap.strahler_order}` : "";
    const d = distanceText(snap.distance_m);
    const nearer = snap.nearer ? distanceText(snap.nearer.distance_m) : "";
    let line;
    if (snap.choice === "main_channel") {
      line = `${gauge ? `This gauge is ${d} from` : `Snapped ${d} to`} the main channel (order ${snap.strahler_order}); ` +
        `a smaller stream is ${nearer} away.`;
    } else if (snap.choice === "area") {
      line = `${gauge ? `This gauge is ${d} from` : `Snapped ${d} to`} river reach ${snap.river_id}${order}, the one ` +
        `whose upstream area matches the catchment${nearer ? ` (the nearest line is ${nearer} away)` : ""}.`;
    } else {
      line = gauge
        ? `This gauge is ${d} from river reach ${snap.river_id}${order}.`
        : `Snapped ${d} to river reach ${snap.river_id}${order}.`;
    }
    return line + largerText(snap);
  }
  const tol = distanceText(snap.max_distance_m);
  if (snap.nearest) {
    return `No stream within ${tol}. The nearest mapped reach is ${distanceText(snap.nearest.distance_m)} away.` + largerText(snap);
  }
  return `No stream within ${distanceText(snap.searched_m || snap.max_distance_m)} of this point.`;
}

function segKm(a, b) {
  const d2r = Math.PI / 180;
  const dLat = (b[1] - a[1]) * d2r, dLon = (b[0] - a[0]) * d2r;
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * d2r) * Math.cos(b[1] * d2r) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(Math.min(1, s)));
}

// Cumulative distance along a line, km, one entry per vertex.
export function cumulativeKm(coords) {
  const out = [0];
  for (let i = 1; i < (coords || []).length; i++) out.push(out[i - 1] + segKm(coords[i - 1], coords[i]));
  return out;
}

// The part of a line from its start to `fraction` of its length, with the
// last point interpolated: the trace grows at an even speed along the river,
// not vertex by vertex (vertices are dense in bends and sparse on the plains).
export function lineUpTo(coords, fraction, cum = null) {
  if (!coords || coords.length < 2) return coords ? coords.slice() : [];
  const f = Math.max(0, Math.min(1, Number(fraction) || 0));
  if (f >= 1) return coords.slice();
  const c = cum || cumulativeKm(coords);
  const target = c[c.length - 1] * f;
  let i = 1;
  while (i < c.length && c[i] < target) i++;
  if (i >= c.length) return coords.slice();
  const span = c[i] - c[i - 1];
  const t = span > 0 ? (target - c[i - 1]) / span : 0;
  const a = coords[i - 1], b = coords[i];
  return [...coords.slice(0, i), [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]];
}

// [west, south, east, north] of a line, for fitting the map to it.
export function lineBounds(coords) {
  if (!coords || !coords.length) return null;
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const [x, y] of coords) {
    if (x < w) w = x;
    if (x > e) e = x;
    if (y < s) s = y;
    if (y > n) n = y;
  }
  return [w, s, e, n];
}

// Dams on the trace (Global Dam Watch v1.0, CC BY 4.0) and the borders it
// crosses (Natural Earth, public domain). The dams themselves, their km along
// the path and the country list all come from aquascope.river_path.
export const DAMS_CREDIT = "Dams: Global Dam Watch v1.0 (Lehner et al. 2024), CC BY 4.0";
export const BORDERS_CREDIT = "Borders: Natural Earth 1:50m, public domain";

// GDW leaves most small barriers unnamed; a reservoir name is the next best thing.
export function damName(d) {
  if (!d) return "";
  return d.name && d.name !== "unnamed" ? d.name : d.reservoir ? `${d.reservoir} dam` : "Unnamed dam";
}

// Storage and main use, in a few words: "25 million m³, hydroelectricity".
export function damFacts(d) {
  if (!d) return "";
  const bits = [];
  const cap = Number(d.capacity_mcm);
  if (d.capacity_mcm !== null && d.capacity_mcm !== undefined && Number.isFinite(cap) && cap > 0) {
    bits.push(`${cap >= 10 ? Math.round(cap).toLocaleString("en-US") : Number(cap.toFixed(1))} million m³`);
  }
  if (d.purpose) bits.push(String(d.purpose).toLowerCase());
  return bits.join(", ");
}

// The dams as map points, each carrying its index in the list it came from.
export function damsGeoJSON(dams) {
  const features = [];
  (dams || []).forEach((d, i) => {
    const lat = Number(d && d.lat), lon = Number(d && d.lon);
    if (d && d.lat !== null && d.lon !== null && Number.isFinite(lat) && Number.isFinite(lon)) {
      features.push({ type: "Feature", properties: { i, name: damName(d) }, geometry: { type: "Point", coordinates: [lon, lat] } });
    }
  });
  return { type: "FeatureCollection", features };
}

// The dams worth a line in the list: those GDW names or gives a storage for.
// The many unnamed weirs stay on the map and are counted, not listed.
export function notableDams(dams) {
  const all = dams || [];
  const notable = all.filter((d) => (d.name && d.name !== "unnamed") || d.reservoir || Number(d.capacity_mcm) > 0);
  return { notable, others: all.length - notable.length };
}
