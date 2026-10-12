// Place context (#520): the pure presentation helpers for the Context card.
// No DOM and no map, so they run under node in the tests. Every number comes
// from aquascope.context in the worker; this file only picks words and shapes.

// The order the lines are asked for and shown: cheap and most asked-for first.
export const CONTEXT_LAYERS = [
  { id: "flood_history", label: "Flood history" },
  { id: "surface_water", label: "Surface water" },
  { id: "flood_hazard", label: "Flood depth" },
  { id: "dams", label: "Dams" },
  { id: "rain_gauge", label: "Rain gauge" },
  { id: "actual_et", label: "Evaporation" },
  { id: "soil", label: "Soil" },
];

// The order the layers are asked for (they are shown in CONTEXT_LAYERS order whatever order they land in).
// By the requests each makes, one after another in a worker: Dams and the rain gauge a few, Flood history and
// Evaporation a handful, Flood depth and Surface water about ten, Soil sixteen (its texture and water layers
// are separate files). Quickest first, so the first lines come within seconds; with more than one light
// worker Soil goes second, so the longest read starts at once rather than last.
const QUICKEST_FIRST = ["dams", "rain_gauge", "flood_history", "actual_et", "flood_hazard", "surface_water", "soil"];
export function fetchOrder(workers = 1) {
  if (workers <= 1) return QUICKEST_FIRST.slice();
  const rest = QUICKEST_FIRST.filter((id) => id !== "soil");
  return [rest[0], "soil", ...rest.slice(1)];
}

export const layerLabel = (id) => (CONTEXT_LAYERS.find((l) => l.id === id) || { label: id }).label;

// One line of the card: what to say for a layer's result (or its error).
export function contextLine(id, res) {
  const label = layerLabel(id);
  if (!res) return { id, label, state: "loading", text: "Looking…" };
  if (res.error && !res.summary) return { id, label, state: "error", text: String(res.error) };
  if (res.ok === false) return { id, label, state: "error", text: res.summary || "Could not read this layer." };
  const quiet = res.available === false || (res.news && res.news.available === false && !(res.radar && res.radar.available));
  return { id, label, state: quiet ? "empty" : "ok", text: res.summary || "" };
}

// Counts per year from a flood_history result: the news events when the
// mirror has them, else the radar months. Null when there is nothing to draw.
export function floodYears(res) {
  if (!res) return null;
  const news = res.news || {};
  if (news.available && news.by_year && Object.keys(news.by_year).length) {
    const years = Object.keys(news.by_year).sort();
    return { x: years, y: years.map((y) => news.by_year[y]), what: "flood events in the news", unit: "events" };
  }
  const radar = res.radar || {};
  if (radar.available && radar.months && Object.keys(radar.months).length) {
    const per = {};
    for (const m of Object.keys(radar.months)) per[m.slice(0, 4)] = (per[m.slice(0, 4)] || 0) + 1;
    const years = Object.keys(per).sort();
    return { x: years, y: years.map((y) => per[y]), what: "months with radar-detected flooding", unit: "months" };
  }
  return null;
}

// GHCN station names are upper case ("NIJMEGEN"); the card writes them as the summary line does.
export const titleCase = (name) => (name === name.toUpperCase()
  ? name.toLowerCase().replace(/(^|[\s\-/(])(\p{L})/gu, (_, sep, ch) => sep + ch.toUpperCase())
  : name);

// Annual rainfall at the nearest gauge, for the chart when there is no flood history.
export function rainYears(res) {
  const annual = res && res.record && res.record.annual_mm;
  if (!annual || !Object.keys(annual).length) return null;
  const years = Object.keys(annual).sort();
  const name = res.station && (res.station.name ? titleCase(res.station.name) : res.station.id);
  return { x: years, y: years.map((y) => annual[y]), what: `yearly rainfall at ${name || "the nearest gauge"}`, unit: "mm" };
}

// The card's one chart: flood history first, the rain gauge's years second.
export function pickChart(results) {
  return floodYears(results.flood_history) || rainYears(results.rain_gauge);
}

// Flood events as map points: a point's recent list, or an area's points.
export function floodPoints(res) {
  const features = [];
  if (res && Array.isArray(res.points)) {
    for (const [lat, lon, start] of res.points) {
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        features.push({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: { start: start || "" } });
      }
    }
  } else if (res && res.news && Array.isArray(res.news.recent)) {
    for (const e of res.news.recent) {
      if (Number.isFinite(e.lat) && Number.isFinite(e.lon)) {
        features.push({ type: "Feature", geometry: { type: "Point", coordinates: [e.lon, e.lat] }, properties: { start: e.start || "" } });
      }
    }
  }
  return { type: "FeatureCollection", features };
}

// The sources behind every line, once each, for the card's foot.
export function credits(results) {
  const seen = new Map();
  for (const res of Object.values(results)) {
    for (const s of (res && res.sources) || []) {
      if (!seen.has(s.key)) {
        seen.set(s.key, { short: s.short || s.label, label: s.label, attribution: s.attribution, licence: s.licence, homepage: s.homepage });
      }
    }
  }
  return [...seen.values()];
}

// The map can hand over longitudes from a neighbouring world copy (past 180);
// fold them back and clamp the latitudes, as the engine wants degrees on Earth.
export function normaliseBox(bbox) {
  const wrap = (x) => ((((x + 180) % 360) + 360) % 360) - 180;
  const clamp = (y) => Math.max(-90, Math.min(90, y));
  const width = bbox.east - bbox.west;
  const west = wrap(bbox.west);
  // A box a world wide or more keeps its full span; otherwise east follows west.
  const east = width >= 360 ? 180 : west + width > 180 ? wrap(bbox.east) : west + width;
  return { west: width >= 360 ? -180 : west, south: clamp(bbox.south), east, north: clamp(bbox.north) };
}

// A box the engine accepts: west < east (no date-line crossing), south < north.
export function boxProblem(bbox) {
  if (!bbox) return "No box.";
  if (bbox.west >= bbox.east) return "Context does not work on a box across the date line yet.";
  if (bbox.north - bbox.south > 16 || bbox.east - bbox.west > 16) return "Draw a smaller box (under about 16 degrees on a side) for its context.";
  return null;
}
