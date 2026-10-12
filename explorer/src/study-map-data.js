// The study map's data, with no map and no DOM: what the engine placed
// (aquascope.study_map, as GeoJSON on each run record and in the
// study_map.geojson artifact), gathered for the layer in study-map.js, the
// bounds of a step, the step list for the finished board, and which figures
// the map now shows instead. Nothing here decides where a step happened; the
// engine did. Node-importable (explorer/tests/study-map.test.mjs).

export const STUDY_MAP_ARTIFACT = "study-map";

// Figures that draw the site or the donors on bare longitude and latitude axes: the map shows those now, so
// the drawer leaves them out. They stay in the bundle.
export const ON_MAP_FIGURE_KINDS = ["site_map", "donors_map"];

const ROLE_WORD = {
  site: "site", gauge: "gauge", catchment: "catchment", grid_cell: "grid cell", donor: "donor", station: "station",
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const isFeature = (f) => Boolean(f && f.type === "Feature" && f.geometry && f.properties);

// A figure the map shows instead: by its kind when the artifact says, else by the fig-{step}-{kind} id.
export function drawnOnMap(fig) {
  if (!fig) return false;
  const kind = fig.meta && fig.meta.kind;
  if (kind) return ON_MAP_FIGURE_KINDS.includes(kind);
  const id = String(fig.id || "").replace(/-svg$/, "");
  return id.startsWith("fig-") && ON_MAP_FIGURE_KINDS.includes(id.slice(id.lastIndexOf("-") + 1));
}

// The FeatureCollection a streamed study_map.geojson artifact carries, or null.
export function featuresFromArtifact(artifact, decode = globalThis.atob) {
  if (!artifact || artifact.id !== STUDY_MAP_ARTIFACT || !artifact.data || typeof decode !== "function") return null;
  try {
    const bin = decode(artifact.data);
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    const fc = JSON.parse(new TextDecoder().decode(bytes));
    return fc && Array.isArray(fc.features) ? { type: "FeatureCollection", features: fc.features.filter(isFeature) } : null;
  } catch {
    return null;
  }
}

// The study map of a workspace the page holds (no bytes): the site, then each step's own features from its
// run record, the order the engine writes them in.
export function featuresFromWorkspace(ws) {
  const features = [];
  const results = (ws && ws.run && ws.run.results) || [];
  for (const r of results) {
    const fc = r && r.map;
    if (fc && Array.isArray(fc.features)) features.push(...fc.features.filter(isFeature));
  }
  const site = ws && ws.site;
  if (features.length && site && Number.isFinite(Number(site.lat)) && Number.isFinite(Number(site.lon))) {
    features.unshift({ type: "Feature", geometry: { type: "Point", coordinates: [Number(site.lon), Number(site.lat)] },
      properties: { role: "site", label: "Study site", step_id: null } });
  }
  return { type: "FeatureCollection", features };
}

// The layers study-map.js adds, as plain MapLibre specs (checked against the style spec): areas (the
// catchment polygon, the grid cells) as a wash and a dashed line, points by role, and a highlight pair
// filtered to one step.
const INK = "#c2185b";      // the study's colour: distinct from the gauge styles and the catchment blue
const DONOR = "#6a1b9a";
const HL = "#ff9800";
const IS_AREA = ["in", ["geometry-type"], ["literal", ["Polygon", "MultiPolygon"]]];
const IS_POINT = ["==", ["geometry-type"], "Point"];
const stepIs = (id) => ["==", ["get", "step_id"], id === null || id === undefined ? "" : String(id)];

export function highlightFilters(stepId) {
  return { line: ["all", IS_AREA, stepIs(stepId)], points: ["all", IS_POINT, stepIs(stepId)] };
}

export function studyLayers(source) {
  const hl = highlightFilters(null);
  return [
    { id: "study-fill", type: "fill", source, filter: IS_AREA,
      paint: { "fill-color": INK, "fill-opacity": ["case", ["==", ["get", "role"], "catchment"], 0.12, 0.06] } },
    { id: "study-line", type: "line", source, filter: IS_AREA,
      paint: { "line-color": INK, "line-width": 1.4, "line-dasharray": [3, 2] } },
    { id: "study-points", type: "circle", source, filter: IS_POINT,
      paint: {
        "circle-radius": ["match", ["get", "role"], "site", 5, "gauge", 7, "catchment", 6, 4.5],
        "circle-color": ["match", ["get", "role"], "donor", DONOR, "station", "#ffffff", "site", "#ffffff", INK],
        "circle-stroke-color": ["match", ["get", "role"], "donor", "#ffffff", INK],
        "circle-stroke-width": ["match", ["get", "role"], "site", 2.5, "station", 1.5, 1.2],
      } },
    { id: "study-hl-line", type: "line", source, filter: hl.line, paint: { "line-color": HL, "line-width": 3 } },
    { id: "study-hl-points", type: "circle", source, filter: hl.points,
      paint: { "circle-radius": 10, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": HL, "circle-stroke-width": 3 } },
  ];
}

function walk(coords, out) {
  if (!Array.isArray(coords)) return;
  if (coords.length >= 2 && typeof coords[0] === "number" && typeof coords[1] === "number") { out.push(coords); return; }
  for (const c of coords) walk(c, out);
}

// [west, south, east, north] over the features (those of one step when stepId is given), or null.
export function boundsOf(features, stepId = null) {
  const pts = [];
  for (const f of features || []) {
    if (!isFeature(f)) continue;
    if (stepId !== null && f.properties.step_id !== stepId) continue;
    walk(f.geometry.coordinates, pts);
  }
  if (!pts.length) return null;
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const [x, y] of pts) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
  return [w, s, e, n];
}

// The steps that put something on the map, in run order: [{ id, tool, words }].
export function stepsOnMap(ws) {
  const out = [];
  for (const r of (ws && ws.run && ws.run.results) || []) {
    const feats = ((r && r.map && r.map.features) || []).filter(isFeature);
    if (!r || !r.id || !feats.length) continue;
    const counts = new Map();
    for (const f of feats) {
      const word = ROLE_WORD[f.properties.role] || String(f.properties.role || "place");
      counts.set(word, (counts.get(word) || 0) + 1);
    }
    const words = [...counts].map(([w, n]) => (n > 1 ? `${n} ${w}s` : w)).join(", ");
    out.push({ id: String(r.id), tool: r.tool, words });
  }
  return out;
}

// The finished board's "On the map" row: one button per step, which the page wires to focus that step.
export function stepsOnMapHtml(ws, label = (t) => String(t || "")) {
  const steps = stepsOnMap(ws);
  if (!steps.length) return "";
  return `<div class="study-map-steps"><span class="muted">On the map:</span> ` +
    steps.map((s) => `<button type="button" class="chip" data-map-step="${esc(s.id)}" title="${esc(s.words)}">` +
      `${esc(s.id)} ${esc(label(s.tool))}</button>`).join(" ") +
    `</div>`;
}
