// The study on the map: one GeoJSON source and a few layers that show where
// the crew looked (the site, the gauge, the catchment, the ERA5 and GloFAS
// cells as boxes, the donor gauges), drawn as each step lands, one step
// picked out when its row is clicked, cleared on New study. The features are
// the engine's (aquascope.study_map); this module only draws them. The
// source and layers are named study-*, which map.js carries across a
// basemap change.

import { state } from "./core.js?v=__BUILD__";
import { fitBoundsTo, map } from "./map.js?v=__BUILD__";
import { boundsOf, featuresFromArtifact, featuresFromWorkspace, highlightFilters, studyLayers } from "./study-map-data.js?v=__BUILD__";

const SRC = "study-map";
const EMPTY = { type: "FeatureCollection", features: [] };
let current = EMPTY;
let focused = null;

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  try {
    if (!map.getSource(SRC)) map.addSource(SRC, { type: "geojson", data: EMPTY });
    for (const layer of studyLayers(SRC)) if (!map.getLayer(layer.id)) map.addLayer(layer);
    return true;
  } catch (err) {
    console.info("study map unavailable:", err && err.message);
    return false;
  }
}

function setHighlight(id) {
  if (!map.getLayer("study-hl-line")) return;
  const f = highlightFilters(id);
  map.setFilter("study-hl-line", f.line);
  map.setFilter("study-hl-points", f.points);
}

// Draw a FeatureCollection as the study map (it replaces what was drawn: the engine sends the whole study).
export function showStudyFeatures(fc) {
  current = fc && Array.isArray(fc.features) ? fc : EMPTY;
  if (!ensureLayers()) return;
  map.getSource(SRC).setData(current);
  if (focused !== null && !current.features.some((f) => f.properties.step_id === focused)) focused = null;
  setHighlight(focused);
}

// The map of a workspace the page holds (a finished, resumed or recorded study); an empty one clears it.
export function showStudyMapFor(ws) {
  showStudyFeatures(featuresFromWorkspace(ws));
}

// A streamed artifact during a run: study_map.geojson is drawn; anything else is ignored. True when drawn.
export function studyMapArtifact(artifact) {
  const fc = featuresFromArtifact(artifact);
  if (!fc) return false;
  showStudyFeatures(fc);
  return true;
}

// Pick out one step's features and bring them into view; the same step again lets go.
export function focusStudyStep(stepId) {
  // a fallback's events name "s3.fallback"; its features ride on s3
  const id = stepId === undefined || stepId === null ? null : String(stepId).replace(/\.fallback$/, "");
  focused = focused === id ? null : id;
  if (!ensureLayers()) return;
  setHighlight(focused);
  if (focused === null) return;
  const b = boundsOf(current.features, focused);
  if (!b) return;
  if (b[0] === b[2] && b[1] === b[3]) {
    map.flyTo({ center: [b[0], b[1]], zoom: Math.max(map.getZoom(), 10), duration: 700 });
  } else {
    fitBoundsTo([[b[0], b[1]], [b[2], b[3]]]);
  }
}

export function clearStudyMap() {
  focused = null;
  showStudyFeatures(EMPTY);
}

export const studyMapFocused = () => focused;
