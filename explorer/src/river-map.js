// Rivers on the map (#516): the GEOGLOWS v2 stream network as a toggleable
// vector layer, read in place from streams.pmtiles, and the trace to the sea
// drawn as a line that grows from the click to the outlet, with the dams on
// it as small squares.

import { $, EMPTY_FC, state } from "./core.js?v=__BUILD__";
import { ensureShapeImages, fitBoundsTo, map } from "./map.js?v=__BUILD__";
import {
  RIVERS_MINZOOM, STREAMS_PMTILES, cumulativeKm, damsGeoJSON, lineBounds, lineUpTo, riverWidth,
} from "./river-core.js?v=__BUILD__";

let added = false;
let animation = 0;

// Under Floods ahead (river-fa-*) and the gauges: the order on the map is basemap, river status,
// Floods past, rivers, Floods ahead, gauges (#543).
function beforeGauges() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-fa-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

export function ensureRiverLayers() {
  if (!state.mapOk || !map) return false;
  if (added && map.getSource("river-trace") && map.getSource("river-dams")) return true;
  try {
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const before = beforeGauges();
    if (globalThis.pmtiles && !map.getSource("river-net")) {
      map.addSource("river-net", { type: "vector", url: `pmtiles://${STREAMS_PMTILES}` });
      map.addLayer({
        id: "river-net-line", type: "line", source: "river-net", "source-layer": "streams", minzoom: RIVERS_MINZOOM,
        layout: { visibility: state.riversOn ? "visible" : "none", "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#1e88e5", "line-opacity": 0.7, "line-width": riverWidth() },
      }, before);
    }
    if (!map.getSource("river-trace")) {
      map.addSource("river-trace", { type: "geojson", data: EMPTY_FC });
      map.addLayer({ id: "river-trace-casing", type: "line", source: "river-trace",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#ffffff", "line-width": 6, "line-opacity": 0.9 } }, before);
      map.addLayer({ id: "river-trace-line", type: "line", source: "river-trace",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#0d47a1", "line-width": 3 } }, before);
    }
    if (!map.getSource("river-dams")) {
      // The square of the gauge shapes, in a dark brown with a white halo: a structure on the line, not a gauge.
      ensureShapeImages();
      map.addSource("river-dams", { type: "geojson", data: EMPTY_FC });
      map.addLayer({ id: "river-dams-icon", type: "symbol", source: "river-dams",
        layout: { "icon-image": "gauge-square", "icon-size": ["interpolate", ["linear"], ["zoom"], 4, 0.55, 10, 0.9],
          "icon-allow-overlap": true },
        paint: { "icon-color": "#5d4037", "icon-halo-color": "#ffffff", "icon-halo-width": 1.5 } }, before);
    }
    added = true;
    return true;
  } catch (err) {
    console.info("river layers unavailable:", err && err.message);
    return false;
  }
}

export function setRiversVisible(on) {
  state.riversOn = Boolean(on);
  const toggle = $("toggle-rivers");
  if (toggle) toggle.checked = state.riversOn;
  if (!ensureRiverLayers() || !map.getLayer("river-net-line")) return;
  map.setLayoutProperty("river-net-line", "visibility", state.riversOn ? "visible" : "none");
}

export function clearRiverTrace() {
  animation++;
  if (state.mapOk && map && map.getSource("river-trace")) map.getSource("river-trace").setData(EMPTY_FC);
  if (state.mapOk && map && map.getSource("river-dams")) map.getSource("river-dams").setData(EMPTY_FC);
}

// The dams on the traced path, drawn once the line is there.
export function drawRiverDams(dams) {
  if (!ensureRiverLayers() || !map.getSource("river-dams")) return;
  map.getSource("river-dams").setData(damsGeoJSON(dams));
}

// Draw the path, growing from the start to the outlet over about two seconds
// (at once when the reader prefers less motion), then frame it.
export function drawRiverTrace(coords) {
  if (!coords || coords.length < 2 || !ensureRiverLayers()) return;
  const my = ++animation;
  const src = map.getSource("river-trace");
  const set = (c) => src.setData({ type: "FeatureCollection", features: [
    { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: c } }] });
  const box = lineBounds(coords);
  if (box) fitBoundsTo(box);
  const still = globalThis.matchMedia && globalThis.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (still) { set(coords); return; }
  const cum = cumulativeKm(coords);
  const ms = 2200;
  const t0 = performance.now();
  const step = (now) => {
    if (my !== animation) return;
    const f = Math.min(1, (now - t0) / ms);
    const part = lineUpTo(coords, f, cum);
    if (part.length >= 2) set(part);
    if (f < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
