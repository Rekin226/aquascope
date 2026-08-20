// The map: basemap, gauge layers, the catchment overlay, and the view part of
// the URL state. Layer choice (basemaps, imagery, terrain, climate rasters)
// is #232; this module keeps the seams for it (setBasemap, overlay helpers).

import { EMPTY_FC, actions, dbg, escapeHtml, sourceStyle, state, trace } from "./core.js?v=__BUILD__";
import { toFeatureCollection } from "./catalog.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

export let map = null;

// WebGL is what the map actually needs; test it directly instead of blaming it
// for every slow style load (the old code said "WebGL is off" after a 12 s
// timeout, which was often just a slow network).
export function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return Boolean(c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl"));
  } catch {
    return false;
  }
}

const BASEMAP = {
  version: 8,
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  sources: {
    carto: {
      type: "raster", tileSize: 256,
      tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"],
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
    },
  },
  layers: [{ id: "carto", type: "raster", source: "carto" }],
};

export function initMap(initialView) {
  if (!webglAvailable()) {
    trace("map: no webgl");
    return Promise.resolve({ ok: false, reason: "webgl" });
  }
  try {
    map = dbg.map = new maplibregl.Map({
      container: "map",
      style: BASEMAP,
      center: initialView ? [initialView.lon, initialView.lat] : [0, 30],
      zoom: initialView ? initialView.zoom : 1.6,
      attributionControl: { compact: true },
    });
  } catch (err) {
    console.warn("map unavailable:", err && err.message);
    map = null;
    return Promise.resolve({ ok: false, reason: "init" });
  }
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.ScaleControl(), "bottom-left");
  map.addControl(new maplibregl.FullscreenControl(), "top-left");
  map.addControl(new maplibregl.GeolocateControl({ trackUserLocation: false }), "top-left");
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ ok: false, reason: "slow" }), 20000);
    map.once("load", () => { clearTimeout(timer); resolve({ ok: true }); });
    map.once("error", (e) => { console.warn("map error", e && e.error); clearTimeout(timer); resolve({ ok: false, reason: "error" }); });
  });
}

export function addStationLayers(fc) {
  map.addSource("catchment", { type: "geojson", data: EMPTY_FC });
  map.addLayer({ id: "catchment-fill", type: "fill", source: "catchment", paint: { "fill-color": "#1565c0", "fill-opacity": 0.14 } });
  map.addLayer({ id: "catchment-line", type: "line", source: "catchment", paint: { "line-color": "#0d47a1", "line-width": 1.6, "line-dasharray": [2, 1.5] } });
  map.addSource("stations", { type: "geojson", data: fc, cluster: true, clusterMaxZoom: 9, clusterRadius: 38 });
  map.addLayer({
    id: "clusters", type: "circle", source: "stations", filter: ["has", "point_count"],
    paint: {
      "circle-color": "#1565c0", "circle-opacity": 0.7, "circle-stroke-color": "#fff", "circle-stroke-width": 1.5,
      "circle-radius": ["step", ["get", "point_count"], 13, 50, 17, 250, 22, 1000, 28],
    },
  });
  map.addLayer({
    id: "cluster-count", type: "symbol", source: "stations", filter: ["has", "point_count"],
    layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 11, "text-font": ["Open Sans Semibold"] },
    paint: { "text-color": "#fff" },
  });
  map.addLayer({
    id: "points", type: "circle", source: "stations", filter: ["!", ["has", "point_count"]],
    paint: { "circle-color": ["get", "color"], "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 6, 14, 9], "circle-stroke-color": "#fff", "circle-stroke-width": 1 },
  });
  map.addLayer({
    id: "selected", type: "circle", source: "stations", filter: ["==", ["get", "key"], "__none__"],
    paint: { "circle-color": "#ffd600", "circle-radius": 11, "circle-stroke-color": "#212121", "circle-stroke-width": 2 },
  });

  map.on("click", "clusters", async (e) => {
    const f = map.queryRenderedFeatures(e.point, { layers: ["clusters"] })[0];
    const zoom = await map.getSource("stations").getClusterExpansionZoom(f.properties.cluster_id);
    map.easeTo({ center: f.geometry.coordinates, zoom });
  });
  map.on("click", "points", (e) => actions.selectStation(e.features[0].properties.key, { fly: false }));
  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
  map.on("mouseenter", "points", (e) => {
    map.getCanvas().style.cursor = "pointer";
    const p = e.features[0].properties;
    popup.setLngLat(e.features[0].geometry.coordinates)
      .setHTML(`<strong>${escapeHtml(p.name || p.key.split("/")[1])}</strong><br><span class="muted">${escapeHtml(sourceStyle(p.source).label)}</span>`)
      .addTo(map);
  });
  map.on("mouseleave", "points", () => { map.getCanvas().style.cursor = ""; popup.remove(); });
  map.on("click", (e) => {
    const hit = map.queryRenderedFeatures(e.point, { layers: ["points", "clusters"] });
    if (hit.length) return; // handled by the layer handlers
    actions.selectPoint(e.lngLat.lat, e.lngLat.lng);
  });
  map.on("mouseenter", "clusters", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "clusters", () => (map.getCanvas().style.cursor = ""));

  // The view is part of the shareable state; write it back, debounced.
  let t;
  map.on("moveend", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      const c = map.getCenter();
      state.view = { zoom: map.getZoom(), lat: c.lat, lon: c.lng };
      writeUrl({ view: state.view });
    }, 350);
  });
}

export function refreshMapData() {
  if (state.mapOk && map.getSource("stations")) map.getSource("stations").setData(toFeatureCollection(state.stations));
}

export function highlightStation(key) {
  if (state.mapOk) map.setFilter("selected", ["==", ["get", "key"], key || "__none__"]);
}

export function flyToStation(r) {
  if (state.mapOk) map.flyTo({ center: [r.lon, r.lat], zoom: Math.max(map.getZoom(), 9) });
}

export function setPointMarker(lat, lon) {
  if (!state.mapOk) return;
  if (state.marker) state.marker.remove();
  state.marker = new maplibregl.Marker({ color: "#455a64" }).setLngLat([lon, lat]).addTo(map);
}

export function clearPointMarker() {
  if (state.marker) { state.marker.remove(); state.marker = null; }
}

export function setView(view) {
  if (state.mapOk && view) map.jumpTo({ center: [view.lon, view.lat], zoom: view.zoom });
}

export function setCatchmentGeometry(feature) {
  if (!state.mapOk || !map.getSource("catchment")) return;
  map.getSource("catchment").setData(feature ? { type: "FeatureCollection", features: [feature] } : EMPTY_FC);
}

export function fitBoundsTo(bbox) {
  if (state.mapOk) map.fitBounds(bbox, { padding: 40, maxZoom: 11, duration: 700 });
}
