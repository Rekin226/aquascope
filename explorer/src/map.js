// The map: basemap, gauge layers, the catchment overlay, and the view part of
// the URL state. Layer choice (basemaps, imagery, terrain, climate rasters)
// is #232; this module keeps the seams for it (setBasemap, overlay helpers).

import { EMPTY_FC, actions, dbg, escapeHtml, sourceStyle, state, trace } from "./core.js?v=__BUILD__";
import { toFeatureCollection } from "./catalog.js?v=__BUILD__";
import { TERRAIN_DEM, basemapById, overlayById, tileUrls } from "./layers.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

export let map = null;

// Our own sources and layers, which must survive a basemap change: setStyle
// replaces the whole style, so they are carried across explicitly.
const OUR_SOURCES = ["stations", "catchment", "basins6", "basins12", "terrain-dem"];
const OUR_LAYERS = ["catchment-fill", "catchment-line", "basins6-line", "basins12-line", "basins12-up",
  "gauge-heat", "clusters", "cluster-count", "points", "selected", "hillshade"];
const isOurs = (id) => OUR_SOURCES.includes(id) || id.startsWith("ov-");
const isOurLayer = (id) => OUR_LAYERS.includes(id) || id.startsWith("ov-");

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

// The same glyph server the vector basemaps use, so the cluster labels have a
// font whichever basemap is on (OpenFreeMap serves Noto Sans; the MapLibre
// demo tiles served Open Sans Semibold, which 404s there).
const GLYPHS = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";
const LABEL_FONT = ["Noto Sans Bold"];

// A raster basemap as a whole style, so switching basemaps is always the same
// operation whether the target is a vector style URL or a tile template.
function rasterStyle(b, date) {
  return {
    version: 8,
    glyphs: GLYPHS,
    sources: {
      base: {
        type: "raster", tileSize: 256, tiles: tileUrls(b, date),
        maxzoom: b.maxzoom || 19, attribution: b.attribution,
      },
    },
    layers: [{ id: "base", type: "raster", source: "base" }],
  };
}

// The last-resort basemap if the chosen one will not load at all.
const FALLBACK_STYLE = {
  version: 8,
  glyphs: GLYPHS,
  sources: {
    base: {
      type: "raster", tileSize: 256,
      tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"],
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
    },
  },
  layers: [{ id: "base", type: "raster", source: "base" }],
};

function styleFor(basemapId, date) {
  const b = basemapById(basemapId);
  return b.kind === "style" ? b.url : rasterStyle(b, date);
}

export function initMap(initialView, { basemap = "light", date = null } = {}) {
  if (!webglAvailable()) {
    trace("map: no webgl");
    return Promise.resolve({ ok: false, reason: "webgl" });
  }
  try {
    map = dbg.map = new maplibregl.Map({
      container: "map",
      style: styleFor(basemap, date),
      center: initialView ? [initialView.lon, initialView.lat] : [0, 30],
      zoom: initialView ? initialView.zoom : 1.6,
      attributionControl: { compact: true },
    });
  } catch (err) {
    console.warn("map unavailable:", err && err.message);
    map = null;
    return Promise.resolve({ ok: false, reason: "init" });
  }
  appliedBasemap = basemap;
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.ScaleControl(), "bottom-left");
  map.addControl(new maplibregl.FullscreenControl(), "top-left");
  map.addControl(new maplibregl.GeolocateControl({ trackUserLocation: false }), "top-left");
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ ok: false, reason: "slow" }), 20000);
    let settled = false;
    const done = (v) => { if (!settled) { settled = true; clearTimeout(timer); resolve(v); } };
    map.once("load", () => done({ ok: true }));
    map.once("error", (e) => {
      // A basemap host can be down (none of these have an SLA). Fall back to a
      // raster basemap once rather than losing the whole map.
      console.warn("map error", e && e.error);
      if (!settled && !map.__fellBack) {
        map.__fellBack = true;
        trace("basemap failed, falling back");
        try {
          map.setStyle(FALLBACK_STYLE, { diff: false });
          map.once("load", () => done({ ok: true, fellBack: true }));
          return;
        } catch { /* fall through */ }
      }
      done({ ok: false, reason: "error" });
    });
  });
}

// ── basemap, overlays, terrain ──────────────────────────────────────────────

// setStyle throws our sources and layers away, so carry them over. MapLibre's
// transformStyle hook hands us the old and the new style to merge.
function carryOver(previous, next) {
  if (!previous) return next;
  const sources = { ...next.sources };
  for (const [id, src] of Object.entries(previous.sources || {})) {
    if (isOurs(id)) sources[id] = src;
  }
  const kept = (previous.layers || []).filter((l) => isOurLayer(l.id));
  // Overlays sit under the gauges, which are the point of the page.
  const overlays = kept.filter((l) => l.id.startsWith("ov-") || l.id === "hillshade");
  const data = kept.filter((l) => !overlays.includes(l));
  return { ...next, sources, layers: [...(next.layers || []), ...overlays, ...data] };
}

// Which basemap the style currently is, so callers can skip a swap they do not
// need (adding layers while setStyle is in flight loses them).
let appliedBasemap = null;
export const currentBasemap = () => appliedBasemap;

export function setBasemap(id, { date = null, then = null } = {}) {
  if (!state.mapOk) return;
  const terrainOn = Boolean(map.getTerrain && map.getTerrain());
  appliedBasemap = id;
  map.setStyle(styleFor(id, date), { diff: false, transformStyle: carryOver });
  // The new style is not usable until it has loaded; terrain in particular is a
  // style property and has to be set again afterwards.
  map.once("idle", () => {
    if (terrainOn && map.getSource("terrain-dem")) {
      map.setTerrain({ source: "terrain-dem", exaggeration: 1.3 });
    }
    if (then) then();
  });
}

function firstDataLayerId() {
  for (const id of ["gauge-heat", "clusters", "points", "catchment-fill"]) {
    if (map.getLayer(id)) return id;
  }
  return undefined;
}

export function setOverlay(id, on, { date = null, opacity = null } = {}) {
  if (!state.mapOk) return;
  const layerId = `ov-${id}`;
  const spec = overlayById(id);
  if (!spec) return;
  if (!on) {
    if (map.getLayer(layerId)) map.removeLayer(layerId);
    if (map.getSource(layerId)) map.removeSource(layerId);
    return;
  }
  if (map.getLayer(layerId)) return;
  map.addSource(layerId, {
    type: "raster", tileSize: 256, tiles: tileUrls(spec, date),
    minzoom: spec.minzoom || 0, maxzoom: spec.maxzoom || 9, attribution: spec.attribution,
  });
  map.addLayer({
    id: layerId, type: "raster", source: layerId,
    paint: { "raster-opacity": opacity === null ? (spec.opacity ?? 0.8) : opacity },
  }, firstDataLayerId());
}

export function setOverlayOpacity(id, opacity) {
  const layerId = `ov-${id}`;
  if (state.mapOk && map.getLayer(layerId)) map.setPaintProperty(layerId, "raster-opacity", opacity);
}

// Move every time-driven layer (and the daily basemap) to a new date.
export function applyDate(date, activeOverlays, basemapId) {
  if (!state.mapOk) return;
  for (const id of activeOverlays || []) {
    const spec = overlayById(id);
    const src = map.getSource(`ov-${id}`);
    if (spec && spec.time && src && src.setTiles) src.setTiles(tileUrls(spec, date));
  }
  const b = basemapById(basemapId);
  if (b && b.time) {
    const src = map.getSource("base");
    if (src && src.setTiles) src.setTiles(tileUrls(b, date));
  }
}

export function ensureTerrainSource() {
  if (!state.mapOk || map.getSource("terrain-dem")) return;
  map.addSource("terrain-dem", {
    type: "raster-dem", tiles: TERRAIN_DEM.tiles, tileSize: TERRAIN_DEM.tileSize,
    encoding: TERRAIN_DEM.encoding, maxzoom: TERRAIN_DEM.maxzoom, attribution: TERRAIN_DEM.attribution,
  });
}

export function setTerrain(on) {
  if (!state.mapOk) return;
  if (!on) { map.setTerrain(null); return; }
  ensureTerrainSource();
  map.setTerrain({ source: "terrain-dem", exaggeration: 1.3 });
  // Tilt so the relief is visible. setPitch rather than an eased camera move:
  // restoring a view from the URL jumps the camera and would cancel an
  // animation mid-flight, leaving 3D terrain looking flat.
  if (map.getPitch() < 20) map.setPitch(55);
}

export function setHillshade(on) {
  if (!state.mapOk) return;
  if (!on) { if (map.getLayer("hillshade")) map.removeLayer("hillshade"); return; }
  ensureTerrainSource();
  if (map.getLayer("hillshade")) return;
  map.addLayer({
    id: "hillshade", type: "hillshade", source: "terrain-dem",
    paint: { "hillshade-exaggeration": 0.5, "hillshade-shadow-color": "#4a4a4a" },
  }, firstDataLayerId());
}

export function setGlobe(on) {
  if (!state.mapOk || !map.setProjection) return false;
  try {
    map.setProjection({ type: on ? "globe" : "mercator" });
    return true;
  } catch (err) {
    console.info("globe projection unavailable:", err && err.message);
    return false;
  }
}

export function globeSupported() {
  return Boolean(state.mapOk && map && map.setProjection);
}

// ── how the gauges are drawn ────────────────────────────────────────────────

const COLOR_FIELD = { source: "color", record: "colorRecord", recent: "colorRecent" };

export function setGaugeStyle(mode) {
  if (!state.mapOk || !map.getLayer("points")) return;
  map.setPaintProperty("points", "circle-color", ["get", COLOR_FIELD[mode] || "color"]);
}

export function setHeatmap(on) {
  if (!state.mapOk) return;
  if (!on) { if (map.getLayer("gauge-heat")) map.removeLayer("gauge-heat"); return; }
  if (map.getLayer("gauge-heat")) return;
  map.addLayer({
    id: "gauge-heat", type: "heatmap", source: "stations", maxzoom: 11,
    paint: {
      "heatmap-weight": 1,
      "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.6, 9, 2.2],
      "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 6, 4, 14, 9, 28],
      "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0.85, 11, 0],
      "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
        0, "rgba(0,0,0,0)", 0.2, "#2c7fb8", 0.45, "#41b6c4", 0.65, "#a1dab4", 0.85, "#ffffcc", 1, "#fff7bc"],
    },
  }, map.getLayer("clusters") ? "clusters" : undefined);
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
    layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 11, "text-font": LABEL_FONT },
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

// ── select an area ──────────────────────────────────────────────────────────
// Drag a box over the map and get the gauges inside it. No drawing library:
// one div and three listeners, which is all a rectangle needs.

let selecting = false;

export function areaSelectActive() { return selecting; }

export function startAreaSelect(onDone) {
  if (!state.mapOk || selecting) return;
  selecting = true;
  const canvas = map.getCanvasContainer();
  const box = document.createElement("div");
  box.className = "area-box";
  box.hidden = true;
  canvas.appendChild(box);
  map.dragPan.disable();
  map.getCanvas().style.cursor = "crosshair";
  let start = null;

  const point = (e) => {
    const rect = map.getCanvas().getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return { x: src.clientX - rect.left, y: src.clientY - rect.top };
  };
  const draw = (a, b) => {
    box.hidden = false;
    box.style.left = `${Math.min(a.x, b.x)}px`;
    box.style.top = `${Math.min(a.y, b.y)}px`;
    box.style.width = `${Math.abs(a.x - b.x)}px`;
    box.style.height = `${Math.abs(a.y - b.y)}px`;
  };
  const onDown = (e) => { start = point(e); draw(start, start); };
  const onMove = (e) => { if (start) draw(start, point(e)); };
  const onUp = (e) => {
    if (!start) return;
    // Keep the origin: finish() clears `start` as part of tearing the drag down.
    const from = start;
    const end = point(e);
    finish();
    const sw = map.unproject([Math.min(from.x, end.x), Math.max(from.y, end.y)]);
    const ne = map.unproject([Math.max(from.x, end.x), Math.min(from.y, end.y)]);
    const tiny = Math.abs(from.x - end.x) < 6 && Math.abs(from.y - end.y) < 6;
    onDone(tiny ? null : { west: sw.lng, south: sw.lat, east: ne.lng, north: ne.lat });
  };
  const onKey = (e) => { if (e.key === "Escape") { finish(); onDone(null); } };

  function finish() {
    selecting = false;
    start = null;
    box.remove();
    map.dragPan.enable();
    map.getCanvas().style.cursor = "";
    canvas.removeEventListener("mousedown", onDown);
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    window.removeEventListener("keydown", onKey);
  }

  canvas.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  window.addEventListener("keydown", onKey);
}
