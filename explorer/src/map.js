// The map: basemap, gauge layers, the catchment overlay, and the view part of
// the URL state. Layer choice (basemaps, imagery, terrain, climate rasters)
// is #232; this module keeps the seams for it (setBasemap, overlay helpers).

import { EMPTY_FC, actions, dbg, escapeHtml, sourceStyle, state, trace } from "./core.js?v=__BUILD__";
import { toFeatureCollection } from "./catalog.js?v=__BUILD__";
import { TERRAIN_DEM, basemapById, overlayById, tileUrls } from "./layers.js?v=__BUILD__";
import { SHAPE_NAMES, shapeSdf } from "./shapes.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

export let map = null;

// Set by the app to finish setting the map up if it loads after we gave up.
let onLate = null;
export function whenMapLoadsLate(cb) { onLate = cb; }

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

// Centred on the Atlantic: the one framing that holds North America, Europe and
// west Africa at once, which is where all six agencies are. The old default
// fitted a box from California to Taiwan, which cropped both ends of it.
export const DEFAULT_CENTER = [-38, 28];

// How much of the map is hidden behind the floating cards. MapLibre calls this
// padding, and once it knows about it "centre" means the middle of what you can
// actually see: the globe is not framed half under the inspector, and flying to
// a gauge does not land it beneath the panel either.
export function panelPadding() {
  const pad = { top: 0, right: 0, bottom: 0, left: 0 };
  if (!map) return pad;
  const frame = map.getContainer().getBoundingClientRect();
  if (!frame.width || !frame.height) return pad;
  if (document.body.classList.contains("workbench-mode")) return pad;
  const cards = [document.getElementById("panel"), document.getElementById("drawer")];
  for (const el of cards) {
    if (!el || el.hidden || el.offsetParent === null) continue;
    if (el.id === "panel" && document.body.classList.contains("panel-collapsed")) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    // A card narrower than the map is a side panel; a full-width one is the
    // bottom sheet the narrow layout uses.
    if (r.width < frame.width * 0.75) pad.right = Math.max(pad.right, frame.right - r.left);
    else pad.bottom = Math.max(pad.bottom, frame.bottom - r.top);
  }
  // Never more than a bit under half. Past roughly 52 % of the width the camera
  // centre is far enough off that MapLibre's own scale control cannot unproject
  // on a globe and prints "NaN m" (measured: fine at 750 px of 1,440, broken at
  // 800). With the inspector and the Analyst both open on a 1,440 px window the
  // uncapped figure is 859. The globe then sits partly behind a card, which is
  // what a map under a floating panel is supposed to do.
  pad.right = Math.min(pad.right, frame.width * 0.48);
  pad.bottom = Math.min(pad.bottom, frame.height * 0.48);
  return pad;
}

export function syncMapPadding() {
  if (!state.mapOk || !map) return;
  const next = panelPadding();
  // setPadding moves the camera, so only when it would actually change
  // something: otherwise a resize or a re-render nudges the map for no reason,
  // and can cut a fly-to short.
  const now = map.getPadding ? map.getPadding() : null;
  if (now && ["top", "right", "bottom", "left"].every((k) => Math.abs((now[k] || 0) - next[k]) < 2)) return;
  try { map.setPadding(next, { duration: 0 }); } catch { /* nothing to do */ }
}

// The cards resize for reasons no click tells us about: the workbench widening
// the inspector, a long station name wrapping, the window changing. Watch them
// instead of trying to name every occasion.
export function watchPanelSizes() {
  if (typeof ResizeObserver === "undefined") return;
  let timer = null;
  const ro = new ResizeObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(syncMapPadding, 120);
  });
  for (const id of ["panel", "drawer"]) {
    const el = document.getElementById(id);
    if (el) ro.observe(el);
  }
}

// The globe does not grow as 2^zoom: MapLibre eases it towards Mercator as you
// zoom in, so the sphere widens by about 1.78x per zoom level rather than 2x.
// Measured on MapLibre 5.6 at zooms 0.8 to 2.4 (268, 344, 432, 544, 686 px).
const GLOBE_PX_AT_Z0 = 169;
const GLOBE_GROWTH = Math.log(1.782);

// Zoom so the globe (or the world) fills the part of the map you can see.
// It has to be derived from the container: a hard-coded zoom leaves a marble in
// a sea of white on a monitor and overflows on a phone.
export function fitWorldZoom(container, { globe = true } = {}) {
  const pad = panelPadding();
  const w = Math.max(220, (container ? container.clientWidth : 900) - pad.right);
  const h = Math.max(220, (container ? container.clientHeight : 700) - pad.bottom);
  const zoom = globe
    ? Math.log((Math.min(w, h) * 0.76) / GLOBE_PX_AT_Z0) / GLOBE_GROWTH
    : Math.log2((Math.max(w, Math.min(w, h)) * 1.02) / 256);
  return Math.max(0, Math.min(4, zoom));
}

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

export function initMap(initialView, { basemap = "light", date = null, globe = false } = {}) {
  if (!webglAvailable()) {
    trace("map: no webgl");
    return Promise.resolve({ ok: false, reason: "webgl" });
  }
  try {
    map = dbg.map = new maplibregl.Map({
      container: "map",
      style: styleFor(basemap, date),
      center: initialView ? [initialView.lon, initialView.lat] : DEFAULT_CENTER,
      zoom: initialView ? initialView.zoom : 1.6,
      // Added by hand below, at the bottom left, where the inspector is not.
      attributionControl: false,
    });
  } catch (err) {
    console.warn("map unavailable:", err && err.message);
    map = null;
    return Promise.resolve({ ok: false, reason: "init" });
  }
  appliedBasemap = basemap;
  // Coverage is global and sparse, so the world view is a globe: it drops
  // Mercator's polar exaggeration, and MapLibre flattens it back to Mercator on
  // its own past about zoom 5, where the work happens. It has to be set on the
  // map rather than passed to the constructor, which ignores a `projection`
  // option (getProjection() comes back undefined), and setting it as soon as
  // the style is ready avoids a flat frame before the globe appears.
  // applyLayerState() is what decides state.globe; this only gets the sphere on
  // screen sooner, so it must not write back a failure over the intent.
  if (globe) map.once("style.load", () => setGlobe(true));
  // One column at the top left: our layers and projection buttons first (see
  // .map-tools), then MapLibre's own, which the stylesheet pushes below them.
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.GeolocateControl({ trackUserLocation: false }), "top-left");
  map.addControl(new maplibregl.FullscreenControl(), "top-left");
  // Bottom left, both of them: the inspector floats over the bottom right.
  map.addControl(new maplibregl.ScaleControl(), "bottom-left");
  map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
  return new Promise((resolve) => {
    let settled = false;
    let timer = null;
    const done = (v) => { if (!settled) { settled = true; clearTimeout(timer); resolve(v); } };
    // Through done(), not resolve(): giving up has to record that we gave up,
    // or the late-load branch below cannot tell that it is late.
    timer = setTimeout(() => done({ ok: false, reason: "slow" }), 20000);
    map.once("load", () => {
      // The twenty seconds are a promise to the reader, not a verdict on the
      // map: a cold cache with Pyodide's 15 MB downloading alongside it can
      // take longer, and a tab that was in the background renders nothing at
      // all until it is looked at, so `load` can arrive minutes late. It used
      // to be terminal, which left a working map with no gauges on it under a
      // notice saying it had failed, recoverable only by reloading.
      if (settled) { const late = onLate; onLate = null; if (late) late(); return; }
      done({ ok: true });
    });
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
  const globeOn = Boolean(state.globe);
  appliedBasemap = id;
  map.setStyle(styleFor(id, date), { diff: false, transformStyle: carryOver });
  // The new style is not usable until it has loaded; terrain in particular is a
  // style property and has to be set again afterwards.
  map.once("idle", () => {
    if (terrainOn && map.getSource("terrain-dem")) {
      map.setTerrain({ source: "terrain-dem", exaggeration: 1.3 });
    }
    // setStyle replaces the projection and the image table along with the rest.
    ensureShapeImages();
    if (globeOn) setGlobe(true);
    if (then) then();
  });
}

// One SDF image per shape, registered on the current style. setStyle throws the
// image table away with everything else, so this runs again after a basemap
// change; the shapes themselves are computed once and reused.
const sdfCache = new Map();

export function ensureShapeImages() {
  if (!map) return;
  for (const name of SHAPE_NAMES) {
    const id = `gauge-${name}`;
    if (map.hasImage && map.hasImage(id)) continue;
    if (!sdfCache.has(name)) sdfCache.set(name, shapeSdf(name));
    try { map.addImage(id, sdfCache.get(name), { sdf: true, pixelRatio: 2 }); } catch { /* already there */ }
  }
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
    paint: { "hillshade-exaggeration": 0.38, "hillshade-shadow-color": "#3f5566", "hillshade-accent-color": "#5b7286" },
  }, firstDataLayerId());
}

export function setGlobe(on) {
  // `map`, not `state.mapOk`: the projection can be set as soon as the map
  // object exists, and mapOk is only true once the app has finished booting it.
  if (!map || !map.setProjection) return false;
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
  map.setPaintProperty("points", "icon-color", ["get", COLOR_FIELD[mode] || "color"]);
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
  map.addLayer({ id: "catchment-fill", type: "fill", source: "catchment", paint: { "fill-color": "#0b6bb8", "fill-opacity": 0.15 } });
  map.addLayer({ id: "catchment-line", type: "line", source: "catchment", paint: { "line-color": "#075390", "line-width": 1.8, "line-dasharray": [2, 1.5] } });
  // clusterRadius 38 with clusterMaxZoom 9 tiled the United States with
  // touching bubbles right down to state level. Wider grouping makes fewer,
  // separated circles, and handing over to real dots at zoom 6 means you are
  // looking at gauges as soon as you are looking at a river basin.
  map.addSource("stations", { type: "geojson", data: fc, cluster: true, clusterMaxZoom: 6, clusterRadius: 92 });
  map.addLayer({
    id: "clusters", type: "circle", source: "stations", filter: ["has", "point_count"],
    paint: {
      "circle-color": ["interpolate", ["linear"], ["get", "point_count"],
        1, "#4d9fe0", 100, "#2b7fc4", 1000, "#125ea3", 10000, "#0b3f76"],
      "circle-opacity": 0.92,
      "circle-stroke-color": "rgba(255,255,255,.85)",
      "circle-stroke-width": ["step", ["get", "point_count"], 1.5, 250, 2],
      "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "point_count"]],
        1, 11, 10, 17, 40, 23, 130, 29],
    },
  });
  map.addLayer({
    id: "cluster-count", type: "symbol", source: "stations", filter: ["has", "point_count"],
    layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": ["step", ["get", "point_count"], 11, 250, 12.5], "text-font": LABEL_FONT },
    paint: { "text-color": "#fff", "text-halo-color": "rgba(10,45,80,.35)", "text-halo-width": 0.6 },
  });
  // A symbol layer rather than a circle, so which agency a gauge belongs to is
  // carried by outline as well as by hue (#283). SDF icons keep the colour
  // data-driven, so "colour by record length" still works and the shape goes on
  // saying the source underneath. allow-overlap because every gauge counts:
  // symbol layers hide colliding icons by default, and a circle layer does not.
  ensureShapeImages();
  map.addLayer({
    id: "points", type: "symbol", source: "stations", filter: ["!", ["has", "point_count"]],
    layout: {
      "icon-image": ["concat", "gauge-", ["get", "shape"]],
      // The icon is 24 px of shape in a 40 px field at pixelRatio 2, so 12 CSS
      // px natural; these factors put it back on the diameters the circle layer
      // used (6.8 px at z4 up to 18 px at z14).
      "icon-size": ["interpolate", ["linear"], ["zoom"], 4, 0.57, 7, 0.77, 10, 1.08, 14, 1.5],
      "icon-allow-overlap": true,
      "icon-ignore-placement": true,
    },
    paint: {
      "icon-color": ["get", "color"],
      "icon-halo-color": "rgba(255,255,255,.92)",
      "icon-halo-width": ["interpolate", ["linear"], ["zoom"], 4, 1, 10, 1.6],
      "icon-opacity": 0.98,
    },
  });
  map.addLayer({
    id: "selected", type: "circle", source: "stations", filter: ["==", ["get", "key"], "__none__"],
    paint: {
      "circle-color": "#ffc400", "circle-radius": 10,
      "circle-stroke-color": "#10222f", "circle-stroke-width": 2.5, "circle-opacity": 1,
    },
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

// A gauge gets flown to; a point never was, so "the climate of Taipei" and any
// #p= link dropped a marker somewhere off screen and left you on the world view.
export function flyToPoint(lat, lon, { zoom = 8 } = {}) {
  if (state.mapOk) map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), zoom) });
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
