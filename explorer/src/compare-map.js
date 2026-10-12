// Swipe compare (#522): a second map laid over the first, cut by a slider.
// Left of the line is the map as it is; right of it is the same view on another
// date, or with one other dated layer. No plugin: a MapLibre map that follows the
// main one's camera, a CSS clip, and a handle to drag.
//
// The right-hand map draws the basemap, the relief and the dated layers only.
// The gauges stay on the left, where clicking them still works: the overlay
// takes no pointer events except on the handle.

import { $, onTime, state } from "./core.js?v=__BUILD__";
import { TERRAIN_DEM, overlayById, tileUrls } from "./layers.js?v=__BUILD__";
import { map, panelPadding, styleFor } from "./map.js?v=__BUILD__";
import { shortDate } from "./timeline.js?v=__BUILD__";

let cmap = null;          // the right-hand map
let built = null;         // what it was built for: "basemap|date|layers"
let split = null;         // where the line is, as a share of the width (null: the middle of what you can see)

const rightLayers = () => {
  const c = state.compare;
  if (c && c.layer) return [c.layer];
  return [...state.overlays];
};

function sync() {
  if (!cmap || !map) return;
  cmap.jumpTo({
    center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch(),
    padding: map.getPadding ? map.getPadding() : undefined,
  });
}

function clip() {
  const box = $("compare");
  const w = box.clientWidth;
  if (split === null) {   // start in the middle of the map you can see, not under the inspector
    const pad = panelPadding();
    split = w ? Math.min(0.95, Math.max(0.05, (w - pad.right) / 2 / w)) : 0.5;
  }
  const x = Math.round(w * split);
  box.querySelector(".compare-map").style.clipPath = `inset(0 0 0 ${x}px)`;
  const handle = $("compare-handle");
  handle.style.left = `${x}px`;
  $("compare-left").style.left = $("compare-right").style.left = `${x}px`;
  handle.setAttribute("aria-valuenow", String(Math.round(split * 100)));
}

function labels() {
  const c = state.compare;
  if (!c) return;
  const left = [...state.overlays].map((id) => (overlayById(id) || {}).label).filter(Boolean);
  $("compare-left").textContent = `${shortDate(state.date)}${left.length === 1 ? ` · ${left[0]}` : ""}`;
  const right = rightLayers().map((id) => (overlayById(id) || {}).label).filter(Boolean);
  $("compare-right").textContent = `${shortDate(c.date)}${right.length === 1 ? ` · ${right[0]}` : ""}`;
}

// The right map's own layers: relief like the left, and the dated rasters for its date.
function dress() {
  const c = state.compare;
  if (!cmap || !c) return;
  if (state.hillshade || state.terrain) {
    if (!cmap.getSource("terrain-dem")) {
      cmap.addSource("terrain-dem", {
        type: "raster-dem", tiles: TERRAIN_DEM.tiles, tileSize: TERRAIN_DEM.tileSize,
        encoding: TERRAIN_DEM.encoding, maxzoom: TERRAIN_DEM.maxzoom,
      });
    }
    if (state.hillshade && !cmap.getLayer("hillshade")) {
      cmap.addLayer({
        id: "hillshade", type: "hillshade", source: "terrain-dem",
        paint: { "hillshade-exaggeration": 0.38, "hillshade-shadow-color": "#3f5566", "hillshade-accent-color": "#5b7286" },
      });
    }
  }
  for (const id of rightLayers()) {
    const spec = overlayById(id);
    if (!spec || cmap.getSource(`ov-${id}`)) continue;   // a second style.load must not add it twice
    cmap.addSource(`ov-${id}`, {
      type: "raster", tileSize: 256, tiles: tileUrls(spec, c.date),
      minzoom: spec.minzoom || 0, maxzoom: spec.maxzoom || 9,
    });
    cmap.addLayer({
      id: `ov-${id}`, type: "raster", source: `ov-${id}`,
      paint: { "raster-opacity": state.opacity[id] ?? spec.opacity ?? 0.8 },
    });
  }
  if (state.globe && cmap.setProjection) {
    try { cmap.setProjection({ type: "globe" }); } catch { /* flat is fine */ }
  }
  sync();
}

function onResize() {
  if (cmap) { cmap.resize(); clip(); }
}

function build() {
  const c = state.compare;
  const key = `${state.basemap}|${c.date}|${rightLayers().join(",")}|${state.hillshade}|${state.globe}`;
  if (cmap && built === key) return;
  built = key;
  const holder = $("compare").querySelector(".compare-map");
  if (!cmap) {
    cmap = new maplibregl.Map({
      container: holder, style: styleFor(state.basemap, c.date), interactive: false, attributionControl: false,
      center: map.getCenter(), zoom: map.getZoom(),
    });
    cmap.once("style.load", dress);
    map.on("move", sync);
    map.on("resize", onResize);
  } else {
    cmap.setStyle(styleFor(state.basemap, c.date), { diff: false });
    cmap.once("style.load", dress);
  }
}

function open() {
  if (!state.mapOk || !map || typeof maplibregl === "undefined") return;
  $("compare").hidden = false;
  build();
  clip();
  labels();
  cmap.resize();
  sync();
}

function close() {
  $("compare").hidden = true;
  if (cmap) {
    map.off("move", sync);
    map.off("resize", onResize);
    cmap.remove();
    cmap = null;
    built = null;
  }
}

function drag(handle) {
  const box = $("compare");
  const move = (clientX) => {
    const r = box.getBoundingClientRect();
    split = Math.min(0.95, Math.max(0.05, (clientX - r.left) / r.width));
    clip();
  };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const onMove = (ev) => move(ev.clientX);
    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });
  handle.addEventListener("keydown", (e) => {
    const stepBy = e.key === "ArrowLeft" ? -0.05 : e.key === "ArrowRight" ? 0.05 : 0;
    if (!stepBy) return;
    e.preventDefault();
    split = Math.min(0.95, Math.max(0.05, (split ?? 0.5) + stepBy));
    clip();
  });
}

/** Re-dress the right side when the left's basemap, overlays or relief change. */
export function syncCompare() {
  if (!state.compare) { close(); return; }
  open();
}

export function initCompare() {
  if (!$("compare")) return;
  drag($("compare-handle"));
  onTime((t) => {
    if (!t.compare) { close(); return; }
    if (t.date !== t.prev.date && cmap) labels();
    if (!t.prev.compare || t.compare.date !== t.prev.compare.date || t.compare.layer !== t.prev.compare.layer) open();
  });
  if (state.compare) open();
}
