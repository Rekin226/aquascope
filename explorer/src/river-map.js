// Rivers on the map (#516, #545): the GEOGLOWS v2 stream network, on by
// default and read in place from streams.pmtiles, with the water's direction
// drawn as a moving dash; a click lights up what drains to the reach and its
// way to the sea (feature-state on riverId); the trace to the sea grows from
// the click to the outlet, with the dams on it as small squares.

import { $, EMPTY_FC, state } from "./core.js?v=__BUILD__";
import { currentBasemap, ensureShapeImages, fitBoundsTo, map, waitingForIdle } from "./map.js?v=__BUILD__";
import {
  FLOW_FPS, FLOW_STEPS, RIVERS_ATTRIBUTION, STREAMS_PMTILES, cumulativeKm, damsGeoJSON, flowDash, flowOpacity,
  highlightColor, highlightOpacity, highlightWidth, lineBounds, lineUpTo, networkStates, networkSummary, riverOpacity,
  riverTheme, riverWidth,
} from "./river-core.js?v=__BUILD__";

let added = false;
let animation = 0;
let hooked = false;

const NET = { source: "river-net", sourceLayer: "streams" };
// The plain network, the moving dash, then the lit network: all one source and one layout, so MapLibre
// cuts each tile once for the lot. Butt caps, because a round cap turns a zero-length dash into a dot.
const LAYOUT = { "line-cap": "butt", "line-join": "round" };
// No transitions on the dash layer, for any of its properties: a paint change restarts MapLibre's default
// 300 ms transition on every transitionable property of the layer, not only the one changed. (The style's sky
// and projection transitions restart too and cannot be turned off per layer; map.js waitingForIdle covers them.)
const FLOW_STILL = Object.fromEntries(["color", "opacity", "width", "gap-width", "offset", "blur", "translate", "dasharray"]
  .map((p) => [`line-${p}-transition`, { duration: 0, delay: 0 }]));

const reducedMotion = () => Boolean(globalThis.matchMedia && globalThis.matchMedia("(prefers-reduced-motion: reduce)").matches);
// The flow animation is on unless the reader asked for less motion; the layer menu turns it either way.
if (state.flowOn === undefined) state.flowOn = !reducedMotion();

// Under Floods ahead (river-fa-*) and the gauges: the order on the map is basemap, river status,
// Floods past, rivers, Floods ahead, gauges (#543).
function beforeGauges() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-fa-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function theme() { return riverTheme(currentBasemap() || state.basemap); }

export function ensureRiverLayers() {
  if (!state.mapOk || !map) return false;
  if (added && map.getSource("river-trace") && map.getSource("river-dams")) return true;
  try {
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const before = beforeGauges();
    const th = theme();
    if (globalThis.pmtiles && !map.getSource("river-net")) {
      // promoteId: the tiles carry riverId as a property, and feature-state needs it as the feature's id.
      // attribution: the network is on every visitor's screen now, so its CC BY-SA credit sits in the map's own
      // attribution line, not only in the layer rail's credits.
      map.addSource("river-net", { type: "vector", url: `pmtiles://${STREAMS_PMTILES}`, promoteId: "riverId",
        attribution: RIVERS_ATTRIBUTION });
      const line = (id, paint) => map.addLayer({
        id, type: "line", source: "river-net", "source-layer": "streams",
        layout: { ...LAYOUT, visibility: "none" }, paint,
      }, before);
      line("river-net-line", { "line-color": th.line, "line-opacity": riverOpacity(), "line-width": riverWidth() });
      line("river-flow", { "line-color": th.flow, "line-opacity": flowOpacity(), "line-width": riverWidth(),
        "line-dasharray": flowDash(0), ...FLOW_STILL });
      line("river-hl-casing", { "line-color": th.casing, "line-opacity": highlightOpacity(0.85),
        "line-width": highlightWidth(2) });
      line("river-hl-line", { "line-color": highlightColor(th), "line-opacity": highlightOpacity(1),
        "line-width": highlightWidth() });
    }
    if (!map.getSource("river-trace")) {
      map.addSource("river-trace", { type: "geojson", data: EMPTY_FC });
      map.addLayer({ id: "river-trace-casing", type: "line", source: "river-trace",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": th.casing, "line-width": 6, "line-opacity": 0.9 } }, before);
      map.addLayer({ id: "river-trace-line", type: "line", source: "river-trace",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": th.down, "line-width": 3 } }, before);
    }
    if (!map.getSource("river-here")) {
      // The reach the click (or the gauge) snapped to: a ring on the river, over the lit lines.
      map.addSource("river-here", { type: "geojson", data: EMPTY_FC });
      map.addLayer({ id: "river-here-ring", type: "circle", source: "river-here",
        paint: { "circle-radius": 7, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2.5,
          "circle-stroke-color": th.down } }, before);
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
    // Relief shades the land under the rivers, not the rivers (after a basemap change map.js puts it there too).
    if (map.getLayer("hillshade") && map.getLayer("river-net-line")) map.moveLayer("hillshade", "river-net-line");
    hookMap();
    added = true;
    syncVisibility();
    return true;
  } catch (err) {
    console.info("river layers unavailable:", err && err.message);
    return false;
  }
}

// A basemap change rebuilds the style: the colours follow the new basemap and the lit network is set again.
function hookMap() {
  if (hooked) return;
  hooked = true;
  map.on("style.load", () => {
    if (!map.getLayer("river-net-line")) return;
    restyle();
    applyStates();
    syncVisibility();
  });
  document.addEventListener("visibilitychange", syncFlow);
  const mq = globalThis.matchMedia && globalThis.matchMedia("(prefers-reduced-motion: reduce)");
  if (mq && mq.addEventListener) mq.addEventListener("change", () => { setFlowOn(!mq.matches); });
}

function restyle() {
  const th = theme();
  const paint = (id, prop, v) => { if (map.getLayer(id)) map.setPaintProperty(id, prop, v); };
  paint("river-net-line", "line-color", th.line);
  paint("river-flow", "line-color", th.flow);
  for (const [prop, v] of Object.entries(FLOW_STILL)) paint("river-flow", prop, v);
  paint("river-hl-casing", "line-color", th.casing);
  paint("river-hl-line", "line-color", highlightColor(th));
  paint("river-trace-casing", "line-color", th.casing);
  paint("river-trace-line", "line-color", th.down);
  paint("river-here-ring", "circle-stroke-color", th.down);
}

function syncVisibility() {
  if (!map || !map.getLayer("river-net-line")) return;
  const vis = (id, on) => { if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none"); };
  vis("river-net-line", state.riversOn);
  vis("river-flow", state.riversOn && state.flowOn);
  // A lit network stays when the plain one is off: it is the answer to a click.
  vis("river-hl-casing", state.riversOn || lit.size > 0);
  vis("river-hl-line", state.riversOn || lit.size > 0);
  syncFlow();
}

export function setRiversVisible(on) {
  state.riversOn = Boolean(on);
  const toggle = $("toggle-rivers");
  if (toggle) toggle.checked = state.riversOn;
  if (!ensureRiverLayers() || !map.getLayer("river-net-line")) return;
  syncVisibility();
}

// ── the flow animation ───────────────────────────────────────────────────────

let flowFrame = 0;
let flowStep = 0;
let flowLast = 0;

export function setFlowOn(on) {
  state.flowOn = Boolean(on);
  const toggle = $("toggle-flow");
  if (toggle) toggle.checked = state.flowOn;
  syncVisibility();
}

// Runs only while it can be seen: rivers on, flow on, the tab in front. A steady FLOW_FPS whatever the
// screen's refresh rate, so a 120 Hz laptop does not draw the map twice as often for the same motion.
function syncFlow() {
  const run = Boolean(state.mapOk && map && map.getLayer("river-flow") && state.riversOn && state.flowOn
    && !document.hidden);
  if (!run) {
    if (flowFrame) cancelAnimationFrame(flowFrame);
    flowFrame = 0;
    return;
  }
  if (flowFrame) return;
  const tick = (now) => {
    flowFrame = requestAnimationFrame(tick);
    // The dash holds still while anything waits for the map to settle (map.js waitingForIdle): the basemap swap,
    // the time bar's play, a GIF frame. Each step is a style change, and the map is never idle while they come.
    if (now - flowLast < 1000 / FLOW_FPS || waitingForIdle()) return;
    flowLast = now;
    flowStep = (flowStep + 1) % FLOW_STEPS;
    try { map.setPaintProperty("river-flow", "line-dasharray", flowDash(flowStep)); } catch { /* style swapping */ }
  };
  flowFrame = requestAnimationFrame(tick);
}

// ── the lit network: what drains here and the way to the sea ────────────────

let lit = new Map();     // riverId -> HL role, as set on the map
let litFor = null;       // the reach it belongs to
let hereFeature = null;  // the ring

function applyStates() {
  if (!map || !map.getSource("river-net")) return;
  if (map.getLayer("river-net-line")) map.setPaintProperty("river-net-line", "line-opacity", riverOpacity(lit.size > 1 ? 0.5 : 1));
  try { map.removeFeatureState(NET); } catch { /* nothing set yet */ }
  for (const [id, hl] of lit) map.setFeatureState({ ...NET, id }, { hl });
  if (map.getSource("river-here")) {
    map.getSource("river-here").setData(hereFeature ? { type: "FeatureCollection", features: [hereFeature] } : EMPTY_FC);
  }
}

export function clearRiverNetwork() {
  litFor = null;
  lit = new Map();
  hereFeature = null;
  if (state.mapOk && map) applyStates();
  showKey(null);
  if (state.mapOk && map) syncVisibility();
}

// Mark the reach at once (the ring and the reach itself), then light the rest when Python has walked the
// routing tables. `load` returns a promise of { upstream, downstream } from aquascope.rivers; `reach` is
// { river_id, lat, lon }. A newer reach wins: an answer for an older click is dropped.
export function lightRiverNetwork(reach, load) {
  if (!reach || reach.river_id === null || reach.river_id === undefined || !ensureRiverLayers()) return;
  const rid = Number(reach.river_id);
  litFor = rid;
  lit = networkStates(rid);
  hereFeature = Number.isFinite(reach.lat) && Number.isFinite(reach.lon)
    ? { type: "Feature", properties: { river_id: rid }, geometry: { type: "Point", coordinates: [reach.lon, reach.lat] } }
    : null;
  applyStates();
  syncVisibility();
  showKey({ loading: true });
  Promise.resolve().then(load).then((net) => {
    if (litFor !== rid || !net) return;
    lit = networkStates(rid, net.upstream && net.upstream.ids, net.downstream && net.downstream.ids);
    applyStates();
    showKey({ net });
  }).catch((err) => {
    if (litFor !== rid) return;
    console.info("river network:", err && err.message);
    showKey({ error: "Could not read this basin's routing tables just now." });
  });
}

// The key on the map: while the routing table loads, a progress line; then what the colours mean, in a few
// words, with where the network comes from; a close button clears it.
function showKey(what) {
  const el = $("river-key");
  if (!el) return;
  if (!what) { el.hidden = true; el.replaceChildren(); return; }
  const th = theme();
  const close = `<button type="button" class="river-key-x" aria-label="Clear the lit river" title="Clear">×</button>`;
  if (what.loading) {
    const first = state.workerReady ? "" : "Starting the engine, then ";
    el.innerHTML = `<span class="spinner" aria-hidden="true"></span>` +
      `<span>${first}${first ? "finding" : "Finding"} what drains here and the way to the sea…</span>${close}`;
  } else if (what.error) {
    el.innerHTML = `<span>${what.error}</span>${close}`;
  } else {
    const s = networkSummary(what.net);
    el.innerHTML =
      `<span class="rk-row"><i class="rk-sw" style="background:${th.up}"></i><b>Drains here</b> <span class="muted">${s.up}</span></span>` +
      `<span class="rk-row"><i class="rk-sw" style="background:${th.down}"></i><b>To the sea</b> <span class="muted">${s.down}</span></span>` +
      `<span class="rk-src muted">GEOGLOWS v2 routing, modelled${s.cut ? `; ${s.cut}` : ""}</span>${close}`;
  }
  el.querySelector(".river-key-x").addEventListener("click", () => clearRiverNetwork());
  el.hidden = false;
}

// ── the trace to the sea ─────────────────────────────────────────────────────

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
  if (reducedMotion()) { set(coords); return; }
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
