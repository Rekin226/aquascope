// The flood views of a model result on the globe (#619, #620): what RAS Mapper and Crayfish show, drawn on the map.
//
// * Smooth depth: the depth interpolated between cell centres instead of square cells, with depth contour lines
//   at a round interval (labelled) and the edge of the water, redrawn at every step as the time bar plays.
// * 3D: the map's terrain on, the camera tilted, and the water draped over the relief.
// * Lenses, one at a time from the model's legend row: the flood hazard H1 to H6 (AIDR Guideline 7-3, from depth
//   and speed at every step), the arrival time (with lines of equal arrival), the largest depth and the largest
//   speed, each with its key and its words in the map card.
//
// Every number is the package's: aquascope.mesh_views runs in the mesh worker beside the dropped file (which
// never leaves the browser), builds the lenses and the grid once, then one smooth picture and its lines per step.
// This module draws them; mesh-views-core.js decides the colours and the words.

import { escapeHtml, state } from "./core.js?v=__BUILD__";
import { map, setHillshade, setTerrain } from "./map.js?v=__BUILD__";
import { renderCredits, syncRailControls } from "./layer-ui.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { MESH_LAYERS, currentMesh, meshAsk, meshState, onMeshCleared, refreshMeshLegend, repaintMesh, setMeshViews } from "./mesh-view.js?v=__BUILD__";
import {
  HAZARD_COLOURS, HAZARD_SHORT, areaWords, hazardPacked, hoursWords, lensFacts, lensLine, lensScale, lensStops, lensTicks,
  lensesFor, packedRamp, paintGrid, rampCssOf,
} from "./mesh-views-core.js?v=__BUILD__";

const SMOOTH = "mesh-smooth";
const LINES = "mesh-contours";
const EDGE = "mesh-edge";
const LABELS = "mesh-contour-labels";
const LABEL_FONT = ["Noto Sans Bold"];
const NRAMP = 64;

let views = null;        // { meta, arrays } from aquascope.mesh_views, or null while they are made
let preparing = null;    // the promise of the views for the open model
let failed = "";         // why the views could not be made
let lens = "depth";
let smoothOn = true;
let contoursOn = true;
let threeD = false;
let before3d = null;     // { terrain, hillshade, pitch, bearing } to put back
let lastStep = null;
let wanted = "";         // the picture the map should show
let shown = "";          // the picture on the map
const cache = new Map(); // key -> { image, lines }
let pic = null;          // { canvas, ctx, image, pixels }
let generation = 0;

const fc = (features = []) => ({ type: "FeatureCollection", features });

// ── the views of a model, from the worker ──────────────────────────────────

function reset() {
  generation += 1;
  views = null; preparing = null; failed = ""; lens = "depth"; lastStep = null;
  wanted = ""; shown = ""; cache.clear(); pic = null;
  removeLayers();
}

function prepare() {
  const my = generation;
  preparing = meshAsk("views").then((res) => {
    if (my !== generation) return;
    views = { meta: res.meta, arrays: res.arrays };
    globalThis.__aq = globalThis.__aq || {};
    globalThis.__aq.meshViews = { meta: () => (views ? views.meta : null), lens: () => lens, shown: () => shown, setLens };
    repaintMesh();
    refreshMeshLegend();
  }).catch((err) => {
    if (my !== generation) return;
    failed = String((err && err.message) || err);
    console.warn("mesh views", failed);
    refreshMeshLegend();
  });
}

// ── the smooth picture and the lines ────────────────────────────────────────

function layerVis() {
  const s = meshState();
  return s && s.visible ? "visible" : "none";
}

function ensureLayers() {
  if (!map || !views) return;
  const g = views.meta.grid;
  if (!pic) {
    const canvas = document.createElement("canvas");
    canvas.width = g.width; canvas.height = g.height;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(g.width, g.height);
    pic = { canvas, ctx, image, pixels: new Uint32Array(image.data.buffer) };
  }
  const under = map.getLayer(MESH_LAYERS.HIT) ? MESH_LAYERS.HIT : undefined;
  if (!map.getSource(SMOOTH)) {
    map.addSource(SMOOTH, { type: "canvas", canvas: pic.canvas, coordinates: g.corners, animate: false });
    map.addLayer({ id: SMOOTH, type: "raster", source: SMOOTH, layout: { visibility: "none" },
      paint: { "raster-opacity": 0.95, "raster-fade-duration": 0, "raster-resampling": "linear" } }, under);
  }
  const above = map.getLayer(MESH_LAYERS.SEL) ? MESH_LAYERS.SEL : undefined;
  if (!map.getSource(LINES)) {
    map.addSource(LINES, { type: "geojson", data: fc() });
    map.addLayer({ id: EDGE, type: "line", source: LINES, filter: ["==", ["get", "kind"], "edge"], layout: { visibility: "none", "line-join": "round" },
      paint: { "line-color": "#0b3a63", "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.8, 15, 1.6], "line-opacity": 0.75 } }, above);
    map.addLayer({ id: LINES, type: "line", source: LINES, filter: ["!=", ["get", "kind"], "edge"], layout: { visibility: "none", "line-join": "round" },
      paint: { "line-color": ["match", ["get", "kind"], "arrival", "#20123f", "#0c2238"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.5, 15, 1.1], "line-opacity": 0.6 } }, above);
    map.addLayer({ id: LABELS, type: "symbol", source: LINES, filter: ["!=", ["get", "kind"], "edge"],
      layout: { visibility: "none", "symbol-placement": "line", "text-field": ["get", "label"], "text-font": LABEL_FONT,
        "text-size": 10.5, "symbol-spacing": 260, "text-max-angle": 30, "text-keep-upright": true, "text-padding": 2 },
      paint: { "text-color": "#0c2238", "text-halo-color": "rgba(255,255,255,0.92)", "text-halo-width": 1.4 } }, overArrows());
  }
}

// The contour labels go over the model's arrows (so an arrow never strikes through "2 m"): just above them.
function overArrows() {
  const ids = map.getStyle().layers.map((l) => l.id);
  const k = ids.indexOf(MESH_LAYERS.ARROWS);
  if (k < 0) return map.getLayer(MESH_LAYERS.SEL) ? MESH_LAYERS.SEL : undefined;
  return ids[k + 1];
}

function removeLayers() {
  if (!map) return;
  for (const id of [LABELS, LINES, EDGE, SMOOTH]) if (map.getLayer(id)) map.removeLayer(id);
  for (const id of [LINES, SMOOTH]) if (map.getSource(id)) map.removeSource(id);
}

let edgeOpacity = null;   // the cell edges' own opacity, put back when the cells are on show again

function setShown(smooth, lines) {
  if (!map) return;
  const vis = layerVis();
  const edges = MESH_LAYERS.EDGES;
  if (map.getLayer(edges)) {   // a smooth picture has no cells to outline
    if (edgeOpacity === null) edgeOpacity = map.getPaintProperty(edges, "line-opacity") ?? 0.22;
    map.setPaintProperty(edges, "line-opacity", smooth ? 0 : edgeOpacity);
  }
  if (map.getLayer(SMOOTH)) map.setLayoutProperty(SMOOTH, "visibility", smooth ? vis : "none");
  for (const id of [LINES, LABELS, EDGE]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", lines && contoursOn ? vis : "none");
  }
}

function scaleOf(which) {
  const s = meshState();
  return lensScale(which, views.meta, s.scale, views.meta.speed_max || maxOf(views.arrays.max_speed));
}

function maxOf(a) {
  let m = 0;
  if (a) for (const x of a) if (Number.isFinite(x) && x > m) m = x;
  return m;
}

const rampCache = new Map();
function rampOf(which) {
  const k = which === "arrival" || which === "max_speed" ? which : "depth";
  if (!rampCache.has(k)) rampCache.set(k, packedRamp(lensStops(k), NRAMP));
  return rampCache.get(k);
}

function draw(key, entry) {
  ensureLayers();
  const which = key.split(":")[0];
  paintGrid(pic.pixels, entry.image, scaleOf(which), rampOf(which));
  pic.ctx.putImageData(pic.image, 0, 0);
  const src = map.getSource(SMOOTH);
  if (src) {   // a still canvas source uploads its pixels when it plays: play for one frame
    src.play();
    map.triggerRepaint();
    map.once("render", () => { if (map.getSource(SMOOTH) === src) src.pause(); });
  }
  const lines = map.getSource(LINES);
  if (lines) lines.setData(entry.lines || fc());
  shown = key;
  setShown(true, true);
}

function pictureKey(s) {
  if (lens === "depth") return s.view === "step" ? `depth:${s.step}` : "depth:max";
  return `${lens}:all`;
}

function request(key) {
  const [which, at] = key.split(":");
  const what = which === "depth" && at === "max" ? "depth" : which;
  const step = which === "depth" && at !== "max" ? Number(at) : null;
  const my = generation;
  meshAsk("field", { what, step }).then((res) => {
    if (my !== generation) return;
    cache.set(key, { image: res.image, lines: res.lines });
    while (cache.size > 16) cache.delete(cache.keys().next().value);
    if (wanted === key) repaintMesh();
  }).catch((err) => {
    if (my !== generation) return;
    cache.delete(key);
    console.warn("mesh field", key, err && err.message);
  });
  cache.set(key, null);   // asked for
}

// Each cell in a lens's colours, for a lens without its smooth picture (hazard, or smoothing off).
function cellColours(which) {
  const a = views.arrays;
  if (which === "hazard") { const pal = hazardPacked(); return (i) => pal[a.hazard[i]] || 0; }
  const vals = which === "arrival" ? a.arrival_h : which === "max_speed" ? a.max_speed : a.max_depth;
  const sc = scaleOf(which), ramp = rampOf(which), one = new Float32Array(1), out = new Uint32Array(1);
  return (i) => { one[0] = vals[i]; return paintGrid(out, one, sc, ramp)[0]; };
}

// The views' say in mesh-view.js's paint(): the smooth picture, a lens's cell colours, or the model's own blues.
function paint(s) {
  if (!views) return null;
  if (s.view === "step" && lastStep !== null && s.step !== lastStep && lens !== "depth") lens = "depth";   // the time bar means depth
  lastStep = s.view === "step" ? s.step : lastStep;
  if (lens === "hazard" || (!smoothOn && lens !== "depth")) {
    setShown(false, false);
    wanted = "";
    return { colourOf: cellColours(lens) };
  }
  if (!smoothOn) { setShown(false, false); wanted = ""; return null; }
  const key = pictureKey(s);
  wanted = key;
  const entry = cache.get(key);
  if (entry) { if (shown !== key || !map.getLayer(SMOOTH)) draw(key, entry); else setShown(true, true); return { smooth: true }; }
  if (entry === undefined) request(key);
  if (shown && shown.split(":")[0] === key.split(":")[0] && map.getLayer(SMOOTH)) return { smooth: true };   // keep the last step until this one comes
  setShown(false, false);
  shown = "";
  return lens === "depth" ? null : { colourOf: cellColours(lens) };
}

// ── 3D ──────────────────────────────────────────────────────────────────────

function set3d(on) {
  if (!map || on === threeD) return;
  threeD = on;
  if (on) {
    before3d = { terrain: state.terrain, hillshade: state.hillshade, pitch: map.getPitch(), bearing: map.getBearing() };
    state.terrain = true; setTerrain(true);
    if (!state.hillshade) { state.hillshade = true; setHillshade(true); }
    const box = currentMesh() && currentMesh().meta.bbox;
    const cam = { pitch: 60, bearing: before3d.bearing || -20, duration: 900 };
    if (box) map.fitBounds([[box[0], box[1]], [box[2], box[3]]], { ...cam, padding: (globalThis.innerWidth || 1024) <= 520 ? 24 : 50, maxZoom: 16 });
    else map.easeTo(cam);
  } else {
    const b = before3d || { terrain: false, hillshade: state.hillshade, pitch: 0, bearing: 0 };
    state.terrain = b.terrain; setTerrain(b.terrain);
    if (state.hillshade !== b.hillshade) { state.hillshade = b.hillshade; setHillshade(b.hillshade); }
    map.easeTo({ pitch: b.terrain ? Math.max(b.pitch, 20) : b.pitch, bearing: b.bearing, duration: 700 });
  }
  syncRailControls();
  renderCredits();
  try { writeUrl(); } catch { /* the URL is a convenience */ }
  announce(on ? "The model is draped over the terrain in 3D." : "Back to the flat map.");
}

// ── the legend row ──────────────────────────────────────────────────────────

function setLens(id) {
  if (!views || !lensesFor(views.meta).some((l) => l.id === id)) return false;
  lens = id;
  repaintMesh();
  refreshMeshLegend();
  announce(`${lensesFor(views.meta).find((l) => l.id === id).label} lens.`);
  return true;
}

function section() {
  if (!currentMesh()) return "";
  if (!views) {
    return failed ? `<p class="mv-line muted">Smooth depth and lenses unavailable: ${escapeHtml(failed)}</p>`
      : '<p class="mv-line muted mv-prep">Preparing the smooth depth, contours and lenses…</p>';
  }
  const list = lensesFor(views.meta);
  const chips = list.map((l) => `<button type="button" class="mv-lens${l.id === lens ? " on" : ""}" data-act="lens-${l.id}" ` +
    `aria-pressed="${l.id === lens ? "true" : "false"}" title="${escapeHtml(l.title)}">${escapeHtml(l.label)}</button>`).join("");
  const opt = (act, on, label, title) => `<label title="${escapeHtml(title)}"><input type="checkbox" data-act="${act}"${on ? " checked" : ""}> ${label}</label>`;
  return `<div class="mv-lenses" role="group" aria-label="Lens">${chips}</div>` +
    `<div class="mv-opts mv-vopts">${opt("smooth", smoothOn, "Smooth", "Depth interpolated between cell centres, or the cells as the model computed them")}` +
    `${opt("contours", contoursOn, "Contours", "Lines of equal depth (or arrival time), and the edge of the water")}` +
    `${opt("3d", threeD, "3D", "The terrain in 3D, with the water draped over it")}</div>`;
}

function key() {
  if (!views || lens === "depth") return null;
  const meta = views.meta;
  const unitTag = lens === "max_speed" && meta.units.velocity ? `, ${meta.units.velocity}` : lens === "max_depth" && meta.units.length ? `, ${meta.units.length}` : "";
  const line = `<p class="fd-tag">${escapeHtml(lensLine(lens, meta))}${escapeHtml(unitTag)}</p>`;
  if (lens === "hazard") {
    const h = meta.hazard;
    const rows = h.classes.map((c, k) => `<li title="${escapeHtml(c.words)}"><i style="background:${HAZARD_COLOURS[k]}"></i>` +
      `<b>${c.code}</b><span>${escapeHtml(HAZARD_SHORT[k])}</span><em>${escapeHtml(areaWords(c.area_m2))}</em></li>`).join("");
    return line + `<ul class="mv-hazard" aria-label="Hazard classes and the area in each">${rows}</ul>` +
      `<p class="fd-src">${escapeHtml(h.source.short)}, the combined hazard curves (D × V, depth and speed limits). ` +
      `<a href="${escapeHtml(h.source.url)}" target="_blank" rel="noopener">Source</a> · ${escapeHtml(h.source.licence)}</p>`;
  }
  const sc = scaleOf(lens);
  const t = lensTicks(lens, sc).map((k) => `<span style="left:${k.at}%">${escapeHtml(k.label)}</span>`).join("");
  const words = lens === "arrival" ? "early (dark) to late (light)" : "low (light) to high (dark)";
  let extra = "";
  if (lens === "arrival") {
    const a = meta.arrival;
    extra = `<p class="mv-line muted">Arrival is ${escapeHtml(a.rule)}. ${a.wet_at_start ? `${a.wet_at_start.toLocaleString("en-GB")} cells wet from the start; ` : ""}` +
      `the last place floods ${escapeHtml(hoursWords(a.last_h))} after the start. Lines every ${escapeHtml(stepWords(meta.arrival_levels))}; ` +
      `blank where the water never reaches ${escapeHtml(String(a.depth_m))} m.</p>`;
  }
  return line + `<div class="fd-ramp" role="img" aria-label="${escapeHtml(lensLine(lens, meta))}, ${words}" style="background:${rampCssOf(lensStops(lens))}"></div>` +
    `<div class="fd-ticks" aria-hidden="true">${t}</div>` + extra;
}

function stepWords(levels) {
  if (!levels || levels.length < 2) return "few hours";
  return hoursWords(levels[1] - levels[0]);
}

function act(name, el) {
  if (name.startsWith("lens-")) return setLens(name.slice(5)) || true;
  if (name === "smooth") { smoothOn = el.checked; repaintMesh(); refreshMeshLegend(); return true; }
  if (name === "contours") { contoursOn = el.checked; repaintMesh(); return true; }
  if (name === "3d") { set3d(el.checked); return true; }
  return false;
}

function card(i) {
  if (!views || lens === "depth") return null;
  const m = currentMesh();
  const f = lensFacts(lens, views, i, m.meta.times, m.meta.units);
  if (!f) return null;
  return { status: { text: f.status, color: f.color }, figure: f.figure, note: f.note };
}

function summary() {
  if (!views || lens === "depth") return null;
  return { hazard: "hazard H1 to H6", arrival: "arrival time", max_depth: "largest depth", max_speed: "largest speed" }[lens];
}

/** Choose a lens from another module (Flood it's buttons): false until the lenses are made. */
export const setMeshLens = (id) => setLens(id);

// ── boot ────────────────────────────────────────────────────────────────────

export function initMeshViews() {
  setMeshViews({
    opened() { reset(); prepare(); },
    paint,
    key,
    section,
    act,
    card,
    summary,
    visible() { if (shown) setShown(true, true); else setShown(false, false); },
  });
  onMeshCleared(() => reset());
  if (map) {
    map.on("style.load", () => setTimeout(() => {
      if (!currentMesh() || !views) return;
      pic = null; shown = "";
      repaintMesh();
    }, 0));
  }
}
