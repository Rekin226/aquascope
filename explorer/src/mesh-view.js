// Your model results on the globe (#555): the Crayfish view, in the page. Drop a HEC-RAS 2D plan file or a
// UGRID NetCDF mesh result on the map (or Tools, "Open model results") and the mesh is drawn on the globe:
// the largest depth in each cell by default, in blues, with velocity arrows at a readable density, optional
// particle traces, and the time bar switched to the model's own times to step and play through the run.
// A click on a cell opens the map card with the depth, the velocity and the time series there.
//
// The file never leaves the browser: mesh-worker.js mounts it read-only in its own Python and runs
// aquascope.io.mesh (the same reader as `aquascope mesh` and the MCP tool mesh_summary), which reads the
// geometry and the maxima once and one time step when asked. This module draws; mesh-core.js decides the
// colours, the arrows, the click and the words.

import { $, actions, claimClicks, clickLayers, dbg, escapeHtml, state } from "./core.js?v=__BUILD__";
import { CONFIG } from "../config.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { cardOpen, closeCard, openCard, updateCard } from "./map-card.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { modelClockOn, releaseModelClock, setModelStep, useModelClock } from "./time-ui.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { isModflowDrop } from "./groundwater-core.js?v=__BUILD__";
import {
  ACCEPT, advect, arrowSize, bearing, binOf, buildIndex, canvasFrame, cellAt, cellFacts, checkFile, colorAt, friendlyError, legendLine, pickModelFiles,
  metresPer, modelShort, modelTime, packedBins, paintPixels, particleTimeScale, rampCss, rangeWords, rasterizeIds, scaleFor,
  shortName, thinArrows, ticks, variableWords,
} from "./mesh-core.js?v=__BUILD__";

const IMG = "mesh-img";
const HIT = "mesh-hit";
const EDGES = "mesh-edges";
const ARROWS = "mesh-arrows";
const SEL = "mesh-sel";
const ARROW_ICON = "mesh-arrow";
const NBINS = 32;
const ARROW_PX = 34;
const EDGE_MINZOOM = 12.5;
const CREDIT = "Your file, read in your browser by aquascope.io.mesh. Nothing is uploaded.";

let worker = null;
let workerReady = null;
let reqId = 0;
const waiting = new Map();

let mesh = null;        // { meta, arrays, index, scale, frame }
let pending = null;     // { file, geometry } while a file waits for its coordinate system
let view = "max";       // "max" (largest over the run) or "step"
let step = 0;           // the model step on show, and the step the arrows come from in the max view
let frame = null;       // { step, value, u, v } of `step`
let visible = true;
let arrowsOn = true;
let particlesOn = false;
let selected = -1;
let bound = false;

const vis = () => (visible ? "visible" : "none");

// ── the worker ──────────────────────────────────────────────────────────────

function ensureWorker() {
  if (workerReady) return workerReady;
  worker = new Worker(new URL("./mesh-worker.js?v=__BUILD__", import.meta.url), { type: "module" });
  workerReady = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data || {};
      if (m.type === "progress") { showStatus({ text: m.text, busy: true, keep: true }); return; }
      if (m.type === "ready") { resolve(); return; }
      if (m.type === "failed") { reject(new Error(m.error)); return; }
      const job = waiting.get(m.id);
      if (!job) return;
      waiting.delete(m.id);
      if (m.error) job.reject(Object.assign(new Error(m.error), { mesh: m.mesh }));
      else job.resolve(m.result);
    };
    worker.onerror = (e) => {
      const err = new Error((e && e.message) || "the model reader stopped");
      reject(err);
      for (const job of waiting.values()) job.reject(err);
      waiting.clear();
    };
  });
  workerReady.catch(() => { worker = null; workerReady = null; });
  worker.postMessage({ type: "init", pyodideIndexURL: CONFIG.pyodideIndexURL,
    wheelsJson: new URL(CONFIG.wheelsJson, location.href).href });
  return workerReady;
}

async function ask(type, payload = {}) {
  await ensureWorker();
  const id = ++reqId;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...payload });
  });
}

// ── for the review (#564, mesh-review.js): the open mesh, its worker, and a word when it goes ────

/** The model on the map: { meta, arrays } (the arrays are the reader's typed arrays), or null. */
export const currentMesh = () => (mesh ? { meta: mesh.meta, arrays: mesh.arrays } : null);

/** A call to the mesh worker, where the file is open. */
export const meshAsk = (type, payload = {}) => ask(type, payload);

const clearedHooks = new Set();
/** Run fn(name) when the model on the map is removed or replaced. */
export function onMeshCleared(fn) { clearedHooks.add(fn); return () => clearedHooks.delete(fn); }

let reviewSection = () => "";
/** The review's part of the legend row (mesh-review.js sets it). */
export function setReviewSection(fn) { reviewSection = fn; }

/** Redraw the legend row (the review calls it when its result changes). */
export function refreshMeshLegend() { if (mesh) refreshLegend("mesh"); }

/** The layer the review's cell flags go under: the arrows, so they stay readable on top. */
export const meshFlagsBefore = () => (map && map.getLayer(ARROWS) ? ARROWS : beforeId());

// ── the status card: reading, errors, the coordinate system question ──────

function showStatus({ text = "", busy = false, error = false, askCrs = false, keep = false } = {}) {
  const el = $("mesh-status");
  if (!el) return;
  if (keep && !el.classList.contains("busy")) return;   // a late progress line does not reopen a closed card
  if (!text) { el.hidden = true; el.innerHTML = ""; el.className = "mesh-status"; return; }
  el.className = `mesh-status${busy ? " busy" : ""}${error ? " error" : ""}`;
  el.hidden = false;
  el.innerHTML =
    `<div class="ms-line">${busy ? '<span class="ms-spin" aria-hidden="true"></span>' : ""}<p>${escapeHtml(text)}</p>` +
    `<button type="button" class="ms-close" data-act="close" aria-label="Close" title="Close">×</button></div>` +
    (askCrs
      ? `<form class="ms-crs" data-act="crs"><label for="ms-epsg">EPSG code</label>` +
        `<input id="ms-epsg" name="epsg" inputmode="numeric" pattern="[0-9]{4,6}" placeholder="for example 32633" required autocomplete="off">` +
        `<button type="submit" class="btn small primary">Place it</button></form>` +
        `<p class="ms-hint">The code of the projection the model was built in: RAS Mapper's projection, or the ` +
        `grid's EPSG code. <a href="https://epsg.io/" target="_blank" rel="noopener">Find a code</a></p>`
      : "") +
    (error && !askCrs ? '<div class="ms-row"><button type="button" class="btn small" data-act="pick">Choose another file</button></div>' : "") +
    (busy ? `<p class="ms-hint">${escapeHtml(CREDIT)}</p>` : "");
  if (askCrs) setTimeout(() => { const box = el.querySelector("input[name=epsg]"); if (box) box.focus(); }, 0);
}

// ── opening a file ──────────────────────────────────────────────────────────

/** Open what a drop or a pick brought: a model result, with its HEC-RAS geometry file when one came along. */
export function openModelFiles(files) {
  const got = pickModelFiles(files);
  if (got.why) { showStatus({ text: got.why, error: true }); announce(got.why); return; }
  openModelFile(got.file, null, got.geometry);
}

export async function openModelFile(file, crs = null, geometry = null) {
  const why = checkFile(file);
  if (why) { showStatus({ text: why, error: true }); announce(why); return; }
  if (!state.mapOk || !map) { showStatus({ text: "The map could not start in this browser, so a model cannot be drawn.", error: true }); return; }
  pending = null;
  showStatus({ text: `Reading ${shortName(file.name, 40)}${geometry ? ` with ${shortName(geometry.name, 28)}` : ""}…`, busy: true });
  let res;
  try {
    res = await ask("open", { file, crs, geometry });
  } catch (err) {
    const msg = err.mesh ? friendlyError(err.message, file.name) : String(err.message || err);
    if (/EPSG/.test(msg)) {
      pending = { file, geometry };
      showStatus({ text: msg, error: true, askCrs: true });
    } else {
      showStatus({ text: err.mesh ? msg : `Could not read ${file.name}: ${msg}`, error: true });
    }
    announce(msg);
    return;
  }
  clearMesh({ keepStatus: true, keepWorker: true });   // the worker has already put the new file in its place
  const { meta, arrays } = res;
  mesh = { meta, arrays, index: buildIndex(arrays), scale: scaleFor(arrays.max_value, meta.variable) };
  dbg.mesh = { name: meta.name, size_bytes: meta.size_bytes, cells: meta.cells, steps: meta.times.length,
    read_ms: meta.read_ms, heap_mb: meta.heap_mb };   // for the timing checks in the PR
  view = "max";
  step = Number.isInteger(meta.peak_step) ? meta.peak_step : 0;
  frame = null;
  visible = true;
  showStatus();
  registerRow();
  openLegendRow("mesh", true);
  addLayers();
  zoomTo(1400);
  paint();
  if (meta.times.length) {
    takeClock(step);
    if (meta.has_velocity) loadFrame(step).then(() => drawArrows());
  }
  announce(`${meta.name} is on the map: ${Number(meta.cells).toLocaleString("en-GB")} cells.`);
}

// The time bar on the model's own times; taken again by "Step through the run" after "Back to dates".
function takeClock(index) {
  const { meta } = mesh;
  useModelClock({
    times: meta.times, index, label: (i) => modelTime(meta.times[i]),
    summary: `Model time: ${rangeWords(meta.times)}.`,
    onStep: (i) => showStep(i),
    onExit: () => showLargest(),
  });
}

function zoomTo(duration) {
  const b = mesh.meta.bbox;
  const pad = (globalThis.innerWidth || 1024) <= 520 ? 28 : 70;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: pad, maxZoom: 15, duration });
}

function pickFile() {
  const input = $("mesh-file");
  if (input) { input.value = ""; input.click(); }
}

// ── painting the mesh ───────────────────────────────────────────────────────

function values() {
  return view === "max" || !frame ? mesh.arrays.max_value : frame.value;
}

// The picture: which cell covers each pixel is worked out once per mesh (rasterizeIds); a time step only looks
// its colours up and hands the canvas to the map (a canvas source, so no image is encoded).
function setupPicture() {
  const { arrays, meta } = mesh;
  const fr = canvasFrame(meta.bbox, meta.cells);
  const n = arrays.ring_lon.length;
  const px = new Float32Array(n), py = new Float32Array(n);
  for (let k = 0; k < n; k++) { px[k] = fr.x(arrays.ring_lon[k]); py[k] = fr.y(arrays.ring_lat[k]); }
  const canvas = document.createElement("canvas");
  canvas.width = fr.width;
  canvas.height = fr.height;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(fr.width, fr.height);
  mesh.picture = { fr, canvas, ctx, image, pixels: new Uint32Array(image.data.buffer),
    ids: rasterizeIds(arrays.ring_offsets, px, py, fr.width, fr.height), colours: packedBins(NBINS),
    cellColour: new Uint32Array(meta.cells) };
}

function paint() {
  if (!mesh) return;
  if (!mesh.picture) setupPicture();
  const pic = mesh.picture;
  const vals = values();
  for (let i = 0; i < pic.cellColour.length; i++) {
    const b = binOf(vals[i], mesh.scale, NBINS);
    pic.cellColour[i] = b < 0 ? 0 : pic.colours[b];
  }
  paintPixels(pic.pixels, pic.ids, pic.cellColour);
  pic.ctx.putImageData(pic.image, 0, 0);
  const src = map.getSource(IMG);
  if (!src) {
    map.addSource(IMG, { type: "canvas", canvas: pic.canvas, coordinates: pic.fr.corners, animate: false });
    map.addLayer({ id: IMG, type: "raster", source: IMG, layout: { visibility: vis() },
      paint: { "raster-opacity": 0.92, "raster-fade-duration": 0, "raster-resampling": "linear" } }, beforeId());
  } else {
    // A still canvas source uploads its pixels when it plays: play for one frame.
    src.play();
    map.triggerRepaint();
    map.once("render", () => { if (map.getSource(IMG) === src) src.pause(); });
  }
  refreshLegend("mesh");
}

// ── the map layers: the picture, the cell edges, the arrows, the click ─────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => ["clusters", "points", "now-dots", "gauge-heat"].includes(l.id) || /^river-lit/.test(l.id));
  return hit ? hit.id : undefined;
}

function arrowImage() {
  const s = 48;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d");
  g.fillStyle = "#000";
  g.beginPath();
  // a slim arrow pointing up (north), drawn for an SDF icon so its colour and halo come from the style
  g.moveTo(24, 4); g.lineTo(35, 20); g.lineTo(27.5, 18.5); g.lineTo(27.5, 44); g.lineTo(20.5, 44); g.lineTo(20.5, 18.5);
  g.lineTo(13, 20); g.closePath(); g.fill();
  return g.getImageData(0, 0, s, s);
}

function edgesGeoJSON() {
  const { ring_offsets: off, ring_lon: xs, ring_lat: ys } = mesh.arrays;
  const lines = [];
  for (let i = 0; i < off.length - 1; i++) {
    const ring = [];
    for (let k = off[i]; k < off[i + 1]; k++) ring.push([xs[k], ys[k]]);
    if (ring.length) ring.push(ring[0]);
    lines.push(ring);
  }
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: lines } }] };
}

function bboxPolygon(b) {
  const [w, s, e, n] = b;
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] };
}

function addLayers() {
  if (!mesh || !map) return;
  const before = beforeId();
  if (!map.getSource(HIT)) {
    map.addSource(HIT, { type: "geojson", data: bboxPolygon(mesh.meta.bbox) });
    map.addLayer({ id: HIT, type: "fill", source: HIT, layout: { visibility: vis() }, paint: { "fill-color": "#000", "fill-opacity": 0 } }, before);
  }
  if (!map.getSource(EDGES)) {
    map.addSource(EDGES, { type: "geojson", data: edgesGeoJSON() });
    map.addLayer({ id: EDGES, type: "line", source: EDGES, minzoom: EDGE_MINZOOM, layout: { visibility: vis() },
      paint: { "line-color": "#16324f", "line-opacity": ["interpolate", ["linear"], ["zoom"], EDGE_MINZOOM, 0, EDGE_MINZOOM + 1.5, 0.22],
        "line-width": 0.6 } }, before);
  }
  if (!map.getSource(SEL)) {
    map.addSource(SEL, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: SEL, type: "line", source: SEL, layout: { visibility: vis() },
      paint: { "line-color": "#0b1f33", "line-width": 2.2 } }, before);
  }
  if (!map.hasImage(ARROW_ICON)) map.addImage(ARROW_ICON, arrowImage(), { sdf: true, pixelRatio: 1.7 });
  if (!map.getSource(ARROWS)) {
    map.addSource(ARROWS, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: ARROWS, type: "symbol", source: ARROWS,
      layout: { visibility: arrowsOn ? vis() : "none", "icon-image": ARROW_ICON, "icon-rotate": ["get", "b"],
        "icon-rotation-alignment": "map", "icon-size": ["get", "s"], "icon-allow-overlap": true, "icon-ignore-placement": true },
      paint: { "icon-color": "#0c2238", "icon-halo-color": "rgba(255,255,255,0.95)", "icon-halo-width": 1.6, "icon-opacity": 0.95 } }, before);
  }
  if (!bound) {
    bound = true;
    clickLayers.add(HIT);
    map.on("click", HIT, onClick);
    claimClicks((ll) => Boolean(mesh && visible && cellAt(mesh.index, ll.lng, ll.lat) >= 0));
    map.on("mouseenter", HIT, () => { map.getCanvas().style.cursor = "crosshair"; });
    map.on("mouseleave", HIT, () => { map.getCanvas().style.cursor = ""; });
  }
}

function removeLayers() {
  for (const id of [ARROWS, SEL, EDGES, HIT, IMG]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
}

// ── the time steps ──────────────────────────────────────────────────────────

const frames = new Map();   // step -> frame, the last few
async function loadFrame(i) {
  if (frames.has(i)) { frame = frames.get(i); return frame; }
  const res = await ask("frame", { step: i });
  const f = { step: i, ...res.arrays };
  frames.set(i, f);
  while (frames.size > 12) frames.delete(frames.keys().next().value);
  if (step === i) frame = f;
  return f;
}

async function showStep(i) {
  if (!mesh) return;
  step = i;
  view = "step";
  const f = await loadFrame(i);
  if (!mesh || step !== i) return;
  frame = f;
  paint();
  drawArrows();
  syncSelection();
}

async function showLargest() {
  if (!mesh) return;
  view = "max";
  paint();
  drawArrows();
  syncSelection();
}

// ── the arrows ──────────────────────────────────────────────────────────────

function drawArrows() {
  const src = map && map.getSource(ARROWS);
  if (!src || !mesh) return;
  if (!arrowsOn || !frame || !frame.u || frame.step !== step) { src.setData({ type: "FeatureCollection", features: [] }); return; }
  const { lon, lat } = mesh.arrays;
  const b = map.getBounds();
  const W = b.getWest(), E = b.getEast(), S = b.getSouth(), N = b.getNorth();
  const pts = [];
  for (let i = 0; i < lon.length; i++) {
    const u = frame.u[i], v = frame.v[i];
    if (!Number.isFinite(u) || !Number.isFinite(v) || (u === 0 && v === 0)) continue;
    if (lon[i] < W || lon[i] > E || lat[i] < S || lat[i] > N) continue;
    const p = map.project([lon[i], lat[i]]);
    pts.push({ i, x: p.x, y: p.y });
  }
  const top = mesh.meta.speed_max || 1;
  const keep = thinArrows(pts, ARROW_PX);
  src.setData({ type: "FeatureCollection", features: keep.map((i) => ({ type: "Feature",
    properties: { b: Math.round(bearing(frame.u[i], frame.v[i])), s: arrowSize(Math.hypot(frame.u[i], frame.v[i]), top) },
    geometry: { type: "Point", coordinates: [lon[i], lat[i]] } })) });
}

// ── particle traces ─────────────────────────────────────────────────────────

let particles = [];
let rafId = 0;
let moving = false;
let timeScale = 0;

const particleEl = () => (map ? map.getContainer().querySelector(".mesh-particles") : null);

function particleCanvas() {
  let c = particleEl();
  if (!c) {
    c = document.createElement("canvas");
    c.className = "mesh-particles";
    c.setAttribute("aria-hidden", "true");
    map.getContainer().appendChild(c);
  }
  const r = map.getCanvas();
  const dpr = globalThis.devicePixelRatio || 1;
  if (c.width !== Math.round(r.clientWidth * dpr) || c.height !== Math.round(r.clientHeight * dpr)) {
    c.width = Math.round(r.clientWidth * dpr);
    c.height = Math.round(r.clientHeight * dpr);
    c.style.width = `${r.clientWidth}px`;
    c.style.height = `${r.clientHeight}px`;
  }
  return c;
}

function spawn() {
  const { lon, lat } = mesh.arrays;
  for (let tries = 0; tries < 40; tries++) {
    const i = Math.floor(Math.random() * lon.length);
    if (frame && Number.isFinite(frame.u[i]) && Math.hypot(frame.u[i], frame.v[i]) > 0) {
      const box = mesh.index.boxes;
      const jx = (box[i * 4 + 2] - box[i * 4]) * (Math.random() - 0.5) * 0.6;
      const jy = (box[i * 4 + 3] - box[i * 4 + 1]) * (Math.random() - 0.5) * 0.6;
      return { lon: lon[i] + jx, lat: lat[i] + jy, age: Math.floor(Math.random() * 80) };
    }
  }
  return null;
}

function rescale() {
  if (!mesh) return;
  const z = map.getZoom(), c = map.getCenter();
  const mpp = (40075016.686 * Math.cos((c.lat * Math.PI) / 180)) / (512 * 2 ** z);
  timeScale = particleTimeScale(mesh.meta.speed_max, mpp, metresPer(mesh.meta.units.velocity));
}

function tick() {
  rafId = 0;
  if (!particlesOn || !mesh || !frame || !frame.u || !visible) { stopParticles(); return; }
  const c = particleCanvas();
  const g = c.getContext("2d");
  const dpr = globalThis.devicePixelRatio || 1;
  g.save();
  g.globalCompositeOperation = "destination-in";
  g.fillStyle = "rgba(0,0,0,0.9)";
  g.fillRect(0, 0, c.width, c.height);
  g.restore();
  if (!moving) {
    if (particles.length < 1200) for (let k = 0; k < 40; k++) { const p = spawn(); if (p) particles.push(p); }
    g.strokeStyle = "rgba(255,255,255,0.9)";
    g.lineWidth = 1.3 * dpr;
    g.lineCap = "round";
    g.beginPath();
    const unit = metresPer(mesh.meta.units.velocity);
    for (let k = 0; k < particles.length; k++) {
      const p = particles[k];
      const i = cellAt(mesh.index, p.lon, p.lat);
      const u = i >= 0 ? frame.u[i] : NaN, v = i >= 0 ? frame.v[i] : NaN;
      if (!Number.isFinite(u) || !Number.isFinite(v) || p.age > 90) { particles[k] = spawn() || p; particles[k].age = 0; continue; }
      const [lon2, lat2] = advect(p.lon, p.lat, u, v, timeScale / 30, unit);
      const a = map.project([p.lon, p.lat]), b = map.project([lon2, lat2]);
      g.moveTo(a.x * dpr, a.y * dpr);
      g.lineTo(b.x * dpr, b.y * dpr);
      p.lon = lon2; p.lat = lat2; p.age++;
    }
    g.stroke();
  }
  rafId = requestAnimationFrame(tick);
}

function startParticles() {
  if (rafId || !particlesOn) return;
  if (frame && frame.u) { rescale(); rafId = requestAnimationFrame(tick); return; }
  if (mesh && mesh.meta.has_velocity) loadFrame(step).then(() => { if (particlesOn) startParticles(); });
}

function stopParticles() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = 0;
  particles = [];
  const c = particleEl();
  if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

// ── the click: a cell's card ────────────────────────────────────────────────

function ringOf(i) {
  const { ring_offsets: off, ring_lon: xs, ring_lat: ys } = mesh.arrays;
  const ring = [];
  for (let k = off[i]; k < off[i + 1]; k++) ring.push([xs[k], ys[k]]);
  if (ring.length) ring.push(ring[0]);
  return ring;
}

function syncSelection() {
  const src = map && map.getSource(SEL);
  if (!src) return;
  if (selected >= 0 && !cardOpen()) selected = -1;   // the card was closed: so is the outline
  src.setData(selected >= 0 && mesh
    ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ringOf(selected) } }] }
    : { type: "FeatureCollection", features: [] });
  if (selected >= 0 && mesh) updateCard(`mesh:${selected}`, cardSpec(selected, { keepSeries: true }));
}

let seriesCache = new Map();
function cardSpec(i, { keepSeries = false } = {}) {
  const { meta, arrays } = mesh;
  const at = view === "step" && frame && frame.step === step
    ? { value: frame.value[i], u: frame.u ? frame.u[i] : NaN, v: frame.v ? frame.v[i] : NaN } : null;
  const largest = { value: arrays.max_value[i], speed: arrays.max_speed ? arrays.max_speed[i] : NaN };
  const f = cellFacts(meta, i, { at, largest, when: meta.times[step] });
  const c = colorAt(at ? at.value : largest.value, mesh.scale);
  const spec = {
    title: f.title, sub: `${shortName(meta.name, 30)} · ${modelShort(meta.model)}`,
    status: { text: f.status, color: c ? `rgb(${c[0]},${c[1]},${c[2]})` : null },
    figure: f.figure, note: f.note,
    buttons: [{ id: "largest", label: "Show the largest", title: "The largest value over the run, in every cell",
      onClick: () => showLargest(), hidden: view !== "step" }],
  };
  if (!keepSeries) spec.spark = meta.times.length ? { pending: true, label: `${variableWords(meta.variable)} over the run` } : null;
  return spec;
}

const GAUGES = ["points", "clusters", "now-dots"];

function onClick(e) {
  if (!mesh) return;
  // A gauge under the click is what the reader meant; inside the mesh, the mesh answers before the background
  // layers (they step aside through clickClaimed); outside a cell, they answer, or the map's own click.
  const gauges = GAUGES.filter((id) => map.getLayer(id));
  if (gauges.length && map.queryRenderedFeatures(e.point, { layers: gauges }).length) return;
  const { lng, lat } = e.lngLat;
  const i = cellAt(mesh.index, lng, lat);
  if (i < 0) {
    const others = [...clickLayers].filter((id) => id !== HIT && map.getLayer(id));
    if (!others.length || !map.queryRenderedFeatures(e.point, { layers: others }).length) actions.selectPoint(lat, lng);
    return;
  }
  selected = i;
  const { meta } = mesh;
  openCard({
    id: `mesh:${i}`, lngLat: [mesh.arrays.lon[i], mesh.arrays.lat[i]], lift: 6, what: "Your model",
    ...cardSpec(i), credit: "Your file, read in your browser by aquascope. Nothing is uploaded.",
  });
  syncSelection();
  if (!meta.times.length) return;
  const run = seriesCache.get(i) || ask("series", { cell: i });
  seriesCache.set(i, run);
  run.then((s) => {
    if (selected !== i) return;
    const unit = s.units[s.variable] || "";
    const has = s.value.some((x) => x !== null);
    updateCard(`mesh:${i}`, { spark: has
      ? { v: s.value, label: `${variableWords(s.variable)} over the run${unit ? `, ${unit}` : ""}`, fromZero: s.variable === "depth",
        alt: `${variableWords(s.variable)} at this cell over the model's ${s.times.length} steps` }
      : { empty: "Dry at every step." } });
  }).catch((err) => updateCard(`mesh:${i}`, { spark: { empty: `Could not read the series: ${err.message}` } }));
}

// ── the legend row ──────────────────────────────────────────────────────────

function keyHtml() {
  if (!mesh) return "";
  const { meta, scale } = mesh;
  const unit = meta.units[meta.variable];
  const t = ticks(scale).map((k) => `<span style="left:${k.at}%">${escapeHtml(k.label)}</span>`).join("");
  const words = variableWords(meta.variable);
  const notes = (meta.notes || []).map((n) => `<p class="mv-note">${escapeHtml(n)}</p>`).join("");
  const velocity = meta.has_velocity
    ? `<div class="mv-opts"><label><input type="checkbox" data-act="arrows"${arrowsOn ? " checked" : ""}> Arrows</label>` +
      `<label><input type="checkbox" data-act="particles"${particlesOn ? " checked" : ""}> Particles</label></div>` +
      (view === "max" && meta.times.length
        ? `<p class="mv-line muted">Arrows at ${escapeHtml(modelTime(meta.times[step]))}` +
          `${step === meta.peak_step ? ", when the most water is on the mesh" : ""}.</p>` : "")
    : "";
  const reviewBtn = actions.reviewModel
    ? '<button type="button" class="fd-go" data-act="review" title="Checks by code on this run: the deepest water, ' +
      'odd depths, wet and dry cells, the volume balance, observed floods and gauges">Review these results</button>' : "";
  const stepBtn = meta.times.length
    ? (view === "max"
      ? '<button type="button" class="fd-go" data-act="steps">Step through the run</button>'
      : '<button type="button" class="fd-go" data-act="largest">Show the largest</button>') : "";
  return `<p class="fd-tag">${escapeHtml(legendLine(meta, { step: view === "step" ? step : null }))}</p>` +
    `<div class="fd-ramp" role="img" aria-label="${escapeHtml(words)} from ${escapeHtml(String(scale.lo))} to ${escapeHtml(String(scale.hi))}${unit ? ` ${escapeHtml(unit)}` : ""}, light to deep blue" style="background:${rampCss()}"></div>` +
    `<div class="fd-ticks" aria-hidden="true">${t}<em>${escapeHtml(unit || "")}</em></div>` +
    velocity + notes +
    `<p class="mv-actions">${reviewBtn}${stepBtn}<button type="button" class="fd-go" data-act="zoom">Zoom to</button>` +
    `<button type="button" class="fd-go" data-act="clear">Remove</button></p>` +
    `<p class="fd-src" title="${escapeHtml(meta.model)}, ${escapeHtml(meta.crs.label)}">${escapeHtml(modelShort(meta.model))} · ${escapeHtml(crsShort(meta.crs.label))} · ` +
    `stays in your browser</p>` + reviewSection(meta.name);
}

// "NAD83 / Pennsylvania North (ftUS) (EPSG:2271)" -> "EPSG:2271"
function crsShort(label) {
  const m = /EPSG:\d+/.exec(String(label || ""));
  return m ? m[0] : String(label || "");
}

function registerRow() {
  registerLegendRow({
    id: "mesh",
    title: mesh ? shortName(mesh.meta.name, 26) : "Model results",
    mark: () => `<i class="fd-mark" style="background:${rampCss()}"></i>`,
    summary: () => {
      if (!mesh) return "";
      return view === "step" ? modelTime(mesh.meta.times[step], { year: false }) : `largest ${variableWords(mesh.meta.variable)}`;
    },
    on: () => visible,
    shown: () => Boolean(mesh),
    toggle: (on) => setVisible(on),
    body: keyHtml,
    act: (name, el) => {
      if (name === "arrows") { arrowsOn = el.checked; if (map.getLayer(ARROWS)) map.setLayoutProperty(ARROWS, "visibility", arrowsOn ? vis() : "none"); drawArrows(); }
      else if (name === "particles") { particlesOn = el.checked; if (particlesOn) startParticles(); else stopParticles(); }
      else if (name === "steps") { if (!modelClockOn()) takeClock(step); setModelStep(step); }
      else if (name === "largest") { showLargest(); }
      else if (name === "zoom") zoomTo(900);
      else if (name === "open") { pickFile(); }
      else if (name === "review") { if (actions.reviewModel) void actions.reviewModel(); }
      else if (name === "clear") { clearMesh(); }
    },
  });
}

function setVisible(on) {
  visible = Boolean(on);
  for (const id of [IMG, HIT, EDGES, SEL]) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis());
  if (map.getLayer(ARROWS)) map.setLayoutProperty(ARROWS, "visibility", arrowsOn ? vis() : "none");
  if (!visible) stopParticles(); else startParticles();
  refreshLegend("mesh");
}

export function clearMesh({ keepStatus = false, keepWorker = false } = {}) {
  stopParticles();
  if (mesh && selected >= 0) closeCard();
  selected = -1;
  if (map) removeLayers();
  const had = Boolean(mesh);
  const name = mesh ? mesh.meta.name : null;
  mesh = null;
  if (had) for (const fn of clearedHooks) { try { fn(name); } catch (err) { console.warn("mesh cleared", err && err.message); } }
  frame = null;
  frames.clear();
  seriesCache = new Map();
  view = "max";
  releaseModelClock();
  if (!keepStatus) showStatus();
  if (had) { refreshLegend(); if (worker && !keepWorker) ask("close").catch(() => {}); }
}

// ── boot ────────────────────────────────────────────────────────────────────

function hasFiles(e) {
  const types = e.dataTransfer && e.dataTransfer.types;
  return Boolean(types && [...types].includes("Files"));
}

export function initMeshView() {
  const input = $("mesh-file");
  if (input) {
    input.accept = ACCEPT;
    input.multiple = true;   // a stripped plan file and its geometry file, picked together
    input.addEventListener("change", () => { if (input.files && input.files.length) openModelFiles(input.files); });
  }
  const btn = $("btn-mesh");
  if (btn) btn.addEventListener("click", pickFile);
  const status = $("mesh-status");
  if (status) {
    status.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (!b) return;
      if (b.dataset.act === "close") { pending = null; showStatus(); }
      else if (b.dataset.act === "pick") pickFile();
    });
    status.addEventListener("submit", (e) => {
      e.preventDefault();
      const box = status.querySelector("input[name=epsg]");
      const code = ((box && box.value) || "").trim();
      if (pending && /^\d{4,6}$/.test(code)) openModelFile(pending.file, code, pending.geometry);
    });
  }
  const wrap = document.querySelector(".map-wrap");
  const drop = $("mesh-drop");
  if (wrap && drop) {
    let depth = 0;
    wrap.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; drop.hidden = false; });
    wrap.addEventListener("dragover", (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
    wrap.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) drop.hidden = true; });
    wrap.addEventListener("drop", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      drop.hidden = true;
      if (isModflowDrop(e.dataTransfer.files)) return;   // MODFLOW heads: groundwater-view.js reads them (#558)
      if (e.dataTransfer.files && e.dataTransfer.files.length) openModelFiles(e.dataTransfer.files);
    });
  }
  if (!state.mapOk || !map) return;
  map.on("moveend", () => { if (mesh) { drawArrows(); rescale(); } moving = false; });
  map.on("movestart", () => {
    moving = true;
    const c = particleEl();
    if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
    particles = [];
  });
  // A basemap change replaces the style: the mesh's sources go with it.
  map.on("style.load", () => {
    if (!mesh) return;
    setTimeout(() => {
      addLayers();
      paint();
      drawArrows();
      syncSelection();
    }, 0);
  });
  actions.openModelFile = openModelFile;
  actions.hasModel = () => Boolean(mesh);
}
