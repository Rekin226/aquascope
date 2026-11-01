// Flood it (#658): click a river, press Flood it, and a screening flood of the whole valley is drawn on the map.
//
// The package plans the run in the worker (aquascope.floodsim.plan_window and prepare_grid, the same functions as
// `aquascope flood` and the MCP tool flood_reach): a grid that follows the river up and down from the click and is
// cut to the valley, every river in it burnt in and fed by its own flow at the same return period, walls, and a
// normal-depth outfall where the river leaves. This module reads the terrain (the Terrarium tiles the hillshade
// uses), runs the local-inertial scheme on the visitor's GPU (floodsim-gpu.js) or, without one, in a worker on cells
// twice as large (floodsim-worker.js), to a near steady state, and draws every frame as it comes in the flood views'
// smooth depth colours. When it settles the camera frames the water, the time bar steps through the run, and the
// panel docked at the side offers the flood views of #619 to #622 (hazard, arrival, a section, what gets flooded):
// the run is handed to the model viewer as a UGRID file (floodsim-core.js ugridFile), so they are the same code a
// HEC-RAS run gets. Two runs on the same grid compare with a swipe. Every run is an entry in the action log, and Ask
// the map's flood_reach action runs through floodReach here.
//
// Levees and breaches (#663): Draw a levee or Draw a breach on the panel draws a line on the map (floodsim-sketch.js);
// the crest control shows the ground and the flood along it (aquascope.floodsim.levee_profile, in the worker) and
// suggests a crest above the flood; Run again floods the same valley with every drawn line raised or lowered
// (edit_grid), and the new run sits beside the old one in the swipe, with what gets flooded before and after
// (compare_exposure). Each line drawn, moved, re-set or removed is an entry in the action log.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { closeCard } from "./map-card.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { cameraPadding, logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { releaseModelClock, useModelClock } from "./time-ui.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { STOPS, fmt, niceCeil, rampCss, ticks } from "./mesh-core.js?v=__BUILD__";
import { packedRamp, paintGrid } from "./mesh-views-core.js?v=__BUILD__";
import { clearMesh, currentMesh, onMeshCleared, openModelFile, runMeshTool } from "./mesh-view.js?v=__BUILD__";
import { setMeshLens } from "./mesh-views.js?v=__BUILD__";
import {
  FLOODSIM_CREDIT, LABEL, WET_M, corners, depthOverTerrain, pxLonLat, runLoop, summary, terrariumElevation, ugridFile,
  unpackDepth, wallContact, wallWords, wetBox, windowTiles, hm, flowLine, clockLabel, clockSummary, depthTag,
} from "./floodsim-core.js?v=__BUILD__";
import { createGpuSim, gpuAvailable } from "./floodsim-gpu.js?v=__BUILD__";
import {
  changeBox, checkSketch, crestHeights, editsKey, exposureChangeFeatures, exposureChangeRows, profilePaths, sketchEdits, sketchWords,
} from "./floodsim-sketch-core.js?v=__BUILD__";
import { cancelDraw, drawSketches, onSketchClick, onSketchMove, startLineDraw } from "./floodsim-sketch.js?v=__BUILD__";
import { sourceLines } from "./flood-views-core.js?v=__BUILD__";
import { designFlowWords, flowFor } from "./design-flow-core.js?v=__BUILD__";

const SRC = "floodsim-depth";
const ROW = "floodit";
const CAP_GPU_H = 72;      // the cap of a steady run on the GPU (h of model time)
const CAP_CPU_H = 48;      // and in the worker, on cells twice as large
const CHOICES = [2, 10, 25, 50, 100];
const DOCS = "https://rekin226.github.io/aquascope/floodsim/";
const JRC_LINE = "Where it differs from JRC's 100-year map: it floods the small streams JRC leaves out, and it is " +
  "1.5 m to 2.7 m shallower on the plain on the reaches checked, mostly because GEOGLOWS's 100-year flows there are " +
  "1.5 to 3.4 times smaller than GloFAS's, which JRC's maps are made from. No bathymetry, and no defences but the ones you draw.";

let target = null;       // { river_id, lat, lon, name }
let plan = null;         // the worker's plan for target
let runs = [];           // the last runs (at most two), newest last
let shown = null;        // the run on the map
let other = null;        // the run the swipe compares with
let inViews = null;      // the run handed to the model viewer
let running = 0;
let status = null;       // { text, kind }
let washHidden = null;   // the basin layers' visibility to put back
let pic = null;          // { canvas, ctx, image, pixels, nx, ny }
let swipe = { on: false, x: 0.5 };
let sketches = [];       // the drawn levees and breaches: { id, kind, line, crest?, invert? }
let nextSketch = 1;
let selected = null;     // the sketch whose points have handles on the map
let pending = null;      // the crest control: { id, profile, crest }
let expo = null;         // before and after: { status, res, pair }
const EXPO = "floodsim-expo";
const RAMP = packedRamp(STOPS, 64);

const panel = () => $("floodit");
const q = (v) => `${fmt(v, v >= 100 ? 0 : v >= 10 ? 0 : 1)} m³/s`;
const phone = () => (globalThis.innerWidth || 1024) <= 640;

// ── inputs: the plan, the terrain, the hydrographs ──────────────────────────

/** The Terrarium terrain of a window { x0, y0, nx, ny, zoom } as a Float32Array (m), rows north to south; a drawn
 * reservoir's lake (#662) reads its terrain with it too. */
export async function readTerrain(win) {
  const t = windowTiles(win);
  const canvas = new OffscreenCanvas(t.cols * 256, t.rows * 256);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  await Promise.all(t.tiles.map(async (tile) => {
    const r = await fetch(tile.url);
    if (!r.ok) throw new Error(`terrain tile ${tile.x}/${tile.y}: HTTP ${r.status}`);
    const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    ctx.drawImage(bmp, tile.col * 256, tile.row * 256);
  }));
  return Float32Array.from(terrariumElevation(ctx.getImageData(t.ox, t.oy, win.nx, win.ny, { colorSpace: "srgb" }).data, 4));
}

async function floodsim(op, args, z) {
  const res = await call("floodsim", { op, args, z });
  if (!res || res.ok === false) throw new Error((res && res.error) || `the worker could not ${op}`);
  return res;
}

const curveOf = (p) => (p && p.reaches && p.reaches[String(p.river_id)] && p.reaches[String(p.river_id)].curve) || null;

/** The worker's choice from the picker's: { return_period } | { flow_cms } | { series, label }. */
async function resolveChoice(choice) {
  if (choice.return_period) return { pick: { return_period: Number(choice.return_period) }, words: `the ${choice.return_period}-year flow` };
  if (choice.flow_cms) return { pick: { flow_cms: Number(choice.flow_cms) }, words: `${q(Number(choice.flow_cms))}` };
  if (choice.forecast === "peak" || choice.forecast === "hydrograph") {
    setStatus("Reading GEOGLOWS's 15-day forecast here…");
    const f = await floodsim("forecast", { river_id: target.river_id });
    if (choice.forecast === "peak") {
      return { pick: { flow_cms: f.peak_cms }, words: `the forecast peak, ${q(f.peak_cms)} on ${f.peak_at.slice(0, 10)}`, series: f };
    }
    return { pick: { series: { t_s: f.t_s, q: f.q }, label: f.label }, words: f.label, series: f, start: f.start };
  }
  if (choice.routed) {
    setStatus("Routing the wave down the river (Route it)…");
    const r = await floodsim("routed", { river_id: target.river_id, return_period: choice.routed, duration_h: 24 });
    return { pick: { series: { t_s: r.t_s, q: r.q }, label: r.label }, words: r.label, series: r };
  }
  return { pick: { return_period: 100 }, words: "the 100-year flow" };
}

// ── the run ─────────────────────────────────────────────────────────────────

async function hasGpu() {
  if (!gpuAvailable()) return false;
  try { return Boolean(await navigator.gpu.requestAdapter()); } catch { return false; }
}

function runWorker(job, onFrame) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./floodsim-worker.js?v=__BUILD__", import.meta.url), { type: "module" });
    let chain = Promise.resolve();
    w.onmessage = (e) => {
      const m = e.data;
      if (m.frame) { const f = m.frame, h = m.h; chain = chain.then(() => onFrame(f, h)); }
      else if (m.done) chain.then(() => { w.terminate(); resolve(m.done); });
      else if (m.error) { w.terminate(); reject(new Error(m.error)); }
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "the worker failed")); };
    w.postMessage(job);
  });
}

/** Flood a reach: t = { river_id, lat, lon, name }; choice = { return_period } | { flow_cms } | { forecast } | { routed }. */
export async function floodIt(t, { choice = { return_period: 100 }, by = "rules", log = true, compare = false } = {}) {
  const my = ++running;
  if (!target || Number(target.river_id) !== Number(t.river_id)) { plan = null; runs = []; other = null; setSwipe(false); }
  clearExpo();
  target = { river_id: Number(t.river_id), lat: Number(t.lat), lon: Number(t.lon), name: t.name || t.label || "" };
  openPanel();
  closeCard();
  hideBasinWash(true);
  if (inViews) { inViews = null; if (currentMesh()) clearMesh(); }
  setSwipe(false);
  try {
    if (!plan) {
      setStatus("Reading the river network around this reach and every river's flood flows…");
      plan = await floodsim("plan", { river_id: target.river_id, lat: target.lat, lon: target.lon });
      if (my !== running) return null;
      render();
    }
    const c = await resolveChoice(choice);
    if (my !== running) return null;
    setStatus("Reading the terrain of the valley…");
    const z = await readTerrain(plan.window);
    const gpu = await hasGpu();
    setStatus(`Cutting the grid to the valley and burning in ${Object.keys(plan.reaches).length} rivers…`);
    const edits = sketchEdits(sketches);
    if (edits.length) setStatus(`Cutting the grid to the valley, burning in the rivers and raising ${sketchWords(edits).replace(/^with /, "")}…`);
    const g = await floodsim("prepare", { river_id: target.river_id, choice: c.pick, coarse: !gpu, edits }, z);
    if (my !== running) return null;
    const steady = !g.hours;
    const hours = g.hours || (gpu ? CAP_GPU_H : CAP_CPU_H);
    const win = g.window;
    const run = { id: `${target.river_id}:${Date.now()}`, target: { ...target }, choice, words: c.words, g, win,
      frames: [], hMax: null, engine: gpu ? "WebGPU" : "worker (no WebGPU)", start: c.start || new Date().toISOString().slice(0, 10),
      scale: { lo: 0, hi: 1, sqrt: true }, steady, hours, label: labelOf(choice, g) };
    run.edits = edits.filter((e) => !(g.edits || []).some((x) => x.id === e.id && x.skipped));
    run.editInfo = (g.edits || []).filter((x) => !x.skipped);
    run.key = `${JSON.stringify(choiceAction(choice))}|${editsKey(run.edits)}`;
    if (run.edits.length) run.label = `${run.label} ${sketchWords(run.edits)}`;
    const started = performance.now();
    let first = null;
    let lastDraw = 0;
    const onFrame = async (f, h) => {
      if (my !== running) return;
      run.frames.push(f);
      if (first === null && f.t > 0) first = (performance.now() - started) / 1000;
      const over = depthOverTerrain(h, g.z, g.ground);
      const s = summary(over, g.dx);
      run.scale = { lo: 0, hi: Math.max(run.scale.hi, niceCeil(Math.max(s.max_m * 0.8, 0.5))), sqrt: true };
      const now = performance.now();
      if (now - lastDraw > 60 || f.t === 0) { draw(run, over); lastDraw = now; hideBasinWash(true); }
      setStatus(`Flooding on your ${gpu ? "GPU" : "CPU"}… ${hm(f.t)} in, ${fmt(s.wet_km2, 1)} km² under water`, "busy");
    };
    let res;
    if (gpu) {
      const sim = await createGpuSim({ z: g.z, n: g.n, nx: g.nx, ny: g.ny, dx: g.dx, edges: g.edges },
        { sources: g.sources, tableDtS: g.table_dt_s });
      if (!sim) throw new Error("the GPU went away");
      try {
        const engine = { advance: (tt) => sim.advance(tt, 400), depth: () => sim.readDepth(), faces: () => sim.readFaces() };
        const loop = await runLoop(engine, { nx: g.nx, ny: g.ny, dx: g.dx, hours, steady, onFrame, cancelled: () => my !== running });
        const st = await sim.readState();
        res = { ...loop, steps: st.steps, hMax: await sim.readMaxDepth() };
      } finally { sim.destroy(); }
    } else {
      const done = await runWorker({ z: g.z, n: g.n, nx: g.nx, ny: g.ny, dx: g.dx, edges: g.edges, sources: g.sources,
        tableDtS: g.table_dt_s, hours, steady }, onFrame);
      res = { ...done };
    }
    if (my !== running) return null;
    run.wall_s = (performance.now() - started) / 1000;
    run.first_s = first;
    run.steps = res.steps;
    run.steadyAt = res.steadyAt;
    run.t = run.frames.at(-1).t;
    run.hMax = res.hMax;
    run.over = depthOverTerrain(res.hMax, g.z, g.ground);
    run.stats = summary(run.over, g.dx);
    run.wall = wallContact(run.over, g.nx, g.ny, g.edges, g.sources);
    run.atWall = run.wall.held;
    run.scale = { lo: 0, hi: niceCeil(Math.max(quantile(run.over, 0.99), 0.5)), sqrt: true };
    const before = slotValue("floodsim");
    status = null;
    hideBasinWash(true);
    const prev = runs.filter((r) => sameGrid(r, run) && r.key !== run.key).at(-1) || null;
    runs = [...(prev ? [prev] : []), run];
    show(run, { fit: !compare });
    if (compare && prev) { other = prev; swipe.x = 0.5; setSwipe(true); }
    if (log) {
      logAction({ action: { type: "flood_reach", river_id: run.target.river_id, ...choiceAction(choice) }, before,
        after: valueOf(run), label: `Flood it: ${run.label}`, by, detail: `${fmt(run.stats.wet_km2, 1)} km² under water` });
    }
    announce(`Flood it: ${fmt(run.stats.wet_km2, 1)} square kilometres under water${run.steadyAt ? `, steady after ${hm(run.steadyAt)}` : ""}.`);
    globalThis.__aq.floodIt.last = () => ({ engine: run.engine, wall_s: run.wall_s, first_frame_s: run.first_s, steps: run.steps,
      steady_at_h: run.steadyAt ? run.steadyAt / 3600 : null, hours: run.t / 3600, cells: g.nx * g.ny, dx: g.dx,
      wet_km2: run.stats.wet_km2, max_m: run.stats.max_m, sources: g.sources.length, coarse: g.coarse, at_wall: run.atWall,
      wall: run.wall,
      edits: run.editInfo });
    return run;
  } catch (err) {
    if (my === running) setStatus(`Could not flood this reach: ${err.message}`, "error");
    return null;
  }
}

function quantile(a, p) {
  const v = [];
  for (const x of a) if (x > WET_M) v.push(x);
  if (!v.length) return 0;
  v.sort((x, y) => x - y);
  return v[Math.min(v.length - 1, Math.floor(p * (v.length - 1)))];
}

function labelOf(choice, g) {
  if (choice.return_period) return `${choice.return_period}-year flow`;
  if (choice.flow_cms) return `${q(Number(choice.flow_cms))}`;
  if (choice.forecast === "peak") return "forecast peak";
  if (choice.forecast === "hydrograph") return "15-day forecast";
  if (choice.routed) return `Route it's ${choice.routed}-year wave`;
  return g.label;
}

function choiceAction(choice) {
  if (choice.return_period) return { return_period: Number(choice.return_period) };
  if (choice.flow_cms) return { flow_cms: Number(choice.flow_cms) };
  if (choice.forecast) return { flow: choice.forecast === "peak" ? "forecast_peak" : "forecast" };
  if (choice.routed) return { flow: "route", return_period: Number(choice.routed) };
  return {};
}

const valueOf = (run) => ({ id: run.id, target: run.target, choice: run.choice, label: run.label });
const sameGrid = (a, b) => a.g.window.x0 === b.g.window.x0 && a.g.window.y0 === b.g.window.y0 && a.g.nx === b.g.nx && a.g.ny === b.g.ny;

// ── drawing ─────────────────────────────────────────────────────────────────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensurePicture(run) {
  const { nx, ny } = run.g;
  if (!pic || pic.nx !== nx || pic.ny !== ny) {
    const canvas = document.createElement("canvas");
    canvas.width = nx; canvas.height = ny;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(nx, ny);
    pic = { canvas, ctx, image, pixels: new Uint32Array(image.data.buffer), nx, ny, key: "" };
  }
  const key = `${run.win.x0},${run.win.y0},${nx},${ny},${run.win.zoom}`;
  if (map.getSource(SRC) && pic.key !== key) { removeLayer(); }
  if (!map.getSource(SRC)) {
    map.addSource(SRC, { type: "canvas", canvas: pic.canvas, coordinates: corners(run.win), animate: false });
    // Under the rivers and the gauges, as the other depth rasters.
    map.addLayer({ id: SRC, type: "raster", source: SRC,
      paint: { "raster-opacity": 0.95, "raster-fade-duration": 0, "raster-resampling": "linear" } }, beforeId());
    pic.key = key;
  }
}

function removeLayer() {
  if (!state.mapOk || !map) return;
  if (map.getLayer(SRC)) map.removeLayer(SRC);
  if (map.getSource(SRC)) map.removeSource(SRC);
  if (pic) pic.key = "";
}

/** Paint one depth grid (or, with the swipe on, two: each cell from the run on its side of the line). */
function draw(run, h, otherH = null) {
  if (!state.mapOk || !map) return;
  ensurePicture(run);
  const n = h.length, vals = new Float32Array(n);
  let side = null;
  if (otherH && swipe.on) side = splitSide(run);
  for (let i = 0; i < n; i++) {
    const v = side && side(i) ? otherH[i] : h[i];
    vals[i] = v > WET_M ? v : NaN;
  }
  paintGrid(pic.pixels, vals, run.scale, RAMP);
  pic.ctx.putImageData(pic.image, 0, 0);
  const src = map.getSource(SRC);
  if (src) {
    src.play();
    map.triggerRepaint();
    map.once("render", () => { if (map.getSource(SRC) === src) src.pause(); });
  }
}

// Which cells are left of the swipe line: the screen x of a cell, from the grid's three projected corners.
function splitSide(run) {
  const { nx, ny } = run.g;
  const [nw, ne, , sw] = corners(run.win).map((c) => map.project(c));
  const ax = (ne.x - nw.x) / nx, bx = (sw.x - nw.x) / ny;
  const line = map.getContainer().clientWidth * swipe.x;
  return (i) => {
    const r = Math.floor(i / nx), c = i - r * nx;
    return nw.x + ax * (c + 0.5) + bx * (r + 0.5) < line;
  };
}

function frameDepth(run, i) {
  const f = run.frames[i];
  return f && f.idx ? depthOverTerrain(unpackDepth(f, run.g.nx * run.g.ny), run.g.z, run.g.ground) : run.over;
}

function show(run, { fit = false, step = null } = {}) {
  shown = run;
  const last = run.frames.length - 1;
  const at = step === null ? last : step;
  paintStep(run, at);
  useModelClock({
    times: run.frames.map((f) => f.t), index: at,
    label: (i) => clockLabel(run.frames[i].t, { steady: run.steady, steadyAt: run.steadyAt, last: i === last }),
    summary: clockSummary({ steady: run.steady }),
    onStep: (i) => { if (shown === run) paintStep(run, i); },
    onExit: () => { if (shown === run) paintStep(run, last); },
  });
  registerRow();
  openLegendRow(ROW, true);
  render();
  if (fit) fitWater(run);
}

function paintStep(run, i) {
  run.step = i;
  const h = frameDepth(run, i);
  const o = swipe.on && other ? frameDepth(other, Math.min(i, other.frames.length - 1)) : null;
  draw(run, h, o);
  refreshLegend(ROW);
}

function fitWater(run) {
  if (!state.mapOk || !map) return;
  const box = wetBox(run.over, run.g.nx, run.g.ny);
  if (!box) return;
  const [w, n] = pxLonLat(run.win.x0 + box.c0, run.win.y0 + box.r0, run.win.zoom);
  const [e, s] = pxLonLat(run.win.x0 + box.c1 + 1, run.win.y0 + box.r1 + 1, run.win.zoom);
  map.fitBounds([[w, s], [e, n]], { padding: padding(), maxZoom: 15, duration: 900 });
}

function padding() {
  const pad = { ...cameraPadding() };
  const el = panel();
  const frame = map.getContainer().getBoundingClientRect();
  if (el && !el.hidden) {
    const r = el.getBoundingClientRect();
    if (phone()) pad.bottom = Math.max(pad.bottom, Math.round(frame.bottom - r.top) + 12);
    else pad.right = Math.max(pad.right, Math.round(frame.right - r.left) + 16);
  }
  pad.top = Math.max(pad.top, 60);
  pad.bottom = Math.max(pad.bottom, 70);
  return pad;
}

// The purple basin wash, the blue catchment fill and the traced river would tint or cross the flood: hidden while
// one is shown (and put back after). Called again as the run settles, for layers added meanwhile.
const WASH = ["basins12-up", "catchment-fill", "river-trace-casing", "river-trace-line", "river-hl-casing", "river-hl-line"];
function hideBasinWash(hide) {
  if (!state.mapOk || !map) return;
  if (hide) {
    washHidden = washHidden || {};
    for (const id of WASH) {
      if (!map.getLayer(id) || id in washHidden) continue;
      washHidden[id] = map.getLayoutProperty(id, "visibility") || "visible";
      map.setLayoutProperty(id, "visibility", "none");
    }
  } else if (washHidden) {
    for (const [id, vis] of Object.entries(washHidden)) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
    washHidden = null;
  }
}

// ── the flood views of #619 to #622, through the model viewer ───────────────

function runFile(run) {
  if (run.file) return run.file;
  const { bytes } = ugridFile({ win: run.win, bed: run.g.z, ground: run.g.ground, hMax: run.hMax, frames: run.frames, start: run.start,
    title: `Flood it, ${run.target.name || `reach ${run.target.river_id}`}, ${run.label}` });
  run.file = new File([bytes], `Flood it, ${run.label}.nc`, { type: "application/x-netcdf" });
  return run.file;
}

async function openViews(what) {
  const run = shown;
  if (!run) return;
  setSwipe(false);
  if (inViews !== run || !currentMesh()) {
    setStatus("Opening the run in the flood views (the first time loads Python for them)…", "busy");
    inViews = run;
    removeLayer();
    releaseModelClock();
    await openModelFile(runFile(run), null, null, { sample: { credit: FLOODSIM_CREDIT, link: DOCS } });
    if (!currentMesh()) { inViews = null; setStatus("The flood views could not open this run.", "error"); show(run); return; }
    setStatus("");
    render();
  }
  if (what === "section" || what === "exposure") { runMeshTool(what); return; }
  for (let k = 0; k < 120; k++) {           // the lenses are made once per file, in the model viewer's worker
    if (setMeshLens(what)) return;
    await new Promise((r) => setTimeout(r, 500));
  }
}

function backFromViews() {
  if (!inViews) return;
  const run = inViews;
  inViews = null;
  if (shown === run && panel() && !panel().hidden) show(run);
}

// ── the swipe: two runs on the same grid ───────────────────────────────────

function setSwipe(on) {
  swipe.on = Boolean(on) && Boolean(other) && Boolean(shown);
  const el = $("fi-swipe");
  if (el) el.hidden = !swipe.on;
  if (swipe.on) placeSwipe();
  if (shown) paintStep(shown, shown.step ?? shown.frames.length - 1);
  render();
}

function placeSwipe() {
  const el = $("fi-swipe");
  if (!el || !swipe.on) return;
  el.style.setProperty("--x", `${swipe.x * 100}%`);
  el.querySelector(".wi-sw-line").style.left = `${swipe.x * 100}%`;
  el.querySelector(".wi-sw-handle").style.left = `${swipe.x * 100}%`;
  el.querySelector(".wi-sw-before").textContent = other ? other.label : "";
  el.querySelector(".wi-sw-after").textContent = shown ? shown.label : "";
}

function initSwipe() {
  const el = $("fi-swipe");
  if (!el) return;
  const handle = el.querySelector(".wi-sw-handle");
  let frame = 0;
  const redraw = () => { placeSwipe(); if (!frame) frame = requestAnimationFrame(() => { frame = 0; if (shown) paintStep(shown, shown.step); }); };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const box = el.getBoundingClientRect();
    const move = (ev) => { swipe.x = Math.min(0.95, Math.max(0.05, (ev.clientX - box.left) / box.width)); redraw(); };
    const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
  handle.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowLeft" ? -0.05 : e.key === "ArrowRight" ? 0.05 : 0;
    if (!d) return;
    e.preventDefault();
    swipe.x = Math.min(0.95, Math.max(0.05, swipe.x + d));
    redraw();
  });
  if (map) map.on("move", () => { if (swipe.on && shown) redraw(); });
}

// ── the panel ───────────────────────────────────────────────────────────────

function ensurePanel() {
  const el = panel();
  if (!el || el.dataset.ready) return el;
  el.dataset.ready = "1";
  el.addEventListener("click", onPanelClick);
  el.addEventListener("submit", (e) => {
    e.preventDefault();
    if (e.target.classList.contains("fi-crest-form")) { if (pending) setCrest(Number(el.querySelector("#fi-crest-num").value), { commit: true }); return; }
    const v = Number(el.querySelector("#fi-q").value);
    if (v > 0 && target) floodIt(target, { choice: { flow_cms: v } });
  });
  el.addEventListener("input", (e) => {
    if (e.target.id === "fi-crest-range" || e.target.id === "fi-crest-num") setCrest(Number(e.target.value), { commit: false });
  });
  el.addEventListener("change", (e) => {
    if (e.target.id === "fi-crest-range" || e.target.id === "fi-crest-num") { setCrest(Number(e.target.value), { commit: true }); return; }
    if (e.target.id === "fi-hydro" && e.target.value && target) {
      const v = e.target.value;
      floodIt(target, { choice: v === "forecast" ? { forecast: "hydrograph" } : { routed: Number(v.split(":")[1]) } });
    }
  });
  return el;
}

function openPanel() {
  const el = ensurePanel();
  if (!el) return;
  el.hidden = false;
  el.classList.remove("done");
  render();
}

function setStatus(text, kind = "") {
  status = text ? { text, kind } : null;
  render();
}

function chip(T, cur, curve) {
  const v = flowFor(curve, T);
  const on = cur && Number(cur.return_period) === T;
  return `<button type="button" class="fi-chip${on ? " on" : ""}" data-rp="${T}" aria-pressed="${on}" ` +
    `title="${escapeHtml(v ? `${q(v)} at this reach, ${designFlowWords(curve)}` : "")}">` +
    `${T}<small>-yr</small>${v ? `<em>${fmt(v, 0)}</em>` : ""}</button>`;
}

function render() {
  const el = panel();
  if (!el || el.hidden) return;
  const curve = curveOf(plan);
  const run = shown && target && shown.target.river_id === target.river_id ? shown : null;
  const cur = run ? run.choice : null;
  const name = target ? (target.name || `River reach ${target.river_id}`) : "River reach";
  const chips = CHOICES.map((T) => chip(T, cur, curve)).join("");
  const fc = cur && cur.forecast === "peak";
  const fl = run ? flowLine(run, q) : null;
  const flowHtml = run ? `<p class="fi-flow"><b>${escapeHtml(cap(fl.words))}</b>${escapeHtml(fl.rest)}</p>` :
    (curve ? `<p class="fi-flow muted">Flows: ${escapeHtml(designFlowWords(curve))}.</p>` : "");
  const st = status ? `<p class="wi-status${status.kind === "error" ? " error" : ""}" role="status">${status.kind === "busy" || /…$/.test(status.text) ? '<span class="spinner" aria-hidden="true"></span>' : ""}${escapeHtml(status.text)}</p>` : "";
  let result = "";
  if (run && !status) {
    const s = run.stats;
    const head = run.steady
      ? (run.steadyAt ? `steady after ${hm(run.steadyAt)}` : `not yet steady at the ${hm(run.t)} cap`)
      : `over the ${hm(run.t)} hydrograph`;
    const comparable = runs.filter((r) => r !== run && r.g.nx === run.g.nx && r.g.window.x0 === run.g.window.x0);
    const cmp = comparable.length
      ? `<button type="button" class="fd-go${swipe.on ? " on" : ""}" data-act="swipe" aria-pressed="${swipe.on}" title="A swipe between this run and the ${escapeHtml(comparable.at(-1).label)}">Compare with ${escapeHtml(comparable.at(-1).label)}</button>` : "";
    result = `<div class="fi-result">
      <p class="fi-head"><b>${fmt(s.wet_km2, 1)} km²</b> under water, ${escapeHtml(head)}; deepest ${fmt(s.max_m, 1)} m.</p>
      ${run.atWall ? `<p class="fi-wall" role="note">${escapeHtml(wallWords(run.wall, run.g.dx))}</p>` : ""}
      <p class="mv-actions fi-views"><span>Look closer</span>
        <button type="button" class="fd-go" data-act="hazard" title="Hazard H1 to H6 from depth and speed (AIDR Guideline 7-3)">Hazard</button>
        <button type="button" class="fd-go" data-act="arrival" title="When each place first has 10 cm of water">Arrival</button>
        <button type="button" class="fd-go" data-act="section" title="Drag a line across the flood: the ground and the water along it">Section</button>
        <button type="button" class="fd-go" data-act="exposure" title="Buildings, roads and people under the water (Overture, GHS-POP)">What gets flooded</button>
        ${cmp}</p>
      <details class="fi-how"><summary>How it ran, and where it differs from JRC</summary>
        <p>${escapeHtml(run.engine)}: ${fmt(run.steps, 0)} steps on ${fmt(run.g.nx * run.g.ny, 0)} cells of ${fmt(run.g.dx, 1)} m in ${fmt(run.wall_s, 1)} s${run.first_s ? ` (first frame after ${fmt(run.first_s, 1)} s)` : ""}.
        ${run.g.coarse ? "Without WebGPU the worker runs cells twice as large and stops at 48 h." : `A steady run stops when the wet area and the volume each change by under 1% in an hour, or at ${CAP_GPU_H} h.`}
        The grid follows the river and is cut to its valley; walls all round but where the river leaves (a normal-depth outfall).
</p>
        <p>${escapeHtml(JRC_LINE)} <a href="${DOCS}" target="_blank" rel="noopener">The checks</a></p>
      </details>
    </div>`;
  }
  const picker = `<div class="fi-flows" role="group" aria-label="The flow">${chips}
      <button type="button" class="fi-chip fi-wide${fc ? " on" : ""}" data-act="peak" aria-pressed="${fc}" title="The highest flow of GEOGLOWS's 15-day forecast here">Forecast peak</button>
    </div>
    <form class="fi-more">
      <select id="fi-hydro" aria-label="A hydrograph instead of a steady flow">
        <option value="">Hydrograph…</option>
        <option value="forecast"${cur && cur.forecast === "hydrograph" ? " selected" : ""}>The 15-day forecast</option>
        <option value="route:100"${cur && cur.routed ? " selected" : ""}>Route it's 100-year wave</option>
      </select>
      <label class="fi-num"><input id="fi-q" type="number" min="1" step="any" inputmode="decimal" placeholder="m³/s" aria-label="A flow in m³/s"${cur && cur.flow_cms ? ` value="${Number(cur.flow_cms)}"` : ""} /><button type="submit" class="btn small">Run</button></label>
    </form>`;
  panel().innerHTML = `<header class="wi-head"><div><p class="wi-kicker">Flood it · screening flood</p>
      <h2 id="fi-title" class="wi-title">${escapeHtml(name)}</h2></div>
      <button type="button" class="wi-close" data-act="close" aria-label="Close Flood it" title="Close and take the flood off the map">×</button></header>
    ${picker}${flowHtml}${st}${result}${run && !status ? sketchBlock(run) : ""}
    <p class="fi-credit">${escapeHtml(LABEL)}. ${inViews && currentMesh() ? '<button type="button" class="link-btn" data-act="back">Back to Flood it</button> · ' : ""}<button type="button" class="link-btn" data-act="clear">Clear</button></p>`;
  panel().classList.toggle("done", Boolean(run && !status));
  panel().classList.toggle("crest", Boolean(run && !status && pending));
  panel().classList.toggle("mini", Boolean(run && inViews === run && currentMesh()));
}

const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);

function onPanelClick(e) {
  const b = e.target.closest("button");
  if (!b || !target) return;
  const rp = b.dataset.rp, act = b.dataset.act;
  if (rp) { floodIt(target, { choice: { return_period: Number(rp) } }); return; }
  if (!act) return;
  if (act === "peak") floodIt(target, { choice: { forecast: "peak" } });
  else if (act === "close" || act === "clear") clearByUndo({ close: act === "close" });
  else if (act === "back") { if (currentMesh()) clearMesh(); }
  else if (act === "swipe") { other = runs.filter((r) => r !== shown).at(-1) || null; setSwipe(!swipe.on); }
  else if (["hazard", "arrival", "section", "exposure"].includes(act)) openViews(act);
  else if (act === "draw-levee" || act === "draw-breach") drawSketch(act === "draw-levee" ? "levee" : "breach");
  else if (act === "sk-edit") editSketch(Number(b.dataset.id));
  else if (act === "sk-remove") removeSketch(Number(b.dataset.id));
  else if (act === "sk-done") { pending = null; selected = null; drawSketches(sketches); render(); }
  else if (act === "sk-suggest" && pending) setCrest(pending.profile.suggested_crest_m, { commit: true });
  else if (act === "rerun") runAgain();
  else if (act === "expo") exposureBeforeAfter();
}

// ── the legend row ──────────────────────────────────────────────────────────

function registerRow() {
  registerLegendRow({
    id: ROW,
    title: "Flood it",
    mark: () => `<i class="fd-mark" style="background:${rampCss()}"></i>`,
    summary: () => (shown ? `${shown.label}, depth` : ""),
    on: () => Boolean(shown),
    shown: () => Boolean(shown) && !inViews,
    toggle: (on) => { if (!on) clearByUndo({ close: false }); },
    body: () => {
      if (!shown) return "";
      const sc = shown.scale;
      const t = ticks(sc).map((k) => `<span style="left:${k.at}%">${escapeHtml(k.label)}</span>`).join("");
      const f = shown.frames[shown.step ?? shown.frames.length - 1];
      return `<p class="fd-tag">${escapeHtml(depthTag(shown.label, f ? f.t : null, { steady: shown.steady }))}</p>` +
        `<div class="fd-ramp" role="img" aria-label="Depth from 0 to ${sc.hi} m, light to deep blue" style="background:${rampCss()}"></div>` +
        `<div class="fd-ticks" aria-hidden="true">${t}<em>m</em></div>` +
        `<p class="fd-src">${escapeHtml(FLOODSIM_CREDIT)}</p>`;
    },
    act: () => {},
  });
}

// ── levees and breaches (#663) ──────────────────────────────────────────────

const sketchValue = (s) => (s ? { kind: s.kind, line: s.line, ...(s.kind === "levee" ? { crest: s.crest } : { invert: s.invert ?? null }) } : null);
const sketchName = (s) => (s.kind === "levee" ? `levee at ${fmt(s.crest, 1)} m` : "breach");

// The undo log's writer for one drawn line: put it back as it was (or take it off).
function putSketch(id, value) {
  sketches = sketches.filter((x) => x.id !== id);
  if (value) sketches.push(checkSketch({ id, ...value }));
  if (!value && pending && pending.id === id) pending = null;
  if (!value && selected === id) selected = null;
  if (value && pending && pending.id === id && value.crest !== undefined) pending.crest = value.crest;
  drawSketches(sketches, { selected });
  render();
}

function logSketch(id, before, after, { by = "rules", label = "", command = "" } = {}) {
  registerSlotWriter(`sketch:${id}`, (v) => putSketch(id, v || null));
  const s = after || before;
  const what = s.kind === "levee" ? "a levee" : "a breach";
  const words = label || (!before ? `Drew ${what}${s.kind === "levee" ? ` at ${fmt(s.crest, 1)} m` : ""}`
    : !after ? `Removed ${what}` : (before.crest !== after.crest ? `Set ${what}'s crest to ${fmt(after.crest, 1)} m` : `Moved ${what}`));
  logAction({ action: { type: s.kind === "levee" ? "draw_levee" : "draw_breach", sketch: id }, before, after, by, command, label: words });
}

async function profileOf(line, opts = {}) {
  const run = shown;
  return floodsim("profile", { river_id: target.river_id, edits: run ? run.edits : [], line,
    ...(opts.freeboard_m !== undefined && opts.freeboard_m !== null ? { freeboard_m: Number(opts.freeboard_m) } : {}) },
  run && run.hMax ? Float32Array.from(run.hMax) : null);
}

/**
 * Draw a levee or a breach on the valley that is flooded now: `line` ([[lon, lat], ...]; none or "drawn": the reader
 * draws it), a levee's `crest` (m above mean sea level) or `freeboard_m` above the flood along the line (the code
 * suggests the crest), a breach's `invert` (none: the natural ground). `run` floods the valley again at once.
 */
export async function drawSketch(kind, opts = {}) {
  if (!shown || !target) throw new Error("flood a river first (Flood it), then draw on its valley");
  let line = Array.isArray(opts.line) ? opts.line : null;
  if (!line) {
    pending = null;
    selected = null;
    render();
    if (panel()) panel().classList.add("drawing");
    try { line = await startLineDraw(kind); } finally { if (panel()) panel().classList.remove("drawing"); }
  }
  if (!line) { render(); return null; }
  let profile;
  try {
    setStatus(kind === "levee" ? "Reading the ground and the flood along the line…" : "Reading the ground along the line…", "busy");
    profile = await profileOf(line, opts);
  } catch (err) {
    setStatus(`Could not place the ${kind}: ${err.message}`, "error");
    return null;
  }
  status = null;
  const id = nextSketch++;
  const s = checkSketch({ id, kind, line, ...(kind === "levee" ? { crest: opts.crest ?? profile.suggested_crest_m } : { invert: opts.invert ?? null }) });
  sketches.push(s);
  logSketch(id, null, sketchValue(s), opts);
  selected = id;
  pending = { id, profile, crest: s.crest ?? null };
  drawSketches(sketches, { selected });
  render();
  if (phone()) fitLine(line);   // the sheet came back over the map: keep the line in sight above it
  announce(kind === "levee" ? `A levee at ${fmt(s.crest, 1)} metres, ${fmt(profile.length_m, 0)} metres long. Set its crest, then Run again.`
    : "A breach. Run again to see the water come through it.");
  if (opts.run) await runAgain({ by: opts.by });
  return s;
}

function fitLine(line) {
  if (!state.mapOk || !map || !line || !line.length) return;
  const xs = line.map((p) => p[0]), ys = line.map((p) => p[1]);
  map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: padding(), maxZoom: 16, duration: 500 });
}

async function editSketch(id) {
  const s = sketches.find((x) => x.id === id);
  if (!s || !shown) return;
  selected = id;
  drawSketches(sketches, { selected });
  try {
    const profile = await profileOf(s.line);
    pending = { id, profile, crest: s.crest ?? null };
  } catch (err) {
    pending = null;
    setStatus(`Could not read along the line: ${err.message}`, "error");
    return;
  }
  render();
}

function removeSketch(id) {
  const s = sketches.find((x) => x.id === id);
  if (!s) return;
  sketches = sketches.filter((x) => x.id !== id);
  if (pending && pending.id === id) pending = null;
  if (selected === id) selected = null;
  logSketch(id, sketchValue(s), null);
  drawSketches(sketches, { selected });
  render();
}

async function moveSketch(id, line) {
  const s = sketches.find((x) => x.id === id);
  if (!s) return;
  const before = sketchValue(s);
  Object.assign(s, checkSketch({ ...s, line }));
  logSketch(id, before, sketchValue(s));
  drawSketches(sketches, { selected });
  if (pending && pending.id === id) await editSketch(id);
  else render();
}

function setCrest(v, { commit = false } = {}) {
  if (!pending || !Number.isFinite(v)) return;
  const s = sketches.find((x) => x.id === pending.id);
  if (!s || s.kind !== "levee") return;
  v = Math.round(v * 10) / 10;
  pending.crest = v;
  const el = panel();
  if (el) {
    const d = profilePaths(pending.profile, v);
    const c = el.querySelector(".fi-prof .c");
    if (c) c.setAttribute("d", d.crest);
    const w = el.querySelector(".fi-crest-words");
    if (w) w.innerHTML = crestWords(pending.profile, v);
    for (const id of ["fi-crest-range", "fi-crest-num"]) {
      const inp = el.querySelector(`#${id}`);
      if (inp && document.activeElement !== inp) inp.value = String(v);
    }
  }
  if (commit && v !== s.crest) {
    const before = sketchValue(s);
    s.crest = v;
    logSketch(s.id, before, sketchValue(s));
    drawSketches(sketches, { selected });
    render();
  }
}

/** Flood the same valley at the same flow again, with the lines drawn now; the swipe then compares the two runs. */
export async function runAgain({ by = "rules" } = {}) {
  if (!target || !shown) return null;
  pending = null;
  selected = null;
  drawSketches(sketches);
  return floodIt(target, { choice: shown.choice, by, compare: true });
}

function crestWords(p, crest) {
  const [h0, h1] = crestHeights(p, crest);
  const flood = p.highest_water_m === null ? "no flood on the line in this run"
    : `the flood up to <b>${fmt(p.highest_water_m, 1)} m</b>`;
  return `Ground ${fmt(p.lowest_ground_m, 1)} to ${fmt(p.highest_ground_m, 1)} m, ${flood}; crest <b>${fmt(crest, 1)} m</b>, ` +
    `${fmt(h0, 1)} to ${fmt(h1, 1)} m above the ground.`;
}

function crestControl() {
  const s = pending && sketches.find((x) => x.id === pending.id);
  if (!s) return "";
  const p = pending.profile;
  const crest = s.kind === "levee" ? pending.crest ?? s.crest : null;
  const d = profilePaths(p, crest);
  const svg = `<svg class="fi-prof" viewBox="0 0 260 84" preserveAspectRatio="none" role="img"
      aria-label="The ground along the line (brown), the flood level (blue) and the crest (amber)">
      <path class="g" d="${d.ground}"/>${d.now ? `<path class="now" d="${d.now}"/>` : ""}<path class="w" d="${d.water}"/>
      <path class="c" d="${d.crest}"/></svg>
    <p class="fi-prof-ax" aria-hidden="true"><span>${fmt(d.hi, 1)} m</span><span>${fmt(p.length_m, 0)} m along the line</span><span>${fmt(d.lo, 1)} m</span></p>`;
  if (s.kind === "breach") {
    return `<div class="fi-crest">${svg}<p class="fi-crest-words">The gap goes back down to the natural ground, ` +
      `${fmt(p.lowest_ground_m, 1)} to ${fmt(p.highest_ground_m, 1)} m, over ${p.cells.length} cells.</p>` +
      `<p class="fi-crest-row"><button type="button" class="link-btn" data-act="sk-done">Done</button></p></div>`;
  }
  const top = Math.max(p.highest_ground_m, p.highest_water_m ?? p.highest_ground_m, crest);
  const lo = Math.floor(p.lowest_ground_m), hi = Math.ceil(top + 3);
  return `<div class="fi-crest">${svg}
    <p class="fi-crest-words">${crestWords(p, crest)}</p>
    <form class="fi-crest-form"><label for="fi-crest-range">Crest</label>
      <input id="fi-crest-range" type="range" min="${lo}" max="${hi}" step="0.1" value="${crest}" aria-label="The levee's crest, m above mean sea level" />
      <input id="fi-crest-num" type="number" step="0.1" inputmode="decimal" value="${crest}" aria-label="The crest in m" /><span>m</span></form>
    <p class="fi-crest-row"><button type="button" class="link-btn" data-act="sk-suggest" title="The highest flood along the line plus ${fmt(p.freeboard_m, 1)} m">Flood + ${fmt(p.freeboard_m, 1)} m</button>
      <button type="button" class="link-btn" data-act="sk-done">Done</button>
      <small>m above mean sea level, the terrain's datum</small></p>
  </div>`;
}

function sketchBlock(run) {
  const now = sketchEdits(sketches);
  const changed = editsKey(now) !== editsKey(run.edits);
  const items = [...sketches].sort((a, b) => a.id - b.id).map((s) => `<li class="fi-sk-item${s.id === selected ? " on" : ""}">
      <i class="fi-sk-mark ${s.kind}" aria-hidden="true"></i><span>${s.kind === "levee" ? `Levee, crest ${fmt(s.crest, 1)} m` : `Breach${s.invert !== undefined && s.invert !== null ? `, down to ${fmt(s.invert, 1)} m` : ""}`}</span>
      <button type="button" class="link-btn" data-act="sk-edit" data-id="${s.id}" aria-label="Edit the ${escapeHtml(sketchName(s))}">Edit</button>
      <button type="button" class="link-btn" data-act="sk-remove" data-id="${s.id}" aria-label="Remove the ${escapeHtml(sketchName(s))}">Remove</button></li>`).join("");
  const dams = (run.editInfo || []).filter((e) => e.kind === "levee" && e.channel_cells > 0);
  const damNote = dams.length && !changed ? `<p class="fi-sk-note">A levee crosses a river at ${dams.reduce((a, e) => a + e.channel_cells, 0)} cells and dams it there.</p>` : "";
  const rerun = changed ? `<p class="fi-sk-run"><button type="button" class="btn small primary" data-act="rerun">Run again ${escapeHtml(now.length ? sketchWords(now) : "without them")}</button></p>` : "";
  const pair = expoPair();
  const expoBtn = pair && !(expo && (expo.res || expo.status)) ? `<button type="button" class="fd-go" data-act="expo" title="Buildings, roads and people under each run's water (Overture Maps, GHS-POP), and the change">What gets flooded, before and after</button>` : "";
  return `<div class="fi-sketch">
    <p class="mv-actions fi-sk-tools"><span>Change the ground</span>
      <button type="button" class="fd-go" data-act="draw-levee" title="Draw a line on the map: its cells are raised to a crest you set">Draw a levee</button>
      <button type="button" class="fd-go" data-act="draw-breach" title="Draw along a levee where it fails: its cells go back down to the natural ground">Draw a breach</button></p>
    ${items ? `<ul class="fi-sk-list">${items}</ul>` : ""}${pending ? crestControl() : ""}${damNote}${rerun}
    ${expoBtn || expo ? `<div class="fi-expo">${expoBtn}${expoHtml()}</div>` : ""}
  </div>`;
}

// ── before and after: what gets flooded ─────────────────────────────────────

let expoRun = 0;

function expoPair() {
  if (runs.length !== 2 || !shown || !runs.includes(shown) || !sameGrid(runs[0], runs[1])) return null;
  if (editsKey(runs[0].edits) === editsKey(runs[1].edits)) return null;
  return runs;
}

function clearExpo() {
  expoRun++;
  expo = null;
  if (!state.mapOk || !map) return;
  for (const id of [`${EXPO}-fill`, `${EXPO}-line`, `${EXPO}-road`, `${EXPO}-box`]) if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(EXPO)) map.removeSource(EXPO);
}

function drawExpo(res, box) {
  if (!state.mapOk || !map) return;
  const data = exposureChangeFeatures(res);
  if (box) {
    const [w, s, e, n] = box;
    data.features.unshift({ type: "Feature", properties: { change: "box" },
      geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] } });
  }
  if (map.getSource(EXPO)) { map.getSource(EXPO).setData(data); return; }
  map.addSource(EXPO, { type: "geojson", data });
  const color = ["match", ["get", "change"], "kept_dry", "#1a7f37", "#cf222e"];
  map.addLayer({ id: `${EXPO}-box`, type: "line", source: EXPO, filter: ["==", ["get", "change"], "box"],
    paint: { "line-color": "#0f1c26", "line-width": 1.2, "line-opacity": 0.8, "line-dasharray": [3, 2] } });
  map.addLayer({ id: `${EXPO}-road`, type: "line", source: EXPO,
    filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "change"], "box"]],
    paint: { "line-color": color, "line-width": 3, "line-opacity": 0.9 } });
  map.addLayer({ id: `${EXPO}-fill`, type: "fill", source: EXPO, filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "fill-color": color, "fill-opacity": 0.75 } });
  map.addLayer({ id: `${EXPO}-line`, type: "line", source: EXPO, filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "line-color": "#ffffff", "line-width": 0.6 } });
}

function expoHtml() {
  if (!expo) return "";
  if (expo.status) {
    return `<p class="wi-status${expo.error ? " error" : ""}" role="status">${expo.error ? "" : '<span class="spinner" aria-hidden="true"></span>'}${escapeHtml(expo.status)}</p>`;
  }
  const c = expo.res, [b, a] = expo.pair;
  const rows = exposureChangeRows(c).map((r) => `<tr><th scope="row">${r.label}${r.unit ? ` <small>${r.unit}</small>` : ""}</th>` +
    `<td>${r.before}</td><td>${r.after}</td><td class="${r.sign < 0 ? "down" : r.sign > 0 ? "up" : ""}">${r.change}</td></tr>`).join("");
  const src = sourceLines(c).map((l) => escapeHtml(l.text)).join(". ");
  const errs = Object.keys(c.errors || {}).length ? ` Could not read: ${escapeHtml(Object.keys(c.errors).join(", "))}.` : "";
  return `<table class="fi-expo-t"><caption>Under the water at each run's largest extent, inside the dashed box where the two runs differ</caption>
      <thead><tr><th></th><th scope="col" title="${escapeHtml(b.label)}">Before</th><th scope="col" title="${escapeHtml(a.label)}">After</th><th scope="col">Change</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <p class="fi-expo-key"><span><i class="kd" aria-hidden="true"></i>kept dry: ${c.kept_dry.length} buildings</span>
      <span><i class="nf" aria-hidden="true"></i>newly flooded: ${c.newly_flooded.length}</span></p>
    <p class="fi-credit">Estimates, as the flood views' own count: a building counts when the water reaches its centre or a corner. ${src}.${errs}</p>`;
}

async function exposureBeforeAfter() {
  const pair = expoPair();
  if (!pair || !target) return null;
  const [b, a] = pair;
  const my = ++expoRun;
  expo = { status: "Reading the buildings and roads (Overture Maps) and the people (GHS-POP) under both floods…", pair };
  render();
  try {
    const n = a.over.length;
    const box = changeBox(b.over, a.over, a.g.nx, a.g.ny);
    if (!box) throw new Error("the two runs flood the same cells");
    const [w, nn] = pxLonLat(a.win.x0 + box.c0, a.win.y0 + box.r0, a.win.zoom);
    const [e, s] = pxLonLat(a.win.x0 + box.c1 + 1, a.win.y0 + box.r1 + 1, a.win.zoom);
    const read = await callLight("flood_exposure", { bbox: [w, s, e, nn] }, { priority: 2 });
    if (my !== expoRun) return null;
    expo = { status: "Counting before and after, in your browser…", pair };
    render();
    const both = new Float32Array(2 * n);
    both.set(b.over, 0);
    both.set(a.over, n);
    const res = await floodsim("exposure", { river_id: target.river_id, edits: a.edits, read }, both);
    if (my !== expoRun) return null;
    expo = { res, pair, box: [w, s, e, nn] };
    drawExpo(res, expo.box);
    render();
    const bl = res.buildings || { change: 0 };
    announce(`Before and after: ${Math.abs(bl.change)} ${bl.change <= 0 ? "fewer" : "more"} buildings under water; ${res.kept_dry.length} kept dry, ${res.newly_flooded.length} newly flooded.`);
    globalThis.__aq.floodIt.expo = () => res;
    return res;
  } catch (err) {
    if (my === expoRun) { expo = { status: `Could not count what gets flooded: ${err.message}`, error: true, pair }; render(); }
    return null;
  }
}

/**
 * What is under the run on the map (Replay an event, #659): the buildings, the road length and the people under its
 * largest depth, counted by the package as the flood views count them (compare_exposure with a dry grid before),
 * inside the wet cells' box (at most 900 km², around the reach when larger). Returns the package's answer.
 */
export async function exposureOfShown() {
  const run = shown;
  if (!run || !target) return null;
  const n = run.over.length;
  const dry = new Float32Array(n);
  const box = changeBox(dry, run.over, run.g.nx, run.g.ny);
  if (!box) return { buildings: { before: 0, after: 0, change: 0 }, roads_km: { before: 0, after: 0, change: 0 },
    people: { before: 0, after: 0, change: 0 }, kept_dry: [], newly_flooded: [], sources: {} };
  let [w, nn] = pxLonLat(run.win.x0 + box.c0, run.win.y0 + box.r0, run.win.zoom);
  let [e, s] = pxLonLat(run.win.x0 + box.c1 + 1, run.win.y0 + box.r1 + 1, run.win.zoom);
  const kx = 111.32 * Math.cos(((s + nn) / 2) * Math.PI / 180);
  if ((e - w) * kx * (nn - s) * 110.57 > 880) {
    const half = Math.sqrt(880) / 2, lon = Number(target.lon), lat = Number(target.lat);
    w = Math.max(w, lon - half / kx); e = Math.min(e, lon + half / kx);
    s = Math.max(s, lat - half / 110.57); nn = Math.min(nn, lat + half / 110.57);
  }
  const read = await callLight("flood_exposure", { bbox: [w, s, e, nn] }, { priority: 2 });
  const both = new Float32Array(2 * n);
  both.set(run.over, n);
  const res = await floodsim("exposure", { river_id: target.river_id, edits: run.edits, read }, both);
  return { ...res, bbox: [w, s, e, nn] };
}

/** The buildings under the run on the map, red, from exposureOfShown's answer (a replay's exposure stop, #659). */
export function drawExposureOf(res) { if (res && res.newly_flooded) drawExpo(res, null); }

/** Ask the map's draw_levee and draw_breach (aquascope.map_commands): the model picks the line and the crest, this
 * draws it, computes nothing itself, and floods again. */
export async function sketchAction(a, { by = "rules", command = "", label = "" } = {}) {
  const kind = a.type === "draw_breach" ? "breach" : "levee";
  return drawSketch(kind, { line: Array.isArray(a.line) ? a.line : null, crest: a.crest ?? null, freeboard_m: a.freeboard_m ?? null,
    invert: a.invert ?? null, by, command, label, run: true });
}

// ── clearing, the log and the AI ────────────────────────────────────────────

function clearMap({ close = true } = {}) {
  running++;
  cancelDraw();
  clearExpo();
  pending = null;
  shown = null;
  other = null;
  setSwipe(false);
  removeLayer();
  releaseModelClock();
  hideBasinWash(false);
  if (inViews && currentMesh()) { inViews = null; clearMesh(); }
  inViews = null;
  refreshLegend();
  if (close && panel()) panel().hidden = true;
  else render();
}

function clearByUndo({ close = true } = {}) {
  const before = slotValue("floodsim");
  clearMap({ close });
  if (before) logAction({ action: { type: "flood_reach", river_id: before.target.river_id, return_period: null }, before, after: null, label: "Flood it cleared" });
}

// The undo log's writer: put a run back (it is still in memory), run it again, or clear.
function paintFromLog(value) {
  if (!value) { clearMap(); return; }
  const run = runs.find((r) => r.id === value.id);
  if (run) { target = run.target; openPanel(); hideBasinWash(true); show(run, { fit: true }); return; }
  floodIt(value.target, { choice: value.choice, log: false });
}

/** Ask the map's flood_reach (aquascope.map_commands): the model picks the reach and the flow, this runs it. */
export async function floodReach(reach, a = {}, { by = "rules" } = {}) {
  let choice = { return_period: 100 };
  if (a.flow_cms) choice = { flow_cms: Number(a.flow_cms) };
  else if (a.flow === "forecast_peak") choice = { forecast: "peak" };
  else if (a.flow === "forecast") choice = { forecast: "hydrograph" };
  else if (a.flow === "route") choice = { routed: Number(a.return_period || 100) };
  else if (a.return_period) choice = { return_period: Number(a.return_period) };
  return floodIt(reach, { choice, by });
}

export function initFloodIt() {
  actions.openFloodIt = (t) => floodIt(t);
  actions.floodReach = floodReach;
  actions.sketchFlood = sketchAction;
  onSketchClick((id) => { if (shown) editSketch(id); });
  onSketchMove((id, line) => moveSketch(id, line));
  registerSlotWriter("floodsim", (value) => paintFromLog(value || null));
  onMeshCleared(() => backFromViews());
  if (map) {
    initSwipe();
    map.on("style.load", () => setTimeout(() => { pic = null; if (shown && !inViews) paintStep(shown, shown.step ?? 0); }, 0));
  }
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.floodIt = { run: floodIt, shown: () => shown, runs: () => runs, plan: () => plan, openViews,
    compare: () => { other = runs.filter((r) => r !== shown).at(-1) || null; setSwipe(true); }, swipe: (x) => { swipe.x = x; placeSwipe(); if (shown) paintStep(shown, shown.step); },
    // levees and breaches (#663)
    draw: drawSketch, runAgain, sketches: () => sketches, edit: editSketch, crest: (v) => setCrest(v, { commit: true }),
    exposure: exposureBeforeAfter, expo: () => (expo && expo.res) || null, remove: removeSketch };
}
