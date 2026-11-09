// Draw a reservoir (#662). From the sketchpad's menu: draw a lake across a river (tap its corners, or drag a box).
// The worker finds the river and the dam (aquascope.sketch.reservoir.plan: the GEOGLOWS reaches drawn inside, the
// dam at the exit of the river through the lake) and the terrain window; the page reads the Terrarium tiles for it
// (Flood it's own reader) and the worker turns them into the lake's stage-storage curve. Set the spillway's crest,
// the top of the dam and the spillway's width, pick the flood (the T-year flow at the dam, a peak, or the wave or
// storm already on the map), and Route: route_reservoir routes it with level-pool routing at the dam inside Route
// it, so the outflow goes on down the river and the wave on the time bar is the flattened one past the dam. The
// card has the inflow and outflow hydrographs at the dam, the peak reduction and the same run without the
// reservoir furthest down. Dragging a corner of the lake plans it again and, after a run, routes again. The lake
// and every run are in the action log.
//
// reservoir-core.js checks the inputs and draws the lake; every number comes from Python.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { cancelDraw, isDrawing, startLineDraw } from "./floodsim-sketch.js?v=__BUILD__";
import { readTerrain } from "./floodsim-map.js?v=__BUILD__";
import { onWave, playWave, showRouted } from "./route-map.js?v=__BUILD__";
import { registerSketchTool } from "./sketch-menu.js?v=__BUILD__";
import { flowWords } from "./route-core.js?v=__BUILD__";
import { hydroChart, ringBounds } from "./storm-core.js?v=__BUILD__";
import {
  RESERVOIR_START, damHydrographs, lakeFeatures, lakeLogLabel, reservoirArgs, reservoirFacts, reservoirHeadline,
  waveFlood, waveWords,
} from "./reservoir-core.js?v=__BUILD__";

const SRC = "reservoir-lake";
const L = { fill: "reservoir-lake-fill", line: "reservoir-lake-line", dam: "reservoir-dam", label: "reservoir-dam-label" };
const WATER = "#1c6ea4";
const ICON = '<path d="M3 18c2 0 2-1.5 4-1.5s2 1.5 4 1.5 2-1.5 4-1.5 2 1.5 4 1.5"/><path d="M4 13c4-1 6-4 8-8 2 4 4 7 8 8"/><path d="M12 5v8"/>';

let lake = null;        // { polygon, dam, window } on the map, or null
let wave = null;        // the run through it on the map (Route it's slot value), or null
let other = null;       // the wave on the map that is not this reservoir's: the flood to send through it
let handles = [];
let running = 0;
let hooked = false;
const terrain = new Map();   // the window's key -> Float32Array, read once

const panel = () => $("reservoir");

// ── the lake on the map ─────────────────────────────────────────────────────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-rt-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getSource(SRC)) return true;
  const before = beforeId();
  map.addSource(SRC, { type: "geojson", data: lakeFeatures(null) });
  const isLake = ["==", ["get", "kind"], "lake"];
  map.addLayer({ id: L.fill, type: "fill", source: SRC, filter: isLake,
    paint: { "fill-color": WATER, "fill-opacity": ["case", ["==", ["get", "live"], 1], 0.38, 0.28] } }, before);
  map.addLayer({ id: L.line, type: "line", source: SRC, filter: isLake, layout: { "line-join": "round" },
    paint: { "line-color": WATER, "line-width": 2.4 } }, before);
  map.addLayer({ id: L.dam, type: "circle", source: SRC, filter: ["==", ["get", "kind"], "dam"],
    paint: { "circle-radius": 7, "circle-color": "#3d3d3d", "circle-stroke-color": "#fff", "circle-stroke-width": 2.5 } });
  map.addLayer({ id: L.label, type: "symbol", source: SRC, filter: ["==", ["get", "kind"], "dam"],
    layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"], "text-size": 12, "text-anchor": "left",
      "text-offset": [0.9, 0], "text-allow-overlap": true },
    paint: { "text-color": "#222", "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 2 } });
  if (!hooked) {
    hooked = true;
    map.on("style.load", () => { if (lake) { ensureLayers(); drawLake(); } });
    for (const id of [L.fill, L.dam]) map.on("click", id, (e) => { if (!isDrawing()) { e.preventDefault(); openReservoir({ draw: false }); } });
  }
  return true;
}

function drawLake() {
  if (!ensureLayers()) return;
  map.getSource(SRC).setData(lakeFeatures(lake, { live: !panel().hidden }));
  placeHandles();
}

function placeHandles() {
  for (const h of handles) h.remove();
  handles = [];
  if (!lake || panel().hidden || isDrawing() || typeof maplibregl === "undefined") return;
  lake.polygon.forEach((ll, k) => {
    const el = document.createElement("span");
    el.className = "st-handle rv-handle";
    el.title = "Drag to move this corner of the lake";
    el.setAttribute("aria-label", `Corner ${k + 1} of the reservoir`);
    // eslint-disable-next-line no-undef
    const m = new maplibregl.Marker({ element: el, anchor: "center", draggable: true }).setLngLat(ll).addTo(map);
    const moved = () => { const p = m.getLngLat(); return lake.polygon.map((q, j) => (j === k ? [p.lng, p.lat] : q)); };
    m.on("drag", () => map.getSource(SRC).setData(lakeFeatures({ polygon: moved() }, { live: true })));
    m.on("dragend", () => { void moveLake(moved()); });
    handles.push(m);
  });
}

function setLake(value) {
  lake = value ? { polygon: value.polygon, dam: value.dam, window: value.window } : null;
  drawLake();
  syncOutputs();
}

async function planLake(polygon) {
  const res = await call("reservoir", { op: "plan", args: { polygon } });
  if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
  return { polygon: res.polygon, dam: res.dam, window: res.window };
}

function logLake(value, { moved = false, by = "rules", command = "" } = {}) {
  const before = slotValue("reservoir");
  setLake(value);
  logAction({ action: { type: "draw_reservoir", area: "drawn" }, before, after: value, label: lakeLogLabel(value, { moved }), by, command });
}

async function moveLake(polygon) {
  say(`<span class="spinner" aria-hidden="true"></span>Finding the river under the lake…`);
  try {
    logLake(await planLake(polygon), { moved: true });
    say("");
    if (wave) await run({ by: "rules" });
  } catch (err) {
    say(escapeHtml(`Could not move the lake: ${err.message}`), "error");
    drawLake();
  }
}

// ── the panel ───────────────────────────────────────────────────────────────

function inputs() {
  return {
    crest_m: $("rv-crest").value, top_m: $("rv-top").value, width_m: $("rv-width").value,
    kind: document.querySelector('input[name="rv-kind"]:checked')?.value || "rp",
    return_period: $("rv-rp").value, peak_cms: $("rv-peak").value, duration_h: $("rv-dur").value, wave: other,
  };
}

function syncOutputs() {
  const v = inputs();
  $("rv-crest-out").textContent = `${v.crest_m} m`;
  $("rv-top-out").textContent = `${v.top_m} m`;
  $("rv-width-out").textContent = `${v.width_m} m`;
  const ww = waveWords(other);
  $("rv-kind-wave").disabled = !ww;
  $("rv-wave-words").textContent = ww || "the flood wave or storm on the map (none now)";
  if (!ww && $("rv-kind-wave").checked) $("rv-kind-rp").checked = true;
  $("rv-title").textContent = lake ? `A reservoir on reach ${lake.dam.river_id}` : "Draw the lake across a river";
  let ok = Boolean(lake);
  try { if (lake) reservoirArgs(inputs()); } catch { ok = false; }
  $("rv-run").disabled = Boolean(running) || !ok;
  $("rv-run").textContent = running ? "Routing…" : "Route it through";
  $("rv-draw").textContent = lake ? "Draw it again" : "Draw the lake";
}

function say(html, kind = "") {
  const el = $("rv-status");
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

/** Open the panel; with no lake on the map, start drawing one. */
export function openReservoir({ draw = true } = {}) {
  const p = panel();
  if (!p) return;
  for (const id of ["whatif", "route", "storm"]) { const el = $(id); if (el) el.hidden = true; }   // one panel at a time
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  p.hidden = false;
  syncOutputs();
  renderResult();
  drawLake();
  if (!lake && draw) void drawNew();
}

function closePanel() {
  cancelDraw();
  panel().hidden = true;
  drawLake();
}

async function drawNew({ by = "rules", command = "" } = {}) {
  say("");
  const p = panel();
  p.classList.add("drawing");
  for (const h of handles) h.remove();
  handles = [];
  const line = await startLineDraw("levee", { polygon: true, what: "the lake, across the river" });
  p.classList.remove("drawing");
  if (!line) { drawLake(); return null; }
  say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady ? "Finding the river under the lake"
    : "Starting the engine (once), then finding the river")}…`);
  try {
    const value = await planLake(line);
    logLake(value, { by, command });
    say("");
    announce(`The lake is on reach ${value.dam.river_id}; the dam is at its downstream end. Set the dam and Route it through.`);
    return value;
  } catch (err) {
    say(escapeHtml(err.message), "warn");
    return null;
  }
}

function chartSvg(res) {
  const c = hydroChart(damHydrographs(res), { w: 300, h: 96 });
  if (!c) return "";
  const lines = c.paths.map((p) => `<path class="st-q rv-${p.role}" d="${p.d}"/>`).join("");
  const ticks = c.ticks.map((t) => `<text x="${t.x}" y="${c.h - 4}" text-anchor="middle">${escapeHtml(t.label)}</text>`).join("");
  const key = c.paths.map((p) => `<span class="st-key rv-${p.role}"><i></i>${p.role === "inflow" ? "Into the lake" : "Out over the spillway"}, ` +
    `${escapeHtml(flowWords(p.peak_cms))} m³/s</span>`).join("");
  return `<svg class="st-chart" viewBox="0 0 ${c.w} ${c.h}" role="img" aria-label="The flow into the lake and out of it, hour by hour">` +
    `<line class="st-axis" x1="${c.pad.l}" x2="${c.w - c.pad.r}" y1="${c.y0}" y2="${c.y0}"/>` +
    `<text x="${c.pad.l - 4}" y="${c.yTop + 4}" text-anchor="end">${escapeHtml(flowWords(c.top))}</text>` +
    `<text x="${c.pad.l - 4}" y="${c.y0}" text-anchor="end">0</text>${lines}${ticks}</svg>` +
    `<p class="st-keys">${key} <span class="muted">(m³/s at the dam)</span></p>`;
}

function renderResult() {
  const box = $("rv-result");
  if (!box) return;
  const res = wave && wave.result;
  box.hidden = !res;
  panel().classList.toggle("done", Boolean(res));
  if (!res) return;
  const r = res.reservoir;
  $("rv-head").textContent = reservoirHeadline(res);
  $("rv-chart").innerHTML = chartSvg(res);
  $("rv-facts").innerHTML = reservoirFacts(res).map((f) => `<dt>${escapeHtml(f.label)}</dt><dd>${escapeHtml(f.value)}</dd>`).join("");
  const notes = (res.notes || []).filter((n) => /overtopped|edge|terrain|Level pool/.test(n));
  $("rv-notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  $("rv-model").textContent = `${r.method.name}; the storage from the terrain inside the lake (${r.cells.toLocaleString("en-US")} cells of ` +
    `${r.cell_m} m), a broad-crested spillway (C = ${r.coefficient}). ${r.check} Routed on with Muskingum on GEOGLOWS's network.`;
}

// ── run ─────────────────────────────────────────────────────────────────────

// The lake and the river round it, beside the panel: the lake in the middle third of the view, so the wave leaving
// the dam shows and the reader zooms out to follow it.
function frameLake() {
  const box = lake && ringBounds(lake.polygon);
  if (!box || !state.mapOk || !map) return;
  const [[w, s], [e, n]] = box;
  const dx = Math.max(e - w, 0.02), dy = Math.max(n - s, 0.012);
  const wide = [[w - dx, s - dy], [e + dx, n + dy]];
  const p = panel(), phone = globalThis.matchMedia && matchMedia("(max-width: 640px)").matches;
  const pad = { top: 90, left: 40, right: 40, bottom: 60 };
  if (p && !p.hidden) {
    if (phone) {
      // a bottom sheet (sketch.css): keep the lake, and its corners to drag, above the sheet's top edge
      const r = p.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
      pad.bottom = Math.min(Math.max(c.bottom - r.top + 24, pad.bottom), c.height * 0.6);
    } else pad.right = p.offsetWidth + 40;
  }
  try { map.fitBounds(wide, { padding: pad, maxZoom: 13, duration: 0 }); } catch { /* the map is not ready */ }
}

async function terrainFor(win) {
  const key = JSON.stringify([win.x0, win.y0, win.nx, win.ny, win.zoom]);
  if (!terrain.has(key)) {
    if (terrain.size > 3) terrain.delete(terrain.keys().next().value);
    terrain.set(key, await readTerrain(win));
  }
  return terrain.get(key);
}

/** Route the flood the panel says through the lake on the map, and on down the river. */
async function run({ by = "rules", command = "", label = "" } = {}) {
  if (!lake) return null;
  let args;
  try { args = reservoirArgs(inputs()); } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  const my = ++running;
  syncOutputs();
  const started = performance.now();
  let stage = "Reading the terrain under the lake";
  const tick = () => {
    const s = Math.round((performance.now() - started) / 1000);
    say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(stage)}… ${s} s`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const z = await terrainFor(lake.window);
    stage = "Routing the flood through the lake and on down the river";
    const res = await call("reservoir", { op: "run", args: { polygon: lake.polygon, window: lake.window, dam: lake.dam, ...args }, z });
    if (my !== running) return res;
    if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
    showRouted(res, { river_id: lake.dam.river_id, lat: lake.dam.lat, lon: lake.dam.lon, reservoir: true, name: "Reservoir" },
      { frame: false, label, by, command });
    frameLake();
    say(`Routed in ${Math.round((performance.now() - started) / 1000)} s. The wave plays on the time bar; drag a corner to move the lake.`, "ok");
    announce(reservoirHeadline(res));
    return res;
  } catch (err) {
    if (my === running) say(escapeHtml(`Could not route it through: ${err.message}`), "error");
    return null;
  } finally {
    clearInterval(timer);
    if (my === running) running = 0;
    syncOutputs();
  }
}

// The wave on the map changed: this reservoir's run shows its card; any other wave is the flood it can take.
function followWave(value) {
  const mine = value && value.result && value.result.reservoir;
  wave = mine ? value : null;
  if (value && !mine) other = value;
  else if (mine && waveFlood(value)) other = value;
  if (!value) other = null;
  if (wave && panel().hidden) openReservoir({ draw: false });
  syncOutputs();
  renderResult();
}

// ── Ask the map's draw_reservoir (aquascope.map_commands) ───────────────────

/**
 * A reservoir the words asked for: { crest_m, top_m, width_m, return_period | peak_cms | inflow: "wave", polygon |
 * area: "drawn" }. Returns the result, or null when no lake was drawn.
 */
export async function drawReservoir(a, { by = "rules", command = "", label = "" } = {}) {
  for (const [id, key] of [["rv-crest", "crest_m"], ["rv-top", "top_m"], ["rv-width", "width_m"], ["rv-rp", "return_period"],
    ["rv-dur", "duration_h"]]) if (a[key] !== undefined && a[key] !== null) $(id).value = a[key];
  if (a.peak_cms) { $("rv-peak").value = a.peak_cms; $("rv-kind-peak").checked = true; }
  else if (a.inflow === "wave" && waveFlood(other)) $("rv-kind-wave").checked = true;
  else $("rv-kind-rp").checked = true;
  openReservoir({ draw: false });
  if (Array.isArray(a.polygon) && a.polygon.length >= 3) {
    const box = ringBounds(a.polygon);
    if (box && state.mapOk) map.fitBounds(box, { padding: 80, maxZoom: 13, duration: 0 });
    try { logLake(await planLake(a.polygon), { by, command }); } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  } else if (!lake) {
    if (!(await drawNew({ by, command }))) return null;
  }
  return run({ by, command, label });
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initReservoir() {
  const p = panel();
  if (!p) return;
  for (const [id, key] of [["rv-crest", "crest_m"], ["rv-top", "top_m"], ["rv-width", "width_m"], ["rv-rp", "return_period"],
    ["rv-dur", "duration_h"]]) $(id).value = RESERVOIR_START[key];
  registerSlotWriter("reservoir", (value) => setLake(value || null));
  onWave(followWave);
  for (const id of ["rv-crest", "rv-top", "rv-width", "rv-rp", "rv-peak", "rv-dur"]) $(id).addEventListener("input", syncOutputs);
  for (const el of document.querySelectorAll('input[name="rv-kind"]')) el.addEventListener("change", syncOutputs);
  $("rv-peak").addEventListener("focus", () => { $("rv-kind-peak").checked = true; syncOutputs(); });
  $("rv-crest").addEventListener("input", () => {
    if (Number($("rv-top").value) <= Number($("rv-crest").value)) $("rv-top").value = Number($("rv-crest").value) + 1;
    syncOutputs();
  });
  $("rv-form").addEventListener("submit", (e) => { e.preventDefault(); void run({ by: "rules" }); });
  $("rv-draw").addEventListener("click", () => { void drawNew(); });
  p.querySelector(".rv-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape" && !isDrawing()) { e.preventDefault(); closePanel(); } });
  $("rv-edit").addEventListener("click", () => { p.classList.remove("done"); $("rv-crest").focus({ preventScroll: true }); });
  $("rv-replay").addEventListener("click", () => { if (wave) playWave(); });
  registerSketchTool({ id: "reservoir", label: "Draw a reservoir", hint: "A lake on a river, and the flood it flattens",
    icon: ICON, run: () => openReservoir() });
  actions.drawReservoir = drawReservoir;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.reservoir = { open: openReservoir, run, frame: frameLake, lake: () => lake, wave: () => wave, other: () => other,
    drawReservoir };
  syncOutputs();
}
