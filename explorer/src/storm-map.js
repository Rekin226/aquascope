// Draw a storm (#670). From the sketchpad's menu: draw an area on the map (tap its points, or drag a box), give the
// rain in mm, its hours and the share that runs off, and Route. The worker runs aquascope.routing.route_storm (the
// same function as `aquascope route storm` and the MCP tool route_storm): the rain times the runoff coefficient on
// every GEOGLOWS unit catchment under the area, routed down every river below with Muskingum. The wave plays on
// the rivers on the time bar, through Route it's own layers (route-map.js), and the card says the numbers with the
// hydrographs of the reach that gets the most and of the main outlet. The area's points can be dragged, and the
// storm is routed again. The area and each run are entries in the action log, each undoable.
//
// storm-core.js checks the inputs and draws the area and the charts; every number comes from Python.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { cancelDraw, isDrawing, startLineDraw } from "./floodsim-sketch.js?v=__BUILD__";
import { onWave, playWave, showRouted } from "./route-map.js?v=__BUILD__";
import { registerSketchTool } from "./sketch-menu.js?v=__BUILD__";
import { flowWords, hoursWords } from "./route-core.js?v=__BUILD__";
import {
  STORM_START, areaLogLabel, hydroChart, ringBounds, stormArgs, stormFacts, stormFeatures, stormHeadline, stormLabel,
} from "./storm-core.js?v=__BUILD__";

const SRC = "storm-area";
const L = { fill: "storm-area-fill", line: "storm-area-line", label: "storm-area-label" };
const INK = "#1f4e8c";
const ICON = '<path d="M7 15a4 4 0 1 1 .9-7.9A5.5 5.5 0 0 1 18.5 9 3 3 0 0 1 18 15Z"/><path d="M8 18l-1 2.5M12 18l-1 2.5M16 18l-1 2.5"/>';

let area = null;        // the area on the map: { polygon } or null
let wave = null;        // the storm's run on the map (Route it's slot value), or null
let handles = [];
let running = 0;
let hooked = false;

const panel = () => $("storm");
const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

// ── the area on the map ─────────────────────────────────────────────────────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-rt-/.test(l.id) || /^river-fa-/.test(l.id) ||
    ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getSource(SRC)) return true;
  const before = beforeId();
  map.addSource(SRC, { type: "geojson", data: stormFeatures(null) });
  const isArea = ["==", ["geometry-type"], "Polygon"];
  map.addLayer({ id: L.fill, type: "fill", source: SRC, filter: isArea,
    paint: { "fill-color": "#3b82c4", "fill-opacity": ["case", ["==", ["get", "live"], 1], 0.22, 0.14] } }, before);
  map.addLayer({ id: L.line, type: "line", source: SRC, filter: isArea, layout: { "line-join": "round" },
    paint: { "line-color": INK, "line-width": 2.2, "line-dasharray": [2.2, 1.2] } }, before);
  map.addLayer({ id: L.label, type: "symbol", source: SRC, filter: ["==", ["geometry-type"], "Point"],
    layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"], "text-size": 13, "text-allow-overlap": true },
    paint: { "text-color": INK, "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 2 } });
  if (!hooked) {
    hooked = true;
    map.on("style.load", () => { if (area) { ensureLayers(); drawArea(); } });
    map.on("click", L.fill, (e) => { if (!isDrawing()) { e.preventDefault(); openStorm(); } });
  }
  return true;
}

function settings() {
  return { rain_mm: $("st-rain").value, duration_h: $("st-dur").value, runoff_coefficient: $("st-c").value };
}

function drawArea() {
  if (!ensureLayers()) return;
  const shownSettings = wave && wave.result.storm ? wave.result.storm : settings();
  map.getSource(SRC).setData(stormFeatures(area ? { ...area, ...shownSettings } : null, { live: !panel().hidden }));
  placeHandles();
}

function placeHandles() {
  for (const h of handles) h.remove();
  handles = [];
  if (!area || panel().hidden || isDrawing() || typeof maplibregl === "undefined") return;
  area.polygon.forEach((ll, k) => {
    const el = document.createElement("span");
    el.className = "st-handle";
    el.title = "Drag to move this corner of the storm";
    el.setAttribute("aria-label", `Corner ${k + 1} of the storm area`);
    // eslint-disable-next-line no-undef
    const m = new maplibregl.Marker({ element: el, anchor: "center", draggable: true }).setLngLat(ll).addTo(map);
    const moved = () => { const p = m.getLngLat(); return area.polygon.map((q, j) => (j === k ? [p.lng, p.lat] : q)); };
    m.on("drag", () => map.getSource(SRC).setData(stormFeatures({ polygon: moved(), ...settings() }, { live: true })));
    m.on("dragend", () => { void moveArea(moved()); });
    handles.push(m);
  });
}

function setArea(value) {
  area = value ? { polygon: value.polygon } : null;
  drawArea();
  syncOutputs();
}

/** Put a new area on the map, logged; undo puts back the one before. */
function logArea(polygon, { moved = false, by = "rules", command = "" } = {}) {
  const before = slotValue("storm");
  const after = { polygon };
  setArea(after);
  logAction({ action: { type: "draw_storm", area: "drawn" }, before, after, label: areaLogLabel(after, { moved }), by, command });
}

async function moveArea(polygon) {
  let ring;
  try { ring = stormArgs({ ...settings(), polygon }).polygon; } catch { drawArea(); return; }
  logArea(ring, { moved: true });
  // The storm was routed: route it again from where it is now.
  if (wave) await run({ by: "rules" });
}

// ── the panel ───────────────────────────────────────────────────────────────

function syncOutputs() {
  const v = settings();
  $("st-dur-out").textContent = hoursWords(Number(v.duration_h));
  $("st-c-out").textContent = Number(v.runoff_coefficient).toFixed(2);
  $("st-title").textContent = area ? "Rain over the area drawn" : "Draw the storm's area";
  let ok = Boolean(area);
  try { if (area) stormArgs({ ...v, polygon: area.polygon }); } catch { ok = false; }
  $("st-run").disabled = Boolean(running) || !ok;
  $("st-run").textContent = running ? "Routing…" : "Route the storm";
  $("st-draw").textContent = area ? "Draw it again" : "Draw the area";
  if (area && state.mapOk && map && map.getSource(SRC) && !wave) {
    map.getSource(SRC).setData(stormFeatures({ ...area, ...v }, { live: !panel().hidden }));
  }
}

function say(html, kind = "") {
  const el = $("st-status");
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

/** Open the panel; with no area on the map, start drawing one. */
export function openStorm({ draw = true } = {}) {
  const p = panel();
  if (!p) return;
  for (const id of ["whatif", "route", "reservoir"]) { const el = $(id); if (el) el.hidden = true; }   // one panel at a time
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  p.hidden = false;
  p.classList.toggle("done", Boolean(wave));
  syncOutputs();
  renderResult();
  drawArea();
  if (!area && draw) void drawNew();
}

function closePanel() {
  cancelDraw();
  panel().hidden = true;
  drawArea();
}

async function drawNew({ by = "rules", command = "" } = {}) {
  say("");
  const p = panel();
  p.classList.add("drawing");
  for (const h of handles) h.remove();
  handles = [];
  const line = await startLineDraw("levee", { polygon: true, what: "the storm's area" });
  p.classList.remove("drawing");
  if (!line) { drawArea(); return null; }
  let ring;
  try { ring = stormArgs({ ...settings(), polygon: line }).polygon; } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  logArea(ring, { by, command });
  announce("The storm's area is drawn. Set the rain and its hours, then Route the storm.");
  $("st-rain").focus({ preventScroll: true });
  return ring;
}

function chartSvg(res) {
  const c = hydroChart(res.hydrographs, { w: 300, h: 96 });
  if (!c) return "";
  const lines = c.paths.map((p) => `<path class="st-q st-${p.role}" d="${p.d}"/>`).join("");
  const ticks = c.ticks.map((t) => `<text x="${t.x}" y="${c.h - 4}" text-anchor="middle">${escapeHtml(t.label)}</text>`).join("");
  const key = c.paths.map((p) => `<span class="st-key st-${p.role}"><i></i>${p.role === "outlet" ? "Main outlet" : "Reach with the biggest peak"}, ` +
    `${escapeHtml(flowWords(p.peak_cms))} m³/s</span>`).join("");
  return `<svg class="st-chart" viewBox="0 0 ${c.w} ${c.h}" role="img" aria-label="The flow the storm adds, hour by hour">` +
    `<line class="st-axis" x1="${c.pad.l}" x2="${c.w - c.pad.r}" y1="${c.y0}" y2="${c.y0}"/>` +
    `<text x="${c.pad.l - 4}" y="${c.yTop + 4}" text-anchor="end">${escapeHtml(flowWords(c.top))}</text>` +
    `<text x="${c.pad.l - 4}" y="${c.y0}" text-anchor="end">0</text>${lines}${ticks}</svg>` +
    `<p class="st-keys">${key} <span class="muted">(m³/s added)</span></p>`;
}

function renderResult() {
  const box = $("st-result");
  if (!box) return;
  const res = wave && wave.result;
  box.hidden = !res;
  panel().classList.toggle("done", Boolean(res));
  if (!res) return;
  $("st-head").textContent = stormHeadline(res);
  $("st-chart").innerHTML = chartSvg(res);
  $("st-facts").innerHTML = stormFacts(res).map((r) => `<dt>${escapeHtml(r.label)}</dt><dd>${escapeHtml(r.value)}</dd>`).join("");
  const check = res.check && res.check.summary ? ` ${res.check.summary}` : "";
  $("st-rule").textContent = res.storm.rule;
  $("st-model").textContent = `Muskingum on GEOGLOWS's own network, ${res.hours} h routed hourly.${check} ` +
    "Only the storm's runoff is routed: the flows it adds to the rivers.";
}

// ── run ─────────────────────────────────────────────────────────────────────

/** Route the storm over the area on the map with the panel's rain, hours and runoff coefficient. */
async function run({ by = "rules", command = "", label = "" } = {}) {
  if (!area) return null;
  let args;
  try { args = stormArgs({ ...settings(), polygon: area.polygon }); } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  const my = ++running;
  syncOutputs();
  const started = performance.now();
  const tick = () => {
    const s = Math.round((performance.now() - started) / 1000);
    say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady ? "Running the rain off and routing it"
      : "Starting the engine (once), then routing")}… ${s} s`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const res = await call("route", { op: "storm", args });
    if (my !== running) return res;
    if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
    const b = res.biggest || {};
    showRouted(res, { river_id: b.river_id, storm: true, name: stormLabel(res.storm) }, { frame: true, label, by, command });
    const secs = Math.round((performance.now() - started) / 1000);
    say(`Routed in ${secs} s. The wave plays on the time bar; drag a corner to move the storm.`, "ok");
    announce(stormHeadline(res));
    return res;
  } catch (err) {
    if (my === running) say(escapeHtml(`Could not route the storm: ${err.message}`), "error");
    return null;
  } finally {
    clearInterval(timer);
    if (my === running) running = 0;
    syncOutputs();
  }
}

// The wave on the map changed (a run, an undo, a replay's wave): the card follows it.
function followWave(value) {
  // The area stays the action log's (its own slot): an undo of a run leaves the area where the reader last put it.
  wave = value && value.result && value.result.storm && !value.result.reservoir ? value : null;   // a reservoir's is its own
  if (wave) {
    const s = wave.result.storm;
    $("st-rain").value = s.rain_mm;
    $("st-dur").value = s.duration_h;
    $("st-c").value = s.runoff_coefficient;
    if (panel().hidden) openStorm({ draw: false });
  }
  syncOutputs();
  renderResult();
  drawArea();
}

// ── Ask the map's draw_storm (aquascope.map_commands) ───────────────────────

/**
 * A storm the words asked for: { rain_mm, duration_h, runoff_coefficient, polygon | area: "drawn" }. "drawn" takes
 * the area on the map, or lets the reader draw one. Returns the result, or null when no area was drawn.
 */
export async function drawStorm(a, { by = "rules", command = "", label = "" } = {}) {
  $("st-rain").value = a.rain_mm;
  $("st-dur").value = a.duration_h || STORM_START.duration_h;
  $("st-c").value = a.runoff_coefficient || STORM_START.runoff_coefficient;
  openStorm({ draw: false });
  if (Array.isArray(a.polygon) && a.polygon.length >= 3) {
    const box = ringBounds(a.polygon);
    if (box && state.mapOk) map.fitBounds(box, { padding: 80, maxZoom: 10, duration: 0 });
    logArea(stormArgs({ ...settings(), polygon: a.polygon }).polygon, { by, command });
  } else if (!area) {
    const ring = await drawNew({ by, command });
    if (!ring) return null;
  }
  return run({ by, command, label });
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initStorm() {
  const p = panel();
  if (!p) return;
  $("st-rain").value = STORM_START.rain_mm;
  $("st-dur").value = STORM_START.duration_h;
  $("st-c").value = STORM_START.runoff_coefficient;
  registerSlotWriter("storm", (value) => setArea(value || null));
  onWave(followWave);
  for (const id of ["st-rain", "st-dur", "st-c"]) $(id).addEventListener("input", syncOutputs);
  $("st-form").addEventListener("submit", (e) => { e.preventDefault(); void run({ by: "rules" }); });
  $("st-draw").addEventListener("click", () => { void drawNew(); });
  p.querySelector(".st-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape" && !isDrawing()) { e.preventDefault(); closePanel(); } });
  $("st-edit").addEventListener("click", () => { p.classList.remove("done"); $("st-rain").focus({ preventScroll: true }); });
  $("st-replay").addEventListener("click", () => { if (wave) playWave(); });
  registerSketchTool({ id: "storm", label: "Draw a storm", hint: "Rain on an area, routed down the rivers", icon: ICON,
    run: () => openStorm() });
  actions.drawStorm = drawStorm;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.storm = { open: openStorm, run, area: () => area, wave: () => wave, setArea: (polygon) => logArea(polygon),
    drawStorm };
  syncOutputs();
}
