// Change land use (#661). From the sketchpad's menu: draw an area (tap its corners, or drag a box), pick what it
// becomes and the design storm, and Compute. The worker runs aquascope.sketch.landuse.landuse_change (the same
// function as `aquascope landuse` and the MCP tool change_landuse): the cover now from ESA WorldCover, the soil group
// from HYSOGs250m (or the one picked), TR-55's curve numbers, runoff and graphical peak, before and after, and the
// GEOGLOWS reach the area drains to. The area is filled in what it becomes, labelled with its peak's change; the
// reach is lit, with the way to it; the panel (a sheet on a phone) gives the numbers, the cover bar, the sources and
// their licences; the added peak can be routed down the river with Route it. Drag a corner and it is computed
// again. Every change is an entry in the action log (slot "landuse"), and Ask the map's change_landuse runs here.
//
// landuse-core.js checks the inputs and decides the words; every number comes from Python.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { cancelDraw, isDrawing, startLineDraw } from "./floodsim-sketch.js?v=__BUILD__";
import { registerSketchTool } from "./sketch-menu.js?v=__BUILD__";
import { STREAMS_PMTILES } from "./river-core.js?v=__BUILD__";
import {
  DISTRIBUTIONS, START, USES, areaFeatures, compareRows, coverBar, flow, headline, landuseArgs, logLabel, ringBounds,
  routeArgs, soilWords, useShort,
} from "./landuse-core.js?v=__BUILD__";

const SRC = "landuse-area";
const NET = "landuse-net";
const L = { fill: "landuse-fill", line: "landuse-line", label: "landuse-label", path: "landuse-path",
  outlet: "landuse-outlet", outletLabel: "landuse-outlet-label", reachCase: "landuse-reach-case", reach: "landuse-reach" };
const INK = "#1f4e8c", REACH = "#e8590c";
const ICON = '<path d="M3 20h18"/><path d="M5 20v-6l4-3 4 3v6"/><path d="M13 20v-9l4-3 4 3v9"/><path d="M8 20v-3h2v3"/>';
const EMPTY = { type: "FeatureCollection", features: [] };

let value = null;       // the change on the map: { polygon, result } (the action log's slot value), or null
let handles = [];
let running = 0;
let hooked = false;

const panel = () => $("landuse");
const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

// ── the layers ──────────────────────────────────────────────────────────────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-rt-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getSource(SRC)) return true;
  try {
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const before = beforeId();
    if (!map.getSource(NET)) map.addSource(NET, { type: "vector", url: `pmtiles://${STREAMS_PMTILES}` });
    const none = ["boolean", false];
    map.addLayer({ id: L.reachCase, type: "line", source: NET, "source-layer": "streams", filter: none,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 6, 6, 12, 11] } }, before);
    map.addLayer({ id: L.reach, type: "line", source: NET, "source-layer": "streams", filter: none,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": REACH, "line-width": ["interpolate", ["linear"], ["zoom"], 6, 3.5, 12, 7] } }, before);
    map.addSource(SRC, { type: "geojson", data: EMPTY });
    const isKind = (k) => ["==", ["get", "kind"], k];
    map.addLayer({ id: L.fill, type: "fill", source: SRC, filter: isKind("area"),
      paint: { "fill-color": ["get", "color"], "fill-opacity": ["case", ["==", ["get", "live"], 1], 0.42, 0.34] } }, before);
    map.addLayer({ id: L.line, type: "line", source: SRC, filter: isKind("area"), layout: { "line-join": "round" },
      paint: { "line-color": ["get", "color"], "line-width": 2.4 } }, before);
    map.addLayer({ id: L.path, type: "line", source: SRC, filter: isKind("path"),
      paint: { "line-color": REACH, "line-width": 2, "line-dasharray": [1.5, 1.2] } }, before);
    map.addLayer({ id: L.outlet, type: "circle", source: SRC, filter: isKind("outlet"),
      paint: { "circle-radius": 6, "circle-color": REACH, "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });
    map.addLayer({ id: L.label, type: "symbol", source: SRC, filter: isKind("area"),
      layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"], "text-size": 13, "text-allow-overlap": true },
      paint: { "text-color": "#222", "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 2.2 } });
    map.addLayer({ id: L.outletLabel, type: "symbol", source: SRC, filter: isKind("outlet"),
      layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"], "text-size": 12, "text-anchor": "left",
        "text-offset": [0.9, 0], "text-allow-overlap": true },
      paint: { "text-color": REACH, "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 2 } });
    if (!hooked) {
      hooked = true;
      map.on("style.load", () => { if (value) { ensureLayers(); drawArea(); } });
      map.on("click", L.fill, (e) => { if (!isDrawing()) { e.preventDefault(); openLanduse({ draw: false }); } });
    }
    return true;
  } catch (err) {
    console.info("land-use layers unavailable:", err && err.message);
    return false;
  }
}

function drawArea() {
  if (!ensureLayers()) return;
  map.getSource(SRC).setData(areaFeatures(value, { live: !panel().hidden }));
  const id = value && value.result && value.result.reach ? Number(value.result.reach.river_id) : null;
  const f = id ? ["==", ["to-number", ["get", "riverId"]], id] : ["boolean", false];
  for (const l of [L.reachCase, L.reach]) map.setFilter(l, f);
  placeHandles();
}

function placeHandles() {
  for (const h of handles) h.remove();
  handles = [];
  if (!value || panel().hidden || isDrawing() || running || typeof maplibregl === "undefined") return;
  value.polygon.forEach((ll, k) => {
    const el = document.createElement("span");
    el.className = "st-handle lu-handle";
    el.title = "Drag to move this corner, and compute again";
    el.setAttribute("aria-label", `Corner ${k + 1} of the area`);
    // eslint-disable-next-line no-undef
    const m = new maplibregl.Marker({ element: el, anchor: "center", draggable: true }).setLngLat(ll).addTo(map);
    const moved = () => { const p = m.getLngLat(); return value.polygon.map((q, j) => (j === k ? [p.lng, p.lat] : q)); };
    m.on("drag", () => map.getSource(SRC).setData(areaFeatures({ polygon: moved() }, { live: true })));
    m.on("dragend", () => { void moveArea(moved()); });
    handles.push(m);
  });
}

// ── the value: drawn, logged, put back by undo ──────────────────────────────

function paint(next) {
  value = next && next.polygon ? next : null;
  drawArea();
  renderResult();
  syncOutputs();
}

function logChange(next, { by = "rules", command = "", label = "" } = {}) {
  const before = slotValue("landuse");
  paint(next);
  logAction({ action: { type: "change_landuse", area: "drawn", use: next && next.result ? next.result.use : undefined },
    before, after: next, label: label || logLabel(next), by, command });
}

async function moveArea(polygon) {
  let ring;
  try { ring = landuseArgs({ ...inputs(), polygon }).polygon; } catch { drawArea(); return; }
  if (value && value.result) await run({ polygon: ring });
  else logChange({ polygon: ring });
}

// ── the panel ───────────────────────────────────────────────────────────────

function inputs() {
  return { use: $("lu-use").value, rain_mm: $("lu-rain").value, distribution: $("lu-dist").value,
    soil_group: $("lu-soil").value, impervious_pct: $("lu-imp").value };
}

function syncOutputs() {
  const v = inputs();
  $("lu-imp-row").hidden = v.use !== "impervious";
  $("lu-title").textContent = value ? "The area drawn" : "Draw the area";
  let ok = Boolean(value);
  try { if (value) landuseArgs({ ...v, polygon: value.polygon }); } catch { ok = false; }
  $("lu-run").disabled = Boolean(running) || !ok;
  $("lu-run").textContent = running ? "Computing…" : "Compute";
  $("lu-draw").textContent = value ? "Draw it again" : "Draw the area";
}

function say(html, kind = "") {
  const el = $("lu-status");
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

function barHtml(rows) {
  const bar = coverBar(rows);
  const seg = bar.map((b) => `<i style="background:${b.color};flex:${b.share}" title="${escapeHtml(b.label)} ` +
    `${Math.round(b.share * 100)}%"></i>`).join("");
  const key = bar.filter((b) => b.share >= 0.03).map((b) => `<span><i style="background:${b.color}"></i>` +
    `${escapeHtml(b.label)} ${Math.round(b.share * 100)}%${b.cn !== null && b.cn !== undefined ? `, CN ${Math.round(b.cn)}` : ""}</span>`).join("");
  return `<div class="lu-bar" role="img" aria-label="The cover under the area now">${seg}</div><p class="lu-key">${key}</p>`;
}

function renderResult() {
  const box = $("lu-result");
  if (!box) return;
  const res = value && value.result;
  box.hidden = !res;
  panel().classList.toggle("done", Boolean(res));
  if (!res) return;
  $("lu-head").textContent = headline(res);
  $("lu-table").innerHTML = '<tr><th></th><th scope="col">Now</th><th scope="col">' +
    `${escapeHtml(useShort(res.use))}</th></tr>` + compareRows(res).map(([k, b, a]) =>
    `<tr><th scope="row">${escapeHtml(k)}</th><td>${escapeHtml(b)}</td><td>${escapeHtml(a)}</td></tr>`).join("");
  $("lu-cover").innerHTML = barHtml(res.cover);
  const reach = res.reach ? `It drains to GEOGLOWS reach ${res.reach.river_id}, lit on the map.` : "";
  $("lu-soil-out").textContent = `Soil groups: ${soilWords(res.soils)}${res.sources.soil && res.sources.soil.picked ? " (picked)" : " (HYSOGs250m)"}. ` +
    `Tc: ${res.tc_from}. ${reach}`;
  $("lu-notes").textContent = (res.notes || []).join(" ");
  $("lu-notes").hidden = !(res.notes || []).length;
  const ra = routeArgs(res);
  $("lu-route").hidden = !(ra && res.reach && actions.routeFlood);
  if (ra) $("lu-route").textContent = `Route the extra ${flow(ra.peak_cms)} m³/s downstream`;
  $("lu-model").textContent = `${res.method} ${res.assumptions} ${res.check}`;
  $("lu-src").innerHTML = Object.values(res.licences).map((l) =>
    `${escapeHtml(l.name)}: ${escapeHtml(l.licence)}`).join(". ") + ".";
}

/** Open the panel; with no area on the map, start drawing one. */
export function openLanduse({ draw = true } = {}) {
  const p = panel();
  if (!p) return;
  for (const id of ["whatif", "route", "storm", "well"]) { const el = $(id); if (el) el.hidden = true; }   // one at a time
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  p.hidden = false;
  syncOutputs();
  renderResult();
  drawArea();
  if (!value && draw) void drawNew();
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
  const line = await startLineDraw("levee", { polygon: true, what: "the area that changes" });
  p.classList.remove("drawing");
  if (!line) { drawArea(); return null; }
  let ring;
  try { ring = landuseArgs({ ...inputs(), polygon: line }).polygon; } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  logChange({ polygon: ring }, { by, command });
  announce("The area is drawn. Pick what it becomes and the design storm, then Compute.");
  $("lu-use").focus({ preventScroll: true });
  return ring;
}

function frame(res) {
  const box = ringBounds(value && value.polygon, res);
  if (!box || !state.mapOk) return;
  const p = panel();
  const pad = { top: 100, left: 50, right: 50, bottom: 120 };
  const lg = $("map-legend");
  if (lg && !lg.hidden && !narrow()) {   // the legend on the left, as Route it frames its wave
    const r = lg.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
    if (r.width && r.right < c.left + c.width / 2) pad.left = Math.max(pad.left, r.right - c.left + 24);
  }
  if (p && !p.hidden) {
    if (narrow()) {
      // a bottom sheet (landuse.css): keep the area above its top edge
      const r = p.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
      pad.bottom = Math.min(Math.max(c.bottom - r.top + 32, pad.bottom), c.height * 0.6);
      pad.top = Math.min(pad.top, c.height * 0.15);
    } else pad.right = p.offsetWidth + 50;
  }
  try { map.fitBounds(box, { padding: pad, maxZoom: 14, duration: 600 }); } catch { /* not ready */ }
}

// ── run ─────────────────────────────────────────────────────────────────────

/** Compute the change for the area on the map (or `polygon`) with the panel's use and storm; logged. */
async function run({ polygon = null, by = "rules", command = "", label = "" } = {}) {
  const ring = polygon || (value && value.polygon);
  if (!ring) return null;
  let args;
  try { args = landuseArgs({ ...inputs(), polygon: ring }); } catch (err) { say(escapeHtml(err.message), "warn"); return null; }
  const my = ++running;
  syncOutputs();
  for (const h of handles) h.remove();
  handles = [];
  const started = performance.now();
  const tick = () => {
    const s = Math.round((performance.now() - started) / 1000);
    say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady
      ? "Reading the land cover, the soil and the slope" : "Starting the engine (once), then reading")}… ${s} s`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const res = await call("landuse", { op: "change", args });
    if (my !== running) return res;
    if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
    running = 0;
    logChange({ polygon: args.polygon, result: res }, { by, command, label });
    const secs = Math.round((performance.now() - started) / 1000);
    say(`Computed in ${secs} s. Drag a corner to move the area.`, "ok");
    announce(headline(res));
    frame(res);
    return res;
  } catch (err) {
    if (my === running) say(escapeHtml(`Could not compute the change: ${err.message}`), "error");
    return null;
  } finally {
    clearInterval(timer);
    if (my === running) running = 0;
    syncOutputs();
    placeHandles();
  }
}

async function routeExtra() {
  const res = value && value.result;
  const ra = routeArgs(res);
  if (!ra || !res.reach || !actions.routeFlood) return;
  const t = { river_id: res.reach.river_id, lat: res.reach.lat, lon: res.reach.lon,
    name: `Below the ${useShort(res.use).toLowerCase()}` };
  panel().hidden = true;   // Route it's panel takes over, with the wave's places; the area stays on the map
  drawArea();
  if (actions.openRoute) actions.openRoute(t);
  try {
    await actions.routeFlood(t, ra, { by: "rules", label: `Route the extra ${flow(ra.peak_cms)} m³/s of the land-use change` });
  } catch { /* said in Route it's panel */ }
}

// ── Ask the map's change_landuse (aquascope.map_commands) ───────────────────

/**
 * A change the words asked for: { use, rain_mm?, soil_group?, impervious_pct?, polygon | area: "drawn" }. "drawn"
 * takes the area on the map, or lets the reader draw one. Returns the result, or null when no area was drawn.
 */
export async function changeLanduse(a, { by = "rules", command = "", label = "" } = {}) {
  $("lu-use").value = a.use;
  if (a.rain_mm) $("lu-rain").value = a.rain_mm;
  if (a.soil_group) $("lu-soil").value = a.soil_group;
  if (a.impervious_pct !== undefined && a.impervious_pct !== null) $("lu-imp").value = a.impervious_pct;
  openLanduse({ draw: false });
  let ring = null;
  if (Array.isArray(a.polygon) && a.polygon.length >= 3) {
    ring = landuseArgs({ ...inputs(), polygon: a.polygon }).polygon;
    const box = ringBounds(ring);
    if (box && state.mapOk) map.fitBounds(box, { padding: 80, maxZoom: 13, duration: 0 });
  } else if (!value) {
    ring = await drawNew({ by, command });
    if (!ring) return null;
  }
  return run({ polygon: ring, by, command, label });
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initLanduse() {
  const p = panel();
  if (!p) return;
  $("lu-use").innerHTML = USES.map(([k, l]) => `<option value="${k}">${escapeHtml(l)}</option>`).join("");
  $("lu-dist").innerHTML = DISTRIBUTIONS.map((d) => `<option value="${d}">Type ${d}</option>`).join("");
  $("lu-use").value = START.use;
  $("lu-rain").value = START.rain_mm;
  $("lu-dist").value = START.distribution;
  $("lu-imp").value = START.impervious_pct;
  registerSlotWriter("landuse", (v) => paint(v || null));
  for (const id of ["lu-use", "lu-rain", "lu-dist", "lu-soil", "lu-imp"]) $(id).addEventListener("input", syncOutputs);
  $("lu-form").addEventListener("submit", (e) => { e.preventDefault(); void run({ by: "rules" }); });
  $("lu-draw").addEventListener("click", () => { void drawNew(); });
  $("lu-route").addEventListener("click", () => { void routeExtra(); });
  $("lu-edit").addEventListener("click", () => { p.classList.remove("done"); $("lu-use").focus({ preventScroll: true }); });
  $("lu-clear").addEventListener("click", () => { if (value) logChange(null, { label: "Take the land-use change off" }); });
  p.querySelector(".lu-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape" && !isDrawing()) { e.preventDefault(); closePanel(); } });
  registerSketchTool({ id: "landuse", label: "Change land use", hint: "Runoff and peak flow, before and after (TR-55)",
    icon: ICON, run: () => openLanduse() });
  actions.changeLanduse = changeLanduse;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.landuse = { open: openLanduse, run, value: () => value, setArea: (polygon) => logChange({ polygon }),
    change: changeLanduse, route: routeExtra };
  syncOutputs();
}

