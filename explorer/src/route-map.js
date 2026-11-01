// Route it (#656). From a river's map card, "Route a flood" opens a small panel on the map: what enters (the
// reach's T-year flow, or a number in m³/s), over how long, and whether it is a flood or a steady release. Route
// runs Muskingum down GEOGLOWS's own network in the worker (aquascope.routing.route, the same function as
// `aquascope route flood` and the MCP tool route_flood) and the wave is drawn on the river lines: each reach
// coloured and widened by how much of the wave is there, hour by hour, on the time bar's own clock, which plays
// once on its own. The towns (read off the basemap's labels) and the Archive's gauges on the way get the peak and
// the hour it arrives, on the map and in the panel. Every run is an entry in the action log (map-actions.js), and
// the AI's route action (aquascope.map_commands) runs through openRoute and routeFlood here.
//
// route-core.js decides the colours, the widths and the words; every number comes from Python.

import { $, actions, clickLayers, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { STREAMS_PMTILES } from "./river-core.js?v=__BUILD__";
import { clearRiverNetwork, clearRiverTrace } from "./river-map.js?v=__BUILD__";
import { releaseModelClock, setModelStep, useModelClock } from "./time-ui.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { designFlowLine } from "./design-flow-core.js?v=__BUILD__";
import {
  RAMP, busiestFrame, cardPlaces, flatterWords, flowWords, frameFeatureStates, frameLabel, headline, hoursWords, logLabel,
  pathFilterExpr, placeFeatures, reachedBounds, routeArgs, townsFrom, waveColorExpr, waveOpacityExpr, waveWidthExpr,
} from "./route-core.js?v=__BUILD__";

const SRC = "river-rt-net";
const PTS = "river-rt-pts";
const L = { casing: "river-rt-casing", trail: "river-rt-trail", wave: "river-rt-wave", dot: "river-rt-dot",
  label: "river-rt-label" };
const NET = { source: SRC, sourceLayer: "streams" };
const EMPTY = { type: "FeatureCollection", features: [] };
const CREDIT = "Muskingum routing (aquascope) with GEOGLOWS v2's own k, x and network, CC BY 4.0; rivers TDX-Hydro, " +
  "CC BY-SA 4.0. Modelled.";

let target = null;       // { river_id, lat, lon, name }
let shown = null;        // { result, target, frame }: what is drawn
let running = 0;
let hooked = false;
let lastReach = null;    // the reach the selection snapped to, for "from here"
let litIds = [];         // the ids carrying feature-state now
const painted = [];      // fn(value) for each module that follows the wave on the map (Draw a storm, #670)

/** fn(value) whenever the wave on the map changes (null: none); a storm's run (#670) shows its card by it. */
export function onWave(fn) { painted.push(fn); }

const panel = () => $("route");

// ── the layers ──────────────────────────────────────────────────────────────

function beforeGauges() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-fa-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getLayer(L.wave)) return true;
  try {
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const before = beforeGauges();
    // Its own source with riverId promoted, so each reach of the path can carry the wave as feature-state.
    if (!map.getSource(SRC)) map.addSource(SRC, { type: "vector", url: `pmtiles://${STREAMS_PMTILES}`, promoteId: "riverId" });
    if (!map.getSource(PTS)) map.addSource(PTS, { type: "geojson", data: EMPTY });
    const none = ["boolean", false];
    const layout = { "line-cap": "round", "line-join": "round" };
    map.addLayer({ id: L.casing, type: "line", source: SRC, "source-layer": "streams", filter: none, layout,
      paint: { "line-color": "rgba(255, 255, 255, 0.85)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 3, 3, 7, 4.5, 12, 7] } }, before);
    // The trail: every reach in its own peak's colour, faint, so the way the wave flattens is always there.
    map.addLayer({ id: L.trail, type: "line", source: SRC, "source-layer": "streams", filter: none, layout,
      paint: { "line-color": waveColorExpr("rp"), "line-opacity": 0.45,
        "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1.6, 7, 2.6, 12, 4] } }, before);
    map.addLayer({ id: L.wave, type: "line", source: SRC, "source-layer": "streams", filter: none, layout,
      paint: { "line-color": waveColorExpr(), "line-width": waveWidthExpr(), "line-opacity": waveOpacityExpr() } }, before);
    const town = ["==", ["get", "kind"], "town"];
    map.addLayer({ id: L.dot, type: "circle", source: PTS,
      paint: { "circle-radius": ["case", ["get", "labelled"], ["case", town, 5.5, 4.5], 2.6],
        "circle-color": ["case", town, "#ffffff", "#3f007d"],
        "circle-stroke-width": 2, "circle-stroke-color": ["case", town, "#3f007d", "#ffffff"] } });
    map.addLayer({ id: L.label, type: "symbol", source: PTS,
      layout: { "text-field": ["get", "text"], "text-font": ["Noto Sans Bold"], "text-size": 12,
        "text-anchor": "left", "text-offset": [0.9, 0], "symbol-sort-key": ["get", "rank"], "text-optional": true },
      paint: { "text-color": "#24113f", "text-halo-color": "rgba(255,255,255,0.92)", "text-halo-width": 1.8 } });
    hook();
    return true;
  } catch (err) {
    console.info("route layers unavailable:", err && err.message);
    return false;
  }
}

function hook() {
  if (hooked) return;
  hooked = true;
  for (const id of [L.wave, L.dot]) {
    clickLayers.add(id);
    map.on("click", id, onClick);
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
  // A basemap change rebuilds the style: the layers and the frame go back on.
  map.on("style.load", () => { litIds = []; if (shown) { ensureLayers(); draw(); } });
}

function setStates(pairs) {
  const keep = new Set(pairs.map(([id]) => id));
  for (const id of litIds) {
    if (keep.has(id)) continue;
    try { map.removeFeatureState({ ...NET, id }, "rw"); map.removeFeatureState({ ...NET, id }, "rp"); } catch { /* swapping */ }
  }
  for (const [id, st] of pairs) { try { map.setFeatureState({ ...NET, id }, st); } catch { /* swapping */ } }
  litIds = [...keep];
}

function drawFrame(i, { peak = false } = {}) {
  if (!shown || !ensureLayers()) return;
  shown.frame = peak ? null : i;
  setStates(frameFeatureStates(shown.result.frames, i, { peak }));
  refreshLegend("route");
}

function draw() {
  if (!shown || !ensureLayers()) return;
  const { result } = shown;
  const ids = (result.frames && result.frames.ids) || [];
  for (const id of [L.casing, L.trail, L.wave]) map.setFilter(id, pathFilterExpr(ids));
  map.getSource(PTS).setData(placeFeatures(result.places, cardPlaces(result.places, narrow() ? 3 : 8)));
  drawFrame(shown.frame === null || shown.frame === undefined ? 0 : shown.frame, { peak: shown.frame === null });
}

function clearLayers() {
  if (!state.mapOk || !map || !map.getLayer(L.wave)) return;
  for (const id of [L.casing, L.trail, L.wave]) map.setFilter(id, ["boolean", false]);
  setStates([]);
  if (map.getSource(PTS)) map.getSource(PTS).setData(EMPTY);
}

// ── the clock: the wave hour by hour on the time bar ────────────────────────

function takeClock(index = 0) {
  const fr = shown && shown.result.frames;
  if (!fr || !fr.t_h || !fr.t_h.length) return;
  useModelClock({
    times: fr.t_h, index, label: (i) => frameLabel(fr.t_h[i]),
    summary: shown.result.storm ? "The storm's wave: time since the rain started." : "The flood wave: time since the water started to enter.",
    onStep: (i) => drawFrame(i),
    // "Back to dates": the still picture is the peak everywhere, how the wave flattens on its way.
    onExit: () => drawFrame(0, { peak: true }),
  });
}

function playOnce() {
  if (globalThis.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) {
    void setModelStep(busiestFrame(shown.result.frames));
    return;
  }
  const btn = $("tb-play");
  if (btn && btn.getAttribute("aria-pressed") !== "true") btn.click();
}

// ── the painted value: drawn, cleared, put back by undo ─────────────────────

function paint(value, { frame = true, play = false } = {}) {
  shown = value ? { ...value, frame: 0 } : null;
  if (!value) {
    clearLayers();
    releaseModelClock();
    renderResult();
    refreshLegend("route");
    for (const fn of painted) fn(null);
    return;
  }
  if (frame) { clearRiverNetwork(); clearRiverTrace(); }
  draw();
  takeClock(0);
  renderResult();
  if (frame) frameResult(value.result);
  openLegendRow("route", true);
  if (play) playOnce();
  for (const fn of painted) fn(shown);
}

const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

function frameResult(result) {
  // A storm's wave (#670) has no single path: the box of the reaches it reaches.
  const b = result && result.bbox;
  const box = b ? [[b[0], b[1]], [b[2], b[3]]] : reachedBounds(result);
  if (!box || !state.mapOk || !map) return;
  const p = [panel(), $("storm"), $("reservoir")].find((el) => el && !el.hidden) || panel();
  const pad = { top: 110, left: 40, right: 40, bottom: 170 };
  const lg = $("map-legend");
  if (lg && !lg.hidden && !narrow()) {
    const r = lg.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
    if (r.width && r.right < c.left + c.width / 2) pad.left = Math.max(pad.left, r.right - c.left + 24);
  }
  if (p && !p.hidden) {
    if (narrow()) pad.bottom = Math.min(p.offsetHeight + 24, map.getContainer().clientHeight * 0.55);
    else pad.right = p.offsetWidth + 40;
  }
  try { map.fitBounds(box, { padding: pad, maxZoom: 10, duration: 0 }); } catch { /* the map is not ready */ }
}

// ── the panel ───────────────────────────────────────────────────────────────

function inputs() {
  return {
    kind: document.querySelector('input[name="rt-kind"]:checked')?.value || "rp",
    peak: $("rt-peak").value, rp: $("rt-rp").value, duration: $("rt-dur").value, block: $("rt-block").checked,
  };
}

function syncOutputs() {
  const v = inputs();
  $("rt-dur-out").textContent = hoursWords(Number(v.duration));
  let ok = true;
  try { routeArgs(v); } catch { ok = false; }
  $("rt-run").disabled = Boolean(running) || !target || !ok;
  $("rt-run").textContent = running ? "Routing…" : "Route";
}

function say(html, kind = "") {
  const el = $("rt-status");
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

/** Open the panel for a reach: { river_id, lat, lon, name }. */
export function openRoute(t) {
  if (!t || t.river_id === null || t.river_id === undefined) return;
  if (!target || Number(t.river_id) !== target.river_id) say("");
  target = { ...t, river_id: Number(t.river_id) };
  const wi = $("whatif");
  if (wi) wi.hidden = true;     // one panel on the map at a time
  const p = panel();
  p.hidden = false;
  p.classList.remove("done");
  $("rt-title").textContent = t.name || `River reach ${target.river_id}`;
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  syncOutputs();
  renderResult();
  $("rt-run").focus({ preventScroll: true });
}

function closePanel() {
  panel().hidden = true;
}

function placeRow(p) {
  const icon = p.kind === "gauge" ? '<i class="rt-g" aria-hidden="true"></i>' : '<i class="rt-t" aria-hidden="true"></i>';
  return `<li>${icon}<span class="rt-name">${escapeHtml(p.name)}</span>` +
    `<span class="rt-when">${escapeHtml(hoursWords(p.peak_h))}</span>` +
    `<span class="rt-q">${escapeHtml(flowWords(p.peak_cms))}</span>` +
    `<span class="rt-flat">${escapeHtml(flatterWords(p.attenuation_pct))}</span></li>`;
}

function renderResult() {
  const box = $("rt-result");
  if (!box) return;
  const mine = shown && target && !shown.result.storm && !shown.result.reservoir && Number(shown.target.river_id) === target.river_id;
  box.hidden = !mine;
  if (!mine) return;
  const r = shown.result;
  $("rt-head").textContent = headline(r);
  const rows = cardPlaces(r.places, narrow() ? 3 : 8);
  $("rt-places").innerHTML = rows.length
    ? '<li class="rt-hdr"><i></i><span class="rt-name">Where</span><span class="rt-when">peak after</span>' +
      '<span class="rt-q">m³/s</span><span class="rt-flat">flatter</span></li>' + rows.map(placeRow).join("")
    : '<li class="rt-none">No town or gauge on the stretch the wave reached in view.</li>';
  const check = r.check && r.check.summary ? ` ${r.check.summary}` : "";
  const flow = designFlowLine(r.inflow && r.inflow.design);
  $("rt-model").textContent = `${flow ? `${flow} ` : ""}Muskingum, GEOGLOWS's own k and x, ${r.hours} h routed hourly.${check} ` +
    "The flow the wave adds to the river; rivers that join add their own.";
}

// ── run ─────────────────────────────────────────────────────────────────────

function townsInView() {
  if (!state.mapOk || !map || !map.getSource("openmaptiles")) return [];
  try {
    return townsFrom(map.querySourceFeatures("openmaptiles", { sourceLayer: "place" }));
  } catch { return []; }
}

/**
 * Route a flood from a reach and draw it: { river_id, lat, lon, name } and the inputs { peak_cms | return_period,
 * duration_h, shape }. Logged (undo puts back what was drawn before). Returns the result.
 */
export async function routeFlood(t, args, { by = "rules", command = "", label = "" } = {}) {
  const my = ++running;
  syncOutputs();
  const started = performance.now();
  const what = args.return_period ? `Reading the ${args.return_period}-year flow, then routing` : "Routing the wave";
  const tick = () => {
    const s = Math.round((performance.now() - started) / 1000);
    say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady ? what : "Starting the engine (once), then routing")}… ${s} s`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const res = await call("route", { op: "flood", args: { river_id: t.river_id, lat: t.lat, lon: t.lon, ...args,
      places: townsInView() } });
    if (my !== running) return res;
    if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
    const before = slotValue("route");
    const value = { result: res, target: t };
    panel().classList.add("done");
    paint(value, { play: true });
    logAction({ action: { type: "route", river_id: t.river_id, ...args }, before, after: value,
      label: label || logLabel(res), by, detail: res.inflow.source, command });
    const secs = Math.round((performance.now() - started) / 1000);
    say(`Routed in ${secs} s. The wave plays on the time bar.`, "ok");
    announce(headline(res));
    void addTowns(value);
    return res;
  } catch (err) {
    if (my === running) say(escapeHtml(`Could not route: ${err.message}`), "error");
    throw err;
  } finally {
    clearInterval(timer);
    if (my === running) running = 0;
    syncOutputs();
  }
}

/**
 * Draw a wave the package already routed (Replay an event, #659: the observed flow at a gauge, routed with
 * aquascope.routing.route in the record) and play it on the time bar, without the panel. Logged as a route, so undo
 * takes it off. `frame` false leaves the camera to the caller.
 */
export function showRouted(result, t, { frame = false, label = "", by = "rules", command = "" } = {}) {
  const before = slotValue("route");
  const value = { result, target: { ...t, river_id: Number(t.river_id) } };
  target = value.target;
  paint(value, { frame, play: true });
  logAction({ action: { type: "route", river_id: t.river_id, inflow: "series" }, before, after: value,
    label: label || logLabel(result), by, detail: (result.inflow && result.inflow.source) || (result.storm && result.storm.rule) || "",
    command });
  return value;
}

/** Play the wave on the map again from its start (a storm's card, #670). */
export function playWave() { if (shown) { takeClock(0); playOnce(); } }

/** Take the wave off the map (a replay moving on from its wave stop). */
export function clearRouted() { if (shown) paint(null); }

// The path is framed now, so the basemap's labels along it are loaded: the towns not placed yet are placed on it.
async function addTowns(value) {
  await new Promise((r) => setTimeout(r, 1800));
  if (!shown || shown.result !== value.result) return;
  const known = new Set((value.result.places || []).map((p) => p.name));
  const towns = townsInView().filter((p) => !known.has(p.name));
  if (!towns.length) return;
  const slim = { geometry: value.result.geometry, reaches: value.result.reaches.filter((r) => r.along_km !== undefined) };
  try {
    const res = await call("route", { op: "places", args: { result: slim, places: towns } });
    if (!shown || shown.result !== value.result || !res || !res.places || !res.places.length) return;
    value.result.places = [...value.result.places, ...res.places].sort((a, b) => (a.along_km || 0) - (b.along_km || 0));
    map.getSource(PTS).setData(placeFeatures(value.result.places, cardPlaces(value.result.places, narrow() ? 3 : 8)));
    renderResult();
  } catch { /* the gauges stand without the towns */ }
}

async function run() {
  if (!target || running) return;
  let args;
  try { args = routeArgs(inputs()); } catch (err) { say(escapeHtml(err.message), "warn"); return; }
  try { await routeFlood(target, args, { by: "rules" }); } catch { /* said in the panel */ }
}

// ── the AI's route action (map-actions.js) ──────────────────────────────────

/** Where "from here" starts: the reach the selection snapped to, or null. */
export const selectedReach = () => lastReach;

// ── a reach's card ──────────────────────────────────────────────────────────

function onClick(e) {
  if (!shown || !e.features || !e.features.length) return;
  e.preventDefault();
  if (e.originalEvent) e.originalEvent.stopPropagation();
  const f = e.features[0];
  const r = shown.result;
  let row = null, title = "";
  if (f.layer.id === L.dot) {
    const p = (r.places || [])[Number(f.properties.n)];
    if (!p) return;
    row = p;
    title = p.name;
  } else {
    const id = Number(f.properties.riverId);
    row = (r.reaches || []).find((x) => Number(x.river_id) === id);
    title = `River reach ${id}`;
  }
  if (!row || !actions.openMapCard) return;
  const status = row.reached
    ? `The wave's front ${hoursWords(row.front_h)} after it starts, its peak ${row.peak_h === null ? "after the hours routed" : hoursWords(row.peak_h)}.`
    : "Too little of the wave gets here to matter.";
  actions.openMapCard({
    id: `route:${title}`, lngLat: [e.lngLat.lng, e.lngLat.lat], what: "Flood wave, modelled", title,
    sub: row.along_km !== undefined && row.along_km !== null ? `${Math.round(row.along_km)} km down the river` : "",
    status: { text: status, color: RAMP[5][1] },
    figure: row.reached ? { value: flowWords(row.peak_cms), unit: " m³/s", label: r.storm ? "the peak the storm adds here"
      : `peak added, ${flatterWords(row.attenuation_pct)} on the start` } : null,
    note: "Muskingum is linear: this is the flow the wave adds to the river.", credit: CREDIT,
    buttons: [{ id: "rt-again", label: "Route a flood from here", title: "Route a flood from this reach",
      onClick: () => openRoute({ river_id: row.river_id, lat: e.lngLat.lat, lon: e.lngLat.lng, name: title }) }],
  });
}

// ── the legend row ──────────────────────────────────────────────────────────

function legendBody() {
  if (!shown) return "";
  const r = shown.result;
  const ramp = RAMP.map(([, c]) => `<i style="background:${c}"></i>`).join("");
  const when = shown.frame === null ? "Each reach at its own peak" : `At ${frameLabel(r.frames.t_h[shown.frame] || 0)}`;
  const ref = r.storm ? `the biggest peak the storm adds (${flowWords(r.frames.scale_cms)} m³/s)`
    : `the peak that entered (${flowWords(r.inflow.peak_cms)} m³/s)`;
  return `<p class="ml-when">${escapeHtml(when)}: the share of ${escapeHtml(ref)} ` +
    "that is there, wider and darker for more.</p>" +
    `<p class="rk-row"><span class="rt-ramp">${ramp}</span> 0 to 100%</p>` +
    '<p class="ml-src">Muskingum with GEOGLOWS v2\'s k and x (CC BY 4.0), rivers TDX-Hydro (CC BY-SA 4.0)' +
    (r.inflow && r.inflow.design ? ", T-year flows from GEOGLOWS v2's annual maxima (CC BY-NC-SA 4.0)" : "") + '. Modelled.</p>';
}

function registerRow() {
  registerLegendRow({
    id: "route", title: "Flood wave",
    shown: () => Boolean(shown),
    mark: () => `<span class="rt-ramp">${RAMP.filter((_, i) => i % 2 === 0).map(([, c]) => `<i style="background:${c}"></i>`).join("")}</span>`,
    summary: () => (shown ? logLabel(shown.result) : ""),
    body: legendBody,
    act: (name) => { if (name === "clear") clearByUndo(); },
  });
}

function clearByUndo() {
  const value = slotValue("route");
  paint(null);
  if (!value) return;
  logAction({ action: { type: "route", river_id: value.target && value.target.river_id }, before: value, after: null,
    label: "Clear the flood wave", by: "rules" });
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initRoute() {
  const p = panel();
  if (!p) return;
  registerSlotWriter("route", (value) => paint(value || null, { frame: false }));
  registerRow();
  for (const id of ["rt-peak", "rt-rp", "rt-dur", "rt-block"]) $(id).addEventListener("input", syncOutputs);
  for (const el of document.querySelectorAll('input[name="rt-kind"]')) el.addEventListener("change", syncOutputs);
  $("rt-peak").addEventListener("focus", () => { $("rt-kind-peak").checked = true; syncOutputs(); });
  $("rt-rp").addEventListener("focus", () => { $("rt-kind-rp").checked = true; syncOutputs(); });
  $("rt-form").addEventListener("submit", (e) => { e.preventDefault(); void run(); });
  p.querySelector(".rt-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); closePanel(); } });
  $("rt-edit").addEventListener("click", () => { p.classList.remove("done"); $("rt-run").focus({ preventScroll: true }); });
  $("rt-replay").addEventListener("click", () => { if (shown) { takeClock(0); playOnce(); } });
  for (const id of ["panel-point", "panel-station"]) {
    const el = $(id);
    if (el) el.addEventListener("reachchange", (e) => { if (e.detail) lastReach = e.detail; });
  }
  actions.openRoute = openRoute;
  actions.routeFlood = routeFlood;
  actions.selectedReach = selectedReach;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.route = { open: openRoute, run, shown: () => shown, frame: drawFrame, step: (i) => setModelStep(i),
    peak: () => releaseModelClock({ exit: true }) };
  syncOutputs();
}

