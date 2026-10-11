// What if, on the map (#565). From the map card, "What if…" opens a small panel on the map: two sliders and a
// demand (keyless), and one line read into those values by the keyless rules, the on-device model or the
// reader's own key (the model only fills the parameters; aquascope.whatif_map checks them against the words).
// Run calibrates GR4J for the basin in the worker (aquascope.whatif_map.whatif, the same function as
// `aquascope whatif` and the MCP tool) and the answer is painted on the rivers: the basin in its change, the
// river below it to the next major confluences fading as other rivers join, a label with the change where it
// starts and after each confluence, a before/after swipe, and a map card per reach. Every run is an entry in
// the action log (map-actions.js), so undo puts back what was painted before.
//
// whatif-core.js decides the colours, the words and the swipe's split; every number comes from Python.

import { $, actions, clickLayers, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { STREAMS_PMTILES } from "./river-core.js?v=__BUILD__";
import { clearRiverNetwork } from "./river-map.js?v=__BUILD__";
import { clearCatchment } from "./basins.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import {
  CLASSES, MEASURES, NEUTRAL, basinRows, beforeSet, classOf, lineColorExpr, lineFilterExpr, logLabel, outletLines,
  paintedIds, pct, reachCard, resultBounds, scenarioWords, sliderValues,
} from "./whatif-core.js?v=__BUILD__";

const L = {
  net: "river-wi-net", basin: "river-wi-basin", casing: "river-wi-casing", line: "river-wi-line",
  pts: "river-wi-pts", ring: "river-wi-ring", label: "river-wi-label",
};
const EMPTY = { type: "FeatureCollection", features: [] };
const CREDIT = "GR4J (aquascope) with ERA5 via Open-Meteo, CC BY 4.0; GEOGLOWS v2, CC BY 4.0; rivers TDX-Hydro, CC BY-SA 4.0. Modelled.";

let target = null;      // what the panel is about: { kind, river_id, lat, lon, source, station_id, name }
let shown = null;       // the painted value: { result, measure, target }
let calib = null;       // { key, params }: the last calibration, reused for the next scenario on the same basin
let running = 0;
let swipeOn = false;
let split = 0.5;        // the swipe line, as a share of the map's width
let where = new Map();  // river id -> longitude of its drawn line (for the swipe)
let hooked = false;

const panel = () => $("whatif");
const targetKey = (t) => (t ? `${t.kind}:${t.river_id}:${t.source || ""}:${t.station_id || ""}` : "");

// ── the layers ──────────────────────────────────────────────────────────────

function beforeGauges() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-fa-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getLayer(L.line)) return true;
  try {
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const before = beforeGauges();
    if (!map.getSource(L.net)) map.addSource(L.net, { type: "vector", url: `pmtiles://${STREAMS_PMTILES}` });
    if (!map.getSource(L.pts)) map.addSource(L.pts, { type: "geojson", data: EMPTY });
    const none = ["boolean", false];
    const lineLayout = { "line-cap": "round", "line-join": "round" };
    // The basin: every reach in the outlet's change, thinner, the way the lumped model sees it.
    map.addLayer({ id: L.basin, type: "line", source: L.net, "source-layer": "streams", filter: none, layout: lineLayout,
      paint: { "line-color": NEUTRAL, "line-opacity": 0.8,
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.8, 8, 1.6, 12, 3] } }, before);
    // The way down: a dark casing under a wide line, so the pale classes read on a light basemap too.
    map.addLayer({ id: L.casing, type: "line", source: L.net, "source-layer": "streams", filter: none, layout: lineLayout,
      paint: { "line-color": "rgba(15, 28, 38, 0.55)", "line-width": ["interpolate", ["linear"], ["zoom"], 3, 4, 7, 6.5, 12, 10] } }, before);
    map.addLayer({ id: L.line, type: "line", source: L.net, "source-layer": "streams", filter: none, layout: lineLayout,
      paint: { "line-color": NEUTRAL, "line-width": ["interpolate", ["linear"], ["zoom"], 3, 2.4, 7, 4.2, 12, 7] } }, before);
    // The outlet a ring, each major confluence a small dot; beside each, the change there.
    const outlet = ["==", ["get", "kind"], "outlet"];
    map.addLayer({ id: L.ring, type: "circle", source: L.pts,
      paint: { "circle-radius": ["case", outlet, 7, 3.6], "circle-color": ["case", outlet, "rgba(0,0,0,0)", "#0f1c26"],
        "circle-stroke-width": ["case", outlet, 2.5, 1.5], "circle-stroke-color": ["case", outlet, "#0f1c26", "#ffffff"] } });
    map.addLayer({ id: L.label, type: "symbol", source: L.pts,
      layout: { "text-field": ["get", "text"], "text-font": ["Noto Sans Bold"], "text-size": 12.5,
        // The outlet's label clears the click's pin, which stands on the same point.
        "text-anchor": "left", "text-offset": ["case", outlet, ["literal", [1.4, 0.6]], ["literal", [0.9, 0]]],
        "text-allow-overlap": true, "symbol-sort-key": ["get", "rank"] },
      paint: { "text-color": "#0f1c26", "text-halo-color": "rgba(255,255,255,0.92)", "text-halo-width": 1.8 } });
    hook();
    return true;
  } catch (err) {
    console.info("what-if layers unavailable:", err && err.message);
    return false;
  }
}

function hook() {
  if (hooked) return;
  hooked = true;
  for (const id of [L.line, L.basin]) {
    clickLayers.add(id);   // its own handler answers; the map-wide one leaves it alone
    map.on("click", id, onReachClick);
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
  // A basemap change rebuilds the style: put the painting back.
  map.on("style.load", () => { if (shown) { ensureLayers(); draw(); } });
  // Once the tiles are in, the drawn lines say where each reach is, for the swipe and the labels. The flowing
  // rivers keep the map from ever going idle, so this looks a few times after a run or a move instead.
  map.on("moveend", () => { if (shown) scheduleLearn(); });
  map.on("move", () => { if (shown && swipeOn) drawSwipe(); });
}

let learnTimers = [];
function scheduleLearn() {
  for (const t of learnTimers) clearTimeout(t);
  learnTimers = [700, 2000, 4500, 9000].map((ms) => setTimeout(() => { if (shown && learnPlaces()) draw(); }, ms));
}

// Each painted reach's longitude and a point on it, read from the stream tiles in view.
let points = new Map();
function learnPlaces() {
  if (!shown || !map.getSource(L.net)) return false;
  const want = new Set(paintedIds(shown.result));
  let feats = [];
  try {
    feats = map.querySourceFeatures(L.net, { sourceLayer: "streams", filter: lineFilterExpr([...want]) });
  } catch { return false; }
  let learnt = false;
  for (const f of feats) {
    const id = Number(f.properties.riverId);
    if (!want.has(id) || points.has(id)) continue;
    const g = f.geometry;
    const line = g.type === "LineString" ? g.coordinates : g.type === "MultiLineString" ? g.coordinates.flat() : [];
    if (line.length < 2) continue;
    const mid = line[Math.floor(line.length / 2)];
    points.set(id, mid);
    where.set(id, mid[0]);
    learnt = true;
  }
  return learnt;
}

function lonOf(id) {
  if (where.has(id)) return where.get(id);
  const r = shown && shown.result.reaches.find((x) => Number(x.river_id) === id);
  if (r && Number.isFinite(r.lon)) return r.lon;
  const c = shown && shown.result.basin && shown.result.basin.centre;
  return c ? c[1] : null;
}

function splitLon() {
  if (!swipeOn || !map) return null;
  const c = map.getContainer();
  try { return map.unproject([c.clientWidth * split, c.clientHeight / 2]).lng; } catch { return null; }
}

function labelFeatures(result, measure, before) {
  const key = MEASURES[measure].key;
  const feats = [];
  const reaches = result.reaches || [];
  const marks = reaches.filter((r, i) => i === 0 || r.confluence);
  marks.forEach((r, i) => {
    const id = Number(r.river_id);
    // The outlet at the snapped point; a confluence on its drawn line once the tiles have it (the routing
    // tables' position is the unit catchment's centre, which can sit off the river).
    const snapped = i === 0 && result.reach && Number.isFinite(result.reach.lon) ? [result.reach.lon, result.reach.lat] : null;
    const at = snapped || points.get(id);
    if (!at) return;
    if (before.has(id)) return;   // left of the swipe: the river as it is, no change to label
    feats.push({ type: "Feature", geometry: { type: "Point", coordinates: at },
      properties: { kind: i === 0 ? "outlet" : "confluence", rank: i, text: pct(r[key]) } });
  });
  return { type: "FeatureCollection", features: feats };
}

function draw() {
  if (!shown || !ensureLayers()) return;
  const { result, measure } = shown;
  const down = result.reaches || [];
  const basin = basinRows(result);
  const lon = splitLon();
  const before = lon === null ? new Set() : beforeSet(paintedIds(result), lonOf, lon);
  map.setFilter(L.basin, lineFilterExpr(basin.map((r) => r.river_id)));
  map.setPaintProperty(L.basin, "line-color", lineColorExpr(basin, measure, before));
  for (const id of [L.casing, L.line]) map.setFilter(id, lineFilterExpr(down.map((r) => r.river_id)));
  map.setPaintProperty(L.line, "line-color", lineColorExpr(down, measure, before));
  map.getSource(L.pts).setData(labelFeatures(result, measure, before));
}

function clearLayers() {
  if (!state.mapOk || !map || !map.getLayer(L.line)) return;
  for (const id of [L.basin, L.casing, L.line]) map.setFilter(id, ["boolean", false]);
  if (map.getSource(L.pts)) map.getSource(L.pts).setData(EMPTY);
}

// ── the painted value: drawn, cleared, put back by undo ─────────────────────

function paint(value, { frame = true } = {}) {
  shown = value;
  points = new Map();
  where = new Map();
  if (!value) {
    clearLayers();
    setSwipe(false);
    renderResult();
    refreshLegend("whatif");
    return;
  }
  // The click's own lit network (what drains here, the way to the sea) and its HydroBASINS wash would compete
  // with the painting: the basin here is the one the model ran on, drawn in its change.
  if (frame) { clearRiverNetwork(); clearCatchment(); }
  draw();
  scheduleLearn();
  if (frame) frameResult(value.result);
  renderResult();
  openLegendRow("whatif", true);
}

const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

// Frame the basin and the river to the first major confluence in the part of the map the panel leaves free.
function frameResult(result) {
  const box = resultBounds(result);
  if (!box || !state.mapOk || !map) return;
  const p = panel();
  const pad = { top: 70, left: 40, right: 40, bottom: 50 };
  if (p && !p.hidden) {
    if (narrow()) pad.bottom = Math.min(p.offsetHeight + 24, map.getContainer().clientHeight * 0.6);
    else pad.right = p.offsetWidth + 40;
  }
  // The legend sits bottom left on a wide screen: keep the painting clear of it.
  const lg = $("map-legend");
  if (lg && !lg.hidden && !narrow()) {
    const r = lg.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
    if (r.width && r.right < c.left + c.width / 2) pad.left = Math.max(pad.left, r.right - c.left + 24);
  }
  try { map.fitBounds(box, { padding: pad, maxZoom: 11, duration: 700 }); } catch { /* the map is not ready */ }
}

// ── the panel ───────────────────────────────────────────────────────────────

function sliders() {
  return {
    dp_pct: Number($("wi-rain").value) || 0,
    dt_c: Number($("wi-warm").value) || 0,
    demand_cms: Math.max(0, Number($("wi-demand").value) || 0),
  };
}

function showSliders(v) {
  $("wi-rain").value = String(v.dp_pct);
  $("wi-warm").value = String(v.dt_c);
  $("wi-demand").value = v.demand_cms ? String(v.demand_cms) : "";
  syncOutputs();
}

function syncOutputs() {
  const v = sliders();
  $("wi-rain-out").textContent = pct(v.dp_pct);
  $("wi-warm-out").textContent = `${v.dt_c > 0 ? "+" : v.dt_c < 0 ? "−" : ""}${Math.abs(v.dt_c)} °C`;
  $("wi-run").disabled = Boolean(running) || !target || (!v.dp_pct && !v.dt_c && !v.demand_cms);
  $("wi-run").textContent = running ? "Running…" : "Run";
  $("wi-scn").textContent = scenarioWords(v);
}

function say(html, kind = "") {
  const el = $("wi-status");
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

export function openWhatIf(t) {
  if (!t || t.river_id === null || t.river_id === undefined) return;
  if (targetKey(t) !== targetKey(target)) say("");
  target = { ...t, river_id: Number(t.river_id) };
  const p = panel();
  p.hidden = false;
  $("wi-title").textContent = t.name ? `${t.name}` : `River reach ${target.river_id}`;
  $("wi-sub").textContent = t.kind === "gauge"
    ? "GR4J calibrated on this gauge's record"
    : "GR4J calibrated on this reach's GEOGLOWS simulated flow";
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  p.classList.remove("done");
  syncOutputs();
  renderResult();
  $("wi-text").focus({ preventScroll: true });
}

function closePanel() {
  panel().hidden = true;
}

function renderResult() {
  const box = $("wi-result");
  if (!box) return;
  const mine = shown && targetKey(shown.target) === targetKey(target);
  box.hidden = !mine;
  if (!mine) return;
  const r = shown.result;
  const m = r.model || {};
  const v = m.validation || {};
  const lines = outletLines(r);
  $("wi-measures").innerHTML = lines.map((l) =>
    `<button type="button" class="wi-m" data-m="${l.key}" aria-pressed="${shown.measure === l.key}">` +
    `<i style="background:${escapeHtml(l.color)}"></i><span class="wi-m-label">${escapeHtml(l.label)}</span>` +
    `<b>${escapeHtml(l.text)}</b></button>`).join("");
  const kge = Number.isFinite(v.kge) ? v.kge.toFixed(2) : "n/a";
  const ens = m.ensemble ? `; range from ${m.ensemble.kept} of ${m.ensemble.tried} parameter sets` : "";
  $("wi-model").textContent = `At the outlet, ${Math.round(r.outlet.area_km2).toLocaleString("en-US")} km². ` +
    `Validation KGE ${kge} (${m.skill_grade || "ungraded"})${ens}.`;
  $("wi-swipe-on").checked = swipeOn;
}

// ── run ─────────────────────────────────────────────────────────────────────

async function run({ by = "sliders", command = "" } = {}) {
  if (!target || running) return;
  const t = target;
  const v = sliders();
  const my = ++running;
  syncOutputs();
  const started = performance.now();
  const reuse = calib && calib.key === targetKey(t) ? calib.params : null;
  const what = reuse ? "Running the scenario" : `Calibrating GR4J on ${t.kind === "gauge" ? "the gauge" : "the reach's simulated flow"}, then the scenario`;
  const tick = () => {
    const s = Math.round((performance.now() - started) / 1000);
    say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady ? what : "Starting the engine (once), then calibrating")}… ${s} s`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const args = {
      river_id: t.river_id, lat: t.lat, lon: t.lon, dp_pct: v.dp_pct, dt_c: v.dt_c, demand_cms: v.demand_cms,
      ...(t.kind === "gauge" ? { source: t.source, station_id: t.station_id } : {}),
      ...(reuse ? { params: reuse } : {}),
    };
    const res = await call("whatif", { op: "run", args });
    if (my !== running) return;
    if (!res || !res.ok) {
      say(escapeHtml(`Could not run: ${(res && res.error) || "no answer"}`), "warn");
      return;
    }
    if (res.model && res.model.params) calib = { key: targetKey(t), params: res.model.params };
    const before = slotValue("whatif");
    const value = { result: res, measure: (shown && shown.measure) || "mean", target: t };
    panel().classList.add("done");   // a phone folds the sliders away so the map shows the answer
    paint(value);
    logAction({ action: { type: "what_if", river_id: t.river_id, scenario: res.scenario }, before, after: value,
      label: logLabel(res), by: by === "sliders" ? "rules" : by, detail: res.scenario.label, command });
    const secs = Math.round((performance.now() - started) / 1000);
    say(`${escapeHtml(res.scenario.label)}: painted on the rivers in ${secs} s.`, "ok");
    announce(`What if ${res.scenario.label}: mean flow ${pct(res.outlet.mean.change_pct)} at the outlet`);
  } catch (err) {
    if (my === running) say(escapeHtml(`Could not run: ${err.message}`), "error");
  } finally {
    clearInterval(timer);
    if (my === running) running = 0;
    syncOutputs();
  }
}

// ── the line: rules, then the device's model, then the reader's key ─────────

async function readLine(text) {
  const words = String(text || "").trim();
  if (!words) return;
  say(`<span class="spinner" aria-hidden="true"></span>Reading…`);
  try {
    let read = await callLight("whatif_words", { op: "parse", text: words }, { priority: 3 });
    let who = "rules", by = "rules";
    if (!read || !read.matched) {
      read = null;
      if (await localModelReady()) {
        const prompt = await callLight("whatif_words", { op: "prompt", context: currentContext() }, { priority: 3 });
        const reply = await generateJsonLocally({ system: prompt.system, prompt: words, schema: prompt.schema, timeoutMs: 25000 })
          .catch(() => null);
        if (reply) {
          const checked = await callLight("whatif_words", { op: "reply", reply, text: words }, { priority: 3 });
          if (checked && checked.scenario) { read = checked; who = localReaderLabel() || "the on-device model"; by = "device"; }
        }
      }
      const cfg = !read && askModelConfig();
      if (cfg) {
        say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(cfg.label)}…`);
        const res = await call("whatif", { op: "llm", text: words, context: currentContext(), provider: cfg.provider,
          model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url });
        if (res && res.scenario) { read = res; who = res.model || cfg.label; by = "key"; }
      }
    }
    if (!read || !read.scenario) {
      const how = askModelConfig() ? "" : " With Chrome's built-in model, or a key in Ask ✨, freer wording works too.";
      say(`Not read. Try “20% less rain and 2 degrees warmer”, or the sliders.${escapeHtml(how)}`, "warn");
      return;
    }
    const v = sliderValues(read.scenario);
    showSliders(v);
    const refused = (read.errors || []).length ? ` (${read.errors.join("; ")})` : "";
    $("wi-read").textContent = `Read by ${who}: ${scenarioWords(v)}${refused}`;
    $("wi-text").blur();
    $("wi-text").scrollLeft = 0;
    $("wi-read").hidden = false;
    await run({ by, command: words });
  } catch (err) {
    say(escapeHtml(`Could not read that: ${err.message}`), "error");
  }
}

// ── a painted reach's card ──────────────────────────────────────────────────

function onReachClick(e) {
  if (!shown || !e.features || !e.features.length) return;
  e.preventDefault();
  if (e.originalEvent) e.originalEvent.stopPropagation();
  const id = Number(e.features[0].properties.riverId);
  const c = reachCard(shown.result, id);
  if (!c || !actions.openMapCard) return;
  actions.openMapCard({
    id: `whatif:${id}`, lngLat: [e.lngLat.lng, e.lngLat.lat], what: `What if: ${shown.result.scenario.label}`,
    title: c.title, status: { text: c.status, color: c.color }, figure: c.figure, note: c.note, credit: CREDIT,
    buttons: [{ id: "wi-again", label: "What if… here", title: "A scenario for the basin draining to this reach",
      onClick: () => openWhatIf({ kind: "reach", river_id: id, lat: e.lngLat.lat, lon: e.lngLat.lng }) }],
  });
}

// ── the swipe ───────────────────────────────────────────────────────────────

function setSwipe(on) {
  swipeOn = Boolean(on) && Boolean(shown);
  const el = $("wi-swipe");
  if (el) el.hidden = !swipeOn;
  if ($("wi-swipe-on")) $("wi-swipe-on").checked = swipeOn;
  if (swipeOn) { learnPlaces(); drawSwipe(); }
  else if (shown) draw();
  refreshLegend("whatif");
}

function drawSwipe() {
  const el = $("wi-swipe");
  if (!el || !swipeOn) return;
  const x = Math.round(el.clientWidth * split);
  el.querySelector(".wi-sw-line").style.left = `${x}px`;
  el.querySelector(".wi-sw-handle").style.left = `${x}px`;
  el.querySelector(".wi-sw-handle").setAttribute("aria-valuenow", String(Math.round(split * 100)));
  $("wi-sw-after").textContent = `After: ${shown ? shown.result.scenario.label : ""}`;
  el.style.setProperty("--x", `${x}px`);
  draw();
}

function initSwipe() {
  const el = $("wi-swipe");
  const handle = el.querySelector(".wi-sw-handle");
  let frame = 0;
  const move = (clientX) => {
    const box = el.getBoundingClientRect();
    split = Math.min(0.95, Math.max(0.05, (clientX - box.left) / box.width));
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawSwipe(); });
  };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const onMove = (ev) => move(ev.clientX);
    const onUp = () => { handle.removeEventListener("pointermove", onMove); handle.removeEventListener("pointerup", onUp); };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
  });
  handle.addEventListener("keydown", (e) => {
    const step = e.key === "ArrowLeft" ? -0.05 : e.key === "ArrowRight" ? 0.05 : 0;
    if (!step) return;
    e.preventDefault();
    split = Math.min(0.95, Math.max(0.05, split + step));
    drawSwipe();
  });
}

// ── the legend row ──────────────────────────────────────────────────────────

function legendBody() {
  if (!shown) return "";
  const r = shown.result;
  const key = MEASURES[shown.measure].key;
  const used = new Set([...(r.reaches || []).map((x) => classOf(x[key]))]);
  const rows = CLASSES.map((c, i) => (used.has(i)
    ? `<p class="rk-row"><i class="rk-sw" style="background:${c.color}"></i>${escapeHtml(c.label)}</p>` : "")).join("");
  const down = r.downstream || {};
  return `<p class="ml-when">Change in ${escapeHtml(MEASURES[shown.measure].label)}: the basin, then ` +
    `${down.n_reaches || 0} reaches to ${down.major_confluences || 0} major confluence${down.major_confluences === 1 ? "" : "s"}.</p>` +
    rows +
    `<label class="ml-check"><input type="checkbox" data-act="swipe" ${swipeOn ? "checked" : ""}> Before/after swipe</label>` +
    "<p class=\"ml-src\">Modelled: GR4J with ERA5 and GEOGLOWS v2 (CC BY 4.0), rivers TDX-Hydro (CC BY-SA 4.0). " +
    "Downstream by area weighting, the other rivers unchanged.</p>";
}

function registerRow() {
  registerLegendRow({
    id: "whatif", title: "What if",
    shown: () => Boolean(shown),
    mark: () => `<span class="wi-mark">${[0, 2, 4, 6, 8].map((i) => `<i style="background:${CLASSES[i].color}"></i>`).join("")}</span>`,
    summary: () => (shown ? shown.result.scenario.label : ""),
    body: legendBody,
    act: (name, el) => {
      if (name === "swipe") setSwipe(el.checked);
      if (name === "clear") clearByUndo();
    },
  });
}

// The legend's ×: the painting goes the way the log's undo would take it, so the log stays true.
function clearByUndo() {
  const value = slotValue("whatif");
  if (!value) { paint(null); return; }
  const before = value;
  paint(null);
  logAction({ action: { type: "what_if", river_id: before.target && before.target.river_id, scenario: null }, before,
    after: null, label: "Clear the what-if", by: "rules" });
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initWhatIf() {
  const p = panel();
  if (!p) return;
  registerSlotWriter("whatif", (value) => paint(value || null, { frame: false }));
  registerRow();
  for (const id of ["wi-rain", "wi-warm", "wi-demand"]) $(id).addEventListener("input", syncOutputs);
  $("wi-form").addEventListener("submit", (e) => { e.preventDefault(); void run({ by: "sliders" }); });
  $("wi-line").addEventListener("submit", (e) => { e.preventDefault(); void readLine($("wi-text").value); });
  p.querySelector(".wi-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); closePanel(); } });
  $("wi-measures").addEventListener("click", (e) => {
    const b = e.target.closest(".wi-m");
    if (!b || !shown) return;
    shown = { ...shown, measure: b.dataset.m };
    draw();
    renderResult();
    refreshLegend("whatif");
  });
  $("wi-swipe-on").addEventListener("change", (e) => setSwipe(e.target.checked));
  $("wi-edit").addEventListener("click", () => { p.classList.remove("done"); $("wi-text").focus({ preventScroll: true }); });
  initSwipe();
  actions.openWhatIf = openWhatIf;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.whatIf = { open: openWhatIf, run, read: readLine, shown: () => shown, setSwipe, setSplit: (s) => { split = s; drawSwipe(); } };
  syncOutputs();
}
