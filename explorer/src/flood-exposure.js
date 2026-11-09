// What gets flooded (#622): the buildings, the roads and the people under a flood model's water, step by step.
//
// "What gets flooded" on the model's legend row reads, around the model, Overture Maps' buildings and road
// segments (GeoParquet, ODbL) and the GHS-POP 2025 population grid (JRC, CC BY 4.0) in a light worker
// (aquascope.flood_exposure.read_sources), then counts in the mesh worker beside the file
// (aquascope.flood_exposure.exposure_payload). The flooded buildings and stretches of road light up on the map
// and follow the time bar; the legend row counts them at the step on show and at the peak, with each source,
// its licence and what the estimate cannot tell; a summary card opens on the peak. A click on a building says
// when it was under water and how deep.

import { escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { closeCard, openCard, updateCard } from "./map-card.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { addMeshTool, currentMesh, meshAsk, meshStepNow, onMeshCleared, onMeshStep } from "./mesh-view.js?v=__BUILD__";
import { buildingWords, clock, countSvg, exposureRows, exposureSummary, peopleWords, sourceLines, wetFilter } from "./flood-views-core.js?v=__BUILD__";

const BLD = "fx-buildings";
const BLD_LINE = "fx-buildings-line";
const ROADS = "fx-roads";
const ROADS_CASE = "fx-roads-case";
const ROW = "flood-exposure";
const FLOODED = "#d9480f";
const ROAD = "#8f1d06";
const READ_WAIT_MS = 180000;

let res = null;          // exposure_payload's answer
let status = "";         // the reading line while it runs, or an error
let running = 0;
let visible = true;
let bound = false;
let fc = null;           // { buildings, roads } feature collections
let read = null;         // what the light worker read, kept so a new threshold only counts again
// The depth that counts, in metres: 0.15 m by default, where the Environment Agency's surface water map says
// water typically passes the kerb (a rain-on-grid run wets every cell, so "any water" would count everything).
let thrM = 0.15;

const now = () => meshStepNow() || { view: "max", step: 0 };
const vis = () => (visible ? "visible" : "none");
const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error(`${what} took too long`)), ms))]);

// ── reading and counting ────────────────────────────────────────────────────

export async function whatGetsFlooded() {
  const m = currentMesh();
  if (!m) return null;
  if (m.meta.variable !== "depth") {
    status = "This file has no bed levels, so no depths: what gets flooded needs them.";
    registerRow();
    openLegendRow(ROW, true);
    return null;
  }
  const my = ++running;
  const name = m.meta.name;
  clearLayers();
  res = null;
  status = "Reading the buildings and roads (Overture Maps) and the population grid (GHS-POP) around the model…";
  registerRow();
  openLegendRow("mesh", false);   // one key open at a time: the model's folds while this one counts
  openLegendRow(ROW, true);
  try {
    read = await withTimeout(callLight("flood_exposure", { bbox: m.meta.bbox }, { priority: 2 }), READ_WAIT_MS, "Reading the sources");
    if (my !== running || !currentMesh() || currentMesh().meta.name !== name) return null;
    status = "Counting what the water reaches at every step, in your browser…";
    refreshLegend(ROW);
    const out = await meshAsk("exposure", { read, threshold: thrM });
    if (my !== running || !currentMesh() || currentMesh().meta.name !== name) return null;
    res = out;
    status = "";
    buildFeatures();
    drawLayers();
    syncStep();
    openSummary();
    const p = res.buildings ? res.buildings.peak : null;
    announce(p ? `${p.value} buildings flooded at the peak, ${clock(res.times[p.step])}.` : "What gets flooded: counted.");
    globalThis.__aq.exposure.result = () => res;
    return res;
  } catch (err) {
    if (my === running) { status = `Could not count what gets flooded: ${err.message || err}`; refreshLegend(ROW); }
    return null;
  }
}

// Another depth threshold: count again over what was read, and redraw.
async function recount(m) {
  if (!read || !Number.isFinite(m) || m === thrM) return;
  thrM = m;
  const my = ++running;
  try {
    const out = await meshAsk("exposure", { read, threshold: thrM });
    if (my !== running) return;
    res = out;
    buildFeatures();
    for (const [id, data] of [[BLD, fc.buildings], [ROADS, fc.roads]]) { const src = map && map.getSource(id); if (src) src.setData(data); }
    syncStep();
    announce(`Counted again for water deeper than ${thrM} m.`);
  } catch (err) {
    status = `Could not count again: ${err.message || err}`;
    refreshLegend(ROW);
  }
}

function buildFeatures() {
  const b = res.buildings ? res.buildings.features : [];
  const r = res.roads ? res.roads.pieces : [];
  fc = {
    buildings: { type: "FeatureCollection", features: b.map((f, i) => ({ type: "Feature", id: i,
      properties: { i, w: f.wet, d: f.max_depth, c: f.class || "" }, geometry: { type: "Polygon", coordinates: [f.ring] } })) },
    roads: { type: "FeatureCollection", features: r.map((p, i) => ({ type: "Feature", id: i,
      properties: { i, w: p.wet, c: p.class || "" }, geometry: { type: "LineString", coordinates: p.coords } })) },
  };
}

// ── the map ─────────────────────────────────────────────────────────────────

// Above the model's picture, under the gauges (as the mesh's own layers sit).
function aboveMesh() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => ["clusters", "points", "now-dots", "gauge-heat"].includes(l.id) || /^river-lit/.test(l.id));
  return hit ? hit.id : undefined;
}

function drawLayers() {
  if (!map || !map.getStyle() || !fc) return;
  const before = aboveMesh();
  if (!map.getSource(BLD)) {
    map.addSource(BLD, { type: "geojson", data: fc.buildings });
    map.addLayer({ id: BLD, type: "fill", source: BLD, layout: { visibility: vis() },
      paint: { "fill-color": "#5c6b78", "fill-opacity": 0.22 } }, before);
    map.addLayer({ id: BLD_LINE, type: "line", source: BLD, layout: { visibility: vis() },
      paint: { "line-color": "#3c4955", "line-width": ["interpolate", ["linear"], ["zoom"], 13, 0.2, 17, 0.9], "line-opacity": 0.5 } }, before);
  }
  if (!map.getSource(ROADS)) {
    map.addSource(ROADS, { type: "geojson", data: fc.roads });
    map.addLayer({ id: ROADS_CASE, type: "line", source: ROADS, layout: { visibility: vis(), "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1.6, 17, 4.6], "line-opacity": 0.7 } }, before);
    map.addLayer({ id: ROADS, type: "line", source: ROADS, layout: { visibility: vis(), "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ROAD, "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.7, 17, 2.4] } }, before);
  }
  if (!bound) {
    bound = true;
    map.on("click", BLD, onBuilding);
    map.on("mouseenter", BLD, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", BLD, () => { map.getCanvas().style.cursor = ""; });
  }
}

function clearLayers() {
  if (!map) return;
  for (const id of [ROADS, ROADS_CASE, BLD_LINE, BLD]) if (map.getLayer(id)) map.removeLayer(id);
  for (const id of [ROADS, BLD]) if (map.getSource(id)) map.removeSource(id);
}

function syncStep() {
  if (!res || !map) return;
  const { view, step } = now();
  const wet = wetFilter(step, view);
  if (map.getLayer(BLD)) {
    map.setPaintProperty(BLD, "fill-color", ["case", wet, FLOODED, "#5c6b78"]);
    map.setPaintProperty(BLD, "fill-opacity", ["case", wet, 0.78, 0.16]);
    map.setPaintProperty(BLD_LINE, "line-color", ["case", wet, "#7a2604", "#3c4955"]);
  }
  for (const id of [ROADS, ROADS_CASE]) if (map.getLayer(id)) map.setFilter(id, wet);
  refreshLegend(ROW);
  if (summaryOpen) updateCard("fx:summary", summarySpec());
}

function setVisible(on) {
  visible = Boolean(on);
  for (const id of [BLD, BLD_LINE, ROADS, ROADS_CASE]) if (map && map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis());
  refreshLegend(ROW);
}

// ── the cards ───────────────────────────────────────────────────────────────

let summaryOpen = false;

function centreOfFlooded() {
  const f = res.buildings && res.buildings.features.filter((x) => x.wet.includes("1"));
  if (f && f.length) {
    let sx = 0, sy = 0;
    for (const x of f) { sx += x.ring[0][0]; sy += x.ring[0][1]; }
    return [sx / f.length, sy / f.length];
  }
  const b = currentMesh().meta.bbox;
  return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
}

function summarySpec() {
  const { view, step } = now();
  const p = res.buildings ? res.buildings.peak : res.roads ? res.roads.peak : res.population.peak;
  const rows = exposureRows(res, step, view);
  const head = rows.map((r) => r.value).join(" · ");
  const when = view === "max" ? "at some point in the run" : `at ${clock(res.times[step], { day: true })}`;
  const series = res.buildings ? res.buildings.per_step : res.roads ? res.roads.per_step_km : res.population.per_step;
  const peakBits = [];
  if (res.buildings) peakBits.push(`${res.buildings.peak.value.toLocaleString("en-GB")} buildings`);
  if (res.roads) peakBits.push(`${res.roads.peak.value.toLocaleString("en-GB", { maximumFractionDigits: 1 })} km of road`);
  if (res.population) peakBits.push(peopleWords(res.population.peak.value));
  return {
    title: head, sub: `flooded ${when}`,
    chart: { label: `${res.buildings ? "Buildings" : res.roads ? "Road length" : "People"} flooded over the run`,
      svg: countSvg(series, { w: 300, h: 60, step: view === "max" ? -1 : step }),
      alt: `Flooded count at each step, highest at ${clock(res.times[p.step])}`,
      key: escapeHtml(`At the peak, ${clock(res.times[p.step], { day: true })}: ${peakBits.join(", ")}. Water deeper than ${res.threshold_m} m counts.`) },
    note: "Estimates: footprints and population grids are approximate, and a bridge above the water still counts as a flooded road.",
    credit: sourceLines(res).map((s) => s.text).join(" · "),
  };
}

function openSummary() {
  if (!res) return;
  summaryOpen = true;
  openCard({ id: "fx:summary", lngLat: centreOfFlooded(), what: "What gets flooded", ...summarySpec(),
    buttons: [] }, { focus: false });
}

function onBuilding(e) {
  if (!res || !e.features || !e.features.length) return;
  const p = e.features[0].properties;
  const { view, step } = now();
  const words = buildingWords({ w: p.w, d: p.d, c: p.c }, res.times, { unit: res.units.depth || "m", step, view });
  summaryOpen = false;
  openCard({ id: `fx:b${p.i}`, lngLat: [e.lngLat.lng, e.lngLat.lat], what: "Building", title: words.title,
    sub: "Overture Maps footprint", status: { text: words.status, color: String(p.w).includes("1") ? FLOODED : null },
    note: words.note, credit: res.sources.buildings ? `${res.sources.buildings.credit}, ${res.sources.buildings.licence}` : "" });
}

// ── the legend row ──────────────────────────────────────────────────────────

function body() {
  if (!res) return `<p class="fx-status">${status ? `${status.startsWith("Could not") || status.startsWith("This file") ? "" : '<span class="ms-spin" aria-hidden="true"></span>'}${escapeHtml(status)}` : ""}</p>`;
  const { view, step } = now();
  const when = view === "max" ? "Over the whole run" : `At ${clock(res.times[step], { day: true })}`;
  const series = { buildings: res.buildings && res.buildings.per_step, roads: res.roads && res.roads.per_step_km,
    people: res.population && res.population.per_step };
  const rows = exposureRows(res, step, view).map((r) =>
    `<li class="fx-${r.key}"><i class="fx-mark" aria-hidden="true"></i><div><b>${escapeHtml(r.value)}</b>` +
    `<span>${escapeHtml(r.sub)}</span></div>${series[r.key] ? countSvg(series[r.key], { w: 56, h: 22, step: view === "max" ? -1 : step }) : ""}</li>`).join("");
  const errs = Object.entries(res.errors || {}).map(([k, v]) => `<p class="fx-err">${escapeHtml(k)} could not be read: ${escapeHtml(v)}</p>`).join("");
  const limits = Object.entries(res.limits || {}).filter(([k]) => res[k === "people" ? "population" : k])
    .map(([k, v]) => `<li><b>${escapeHtml(k[0].toUpperCase() + k.slice(1))}.</b> ${escapeHtml(v)}</li>`).join("");
  const srcs = sourceLines(res).map((s) => `<li>${escapeHtml(s.text)}. <a href="${escapeHtml(s.link)}" target="_blank" rel="noopener">Attribution</a></li>`).join("");
  const opts = (res.thresholds_m || []).map((t) => `<label title="${escapeHtml(t.words)}"><input type="radio" name="fx-thr" ` +
    `data-act="thr" value="${t.m}"${Math.abs(t.m - res.threshold_m) < 1e-6 ? " checked" : ""}> ${t.m < 0.1 ? "any" : `${t.m.toFixed(2)} m`}</label>`).join("");
  const chosen = (res.thresholds_m || []).find((t) => Math.abs(t.m - res.threshold_m) < 1e-6);
  return `<p class="fx-when">${escapeHtml(when)}</p><ul class="fx-rows">${rows}</ul>${errs}` +
    `<div class="fx-thr"><span>Counts water deeper than</span>${opts}</div>` +
    (chosen ? `<p class="fx-thr-words">${escapeHtml(chosen.words)}${chosen.m >= 0.1 ? ` <a href="${escapeHtml(res.threshold_source)}" target="_blank" rel="noopener">Source</a>` : ""}</p>` : "") +
    `<details class="fx-more"><summary>Sources and limits</summary><ul class="fx-src">${srcs}</ul>` +
    `<ul class="fx-lim">${limits}</ul></details>` +
    `<p class="mv-actions"><button type="button" class="fd-go" data-act="summary">Summary</button>` +
    `<button type="button" class="fd-go" data-act="clear">Remove</button></p>`;
}

function registerRow() {
  registerLegendRow({
    id: ROW,
    title: "What gets flooded",
    mark: () => `<i class="fx-legend-mark" style="background:${FLOODED}"></i>`,
    summary: () => (res ? exposureSummary(res, now().step, now().view) : status ? "reading…" : ""),
    on: () => visible,
    shown: () => Boolean(currentMesh()) && Boolean(res || status),
    toggle: (on) => setVisible(on),
    body,
    act: (name, el) => {
      if (name === "thr") void recount(Number(el.value));
      else if (name === "summary") openSummary();
      else if (name === "clear") clearAll();
    },
  });
}

function clearAll() {
  running++;
  res = null;
  read = null;
  status = "";
  fc = null;
  clearLayers();
  if (summaryOpen) { summaryOpen = false; closeCard(); }
  refreshLegend();
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initFloodExposure() {
  addMeshTool({ id: "exposure", label: "What gets flooded", title: "Buildings, roads and people under the water, step by step, from open data",
    run: () => { void whatGetsFlooded(); } });
  onMeshStep(() => syncStep());
  onMeshCleared(() => clearAll());
  if (state.mapOk && map) map.on("style.load", () => setTimeout(() => { if (res) { drawLayers(); syncStep(); } }, 0));
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.exposure = { run: whatGetsFlooded, result: () => res };
}
