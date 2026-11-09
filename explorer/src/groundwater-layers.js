// Groundwater views (#625, #626): what ModelMuse and Groundwater Vistas show beside the plan view, for the MODFLOW
// model on the map (groundwater-view.js loads it and calls in here).
//
// - The layers in 3D: every layer between its top and bottom, extruded on the map and coloured by head, with the
//   vertical exaggeration said and changeable, and a way back to 2D.
// - A cross-section along a line you draw: the layers' tops and bottoms, each layer's head, the water table,
//   stepping with the time bar.
// - A cell's head over time in every layer, on its map card.
// - The water budget from the cell-by-cell budget file (.cbc), dropped with the other files: in and out by term
//   over time, the current time's numbers beside it.
//
// The worker reads the model again into its own Python (op "open" of "gw_layers"), with aquascope.groundwater.layers
// and .budget: every number here comes from there. groundwater-layers-core.js draws them (node tests it).

import { claimClicks, clickLayers, dbg, escapeHtml, setTime, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { updateCard } from "./map-card.js?v=__BUILD__";
import { refreshLegend } from "./map-legend.js?v=__BUILD__";
import { lengthUnit } from "./groundwater-core.js?v=__BUILD__";
import {
  budgetRows, budgetStep, budgetSvg, dateWords, distWords, everWet, exaggerationChoices, hydrographSvg, hydrographWords,
  headRange, layerLine, prismFeatures, rateWords, sectionAt, sectionRange, sectionSvg, veWords,
} from "./groundwater-layers-core.js?v=__BUILD__";

const L = { prisms: "gwl-3d", line: "gwl-line", pts: "gwl-line-pts", catcher: "gwl-catch" };
const EMPTY = { type: "FeatureCollection", features: [] };
const WORLD = { type: "Feature", properties: {}, geometry: { type: "Polygon",
  coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]] } };

let ids = { fill: "gw-model", line: "gw-model-line" };
let current = null;        // the model groundwater-view.js shows (its object, read, never changed here)
let token = 0;             // which model the worker's answer belongs to
let views = null;          // { geo, budget, layersError, budgetError, busy }
let wet = null;            // which layer-cells ever have a head
let view3d = false;
let ve = 1;
let exploded = false;      // the layers pulled apart in 3D, to see each one's top
let shown = null;          // the layer indexes drawn in 3D
let section = null;        // the last cross-section (with its arrays), and its line
let sectionBusy = false;
let sectionError = null;
let drawing = null;        // { pts: [[lon, lat]], unclaim } while a line is being drawn
let dockTab = "section";
let closeLook = false;     // the section's vertical range around the heads, not the whole depth
let dock = null;
let markers = [];
let saved2d = null;        // the camera before 3D, to go back to

const vis = (on) => (on ? "visible" : "none");
const haveGeo = () => Boolean(views && views.geo);
const haveBudget = () => Boolean(views && views.budget);

// ── the worker ──────────────────────────────────────────────────────────────

/** A model was drawn: read its layers (and budget, when a .cbc came with it) in the worker. */
export async function layersLoaded(model, sorted) {
  current = model;
  const my = ++token;
  views = { geo: null, budget: null, layersError: null, budgetError: null, busy: true };
  wet = everWet(model.heads, model.shape);
  section = null; sectionError = null;
  shown = null;
  refreshLegend("modflow");
  try {
    const read = (f) => f.arrayBuffer();
    const out = await call("gw_layers", {
      op: "open", hds: await read(sorted.hds), grid: sorted.grid ? await read(sorted.grid) : null,
      gridName: sorted.grid ? sorted.grid.name : null, sidecars: await Promise.all(sorted.sidecars.map(read)),
      cbc: sorted.cbc ? await read(sorted.cbc) : null, place: model.place || {},
    });
    if (my !== token) return;
    if (out.error) throw new Error(out.error);
    if (out.layers && out.top && out.botm) {
      views.geo = { ...out.layers, top: out.top, botm: out.botm, tops: out.tops || null, nlay: out.layers.layers.length };
      ve = out.layers.exaggeration || 1;
      shown = new Set(out.layers.layers.map((_, i) => i));
    }
    views.layersError = out.layers_error || null;
    views.budget = out.budget || null;
    views.budgetError = out.budget_error || null;
    if (views.budget && !section) openDock("budget");   // a budget dropped with the heads is what was asked for
  } catch (err) {
    if (my !== token) return;
    views.layersError = err && err.message ? err.message : String(err);
  } finally {
    if (my === token) { views.busy = false; refreshLegend("modflow"); if (view3d) paint3d(); }
  }
}

/** The model left the map. */
export function layersRemoved() {
  token++;
  current = null;
  views = null;
  section = null;
  stopDrawing();
  if (view3d) set3d(false);
  closeDock();
  clearLine();
  void call("gw_layers", { op: "close" }).catch(() => {});
}

/** The model's frame changed (time step, layer) or the style was reloaded: redraw what follows it. */
export function layersFrame(model) {
  current = model;
  if (view3d) paint3d();
  if (section) drawLine(section.line);
  renderDock();
}

/** Whether a line is being drawn (a click on the model then adds a point, it does not open a card). */
export const layersDrawing = () => Boolean(drawing);
/** Whether the 3D stack stands (the plan view's contours, arrows and zones of #623/#624 then step aside: drawn on
 * the ground, they would show through the stack's top as clutter, out of place). */
export const layers3d = () => view3d;
let onView = null;

// ── 3D ──────────────────────────────────────────────────────────────────────

function ensure3d() {
  if (!state.mapOk || !map || !map.getStyle()) return false;
  if (!map.getSource(L.prisms)) {
    map.addSource(L.prisms, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.prisms, type: "fill-extrusion", source: L.prisms, layout: { visibility: vis(view3d) },
      paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-base": ["get", "base"],
        "fill-extrusion-height": ["get", "height"], "fill-extrusion-opacity": 0.92,
        "fill-extrusion-vertical-gradient": true } });
    map.on("click", L.prisms, onPrismClick);
    map.on("mouseenter", L.prisms, () => { if (!drawing) map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", L.prisms, () => { if (!drawing) map.getCanvas().style.cursor = ""; });
    clickLayers.add(L.prisms);
  }
  return true;
}

function paint3d() {
  if (!current || !haveGeo() || !ensure3d()) return;
  const m = { rings: current.rings, offsets: current.offsets, heads: current.heads, shape: current.shape, layers: current.meta.layers };
  const fc = prismFeatures(views.geo, m, current.t, { exaggeration: ve, shown, range: current.meta.range || [0, 1], wet,
    gap: exploded ? explodeGap() : 0 });
  map.getSource(L.prisms).setData(fc);
  map.setLayoutProperty(L.prisms, "visibility", vis(view3d));
  // the plan view under the stack would show through its gaps: hide it while the stack stands
  if (map.getLayer(ids.fill)) map.setPaintProperty(ids.fill, "fill-opacity", view3d ? 0 : 0.86);
  if (map.getLayer(ids.line)) map.setLayoutProperty(ids.line, "visibility", vis(!view3d && Boolean(current)));
}

let onCell = null;
/** groundwater-view.js hands over how a cell's card opens, for clicks on the 3D stack. */
export function setCellOpener(fn) { onCell = fn; }

function onPrismClick(e) {
  if (drawing || !view3d || !onCell) return;
  const f = (e.features || [])[0];
  if (!f) return;
  onCell(Number(f.properties.c), e.lngLat, Number(f.properties.l));
}

// How far apart the layers stand when pulled apart: a fifth of the drawn stack, in metres.
function explodeGap() {
  const g = views.geo;
  return 0.2 * (g.crest - g.base) * (g.metres_per_unit || 1) * ve;
}

function modelBounds() {
  const r = current.rings;
  let w = Infinity, s2 = Infinity, e = -Infinity, n = -Infinity;
  for (let k = 0; k < r.length; k += 2) { w = Math.min(w, r[k]); e = Math.max(e, r[k]); s2 = Math.min(s2, r[k + 1]); n = Math.max(n, r[k + 1]); }
  return [[w, s2], [e, n]];
}

// The map you can see: the left edge past the legend (when it sits on the left), for the camera and the dock.
function visibleLeft() {
  const wrap = document.querySelector(".map-wrap");
  const lg = document.getElementById("map-legend");
  if (!wrap || !lg) return 12;
  const a = wrap.getBoundingClientRect(), r = lg.getBoundingClientRect();
  return r.width && r.right < a.left + a.width / 2 ? Math.round(r.right - a.left + 12) : 12;
}

function set3d(on) {
  view3d = Boolean(on) && haveGeo();
  if (!state.mapOk || !map) return;
  if (view3d) {
    saved2d = { pitch: map.getPitch(), bearing: map.getBearing() };
    paint3d();
    const bearing = -28;
    const cam = map.cameraForBounds(modelBounds(), { bearing, padding: { top: 90, bottom: dockOpen() ? Math.round(dock.getBoundingClientRect().height) + 60 : 170, left: visibleLeft() + 40, right: 60 } });
    if (cam) map.easeTo({ center: cam.center, zoom: cam.zoom - 0.35, bearing, pitch: 58, duration: 1000 });
    else map.easeTo({ pitch: 58, bearing, duration: 900 });
  } else {
    if (map.getLayer(L.prisms)) map.setLayoutProperty(L.prisms, "visibility", "none");
    if (map.getLayer(ids.fill)) map.setPaintProperty(ids.fill, "fill-opacity", 0.86);
    if (map.getLayer(ids.line)) map.setLayoutProperty(ids.line, "visibility", vis(Boolean(current)));
    if (saved2d) map.easeTo({ pitch: saved2d.pitch, bearing: saved2d.bearing, duration: 700 });
    saved2d = null;
  }
  if (onView) onView(view3d);
  refreshLegend("modflow");
}

// ── the cross-section: drawing the line ─────────────────────────────────────

function ensureLineLayers() {
  if (!state.mapOk || !map || !map.getStyle()) return false;
  if (!map.getSource(L.line)) {
    map.addSource(L.line, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.line, type: "line", source: L.line, layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#0b2440", "line-width": 2.2, "line-opacity": 0.9, "line-dasharray": [2, 1.4] } });
    map.addLayer({ id: L.pts, type: "circle", source: L.line, filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 3.6, "circle-color": "#fff", "circle-stroke-color": "#0b2440", "circle-stroke-width": 1.6 } });
  }
  return true;
}

function lineData(pts, hover = null) {
  const feats = pts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
  const path = hover ? [...pts, hover] : pts;
  if (path.length > 1) feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: path } });
  return { type: "FeatureCollection", features: feats };
}

function endMarker(text, lngLat) {
  const el = document.createElement("div");
  el.className = "gwl-mark";
  el.textContent = text;
  return new globalThis.maplibregl.Marker({ element: el, anchor: "bottom", offset: [0, -6] }).setLngLat(lngLat).addTo(map);
}

function drawLine(pts) {
  if (!ensureLineLayers()) return;
  map.getSource(L.line).setData(lineData(pts));
  for (const m of markers) m.remove();
  markers = [];
  if (pts.length > 1 && globalThis.maplibregl) markers = [endMarker("A", pts[0]), endMarker("A′", pts[pts.length - 1])];
}

function clearLine() {
  if (state.mapOk && map && map.getSource(L.line)) map.getSource(L.line).setData(EMPTY);
  for (const m of markers) m.remove();
  markers = [];
}

function startDrawing() {
  if (!state.mapOk || !map || !haveGeo()) return;
  stopDrawing();
  if (view3d) set3d(false);   // a line is drawn on the plan
  ensureLineLayers();
  if (!map.getSource(L.catcher)) {
    map.addSource(L.catcher, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.catcher, type: "fill", source: L.catcher, paint: { "fill-color": "#000", "fill-opacity": 0.001 } });
  }
  map.getSource(L.catcher).setData({ type: "FeatureCollection", features: [WORLD] });
  clickLayers.add(L.catcher);
  drawing = { pts: [], unclaim: claimClicks(() => Boolean(drawing)) };
  map.doubleClickZoom.disable();
  map.getCanvas().style.cursor = "crosshair";
  map.on("click", onDrawClick);
  map.on("mousemove", onDrawMove);
  map.on("dblclick", onDrawDouble);
  document.addEventListener("keydown", onDrawKey);
  clearLine();
  dockTab = "section";
  renderDock();
  clearOfDock();
}

function stopDrawing() {
  if (!drawing) return;
  if (drawing.unclaim) drawing.unclaim();
  drawing = null;
  if (state.mapOk && map) {
    map.off("click", onDrawClick);
    map.off("mousemove", onDrawMove);
    map.off("dblclick", onDrawDouble);
    if (map.getSource(L.catcher)) map.getSource(L.catcher).setData(EMPTY);
    clickLayers.delete(L.catcher);
    map.getCanvas().style.cursor = "";
    setTimeout(() => map.doubleClickZoom.enable(), 300);
  }
  document.removeEventListener("keydown", onDrawKey);
}

function onDrawClick(e) {
  if (!drawing) return;
  const p = [Number(e.lngLat.lng.toFixed(6)), Number(e.lngLat.lat.toFixed(6))];
  const last = drawing.pts[drawing.pts.length - 1];
  if (last) {
    const a = map.project(last), b = e.point;
    if (Math.hypot(a.x - b.x, a.y - b.y) < 4) return;   // the second click of a double click
  }
  drawing.pts.push(p);
  map.getSource(L.line).setData(lineData(drawing.pts));
  renderDock();
}

function onDrawMove(e) {
  if (!drawing || !drawing.pts.length) return;
  map.getSource(L.line).setData(lineData(drawing.pts, [e.lngLat.lng, e.lngLat.lat]));
}

function onDrawDouble(e) {
  if (!drawing) return;
  e.preventDefault();
  finishDrawing();
}

function onDrawKey(e) {
  if (!drawing) return;
  if (e.key === "Escape") { stopDrawing(); clearLine(); if (section) drawLine(section.line); renderDock(); }
  else if (e.key === "Enter") finishDrawing();
}

function finishDrawing() {
  const pts = drawing ? drawing.pts.slice() : [];
  stopDrawing();
  if (pts.length < 2) { clearLine(); renderDock(); return; }
  void cut(pts);
}

async function cut(pts) {
  const was = dockOpen();
  sectionBusy = true;
  sectionError = null;
  drawLine(pts);
  renderDock();
  if (!was) clearOfDock();
  const my = token;
  try {
    const out = await call("gw_layers", { op: "section", line: pts });
    if (my !== token) return;
    if (out.error) { sectionError = out.error; section = null; return; }
    section = { ...out, line: pts };
    section.whole = sectionRange(section);
    section.close = headRange(section);
    section.yRange = closeLook ? section.close : section.whole;
  } catch (err) {
    sectionError = err && err.message ? err.message : String(err);
  } finally {
    sectionBusy = false;
    if (my === token) { dockTab = "section"; renderDock(); refreshLegend("modflow"); }
  }
}

// ── the dock: the section and the budget ───────────────────────────────────

function ensureDock() {
  if (dock && document.contains(dock)) return dock;
  const host = document.querySelector(".map-wrap") || document.body;
  dock = document.createElement("section");
  dock.className = "gwl-dock";
  dock.setAttribute("aria-label", "Model views");
  dock.hidden = true;
  host.appendChild(dock);
  dock.addEventListener("click", (e) => {
    const b = e.target.closest("[data-gwl]");
    if (!b) return;
    const act = b.dataset.gwl;
    if (act === "close") { closeDock(); return; }
    if (act === "tab-section") { dockTab = "section"; renderDock(); }
    else if (act === "tab-budget") { dockTab = "budget"; renderDock(); }
    else if (act === "draw") startDrawing();
    else if (act === "cancel") { stopDrawing(); clearLine(); if (section) drawLine(section.line); renderDock(); }
    else if (act === "done") finishDrawing();
    else if (act === "whole" || act === "near") {
      closeLook = act === "near";
      if (section) section.yRange = closeLook ? section.close : section.whole;
      renderDock();
    }
  });
  dock.addEventListener("mousemove", onDockHover);
  dock.addEventListener("mouseleave", () => hoverOff());
  dock.addEventListener("click", onBudgetPick);
  return dock;
}

function closeDock() {
  if (dock) dock.hidden = true;
  if (!drawing && !section) clearLine();
}

function openDock(tab) {
  const was = dockOpen();
  dockTab = tab;
  ensureDock().hidden = false;
  renderDock();
  if (!was) clearOfDock();
}

function dockOpen() { return Boolean(dock && !dock.hidden); }

function headStep() { return current ? current.t : 0; }

function budgetIndex() {
  if (!haveBudget() || !current) return -1;
  const k = budgetStep(views.budget, current.meta.kstpkper && current.meta.kstpkper[current.t]);
  return k >= 0 ? k : Math.min(current.t, views.budget.kstpkper.length - 1);
}

function tabsHtml() {
  const tab = (id, label, on) => `<button type="button" class="gwl-tab${on ? " on" : ""}" data-gwl="tab-${id}" aria-pressed="${on}">${label}</button>`;
  return `<div class="gwl-tabs" role="group" aria-label="View">${haveGeo() ? tab("section", "Cross-section", dockTab === "section") : ""}` +
    `${haveBudget() ? tab("budget", "Water budget", dockTab === "budget") : ""}</div>`;
}

function sectionBody() {
  const unit = lengthUnit(current.meta);
  if (drawing) {
    const n = drawing.pts.length;
    return `<p class="gwl-hint"><strong>Draw the section.</strong> Click points across the model, from A to A′; ` +
      `double-click or press Enter to finish, Esc to cancel. ${n ? `${n} point${n === 1 ? "" : "s"}.` : ""}</p>` +
      `<p class="gwl-acts">${n >= 2 ? '<button type="button" class="btn primary" data-gwl="done">Cut the section</button> ' : ""}` +
      '<button type="button" class="btn" data-gwl="cancel">Cancel</button></p>';
  }
  if (sectionBusy) return '<p class="gwl-hint muted">Cutting through the layers…</p>';
  if (sectionError) return `<p class="gwl-err">${escapeHtml(sectionError)}</p><p class="gwl-acts"><button type="button" class="btn" data-gwl="draw">Draw again</button></p>`;
  if (!section) {
    return '<p class="gwl-hint">Draw a line across the model to see every layer under it: tops and bottoms, heads ' +
      'and the water table, through time.</p><p class="gwl-acts"><button type="button" class="btn primary" data-gwl="draw">Draw a section</button></p>';
  }
  const width = Math.max(420, Math.min(900, Math.round((dock.clientWidth || 760) - 24)));
  const d = sectionSvg(section, headStep(), { width, height: 214, unit, title: "Cross-section A to A′" });
  dock.__sec = d;
  dock.__secWidth = width;
  return `<div class="gwl-plot" data-plot="section">${d.svg}<div class="gwl-read" hidden></div></div>` +
    `<p class="gwl-foot"><span>${escapeHtml(distWords(section.length_m))} from A to A′, ${section.segments.length} cells` +
    ` · Vertical exaggeration ${escapeHtml(veWords(d.ve))}</span>` +
    `<span class="gwl-keys"><span class="gwl-key"><i class="gwl-k-wt"></i>Water table</span>` +
    `<span class="gwl-key"><i class="gwl-k-sat"></i>Saturated</span>` +
    (d.noHead ? '<span class="gwl-key"><i class="gwl-k-none"></i>No head (inactive or dry)</span>' : "") +
    section.layers.slice(1).map((lay, i) => `<span class="gwl-key"><i class="gwl-k-dash" style="border-color:${layerLine(i + 1)}"></i>Layer ${lay} head</span>`).join("") +
    `</span><span class="gw-segs gwl-depth" role="group" aria-label="Vertical range">` +
    `<button type="button" class="gw-seg${closeLook ? "" : " on"}" data-gwl="whole" aria-pressed="${!closeLook}">Whole depth</button>` +
    `<button type="button" class="gw-seg${closeLook ? " on" : ""}" data-gwl="near" aria-pressed="${closeLook}">Around the heads</button></span>` +
    `<button type="button" class="gw-link" data-gwl="draw">Draw another</button></p>` +
    `<p class="gwl-note muted">${escapeHtml(section.water_table_rule)} ${escapeHtml((section.notes || []).join(" "))}</p>`;
}

function budgetBody() {
  const b = views.budget;
  const k = budgetIndex();
  const width = Math.max(380, Math.min(640, Math.round((dock.clientWidth || 760) - 300)));
  const d = budgetSvg(b, k, { width, height: 196, title: "Water budget over time" });
  dock.__bud = d;
  dock.__budWidth = width;
  const r = budgetRows(b, Math.max(0, k));
  const rows = r.rows.map((x) => `<tr><th><i style="background:${x.color}"></i>${escapeHtml(x.label)}</th>` +
    `<td>${x.in ? escapeHtml(rateWords(x.in)) : ""}</td><td>${x.out ? escapeHtml(rateWords(x.out)) : ""}</td></tr>`).join("");
  const disc = r.discrepancy === null || r.discrepancy === undefined ? "" : `${Number(r.discrepancy).toFixed(2)}%`;
  return `<div class="gwl-budget"><div class="gwl-plot" data-plot="budget">${d.svg}<div class="gwl-read" hidden></div>` +
    `<p class="gwl-foot"><span class="gwl-keys">${d.key}</span></p></div>` +
    `<div class="gwl-table"><p class="gwl-when">${escapeHtml(r.when)}</p><table><thead><tr><th></th><th>In</th><th>Out</th></tr></thead>` +
    `<tbody>${rows}</tbody><tfoot><tr><th>Total</th><td>${escapeHtml(rateWords(r.total_in))}</td><td>${escapeHtml(rateWords(r.total_out))}</td></tr></tfoot></table>` +
    `<p class="gwl-note muted">${escapeHtml(b.unit)}. Discrepancy ${escapeHtml(disc)}. ${escapeHtml((b.notes || []).join(" "))}</p></div></div>` +
    `<p class="gwl-note muted">${escapeHtml(b.method)}${b.internal && b.internal.length ? ` Flows between cells (${escapeHtml(b.internal.join(", ").toLowerCase())}) are not budget terms and are left out.` : ""}</p>`;
}

function placeDock() {
  if (!dock || !globalThis.matchMedia || matchMedia("(max-width: 900px)").matches) {
    if (dock) { dock.style.left = ""; dock.style.width = ""; }
    return;
  }
  const wrap = dock.parentElement.getBoundingClientRect();
  const left = visibleLeft();
  const avail = wrap.width - left - 16;
  if (avail < 480) { dock.style.left = ""; dock.style.width = ""; return; }
  dock.style.left = `${left}px`;
  dock.style.width = `${Math.min(900, avail)}px`;
}

// The time step as the dock says it: "2 Mar 2025", or the model time when the run has no dates.
function stepWords() {
  const d = current.meta.dates && current.meta.dates[current.t];
  if (d) return dateWords(d, { day: true });
  return `t = ${Number(current.meta.times[current.t]).toPrecision(4).replace(/\.?0+$/, "")} ${current.meta.time_units || "days"}`;
}

// Keep the model in the part of the map the dock leaves free (in plan view; a 3D camera is the reader's).
function clearOfDock() {
  if (!dockOpen() || view3d || !state.mapOk || !map || !current) return;
  // Still flying to the model (a budget dropped with the heads opens the dock as the map zooms in): wait for it,
  // or this would stop the flight wherever it had got to, often still at the globe.
  if (map.isMoving()) { map.once("moveend", () => clearOfDock()); return; }
  const h = dock.getBoundingClientRect().height;
  // On a phone the dock fills most of the map: leave the model at least 160 px above it rather than zooming out
  // until it is a speck.
  const free = map.getContainer().clientHeight - 70 - 160;
  const cam = map.cameraForBounds(modelBounds(), { bearing: map.getBearing(),
    padding: { top: 70, bottom: Math.max(40, Math.min(Math.round(h) + 150, free)), left: visibleLeft() + 24, right: 40 } });
  if (cam) map.easeTo({ center: cam.center, zoom: Math.min(cam.zoom, map.getZoom()), duration: 700 });
}

function renderDock() {
  if (!dock || dock.hidden) {
    if (!(drawing || sectionBusy)) return;
    ensureDock().hidden = false;
  }
  if (!current) { dock.hidden = true; return; }
  placeDock();
  if (dockTab === "budget" && !haveBudget()) dockTab = "section";
  const when = dockTab === "budget" && haveBudget() ? "" : escapeHtml(stepWords());
  const title = dockTab === "budget" ? "Water budget" : section ? "Cross-section A–A′" : "Cross-section";
  dock.innerHTML = `<header class="gwl-head"><h2>${title}</h2>${when ? `<span class="gwl-time">${when}</span>` : ""}` +
    `${tabsHtml()}<button type="button" class="gwl-x" data-gwl="close" aria-label="Close">×</button></header>` +
    `<div class="gwl-body">${dockTab === "budget" ? budgetBody() : sectionBody()}</div>`;
}

// The hover: a line through the plot, and what is under it (the section's cell and layers, the budget's time).
function onDockHover(e) {
  const plot = e.target.closest && e.target.closest(".gwl-plot");
  if (!plot) { hoverOff(); return; }
  const svg = plot.querySelector("svg");
  if (!svg) return;
  const box = svg.getBoundingClientRect();
  const kind = plot.dataset.plot;
  const d = kind === "section" ? dock.__sec : dock.__bud;
  const w = kind === "section" ? dock.__secWidth : dock.__budWidth;
  if (!d || !w) return;
  const x = ((e.clientX - box.left) / box.width) * w;
  const read = plot.querySelector(".gwl-read");
  const cursor = svg.querySelector(".gwl-cursor");
  if (x < d.x0 || x > d.x1) { hoverOff(); return; }
  const unit = lengthUnit(current.meta);
  const u = unit ? ` ${unit}` : "";
  let html = "";
  if (kind === "section" && section) {
    const m = ((x - d.x0) / (d.x1 - d.x0)) * section.length_m;
    const at = sectionAt(section, headStep(), m);
    if (cursor) { cursor.setAttribute("x1", x); cursor.setAttribute("x2", x); cursor.setAttribute("visibility", "visible"); }
    hoverOnLine(m);
    if (!at) { html = `<strong>${escapeHtml(distWords(m))}</strong> outside the model`; }
    else {
      const f = (v) => (v === null ? "–" : v.toFixed(2));
      html = `<strong>${escapeHtml(distWords(m))}</strong>${at.row ? ` · row ${at.row}, column ${at.col}` : ` · cell ${at.cell + 1}`}` +
        (at.water_table !== null ? `<br>Water table ${f(at.water_table)}${u}` : "") +
        at.layers.map((l) => `<br><span style="color:${layerLine(l.layer - 1)}">Layer ${l.layer}</span> ${f(l.top)} to ${f(l.bottom)}${u}, head ${f(l.head)}`).join("");
    }
  } else if (kind === "budget" && haveBudget()) {
    const k = Math.min(d.n - 1, Math.max(0, Math.floor(((x - d.x0) / (d.x1 - d.x0)) * d.n)));
    const r = budgetRows(views.budget, k);
    html = `<strong>${escapeHtml(r.when)}</strong> · in ${escapeHtml(rateWords(r.total_in))}, out ${escapeHtml(rateWords(r.total_out))}` +
      (current.meta.dates ? "<br><span class=\"muted\">Click to go to this time</span>" : "");
  }
  read.innerHTML = html;
  read.hidden = !html;
  const px = e.clientX - plot.getBoundingClientRect().left;
  read.style.left = `${Math.min(px + 12, plot.clientWidth - 230)}px`;
}

function hoverOff() {
  if (!dock) return;
  for (const r of dock.querySelectorAll(".gwl-read")) r.hidden = true;
  const c = dock.querySelector(".gwl-cursor");
  if (c) c.setAttribute("visibility", "hidden");
  hoverOnLine(null);
}

let hoverMarker = null;
// Where the hover is along the drawn line, on the map: interpolated between the line's points (for the eye only).
function hoverOnLine(m) {
  if (!state.mapOk || !globalThis.maplibregl) return;
  if (m === null || !section) { if (hoverMarker) { hoverMarker.remove(); hoverMarker = null; } return; }
  const pts = section.line;
  const k = Math.cos((pts[0][1] * Math.PI) / 180);
  const seg = pts.slice(1).map((p, i) => Math.hypot((p[0] - pts[i][0]) * k, p[1] - pts[i][1]));
  const total = seg.reduce((a, b) => a + b, 0);
  let want = (m / section.length_m) * total, i = 0;
  while (i < seg.length - 1 && want > seg[i]) { want -= seg[i]; i++; }
  const f = seg[i] > 0 ? want / seg[i] : 0;
  const at = [pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])];
  if (!hoverMarker) {
    const el = document.createElement("div");
    el.className = "gwl-dot";
    hoverMarker = new globalThis.maplibregl.Marker({ element: el }).setLngLat(at).addTo(map);
  } else hoverMarker.setLngLat(at);
}

function onBudgetPick(e) {
  const plot = e.target.closest && e.target.closest('.gwl-plot[data-plot="budget"]');
  if (!plot || !haveBudget() || !current || !current.meta.dates) return;
  const d = dock.__bud, w = dock.__budWidth;
  const box = plot.querySelector("svg").getBoundingClientRect();
  const x = ((e.clientX - box.left) / box.width) * w;
  if (x < d.x0 || x > d.x1) return;
  const k = Math.min(d.n - 1, Math.max(0, Math.floor(((x - d.x0) / (d.x1 - d.x0)) * d.n)));
  const kk = views.budget.kstpkper[k];
  const t = (current.meta.kstpkper || []).findIndex((x2) => x2[0] === kk[0] && x2[1] === kk[1]);
  if (t >= 0) setTime({ date: String(current.meta.dates[t]).slice(0, 10) }, { source: "chart" });
}

// ── a cell's heads over time, on its card ──────────────────────────────────

/** Fill a model cell's card with its hydrograph: every layer's head over the run. */
export async function layersCellChart(cardId, c) {
  if (!current) return;
  const my = token;
  try {
    const h = await call("gw_layers", { op: "hydrograph", cell: c });
    if (my !== token || h.error) return;
    const unit = lengthUnit(current.meta);
    const chart = hydrographSvg(h, current.t, { unit: unit || "head" });
    if (!chart) return;
    updateCard(cardId, {
      chart: { svg: chart.svg, key: chart.key, alt: h.summary, label: "Head over time, every layer" },
      note: hydrographWords(h, current.t, unit).join(". ") + ".",
    });
  } catch (err) {
    console.warn("hydrograph:", err);
  }
}

// ── the legend's part ───────────────────────────────────────────────────────

/** The model row's views: 2D or 3D, the exaggeration and the layers in 3D, the section and the budget. */
export function layersLegendHtml() {
  if (!current || !views) return "";
  const seg = (act, label, on, title = "") => `<button type="button" class="gw-seg${on ? " on" : ""}" data-act="${act}" aria-pressed="${on}"${title ? ` title="${escapeHtml(title)}"` : ""}>${label}</button>`;
  let html = '<div class="gwl-ctl">';
  if (haveGeo()) {
    html += `<div class="gw-segs" role="group" aria-label="Plan or 3D">${seg("gwl-2d", "Plan", !view3d)}${seg("gwl-3d", "3D layers", view3d)}</div>`;
  }
  const btn = (act, label, on) => `<button type="button" class="gwl-btn${on ? " on" : ""}" data-act="${act}">${label}</button>`;
  html += `${haveGeo() ? btn("gwl-section", "Cross-section", dockOpen() && dockTab === "section") : ""}` +
    `${haveBudget() ? btn("gwl-budget", "Water budget", dockOpen() && dockTab === "budget") : ""}</div>`;
  if (views.busy) html += '<p class="gw-line muted">Reading the layers…</p>';
  if (view3d && haveGeo()) {
    const g = views.geo;
    html += `<div class="gwl-3dctl"><span>Vertical exaggeration</span><span class="gw-segs" role="group" aria-label="Vertical exaggeration">` +
      exaggerationChoices(g.exaggeration).map((v) => seg(`gwl-ve-${v}`, `x${v}`, v === ve)).join("") + "</span></div>";
    if (g.layers.length > 1) {
      html += `<div class="gwl-3dctl"><span>Layers</span><span class="gw-segs" role="group" aria-label="Layers shown">` +
        g.layers.map((lay, i) => seg(`gwl-layer-${i}`, String(lay), shown.has(i), `Show layer ${lay}`)).join("") + "</span></div>";
    }
    html += `<div class="gwl-3dctl"><span>Spacing</span><span class="gw-segs" role="group" aria-label="Layer spacing">` +
      `${seg("gwl-stack", "Stacked", !exploded)}${seg("gwl-explode", "Pulled apart", exploded)}</span></div>`;
    const u = lengthUnit(current.meta);
    html += `<p class="gw-line muted">Layer tops and bottoms from the grid file, ${ve === 1 ? "at true scale" : `${ve} times taller than they are`}` +
      `${exploded ? ", pulled apart to show each top" : ""}. Base ${escapeHtml(g.base.toFixed(1))}${u ? ` ${u}` : ""}. Grey: dry now.</p>`;
  }
  if (views.layersError && !views.busy) html += `<p class="gw-line muted">No 3D or section: ${escapeHtml(views.layersError)}</p>`;
  if (views.budgetError) html += `<p class="gw-err">The budget file did not read: ${escapeHtml(views.budgetError)}</p>`;
  if (!haveBudget() && !views.budgetError && !views.busy) {
    html += '<p class="gw-line muted">Drop the cell-by-cell budget file (.cbc) with the others for the water budget.</p>';
  }
  return html;
}

/** A legend button of ours: handled here (true), or not ours (false). */
export function layersAct(name) {
  if (!name.startsWith("gwl-")) return false;
  if (name === "gwl-3d") set3d(true);
  else if (name === "gwl-2d") set3d(false);
  else if (name === "gwl-section") { if (dockOpen() && dockTab === "section") closeDock(); else { openDock("section"); if (!section) startDrawing(); } }
  else if (name === "gwl-budget") { if (dockOpen() && dockTab === "budget") closeDock(); else openDock("budget"); }
  else if (name.startsWith("gwl-ve-")) { ve = Number(name.slice(7)) || 1; paint3d(); }
  else if (name === "gwl-explode" || name === "gwl-stack") { exploded = name === "gwl-explode"; paint3d(); }
  else if (name.startsWith("gwl-layer-")) {
    const i = Number(name.slice(10));
    if (shown.has(i) && shown.size > 1) shown.delete(i); else shown.add(i);
    paint3d();
  }
  refreshLegend("modflow");
  return true;
}

/** Ids of groundwater-view.js's plan layers (hidden under the 3D stack). */
export function initGwLayers({ fill, line, viewChanged = null } = {}) {
  ids = { fill: fill || ids.fill, line: line || ids.line };
  onView = viewChanged;
  window.addEventListener("resize", () => { if (dockOpen()) renderDock(); });
  // Debug hooks (harmless, like the rest of window.__aq): what the views hold, and a section cut from code.
  dbg.gwLayers = {
    state: () => ({ busy: views ? views.busy : null, geo: haveGeo() ? { ...views.geo, top: undefined, botm: undefined, tops: undefined } : null,
      budget: haveBudget(), layersError: views && views.layersError, budgetError: views && views.budgetError,
      view3d, ve, section: section ? { cells: section.segments.length, length_m: section.length_m } : null,
      sectionError, dock: dockOpen() ? dockTab : null }),
    cut: (pts) => cut(pts),
  };
}
