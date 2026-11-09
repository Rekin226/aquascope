// Your MODFLOW heads, read the way a hydrogeologist reads them (#623, #624): a smooth head surface with labelled
// contours and flow arrows, the drawdown cone, and where each well draws its water from (pathlines and capture
// zones). The numbers come from aquascope.groundwater.views in the worker, one step and layer at a time; this module
// paints them and keeps up with the time bar and the layer chosen. groundwater-view.js owns the model and calls in.
//
// The words, the colours and the features are in groundwater-views-core.js (node tests it).

import { clickLayers, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { openCard } from "./map-card.js?v=__BUILD__";
import { refreshLegend } from "./map-legend.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { frameOf, lengthUnit, modelOutline, timeLabel } from "./groundwater-core.js?v=__BUILD__";
import {
  EMPTY_FC, VIEW_MODES, changeSince, modeOf, paintSmooth, smoothScale, viewFeatures, viewLines, wellCardOf,
} from "./groundwater-views-core.js?v=__BUILD__";

const L = {
  smooth: "gwv-smooth", zones: "gwv-zones", zoneLine: "gwv-zone-line", contours: "gwv-contours",
  labels: "gwv-contour-labels", paths: "gwv-paths", pathArrows: "gwv-path-arrows", arrows: "gwv-arrows",
  wells: "gwv-wells", wellLabels: "gwv-well-labels",
};
const VECTORS = ["zones", "paths", "contours", "arrows", "wells"];
const ARROW = "gwv-arrow";
const CELLS = "gw-model";       // groundwater-view.js's cell fill: see-through while the smooth surface is on
const GRID = "gw-model-line";   // and its cell edges: only a hint, close in, over the smooth surface
const GRID_OPACITY = ["interpolate", ["linear"], ["zoom"], 10, 0, 13, 0.35];
const GRID_FAINT = ["interpolate", ["linear"], ["zoom"], 13, 0, 16, 0.18];

let cur = null;                  // { model, mode, ref, v, key, busy, queued, url, ranges }
let bound = false;

const vis = (on) => (on ? "visible" : "none");

function viewsFor(model) {
  if (!cur || cur.model !== model) {
    if (cur && cur.url) URL.revokeObjectURL(cur.url);
    cur = { model, mode: "head", ref: 0, v: null, key: "", busy: false, queued: false, url: null, ranges: new Map() };
  }
  return cur;
}

const structured = (model) => Boolean(model && model.meta && model.meta.grid && model.meta.grid.type === "dis");

// ── layers ──────────────────────────────────────────────────────────────────

function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^(river-|fd-)/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function arrowImage() {
  const s = 48;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d");
  g.fillStyle = "#000";
  g.beginPath();
  // a slim arrow pointing up (north), an SDF icon so its colour and halo come from the style
  g.moveTo(24, 6); g.lineTo(34, 21); g.lineTo(27, 19.5); g.lineTo(27, 42); g.lineTo(21, 42); g.lineTo(21, 19.5);
  g.lineTo(14, 21); g.closePath(); g.fill();
  return g.getImageData(0, 0, s, s);
}

// True when the layers are there; "fresh" when they were just (re)made, after a basemap change dropped them.
function ensureLayers() {
  if (!state.mapOk || !map || !map.getStyle()) return false;
  const fresh = !map.getSource(L.contours);
  if (!map.hasImage(ARROW)) map.addImage(ARROW, arrowImage(), { sdf: true, pixelRatio: 2 });
  const before = beforeId();
  const add = (id, src, spec, under = before) => {
    if (!map.getSource(src)) map.addSource(src, { type: "geojson", data: EMPTY_FC });
    if (!map.getLayer(id)) map.addLayer({ id, source: src, ...spec }, under);
  };
  add(L.zones, L.zones, { type: "fill", paint: { "fill-color": ["get", "color"], "fill-opacity": 0.16 } });
  add(L.zoneLine, L.zones, { type: "line", paint: { "line-color": ["get", "color"], "line-width": 1.8, "line-opacity": 0.9 } });
  add(L.paths, L.paths, { type: "line", layout: { "line-cap": "round" },
    paint: { "line-color": ["get", "color"], "line-width": 1.1, "line-opacity": 0.75 } });
  add(L.pathArrows, L.paths, { type: "symbol", layout: { "symbol-placement": "line", "symbol-spacing": 70, "icon-image": ARROW,
    "icon-size": 0.32, "icon-rotate": 90, "icon-rotation-alignment": "map", "icon-allow-overlap": true, "icon-ignore-placement": true },
  paint: { "icon-color": ["get", "color"], "icon-opacity": 0.85 } });
  add(L.contours, L.contours, { type: "line", layout: { "line-join": "round" }, paint: {
    "line-color": ["case", ["get", "rise"], "#01665e", "#0b2440"],
    "line-width": ["case", ["get", "index"], 1.5, 0.75],
    "line-opacity": ["case", ["get", "index"], 0.8, 0.55] } });
  add(L.labels, L.contours, { type: "symbol", layout: { "symbol-placement": "line", "symbol-spacing": 260,
    "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": ["case", ["get", "index"], 11.5, 10.5],
    "text-max-angle": 30, "text-keep-upright": true, "text-padding": 4 },
  paint: { "text-color": "#0b2440", "text-halo-color": "rgba(255,255,255,0.92)", "text-halo-width": 1.4 } });
  // Arrows that would touch drop out, so a small model seen from afar, or on a phone, keeps a calm, even scatter and
  // fills in as you zoom. They sit under the contour labels, which are placed first and keep their room.
  add(L.arrows, L.arrows, { type: "symbol", layout: { "icon-image": ARROW, "icon-rotate": ["get", "b"],
    "icon-rotation-alignment": "map", "icon-size": ["get", "s"], "icon-allow-overlap": false, "icon-ignore-placement": false,
    "icon-padding": 3 },
  paint: { "icon-color": "#14324f", "icon-halo-color": "rgba(255,255,255,0.9)", "icon-halo-width": 1.2, "icon-opacity": 0.9 } },
  map.getLayer(L.labels) ? L.labels : before);
  add(L.wells, L.wells, { type: "circle", paint: {
    "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 3.5, 14, 6.5],
    "circle-color": ["case", ["get", "pumping"], ["get", "color"], "#ffffff"],
    "circle-stroke-color": ["case", ["get", "here"], "#0b2440", "#8a96a3"],
    "circle-stroke-width": ["case", ["get", "pumping"], 1.6, 1.2] } });
  add(L.wellLabels, L.wells, { type: "symbol", layout: { "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"],
    "text-size": 11.5, "text-offset": [0.9, 0], "text-anchor": "left", "text-allow-overlap": true },
  paint: { "text-color": "#0b2440", "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 1.5 } });
  if (!bound) {
    bound = true;
    clickLayers.add(L.wells);
    map.on("click", L.wells, onWellClick);
    map.on("mouseenter", L.wells, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", L.wells, () => { map.getCanvas().style.cursor = ""; });
  }
  return fresh ? "fresh" : true;
}

function setVisible(on) {
  if (!state.mapOk || !map) return;
  for (const id of Object.values(L)) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis(on));
  cellsUnder(on && Boolean(map.getLayer(L.smooth)));
}

// The cells stay for the click and the card; under the smooth surface they are see-through and their edges faint.
function cellsUnder(smooth) {
  if (map.getLayer(CELLS)) map.setPaintProperty(CELLS, "fill-opacity", smooth ? 0 : 0.86);
  if (map.getLayer(GRID)) map.setPaintProperty(GRID, "line-opacity", smooth ? GRID_FAINT : GRID_OPACITY);
}

function clearAll() {
  if (!state.mapOk || !map) return;
  for (const k of VECTORS) if (map.getSource(L[k])) map.getSource(L[k]).setData(EMPTY_FC);
  if (map.getLayer(L.smooth)) map.removeLayer(L.smooth);
  if (map.getSource(L.smooth)) map.removeSource(L.smooth);
  cellsUnder(false);
}

// ── the smooth surface ──────────────────────────────────────────────────────

async function toUrl({ px, width, height }) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(px, width, height), 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return URL.createObjectURL(blob);
}

function surfaceValues(model) {
  const frame = frameOf(model.heads, model.shape, model.t, model.l);
  if (cur.mode !== "drawdown") return { values: frame, range: model.meta.range };
  const ref = frameOf(model.heads, model.shape, cur.ref, model.l);
  const range = cur.ranges.get(`${model.l}|${cur.ref}`) || null;
  return range ? { values: changeSince(frame, ref), range } : null;
}

async function paintSurface(model) {
  const g = model.meta.grid;
  const corners = modelOutline(g, model.rings);
  const sv = surfaceValues(model);
  if (!corners || !sv) return;
  const scale = smoothScale(g.drawn_nrow, g.drawn_ncol);
  const url = await toUrl(paintSmooth(sv.values, g.drawn_nrow, g.drawn_ncol, sv.range, cur.mode, scale));
  if (!cur || cur.model !== model) { URL.revokeObjectURL(url); return; }
  const coords = corners.slice(0, 4);
  const src = map.getSource(L.smooth);
  if (src && src.updateImage) src.updateImage({ url, coordinates: coords });
  else {
    map.addSource(L.smooth, { type: "image", url, coordinates: coords });
    map.addLayer({ id: L.smooth, type: "raster", source: L.smooth,
      paint: { "raster-opacity": 1, "raster-fade-duration": 0, "raster-resampling": "linear" } }, map.getLayer(CELLS) ? CELLS : beforeId());
  }
  if (cur.url) setTimeout(((old) => () => URL.revokeObjectURL(old))(cur.url), 1000);
  cur.url = url;
  cellsUnder(true);
}

// ── asking Python, one request in flight ────────────────────────────────────

async function fetchViews(model) {
  const key = `${model.t}|${model.l}|${cur.mode}|${cur.ref}`;
  if (key === cur.key) return;
  if (cur.busy) { cur.queued = true; return; }
  cur.busy = true;
  try {
    const v = await call("groundwater", { op: "views", t: model.t, layer: model.meta.layers[model.l], reference: cur.ref,
      show: modeOf(cur.mode).show });
    if (!cur || cur.model !== model) return;
    cur.v = v;
    cur.key = key;
    if (v.drawdown && v.drawdown.range) cur.ranges.set(`${model.l}|${v.drawdown.reference}`, v.drawdown.range);
    if (v.error) { clearAll(); refreshLegend("modflow"); return; }
    if (cur.mode === "drawdown") await paintSurface(model);
    const f = viewFeatures(v, cur.mode, lengthUnit(model.meta));
    for (const k of VECTORS) if (map.getSource(L[k])) map.getSource(L[k]).setData(f[k]);
    refreshLegend("modflow");
  } catch (err) {
    console.warn("groundwater views:", err);
    if (cur && cur.model === model) cur.v = { error: `Could not draw the views: ${err && err.message ? err.message : err}` };
    refreshLegend("modflow");
  } finally {
    if (cur && cur.model === model) {
      cur.busy = false;
      if (cur.queued) { cur.queued = false; void fetchViews(model); }
    }
  }
}

/** Draw the views for the model's step and layer (called by groundwater-view.js whenever it repaints the cells). */
export async function updateModelViews(model, { shown = true } = {}) {
  if (model && !structured(model) && cur) { clearModelViews(); return; }   // an unstructured grid: the cells only
  const ready = model && structured(model) && ensureLayers();
  if (!ready) return;
  viewsFor(model);
  if (ready === "fresh") cur.key = "";
  setVisible(shown);
  if (!shown) return;
  if (cur.mode !== "drawdown" || cur.ranges.has(`${model.l}|${cur.ref}`)) await paintSurface(model);
  await fetchViews(model);
}

/** Show or hide the views with the model's own row toggle. */
export function showModelViews(on) { if (cur && cur.model) setVisible(on); }

/** The model was removed: take the views off the map. */
export function clearModelViews() {
  clearAll();
  if (cur && cur.url) URL.revokeObjectURL(cur.url);
  cur = null;
}

// ── the legend ──────────────────────────────────────────────────────────────

function stepLabel(model) { return (t) => String(timeLabel(model.meta, t)).replace("T", " "); }

/** The ramp key the legend shows: the model's own for the head, the change ramp for the drawdown (else null, and
 * groundwater-view.js draws its head key). */
export function modelViewsKey(model, rampKey) {
  if (!cur || cur.model !== model || cur.mode !== "drawdown") return null;
  const r = cur.ranges.get(`${model.l}|${cur.ref}`);
  if (!r) return null;
  const unit = lengthUnit(model.meta);
  const m = Math.abs(r[0]).toFixed(Math.abs(r[0]) < 1 ? 2 : 1);
  return rampKey("change", { lo: `fell ${m}`, hi: `rose ${m}` }, unit);
}

/** The views' part of the model's legend body: three ways to look, and two or three short lines. */
export function modelViewsBody(model) {
  if (!structured(model)) return "";
  viewsFor(model);
  const seg = (m) => `<button type="button" class="gw-seg${cur.mode === m.id ? " on" : ""}" data-act="gwv-${m.id}" aria-pressed="${cur.mode === m.id}">${m.label}</button>`;
  let html = `<div class="gw-segs" role="group" aria-label="How to look at the heads">${VIEW_MODES.map(seg).join("")}</div>`;
  const lines = viewLines(cur.v, cur.mode, { unit: lengthUnit(model.meta), stepLabel: stepLabel(model) });
  for (const line of lines) html += `<p class="gw-line">${escapeHtml(line)}</p>`;
  if (cur.mode === "drawdown") {
    const d = (cur.v && cur.v.drawdown) || {};
    const btns = [];
    if (model.t !== cur.ref) btns.push(`<button type="button" class="gw-link" data-act="gwv-ref-here">Measure from this step</button>`);
    if (Number.isInteger(d.suggested_reference) && d.suggested_reference !== cur.ref) {
      btns.push(`<button type="button" class="gw-link" data-act="gwv-ref-best">from ${escapeHtml(stepLabel(model)(d.suggested_reference))}, when the wells' heads are highest</button>`);
    }
    if (cur.ref !== 0) btns.push(`<button type="button" class="gw-link" data-act="gwv-ref-first">from the first step</button>`);
    if (btns.length) html += `<p class="gw-line">${btns.join(" · ")}</p>`;
  }
  html += `<p class="gw-line muted"><button type="button" class="gw-link" data-act="gwv-about">How these are drawn</button></p>`;
  return html;
}

/** The legend's buttons for the views (data-act names starting gwv-); false when the name is not ours. */
export function modelViewsAct(name, model) {
  if (!String(name).startsWith("gwv-") || !model || !cur) return false;
  const what = name.slice(4);
  if (VIEW_MODES.some((m) => m.id === what)) {
    cur.mode = what;
    cur.v = null;
  } else if (what === "ref-here") cur.ref = model.t;
  else if (what === "ref-best" && cur.v && cur.v.drawdown) cur.ref = cur.v.drawdown.suggested_reference;
  else if (what === "ref-first") cur.ref = 0;
  else if (what === "about") { about(); return true; }
  cur.key = "";
  refreshLegend("modflow");
  void updateModelViews(model);
  return true;
}

function about() {
  const v = (cur && cur.v) || {};
  openModal("How the model views are drawn", `
    <p><strong>Contours and arrows.</strong> Head contours at one round interval for the whole run, heavier every fifth,
    drawn between cell centres. The arrows point the way groundwater moves, down the head gradient (Darcy's law), with
    the same conductivity taken in every direction, because a head file does not hold it.</p>
    <p><strong>Drawdown.</strong> The head at the reference step minus the head now: the first step unless you choose
    another. A model that starts already pumping shows its cone best measured from a step when the wells were idle.</p>
    <p><strong>Pathlines and capture zones.</strong> Particles follow the head gradient within the layer, the step's
    heads held still. Forward tracking ends at a pumping well; backward tracking from around each well traces where
    its water comes from, and the area those paths sweep is its capture zone.</p>
    <p><strong>Wells.</strong> ${escapeHtml(v.wells_how ? `${v.wells_how[0].toUpperCase()}${v.wells_how.slice(1)}.` : "From a .wel file dropped with the model, or found in the heads.")}</p>
    <p><strong>What this is not.</strong> ${escapeHtml(v.limits || "")}</p>
    <p class="muted">Computed in your browser by aquascope.groundwater.views; the same in Python:
    <code>aquascope modflow model.hds --grid model.dis --views --wel model.wel</code>.</p>`);
}

// ── clicks ──────────────────────────────────────────────────────────────────

function onWellClick(e) {
  if (!cur || !cur.v || !cur.v.wells) return;
  const f = (e.features || [])[0];
  const w = f && cur.v.wells.find((x) => x.id === f.properties.id);
  if (!w) return;
  const m = cur.model;
  const c = wellCardOf(cur.v, w, { unit: lengthUnit(m.meta), timeUnits: m.meta.time_units, when: timeLabel(m.meta, m.t) });
  openCard({
    id: `gwv-well:${w.id}:${m.t}:${m.l}`, lngLat: [w.lon, w.lat], lift: 6, what: c.what, title: c.title, sub: c.sub,
    status: c.status, figure: c.figure, note: c.note,
    credit: m.sample ? `${m.sample.credit} Read in your browser with aquascope.` : "Your files, read in your browser with aquascope.",
  });
}

/** For tests and the debug hook: what is drawn now. */
export const modelViewsState = () => (cur ? { mode: cur.mode, ref: cur.ref, v: cur.v } : null);
