// Groundwater on the map (#558): the ModelMuse view, keyless.
//
// 1. A head surface from the Archive's wells (the Environment Agency's, England), and its change between two
//    months. Off by default; "Groundwater" in the legend and the rail turns it on. The page reads the wells'
//    readings from the Archive's bundle with DuckDB (once, about 4 MB) and hands the months it needs to Python
//    (aquascope.groundwater.surface, in the worker), which picks the month, interpolates and applies the density
//    mask. Here the result is painted: the surface in a ramp, the "too few wells" cells hatched in grey, contour
//    lines on the head, the wells as small dots. The time bar picks the month.
// 2. Your MODFLOW heads. Drop a .hds (with its .dis, .disv or .dis.grb, and a .nam if FloPy wrote one) on the map,
//    or open them from the legend: the worker reads them with the same package (the files stay in your browser),
//    and the cells are drawn on the model grid, layer and time step selectable and played with the time bar when
//    the model has dates. A small form asks where the model sits when its files do not say.
//
// The colours, the words and the cards are in groundwater-core.js (node tests it).

import { $, actions, clickLayers, datedExtras, dbg, escapeHtml, onTime, setTime, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { openCard } from "./map-card.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { closeModal, openModal } from "./shell.js?v=__BUILD__";
import { syncTimeBar } from "./time-ui.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import {
  GW_EXTENT, GW_SHORT_CREDIT, GW_SINCE, GW_SOURCE, cellsGeoJSON, fmtM, frameOf, gridCellAt, gridCorners,
  headWord, isModflowDrop, keyLines, lengthUnit, maskScale, modelOutline, monthOf, paintMask, paintSurface, rampCss, readGwParam, readPlaceForm, readingsWindow,
  rgb, rowSummary, sortModelFiles, stepForDate, surfaceCard, timeLabel, valueColor, viewTouches, wellCard, wellsNear,
} from "./groundwater-core.js?v=__BUILD__";

const ARCHIVE = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/";
const BUNDLE = `${ARCHIVE}obs/groundwater_level/${GW_SOURCE}.parquet`;
const L = {
  surface: "gw-surface", mask: "gw-mask", contours: "gw-contours", wells: "gw-wells", hit: "gw-hit",
  model: "gw-model", modelLine: "gw-model-line", modelEdge: "gw-model-edge",
};
const EMPTY = { type: "FeatureCollection", features: [] };

let mode = "level";            // "level" (the head) or "change" (against 12 months earlier)
let res = null;                // the last surface from Python
let busy = 0;
let failed = null;
let drawnKey = "";
let runToken = 0;
let wells = null;              // the Archive's groundwater wells: [{ id, name, lon, lat }]
let tableReady = null;         // the bundle in DuckDB
const windows = new Map();     // month -> readings payload for Python
let urls = { surface: null, mask: null };

// Your MODFLOW heads
let model = null;              // { meta, rings, offsets, heads, shape, t, l, name, fc }
let modelOn = true;
let modelBusy = false;
let pending = null;            // dropped files waiting for the place form

const vis = (on) => (on ? "visible" : "none");
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

// ── where our layers go ─────────────────────────────────────────────────────

// Under the rivers, the catchments and the gauges, over the river status and Floods past, like the flood depth.
function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^(river-|fd-)/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureVectorLayers() {
  if (!state.mapOk || !map || !map.getStyle()) return false;
  const before = beforeId();
  if (!map.getSource(L.contours)) {
    map.addSource(L.contours, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.contours, type: "line", source: L.contours, layout: { visibility: vis(state.groundwater) },
      paint: { "line-color": "#0b2440", "line-opacity": 0.38, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 10, 1.1] } }, before);
  }
  if (!map.getSource(L.hit)) {
    map.addSource(L.hit, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.hit, type: "fill", source: L.hit, layout: { visibility: vis(state.groundwater) },
      paint: { "fill-color": "#000", "fill-opacity": 0 } }, before);
  }
  if (!map.getSource(L.wells)) {
    map.addSource(L.wells, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.wells, type: "circle", source: L.wells, layout: { visibility: vis(state.groundwater) },
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 1.8, 8, 3.2, 12, 4.5],
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#16202a", "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 10, 1],
        "circle-stroke-opacity": 0.75,
      } }, before);
  }
  if (!map.getSource(L.model)) {
    map.addSource(L.model, { type: "geojson", data: EMPTY, promoteId: "c" });
    map.addLayer({ id: L.model, type: "fill", source: L.model, layout: { visibility: vis(modelShown()) },
      paint: { "fill-color": ["coalesce", ["get", "color"], "rgba(0,0,0,0)"], "fill-opacity": 0.86, "fill-antialias": false } }, before);
    map.addLayer({ id: L.modelLine, type: "line", source: L.model, layout: { visibility: vis(modelShown()) },
      paint: { "line-color": "#0b2440", "line-width": 0.4, "line-opacity": ["interpolate", ["linear"], ["zoom"], 10, 0, 13, 0.35] } }, before);
    map.addSource(L.modelEdge, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.modelEdge, type: "line", source: L.modelEdge, layout: { visibility: vis(modelShown()) },
      paint: { "line-color": "#0b2440", "line-width": 1.2, "line-opacity": 0.7, "line-dasharray": [3, 2] } }, before);
  }
  return true;
}

function setImage(id, url, corners, resampling) {
  const src = map.getSource(id);
  if (src && src.updateImage) {
    src.updateImage({ url, coordinates: corners });
  } else {
    map.addSource(id, { type: "image", url, coordinates: corners });
    map.addLayer({ id, type: "raster", source: id, layout: { visibility: vis(state.groundwater) },
      paint: { "raster-opacity": 1, "raster-fade-duration": 0, "raster-resampling": resampling } }, map.getLayer(L.contours) ? L.contours : beforeId());
  }
}

async function toUrl(px, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(px, width, height), 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return URL.createObjectURL(blob);
}

// ── the surface ─────────────────────────────────────────────────────────────

function archiveWells() {
  if (wells && wells.length) return wells;
  const list = (state.stations || []).filter((r) => r.source === GW_SOURCE && (r.variables || []).includes("groundwater_level") &&
    Number.isFinite(r.lon) && Number.isFinite(r.lat));
  if (list.length) wells = list.map((r) => ({ id: r.station_id, name: r.name || null, lon: r.lon, lat: r.lat }));
  return wells || [];
}

function ensureTable() {
  if (tableReady) return tableReady;
  tableReady = (async () => {
    const { conn } = await duck();
    await conn.query(`CREATE TABLE IF NOT EXISTS gw_${GW_SOURCE} AS SELECT station_id, CAST(date AS VARCHAR) AS d, value
      FROM read_parquet('${BUNDLE}')`);
    return conn;
  })();
  tableReady.catch(() => { tableReady = null; });
  return tableReady;
}

// The readings a month needs, in the shape Python takes: well indexes into `wells`, dates and values.
async function readingsFor(month) {
  if (windows.has(month)) return windows.get(month);
  const conn = await ensureTable();
  const w = readingsWindow(month);
  const t = await conn.query(`SELECT station_id, d, value FROM gw_${GW_SOURCE} WHERE d >= '${w.from}' AND d < '${w.to}'`);
  const pos = new Map(archiveWells().map((x, i) => [x.id, i]));
  const out = { well: [], date: [], value: [] };
  for (const r of t.toArray()) {
    const i = pos.get(r.station_id);
    if (i === undefined || r.value === null) continue;
    out.well.push(i);
    out.date.push(String(r.d).slice(0, 10));
    out.value.push(Number(r.value));
  }
  windows.set(month, out);
  if (windows.size > 6) windows.delete(windows.keys().next().value);
  return out;
}

function viewBox() {
  const b = map.getBounds();
  const w = Math.max(-180, b.getWest()), e = Math.min(180, b.getEast());
  const s = Math.max(-85, b.getSouth()), n = Math.min(85, b.getNorth());
  return [w, s, e, n].map((v) => Math.round(v * 100) / 100);
}

async function refresh() {
  if (!state.groundwater || !state.mapOk || !map) { renderLegend(); return; }
  const view = viewBox();
  const month = monthOf(state.date || new Date().toISOString());
  if (!viewTouches(view)) {
    res = { kind: "groundwater_surface", grid: null, outside: true, mode, label: "" };
    drawnKey = "";
    clearSurface();
    renderLegend();
    return;
  }
  const key = `${month}|${mode}|${view.join(",")}`;
  if (key === drawnKey) return;
  const all = archiveWells();
  if (!all.length) { setTimeout(() => void refresh(), 1500); renderLegend(); return; }
  const my = ++runToken;
  busy++;
  renderLegend();
  try {
    const readings = await readingsFor(month);
    if (my !== runToken) return;
    const out = await call("groundwater", { op: "surface", wells: all, readings, month, mode, bbox: view });
    if (my !== runToken) return;
    res = out;
    failed = null;
    drawnKey = key;
    await draw();
    syncDated();
  } catch (err) {
    console.warn("groundwater surface:", err);
    if (my === runToken) failed = err && err.message ? err.message : "could not draw";
  } finally {
    busy--;
    renderLegend();
  }
}

function clearSurface() {
  for (const id of [L.surface, L.mask]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
  for (const id of [L.contours, L.wells, L.hit]) if (map.getSource(id)) map.getSource(id).setData(EMPTY);
}

async function draw() {
  if (!ensureVectorLayers()) return;
  const g = res && res.grid;
  if (!g) { clearSurface(); return; }
  const corners = gridCorners(g);
  const sUrl = await toUrl(paintSurface(g, res.range, res.mode), g.nx, g.ny);
  const m = paintMask(g, maskScale(g));
  const mUrl = await toUrl(m.px, m.width, m.height);
  setImage(L.surface, sUrl, corners, "linear");
  setImage(L.mask, mUrl, corners, "nearest");
  for (const k of ["surface", "mask"]) if (urls[k]) URL.revokeObjectURL(urls[k]);
  urls = { surface: sUrl, mask: mUrl };
  const lines = res.mode === "level" && res.contours ? res.contours.levels.map((lv) => ({
    type: "Feature", properties: { level: lv.level }, geometry: { type: "MultiLineString", coordinates: lv.segments },
  })) : [];
  map.getSource(L.contours).setData({ type: "FeatureCollection", features: lines });
  map.getSource(L.wells).setData({ type: "FeatureCollection", features: (res.wells || []).map((w) => ({
    type: "Feature", properties: { id: w.id, color: rgb(valueColor(w.value, res.range, res.mode)) },
    geometry: { type: "Point", coordinates: [w.lon, w.lat] },
  })) });
  const [w, s, e, n] = [g.west, g.south, g.east, g.north];
  map.getSource(L.hit).setData({ type: "FeatureCollection", features: [{ type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] });
}

// The time bar shows itself for a dated layer: the surface is one while it is on, month by month.
function syncDated() {
  const had = datedExtras.has("groundwater");
  if (state.groundwater) {
    datedExtras.set("groundwater", { id: "groundwater", label: "Groundwater", time: true, monthly: true, since: GW_SINCE });
  } else {
    datedExtras.delete("groundwater");
  }
  if (had !== datedExtras.has("groundwater")) syncTimeBar({ layersChanged: true });
}

function setLayerVisibility() {
  if (!state.mapOk || !map) return;
  for (const id of [L.surface, L.mask, L.contours, L.wells, L.hit]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis(state.groundwater));
  }
  for (const id of [L.model, L.modelLine, L.modelEdge]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis(modelShown()));
  }
}

export function setGroundwater(on, { write = true } = {}) {
  state.groundwater = Boolean(on);
  const toggle = $("toggle-groundwater");
  if (toggle) toggle.checked = state.groundwater;
  ensureVectorLayers();
  setLayerVisibility();
  syncDated();
  renderCredits();
  if (write) writeUrl();
  if (state.groundwater) void refresh(); else renderLegend();
}

function setMode(next) {
  if (next === mode) return;
  mode = next;
  drawnKey = "";
  void refresh();
}

// ── clicks: the map card ────────────────────────────────────────────────────

function onSurfaceClick(e) {
  if (!res || !res.grid) return;
  if (map.queryRenderedFeatures(e.point, { layers: ["points", "clusters", L.wells].filter((id) => map.getLayer(id)) }).length) return;
  const { lng, lat } = e.lngLat;
  const cell = gridCellAt(res.grid, lng, lat);
  if (!cell) return;
  const near = wellsNear(res.wells, lng, lat, res.radius_km);
  const c = surfaceCard(res, cell, near);
  openCard({
    id: `gw:${res.mode}:${res.label}:${lng.toFixed(3)},${lat.toFixed(3)}`, lngLat: [lng, lat], lift: 6,
    what: c.what, title: c.title, sub: c.sub, status: c.status, figure: c.figure, note: c.note,
    credit: GW_SHORT_CREDIT,
    buttons: [
      { id: "about", label: "About", title: "How the surface is made, and what it is not", onClick: about },
      ...(near[0] ? [{ id: "well", label: "Nearest well", title: "Open the nearest well's record", onClick: () => openWell(near[0]) }] : []),
    ],
  });
}

function openWell(w) {
  const r = state.byKey && state.byKey.get(`${GW_SOURCE}/${w.id}`);
  if (r) actions.selectStation(r);
}

function onWellClick(e) {
  if (map.queryRenderedFeatures(e.point, { layers: ["points", "clusters"].filter((id) => map.getLayer(id)) }).length) return;
  const f = (e.features || [])[0];
  const w = f && (res.wells || []).find((x) => x.id === f.properties.id);
  if (!w) return;
  const c = wellCard(res, w);
  openCard({
    id: `gw-well:${w.id}:${res.label}`, lngLat: [w.lon, w.lat], lift: 6, what: c.what, title: c.title, sub: c.sub,
    status: c.status, figure: c.figure, credit: GW_SHORT_CREDIT, details: () => openWell(w),
  });
}

function onModelClick(e) {
  if (!model || !modelShown()) return;
  if (map.queryRenderedFeatures(e.point, { layers: ["points", "clusters"].filter((id) => map.getLayer(id)) }).length) return;
  const f = (e.features || [])[0];
  if (!f) return;
  const c = Number(f.properties.c);
  const h = frameOf(model.heads, model.shape, model.t, model.l)[c];
  const lay = model.meta.layers[model.l];
  const g = model.meta.grid;
  const where = g.type === "dis" && !(g.block > 1)
    ? `row ${Math.floor(c / g.drawn_ncol) + 1}, column ${(c % g.drawn_ncol) + 1}` : `cell ${c + 1}`;
  const { lng, lat } = e.lngLat;
  const unit = lengthUnit(model.meta), word = headWord(model.meta);
  openCard({
    id: `gw-model:${c}:${model.t}:${model.l}`, lngLat: [lng, lat], lift: 4,
    what: "Your MODFLOW heads", title: Number.isFinite(h) ? `${word} ${fmtM(h)}${unit ? ` ${unit}` : ""}` : `No ${word.toLowerCase()} (inactive or dry)`,
    sub: `${where}, layer ${lay} · ${timeLabel(model.meta, model.t)}`,
    status: { text: `${model.name}: ${model.meta.text.toLowerCase()} from your file, in the model's length unit.`, color: null },
    figure: Number.isFinite(h) ? { value: fmtM(h), unit, label: word.toLowerCase() } : null,
    note: g.block > 1 ? `Drawn in blocks of ${g.block} x ${g.block} cells: this is their mean.` : "",
    credit: "Your file, read in your browser with aquascope (it did not leave this page).",
  });
}

// ── your MODFLOW heads ──────────────────────────────────────────────────────

const modelShown = () => Boolean(model && modelOn);

function modelRange() {
  return model.meta.range || [0, 1];
}

function paintModel() {
  if (!model || !ensureVectorLayers()) return;
  const frame = frameOf(model.heads, model.shape, model.t, model.l);
  const range = modelRange();
  for (const f of model.fc.features) {
    const h = frame[f.properties.c];
    f.properties.color = Number.isFinite(h) ? rgb(valueColor(h, range, "level")) : null;
  }
  map.getSource(L.model).setData(model.fc);
  setLayerVisibility();
  refreshLegend("modflow");
}

function setModelStep(t) {
  if (!model) return;
  model.t = Math.max(0, Math.min(model.meta.times.length - 1, t));
  paintModel();
}

function edgeOf(rings) {
  // The model's outline: the hull of its corners is enough to show where it sits.
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (let k = 0; k < rings.length; k += 2) {
    w = Math.min(w, rings[k]); e = Math.max(e, rings[k]); s = Math.min(s, rings[k + 1]); n = Math.max(n, rings[k + 1]);
  }
  return { w, s, e, n };
}

// The model's outline: the four outer corners of a structured grid (rotated with it); none for a DISV grid.
function drawOutline() {
  const ring = modelOutline(model.meta.grid, model.rings);
  map.getSource(L.modelEdge).setData(ring ? { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ring } } : EMPTY);
}

async function loadModel(files, place = {}) {
  const sorted = sortModelFiles(files);
  if (!sorted.hds) {
    openModal("Your MODFLOW heads", "<p>Add the binary head file (<code>.hds</code>, <code>.hed</code> or <code>.bhd</code>), with its grid " +
      "(<code>.dis</code>, <code>.disv</code> or <code>.dis.grb</code>).</p>");
    return;
  }
  modelBusy = true;
  registerModelRow();
  renderModelLegend();
  try {
    const read = (f) => f.arrayBuffer();
    const out = await call("groundwater", {
      op: "modflow", hds: await read(sorted.hds), grid: sorted.grid ? await read(sorted.grid) : null,
      gridName: sorted.grid ? sorted.grid.name : null, sidecars: await Promise.all(sorted.sidecars.map(read)), place,
    });
    if (out.error) {
      if (out.needs) { pending = files; placeForm(out, sorted); return; }
      openModal("Your MODFLOW heads", `<p>${escapeHtml(out.error)}</p>`);
      return;
    }
    model = {
      meta: out, rings: out.rings, offsets: out.offsets, heads: out.heads, shape: out.shape,
      t: 0, l: 0, name: sorted.hds.name,
      fc: cellsGeoJSON(out.rings, out.offsets),
    };
    modelOn = true;
    pending = null;
    ensureVectorLayers();
    const b = edgeOf(out.rings);
    drawOutline();
    paintModel();
    syncModelDated();
    openLegendRow("modflow", true);
    map.fitBounds([[b.w, b.s], [b.e, b.n]], { padding: 80, maxZoom: 14, duration: 1200 });
  } catch (err) {
    openModal("Your MODFLOW heads", `<p>Could not read the files: ${escapeHtml(err && err.message ? err.message : String(err))}</p>`);
  } finally {
    modelBusy = false;
    renderModelLegend();
  }
}

// Where the model sits, when its files do not say: the lower-left corner, the rotation, the CRS (and the cell
// size without a grid file). Nothing here is sent anywhere; it goes to the worker with the file.
function placeForm(out, sorted) {
  const needCell = !sorted.grid;
  const field = (name, label, hint, value = "") => `<label class="gw-field"><span>${label}</span>` +
    `<input name="${name}" value="${escapeHtml(String(value ?? ""))}" autocomplete="off" spellcheck="false"><small class="muted">${hint}</small></label>`;
  openModal("Where is this model?", `
    <form class="gw-form" novalidate>
      <p>${escapeHtml(out.error)}</p>
      <p class="muted">${escapeHtml(sorted.hds.name)}: ${out.nrow} rows x ${out.ncol} columns, ${plural(out.layers.length, "layer")},
        ${plural(out.n_times, "time step")}${sorted.grid ? `; grid from ${escapeHtml(out.grid_from || sorted.grid.name)}` : ""}.</p>
      <div class="gw-grid">
        ${field("x", "Lower-left x", "in the CRS's units", out.xorigin || "")}
        ${field("y", "Lower-left y", "", out.yorigin || "")}
        ${field("rotation", "Rotation", "degrees, counter-clockwise", out.angrot || "")}
        ${field("crs", "CRS", "EPSG code, e.g. 27700 or 32631")}
        ${needCell ? field("cell", "Cell size", "one cell's width, in the CRS's units") : ""}
        ${field("start", "Start date", "optional, YYYY-MM-DD, for the time bar")}
      </div>
      <p class="gw-form-err" role="alert" hidden></p>
      <div class="gw-form-row"><button class="btn primary" type="submit">Place it on the map</button>
        <span class="muted">The files stay in your browser.</span></div>
    </form>`);
  const form = $("modal-body").querySelector("form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const v = Object.fromEntries(new FormData(form).entries());
    const place = readPlaceForm(v, { needCell });
    const err = form.querySelector(".gw-form-err");
    if (place.error) { err.textContent = place.error; err.hidden = false; return; }
    if (!place.crs) { err.textContent = "The CRS is needed to place the model."; err.hidden = false; return; }
    closeModal();
    void loadModel(pending || [], place);
  });
}

function pickFiles() {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = ".hds,.hed,.bhd,.ddn,.dis,.disv,.grb,.nam,.tdis";
  input.addEventListener("change", () => { if (input.files && input.files.length) void loadModel([...input.files]); });
  input.click();
}

function removeModel() {
  model = null;
  if (state.mapOk && map) {
    for (const id of [L.model, L.modelEdge]) if (map.getSource(id)) map.getSource(id).setData(EMPTY);
  }
  datedExtras.delete("modflow");
  syncTimeBar({ layersChanged: true });
  refreshLegend("modflow");
}

function syncModelDated() {
  if (model && model.meta.dates && model.meta.dates.length > 1) {
    const d = model.meta.dates;
    datedExtras.set("modflow", { id: "modflow", label: "Your MODFLOW heads", time: true, since: String(d[0]).slice(0, 10),
      until: String(d[d.length - 1]).slice(0, 10) });
  } else {
    datedExtras.delete("modflow");
  }
  syncTimeBar({ layersChanged: true });
}

// Dropping files on the map: MODFLOW files load as your model. The map has one drop zone (#mesh-drop, drawn by
// mesh-view.js, which names MODFLOW too): a HEC-RAS or UGRID result goes to the mesh viewer, these come here.
function initDrop() {
  const wrap = document.querySelector(".map-wrap") || $("map");
  if (!wrap) return;
  wrap.addEventListener("drop", (e) => {
    const files = [...((e.dataTransfer && e.dataTransfer.files) || [])];
    if (!isModflowDrop(files)) return;
    e.preventDefault();
    void loadModel(files);
  });
}

// ── the legend ──────────────────────────────────────────────────────────────

function rampKey(m, range, unit = "m") {
  const u = unit ? ` ${escapeHtml(unit)}` : "";
  return `<div class="gw-ramp" style="background:${rampCss(m)}" role="img" aria-label="${m === "change" ? "Brown fell, blue-green rose" : "Pale is low, deep blue is high"}"></div>` +
    (range ? `<div class="gw-ends"><span>${m === "change" ? "&le; " : ""}${escapeHtml(range.lo)}${u}</span>` +
      `${m === "change" ? "<span>0</span>" : ""}<span>${m === "change" ? "&ge; " : ""}${escapeHtml(range.hi)}${u}</span></div>` : "");
}

function bodyHtml() {
  const seg = (m, label) => `<button type="button" class="gw-seg${mode === m ? " on" : ""}" data-act="mode-${m}" aria-pressed="${mode === m}">${label}</button>`;
  let html = `<div class="gw-segs" role="group" aria-label="What the surface shows">${seg("level", "Level")}${seg("change", "Change over 12 months")}</div>`;
  if (res && res.outside) {
    html += `<p class="gw-line">The Archive's wells are in England. <button type="button" class="gw-link" data-act="go">Go there</button></p>`;
  } else if (res) {
    const k = keyLines(res);
    if (k.head) html += `<p class="gw-head">${escapeHtml(k.head)}</p>`;
    if (res.grid) html += rampKey(res.mode, k.range);
    for (const line of k.lines) html += `<p class="gw-line">${escapeHtml(line)}</p>`;
    if (res.grid) {
      html += `<p class="gw-mask"><i class="gw-hatch" aria-hidden="true"></i><span>${escapeHtml(k.mask)}</span></p>`;
    }
  } else if (busy) {
    html += '<p class="gw-line muted">Reading the wells…</p>';
  }
  if (failed) html += `<p class="gw-err">Could not draw the surface: ${escapeHtml(failed)}</p>`;
  const grace = state.overlays && state.overlays.has("storage");
  html += `<p class="gw-line">${grace ? "GRACE storage is on: the regional view from space."
    : 'From space: <button type="button" class="gw-link" data-act="grace">GRACE water storage</button>'} · ` +
    '<button type="button" class="gw-link" data-act="modflow">Your MODFLOW heads</button></p>';
  html += `<p class="gw-src">Environment Agency, OGL v3.0 · <button type="button" class="gw-link" data-act="about">About</button></p>`;
  return html;
}

function registerRow() {
  registerLegendRow({
    id: "groundwater",
    title: "Groundwater",
    mark: () => `<i class="gw-mark" style="background:${rampCss(mode)}"></i>`,
    summary: () => (res && res.outside ? "wells in England" : rowSummary(res, { busy: busy > 0, error: failed })),
    on: () => Boolean(state.groundwater && state.mapOk),
    empty: () => false,
    toggle: (on) => setGroundwater(on),
    body: bodyHtml,
    act: (name) => {
      if (name === "mode-level") setMode("level");
      else if (name === "mode-change") setMode("change");
      else if (name === "grace") { actions.setOverlay("storage", true); renderLegend(); }
      else if (name === "modflow") pickFiles();
      else if (name === "about") about();
      else if (name === "go") map.fitBounds([[GW_EXTENT[0], GW_EXTENT[1]], [GW_EXTENT[2], GW_EXTENT[3]]], { padding: 40, duration: 1400 });
    },
  });
}

function modelBody() {
  if (modelBusy) return '<p class="gw-line muted">Reading your files…</p>';
  if (!model) return "";
  const m = model.meta;
  const n = m.times.length;
  const layers = m.layers.map((lay, i) => `<option value="${i}"${i === model.l ? " selected" : ""}>Layer ${lay}</option>`).join("");
  const r = modelRange();
  const dated = Boolean(m.dates && n > 1);
  return `<p class="gw-head">${escapeHtml(model.name)}: ${escapeHtml(m.text.toLowerCase())}, ${escapeHtml(timeLabel(m, model.t))}</p>` +
    rampKey("level", { lo: fmtM(r[0]), hi: fmtM(r[1]) }, lengthUnit(m)) +
    `<div class="gw-model-ctl">` +
    (m.layers.length > 1 ? `<select class="gw-layer" aria-label="Model layer">${layers}</select>` : `<span class="muted">Layer ${m.layers[0]}</span>`) +
    (n > 1 ? `<input class="gw-step" type="range" min="0" max="${n - 1}" step="1" value="${model.t}" aria-label="Time step">` +
      `<span class="gw-step-n muted">${model.t + 1} / ${n}</span>` : "") +
    `</div>` +
    (n > 1 ? `<p class="gw-line">${dated ? 'Plays with the time bar. <button type="button" class="gw-link" data-act="play">Play the run</button>'
      : "No start date, so the slider steps through the model times."}</p>` : "") +
    `<p class="gw-line muted">Colours span the whole run. ${escapeHtml(m.grid.from)}, placed by ${escapeHtml(m.grid.placed_by)}` +
    `${/^EPSG:\d+$/.test(m.grid.crs) ? ` (${escapeHtml(m.grid.crs)})` : ""}.` +
    `${(m.notes || []).map((x) => ` ${escapeHtml(x)}`).join("")}</p>` +
    '<p class="gw-src">Read in your browser; the file did not leave it · <button type="button" class="gw-link" data-act="fit">Zoom to it</button> · ' +
    '<button type="button" class="gw-link" data-act="remove">Remove</button></p>';
}

function registerModelRow() {
  registerLegendRow({
    id: "modflow",
    title: "Your MODFLOW heads",
    mark: () => `<i class="gw-mark gw-mark-grid" style="background:${rampCss("level")}"></i>`,
    summary: () => (modelBusy ? "reading…" : model ? timeLabel(model.meta, model.t) : ""),
    on: () => modelShown(),
    empty: () => !model && !modelBusy,
    toggle: (on) => { modelOn = Boolean(on); setLayerVisibility(); refreshLegend("modflow"); },
    body: modelBody,
    shown: () => Boolean(model || modelBusy),
    act: (name) => {
      if (name === "remove") removeModel();
      else if (name === "fit" && model) { const b = edgeOf(model.rings); map.fitBounds([[b.w, b.s], [b.e, b.n]], { padding: 80, duration: 1000 }); }
      else if (name === "play" && model && model.meta.dates) playModel();
    },
  });
}

// The legend body's own controls (the select and the slider) are not data-act buttons: bind them on input.
function bindModelControls() {
  const host = $("map-legend");
  if (!host || host.__gwBound) return;
  host.__gwBound = true;
  host.addEventListener("input", (e) => {
    if (!model) return;
    if (e.target.classList.contains("gw-step")) {
      model.t = Number(e.target.value);
      paintModel();
      if (model.meta.dates) setTime({ date: String(model.meta.dates[model.t]).slice(0, 10) }, { source: "chart" });
    }
  });
  host.addEventListener("change", (e) => {
    if (model && e.target.classList.contains("gw-layer")) { model.l = Number(e.target.value); paintModel(); }
  });
}

function playModel() {
  const d = model.meta.dates;
  const tbPlay = $("tb-play");
  if (!tbPlay) return;
  if (state.playing) { tbPlay.click(); return; }
  const from = String(d[0]).slice(0, 10), to = String(d[d.length - 1]).slice(0, 10);
  const days = (Date.parse(to) - Date.parse(from)) / 86400000;
  const step = days > 3 * 365 ? "month" : days > 120 ? "week" : "day";
  setTime({ step, range: { from, to }, date: from }, { source: "bar" });
  tbPlay.click();
}

function renderLegend() { refreshLegend("groundwater"); }
function renderModelLegend() { refreshLegend("modflow"); }

function about() {
  openModal("Groundwater", `
    <p><strong>The surface.</strong> The Environment Agency measures groundwater levels in about 380 wells in England,
    and the AquaScope Archive mirrors their records. Their values are heads in metres above Ordnance Datum (mAOD),
    a datum common to every well, so the level is a head surface; it is not the depth to water, which would need
    each well's ground level. For a month, each well's value is the mean of its readings in it, or the reading
    nearest the 15th within a month. When few wells have reported yet (the Archive runs a few weeks behind), the
    latest month most wells reported in is drawn, and the legend says so.</p>
    <p><strong>How it is drawn.</strong> Inverse distance weighting between the wells (power 2, fading to nothing
    at 40 km), drawn only where at least 3 wells lie within 20 km. Where only one or two do, the map is hatched grey:
    too few wells to draw anything honest. Where none do, nothing is drawn. The change is this month minus the same
    month a year before, at the wells that have both.</p>
    <p><strong>What it is not.</strong> A picture of the measurements, not a groundwater model: wells in different
    aquifers are mixed, and between wells the surface is a guess. GRACE (Layers, Water storage anomaly) shows the
    regional change in total water storage from space, at about 300 km, from 2002 to mid-2022.</p>
    <p><strong>Your MODFLOW heads.</strong> Drop a binary head file (MODFLOW-2005 or 6) on the map with its grid
    (<code>.dis</code>, <code>.disv</code> or <code>.dis.grb</code>), and the <code>.nam</code> FloPy writes if it
    carries the model's place. The file is read in your browser by aquascope's own reader and does not leave it.</p>
    <p class="muted">Data: Environment Agency, Open Government Licence v3.0, via the AquaScope Archive
    (obs/groundwater_level/uk_ea). The same in Python: <code>aquascope groundwater surface</code> and
    <code>aquascope modflow model.hds --grid model.dis.grb</code>.</p>`);
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initGroundwater() {
  // Debug hooks (harmless in production, like the rest of window.__aq): the surface and the cell under a point.
  dbg.groundwater = () => res;
  dbg.gwCell = (lon, lat) => (res && res.grid ? gridCellAt(res.grid, lon, lat) : null);
  registerRow();
  const fromUrl = readGwParam(location.hash);
  if (fromUrl !== null) state.groundwater = fromUrl;
  const toggle = $("toggle-groundwater");
  if (toggle) {
    toggle.checked = Boolean(state.groundwater);
    toggle.addEventListener("change", (e) => setGroundwater(e.target.checked));
  }
  bindModelControls();
  if (!state.mapOk || !map) return;
  initDrop();
  let t = 0;
  map.on("moveend", () => { clearTimeout(t); t = setTimeout(() => void refresh(), 250); });
  // A basemap change replaces the style and drops our sources: put them back with what they showed.
  map.on("style.load", () => {
    setTimeout(() => {
      ensureVectorLayers();
      drawnKey = "";
      if (state.groundwater) void refresh();
      if (model) { drawOutline(); paintModel(); }
    }, 0);
  });
  clickLayers.add(L.hit);
  clickLayers.add(L.wells);
  clickLayers.add(L.model);
  map.on("click", L.wells, onWellClick);
  map.on("click", L.hit, (e) => {
    if (map.queryRenderedFeatures(e.point, { layers: [L.wells, L.model].filter((id) => map.getLayer(id)) }).length) return;
    onSurfaceClick(e);
  });
  map.on("click", L.model, onModelClick);
  for (const id of [L.wells, L.model]) {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
  onTime((tm) => {
    if (tm.date === tm.prev.date) return;
    if (state.groundwater && monthOf(tm.date) !== monthOf(tm.prev.date)) void refresh();
    if (model && model.meta.dates) {
      const next = stepForDate(model.meta.dates, tm.date);
      if (next !== model.t) setModelStep(next);
    }
  });
  window.addEventListener("hashchange", () => {
    const want = readGwParam(location.hash);
    const next = want === null ? false : want;
    if (next !== Boolean(state.groundwater)) setGroundwater(next, { write: false });
  });
  ensureVectorLayers();
  if (state.groundwater) setGroundwater(true, { write: false });
}

/** What the layer shows now, for Explain and Scout: the surface's summary (no grid), or null when off. */
export function groundwaterShown() {
  if (!state.groundwater || !res || res.outside) return null;
  const { grid, contours, ...rest } = res;
  void grid; void contours;
  return rest;
}
