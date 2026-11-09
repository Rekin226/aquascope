// A cross-section and a hydrograph through a flood model's results (#621): RAS Mapper's profile line and time
// series plot, on the globe.
//
// "Draw a section" on the model's legend row: drag a line across the flood (or tap its two ends; mouse, pen and
// touch alike) and a panel under the map draws the ground and the water along it, A on the left and A′ on the
// right, stepping with the time bar; the highest water of the run is dashed. Hovering the chart marks the same
// spot on the map. A click on a cell puts its hydrograph in the cell's card: the depth (left axis) and the water
// level (right axis) over the run, with the arrival, the peak and the time under water marked.
//
// The numbers are aquascope.flood_section's (run in the mesh worker beside the file, which never leaves the
// browser); flood-views-core.js draws them.

import { escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { cardOpen, updateCard } from "./map-card.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { addMeshTool, currentMesh, meshAsk, meshStepNow, onMeshCell, onMeshCleared, onMeshStep } from "./mesh-view.js?v=__BUILD__";
import { hydroAlt, hydroKey, hydrographSvg, sectionHeading, sectionIndexAt, sectionNote, sectionReadout, sectionSvg } from "./flood-views-core.js?v=__BUILD__";

const LINE = "fs-line";
const LINE_CASE = "fs-line-case";
const SPOT = "fs-spot";
const TAP_PX = 10;

let sec = null;          // the section on show: aquascope.flood_section.section's answer
let line = null;         // [[lon, lat], [lon, lat]]
let drawing = null;      // { overlay, start, moved } while a line is being drawn
let panel = null;
let hover = -1;
let ends = [];           // the A and A′ markers
let hydro = null;        // { id, cell, res } of the card's hydrograph
let busy = 0;

const unitOf = () => { const m = currentMesh(); return (m && m.meta.units && (m.meta.units.depth || m.meta.units.water_surface)) || "m"; };
const now = () => meshStepNow() || { view: "max", step: 0 };

// ── the line on the map ─────────────────────────────────────────────────────

function lineData(coords) {
  return { type: "FeatureCollection", features: coords && coords.length > 1
    ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } }] : [] };
}

function ensureLayers() {
  if (!map || !map.getStyle()) return;
  if (!map.getSource(LINE)) {
    map.addSource(LINE, { type: "geojson", data: lineData(line) });
    map.addLayer({ id: LINE_CASE, type: "line", source: LINE, layout: { "line-cap": "round" },
      paint: { "line-color": "#ffffff", "line-width": 6, "line-opacity": 0.9 } });
    map.addLayer({ id: LINE, type: "line", source: LINE, layout: { "line-cap": "round" },
      paint: { "line-color": "#0f1c26", "line-width": 2.4 } });
  }
  if (!map.getSource(SPOT)) {
    map.addSource(SPOT, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: SPOT, type: "circle", source: SPOT,
      paint: { "circle-radius": 5, "circle-color": "#0b6bb8", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
  }
}

function setLine(coords) {
  ensureLayers();
  const src = map && map.getSource(LINE);
  if (src) src.setData(lineData(coords));
}

function setSpot(lonlat) {
  const src = map && map.getSource(SPOT);
  if (src) src.setData({ type: "FeatureCollection", features: lonlat ? [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: lonlat } }] : [] });
}

function placeEnds(coords) {
  for (const m of ends) m.remove();
  ends = [];
  if (!coords || typeof maplibregl === "undefined") return;
  [["A", coords[0]], ["A′", coords[coords.length - 1]]].forEach(([label, ll]) => {
    const el = document.createElement("span");
    el.className = "fs-end";
    el.textContent = label;
    ends.push(new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat(ll).addTo(map));
  });
}

function removeLine() {
  for (const id of [SPOT, LINE, LINE_CASE]) if (map && map.getLayer(id)) map.removeLayer(id);
  for (const id of [SPOT, LINE]) if (map && map.getSource(id)) map.removeSource(id);
  for (const m of ends) m.remove();
  ends = [];
}

// ── drawing: drag across, or tap the two ends ───────────────────────────────

function hint(text) {
  if (!drawing) return;
  drawing.hint.textContent = text;
}

function lngLatAt(e) {
  const r = map.getContainer().getBoundingClientRect();
  const ll = map.unproject([e.clientX - r.left, e.clientY - r.top]);
  return [ll.lng, ll.lat];
}

export function startDrawing() {
  if (!currentMesh() || !state.mapOk || !map) return;
  if (drawing) return;
  const wrap = map.getContainer();
  const overlay = document.createElement("div");
  overlay.className = "fs-draw";
  overlay.setAttribute("role", "application");
  overlay.setAttribute("aria-label", "Draw the section: drag across the flood, or tap its two ends. Escape cancels.");
  overlay.tabIndex = 0;
  const tip = document.createElement("p");
  tip.className = "fs-hint";
  overlay.appendChild(tip);
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "btn small fs-cancel";
  cancel.textContent = "Cancel";
  overlay.appendChild(cancel);
  wrap.appendChild(overlay);
  drawing = { overlay, hint: tip, start: null, down: null, tapped: null };
  hint("Drag a line across the flood, or tap its two ends.");
  overlay.focus({ preventScroll: true });
  cancel.addEventListener("click", (e) => { e.stopPropagation(); stopDrawing(); });
  cancel.addEventListener("pointerdown", (e) => e.stopPropagation());
  overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") stopDrawing(); });
  overlay.addEventListener("pointerdown", (e) => {
    if (e.button > 0) return;
    e.preventDefault();
    overlay.setPointerCapture(e.pointerId);
    drawing.down = { x: e.clientX, y: e.clientY, ll: lngLatAt(e) };
  });
  overlay.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const ll = lngLatAt(e);
    if (drawing.down) {
      const moved = Math.hypot(e.clientX - drawing.down.x, e.clientY - drawing.down.y) > TAP_PX;
      if (moved) setLine([drawing.down.ll, ll]);
    } else if (drawing.tapped) {
      setLine([drawing.tapped, ll]);
    }
  });
  overlay.addEventListener("pointerup", (e) => {
    if (!drawing || !drawing.down) return;
    const ll = lngLatAt(e);
    const moved = Math.hypot(e.clientX - drawing.down.x, e.clientY - drawing.down.y) > TAP_PX;
    const from = drawing.down.ll;
    drawing.down = null;
    if (moved) { finish([from, ll]); return; }
    if (!drawing.tapped) {
      drawing.tapped = ll;
      setLine([ll, ll]);
      hint("Now tap the other end.");
      return;
    }
    finish([drawing.tapped, ll]);
  });
  overlay.addEventListener("pointercancel", () => { if (drawing) drawing.down = null; });
  // a tap's compatibility mouse events and click must not reach the map under it (they would open a cell's card)
  for (const type of ["touchstart", "touchend", "click", "mousedown", "mouseup"]) {
    overlay.addEventListener(type, (e) => { if (e.target === overlay || type !== "click") e.preventDefault(); e.stopPropagation(); },
      { passive: false });
  }
  // the wheel still zooms the map under the overlay
  overlay.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = map.getContainer().getBoundingClientRect();
    const around = map.unproject([e.clientX - r.left, e.clientY - r.top]);
    map.easeTo({ zoom: map.getZoom() - Math.sign(e.deltaY) * 0.5, around, duration: 150 });
  }, { passive: false });
  announce("Drawing a section: drag a line across the flood, or tap its two ends.");
}

function stopDrawing() {
  if (!drawing) return;
  const { overlay } = drawing;
  drawing = null;
  // Kept a moment, out of sight, so the click that ends a tap lands on it and not on the map.
  overlay.classList.add("done");
  overlay.replaceChildren();
  setTimeout(() => overlay.remove(), 400);
  setLine(line);
}

async function finish(coords) {
  stopDrawing();
  line = coords;
  setLine(line);
  placeEnds(line);
  await loadSection();
}

// ── the section ─────────────────────────────────────────────────────────────

async function loadSection() {
  const my = ++busy;
  openPanel();
  panel.body.innerHTML = '<p class="fs-wait"><span class="ms-spin" aria-hidden="true"></span>Reading the water along the line…</p>';
  try {
    const res = await meshAsk("section", { line });
    if (my !== busy) return;
    sec = res;
    hover = -1;
    renderPanel();
    keepLineInSight();
    announce(`Section drawn: ${sectionHeading(sec, now())}.`);
  } catch (err) {
    if (my !== busy) return;
    sec = null;
    panel.body.innerHTML = `<p class="fs-error">${escapeHtml(err.message || String(err))}</p>`;
  }
}

function openPanel() {
  if (panel) { panel.el.hidden = false; return; }
  const el = document.createElement("section");
  el.className = "fs-panel";
  el.setAttribute("aria-label", "Cross-section");
  el.innerHTML = '<div class="fs-head"><p class="fs-title"><b>Section A–A′</b><span class="fs-sub"></span></p>' +
    '<div class="fs-btns"><button type="button" class="fd-go" data-act="redraw">Redraw</button>' +
    '<button type="button" class="fs-x" data-act="close" aria-label="Close the section" title="Close">×</button></div></div>' +
    '<div class="fs-body"></div>';
  map.getContainer().parentElement.appendChild(el);
  panel = { el, body: el.querySelector(".fs-body"), sub: el.querySelector(".fs-sub") };
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    if (b.dataset.act === "close") closeSection();
    else if (b.dataset.act === "redraw") startDrawing();
  });
  el.addEventListener("pointermove", (e) => {
    const svg = e.target.closest && e.target.closest("svg.fv-chart");
    if (!svg || !sec) return;
    const r = svg.getBoundingClientRect();
    const w = Number(svg.getAttribute("width"));
    const i = sectionIndexAt(sec, ((e.clientX - r.left) / r.width) * w, { w, h: Number(svg.getAttribute("height")) });
    if (i !== hover) { hover = i; renderPanel({ keepSize: true }); }
  });
  el.addEventListener("pointerleave", () => { if (hover >= 0) { hover = -1; renderPanel({ keepSize: true }); } });
  globalThis.addEventListener("resize", () => { if (sec && panel && !panel.el.hidden) renderPanel(); });
}

// Sit above the time bar (and the card sheet on a phone), whatever their height.
function placePanel() {
  const wrap = map.getContainer().parentElement.getBoundingClientRect();
  const bar = document.getElementById("timebar-wrap");
  let bottom = 16;
  if (bar && bar.offsetParent !== null) {
    const r = bar.getBoundingClientRect();
    if (r.height) bottom = Math.max(bottom, wrap.bottom - r.top + 10);
  }
  panel.el.style.setProperty("--fs-bottom", `${Math.round(bottom)}px`);
}

function chartSize() {
  const w = Math.max(260, Math.round(panel.body.clientWidth || 600));
  const h = (globalThis.innerWidth || 1024) <= 640 ? 170 : 200;
  return { w, h };
}

function renderPanel() {
  if (!panel || !sec) return;
  const { view, step } = now();
  const unit = unitOf();
  placePanel();
  const size = chartSize();
  panel.sub.textContent = ` · ${sectionHeading(sec, { view, step })}`;
  const read = hover >= 0 ? sectionReadout(sec, hover, { view, step, unit }) : "Hover or touch the chart to read a point.";
  panel.body.innerHTML = sectionSvg(sec, { view, step, hover, unit, ...size }) +
    `<p class="fs-read" aria-live="polite">${escapeHtml(read)}</p>` +
    `<p class="fs-note">${escapeHtml(sectionNote(sec, { unit }))}</p>`;
  panel.body.querySelector("svg").setAttribute("aria-label", `Cross-section A to A′, ${sectionHeading(sec, { view, step })}`);
  setSpot(hover >= 0 ? [sec.lon[hover], sec.lat[hover]] : null);
}

// When the panel covers part of the line (a phone, a long line low on the map), move the map so all of it shows
// above the panel.
function keepLineInSight() {
  if (!line || !panel || panel.el.hidden) return;
  const r = panel.el.getBoundingClientRect();
  const box = map.getContainer().getBoundingClientRect();
  const hidden = line.some((ll) => { const p = map.project(ll); return box.top + p.y > r.top - 12; });
  if (!hidden) return;
  const lons = line.map((p) => p[0]), lats = line.map((p) => p[1]);
  const pad = Math.min(48, box.width / 8);
  map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], {
    padding: { top: box.width > 640 ? 130 : 70, left: pad, right: pad, bottom: Math.max(pad, box.bottom - r.top + 24) }, maxZoom: map.getZoom() + 1, duration: 700 });
}

function closeSection() {
  busy++;
  sec = null;
  line = null;
  if (panel) panel.el.hidden = true;
  stopDrawing();
  removeLine();
}

// ── the hydrograph in a cell's card ─────────────────────────────────────────

function hydroChart() {
  const { step, view } = now();
  const unit = unitOf();
  return { label: "Depth and water level over the run", svg: hydrographSvg(hydro.res, { step: view === "max" ? -1 : step, unit }),
    alt: hydroAlt(hydro.res, { unit }), key: escapeHtml(hydroKey(hydro.res, { unit })) };
}

async function showHydrograph(cell, id) {
  const m = currentMesh();
  if (!m || !m.meta.times.length) return;
  hydro = { id, cell, res: null };
  try {
    const res = await meshAsk("hydrograph", { cell });
    if (!hydro || hydro.id !== id) return;
    hydro.res = res;
    if (res.wet) updateCard(id, { chart: hydroChart() });   // a dry cell keeps the card's own "dry" line
  } catch (err) {
    console.warn("hydrograph", err && err.message);
  }
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initFloodSection() {
  addMeshTool({ id: "section", label: "Draw a section", title: "Drag a line across the flood for the ground and the water along it, step by step",
    run: () => startDrawing() });
  onMeshCell((cell, id) => { void showHydrograph(cell, id); });
  onMeshStep(() => {
    if (sec) renderPanel();
    if (hydro && hydro.res && hydro.res.wet && cardOpen()) updateCard(hydro.id, { chart: hydroChart() });
  });
  onMeshCleared(() => { closeSection(); hydro = null; });
  if (state.mapOk && map) {
    map.on("style.load", () => setTimeout(() => { if (line) { setLine(line); } }, 0));
  }
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.section = { draw: (coords) => finish(coords), start: startDrawing, result: () => sec, hydrograph: () => (hydro ? hydro.res : null) };
}
