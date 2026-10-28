// Levees and breaches on Flood it's grid (#663), on the map: the drawn lines (a levee in amber with its crest along
// it, a breach dashed red), the line being drawn, the handles that move a selected line's points, and the drawing
// tool itself (tap the points, or drag a straight line; Done, a double tap or Enter ends it, Escape cancels). The
// lines become the package's edits in floodsim-map.js; nothing here computes a level. The same tool draws an area
// for the sketchpad (#670 and on): `polygon: true` closes the outline, needs three points, and a drag draws a box.

import { state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { sketchFeatures } from "./floodsim-sketch-core.js?v=__BUILD__";

const SRC = "floodsim-sketch";
const L = { cas: "floodsim-sketch-case", levee: "floodsim-sketch-levee", breach: "floodsim-sketch-breach",
  draft: "floodsim-sketch-draft", pt: "floodsim-sketch-draft-pt", label: "floodsim-sketch-label" };
const TAP_PX = 8;
const LEVEE = "#9a5a00", LEVEE_ON = "#d87400", BREACH = "#d1242f";

let shown = { sketches: [], selected: null, draft: null };
let handles = [];
let drawing = null;
let clickHook = null;
let moveHook = null;

function ensureLayers() {
  if (!state.mapOk || !map || map.getSource(SRC)) return;
  map.addSource(SRC, { type: "geojson", data: sketchFeatures([]) });
  const isKind = (k) => ["==", ["get", "kind"], k];
  map.addLayer({ id: L.cas, type: "line", source: SRC, filter: ["any", isKind("levee"), isKind("breach")],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ffffff", "line-width": ["case", ["==", ["get", "selected"], 1], 10, 8], "line-opacity": 0.9 } });
  map.addLayer({ id: L.levee, type: "line", source: SRC, filter: isKind("levee"),
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["case", ["==", ["get", "selected"], 1], LEVEE_ON, LEVEE], "line-width": ["case", ["==", ["get", "selected"], 1], 6, 4.5] } });
  map.addLayer({ id: L.breach, type: "line", source: SRC, filter: isKind("breach"),
    layout: { "line-cap": "butt", "line-join": "round" },
    paint: { "line-color": BREACH, "line-width": ["case", ["==", ["get", "selected"], 1], 6, 4.5], "line-dasharray": [1.2, 0.8] } });
  map.addLayer({ id: L.draft, type: "line", source: SRC, filter: isKind("draft"),
    paint: { "line-color": "#0f1c26", "line-width": 2.5, "line-dasharray": [2, 1] } });
  map.addLayer({ id: L.pt, type: "circle", source: SRC, filter: ["all", isKind("draft"), ["==", ["geometry-type"], "Point"]],
    paint: { "circle-radius": 5, "circle-color": "#0f1c26", "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });
  map.addLayer({ id: L.label, type: "symbol", source: SRC, filter: ["any", isKind("levee"), isKind("breach")],
    layout: { "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"],
      "text-size": 12, "text-offset": [0, -1.1], "text-allow-overlap": true },
    paint: { "text-color": ["case", ["==", ["get", "kind"], "breach"], BREACH, LEVEE], "text-halo-color": "#fff", "text-halo-width": 2 } });
  for (const id of [L.levee, L.breach, L.cas]) {
    map.on("click", id, (e) => {
      const f = e.features && e.features[0];
      if (f && clickHook && !drawing) { e.preventDefault(); clickHook(Number(f.properties.id)); }
    });
    map.on("mouseenter", id, () => { if (!drawing) map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { if (!drawing) map.getCanvas().style.cursor = ""; });
  }
}

/** Draw the lines: every sketch, the selected one with handles on its points, and the line being drawn. */
export function drawSketches(sketches, { selected = null, draft = null } = {}) {
  shown = { sketches: sketches || [], selected, draft };
  if (!state.mapOk || !map) return;
  ensureLayers();
  const src = map.getSource(SRC);
  if (src) src.setData(sketchFeatures(shown.sketches, { selected, draft }));
  placeHandles();
}

export function removeSketchLayers() {
  for (const h of handles) h.remove();
  handles = [];
  if (!state.mapOk || !map) return;
  for (const id of Object.values(L).reverse()) if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(SRC)) map.removeSource(SRC);
}

/** fn(id) when a drawn line is clicked on the map; fn(id, line) when a selected line's point is dragged. */
export function onSketchClick(fn) { clickHook = fn; }
export function onSketchMove(fn) { moveHook = fn; }

function placeHandles() {
  for (const h of handles) h.remove();
  handles = [];
  const s = shown.sketches.find((x) => x.id === shown.selected);
  if (!s || drawing || typeof maplibregl === "undefined") return;
  s.line.forEach((ll, k) => {
    const el = document.createElement("span");
    el.className = `fsk-handle fsk-${s.kind}`;
    el.title = "Drag to move this point of the line";
    el.setAttribute("aria-label", `Point ${k + 1} of the ${s.kind}`);
    // eslint-disable-next-line no-undef
    const m = new maplibregl.Marker({ element: el, anchor: "center", draggable: true }).setLngLat(ll).addTo(map);
    m.on("drag", () => {
      const p = m.getLngLat();
      const line = s.line.map((q, j) => (j === k ? [p.lng, p.lat] : q));
      const src = map.getSource(SRC);
      if (src) src.setData(sketchFeatures(shown.sketches.map((x) => (x.id === s.id ? { ...x, line } : x)), { selected: s.id }));
    });
    m.on("dragend", () => {
      const p = m.getLngLat();
      const line = s.line.map((q, j) => (j === k ? [p.lng, p.lat] : q));
      if (moveHook) moveHook(s.id, line);
    });
    handles.push(m);
  });
}

// ── the drawing tool ────────────────────────────────────────────────────────

function lngLatAt(e) {
  const r = map.getContainer().getBoundingClientRect();
  const ll = map.unproject([e.clientX - r.left, e.clientY - r.top]);
  return [ll.lng, ll.lat];
}

export const isDrawing = () => Boolean(drawing);

/** Cancel a line being drawn (its promise gives null). */
export function cancelDraw() { if (drawing) drawing.finish(null); }

/**
 * Draw a line on the map: resolves with [[lon, lat], ...] (two points or more) or null when cancelled. `kind` is
 * "levee" or "breach" (for the words); the other lines stay on the map while it is drawn. With `polygon` it draws
 * an area instead (three points or more, the outline shown closed; a drag draws a box), and `what` names it.
 */
export function startLineDraw(kind = "levee", { polygon = false, what: named = "" } = {}) {
  if (!state.mapOk || !map) return Promise.resolve(null);
  if (drawing) drawing.finish(null);
  ensureLayers();
  for (const h of handles) h.remove();
  handles = [];
  const wrap = map.getContainer();
  const overlay = document.createElement("div");
  overlay.className = "fs-draw fsk-draw";
  overlay.setAttribute("role", "application");
  const what = named || (kind === "breach" ? "the breach, along the levee where it fails" : "the levee");
  const least = polygon ? 3 : 2;
  const dragWords = polygon ? "drag a box" : "drag a straight line";
  const box = (a, b) => [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
  overlay.setAttribute("aria-label", `Draw ${what}: tap its points and press Done, or ${dragWords}. Escape cancels.`);
  overlay.tabIndex = 0;
  const tip = document.createElement("p");
  tip.className = "fs-hint";
  const bar = document.createElement("div");
  bar.className = "fsk-bar";
  const undo = Object.assign(document.createElement("button"), { type: "button", className: "btn small ghost", textContent: "Undo point" });
  const done = Object.assign(document.createElement("button"), { type: "button", className: "btn small primary", textContent: "Done" });
  const cancel = Object.assign(document.createElement("button"), { type: "button", className: "btn small", textContent: "Cancel" });
  bar.append(undo, cancel, done);
  overlay.append(tip, bar);
  wrap.appendChild(overlay);
  const pts = [];
  let down = null, hover = null, lastTap = 0;
  const hint = () => {
    tip.textContent = pts.length === 0 ? `Draw ${what}: tap where it starts, or ${dragWords}.`
      : pts.length < least ? "Tap the next point." : `${pts.length} points. Tap more, or Done.`;
    done.disabled = pts.length < least;
    undo.disabled = pts.length === 0;
  };
  const ring = (list) => (polygon && list.length >= 3 ? [...list, list[0]] : list);
  const paint = () => drawSketches(shown.sketches, { selected: null, draft: hover && pts.length ? ring([...pts, hover]) : pts.length ? ring([...pts]) : null });
  return new Promise((resolve) => {
    const finish = (line) => {
      if (!drawing) return;
      drawing = null;
      overlay.classList.add("done");
      overlay.replaceChildren();
      setTimeout(() => overlay.remove(), 400);   // the click that ends a tap lands on it, not on the map
      drawSketches(shown.sketches, { selected: shown.selected, draft: null });
      resolve(line && line.length >= least ? line : null);
    };
    drawing = { finish };
    hint();
    overlay.focus({ preventScroll: true });
    const stop = (e) => e.stopPropagation();
    for (const b of [undo, done, cancel]) b.addEventListener("pointerdown", stop);
    undo.addEventListener("click", (e) => { e.stopPropagation(); pts.pop(); hint(); paint(); });
    done.addEventListener("click", (e) => { e.stopPropagation(); finish(pts); });
    cancel.addEventListener("click", (e) => { e.stopPropagation(); finish(null); });
    overlay.addEventListener("keydown", (e) => {
      if (e.key === "Escape") finish(null);
      else if (e.key === "Enter" && pts.length >= least) finish(pts);
      else if (e.key === "Backspace") { pts.pop(); hint(); paint(); }
    });
    overlay.addEventListener("pointerdown", (e) => {
      if (e.button > 0 || e.target !== overlay) return;
      e.preventDefault();
      overlay.setPointerCapture(e.pointerId);
      down = { x: e.clientX, y: e.clientY, ll: lngLatAt(e) };
    });
    overlay.addEventListener("pointermove", (e) => {
      if (!drawing) return;
      const ll = lngLatAt(e);
      if (down && !pts.length && Math.hypot(e.clientX - down.x, e.clientY - down.y) > TAP_PX) {
        drawSketches(shown.sketches, { draft: polygon ? ring(box(down.ll, ll)) : [down.ll, ll] });
        return;
      }
      if (pts.length && e.pointerType === "mouse") { hover = ll; paint(); }
    });
    overlay.addEventListener("pointerup", (e) => {
      if (!drawing || !down) return;
      const ll = lngLatAt(e);
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > TAP_PX;
      const from = down.ll;
      down = null;
      if (moved && !pts.length) { finish(polygon ? box(from, ll) : [from, ll]); return; }
      const now = Date.now();
      if (now - lastTap < 350 && pts.length >= least) { finish(pts); return; }   // a double tap ends the line
      lastTap = now;
      pts.push(ll);
      hint();
      paint();
    });
    overlay.addEventListener("pointercancel", () => { down = null; });
    for (const type of ["touchstart", "touchend", "click", "mousedown", "mouseup", "dblclick"]) {
      overlay.addEventListener(type, (e) => {
        if (bar.contains(e.target)) return;   // the buttons keep their own taps
        if (e.target === overlay || type !== "click") e.preventDefault();
        e.stopPropagation();
      },
        { passive: false });
    }
    overlay.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = map.getContainer().getBoundingClientRect();
      const around = map.unproject([e.clientX - r.left, e.clientY - r.top]);
      map.easeTo({ zoom: map.getZoom() - Math.sign(e.deltaY) * 0.5, around, duration: 150 });
    }, { passive: false });
    announce(`Drawing ${what}: tap its points and press Done, or ${dragWords}.`);
  });
}
