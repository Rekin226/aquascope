// Drop a well (#660), on the map. Drop a well in the sketchpad's menu drops a pumping well where you click; the drawdown
// cone is drawn round it as contour lines at round levels (5 cm out to the deepest) over a fill, recomputed as you
// drag a well, change its rate, the aquifer or the days. Several wells add (superposition). The cone grows on the
// time bar ("Play the cone growing"). A card on a well gives its drawdown, its own share and the others', and, when
// your MODFLOW model is under it, the model's cell and layer and what its well cell would read. Every change is an
// entry in the action log (slot "wells"); Ask the map's drop_well action runs through dropWell here.
//
// Every number comes from aquascope.sketch.drawdown in the worker (the same function as `aquascope well` and the
// MCP tool drop_well); well-core.js decides the colours and the words.

import { $, actions, clickLayers, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { releaseModelClock, setModelStep, useModelClock } from "./time-ui.js?v=__BUILD__";
import { modflowContext } from "./groundwater-view.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { registerSketchTool } from "./sketch-menu.js?v=__BUILD__";
import {
  DEFAULTS, LEVEL_COLORS, MAX_WELLS, SPANS, TYPICAL, addWell, contourFeatures, corners, daysWords, distWords,
  editWell, fillPixels, frameLabel, fromSlider, headline, logLabel, mapArgs, metres, modflowCellAt, modflowWords,
  peacemanRadius, rateWords, sWords, toSlider, wellsValue,
} from "./well-core.js?v=__BUILD__";

const SRC = "well-cone";
const FILL = "well-fill";
const L = { fill: "well-fill", casing: "well-cone-casing", edge: "well-cone-edge", line: "well-cone-line",
  label: "well-cone-label" };
const CREDIT = "Theis (1935) with superposition, computed in your browser by aquascope.sketch.drawdown; checked " +
  "against Fetter (2001)'s worked example and USGS Professional Paper 708's W(u) table.";
const EMPTY = { type: "FeatureCollection", features: [] };

let value = null;          // the wells on the map: { wells: [{ id, lon, lat, rate }], T, S, days } (the slot's value)
let result = null;         // the package's answer for `value`
let frame = null;          // the frame of result.frames shown, or null for the days asked
let markers = new Map();   // well id -> maplibregl.Marker
let placing = null;        // { finish } while the reader places a well
let inflight = null, queued = null;
let dragging = false;      // a well is being dragged: the live answers draw, the others wait
let hooked = false;

const panel = () => $("well");
const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

// ── the layers ──────────────────────────────────────────────────────────────

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getLayer(L.line)) return true;
  try {
    if (!map.getSource(SRC)) map.addSource(SRC, { type: "geojson", data: EMPTY });
    map.addLayer({ id: L.casing, type: "line", source: SRC, layout: { "line-join": "round", "line-cap": "round" },
      paint: { "line-color": "rgba(255,255,255,0.85)", "line-width": ["case", ["==", ["get", "edge"], 1], 3.2, 4] } });
    // the 5 cm line, the cone's edge, dashed; the others solid
    map.addLayer({ id: L.edge, type: "line", source: SRC, filter: ["==", ["get", "edge"], 1],
      paint: { "line-color": ["get", "color"], "line-width": 1.6, "line-dasharray": [3, 2] } });
    map.addLayer({ id: L.line, type: "line", source: SRC, filter: ["==", ["get", "edge"], 0],
      layout: { "line-join": "round", "line-cap": "round" },
      paint: { "line-color": ["get", "color"], "line-width": 2.2 } });
    map.addLayer({ id: L.label, type: "symbol", source: SRC,
      layout: { "symbol-placement": "line", "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"],
        "text-size": 11.5, "symbol-spacing": 260, "text-keep-upright": true, "text-allow-overlap": false },
      paint: { "text-color": ["get", "color"], "text-halo-color": "rgba(255,255,255,0.95)", "text-halo-width": 2 } });
    hook();
    return true;
  } catch (err) {
    console.info("well layers unavailable:", err && err.message);
    return false;
  }
}

function hook() {
  if (hooked) return;
  hooked = true;
  for (const id of [L.line, L.edge, L.casing]) {
    clickLayers.add(id);
    map.on("click", id, onLineClick);
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
  map.on("style.load", () => { if (value) { ensureLayers(); drawCone(); } });
}

function setFill(res) {
  const img = res && res.image;
  if (!img) {
    if (map.getLayer(FILL)) map.removeLayer(FILL);
    if (map.getSource(FILL)) map.removeSource(FILL);
    return;
  }
  const { px, width, height } = fillPixels(img);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(px, width, height), 0, 0);
  const url = canvas.toDataURL("image/png");
  const coords = corners(img.bbox);
  const src = map.getSource(FILL);
  if (src && src.updateImage) src.updateImage({ url, coordinates: coords });
  else {
    map.addSource(FILL, { type: "image", url, coordinates: coords });
    map.addLayer({ id: L.fill, type: "raster", source: FILL,
      paint: { "raster-opacity": 0.75, "raster-fade-duration": 0, "raster-resampling": "linear" } }, L.casing);
  }
}

function drawCone() {
  if (!ensureLayers()) return;
  const src = map.getSource(SRC);
  if (!result || !value || !value.wells.length) {
    if (src) src.setData(EMPTY);
    setFill(null);
    return;
  }
  const fr = frame !== null && result.frames ? result.frames[frame] : null;
  src.setData(contourFeatures(fr ? fr.contours : result.contours));
  setFill(fr && fr.image ? fr : result);
}

// ── the wells' markers ──────────────────────────────────────────────────────

function markerEl(w) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "wl-marker";
  el.title = "Drag to move the well; click for its numbers";
  el.setAttribute("aria-label", `Well ${w.id}, pumping ${rateWords(w.rate)}. Drag to move it.`);
  el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/>' +
    '<path d="M12 5v9M8.5 10.5 12 14l3.5-3.5"/></svg>' + `<span>${w.id}</span>`;
  return el;
}

function placeMarkers() {
  const keep = new Set((value ? value.wells : []).map((w) => w.id));
  for (const [id, m] of markers) if (!keep.has(id)) { m.remove(); markers.delete(id); }
  if (!value || typeof maplibregl === "undefined" || !state.mapOk) return;
  for (const w of value.wells) {
    let m = markers.get(w.id);
    if (!m) {
      const el = markerEl(w);
      // eslint-disable-next-line no-undef
      m = new maplibregl.Marker({ element: el, anchor: "center", draggable: true }).setLngLat([w.lon, w.lat]).addTo(map);
      let moved = false;
      m.on("dragstart", () => { moved = true; dragging = true; });
      m.on("drag", () => {
        const p = m.getLngLat();
        compute(editWell(value, w.id, { lon: p.lng, lat: p.lat }), { live: true });
      });
      m.on("dragend", () => {
        dragging = false;
        const p = m.getLngLat();
        change(editWell(value, w.id, { lon: round6(p.lng), lat: round6(p.lat) }));
        setTimeout(() => { moved = false; }, 0);
      });
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!moved) void wellCard(w.id);
      });
      markers.set(w.id, m);
    } else {
      m.setLngLat([w.lon, w.lat]);
      m.getElement().setAttribute("aria-label", `Well ${w.id}, pumping ${rateWords(w.rate)}. Drag to move it.`);
    }
  }
}

const round6 = (x) => Math.round(x * 1e6) / 1e6;

// ── computing: the worker, one request at a time, the newest wins ───────────

async function compute(v, { live = false } = {}) {
  if (!v || !v.wells.length) { result = null; drawCone(); renderPanel(); return null; }
  const args = mapArgs(v, { frames: !live });
  if (inflight) { queued = { v, live }; return inflight; }
  const run = async () => {
    say(live ? "" : `<span class="spinner" aria-hidden="true"></span>${escapeHtml(state.workerReady ? "Computing the cone" : "Starting the engine (once), then computing")}…`);
    try {
      const res = await call("well", { op: "map", args });
      if (!res || !res.ok) throw new Error((res && res.error) || "no answer");
      if ((live && dragging) || (!live && v === value)) {
        result = live ? { ...res, frames: null, image: null } : res;
        frame = null;
        drawCone();
        if (!live) { takeClock(); renderPanel(); say(""); }
        else renderPanel({ live: true });
      }
      return res;
    } catch (err) {
      say(escapeHtml(`Could not compute the cone: ${err.message}`), "error");
      return null;
    }
  };
  inflight = run();
  const out = await inflight;
  inflight = null;
  if (queued) {
    const q = queued;
    queued = null;
    // a live request is stale once the drag has ended and the full one queued behind it
    if (!q.live || q.v === value || !value) return compute(q.v, { live: q.live });
    return compute(value);
  }
  return out;
}

/** Make `next` the wells on the map, log it with what it replaced, and compute it. */
function change(next, { by = "rules", label = "", command = "" } = {}) {
  const before = value;
  paint(next);
  logAction({ action: { type: "drop_well", wells: next ? next.wells.length : 0 }, before, after: next,
    label: label || logLabel(before, next), by, command });
  return next;
}

/** Put a value on the map (from a change, or from undo) and compute it. */
function paint(next) {
  value = next && next.wells.length ? wellsValue(next) : null;
  placeMarkers();
  if (!value) {
    result = null;
    frame = null;
    drawCone();
    releaseModelClock();
    renderPanel();
    refreshLegend("wells");
    return;
  }
  syncInputs();
  renderPanel();
  void compute(value).then(() => refreshLegend("wells"));
  openLegendRow("wells", true);
}

// ── the time bar: the cone growing ──────────────────────────────────────────

function takeClock() {
  const fr = result && result.frames;
  if (!fr || fr.length < 2) return;
  useModelClock({
    times: fr.map((f) => f.days), index: fr.length - 1, label: (i) => frameLabel(fr[i].days),
    summary: "The drawdown cone: days since the wells started pumping.",
    onStep: (i) => { frame = i === fr.length - 1 ? null : i; drawCone(); refreshLegend("wells"); },
    onExit: () => { frame = null; drawCone(); },
  });
}

function playCone() {
  if (!result || !result.frames) return;
  takeClock();
  if (globalThis.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  void setModelStep(0).then(() => {
    const btn = $("tb-play");
    if (btn && btn.getAttribute("aria-pressed") !== "true") btn.click();
  });
}

// ── placing a well: click the map ───────────────────────────────────────────

/** Let the reader click the map to drop a well; resolves with [lon, lat] or null (Escape, Cancel). */
function startPlacing() {
  if (!state.mapOk || !map) return Promise.resolve(null);
  if (placing) placing.finish(null);
  const wrap = map.getContainer();
  const overlay = document.createElement("div");
  overlay.className = "fs-draw wl-place";
  overlay.setAttribute("role", "application");
  overlay.setAttribute("aria-label", "Click the map where the well goes. Escape cancels.");
  overlay.tabIndex = 0;
  const tip = Object.assign(document.createElement("p"), { className: "fs-hint",
    textContent: "Click the map where the well goes." });
  const bar = document.createElement("div");
  bar.className = "fsk-bar";
  const cancel = Object.assign(document.createElement("button"), { type: "button", className: "btn small", textContent: "Cancel" });
  bar.append(cancel);
  overlay.append(tip, bar);
  wrap.appendChild(overlay);
  return new Promise((resolve) => {
    let down = null;
    const finish = (ll) => {
      if (!placing) return;
      placing = null;
      overlay.classList.add("done");
      overlay.replaceChildren();
      setTimeout(() => overlay.remove(), 400);
      resolve(ll);
    };
    placing = { finish };
    overlay.focus({ preventScroll: true });
    cancel.addEventListener("pointerdown", (e) => e.stopPropagation());
    cancel.addEventListener("click", (e) => { e.stopPropagation(); finish(null); });
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") finish(null); });
    overlay.addEventListener("pointerdown", (e) => {
      if (e.button > 0 || e.target !== overlay) return;
      down = { x: e.clientX, y: e.clientY };
    });
    overlay.addEventListener("pointerup", (e) => {
      if (!down || e.target !== overlay) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > 8;
      down = null;
      if (moved) return;
      const r = wrap.getBoundingClientRect();
      const ll = map.unproject([e.clientX - r.left, e.clientY - r.top]);
      finish([round6(ll.lng), round6(ll.lat)]);
    });
    // the click that ends a tap lands on the overlay, not on the map (as the levee tool does)
    for (const type of ["touchstart", "touchend", "click", "mousedown", "mouseup", "dblclick"]) {
      overlay.addEventListener(type, (e) => {
        if (bar.contains(e.target)) return;
        if (e.target === overlay || type !== "click") e.preventDefault();
        e.stopPropagation();
      }, { passive: false });
    }
    overlay.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = wrap.getBoundingClientRect();
      const around = map.unproject([e.clientX - r.left, e.clientY - r.top]);
      map.easeTo({ zoom: map.getZoom() - Math.sign(e.deltaY) * 0.5, around, duration: 150 });
    }, { passive: false });
    announce("Click the map where the well goes. Escape cancels.");
  });
}

async function placeOne() {
  if (value && value.wells.length >= MAX_WELLS) { say(`At most ${MAX_WELLS} wells.`, "warn"); return; }
  const ll = await startPlacing();
  if (!ll) return;
  // the first well's rate is the one typed; a later one starts at the last well's
  const typed = Number($("wl-rate-new") && $("wl-rate-new").value);
  const rate = value && value.wells.length ? value.wells[value.wells.length - 1].rate : typed > 0 ? typed : DEFAULTS.rate;
  const base = value || { wells: [], ...readAquifer() };
  change(addWell(base, { lon: ll[0], lat: ll[1], rate }));
  openPanel();
  frameCone();
}

function frameCone() {
  if (!value || !state.mapOk) return;
  const once = () => {
    if (!result || !result.bbox) return;
    const [w, s, e, n] = result.bbox;
    // the cone's inner lines: frame the 0.2 m line's reach at most, so the wells are not lost in a 50 km box
    const lons = value.wells.map((x) => x.lon), lats = value.wells.map((x) => x.lat);
    const r = Math.min(result.reach_m, Math.max(...result.wells.map((x) => x.reach_m))) / 111195;
    const k = Math.cos((lats[0] * Math.PI) / 180) || 1;
    const box = [[Math.max(w, Math.min(...lons) - r / k), Math.max(s, Math.min(...lats) - r)],
      [Math.min(e, Math.max(...lons) + r / k), Math.min(n, Math.max(...lats) + r)]];
    const p = panel();
    const pad = { top: 90, left: 40, right: 40, bottom: 120 };
    if (p && !p.hidden) {
      if (narrow()) pad.bottom = Math.min(p.offsetHeight + 24, map.getContainer().clientHeight * 0.55);
      else pad.right = p.offsetWidth + 40;
    }
    try { map.fitBounds(box, { padding: pad, maxZoom: 13, duration: 700 }); } catch { /* not ready */ }
  };
  if (result) once();
  else setTimeout(() => { if (result) once(); }, 1500);
}

// ── the panel ───────────────────────────────────────────────────────────────

function readAquifer() {
  return {
    T: fromSlider(Number($("wl-t").value), SPANS.T),
    S: fromSlider(Number($("wl-s").value), SPANS.S),
    days: fromSlider(Number($("wl-days").value), SPANS.days),
  };
}

function syncInputs() {
  const v = value || { ...DEFAULTS };
  $("wl-t").value = toSlider(v.T, SPANS.T);
  $("wl-s").value = toSlider(v.S, SPANS.S);
  $("wl-days").value = toSlider(v.days, SPANS.days);
  syncOutputs();
}

function syncOutputs() {
  const a = readAquifer();
  $("wl-t-out").textContent = `${a.T.toLocaleString("en-US")} m²/day`;
  $("wl-s-out").textContent = sWords(a.S);
  $("wl-days-out").textContent = daysWords(a.days);
  const odd = [];
  if (a.T < TYPICAL.T[0] || a.T > TYPICAL.T[1]) odd.push("transmissivity");
  if (a.S < TYPICAL.S[0] || a.S > TYPICAL.S[1]) odd.push("storativity");
  const note = $("wl-odd");
  note.hidden = !odd.length;
  note.textContent = odd.length ? `Outside the typical range of a confined aquifer: ${odd.join(" and ")}.` : "";
}

function say(html, kind = "") {
  const el = $("wl-status");
  if (!el) return;
  el.className = `wi-status ${kind}`;
  el.innerHTML = html || "";
  el.hidden = !html;
}

function wellRow(w, res) {
  const r = res && res.wells ? res.wells.find((x) => x.name === `Well ${w.id}`) : null;
  return `<li data-id="${w.id}"><span class="wl-n" aria-hidden="true">${w.id}</span>` +
    `<label class="wl-q"><span class="visually-hidden">Well ${w.id}'s rate, m³/day</span>` +
    `<input type="number" min="1" step="any" inputmode="decimal" value="${w.rate}" data-id="${w.id}" /> m³/day</label>` +
    `<span class="wl-at">${r ? `${escapeHtml(metres(r.at_well_m))} m` : "…"}</span>` +
    `<button type="button" class="wl-del" data-id="${w.id}" aria-label="Remove well ${w.id}" title="Remove this well">×</button></li>`;
}

function renderPanel({ live = false } = {}) {
  const p = panel();
  if (!p) return;
  const list = $("wl-wells");
  const has = Boolean(value && value.wells.length);
  $("wl-empty").hidden = has;
  $("wl-result").hidden = !has;
  if (!has) { list.innerHTML = ""; return; }
  if (!live || !list.children.length) list.innerHTML = value.wells.map((w) => wellRow(w, result)).join("");
  else {
    for (const li of list.querySelectorAll("li")) {
      const r = result && result.wells.find((x) => x.name === `Well ${li.dataset.id}`);
      const at = li.querySelector(".wl-at");
      if (r && at) at.textContent = `${metres(r.at_well_m)} m`;
    }
  }
  $("wl-head").textContent = result ? headline(result) : "Computing the cone…";
  $("wl-add").disabled = value.wells.length >= MAX_WELLS;
  if (result && !live) {
    $("wl-model").textContent = `${result.method} ${result.assumptions} ${result.check}`;
    $("wl-src").textContent = `Typical ranges: ${result.ranges_source}`;
  }
}

function openPanel() {
  const p = panel();
  for (const id of ["whatif", "route"]) { const o = $(id); if (o) o.hidden = true; }   // one panel at a time
  p.hidden = false;
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });
  syncInputs();
  renderPanel();
}

function closePanel() { panel().hidden = true; }

// ── cards ───────────────────────────────────────────────────────────────────

// On a phone the card is a bottom sheet under the panel: the panel steps aside while a card is open.
function cardOpened() {
  if (!narrow()) return;
  const p = panel();
  if (p && !p.hidden) p.classList.add("carded");
}

function modflowAt(lon, lat) {
  const ctx = modflowContext && modflowContext();
  if (!ctx || !ctx.visible) return null;
  const cell = modflowCellAt(ctx.rings, ctx.offsets, lon, lat);
  return cell ? { ctx, cell } : null;
}

async function wellCard(id) {
  if (!value || !actions.openMapCard) return;
  const w = value.wells.find((x) => x.id === id);
  if (!w) return;
  const r = result && result.wells.find((x) => x.name === `Well ${id}`);
  const mf = modflowAt(w.lon, w.lat);
  let note = r && value.wells.length > 1
    ? `Its own ${metres(r.own_m)} m and ${metres(r.others_m)} m from the other wells.` : "";
  cardOpened();
  const card = actions.openMapCard({
    id: `well:${id}`, lngLat: [w.lon, w.lat], lift: 16, what: "Pumping well, modelled", title: `Well ${id}`,
    sub: `${rateWords(w.rate)} for ${daysWords(value.days)}`,
    status: { text: r ? `The 5 cm line reaches ${distWords(r.reach_m)} from this well alone.` : "Computing…", color: LEVEL_COLORS[4][1] },
    figure: r ? { value: metres(r.at_well_m), unit: " m", label: `drawdown at the well's face (${result.well_radius_m} m radius, no well loss)` } : { pending: true },
    note: [note, mf ? modflowWords(mf.ctx, mf.cell) : ""].filter(Boolean).join(" "), credit: CREDIT,
    buttons: [{ id: "wl-card-del", label: "Remove this well", title: "Take this well off the map (undoable)",
      onClick: () => { change(editWell(value, id, null)); if (actions.closeMapCard) actions.closeMapCard(); } }],
  });
  if (mf && card && card.update) {
    // what the model's well cell would read: the cone at Peaceman's radius, from the package
    const rr = peacemanRadius(mf.cell);
    const k = Math.cos((w.lat * Math.PI) / 180) || 1;
    try {
      const at = await call("well", { op: "point", args: { lon: w.lon + rr / (111195 * k), lat: w.lat,
        wells: mapArgs(value).wells, days: value.days, transmissivity: value.T, storativity: value.S } });
      if (at && at.drawdown_m !== undefined) {
        note = [note, modflowWords(mf.ctx, mf.cell, at.drawdown_m)].filter(Boolean).join(" ");
        card.update({ note });
      }
    } catch { /* the card stands without it */ }
  }
}

async function onLineClick(e) {
  if (!result || !e.features || !e.features.length || !actions.openMapCard) return;
  e.preventDefault();
  if (e.originalEvent) e.originalEvent.stopPropagation();
  const f = e.features[0];
  const level = Number(f.properties.level);
  const when = frame !== null && result.frames ? result.frames[frame].days : value.days;
  cardOpened();
  const card = actions.openMapCard({
    id: `well-line:${level}`, lngLat: [e.lngLat.lng, e.lngLat.lat], what: "Drawdown, modelled",
    title: `${f.properties.label} line`, sub: `after ${daysWords(when)}`,
    status: { text: `Along this line the water stands ${f.properties.label} lower than before pumping.`, color: f.properties.color },
    figure: { pending: true }, credit: CREDIT,
  });
  try {
    const at = await call("well", { op: "point", args: { lon: e.lngLat.lng, lat: e.lngLat.lat, wells: mapArgs(value).wells,
      days: when, transmissivity: value.T, storativity: value.S } });
    if (card && card.update && at && at.wells) {
      const near = at.wells.reduce((a, b) => (b.distance_m < a.distance_m ? b : a));
      card.update({ figure: { value: metres(at.drawdown_m), unit: " m", label: "drawdown here" },
        note: `${distWords(near.distance_m)} from ${near.name}.` });
    }
  } catch { /* the card stands without the number */ }
}

// ── the legend row ──────────────────────────────────────────────────────────

function legendBody() {
  if (!result) return "";
  const lv = (result.levels || []).map((l) => `<i style="background:${LEVEL_COLORS.find(([x]) => x >= l - 1e-9)[1]}"></i>`).join("");
  const when = frame !== null && result.frames ? frameLabel(result.frames[frame].days) : `after ${daysWords(result.days)}`;
  return `<p class="ml-when">Drawdown ${escapeHtml(when)}: lines at ${escapeHtml((result.levels || []).map((l) => `${l}`).join(", "))} m, ` +
    "the 5 cm line dashed.</p>" +
    `<p class="rk-row"><span class="wl-ramp">${lv}</span> 5 cm to ${escapeHtml(String(result.levels[result.levels.length - 1]))} m</p>` +
    `<p class="ml-src">Theis (1935), T ${escapeHtml(String(result.transmissivity_m2_day))} m²/day, S ${escapeHtml(sWords(result.storativity))}. Modelled, computed in your browser.</p>`;
}

function registerRow() {
  registerLegendRow({
    id: "wells", title: "Drawdown",
    shown: () => Boolean(result && value),
    mark: () => `<span class="wl-ramp">${LEVEL_COLORS.slice(0, 6).filter((_, i) => i % 2 === 0).map(([, c]) => `<i style="background:${c}"></i>`).join("")}</span>`,
    summary: () => (value ? `${value.wells.length} well${value.wells.length === 1 ? "" : "s"}, ${daysWords(value.days)}` : ""),
    body: legendBody,
    act: (name) => { if (name === "clear" && value) change(null, { label: "Take the wells off the map" }); },
  });
}

// ── the AI's drop_well action (map-actions.js) ──────────────────────────────

/**
 * Drop a well for an action: { lat, lon } or { at: "center" }, rate_m3_day, and when given days, transmissivity,
 * storativity. Logged with `by`. Returns the new value.
 */
export async function dropWell(a, { by = "rules", label = "", command = "" } = {}) {
  let lon = a.lon, lat = a.lat;
  if (a.at === "center" || lon === undefined || lat === undefined) {
    const c = map.getCenter();
    lon = round6(c.lng);
    lat = round6(c.lat);
  }
  const base = value || { wells: [], ...DEFAULTS };
  const next = addWell({ ...base, ...(a.days ? { days: a.days } : {}), ...(a.transmissivity ? { T: a.transmissivity } : {}),
    ...(a.storativity ? { S: a.storativity } : {}) }, { lon, lat, rate: a.rate_m3_day || DEFAULTS.rate });
  change(next, { by, label, command });
  openPanel();
  frameCone();
  return next;
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initWell() {
  const p = panel();
  if (!p) return;
  registerSlotWriter("wells", (v) => paint(v || null));
  registerRow();
  registerSketchTool({
    id: "well", label: "Drop a well", hint: "The drawdown cone of a pumping well (Theis)",
    icon: '<path d="M12 3v13"/><path d="M9 6h6"/><path d="M3 12c3 0 5 1 7 5h4c2-4 4-5 7-5"/><path d="M3 20h18"/>',
    run: () => { if (placing) placing.finish(null); else void placeOne(); },
  });
  $("wl-add").addEventListener("click", () => void placeOne());
  $("wl-place").addEventListener("click", () => void placeOne());
  for (const id of ["wl-t", "wl-s", "wl-days"]) {
    $(id).addEventListener("input", syncOutputs);
    $(id).addEventListener("change", () => {
      if (!value) return;
      change(wellsValue({ ...value, ...readAquifer() }));
    });
  }
  $("wl-wells").addEventListener("change", (e) => {
    const el = e.target;
    if (!el.matches("input[data-id]") || !value) return;
    const q = Number(el.value);
    const was = value.wells.find((w) => w.id === Number(el.dataset.id));
    if (!was) return;
    if (!(q > 0)) { el.value = was.rate; return; }
    if (q === was.rate) return;
    change(editWell(value, Number(el.dataset.id), { rate: q }));
  });
  $("wl-wells").addEventListener("click", (e) => {
    const b = e.target.closest(".wl-del");
    if (b && value) change(editWell(value, Number(b.dataset.id), null));
  });
  $("wl-play").addEventListener("click", playCone);
  $("wl-clear").addEventListener("click", () => { if (value) change(null, { label: "Take the wells off the map" }); });
  p.querySelector(".wl-close").addEventListener("click", closePanel);
  p.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); closePanel(); } });
  const mc = $("map-card");
  if (mc && typeof MutationObserver !== "undefined") {
    new MutationObserver(() => { if (mc.hidden) p.classList.remove("carded"); })
      .observe(mc, { attributes: true, attributeFilter: ["hidden"] });
  }
  syncInputs();
  actions.dropWell = dropWell;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.well = { value: () => value, result: () => result, drop: dropWell, place: placeOne, card: wellCard,
    move: (id, lon, lat) => change(editWell(value, id, { lon, lat })), drag: (id, lon, lat) => { dragging = true; return compute(editWell(value, id, { lon, lat }), { live: true }); },
    dragEnd: () => { dragging = false; },
    play: playCone, slot: () => slotValue("wells") };
}
