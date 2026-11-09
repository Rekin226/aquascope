// AI on the map (#561): apply map actions, and the action log with one-click undo.
//
// An action is one of aquascope.map_commands' types (fly_to, set_time, set_layer, focus_status, set_basemap,
// highlight_river, draw_area, add_pin, route, paint, draw_storm and the rest), checked by the package before it gets here and with its place names
// already looked up. Whoever wrote it (the keyless rules, the on-device model, the reader's own key, an
// agent over WebMCP, or Explain and Scout through addPin), it is applied the way the reader's own click
// would apply it, and listed in the log on the map with an undo for each and one for all.
//
// map-actions-core.js keeps the log (pure, tested in node); this module touches the map and the page.

import { $, actions, escapeHtml, setTime, state, timeState } from "./core.js?v=__BUILD__";
import { DEFAULT_CENTER, fitWorldZoom, map } from "./map.js?v=__BUILD__";
import {
  ActionLog, BY_LABEL, LAYER_CONTROLS, agoWords, checkPin, inkOn, noteBody, placeNotes, slotOf,
} from "./map-actions-core.js?v=__BUILD__";
import { getStatusFocus, setStatusFocus } from "./status-layer.js?v=__BUILD__";
import { SEVERITY as REVIEW_SEVERITY } from "./mesh-review-core.js?v=__BUILD__";
import { clearRiverNetwork, lightRiverNetwork } from "./river-map.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";

export const log = new ActionLog();

const AREA_SRC = "ai-areas";
const PIN_PATH = '<path d="M12 31s10-10.2 10-18A10 10 0 0 0 2 13c0 7.8 10 18 10 18Z"/>';
const PIN_SVG = `<svg viewBox="0 0 24 32" aria-hidden="true">${PIN_PATH}<circle cx="12" cy="12.5" r="3.6"/></svg>`;
// A ranked pin (Scout, #563) carries its number where the dot was.
const rankedPin = (n) => `<svg viewBox="0 0 24 32" aria-hidden="true">${PIN_PATH}` +
  `<text x="12" y="13" text-anchor="middle" dominant-baseline="central">${n}</text></svg>`;

let river = null;              // the river this module lit: { lat, lon, river_id, direction, label } or null
const areas = new Map();       // entry id -> { bbox, label, marker }
const pinMarkers = new Map();  // entry id -> maplibregl.Marker

// ── reading and writing each slot ───────────────────────────────────────────

function layerOn(id) {
  if (id === "globe") return Boolean(state.globe);
  const el = $(LAYER_CONTROLS[id]);
  return Boolean(el && el.checked);
}

// Flip a layer exactly as its control would, so the URL, the credits, the legends and the time bar follow.
function setLayer(id, on) {
  if (id === "globe") {
    if (Boolean(state.globe) !== Boolean(on)) $("btn-globe").click();
    return;
  }
  const el = $(LAYER_CONTROLS[id]);
  if (!el) throw new Error(`the ${id} layer is not on this page`);
  if (el.checked === Boolean(on)) return;
  el.checked = Boolean(on);
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

/** A layer on or off as its control would (Replay an event, #659, turns the rain and Floods past on per stop). */
export function setMapLayer(id, on) { setLayer(id, on); }

function readView() {
  const c = map.getCenter();
  return { center: [c.lng, c.lat], zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch() };
}

function stopPlaying() {
  if (state.playing) $("tb-play").click();
}

function writeTime(v) {
  stopPlaying();
  setTime({ date: v.date, step: v.step, range: v.range || null }, { source: "agent" });
  if (v.playing) $("tb-play").click();
}

function lightRiver(spec) {
  river = spec;
  const reach = { river_id: spec.river_id, lat: spec.lat, lon: spec.lon };
  lightRiverNetwork(reach, () => call("river", { op: "network", args: reach }).then((net) => {
    if (!net || spec.direction === "both") return net;
    // One way only: what drains here, or the way to the sea.
    return spec.direction === "upstream" ? { ...net, downstream: null } : { ...net, upstream: null };
  }));
}

const SLOT_WRITERS = {
  view: (v) => map.easeTo({ ...v, duration: 800 }),
  time: writeTime,
  "status-focus": (v) => setStatusFocus(v || []),
  basemap: (v) => actions.setBasemap(v),
  river: (v) => { if (v) lightRiver(v); else { river = null; clearRiverNetwork(); } },
};

/**
 * A module that paints its own slot (What if, #565) hands the writer that puts a value back on the map, so the
 * log's undo can restore what was there before: `fn(value)` with null clears it.
 */
export function registerSlotWriter(slot, fn) {
  SLOT_WRITERS[slot] = fn;
}

/**
 * Log an action a module has already applied itself (its slot needs a writer: registerSlotWriter), with what it
 * replaced, so the log lists it and undo puts `before` back. Returns the entry.
 */
export function logAction({ action, before = null, after = null, label = "", by = "rules", detail = "", command = "" }) {
  const slot = slotOf(action);
  if (!slot || !SLOT_WRITERS[slot]) throw new Error(`no writer for ${action && action.type}`);
  const entry = log.add({ action, slot, before, after, label: label || action.type, by, detail, command });
  renderLog();
  return entry;
}

/** The value in force on a slot now: the newest live entry's `after`, or null. */
export function slotValue(slot) {
  const live = log.live().filter((e) => e.slot === slot);
  return live.length ? live[live.length - 1].after : null;
}

function writeSlot(slot, value) {
  if (SLOT_WRITERS[slot]) return SLOT_WRITERS[slot](value);
  if (slot.startsWith("layer:")) return setLayer(slot.slice(6), value);
  if (slot.startsWith("area:")) return value ? drawArea(Number(slot.slice(5)), value) : removeArea(Number(slot.slice(5)));
  if (slot.startsWith("pin:")) return value ? drawPin(Number(slot.slice(4)), value) : removePin(Number(slot.slice(4)));
  return undefined;
}

// ── drawn areas ─────────────────────────────────────────────────────────────

function areaFeatures() {
  return [...areas].map(([id, a]) => {
    const [w, s, e, n] = a.bbox;
    return {
      type: "Feature", id, properties: { id, label: a.label || "" },
      geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
    };
  });
}

// Under the gauges (the layer order is basemap, status, floods past, rivers, floods ahead, gauges, card).
function ensureAreaLayers() {
  if (!state.mapOk || !map) return false;
  if (!map.getSource(AREA_SRC)) {
    map.addSource(AREA_SRC, { type: "geojson", data: { type: "FeatureCollection", features: areaFeatures() } });
    const before = ["gauge-heat", "clusters", "points"].find((id) => map.getLayer(id));
    map.addLayer({ id: "ai-area-fill", type: "fill", source: AREA_SRC, paint: { "fill-color": "#0b6bb8", "fill-opacity": 0.05 } }, before);
    map.addLayer({
      id: "ai-area-line", type: "line", source: AREA_SRC,
      paint: { "line-color": "#0b6bb8", "line-width": 1.6, "line-dasharray": [3, 2], "line-opacity": 0.9 },
    }, before);
  }
  return true;
}

function refreshAreas() {
  if (!ensureAreaLayers()) return;
  map.getSource(AREA_SRC).setData({ type: "FeatureCollection", features: areaFeatures() });
}

function drawArea(id, area) {
  const old = areas.get(id);
  if (old && old.marker) old.marker.remove();
  const tag = document.createElement("span");
  tag.className = "ma-area-tag";
  tag.textContent = area.label || "Area";
  const marker = new maplibregl.Marker({ element: tag, anchor: "bottom-left", offset: [2, -2] })
    .setLngLat([area.bbox[0], area.bbox[3]]).addTo(map);
  areas.set(id, { ...area, marker });
  refreshAreas();
}

function removeArea(id) {
  const a = areas.get(id);
  if (a && a.marker) a.marker.remove();
  areas.delete(id);
  refreshAreas();
}

// ── pins ────────────────────────────────────────────────────────────────────

function factLine(f) {
  const v = typeof f.value === "number" ? f.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(f.value);
  return `${f.label}: ${v}${f.unit ? ` ${f.unit}` : ""}${f.when ? ` (${f.when})` : ""}`;
}

function openPinCard(id) {
  const e = log.get(id);
  if (!e || e.undone) return;
  const p = e.after;
  const [first, ...rest] = p.facts || [];
  const v = first && (typeof first.value === "number" ? first.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(first.value));
  const isNote = p.kind === "note";
  const isReview = p.kind === "review";
  // A note's card leads with its words; the facts they do not already give follow (#562). A scout pin's reason
  // already says its numbers (all formatted by the package); its facts stay in pins() (#563).
  const note = isNote ? noteBody(p) : p.rank ? p.text : [p.text, ...rest.map(factLine)].filter(Boolean).join(" · ");
  const sev = isReview ? REVIEW_SEVERITY[p.severity] || REVIEW_SEVERITY.info : null;
  // The pin's card is what was asked for now: the log folds so the two do not stack (its count reopens it).
  if (logOpen) setLogOpen(false);
  actions.openMapCard({
    id: `pin:${id}`, lngLat: [p.lon, p.lat], lift: isNote ? 14 : 30,
    what: isReview ? `Review ${p.rank}, ${p.check} · ${BY_LABEL[e.by] || e.by}`
      : !isNote && p.rank ? `Scout ${p.rank}${p.kind ? `, ${p.kind}` : ""} · ${BY_LABEL[e.by] || e.by}` : `Note, ${BY_LABEL[e.by] || e.by}`,
    ...(sev ? { status: { text: sev.words, color: sev.colour } } : {}),
    whatIcon: `<span class="ma-pin-mini" aria-hidden="true">${PIN_SVG}</span>`,
    title: p.title, sub: [p.place || `${p.lat.toFixed(3)}, ${p.lon.toFixed(3)}`, p.when].filter(Boolean).join(" · "),
    figure: !isNote && first ? { value: v, unit: first.unit || "", label: first.label } : null,
    note, credit: p.credit || p.source || "",
    buttons: [{ id: "unpin", label: isNote ? "Remove note" : isReview ? "Remove this finding" : "Remove pin", title: "Undo this pin",
      onClick: () => { undoEntry(id); actions.closeMapCard(); } }],
  });
  // Its callout steps back while the card says the same thing in full.
  for (const [nid, n] of noteMarkers) n.el.classList.toggle("carded", nid === id);
}

// A note (Explain, #562): a numbered dot at the place and a short card beside it, on whichever side keeps the
// cards off each other and off the legend, the box and the time bar (map-actions-core.js placeNotes).
const noteMarkers = new Map();  // entry id -> { marker, el, pin }

function drawNote(id, pin) {
  const el = document.createElement("div");
  el.className = "ma-note side-right";
  if (pin.tone) { el.style.setProperty("--tone", pin.tone); el.style.setProperty("--tone-ink", inkOn(pin.tone)); }
  el.innerHTML = `<button type="button" class="ma-note-dot" aria-label="Note ${pin.n || ""}: ${escapeHtml(pin.title)}">` +
    `${pin.n || ""}</button><div class="ma-note-card"><b class="ma-note-title">${escapeHtml(pin.title)}</b>` +
    (pin.text ? `<span class="ma-note-text">${escapeHtml(pin.text)}</span>` : "") +
    (pin.source ? `<span class="ma-note-src">${escapeHtml(pin.source)}</span>` : "") + "</div>";
  el.addEventListener("click", (ev) => { ev.stopPropagation(); openPinCard(id); });
  const marker = new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([pin.lon, pin.lat]).addTo(map);
  noteMarkers.set(id, { marker, el, pin });
  pinMarkers.set(id, marker);
  placeNoteCards();
}

// The furniture a card should not sit on, in the map's own pixels.
function furniture(frame) {
  const out = [];
  for (const sel of ["#map-legends", "#map-ask .ma-bar", "#ma-status", "#ma-log", "#timebar", "#map-tools", "#map-card",
    ".map-lead", ".maplibregl-ctrl-bottom-left", ".maplibregl-ctrl-bottom-right"]) {
    const el = document.querySelector(sel);
    if (!el || el.hidden) continue;
    const r = el.getBoundingClientRect();
    if (r.width && r.height) out.push({ x: r.left - frame.left, y: r.top - frame.top, w: r.width, h: r.height });
  }
  return out;
}

let placing = 0;
function placeNoteCards() {
  if (placing || !noteMarkers.size) return;
  placing = requestAnimationFrame(() => {
    placing = 0;
    const frame = map.getContainer().getBoundingClientRect();
    const list = [...noteMarkers.values()];
    const points = list.map((n) => map.project([n.pin.lon, n.pin.lat]));
    const sizes = list.map((n) => {
      const c = n.el.querySelector(".ma-note-card");
      return { w: (c && c.offsetWidth) || 200, h: (c && c.offsetHeight) || 60 };
    });
    const sides = placeNotes(points, sizes, { w: frame.width, h: frame.height }, furniture(frame));
    list.forEach((n, i) => {
      for (const side of ["right", "left", "top", "bottom"]) n.el.classList.toggle(`side-${side}`, side === sides[i]);
    });
  });
}

// A pin from the log: brought into view if it is off screen, then its card opened.
function showPin(id) {
  const e = log.get(id);
  if (!e || e.undone) return;
  const p = e.after;
  const b = map.getBounds();
  if (b && !b.contains([p.lon, p.lat])) map.easeTo({ center: [p.lon, p.lat], duration: 700 });
  openPinCard(id);
}

function drawPin(id, pin) {
  if (!state.mapOk || !map) return;
  const old = pinMarkers.get(id);
  if (old) old.remove();
  noteMarkers.delete(id);
  if (pin.kind === "note") { drawNote(id, pin); return; }
  const el = document.createElement("button");
  el.type = "button";
  el.className = pin.rank ? "ma-pin ranked" : "ma-pin";
  el.innerHTML = pin.rank ? rankedPin(pin.rank) : PIN_SVG;
  el.setAttribute("aria-label", pin.rank ? `${pin.kind === "review" ? "Review" : "Scout"} ${pin.rank}: ${pin.title}` : `Note: ${pin.title}`);
  if (pin.kind === "review") {
    // A review pin (#564) wears its severity: red a problem, amber worth a look, blue for information.
    el.classList.add("review", `sev-${pin.severity}`);
    el.style.setProperty("--pin", (REVIEW_SEVERITY[pin.severity] || REVIEW_SEVERITY.info).colour);
  }
  el.title = pin.title;
  el.addEventListener("click", (ev) => { ev.stopPropagation(); openPinCard(id); });
  const marker = new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([pin.lon, pin.lat]).addTo(map);
  pinMarkers.set(id, marker);
}

function removePin(id) {
  const m = pinMarkers.get(id);
  if (m) m.remove();
  pinMarkers.delete(id);
  if (noteMarkers.delete(id)) placeNoteCards();
}

// ── applying one action ─────────────────────────────────────────────────────

// "From here" (#656): the reach the selection snapped to, else none (the action says to click a river first).
function hereReach() {
  const r = actions.selectedReach ? actions.selectedReach() : null;
  return r && r.river_id !== null && r.river_id !== undefined ? { river_id: r.river_id, lat: r.lat, lon: r.lon } : null;
}

async function snapRiver(a) {
  const snap = await callLight("river", { op: "snap", args: { lat: a.lat, lon: a.lon, prefer: "main", max_distance_m: 5000 } },
    { priority: 2 });
  const reach = snap && snap.snapped ? { river_id: snap.river_id, lat: snap.snap_lat, lon: snap.snap_lon }
    : snap && snap.nearest ? { river_id: snap.nearest.river_id, lat: snap.nearest.lat, lon: snap.nearest.lon } : null;
  if (!reach) throw new Error(`no mapped river within 5 km of ${a.label || "that point"}`);
  return reach;
}

// What the floating pieces cover, so a place lands in the part of the map you can see: the box and its log
// at the top, the legend stack on the left of a wide screen, the time bar at the bottom.
export function cameraPadding() {
  const frame = map.getContainer().getBoundingClientRect();
  const pad = { top: 70, bottom: 120, left: 60, right: 60 };
  // The box itself, not its log: the log is short-lived and folds away, the place is what you asked for.
  const bar = document.querySelector("#map-ask .ma-bar");
  if (bar && frame.width > 640) {
    const r = bar.getBoundingClientRect();
    if (r.height) pad.top = Math.max(pad.top, Math.round(r.bottom - frame.top) + 24);
  }
  const legends = $("map-legends");
  if (legends && frame.width > 1100) {
    const r = legends.getBoundingClientRect();
    if (r.width && r.height) pad.left = Math.max(pad.left, Math.round(r.right - frame.left) + 16);
  }
  // Never more than the map can give: a small window keeps half its width and height for the place.
  pad.top = Math.min(pad.top, frame.height * 0.45);
  pad.left = Math.min(pad.left, frame.width * 0.35);
  return pad;
}

function camera(a) {
  const pad = cameraPadding();
  if (a.region === "world" && !a.bbox) {
    map.easeTo({ center: DEFAULT_CENTER, zoom: fitWorldZoom($("map"), { globe: state.globe }), bearing: 0, pitch: 0, duration: 900 });
  } else if (a.bbox) {
    const [w, s, e, n] = a.bbox;
    map.fitBounds([[w, s], [e, n]], { padding: pad, maxZoom: 11, duration: 900 });
  } else if (a.center) {
    map.flyTo({ center: [a.center[1], a.center[0]], zoom: a.zoom ?? 8, duration: 1100 });
  } else if (a.zoom_by) {
    map.easeTo({ zoom: Math.max(0, Math.min(18, map.getZoom() + a.zoom_by)), duration: 500 });
  }
}

/**
 * Apply one checked, resolved action and log it. `by` says who asked ("rules", "device", "key", "agent",
 * "api"), `label` is the line the log shows (the package's describe_action), `command` the words typed.
 * Returns the log entry; throws when the action cannot run here.
 */
export async function applyAction(a, { by = "rules", label = "", command = "" } = {}) {
  if (!state.mapOk || !map) throw new Error("the map is not ready");
  const id = log.peekId();
  const slot = slotOf(a, id);
  if (!slot) throw new Error(`unknown action ${a && a.type}`);
  let before = null, after = null;
  switch (a.type) {
    case "fly_to":
      before = readView();
      camera(a);
      after = { target: a.bbox || a.center || a.region || a.zoom_by };
      break;
    case "set_time": {
      before = timeState();
      if (state.playing) stopPlaying();
      const patch = {};
      if (a.date) patch.date = a.date;
      if (a.step) patch.step = a.step;
      if (a.range) patch.range = a.range;
      if (Object.keys(patch).length) setTime(patch, { source: "agent" });
      if (a.playing === true) $("tb-play").click();
      after = timeState();
      break;
    }
    case "set_layer":
      before = layerOn(a.layer);
      setLayer(a.layer, a.on);
      after = a.on;
      break;
    case "focus_status":
      before = getStatusFocus();
      after = setStatusFocus(a.classes || []);
      break;
    case "set_basemap":
      before = state.basemap;
      actions.setBasemap(a.basemap);
      after = a.basemap;
      break;
    case "highlight_river": {
      before = river;
      const reach = await snapRiver(a);
      after = { ...reach, direction: a.direction || "both", label: a.label || "" };
      lightRiver(after);
      break;
    }
    case "route": {
      // Route it (#656): route-map.js runs it, draws it and logs it with what it replaced; nothing to log here.
      if (!actions.routeFlood) throw new Error("routing is not on this page");
      const reach = a.at === "here" ? hereReach() : await snapRiver(a);
      if (!reach) throw new Error("select a river first (click one), or name a place");
      const args = { duration_h: a.duration_h || 24, shape: a.shape || "triangle",
        ...(a.peak_cms ? { peak_cms: a.peak_cms } : { return_period: a.return_period || 10 }) };
      if (actions.openRoute) actions.openRoute({ ...reach, name: a.label || a.where || "" });
      await actions.routeFlood({ ...reach, name: a.label || a.where || "" }, args, { by, command, label });
      return log.live().slice(-1)[0];
    }
    case "flood_reach": {
      // Flood it (#658): floodsim-map.js runs it (the grid and the flows from aquascope.floodsim in the worker),
      // draws it and logs it with the run it replaced; nothing to log here.
      if (!actions.floodReach) throw new Error("Flood it is not on this page");
      const reach = a.at === "here" ? hereReach() : await snapRiver(a);
      if (!reach) throw new Error("select a river first (click one), or name a place");
      await actions.floodReach({ ...reach, name: a.label || a.where || "" }, a, { by, command, label });
      return log.live().slice(-1)[0];
    }
    case "replay": {
      // Replay an event (#659): the code lists Floods past's floods there that month and replay.js assembles and
      // plays the one picked (the most reported unless the action says another); it logs it; nothing to log here.
      if (!actions.startReplay) throw new Error("Replay is not on this page");
      const at = a.at === "here" ? hereReach() : { lat: a.lat, lon: a.lon };
      if (!at || at.lat === undefined || at.lat === null) throw new Error("select a river first (click one), or name a place");
      const res = await call("replay", { op: "candidates", args: { lat: at.lat, lon: at.lon, month: a.month, radius_km: 40 } });
      const list = (res && res.candidates) || [];
      if (!list.length) throw new Error((res && res.note) || `no flood in Floods past near there in ${a.month}`);
      const c = list[Math.min(list.length, a.pick || 1) - 1];
      const story = await actions.startReplay({ ...(at.river_id ? { river_id: at.river_id } : {}), lat: c.lat, lon: c.lon,
        start: c.start, end: c.end, reports: c.reports }, { by, label, command });
      if (!story) throw new Error("the replay could not be assembled");
      return log.live().slice(-1)[0];
    }
    case "draw_levee":
    case "draw_breach": {
      // Levees and breaches (#663): floodsim-map.js draws the line (or lets the reader draw it), asks the package for
      // the crest when the words gave a freeboard, logs the line and floods again; nothing to log here.
      if (!actions.sketchFlood) throw new Error("Flood it is not on this page");
      const s = await actions.sketchFlood(a, { by, command, label });
      if (!s) throw new Error("no line was drawn");
      return log.live().slice(-1)[0];
    }
    case "drop_well": {
      // Drop a well (#660): well-map.js adds it to the wells on the map, computes the cone (aquascope.sketch.drawdown
      // in the worker) and logs the wells with what they replaced; nothing to log here.
      if (!actions.dropWell) throw new Error("Drop a well is not on this page");
      await actions.dropWell(a, { by, command, label });
      return log.live().slice(-1)[0];
    }
    case "change_landuse": {
      // Change land use (#661): landuse-map.js takes the area (the one given, the one on the map, or one the reader
      // draws), computes the change (aquascope.sketch.landuse in the worker) and logs it; nothing to log here.
      if (!actions.changeLanduse) throw new Error("Change land use is not on this page");
      const res = await actions.changeLanduse(a, { by, command, label });
      if (!res) throw new Error("no area was drawn");
      return log.live().slice(-1)[0];
    }
    case "draw_storm": {
      // Draw a storm (#670): storm-map.js takes the area (the words', the one on the map, or one the reader draws),
      // logs it, routes the storm and logs the run as a route; nothing to log here.
      if (!actions.drawStorm) throw new Error("Draw a storm is not on this page");
      const res = await actions.drawStorm(a, { by, command, label });
      if (!res) throw new Error("no storm was routed (no area drawn, or no river under it)");
      return log.live().slice(-1)[0];
    }
    case "draw_reservoir": {
      // Draw a reservoir (#662): reservoir-map.js takes the lake (the words', the one on the map, or one the reader
      // draws), logs it, routes the flood through it and logs the run as a route; nothing to log here.
      if (!actions.drawReservoir) throw new Error("Draw a reservoir is not on this page");
      const res = await actions.drawReservoir(a, { by, command, label });
      if (!res) throw new Error("no flood was routed through a reservoir (no lake drawn, or no river under it)");
      return log.live().slice(-1)[0];
    }
    case "draw_area":
      after = { bbox: a.bbox, label: a.label || a.where || "" };
      drawArea(id, after);
      break;
    case "add_pin": {
      const c = map.getCenter();
      after = checkPin(a.at === "center" ? { ...a, lat: c.lat, lon: c.lng } : a);
      drawPin(id, after);
      break;
    }
    case "paint": {
      // Paint the globe (#657): paint.js runs it (the numbers come from aquascope.paint in the worker); the box
      // is framed first so the reader sees what is painted. Its slot writer puts the painting before it back.
      if (!actions.paintMap) throw new Error("painting is not available on this page");
      before = slotValue("paint");
      if (a.bbox) {
        const [w, s, e, n] = a.bbox;
        const moved = new Promise((resolve) => { map.once("moveend", resolve); setTimeout(resolve, 2500); });
        map.fitBounds([[w, s], [e, n]], { padding: cameraPadding(), maxZoom: 9, duration: 700 });
        await moved;
      }
      after = await actions.paintMap({ question: a.question, bbox: a.bbox || null, min_years: a.min_years },
        { log: false });
      if (a.label) after = { ...after, label: a.label };
      break;
    }
    default:
      throw new Error(`unknown action ${a.type}`);
  }
  return log.add({ action: a, slot, before, after, label: label || a.type, by, detail: a.where || "", command });
}

/**
 * Apply several actions in order (each one logged). `said` holds the log line for each. Returns
 * { applied: [entries], failed: [{ action, error }] }: one that fails does not stop the rest.
 */
export async function applyActions(list, { by = "rules", said = [], command = "" } = {}) {
  const applied = [], failed = [];
  for (const [i, a] of (list || []).entries()) {
    try {
      applied.push(await applyAction(a, { by, label: said[i] || "", command }));
    } catch (err) {
      failed.push({ action: a, error: err.message });
    }
  }
  return { applied, failed };
}

// ── undo ────────────────────────────────────────────────────────────────────

export function undoEntry(id) {
  const res = log.undo(id);
  if (!res) return false;
  if (res.restore) writeSlot(res.restore.slot, res.restore.value);
  announce(`Undone: ${res.entry.label}`);
  return true;
}

export function undoLast() {
  const live = log.live();
  return live.length ? undoEntry(live[live.length - 1].id) : false;
}

export function undoAll() {
  const restores = log.undoAll();
  for (const r of restores) {
    try { writeSlot(r.slot, r.value); } catch (err) { console.warn("undo all:", r.slot, err && err.message); }
  }
  if (restores.length) announce("Every map action undone");
  return restores.length;
}

// ── pins as an API (Explain #562 and Scout #563 use it) ─────────────────────

/**
 * Drop a pin with a note: addPin({ lat, lon, title, text, facts: [{label, value, unit}], source }).
 * It goes in the action log like any other action (undo removes it). Returns the pin's id.
 */
export function addPin(pin, { by = "api", label = "" } = {}) {
  if (!state.mapOk || !map) throw new Error("the map is not ready");
  const p = checkPin(pin);
  const id = log.peekId();
  drawPin(id, p);
  const entry = log.add({ action: { type: "add_pin", ...p }, slot: `pin:${id}`, before: null, after: p,
    label: label || `Pin: ${p.title}`, by, detail: p.source || "" });
  return entry.id;
}

/**
 * Refine a pin already on the map (its title or note, as a place name arrives), without a new log entry: the
 * pin stays where it is and keeps its undo. Returns false for a pin that is gone.
 */
export function updatePin(id, pin, { label = "" } = {}) {
  const e = log.get(id);
  if (!e || e.undone || !e.action || e.action.type !== "add_pin") return false;
  const p = checkPin({ ...e.after, ...pin, lat: e.after.lat, lon: e.after.lon });
  e.after = p;
  if (label) e.label = label;
  const m = pinMarkers.get(id);
  if (m) {
    const el = m.getElement();
    el.title = p.title;
    el.setAttribute("aria-label", p.rank ? `Scout ${p.rank}: ${p.title}` : `Note: ${p.title}`);
  }
  log.emit();
  return true;
}

/** The pins on the map, oldest first: { id, lat, lon, title, text, facts, source, by, at }. */
export const pins = () => log.pins();

export function removePinById(id) { return undoEntry(id); }

// ── the log on the map ──────────────────────────────────────────────────────

const ICON = {
  fly_to: '<path d="M4 12h12M12 6l6 6-6 6"/>',
  set_time: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
  set_layer: '<path d="M12 4 3 9l9 5 9-5-9-5Z"/><path d="m3 14 9 5 9-5"/>',
  focus_status: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>',
  set_basemap: '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M4 15l5-5 5 5 2-2 4 4"/>',
  highlight_river: '<path d="M4 18c3 0 3-4 6-4s3 4 6 4 2-3 4-3"/><path d="M4 11c3 0 3-4 6-4s3 4 6 4 2-3 4-3"/>',
  draw_area: '<rect x="4.5" y="5.5" width="15" height="13" rx="1" stroke-dasharray="3 2"/>',
  add_pin: '<path d="M12 21s7-7 7-12a7 7 0 0 0-14 0c0 5 7 12 7 12Z"/><circle cx="12" cy="9" r="2.2"/>',
  paint: '<path d="M19.5 3.5c-3.2 1.2-7.4 5.1-9.6 8.3l1.9 1.9c3.2-2.2 7.1-6.4 8.3-9.6Z"/><path d="M9.4 12.6c-1.9 0-3.3 1.4-3.5 3.3-.2 1.7-.9 2.6-2.4 3.1 3.4 1.4 7.3.3 7.7-3.2.1-1.2-.5-2.3-1.8-3.2Z"/>',
  what_if: '<path d="M4 17c3 0 3-3 6-3s3 3 6 3 2-2 4-2"/><path d="M9 4.5a3 3 0 1 1 3 3.5v1.6"/><path d="M12 11.6h.01"/>',
  route: '<path d="M3 16c3 0 3-3 6-3s3 3 6 3 3-3 6-3"/><path d="M6 9c1.5-3 3-4 4.5-4S13 7 14 9s2.5 3 4 2"/>',
  flood_reach: '<path d="M3 15c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2 1 3 1"/><path d="M3 19c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2 1 3 1"/><path d="M12 3v7M9 7l3 3 3-3"/>',
  draw_levee: '<path d="M3 17h18"/><path d="M5 17l3-6h8l3 6"/><path d="M3 21c2.5 0 2.5-1.5 5-1.5S10.5 21 13 21"/>',
  change_landuse: '<path d="M3 20h18"/><path d="M5 20v-6l4-3 4 3v6"/><path d="M13 20v-9l4-3 4 3v9"/>',
  drop_well: '<path d="M12 3v13"/><path d="M9 6h6"/><path d="M3 12c3 0 5 1 7 5h4c2-4 4-5 7-5"/><path d="M3 20h18"/>',
  draw_storm: '<path d="M7 15a4 4 0 1 1 .9-7.9A5.5 5.5 0 0 1 18.5 9 3 3 0 0 1 18 15Z"/><path d="M8 18l-1 2.5M12 18l-1 2.5M16 18l-1 2.5"/>',
  draw_reservoir: '<path d="M3 18c2 0 2-1.5 4-1.5s2 1.5 4 1.5 2-1.5 4-1.5 2 1.5 4 1.5"/><path d="M4 13c4-1 6-4 8-8 2 4 4 7 8 8"/><path d="M12 5v8"/>',
  draw_breach: '<path d="M3 17h7M14 17h7"/><path d="M5 17l3-6h2M14 11h2l3 6"/><path d="M11 13l1 4 1-4"/>',
};

const icon = (type) => `<svg class="ma-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ` +
  `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[type] || ICON.fly_to}</svg>`;

let logOpen = false;

export function setLogOpen(open) {
  logOpen = Boolean(open);
  renderLog();
}

export function renderLog() {
  const box = $("ma-log"), toggle = $("ma-log-toggle");
  if (!box) return;
  const live = log.live();
  const n = live.length;
  if (toggle) {
    toggle.hidden = n === 0;
    toggle.textContent = `${n} map action${n === 1 ? "" : "s"}`;
    toggle.setAttribute("aria-expanded", logOpen && n ? "true" : "false");
  }
  box.hidden = !logOpen || n === 0;
  if (box.hidden) return;
  const now = Date.now();
  const rows = live.slice().reverse().map((e) => {
    const by = BY_LABEL[e.by] || e.by;
    const title = [e.command ? `“${e.command}”` : "", e.detail, agoWords(e.at, now)].filter(Boolean).join(" · ");
    const pin = e.action && e.action.type === "add_pin";
    const label = pin
      ? `<button type="button" class="ma-label ma-open-pin" data-pin="${e.id}" title="${escapeHtml(title)}">${escapeHtml(e.label)}</button>`
      : `<span class="ma-label" title="${escapeHtml(title)}">${escapeHtml(e.label)}</span>`;
    return `<li class="ma-row" data-id="${e.id}">${icon(e.action && e.action.type)}${label}` +
      `<span class="ma-by ma-by-${escapeHtml(e.by)}">${escapeHtml(by)}</span>` +
      `<button type="button" class="ma-undo" data-undo="${e.id}" aria-label="Undo: ${escapeHtml(e.label)}" title="Undo">` +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>' +
      "</button></li>";
  }).join("");
  box.innerHTML = `<header class="ma-log-head"><b>Map actions</b><span class="muted">newest first</span>` +
    `<button type="button" class="link-btn ma-undo-all" data-undo-all>Undo all</button></header>` +
    `<ol class="ma-rows">${rows}</ol>`;
}

export function initMapActions() {
  const box = $("ma-log");
  if (box) {
    box.addEventListener("click", (e) => {
      const one = e.target.closest("[data-undo]");
      if (one) { undoEntry(Number(one.dataset.undo)); return; }
      const pin = e.target.closest("[data-pin]");
      if (pin) { showPin(Number(pin.dataset.pin)); return; }
      if (e.target.closest("[data-undo-all]")) undoAll();
    });
  }
  const toggle = $("ma-log-toggle");
  if (toggle) toggle.addEventListener("click", () => setLogOpen(!logOpen));
  log.subscribe(() => renderLog());
  // A basemap swap replaces the style: put the drawn areas back on the new one.
  // A closed map card gives a note back its callout.
  const card = $("map-card");
  if (card && globalThis.MutationObserver) {
    new MutationObserver(() => {
      if (card.hidden) for (const n of noteMarkers.values()) n.el.classList.remove("carded");
    }).observe(card, { attributes: true, attributeFilter: ["hidden"] });
  }
  if (state.mapOk && map) {
    map.on("style.load", () => { if (areas.size) refreshAreas(); });
    // The notes' cards find their sides again once the map has moved (#562).
    map.on("moveend", placeNoteCards);
    map.on("resize", placeNoteCards);
  }
  actions.addPin = addPin;
  actions.mapPins = pins;
  actions.removePin = removePinById;
  actions.applyMapActions = applyActions;
  actions.undoMapAction = undoEntry;
  globalThis.__aq.mapActions = { log, addPin, pins, applyActions, undoEntry, undoAll };
  renderLog();
}
