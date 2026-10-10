// AI on the map (#561): apply map actions, and the action log with one-click undo.
//
// An action is one of aquascope.map_commands' types (fly_to, set_time, set_layer, focus_status, set_basemap,
// highlight_river, draw_area, add_pin), checked by the package before it gets here and with its place names
// already looked up. Whoever wrote it (the keyless rules, the on-device model, the reader's own key, an
// agent over WebMCP, or Explain and Scout through addPin), it is applied the way the reader's own click
// would apply it, and listed in the log on the map with an undo for each and one for all.
//
// map-actions-core.js keeps the log (pure, tested in node); this module touches the map and the page.

import { $, actions, escapeHtml, setTime, state, timeState } from "./core.js?v=__BUILD__";
import { DEFAULT_CENTER, fitWorldZoom, map } from "./map.js?v=__BUILD__";
import { ActionLog, BY_LABEL, LAYER_CONTROLS, agoWords, checkPin, slotOf } from "./map-actions-core.js?v=__BUILD__";
import { getStatusFocus, setStatusFocus } from "./status-layer.js?v=__BUILD__";
import { clearRiverNetwork, lightRiverNetwork } from "./river-map.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";

export const log = new ActionLog();

const AREA_SRC = "ai-areas";
const PIN_SVG = '<svg viewBox="0 0 24 32" aria-hidden="true"><path d="M12 31s10-10.2 10-18A10 10 0 0 0 2 13c0 7.8 10 18 10 18Z"/>' +
  '<circle cx="12" cy="12.5" r="3.6"/></svg>';

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
  return `${f.label}: ${v}${f.unit ? ` ${f.unit}` : ""}`;
}

function openPinCard(id) {
  const e = log.get(id);
  if (!e || e.undone) return;
  const p = e.after;
  const [first, ...rest] = p.facts || [];
  const v = first && (typeof first.value === "number" ? first.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(first.value));
  const note = [p.text, ...rest.map(factLine)].filter(Boolean).join(" · ");
  actions.openMapCard({
    id: `pin:${id}`, lngLat: [p.lon, p.lat], lift: 30,
    what: `Note, ${BY_LABEL[e.by] || e.by}`, whatIcon: `<span class="ma-pin-mini" aria-hidden="true">${PIN_SVG}</span>`,
    title: p.title, sub: `${p.lat.toFixed(3)}, ${p.lon.toFixed(3)}`,
    figure: first ? { value: v, unit: first.unit || "", label: first.label } : null,
    note, credit: p.source || "",
    buttons: [{ id: "unpin", label: "Remove pin", title: "Undo this pin", onClick: () => { undoEntry(id); actions.closeMapCard(); } }],
  });
}

function drawPin(id, pin) {
  if (!state.mapOk || !map) return;
  const old = pinMarkers.get(id);
  if (old) old.remove();
  const el = document.createElement("button");
  el.type = "button";
  el.className = "ma-pin";
  el.innerHTML = PIN_SVG;
  el.setAttribute("aria-label", `Note: ${pin.title}`);
  el.title = pin.title;
  el.addEventListener("click", (ev) => { ev.stopPropagation(); openPinCard(id); });
  const marker = new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([pin.lon, pin.lat]).addTo(map);
  pinMarkers.set(id, marker);
}

function removePin(id) {
  const m = pinMarkers.get(id);
  if (m) m.remove();
  pinMarkers.delete(id);
}

// ── applying one action ─────────────────────────────────────────────────────

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
function cameraPadding() {
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
    return `<li class="ma-row" data-id="${e.id}">${icon(e.action && e.action.type)}` +
      `<span class="ma-label" title="${escapeHtml(title)}">${escapeHtml(e.label)}</span>` +
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
      if (e.target.closest("[data-undo-all]")) undoAll();
    });
  }
  const toggle = $("ma-log-toggle");
  if (toggle) toggle.addEventListener("click", () => setLogOpen(!logOpen));
  log.subscribe(() => renderLog());
  // A basemap swap replaces the style: put the drawn areas back on the new one.
  if (state.mapOk && map) map.on("style.load", () => { if (areas.size) refreshAreas(); });
  actions.addPin = addPin;
  actions.mapPins = pins;
  actions.removePin = removePinById;
  actions.applyMapActions = applyActions;
  actions.undoMapAction = undoEntry;
  globalThis.__aq.mapActions = { log, addPin, pins, applyActions, undoEntry, undoAll };
  renderLog();
}
