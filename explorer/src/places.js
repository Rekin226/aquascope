// My places: gauges the reader has starred, kept in this browser's
// localStorage. Every read and write is wrapped, so a private window, blocked
// storage or a full quota leaves the page working with an in-memory list.
//
// The list opens as its own surface from the header ("My places"); ticking
// two to five places and pressing Compare hands them to compare.js.

import { $, actions, escapeHtml, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { initTabs, setStatusEl, showSurface } from "./shell.js?v=__BUILD__";
import { MAX_COMPARE, MIN_COMPARE, runCompare } from "./compare.js?v=__BUILD__";

export const PLACES_KEY = "aquascope-places";
export const MAX_PLACES = 50;

// ── the store (pure, so node:test can drive it with a fake storage) ─────────

let memory = [];
let writeFailed = false;

function storage() {
  try { return globalThis.localStorage || null; } catch { return null; }
}

function valid(p) {
  return p && typeof p === "object" && typeof p.source === "string" && p.source
    && typeof p.station_id === "string" && p.station_id;
}

// Storage wins when it answers; the in-memory copy is what is left when it
// does not (denied, corrupt, or a write that failed).
export function readPlaces(store = storage()) {
  if (store) {
    try {
      const raw = store.getItem(PLACES_KEY);
      const list = raw ? JSON.parse(raw) : (writeFailed ? null : []);
      if (Array.isArray(list)) memory = list.filter(valid);
    } catch { /* storage denied or corrupt: keep the memory copy */ }
  }
  return memory.slice();
}

export function writePlaces(list, store = storage()) {
  memory = list.filter(valid).slice(0, MAX_PLACES);
  if (store) {
    try {
      store.setItem(PLACES_KEY, JSON.stringify(memory));
      writeFailed = false;
    } catch { writeFailed = true; /* quota or denied: the memory copy carries on */ }
  }
  return memory.slice();
}

export const placeKey = (p) => `${p.source}/${p.station_id}`;

export function hasPlace(key, store = storage()) {
  return readPlaces(store).some((p) => placeKey(p) === key);
}

// A place keeps only what the list needs to draw itself without the catalog.
export function placeFromStation(r) {
  return {
    source: String(r.source), station_id: String(r.station_id), name: r.name || r.station_id,
    lat: Number.isFinite(Number(r.lat)) ? Number(r.lat) : null,
    lon: Number.isFinite(Number(r.lon)) ? Number(r.lon) : null,
    period_start: r.period_start || null,
    added: new Date().toISOString().slice(0, 10),
  };
}

export function addPlace(place, store = storage()) {
  const list = readPlaces(store);
  if (list.some((p) => placeKey(p) === placeKey(place))) return list;
  return writePlaces([place, ...list], store);
}

export function removePlace(key, store = storage()) {
  return writePlaces(readPlaces(store).filter((p) => placeKey(p) !== key), store);
}

export function togglePlace(place, store = storage()) {
  const key = placeKey(place);
  return hasPlace(key, store) ? removePlace(key, store) : addPlace(place, store);
}

/** Only for tests: forget the in-memory fallback. */
export function _resetMemory() { memory = []; writeFailed = false; }

// ── the page ────────────────────────────────────────────────────────────────

const checked = new Set();
let returnTo = "panel-empty";

function say(text, kind = "info") { setStatusEl($("places-status"), text, kind); }

export function syncPlaceButton() {
  const btn = $("btn-place");
  if (!btn) return;
  const on = Boolean(state.selected) && hasPlace(stationKey(state.selected));
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.textContent = on ? "★ Saved" : "☆ Save";
  btn.title = on ? "Remove this gauge from My places" : "Add this gauge to My places";
  syncPlacesCount();
}

function syncPlacesCount() {
  const n = readPlaces().length;
  const badge = $("places-count");
  if (badge) { badge.textContent = n ? String(n) : ""; badge.hidden = !n; }
}

function syncCompareButton() {
  const n = checked.size;
  const btn = $("btn-compare");
  btn.disabled = n < MIN_COMPARE || n > MAX_COMPARE;
  btn.textContent = n ? `Compare ${n}` : "Compare";
  $("places-hint").textContent = n > MAX_COMPARE
    ? `Up to ${MAX_COMPARE} at a time; untick ${n - MAX_COMPARE}.`
    : `Tick ${MIN_COMPARE} to ${MAX_COMPARE} places to compare them.`;
}

export function renderPlaces() {
  const list = readPlaces();
  const ul = $("places-list");
  for (const k of [...checked]) if (!list.some((p) => placeKey(p) === k)) checked.delete(k);
  $("places-empty").hidden = list.length > 0;
  ul.replaceChildren();
  for (const p of list) {
    const key = placeKey(p);
    const st = sourceStyle(p.source);
    const li = document.createElement("li");
    li.className = "place-row";
    li.innerHTML =
      `<label class="place-pick"><input type="checkbox" ${checked.has(key) ? "checked" : ""} ` +
      `aria-label="Compare ${escapeHtml(p.name)}"></label>` +
      `<button type="button" class="place-open link" title="Open this gauge">` +
      `<span class="badge" style="background:${st.color}">${escapeHtml(st.label)}</span> ` +
      `<span class="place-name">${escapeHtml(p.name)}</span> <span class="muted">${escapeHtml(p.station_id)}</span></button>` +
      `<button type="button" class="btn tiny place-remove" title="Remove from My places" ` +
      `aria-label="Remove ${escapeHtml(p.name)}">Remove</button>`;
    li.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) checked.add(key); else checked.delete(key);
      syncCompareButton();
    });
    li.querySelector(".place-open").addEventListener("click", () => openPlace(p));
    li.querySelector(".place-remove").addEventListener("click", () => {
      removePlace(key);
      checked.delete(key);
      renderPlaces();
      syncPlaceButton();
    });
    ul.appendChild(li);
  }
  syncCompareButton();
  syncPlacesCount();
}

function openPlace(p) {
  const key = placeKey(p);
  if (!state.byKey.has(key)) {
    say("This gauge is not in the current catalog, so it cannot be opened. You can remove it.", "warn");
    return;
  }
  actions.selectStation(key, { fly: true });
}

export function openPlaces() {
  if (!$("panel-places").hidden) return;
  for (const id of ["panel-station", "panel-point", "panel-workbench", "panel-empty"]) {
    const el = $(id);
    if (el && !el.hidden) { returnTo = id; break; }
  }
  say("");
  renderPlaces();
  showSurface("panel-places");
}

function closePlaces() {
  showSurface(returnTo && $(returnTo) ? returnTo : "panel-empty");
}

export function initPlaces() {
  initTabs($("panel-places"));
  // Plotly sizes a figure drawn in a hidden pane wrongly; nudge it when shown.
  $("panel-places").addEventListener("tabchange", () => {
    for (const id of ["plot-cmp-hydro", "plot-cmp-fdc", "plot-cmp-ffa"]) {
      const el = $(id);
      if (el && el.offsetParent !== null && el.data && globalThis.Plotly) Plotly.Plots.resize(el);
    }
  });
  $("btn-places").addEventListener("click", () => openPlaces());
  $("places-back").addEventListener("click", closePlaces);
  $("btn-place").addEventListener("click", () => {
    if (!state.selected) return;
    togglePlace(placeFromStation(state.selected));
    syncPlaceButton();
  });
  $("btn-compare").addEventListener("click", () => {
    const picked = readPlaces().filter((p) => checked.has(placeKey(p)));
    runCompare(picked);
  });
  // Another tab changed the list: follow it.
  globalThis.addEventListener?.("storage", (e) => {
    if (e.key !== PLACES_KEY) return;
    syncPlaceButton();
    if (!$("panel-places").hidden) renderPlaces();
  });
  syncPlacesCount();
}
