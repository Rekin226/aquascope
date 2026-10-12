// Watch (#521): the watch list kept in this browser, as pure functions so
// node:test can drive them with a fake storage. Every read and write is
// wrapped: a private window, blocked storage or a full quota leaves the page
// working with an in-memory copy. What changed is never worked out here; that
// is aquascope.watch.watch_digest in the worker. This module only keeps the
// list, the thresholds and what each item looked like when last seen.

export const WATCH_KEY = "aquascope-watch";
export const MAX_WATCH = 30;
// The return periods the threshold menu offers (the forecast archive carries the gauge's flow for each).
export const RETURN_PERIODS = [2, 5, 10, 25, 50, 100];

const empty = () => ({ v: 1, items: [], seen: {}, lastVisit: null });
let memory = empty();
let writeFailed = false;

function storage() {
  try { return globalThis.localStorage || null; } catch { return null; }
}

const round4 = (x) => Math.round(Number(x) * 1e4) / 1e4;
const coord = (x) => { const v = round4(x); return String(Object.is(v, -0) ? 0 : v); };

// The ids aquascope.watch.parse_item gives: source/station_id, river:<id>, area:w,s,e,n.
export function watchId(item) {
  if (!item) return null;
  if (item.kind === "gauge") return `${item.source}/${item.station_id}`;
  if (item.kind === "reach") return `river:${item.river_id}`;
  if (item.kind === "area" && Array.isArray(item.bbox) && item.bbox.length === 4) return `area:${item.bbox.map(coord).join(",")}`;
  return null;
}

function validItem(it) {
  return it && typeof it === "object" && ["gauge", "reach", "area"].includes(it.kind) && watchId(it) === it.id;
}

function clean(raw) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.items)) return null;
  const seen = raw.seen && typeof raw.seen === "object" ? raw.seen : {};
  return {
    v: 1,
    items: raw.items.filter(validItem).slice(0, MAX_WATCH),
    seen,
    lastVisit: typeof raw.lastVisit === "string" ? raw.lastVisit : null,
  };
}

// Storage wins when it answers (another tab may have changed it), unless the last write failed: then the
// memory copy is the newer one.
export function readWatch(store = storage()) {
  if (store && !writeFailed) {
    try {
      const raw = store.getItem(WATCH_KEY);
      const got = raw ? clean(JSON.parse(raw)) : empty();
      if (got) memory = got;
    } catch { /* denied or corrupt: keep the memory copy */ }
  }
  return structuredCopy(memory);
}

export function writeWatch(next, store = storage()) {
  memory = clean(next) || empty();
  if (store) {
    try {
      store.setItem(WATCH_KEY, JSON.stringify(memory));
      writeFailed = false;
    } catch { writeFailed = true; /* the memory copy carries on */ }
  }
  return structuredCopy(memory);
}

function structuredCopy(x) { return JSON.parse(JSON.stringify(x)); }

export function isWatched(id, store = storage()) {
  return readWatch(store).items.some((it) => it.id === id);
}

export function findItem(id, store = storage()) {
  return readWatch(store).items.find((it) => it.id === id) || null;
}

// A new item keeps what the list needs to draw and to be checked without the catalog.
export function makeItem(kind, fields, today = new Date().toISOString().slice(0, 10)) {
  const item = { kind, added: today, threshold: null };
  if (kind === "gauge") {
    Object.assign(item, { source: String(fields.source), station_id: String(fields.station_id),
      name: fields.name || fields.station_id });
    if (Number.isFinite(Number(fields.river_id)) && fields.river_id !== null) item.river_id = Number(fields.river_id);
  } else if (kind === "reach") {
    Object.assign(item, { river_id: Number(fields.river_id), name: fields.name || `River reach ${fields.river_id}` });
  } else if (kind === "area") {
    item.bbox = areaBox(fields.bbox || []);
    item.name = fields.name || areaName(item.bbox);
  }
  if (kind !== "area") {
    for (const k of ["lat", "lon"]) if (Number.isFinite(Number(fields[k])) && fields[k] !== null) item[k] = round4(fields[k]);
  }
  item.id = watchId(item);
  return item;
}

// A drawn box on a wrapped map can reach past ±180°; the package takes longitudes in -180..180.
const wrapLon = (x) => { const v = ((Number(x) + 180) % 360 + 360) % 360 - 180; return v === -180 && Number(x) > 0 ? 180 : v; };
export function areaBox(bbox) {
  const [w, s, e, n] = bbox.map(Number);
  if (!(e - w < 360)) return [-180, s, 180, n].map(round4);
  return [wrapLon(w), s, wrapLon(e), n].map(round4);
}

export function areaName(bbox) {
  const [w, s, e, n] = bbox.map(Number);
  return `Area ${s.toFixed(2)} to ${n.toFixed(2)} °N, ${w.toFixed(2)} to ${e.toFixed(2)} °E`;
}

export function addItem(item, store = storage()) {
  const st = readWatch(store);
  if (!item || !item.id || st.items.some((it) => it.id === item.id)) return st;
  st.items = [item, ...st.items].slice(0, MAX_WATCH);
  return writeWatch(st, store);
}

export function removeItem(id, store = storage()) {
  const st = readWatch(store);
  st.items = st.items.filter((it) => it.id !== id);
  delete st.seen[id];
  return writeWatch(st, store);
}

export function toggleItem(item, store = storage()) {
  return isWatched(item.id, store) ? removeItem(item.id, store) : addItem(item, store);
}

// A threshold is { return_period: T } or { value: x }; null clears it (the 2-year flow is then used).
export function setThreshold(id, threshold, store = storage()) {
  const st = readWatch(store);
  const it = st.items.find((x) => x.id === id);
  if (!it) return st;
  it.threshold = normaliseThreshold(threshold);
  return writeWatch(st, store);
}

export function normaliseThreshold(t) {
  if (!t || typeof t !== "object") return null;
  const rp = Number(t.return_period);
  if (t.return_period !== undefined && t.return_period !== null && Number.isFinite(rp) && rp >= 1.01 && rp <= 1000) {
    return { return_period: rp };
  }
  const v = Number(t.value);
  if (t.value !== undefined && t.value !== null && t.value !== "" && Number.isFinite(v)) return { value: v };
  return null;
}

// The menu value for a threshold ("rp:10", "value") and back.
export function thresholdChoice(t) {
  const n = normaliseThreshold(t);
  if (!n) return "rp:2";
  return n.return_period !== undefined ? `rp:${n.return_period}` : "value";
}

export function thresholdLabel(t, unit = "m³/s") {
  const n = normaliseThreshold(t);
  if (!n) return "the 2-year flow";
  if (n.return_period !== undefined) return `the ${n.return_period}-year flow`;
  return `${n.value} ${unit}`;
}

// What the worker needs: the items (without anything it would not use) and, per item, what was seen last:
// the stored state, else the day it was added when that is after the last visit, else the last visit.
export function digestRequest(st, today = new Date().toISOString().slice(0, 10)) {
  const lastSeen = {};
  for (const it of st.items) {
    const seen = st.seen[it.id];
    if (seen && seen.date) lastSeen[it.id] = seen;
    else if (st.lastVisit && (!it.added || it.added <= st.lastVisit)) lastSeen[it.id] = { date: st.lastVisit };
    else if (it.added && it.added < today) lastSeen[it.id] = { date: it.added };
    else lastSeen[it.id] = { date: today };
  }
  const items = st.items.map((it) => {
    const out = { id: it.id, kind: it.kind, name: it.name, threshold: it.threshold || null };
    for (const k of ["source", "station_id", "river_id", "lat", "lon", "bbox"]) if (it[k] !== undefined) out[k] = it[k];
    return out;
  });
  return { items, last_seen: lastSeen, today };
}

// After a digest: keep each item's "seen" state, and today as the last visit once every item answered.
export function applyDigest(st, results, { complete = true, today = new Date().toISOString().slice(0, 10) } = {}) {
  const next = structuredCopy(st);
  for (const r of results || []) {
    if (r && r.id && r.seen && !r.error && next.items.some((it) => it.id === r.id)) next.seen[r.id] = r.seen;
  }
  if (complete) next.lastVisit = today;
  return next;
}

// Whether the "Since you were here" panel has anything to say on load.
export function shouldGreet(st, today = new Date().toISOString().slice(0, 10)) {
  return st.items.length > 0 && st.lastVisit !== today;
}

/** Only for tests: forget the in-memory fallback. */
export function _resetMemory() { memory = empty(); writeFailed = false; }
