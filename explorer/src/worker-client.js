// Talking to the Pyodide worker. Progress goes to the boot bar wherever the
// user is (it used to be dropped unless a station was selected), and every
// call can be abandoned: Python keeps running to completion in the worker, but
// a cancelled call never lands on the page.

import { metrics } from "./metrics.js?v=__BUILD__";
import { CONFIG } from "../config.js?v=__BUILD__";
import { sourceStyle, state } from "./core.js?v=__BUILD__";
import { bootDone, bootProgress } from "./shell.js?v=__BUILD__";

let worker = null;
const askListeners = new Set();
const solveListeners = new Set();
const studioListeners = new Set();
const artifactListeners = new Set();
const restartListeners = new Set();

export function onAskProgress(fn) { askListeners.add(fn); return () => askListeners.delete(fn); }
// Solve's timeline events ({role, step, event, detail}) with the id of the call they belong to.
export function onSolveProgress(fn) { solveListeners.add(fn); return () => solveListeners.delete(fn); }
// The Study crew's events ({role, step, event, detail, at}) and its figures as they are drawn
// ({id, kind, name, media_type, caption, step, data} with the PNG bytes as base64), each with
// the id of the call they belong to.
export function onStudioProgress(fn) { studioListeners.add(fn); return () => studioListeners.delete(fn); }
export function onStudioArtifact(fn) { artifactListeners.add(fn); return () => artifactListeners.delete(fn); }
// The worker was terminated and a fresh one is booting: a module that had handed it state (a table, a
// study's bytes) re-sends what it can, or marks what is gone.
export function onWorkerRestart(fn) { restartListeners.add(fn); return () => restartListeners.delete(fn); }
// Study this area: the engine's progress events ({phase, done, total, site}) with the call's id.
const areaListeners = new Set();
export function onAreaProgress(fn) { areaListeners.add(fn); return () => areaListeners.delete(fn); }

export function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(`./worker.js?v=${CONFIG.build}`, { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { if (!state.workerReady) bootProgress(m.text); return; }
    if (m.type === "ask_progress") { for (const fn of askListeners) fn(m.text); return; }
    if (m.type === "solve_progress") { for (const fn of solveListeners) fn(m.event, m.id); return; }
    if (m.type === "studio_progress") { for (const fn of studioListeners) fn(m.event, m.id); return; }
    if (m.type === "studio_artifact") { for (const fn of artifactListeners) fn(m.artifact, m.id); return; }
    if (m.type === "area_progress") { for (const fn of areaListeners) fn(m.event, m.id); return; }
    if (m.type === "ready") {
      state.workerReady = true;
      bootDone();
      warmLightPool();  // the light workers boot once Python is up here, so the first click finds them ready
      return;
    }
    const pending = state.pending.get(m.id);
    if (!pending) return;                       // cancelled: drop it
    state.pending.delete(m.id);
    if (m.type === "error") pending.reject(new Error(m.message));
    else pending.resolve(m.result);
  };
  worker.onerror = (e) => {
    console.error(e);
    bootProgress("");
    for (const [, p] of state.pending) p.reject(new Error(e.message || "worker error"));
    state.pending.clear();
  };
  worker.postMessage({ type: "init", pyodideIndexURL: CONFIG.pyodideIndexURL, wheelsJson: new URL(CONFIG.wheelsJson, location.href).href, build: CONFIG.build });
  return worker;
}

export class Cancelled extends Error {
  constructor() { super("cancelled"); this.name = "Cancelled"; }
}

// Stop that means stop. Python cannot be interrupted mid-call (a study's run
// is one synchronous call that keeps going after the page abandons it, and
// the next message queues behind it), so the worker is terminated and a fresh
// one boots, with the progress bar as at first load. Every call in flight is
// rejected as Cancelled. The listeners above are this module's and carry
// over; the pending map is the page's and is emptied here; what the old
// worker held (the catalog, a table, a study's bytes) is gone, and the owners
// re-send it on the restart event.
export function restartWorker() {
  if (worker) {
    try { worker.terminate(); } catch (err) { console.warn("terminate:", err && err.message); }
  }
  worker = null;
  state.workerReady = false;
  state.ask.catalogSent = false;
  state.workerEpoch = (state.workerEpoch || 0) + 1;
  const pending = [...state.pending.values()];
  state.pending.clear();
  for (const p of pending) p.reject(new Cancelled());
  ensureWorker();
  for (const fn of restartListeners) {
    try { fn(); } catch (err) { console.warn("on restart:", err && err.message); }
  }
  return worker;
}

// Returns a promise plus a cancel() that rejects it and forgets the reply.
export function callCancelable(type, payload = {}) {
  const started = performance.now(), runtime = state.workerReady ? "warm" : "cold";
  const kind = { analyze: "station", ingest: "table", load_table: "table", studio: "study" }[type];
  ensureWorker();
  const id = ++state.reqId;
  let reject_;
  const promise = new Promise((resolve, reject) => {
    reject_ = reject;
    state.pending.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...payload });
  });
  const cancel = () => {
    if (!state.pending.has(id)) return false;
    state.pending.delete(id);
    reject_(new Cancelled());
    return true;
  };
  const measured = promise.then(result => {
    if (kind) metrics.record(result?.error ? "operation_error" : "operation_ok", { kind, runtime, durationMs: performance.now() - started });
    return result;
  }, error => {
    if (kind && !(error instanceof Cancelled)) metrics.record("operation_error", { kind, runtime, durationMs: performance.now() - started });
    throw error;
  });
  return { promise: measured, cancel, id };
}

export function call(type, payload = {}) {
  return callCancelable(type, payload).promise;
}

// The catalog the page holds, handed to Python once so find_stations() and
// assess_site() answer from memory (the worker cannot read the Hub).
export async function ensureCatalogInWorker() {
  if (state.ask.catalogSent) return;
  const rows = state.stations.map((r) => ({
    source: r.source, station_id: r.station_id, name: r.name, latitude: r.lat, longitude: r.lon,
    site_id: r.site_id || r.station_id,
    variables: r.variables || [], period_start: r.period_start, period_end: r.period_end, url: r.url,
    agency: sourceStyle(r.source).label,
  }));
  await call("catalog", { rows });
  state.ask.catalogSent = true;
}

export function workerBusyMessage() {
  return state.workerReady ? null : "Loading Python in your browser (about 30 MB, once).";
}

// ── the light pool ──────────────────────────────────────────────────────────
// Pyodide's HTTP is synchronous, so one worker reads one URL at a time and every call queues behind the one
// before it. A few light workers (worker.js with `lite`: numpy but no pandas or scipy, so they boot in a few
// seconds) take the calls that only read the network: a place's context layers, a click's river snap and the
// quick forecast. They run side by side, and beside the main worker. Calls are queued here and handed to
// whichever light worker is free, higher `priority` first; a light worker that cannot boot hands its calls to
// the main worker, so nothing is lost on a browser that cannot run several.

const POOL_MAX = 3;
const pool = { slots: [], queue: [], warmed: false, broken: false };

export function lightPoolSize() {
  if (typeof navigator === "undefined") return 1;
  const mobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent || "");
  const memory = Number(navigator.deviceMemory) || 4;   // GB, where the browser says
  if (mobile || memory <= 2) return 1;
  return Math.max(1, Math.min(POOL_MAX, (Number(navigator.hardwareConcurrency) || 2) - 1));
}

function startLight() {
  const slot = { worker: new Worker(`./worker.js?v=${CONFIG.build}`, { type: "module" }), ready: false, job: null };
  slot.worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { slot.ready = true; pump(); return; }
    if (m.type === "error" && m.id === undefined) { dropLight(slot); return; }  // the boot failed
    if (!slot.job || m.id !== slot.job.id) return;
    const job = slot.job;
    slot.job = null;
    if (job.cancelled) { /* the reader moved on: dropped */ } else if (m.type === "error") job.reject(new Error(m.message));
    else if (m.type === "result") job.resolve(m.result);
    pump();
  };
  slot.worker.onerror = (e) => { console.warn("light worker:", e && e.message); dropLight(slot); };
  slot.worker.postMessage({
    type: "init", pyodideIndexURL: CONFIG.pyodideIndexURL, lite: true,
    wheelsJson: new URL(CONFIG.wheelsJson, location.href).href, build: CONFIG.build,
  });
  pool.slots.push(slot);
}

function toMain(job) {
  const { promise } = callCancelable(job.type, job.payload);
  promise.then(job.resolve, job.reject);
}

function dropLight(slot) {
  try { slot.worker.terminate(); } catch { /* already gone */ }
  pool.slots = pool.slots.filter((s) => s !== slot);
  if (slot.job) { const job = slot.job; slot.job = null; if (!job.cancelled) toMain(job); }
  if (!pool.slots.length) {
    pool.broken = true;
    while (pool.queue.length) toMain(pool.queue.shift());
  }
}

function pump() {
  for (const slot of pool.slots) {
    if (!slot.ready || slot.job || !pool.queue.length) continue;
    const job = pool.queue.shift();
    slot.job = job;
    slot.worker.postMessage({ type: job.type, id: job.id, ...job.payload });
  }
}

export function warmLightPool() {
  if (pool.warmed || typeof Worker === "undefined") return;
  pool.warmed = true;
  const n = lightPoolSize();
  for (let i = 0; i < n; i++) startLight();
}

// Like callCancelable, on a light worker. cancel() drops a call still waiting for a worker (one already
// running finishes and is forgotten), so a place the reader has left does not hold up the next one.
export function callLightCancelable(type, payload = {}, { priority = 0 } = {}) {
  if (pool.broken || typeof Worker === "undefined") return callCancelable(type, payload);
  warmLightPool();
  const id = ++state.reqId;
  let job;
  const promise = new Promise((resolve, reject) => {
    job = { id, type, payload, priority, resolve, reject };
    let at = pool.queue.findIndex((j) => j.priority < priority);
    if (at < 0) at = pool.queue.length;
    pool.queue.splice(at, 0, job);
  });
  pump();
  const cancel = () => {
    if (job.cancelled) return false;
    const at = pool.queue.indexOf(job);
    if (at >= 0) pool.queue.splice(at, 1);
    else if (!pool.slots.some((s) => s.job === job)) return false;   // answered already
    job.cancelled = true;   // a running one finishes and its answer is dropped on arrival
    job.reject(new Cancelled());
    return true;
  };
  return { promise, cancel, id };
}

export function callLight(type, payload = {}, opts = {}) {
  return callLightCancelable(type, payload, opts).promise;
}
