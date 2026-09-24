// Talking to the Pyodide worker. Progress goes to the boot bar wherever the
// user is (it used to be dropped unless a station was selected), and every
// call can be abandoned: Python keeps running to completion in the worker, but
// a cancelled call never lands on the page.

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
  worker = new Worker(`./worker.js?v=${CONFIG.build}`);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { if (!state.workerReady) bootProgress(m.text); return; }
    if (m.type === "ask_progress") { for (const fn of askListeners) fn(m.text); return; }
    if (m.type === "solve_progress") { for (const fn of solveListeners) fn(m.event, m.id); return; }
    if (m.type === "studio_progress") { for (const fn of studioListeners) fn(m.event, m.id); return; }
    if (m.type === "studio_artifact") { for (const fn of artifactListeners) fn(m.artifact, m.id); return; }
    if (m.type === "area_progress") { for (const fn of areaListeners) fn(m.event, m.id); return; }
    if (m.type === "ready") { state.workerReady = true; bootDone(); return; }
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
  return { promise, cancel, id };
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
  return state.workerReady ? null : "Loading Python in your browser (about 15 MB, once).";
}
