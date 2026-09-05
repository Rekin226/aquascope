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

export function onAskProgress(fn) { askListeners.add(fn); return () => askListeners.delete(fn); }
// Solve's timeline events ({role, step, event, detail}) with the id of the call they belong to.
export function onSolveProgress(fn) { solveListeners.add(fn); return () => solveListeners.delete(fn); }

export function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(`./worker.js?v=${CONFIG.build}`);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { if (!state.workerReady) bootProgress(m.text); return; }
    if (m.type === "ask_progress") { for (const fn of askListeners) fn(m.text); return; }
    if (m.type === "solve_progress") { for (const fn of solveListeners) fn(m.event, m.id); return; }
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
    variables: r.variables || [], period_start: r.period_start, period_end: r.period_end, url: r.url,
    agency: sourceStyle(r.source).label,
  }));
  await call("catalog", { rows });
  state.ask.catalogSent = true;
}

export function workerBusyMessage() {
  return state.workerReady ? null : "Loading Python in your browser (about 15 MB, once).";
}
