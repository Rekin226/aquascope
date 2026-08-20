// Talking to the Pyodide worker. Progress goes to the boot bar wherever the
// user is (it used to be dropped unless a station was selected), and every
// call can be abandoned: Python keeps running to completion in the worker, but
// a cancelled call never lands on the page.

import { CONFIG } from "../config.js?v=__BUILD__";
import { state } from "./core.js?v=__BUILD__";
import { bootDone, bootProgress } from "./shell.js?v=__BUILD__";

let worker = null;
const askListeners = new Set();

export function onAskProgress(fn) { askListeners.add(fn); return () => askListeners.delete(fn); }

export function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(`./worker.js?v=${CONFIG.build}`);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { if (!state.workerReady) bootProgress(m.text); return; }
    if (m.type === "ask_progress") { for (const fn of askListeners) fn(m.text); return; }
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

export function workerBusyMessage() {
  return state.workerReady ? null : "Loading Python in your browser (about 15 MB, once).";
}
