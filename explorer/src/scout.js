// Scout (#563): one button beside "Ask the map" that drops up to ten numbered pins on what stands out in the
// view, each with its reason, its numbers and its source.
//
// aquascope.map_scout finds them in a light worker: from the daily scout file (scout/latest.json, written by the
// flood-warnings workflow, with record checks against 1990 on) when the view is half the globe at the newest
// month, otherwise by a live scan of the view at the map's month. With the on-device model ready, or a key in
// Ask, the model picks and the code writes (#590): it orders the findings and picks one of the package's fixed
// variants for each, as ids held to a JSON Schema of enums (map_scout.wording_schema, Chrome's
// responseConstraint); a reply outside it is refused whole and the rules' order and wording stay. Keyless,
// the rules' templates say it. The pins go in the action log, so each one, or all, can be undone.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { addPin, log, setLogOpen, undoEntry, updatePin } from "./map-actions.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { ensureNowStatus } from "./now-map.js?v=__BUILD__";
import { skillUrl } from "./evidence-core.js?v=__BUILD__";
import {
  SCOUT_EXTRA, SCOUT_MAX, askedFailed, keepVisible, scoutLine, scoutMonth, scoutPin, scoutView, wordingLine,
} from "./scout-core.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";

let running = 0;
let lastPins = [];      // the log ids of the last scout's pins: a new scout replaces them
let skillRows = null;   // the evidence table's best rows, read once
let lastResult = null;  // the package's answer, for a look in the console (__aq.scout.result())

function say(html, kind = "info") {
  const el = $("ma-status");
  if (!el) return;
  el.hidden = !html;
  el.className = `ma-status ${kind}`;
  el.innerHTML = html || "";
}

function busy(on) {
  const b = $("btn-scout");
  if (!b) return;
  b.classList.toggle("busy", on);
  b.setAttribute("aria-busy", on ? "true" : "false");
}

// How far down the map the bars at the top reach, and where the time bar starts, in the map's own pixels: a pin
// stands 37 px tall above its point, so its point sits a pin's height under the lowest bar (the line over the
// globe comes back when the scout's line steps aside).
function frameEdges(el, h) {
  const f = el.getBoundingClientRect();
  let top = 56, bottom = h - 110;
  for (const sel of [".ma-bar", ".map-lead", "#ma-status"]) {
    const e = document.querySelector(sel);
    const r = e && !e.hidden ? e.getBoundingClientRect() : null;
    if (r && r.height && r.bottom - f.top < h / 3) top = Math.max(top, r.bottom - f.top);
  }
  const tb = $("timebar");
  const r = tb && !tb.hidden ? tb.getBoundingClientRect() : null;
  if (r && r.height && r.top - f.top > h / 2) bottom = Math.min(bottom, r.top - f.top - 8);
  return [top + 42, bottom];
}

// The part of the map a pin can be seen in: under the bars at the top, above the time bar, inside the side
// controls and around the legend stack, read back to degrees at points close together around that inner frame
// (on the globe a straight line on screen is a curve in degrees, so the points are dense).
function innerOutline(el) {
  const w = el.clientWidth, h = el.clientHeight;
  const wide = w > 700;
  const [y0, y1] = frameEdges(el, h);
  const x0 = wide ? 70 : 16, x1 = w - (wide ? 70 : 16);
  // The legends sit bottom left on a wide screen: the frame steps around them.
  let corner = [[x1, y1], [x0, y1]];
  const leg = $("map-legends");
  if (wide && leg) {
    const r = leg.getBoundingClientRect(), f = el.getBoundingClientRect();
    const lx = r.right - f.left + 12, ly = r.top - f.top - 6;
    if (r.width && r.height && lx > x0 && lx < w / 2 && ly > y0 + 40 && ly < y1) corner = [[x1, y1], [lx, y1], [lx, ly], [x0, ly]];
  }
  const ring = [[x0, y0], [x1, y0], ...corner];
  const pts = [];
  ring.forEach((a, i) => {
    const b = ring[(i + 1) % ring.length];
    for (let k = 0; k < 10; k++) pts.push([a[0] + ((b[0] - a[0]) * k) / 10, a[1] + ((b[1] - a[1]) * k) / 10]);
  });
  const out = [];
  for (const p of pts) {
    const ll = map.unproject(p);
    if (!ll || !Number.isFinite(ll.lat)) return null;
    out.push([ll.lng, ll.lat]);
  }
  return out;
}

// Whether a finding's pin stands where it can be seen: its point inside the frame innerOutline traces (under
// the bars, above the time bar, clear of the side controls) and, on the globe, on the side facing the reader.
function pinVisible(f) {
  const el = map.getContainer();
  const w = el.clientWidth, h = el.clientHeight;
  const [y0, y1] = frameEdges(el, h);
  const pad = w > 700 ? 70 : 16;
  const p = map.project([f.lon, f.lat]);
  if (!p || !(p.x >= pad && p.x <= w - pad && p.y >= y0 && p.y <= y1)) return false;
  // Not under the legend stack either (a pin stands 37 px above its point).
  const leg = $("map-legends");
  const r = leg && !leg.hidden ? leg.getBoundingClientRect() : null;
  if (r && r.width && r.height) {
    const f0 = el.getBoundingClientRect();
    const lx = r.left - f0.left, rx = r.right - f0.left, ty = r.top - f0.top, by = r.bottom - f0.top;
    if (p.x >= lx - 14 && p.x <= rx + 14 && p.y >= ty && p.y <= by + 42) return false;
  }
  if (state.globe && typeof map.transform?.isLocationOccluded === "function") {
    return !map.transform.isLocationOccluded({ lng: f.lon, lat: f.lat });
  }
  return true;
}

function currentView() {
  const c = map.getCenter();
  const el = map.getContainer();
  const b = map.getBounds();
  return scoutView({
    globe: Boolean(state.globe), zoom: map.getZoom(), center: [c.lng, c.lat], outline: innerOutline(el),
    bounds: b ? [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] : null,
    width: el.clientWidth, height: el.clientHeight,
  });
}

// ── what the page reads for the worker (the light worker has no parquet reader) ──

async function gaugeRows() {
  try { await ensureNowStatus(); } catch { /* the scout says it */ }
  if (!state.nowStatus) return null;
  const rows = [];
  for (const [key, v] of state.nowStatus) {
    const st = state.byKey.get(key);
    if (!st || !Number.isFinite(st.lat)) continue;
    rows.push({ source: st.source, station_id: st.station_id, name: st.name, lat: st.lat, lon: st.lon,
      class: v.cls, percentile: v.pct, value: v.value, n_years: v.n_years, value_date: v.date });
  }
  return rows;
}

async function evidenceRows() {
  if (skillRows) return skillRows;
  try {
    const { conn } = await duck();
    const url = skillUrl(CONFIG.stationsParquet).replace(/'/g, "''");
    const t = await conn.query(`SELECT source, station_id, lat, lon, model, label, is_best, kge, pbias, n_days,
        mean_gauge, area_ratio, CAST("start" AS VARCHAR) AS "start", CAST("end" AS VARCHAR) AS "end",
        CAST(computed_at AS VARCHAR) AS computed_at, COUNT(kge) OVER (PARTITION BY source, station_id) AS n_models
      FROM read_parquet('${url}') QUALIFY is_best`);
    skillRows = t.toArray().map((r) => {
      const o = {};
      for (const [k, v] of Object.entries(r.toJSON())) o[k] = typeof v === "bigint" ? Number(v) : v;
      const st = state.byKey.get(`${o.source}/${o.station_id}`);
      o.name = st ? st.name : null;
      return o;
    });
  } catch (err) {
    console.warn("scout: evidence table", err && err.message);
    return null;
  }
  return skillRows;
}

// ── place names: the page asks the gazetteer for all of them at once, the package names them ──

// Photon answers one request at a time (about a second each, measured 2026-10-11) and asks to be used fairly,
// so the names are asked for one after another, and not at all once it has failed twice in a row. Without a
// model the pins drop at once under a region's name or their coordinates and take their names afterwards.
async function namePlaces(res) {
  const cands = res.candidates || res.picks || [];
  if (!cands.some((f) => f.place_url)) return false;
  const ask = async (url) => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 8000);
    try {
      const r = await fetch(url, { signal: ctl.signal });
      return r.ok ? await r.json() : null;
    } catch {
      return null;    // a name is a nicety: the region or the coordinates stay
    } finally {
      clearTimeout(timer);
    }
  };
  const answers = [];
  let misses = 0;
  for (const f of cands) {
    const a = f.place_url && misses < 2 ? await ask(f.place_url) : null;
    if (f.place_url) misses = a ? 0 : misses + 1;
    answers.push(a);
  }
  if (!answers.some(Boolean)) return false;
  const named = await callLight("scout", { op: "places", findings: cands, answers }, { priority: 2 });
  if (!named || !named.findings) return false;
  const byId = new Map(named.findings.map((f) => [f.id, f]));
  res.candidates = cands.map((f) => byId.get(f.id) || f);
  res.picks = (res.picks || []).map((f) => byId.get(f.id) || f);
  return true;
}

// The names, once they are in, on the pins already on the map (a newer scout's pins are left alone).
async function renamePins(my, res, pinOf) {
  const t = performance.now();
  if (!(await namePlaces(res)) || my !== running) return;
  for (const f of res.picks) {
    const id = pinOf.get(f.id);
    if (id) updatePin(id, scoutPin(f, { mode: res.mode, made: res.published || "" }), { label: `Scout ${f.rank}: ${f.title}` });
  }
  if (res.page_ms) res.page_ms.names = Math.round(performance.now() - t);
}

// ── a model picks the order and a variant of each (the words stay the package's) ──

const DEVICE_WAIT_MS = 40000;

async function wordWithModel(res) {
  const findings = res.candidates || res.picks;
  let asked = null;
  if (await localModelReady()) {
    const name = localReaderLabel() || "the on-device model";
    try {
      const p = await callLight("scout", { op: "prompt", findings, context: currentContext() }, { priority: 2 });
      say(`<span class="spinner" aria-hidden="true"></span>Ordering the pins on your device…`);
      const reply = await generateJsonLocally({ system: p.system, prompt: p.prompt, schema: p.schema, timeoutMs: DEVICE_WAIT_MS });
      const out = await callLight("scout", { op: "words", findings, reply, by: "device" }, { priority: 2 });
      return { ...out, by: "device", model: name };
    } catch (err) {
      console.warn("scout: on-device pick", err && err.message);
      // The device was asked and gave nothing (a timeout, an error): said on the line, not "no model" (#590).
      asked = { findings: null, by: "device", model: name, failed: askedFailed(err, DEVICE_WAIT_MS) };
    }
  }
  const cfg = askModelConfig();
  if (cfg) {
    say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(cfg.label)} to order and word them…`);
    // The main worker: the provider client lives beside the Analyst there.
    const out = await call("scout", { op: "llm", findings, context: currentContext(), provider: cfg.provider,
      model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url });
    if (out && out.error) throw new Error(out.error);
    return { ...out, by: "key" };
  }
  return asked;
}

// ── the button ──────────────────────────────────────────────────────────────

function clearLast() {
  for (const id of lastPins) {
    const e = log.get(id);
    if (e && !e.undone) undoEntry(id);
  }
  lastPins = [];
  // Explain's numbered notes (#562) step aside too: two sets of numbers on one map read as one list.
  for (const e of log.live().filter((x) => x.action && x.action.type === "add_pin" && x.after && x.after.kind === "note").reverse()) {
    undoEntry(e.id);
  }
}

// The line steps aside after a while (the caption it covers comes back); the pins and their undo stay.
function fadeLater(my, ms) {
  setTimeout(() => {
    const el = $("ma-status");
    const ask = $("map-ask");
    if (my !== running || !el || el.hidden || (ask && ask.classList.contains("open"))) return;
    if (el.matches(":hover")) { fadeLater(my, 4000); return; }
    say("");
  }, ms);
}

export async function runScout() {
  if (!state.mapOk || !map) return null;
  const my = ++running;
  busy(true);
  say(`<span class="spinner" aria-hidden="true"></span>${state.workerReady ? "Scouting this view…" : "Starting the engine (once), then scouting…"}`);
  try {
    const modelReady = Boolean(askModelConfig()) || await localModelReady();
    const t0 = performance.now();
    const [gauges, skill] = await Promise.all([gaugeRows(), evidenceRows()]);
    if (my !== running) return null;
    const t1 = performance.now();
    const res = await callLight("scout", {
      op: "view", view: currentView(), month: scoutMonth(state.date), gauges, skill,
      // A few more than ten always: the ones whose pin would sit under a bar are left out on the page.
      extra: SCOUT_EXTRA, max_pins: SCOUT_MAX, places: "page",
    }, { priority: 2 });
    if (my !== running) return null;
    if (!res || res.error) throw new Error((res && res.error) || "the scout returned nothing");
    Object.assign(res, keepVisible(res, pinVisible, SCOUT_MAX, modelReady ? SCOUT_EXTRA : 0));
    // How long each part took, in ms, for a look in the console (names: filled in when they arrive).
    res.page_ms = { rows: Math.round(t1 - t0), scan: Math.round(performance.now() - t1), names: null };
    lastResult = res;
    let picks = res.picks || [];
    let by = "rules", model = "", refused = [], unknown = [], failed = "";
    if (modelReady && picks.length) {
      // A model words them with their place names, so it waits for the gazetteer.
      await namePlaces(res);
      if (my !== running) return null;
      picks = res.picks || [];
      try {
        const worded = await wordWithModel(res);
        if (my !== running) return null;
        if (worded && worded.findings && worded.findings.length) {
          picks = worded.findings;
          by = worded.by;
          model = worded.model || "";
          refused = worded.refused || [];
          unknown = worded.unknown || [];
        } else if (worded && worded.failed) {
          by = worded.by;
          model = worded.model || "";
          failed = worded.failed;
        }
      } catch (err) {
        failed = `the model could not be asked (${err.message}); the rules' wording is kept`;
      }
    }
    clearLast();
    const made = res.published || "";
    const pinOf = new Map();
    // The last first: pin 1 is drawn last, so it sits on top where pins crowd, and heads the log (newest first).
    for (const f of picks.slice().reverse()) {
      try {
        const id = addPin(scoutPin(f, { mode: res.mode, made }), { by: f.by === "rules" ? "rules" : by,
          label: `Scout ${f.rank}: ${f.title}` });
        lastPins.push(id);
        pinOf.set(f.id, id);
      } catch (err) {
        console.warn("scout pin", err && err.message);
      }
    }
    if (by === "rules") void renamePins(my, { ...res, picks }, pinOf);
    // Who worded the pins: a model whose wording is on no pin is named as asked, not as the one that worded them.
    const line = wordingLine({ by, model, picks, refused, unknown, failed });
    const [who, how] = [line.who, line.how];
    const n = lastPins.length;
    const notes = [...(res.notes || []), ...line.notes];
    // One line: who, how many. Each pin's card carries its own reason and source.
    say(n
      ? `<b class="ma-who">Scout</b><span class="ma-how">${escapeHtml(who)}, ${escapeHtml(how)}</span>` +
        `<span>${escapeHtml(scoutLine(res, n))}</span>` +
        (notes.length ? `<span class="ma-note">${escapeHtml(notes.join("; "))}</span>` : "")
      : escapeHtml(scoutLine(res, 0)), n ? "ok" : "warn");
    if (n) setLogOpen(false);
    fadeLater(my, notes.length ? 16000 : 9000);
    announce(n ? `Scout dropped ${n} pins` : "Scout found nothing in this view");
    return { picks, mode: res.mode, by };
  } catch (err) {
    if (my === running) say(`The scout could not run: ${escapeHtml(err.message)}`, "error");
    return null;
  } finally {
    if (my === running) busy(false);
  }
}

export function initScout() {
  const b = $("btn-scout");
  if (!b) return;
  b.addEventListener("click", () => { void runScout(); });
  actions.runScout = runScout;   // the WebMCP tool aquascope_scout (webmcp.js)
  globalThis.__aq.scout = { run: runScout, last: () => lastPins.slice(), result: () => lastResult };
}
