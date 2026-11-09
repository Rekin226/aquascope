// Explain what is on screen (#562): "what am I seeing?" answered with three to five short notes pinned on the
// map, each at the place it describes and each naming its source.
//
// The facts come from the package (aquascope.map_explain, in a light worker): this module only hands it what
// the page already shows (the view, the map date, the layers that are on, the month of the river status, the
// Floods ahead issue, the Floods past cells and today's gauge status) and pins what comes back. With the
// on-device model, or the reader's own key from Ask, the model picks and the code writes (#590): it chooses
// which of the package's notes to show, in which order, and one of each note's fixed variants, as ids held to
// a JSON Schema of enums (passed to Chrome's Prompt API as responseConstraint). Every word on a note is the
// package's; a reply outside the enums is refused whole, said in one line, and the fixed wording stays. With
// neither, the package's fixed wording is used. Every note goes in the action log (map-actions.js addPin),
// so each one, or all, can be undone; asking again replaces the notes of the time before.

import { $, escapeHtml, sourceStyle, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { addPin, cameraPadding, log, renderLog, undoEntry } from "./map-actions.js?v=__BUILD__";
import { explainWho } from "./map-actions-core.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { statusReady, statusShown } from "./status-layer.js?v=__BUILD__";
import { floodsAheadReady, floodsAheadShown, floodsVisible } from "./floods-ahead.js?v=__BUILD__";
import { floodsPastReady, floodsPastShown } from "./floods-past.js?v=__BUILD__";
import { QUICK_PRIORITY, call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";

const GROUP = "explain";
let run = 0;

// ── what is on screen ───────────────────────────────────────────────────────

// How far from the middle of a globe seen whole a note may sit, in degrees of arc: past it, toward the rim,
// the map is too foreshortened to read and MapLibre fades the pins out.
const RIM = 62;

/** On a globe seen whole, the side facing the reader ({lat, lon, deg}); null on a flat map or close in. */
export function viewDisc() {
  if (!state.globe || map.getZoom() >= 3) return null;
  const c = map.getCenter();
  return { lat: c.lat, lon: c.lng, deg: RIM };
}

// The map's own furniture: a note's pin should not sit under any of it (the package keeps pins out of these).
const FURNITURE = ["#map-legends", "#timebar", "#map-ask .ma-bar", "#ma-status", "#map-tools", ".map-lead"];

// The screen box of the map minus a thin edge, and the top under the box.
function screenBox() {
  const frame = map.getContainer().getBoundingClientRect();
  const pad = cameraPadding();
  return { frame, x0: 56, x1: frame.width - 16, y0: Math.min(pad.top, frame.height * 0.3), y1: frame.height - 34 };
}

/**
 * The part of the map you can see as [west, south, east, north]. The globe far out shows a hemisphere, not the
 * whole world its bounds would say. What sits under the legend, the time bar and the box is still in view
 * (they are translucent and fold away); viewAvoid() keeps the notes' pins off them.
 */
export function viewBbox() {
  const z = map.getZoom();
  const c = map.getCenter();
  const { width, height } = map.getContainer().getBoundingClientRect();
  const pad = cameraPadding();
  if (state.globe && z < 3) {
    // The whole globe, or most of it, is on screen: as far from the centre as the free part of the screen
    // reaches on a sphere of the map's radius, short of the rim, where nothing can be read.
    const R = (512 * 2 ** z) / (2 * Math.PI);
    const reach = (px) => Math.min(RIM, (Math.asin(Math.min(1, px / R)) * 180) / Math.PI);
    const lon = reach((width - pad.left - pad.right) / 2), lat = reach((height - pad.top - pad.bottom) / 2);
    return [c.lng - lon, Math.max(-85, c.lat - lat), c.lng + lon, Math.min(85, c.lat + lat)];
  }
  const { x0, x1, y0, y1 } = screenBox();
  // The globe bends the meridians: the box is the one inside every point of each edge, not around the corners.
  const along = (a, b) => [0, 0.25, 0.5, 0.75, 1].map((t) => a + (b - a) * t);
  const at = (pts) => pts.map(([x, y]) => map.unproject([x, y]));
  const west = Math.max(...at(along(y0, y1).map((y) => [x0, y])).map((p) => p.lng));
  const east = Math.min(...at(along(y0, y1).map((y) => [x1, y])).map((p) => p.lng));
  const north = Math.min(...at(along(x0, x1).map((x) => [x, y0])).map((p) => p.lat));
  const south = Math.max(...at(along(x0, x1).map((x) => [x, y1])).map((p) => p.lat));
  return [west, Math.max(-85, south), east, Math.min(85, north)];
}

/** The furniture's boxes in degrees, [[w, s, e, n], ...], for the package to keep pins out of; [] on the globe far out. */
export function viewAvoid() {
  if (state.globe && map.getZoom() < 3) return [];
  const { frame } = screenBox();
  const out = [];
  for (const sel of FURNITURE) {
    const el = document.querySelector(sel);
    if (!el || el.hidden) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    // A little round it: a pin right at the edge would put its card under it.
    const x0 = r.left - frame.left - 24, x1 = r.right - frame.left + 24, y0 = r.top - frame.top - 24, y1 = r.bottom - frame.top + 24;
    const pts = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map((p) => map.unproject(p));
    out.push([Math.min(...pts.map((p) => p.lng)), Math.min(...pts.map((p) => p.lat)),
      Math.max(...pts.map((p) => p.lng)), Math.max(...pts.map((p) => p.lat))]);
  }
  return out;
}

// A first cut of each layer's data to the view, so less crosses to the worker; the package cuts again.
function inBoxOf(bbox) {
  const [w, s, e, n] = bbox;
  if (e - w >= 360) return (lat) => lat >= s && lat <= n;
  const wrap = (x) => ((((x + 180) % 360) + 360) % 360) - 180;
  const W = wrap(w), E = wrap(e);
  return (lat, lon) => lat >= s && lat <= n && (W <= E ? lon >= W && lon <= E : lon >= W || lon <= E);
}

// Today's status of the gauges in view, when the gauges are on and coloured by it.
function gaugesShown(inBox) {
  if (state.gaugesOn === false || state.gaugeStyle !== "now" || !state.nowStatus) return null;
  const out = [];
  for (const r of state.stations) {
    if (state.hidden.has(r.source) || !inBox(r.lat, r.lon)) continue;
    const st = state.nowStatus.get(`${r.source}/${r.station_id}`);
    if (!st || !st.cls) continue;
    out.push({
      source: r.source, station_id: r.station_id, name: r.name || r.station_id, agency: sourceStyle(r.source).label,
      lat: r.lat, lon: r.lon, cls: st.cls, pct: st.pct, date: st.date, value: st.value, n_years: st.n_years,
    });
  }
  return out;
}

/** Everything the package needs to read the view: what is on, and what each layer holds inside it. */
export function gatherView() {
  const bbox = viewBbox();
  const inBox = inBoxOf(bbox);
  const status = statusShown();
  const ahead = floodsVisible() ? floodsAheadShown(inBox) : null;
  const past = floodsPastShown(inBox);
  const gauges = gaugesShown(inBox);
  const layers = [];
  if (state.status) layers.push("status");
  if (floodsVisible()) layers.push("floods_ahead");
  if (state.floodsPast) layers.push("floods_past");
  if (gauges) layers.push("gauges");
  return {
    view: { bbox, disc: viewDisc(), avoid: viewAvoid(), date: state.date, layers, zoom: Math.round(map.getZoom() * 10) / 10 },
    status_url: status ? status.url : null, status_month: status ? status.month : null,
    warnings: ahead, floods_past: past, gauges: gauges || [],
  };
}

// ── who words the notes ─────────────────────────────────────────────────────

async function byDevice(res) {
  if (!res.prompt || !(await localModelReady())) return null;
  const reply = await generateJsonLocally({
    system: res.prompt.system, prompt: res.prompt.prompt || "Choose the notes.", schema: res.prompt.schema, timeoutMs: 30000,
  });
  const checked = await callLight("map_explain", { op: "check", reply, facts: res.facts, view: res.view }, { priority: QUICK_PRIORITY });
  return { ...checked, model: localReaderLabel() || "the on-device model" };
}

async function byKey(res) {
  const cfg = askModelConfig();
  if (!cfg || !res.facts.length) return null;
  const out = await call("map_explain", {
    op: "llm", facts: res.facts, view: res.view, context: currentContext(),
    provider: cfg.provider, model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url,
  });
  if (out && out.error) throw new Error(out.error);
  return out;
}

// ── pinning ─────────────────────────────────────────────────────────────────

/** The notes Explain put on the map last time that are still there. */
const explainPins = () => log.live().filter((e) => e.action && e.action.type === "add_pin" && e.after && e.after.group === GROUP);

function clearNotes() {
  for (const e of explainPins().reverse()) undoEntry(e.id);
  // Scout's numbered pins (#563) step aside too: two sets of numbers on one map read as one list.
  for (const e of log.live().filter((x) => x.action && x.action.type === "add_pin" && x.after && x.after.rank).reverse()) {
    undoEntry(e.id);
  }
}

function pinNotes(notes, by) {
  const ids = [];
  notes.forEach((n, i) => {
    try {
      ids.push(addPin({
        lat: n.lat, lon: n.lon, title: n.title, text: n.text, facts: n.facts, source: n.source, credit: n.credit,
        place: n.place, when: n.when, tone: n.tone, kind: "note", n: i + 1, group: GROUP,
      }, { by: n.by || by, label: `Note: ${n.title}` }));
    } catch (err) {
      console.warn("explain: a note could not be pinned", err && err.message);
    }
  });
  return ids;
}

// ── the one entry point ─────────────────────────────────────────────────────

/**
 * Explain the view: read it into facts, word them (on-device model, the reader's key, or the fixed wording),
 * and pin three to five notes. `say(html, kind)` writes the line under the box. Returns the pinned entry ids.
 */
export async function explainView({ say = () => {} } = {}) {
  if (!state.mapOk || !map) { say("The map is not ready yet.", "warn"); return []; }
  const my = ++run;
  say(`<span class="spinner" aria-hidden="true"></span>${state.workerReady ? "Reading the view…" : "Starting the engine (once), then reading the view…"}`);
  // The layers still reading their file (just after the page opens, or a new date) are waited for, a while.
  await Promise.race([
    Promise.all([floodsAheadReady().catch(() => {}), statusReady(), floodsPastReady()]),
    new Promise((r) => setTimeout(r, 15000)),
  ]);
  const res = await callLight("map_explain", { op: "facts", ...gatherView(), context: currentContext() }, { priority: 3 });
  if (my !== run) return [];
  if (!res || res.error) { say(`Could not read the view: ${escapeHtml((res && res.error) || "no answer")}.`, "error"); return []; }
  if (!res.facts.length || !res.notes.length) {
    const why = res.skipped && res.skipped.length ? ` (${res.skipped.join("; ")})` : "";
    say(`Nothing on screen to explain${escapeHtml(why)}. Turn a layer on in "On the map", or move the map.`, "warn");
    return [];
  }
  let notes = res.notes, by = "rules", who = ["Rules", "no model"], refused = [];
  try {
    const device = await byDevice(res).catch((err) => ({ error: err.message, model: localReaderLabel() || "the on-device model" }));
    if (my !== run) return [];
    if (device && device.notes && device.notes.length) {
      notes = device.notes; by = "device"; who = [device.model, "chose on this device; words by code"]; refused = device.refused || [];
    } else {
      // The device was asked and none of its picks was used: the rules' notes stand, and the line says the
      // model's reply was refused (and why), not that there was no model (#590).
      if (device) { who = explainWho(device); refused = device.refused || []; }
      if (askModelConfig()) say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(askModelConfig().label)}…`);
      const key = await byKey(res).catch((err) => ({ error: err.message, model: askModelConfig() && askModelConfig().label }));
      if (my !== run) return [];
      if (key && key.notes && key.notes.length) {
        notes = key.notes; by = "key"; who = [key.model, "chose with your key; words by code"]; refused = key.refused || [];
      } else if (key) {
        who = explainWho(key);
        refused = [...refused, ...(key.refused || [])];
      }
    }
  } catch (err) {
    console.info("explain: the model's notes were not used:", err && err.message);
  }
  clearNotes();
  const ids = pinNotes(notes, by);
  renderLog();
  const n = ids.length;
  // A refusal in one short line: the whole reply, or how many of its picks, and why (#590).
  const whole = refused.length === 1 && /^the model's reply was refused/.test(refused[0]);
  const refusedLine = !refused.length ? "" : whole
    ? `<span class="ma-note">${escapeHtml(refused[0].charAt(0).toUpperCase() + refused[0].slice(1))}; the fixed wording is shown</span>`
    : `<span class="ma-note">${refused.length} of the model's picks not used: ${escapeHtml(refused.join("; "))}</span>`;
  const credit = (res.credits || []).find((c) => c.startsWith("Places:"));
  say(`<b class="ma-who">${escapeHtml(who[0])}</b><span class="ma-how">${escapeHtml(who[1])}</span>` +
    `<span>${n} note${n === 1 ? "" : "s"} on the map, each with its source</span>${refusedLine}` +
    (credit ? `<span class="ma-credit">${escapeHtml(credit)}</span>` : ""), n ? "ok" : "warn");
  announce(`${n} notes pinned on the map`);
  // A phone: the line sits over the map, so once read it steps aside (the log keeps every note a tap away).
  const narrow = globalThis.matchMedia && matchMedia("(max-width: 640px)").matches;
  if (narrow && n) setTimeout(() => { if (my === run && !$("map-ask").classList.contains("open")) say(""); }, 6000);
  return ids;
}

export function initMapExplain({ say }) {
  const btn = $("btn-map-explain");
  if (btn) btn.addEventListener("click", () => { void explainView({ say }); });
  globalThis.__aq.mapExplain = { run: () => explainView({ say }), gather: gatherView };
}
