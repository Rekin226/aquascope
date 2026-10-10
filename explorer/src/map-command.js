// Ask the map (#561): one line on the map, opened from its button or the / key. A request is read by the
// keyless rules first (aquascope.map_commands in a light worker), then by the on-device model where the
// browser has one ready, then by the reader's own model through Ask's key; the line under the box says
// which answered. The actions are checked by the package, their places looked up in the gazetteer (Photon,
// the one the search uses; a gauge name falls back to the catalogue search), and applied through
// map-actions.js, which lists each in the action log with an undo.
//
// Nothing here ever uses a key of ours: with no rule match, no model on the device and no key of the
// reader's, the box says so and offers examples.

import { $, escapeHtml, state } from "./core.js?v=__BUILD__";
import { applyActions, renderLog, setLogOpen, undoAll, undoLast } from "./map-actions.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { searchStations } from "./search.js?v=__BUILD__";
import { todayIso } from "./timeline.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";

// A few that show the range, offered when the box opens empty and after a miss.
export const TRIES = [
  "Trace the Nile to the sea",
  "Where are rivers much above normal?",
  "Play the last 5 years",
  "Go to Bangladesh",
];

let open = false;
let run = 0;

const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

const box = () => $("map-ask");

function say(html, kind = "info") {
  const el = $("ma-status");
  if (!el) return;
  el.hidden = !html;
  el.className = `ma-status ${kind}`;
  el.innerHTML = html || "";
}

function showTries(on) {
  const el = $("ma-tries");
  if (!el) return;
  el.hidden = !on;
  if (on && !el.childElementCount) {
    el.innerHTML = TRIES.map((t) => `<button type="button" class="chip ma-try">${escapeHtml(t)}</button>`).join("");
  }
}

export function openMapAsk({ text = "" } = {}) {
  open = true;
  box().classList.add("open");
  $("ma-form").hidden = false;
  $("btn-map-ask").hidden = true;
  $("btn-map-ask").setAttribute("aria-expanded", "true");
  const hint = $("map-hint");
  if (hint) hint.hidden = true;
  const input = $("ma-input");
  if (text) input.value = text;
  showTries(!input.value && !$("ma-status").innerHTML);
  setLogOpen(true);
  input.focus();
  input.select();
}

export function closeMapAsk() {
  open = false;
  box().classList.remove("open");
  $("ma-form").hidden = true;
  $("btn-map-ask").hidden = false;
  $("btn-map-ask").setAttribute("aria-expanded", "false");
  showTries(false);
  say("");
  setLogOpen(false);
}

// ── who reads the words ─────────────────────────────────────────────────────

function context() {
  const bits = [currentContext()];
  if (state.date) bits.push(`The map date is ${state.date}.`);
  return bits.filter(Boolean).join(" ");
}

async function byRules(text) {
  return callLight("map_command", { op: "parse", text, today: todayIso() }, { priority: 3 });
}

async function byDevice(text) {
  if (!(await localModelReady())) return null;
  const prompt = await callLight("map_command", { op: "prompt", context: context(), today: todayIso() }, { priority: 3 });
  const reply = await generateJsonLocally({ system: prompt.system, prompt: text, schema: prompt.schema, timeoutMs: 25000,
    onProgress: (m) => say(`<span class="spinner" aria-hidden="true"></span>${escapeHtml(m)}`) });
  const checked = await callLight("map_command", { op: "reply", reply, today: todayIso() }, { priority: 3 });
  return { ...checked, model: localReaderLabel() || "the on-device model" };
}

async function byKey(text) {
  const cfg = askModelConfig();
  if (!cfg) return null;
  say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(cfg.label)}…`);
  // The main worker: the provider client lives beside the Analyst there.
  const res = await call("map_command", {
    op: "llm", text, context: context(), today: todayIso(),
    provider: cfg.provider, model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url,
  });
  if (res && res.error) throw new Error(res.error);
  return res;
}

// Place names to boxes and points (Photon, in the worker). A name the gazetteer does not know may be a
// gauge: the catalogue search the header uses answers that.
async function resolve(actionsIn) {
  if (!actionsIn.some((a) => a.place)) return { actions: actionsIn, said: null, notes: [], credit: null };
  const res = await callLight("map_command", { op: "resolve", actions: actionsIn }, { priority: 3 });
  const gone = actionsIn.filter((a) => a.type === "fly_to" && a.place)
    .filter((a) => !res.actions.some((b) => b.type === "fly_to" && b.label));
  for (const a of gone) {
    const hit = searchStations(a.place, 1)[0];
    if (!hit) continue;
    res.actions.unshift({ type: "fly_to", center: [hit.lat, hit.lon], zoom: 11, label: hit.name || hit.station_id, where: "gauge" });
    res.said = [`Fly to ${hit.name || hit.station_id} (gauge)`, ...(res.said || [])];
    res.notes = res.notes.filter((n) => !n.includes(a.place));
  }
  return res;
}

async function runCommand(text) {
  const my = ++run;
  const words = String(text || "").trim();
  if (!words) return;
  showTries(false);
  say(`<span class="spinner" aria-hidden="true"></span>${state.workerReady ? "Reading…" : "Starting the engine (once), then reading…"}`);
  try {
    let res = await byRules(words);
    if (my !== run) return;
    if (res && res.control) {
      const n = res.control === "undo_all" ? undoAll() : (undoLast() ? 1 : 0);
      say(n ? (res.control === "undo_all" ? "Every map action undone." : "Undone.") : "Nothing to undo.", "ok");
      $("ma-input").value = "";
      return;
    }
    let by = "rules", who = ["Rules", "no model"];
    let resolved = res && res.matched ? await resolve(res.actions) : null;
    // Matched words whose places the gazetteer does not hold ("show me something clever") go to a model.
    if (!resolved || !resolved.actions.length) {
      const device = await byDevice(words).catch((err) => ({ error: err.message }));
      if (my !== run) return;
      if (device && device.actions && device.actions.length) {
        res = device; by = "device"; who = [device.model, "on this device"];
      } else {
        const key = await byKey(words).catch((err) => ({ error: err.message }));
        if (my !== run) return;
        if (key && key.actions && key.actions.length) { res = key; by = "key"; who = [key.model, "your key"]; }
        else {
          const why = (resolved && resolved.notes && resolved.notes[0]) || (key && key.error) || (device && device.error) || "";
          const how = askModelConfig() ? "" : " With Chrome's built-in model, or a key in Ask ✨, free-form requests work too.";
          say(`Not understood${why ? ` (${escapeHtml(why)})` : ""}. Try one of these.${escapeHtml(how)}`, "warn");
          showTries(true);
          return;
        }
      }
      resolved = await resolve(res.actions);
      if (my !== run) return;
    }
    const said = resolved.said || res.said || [];
    const { applied, failed } = await applyActions(resolved.actions, { by, said, command: words });
    if (my !== run) return;
    // What the checks refused from a model's answer is said, not dropped in silence.
    const refused = by === "rules" ? [] : (res.errors || []).map((e) => `refused ${e.replace(/^action \d+: /, "")}`);
    const notes = [...refused, ...(resolved.notes || []), ...failed.map((f) => f.error)];
    const credit = resolved.credit ? `<span class="ma-credit">${escapeHtml(resolved.credit)}</span>` : "";
    const n = applied.length;
    say(n
      ? `<b class="ma-who">${escapeHtml(who[0])}</b><span class="ma-how">${escapeHtml(who[1])}</span>` +
        `<span>${n} action${n === 1 ? "" : "s"} on the map</span>` +
        (notes.length ? `<span class="ma-note">${escapeHtml(notes.join("; "))}</span>` : "") + credit
      : `Nothing could be done: ${escapeHtml(notes.join("; ") || "no action ran")}.`, n ? "ok" : "warn");
    if (applied.length) $("ma-input").value = "";
    renderLog();
    // A phone: the open box covers the top of a small map, so it steps aside once the map has answered.
    // The count beside the button keeps every action a tap from its undo.
    if (n && narrow()) setTimeout(() => { if (my === run && open) closeMapAsk(); }, 1600);
  } catch (err) {
    if (my === run) say(`Could not run that: ${escapeHtml(err.message)}`, "error");
  }
}

export function initMapCommand() {
  const root = box();
  if (!root) return;
  $("btn-map-ask").addEventListener("click", () => openMapAsk());
  if (narrow()) $("ma-input").placeholder = "Ask the map…";
  // On a phone the count opens the box with its log (a log alone would hang off a button).
  $("ma-log-toggle").addEventListener("click", (e) => {
    if (!narrow() || open) return;
    e.stopImmediatePropagation();
    openMapAsk();
  }, true);
  $("ma-form").addEventListener("submit", (e) => { e.preventDefault(); void runCommand($("ma-input").value); });
  root.querySelector(".ma-x").addEventListener("click", closeMapAsk);
  $("ma-input").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closeMapAsk(); $("btn-map-ask").focus(); }
  });
  $("ma-input").addEventListener("input", () => { if (!$("ma-input").value) showTries(true); });
  $("ma-tries").addEventListener("click", (e) => {
    const b = e.target.closest(".ma-try");
    if (!b) return;
    $("ma-input").value = b.textContent;
    void runCommand(b.textContent);
  });
  // The / key opens the box anywhere outside a field (the search keeps its own box a click away).
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
    const a = document.activeElement;
    if (a && (/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) || a.isContentEditable)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    openMapAsk();
  }, true);
  globalThis.__aq.mapAsk = { run: runCommand, open: openMapAsk, close: closeMapAsk };
}
