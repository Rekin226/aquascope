// The time bar (#522): one date control on the map that every dated layer
// follows. Back and forward a step, play through a range, and behind "more":
// the step, the range, swipe compare and a GIF of the range.
//
// The date itself is state.date, changed only through setTime() in core.js.
// This module listens like any other subscriber; a chart click (charts.js), a
// pasted link (url.js) or an agent (webmcp.js) moves the date the same way.
// The date arithmetic is in timeline.js, which node tests directly.
//
// A model result dropped on the map (#555, mesh-view.js) brings its own clock: hourly or finer steps over a
// few days, often decades ago. While it is on, the bar steps and plays through the model's own times
// (useModelClock), says so in its note, and "Back to dates" returns it to the map date.

import { $, actions, datedExtras, onTime, setTime, state } from "./core.js?v=__BUILD__";
import { OVERLAYS, datedLayersOn } from "./layers.js?v=__BUILD__";
import { applyDate, whenSettled } from "./map.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { initCompare, syncCompare } from "./compare-map.js?v=__BUILD__";
import {
  GIF_FRAMES, MAX_FRAMES, addStep, clampDate, defaultRange, frameDates, isIsoDate, layersMissing,
  missingNote, nextFrame, normaliseRange, shortDate, todayIso,
} from "./timeline.js?v=__BUILD__";

// null follows the layers (shown while a dated layer is on); true or false is
// the reader's own choice from the clock button.
let userOpen = null;
let note = null;          // { text, until } a passing note: "Map set to 14 Jul 2021"
let noteTimer = null;
let playToken = 0;
let gifRun = null;        // { cancel } while a GIF is being made
// The model's own clock while a model result is on (#555): { times, index, label(i), summary, onStep(i) ->
// promise, onExit() }; null on the map date.
let clock = null;
let clockToken = 0;
let clockPlaying = false;

const datedOn = () => [...datedLayersOn(state.basemap, state.overlays), ...datedExtras.values()];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const effectiveRange = () => normaliseRange(state.timeRange) || defaultRange(state.date, state.timeStep);

function barVisible() {
  if (!state.mapOk) return false;
  if (clock) return userOpen !== false;
  return userOpen === null ? datedOn().length > 0 : userOpen;
}

// ── the note under the date ─────────────────────────────────────────────────

function showNote(text, ms = 5000) {
  note = { text };
  clearTimeout(noteTimer);
  if (ms) noteTimer = setTimeout(() => { note = null; renderNote(); }, ms);
  renderNote();
}

function renderNote() {
  const box = $("tb-note");
  if (!box) return;
  box.innerHTML = "";
  let text = note && note.text;
  if (clock) {
    const back = document.createElement("button");
    back.type = "button";
    back.className = "tb-chip";
    back.textContent = "Back to dates";
    back.addEventListener("click", () => releaseModelClock({ exit: true }));
    box.append(document.createTextNode(text || clock.summary), back);
    box.hidden = false;
    return;
  }
  if (!text) {
    const dated = datedOn();
    if (!dated.length) {
      text = "No dated layer on.";
      for (const [label, apply] of [["Rain", () => actions.setOverlay("precip", true)],
        ["Satellite", () => actions.setBasemap("daily")]]) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "tb-chip";
        chip.textContent = label;
        chip.addEventListener("click", apply);
        box.append(chip);
      }
    } else {
      const missing = layersMissing(dated, state.date);
      if (missing.length) text = missingNote(missing[0], state.date);
    }
  }
  if (text) box.prepend(document.createTextNode(text));
  box.hidden = !box.childNodes.length;
}

// ── keeping the bar in sync with the state ──────────────────────────────────

export function syncTimeBar({ layersChanged = false } = {}) {
  const bar = $("timebar");
  if (!bar) return;
  // Turning a dated layer on brings a bar the reader had closed back.
  if (layersChanged && userOpen === false && datedOn().length) userOpen = null;
  if (layersChanged) syncCompare();
  bar.hidden = !barVisible();
  const btn = $("btn-time");
  if (btn) {
    btn.hidden = !state.mapOk;
    btn.setAttribute("aria-pressed", bar.hidden ? "false" : "true");
  }
  if (bar.hidden) { closeMore(); return; }
  bar.classList.toggle("tb-clock", Boolean(clock));
  $("tb-date").hidden = Boolean(clock);
  $("tb-more").hidden = Boolean(clock);
  const model = $("tb-model");
  if (model) model.hidden = !clock;
  if (clock) {
    closeMore();
    model.textContent = clock.label(clock.index);
    const label = clockPlaying ? "Pause" : "Play";
    $("tb-play").setAttribute("aria-pressed", clockPlaying ? "true" : "false");
    $("tb-play").setAttribute("aria-label", label);
    $("tb-play").title = label;
    $("tb-prev").disabled = clock.index <= 0;
    $("tb-next").disabled = clock.index >= clock.times.length - 1;
    renderNote();
    placeBar();
    return;
  }
  $("tb-prev").disabled = false;
  const today = todayIso();
  $("tb-date").value = state.date || "";
  $("tb-date").max = today;
  const label = state.playing ? "Pause" : "Play";
  $("tb-play").setAttribute("aria-pressed", state.playing ? "true" : "false");
  $("tb-play").setAttribute("aria-label", label);
  $("tb-play").title = label;
  $("tb-next").disabled = Boolean(state.date && addStep(state.date, state.timeStep, 1) > today);
  for (const input of document.querySelectorAll("#tb-steps input")) input.checked = input.value === state.timeStep;
  const range = effectiveRange();
  $("tb-from").value = range.from;
  $("tb-to").value = range.to;
  $("tb-from").max = $("tb-to").max = today;
  const cmp = state.compare;
  $("tb-compare").checked = Boolean(cmp);
  $("tb-cmp-date").disabled = $("tb-cmp-layer").disabled = !cmp;
  $("tb-cmp-date").value = cmp ? cmp.date : "";
  $("tb-cmp-date").max = today;
  $("tb-cmp-layer").value = cmp && cmp.layer ? cmp.layer : "";
  const frames = frameDates(range.from, range.to, state.timeStep, GIF_FRAMES);
  if (!gifRun) {
    $("tb-gif").textContent = "Make a GIF";
    $("tb-gif-status").textContent = `${frames.dates.length}${frames.truncated ? "+" : ""} frames`;
  }
  renderNote();
  placeBar();
}

// The bar sits centred under the part of the map you can see: clear of the
// inspector on a wide screen, above the bottom sheet on a phone. Measured here
// rather than taken from panelPadding(), which caps itself for the camera's sake
// and would leave the bar under a tall sheet.
function placeBar() {
  const wrap = $("timebar-wrap");
  const mapEl = $("map");
  if (!wrap || !mapEl) return;
  const frame = mapEl.getBoundingClientRect();
  let right = 0, bottom = 0;
  for (const id of ["panel", "drawer"]) {
    const el = $(id);
    if (!el || el.hidden || el.offsetParent === null) continue;
    if (id === "panel" && document.body.classList.contains("panel-collapsed")) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height || r.top >= frame.bottom || r.left >= frame.right) continue;
    if (r.width < frame.width * 0.75) right = Math.max(right, frame.right - r.left);
    else bottom = Math.max(bottom, frame.bottom - r.top);
  }
  wrap.style.right = `${Math.round(Math.min(right, Math.max(0, frame.width - 320)))}px`;
  wrap.style.bottom = `${Math.round(bottom) + (bottom ? 10 : 64)}px`;
}

// ── moving the date ─────────────────────────────────────────────────────────

function stopPlay() {
  if (!state.playing) return;
  playToken++;
  setTime({ playing: false }, { source: "bar" });
}

// ── the model's own clock (#555) ────────────────────────────────────────────

/** Hand the bar to a model's own times; the previous clock, if any, is let go. */
export function useModelClock(next) {
  stopPlay();
  stopClock();
  clock = { index: 0, ...next };
  userOpen = null;
  syncTimeBar();
}

/** Back to the map date; exit: true also tells the model view (the reader pressed "Back to dates"). */
export function releaseModelClock({ exit = false } = {}) {
  if (!clock) return;
  const was = clock;
  stopClock();
  clock = null;
  syncTimeBar();
  if (exit && was.onExit) was.onExit();
}

export const modelClockOn = () => Boolean(clock);

/** Move the model clock (the legend's buttons, the card): draws through the clock's own onStep. */
export function setModelStep(i, { fromPlay = false } = {}) {
  if (!clock) return Promise.resolve();
  const k = Math.max(0, Math.min(clock.times.length - 1, Math.round(i)));
  if (!fromPlay) stopClock();
  clock.index = k;
  syncTimeBar();
  return Promise.resolve(clock.onStep ? clock.onStep(k) : null);
}

function stopClock() {
  clockToken++;
  if (clockPlaying) { clockPlaying = false; syncTimeBar(); }
}

async function playClock() {
  const token = ++clockToken;
  clockPlaying = true;
  let i = clock.index >= clock.times.length - 1 ? 0 : clock.index;
  while (clock && token === clockToken) {
    const started = performance.now();
    await setModelStep(i, { fromPlay: true });
    await sleep(Math.max(60, 380 - (performance.now() - started)));
    if (!clock || token !== clockToken) return;
    i = (i + 1) % clock.times.length;
  }
}

function step(n) {
  if (clock) { setModelStep(clock.index + n); return; }
  if (!state.date) return;
  stopPlay();
  setTime({ date: clampDate(addStep(state.date, state.timeStep, n), null, todayIso()) }, { source: "bar" });
}

async function play() {
  const token = ++playToken;
  let range = normaliseRange(state.timeRange);
  if (!range) {   // a play needs a range; the default one becomes the reader's, so the link replays it
    range = defaultRange(state.date, state.timeStep);
    setTime({ range }, { source: "bar" });
  }
  const { dates } = frameDates(range.from, range.to, state.timeStep, MAX_FRAMES);
  if (!dates.length) return;
  if (!datedOn().length) actions.setOverlay("precip", true);
  setTime({ playing: true }, { source: "play" });
  let d = dates.includes(state.date) && state.date !== dates[dates.length - 1] ? state.date : dates[0];
  while (token === playToken) {
    setTime({ date: d }, { source: "play" });
    await whenSettled(3000);
    await sleep(state.timeStep === "day" ? 450 : 700);
    if (token !== playToken) return;
    d = nextFrame(dates, d);
  }
}

// ── "more": step, range, compare, GIF ───────────────────────────────────────

function closeMore() {
  const panel = $("tb-panel");
  if (panel) panel.hidden = true;
  const more = $("tb-more");
  if (more) more.setAttribute("aria-expanded", "false");
}

function toggleMore() {
  const panel = $("tb-panel");
  const open = panel.hidden;
  panel.hidden = !open;
  $("tb-more").setAttribute("aria-expanded", open ? "true" : "false");
}

function readRangeInputs() {
  const from = $("tb-from").value, to = $("tb-to").value;
  if (isIsoDate(from) && isIsoDate(to)) setTime({ range: normaliseRange({ from, to }) }, { source: "bar" });
}

async function gif() {
  if (gifRun) { gifRun.cancel(); return; }
  stopPlay();
  const range = effectiveRange();
  const { dates, truncated } = frameDates(range.from, range.to, state.timeStep, GIF_FRAMES);
  if (!dates.length) return;
  if (!datedOn().length) actions.setOverlay("precip", true);
  const status = $("tb-gif-status");
  const run = { cancelled: false, cancel() { this.cancelled = true; } };
  gifRun = run;
  $("tb-gif").textContent = "Stop";
  const back = state.date;
  let message = "";
  try {
    const { makeGif } = await import("./gif.js?v=__BUILD__");
    const label = datedOn().map((l) => l.label).join(" + ");
    // Who to credit on each frame: NASA GIBS for the satellite layers, a layer's own `credit` otherwise.
    const credit = [...new Set(datedOn().map((l) => l.credit || "NASA GIBS"))].join(" · ");
    const done = await makeGif({
      dates, label, credit, step: state.timeStep, isCancelled: () => run.cancelled,
      setDate: (d) => setTime({ date: d }, { source: "gif" }),
      onProgress: (i, n) => { status.textContent = `Frame ${i} of ${n}${truncated ? ` (first ${n})` : ""}`; },
    });
    message = done ? `Saved, ${dates.length} frames.` : "Stopped.";
  } catch (err) {
    console.error(err);
    message = `Could not make the GIF: ${err.message}`;
  } finally {
    gifRun = null;
    setTime({ date: back }, { source: "gif" });   // redraws the bar, so the message goes after it
    $("tb-gif").textContent = "Make a GIF";
    status.textContent = message;
  }
}

function fillCompareLayers() {
  const select = $("tb-cmp-layer");
  select.innerHTML = '<option value="">Same layers</option>' +
    OVERLAYS.filter((o) => o.time).map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
}

function setCompare(on) {
  if (!on) { setTime({ compare: null }, { source: "bar" }); return; }
  const date = (state.compare && state.compare.date) || clampDate(addStep(state.date, "month", -12), null, todayIso());
  setTime({ compare: { date, layer: $("tb-cmp-layer").value || null } }, { source: "bar" });
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initTimeBar() {
  if (!$("timebar")) return;
  fillCompareLayers();
  $("btn-time").addEventListener("click", () => {
    userOpen = $("timebar").hidden;
    syncTimeBar();
  });
  $("tb-prev").addEventListener("click", () => step(-1));
  $("tb-next").addEventListener("click", () => step(1));
  $("tb-date").addEventListener("change", (e) => {
    if (!isIsoDate(e.target.value)) return;
    stopPlay();
    setTime({ date: clampDate(e.target.value, null, todayIso()) }, { source: "bar" });
  });
  $("tb-play").addEventListener("click", () => {
    if (clock) { if (clockPlaying) stopClock(); else playClock(); return; }
    if (state.playing) stopPlay(); else play();
  });
  $("tb-more").addEventListener("click", toggleMore);
  $("tb-panel").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeMore(); $("tb-more").focus(); }
  });
  for (const input of document.querySelectorAll("#tb-steps input")) {
    input.addEventListener("change", () => { stopPlay(); setTime({ step: input.value }, { source: "bar" }); });
  }
  $("tb-from").addEventListener("change", readRangeInputs);
  $("tb-to").addEventListener("change", readRangeInputs);
  $("tb-compare").addEventListener("change", (e) => setCompare(e.target.checked));
  $("tb-cmp-date").addEventListener("change", (e) => {
    if (state.compare && isIsoDate(e.target.value)) setTime({ compare: { ...state.compare, date: e.target.value } });
  });
  $("tb-cmp-layer").addEventListener("change", (e) => {
    if (state.compare) setTime({ compare: { ...state.compare, layer: e.target.value || null } });
  });
  $("tb-gif").addEventListener("click", gif);

  onTime((t) => {
    if (t.date !== t.prev.date) applyDate(t.date, [...state.overlays], state.basemap);
    if (t.source === "chart" || t.source === "agent") {
      stopPlay();
      userOpen = true;
      // Something on the map has to change with the date, or the jump is invisible.
      if (!datedOn().length) actions.setOverlay("precip", true);
      // A hydrograph runs back further than the satellites: say so when nothing on the map has that day.
      const dated = datedOn();
      const missing = layersMissing(dated, t.date);
      const why = dated.length && missing.length === dated.length ? ` ${missingNote(missing[0], t.date)}` : "";
      showNote(`Map set to ${shortDate(t.date)}.${why}`, why ? 8000 : 5000);
    }
    syncTimeBar();
    if (t.source !== "play" && t.source !== "gif") writeUrl();
  });

  // The cards move for reasons no click reports (a resize, the panel collapsing).
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => placeBar());
    for (const id of ["panel", "drawer", "map"]) { const el = $(id); if (el) ro.observe(el); }
  }
  if (typeof MutationObserver !== "undefined") {
    new MutationObserver(() => placeBar()).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  }
  initCompare();
  syncTimeBar();
}
