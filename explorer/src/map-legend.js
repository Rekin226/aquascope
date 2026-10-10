// "On the map" (#543 design pass): one compact legend for every layer on the
// globe, bottom left (top right on a phone), in place of a card per layer.
//
// A layer module registers a row: a line with a visibility toggle, the layer's
// mark, its name and a few words (the month, a count), which opens on its key
// (the layer's own body: colours, counts, play, about). A layer with nothing to
// draw is one muted line. On a phone the legend starts folded to a small chip.
// The rules (order, states, the chip's words) are in map-legend-core.js.

import { $, escapeHtml } from "./core.js?v=__BUILD__";
import { chipLabel, rowState, sortRows, startsFolded, startsOpen } from "./map-legend-core.js?v=__BUILD__";

const rows = new Map();   // id -> { def, open }
let folded = null;        // null until the first render decides from the screen width

const host = () => $("map-legend");
const width = () => (globalThis.innerWidth || 1024);

const CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" ' +
  'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const LAYERS = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4 3 9l9 5 9-5-9-5Z" fill="none" stroke="currentColor" ' +
  'stroke-width="1.8" stroke-linejoin="round"/><path d="m3 14 9 5 9-5" fill="none" stroke="currentColor" stroke-width="1.8" ' +
  'stroke-linejoin="round"/></svg>';

/**
 * Add a layer's row. `def`: { id, title, mark() -> html, summary() -> text, on() -> bool, empty() -> bool,
 * toggle(on) (left out: no toggle, as for a lit river, which `clear()` takes away), body() -> html (its key,
 * shown when the row is open), act(name, button) for the body's buttons (data-act), shown() -> bool
 * (left out: always listed) }.
 */
export function registerLegendRow(def) {
  if (!def || !def.id) return;
  const had = rows.get(def.id);
  rows.set(def.id, { def, open: had ? had.open : startsOpen(def.id, width()) });
  render();
}

/** Re-draw one row (or all of them): a module calls this wherever it used to redraw its own card. */
export function refreshLegend(id) {
  const el = host();
  if (!el) return;
  if (!id || !el.querySelector(`.ml-row[data-id="${CSS.escape(id)}"]`)) { render(); return; }
  const r = rows.get(id);
  if (!r) return;
  const row = el.querySelector(`.ml-row[data-id="${CSS.escape(id)}"]`);
  const fresh = rowNode(r);
  if (!fresh) { render(); return; }
  row.replaceWith(fresh);
  syncChip();
}

/** Open a row's key, as when a lit river wants its words seen. */
export function openLegendRow(id, open = true) {
  const r = rows.get(id);
  if (!r) return;
  r.open = Boolean(open);
  if (open && folded && !startsFolded(width())) folded = false;
  render();
}

/** The body element of a row, while it is open: for a module that patches a busy mark in place. */
export function legendBody(id) {
  const el = host();
  return el ? el.querySelector(`.ml-row[data-id="${CSS.escape(id)}"] .ml-body`) : null;
}

function stateOf(def) {
  return rowState({ on: def.toggle ? def.on() : true, empty: def.empty ? def.empty() : false });
}

function visibleRows() {
  return sortRows([...rows.values()].map((r) => ({ id: r.def.id, r })))
    .map((x) => x.r).filter((r) => !r.def.shown || r.def.shown());
}

function rowNode(r) {
  const d = r.def;
  if (d.shown && !d.shown()) return null;
  const st = stateOf(d);
  const canOpen = st === "on" && Boolean(d.body);
  const open = canOpen && r.open;
  const bodyId = `ml-body-${d.id}`;
  const node = document.createElement("div");
  node.className = `ml-row ${st}${open ? " open" : ""}`;
  node.dataset.id = d.id;
  const sum = st === "off" ? "off" : (d.summary ? d.summary() : "");
  const toggle = d.toggle
    ? `<input type="checkbox" class="ml-toggle" ${st === "off" ? "" : "checked"} aria-label="Show ${escapeHtml(d.title)}" ` +
      `title="${st === "off" ? "Show" : "Hide"} ${escapeHtml(d.title)}">`
    : '<button type="button" class="ml-clear" data-act="clear" aria-label="Clear" title="Clear">×</button>';
  const name = `<span class="ml-title">${escapeHtml(d.title)}</span>` +
    (sum ? `<span class="ml-sum">${escapeHtml(sum)}</span>` : "");
  node.innerHTML =
    `<div class="ml-line">${toggle}<span class="ml-mark" aria-hidden="true">${d.mark ? d.mark() : ""}</span>` +
    (canOpen
      ? `<button type="button" class="ml-name" aria-expanded="${open ? "true" : "false"}" aria-controls="${bodyId}" ` +
        `title="${open ? "Fold" : "What it shows"}">${name}<span class="ml-chev">${CHEVRON}</span></button>`
      : `<span class="ml-name static">${name}</span>`) +
    `</div>` +
    (open ? `<div class="ml-body" id="${bodyId}">${d.body()}</div>` : "");
  return node;
}

function syncChip() {
  const el = host();
  if (!el) return;
  const chip = el.querySelector(".ml-chip");
  if (!chip) return;
  const label = chipLabel(visibleRows().map((r) => stateOf(r.def)));
  chip.querySelector(".ml-chip-text").textContent = label;
  chip.setAttribute("aria-expanded", folded ? "false" : "true");
}

function render() {
  const el = host();
  if (!el) return;
  if (folded === null) folded = startsFolded(width());
  const list = visibleRows();
  el.hidden = !list.length;
  el.classList.toggle("folded", folded);
  el.innerHTML =
    `<button type="button" class="ml-chip" aria-expanded="${folded ? "false" : "true"}" aria-controls="ml-rows" ` +
    `title="${folded ? "Show the legend" : "Fold the legend"}">${LAYERS}<span class="ml-chip-text"></span>` +
    `<span class="ml-chev">${CHEVRON}</span></button>` +
    `<div class="ml-rows" id="ml-rows"${folded ? " hidden" : ""}></div>`;
  const box = el.querySelector(".ml-rows");
  for (const r of list) {
    const n = rowNode(r);
    if (n) box.appendChild(n);
  }
  syncChip();
}

export function initMapLegend() {
  const el = host();
  if (!el) return;
  el.addEventListener("click", (e) => {
    const chip = e.target.closest(".ml-chip");
    if (chip) { folded = !folded; render(); el.querySelector(".ml-chip")?.focus(); return; }
    const row = e.target.closest(".ml-row");
    if (!row) return;
    const r = rows.get(row.dataset.id);
    if (!r) return;
    const name = e.target.closest("button.ml-name");
    if (name) {
      r.open = !r.open;
      refreshLegend(r.def.id);
      el.querySelector(`.ml-row[data-id="${CSS.escape(r.def.id)}"] button.ml-name`)?.focus();
      return;
    }
    const btn = e.target.closest("button[data-act]");
    if (btn && r.def.act) r.def.act(btn.dataset.act, btn, e);
  });
  el.addEventListener("change", (e) => {
    const input = e.target.closest("input[data-act]");
    if (input) {
      const r = rows.get(input.closest(".ml-row").dataset.id);
      if (r && r.def.act) r.def.act(input.dataset.act, input, e);
      return;
    }
    const t = e.target.closest(".ml-toggle");
    if (!t) return;
    const r = rows.get(t.closest(".ml-row").dataset.id);
    if (r && r.def.toggle) r.def.toggle(t.checked);
    refreshLegend(r && r.def.id);
  });
  render();
}
