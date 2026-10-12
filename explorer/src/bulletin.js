// The monthly bulletin (#523) in the Explorer: the Bulletin entry in Tools opens
// the latest one in a reader (the Studio report reader's layout: the document in
// a sandboxed frame, Print or save as PDF beside it), and "Last month's status"
// colours the gauges from bulletins/<month>/status.parquet. Both read what the
// monthly workflow published (aquascope.bulletin); nothing is computed here.
// Before the first bulletin exists, both say so and the gauges keep their colours.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, escapeHtml, state } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { STATUS_CLASSES, NO_STATUS_COLOR } from "./now-core.js?v=__BUILD__";
import { NO_BULLETIN_TEXT, bulletinLegendLine, bulletinPaths, latestEntry, monthLabel } from "./bulletin-core.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { announce, captureFocus } from "./a11y.js?v=__BUILD__";

let indexLoading = null;
let statusLoading = null;
let reader = null;   // the reader's modal, made on first open

// bulletins/index.json, read once per visit; null when nothing is published yet.
function loadIndex() {
  if (!indexLoading) {
    indexLoading = fetch(`${CONFIG.bulletinsBase}index.json`, { cache: "no-cache" })
      .then((res) => (res.ok ? res.json() : null))
      .then((idx) => latestEntry(idx))
      .catch((err) => { console.warn("bulletin index:", err && err.message); indexLoading = null; return null; });
  }
  return indexLoading;
}

// Fills state.bulletinStatus (key -> { cls, pct }) and state.bulletinMeta (the
// index entry), or bulletinMeta = { missing: true } before the first bulletin.
export function ensureBulletinStatus() {
  if (statusLoading) return statusLoading;
  statusLoading = (async () => {
    const entry = await loadIndex();
    if (!entry) {
      state.bulletinMeta = { missing: true };
      return;
    }
    const { conn } = await duck();
    const url = bulletinPaths(CONFIG.bulletinsBase, entry.month).status;
    const table = await conn.query(`SELECT source, station_id, percentile, "class" AS cls FROM read_parquet('${url}')`);
    const map = new Map();
    for (const row of table.toArray().map((r) => r.toJSON())) {
      map.set(`${row.source}/${row.station_id}`, { cls: row.cls, pct: Number(row.percentile) });
    }
    state.bulletinMeta = entry;
    state.bulletinStatus = map;
  })().catch((err) => {
    console.warn("bulletin status:", err && err.message);
    state.bulletinMeta = { missing: true };
    statusLoading = null;   // a later pick tries again
  });
  return statusLoading;
}

export function bulletinLegendHtml() {
  const swatch = (c, l) => `<span class="sw"><i style="background:${c}"></i>${escapeHtml(l)}</span>`;
  const meta = state.bulletinMeta;
  if (!meta) return `<p class="muted now-legend-note">Loading last month's bulletin…</p>`;
  const line = escapeHtml(bulletinLegendLine(meta));
  if (meta.missing) return `<p class="muted now-legend-note">${line}</p>`;
  return STATUS_CLASSES.map((c) => swatch(c.color, c.label)).join("") + swatch(NO_STATUS_COLOR, "not classed") +
    `<p class="muted now-legend-note">${line} <button type="button" class="link-btn" data-open-bulletin>Read it</button></p>`;
}

// ── the reader ──────────────────────────────────────────────────────────────

function readerBox() {
  if (reader) return reader;
  const box = document.createElement("div");
  box.id = "bulletin-reader";
  box.className = "modal study-reader";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.setAttribute("aria-label", "Monthly bulletin");
  box.hidden = true;
  document.body.appendChild(box);
  box.addEventListener("click", (e) => {
    const act = e.target.closest("[data-reader]");
    if (e.target === box || (act && act.dataset.reader === "close")) { closeReader(); return; }
    if (act && act.dataset.reader === "print") {
      const frame = box.querySelector("iframe");
      if (frame && frame.contentWindow) frame.contentWindow.print();
    }
  });
  reader = box;
  return box;
}

function closeReader() {
  const box = reader;
  if (!box || box.hidden) return;
  box.hidden = true;
  if (box.__release) { box.__release(); box.__release = null; }
}

export async function openBulletin() {
  const entry = await loadIndex();
  if (!entry) {
    openModal("Bulletin", `<p>${escapeHtml(NO_BULLETIN_TEXT)}</p>`);
    return;
  }
  const paths = bulletinPaths(CONFIG.bulletinsBase, entry.month);
  const label = entry.label || monthLabel(entry.month);
  let html;
  try {
    const res = await fetch(paths.html);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    html = await res.text();
  } catch (err) {
    openModal("Bulletin", `<p>The ${escapeHtml(label)} bulletin could not be opened (${escapeHtml(err.message)}). ` +
      `It is also at <a href="${escapeHtml(paths.html)}" target="_blank" rel="noopener">the Archive</a>.</p>`);
    return;
  }
  const box = readerBox();
  box.innerHTML =
    `<div class="modal-box study-reader-box"><div class="modal-head"><h2>State of the rivers, ${escapeHtml(label)}</h2>` +
    `<div class="row-actions"><button type="button" class="btn" data-reader="print">Print or save as PDF</button>` +
    `<a class="btn" href="${escapeHtml(paths.markdown)}" target="_blank" rel="noopener">Markdown</a>` +
    `<button type="button" class="btn" data-reader="close" aria-label="Close">Close</button></div></div>` +
    `<div class="study-reader-body"><iframe title="Monthly bulletin" sandbox="allow-same-origin allow-modals"></iframe></div></div>`;
  box.querySelector("iframe").srcdoc = html;
  box.hidden = false;
  box.__release = captureFocus(box, { onEscape: closeReader, trap: true });
  box.querySelector('[data-reader="close"]').focus();
  announce(`State of the rivers, ${label}`);
}

export function initBulletin() {
  const btn = $("btn-bulletin");
  if (btn) {
    btn.addEventListener("click", () => {
      const menu = btn.closest("details");
      if (menu) menu.open = false;
      openBulletin();
    });
  }
  // "Read it" in the map legend.
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-open-bulletin]")) openBulletin();
  });
}
