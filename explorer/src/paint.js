// Paint the globe (#657): one question answered at every gauge in view, drawn on the map as it comes in.
//
// "Paint this view" (the brush in the map tools, or the AI action paint) plans the view from the catalog and
// the Archive's signatures.parquet (which gauges the Archive mirrors and for how long), downloads each
// gauge's daily file from the Archive several at a time, and hands them to the worker in batches:
// aquascope.paint answers each gauge (the same functions as `aquascope paint` and the MCP tool paint_map) and
// the field test over every gauge so far. Dots are painted batch by batch; gauges with no Archive record, too
// short a record or past the cap are drawn too, so the map says where it has no answer. A click on a dot opens
// its card (and from there its own record and study); the legend row carries the key, the verdict and the
// downloads: a PNG figure of the map, the table as CSV, and a methods paragraph with its references.
//
// paint-core.js decides the gauges, colours and words; every number comes from Python.

import { $, EMPTY_FC, actions, clickLayers, downloadBlob, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { CONFIG } from "../config.js?v=__BUILD__";
import { Cancelled, callCancelable, onWorkerRestart } from "./worker-client.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { logAction, registerSlotWriter, slotValue } from "./map-actions.js?v=__BUILD__";
import { signaturesUrl } from "./signature-filter-core.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import {
  ALPHA_GLOBAL, MAX_GAUGES, MAX_GAUGES_PHONE, MIN_YEARS, PERIODS, QUESTIONS, STATUS_STYLE, YEAR_CHOICES,
  archiveFileUrl, batchesOf, cardFor, figureTitle, fileStem, legendRows, paintLayout, paintProps, palette, planGauges,
  progressWords, questionById, summaryWords, viewBbox, wrapText,
} from "./paint-core.js?v=__BUILD__";

const SRC = "paint-gauges";
const LAYER = "paint-dots";
const GAUGE_LAYERS = ["clusters", "cluster-count", "points", "now-dots", "gauge-heat"];
const CONCURRENCY = 10;     // Archive files downloading at once (HTTP/2: one connection)
const BATCH = 8;            // gauges per worker call

// The painting on the map: its spec, the gauges by key in plan order, the field test and the final summary.
let shown = null;    // { spec, q, bbox, order: [keys], rows: Map, field, result, done, total, failed, phase }
let runId = 0;
let aborter = null;
let sigPromise = null;

const panel = () => $("paint");
const isPhone = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 640px)").matches);

// ── the Archive's signatures (which gauges it mirrors, and for how long) ────

function loadSignatures() {
  if (sigPromise) return sigPromise;
  sigPromise = (async () => {
    const { conn } = await duck();
    const url = signaturesUrl(CONFIG.stationsParquet).replace(/'/g, "''");
    const t = await conn.query(`SELECT source, station_id, record_years, data_years, bfi, record_start, record_end, notes
      FROM read_parquet('${url}') WHERE variable = 'discharge'`);
    const out = new Map();
    for (const r of t.toArray()) {
      const o = r.toJSON();
      out.set(`${o.source}/${o.station_id}`, {
        record_years: o.record_years === null ? null : Number(o.record_years),
        data_years: o.data_years === null ? null : Number(o.data_years),
        bfi: o.bfi === null ? null : Number(o.bfi),
        record_start: o.record_start || null, record_end: o.record_end || null, notes: o.notes || "",
      });
    }
    return out;
  })();
  sigPromise.catch(() => { sigPromise = null; });
  return sigPromise;
}

// ── the dots ────────────────────────────────────────────────────────────────

function features() {
  if (!shown) return EMPTY_FC;
  const q = shown.q;
  return {
    type: "FeatureCollection",
    features: shown.order.map((k) => shown.rows.get(k)).filter((r) => r && Number.isFinite(r.longitude)).map((r) => ({
      type: "Feature", geometry: { type: "Point", coordinates: [r.longitude, r.latitude] },
      properties: {
        key: r.key, status: r.status, fdr_significant: r.fdr_significant === true,
        cls: r.status === "painted" && r.value !== null && r.value !== undefined ? clsOf(q, r.value) : null,
      },
    })),
  };
}

function clsOf(q, v) {
  let i = 0;
  while (i < q.breaks.length && Number(v) >= q.breaks[i]) i++;
  return i;
}

function setGaugeMarks(visible) {
  if (!state.mapOk || !map) return;
  for (const id of GAUGE_LAYERS) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", visible && state.gaugesOn !== false ? "visible" : "none");
  }
}

function draw() {
  if (!state.mapOk || !map) return;
  const fc = features();
  if (!map.getSource(SRC)) {
    map.addSource(SRC, { type: "geojson", data: fc });
    map.addLayer({ id: LAYER, type: "circle", source: SRC, paint: shown ? paintProps(shown.q) : {} });
    clickLayers.add(LAYER);
    map.on("click", LAYER, onDotClick);
    map.on("mouseenter", LAYER, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", LAYER, () => { map.getCanvas().style.cursor = ""; });
  } else {
    map.getSource(SRC).setData(fc);
  }
  if (shown) {
    for (const [k, v] of Object.entries(paintProps(shown.q))) map.setPaintProperty(LAYER, k, v);
    for (const [k, v] of Object.entries(paintLayout(shown.q))) map.setLayoutProperty(LAYER, k, v);
    map.setLayoutProperty(LAYER, "visibility", "visible");
    if (map.getLayer(LAYER) && map.getStyle().layers.slice(-1)[0].id !== LAYER) map.moveLayer(LAYER);
  } else if (map.getLayer(LAYER)) {
    map.setLayoutProperty(LAYER, "visibility", "none");
  }
  // The painted dots stand in for the gauge marks while the painting is on.
  setGaugeMarks(!shown);
}

let watching = false;
function watchStyle() {
  if (watching || !state.mapOk || !map) return;
  watching = true;
  map.on("styledata", () => {
    if (shown && !map.getSource(SRC)) {
      try { draw(); } catch (err) { console.info("paint:", err && err.message); }
    } else if (shown) setGaugeMarks(false);
  });
}

function onDotClick(e) {
  if (!shown || !e.features || !e.features.length) return;
  if (e.originalEvent) e.originalEvent.stopPropagation();
  const key = e.features[0].properties.key;
  const row = shown.rows.get(key);
  if (!row || !actions.openMapCard) return;
  const c = cardFor(shown.q, row);
  const known = state.byKey && state.byKey.has(key);
  actions.openMapCard({
    id: `paint:${key}`, lngLat: [row.longitude, row.latitude], what: c.what, title: c.title, sub: c.sub,
    status: c.status, figure: c.figure, note: c.note, credit: creditFor(row.source),
    buttons: [
      { id: "pt-record", label: "Its record", title: "This gauge's record, floods and trend in the panel", hidden: !known,
        onClick: () => { actions.closeMapCard(); actions.selectStation(key, { fly: false }); } },
      { id: "pt-study", label: "Study", title: "A complete study at this gauge, with a report", hidden: !known,
        onClick: () => { actions.selectStation(key, { fly: false }); actions.openStudy({ fresh: true }); } },
    ],
  });
}

function creditFor(source) {
  const lic = shown && shown.result && (shown.result.licences || []).find((l) => l.source === source);
  return lic ? `AquaScope Archive: ${lic.label} (${lic.license})` : "AquaScope Archive";
}

// ── running a painting ──────────────────────────────────────────────────────

function currentBbox() {
  const b = map.getBounds();
  return viewBbox({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() });
}

function normSpec(spec) {
  const q = questionById(spec && spec.question) || QUESTIONS[0];
  const minYears = Number(spec && spec.min_years) > 0 ? Math.round(Number(spec.min_years)) : MIN_YEARS;
  const start = Number(spec && spec.start_year) > 0 ? Math.round(Number(spec.start_year)) : null;
  const cap = Number(spec && spec.max_gauges) > 0 ? Math.round(Number(spec.max_gauges)) : null;
  return { question: q.id, min_years: minYears, start_year: start, bbox: (spec && spec.bbox) || null, max_gauges: cap };
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchFile(url, signal) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(url, { signal });
    if (res.ok) return new Uint8Array(await res.arrayBuffer());
    if (res.status !== 429 && res.status < 500) throw new Error(`the Archive answered ${res.status}`);
    await wait(800 * (attempt + 1));
  }
  throw new Error("the Archive is busy; try again in a minute");
}

/**
 * Paint the view (or spec.bbox) with spec.question. Returns the spec painted. `log` puts it in the action log
 * (undo puts back the painting before it); `by` says who asked.
 */
export async function paintView(spec, { log = true, by = "rules", label = "", command = "" } = {}) {
  if (!state.mapOk || !map) throw new Error("the map is not ready");
  const s = normSpec(spec);
  const q = questionById(s.question);
  const bbox = s.bbox || currentBbox();
  const done = { ...s, bbox };
  if (log) {
    const before = slotValue("paint");
    logAction({ action: { type: "paint", question: q.id, bbox, min_years: s.min_years }, before, after: done,
      label: label || `Paint: ${q.short}`, by, command });
  }
  void run(done);
  return done;
}

async function run(spec) {
  const my = ++runId;
  if (aborter) aborter.abort();
  aborter = new AbortController();
  const signal = aborter.signal;
  const q = questionById(spec.question);
  watchStyle();
  shown = { spec, q, bbox: spec.bbox, order: [], readKeys: [], rows: new Map(), field: null, result: null, done: 0, total: 0, failed: 0,
    phase: "plan" };
  showPanel(true);
  renderPanel();
  draw();
  openLegendRow("paint", true);
  const live = () => my === runId && shown && shown.spec === spec;
  try {
    const sig = await loadSignatures();
    if (!live()) return;
    const plan = planGauges(state.stations || [], sig, {
      bbox: spec.bbox, minYears: spec.min_years, maxGauges: spec.max_gauges || capFor(),
    });
    shown.order = plan.rows.map((r) => r.key);
    for (const r of plan.rows) shown.rows.set(r.key, r);
    shown.total = plan.read.length;
    shown.readKeys = plan.read.map((r) => r.key);
    shown.phase = state.workerReady ? "read" : "boot";
    draw();
    renderPanel();
    refreshLegend("paint");
    const params = { min_years: spec.min_years, alpha_global: ALPHA_GLOBAL, ...(spec.start_year ? { start_year: spec.start_year } : {}) };
    await callCancelable("paint", { op: "start", question: q.id, params, bbox: spec.bbox, rows: plan.rows }).promise;
    if (!live()) return;
    shown.phase = "read";
    renderPanel();
    if (q.needs === "signatures") {
      for (const batch of batchesOf(plan.read, 200)) {
        const items = batch.map((r) => ({ ...meta(r), signature: sig.get(r.key) || {} }));
        const out = await callCancelable("paint", { op: "signatures", items }).promise;
        if (!live()) return;
        take(out);
      }
    } else {
      await readAndAnswer(plan.read, signal, live);
      if (!live()) return;
    }
    await finish(live);
  } catch (err) {
    failed(err, live);
  }
}

const capFor = () => (isPhone() ? MAX_GAUGES_PHONE : MAX_GAUGES);

function failed(err, live) {
  if (!live() || err instanceof Cancelled || (err && err.name === "AbortError")) return;
  shown.phase = "error";
  shown.error = err.message || String(err);
  renderPanel();
  refreshLegend("paint");
}

// The whole painting from the worker: the field test over every gauge, the counts, licences and methods.
async function finish(live) {
  shown.phase = "field";
  renderPanel();
  const result = await callCancelable("paint", { op: "summary" }).promise;
  if (!live()) return;
  shown.result = result;
  for (const r of result.rows || []) shown.rows.set(r.key, r);
  shown.field = result.field;
  shown.phase = "done";
  draw();
  renderPanel();
  refreshLegend("paint");
  announce(result.headline || "Painted");
}

/** Read the next gauges past the cap (the longest records first), as many again as the cap. */
export async function paintMore() {
  if (!shown || shown.phase !== "done" || shown.q.needs !== "daily") return;
  const more = shown.order.map((k) => shown.rows.get(k)).filter((r) => r && r.status === "over_cap").slice(0, capFor());
  if (!more.length) return;
  const my = runId;
  const spec = shown.spec;
  const live = () => my === runId && shown && shown.spec === spec;
  spec.max_gauges = (spec.max_gauges || capFor()) + more.length;   // the log's entry repaints the same gauges
  for (const r of more) Object.assign(r, { status: "queued", note: "" });
  shown.readKeys.push(...more.map((r) => r.key));
  shown.total += more.length;
  shown.result = null;
  shown.phase = "read";
  draw();
  renderPanel();
  refreshLegend("paint");
  try {
    await readAndAnswer(more, aborter.signal, live);
    if (!live()) return;
    await finish(live);
  } catch (err) {
    failed(err, live);
  }
}

function meta(r) {
  return { key: r.key, source: r.source, station_id: r.station_id, name: r.name, latitude: r.latitude,
    longitude: r.longitude, record_years: r.record_years };
}

// Download with a few at once, answer in batches as they land: the first dots appear within seconds.
async function readAndAnswer(list, signal, live) {
  const ready = [];
  let next = 0, landed = 0;
  let wake = null;
  const worker = async () => {
    while (next < list.length && live()) {
      const r = list[next++];
      const url = archiveFileUrl(CONFIG.stationsParquet, r.source, r.station_id);
      try {
        ready.push({ item: meta(r), buf: await fetchFile(url, signal) });
      } catch (err) {
        if (err && err.name === "AbortError") return;
        ready.push({ item: { ...meta(r), fetch_error: `could not download the Archive file (${err.message})` }, buf: new Uint8Array(0) });
      }
      landed++;
      if (wake) { const w = wake; wake = null; w(); }
    }
  };
  const fetchers = Array.from({ length: Math.min(CONCURRENCY, list.length) }, worker);
  const all = Promise.all(fetchers);
  const nap = () => new Promise((r) => { wake = r; setTimeout(r, 300); });
  let answered = 0;
  while (answered < list.length && live()) {
    if (!ready.length) { if (landed >= list.length) break; await nap(); continue; }
    // A short wait fills the batch; the worker is never left idle with files in hand for long.
    if (ready.length < BATCH && landed < list.length) await nap();
    const batch = ready.splice(0, BATCH);
    const out = await callCancelable("paint", {
      op: "gauges", items: batch.map((b) => b.item), buffers: batch.map((b) => b.buf),
    }).promise;
    if (!live()) return;
    answered += batch.length;
    take(out);
  }
  await all;
}

function take(out) {
  for (const r of (out && out.rows) || []) shown.rows.set(r.key, r);
  if (out && out.field) {
    shown.field = out.field;
    const keep = new Set(out.field.significant || []);
    for (const r of shown.rows.values()) {
      if (r.status === "painted" && r.p !== null && r.p !== undefined) r.fdr_significant = keep.has(r.key);
    }
  }
  shown.done = shown.readKeys.filter((k) => (shown.rows.get(k) || {}).status !== "queued").length;
  shown.failed = [...shown.rows.values()].filter((r) => r.status === "failed").length;
  draw();
  renderPanel();
  refreshLegend("paint");
}

export function stopPainting() {
  runId++;
  if (aborter) aborter.abort();
  if (shown && shown.phase !== "done") {
    shown.phase = "stopped";
    renderPanel();
    refreshLegend("paint");
  }
}

function clear() {
  runId++;
  if (aborter) aborter.abort();
  shown = null;
  draw();
  renderPanel();
  refreshLegend("paint");
  if (actions.closeMapCard) actions.closeMapCard();
}

// The legend's ×: the painting goes the way the log's undo would take it, so the log stays true.
function clearByUndo() {
  const before = slotValue("paint");
  clear();
  if (before) logAction({ action: { type: "paint", question: before.question, bbox: before.bbox }, before, after: null,
    label: "Clear the painting", by: "rules" });
}

// A restarted worker has lost the painting it held: hand it the rows back, so the downloads still work.
async function resync() {
  const my = runId, spec = shown.spec;
  try {
    const rows = shown.order.map((k) => shown.rows.get(k)).filter(Boolean);
    const params = { min_years: spec.min_years, alpha_global: ALPHA_GLOBAL, ...(spec.start_year ? { start_year: spec.start_year } : {}) };
    await callCancelable("paint", { op: "start", question: spec.question, params, bbox: spec.bbox, rows }).promise;
    if (my === runId) await callCancelable("paint", { op: "summary" }).promise;
  } catch (err) {
    console.info("paint: could not hand the painting back to the worker:", err && err.message);
  }
}

// ── the panel: the picker, the progress and the verdict ─────────────────────

function showPanel(on) {
  const p = panel();
  if (!p) return;
  p.hidden = !on;
  const btn = $("btn-paint");
  if (btn) btn.setAttribute("aria-expanded", on ? "true" : "false");
}

function picked() {
  const f = $("pt-form");
  const q = f.querySelector('input[name="pt-q"]:checked');
  const period = PERIODS.find((p) => p.id === $("pt-period").value) || PERIODS[0];
  return { question: q ? q.value : QUESTIONS[0].id, min_years: Number($("pt-years").value) || MIN_YEARS,
    start_year: period.start_year };
}

function askedWords(spec) {
  const q = questionById(spec.question);
  const period = PERIODS.find((p) => p.start_year === (spec.start_year || null));
  return `${q.short} · at least ${spec.min_years} years · ${period ? period.label.toLowerCase() : `from ${spec.start_year}`}`;
}

function renderPanel() {
  const p = panel(), st = $("pt-status"), res = $("pt-result"), bar = $("pt-bar");
  if (!st) return;
  // Once painted, the form folds to one line: the map is the answer, the panel only says how it is going.
  p.classList.toggle("ran", Boolean(shown) && !p.classList.contains("editing"));
  $("pt-asked").textContent = shown ? askedWords(shown.spec) : "";
  if (!shown) {
    st.hidden = true; res.hidden = true; bar.hidden = true; $("pt-stop").hidden = true;
    return;
  }
  const phase = shown.phase;
  const reading = ["plan", "boot", "read", "field"].includes(phase);
  bar.hidden = !reading;
  const share = shown.total ? Math.min(1, shown.done / shown.total) : (phase === "plan" ? 0.02 : 0);
  bar.querySelector("span").style.width = `${Math.round(share * 100)}%`;
  st.className = `pt-status${phase === "error" ? " error" : ""}`;
  if (phase === "error") st.textContent = `The painting stopped: ${shown.error}`;
  else if (phase === "stopped") st.textContent = `Stopped after ${shown.done} of ${shown.total} gauges.`;
  else st.textContent = progressWords({ done: shown.done, total: shown.total, failed: shown.failed, phase });
  st.hidden = phase === "done";
  $("pt-stop").hidden = !reading;
  const r = shown.result;
  res.hidden = !r;
  if (r) {
    $("pt-headline").textContent = r.headline || "";
    const f = r.field;
    $("pt-verdict").hidden = !f;
    if (f) {
      $("pt-verdict").className = `pt-verdict ${f.field_significant ? "sig" : "nsig"}`;
      $("pt-verdict").textContent = f.field_significant ? "Field significant" : "Field not significant";
    }
    const over = (r.counts && r.counts.over_cap) || 0;
    $("pt-more").hidden = !over || shown.q.needs !== "daily";
    $("pt-more").textContent = `Read ${Math.min(over, capFor())} more`;
  }
}

// ── the legend row ──────────────────────────────────────────────────────────

function legendHtml() {
  if (!shown) return "";
  const q = shown.q;
  const rows = [...shown.rows.values()];
  const L = legendRows(q, rows);
  const classes = L.classes.slice().reverse().map((c) =>
    `<p class="pt-key"><i class="pt-sw" style="background:${c.color}"></i><span>${escapeHtml(c.label)}</span><b>${c.n || ""}</b></p>`).join("");
  const sigKey = q.kind === "trend"
    ? '<p class="pt-key"><i class="pt-sw sig"></i><span>Significant after the false discovery rate</span></p>' +
      '<p class="pt-key"><i class="pt-sw nsig"></i><span>Not significant</span></p>' : "";
  const status = L.status.map((s) =>
    `<p class="pt-key"><i class="pt-sw st${s.hollow ? " hollow" : ""}" style="${s.hollow ? `border-color:${s.color}` : `background:${s.color}`}"></i>` +
    `<span>${escapeHtml(s.label)}</span><b>${s.n}</b></p>`).join("");
  const f = shown.field;
  const verdict = f && f.n_tested ? `<p class="pt-field ${f.field_significant ? "sig" : "nsig"}">${escapeHtml(f.verdict)}</p>` : "";
  const reading = L.reading ? `<p class="ml-when">${escapeHtml(progressWords({ done: shown.done, total: shown.total, failed: shown.failed, phase: shown.phase }))}</p>` : "";
  const unit = `<p class="ml-when">${escapeHtml(q.label)}${q.unit ? `, ${escapeHtml(q.unit)}` : ""}. The painted dots stand in ` +
    "for the gauge marks.</p>";
  const lic = shown.result ? (shown.result.licences || []).map((l) => `${l.label} (${l.license})`).join("; ") : "";
  const done = shown.phase === "done";
  return unit + reading + classes + sigKey + status + verdict +
    (done ? '<p class="pt-acts"><button type="button" class="ml-btn" data-act="png">Figure (PNG)</button>' +
      '<button type="button" class="ml-btn" data-act="csv">Table (CSV)</button>' +
      '<button type="button" class="ml-btn" data-act="methods">Methods</button>' +
      ((shown.result && shown.result.counts.over_cap && q.needs === "daily")
        ? `<button type="button" class="ml-btn" data-act="more">Read ${Math.min(shown.result.counts.over_cap, capFor())} more</button>` : "") +
      '<button type="button" class="ml-btn quiet" data-act="again">Change</button></p>' : "") +
    `<p class="ml-src">Daily discharge from the AquaScope Archive${lic ? `: ${escapeHtml(lic)}` : ""}. ` +
    "Mirrored gauges only: where the Archive has no record, the map says so.</p>";
}

function registerRow() {
  registerLegendRow({
    id: "paint", title: "Painted",
    shown: () => Boolean(shown),
    mark: () => `<span class="pt-mark">${palette(shown ? shown.q : QUESTIONS[0]).map((c) => `<i style="background:${c}"></i>`).join("")}</span>`,
    summary: () => (shown ? `${shown.q.short} · ${shown.result ? summaryWords(shown.q, shown.result) : `${shown.done} of ${shown.total}`}` : ""),
    body: legendHtml,
    act: (name, el) => {
      if (name === "clear") clearByUndo();
      if (name === "png") void exportPng(el);
      if (name === "csv") void exportText("csv", el);
      if (name === "methods") void exportText("methods", el);
      if (name === "again") { panel().classList.add("editing"); openPanel(); }
      if (name === "more") void paintMore();
    },
  });
}

// ── downloads ───────────────────────────────────────────────────────────────

async function exportText(kind, btn) {
  if (!shown || !shown.result) return;
  const label = btn ? btn.textContent : "";
  try {
    if (btn) { btn.disabled = true; btn.textContent = "Preparing…"; }
    const text = await callCancelable("paint", { op: kind }).promise;
    const stem = fileStem(shown.q, shown.bbox);
    if (kind === "csv") downloadBlob(`${stem}.csv`, text, "text/csv");
    else downloadBlob(`${stem}-methods.txt`, text, "text/plain");
  } catch (err) {
    if (!(err instanceof Cancelled)) announce(`Download failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// The map as it is drawn now: copied out of the WebGL buffer inside a render, before it is cleared.
function grabMap() {
  return new Promise((resolve) => {
    map.once("render", () => {
      const src = map.getCanvas();
      const c = document.createElement("canvas");
      c.width = src.width;
      c.height = src.height;
      try { c.getContext("2d").drawImage(src, 0, 0); resolve(c); } catch { resolve(null); }
    });
    map.triggerRepaint();
  });
}

/** A figure of the painting: the map, a title, the legend with its counts, the field verdict and the credits. */
export async function figureCanvas() {
  if (!shown || !shown.result) return null;
  const shot = await grabMap();
  if (!shot) return null;
  const dpr = shot.width / Math.max(1, map.getCanvas().clientWidth);
  const W = shot.width, mapH = shot.height;
  const pad = 14 * dpr;
  const q = shown.q, res = shown.result;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = (px, w = 400) => `${w} ${px * dpr}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  // The text under the map: the verdict and the credits, wrapped to the width.
  ctx.font = font(12);
  const verdict = res.field ? res.field.verdict : res.headline;
  const vLines = wrapText(verdict, W - 2 * pad, (t) => ctx.measureText(t).width);
  ctx.font = font(10);
  const attrib = (document.querySelector(".maplibregl-ctrl-attrib-inner") || {}).textContent || "";
  const lic = (res.licences || []).map((l) => `${l.label} (${l.license})`).join("; ");
  const credit = `Data: AquaScope Archive${lic ? `, ${lic}` : ""}. Map: ${attrib.trim() || "MapLibre"}. ` +
    "Computed in the browser with AquaScope (doi.org/10.5281/zenodo.21903143).";
  const cLines = wrapText(credit, W - 2 * pad, (t) => ctx.measureText(t).width);
  const footH = pad + vLines.length * 17 * dpr + 6 * dpr + cLines.length * 14 * dpr + pad;
  c.width = W;
  c.height = mapH + footH;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(shot, 0, 0);
  // Title box, top left.
  const t = figureTitle(q, res);
  ctx.font = font(15, 650);
  const tw = Math.min(W - 2 * pad, Math.max(ctx.measureText(t.title).width, 260 * dpr) + 2 * pad);
  ctx.font = font(11);
  const subLines = wrapText(t.sub, tw - 2 * pad, (s) => ctx.measureText(s).width);
  const th = pad + 18 * dpr + subLines.length * 14 * dpr + pad * 0.6;
  ctx.fillStyle = "rgba(255,255,255,0.93)";
  ctx.fillRect(pad, pad, tw, th);
  ctx.fillStyle = "#111827";
  ctx.font = font(15, 650);
  ctx.fillText(t.title, 2 * pad, pad + 18 * dpr);
  ctx.font = font(11);
  ctx.fillStyle = "#374151";
  subLines.forEach((s, i) => ctx.fillText(s, 2 * pad, pad + 18 * dpr + (i + 1) * 14 * dpr));
  // Legend box, bottom left of the map.
  const L = legendRows(q, [...shown.rows.values()]);
  const items = [...L.classes.slice().reverse().map((x) => ({ ...x, kind: "class" })),
    ...(q.kind === "trend" ? [{ kind: "sig", label: "significant (FDR)" }, { kind: "nsig", label: "not significant" }] : []),
    ...L.status.map((x) => ({ ...x, kind: "status" }))];
  const rowH = 15 * dpr;
  const lw = 230 * dpr, lh = pad + items.length * rowH + pad * 0.5;
  const ly = Math.max(th + 2 * pad, mapH - lh - pad);
  ctx.fillStyle = "rgba(255,255,255,0.93)";
  ctx.fillRect(pad, ly, lw, lh);
  ctx.font = font(11);
  items.forEach((it, i) => {
    const y = ly + pad * 0.7 + i * rowH + rowH / 2;
    const x = 2 * pad;
    ctx.beginPath();
    ctx.arc(x, y, 5 * dpr, 0, 2 * Math.PI);
    if (it.kind === "class") { ctx.fillStyle = it.color; ctx.fill(); ctx.strokeStyle = "#33414f"; ctx.lineWidth = 0.8 * dpr; ctx.stroke(); }
    else if (it.kind === "sig") { ctx.fillStyle = palette(q)[5]; ctx.fill(); ctx.strokeStyle = "#111827"; ctx.lineWidth = 1.8 * dpr; ctx.stroke(); }
    else if (it.kind === "nsig") { ctx.globalAlpha = 0.85; ctx.fillStyle = palette(q)[5]; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = "#5b6673"; ctx.lineWidth = 0.7 * dpr; ctx.stroke(); }
    else if (it.hollow) { ctx.strokeStyle = it.color; ctx.lineWidth = 1.4 * dpr; ctx.stroke(); }
    else { ctx.fillStyle = it.color; ctx.fill(); }
    ctx.fillStyle = "#111827";
    ctx.fillText(`${it.label}${it.n !== undefined ? `  (${it.n})` : ""}`, x + 11 * dpr, y + 4 * dpr);
  });
  // Under the map: the verdict and the credits.
  let y = mapH + pad + 12 * dpr;
  ctx.fillStyle = "#111827";
  ctx.font = font(12);
  for (const line of vLines) { ctx.fillText(line, pad, y); y += 17 * dpr; }
  y += 6 * dpr - 3 * dpr;
  ctx.fillStyle = "#4b5563";
  ctx.font = font(10);
  for (const line of cLines) { ctx.fillText(line, pad, y); y += 14 * dpr; }
  return c;
}

async function exportPng(btn) {
  const label = btn ? btn.textContent : "";
  try {
    if (btn) { btn.disabled = true; btn.textContent = "Drawing…"; }
    const c = await figureCanvas();
    if (!c) throw new Error("the map could not be copied");
    const blob = await new Promise((resolve) => c.toBlob(resolve, "image/png"));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${fileStem(shown.q, shown.bbox)}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } catch (err) {
    announce(`The figure failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// ── wiring ──────────────────────────────────────────────────────────────────

function openPanel() {
  showPanel(true);
  renderPanel();
  const first = $("pt-form").querySelector('input[name="pt-q"]:checked');
  if (first) first.focus({ preventScroll: true });
}

function buildForm() {
  const qs = $("pt-qs");
  qs.innerHTML = QUESTIONS.map((q, i) =>
    `<label class="pt-q"><input type="radio" name="pt-q" value="${q.id}"${i === 0 ? " checked" : ""}>` +
    `<span><b>${escapeHtml(q.label)}</b><small>${escapeHtml(q.describe)}</small></span></label>`).join("");
  $("pt-years").innerHTML = YEAR_CHOICES.map((y) => `<option value="${y}"${y === MIN_YEARS ? " selected" : ""}>${y}</option>`).join("");
  $("pt-period").innerHTML = PERIODS.map((p) => `<option value="${p.id}">${escapeHtml(p.label)}</option>`).join("");
  $("pt-note").textContent = `Archive gauges only; up to ${capFor()} a view, the longest records first.`;
}

export function initPaint() {
  const p = panel();
  if (!p) return;
  buildForm();
  registerRow();
  registerSlotWriter("paint", (value) => { if (value) void run(value); else clear(); });
  $("btn-paint").addEventListener("click", () => {
    if (!p.hidden) { showPanel(false); return; }
    openPanel();
  });
  p.querySelector(".pt-close").addEventListener("click", () => showPanel(false));
  p.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); showPanel(false); } });
  $("pt-form").addEventListener("submit", (e) => {
    e.preventDefault();
    p.classList.remove("editing");
    void paintView(picked(), { by: "rules", label: `Paint this view: ${questionById(picked().question).short}` });
  });
  $("pt-stop").addEventListener("click", stopPainting);
  $("pt-more").addEventListener("click", () => void paintMore());
  $("pt-change").addEventListener("click", () => { p.classList.add("editing"); renderPanel(); });
  onWorkerRestart(() => {
    if (shown && shown.phase !== "done") stopPainting();
    else if (shown) void resync();
  });
  // The AI action paint (map-actions.js) and agents reach it here.
  actions.paintMap = (spec, opts) => paintView(spec, opts);
  actions.paintState = () => (shown ? { ...shown.spec } : null);
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.paint = {
    paint: paintView, more: paintMore, stop: stopPainting, clear, shown: () => shown, figure: figureCanvas,
    status: () => (shown ? STATUS_STYLE : null),
  };
}
