// Study this area: a multi-gauge flood study over the gauges an area select
// picked. The engine is aquascope.area_study (in the worker): it reads the
// Archive first, fetches a capped number of gauges live, and returns per-site
// values plus two regional methods. This module is the face: progress, the
// pins coloured by a chosen result, a sortable table and the downloads.

import { $, EMPTY_FC, actions, downloadBlob, escapeHtml, fmt, fmtP, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { stationArea } from "./basins.js?v=__BUILD__";
import { Cancelled, callCancelable, onAreaProgress, restartWorker } from "./worker-client.js?v=__BUILD__";
import { COLOR_BY, TABLE, legend, pinColor, progressShare, progressText, sortRows } from "./area-study-view.js?v=__BUILD__";

const SOURCE = "area-study";
const LAYER = "area-study-pins";

const view = { result: null, colorBy: "trend", sort: { field: "record_years", dir: -1 }, call: null, bbox: null };

// ── the pins ────────────────────────────────────────────────────────────────

function drawPins() {
  if (!state.mapOk || !map) return;
  const fc = view.result ? view.result.geojson : EMPTY_FC;
  if (!map.getSource(SOURCE)) {
    map.addSource(SOURCE, { type: "geojson", data: fc });
    map.addLayer({
      id: LAYER, type: "circle", source: SOURCE,
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 5, 10, 9],
        "circle-color": pinColor(view.colorBy, fc.features),
        "circle-stroke-color": ["case", ["==", ["get", "status"], "studied"], "#0f1c26", "#9aa9b6"],
        "circle-stroke-width": ["case", ["==", ["get", "fdr_significant"], true], 2.6, 1.2],
      },
    });
    map.on("click", LAYER, (e) => {
      const key = e.features && e.features[0] && e.features[0].properties.key;
      if (key && state.byKey.has(key)) actions.selectStation(key, { fly: true });
    });
    map.on("mouseenter", LAYER, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", LAYER, () => { map.getCanvas().style.cursor = ""; });
  } else {
    map.getSource(SOURCE).setData(fc);
    map.setPaintProperty(LAYER, "circle-color", pinColor(view.colorBy, fc.features));
  }
}

// A basemap change replaces the whole style; put the study's pins back.
let watching = false;
function watchStyle() {
  if (watching || !state.mapOk || !map) return;
  watching = true;
  map.on("styledata", () => {
    if (view.result && !map.getSource(SOURCE)) {
      try { drawPins(); } catch (err) { console.info("area study pins:", err && err.message); }
    }
  });
}

function clearPins() {
  if (state.mapOk && map && map.getSource(SOURCE)) map.getSource(SOURCE).setData(EMPTY_FC);
}

// ── the panel ───────────────────────────────────────────────────────────────

function panel() { return $("area-study"); }

function shell(title) {
  const box = panel();
  box.hidden = false;
  box.innerHTML = `
    <div class="as-head">
      <strong>${escapeHtml(title)}</strong>
      <button class="icon-btn tiny as-close" type="button" aria-label="Close the area study">×</button>
    </div>
    <div class="as-body"></div>`;
  box.querySelector(".as-close").addEventListener("click", closeAreaStudy);
  return box.querySelector(".as-body");
}

export function closeAreaStudy() {
  if (view.call && view.call.cancel()) restartWorker();
  view.call = null;
  view.result = null;
  clearPins();
  const box = panel();
  if (box) { box.hidden = true; box.innerHTML = ""; }
}

function showProgress(body, e) {
  const bar = body.querySelector(".as-bar > span");
  const text = body.querySelector(".as-progress-text");
  if (bar) bar.style.width = `${Math.round(progressShare(e) * 100)}%`;
  if (text) text.textContent = progressText(e);
}

function renderLegend(body) {
  const fmtLegend = (x) => fmt(x);
  const rows = legend(view.colorBy, view.result.geojson.features, fmtLegend);
  body.querySelector(".as-legend").innerHTML = rows
    .map((r) => `<span class="as-key"><i style="background:${r.color}"></i>${escapeHtml(r.label)}</span>`).join("");
}

function cell(row, field, kind) {
  const v = row[field];
  if (field === "name") {
    const label = v || row.station_id;
    return `<button class="link as-site" type="button" data-key="${escapeHtml(row.key)}">${escapeHtml(label)}</button>`;
  }
  if (field === "status" && row.note) return `<span title="${escapeHtml(row.note)}">${escapeHtml(v)}</span>`;
  if (kind === "num") return fmt(v);
  if (kind === "p") return fmtP(v);
  if (kind === "bool") return v === true ? "yes" : v === false ? "no" : "";
  return escapeHtml(v ?? "");
}

function renderTable(body) {
  const rows = sortRows(view.result.sites, view.sort.field, view.sort.dir);
  const arrow = (f) => (view.sort.field === f ? (view.sort.dir > 0 ? " ▲" : " ▼") : "");
  const head = TABLE.map(([f, label]) =>
    `<th scope="col"><button type="button" class="as-sort" data-field="${f}">${escapeHtml(label)}${arrow(f)}</button></th>`).join("");
  const bodyRows = rows.map((r) =>
    `<tr class="as-${escapeHtml(r.status)}">${TABLE.map(([f, , kind]) => `<td>${cell(r, f, kind)}</td>`).join("")}</tr>`).join("");
  const wrap = body.querySelector(".as-table");
  wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${bodyRows}</tbody></table>`;
  for (const b of wrap.querySelectorAll(".as-sort")) {
    b.addEventListener("click", () => {
      const f = b.dataset.field;
      view.sort = { field: f, dir: view.sort.field === f ? -view.sort.dir : (f === "name" ? 1 : -1) };
      renderTable(body);
    });
  }
  for (const b of wrap.querySelectorAll(".as-site")) {
    b.addEventListener("click", () => { if (state.byKey.has(b.dataset.key)) actions.selectStation(b.dataset.key, { fly: true }); });
  }
}

function regionalHtml(res) {
  const fs = res.field_significance || {};
  const rfa = res.regional_frequency || {};
  const parts = [];
  if (fs.verdict) {
    parts.push(`<p><b>Trend field.</b> ${escapeHtml(fs.verdict)}</p>`);
    if (fs.n_tested) parts.push(`<p class="muted small">${escapeHtml(fs.caveat || "")}</p>`);
  }
  if (rfa.regional) {
    const g = rfa.regional.growth_curve || {};
    const het = rfa.heterogeneity || {};
    const curve = Object.entries(g).map(([t, v]) => `T${t}: ${fmt(v, 2)}`).join(", ");
    parts.push(`<p><b>Regional growth curve</b> from ${rfa.n_sites} sites (times the index flood): ${escapeHtml(curve)}. ` +
      `H = ${fmt(het.H, 2)}, ${escapeHtml(het.class || "")}.</p>`);
  } else if (rfa.error) {
    parts.push(`<p><b>Regional growth curve.</b> ${escapeHtml(rfa.error)}</p>`);
  }
  return parts.join("");
}

function renderResult(body) {
  const res = view.result;
  const notes = (res.notes || []).map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  body.innerHTML = `
    <p class="as-headline">${escapeHtml(res.headline || "")}</p>
    ${regionalHtml(res)}
    <div class="as-controls">
      <label>Colour pins by <select class="as-color">${COLOR_BY.map((c) =>
        `<option value="${c.id}"${c.id === view.colorBy ? " selected" : ""}>${escapeHtml(c.label)}</option>`).join("")}</select></label>
      <button class="btn tiny as-csv" type="button">CSV</button>
      <button class="btn tiny as-xlsx" type="button">XLSX</button>
    </div>
    <div class="as-legend"></div>
    <div class="as-table"></div>
    ${notes ? `<details class="as-notes"><summary>Notes (${res.notes.length})</summary><ul>${notes}</ul></details>` : ""}
    <details class="as-notes"><summary>Methods</summary><ul>${(res.methods || []).map((m) =>
      `<li><b>${escapeHtml(m.name)}</b>. ${escapeHtml(m.text)} <span class="muted">${escapeHtml(m.citation)}</span></li>`).join("")}</ul></details>`;
  body.querySelector(".as-color").addEventListener("change", (e) => {
    view.colorBy = e.target.value;
    drawPins();
    renderLegend(body);
  });
  body.querySelector(".as-csv").addEventListener("click", () => download("csv"));
  body.querySelector(".as-xlsx").addEventListener("click", (e) => download("xlsx", e.target));
  renderLegend(body);
  renderTable(body);
}

function fileStem() {
  const b = view.bbox;
  return b ? `aquascope-area-${b.south.toFixed(2)}_${b.west.toFixed(2)}` : "aquascope-area";
}

async function download(kind, btn) {
  const label = btn ? btn.textContent : "";
  try {
    if (btn) { btn.disabled = true; btn.textContent = "Preparing…"; }
    const out = await callCancelable("area_study", { op: kind }).promise;
    if (kind === "csv") {
      downloadBlob(`${fileStem()}.csv`, out, "text/csv");
    } else {
      const bytes = Uint8Array.from(atob(out), (c) => c.charCodeAt(0));
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }));
      a.download = `${fileStem()}.xlsx`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }
  } catch (err) {
    if (!(err instanceof Cancelled)) alert(`Download failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// Catchment areas from the Archive's station_catchments table, for Q100 per km².
async function areasFor(sites) {
  const out = {};
  await Promise.all(sites.filter((s) => s.status === "studied").map(async (s) => {
    const a = await stationArea(s.key);
    if (a && a.area) out[s.key] = a.area;
  }));
  return out;
}

// ── entry point: called by the area select with the rows in the box ────────

export async function openAreaStudy(rows, bbox) {
  if (view.call) view.call.cancel();
  view.bbox = bbox || null;
  view.result = null;
  clearPins();
  watchStyle();
  const body = shell("Study this area");
  body.innerHTML = `
    <p class="muted small">${rows.length.toLocaleString()} gauges in the box. Records come from the AquaScope Archive first;
    at most 25 are fetched live from the agencies (keyless limits), the rest are listed as skipped.</p>
    <div class="as-bar"><span></span></div>
    <p class="as-progress-text muted small">Starting…</p>
    <button class="btn tiny as-stop" type="button">Stop</button>`;
  const stations = rows.map((r) => ({
    source: r.source, station_id: r.station_id, site_id: r.site_id || r.station_id, name: r.name || "",
    latitude: r.lat, longitude: r.lon, variables: r.variables || [], period_start: r.period_start || null,
    period_end: r.period_end || null,
  }));
  const call = callCancelable("area_study", { op: "run", stations, question: "Study this area" });
  view.call = call;
  // Python cannot be interrupted mid-call, so Stop restarts the worker (as the Study drawer does).
  body.querySelector(".as-stop").addEventListener("click", () => { if (call.cancel()) restartWorker(); });
  const off = onAreaProgress((e, id) => { if (id === call.id) showProgress(body, e); });
  try {
    view.result = await call.promise;
  } catch (err) {
    off();
    if (err instanceof Cancelled) { body.innerHTML = `<p class="muted">Stopped.</p>`; return; }
    body.innerHTML = `<p class="status warn">The area study failed: ${escapeHtml(err.message)}</p>`;
    return;
  } finally {
    if (view.call === call) view.call = null;
  }
  off();
  renderResult(body);
  drawPins();
  // Areas arrive after the study (a DuckDB read on this thread); Q100 per km² fills in when they do.
  try {
    const areas = await areasFor(view.result.sites);
    if (Object.keys(areas).length && view.result) {
      view.result = await callCancelable("area_study", { op: "areas", areas }).promise;
      renderResult(body);
      drawPins();
    }
  } catch (err) {
    console.info("area study: catchment areas unavailable:", err && err.message);
  }
}

// For the debug hooks in a browser.
export const _areaStudyView = view;
