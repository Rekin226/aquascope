// The layer panel in the left rail: basemap, terrain, overlays with opacity
// and legends, the shared date for the time-driven layers, how the gauges are
// coloured, and "select an area".

import { $, downloadBlob, escapeHtml, sourceStyle, state, stationKey, toCsv } from "./core.js?v=__BUILD__";
import {
  BASEMAPS, GAUGE_STYLES, OVERLAYS, OVERLAY_GROUPS, RECENT_BREAKS, RECORD_BREAKS,
  basemapById, creditLines, defaultDate, overlayById, recordYears, yearsSinceLast,
} from "./layers.js?v=__BUILD__";
import {
  applyDate, areaSelectActive, currentBasemap, globeSupported, setBasemap, setGaugeStyle, setGlobe,
  setHeatmap, setHillshade, setOverlay, setOverlayOpacity, setTerrain, startAreaSelect,
} from "./map.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

const anyTimeLayer = () =>
  [...state.overlays].some((id) => (overlayById(id) || {}).time) || Boolean(basemapById(state.basemap).time);

function radioRow(name, value, label, checked, title) {
  const row = document.createElement("label");
  row.className = "rail-row";
  row.title = title || "";
  row.innerHTML = `<input type="radio" name="${name}" value="${escapeHtml(value)}" ${checked ? "checked" : ""}>` +
    `<span class="rail-label">${escapeHtml(label)}</span>`;
  return row;
}

function buildBasemaps() {
  const box = $("rail-basemaps");
  box.innerHTML = "";
  for (const b of BASEMAPS) {
    const row = radioRow("basemap", b.id, b.label, b.id === state.basemap, `${b.attribution} · ${b.licence}`);
    if (/non-commercial/i.test(b.licence)) {
      row.querySelector(".rail-label").insertAdjacentHTML("beforeend", ' <span class="tag">NC</span>');
    }
    row.querySelector("input").addEventListener("change", () => {
      state.basemap = b.id;
      setBasemap(b.id, { date: state.date });
      renderCredits();
      syncDateRow();
      writeUrl();
    });
    box.appendChild(row);
  }
}

function buildTerrain() {
  const box = $("rail-terrain");
  box.innerHTML = "";
  const rows = [
    ["terrain", "3D terrain", state.terrain, (on) => { state.terrain = on; setTerrain(on); renderCredits(); }],
    ["hillshade", "Hillshade", state.hillshade, (on) => { state.hillshade = on; setHillshade(on); renderCredits(); }],
    ["globe", "Globe", state.globe, (on) => { state.globe = setGlobe(on) ? on : false; }],
  ];
  for (const [id, label, checked, apply] of rows) {
    if (id === "globe" && !globeSupported()) continue;
    const row = document.createElement("label");
    row.className = "rail-row";
    row.innerHTML = `<input type="checkbox" id="toggle-${id}" ${checked ? "checked" : ""}><span class="rail-label">${label}</span>`;
    row.querySelector("input").addEventListener("change", (e) => { apply(e.target.checked); writeUrl(); });
    box.appendChild(row);
  }
  $("rail-terrain-note").textContent = "Elevation from AWS Terrain Tiles (Mapzen/Tilezen), open data.";
}

function overlayRow(o) {
  const wrap = document.createElement("div");
  wrap.className = "overlay-row";
  const on = state.overlays.has(o.id);
  const opacity = state.opacity[o.id] ?? o.opacity ?? 0.8;
  wrap.innerHTML =
    `<label class="rail-row"><input type="checkbox" id="ov-${escapeHtml(o.id)}" ${on ? "checked" : ""}>` +
    `<span class="rail-label">${escapeHtml(o.label)}</span>` +
    `<button class="icon-btn tiny info" type="button" title="About this layer" aria-label="About ${escapeHtml(o.label)}">i</button></label>` +
    `<div class="overlay-controls" ${on ? "" : "hidden"}>` +
    `<input type="range" min="0" max="1" step="0.05" value="${opacity}" aria-label="${escapeHtml(o.label)} opacity">` +
    (o.legend ? `<img class="legend" src="${o.legend}" alt="Colour scale for ${escapeHtml(o.label)}" loading="lazy">` : "") +
    `</div>`;
  const check = wrap.querySelector("input[type=checkbox]");
  const controls = wrap.querySelector(".overlay-controls");
  check.addEventListener("change", (e) => {
    if (e.target.checked) state.overlays.add(o.id); else state.overlays.delete(o.id);
    controls.hidden = !e.target.checked;
    setOverlay(o.id, e.target.checked, { date: state.date, opacity: state.opacity[o.id] ?? null });
    renderCredits();
    syncDateRow();
    writeUrl();
  });
  wrap.querySelector("input[type=range]").addEventListener("input", (e) => {
    const v = Number(e.target.value);
    state.opacity[o.id] = v;
    setOverlayOpacity(o.id, v);
  });
  wrap.querySelector("input[type=range]").addEventListener("change", () => writeUrl());
  wrap.querySelector("button.info").addEventListener("click", () => {
    openModal(o.label, `
      <p>${escapeHtml(o.note || "")}</p>
      <p class="muted">${o.attribution}</p>
      <p class="muted">Licence: ${escapeHtml(o.licence)}</p>
      ${o.legend ? `<img class="legend-big" src="${o.legend}" alt="Colour scale">` : ""}
    `);
  });
  return wrap;
}

function buildOverlays() {
  const box = $("rail-overlays");
  box.innerHTML = "";
  for (const group of OVERLAY_GROUPS) {
    const layers = OVERLAYS.filter((o) => o.group === group);
    if (!layers.length) continue;
    const h = document.createElement("div");
    h.className = "rail-subhead";
    h.textContent = group;
    box.appendChild(h);
    for (const o of layers) box.appendChild(overlayRow(o));
  }
}

// ── the shared date ─────────────────────────────────────────────────────────

function syncDateRow() {
  const row = $("rail-date");
  row.hidden = !anyTimeLayer();
  $("date-input").value = state.date;
}

function stepDate(days) {
  const d = new Date(`${state.date}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  const iso = d.toISOString().slice(0, 10);
  const today = new Date().toISOString().slice(0, 10);
  state.date = iso > today ? today : iso;
  $("date-input").value = state.date;
  applyDate(state.date, [...state.overlays], state.basemap);
  writeUrl();
}

function buildDate() {
  $("date-input").value = state.date;
  $("date-input").max = new Date().toISOString().slice(0, 10);
  $("date-input").addEventListener("change", (e) => {
    state.date = e.target.value;
    applyDate(state.date, [...state.overlays], state.basemap);
    writeUrl();
  });
  $("date-prev").addEventListener("click", () => stepDate(-1));
  $("date-next").addEventListener("click", () => stepDate(1));
  syncDateRow();
}

// ── gauge styling ───────────────────────────────────────────────────────────

function gaugeLegendHtml(mode) {
  const swatch = (c, l) => `<span class="sw"><i style="background:${c}"></i>${escapeHtml(l)}</span>`;
  if (mode === "record") return RECORD_BREAKS.map((b) => swatch(b.color, b.label)).join("");
  if (mode === "recent") return RECENT_BREAKS.map((b) => swatch(b.color, b.label)).join("");
  return "";
}

function buildGaugeStyle() {
  const select = $("gauge-style");
  select.innerHTML = GAUGE_STYLES.map((g) => `<option value="${g.id}">${escapeHtml(g.label)}</option>`).join("");
  select.value = state.gaugeStyle;
  const apply = () => {
    setGaugeStyle(state.gaugeStyle);
    $("gauge-legend").innerHTML = gaugeLegendHtml(state.gaugeStyle);
    $("gauge-legend").hidden = state.gaugeStyle === "source";
    $("rail-sources").classList.toggle("dimmed", state.gaugeStyle !== "source");
  };
  select.addEventListener("change", (e) => { state.gaugeStyle = e.target.value; apply(); writeUrl(); });
  const heat = $("toggle-heat");
  heat.checked = state.heat;
  heat.addEventListener("change", (e) => { state.heat = e.target.checked; setHeatmap(e.target.checked); writeUrl(); });
  apply();
}

// ── select an area ──────────────────────────────────────────────────────────

function stationsIn(bbox) {
  return state.stations.filter((r) =>
    !state.hidden.has(r.source) &&
    r.lat >= bbox.south && r.lat <= bbox.north &&
    (bbox.west <= bbox.east
      ? r.lon >= bbox.west && r.lon <= bbox.east
      : r.lon >= bbox.west || r.lon <= bbox.east));   // across the antimeridian
}

function showSelection(bbox) {
  const btn = $("btn-area");
  btn.classList.remove("active");
  btn.textContent = "Select an area";
  if (!bbox) { $("area-result").hidden = true; return; }
  const rows = stationsIn(bbox);
  const box = $("area-result");
  box.hidden = false;
  const now = new Date();
  box.innerHTML = `<div><strong>${rows.length.toLocaleString()}</strong> gauges in this box</div>` +
    `<div class="muted">${bbox.south.toFixed(2)} to ${bbox.north.toFixed(2)} °N, ${bbox.west.toFixed(2)} to ${bbox.east.toFixed(2)} °E</div>`;
  const dl = document.createElement("button");
  dl.className = "btn tiny";
  dl.textContent = "Download CSV";
  dl.disabled = rows.length === 0;
  dl.addEventListener("click", () => {
    const csv = toCsv(
      ["source", "station_id", "name", "latitude", "longitude", "variables", "period_start", "period_end", "record_years", "url"],
      rows.map((r) => [
        r.source, r.station_id, r.name || "", r.lat, r.lon, (r.variables || []).join(" "),
        r.period_start || "", r.period_end || "",
        (recordYears(r, now) ?? "") === "" ? "" : (recordYears(r, now)).toFixed(1), r.url || "",
      ]),
    );
    downloadBlob(`aquascope-gauges-${bbox.south.toFixed(2)}_${bbox.west.toFixed(2)}.csv`, csv, "text/csv");
  });
  box.appendChild(dl);
  const clear = document.createElement("button");
  clear.className = "btn tiny";
  clear.textContent = "Clear";
  clear.addEventListener("click", () => { box.hidden = true; });
  box.appendChild(clear);
}

function buildAreaSelect() {
  const btn = $("btn-area");
  btn.addEventListener("click", () => {
    if (areaSelectActive()) return;
    btn.classList.add("active");
    btn.textContent = "Drag a box on the map (Esc to cancel)";
    startAreaSelect(showSelection);
  });
}

// ── credits ─────────────────────────────────────────────────────────────────

export function renderCredits() {
  const lines = creditLines(state.basemap, [...state.overlays], { terrain: state.terrain || state.hillshade });
  $("rail-credits").innerHTML = lines
    .map((l) => `<div><b>${escapeHtml(l.label)}</b>: ${l.attribution} <span class="muted">(${escapeHtml(l.licence)})</span></div>`)
    .join("");
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initLayerUI() {
  if (!state.date) state.date = defaultDate();
  buildBasemaps();
  buildTerrain();
  buildOverlays();
  buildDate();
  buildGaugeStyle();
  buildAreaSelect();
  renderCredits();
}

// Apply layer state that arrived from the URL (or from Back). A basemap swap
// is asynchronous, so anything added to the style has to wait for it: adding
// an overlay while setStyle is in flight silently loses it.
export function applyLayerState() {
  const rest = () => {
    for (const o of OVERLAYS) {
      const on = state.overlays.has(o.id);
      setOverlay(o.id, on, { date: state.date, opacity: state.opacity[o.id] ?? null });
    }
    setTerrain(state.terrain);
    setHillshade(state.hillshade);
    if (state.globe) state.globe = setGlobe(true); else setGlobe(false);
    setHeatmap(state.heat);
    setGaugeStyle(state.gaugeStyle);
    syncRailControls();
    renderCredits();
    syncDateRow();
  };
  if (currentBasemap() === state.basemap) rest();
  else setBasemap(state.basemap, { date: state.date, then: rest });
}

export function syncRailControls() {
  for (const input of document.querySelectorAll('#rail-basemaps input[type=radio]')) {
    input.checked = input.value === state.basemap;
  }
  for (const o of OVERLAYS) {
    const el = $(`ov-${o.id}`);
    if (!el) continue;
    el.checked = state.overlays.has(o.id);
    const controls = el.closest(".overlay-row").querySelector(".overlay-controls");
    controls.hidden = !el.checked;
    const range = controls.querySelector("input[type=range]");
    if (range) range.value = state.opacity[o.id] ?? o.opacity ?? 0.8;
  }
  for (const [id, val] of [["terrain", state.terrain], ["hillshade", state.hillshade], ["globe", state.globe]]) {
    const el = $(`toggle-${id}`);
    if (el) el.checked = Boolean(val);
  }
  const gs = $("gauge-style");
  if (gs) gs.value = state.gaugeStyle;
  const heat = $("toggle-heat");
  if (heat) heat.checked = state.heat;
}

// Used by the Ask drawer so the model knows what is on the map.
export function visibleLayerSummary() {
  const base = basemapById(state.basemap);
  const bits = [`basemap ${base.label}`];
  for (const id of state.overlays) {
    const o = overlayById(id);
    if (o) bits.push(`${o.label}${o.time ? ` for ${state.date}` : ""}`);
  }
  if (state.terrain) bits.push("3D terrain");
  if (state.gaugeStyle !== "source") bits.push(`gauges coloured by ${(GAUGE_STYLES.find((g) => g.id === state.gaugeStyle) || {}).label}`);
  return bits.join(", ");
}

export { stationsIn, yearsSinceLast };
