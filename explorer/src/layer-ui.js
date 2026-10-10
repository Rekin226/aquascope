// The layer panel in the left rail: basemap, terrain, overlays with opacity
// and legends, how the gauges are coloured, and "select an area". The date the
// dated layers follow is not here: it is the time bar on the map (time-ui.js).

import { $, actions, downloadBlob, escapeHtml, onTime, state, toCsv } from "./core.js?v=__BUILD__";
import {
  BASEMAPS, GAUGE_STYLES, OVERLAYS, OVERLAY_GROUPS, RECENT_BREAKS, RECORD_BREAKS,
  basemapById, creditLines, defaultDate, overlayById, recordYears, yearsSinceLast,
} from "./layers.js?v=__BUILD__";
import {
  areaSelectActive, currentBasemap, globeSupported, refreshMapData, setBasemap, setGaugeStyle, setGaugesVisible, setGlobe,
  setHeatmap, setHillshade, setOverlay, setOverlayOpacity, setTerrain, startAreaSelect,
} from "./map.js?v=__BUILD__";
import { syncTimeBar } from "./time-ui.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { RIVERS_CREDIT } from "./river-core.js?v=__BUILD__";
import { FLOODS_CREDIT } from "./floods-ahead-core.js?v=__BUILD__";
import { DEPTH_CREDIT } from "./flood-depth-core.js?v=__BUILD__";
import { NEWS_CREDIT, RADAR_CREDIT } from "./floods-past-core.js?v=__BUILD__";
import { STATUS_CREDIT, monthLabel } from "./status-core.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { openAreaStudy } from "./area-study.js?v=__BUILD__";
import { cancelAreaContext, openAreaContext } from "./context.js?v=__BUILD__";
import { loadSkillGrades, skillLegendHtml } from "./evidence.js?v=__BUILD__";
import { ensureNowStatus, nowLegendHtml } from "./now-map.js?v=__BUILD__";
import { bulletinLegendHtml, ensureBulletinStatus } from "./bulletin.js?v=__BUILD__";
import { areaWatchButton } from "./watch.js?v=__BUILD__";
import { refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { STATUS_CLASSES } from "./now-core.js?v=__BUILD__";

// A tiny swatch standing in for each basemap, so eight radio rows become two
// columns of chips you can pick from at a glance.
const BASEMAP_SWATCH = {
  light: "linear-gradient(135deg,#fbfbfb,#e7edf2)",
  dark: "linear-gradient(135deg,#3b444d,#12181e)",
  streets: "linear-gradient(135deg,#f7f3ea,#d9e6cf)",
  satellite: "linear-gradient(135deg,#2e5f3a,#123049)",
  "satellite-recent": "linear-gradient(135deg,#3a7a45,#0f2b41)",
  terrain: "linear-gradient(135deg,#e6ddc6,#a3b58c)",
  daily: "linear-gradient(135deg,#5d7fa8,#1b2a3d)",
  usgs: "linear-gradient(135deg,#6b7f52,#2b3d2a)",
};

// The space behind the globe follows the basemap: a soft sky under the paper
// styles, deep space under the imagery ones (see body[data-basemap] in the CSS).
function reflectBasemap() {
  document.body.dataset.basemap = state.basemap;
}

function basemapChip(b) {
  const row = document.createElement("label");
  row.className = "basemap-chip";
  row.title = `${b.attribution} · ${b.licence}`;
  const nc = /non-commercial/i.test(b.licence) ? ' <span class="tag">NC</span>' : "";
  row.innerHTML =
    `<input type="radio" name="basemap" value="${escapeHtml(b.id)}" ${b.id === state.basemap ? "checked" : ""}>` +
    `<span class="sw-thumb" style="background:${BASEMAP_SWATCH[b.id] || "var(--bg-sunken)"}" aria-hidden="true"></span>` +
    `<span class="rail-label">${escapeHtml(b.label)}${nc}</span>`;
  return row;
}

// Switch the basemap exactly as its chip would (the time bar's "Satellite" uses it too).
export function chooseBasemap(id) {
  const b = basemapById(id);
  state.basemap = b.id;
  for (const input of document.querySelectorAll('#rail-basemaps input[type=radio]')) input.checked = input.value === b.id;
  reflectBasemap();
  setBasemap(b.id, { date: state.date });
  renderCredits();
  syncTimeBar({ layersChanged: true });
  writeUrl();
}

function buildBasemaps() {
  const box = $("rail-basemaps");
  box.innerHTML = "";
  for (const b of BASEMAPS) {
    const row = basemapChip(b);
    row.querySelector("input").addEventListener("change", () => chooseBasemap(b.id));
    box.appendChild(row);
  }
  reflectBasemap();
}

function buildTerrain() {
  const box = $("rail-terrain");
  box.innerHTML = "";
  const rows = [
    ["terrain", "3D terrain", state.terrain, (on) => { state.terrain = on; setTerrain(on); renderCredits(); }],
    ["hillshade", "Hillshade", state.hillshade, (on) => { state.hillshade = on; setHillshade(on); renderCredits(); }],
  ];
  for (const [id, label, checked, apply] of rows) {
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
  check.addEventListener("change", (e) => toggleOverlay(o.id, e.target.checked));
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

// Turn an overlay on or off exactly as its checkbox would. The time bar uses it
// for "show rain" when the reader picks a date with no dated layer on.
export function toggleOverlay(id, on) {
  const o = overlayById(id);
  if (!o) return;
  if (on) state.overlays.add(id); else state.overlays.delete(id);
  const check = $(`ov-${id}`);
  if (check) {
    check.checked = on;
    check.closest(".overlay-row").querySelector(".overlay-controls").hidden = !on;
  }
  setOverlay(id, on, { date: state.date, opacity: state.opacity[id] ?? null });
  renderCredits();
  syncTimeBar({ layersChanged: true });
  writeUrl();
}

// ── gauge styling ───────────────────────────────────────────────────────────

function gaugeLegendHtml(mode) {
  const swatch = (c, l) => `<span class="sw"><i style="background:${c}"></i>${escapeHtml(l)}</span>`;
  if (mode === "record") return RECORD_BREAKS.map((b) => swatch(b.color, b.label)).join("");
  if (mode === "recent") return RECENT_BREAKS.map((b) => swatch(b.color, b.label)).join("");
  if (mode === "skill") return skillLegendHtml();
  if (mode === "now") return nowLegendHtml();
  if (mode === "bulletin") return bulletinLegendHtml();
  return "";
}

// ── the Gauges row in "On the map" (map-legend.js) ──────────────────────────

const fmtK = (n) => (n >= 10000 ? `${Math.round(n / 1000)}k` : n.toLocaleString("en-GB"));
const nowReady = () => state.gaugeStyle === "now" && state.nowStatus && state.nowMeta && !state.nowMeta.missing;

function gaugesMark() {
  if (nowReady()) return '<i class="ml-dot now" aria-hidden="true"></i>';
  return '<i class="ml-dot" aria-hidden="true"></i>';
}

// While the time bar replays a past month, the dots still show today: the row says so (a month or two back
// is as good as today, since the newest river status month trails the calendar).
function pastMonth() {
  const t = Date.parse(String(state.date || ""));
  return Number.isFinite(t) && t < Date.now() - 62 * 86400e3 ? String(state.date).slice(0, 7) : "";
}

function gaugesSummary() {
  if (nowReady()) {
    const past = pastMonth();
    return `${fmtK(state.nowStatus.size)} today vs normal${past ? `, not ${monthLabel(past)}` : ""}`;
  }
  const style = GAUGE_STYLES.find((g) => g.id === state.gaugeStyle);
  const n = state.stations.length;
  return `${n ? `${fmtK(n)}, ` : ""}by ${(style ? style.label : "agency").toLowerCase()}`;
}

function gaugesBody() {
  if (nowReady()) {
    const dots = STATUS_CLASSES.map((c) => `<i style="--c:${c.color}" title="${escapeHtml(c.label)}"></i>`).join("");
    return `<span class="ml-dots">${dots}</span>` +
      '<div class="sl-ends"><span>much below</span><span>normal</span><span>much above</span></div>' +
      `<p class="ml-when">Measured flow today against the same day in other years, at ${state.nowStatus.size.toLocaleString("en-GB")} ` +
      "gauges with a fresh record. The rest wait in the light clusters; zoom in to see them.</p>" +
      '<p class="ml-src">Colour the gauges another way in Layers.</p>';
  }
  const html = gaugeLegendHtml(state.gaugeStyle);
  return (html ? `<div class="swatches">${html}</div>` : "") +
    '<p class="ml-when">The light circles are groups of gauges: click one to zoom in.</p>' +
    '<p class="ml-src">Colour the gauges another way in Layers.</p>';
}

function registerGaugesRow() {
  registerLegendRow({
    id: "gauges", title: "Gauges",
    mark: gaugesMark,
    summary: gaugesSummary,
    on: () => state.gaugesOn !== false,
    toggle: (on) => { setGaugesVisible(on); refreshLegend("gauges"); },
    body: gaugesBody,
  });
  onTime((t) => {
    if (String(t.date).slice(0, 7) !== String(t.prev.date).slice(0, 7)) refreshLegend("gauges");
  });
}

// "Best model skill" (#518) reads skill/model_skill.parquet on first use; the dots are grey until it has
// loaded, and stay grey (with a legend that says why) when the table is not published yet.
function ensureSkillColours() {
  if (state.gaugeStyle !== "skill") return;
  loadSkillGrades().then(() => {
    actions.refreshMapData();
    if (state.gaugeStyle === "skill" && $("gauge-legend")) $("gauge-legend").innerHTML = gaugeLegendHtml("skill");
  });
}

function buildGaugeStyle() {
  const select = $("gauge-style");
  select.innerHTML = GAUGE_STYLES.map((g) => `<option value="${g.id}">${escapeHtml(g.label)}</option>`).join("");
  select.value = state.gaugeStyle;
  const apply = () => {
    setGaugeStyle(state.gaugeStyle);
    refreshLegend("gauges");
    $("gauge-legend").innerHTML = gaugeLegendHtml(state.gaugeStyle);
    $("gauge-legend").hidden = state.gaugeStyle === "source";
    $("rail-sources").classList.toggle("dimmed", !["source", "now"].includes(state.gaugeStyle));
    ensureSkillColours();
    // Today vs normal reads the daily snapshot the first time it is picked, then colours the dots.
    if (state.gaugeStyle === "now" && !state.nowStatus) {
      ensureNowStatus().then(() => {
        if (state.gaugeStyle !== "now") return;
        refreshMapData();
        setGaugeStyle("now");
        $("gauge-legend").innerHTML = gaugeLegendHtml("now");
        refreshLegend("gauges");
      });
    }
    // Last month's status reads the latest bulletin the first time it is picked (#523).
    if (state.gaugeStyle === "bulletin" && !state.bulletinStatus) {
      ensureBulletinStatus().then(() => {
        if (state.gaugeStyle !== "bulletin") return;
        refreshMapData();
        setGaugeStyle("bulletin");
        $("gauge-legend").innerHTML = gaugeLegendHtml("bulletin");
      });
    }
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
  cancelAreaContext();
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
  // Study this area (area-study.js): a multi-gauge flood study over these gauges.
  const study = document.createElement("button");
  study.className = "btn tiny primary";
  study.textContent = "Study this area";
  study.disabled = rows.length === 0;
  study.addEventListener("click", () => openAreaStudy(rows, bbox));
  box.appendChild(study);
  // Place context (#520): flood history, surface water, flood depth, dams, rain gauges, ET and soil in the box.
  const ctx = document.createElement("button");
  ctx.className = "btn tiny";
  ctx.textContent = "Context";
  ctx.title = "Flood history, surface water, flood depth, dams, rain gauges, evaporation and soil in this box";
  ctx.addEventListener("click", () => { void openAreaContext(bbox, box); });
  box.appendChild(ctx);
  // Watch (#521): new flood events and gauges above normal here, on the next visit.
  box.appendChild(areaWatchButton(bbox));
  const clear = document.createElement("button");
  clear.className = "btn tiny";
  clear.textContent = "Clear";
  clear.addEventListener("click", () => { box.hidden = true; cancelAreaContext(); });
  box.appendChild(clear);
}

function buildAreaSelect() {
  actions.showArea = showSelection;  // a watched area, opened from the Watched list (watch.js)
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
  if (state.riversOn) lines.push(RIVERS_CREDIT);
  if (state.floodsOn) lines.push(FLOODS_CREDIT);
  if (state.depthOn) lines.push(DEPTH_CREDIT);
  if (state.floodsPast) lines.push(NEWS_CREDIT, RADAR_CREDIT);
  if (state.status) lines.push(STATUS_CREDIT);
  $("rail-credits").innerHTML = lines
    .map((l) => `<div><b>${escapeHtml(l.label)}</b>: ${l.attribution} <span class="muted">(${escapeHtml(l.licence)})</span></div>`)
    .join("");
}

// ── boot ────────────────────────────────────────────────────────────────────

// The projection button in the map's own tool stack. Globe is a property of
// the view, not a layer, so it belongs next to zoom rather than in a list of
// checkboxes three groups down.
function syncGlobeButton() {
  const btn = $("btn-globe");
  if (!btn) return;
  btn.hidden = !globeSupported();
  btn.setAttribute("aria-pressed", state.globe ? "true" : "false");
  btn.title = state.globe ? "Switch to a flat map" : "Switch to the globe";
}

function buildGlobeButton() {
  const btn = $("btn-globe");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const want = !state.globe;
    state.globe = setGlobe(want) ? want : false;
    syncGlobeButton();
    writeUrl();
  });
  syncGlobeButton();
}

export function initLayerUI() {
  if (!state.date) state.date = defaultDate();
  buildBasemaps();
  buildGlobeButton();
  buildTerrain();
  buildOverlays();
  actions.setOverlay = toggleOverlay;
  actions.setBasemap = chooseBasemap;
  registerGaugesRow();
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
    actions.setStatus(state.status);
    ensureSkillColours();
    syncRailControls();
    renderCredits();
    syncTimeBar({ layersChanged: true });
  };
  if (currentBasemap() === state.basemap) rest();
  else setBasemap(state.basemap, { date: state.date, then: rest });
}

export function syncRailControls() {
  for (const input of document.querySelectorAll('#rail-basemaps input[type=radio]')) {
    input.checked = input.value === state.basemap;
  }
  reflectBasemap();
  for (const o of OVERLAYS) {
    const el = $(`ov-${o.id}`);
    if (!el) continue;
    el.checked = state.overlays.has(o.id);
    const controls = el.closest(".overlay-row").querySelector(".overlay-controls");
    controls.hidden = !el.checked;
    const range = controls.querySelector("input[type=range]");
    if (range) range.value = state.opacity[o.id] ?? o.opacity ?? 0.8;
  }
  for (const [id, val] of [["terrain", state.terrain], ["hillshade", state.hillshade]]) {
    const el = $(`toggle-${id}`);
    if (el) el.checked = Boolean(val);
  }
  syncGlobeButton();
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
