// Water balance on the map (#557): the HEC-HMS and QSWAT basin map, for the whole world.
//
// Every HydroBASINS level-4 subbasin coloured by its month: rain, evaporation, runoff or the closure error
// (P − ET − Q), as an amount or against the subbasin's 1991-2020 normal. Off by default; "Water balance" in the
// legend and the rail turns it on. It follows the time bar month by month, so play walks the months. A click opens
// the map card with the subbasin's year as bars.
//
// The numbers come from the Archive (waterbalance/, made monthly in CI by aquascope.waterbalance): the outlines
// once (about 2 MB), one file per year, the normals for the card. Colours and words are in waterbalance-core.js.

import { $, actions, clickLayers, dbg, escapeHtml, onTime, setTime, state, trace } from "./core.js?v=__BUILD__";
import { registerDatedLayer } from "./layers.js?v=__BUILD__";
import { holdSettle, map } from "./map.js?v=__BUILD__";
import { openCard } from "./map-card.js?v=__BUILD__";
import { refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { syncTimeBar } from "./time-ui.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import {
  MODES, NO_VALUE, VARIABLES, WB_BASE, WB_SHORT_CREDIT, basinCard, fillColorExpression, frameValues, hasNormals,
  monthLabel, rampCss, rampEnds, readWbParam, rowSummary, variableNote, variableOf, wbDatedLayer, wbMonthFor,
} from "./waterbalance-core.js?v=__BUILD__";

const SRC = "wb-basins";
const FILL = "wb-fill";
const LINE = "wb-line";
const KEEP_YEARS = 4;

let manifest = null;
let manifestLoad = null;
let outlines = null;          // promise of the GeoJSON
let normals = null;           // the normals file, once read
let normalsLoad = null;
const years = new Map();      // "2026" -> promise of the year file (most recent last)
let shown = null;             // the month painted
let wanted = null;
let failed = null;
let painted = new Map();      // k -> value set as feature-state
let sourceReady = false;

if (!state.wbVar) state.wbVar = "p";
if (!state.wbMode) state.wbMode = "amount";

const fetchJson = async (path) => {
  const res = await fetch(`${WB_BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
};

function loadManifest() {
  if (manifestLoad) return manifestLoad;
  manifestLoad = fetchJson("manifest.json").then((m) => {
    manifest = m;
    trace(`water balance: ${m.months} months, ${m.first_month} to ${m.last_month}`);
    return m;
  });
  manifestLoad.catch((err) => { console.warn("water balance manifest:", err && err.message); manifestLoad = null; failed = { manifest: true }; renderLegend(); });
  return manifestLoad;
}

function loadYear(y) {
  if (years.has(y)) {
    const p = years.get(y);
    years.delete(y);
    years.set(y, p);
    return p;
  }
  const p = fetchJson(`years/${y}.json`);
  years.set(y, p);
  p.catch(() => years.delete(y));
  while (years.size > KEEP_YEARS) years.delete(years.keys().next().value);
  return p;
}

function loadNormals() {
  if (normalsLoad) return normalsLoad;
  normalsLoad = fetchJson("normals.json").then((n) => { normals = n; renderLegend(); return n; });
  normalsLoad.catch(() => { normalsLoad = null; });
  return normalsLoad;
}

// ── the map layer ───────────────────────────────────────────────────────────

// Under the rivers, the flood layers, the groundwater and the gauges; over the river status and Floods past.
function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^(river-|fd-|gw-|fp-)/.test(l.id) || l.type === "symbol" ||
    ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function opacityByZoom() {
  return ["interpolate", ["linear"], ["zoom"], 2, 0.86, 6, 0.6, 9, 0.35];
}

function ensureLayer() {
  if (!state.mapOk || !map || !map.getStyle()) return false;
  if (map.getSource(SRC)) return true;
  if (!outlines) {
    outlines = fetchJson("basins.geojson");
    outlines.catch((err) => { console.warn("water balance outlines:", err && err.message); outlines = null; failed = { outlines: true }; renderLegend(); });
  }
  outlines.then((fc) => {
    if (!map.getStyle() || map.getSource(SRC)) return;
    map.addSource(SRC, { type: "geojson", data: fc, promoteId: "k", tolerance: 0.6 });
    const before = beforeId();
    map.addLayer({ id: FILL, type: "fill", source: SRC, layout: { visibility: state.waterBalance ? "visible" : "none" },
      paint: { "fill-color": fillColorExpression(state.wbVar, state.wbMode), "fill-opacity": opacityByZoom(),
        "fill-antialias": false } }, before);
    map.addLayer({ id: LINE, type: "line", source: SRC, layout: { visibility: state.waterBalance ? "visible" : "none" },
      paint: { "line-color": "#ffffff", "line-opacity": ["interpolate", ["linear"], ["zoom"], 1, 0.35, 5, 0.6],
        "line-width": ["interpolate", ["linear"], ["zoom"], 1, 0.3, 6, 0.8] } }, before);
    sourceReady = true;
    painted = new Map();
    shown = null;
    update();
  }).catch(() => {});
  return false;
}

// Drawn while on and while the map date has a month (a date outside the table draws nothing; the row says why).
function setVisibility() {
  const on = Boolean(state.waterBalance && (wanted || !manifest));
  for (const id of [FILL, LINE]) {
    if (map && map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  }
}

// Paint a month: one feature-state per subbasin, changed only where the value changed.
function paint(values) {
  if (!map.getSource(SRC)) return;
  const n = values.length;
  for (let k = 0; k < n; k++) {
    const v = values[k];
    const val = Number.isFinite(v) ? v : null;
    if (painted.get(k) === val) continue;
    map.setFeatureState({ source: SRC, id: k }, { v: val });
    painted.set(k, val);
  }
}

function restyle() {
  if (map && map.getLayer(FILL)) map.setPaintProperty(FILL, "fill-color", fillColorExpression(state.wbVar, state.wbMode));
}

function update() {
  if (!state.waterBalance || !state.mapOk) { renderLegend(); return; }
  if (!manifest) { loadManifest().then(update).catch(() => {}); renderLegend(); return; }
  const month = wbMonthFor(state.date, manifest);
  wanted = month;
  setVisibility();
  if (!month) { shown = null; renderLegend(); return; }
  if (!sourceReady) { ensureLayer(); renderLegend(); return; }
  const key = `${month}|${state.wbVar}|${state.wbMode}`;
  if (key === shown) { renderLegend(); return; }
  renderLegend();
  const job = loadYear(month.slice(0, 4)).then((doc) => {
    if (wanted !== month || !state.waterBalance) return;
    const values = frameValues(doc, month, state.wbVar, state.wbMode);
    paint(values || []);
    shown = key;
    failed = null;
    renderLegend();
    preload(month);
  }).catch((err) => {
    console.warn(`water balance ${month}:`, err && err.message);
    if (wanted === month) { failed = { month }; renderLegend(); }
  });
  holdSettle(job);   // a play waits for the frame
}

// While playing into December, read the next year's file ahead.
function preload(month) {
  if (!state.playing || !month.endsWith("-12")) return;
  const next = String(Number(month.slice(0, 4)) + 1);
  if (manifest && (manifest.years || []).includes(next)) loadYear(next).catch(() => {});
}

// ── the row in "On the map" ─────────────────────────────────────────────────

function renderLegend() { refreshLegend("water-balance"); }

function rowBody() {
  const v = state.wbVar, m = state.wbMode;
  const seg = (list, cur, act) => `<div class="wb-seg wb-seg-${list.length}" role="group">${list.map((x) =>
    `<button type="button" class="wb-seg-b${x.id === cur ? " on" : ""}" data-act="${act}:${x.id}" aria-pressed="${x.id === cur}">${escapeHtml(x.label)}</button>`).join("")}</div>`;
  const [lo, mid, hi] = rampEnds(v, m);
  const noNormal = m === "anom" && normals && !hasNormals(normals);
  if (m === "anom" && !normals) loadNormals().catch(() => {});
  const era5t = manifest && shown && manifest.era5_final_until && wanted && wanted > String(manifest.era5_final_until).slice(0, 7);
  return seg(VARIABLES, v, "var") + seg(MODES, m, "mode") +
    `<span class="wb-ramp" style="background:${rampCss(v, m)}"></span>` +
    `<div class="wb-ends"><span>${escapeHtml(lo)}</span><span>${escapeHtml(mid)}</span><span>${escapeHtml(hi)}</span></div>` +
    `<p class="wb-note">${escapeHtml(variableNote(v, m, { normals: !noNormal }))}</p>` +
    (v === "q" || v === "closure" || m === "anom"
      ? `<p class="wb-note muted"><i class="wb-none" style="background:${NO_VALUE}"></i>${m === "anom" && v !== "q" && v !== "closure"
        ? "no normal yet" : "no runoff worked out here (the card says why)"}${era5t ? "; ERA5T, preliminary" : ""}</p>`
      : era5t ? '<p class="wb-note muted">ERA5T, preliminary</p>' : "") +
    `<p class="ml-src">${escapeHtml(WB_SHORT_CREDIT)} <button type="button" class="ml-link" data-act="info">About</button></p>`;
}

function registerRow() {
  registerLegendRow({
    id: "water-balance", title: "Water balance",
    mark: () => `<i class="wb-mark" style="background:${rampCss(state.wbVar, state.wbMode)}"></i>`,
    summary: () => rowSummary({ manifest, month: wanted, variable: state.wbVar, mode: state.wbMode,
      loading: Boolean(wanted && shown !== `${wanted}|${state.wbVar}|${state.wbMode}` && !failed),
      failed: Boolean(failed), date: state.date }),
    on: () => Boolean(state.waterBalance && state.mapOk),
    empty: () => Boolean(manifest && !wanted),
    toggle: (on) => chooseWaterBalance(on),
    body: rowBody,
    act: (name) => {
      if (name === "info") about();
      else if (name.startsWith("var:")) setWaterBalanceView({ variable: name.slice(4) });
      else if (name.startsWith("mode:")) setWaterBalanceView({ mode: name.slice(5) });
    },
  });
}

/** Switch what is painted: { variable: "p" | "et" | "q" | "closure", mode: "amount" | "anom" } (Ask the map too). */
export function setWaterBalanceView({ variable, mode } = {}) {
  if (variable && VARIABLES.some((x) => x.id === variable)) state.wbVar = variable;
  if (mode && MODES.some((x) => x.id === mode)) state.wbMode = mode;
  if (state.wbMode === "anom") loadNormals().catch(() => {});
  restyle();
  update();
  renderLegend();
  writeUrl();
  return { variable: state.wbVar, mode: state.wbMode };
}

function about() {
  const m = manifest || {};
  openModal("Water balance", `
    <p>Each HydroBASINS level-4 subbasin (1,342 of them, about 100,000 km² on average) is coloured by its month:</p>
    <p><strong>Rain</strong>: ERA5 total precipitation. <strong>Evaporation</strong>: ERA5 total evaporation, which over
    land is the actual evapotranspiration. Both are summed over the month's hours and averaged over the 0.25° cells whose
    centres fall in the subbasin, weighted by area. <strong>Runoff</strong>: the runoff the subbasin makes itself, the
    GEOGLOWS v2 monthly flow at its outlet reach minus the flow coming in from the subbasins upstream, spread over its
    area. <strong>Closure</strong>: P − ET − Q.</p>
    <p>Runoff is withheld, and the card says why, where that difference cannot be trusted: the reach paired with the
    outlet carries less than half the inflow (a delta, a coast with several outlets), the river brings in more than a
    metre of water a month over the subbasin, the outlet carries more than the rain over the record, or the inflow
    comes from a subbasin that fails one of these. The flows stay in the published table.</p>
    <p><strong>The closure error is shown, not hidden.</strong> It holds the change in storage (soil, snow, groundwater,
    lakes and reservoirs), which swings with the seasons, and the errors of three products made separately. Over whole
    years it should be small; where it is not, one of the three is off. GEOGLOWS is itself driven by ERA5 runoff, so the
    three are not independent.</p>
    <p><strong>vs normal</strong> is the month minus the subbasin's 1991-2020 mean for that calendar month.
    ET is ERA5's rather than WaPOR's: WaPOR covers Africa and the Near East since 2018, this map the world since 1990.</p>
    <p>${m.first_month ? `${escapeHtml(monthLabel(m.first_month))} to ${escapeHtml(monthLabel(m.last_month))}, ` +
      `${Number(m.runoff_ok || 0).toLocaleString()} of ${Number(m.basins || 0).toLocaleString()} subbasins with runoff. ` : ""}
    Months after ERA5's final data are ERA5T (preliminary) and are recomputed when final. Made monthly in CI;
    the same in Python: <code>aquascope waterbalance --at LAT LON</code>.</p>
    <p class="muted">ERA5 hourly single levels (ARCO-ERA5 on Google Cloud), CC BY 4.0, Copernicus Climate Change Service /
    ECMWF, Hersbach et al. (2020); contains modified Copernicus Climate Change Service information. GEOGLOWS v2
    retrospective simulation and HydroSOS tables, CC BY 4.0. HydroBASINS level-4 subbasins as published in BasinATLAS
    v1.0 (Linke et al. 2019), CC BY 4.0. Modelled, not measured.</p>`);
}

// ── the card ────────────────────────────────────────────────────────────────

const OWN = new Set([FILL, LINE]);
function othersHit(e) {
  const others = ["points", "clusters", "now-dots", ...[...clickLayers].filter((id) => !OWN.has(id))].filter((id) => map.getLayer(id));
  return others.length && map.queryRenderedFeatures(e.point, { layers: others }).length > 0;
}

async function onClick(e) {
  if (!state.waterBalance || !e.features || !e.features.length || othersHit(e)) return;
  const props = e.features[0].properties || {};
  const k = Number(props.k);
  const month = wanted;
  const id = `wb:${props.id}:${month}`;
  const lngLat = [e.lngLat.lng, e.lngLat.lat];
  const base = { id, lngLat, lift: 6, what: "Water balance", credit: WB_SHORT_CREDIT,
    buttons: [
      { id: "here", label: "This place", title: "The river and the gauges here", onClick: () => actions.selectPoint(lngLat[1], lngLat[0]) },
      { id: "about", label: "About", title: "How the balance is made, and what the closure error is", onClick: about },
    ] };
  if (!month) {
    openCard({ ...base, title: "Level-4 subbasin", sub: `HydroBASINS ${props.id}`,
      status: { text: rowSummary({ manifest, month: null, variable: state.wbVar, mode: state.wbMode, date: state.date }), muted: true } });
    return;
  }
  openCard({ ...base, title: "Level-4 subbasin", sub: `HydroBASINS ${props.id}`, status: { pending: "Reading the year…" } });
  try {
    const [doc] = await Promise.all([loadYear(month.slice(0, 4)), loadNormals().catch(() => null)]);
    const c = basinCard(doc, k, month, props, { normals, variable: state.wbVar, mode: state.wbMode });
    openCard({ ...base, title: c.title, sub: c.sub, status: c.status, chart: c.chart, note: c.note });
  } catch (err) {
    openCard({ ...base, title: "Level-4 subbasin", sub: `HydroBASINS ${props.id}`,
      status: { text: `Could not read ${month.slice(0, 4)} just now.`, muted: true } });
  }
}

// ── on and off ──────────────────────────────────────────────────────────────

/** Show or hide the layer (from the URL, Back, the rail or the legend). */
export function setWaterBalance(on, { write = true } = {}) {
  state.waterBalance = Boolean(on);
  const toggle = $("toggle-water-balance");
  if (toggle) toggle.checked = state.waterBalance;
  if (state.mapOk && map) {
    if (state.waterBalance) {
      ensureLayer();
      loadManifest().then((m) => {
        // A date past the newest month (today, at first) moves to the newest month, by the month.
        if (state.waterBalance && !wbMonthFor(state.date, m) && String(state.date || "").slice(0, 7) > m.last_month) {
          setTime({ date: `${m.last_month}-15`, step: "month" }, { source: "layer" });
        } else if (state.waterBalance && state.timeStep === "day") {
          setTime({ step: "month" }, { source: "layer" });
        }
        syncTimeBar({ layersChanged: true });
        update();
      }).catch(() => {});
    }
    setVisibility();
  }
  renderCredits();
  syncTimeBar({ layersChanged: true });
  renderLegend();
  if (write) writeUrl();
}

// The river status colours the same level-4 basins and would sit under this layer, its caption talking about
// another map: the reader's choice turns it off while the water balance is on, and back on after.
let statusWasOn = false;
function chooseWaterBalance(on) {
  if (on && state.status && actions.setStatus) { statusWasOn = true; actions.setStatus(false); }
  setWaterBalance(on);
  if (!on && statusWasOn && actions.setStatus) { statusWasOn = false; actions.setStatus(true); renderCredits(); writeUrl(); }
}

export function initWaterBalance() {
  dbg.waterBalance = () => ({ manifest, shown, wanted, variable: state.wbVar, mode: state.wbMode });
  registerRow();
  registerDatedLayer(() => (state.waterBalance ? wbDatedLayer(manifest) : null));
  const fromUrl = readWbParam(location.hash);
  if (fromUrl) {
    state.waterBalance = fromUrl.on;
    state.wbVar = fromUrl.variable;
    state.wbMode = fromUrl.mode;
  }
  const toggle = $("toggle-water-balance");
  if (toggle) {
    toggle.checked = Boolean(state.waterBalance);
    toggle.addEventListener("change", (e) => chooseWaterBalance(e.target.checked));
  }
  actions.setWaterBalance = setWaterBalance;
  actions.setWaterBalanceView = setWaterBalanceView;
  if (!state.mapOk || !map) return;
  clickLayers.add(FILL);
  map.on("click", FILL, (e) => { void onClick(e); });
  map.on("style.load", () => {
    sourceReady = false;
    painted = new Map();
    shown = null;
    setTimeout(() => { if (state.waterBalance) { ensureLayer(); } }, 0);
  });
  onTime((t) => {
    if (t.date !== t.prev.date || t.playing !== t.prev.playing) update();
  });
  window.addEventListener("hashchange", () => {
    const want = readWbParam(location.hash);
    const next = want ? want.on : false;
    if (next !== Boolean(state.waterBalance)) setWaterBalance(next, { write: false });
  });
  if (state.waterBalance) setWaterBalance(true, { write: false });
}
