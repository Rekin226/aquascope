// The Now tab (#517): today against normal in one sentence, and the next 15
// days from GEOGLOWS and GloFAS with the return-period lines, corrected to the
// gauge where there is one. Every number comes from aquascope.nownext in the
// worker; this module asks for it and lays it out. The map date (onTime)
// moves a marker across the forecast plot.

import { $, escapeHtml, onTime, state } from "./core.js?v=__BUILD__";
import { emphasisColor, plot } from "./charts.js?v=__BUILD__";
import { addMethodOnce } from "./methods.js?v=__BUILD__";
import { FORECAST_CREDIT, forecastTraces, markerShapes, plotRange, skillText, statusClass } from "./now-core.js?v=__BUILD__";
import { setCard, setTab } from "./shell.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";

const TARGETS = {
  st: { panel: "panel-station", methods: "methods" },
  pt: { panel: "panel-point", methods: "pt-methods" },
};
// Records that have a "normal" for the day; rainfall is mostly dry days.
const STATUS_VARIABLES = new Set(["discharge", "water_level", "groundwater_level"]);
const runs = { st: { run: 0 }, pt: { run: 0 } };

const panel = (t) => $(TARGETS[t].panel);
const box = (t) => $(`${t}-now`);
const part = (t, cls) => box(t) && box(t).querySelector(`.${cls}`);

function skeleton(t) {
  const el = box(t);
  if (!el) return;
  el.innerHTML = `
    <div class="card now-status" hidden>
      <div class="card-body">
        <p class="now-line"><span class="now-chip" aria-hidden="true"></span><span class="now-text"></span></p>
        <p class="muted now-sub"></p>
      </div>
    </div>
    <div class="card now-forecast" hidden>
      <div class="card-title"><h3>Next 15 days <span class="muted">modelled</span></h3></div>
      <div class="card-body">
        <p class="now-peak"></p>
        <p class="muted now-pending" role="status" hidden></p>
        <div id="plot-${t}-now" class="plot"></div>
        <p class="muted now-skill"></p>
        <p class="basin-foot muted now-foot"></p>
      </div>
    </div>`;
}

function isShown(t) {
  const tab = panel(t) && panel(t).querySelector('[role="tab"][data-tab="now"]');
  return Boolean(tab && tab.getAttribute("aria-selected") === "true");
}

// A new gauge or point: forget the last one and wait for what this one needs.
export function resetNow(t, reason = "Loading…") {
  const r = runs[t];
  r.run++;
  Object.assign(r, { ctx: null, loaded: false, plotted: null, fc: null, recent: null, range: null });
  skeleton(t);
  setTab(panel(t), "now", { enabled: false, reason });
}

// ctx for a gauge: { station, result, snap }; for a point: { lat, lon, snap }. `snap` is the river snap's
// promise (river.js), which the forecast waits on to know the reach.
export function startNow(t, ctx) {
  const r = runs[t];
  r.ctx = ctx;
  r.loaded = false;
  if (t === "st") {
    const res = ctx.result || {};
    if (res.error || !STATUS_VARIABLES.has(res.variable)) {
      setTab(panel(t), "now", { enabled: false, reason: "Today against normal needs a flow or level record." });
      return;
    }
  }
  setTab(panel(t), "now", { enabled: true });
  if (isShown(t)) load(t);
}

function load(t) {
  const r = runs[t];
  if (!r.ctx || r.loaded) return;
  r.loaded = true;
  const my = r.run;
  if (t === "st") loadStatus(t, my);
  loadForecast(t, my);
}

function renderStatus(t, st, { modelled = false } = {}) {
  const card = part(t, "now-status");
  if (!st || (!st.sentence && !st.error)) { card.hidden = true; return; }
  const cls = statusClass(st.class);
  const chip = part(t, "now-chip");
  chip.style.background = cls ? cls.color : "transparent";
  chip.hidden = !cls;
  part(t, "now-text").textContent = st.sentence || st.error;
  const bits = [];
  if (st.top_up) bits.push(st.top_up);
  if (cls) {
    bits.push(modelled
      ? "Against the reach's own simulated record, 7 days either side of this date in every year."
      : "Against this record, 7 days either side of this date in every other year (USGS and WMO HydroSOS classes).");
  }
  part(t, "now-sub").textContent = bits.join(" ");
  setCard(card, "ready");
}

async function loadStatus(t, my) {
  const r = runs[t];
  const card = part(t, "now-status");
  const s = r.ctx.station;
  setCard(card, "loading", { message: "Placing today in the record (and asking the agency for its newest days)…" });
  try {
    const st = await call("now", { op: "status", args: { source: s.source, station_id: s.station_id } });
    if (my !== r.run) return;
    if (st.error && !st.sentence) { setCard(card, "empty", { message: st.error }); return; }
    renderStatus(t, st);
    r.recent = st.recent || null;
    if (r.fc) drawForecast(t);
  } catch (err) {
    if (my !== r.run) return;
    setCard(card, "error", { message: `Could not place today in the record: ${err.message}`, retry: () => loadStatus(t, my) });
  }
}

// Full forecasts already answered this session, by what was asked: going back to a place is instant.
const answered = new Map();
const ANSWERED_MAX = 24;
function remember(key, fc) {
  answered.delete(key);
  answered.set(key, fc);
  while (answered.size > ANSWERED_MAX) answered.delete(answered.keys().next().value);
}

// The forecast in two steps. A light worker reads the next 15 days of GEOGLOWS at once (the quick forecast:
// no simulated record, so no thresholds), and the plot and its sentence go up as soon as it lands. The main
// worker then adds what needs the reach's simulated record since 1940 (the thresholds, the status of a
// click's river) and, for a gauge, its record (GloFAS's cell picked by flow, the correction), reusing the
// GEOGLOWS part rather than reading it again.
async function loadForecast(t, my) {
  const r = runs[t];
  const card = part(t, "now-forecast");
  setCard(card, "loading", { message: "Asking GEOGLOWS for the next 15 days…" });
  try {
    const snap = await (r.ctx.snap || Promise.resolve(null));
    if (my !== r.run) return;
    const args = {};
    if (t === "st") {
      const s = r.ctx.station, res = r.ctx.result || {};
      Object.assign(args, { lat: s.lat, lon: s.lon, source: s.source, station_id: s.station_id });
      if (res.variable === "discharge") {
        args.use_gauge = true;
        if (res.stats && Number.isFinite(res.stats.mean)) args.match_mean_flow = res.stats.mean;
      }
    } else {
      Object.assign(args, { lat: r.ctx.lat, lon: r.ctx.lon });
    }
    if (snap && snap.snapped) {
      args.river_id = snap.river_id;
      // A click's GloFAS cell is read where the river is, not on the hillside beside it.
      if (t === "pt") Object.assign(args, { lat: snap.snap_lat, lon: snap.snap_lon });
    }
    // A gauge's corrected forecast depends on the record loaded (its period), so only the rest are kept.
    const key = args.use_gauge ? null : JSON.stringify(args);
    if (key && answered.has(key)) { showForecast(t, answered.get(key), { done: true }); return; }
    // Without a reach there is only GloFAS, and the quick answer is the whole answer. A gauge's GloFAS cell is
    // picked by its mean flow (aquascope.explore, which needs pandas), so that one goes to the main worker.
    if (!args.river_id) {
      const fc = args.match_mean_flow != null
        ? await call("now", { op: "forecast", args })
        : await callLight("now", { op: "forecast", args }, { priority: 1 });
      if (my !== r.run) return;
      if (key) remember(key, fc);
      showForecast(t, fc, { done: true });
      return;
    }
    let quick = null;
    try {
      quick = await callLight("now", { op: "forecast", args: { ...args, history: false, glofas: false } }, { priority: 1 });
    } catch (err) {
      console.info("quick forecast:", err && err.message);
    }
    if (my !== r.run) return;
    if (quick && quick.geoglows && !quick.geoglows.error) showForecast(t, quick, { done: false });
    const full = await call("now", { op: "forecast", args: { ...args, known_geoglows: quick && !(quick.geoglows || {}).error ? quick.geoglows : null } });
    if (my !== r.run) return;
    if (key) remember(key, full);
    showForecast(t, full, { done: true });
  } catch (err) {
    if (my !== r.run) return;
    if (r.fc) { pending(t, ""); return; }   // the quick forecast is up; the rest did not come
    setCard(card, "error", { message: "The forecast services did not answer this time; GEOGLOWS is slow now and then.",
      retry: () => loadForecast(t, my) });
  }
}

function pending(t, text) {
  const el = part(t, "now-pending");
  if (!el) return;
  el.textContent = text;
  el.hidden = !text;
}

function showForecast(t, fc, { done }) {
  const r = runs[t];
  r.fc = fc;
  if (t === "pt" && done) renderStatus(t, fc.status, { modelled: true });
  drawForecast(t);
  pending(t, done ? "" : (t === "st"
    ? "Adding GloFAS, the correction to this gauge and the flood thresholds (they need the reach's simulated record since 1940)…"
    : "Adding GloFAS and the flood thresholds (they need the reach's simulated record since 1940)…"));
  if (done) for (const m of fc.methods || []) if (m && m.name) addMethodOnce(TARGETS[t].methods, m);
}

function layoutFor(t) {
  const r = runs[t];
  // The legend sits under the plot: on a phone-width panel a legend above wraps down over the lines.
  return { height: 300, margin: { b: 20 }, yaxis: { title: { text: "m³/s" }, rangemode: "tozero" },
    legend: { orientation: "h", x: 0, y: -0.14, yanchor: "top" },
    shapes: markerShapes(state.date, r.range, emphasisColor()) };
}

function drawForecast(t) {
  const r = runs[t];
  const fc = r.fc;
  const card = part(t, "now-forecast");
  const traces = forecastTraces(fc, { recent: t === "st" && r.ctx.result?.variable === "discharge" ? r.recent : null,
    ink: emphasisColor() });
  if (!traces.length) {
    setCard(card, "empty", { message: fc.sentence || "No forecast answered for this place." });
    return;
  }
  r.range = plotRange(fc, r.recent);
  setCard(card, "ready");
  part(t, "now-peak").textContent = fc.sentence || "";
  plot(`plot-${t}-now`, traces, layoutFor(t), `forecast-${fc.river_id || "point"}`);
  r.plotted = traces;
  const skill = skillText(fc);
  part(t, "now-skill").textContent = skill;
  part(t, "now-skill").hidden = !skill;
  const notes = [];
  for (const k of ["geoglows", "glofas"]) if (fc[k] && fc[k].error) notes.push(fc[k].error);
  if (fc.glofas && fc.glofas.cell_note) notes.push(fc.glofas.cell_note);
  const shaded = fc.geoglows && !fc.geoglows.error ? "Shaded: the middle half and the full range of the GEOGLOWS ensemble. " : "";
  part(t, "now-foot").innerHTML = `${shaded}${notes.map((n) => `${escapeHtml(n)} `).join("")}Model output, not measurements. ` +
    `Data: ${escapeHtml(FORECAST_CREDIT)}.`;
}

// The map date moves the marker; the plot is re-drawn so a theme change keeps it.
function onDate() {
  for (const t of Object.keys(runs)) {
    const r = runs[t];
    if (!r.plotted || !$(`plot-${t}-now`)) continue;
    plot(`plot-${t}-now`, r.plotted, layoutFor(t), `forecast-${(r.fc && r.fc.river_id) || "point"}`);
  }
}

export function initNow() {
  for (const t of Object.keys(TARGETS)) {
    const root = panel(t);
    if (!root) continue;
    // "Use the nearest reach" on a click away from any river: the forecast is asked again for that reach
    // (GEOGLOWS and the GloFAS cell on the river), not left as the hillside's GloFAS-only answer.
    root.addEventListener("reachchange", (e) => {
      const reach = e.detail;
      const r = runs[t];
      if (!reach || !reach.chosen || !r.ctx) return;
      const snap = { snapped: true, river_id: reach.river_id, snap_lat: reach.lat, snap_lon: reach.lon };
      r.run++;
      Object.assign(r, { ctx: { ...r.ctx, snap: Promise.resolve(snap) }, loaded: false, plotted: null, fc: null, recent: null, range: null });
      skeleton(t);
      if (isShown(t)) load(t);
    });
    root.addEventListener("tabchange", (e) => {
      if (e.detail.tab !== "now") return;
      load(t);
      const el = $(`plot-${t}-now`);
      if (el && el.offsetParent !== null && el.data) Plotly.Plots.resize(el);
    });
  }
  onTime((change) => { if (change.date !== change.prev.date) onDate(); });
}
