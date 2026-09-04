// The station inspector: pick a gauge, run aquascope on it in the worker, and
// fill the tabs (Overview, Floods, Flows, Model, Catchment, Similar, Methods).
// The science is unchanged; what is new is that each tab reports its own state
// and can be cancelled, and that the record and every table export.

import { CONFIG } from "../config.js?v=__BUILD__";
import {
  $, VAR_LABEL, actions, article, copyText, downloadBlob, escapeHtml, fmt, fmtP, sourceStyle, state, stationKey,
} from "./core.js?v=__BUILD__";
import { addTableDownload, emphasisColor, plot, surfaceColor } from "./charts.js?v=__BUILD__";
import { requestAssess } from "./assess.js?v=__BUILD__";
import { clearCatchment, requestBasin, requestCatchment, stationArea } from "./basins.js?v=__BUILD__";
import { flyToStation, highlightStation, clearPointMarker } from "./map.js?v=__BUILD__";
import { GR4J_METHODS, addMethodOnce, methodsOnPage, openCite, renderMethodList } from "./methods.js?v=__BUILD__";
import { hideCard, selectTab, setCard, setStatusEl, setTab, showSurface } from "./shell.js?v=__BUILD__";
import { Cancelled, call, callCancelable } from "./worker-client.js?v=__BUILD__";
import { canonicalUrl, writeUrl } from "./url.js?v=__BUILD__";

let analysisRun = 0;
let gr4jRun = 0;
let ciCall = null;
let gr4jCancel = null;

const root = () => $("panel-station");
const setStatus = (text, kind = "info") => setStatusEl($("status"), text, kind);

// Display units (#316). Discharge can be shown in ft³/s (cfs), but only shown:
// the worker, the GR4J model and the CSV downloads never see anything but the
// agency's unit, so the conversion lives here at render time and nowhere else.
const CFS_PER_CMS = 35.314666721;
const UNIT_PREF_KEY = "aquascope-unit";
const isCms = (u) => /m3\/s|m³\/s/.test(u || "");
let unitPref = "m3s";
try { if (localStorage.getItem(UNIT_PREF_KEY) === "cfs") unitPref = "cfs"; } catch { /* storage denied */ }
const cfsOn = (rawUnit) => unitPref === "cfs" && isCms(rawUnit);
const dUnit = (rawUnit) => (cfsOn(rawUnit) ? "ft³/s" : rawUnit);
const dVal = (x, rawUnit) => (cfsOn(rawUnit) && x !== null && x !== undefined ? x * CFS_PER_CMS : x);
const dArr = (a, rawUnit) => (cfsOn(rawUnit) ? Array.from(a, (v) => (v === null || v === undefined ? v : v * CFS_PER_CMS)) : a);

export function selectStation(key, { fly = false, tab = null, push = true } = {}) {
  const r = state.byKey.get(key);
  if (!r) return;
  const my = ++analysisRun;
  state.selected = r;
  state.point = null;
  state.result = null;
  highlightStation(key);
  clearPointMarker();
  if (fly) flyToStation(r);

  showSurface("panel-station");
  const st = sourceStyle(r.source);
  const badge = $("st-source");
  badge.textContent = st.label;
  badge.style.background = st.color;
  $("st-name").textContent = r.name || r.station_id;
  $("st-id").textContent = r.station_id;
  $("st-vars").textContent = (r.variables || []).map((v) => VAR_LABEL[v] || v).join(", ") || "—";
  $("st-period").textContent = r.period_start ? ` · ${r.period_start} → ${r.period_end || "present"}` : "";
  const agency = $("st-agency");
  if (r.url) { agency.href = r.url; agency.hidden = false; } else agency.hidden = true;
  $("btn-csv").disabled = true;
  $("btn-unit").hidden = true;

  // Reset the tabs to "loading" so nothing from the last station lingers.
  for (const id of ["st-kpis-card", "st-hydro-card", "st-ffa-card", "st-fdc-card", "st-trend-card", "st-gr4j-card", "st-notes-card", "st-assess-card"]) {
    hideCard($(id));
  }
  renderMethodList("methods", []);
  $("attribution").textContent = "";
  resetGr4j();
  clearCatchment();
  for (const name of ["floods", "flows", "model", "catchment", "similar"]) {
    setTab(root(), name, { enabled: false, reason: "Loading the record…", count: null });
  }
  setTab(root(), "overview", { enabled: true });
  setTab(root(), "methods", { enabled: true });
  // Record the selection before the tab is applied: the tab change only ever
  // replaces the URL, so pushing first is what makes Back leave this station.
  state.activeTab = tab && tabExists(tab) ? tab : "overview";
  writeUrl({ push });
  selectTab(root(), state.activeTab);

  setCard($("st-kpis-card"), "loading", { message: state.workerReady ? "Fetching the record from the agency…" : "Loading Python in your browser (about 15 MB, once)…" });
  requestAnalysis(r, my);
  requestCatchment({ station: r, target: "st" });
  requestBasin(r.lat, r.lon, "st");
  requestAssess({ lat: r.lat, lon: r.lon, target: "st", key });
}

function tabExists(name) {
  return Boolean(root().querySelector(`[role="tab"][data-tab="${name}"]`));
}

async function requestAnalysis(r, my) {
  const key = stationKey(r);
  setStatus("");
  try {
    const result = await call("analyze", {
      source: r.source, station_id: r.station_id, years: CONFIG.years, period_start: r.period_start || null,
    });
    if (my !== analysisRun || !state.selected || stationKey(state.selected) !== key) return; // user moved on
    state.result = result;
    render(result, r);
  } catch (err) {
    if (my !== analysisRun) return;
    setCard($("st-kpis-card"), "error", {
      message: `Could not analyse this station: ${err.message}`,
      retry: () => requestAnalysis(r, my),
    });
  }
}

function render(res, r) {
  const st = sourceStyle(r.source);
  const rawUnit = res.unit || "";
  const unit = dUnit(rawUnit);
  const varLabel = VAR_LABEL[res.variable] || res.variable;
  const ub = $("btn-unit");
  if (ub) {
    ub.hidden = !isCms(rawUnit);
    ub.textContent = unitPref === "cfs" ? "Show m³/s" : "Show ft³/s";
  }

  if (res.error || !res.n) {
    setCard($("st-kpis-card"), "empty", { message: res.error || "The agency returned no observations for this station." });
    renderNotes(res);
    renderMethods(res);
    return;
  }
  $("btn-csv").disabled = false;

  // Overview: KPIs + hydrograph
  const k = res.stats || {};
  $("kpis").innerHTML = [
    // Years, not two full dates: "1986-08-26 → 2026-08-22" wrapped onto a second
    // line while the three tiles beside it sat on one, and the exact days are
    // already on the line under the station name.
    ["record", `${String(res.start).slice(0, 4)}–${String(res.end).slice(0, 4)}`,
      `${res.years} yr · ${res.n.toLocaleString()} obs`],
    ["mean", `${fmt(dVal(k.mean, rawUnit))} ${unit}`, varLabel],
    ["max", `${fmt(dVal(k.max, rawUnit))} ${unit}`, "observed"],
    ["min", `${fmt(dVal(k.min, rawUnit))} ${unit}`, "observed"],
  ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  setCard($("st-kpis-card"), "ready");

  const traces = [{
    x: res.series.t, y: dArr(res.series.v, rawUnit), mode: "lines", line: { width: 1, color: st.color }, name: varLabel,
    hovertemplate: "%{x}<br>%{y:.3~f} " + unit + "<extra></extra>",
  }];
  if (res.annual_max && res.annual_max.year.length > 1) {
    traces.push({
      x: res.annual_max.year.map((y) => `${y}-07-01`), y: dArr(res.annual_max.v, rawUnit), mode: "markers",
      // Ink with a ring punched out of the card, not a second hue: these mark
      // the same series, and they have to read beside any of the six agency
      // colours (red markers on the UK's green line are ΔE 5.5 under protanopia).
      marker: { color: emphasisColor(), size: 6, line: { color: surfaceColor(), width: 1.5 } },
      name: "annual max",
      hovertemplate: "%{x|%Y} annual max<br>%{y:.3~f} " + unit + "<extra></extra>",
    });
  }
  setCard($("st-hydro-card"), "ready");
  plot("plot-hydro", traces, { yaxis: { title: { text: unit }, rangemode: "tozero" }, showlegend: false },
    `${r.source}-${r.station_id}-record`);

  // Floods
  if (res.ffa && res.ffa.fits) {
    setTab(root(), "floods", { enabled: true });
    setCard($("st-ffa-card"), "ready");
    $("ffa-years").textContent = `(${res.ffa.n_years} annual maxima)`;
    renderFfaTable(res.ffa, rawUnit);
    renderFfaPlot(res.ffa, rawUnit, st.color, r);
    $("btn-ci").disabled = Boolean(res.ffa.fits.gev_bootstrap);
  } else {
    setTab(root(), "floods", { enabled: false, reason: "Flood frequency needs annual maxima from a multi-year daily record." });
  }

  // Flows: FDC + trend
  const hasFdc = Boolean(res.fdc), hasTrend = Boolean(res.trend);
  if (hasFdc) {
    setCard($("st-fdc-card"), "ready");
    const q95 = dVal(res.fdc.q95, rawUnit), q10 = dVal(res.fdc.q10, rawUnit);
    plot("plot-fdc", [{
      x: res.fdc.exceedance, y: dArr(res.fdc.q, rawUnit), mode: "lines", line: { color: st.color, width: 2 },
      hovertemplate: "%{x:.1f} % exceedance<br>%{y:.3~f} " + unit + "<extra></extra>",
    }], {
      xaxis: { title: { text: "% of time exceeded" }, range: [0, 100] },
      yaxis: { title: { text: unit }, type: "log" }, showlegend: false,
      annotations: [
        { x: 95, y: Math.log10(q95 || 1), text: `Q95 ${fmt(q95)}`, showarrow: true, arrowhead: 2, ax: -40, ay: -30 },
        { x: 10, y: Math.log10(q10 || 1), text: `Q10 ${fmt(q10)}`, showarrow: true, arrowhead: 2, ax: 40, ay: -30 },
      ],
    }, `${r.source}-${r.station_id}-flow-duration`);
  } else hideCard($("st-fdc-card"));

  if (hasTrend) {
    const t = res.trend;
    const dir = t.trend === "no trend" ? "no significant trend" : `${article(t.trend)} ${t.trend} trend`;
    $("trend-text").innerHTML = `Mann-Kendall on ${t.n_years} annual means: <strong>${dir}</strong> ` +
      `(p = ${fmtP(t.p_value)}, τ = ${fmt(t.tau, 2)}). Sen's slope ${fmt(dVal(t.sens_slope_per_year, rawUnit), 3)} ${escapeHtml(unit)}/yr.`;
    setCard($("st-trend-card"), "ready");
  } else hideCard($("st-trend-card"));

  setTab(root(), "flows", hasFdc || hasTrend
    ? { enabled: true }
    : { enabled: false, reason: "Flow duration and trend need a daily record." });

  // Model (GR4J): discharge in m3/s, long enough, and the JS model loaded
  const modelOk = res.variable === "discharge" && isCms(rawUnit) && res.series && res.series.t.length > 365 * 4 && window.GR4J;
  setTab(root(), "model", modelOk
    ? { enabled: true }
    : { enabled: false, reason: "GR4J needs four or more years of daily discharge in m³/s." });
  if (modelOk) setCard($("st-gr4j-card"), "ready");

  renderNotes(res);
  renderMethods(res);
}

function renderNotes(res) {
  const notes = [...(res.notes || [])];
  if (res.fetch_note) notes.unshift(res.fetch_note);
  if (!notes.length) { hideCard($("st-notes-card")); return; }
  $("notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  setCard($("st-notes-card"), "ready");
}

function renderMethods(res) {
  renderMethodList("methods", res.methods || []);
  $("attribution").textContent = res.attribution
    ? `Data: ${res.attribution}. Licence: ${res.license}. Computed with aquascope in your browser.`
    : "";
}

function renderFfaTable(ffa, rawUnit) {
  const unit = dUnit(rawUnit);
  const cv = (x) => dVal(x, rawUnit);
  const rps = ffa.return_periods;
  const g = ffa.fits.gev_lmoments || {}, l = ffa.fits.lp3 || {}, b = ffa.fits.gev_bootstrap;
  const head = `<tr><th>T (yr)</th><th>GEV L-moments</th><th>LP3 (90 % CI)</th>${b ? "<th>GEV bootstrap (90 % CI)</th>" : ""}</tr>`;
  const rows = rps.map((rp, i) => {
    const gq = g.q ? fmt(cv(g.q[i])) : (g.error ? "n/a" : "—");
    const lq = l.q ? `${fmt(cv(l.q[i]))} <span class="ci">${l.ci && l.ci[i] ? `[${fmt(cv(l.ci[i][0]))}, ${fmt(cv(l.ci[i][1]))}]` : ""}</span>` : (l.error ? "n/a" : "—");
    const bq = b ? `${fmt(cv(b.q[i]))} <span class="ci">${b.ci && b.ci[i] ? `[${fmt(cv(b.ci[i][0]))}, ${fmt(cv(b.ci[i][1]))}]` : ""}</span>` : "";
    return `<tr><td>${rp}</td><td>${gq}</td><td>${lq}</td>${b ? `<td>${bq}</td>` : ""}</tr>`;
  }).join("");
  const table = $("ffa-table");
  let caveat = "";
  if (b && typeof b.n_bootstrap === "number" && typeof b.n_bootstrap_discarded === "number" && b.n_bootstrap > 0) {
    const kept = b.n_bootstrap - b.n_bootstrap_discarded;
    const pct = (b.n_bootstrap_discarded / b.n_bootstrap) * 100;
    if (pct >= 5) {
      caveat = `<br><span class="muted">90 % CI from ${fmt(kept)} of ${fmt(b.n_bootstrap)} resamples; ${fmt(b.n_bootstrap_discarded)} fits fell outside the shape bounds (|c| ≤ 0.50).</span>`;
    }
  }
  table.innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>` +
    `<tfoot><tr><td colspan="${b ? 4 : 3}" class="muted">Return levels in ${escapeHtml(unit)}. T = return period.${caveat}</td></tr></tfoot>`;
  const r = state.selected;
  addTableDownload($("ffa-actions"), table, r ? `${r.source}-${r.station_id}-flood-frequency.csv` : "flood-frequency.csv");
}

function renderFfaPlot(ffa, rawUnit, color, r) {
  const unit = dUnit(rawUnit);
  const cv = (x) => dVal(x, rawUnit);
  const rps = ffa.return_periods;
  const traces = [];
  if (ffa.fits.gev_lmoments && ffa.fits.gev_lmoments.q) {
    traces.push({ x: rps, y: ffa.fits.gev_lmoments.q.map(cv), mode: "lines+markers", name: "GEV (L-moments)", line: { color } });
  }
  if (ffa.fits.lp3 && ffa.fits.lp3.q) {
    traces.push({ x: rps, y: ffa.fits.lp3.q.map(cv), mode: "lines+markers", name: "LP3", line: { color: "#8e24aa" } });
    if (ffa.fits.lp3.ci) {
      traces.push({
        x: [...rps, ...rps.slice().reverse()],
        y: [...ffa.fits.lp3.ci.map((c) => cv(c[1])), ...ffa.fits.lp3.ci.map((c) => cv(c[0])).reverse()],
        fill: "toself", fillcolor: "rgba(142,36,170,0.12)", line: { color: "transparent" }, name: "LP3 90 % CI", hoverinfo: "skip",
      });
    }
  }
  if (ffa.fits.gev_bootstrap) {
    const b = ffa.fits.gev_bootstrap;
    traces.push({ x: rps, y: b.q.map(cv), mode: "lines+markers", name: "GEV (MLE)", line: { color: "#f57c00", dash: "dot" } });
    traces.push({
      x: [...rps, ...rps.slice().reverse()],
      y: [...b.ci.map((c) => cv(c[1])), ...b.ci.map((c) => cv(c[0])).reverse()],
      fill: "toself", fillcolor: "rgba(245,124,0,0.12)", line: { color: "transparent" }, name: "GEV 90 % CI", hoverinfo: "skip",
    });
  }
  plot("plot-ffa", traces, {
    height: 260, xaxis: { title: { text: "return period (years)" }, type: "log" },
    yaxis: { title: { text: unit } }, legend: { orientation: "h", y: 1.15 },
  }, r ? `${r.source}-${r.station_id}-flood-frequency` : "flood-frequency");
}

// ── GR4J ────────────────────────────────────────────────────────────────────

function resetGr4j() {
  gr4jRun++;
  if (gr4jCancel) { gr4jCancel(); gr4jCancel = null; }
  const out = $("gr4j-out");
  if (out) out.hidden = true;
  setStatusEl($("gr4j-status"), "");
  const btn = $("btn-gr4j");
  if (btn) { btn.disabled = false; btn.textContent = "Calibrate GR4J"; }
  const stop = $("btn-gr4j-stop");
  if (stop) stop.hidden = true;
}

async function runGr4j() {
  const r = state.selected, res = state.result;
  if (!r || !res || !res.series) return;
  const key = stationKey(r);
  const my = ++gr4jRun;
  const btn = $("btn-gr4j"), stop = $("btn-gr4j-stop");
  btn.disabled = true;
  btn.textContent = "Calibrating…";
  stop.hidden = false;
  let cancelled = false;
  gr4jCancel = () => { cancelled = true; };
  const say = (t, kind = "info") => { if (my === gr4jRun) setStatusEl($("gr4j-status"), t, kind); };
  const done = (label = "Calibrate GR4J") => { btn.disabled = false; btn.textContent = label; stop.hidden = true; };
  try {
    say("Looking up the catchment area…");
    const area = await stationArea(key);
    if (my !== gr4jRun || cancelled) return;
    if (!area) {
      say("No catchment area for this gauge (it is not in the station catchments table), so flow cannot be expressed in mm/d.", "warn");
      done();
      return;
    }
    const t = res.series.t, v = res.series.v;
    let start = t[0];
    const end = t[t.length - 1];
    if (start < "1940-01-02") start = "1940-01-02";
    say("Fetching ERA5-Land/ERA5 rainfall and ET0 at the gauge (Open-Meteo)…");
    const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${r.lat}&longitude=${r.lon}&start_date=${start}&end_date=${end}` +
      `&daily=precipitation_sum,et0_fao_evapotranspiration&models=best_match&timezone=UTC`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Open-Meteo ${resp.status}${resp.status === 429 ? " (rate limit, try again in a minute)" : ""}`);
    const om = await resp.json();
    if (my !== gr4jRun || cancelled) return;
    const days = om.daily.time, P = om.daily.precipitation_sum, E = om.daily.et0_fao_evapotranspiration;
    const obsByDay = new Map();
    for (let i = 0; i < t.length; i++) obsByDay.set(t[i], v[i]);
    const n = days.length;
    const precip = new Float64Array(n), pet = new Float64Array(n), obs = new Array(n).fill(NaN);
    let lastE = 2.0;
    for (let i = 0; i < n; i++) {
      precip[i] = Number.isFinite(P[i]) && P[i] > 0 ? P[i] : 0;
      if (Number.isFinite(E[i])) lastE = Math.max(0, E[i]);
      pet[i] = lastE;
      const o = obsByDay.get(days[i]);
      if (o !== undefined && o !== null && Number.isFinite(o) && o >= 0) obs[i] = o * 86.4 / area.area;  // m3/s -> mm/d
    }
    const nObs = obs.filter(Number.isFinite).length;
    if (nObs < 365 * 3) {
      say(`Only ${nObs} days of overlap between the record and the forcing; three years are needed to calibrate.`, "warn");
      done();
      return;
    }
    const calEnd = Math.floor(n * 0.65);
    const t0 = performance.now();
    const fit = await GR4J.calibrate(precip, pet, obs, {
      objective: "kge", warmup: 365, calEnd, popsize: 20, generations: 40, seed: 1,
      onProgress: (g, best, G) => {
        if (cancelled) throw new Cancelled();
        say(`Differential evolution: generation ${g}/${G}, best KGE ${best.toFixed(3)}…`);
      },
    });
    if (my !== gr4jRun || cancelled) return;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    say("");
    const c = fit.calibration, val = fit.validation || {};
    const f3 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(3));
    const f1 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(1));
    $("gr4j-table").innerHTML =
      `<thead><tr><th>Parameter</th><th>Value</th><th>Metric</th>` +
      `<th>Calibration<br><span class="ci">${days[365]} to ${days[calEnd - 1]}</span></th>` +
      `<th>Validation<br><span class="ci">${days[calEnd]} to ${days[n - 1]}</span></th></tr></thead><tbody>` +
      `<tr><td>X1 production store</td><td>${fit.params.X1.toFixed(0)} mm</td><td>KGE</td><td>${f3(c.kge)}</td><td>${f3(val.kge)}</td></tr>` +
      `<tr><td>X2 exchange</td><td>${fit.params.X2.toFixed(2)} mm/d</td><td>NSE</td><td>${f3(c.nse)}</td><td>${f3(val.nse)}</td></tr>` +
      `<tr><td>X3 routing store</td><td>${fit.params.X3.toFixed(0)} mm</td><td>log-NSE</td><td>${f3(c.log_nse)}</td><td>${f3(val.log_nse)}</td></tr>` +
      `<tr><td>X4 UH time base</td><td>${fit.params.X4.toFixed(2)} d</td><td>PBIAS</td><td>${f1(c.pbias)} %</td><td>${f1(val.pbias)} %</td></tr></tbody>`;
    addTableDownload($("gr4j-actions"), $("gr4j-table"), `${r.source}-${r.station_id}-gr4j.csv`);
    // plot: last 6 years, observed vs simulated, in the station's unit
    const from = Math.max(0, n - 365 * 6);
    const toUnit = (mm) => dVal(mm * area.area / 86.4, res.unit);
    const gUnit = dUnit(res.unit);
    const st = sourceStyle(r.source);
    plot("plot-gr4j", [
      {
        x: days.slice(from), y: Array.from(obs.slice(from), (o) => (Number.isFinite(o) ? toUnit(o) : null)),
        mode: "lines", line: { width: 1, color: st.color }, name: "observed",
        hovertemplate: "%{x}<br>obs %{y:.3~f} " + gUnit + "<extra></extra>",
      },
      {
        x: days.slice(from), y: Array.from(fit.sim.slice(from), toUnit), mode: "lines",
        line: { width: 1.4, color: emphasisColor(), dash: "dot" }, name: "GR4J",
        hovertemplate: "%{x}<br>sim %{y:.3~f} " + gUnit + "<extra></extra>",
      },
    ], {
      yaxis: { title: { text: gUnit }, rangemode: "tozero" }, legend: { orientation: "h", y: 1.15 },
      shapes: calEnd > from ? [{ type: "line", x0: days[calEnd], x1: days[calEnd], y0: 0, y1: 1, yref: "paper", line: { color: emphasisColor(), dash: "dot", width: 1 } }] : [],
    }, `${r.source}-${r.station_id}-gr4j`);
    $("gr4j-foot").textContent =
      `Last six years shown (dotted line: start of the validation period). Area ${area.area.toLocaleString(undefined, { maximumFractionDigits: 0 })} km² ` +
      `(${area.source}); forcing at the gauge point, not catchment-averaged, so wet mountainous catchments will under-run. ` +
      `${fit.simulations} simulations in ${secs} s. Point values, not a forecast.`;
    $("gr4j-out").hidden = false;
    // The simulated series is worth keeping.
    $("btn-gr4j-csv").hidden = false;
    $("btn-gr4j-csv").onclick = () => {
      const lines = ["date,observed_mm_per_day,simulated_mm_per_day"];
      for (let i = 0; i < n; i++) lines.push(`${days[i]},${Number.isFinite(obs[i]) ? obs[i].toFixed(4) : ""},${fit.sim[i].toFixed(4)}`);
      downloadBlob(`${r.source}-${r.station_id}-gr4j-simulation.csv`, lines.join("\n") + "\n", "text/csv");
    };
    done("Recalibrate");
    for (const m of GR4J_METHODS) addMethodOnce("methods", m);
  } catch (err) {
    if (my !== gr4jRun) return;
    if (cancelled || err instanceof Cancelled) { say("Calibration stopped."); done(); return; }
    say(`GR4J failed: ${err.message}`, "error");
    done();
  } finally {
    if (my === gr4jRun) gr4jCancel = null;
  }
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initStationPanel() {
  const r = root();
  r.addEventListener("tabchange", (e) => {
    state.activeTab = e.detail.tab;
    writeUrl();
    // Plotly needs a nudge when a figure becomes visible for the first time.
    for (const id of ["plot-hydro", "plot-ffa", "plot-fdc", "plot-gr4j"]) {
      const el = $(id);
      if (el && el.offsetParent !== null && el.data) Plotly.Plots.resize(el);
    }
  });

  $("btn-unit").addEventListener("click", () => {
    unitPref = unitPref === "cfs" ? "m3s" : "cfs";
    try { localStorage.setItem(UNIT_PREF_KEY, unitPref); } catch { /* storage denied */ }
    // Re-render from the untouched result; the GR4J card redraws on recalibrate.
    if (state.result && state.selected) render(state.result, state.selected);
  });

  $("btn-share").addEventListener("click", (e) => copyText(canonicalUrl(), e.currentTarget, "Link copied"));
  $("btn-cite").addEventListener("click", () => openCite(methodsOnPage("methods")));
  $("btn-to-workbench").addEventListener("click", () => actions.openStationInWorkbench());

  $("btn-csv").addEventListener("click", async () => {
    if (!state.result || !state.selected) return;
    const btn = $("btn-csv");
    btn.disabled = true;
    try {
      const csv = await call("csv", {});
      downloadBlob(`${state.selected.source}_${state.selected.station_id}.csv`, csv, "text/csv");
    } catch (err) {
      setStatus(`Could not build the CSV: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
    }
  });

  $("btn-gr4j").addEventListener("click", () => runGr4j());
  $("btn-gr4j-stop").addEventListener("click", () => { if (gr4jCancel) gr4jCancel(); });

  $("btn-ci").addEventListener("click", async () => {
    if (!state.result || !state.result.ffa) return;
    const btn = $("btn-ci"), stop = $("btn-ci-stop");
    btn.disabled = true;
    btn.textContent = "Bootstrapping 1,000 GEV fits…";
    stop.hidden = false;
    const job = callCancelable("flood_ci", {});
    ciCall = job.cancel;
    try {
      const ci = await job.promise;
      state.result.ffa.fits.gev_bootstrap = ci;
      if (!state.result.methods.some((m) => m.name === ci.method.name)) state.result.methods.push(ci.method);
      renderFfaTable(state.result.ffa, state.result.unit);
      renderFfaPlot(state.result.ffa, state.result.unit, sourceStyle(state.selected.source).color, state.selected);
      renderMethods(state.result);
      btn.textContent = "Bootstrap CI added";
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Add bootstrap 90 % CI (GEV, slow)";
      if (!(err instanceof Cancelled)) setStatus(`Bootstrap failed: ${err.message}`, "error");
    } finally {
      stop.hidden = true;
      ciCall = null;
    }
  });
  $("btn-ci-stop").addEventListener("click", () => { if (ciCall) ciCall(); });

  actions.selectStation = selectStation;
}
