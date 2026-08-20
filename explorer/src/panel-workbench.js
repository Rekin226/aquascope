// The Workbench: your own data, analysed by the same code as everything else.
//
// Drop a CSV or an Excel export (aquascope.ingest works out the date and value
// columns, converts to SI and writes a QA report), or send the gauge you are
// looking at straight here. The analyses are aquascope.workbench, running in
// the same Pyodide worker as the rest of the page, so the numbers match the
// CLI and the MCP server exactly.

import { $, actions, downloadBlob, escapeHtml, fmt, state } from "./core.js?v=__BUILD__";
import { plot } from "./charts.js?v=__BUILD__";
import { addMethodOnce, renderMethodList } from "./methods.js?v=__BUILD__";
import { hideCard, selectTab, setCard, setTab, showSurface } from "./shell.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

const root = () => $("panel-workbench");

// Which analyses each tab offers, and the controls each one needs.
const ANALYSES = {
  quality: [
    { id: "eda", label: "Exploratory summary" },
    { id: "quality", label: "Data quality" },
    { id: "who_screen", label: "WHO drinking-water screen" },
  ],
  hydrology: [
    { id: "flow_duration", label: "Flow duration" },
    {
      id: "baseflow", label: "Baseflow separation", params: [
        { name: "method", type: "select", options: ["lyne_hollick", "eckhardt", "ukih"], value: "lyne_hollick" },
        { name: "alpha", type: "number", value: 0.925, min: 0.9, max: 0.99, step: 0.005 },
      ],
    },
    { id: "recession", label: "Recession", params: [{ name: "min_length", type: "number", value: 5, min: 3, max: 30, step: 1 }] },
    { id: "signatures", label: "Flow signatures" },
  ],
  extremes: [
    { id: "flood_frequency", label: "GEV flood frequency" },
    {
      id: "return_periods", label: "Return periods", params: [
        { name: "distribution", type: "select", options: ["gev", "lp3", "gumbel"], value: "gev" },
        { name: "n_bootstrap", type: "number", value: 200, min: 50, max: 1000, step: 50 },
      ],
    },
  ],
  groundwater: [
    { id: "sgi_drought", label: "Groundwater drought (SGI)", params: [{ name: "threshold", type: "number", value: -1, min: -3, max: 0, step: 0.1 }] },
    { id: "recharge", label: "Recharge (water-table fluctuation)", params: [{ name: "specific_yield", type: "number", value: 0.15, min: 0.01, max: 0.5, step: 0.01 }] },
    {
      id: "aquifer_drawdown", label: "Theis drawdown", noData: true, params: [
        { name: "transmissivity", type: "number", value: 500, min: 0.01, step: 10 },
        { name: "storativity", type: "number", value: 0.001, min: 0.000001, step: 0.0001 },
        { name: "pumping_rate", type: "number", value: 1000, min: 0.01, step: 50 },
        { name: "distance", type: "number", value: 100, min: 0.01, step: 10 },
        { name: "time_days", type: "number", value: 10, min: 0.01, step: 1 },
      ],
    },
  ],
};

let table = null;      // { n, columns, insights, label }
let runSeq = 0;

export function hasTable() { return Boolean(table); }

// ── loading data in ─────────────────────────────────────────────────────────

async function afterLoad(result, label) {
  table = { ...result, label };
  $("wb-label").textContent = label;
  $("wb-meta").textContent = `${result.n.toLocaleString()} rows · ${result.columns.length} columns: ${result.columns.slice(0, 8).join(", ")}`;
  $("wb-empty").hidden = true;
  $("wb-loaded").hidden = false;
  renderInsights(result.insights);
  for (const name of ["quality", "hydrology", "extremes", "groundwater", "methods"]) {
    setTab(root(), name, { enabled: true });
  }
  buildAnalyses();
  writeUrl();
}

async function loadFile(file) {
  setCard($("wb-load-card"), "loading", { message: `Reading ${file.name}…` });
  try {
    const text = await file.text();
    const res = await call("ingest", { text, filename: file.name });
    // The cleaned series is now the worker's table; ask it to describe itself.
    const loaded = await call("load_table", { csv: seriesToCsv(res), label: file.name });
    setCard($("wb-load-card"), "ready");
    renderQa(res);
    await afterLoad(loaded, file.name);
  } catch (err) {
    setCard($("wb-load-card"), "error", { message: `Could not read that file: ${err.message}` });
  }
}

// The ingest result carries the cleaned series; hand it back as CSV so the
// workbench sees a plain table with a date and a value column.
function seriesToCsv(res) {
  const a = res.analysis || {};
  const t = (a.series && a.series.t) || [];
  const v = (a.series && a.series.v) || [];
  const variable = a.variable || "value";
  const lines = [`date,${variable}`];
  for (let i = 0; i < t.length; i++) lines.push(`${t[i]},${v[i]}`);
  return lines.join("\n");
}

async function loadPasted(text) {
  setCard($("wb-load-card"), "loading", { message: "Reading what you pasted…" });
  try {
    const loaded = await call("load_table", { csv: text, label: "pasted table" });
    setCard($("wb-load-card"), "ready");
    hideCard($("wb-qa-card"));
    await afterLoad(loaded, "pasted table");
  } catch (err) {
    setCard($("wb-load-card"), "error", { message: `Could not read that: ${err.message}` });
  }
}

export async function openStationInWorkbench() {
  if (!state.selected || !state.result) return;
  openWorkbench();
  setCard($("wb-load-card"), "loading", { message: "Handing this record to the workbench…" });
  try {
    const loaded = await call("frame_from_station", {});
    setCard($("wb-load-card"), "ready");
    hideCard($("wb-qa-card"));
    await afterLoad(loaded, state.selected.name || state.selected.station_id);
  } catch (err) {
    setCard($("wb-load-card"), "error", { message: `Could not open this record: ${err.message}` });
  }
}

function renderQa(res) {
  const qa = res.qa || {};
  const m = res.mapping || {};
  const warnings = (qa.warnings || []).map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  $("wb-qa-card").querySelector(".card-body").innerHTML =
    `<div class="kpis">
      ${kpi("kept", `${(qa.n_values || 0).toLocaleString()} values`, `of ${(qa.n_rows_in || 0).toLocaleString()} rows`)}
      ${kpi("coverage", `${fmt(qa.coverage_pct, 1)} %`, `${qa.start || "?"} to ${qa.end || "?"}`)}
      ${kpi("dropped", `${(qa.n_sentinels_dropped || 0) + (qa.n_duplicates_dropped || 0)}`, "sentinels and duplicates")}
      ${kpi("flagged", `${qa.n_spikes_flagged || 0}`, "spikes")}
     </div>
     <p class="muted">Read as <strong>${escapeHtml(m.variable || "value")}</strong> in
       <strong>${escapeHtml(m.unit || "?")}</strong> from columns
       <code>${escapeHtml(m.datetime_column || "?")}</code> and <code>${escapeHtml(m.value_column || "?")}</code>.</p>
     ${warnings ? `<ul class="wb-warnings">${warnings}</ul>` : ""}`;
  setCard($("wb-qa-card"), "ready");
}

const kpi = (l, v, s) => `<div class="kpi"><div class="l">${escapeHtml(l)}</div><div class="v">${v}</div><div class="s">${escapeHtml(s)}</div></div>`;

function renderInsights(ins) {
  if (!ins) { hideCard($("wb-insights-card")); return; }
  const prof = ins.profile || {};
  const notes = (ins.quality_notes || []).map((n) => escapeHtml(n)).join("; ");
  const detected = [
    prof.datetime_col ? `time <code>${escapeHtml(prof.datetime_col)}</code>` : null,
    prof.discharge_col ? `discharge <code>${escapeHtml(prof.discharge_col)}</code>` : null,
    prof.param_col ? `parameter <code>${escapeHtml(prof.param_col)}</code>` : null,
    prof.value_col ? `value <code>${escapeHtml(prof.value_col)}</code>` : null,
  ].filter(Boolean).join(", ");
  $("wb-insights-card").querySelector(".card-body").innerHTML =
    `<div class="kpis">
      ${kpi("rows", (prof.n_records || 0).toLocaleString(), `${prof.n_stations || 0} station(s)`)}
      ${kpi("completeness", `${fmt(prof.completeness_pct, 1)} %`, notes || "no problems found")}
      ${kpi("quality", `${ins.quality_score}/100`, `${ins.n_duplicates} duplicate rows`)}
      ${kpi("WHO screen", `${ins.who_alerts}/${ins.who_checked}`, "parameters over the guideline")}
     </div>
     <p class="muted">Detected: ${detected || "no recognised columns"}.</p>
     <div class="wb-suggestions"></div>`;
  const box = $("wb-insights-card").querySelector(".wb-suggestions");
  for (const s of ins.suggestions || []) {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = s.label;
    b.title = s.reason;
    b.addEventListener("click", () => {
      const tab = { alerts: "quality", analysis: "quality", extremes: "extremes", hydrology: "hydrology", visualize: "quality" }[s.key] || "quality";
      selectTab(root(), tab, true);
    });
    box.appendChild(b);
  }
  setCard($("wb-insights-card"), "ready");
}

// ── running an analysis ─────────────────────────────────────────────────────

function controlHtml(a, p) {
  const id = `wb-${a.id}-${p.name}`;
  const label = p.name.replace(/_/g, " ");
  if (p.type === "select") {
    return `<label class="wb-field">${escapeHtml(label)}
      <select id="${id}">${p.options.map((o) => `<option value="${o}" ${o === p.value ? "selected" : ""}>${o.replace(/_/g, " ")}</option>`).join("")}</select></label>`;
  }
  return `<label class="wb-field">${escapeHtml(label)}
    <input id="${id}" type="number" value="${p.value}" ${p.min !== undefined ? `min="${p.min}"` : ""} ${p.max !== undefined ? `max="${p.max}"` : ""} step="${p.step || 1}"></label>`;
}

function readParams(a) {
  const out = {};
  for (const p of a.params || []) {
    const el = $(`wb-${a.id}-${p.name}`);
    if (!el) continue;
    out[p.name] = p.type === "select" ? el.value : Number(el.value);
  }
  return out;
}

function buildAnalyses() {
  for (const [tab, list] of Object.entries(ANALYSES)) {
    const box = $(`wb-${tab}-list`);
    if (!box || box.dataset.built === "1") continue;
    box.innerHTML = "";
    for (const a of list) {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML =
        `<div class="card-title"><h3>${escapeHtml(a.label)}</h3></div>` +
        `<div class="card-body">` +
        (a.params ? `<div class="wb-params">${a.params.map((p) => controlHtml(a, p)).join("")}</div>` : "") +
        `<div class="row-actions"><button class="btn small" data-run="${a.id}">Run</button></div>` +
        `<div class="wb-result" id="wb-out-${a.id}" hidden></div>` +
        `</div>`;
      card.querySelector("[data-run]").addEventListener("click", () => runAnalysis(a));
      box.appendChild(card);
    }
    box.dataset.built = "1";
  }
}

async function runAnalysis(a) {
  const out = $(`wb-out-${a.id}`);
  const my = ++runSeq;
  out.hidden = false;
  out.innerHTML = `<div class="card-note"><span class="spinner"></span><span>Running…</span></div>`;
  try {
    const res = await call("workbench", { analysis: a.id, params: readParams(a) });
    if (my !== runSeq && a.id !== "aquifer_drawdown") return;
    renderResult(a, res, out);
    for (const m of res.methods || []) addMethodOnce("wb-methods", m);
    setTab(root(), "methods", { enabled: true });
  } catch (err) {
    out.innerHTML = `<div class="card-note"><span class="note-icon error">!</span><span>${escapeHtml(err.message)}</span></div>`;
  }
}

function table2(rows) {
  return `<table class="ffa">${rows.map((r) => `<tr><td>${escapeHtml(r[0])}</td><td>${r[1]}</td></tr>`).join("")}</table>`;
}

function renderResult(a, res, out) {
  out.innerHTML = "";
  const plotId = `wb-plot-${a.id}`;
  const addPlot = () => {
    const d = document.createElement("div");
    d.className = "plot";
    d.id = plotId;
    out.appendChild(d);
  };
  const say = (html) => { const d = document.createElement("div"); d.innerHTML = html; out.appendChild(d); };

  if (a.id === "flow_duration") {
    const p = res.percentiles || {};
    say(table2([["Q5", fmt(p["5"])], ["Q50 (median)", fmt(p["50"])], ["Q95", fmt(p["95"])], ["values", res.n.toLocaleString()]]));
    addPlot();
    plot(plotId, [{ x: res.exceedance, y: res.discharge, mode: "lines", line: { color: "#1565c0" } }],
      { xaxis: { title: { text: "% of time exceeded" } }, yaxis: { title: { text: "flow" }, type: "log" }, showlegend: false },
      "flow-duration");
  } else if (a.id === "baseflow") {
    say(table2([["Baseflow index", fmt(res.bfi, 3)], ["Method", res.method]]));
    addPlot();
    plot(plotId, [
      { x: res.series.index, y: res.series.total, mode: "lines", name: "total", line: { width: 1, color: "#1565c0" } },
      { x: res.series.index, y: res.series.baseflow, mode: "lines", name: "baseflow", line: { width: 1.4, color: "#2e7d32" } },
    ], { legend: { orientation: "h", y: 1.15 }, yaxis: { title: { text: "flow" } } }, "baseflow");
  } else if (a.id === "recession") {
    say(table2([["Recession constant", fmt(res.recession_constant, 4)], ["R²", fmt(res.r_squared, 3)],
      ["Half-life (days)", fmt(res.half_life_days, 1)], ["Segments", res.n_segments]]));
  } else if (a.id === "signatures") {
    const s = res.signatures || {};
    say(table2(Object.entries(s).filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => [k.replace(/_/g, " "), typeof v === "number" ? fmt(v, 3) : String(v)])));
  } else if (a.id === "flood_frequency" || a.id === "return_periods") {
    const rp = a.id === "flood_frequency"
      ? Object.entries(res.return_periods).map(([t, q]) => [t, q, null, null])
      : res.return_periods.map((t, i) => [t, res.return_levels[i], res.lower_bound[i], res.upper_bound[i]]);
    say(`<table class="ffa"><thead><tr><th>T (yr)</th><th>Level</th>${rp[0] && rp[0][2] !== null ? "<th>CI</th>" : ""}</tr></thead><tbody>` +
      rp.map(([t, q, lo, hi]) => `<tr><td>${t}</td><td>${fmt(q)}</td>${lo !== null && lo !== undefined ? `<td class="ci">${fmt(lo)} to ${fmt(hi)}</td>` : ""}</tr>`).join("") +
      `</tbody></table>`);
    addPlot();
    const traces = [{ x: rp.map((r) => Number(r[0])), y: rp.map((r) => r[1]), mode: "lines+markers", name: "fitted", line: { color: "#1565c0" } }];
    if (res.empirical) {
      traces.push({ x: res.empirical.return_period, y: res.empirical.value, mode: "markers", name: "observed", marker: { color: "#e53935", size: 6 } });
    }
    plot(plotId, traces, { xaxis: { title: { text: "return period (years)" }, type: "log" }, legend: { orientation: "h", y: 1.15 } }, "return-periods");
  } else if (a.id === "sgi_drought") {
    say(table2([["Current SGI", fmt(res.current, 2)], ["Worst", fmt(res.worst, 2)], ["Droughts", res.events.length]]));
    addPlot();
    plot(plotId, [{ x: res.sgi.index, y: res.sgi.values, type: "bar", marker: { color: "#1565c0" } }],
      { yaxis: { title: { text: "SGI" } }, showlegend: false }, "sgi");
  } else if (a.id === "recharge") {
    say(table2([["Recharge", `${fmt(res.value_mm_per_year, 1)} mm/yr`],
      ["Total rise", `${fmt((res.metadata || {}).total_rise_m, 2)} m`],
      ["Period", `${fmt((res.metadata || {}).period_years, 1)} yr`]]));
  } else if (a.id === "aquifer_drawdown") {
    say(table2([["Drawdown", `${fmt(res.drawdown_m, 4)} m`]]));
  } else if (a.id === "eda") {
    say(table2([["Records", res.n_records.toLocaleString()], ["Stations", res.n_stations],
      ["Parameters", res.n_parameters], ["Completeness", `${fmt(res.completeness_pct, 1)} %`]]));
    if ((res.parameters || []).length) {
      say(`<table class="ffa"><thead><tr><th>Parameter</th><th>n</th><th>mean</th><th>min</th><th>max</th><th>outliers</th></tr></thead><tbody>` +
        res.parameters.map((p) => `<tr><td>${escapeHtml(p.name)}</td><td>${p.count}</td><td>${fmt(p.mean, 3)}</td><td>${fmt(p.min, 3)}</td><td>${fmt(p.max, 3)}</td><td>${p.outlier_count}</td></tr>`).join("") +
        `</tbody></table>`);
    }
  } else if (a.id === "quality") {
    say(table2([["Completeness", `${fmt(res.completeness_pct, 1)} %`], ["Duplicates", res.n_duplicates],
      ["Gaps", (res.temporal_gaps || []).length]]));
    if ((res.recommended_steps || []).length) say(`<p class="muted">Suggested: ${res.recommended_steps.join(", ")}</p>`);
  } else if (a.id === "who_screen") {
    if (!res.rows.length) say(`<p class="muted">${escapeHtml(res.note || "Nothing to screen in this table.")}</p>`);
    else {
      say(`<table class="ffa"><thead><tr><th>Parameter</th><th>Guideline</th><th>n</th><th>over</th><th>%</th><th>status</th></tr></thead><tbody>` +
        res.rows.map((r) => `<tr><td>${escapeHtml(r.parameter)}</td><td>${escapeHtml(r.rule)}</td><td>${r.n}</td><td>${r.n_exceed}</td><td>${fmt(r.pct, 1)}</td><td>${r.status}</td></tr>`).join("") +
        `</tbody></table>`);
    }
  } else {
    say(`<pre class="wb-json">${escapeHtml(JSON.stringify(res, null, 1).slice(0, 4000))}</pre>`);
  }

  const dl = document.createElement("button");
  dl.className = "btn tiny";
  dl.textContent = "JSON";
  dl.title = "Download this result";
  dl.addEventListener("click", () => downloadBlob(`aquascope-${a.id}.json`, JSON.stringify(res, null, 2), "application/json"));
  out.appendChild(dl);
}

// ── the surface ─────────────────────────────────────────────────────────────

export function openWorkbench() {
  showSurface("panel-workbench");
  state.mode = "workbench";
  state.selected = null;
  state.point = null;
  document.body.classList.add("workbench-mode");
  writeUrl({ push: true });
}

export function closeWorkbench() {
  document.body.classList.remove("workbench-mode");
  state.mode = "map";
}

export function initWorkbench() {
  const r = root();
  r.addEventListener("tabchange", () => {
    for (const el of r.querySelectorAll(".plot")) {
      if (el.offsetParent !== null && el.data) Plotly.Plots.resize(el);
    }
  });
  $("wb-file").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) loadFile(file);
  });
  const drop = $("wb-drop");
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("over"); });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  }
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFile(file);
  });
  $("wb-paste-run").addEventListener("click", () => {
    const text = $("wb-paste").value.trim();
    if (text) loadPasted(text);
  });
  $("btn-workbench").addEventListener("click", () => openWorkbench());
  $("wb-back").addEventListener("click", () => {
    closeWorkbench();
    showSurface("panel-empty");
    writeUrl({ push: true });
  });
  renderMethodList("wb-methods", []);
  actions.openWorkbench = openWorkbench;
  actions.openStationInWorkbench = openStationInWorkbench;
}
