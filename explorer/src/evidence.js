// The Evidence tab and the map's "Best model skill" colouring (#518).
//
// The tab puts the gauge's record and the global models on one plot and says
// how each one scores: aquascope.evidence.model_skill in the worker computes
// GEOGLOWS and GloFAS live for this gauge; NWM v3 and Google GRRR are scored by
// the monthly CI run and read here from skill/model_skill.parquet with
// DuckDB-WASM, then handed to the same Python function. Before that table is
// published the tab still works (two models) and the map colouring says so.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, escapeHtml, sourceStyle, state, stationKey, trace } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { addTableDownload, plot } from "./charts.js?v=__BUILD__";
import { hideCard, setCard, setStatusEl } from "./shell.js?v=__BUILD__";
import { Cancelled, callCancelable } from "./worker-client.js?v=__BUILD__";
import { stationArea } from "./basins.js?v=__BUILD__";
import {
  MODEL_COLORS, badge, bestByGauge, bestSql, gaugeSql, gradeColor, plainRow, skillLegend, skillTableRows, skillUrl,
} from "./evidence-core.js?v=__BUILD__";

const URL_ = CONFIG.skillParquet || skillUrl(CONFIG.stationsParquet);

let grades = null;            // Map key -> {grade, model, kge} once loaded
let gradesStatus = "idle";    // idle | loading | ready | missing
let gradesPromise = null;
let run = 0;
let cancel = null;
let doneKey = null;

// ── the published table ─────────────────────────────────────────────────────

/** Load the best grade per gauge once; resolves to the Map, or null when the table is not published. */
export function loadSkillGrades() {
  if (gradesPromise) return gradesPromise;
  gradesStatus = "loading";
  gradesPromise = (async () => {
    try {
      const { conn } = await duck();
      const t = await conn.query(bestSql(URL_));
      grades = bestByGauge(t.toArray().map((r) => plainRow(r.toJSON())));
      state.skillGrades = grades;
      gradesStatus = "ready";
      trace(`skill table: ${grades.size} graded gauges`);
    } catch (err) {
      grades = new Map();
      gradesStatus = "missing";
      trace(`skill table not available: ${err && err.message}`);
    }
    return gradesStatus === "ready" ? grades : null;
  })();
  return gradesPromise;
}

export const skillGradesStatus = () => gradesStatus;
export const skillGradeFor = (key) => (grades && grades.get(key)) || null;

/** The legend under the gauge-style menu while "Best model skill" is on. */
export function skillLegendHtml() {
  return skillLegend(gradesStatus).map((b) =>
    `<span class="sw"><i style="background:${b.color}"></i>${escapeHtml(b.label)}</span>`).join("");
}

async function publishedRows(r) {
  if (await loadSkillGrades() === null) return [];   // not published yet: nothing to ask for
  try {
    const { conn } = await duck();
    const t = await conn.query(gaugeSql(URL_, r.source, r.station_id));
    return t.toArray().map((row) => plainRow(row.toJSON()));
  } catch (err) {
    trace(`published skill rows unavailable: ${err && err.message}`);
    return [];
  }
}

// ── the header badge ────────────────────────────────────────────────────────

function setBadge(best) {
  const el = $("st-grade");
  if (!el) return;
  const b = badge(best);
  el.hidden = !b;
  if (!b) return;
  el.textContent = b.text;
  el.style.background = b.color;
  el.title = b.title;
}

/** The published best grade for a gauge, as soon as the table has loaded. */
export async function showSkillBadge(r) {
  setBadge(null);
  const key = stationKey(r);
  await loadSkillGrades();
  if (state.selected && stationKey(state.selected) === key) setBadge(skillGradeFor(key));
}

// ── the tab ─────────────────────────────────────────────────────────────────

const card = () => $("st-evidence");
const say = (text, kind = "info") => setStatusEl($("ev-status"), text, kind);

export function resetEvidence() {
  run++;
  if (cancel) { cancel(); cancel = null; }
  doneKey = null;
  hideCard(card());
  say("");
}

/** Score the models for the selected gauge (once per selection; the tab calls it when opened). */
export async function startEvidence(r) {
  if (!r) return;
  const key = stationKey(r);
  if (doneKey === key) return;
  doneKey = key;
  const my = ++run;
  setCard(card(), "loading", { message: "Reading GEOGLOWS and GloFAS for this gauge (about half a minute)…" });
  try {
    const [area, published] = await Promise.all([stationArea(key), publishedRows(r)]);
    if (my !== run) return;
    const job = callCancelable("evidence", {
      args: {
        source: r.source, station_id: r.station_id, lat: r.lat, lon: r.lon,
        area_km2: area ? area.area : null, published,
      },
    });
    cancel = job.cancel;
    const res = await job.promise;
    if (my !== run) return;
    cancel = null;
    render(res, r);
  } catch (err) {
    if (my !== run || err instanceof Cancelled) return;
    doneKey = null;
    setCard(card(), "error", { message: `Could not score the models: ${err && err.message}`, retry: () => startEvidence(r) });
  }
}

function render(res, r) {
  if (res.error) {
    setCard(card(), "empty", { message: res.error });
    return;
  }
  setCard(card(), "ready");
  setBadge({ grade: res.best_grade });
  $("ev-sentence").textContent = res.sentence || "";
  const st = sourceStyle(r.source);
  const series = res.series || {};
  const traces = [];
  if (series.observed) {
    traces.push({ x: series.observed.t, y: series.observed.v, mode: "lines", name: "gauge", line: { width: 2.2, color: st.color },
      hovertemplate: "%{x}<br>gauge %{y:.3~f} m³/s<extra></extra>" });
  }
  for (const m of ["geoglows", "glofas"]) {
    if (!series[m] || !series[m].t.length) continue;
    const label = (res.models.find((x) => x.model === m) || {}).label || m;
    traces.push({ x: series[m].t, y: series[m].v, mode: "lines", name: label, line: { width: 1, color: MODEL_COLORS[m] },
      hovertemplate: `%{x}<br>${escapeHtml(label)} %{y:.3~f} m³/s<extra></extra>` });
  }
  plot("plot-evidence", traces, { yaxis: { title: { text: "m³/s" }, rangemode: "tozero" }, legend: { orientation: "h", y: 1.15 } },
    `${r.source}-${r.station_id}-models`);

  const rows = skillTableRows(res.models);
  const head = "<tr><th>Model</th><th>Grade</th><th>KGE</th><th class='sym'>r</th><th class='sym'>α</th><th class='sym'>β</th><th>NSE</th><th>Bias</th><th>Q2</th><th>Q10</th><th>Q100</th></tr>";
  const body = rows.map((x) => (x.scored
    ? `<tr><td>${escapeHtml(x.label)}${x.published ? ' <span class="muted">(monthly)</span>' : ""}</td>` +
      `<td><span class="grade-chip" style="background:${gradeColor(x.grade)}" title="${escapeHtml(x.why)}">${escapeHtml(x.grade || "–")}</span></td>` +
      `<td>${x.kge}</td><td>${x.r}</td><td>${x.alpha}</td><td>${x.beta}</td><td>${x.nse}</td><td>${x.pbias}</td>` +
      `<td>${x.q2}</td><td>${x.q10}</td><td>${x.q100}</td></tr>`
    : `<tr><td>${escapeHtml(x.label)}</td><td colspan="10" class="muted">${escapeHtml(x.why || "not scored")}</td></tr>`)).join("");
  const table = $("ev-table");
  table.innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>` +
    `<tfoot><tr><td colspan="11" class="muted">Q2, Q10, Q100: the model's 2-, 10- and 100-year flow against the gauge's, each from its own fit. ` +
    `Grade A: KGE 0.75 and up, B: 0.5, C: above −0.41, D: no better than the mean flow; one letter lower when the 100-year flow is off by more than half.</td></tr></tfoot>`;
  addTableDownload($("ev-actions"), table, `${r.source}-${r.station_id}-model-skill.csv`);
  const notes = [...(res.notes || [])];
  notes.push(`Gauge record ${res.obs_start} to ${res.obs_end}; each model is scored over the days it shares with it. ` +
    "Models are modelled, not measured.");
  notes.push(`Data: ${res.attribution}.`);
  $("ev-foot").innerHTML = notes.map((n) => `<p>${escapeHtml(n)}</p>`).join("");
  say("");
}
