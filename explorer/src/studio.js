// Study: a complete study at a place, done by a crew of roles in the Pyodide
// worker (aquascope.studio, the same Coordinator the CLI and the MCP tools
// run). The page is a conversation and a board. The conversation is the
// workspace's messages: what you said, what the Consultant asked, what the
// crew wrote. The board above the input shows one thing at a time, by status:
// where and what to bring (intake), the plan to approve (review), the timeline
// and the figures as they land (running), the answer and the bundle (done), or
// why the crew declined. Keyless by default: the key Ask holds is offered on
// one line, and an on-device model, when one is already there, joins the crew:
// it reads the first sentence into a brief, writes the plan (the engine's
// validator checks it, the playbook tree stands when it fails) and, after the
// run, the prose (the Critic's checks drop what the results do not carry).
// The card says who wrote what. Nothing is downloaded for that; nothing is
// uploaded, ever. A study is saved in this browser after every reply, so a
// reload offers it back; Stop terminates the worker and keeps the plan. The
// recorded studies (explorer/showcase/studies, made once with a model) are
// chips on the intake board: one opens as its finished board, as recorded,
// and Re-run live runs its plan again here, keyless.

import { $, actions, escapeHtml, fmt, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { shapeSvg } from "./shapes.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { catchmentAreaAt, catchmentForWorker, donorPoolSize, donorTablesForWorker, stationArea } from "./basins.js?v=__BUILD__";
import { askModelConfig, mdToHtml } from "./ask.js?v=__BUILD__";
import { closeDrawer, drawerMode, drawerOpen, openDrawer, setStatusEl } from "./shell.js?v=__BUILD__";
import {
  Cancelled, call, callCancelable, ensureCatalogInWorker, onStudioArtifact, onStudioProgress, restartWorker,
} from "./worker-client.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { briefPrompt, briefSchema, parseBriefReply } from "./intake.js?v=__BUILD__";
import { hasTable, tableLabel } from "./panel-workbench.js?v=__BUILD__";
import { narrateOnDevice, planLine, planOnDevice, proseLine } from "./studio-device.js?v=__BUILD__";
import { agoWords, latestStudy, loadStudy, saveStudy } from "./study-store.js?v=__BUILD__";
import { loadRecorded, recordedIndex, replayBrief, replayPlan } from "./studio-showcase.js?v=__BUILD__";
import {
  recordedChipsHtml, recordedFigures, recordedFilesHtml, recordedNoteHtml, recordedPlanLine,
} from "./studio-recorded.js?v=__BUILD__";
import { canonicalUrl, writeUrl } from "./url.js?v=__BUILD__";
import { fetchStudyYaml, linkUrl, sharedBoardHtml, sharedPlanLine } from "./study-link.js?v=__BUILD__";
// study map: the crew's places drawn on the map (study-map.js)
import { clearStudyMap, focusStudyStep, showStudyMapFor, studyMapArtifact } from "./study-map.js?v=__BUILD__";
import { drawnOnMap, stepsOnMapHtml } from "./study-map-data.js?v=__BUILD__";
import { initStudyControls, rememberPlain, steerHtml, stepChecksHtml } from "./study-controls.js?v=__BUILD__";

const RECORDED_BASE = "./showcase/studies/";

const DONOR_K = 10;
const BRIEF_TIMEOUT_MS = 25000;
const TIMELINE_MAX = 80;

// Tool names in words for the plan and the timeline; anything else is its id
// with the underscores spaced out.
const TOOL_LABEL = {
  describe_catchment: "Describe the catchment",
  analyze_station: "Analyse the gauge record",
  flood_frequency: "Fit the flood frequency",
  return_periods: "Return periods",
  get_timeseries: "Fetch the series",
  similar_basins: "Find donor gauges",
  regionalize_signatures: "Transfer flow signatures from donors",
  anywhere: "ERA5 climate and GloFAS for this cell",
  find_stations: "Find gauges nearby",
  assess_site: "Reconnaissance",
  drought_indices: "Drought indices (SPI, SPEI)",
  drought_propagation: "Drought propagation",
  sgi_drought: "Standardised Groundwater Index",
  recharge: "Water-table-fluctuation recharge",
  low_flow_context: "Low-flow context",
  supply_reliability: "Supply reliability",
  crop_water_demand: "Crop water demand",
  irrigation: "Irrigation schedule",
  reference_et: "Reference evapotranspiration",
  water_quality_samples: "Fetch water-quality samples",
  who_screen: "WHO drinking-water screen",
  wqi: "Water-quality index",
  iwqi: "Irrigation water-quality index",
  load_table: "Read your table",
  flow_duration: "Flow duration",
  baseflow: "Baseflow separation",
  recession: "Recession",
  signatures: "Flow signatures",
  eda: "Exploratory summary",
  quality: "Data quality",
  preprocess: "Preprocess",
  insights: "Insights",
  spei: "SPEI",
  aquifer_drawdown: "Theis drawdown",
};
const toolLabel = (t) => TOOL_LABEL[t] || String(t || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const ARG_LABEL = {
  return_period: "return period", station_id: "station", n_bootstrap: "resamples", area_km2: "area",
  timescales: "timescales", demand_m3s: "demand", value_column: "value column", datetime_column: "date column",
};
const argWord = (k) => ARG_LABEL[k] || String(k).replace(/_/g, " ");

const ROLE_NAME = {
  user: "you", consultant: "consultant", methodologist: "methodologist", author: "author",
  coordinator: "coordinator", scout: "scout", analyst: "analysts", critic: "critic",
};

const DOCS = [["report-docx", "Word"], ["workbook", "Excel"], ["report-md", "Markdown"], ["notebook", "notebook"],
  ["study", "study.yaml"]];

const S = {
  ws: null,             // the workspace dict, without the artifact bytes
  site: null,           // where the study was started: { key, lat, lon, text, html }
  catchment: null, area: null, donors: null,
  siteReady: false,     // the catchment row, the area and the donor pool are loaded for S.site
  busy: false,          // a call is in flight: the board shows the timeline
  writing: false,       // the device model is writing the prose: the input waits
  declined: null,       // the reader's own decline of the plan (the crew's is ws.status)
  editing: false,
  useKey: true,         // the key Ask holds, when there is one
  files: [],            // [{ id, csv }] attached before the study starts
  useMyData: false,
  figures: new Map(),   // artifact id -> { id, src, caption, step, job }
  events: [],
  jobId: null, cancel: null, run: 0,
  proposal: null,       // the device model's plan shown on the card: { plan, steps, model }
  planLine: null,       // who planned, in words, for the card and the foot
  proseLine: null,      // who wrote the prose, in words
  prompts: undefined,   // the crew's exported prompts and schemas: undefined (not asked), null (none), or the dict
  resume: null,         // a saved study offered on the intake board: { id, at, site }
  index: undefined,     // the recorded studies' index rows: undefined (not asked), else the list
  moreRecorded: false,  // every recorded chip shown, not just the first six
  recorded: null,       // the recording on the board: { id, meta, workspace, report, figures, files }
  planSource: null,     // "recorded" while a re-run's plan is with the engine, for the words on the foot
  planModel: null,      // the model that wrote the recorded plan being re-run
  shared: null,         // a shared study on the board: the worker's open_link reply, or { loading: true }
};

const note = (text, kind = "info") => setStatusEl($("study-status"), text, kind);
const board = () => $("study-board");

// ── where, and with what ────────────────────────────────────────────────────

function where() {
  if (state.selected) {
    const r = state.selected;
    const st = sourceStyle(r.source);
    return {
      key: stationKey(r), lat: r.lat, lon: r.lon, text: r.name || r.station_id,
      html: `${shapeSvg(st.shape, st.color)}${escapeHtml(r.name || r.station_id)} <span class="muted">${escapeHtml(st.label)}</span>`,
    };
  }
  if (state.point) {
    const { lat, lon } = state.point;
    const text = `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
    return { key: `p/${lat},${lon}`, lat, lon, text, html: `${escapeHtml(text)} <span class="muted">a point, no gauge</span>` };
  }
  return null;
}

// A site from a saved record or a workspace: the page's own when it is the
// same place, else the bare coordinates.
function siteFrom(rec) {
  const w = where();
  if (w && rec && Number(rec.lat) === w.lat && Number(rec.lon) === w.lon) return w;
  const lat = Number(rec.lat), lon = Number(rec.lon);
  const text = rec.text || `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
  return { key: rec.key || `p/${lat},${lon}`, lat, lon, text, html: `${escapeHtml(text)} <span class="muted">saved study</span>` };
}

function modelForRun() {
  const cfg = S.useKey ? askModelConfig() : null;
  return cfg ? { provider: cfg.provider, model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url } : {};
}

// What only the page can read for a site: the sub-basin row for
// describe_catchment, the catchment area and the donor pool for the
// reconnaissance. Loaded once per site, before the first call that needs it.
async function loadSiteInfo(w, my) {
  if (S.siteReady) return;
  const [catchment, area, pool] = await Promise.all([
    catchmentForWorker(w.lat, w.lon).catch(() => null),
    (state.selected && w.key === stationKey(state.selected) ? stationArea(w.key).then((a) => (a ? a.area : null))
      : catchmentAreaAt(w.lat, w.lon)).catch(() => null),
    donorPoolSize().catch(() => 0),
  ]);
  if (my !== S.run) return;
  S.catchment = catchment;
  S.area = area;
  S.donors = area ? Math.min(DONOR_K, pool) : null;
  S.siteReady = true;
}

// ── the board: one thing, by status ─────────────────────────────────────────

function boardStatus() {
  if (S.busy) return "running";
  if (S.declined) return "declined";
  if (S.shared && !S.ws) return "shared";
  if (!S.ws) return "intake";
  const st = S.ws.status;
  if (st === "declined" || st === "review" || st === "done" || st === "waiting") return st;
  if (["running", "critique", "authoring", "scouting", "planning"].includes(st)) return "running";
  return "intake";
}

// The file drop area once a study exists (any status but a running one, where it is not offered): the
// tables already in the workspace (not S.files, which only stages tables before a study starts) and, at
// waiting, review or done, a line saying what dropping one now does (#419).
const DROP_NOTE = {
  waiting: "A table dropped now is inventoried and the plan is written again.",
  review: "A table dropped now is inventoried and the plan is written again.",
  done: "A table dropped now runs as a follow-up.",
};

function fileDropHtml(status) {
  const attached = Object.keys((S.ws && S.ws.tables) || {});
  const note = DROP_NOTE[status];
  return `<div class="study-data">` +
    `<label class="link" for="study-file">Add a CSV or XLSX</label>` +
    `<input type="file" id="study-file" accept=".csv,.txt,.xlsx,.xls,.json" multiple hidden>` +
    (attached.length ? `<p class="study-line muted">with ${attached.map(escapeHtml).join(", ")}</p>` : "") +
    (note ? `<p class="study-line muted">${escapeHtml(note)}</p>` : "") +
    `</div>`;
}

function intakeHtml() {
  const w = S.ws ? S.site : where();
  const cfg = askModelConfig();
  const started = Boolean(S.ws);
  let model;
  if (!cfg) model = `<p class="study-line muted">No key: the playbook tree plans, templates write.</p>`;
  else if (started) model = `<p class="study-line muted">${S.useKey ? escapeHtml(cfg.label) : "no model"}</p>`;
  else model = `<label class="study-line ask-context-toggle"><input type="checkbox" data-opt="key" ${S.useKey ? "checked" : ""}> use ${escapeHtml(cfg.label)} for the prose</label>`;
  const data = started ? fileDropHtml("intake") : `<div class="study-data">` +
      `<label class="link" for="study-file">Add a CSV or XLSX</label>` +
      `<input type="file" id="study-file" accept=".csv,.txt,.xlsx,.xls,.json" multiple hidden>` +
      (hasTable()
        ? ` <button type="button" class="btn tiny" data-act="my-data" aria-pressed="${S.useMyData}">${S.useMyData ? "Using" : "Use"} the table in My data</button>`
        : "") +
      (S.files.length
        ? `<ul class="study-files">${S.files.map((f, i) => `<li>${escapeHtml(f.id)} <button type="button" class="study-x" data-remove="${i}" aria-label="Remove ${escapeHtml(f.id)}">×</button></li>`).join("")}</ul>`
        : "") +
      `</div>`;
  // A study saved in this browser at this place (or, with nothing picked, the last one anywhere).
  const resume = !started && S.resume
    ? `<p class="study-resume"><button type="button" class="chip" data-act="resume" title="${escapeHtml(`${S.resume.site.text}, ${agoWords(S.resume.at)}`)}">Resume the last study${w ? "" : ` at ${escapeHtml(S.resume.site.text)}`}</button></p>`
    : "";
  const recorded = !started ? recordedChipsHtml(S.index, { showAll: S.moreRecorded }) : "";
  return `<p class="study-where">${w ? w.html : `<span class="muted">Pick a gauge or a spot on the map first.</span>`}</p>${model}${data}${resume}${recorded}`;
}

const fmtArg = (v) => (typeof v === "string" ? v : JSON.stringify(v));

// "return period 100 · source uk_ea · station 3400TH": a step's arguments in words.
function argValue(v) {
  if (typeof v === "string") {
    const m = v.match(/\{\{\s*result\.(\w+)/);
    if (m) return `from ${m[1]}`;
    return v.length > 28 ? `${v.slice(0, 14)}…` : v;
  }
  if (Array.isArray(v)) return v.map(argValue).join(", ");
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}
const argsWords = (args) => Object.entries(args || {}).map(([k, v]) => `${argWord(k)} ${argValue(v)}`).join(" · ");

function gateChip(g) {
  const label = `${g.check}${g.value !== undefined && g.value !== null ? ` ${fmtArg(g.value)}` : ""}`;
  const on = g.path || (g.paths || []).join(", ") || "";
  return `<span class="gate" title="${escapeHtml(on)}">${escapeHtml(label)}</span>`;
}

function stepHtml(s) {
  const args = s.arguments || {};
  const gates = (s.expects || []).map(gateChip).join("");
  const inputs = S.editing
    ? Object.entries(args).map(([k, v]) =>
      `<label class="study-arg">${escapeHtml(argWord(k))}<input data-step="${escapeHtml(s.id || "")}" data-arg="${escapeHtml(k)}" value="${escapeHtml(fmtArg(v))}"></label>`).join("")
    : "";
  return `<li>` +
    `<div class="step-main"><span class="step-tool">${escapeHtml(toolLabel(s.tool))}</span>` +
    (S.editing ? "" : ` <span class="step-args">${escapeHtml(argsWords(args))}</span>`) + `</div>` +
    (inputs ? `<div class="study-args">${inputs}</div>` : "") +
    stepChecksHtml(s, gates) + // steering: the gates in plain words, the raw chips behind "details"
    (s.rationale ? `<details class="step-why"><summary>why</summary>${escapeHtml(s.rationale)}</details>` : "") +
    `</li>`;
}

// The steps on the card: the device model's proposal when there is one, else the study's.
function shownSteps() {
  if (S.proposal) return S.proposal.steps;
  return ((S.ws && S.ws.study) || {}).steps || [];
}

// A compound brief: the companion playbook's steps, appended to the primary plan (#419).
function companionsHtml(plan) {
  const list = plan.companions;
  if (!plan.compound || !Array.isArray(list) || !list.length) return "";
  const bits = list.map((c) => {
    const n = (c.steps || []).length;
    return `${String(c.playbook || "").replace(/_/g, " ")} adds ${n} step${n === 1 ? "" : "s"}`;
  });
  return `<p class="study-line muted">This brief spans two playbooks: ${escapeHtml(bits.join("; "))}</p>`;
}

function planHtml() {
  const study = S.ws.study || {};
  const plan = S.proposal ? { ...study.plan, ...S.proposal.plan } : (study.plan || {});
  const steps = shownSteps();
  const notes = [...(plan.assumptions || []), ...(plan.caveats || [])];
  return `<article class="study-plan" tabindex="-1" aria-label="The plan">` +
    (plan.objective ? `<p class="study-objective">${escapeHtml(plan.objective)}</p>` : "") +
    companionsHtml(plan) +
    `<ol class="study-steps">${steps.map(stepHtml).join("")}</ol>` +
    (notes.length
      ? `<details class="study-notes"><summary>${notes.length} note${notes.length === 1 ? "" : "s"}</summary><ul>${notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul></details>`
      : "") +
    (S.planLine ? `<p class="study-by muted">${escapeHtml(S.planLine)}</p>` : "") +
    fileDropHtml("review") +
    `<div class="row-actions">` +
      `<button type="button" class="btn primary" data-act="approve">Approve</button>` +
      `<button type="button" class="btn" data-act="edit" aria-pressed="${S.editing}">${S.editing ? "Cancel edits" : "Edit"}</button>` +
      `<button type="button" class="btn" data-act="decline">Decline</button>` +
    `</div></article>`;
}

function statusWord(s) {
  return { scouting: "the Scout takes the inventory", planning: "the Methodologist plans", review: "the plan is ready",
    running: "the Analysts run the plan", critique: "the Critic reads the draft", authoring: "the Author writes",
    done: "done", declined: "declined" }[s] || s;
}

function eventHtml(e) {
  const detail = String(e.detail || "");
  let cls = "";
  if (e.event === "gate") cls = /fail/i.test(detail) ? "fail" : "ok";
  if (["error", "stop", "declined", "skipped", "model_error", "invalid", "no_fallback", "replan_declined"].includes(e.event)) cls = "fail";
  if (["done", "reused", "figures", "artifact", "deliverables"].includes(e.event)) cls = "ok";
  let text = detail;
  if (e.event === "start" || e.event === "fallback") {
    const m = detail.match(/^(\w+)\(/);
    if (m) text = `${e.event === "fallback" ? "fallback: " : ""}${toolLabel(m[1])}`;
  }
  if (e.event === "status") text = statusWord(detail);
  return `<li class="${cls}"${e.step ? ` data-map-step="${escapeHtml(e.step)}"` : ""}><span class="tl-role">${escapeHtml(e.role || "")}</span>` +
    `<span class="tl-text" title="${escapeHtml(detail)}">${e.step ? `<span class="tl-step">${escapeHtml(e.step)}</span> ` : ""}${escapeHtml(text)}</span></li>`;
}

function figHtml(f) {
  if (drawnOnMap(f)) return "";   // study map: the map shows the site and the donors
  const cap = escapeHtml(f.caption || "");
  const img = f.src
    ? `<img src="${f.src}" alt="${cap}" loading="lazy">`
    : `<img data-art="${escapeHtml(f.id)}" alt="${cap}">`;
  return `<figure class="study-fig">${img}${cap ? `<figcaption>${cap}</figcaption>` : ""}</figure>`;
}

// Waiting for data (#419): what the crew needs, why, and what it changes; "Continue without" only when the
// rule allows it, else a line saying the study cannot answer without the data. The file drop area is
// offered here too, and dropping a table plans again on it (Studio.add_table).
function waitingHtml() {
  const req = S.ws.pending_request || {};
  return `<div class="study-waiting">` +
    (req.what ? `<p class="study-line">${escapeHtml(req.what)}</p>` : "") +
    (req.why ? `<p class="study-line muted">Why: ${escapeHtml(req.why)}</p>` : "") +
    (req.effect ? `<p class="study-line muted">What it changes: ${escapeHtml(req.effect)}</p>` : "") +
    (req.can_continue
      ? `<div class="row-actions"><button type="button" class="btn" data-act="continue">Continue without</button></div>`
      : `<p class="study-line muted">Without it, the study cannot answer this brief.</p>`) +
    fileDropHtml("waiting") +
    `</div>`;
}

function runningHtml() {
  const figs = [...S.figures.values()].filter((f) => f.job === S.jobId);
  return `<div class="study-run">` +
    `<ol class="study-timeline" role="log" aria-live="polite" aria-label="What the crew is doing">${S.events.slice(-TIMELINE_MAX).map(eventHtml).join("")}</ol>` +
    `<div class="study-figs">${figs.map(figHtml).join("")}</div>` +
    `<div class="row-actions"><button type="button" class="btn" data-act="stop">Stop</button></div></div>`;
}

const numValue = (k) => `${typeof k.value === "number" ? fmt(k.value) : String(k.value)}${k.unit ? ` ${k.unit}` : ""}`;

function footLine() {
  const run = S.ws.run || {};
  const gates = (run.gates || []).length;
  const failed = (run.failed_gates || []).length;
  const steps = ((S.ws.study || {}).steps || []).length;
  const model = S.ws.model ? `${S.ws.model} via ${S.ws.provider}` : "no model";
  return `${steps} step${steps === 1 ? "" : "s"} · ${gates - failed} of ${gates} gates passed · ${model}`;
}

// Who planned and who wrote, from the replies when they said, else from the workspace.
function crewLine() {
  const study = S.ws.study || {};
  const plan = S.planLine || planLine({ author: (study.plan || {}).author, model: deviceLabel() });
  return [plan, S.proseLine].filter(Boolean).join(" · ");
}

// The engineer's view, above the answer: the grade (with what it means on a title), the decision (the
// headline value, what has to hold, what would change it, what the crew would ask for) and the findings,
// collapsed (#417, #419). Absent on an older report (a recording made before this shipped): every helper
// below is a no-op then, so nothing new renders and nothing breaks (studio-recorded.js).
const GRADE_TITLE = {
  established: "established: at-site data, every gate passed",
  indicative: "indicative: a fallback, a donor transfer or a marginal method",
  screening: "screening: regional or reanalysis data only",
  not_established: "not established: no number could be established for the decision",
};

function gradeBadgeHtml(grade) {
  if (!grade) return "";
  const word = String(grade).replace(/_/g, " ");
  return `<p><span class="study-grade grade-${escapeHtml(String(grade))}" title="${escapeHtml(GRADE_TITLE[grade] || word)}">${escapeHtml(word)}</span></p>`;
}

function decisionHtml(report) {
  const d = report.decision;
  if (!d) return "";
  const conditions = d.conditions || [];
  const changes = d.what_would_change_it || [];
  const requests = report.data_requests || [];
  return `<div class="study-decision">` +
    (d.answer ? `<p class="study-decision-answer">${escapeHtml(d.answer)}</p>` : "") +
    (conditions.length
      ? `<p class="study-line muted">Holds if:</p><ul>${conditions.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>` : "") +
    (changes.length
      ? `<p class="study-line muted">Would change it:</p><ul>${changes.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>` : "") +
    (requests.length
      ? `<p class="study-line muted">The crew would ask for:</p><ul>${requests.map((r) =>
        `<li>${escapeHtml(r.what || "")}${r.effect_on_grade ? `: ${escapeHtml(r.effect_on_grade)}` : ""}</li>`).join("")}</ul>` : "") +
    `</div>`;
}

function findingsHtml(report) {
  const findings = report.findings || [];
  if (!findings.length) return "";
  const rows = findings.map((f) => `<li>[${escapeHtml(String(f.grade || "").replace(/_/g, " "))}] ${escapeHtml(f.claim || "")}` +
    ((f.basis || []).length ? `<br><code class="study-basis">${escapeHtml((f.basis || []).join(", "))}</code>` : "") +
    `</li>`).join("");
  return `<details class="study-findings"><summary>Findings</summary><ul>${rows}</ul></details>`;
}

function doneHtml() {
  const report = S.ws.report || {};
  const numbers = (report.key_numbers || []).slice(0, 12);
  const artifacts = S.ws.artifacts || [];
  const not = report.not_established || [];
  if (S.recorded) return recordedDoneHtml(report, numbers, not);
  const figs = artifacts.filter((a) => a.kind === "figure" && a.media_type === "image/png");
  const docs = DOCS.filter(([id]) => artifacts.some((a) => a.id === id));
  return gradeBadgeHtml(report.grade) +
    decisionHtml(report) +
    findingsHtml(report) +
    `<article class="study-answer ask-result" tabindex="-1" aria-label="The answer">${mdToHtml(report.answer || "No answer was produced.")}</article>` +
    (numbers.length
      ? `<table class="ffa study-numbers"><tbody>${numbers.map((k) => `<tr><td>${escapeHtml(k.label)}</td><td>${escapeHtml(numValue(k))}</td></tr>`).join("")}</tbody></table>`
      : "") +
    (figs.length
      ? `<div class="study-figs">${figs.map((a) => figHtml(S.figures.get(a.id) || { id: a.id, caption: a.caption })).join("")}</div>`
      : "") +
    stepsOnMapHtml(S.ws, toolLabel) +
    steerHtml(S.ws.study, { label: toolLabel, busy: S.busy }) + // steering: per-step controls
    (not.length
      ? `<div class="ask-checks warn"><strong>Not established</strong><ul>${not.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul></div>`
      : "") +
    `<div class="row-actions"><button type="button" class="btn primary" data-act="bundle">Download bundle</button>` +
    `<button type="button" class="btn" data-act="copy-link" title="A link to this plan; whoever opens it reruns it keyless">Copy link</button>` +
    `<button type="button" class="btn" data-act="again">New study</button></div>` +
    (docs.length ? `<p class="study-docs muted">${docs.map(([id, label]) => `<a href="#" data-file="${id}">${label}</a>`).join(" · ")}</p>` : "") +
    `<p class="study-foot muted">${escapeHtml(footLine())}</p>` +
    `<p class="study-by muted">${escapeHtml(crewLine())}</p>`;
}

// A recording, as recorded: the note first (the numbers are the recording's), the answer, the key numbers,
// the figures from their PNG urls, the recorded files as links, Re-run live. No worker call is made.
function recordedDoneHtml(report, numbers, not) {
  const rec = S.recorded;
  return recordedNoteHtml(rec.meta) +
    gradeBadgeHtml(report.grade) +
    decisionHtml(report) +
    findingsHtml(report) +
    `<article class="study-answer ask-result" tabindex="-1" aria-label="The answer">${mdToHtml(report.answer || "No answer was produced.")}</article>` +
    (numbers.length
      ? `<table class="ffa study-numbers"><tbody>${numbers.map((k) => `<tr><td>${escapeHtml(k.label)}</td><td>${escapeHtml(numValue(k))}</td></tr>`).join("")}</tbody></table>`
      : "") +
    (S.figures.size ? `<div class="study-figs">${[...S.figures.values()].map(figHtml).join("")}</div>` : "") +
    stepsOnMapHtml(S.ws, toolLabel) +
    (not.length
      ? `<div class="ask-checks warn"><strong>Not established</strong><ul>${not.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul></div>`
      : "") +
    `<div class="row-actions"><button type="button" class="btn primary" data-act="rerun" title="Run the recorded plan again here, keyless">Re-run live</button>` +
    `<button type="button" class="btn" data-act="again">New study</button></div>` +
    recordedFilesHtml(rec.files) +
    `<p class="study-foot muted">${escapeHtml(footLine())}</p>`;
}

function declinedHtml() {
  const reason = S.declined || (S.ws && S.ws.declined_reason) || "The crew declined.";
  return `<p class="study-declined">${escapeHtml(reason)}</p>` +
    `<div class="row-actions"><button type="button" class="btn primary" data-act="again">Start again</button></div>`;
}

const BOARDS = {
  intake: intakeHtml, review: planHtml, waiting: waitingHtml, running: runningHtml, done: doneHtml,
  declined: declinedHtml,
};
BOARDS.shared = () => sharedBoardHtml(S.shared, { escapeHtml, stepHtml });   // study links (study-link.js)

function renderBoard() {
  const status = boardStatus();
  const el = board();
  el.dataset.status = status;
  el.innerHTML = BOARDS[status]();
  if (status === "done") loadMissingFigures();
  renderCompose(status);
}

function openQuestions() {
  if (!S.ws || S.ws.status !== "intake") return [];
  return ((S.ws.brief || {}).questions || []).filter((q) => q.answer === null || q.answer === undefined);
}

const PLACEHOLDER = {
  intake: "What do you need to know here?",
  questions: "Your answer",
  review: "Or say what to change",
  waiting: "Say \"continue without\", or what you are attaching",
  done: "A question, or a change",
  declined: "Say it another way",
};

function renderCompose(status) {
  const text = $("study-text"), send = $("study-send"), go = $("study-go");
  const questions = openQuestions().length > 0;
  const canType = !S.busy && !S.writing && !S.recorded && status !== "running";
  text.disabled = !canType;
  send.disabled = !canType || (!S.ws && !where());
  go.hidden = !(questions && canType);
  text.placeholder = S.recorded ? "Re-run live to go on here" : PLACEHOLDER[questions ? "questions" : status] || PLACEHOLDER.intake;
}

// A figure the page has not seen (the worker was restarted, or the study came
// from elsewhere) is fetched by id when the report shows it; one the worker
// cannot serve either (its bytes were never in this session) leaves the board
// rather than standing as a broken image. The next run draws it again.
function loadMissingFigures() {
  for (const img of board().querySelectorAll("img[data-art]")) {
    const id = img.dataset.art;
    const drop = () => { const fig = img.closest("figure"); if (fig) fig.remove(); };
    call("studio", { op: "file", workspace: S.ws, artifact_id: id }).then((res) => {
      if (!res || res.error || !res.data) { drop(); return; }
      const f = { id, src: `data:${res.media_type};base64,${res.data}`, caption: img.alt, job: null };
      S.figures.set(id, f);
      if (img.isConnected) img.src = f.src;
    }).catch((err) => { console.info("figure unavailable:", err && err.message); drop(); });
  }
}

// ── the thread ──────────────────────────────────────────────────────────────

function questionsHtml(m, live) {
  const qs = (m.payload || {}).questions || [];
  // Chips answer the first open question: the Consultant reads answers in order.
  const first = live ? openQuestions()[0] : null;
  return qs.map((q) => {
    const opts = first && first.id === q.id ? (q.options || []).slice(0, 8) : [];
    const chips = opts.length
      ? `<div class="study-chips">${opts.map((o) => `<button type="button" class="chip" data-answer="${escapeHtml(String(o))}">${escapeHtml(String(o).replace(/_/g, " "))}</button>`).join("")}</div>`
      : "";
    // With the options as chips and Just go beside Send, the question is the question: the sentence that
    // lists the options and says how to proceed is not repeated in prose.
    const text = opts.length && /\?/.test(q.text) ? q.text.slice(0, q.text.indexOf("?") + 1) : q.text;
    return `<div class="study-q">${escapeHtml(text)}${chips}</div>`;
  }).join("");
}

// The Consultant's brief, from its payload rather than its sentence: the decision, the playbook, the intake
// in words, and what was assumed on a second line.
function briefHtml(m) {
  const b = (m.payload || {}).brief || {};
  const bits = [b.decision || b.problem || ""];
  if (b.playbook) bits.push(String(b.playbook).replace(/_/g, " "));
  for (const [k, v] of Object.entries(b.intake || {})) {
    if (v !== null && v !== undefined && v !== "") bits.push(`${argWord(k)} ${argValue(v)}`);
  }
  const assumed = (b.assumptions || []).slice(-3);
  return `Brief: ${escapeHtml(bits.filter(Boolean).join(" · "))}` +
    (assumed.length ? `<div class="msg-sub muted">assumed: ${escapeHtml(assumed.join("; "))}</div>` : "");
}

function msgHtml(m, live) {
  let body;
  if (m.kind === "brief" && m.payload && m.payload.brief) {
    body = briefHtml(m);
  } else if (m.kind === "plan") {
    const n = (((m.payload || {}).study || {}).steps || []).length;
    body = `Plan: ${n} step${n === 1 ? "" : "s"}`;
  } else if (m.kind === "report") {
    body = escapeHtml((m.payload || {}).title || "Report");
  } else if (m.kind === "questions") {
    body = questionsHtml(m, live);
  } else {
    body = escapeHtml(m.text || "").replace(/\n/g, "<br>");
  }
  return `<li class="msg ${escapeHtml(m.role || "")}"><span class="msg-role">${escapeHtml(ROLE_NAME[m.role] || m.role || "")}</span><div class="msg-body">${body}</div></li>`;
}

function renderThread() {
  const ol = $("study-thread");
  const msgs = (S.ws && S.ws.messages) || [];
  ol.hidden = !msgs.length;
  const last = msgs[msgs.length - 1];
  ol.innerHTML = msgs.map((m) => msgHtml(m, m === last || (m.kind === "questions" && last && last.role === "user"))).join("");
}

function renderAll() {
  renderThread();
  renderBoard();
}

function focusBoard(selector) {
  const el = board().querySelector(selector);
  if (!el) return;
  try { el.scrollIntoView({ block: "nearest", behavior: "smooth" }); } catch { /* fine */ }
  el.focus({ preventScroll: true });
}

// ── the calls ───────────────────────────────────────────────────────────────

function setBusy(on) {
  S.busy = on;
  state.study.running = on;
  renderBoard();
}

function job(op, extra = {}) {
  const j = callCancelable("studio", {
    op, workspace: S.ws, catchment: S.catchment, area_km2: S.area, donors: S.donors, ...modelForRun(), ...extra,
  });
  S.cancel = j.cancel;
  S.jobId = j.id;
  return j.promise.finally(() => { if (S.jobId === j.id) { S.cancel = null; S.jobId = null; } });
}

function applyReply(res, op) {
  S.ws = res.workspace || S.ws;
  rememberPlain(res.plain); // steering: the plan in plain words and the step controls
  S.busy = false;
  state.study.running = false;
  S.editing = false;
  $("study-text").value = "";
  const r = res.reply || {};
  const p = r.payload || {};
  note("");
  if (r.kind === "plan" && p.errors) note(`The edit was not accepted: ${p.errors.join("; ")}`, "warn");
  if (r.kind === "plan") { S.proposal = null; S.planLine = null; S.proseLine = null; }
  if (r.kind === "report" && (op === "approve" || op === "follow_up")) {
    S.proposal = null;
    if (p.plan_used && S.planSource === "shared") {
      S.planLine = sharedPlanLine({ used: p.plan_used, errors: p.plan_errors || [] });
    } else if (p.plan_used && S.planSource === "recorded") {
      S.planLine = recordedPlanLine({ used: p.plan_used, errors: p.plan_errors || [], model: S.planModel });
    } else if (p.plan_used) {
      S.planLine = planLine({ used: p.plan_used, errors: p.plan_errors || [], model: deviceLabel() });
    }
    S.planSource = null;
    S.proseLine = null;
  }
  renderAll();
  showStudyMapFor(S.ws);
  persist();
  if (r.kind === "plan") { focusBoard(".study-plan"); announce("The plan is ready to approve."); maybePlanOnDevice(); return; }
  if (r.kind === "report") {
    focusBoard(".study-answer");
    announce("The report is ready.");
    if (op === "approve" || op === "follow_up") maybeNarrateOnDevice();
    return;
  }
  if (r.kind === "declined") { announce(r.text || "Declined."); return; }
  $("study-text").focus({ preventScroll: true });
}

function failed(err) {
  S.busy = false;
  state.study.running = false;
  if (err instanceof Cancelled) note("Stopped.", "warn");
  else note(`The crew could not continue: ${err.message}`, "error");
  renderBoard();
}

// ── the device model on the crew ────────────────────────────────────────────
// Keyless, with a model already on the machine: the brief before the
// Consultant sees it, the plan at review (validated by the engine before it
// is shown; the tree stands when it fails), the prose after the run (the
// Critic's checks drop what the results do not carry). Every phase is bounded
// (one or a few calls, 25 s each), never starts a download, and on failure
// leaves the keyless result in place and says so in one line.

const deviceLabel = () => localReaderLabel() || "this device";

async function deviceOnCrew() {
  return !modelForRun().provider && await localModelReady();
}

// The crew's prompts and schemas: the exported file next to the page when the
// build made one, else what the engine in the worker exports, else none.
async function loadPrompts() {
  if (S.prompts !== undefined) return S.prompts;
  let prompts = null;
  try {
    const resp = await fetch("./prompts.json?v=__BUILD__", { cache: "force-cache" });
    if (resp.ok) prompts = await resp.json();
  } catch { /* not shipped */ }
  if (!prompts) {
    try { prompts = await call("studio", { op: "prompts" }); } catch (err) { console.info("no prompts from the worker:", err && err.message); }
  }
  S.prompts = prompts && typeof prompts === "object" ? prompts : null;
  return S.prompts;
}

// The first sentence, read on the device when a small model is already there
// and no key is in play. What it states goes in as the brief and the intake;
// the Consultant's rules fill the rest and ask for what is missing.
async function readOnDevice(text) {
  note("Reading your words on your device…");
  try {
    const reply = await generateJsonLocally({
      system: briefPrompt(), prompt: text, schema: briefSchema(), temperature: 0.1, timeoutMs: BRIEF_TIMEOUT_MS,
      onProgress: (m) => note(m),
    });
    return parseBriefReply(reply);
  } catch (err) {
    console.info("on-device brief unavailable:", err && err.message);
    return null;
  }
}

// At review: the plan from the device, checked by the engine's validator
// before the card shows it. Any call the reader makes meanwhile (Approve,
// a change of brief) moves S.run on, and what arrives late is dropped.
async function maybePlanOnDevice() {
  if (!S.ws || S.ws.status !== "review" || ((S.ws.study || {}).plan || {}).author === "device") return;
  if (!(await deviceOnCrew())) return;
  const my = S.run;
  let ctx;
  try {
    ctx = await call("studio", { op: "context", role: "methodologist", workspace: S.ws });
  } catch (err) {
    console.info("no methodologist context:", err && err.message);
    return;
  }
  if (my !== S.run || !ctx || ctx.error) { if (ctx && ctx.error) console.info("methodologist context:", ctx.error); return; }
  const prompts = await loadPrompts();
  if (my !== S.run) return;
  note("Planning on your device…");
  const label = deviceLabel();
  const { plan, error } = await planOnDevice({ context: ctx, prompts, generate: generateJsonLocally, onProgress: (m) => note(m) });
  if (my !== S.run) return;
  if (!plan) {
    S.planLine = `the playbook's plan (the device model gave no plan: ${error})`;
    note("");
    renderBoard();
    return;
  }
  let chk = null;
  try {
    chk = await call("studio", { op: "check_plan", workspace: S.ws, plan });
  } catch (err) {
    chk = { ok: false, errors: [err.message] };
  }
  if (my !== S.run) return;
  note("");
  if (chk && chk.ok === false) {
    S.planLine = planLine({ used: "tree", errors: chk.errors && chk.errors.length ? chk.errors : ["no reason given"], model: label });
    renderBoard();
    return;
  }
  if (S.editing) {   // the reader is editing the tree's plan: that is the plan
    S.planLine = "the playbook's plan";
    return;
  }
  const steps = chk && Array.isArray(chk.steps) && chk.steps.length ? chk.steps : plan.steps;
  S.proposal = { plan: { ...plan, steps }, steps, model: label };
  S.planLine = planLine({ used: "proposed", model: label });
  renderBoard();
  focusBoard(".study-plan");
  announce("The device wrote a plan; it is on the card.");
}

// After the run: the summary and the recommendations from the device, then
// the steps while it is quick, handed to the engine's narrate, which keeps
// only what the checks allow and remakes the documents.
async function maybeNarrateOnDevice() {
  if (!S.ws || S.ws.status !== "done") return;
  if (!(await deviceOnCrew())) return;
  const my = S.run;
  let ctx;
  try {
    ctx = await call("studio", { op: "context", role: "author", workspace: S.ws });
  } catch (err) {
    console.info("no author context:", err && err.message);
    return;
  }
  if (my !== S.run || !ctx || ctx.error) { if (ctx && ctx.error) console.info("author context:", ctx.error); return; }
  const prompts = await loadPrompts();
  if (my !== S.run) return;
  const label = deviceLabel();
  S.writing = true;
  renderCompose("done");
  note("Writing on your device…");
  try {
    const { sections, error } = await narrateOnDevice({ context: ctx, prompts, generate: generateJsonLocally, onProgress: (m) => note(m) });
    if (my !== S.run) return;
    if (!Object.keys(sections).length) {
      S.proseLine = proseLine({ failed: `the device model ${error}` });
      return;
    }
    note("Checking the prose…");
    const res = await call("studio", { op: "narrate", workspace: S.ws, sections, source: "device" });
    if (my !== S.run) return;
    if (!res || res.error) {
      S.proseLine = proseLine({ failed: (res && res.error) || "the engine kept the template text" });
      return;
    }
    S.ws = res.workspace || S.ws;
    const p = ((res.reply || {}).payload) || {};
    S.proseLine = proseLine({ writtenBy: p.written_by || "device", dropped: p.dropped, model: label });
    persist();
  } catch (err) {
    if (my !== S.run) return;
    S.proseLine = proseLine({ failed: err.message });
  } finally {
    if (my === S.run) {
      S.writing = false;
      note("");
      renderAll();
    }
  }
}

// ── start, say, approve ─────────────────────────────────────────────────────

async function start(text, { intake: given = null } = {}) {
  const w = where();
  if (!w) { note("Pick a gauge or a spot on the map first.", "warn"); return; }
  const my = ++S.run;
  S.site = w;
  S.siteReady = false;
  S.declined = null;
  S.editing = false;
  S.events = [];
  S.figures.clear();
  clearStudyMap();
  S.proposal = null;
  S.planLine = null;
  S.proseLine = null;
  setBusy(true);
  note(state.workerReady ? "" : "Loading Python in your browser (about 15 MB, once)…");
  try {
    let intake = given, brief = null;
    if (!given && await deviceOnCrew()) {
      const read = await readOnDevice(text);
      if (my !== S.run) return;
      if (read) ({ intake, brief } = read);
    }
    await ensureCatalogInWorker();
    await loadSiteInfo(w, my);
    if (my !== S.run) return;
    const tables = {};
    for (const f of S.files) tables[f.id] = f.csv;
    note("");
    const res = await job("start", {
      lat: w.lat, lon: w.lon, text, tables, use_frame: Boolean(S.useMyData && hasTable()), frame_label: tableLabel(),
      intake, proposed: brief ? { brief, source: "device" } : null,
    });
    if (my !== S.run) return;
    applyReply(res, "start");
  } catch (err) {
    if (my !== S.run) return;
    failed(err);
  }
}

async function callStudio(op, extra = {}) {
  if (S.busy || S.writing || !S.ws) return;
  const my = ++S.run;
  S.events = [];
  S.declined = null;
  setBusy(true);
  try {
    if (op === "approve" || op === "follow_up" || op === "add_table" || op === "steer") {
      await ensureCatalogInWorker();
      // A resumed study (a reload, a dropped workspace.json) has not read its site yet.
      if (S.site && !S.siteReady) await loadSiteInfo(S.site, my);
      if (my !== S.run) return;
      // A run may reach for donors (similar_basins, regionalize_signatures); the worker cannot read the
      // parquet tables, so the page hands over the ones it holds, once.
      if (S.catchment && !("donors_tables" in extra)) {
        extra = { ...extra, donors_tables: await donorTablesForWorker().catch((err) => {
          console.warn("donor tables unavailable, the donor steps will say so:", err && err.message);
          return null;
        }) };
        if (my !== S.run) return;
      }
    }
    const res = await job(op, extra);
    if (my !== S.run) return;
    applyReply(res, op);
  } catch (err) {
    if (my !== S.run) return;
    failed(err);
  }
}

function send(given) {
  const text = (given !== undefined ? given : $("study-text").value).trim();
  if (!text || S.busy || S.writing) return;
  if (!S.ws) { start(text); return; }
  const st = S.ws.status;
  if (st === "done") callStudio("follow_up", { text });
  else if (st === "intake" || st === "review" || st === "waiting") callStudio("say", { text });
  else if (st === "declined") { reset(); start(text); }
}

function coerce(raw) {
  const t = raw.trim();
  if (/^-?\d+(\.\d+)?$/.test(t)) return Number(t);
  if (t === "true") return true;
  if (t === "false") return false;
  if (/^[[{]/.test(t)) { try { return JSON.parse(t); } catch { return t; } }
  return t;
}

// The inputs that differ from the steps on the card, as the Methodologist's
// edits: {step id: {arguments: {name: value}}}, revalidated in the worker.
function readEdits(steps) {
  const edits = {};
  const byId = new Map(steps.map((s) => [s.id, s]));
  for (const input of board().querySelectorAll("input[data-step]")) {
    const s = byId.get(input.dataset.step);
    if (!s || !input.value.trim()) continue;
    const old = (s.arguments || {})[input.dataset.arg];
    if (fmtArg(old) === input.value.trim()) continue;
    (edits[s.id] = edits[s.id] || { arguments: {} }).arguments[input.dataset.arg] = coerce(input.value);
  }
  return Object.keys(edits).length ? edits : null;
}

function approve() {
  if (S.proposal) {
    // The device's plan, with the reader's edits folded in, goes as the proposed plan: the engine validates it
    // again and runs it, or runs the tree and says why.
    const edits = S.editing ? readEdits(S.proposal.steps) : null;
    const steps = S.proposal.steps.map((s) => (edits && edits[s.id]
      ? { ...s, arguments: { ...(s.arguments || {}), ...edits[s.id].arguments } } : s));
    callStudio("approve", { plan: { ...S.proposal.plan, steps }, edits: null });
    return;
  }
  const edits = S.editing ? readEdits(((S.ws.study || {}).steps || [])) : null;
  callStudio("approve", { edits });
}

function decline() {
  S.declined = "You declined the plan. Say what to change, or start again.";
  S.editing = false;
  renderBoard();
  $("study-text").focus({ preventScroll: true });
}

// Stop that means stop: the worker is terminated and boots again (the
// progress bar as at first load). The page's copy of the workspace is the one
// from before the run, so the plan is kept; the figures the run had drawn are
// gone with the worker; the next Approve rebuilds the study from this copy.
function stop() {
  if (!S.busy) return;
  const stoppedJob = S.jobId;
  S.run++;
  restartWorker();
  S.busy = false;
  state.study.running = false;
  S.cancel = null;
  S.jobId = null;
  for (const [id, f] of S.figures) if (f.job === stoppedJob) S.figures.delete(id);
  S.events = [];
  renderAll();
  showStudyMapFor(S.ws);
  note(S.ws ? "Stopped; the figures made so far are gone, the plan is kept." : "Stopped.", "warn");
}

function reset() {
  S.run++;
  if (S.cancel) S.cancel();
  S.ws = null;
  S.site = null;
  S.siteReady = false;
  S.declined = null;
  S.editing = false;
  S.busy = false;
  S.writing = false;
  state.study.running = false;
  S.files = [];
  S.useMyData = false;
  S.figures.clear();
  S.events = [];
  S.proposal = null;
  S.planLine = null;
  S.proseLine = null;
  S.planSource = null;
  S.recorded = null;
  state.study.recorded = null;
  clearStudyMap();
  S.shared = null;
  state.study.link = null;
  note("");
  renderAll();
  refreshResume();
  if (drawerOpen()) writeUrl();
}

// ── saved studies ───────────────────────────────────────────────────────────
// After every reply the workspace (without bytes) and the PNG figures go to
// IndexedDB under the workspace id; the last five are kept. The intake board
// offers the latest one at the place on screen (or, with nothing picked, the
// latest anywhere), and a dropped workspace.json from a bundle resumes too.

async function blobOf(src) {
  const r = await fetch(src);
  return r.blob();
}

async function persist() {
  if (!S.ws || !S.site) return;
  try {
    const figures = {};
    for (const [id, f] of S.figures) {
      if (f.src) figures[id] = { blob: await blobOf(f.src), caption: f.caption || "", step: f.step || null };
    }
    const { key, lat, lon, text } = S.site;
    await saveStudy({ id: S.ws.id, at: Date.now(), site: { key, lat, lon, text }, ws: S.ws, figures,
                      lines: { plan: S.planLine, prose: S.proseLine } });
  } catch (err) {
    console.info("study not saved:", err && err.message);
  }
}

async function refreshResume() {
  const w = where();
  let rec = null;
  try { rec = await latestStudy(w ? w.key : null); } catch { rec = null; }
  S.resume = rec && rec.site ? rec : null;
  if (!S.ws && !S.busy && drawerOpen() && drawerMode() === "study") renderBoard();
}

function openWorkspace(ws, figures, lines = {}) {
  S.run++;
  S.ws = ws;
  S.site = siteFrom({ ...(ws.site || {}), ...(S.resume && S.resume.id === ws.id ? S.resume.site : {}) });
  S.siteReady = false;
  S.declined = null;
  S.editing = false;
  S.busy = false;
  S.writing = false;
  state.study.running = false;
  S.events = [];
  S.figures = figures;
  S.proposal = null;
  S.planLine = lines.plan || null;
  S.proseLine = lines.prose || null;
  S.recorded = null;
  state.study.recorded = null;
  note("");
  renderAll();
  showStudyMapFor(ws);
  announce(`Study resumed: ${statusWord(ws.status)}.`);
}

// ── the recorded studies ────────────────────────────────────────────────────
// Made once with a model and committed under showcase/studies/ (docs/studio.md,
// "Recorded studies"). The index is read once and offered as chips; a chip
// opens the recording as its finished board, with no worker call; Re-run live
// starts the same study at the recorded site and approves the recorded plan,
// which runs here keyless through the validator and the gates.

async function loadIndex() {
  if (S.index !== undefined) return S.index;
  try {
    const data = await recordedIndex(RECORDED_BASE, { version: "__BUILD__" });
    S.index = data.studies.filter((r) => r && r.id);
  } catch (err) {
    console.info("recorded studies unavailable:", err && err.message);
    S.index = [];
  }
  if (!S.ws && !S.busy && drawerOpen() && drawerMode() === "study") renderBoard();
  return S.index;
}

async function openRecorded(id) {
  note("Opening the recorded study…");
  let rec;
  try {
    rec = await loadRecorded(RECORDED_BASE, id, { version: "__BUILD__" });
  } catch (err) {
    note(`Could not open that recording: ${err.message}`, "error");
    return;
  }
  if (!rec.workspace || !rec.workspace.status) { note("That recording carries no study.", "warn"); return; }
  S.run++;
  if (S.cancel) S.cancel();
  const site = (rec.meta && rec.meta.site) || rec.workspace.site || {};
  S.ws = rec.workspace;
  S.recorded = rec;
  S.site = siteFrom({ lat: site.lat, lon: site.lon, text: site.name, key: `rec/${rec.id}` });
  S.siteReady = false;
  S.declined = null;
  S.editing = false;
  S.busy = false;
  S.writing = false;
  state.study.running = false;
  state.study.recorded = rec.id;
  S.files = [];
  S.events = [];
  S.figures = new Map(recordedFigures(rec).map((f) => [f.id, f]));
  S.proposal = null;
  S.planLine = null;
  S.proseLine = null;
  S.planSource = null;
  note("");
  renderAll();
  showStudyMapFor(S.ws);
  if (drawerOpen()) writeUrl();
  focusBoard(".study-answer");
  announce(`Recorded study opened: ${(rec.meta && rec.meta.title) || rec.id}.`);
}

// The same study at the recorded site, live: the point on the map (so the URL and the header say where),
// the recorded text and intake as the brief, the defaults for any question, the recorded plan approved.
async function rerunRecorded() {
  const rec = S.recorded;
  if (!rec || S.busy) return;
  const b = replayBrief(rec.workspace);
  const plan = replayPlan(rec.workspace);
  if (!Number.isFinite(Number(b.lat)) || !Number.isFinite(Number(b.lon))) { note("This recording has no site.", "warn"); return; }
  const model = (rec.meta && rec.meta.model) || null;
  S.recorded = null;
  state.study.recorded = null;
  S.ws = null;
  S.figures = new Map();
  S.files = Object.entries(b.tables || {}).filter(([, csv]) => typeof csv === "string" && csv).map(([id, csv]) => ({ id, csv }));
  S.useMyData = false;
  try { actions.selectPoint(Number(b.lat), Number(b.lon), { fly: true, push: true }); } catch (err) { console.warn(err); }
  openDrawer({ mode: "study" });
  await start(b.text || (rec.meta && rec.meta.problem) || "", { intake: b.intake });
  if (S.ws && S.ws.status === "intake" && openQuestions().length) await callStudio("say", { text: "just go" });
  if (!S.ws || S.ws.status !== "review") return;
  S.planSource = plan ? "recorded" : null;
  S.planModel = model;
  await callStudio("approve", plan ? { plan: { ...plan, source: "recorded" }, edits: null } : { edits: null });
}

async function resumeSaved() {
  if (!S.resume) return;
  const rec = await loadStudy(S.resume.id);
  if (!rec || !rec.ws) { note("That study is no longer in this browser.", "warn"); S.resume = null; renderBoard(); return; }
  const figures = new Map();
  for (const [id, f] of Object.entries(rec.figures || {})) {
    try {
      figures.set(id, { id, src: URL.createObjectURL(f.blob), caption: f.caption || "", step: f.step, job: null });
    } catch { /* a figure that cannot be shown is fetched by id */ }
  }
  openWorkspace(rec.ws, figures, rec.lines || {});
}

// A workspace.json from a bundle (or the CLI): the study as it was, without
// bytes; the figures and the documents come back with the next run.
function resumeWorkspace(obj) {
  const ws = { ...obj };
  const figures = new Map();
  ws.artifacts = (ws.artifacts || []).map((a) => {
    if (a && a.data && a.media_type === "image/png") {
      figures.set(a.id, { id: a.id, src: `data:image/png;base64,${a.data}`, caption: a.caption || "", step: a.step, job: null });
    }
    const { data: _bytes, ...rest } = a || {};
    return rest;
  });
  S.resume = null;
  openWorkspace(ws, figures);
  persist();
}

const looksLikeWorkspace = (obj) => Boolean(obj && typeof obj === "object" && obj.id && obj.status && obj.brief && obj.site);

// ── files in, files out ─────────────────────────────────────────────────────

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function saveBytes(name, b64, type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes], { type }));
  a.download = name.replace(/[^\w.-]+/g, "_");
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// A CSV goes in as it is; an Excel file is turned into CSV in the worker, so
// every table travels in the workspace the same way and round-trips. A
// workspace.json resumes the study it carries. Before a study starts a table
// only stages in S.files, sent along with "start"; once one exists, a table
// is attached to it directly (Studio.add_table, #419) and the reply it comes
// back with is applied like any other, so a plan or a report renders.
async function addFiles(list) {
  for (const file of list) {
    const id = file.name.replace(/\s+/g, "_");
    try {
      if (/\.json$/i.test(file.name)) {
        let obj = null;
        try { obj = JSON.parse(await file.text()); } catch { obj = null; }
        if (!looksLikeWorkspace(obj)) throw new Error("not a workspace.json from a study bundle");
        resumeWorkspace(obj);
        return;
      }
      let csv, tableId = id;
      if (/\.xlsx?$|\.xls$/i.test(file.name)) {
        note(`Reading ${file.name}…`);
        const res = await call("studio", { op: "table", name: file.name, data: toBase64(await file.arrayBuffer()) });
        csv = res.csv;
        tableId = id.replace(/\.xlsx?$|\.xls$/i, ".csv");
      } else {
        csv = await file.text();
      }
      if (S.ws) {
        note(`Attaching ${tableId}…`);
        await callStudio("add_table", { name: tableId, csv });
      } else {
        S.files.push({ id: tableId, csv });
      }
    } catch (err) {
      note(`Could not read ${file.name}: ${err.message}`, "error");
      renderBoard();
      return;
    }
  }
  note("");
  renderBoard();
}

async function downloadArtifact(id) {
  if (!S.ws) return;
  note("Preparing the file…");
  try {
    const res = await call("studio", id === "bundle"
      ? { op: "export", workspace: S.ws }
      : { op: "file", workspace: S.ws, artifact_id: id });
    if (!res || res.error) throw new Error((res && res.error) || "no file");
    const name = id === "bundle" ? `study-${S.ws.id}.zip` : String(res.name || id).split("/").pop();
    saveBytes(name, res.data, res.media_type);
    note("");
  } catch (err) {
    note(`Could not prepare the file: ${err.message}`, "error");
  }
}

// ── the live run ────────────────────────────────────────────────────────────

function appendEvent(e) {
  const tl = board().querySelector(".study-timeline");
  if (!tl) return;
  tl.insertAdjacentHTML("beforeend", eventHtml(e));
  while (tl.children.length > TIMELINE_MAX) tl.firstElementChild.remove();
  tl.scrollTop = tl.scrollHeight;
}

function appendFigure(f) {
  const box = board().querySelector(".study-run .study-figs");
  if (box) box.insertAdjacentHTML("beforeend", figHtml(f));
}

// ── study links ─────────────────────────────────────────────────────────────
// Copy link at the end of a study: the worker turns the approved plan (never
// the results) into a token and checks it the way a recipient would. Opening
// #study=<token> or ?study_url=<https study.yaml> shows the plan, checked
// against the method catalogue in the worker, with a "shared study" label;
// Run starts the same study at the same place and approves that plan, which
// the engine validates again at the site, keyless (aquascope.study_link).

async function copyStudyLink() {
  if (!S.ws) return;
  note("Making the link…");
  let res;
  try {
    res = await call("studio", { op: "link", workspace: S.ws, name: S.site && S.site.text });
  } catch (err) {
    res = { ok: false, errors: [err.message] };
  }
  if (!res || !res.ok) { note(`No link: ${((res && res.errors) || ["unknown reason"])[0]}`, "warn"); return; }
  const url = linkUrl(canonicalUrl(), res.token);
  try {
    await navigator.clipboard.writeText(url);
    note("Link copied. Whoever opens it sees this plan and can rerun it in their browser.");
  } catch {
    window.prompt("Copy this link", url);
    note("");
  }
}

export async function openSharedStudy({ link = null, studyUrl = null } = {}) {
  if (S.busy) { note("A study is running; stop it before opening a shared one.", "warn"); return; }
  reset();
  const my = S.run;
  state.study.link = link;
  S.shared = { loading: true };
  openDrawer({ mode: "study" });
  renderAll();
  if (studyUrl && studyUrl.error) {
    S.shared = { ok: false, errors: [studyUrl.error] };
    renderAll();
    return;
  }
  note(state.workerReady ? "" : "Loading Python in your browser (about 15 MB, once)…");
  let res;
  try {
    res = await call("studio", link ? { op: "open_link", token: link }
      : { op: "open_link", yaml: await fetchStudyYaml(studyUrl) });
  } catch (err) {
    res = { ok: false, errors: [err.message] };
  }
  if (my !== S.run) return;
  S.shared = { ...(res || { ok: false, errors: ["no reply"] }), from: studyUrl || null };
  note("");
  if (S.shared.ok) {
    try { actions.selectPoint(S.shared.study.lat, S.shared.study.lon, { fly: true, push: false }); } catch (err) { console.warn(err); }
  }
  openDrawer({ mode: "study" });
  renderAll();
  writeUrl();   // the point and the shared study, so a reload opens it again
  focusBoard(".study-shared");
  announce(S.shared.ok ? "A shared study is open; Run reruns it here." : "The shared study cannot be opened.");
}

// The shared plan, run here: the same brief at the same place, the defaults for any question, the plan
// approved as source "shared" (the engine's validator decides whether it runs or the tree stands).
async function runShared() {
  const sh = S.shared && S.shared.ok ? S.shared.study : null;
  if (!sh || S.busy) return;
  S.shared = null;
  S.ws = null;
  S.figures = new Map();
  S.files = [];
  S.useMyData = false;
  try { actions.selectPoint(sh.lat, sh.lon, { fly: true, push: true }); } catch (err) { console.warn(err); }
  openDrawer({ mode: "study" });
  await start(sh.text, { intake: sh.intake });
  if (S.ws && S.ws.status === "intake" && openQuestions().length) await callStudio("say", { text: "just go" });
  if (!S.ws || S.ws.status !== "review") return;
  S.planSource = "shared";
  await callStudio("approve", { plan: { ...sh.plan, source: "shared" }, edits: null });
}

// ── open, wire ──────────────────────────────────────────────────────────────

export function openStudy({ fresh = false, recorded = null } = {}) {
  const w = where();
  // "Study this place" on a panel: a study made elsewhere is finished with; one made here goes on.
  if (fresh && S.ws && (!w || !S.site || w.key !== S.site.key)) reset();
  openDrawer({ mode: "study" });
  renderAll();
  refreshResume();
  loadIndex();
  if (recorded) { openRecorded(recorded); return; }
  if (!S.busy) $("study-text").focus({ preventScroll: true });
}

export function toggleStudy() {
  if (drawerOpen() && drawerMode() === "study") closeDrawer(); else openStudy();
}

function onBoardClick(e) {
  const act = e.target.closest("[data-act]");
  if (act) {
    const what = act.dataset.act;
    if (what === "approve") approve();
    else if (what === "edit") { S.editing = !S.editing; renderBoard(); }
    else if (what === "decline") decline();
    else if (what === "continue") callStudio("say", { text: "continue without" });
    else if (what === "stop") stop();
    else if (what === "bundle") downloadArtifact("bundle");
    else if (what === "again") reset();
    else if (what === "my-data") { S.useMyData = !S.useMyData; renderBoard(); }
    else if (what === "resume") resumeSaved();
    else if (what === "rerun") rerunRecorded();
    else if (what === "more-recorded") { S.moreRecorded = true; renderBoard(); }
    else if (what === "copy-link") copyStudyLink();
    else if (what === "run-shared") runShared();
    return;
  }
  const onMap = e.target.closest("[data-map-step]");
  if (onMap) { focusStudyStep(onMap.dataset.mapStep); return; }
  const chip = e.target.closest("[data-recorded]");
  if (chip) { openRecorded(chip.dataset.recorded); return; }
  const file = e.target.closest("[data-file]");
  if (file) { e.preventDefault(); downloadArtifact(file.dataset.file); return; }
  const remove = e.target.closest("[data-remove]");
  if (remove) { S.files.splice(Number(remove.dataset.remove), 1); renderBoard(); }
}

function onBoardChange(e) {
  if (e.target.id === "study-file") {
    if (e.target.files && e.target.files.length) addFiles([...e.target.files]);
  } else if (e.target.dataset.opt === "key") {
    S.useKey = e.target.checked;
  }
}

// Wired by app.js on first use (the Study button, the drawer's radio, "Study
// this place", a #study=1 link); the Study button itself is wired there,
// before anything is awaited (#271).
export function initStudy() {
  $("study-send").addEventListener("click", () => send());
  $("study-go").addEventListener("click", () => send("just go"));
  $("study-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  const el = board();
  el.addEventListener("click", onBoardClick);
  el.addEventListener("change", onBoardChange);
  initStudyControls(el, (extra) => callStudio("steer", extra)); // steering: Rerun this step
  for (const type of ["dragenter", "dragover"]) {
    el.addEventListener(type, (e) => { if (S.busy || S.writing) return; e.preventDefault(); el.classList.add("over"); });
  }
  for (const type of ["dragleave", "drop"]) {
    el.addEventListener(type, (e) => { e.preventDefault(); el.classList.remove("over"); });
  }
  el.addEventListener("drop", (e) => {
    if (S.busy || S.writing) return;
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) addFiles([...files]);
  });
  $("study-thread").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-answer]");
    if (chip) send(chip.dataset.answer);
  });
  $("drawer").addEventListener("drawermode", (e) => { if (e.detail.mode === "study") { renderAll(); refreshResume(); loadIndex(); } });
  onStudioProgress((event, id) => {
    if (id !== S.jobId) return;
    S.events.push(event);
    appendEvent(event);
  });
  onStudioArtifact((artifact, id) => {
    if (id === S.jobId && studyMapArtifact(artifact)) return;   // study map: drawn as the steps land
    if (id !== S.jobId || artifact.media_type !== "image/png" || !artifact.data) return;
    const f = { id: artifact.id, src: `data:image/png;base64,${artifact.data}`, caption: artifact.caption || "",
                step: artifact.step, job: id };
    S.figures.set(artifact.id, f);
    appendFigure(f);
  });
  actions.openStudy = openStudy;
  renderAll();
}

// Exported for explorer/tests/studio-requests.test.mjs: the board state, the status it maps to, the board
// renderers by status and the file intake, none of which otherwise leave this module.
export { S, boardStatus, BOARDS, addFiles };
