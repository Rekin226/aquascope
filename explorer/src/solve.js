// Solve: a problem at a place, planned first, checked at every step. The page
// is a thin face over aquascope.ai_engine.team, running in the Pyodide worker:
// solve(execute=False) plans, run_reviewed() runs what the reader approved.
// Six stages, one on screen at a time: where, what, whether it can be answered
// here, the plan, the run, the result. A finished stage folds to one line;
// reopening an earlier one clears everything after it. No key is ever needed:
// the tree fills the plan and a template writes the prose. The key Ask holds
// is borrowed when there is one, never asked for twice. A sentence typed with
// no chip chosen is read on the reader's device when a small model is already
// there (Chrome's built-in, or the one Ask loaded), and by the keyword rules
// otherwise; either way the chip and the fields it fills are the ones shown.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, copyText, downloadBlob, escapeHtml, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { shapeSvg } from "./shapes.js?v=__BUILD__";
import { requestAssess } from "./assess.js?v=__BUILD__";
import { catchmentForWorker } from "./basins.js?v=__BUILD__";
import { askModelConfig, mdToHtml } from "./ask.js?v=__BUILD__";
import { closeDrawer, drawerMode, drawerOpen, openDrawer, setStatusEl } from "./shell.js?v=__BUILD__";
import { Cancelled, call, callCancelable, ensureCatalogInWorker, onSolveProgress } from "./worker-client.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { intakePrompt, intakeSchema, parseIntakeReply } from "./intake.js?v=__BUILD__";

const STAGES = ["where", "what", "recon", "plan", "run", "result"];

// Tool names in words for the checklist; anything else is its id with the
// underscores spaced out.
const TOOL_LABEL = {
  describe_catchment: "Describe the catchment",
  analyze_station: "Analyse the gauge record",
  flood_frequency: "Fit the flood frequency",
  get_timeseries: "Fetch the series",
  similar_basins: "Find donor gauges",
  regionalize_signatures: "Transfer flow signatures from donors",
  anywhere: "ERA5 climate and GloFAS for this cell",
  find_stations: "Find gauges nearby",
  assess_site: "Reconnaissance",
  sgi_drought: "Standardised Groundwater Index",
  recharge: "Water-table-fluctuation recharge",
};
const toolLabel = (t) => TOOL_LABEL[t] || String(t || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

const S = {
  playbooks: [], playbook: null, pendingChip: null,
  recon: null, plan: null, study: null, result: null,
  editing: false, cancel: null, jobId: null, run: 0, startedAt: 0, whereKey: null,
  intakeNote: null,   // who read the sentence: the device, or the keyword rules, and why
};

// ── stages ──────────────────────────────────────────────────────────────────

const stageEl = (name) => $(`stage-${name}`);

function setStage(name, kind, summary) {
  const el = stageEl(name);
  el.dataset.state = kind;
  if (summary !== undefined) el.querySelector(".stage-sum").textContent = summary || "";
}

// Open `name`; fold what came before it, clear what comes after.
function openStage(name) {
  const i = STAGES.indexOf(name);
  STAGES.forEach((s, k) => {
    if (k < i) { if (stageEl(s).dataset.state === "open") setStage(s, "done"); }
    else if (k === i) setStage(s, "open");
    else setStage(s, "pending", "");
  });
  S.current = name;
}

// Back to the start: where and what, both open, nothing after them.
function openIntake() {
  stopRun();
  setStage("where", "open", "");
  setStage("what", "open", "");
  for (const s of STAGES.slice(2)) setStage(s, "pending", "");
  S.current = "what";
  S.recon = null; S.plan = null; S.study = null; S.result = null; S.editing = false;
  renderWhere();
}

function onHeadClick(name) {
  const el = stageEl(name);
  if (el.dataset.state === "pending") return;
  if (name === "where" || name === "what") { if (el.dataset.state === "done") openIntake(); return; }
  if (name === "plan") { if (el.dataset.state === "done" && S.study) { stopRun(); openStage("plan"); } return; }
  // The reconnaissance and the timeline are there to be looked at again;
  // unfolding them takes nothing away from the stage that is current.
  if ((name === "recon" || name === "run") && name !== S.current) {
    el.dataset.state = el.dataset.state === "open" ? "done" : "open";
  }
}

// ── where ───────────────────────────────────────────────────────────────────

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

function renderWhere() {
  const w = where();
  $("solve-where-text").innerHTML = w ? w.html : `<span class="muted">Click a gauge or a spot on the map first.</span>`;
  $("solve-plan-btn").disabled = !w;
}

// ── what ────────────────────────────────────────────────────────────────────

const chosen = () => S.playbooks.find((p) => p.id === S.playbook) || null;

function renderChips() {
  const box = $("solve-chips");
  box.innerHTML = "";
  for (const pb of S.playbooks) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = pb.title;
    b.title = pb.description || "";
    b.setAttribute("aria-pressed", pb.id === S.playbook ? "true" : "false");
    b.addEventListener("click", () => chooseChip(pb.id === S.playbook ? null : pb.id));
    box.appendChild(b);
  }
}

function chooseChip(id) {
  S.playbook = id;
  state.solve.playbook = id;
  renderChips();
  renderIntake(chosen());
  if (state.drawerOpen) writeUrl();
}

// The playbook's intake fields as compact inputs with their defaults.
function renderIntake(pb) {
  const box = $("solve-intake");
  box.innerHTML = "";
  box.hidden = !pb || !(pb.intake || []).length;
  if (!pb) return;
  for (const f of pb.intake || []) {
    const label = document.createElement("label");
    let input;
    if (f.type === "choice") {
      input = document.createElement("select");
      for (const o of f.options || []) {
        const opt = document.createElement("option");
        opt.value = String(o);
        opt.textContent = String(o);
        opt.selected = o === f.default;
        input.appendChild(opt);
      }
    } else if (f.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(f.default);
      label.className = "solve-bool";
    } else {
      input = document.createElement("input");
      input.type = f.type === "int" || f.type === "float" ? "number" : "text";
      if (f.type === "int") input.step = "1";
      if (f.type === "list") input.placeholder = "comma-separated";
      if (f.default !== null && f.default !== undefined) {
        input.value = Array.isArray(f.default) ? f.default.join(", ") : String(f.default);
      }
    }
    input.dataset.name = f.name;
    input.dataset.type = f.type;
    input.title = f.help || "";
    if (f.type === "bool") { label.appendChild(input); label.append(` ${f.label || f.name}`); }
    else { label.textContent = f.label || f.name; label.appendChild(input); }
    box.appendChild(label);
  }
}

// The values a model read off the sentence, put into the inputs so the
// reader sees them and can change them before anything runs.
function writeIntake(values) {
  for (const el of $("solve-intake").querySelectorAll("[data-name]")) {
    const v = (values || {})[el.dataset.name];
    if (v === undefined || v === null) continue;
    if (el.dataset.type === "bool") el.checked = Boolean(v);
    else el.value = String(v);
  }
}

function readIntake() {
  const out = {};
  for (const el of $("solve-intake").querySelectorAll("[data-name]")) {
    const name = el.dataset.name, type = el.dataset.type;
    if (type === "bool") out[name] = el.checked;
    else if (type === "int" || type === "float") {
      const v = type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
      if (Number.isFinite(v)) out[name] = v;
    } else if (type === "list") {
      // "1, 3, 12" -> [1, 3, 12]; Python's fill_intake coerces the same way.
      const items = el.value.split(/[,;]/).map((s) => s.trim()).filter(Boolean)
        .map((s) => (Number.isFinite(Number(s)) ? Number(s) : s));
      if (items.length) out[name] = items;
    } else if (el.value !== "") out[name] = el.value;
  }
  return out;
}

// "Flood risk at a site · return period 100 · design flow": the values, in order.
function whatSummary(pb, text) {
  if (!pb) return text.length > 60 ? `${text.slice(0, 57)}...` : text;
  const bits = [pb.title];
  const intake = readIntake();
  for (const f of pb.intake || []) {
    const v = intake[f.name];
    if (v === undefined || v === "") continue;
    const short = String(f.label || f.name).replace(/\s*\(.*\)$/, "").toLowerCase();
    if (f.type === "bool") { if (v) bits.push(short); }
    else if (f.type === "int" || f.type === "float") bits.push(`${short} ${v}`);
    else if (Array.isArray(v)) bits.push(`${short} ${v.join("/")}`);
    else bits.push(String(v));
  }
  return bits.join(" · ");
}

// The key Ask holds, if any, offered as one line: unticked or absent, the run is keyless.
function renderModelRow() {
  const cfg = askModelConfig();
  const row = $("solve-model-row");
  row.hidden = !cfg;
  if (cfg) $("solve-model-label").textContent = `use ${cfg.label} for the prose`;
}

function modelForRun() {
  const cfg = $("solve-use-key").checked ? askModelConfig() : null;
  return cfg ? { provider: cfg.provider, model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url } : {};
}

// ── the sentence, read on the device ────────────────────────────────────────

const INTAKE_TIMEOUT_MS = 25000;

// A sentence with no chip chosen and no key: one small call to the model on
// this device, then the package's own rules on what it wrote (coerce_intake in
// the worker). Returns { playbook, intake, note } when a playbook was read,
// { note } otherwise; the note says which path read the words.
async function readOnDevice(text) {
  if (!(await localModelReady())) {
    return { note: "No on-device model in this browser; the keyword rules read your words." };
  }
  let reply;
  try {
    reply = await generateJsonLocally({
      system: intakePrompt(S.playbooks), prompt: text, schema: intakeSchema(S.playbooks),
      temperature: 0.1, timeoutMs: INTAKE_TIMEOUT_MS, onProgress: (m) => whatStatus(m, "info"),
    });
  } catch (err) {
    console.info("on-device intake unavailable:", err && err.message);
    return { note: "The on-device model could not read your words; the keyword rules did." };
  }
  const parsed = parseIntakeReply(reply, S.playbooks);
  if (!parsed) return { note: "The on-device model found no playbook in your words; the keyword rules read them." };
  whatStatus(state.workerReady ? "Checking the fields..." : "Loading Python in your browser (about 15 MB, once)...", "info");
  let safe;
  try {
    safe = await call("coerce_intake", parsed);
  } catch (err) {
    console.info("coerce_intake failed:", err && err.message);
    return { note: "The on-device model's reading could not be checked; the keyword rules read your words." };
  }
  if (!safe || !safe.playbook) return { note: "The on-device model named no playbook; the keyword rules read your words." };
  return { playbook: safe.playbook, intake: safe.intake || {},
           note: `Your words were read on your device by ${localReaderLabel() || "the on-device model"}.` };
}

// ── recon and plan ──────────────────────────────────────────────────────────

const whatStatus = (t, k = "warn") => setStatusEl($("solve-what-status"), t, k);
const planStatus = (t, k = "info") => setStatusEl($("solve-plan-status"), t, k);
const runStatus = (t, k = "info") => setStatusEl($("solve-run-status"), t, k);

// "4 defensible · 1 marginal": the zero counts say nothing, so they are left out.
function reconSummary(recon) {
  if (!recon) return "not assessed";
  const n = { defensible: 0, marginal: 0, not_defensible: 0 };
  for (const r of recon.sufficiency || []) n[r.status] = (n[r.status] || 0) + 1;
  const bits = [[n.defensible, "defensible"], [n.marginal, "marginal"], [n.not_defensible, "not defensible"]]
    .filter(([k]) => k > 0).map(([k, w]) => `${k} ${w}`);
  return bits.length ? bits.join(" · ") : "no method assessed";
}

async function plan() {
  const w = where();
  if (!w) { whatStatus("Click a gauge or a spot on the map first."); return; }
  let pb = chosen();
  const text = $("solve-text").value.trim();
  if (!pb && !text) { whatStatus("Pick a problem, or say it in your words."); return; }
  whatStatus("");
  const my = ++S.run;
  S.whereKey = w.key;
  S.recon = null; S.plan = null; S.study = null; S.result = null; S.editing = false;
  S.intakeNote = null;

  // 2. a sentence and no chip: read it here when a model is on the device
  // (with a key the Python Coordinator may read it instead).
  if (!pb && text && !modelForRun().provider) {
    whatStatus("Reading your words on your device...", "info");
    const read = await readOnDevice(text);
    if (my !== S.run) return;
    whatStatus("");
    S.intakeNote = read.note;
    if (read.playbook) {
      chooseChip(read.playbook);
      writeIntake(read.intake);
      pb = chosen();
    }
  }
  setStage("where", "done", w.text);
  setStage("what", "done", whatSummary(pb, text));
  openStage("recon");

  // 3. the card the inspector shows, narrowed to this problem; the worker gets
  // the catchment area and donor count only the page can read.
  const recon = await requestAssess({ lat: w.lat, lon: w.lon, target: "solve", key: state.selected ? w.key : null,
                                      problem: pb ? pb.problem : null });
  if (my !== S.run) return;
  S.recon = recon;
  setStage("recon", "done", reconSummary(recon));

  // 4. the plan, from the tree (and the Coordinator when there is a model).
  openStage("plan");
  $("solve-declined").hidden = true;
  $("solve-plan-head").innerHTML = "";
  $("solve-steps").innerHTML = "";
  $("solve-run").hidden = true;
  $("solve-edit").hidden = true;
  planStatus(state.workerReady ? "Planning..." : "Loading Python in your browser (about 15 MB, once)...");
  try {
    await ensureCatalogInWorker();
    const job = callCancelable("solve_plan", {
      problem: text, lat: w.lat, lon: w.lon, playbook: pb ? pb.id : null, intake: pb ? readIntake() : null,
      recon, ...modelForRun(),
    });
    S.cancel = job.cancel;
    const res = await job.promise;
    if (my !== S.run) return;
    S.plan = res;
    S.study = res.study;
    // The keyword rules read the sentence in Python: show their reading the
    // way the device's is shown, the chip pressed and the fields filled, so
    // reopening What starts from what was planned rather than from a blank.
    const picked = res.study && res.study.plan && res.study.plan.playbook;
    if (!pb && picked && !res.declined && S.playbooks.some((p) => p.id === picked)) {
      chooseChip(picked);
      writeIntake((res.study.problem || {}).params || {});
      setStage("what", "done", whatSummary(chosen(), text));
    }
    renderPlan(res);
    if (S.intakeNote) planStatus(S.intakeNote, "info");
  } catch (err) {
    if (my !== S.run || err instanceof Cancelled) return;
    planStatus(`Could not plan: ${err.message}`, "error");
  } finally {
    if (my === S.run) S.cancel = null;
  }
}

function declineSentence(res) {
  const kind = (res.study && res.study.plan && res.study.plan.kind) || "";
  if (kind === "no_playbook") return "Those words did not name a problem the playbooks cover. Pick one above.";
  return res.declined_reason || "The playbook declines this problem here.";
}

function firstSentence(t) {
  const s = String(t || "").trim();
  const m = s.match(/^.*?[.!?](?=\s|$)/);
  return m ? m[0] : s;
}

const fmtArg = (v) => (typeof v === "string" ? v : JSON.stringify(v));
// A UUID station id three times over is most of a checklist; the first
// characters tell two apart, the full arguments sit in the title and in Edit.
const shortArg = (v) => { const s = fmtArg(v); return s.length > 24 ? `${s.slice(0, 12)}…` : s; };

function gateChip(g, outcome) {
  const label = `${g.check}${g.value !== undefined && g.value !== null ? ` ${fmtArg(g.value)}` : ""}`;
  const cls = outcome ? (outcome.passed ? " ok" : " fail") : "";
  const title = outcome ? `${g.path || ""}: ${outcome.detail || ""}`.trim() : (g.path || g.paths || "").toString();
  return `<span class="gate${cls}" title="${escapeHtml(title)}">${escapeHtml(label)}${outcome ? (outcome.passed ? " ✓" : " ✗") : ""}</span>`;
}

// One checklist entry; `rec` (the runner's entry for the step) adds the outcome.
function stepHtml(s, rec, { editing = false } = {}) {
  const full = Object.entries(s.arguments || {}).map(([k, v]) => `${k}=${fmtArg(v)}`).join(", ");
  const args = Object.entries(s.arguments || {}).map(([k, v]) => `${k}=${shortArg(v)}`).join(", ");
  const gates = (s.expects || []).map((g, i) => gateChip(g, rec && rec.gates ? rec.gates[i] : null)).join("");
  const cls = rec ? (rec.ok && (rec.gates || []).every((g) => g.passed) ? "ok" : "fail") : "";
  let more = "";
  if (rec && rec.fallback_used && rec.fallback) {
    const fb = rec.fallback;
    const fgates = (fb.gates || []).map((g) => gateChip(g, g)).join("");
    more += `<div class="step-more">Fallback ${escapeHtml(toolLabel(fb.tool))}: ${fb.ok ? "ran" : "failed"}${fgates ? ` ${fgates}` : ""}</div>`;
  }
  if (rec && !rec.ok && rec.error) more += `<div class="step-more">${escapeHtml(rec.error)}</div>`;
  const edit = editing
    ? `<textarea class="step-edit" rows="3" aria-label="Arguments of this step, as JSON">${escapeHtml(JSON.stringify(s.arguments || {}, null, 1))}</textarea>`
    : "";
  return `<li class="${cls}" data-step="${escapeHtml(s.id || "")}">` +
    `<div class="step-main"><span class="step-tool">${escapeHtml(toolLabel(s.tool))}</span> <span class="step-args" title="${escapeHtml(full)}">${escapeHtml(args)}</span></div>` +
    (s.rationale ? `<div class="step-why">${escapeHtml(firstSentence(s.rationale))}</div>` : "") +
    (gates ? `<div class="step-gates">${gates}</div>` : "") + more + edit + "</li>";
}

function renderPlan(res) {
  planStatus("");
  const study = res.study || {};
  const plan = study.plan || {};
  if (res.declined) {
    const d = $("solve-declined");
    d.textContent = declineSentence(res);
    d.hidden = false;
    setStage("plan", "open", "declined");
    return;
  }
  // The notes (what the reconnaissance assumed, what was dropped) are in the
  // report anyway; on screen they fold behind a count.
  const notes = [...(plan.recon_notes || []), ...(plan.notes || [])];
  $("solve-plan-head").innerHTML =
    `<div class="solve-branch">Branch <b>${escapeHtml(plan.branch || "")}</b><span class="muted"> · ${study.steps.length} step${study.steps.length === 1 ? "" : "s"}</span></div>` +
    (plan.rationale ? `<p class="solve-why muted" title="Show the whole rationale">${escapeHtml(plan.rationale)}</p>` : "") +
    (notes.length
      ? `<details class="solve-notes"><summary>${notes.length} note${notes.length === 1 ? "" : "s"}</summary>` +
        notes.map((n) => `<p class="solve-note muted">${escapeHtml(n)}</p>`).join("") + "</details>"
      : "");
  renderSteps();
  $("solve-run").hidden = false;
  $("solve-edit").hidden = false;
  $("solve-edit").textContent = "Edit";
}

function renderSteps() {
  $("solve-steps").innerHTML = (S.study.steps || []).map((s) => stepHtml(s, null, { editing: S.editing })).join("");
}

// Edit: every step's arguments as a small JSON box; Done (or Run) reads them back.
function toggleEdit() {
  if (!S.study) return;
  if (S.editing) { if (!applyEdits()) return; }
  S.editing = !S.editing;
  $("solve-edit").textContent = S.editing ? "Done" : "Edit";
  renderSteps();
}

function applyEdits() {
  const boxes = [...$("solve-steps").querySelectorAll(".step-edit")];
  for (let i = 0; i < boxes.length; i++) {
    try {
      const v = JSON.parse(boxes[i].value);
      if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("not an object");
      S.study.steps[i].arguments = v;
    } catch (err) {
      planStatus(`Step ${i + 1}: the arguments are not a JSON object (${err.message}).`, "warn");
      boxes[i].focus();
      return false;
    }
  }
  planStatus("");
  return true;
}

// ── run ─────────────────────────────────────────────────────────────────────

function planSummary(study) {
  const plan = (study && study.plan) || {};
  const n = (study && study.steps || []).length;
  return `branch ${plan.branch || "?"} · ${n} step${n === 1 ? "" : "s"}`;
}

function runSummary(res, secs) {
  const gates = res.gates || [];
  const passed = gates.filter((g) => g.passed).length;
  const run = res.run || {};
  const bits = [`${passed} of ${gates.length} gates passed`];
  if (run.stop_reason) bits.push(`stopped at ${run.stopped_at}`);
  bits.push(`${secs} s`);
  return bits.join(" · ");
}

async function runPlan() {
  if (!S.study || state.solve.running) return;
  if (S.editing) { if (!applyEdits()) return; S.editing = false; $("solve-edit").textContent = "Edit"; }
  const my = ++S.run;
  S.result = null;
  setStage("plan", "done", planSummary(S.study));
  openStage("run");
  $("solve-timeline").innerHTML = "";
  runStatus(state.workerReady ? "Reading the catchment..." : "Loading Python in your browser (about 15 MB, once)...");
  $("solve-stop").hidden = false;
  S.startedAt = Date.now();
  state.solve.running = true;
  try {
    await ensureCatalogInWorker();
    // The worker cannot read BasinATLAS; the page hands it the sub-basin it found.
    const site = (S.study.problem && S.study.problem.site) || {};
    const catchment = Number.isFinite(Number(site.lat)) && Number.isFinite(Number(site.lon))
      ? await catchmentForWorker(Number(site.lat), Number(site.lon)).catch(() => null)
      : null;
    if (my !== S.run) return;
    runStatus("");
    const job = callCancelable("solve_run", { study: S.study, recon: S.recon, catchment, ...modelForRun() });
    S.cancel = job.cancel;
    S.jobId = job.id;
    const res = await job.promise;
    if (my !== S.run) return;
    runStatus("");
    S.result = res;
    setStage("run", "done", runSummary(res, Math.round((Date.now() - S.startedAt) / 1000)));
    renderResult(res);
    openStage("result");
    const out = $("solve-answer");
    try { out.scrollIntoView({ block: "nearest", behavior: "smooth" }); } catch { /* fine */ }
    out.focus({ preventScroll: true });
  } catch (err) {
    if (my !== S.run) return;
    if (err instanceof Cancelled) runStatus("Stopped.", "warn");
    else runStatus(`The run failed: ${err.message}`, "error");
  } finally {
    if (my === S.run) { S.cancel = null; S.jobId = null; }
    state.solve.running = false;
    $("solve-stop").hidden = true;
  }
}

function stopRun() {
  if (S.cancel) S.cancel();
}

// One line per timeline event: the role, the step, what happened.
function addEvent(e) {
  const tl = $("solve-timeline");
  const li = document.createElement("li");
  const detail = String(e.detail || "");
  if (e.event === "gate") li.classList.add(/FAILED/.test(detail) ? "fail" : "ok");
  if (["error", "stop", "declined", "skipped", "model_error"].includes(e.event)) li.classList.add("fail");
  if (["done", "reused"].includes(e.event)) li.classList.add("ok");
  let text = detail;
  if (e.event === "start" || e.event === "fallback") {
    const m = detail.match(/(\w+)\((.*)\)$/);
    if (m) text = `${e.event === "fallback" ? "fallback: " : ""}${toolLabel(m[1])} (${m[2]})`;
  }
  li.innerHTML = `<span class="tl-role">${escapeHtml(e.role || "")}</span>` +
    `<span class="tl-text" title="${escapeHtml(detail)}">${e.step ? `<span class="tl-step">${escapeHtml(e.step)}</span> ` : ""}${escapeHtml(text)}</span>`;
  if (e.event === "start" || e.event === "fallback" || e.event === "done" || e.event === "error") {
    for (const c of tl.querySelectorAll(".current")) c.classList.remove("current");
  }
  if (e.event === "start" || e.event === "fallback") li.classList.add("current");
  tl.appendChild(li);
  tl.scrollTop = tl.scrollHeight;
}

// ── result ──────────────────────────────────────────────────────────────────

function renderResult(res) {
  const study = res.study || {};
  $("solve-answer").innerHTML = mdToHtml(res.answer || (res.declined ? res.declined_reason : "No answer was produced."));

  // Always there, never folded: what the run did not establish.
  const items = res.not_established || [];
  const not = $("solve-not");
  not.className = `ask-checks ${items.length ? "warn" : "ok"}`;
  not.innerHTML = `<strong>What this answer does not establish</strong>` + (items.length
    ? `<ul>${items.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>`
    : `<div>Nothing beyond the caveats: every gate and every check passed.</div>`);

  const caveats = res.caveats || [];
  $("solve-caveats").innerHTML = caveats.length
    ? `<div class="solve-h">Caveats</div><ul>${caveats.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>` : "";

  const results = study.results || {};
  $("solve-outcomes").innerHTML = (study.steps || []).map((s) => stepHtml(s, results[s.id] || null)).join("");

  const data = (res.data_used || []).map((d) => {
    const bits = [`<b>${escapeHtml(d.label || "")}</b>`];
    if (d.period) bits.push(escapeHtml(String(d.period)));
    if (d.license) bits.push(`licence ${escapeHtml(d.license)}`);
    if (d.attribution) bits.push(escapeHtml(String(d.attribution)));
    return `<li>${bits.join(" · ")}</li>`;
  });
  const methods = [
    ...(res.methods || []).map((m) => `<li><b>${escapeHtml(m.name)}.</b> ${escapeHtml(m.text || "")} <span class="muted">${escapeHtml(m.citation || "")}</span></li>`),
    ...(res.citations || []).map((c) => `<li>${escapeHtml(c)}</li>`),
  ];
  $("solve-data-body").innerHTML =
    (data.length ? `<div class="solve-h">Data</div><ul>${data.join("")}</ul>` : "") +
    (methods.length ? `<div class="solve-h">Methods and citations</div><ol>${methods.join("")}</ol>` : "") ||
    `<span class="muted">No tool result to cite.</span>`;
  $("solve-data").open = false;

  const calls = Object.values(res.cost || {}).reduce((n, c) => n + (c.calls || 0), 0);
  $("solve-foot").textContent = res.model
    ? `${res.model} via ${res.provider}, ${calls} model call${calls === 1 ? "" : "s"}. Every number came through a gate.`
    : "No model: the playbook tree planned, a template wrote the prose. Every number came through a gate.";
}

// ── open, init ──────────────────────────────────────────────────────────────

function refresh() {
  renderModelRow();
  const w = where();
  // A different place than the plan was made for: start again from the intake.
  if (S.whereKey && (!w || w.key !== S.whereKey) && S.current !== "what") openIntake();
  else renderWhere();
}

export function openSolve(playbook = null) {
  openDrawer({ mode: "solve" });
  if (playbook) {
    if (S.playbooks.some((p) => p.id === playbook)) chooseChip(playbook);
    else S.pendingChip = playbook;
  }
  refresh();
  const first = $("solve-chips").querySelector('[aria-pressed="true"]') || $("solve-chips").querySelector(".chip");
  if (first && S.current === "what") first.focus({ preventScroll: true });
}

async function loadPlaybooks() {
  try {
    const res = await fetch(`./playbooks.json?v=${CONFIG.build}`);
    if (!res.ok) throw new Error(String(res.status));
    S.playbooks = (await res.json()).playbooks || [];
  } catch (err) {
    console.info("playbook list unavailable:", err && err.message);
    whatStatus("The playbook list did not load; say the problem in your words.");
    return;
  }
  renderChips();
  if (S.pendingChip) { const id = S.pendingChip; S.pendingChip = null; chooseChip(id); }
}

export async function initSolve() {
  // Wired before anything is awaited, like the Ask button (#271).
  $("btn-solve").addEventListener("click", () => {
    if (drawerOpen() && drawerMode() === "solve") closeDrawer(); else openSolve();
  });
  for (const name of STAGES) stageEl(name).querySelector(".stage-head").addEventListener("click", () => onHeadClick(name));
  $("solve-plan-btn").addEventListener("click", plan);
  $("solve-text").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); plan(); } });
  $("solve-plan-head").addEventListener("click", (e) => {
    const why = e.target.closest(".solve-why");
    if (why) why.classList.toggle("open");
  });
  $("solve-run").addEventListener("click", runPlan);
  $("solve-edit").addEventListener("click", toggleEdit);
  $("solve-stop").addEventListener("click", stopRun);
  $("solve-study").addEventListener("click", () => { if (S.result) downloadBlob("study.yaml", S.result.study_yaml, "text/yaml"); });
  $("solve-copy").addEventListener("click", (e) => { if (S.result) copyText(S.result.report, e.currentTarget, "Copied!"); });
  $("solve-rerun").addEventListener("click", runPlan);
  $("drawer").addEventListener("drawermode", (e) => { if (e.detail.mode === "solve") refresh(); });
  onSolveProgress((event, id) => { if (id === S.jobId) addEvent(event); });
  actions.openSolve = openSolve;
  openIntake();
  await loadPlaybooks();
}

