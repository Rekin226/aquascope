// Steerable steps on the Study board: the plan's gates in plain words (the
// raw gates behind "details"), and after the report, small controls for the
// parameters a step declares steerable (aquascope.studio.steering). Both come
// from the worker with every reply as `plain`; this module only draws them
// and hands a change back. The engine validates the change, reruns the step
// and its dependants, and records it in study.yaml. Pure HTML builders plus
// one listener; node-importable (no imports, no DOM at load).

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const same = (a, b) => JSON.stringify(a || []) === JSON.stringify(b || []);

let PLAIN = null; // the worker's plain_plan for the study on the board: { steps: [{ id, tool, checks, gates, controls }] }

/** Keep the worker's plain plan from a reply (null clears it). */
export function rememberPlain(plain) {
  PLAIN = plain && Array.isArray(plain.steps) ? plain : null;
}

// The plain row for a step, only when it describes this very step (same id, tool and gates): a device
// proposal or a resumed study that the worker has not described falls back to the raw gates.
function rowFor(step) {
  if (!PLAIN || !step) return null;
  const row = PLAIN.steps.find((r) => r.id === step.id && r.tool === step.tool);
  return row && same(row.gates, step.expects) ? row : null;
}

/**
 * The checks under a plan step: one plain sentence per gate, the raw gate
 * chips behind a "details" toggle. `chipsHtml` is the page's own chips; they
 * are shown as they are when the worker gave no sentences.
 */
export function stepChecksHtml(step, chipsHtml) {
  const row = rowFor(step);
  const checks = row ? (row.checks || []).filter(Boolean) : [];
  if (!checks.length) return chipsHtml ? `<div class="step-gates">${chipsHtml}</div>` : "";
  return `<ul class="step-checks">${checks.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` +
    (chipsHtml ? `<details class="step-raw"><summary>details</summary><div class="step-gates">${chipsHtml}</div></details>` : "");
}

function inputHtml(stepId, c) {
  const id = `steer-${esc(stepId)}-${esc(c.param)}`;
  const cur = c.value === undefined ? null : c.value;
  const data = `data-param="${esc(c.param)}" data-type="${esc(c.type)}" data-current="${esc(JSON.stringify(cur))}"`;
  let field;
  if (c.type === "boolean") {
    field = `<input type="checkbox" id="${id}" ${data}${cur ? " checked" : ""}>`;
  } else if (c.type === "choice") {
    const opts = (c.choices || []).map((v) =>
      `<option value="${esc(v)}"${String(v) === String(cur) ? " selected" : ""}>${esc(v)}</option>`).join("");
    field = `<select id="${id}" ${data}>${opts}</select>`;
  } else {
    const bounds = (c.min !== undefined ? ` min="${esc(c.min)}"` : "") + (c.max !== undefined ? ` max="${esc(c.max)}"` : "");
    const step = c.type === "integer" ? "1" : "any";
    const hint = c.optional ? ` placeholder="full record"` : "";
    field = `<input type="number" id="${id}" ${data} step="${step}"${bounds}${hint} value="${cur === null ? "" : esc(cur)}">`;
  }
  return `<label class="steer-field" for="${id}" title="${esc(c.help || "")}">${esc(c.label)}${field}</label>`;
}

/**
 * The controls after the report: one small form per step that has steerable
 * parameters, each with a Rerun button. `study` is the workspace's study (the
 * controls are drawn only while the worker's description matches it), `label`
 * turns a tool id into words, `busy` disables the buttons during a run.
 */
export function steerHtml(study, { label = (t) => t, busy = false } = {}) {
  const steps = ((study || {}).steps || []);
  const forms = steps.map((s) => {
    const row = rowFor(s);
    const controls = row ? (row.controls || []).filter((c) => !c.locked) : [];
    if (!controls.length) return "";
    return `<fieldset class="steer-step" data-steer-step="${esc(s.id)}">` +
      `<legend>${esc(s.id)} · ${esc(label(s.tool))}</legend>` +
      `<div class="steer-fields">${controls.map((c) => inputHtml(s.id, c)).join("")}</div>` +
      `<button type="button" class="btn" data-steer-run="${esc(s.id)}"${busy ? " disabled" : ""}>Rerun this step</button>` +
      `</fieldset>`;
  }).filter(Boolean);
  if (!forms.length) return "";
  const n = ((PLAIN && PLAIN.steering) || []).length;
  return `<details class="study-steer"><summary>Adjust a step${n ? ` (${n} change${n === 1 ? "" : "s"} so far)` : ""}</summary>` +
    `<p class="muted steer-note">A change reruns that step and the steps that use it, then rewrites the report. It is recorded in study.yaml.</p>` +
    forms.join("") + `</details>`;
}

// One field's value in the control's type; undefined when it is unchanged.
function valueOf(el) {
  const type = el.dataset.type;
  let current = null;
  try { current = JSON.parse(el.dataset.current); } catch { current = null; }
  let value;
  if (type === "boolean") value = el.checked;
  else if (type === "choice") value = typeof current === "number" || /^-?\d+(\.\d+)?$/.test(el.value) ? Number(el.value) : el.value;
  else value = el.value.trim() === "" ? null : Number(el.value);
  if (value === current || (value !== null && current !== null && String(value) === String(current))) return undefined;
  return value;
}

/** The changes in one step's form: {param: value} for the fields that differ from the current values. */
export function readChanges(fieldset) {
  const changes = {};
  for (const el of fieldset.querySelectorAll("[data-param]")) {
    const v = valueOf(el);
    if (v !== undefined) changes[el.dataset.param] = v;
  }
  return changes;
}

/**
 * Wire the Rerun buttons on the board: `run({ step_id, changes })` is the
 * page's call to the worker. A form with nothing changed says so and sends nothing.
 */
export function initStudyControls(boardEl, run) {
  boardEl.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-steer-run]");
    if (!btn) return;
    const form = btn.closest("[data-steer-step]");
    if (!form) return;
    const changes = readChanges(form);
    if (!Object.keys(changes).length) {
      btn.textContent = "Nothing changed";
      setTimeout(() => { btn.textContent = "Rerun this step"; }, 1600);
      return;
    }
    run({ step_id: btn.dataset.steerRun, changes });
  });
}
