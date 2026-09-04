// Worked examples: what the Analyst does, without a key.
//
// CI records these once with the maintainer's key (aquascope.showcase). The
// page replays the trace: the question, every tool call, the answer with its
// Data and Methods sections, and the checks. The prose is a recording and the
// page says so; the tools are not, and "run the tools again" re-runs the
// deterministic half live in this browser, with no key at all.

import { $, escapeHtml, state } from "./core.js?v=__BUILD__";
import { CONFIG } from "../config.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { setStatusEl } from "./shell.js?v=__BUILD__";

let index = null;
const cache = new Map();

export async function loadIndex() {
  if (index) return index;
  try {
    const res = await fetch(`./showcase/index.json?v=${CONFIG.build}`);
    if (!res.ok) throw new Error(String(res.status));
    index = await res.json();
  } catch (err) {
    console.info("no recorded examples published yet:", err && err.message);
    index = { examples: [] };
  }
  return index;
}

async function loadEntry(id) {
  if (cache.has(id)) return cache.get(id);
  const res = await fetch(`./showcase/${id}.json?v=${CONFIG.build}`);
  if (!res.ok) throw new Error(`example ${id}: ${res.status}`);
  const entry = await res.json();
  cache.set(id, entry);
  return entry;
}

export async function renderList() {
  const box = $("ask-showcase");
  const data = await loadIndex();
  if (!data.examples || !data.examples.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML =
    `<div class="showcase-head" title="Recorded ${escapeHtml((data.generated || "").slice(0, 10))}">` +
    `${data.examples.length} worked examples</div>` +
    `<div class="showcase-list"></div>`;
  const list = box.querySelector(".showcase-list");
  for (const ex of data.examples) {
    const b = document.createElement("button");
    b.className = "chip showcase-chip";
    b.type = "button";
    b.title = ex.shows || "";
    b.textContent = ex.question;
    b.addEventListener("click", () => open(ex.id));
    list.appendChild(b);
  }
  // The tier decides whether this box is on screen, and it could not know
  // whether there were any examples until now.
  const checked = document.querySelector('input[name="ask-tier"]:checked');
  if (checked) checked.dispatchEvent(new Event("change", { bubbles: true }));
}

function toolLine(call_) {
  const args = Object.entries(call_.arguments || {})
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  if (call_.name === "run_python") {
    const code = (call_.arguments || {}).code || "";
    return `<li><code>run_python</code><pre class="ask-code">${escapeHtml(code.slice(0, 800))}</pre></li>`;
  }
  return `<li><code>${escapeHtml(call_.name)}</code>(${escapeHtml(args).slice(0, 200)})` +
    `${call_.ok ? "" : ' <span class="muted">failed</span>'}</li>`;
}

export async function open(id) {
  const status = $("ask-status");
  setStatusEl(status, "Loading the recorded answer…");
  try {
    const entry = await loadEntry(id);
    setStatusEl(status, "");
    state.ask.showcase = entry;
    state.ask.markdown = entry.markdown || "";
    state.ask.study = entry.study || "";
    $("ask-question").value = entry.question;
    $("ask-log").hidden = false;
    $("ask-log").innerHTML = (entry.tool_calls || []).map(toolLine).join("");
    const out = $("ask-result");
    const { mdToHtml } = await import("./ask.js?v=__BUILD__");
    out.innerHTML =
      `<div class="showcase-note">A recorded answer. ${escapeHtml(entry.model || "a model")} wrote the prose once, ` +
      `on ${escapeHtml((entry.recorded || "").slice(0, 10))}; the numbers came from the tool calls above, which you ` +
      `can run again here, live, with no key.</div>` + mdToHtml(entry.markdown || entry.answer || "");
    out.hidden = false;
    $("ask-copy").hidden = false;
    $("ask-download").hidden = false;
    $("ask-study").hidden = !entry.study;
    $("ask-rerun").hidden = false;
    renderChecksFromEntry(entry);
    // Without this the replay lands below the fold and the click reads as a
    // no-op (#271).
    const { revealResult } = await import("./ask.js?v=__BUILD__");
    revealResult();
  } catch (err) {
    setStatusEl(status, `Could not load that example: ${err.message}`, "error");
  }
}

function renderChecksFromEntry(entry) {
  const box = $("ask-checks");
  const checks = entry.checks || [];
  if (!checks.length) { box.hidden = true; return; }
  const unmet = checks.filter((c) => !c.passed);
  box.hidden = false;
  box.className = `ask-checks ${unmet.length ? "warn" : "ok"}`;
  box.innerHTML = unmet.length
    ? `<strong>${unmet.length} of ${checks.length} checks not met</strong><ul>` +
      unmet.map((c) => `<li>${escapeHtml(c.detail || c.name)}</li>`).join("") + "</ul>"
    : `<strong>All ${checks.length} checks passed</strong> when this was recorded.`;
}

// Re-run the tool calls of the open example in this browser. No model, no key:
// this is the half that is deterministic, and it is most of the answer.
export async function rerun() {
  const entry = state.ask.showcase;
  if (!entry) return;
  const btn = $("ask-rerun");
  const log = $("ask-log");
  btn.disabled = true;
  btn.textContent = "Running the tools…";
  const calls = (entry.tool_calls || []).filter((c) => c.ok && c.name !== "run_python");
  log.innerHTML = "";
  let same = 0;
  try {
    for (const c of calls) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${escapeHtml(c.name)}</code> running…`;
      log.appendChild(li);
      try {
        const res = await call("tool", { name: c.name, arguments: c.arguments });
        const now = JSON.stringify(res).slice(0, 160);
        const matched = now === (c.summary || "").slice(0, 160);
        if (matched) same += 1;
        li.innerHTML = `<code>${escapeHtml(c.name)}</code> ` +
          `<span class="${matched ? "muted" : "changed"}">${matched ? "same as recorded" : "ran, result differs from the recording"}</span>`;
      } catch (err) {
        li.innerHTML = `<code>${escapeHtml(c.name)}</code> <span class="changed">${escapeHtml(err.message)}</span>`;
      }
    }
    setStatusEl($("ask-status"),
      `Ran ${calls.length} tool call(s) in this browser: ${same} returned exactly what was recorded. ` +
      "Records grow, so a difference usually means new observations, not an error.", "info");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run the tools again";
  }
}

export function initShowcase() {
  $("ask-rerun").addEventListener("click", rerun);
  renderList().catch((err) => console.info("showcase list unavailable:", err && err.message));
}
