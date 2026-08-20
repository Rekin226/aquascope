// Ask ✨, the Analyst in the page. aquascope.ai_engine.analyst.ask runs inside
// the Pyodide worker: the model call goes from the worker straight to the
// provider (bring your own key), the tools run on the catalog held here plus
// the agencies' APIs, and the report ends with Data and Methods sections
// assembled from tool results.
//
// New here: the drawer sits beside the inspector instead of replacing it, the
// question carries what you are looking at, the run can be stopped, and the
// key defaults to this tab only.

import { CONFIG } from "../config.js?v=__BUILD__";
import {
  $, actions, copyText, downloadBlob, escapeHtml, sourceStyle, state, stationKey,
} from "./core.js?v=__BUILD__";
import { closeDrawer, drawerOpen, openDrawer, setStatusEl } from "./shell.js?v=__BUILD__";
import { Cancelled, callCancelable, call, onAskProgress } from "./worker-client.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { visibleLayerSummary } from "./layer-ui.js?v=__BUILD__";
import { initShowcase } from "./showcase.js?v=__BUILD__";
import { askLocally, describeLocal, localAnswerHtml, localModelPossible } from "./local-model.js?v=__BUILD__";

// Providers come from the package's registry (aquascope/ai_engine/providers.py),
// written to explorer/providers.json by `python -m aquascope.ai_engine.providers`.
// The page used to keep its own copy, which drifted from the Python one until a
// retired Groq model broke both. This small fallback only covers the case where
// the JSON did not load at all.
let ASK_PROVIDERS = {
  groq: { label: "Groq (free tier, fast)", base_url: "https://api.groq.com/openai/v1", model: "openai/gpt-oss-120b" },
  custom: { label: "Custom OpenAI-compatible endpoint", base_url: "", model: "" },
};

async function loadProviders() {
  try {
    const res = await fetch(`./providers.json?v=${CONFIG.build}`);
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    const next = {};
    for (const p of data.providers || []) {
      next[p.id] = { label: p.label, base_url: p.base_url || "", model: p.model || "", models: p.models || [], free: p.free, signup: p.signup, note: p.note };
    }
    if (Object.keys(next).length) ASK_PROVIDERS = next;
  } catch (err) {
    console.info("provider registry unavailable, using the built-in defaults:", err && err.message);
  }
}

const ASK_EXAMPLES = [
  "What is the 100-year flood of the Thames at Kingston, and how sure can we be?",
  "Compare the low flows (Q95) of the Seine at Paris and the Loire at Blois.",
  "Is the Potomac at Little Falls getting drier? Use the annual-mean trend.",
  "How wet is Taipei compared with London, and what is the aridity class of each?",
  "Which UK boreholes near Cambridge have the longest groundwater records?",
];

const ASK_STORE = "aquascope.ask.settings";
let cancelAsk = null;

function askSettings() {
  try {
    return JSON.parse(sessionStorage.getItem(ASK_STORE) || localStorage.getItem(ASK_STORE) || "{}");
  } catch {
    return {};
  }
}

// The key lives in this tab unless the user asks for it to be remembered, and
// "Forget" clears both stores.
function saveAskSettings() {
  const remember = $("ask-remember").checked;
  const data = { provider: $("ask-provider").value, model: $("ask-model").value, base_url: $("ask-base-url").value, remember };
  const withKey = { ...data, key: $("ask-key").value };
  try {
    sessionStorage.setItem(ASK_STORE, JSON.stringify(withKey));
    if (remember) localStorage.setItem(ASK_STORE, JSON.stringify(withKey));
    else localStorage.setItem(ASK_STORE, JSON.stringify(data));
  } catch { /* storage blocked: fine, the tab keeps the values */ }
  updateForgetButton();
}

function forgetKey() {
  try {
    sessionStorage.removeItem(ASK_STORE);
    localStorage.removeItem(ASK_STORE);
  } catch { /* ignore */ }
  $("ask-key").value = "";
  $("ask-remember").checked = false;
  updateForgetButton();
  askStatus("Key forgotten in this browser.", "info");
}

function updateForgetButton() {
  const has = Boolean($("ask-key").value);
  $("ask-forget").hidden = !has;
}

// What the user is looking at, in one line the model can act on. Shown in the
// drawer so it is never a hidden prompt.
export function currentContext() {
  const bits = [];
  if (state.selected) {
    const r = state.selected;
    bits.push(`the gauge ${r.name || r.station_id} (source ${r.source}, id ${r.station_id}, ${r.lat.toFixed(3)}, ${r.lon.toFixed(3)})`);
  } else if (state.point) {
    bits.push(`the point ${state.point.lat}, ${state.point.lon}`);
  }
  if (state.mapOk && map) {
    const c = map.getCenter();
    bits.push(`a map centred on ${c.lat.toFixed(2)}, ${c.lng.toFixed(2)} at zoom ${map.getZoom().toFixed(1)}`);
    const layers = visibleLayerSummary();
    if (layers) bits.push(`layers on the map: ${layers}`);
  }
  const hidden = [...state.hidden];
  if (hidden.length) bits.push(`sources hidden on the map: ${hidden.join(", ")}`);
  return bits.length ? `The user is looking at ${bits.join("; ")}.` : "";
}

function contextLine() {
  const ctx = currentContext();
  const el = $("ask-context-text");
  el.textContent = ctx || "nothing selected yet";
  $("ask-context").hidden = !ctx;
  return ctx;
}

export function openAsk() {
  openDrawer();
  const q = $("ask-question");
  contextLine();
  if (state.selected && !q.value.trim()) {
    const r = state.selected;
    q.value = `Summarise the record of ${r.name || r.station_id} (${r.source} / ${r.station_id}): period, mean, trend, and the flood frequency if the record allows it.`;
  }
  const chip = $("ask-this-station");
  if (chip) chip.remove();
  if (state.selected) {
    const r = state.selected;
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.id = "ask-this-station";
    b.textContent = `About ${r.name || r.station_id}`;
    b.addEventListener("click", () => {
      q.value = `Summarise the record of ${r.name || r.station_id} (${r.source} / ${r.station_id}): period, mean, trend, and the flood frequency if the record allows it.`;
      q.focus();
    });
    $("ask-examples").prepend(b);
  }
  q.focus();
}

function askStatus(text, kind = "info") {
  setStatusEl($("ask-status"), text, kind);
}

function askLog(text) {
  const log = $("ask-log");
  log.hidden = false;
  const li = document.createElement("li");
  const m = String(text).match(/^tool (\w+)\((.*)\)$/);
  if (m && m[1] === "run_python") {
    // Show the code it wrote: this is the step a reader most wants to see.
    const code = (m[2].match(/code='([\s\S]*)'$/) || [])[1] || m[2];
    li.innerHTML = `<code>run_python</code><pre class="ask-code">${escapeHtml(code.slice(0, 800))}</pre>`;
  } else {
    li.innerHTML = m ? `<code>${escapeHtml(m[1])}</code>(${escapeHtml(m[2]).slice(0, 160)})` : escapeHtml(text);
  }
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
}

async function ensureCatalogInWorker() {
  if (state.ask.catalogSent) return;
  const rows = state.stations.map((r) => ({
    source: r.source, station_id: r.station_id, name: r.name, latitude: r.lat, longitude: r.lon,
    variables: r.variables || [], period_start: r.period_start, period_end: r.period_end, url: r.url,
    agency: sourceStyle(r.source).label,
  }));
  await call("catalog", { rows });
  state.ask.catalogSent = true;
}

function currentTier() {
  const el = document.querySelector('input[name="ask-tier"]:checked');
  return el ? el.value : "key";
}

// The device tier: a small model here, a reduced tool set, tools run in the
// worker exactly as they do for the full loop.
async function runLocally(question) {
  const ctx = $("ask-use-context").checked ? contextLine() : "";
  state.ask.running = true;
  $("ask-run").disabled = true;
  $("ask-run").textContent = "Working…";
  $("ask-log").innerHTML = "";
  $("ask-log").hidden = false;
  $("ask-result").hidden = true;
  $("ask-checks").hidden = true;
  askStatus("Starting the model on your device…");
  try {
    const res = await askLocally(question, {
      callTool: (name, args) => call("tool", { name, arguments: args }),
      onEvent: (m) => (m.startsWith("tool ") ? askLog(m) : askStatus(m)),
      context: ctx,
    });
    askStatus("");
    state.ask.markdown = `# ${question}\n\n${res.answer}\n`;
    $("ask-result").innerHTML = localAnswerHtml(res);
    $("ask-result").hidden = false;
    $("ask-copy").hidden = false;
    $("ask-download").hidden = false;
    $("ask-study").hidden = true;
    $("ask-rerun").hidden = true;
  } catch (err) {
    askStatus(`The on-device model could not run: ${err.message}`, "error");
  } finally {
    state.ask.running = false;
    $("ask-run").disabled = false;
    $("ask-run").textContent = "Ask";
  }
}

async function runAsk() {
  if (state.ask.running) return;
  const question = $("ask-question").value.trim();
  if (currentTier() === "local") {
    if (!question) { askStatus("Type a question first.", "warn"); return; }
    await runLocally(question);
    return;
  }
  const provider = $("ask-provider").value;
  const chosen = ASK_PROVIDERS[provider];
  if (!chosen) {
    // The picker fills from providers.json a moment after load. Running before
    // that used to throw on an undefined provider with nothing on screen.
    askStatus("Still loading the provider list, try again in a second.", "warn");
    return;
  }
  const key = $("ask-key").value.trim();
  const model = $("ask-model").value.trim() || chosen.model;
  const base_url = provider === "custom" ? $("ask-base-url").value.trim() : chosen.base_url;
  if (!question) { askStatus("Type a question first.", "warn"); return; }
  if (!key && provider !== "custom") {
    askStatus("Paste an API key (Groq and Hugging Face give free ones).", "warn");
    $("ask-settings").open = true;
    return;
  }
  if (provider === "custom" && !base_url) { askStatus("A custom endpoint needs its base URL (ending in /v1).", "warn"); return; }
  saveAskSettings();

  const ctx = $("ask-use-context").checked ? contextLine() : "";
  const prompt = ctx ? `${question}\n\nContext: ${ctx}` : question;

  state.ask.running = true;
  state.ask.markdown = null;
  $("ask-run").disabled = true;
  $("ask-run").textContent = "Working…";
  $("ask-stop").hidden = false;
  $("ask-copy").hidden = true;
  $("ask-download").hidden = true;
  $("ask-result").hidden = true;
  $("ask-stations").hidden = true;
  $("ask-log").innerHTML = "";
  $("ask-log").hidden = true;
  askStatus(state.workerReady ? "Preparing…" : "Loading Python in your browser (about 15 MB, once)…");
  try {
    await ensureCatalogInWorker();
    askStatus(`Asking ${model} via ${provider}…`);
    const job = callCancelable("ask", {
      question: prompt,
      provider: provider === "custom" ? "custom" : provider,
      model,
      api_key: key || (provider === "custom" ? "none" : null),
      base_url,
      max_steps: 8,
    });
    cancelAsk = job.cancel;
    const res = await job.promise;
    askStatus("");
    state.ask.markdown = res.markdown;
    renderAsk(res);
  } catch (err) {
    if (err instanceof Cancelled) askStatus("Stopped. The model call was abandoned; nothing was charged after this point.", "warn");
    else askStatus(`The analyst could not finish: ${err.message}`, "error");
  } finally {
    cancelAsk = null;
    state.ask.running = false;
    $("ask-run").disabled = false;
    $("ask-run").textContent = "Ask";
    $("ask-stop").hidden = true;
  }
}

function renderAsk(res) {
  const out = $("ask-result");
  out.innerHTML = mdToHtml(res.markdown);
  out.hidden = false;
  $("ask-copy").hidden = false;
  $("ask-download").hidden = false;
  renderChecks(res.checks || [], res.verified);
  state.ask.study = res.study || "";
  $("ask-study").hidden = !state.ask.study;
  // chips for every station the tools touched: click to open it on the map
  const chips = [];
  for (const d of res.data_used || []) {
    const m = String(d.label || "").match(/^(\S+) \/ (.+)$/);
    if (!m) continue;
    const key = `${m[1]}/${m[2]}`;
    const r = state.byKey.get(key);
    if (!r) continue;
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.innerHTML = `<i style="background:${sourceStyle(r.source).color}"></i>${escapeHtml(r.name || r.station_id)}`;
    b.title = "Open this station on the map";
    b.addEventListener("click", () => actions.selectStation(stationKey(r), { fly: true }));
    chips.push(b);
  }
  const box = $("ask-stations");
  box.innerHTML = "";
  if (chips.length) {
    const l = document.createElement("span");
    l.className = "muted";
    l.textContent = "Stations used: ";
    box.appendChild(l);
    for (const c of chips) box.appendChild(c);
  }
  box.hidden = chips.length === 0;
}

// What the answer did and did not establish, from the deterministic checks in
// aquascope.ai_engine.verify (not a model grading a model).
function renderChecks(checks, verified) {
  const box = $("ask-checks");
  if (!checks.length) { box.hidden = true; return; }
  const unmet = checks.filter((c) => !c.passed);
  box.hidden = false;
  box.className = `ask-checks ${unmet.length ? "warn" : "ok"}`;
  box.innerHTML = unmet.length
    ? `<strong>${unmet.length} of ${checks.length} checks not met</strong><ul>` +
      unmet.map((c) => `<li>${escapeHtml(c.detail || c.name)}</li>`).join("") + "</ul>"
    : `<strong>All ${checks.length} checks passed</strong>: every number in the answer appears in a tool result, ` +
      `the record is named, and units and uncertainty are quoted.`;
}

// A small Markdown renderer for the analyst's reports (headings, lists,
// tables, code, emphasis, links). Input is escaped first; only our own tags
// come out.
export function mdToHtml(md) {
  const lines = String(md || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let i = 0, list = null, para = [];
  const inline = (t) => escapeHtml(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const flushPara = () => { if (para.length) { html.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flushPara(); closeList();
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      html.push(`<pre><code>${escapeHtml(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); closeList(); html.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushPara(); closeList(); html.push("<hr>"); i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(); closeList();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      html.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        `${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/), ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { closeList(); html.push(`<${kind}>`); list = kind; }
      html.push(`<li>${inline((ul || ol)[1])}</li>`); i++;
      continue;
    }
    if (!line.trim()) { flushPara(); closeList(); i++; continue; }
    para.push(line.trim()); i++;
  }
  flushPara(); closeList();
  return html.join("\n").replace(/<p>(Produced by aquascope[^<]*)<\/p>/, '<p class="foot">$1</p>');
}

export async function initAsk() {
  // Wire the button that opens the drawer before anything is awaited. This used
  // to sit after `await loadProviders()`, so between load and that fetch coming
  // back, clicking Ask did nothing at all and said nothing about why. The
  // picker below fills in a moment later; the drawer does not need it to open.
  $("btn-ask").addEventListener("click", () => { if (drawerOpen()) closeDrawer(); else openAsk(); });

  const ex = $("ask-examples");
  for (const q of ASK_EXAMPLES) {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => { $("ask-question").value = q; $("ask-question").focus(); });
    ex.appendChild(b);
  }

  $("ask-run").addEventListener("click", runAsk);
  $("ask-stop").addEventListener("click", () => { if (cancelAsk) cancelAsk(); });
  $("ask-question").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAsk(); });
  $("ask-use-context").addEventListener("change", contextLine);
  $("ask-copy").addEventListener("click", (e) => { if (state.ask.markdown) copyText(state.ask.markdown, e.currentTarget, "Copied!"); });
  $("ask-download").addEventListener("click", () => {
    if (state.ask.markdown) downloadBlob("aquascope-answer.md", state.ask.markdown, "text/markdown");
  });
  $("ask-study").addEventListener("click", () => {
    if (state.ask.study) downloadBlob("study.yaml", state.ask.study, "text/yaml");
  });
  onAskProgress(askLog);
  initShowcase();

  // The device tier is offered only where it can actually run.
  const possible = localModelPossible();
  $("ask-local-note").textContent = describeLocal();
  const localRadio = document.querySelector('input[name="ask-tier"][value="local"]');
  localRadio.disabled = !possible;
  for (const r of document.querySelectorAll('input[name="ask-tier"]')) {
    r.addEventListener("change", () => {
      const local = currentTier() === "local";
      $("ask-settings").hidden = local;
      askStatus(local
        ? "Nothing leaves this tab: the model runs here, and so do the tools."
        : "");
    });
  }
  actions.openAsk = openAsk;

  // Last, because it is the only part that waits on the network: the provider
  // list. Everything above works without it.
  await loadProviders();
  const provider = $("ask-provider"), model = $("ask-model"), baseRow = $("ask-base-url-row"), base = $("ask-base-url");
  provider.innerHTML = Object.entries(ASK_PROVIDERS)
    .map(([id, p]) => `<option value="${id}">${escapeHtml(p.label || id)}</option>`).join("");
  const saved = askSettings();
  if (saved.provider && ASK_PROVIDERS[saved.provider]) provider.value = saved.provider;
  const applyProvider = (keepModel) => {
    const p = ASK_PROVIDERS[provider.value];
    baseRow.hidden = provider.value !== "custom";
    if (!keepModel) model.value = p.model;
    if (provider.value !== "custom") base.value = p.base_url;
    model.placeholder = p.model || "model id";
    const note = $("ask-provider-note");
    const bits = [p.free, p.note].filter(Boolean);
    note.textContent = bits.join(" ");
    note.hidden = !bits.length;
  };
  applyProvider(Boolean(saved.model));
  if (saved.model) model.value = saved.model;
  if (saved.base_url && provider.value === "custom") base.value = saved.base_url;
  if (saved.key) $("ask-key").value = saved.key;
  $("ask-remember").checked = Boolean(saved.remember);
  $("ask-settings").open = !saved.key;
  updateForgetButton();
  provider.addEventListener("change", () => applyProvider(false));
  for (const el of [provider, model, base, $("ask-key"), $("ask-remember")]) el.addEventListener("change", saveAskSettings);
  $("ask-forget").addEventListener("click", forgetKey);
}
