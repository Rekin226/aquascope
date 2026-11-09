// Tools, Open model results (#591): your files, or a sample model. The chooser opens in the page's dialog: choose
// files from your computer (or drop them on the map), or try one of two samples, MODFLOW (FloPy's Freyberg model)
// and HEC-RAS 2D (a HydroShare run, from the Archive's copy). A sample is fetched by the page, its SHA-256 checked,
// and then read like a file you dropped: it stays in the browser, and its credit shows on the legend row.
// What the samples are, their sizes and the words: model-samples-core.js.

import { $, actions, escapeHtml } from "./core.js?v=__BUILD__";
import { closeModal, openModal } from "./shell.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { openModelFiles, pickFile, showModelStatus } from "./mesh-view.js?v=__BUILD__";
import { SAMPLES, hex, manifestFiles, sampleById, sampleButton, sizeWords, totalBytes } from "./model-samples-core.js?v=__BUILD__";

const fetched = new Map();      // sample id -> File[] already fetched in this page
const states = new Map();       // sample id -> { state, files, done }
let manifestRead = null;        // the HEC-RAS manifest, read once per page

function stateOf(s) {
  return states.get(s.id) || { state: s.files ? "ready" : "checking", files: s.files || null, done: 0 };
}

function setState(s, next) {
  states.set(s.id, { ...stateOf(s), ...next });
  renderSamples();
}

// The Archive's manifest for a mirrored sample: its files, or "missing" while it is not published (a 404).
function readManifest(s) {
  if (manifestRead) return manifestRead;
  manifestRead = (async () => {
    const resp = await fetch(s.manifest, { cache: "no-cache" });
    if (resp.status === 404) return { state: "missing", files: null };
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const files = manifestFiles(await resp.json());
    return files.length ? { state: "ready", files } : { state: "missing", files: null };
  })();
  manifestRead.catch(() => { manifestRead = null; });
  return manifestRead;
}

function sampleHtml(s) {
  const st = stateOf(s);
  const b = sampleButton(s, st.state, { files: st.files || s.files || [], done: st.done });
  return `<li class="ms-sample" data-id="${escapeHtml(s.id)}">` +
    `<button type="button" class="btn${b.disabled ? "" : " primary"}" data-sample="${escapeHtml(s.id)}"${b.disabled ? " disabled" : ""}>` +
    `${escapeHtml(b.label)}</button>` +
    `<p class="ms-note muted" aria-live="polite">${escapeHtml(b.note)}</p>` +
    `<p class="ms-about">${escapeHtml(s.about)}</p>` +
    `<p class="ms-credit muted">${escapeHtml(s.credit)} <a href="${escapeHtml(s.link)}" target="_blank" rel="noopener">Source</a></p>` +
    "</li>";
}

function renderSamples() {
  const list = document.querySelector("#modal-body .ms-samples");
  if (list) list.innerHTML = SAMPLES.map(sampleHtml).join("");
}

export function openModelChooser() {
  openModal("Your model results", `
    <div class="ms-chooser">
      <p>A HEC-RAS 2D plan result (<code>.p01.hdf</code>), a UGRID mesh (<code>.nc</code>), or MODFLOW heads
        (<code>.hds</code> with its <code>.dis</code> or <code>.dis.grb</code>, and the <code>.nam</code> that says
        where the model is). Drop the files on the map, or:</p>
      <p><button type="button" class="btn primary" data-act="pick">Choose files</button></p>
      <h3 class="ms-h">Or try a sample model</h3>
      <ul class="ms-samples"></ul>
      <p class="muted">A sample is fetched from its source into this page and read like a file you dropped. Nothing
        is uploaded.</p>
    </div>`);
  const body = $("modal-body");
  renderSamples();
  body.querySelector(".ms-chooser").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.act === "pick") { closeModal(); pickFile(); return; }
    if (b.dataset.sample) void openSample(b.dataset.sample);
  });
  for (const s of SAMPLES) {
    if (!s.manifest || stateOf(s).state === "ready" || stateOf(s).state === "loading") continue;
    setState(s, { state: "checking" });
    readManifest(s).then((m) => setState(s, m)).catch(() => setState(s, { state: "offline" }));
  }
}

// Fetch one file with its progress, and check its SHA-256 against the one recorded for it.
async function fetchFile(f, onBytes) {
  const resp = await fetch(f.url);
  if (!resp.ok) throw new Error(`${f.name}: HTTP ${resp.status}`);
  let buf;
  if (resp.body && resp.body.getReader) {
    const reader = resp.body.getReader();
    const parts = [];
    let n = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      n += value.length;
      onBytes(n);
    }
    buf = await new Blob(parts).arrayBuffer();
  } else {
    buf = await resp.arrayBuffer();
    onBytes(buf.byteLength);
  }
  if (globalThis.crypto && crypto.subtle) {
    const got = hex(await crypto.subtle.digest("SHA-256", buf));
    if (got !== f.sha256) throw new Error(`${f.name} is not the file recorded for this sample (its SHA-256 differs)`);
  }
  return new File([buf], f.name, { type: "application/octet-stream" });
}

export async function openSample(id) {
  const s = sampleById(id);
  if (!s) return;
  const sample = { id: s.id, credit: s.credit, link: s.link, licence: s.licence };
  if (fetched.has(id)) { closeModal(); openModelFiles(fetched.get(id), { sample }); return; }
  const files = stateOf(s).files || s.files;
  if (!files || !files.length) return;
  const total = totalBytes(files);
  const doneBefore = [];
  setState(s, { state: "loading", done: 0 });
  closeModal();
  showModelStatus({ text: `Fetching the sample (${sizeWords(total)})…`, busy: true });
  try {
    const out = [];
    for (const [k, f] of files.entries()) {
      out.push(await fetchFile(f, (n) => {
        const done = doneBefore.reduce((a, b) => a + b, 0) + n;
        states.set(s.id, { ...stateOf(s), done });
        showModelStatus({ text: `Fetching the sample: ${sizeWords(done)} of ${sizeWords(total)}`, busy: true, keep: true });
      }));
      doneBefore[k] = f.bytes;
    }
    fetched.set(id, out);
    states.set(s.id, { ...stateOf(s), state: "ready" });
    showModelStatus();
    announce(`${s.title}: fetched, opening it.`);
    openModelFiles(out, { sample });
  } catch (err) {
    states.set(s.id, { ...stateOf(s), state: "ready" });
    const msg = `Could not fetch the sample: ${err && err.message ? err.message : String(err)}`;
    showModelStatus({ text: msg, error: true });
    announce(msg);
  }
}

export function initModelSamples() {
  actions.openModelChooser = openModelChooser;
  actions.openSample = openSample;
}
