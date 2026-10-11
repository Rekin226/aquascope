// Review my model results (#564): "Review these results" on the model's legend row, or "review my model" in Ask
// the map. The checks are aquascope.mesh_review's, run in the mesh worker beside the dropped file (which never
// leaves the browser): the deepest water, negative and film depths, isolated wet cells, wet and dry flicker, the
// model's own volume error, the modelled extent against floods seen by radar and in the news, and the modelled
// peak against the Archive's gauges inside the mesh.
//
// What lies outside the file comes from the workers: the radar and news flood events around the mesh from the
// Archive's context mirror (a light worker), each gauge's record over the run (the main worker). The findings
// drop as numbered pins, coloured by severity, each with its reason, its numbers and its source, and flag the
// cells they are about. Every pin goes in the action log: undo one and its flags go with it. With the on-device
// model, or a key in Ask, a model may reorder and reword the findings, held to the numbers by the claim lock
// (aquascope.map_explain.claim_lock); without one, the package's templates say it.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { addPin, log, setLogOpen, undoEntry } from "./map-actions.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { currentMesh, meshAsk, meshFlagsBefore, onMeshCleared, refreshMeshLegend, setReviewSection } from "./mesh-view.js?v=__BUILD__";
import { SEVERITY, checkRows, flaggedCells, gaugesInBox, reviewLine, reviewPin, reviewWho, runSpan } from "./mesh-review-core.js?v=__BUILD__";

const FLAGS = "mesh-flags";
const FLAGS_LINE = "mesh-flags-line";
// A gauge record that takes longer than this to read is left out (the review says so).
const GAUGE_WAIT_MS = 45000;

let running = 0;
let lastPins = [];     // log ids of the last review's pins
let last = null;       // { name, res, who, how, refused }

function say(html, kind = "info") {
  const el = $("ma-status");
  if (!el) return;
  el.hidden = !html;
  el.className = `ma-status ${kind}`;
  el.innerHTML = html || "";
}

const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error(`${what} took too long`)), ms))]);

// ── what lies outside the file ──────────────────────────────────────────────

async function gatherContext(meta) {
  const span = runSpan(meta.times);
  const rows = gaugesInBox(state.stations, meta.bbox);
  const inside = rows.length ? await meshAsk("inside", { rows }) : [];
  const floods = callLight("mesh_review", { op: "floods", bbox: meta.bbox }, { priority: 2 })
    .catch((err) => ({ radar: { rows: null, error: err.message }, news: { events: null, error: err.message } }));
  let gauges = Promise.resolve(inside);
  if (inside.length && span) {
    gauges = withTimeout(call("mesh_review", { op: "gauges", gauges: inside, start: span.start, end: span.end }), GAUGE_WAIT_MS,
      "the gauge records")
      .then((r) => (r && r.gauges) || inside.map((g) => ({ ...g, note: "its record could not be read" })))
      .catch((err) => inside.map((g) => ({ ...g, series: [], note: `its record could not be read (${err.message})` })));
  }
  const [f, g] = await Promise.all([floods, gauges]);
  return { floods: f && !f.error ? f : { radar: { rows: null, error: f && f.error }, news: { events: null, error: f && f.error } }, gauges: g };
}

// ── a model orders and words (numbers stay the code's) ─────────────────────

async function wordWithModel(findings) {
  if (await localModelReady()) {
    try {
      const p = await meshAsk("prompt", { findings, context: currentContext() });
      say('<span class="spinner" aria-hidden="true"></span>Wording the review on your device…');
      const reply = await generateJsonLocally({ system: p.system, prompt: p.prompt, schema: p.schema, timeoutMs: 45000 });
      const out = await meshAsk("words", { findings, reply, by: "device" });
      return { ...out, by: "device", model: localReaderLabel() || "the on-device model" };
    } catch (err) {
      console.warn("review: on-device wording", err && err.message);
    }
  }
  const cfg = askModelConfig();
  if (cfg) {
    say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(cfg.label)} to order and word the review…`);
    const out = await call("mesh_review", { op: "llm", findings, context: currentContext(), provider: cfg.provider,
      model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url });
    if (out && out.error) throw new Error(out.error);
    return { ...out, by: "key" };
  }
  return null;
}

// ── the flags on the cells ──────────────────────────────────────────────────

function flagFeatures() {
  const m = currentMesh();
  if (!m) return [];
  const cells = flaggedCells(log.pins(), m.meta.name);
  const { ring_offsets: off, ring_lon: xs, ring_lat: ys } = m.arrays;
  const out = [];
  for (const [c, sev] of cells) {
    if (c < 0 || c >= off.length - 1) continue;
    const ring = [];
    for (let k = off[c]; k < off[c + 1]; k++) ring.push([xs[k], ys[k]]);
    if (ring.length < 3) continue;
    ring.push(ring[0]);
    out.push({ type: "Feature", properties: { sev, colour: (SEVERITY[sev] || SEVERITY.info).colour },
      geometry: { type: "Polygon", coordinates: [ring] } });
  }
  return out;
}

function drawFlags() {
  if (!state.mapOk || !map || !map.getStyle()) return;
  const data = { type: "FeatureCollection", features: flagFeatures() };
  const src = map.getSource(FLAGS);
  if (src) { src.setData(data); return; }
  if (!data.features.length) return;
  const before = meshFlagsBefore();
  map.addSource(FLAGS, { type: "geojson", data });
  map.addLayer({ id: FLAGS, type: "fill", source: FLAGS,
    paint: { "fill-color": ["get", "colour"], "fill-opacity": ["match", ["get", "sev"], "info", 0.3, 0.45] } }, before);
  map.addLayer({ id: FLAGS_LINE, type: "line", source: FLAGS,
    paint: { "line-color": ["get", "colour"], "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.8, 15, 2],
      "line-opacity": 0.95 } }, before);
}

function removeFlags() {
  if (!map) return;
  for (const id of [FLAGS_LINE, FLAGS]) if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(FLAGS)) map.removeSource(FLAGS);
}

// ── the legend row's part ───────────────────────────────────────────────────

function section(name) {
  if (!last || last.name !== name) return "";
  const live = lastPins.filter((id) => { const e = log.get(id); return e && !e.undone; }).length;
  const rows = checkRows(last.res.checks).map((c) =>
    `<li class="${escapeHtml(c.status)}"><i class="mr-mark" aria-hidden="true">${escapeHtml(c.mark)}</i>` +
    `<b>${escapeHtml(c.label)}</b><span>${escapeHtml(c.line)}</span></li>`).join("");
  const key = Object.entries(SEVERITY).map(([, v]) => `<span><i style="background:${v.colour}"></i>${escapeHtml(v.words)}</span>`).join("");
  return `<div class="mv-review"><p class="mr-head">Review: ${escapeHtml(last.res.summary)}</p>` +
    `<p class="mr-who">${escapeHtml(last.who)}, ${escapeHtml(last.how)}. ${live} pin${live === 1 ? "" : "s"} on the map.</p>` +
    `<p class="mr-key">${key}</p>` +
    `<details><summary>The ${last.res.checks.length} checks</summary><ul>${rows}</ul></details></div>`;
}

// ── the review ──────────────────────────────────────────────────────────────

function clearLast() {
  for (const id of lastPins) {
    const e = log.get(id);
    if (e && !e.undone) undoEntry(id);
  }
  lastPins = [];
}

export async function reviewModel({ say: sayOut = null } = {}) {
  const m = currentMesh();
  const tell = sayOut || say;
  if (!m) {
    tell("No model result on the map to review. Drop a HEC-RAS plan or a UGRID NetCDF file on the map first.", "warn");
    return null;
  }
  const my = ++running;
  const { meta } = m;
  tell(`<span class="spinner" aria-hidden="true"></span>Reviewing ${escapeHtml(meta.name)}: reading the floods and gauges around it…`);
  try {
    const context = await gatherContext(meta);
    if (my !== running) return null;
    tell('<span class="spinner" aria-hidden="true"></span>Running the checks on your file, in your browser…');
    const res = await meshAsk("review", { context });
    if (my !== running || !currentMesh() || currentMesh().meta.name !== meta.name) return null;
    let findings = res.findings || [];
    let by = "rules", model = "", refused = [];
    const modelReady = Boolean(askModelConfig()) || await localModelReady();
    if (modelReady && findings.length) {
      try {
        const worded = await wordWithModel(findings);
        if (my !== running) return null;
        if (worded && worded.findings && worded.findings.length) {
          findings = worded.findings;
          by = worded.by;
          model = worded.model || "";
          refused = worded.refused || [];
        }
      } catch (err) {
        refused = [`the model could not be asked (${err.message}); the rules' wording is kept`];
      }
    }
    clearLast();
    // The last first: pin 1 is drawn last, so it sits on top where pins crowd.
    for (const f of findings.slice().reverse()) {
      try {
        const id = addPin(reviewPin(f, { mesh: meta.name }), { by: f.by === "rules" ? "rules" : by,
          label: `Review ${f.rank}: ${f.title}` });
        lastPins.push(id);
      } catch (err) {
        console.warn("review pin", err && err.message);
      }
    }
    const [who, how] = reviewWho(by, model);
    last = { name: meta.name, res, who, how, refused };
    drawFlags();
    refreshMeshLegend();
    const n = lastPins.length;
    const notes = refused.length ? [`${refused.length} wording${refused.length === 1 ? "" : "s"} refused by the claim lock, rules kept`] : [];
    tell(`<b class="ma-who">Review</b><span class="ma-how">${escapeHtml(who)}, ${escapeHtml(how)}</span>` +
      `<span>${escapeHtml(reviewLine(res, n))}</span>` +
      (notes.length ? `<span class="ma-note">${escapeHtml(notes.join("; "))}</span>` : ""), res.counts && res.counts.problem ? "warn" : "ok");
    if (n) setLogOpen(false);
    announce(`Review dropped ${n} pins: ${res.summary}`);
    globalThis.__aq.review.result = () => res;
    return { findings, by, summary: res.summary };
  } catch (err) {
    if (my === running) tell(`The review could not run: ${escapeHtml(err.message)}`, "error");
    return null;
  }
}

export function initMeshReview() {
  setReviewSection(section);
  actions.reviewModel = reviewModel;   // the legend row's button, Ask the map's "review my model", WebMCP
  log.subscribe(() => { if (currentMesh()) { drawFlags(); if (last) refreshMeshLegend(); } });
  onMeshCleared(() => {
    clearLast();
    last = null;
    removeFlags();
  });
  if (state.mapOk && map) map.on("style.load", () => setTimeout(() => { if (currentMesh()) drawFlags(); }, 0));
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.review = { run: reviewModel, last: () => lastPins.slice(), result: () => (last ? last.res : null) };
}
