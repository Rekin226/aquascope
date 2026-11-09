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
// model, or a key in Ask, the model picks and the code writes (#590): it chooses which findings to pin, in what
// order, and one of three fixed variants of each, as ids held to a JSON Schema of enums
// (aquascope.mesh_review.review_schema, Chrome's responseConstraint). Every word is the package's, a problem is
// always pinned and the counts are the rules'. A reply outside the schema is refused whole and said in one line.

import { $, actions, escapeHtml, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { addPin, log, setLogOpen, undoEntry } from "./map-actions.js?v=__BUILD__";
import { askModelConfig, currentContext } from "./ask.js?v=__BUILD__";
import { generateJsonLocally, localModelReady, localReaderLabel } from "./local-model.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { currentMesh, meshAsk, meshFlagsBefore, onMeshCleared, refreshMeshLegend, setReviewSection } from "./mesh-view.js?v=__BUILD__";
import { modflowShown } from "./groundwater-view.js?v=__BUILD__";
import {
  SEVERITY, checkRows, flaggedCells, gaugesInBox, reviewLine, reviewPin, reviewTarget, reviewWho, runSpan,
} from "./mesh-review-core.js?v=__BUILD__";
import { askedFailed, refusedNote } from "./scout-core.js?v=__BUILD__";

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

// ── a model picks the pins, their order and a variant of each (the words stay the package's) ──

const DEVICE_WAIT_MS = 45000;

async function wordWithModel(findings, question = "") {
  let asked = null;
  if (await localModelReady()) {
    const name = localReaderLabel() || "the on-device model";
    try {
      const p = await meshAsk("prompt", { findings, context: currentContext(), question });
      say('<span class="spinner" aria-hidden="true"></span>Picking the pins on your device…');
      const reply = await generateJsonLocally({ system: p.system, prompt: p.prompt, schema: p.schema, timeoutMs: DEVICE_WAIT_MS });
      const out = await meshAsk("words", { findings, reply, by: "device", question });
      return { ...out, by: "device", model: name };
    } catch (err) {
      console.warn("review: on-device pick", err && err.message);
      asked = { findings: null, by: "device", model: name, refused: [askedFailed(err, DEVICE_WAIT_MS)] };
    }
  }
  const cfg = askModelConfig();
  if (cfg) {
    say(`<span class="spinner" aria-hidden="true"></span>Asking ${escapeHtml(cfg.label)} to order and word the review…`);
    const out = await call("mesh_review", { op: "llm", findings, context: currentContext(), question,
      provider: cfg.provider, model: cfg.model, api_key: cfg.api_key, base_url: cfg.base_url });
    if (out && out.error) throw new Error(out.error);
    return { ...out, by: "key" };
  }
  return asked;
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

function viewBox() {
  if (!state.mapOk || !map) return null;
  try {
    const b = map.getBounds();
    return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  } catch {
    return null;
  }
}

/**
 * Review the model result on screen. With `anywhere` (the mesh's own legend row, which names its mesh, and the
 * WebMCP tool) the loaded mesh is reviewed wherever the map is. From Ask the map, the model on screen decides
 * (#590): a MODFLOW model there, or a mesh off screen, is said in the line rather than an older mesh reviewed out
 * of sight. `question` is what the reader asked, verbatim: a question about stability or instabilities names the
 * wet and dry flicker, which is then pinned whatever the model chose (aquascope.mesh_review.named_findings).
 */
export async function reviewModel({ say: sayOut = null, anywhere = false, question = "" } = {}) {
  const m = currentMesh();
  const tell = sayOut || say;
  const target = reviewTarget({
    mesh: m ? { name: m.meta.name, bbox: m.meta.bbox, openedAt: m.openedAt, visible: m.visible } : null,
    modflow: anywhere ? null : modflowShown(), view: anywhere ? null : viewBox(),
  });
  if (!m || !target.review) {
    tell(escapeHtml(target.line), "warn");
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
    let by = "rules", model = "", refused = [], leftOut = 0, asked = "";
    const modelReady = Boolean(askModelConfig()) || await localModelReady();
    if (modelReady && findings.length) {
      try {
        const worded = await wordWithModel(findings, question);
        if (my !== running) return null;
        if (worded && worded.findings && worded.findings.length && worded.ordered !== false) {
          findings = worded.findings;
          by = worded.by;
          model = worded.model || "";
          refused = worded.refused || [];
          leftOut = (worded.left_out || []).length;
        } else if (worded) {
          // Asked, and its reply refused whole or not given: the rules' pins, and the line says the model was asked.
          refused = worded.refused || [];
          asked = worded.model || (worded.by === "key" ? "your model" : "the on-device model");
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
    const [who, how] = reviewWho(by, model, asked);
    last = { name: meta.name, res, who, how, refused };
    drawFlags();
    refreshMeshLegend();
    const n = lastPins.length;
    // The count and the first refusal's reason (#590), as Scout's line gives them.
    const notes = refused.length ? [/^it was asked/.test(refused[0]) ? refused[0] : refusedNote(refused)] : [];
    tell(`<b class="ma-who">Review</b><span class="ma-how">${escapeHtml(who)}, ${escapeHtml(how)}</span>` +
      `<span>${escapeHtml(reviewLine(res, n, meta.name, { picked: by !== "rules", leftOut }))}</span>` +
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
