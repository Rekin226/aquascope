// Flow vs normal on the rivers (#559): the living-rivers network coloured by
// each reach's forecast flow for the day against its own normal for the month,
// the NOAA National Water Model map for the whole world.
//
// The daily job (flood-warnings.yml, aquascope/flowstatus.py) publishes
// flowstatus/latest.bin.gz: about 840,000 reaches as sorted river ids and one
// class byte each, a few hundred kB. This module reads it once, and as the
// stream tiles load it sets each reach's class as feature-state "fs" on riverId
// (the network's promoteId); river-map.js colours the line by it. A choice in
// the Rivers row of "On the map" turns it on; off by default.

import { CONFIG } from "../config.js?v=__BUILD__";
import { escapeHtml, state } from "./core.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { openLegendRow, refreshLegend } from "./map-legend.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { STATUS_CLASSES } from "./now-core.js?v=__BUILD__";
import { ensureRiverLayers, setRiverColouring, setRiverRowExtras, setRiversVisible } from "./river-map.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { refreshStatusOpacity } from "./status-layer.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import {
  FS_CREDIT, FS_MIN_ORDER, classOf, decodeFlowStatus, isGzip, issueLine, readFlowNormalParam, rowSummary, sharesLine,
} from "./flowstatus-core.js?v=__BUILD__";

export { FS_CREDIT };

const BASE = CONFIG.flowstatusBase || "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/flowstatus/";
const NET = { source: "river-net", sourceLayer: "streams" };

let data = null;          // { manifest, ids, classes } | { manifest: { missing: true } }
let loading = null;
let failed = false;
let applied = new Set();  // reaches whose class is set on the map
let timer = 0;

// ── data ────────────────────────────────────────────────────────────────────

async function gunzip(bytes) {
  if (!isGzip(bytes)) return bytes;   // a server that already undid the gzip
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function load() {
  if (loading) return loading;
  loading = (async () => {
    const res = await fetch(`${BASE}manifest.json`, { cache: "no-cache" });
    if (!res.ok) { data = { manifest: { missing: true } }; return data; }
    const manifest = await res.json();
    const bin = await fetch(`${BASE}latest.bin.gz`, { cache: "no-cache" });
    if (!bin.ok) throw new Error(`latest.bin.gz answered ${bin.status}`);
    const decoded = decodeFlowStatus(await gunzip(new Uint8Array(await bin.arrayBuffer())));
    data = { manifest, ...decoded };
    return data;
  })().catch((err) => {
    console.info("flow vs normal:", err && err.message);
    failed = true;
    loading = null;   // a later turn-on tries again
    return null;
  }).finally(() => { refreshLegend("rivers"); });
  return loading;
}

const ready = () => Boolean(data && data.ids);

// ── the classes on the map ──────────────────────────────────────────────────

// Every reach of order 5 and up in the tiles loaded so far gets its class once; feature-state stays with the
// source, so a tile that is dropped and read again keeps it.
function applyLoaded() {
  timer = 0;
  if (!state.flowNormal || !ready() || !map || !map.getSource(NET.source)) return;
  let feats = [];
  try {
    feats = map.querySourceFeatures(NET.source, { sourceLayer: NET.sourceLayer,
      filter: [">=", ["get", "strahlerOrder"], FS_MIN_ORDER] });
  } catch { return; }
  for (const f of feats) {
    const id = f.id;
    if (id === undefined || id === null || applied.has(id)) continue;
    applied.add(id);
    const code = classOf(data, id);
    if (code) map.setFeatureState({ ...NET, id }, { fs: code });
  }
}

function schedule() {
  if (!timer) timer = setTimeout(applyLoaded, 150);
}

function sync() {
  const on = Boolean(state.flowNormal && state.riversOn && ready());
  state.flowNormalShown = on;   // the classes are on the lines (not before the first issue is published)
  setRiverColouring(on ? "flow" : null);
  refreshStatusOpacity();   // the basins step back while the rivers carry the classes
  if (on) schedule();
}

/** Turn the colouring on or off (the Rivers row, a link, Ask the map). */
export function setFlowNormal(on, { write = true } = {}) {
  state.flowNormal = Boolean(on);
  if (state.flowNormal) {
    if (!state.riversOn) setRiversVisible(true);   // the colours live on the rivers
    ensureRiverLayers();
    void load().then(sync);
  }
  sync();
  renderCredits();
  refreshLegend("rivers");
  if (state.flowNormal) openLegendRow("rivers", true);
  if (write) writeUrl();
}

// ── the Rivers row ──────────────────────────────────────────────────────────

const bar = (cls) => `<span class="${cls}">${STATUS_CLASSES.map((c) =>
  `<i style="--c:${c.color}" title="${escapeHtml(c.label)}"></i>`).join("")}</span>`;

function rowBody() {
  const check = `<label class="ml-check"><input type="checkbox" data-act="flow-normal" ${state.flowNormal ? "checked" : ""}>` +
    " Colour by flow against normal</label>";
  if (!state.flowNormal) return check;
  const m = data && data.manifest;
  if (!m) {
    const wait = failed ? "Could not read today's classes just now." :
      '<span class="spinner" aria-hidden="true"></span> Reading today\'s classes…';
    return `${check}<p class="ml-when">${wait}</p>`;
  }
  if (m.missing) return `${check}<p class="ml-when">${escapeHtml(issueLine(m))}</p>`;
  const shares = sharesLine(m);
  return check + bar("sl-bar") +
    '<div class="sl-ends"><span>much below</span><span>normal</span><span>much above</span></div>' +
    `<p class="ml-when">${escapeHtml(issueLine(m))}${shares ? ` ${escapeHtml(shares)}` : ""}</p>` +
    `<p class="ml-src">Order ${FS_MIN_ORDER} and up; faint lines have no class${state.status ? "; river status fades while this is on" : ""}. ` +
    "Modelled: GEOGLOWS, CC BY 4.0. " +
    '<button type="button" class="ml-link" data-act="flow-normal-info">About</button></p>';
}

function openAbout() {
  const m = (data && data.manifest) || {};
  const swatches = STATUS_CLASSES.map((c) =>
    `<span class="sw"><i style="background:${c.color}"></i>${escapeHtml(c.label)}</span>`).join("");
  openModal("Flow vs normal", `
    <p>Each river reach of Strahler order ${FS_MIN_ORDER} and up is coloured by its flow for the day against its own
    normal for the month: below the 10th percentile of daily flow is much below normal, 10th to 25th below, 25th to
    75th normal, 75th to 90th above, and over the 90th much above. The same classes and colours as the river status
    layer and the gauges' "Today vs normal".</p>
    <div class="swatches">${swatches}</div>
    <p>The day's flow is the GEOGLOWS v2 forecast's first day (the mean of its 51 members); the normal is the reach's
    percentiles of daily flow in the same calendar month of the GEOGLOWS retrospective simulation, ${escapeHtml(m.base_period || "1950 to 2019")}
    (<code>retrospective/fdc.zarr</code>). Both are modelled, not measured: where the model is wrong it is wrong in
    both, and the colour says where today sits in the model's own climate.</p>
    ${m.issue_date ? `<p>${escapeHtml(issueLine(m))} ${escapeHtml(sharesLine(m))}</p>` : ""}
    <p class="muted">${FS_CREDIT.attribution}. Licence: ${escapeHtml(FS_CREDIT.licence)}. Rebuilt every day by the
    flood-warnings workflow. The same in Python: <code>aquascope flowstatus</code>, and the MCP tool
    <code>flow_vs_normal</code>.</p>`);
}

function registerExtras() {
  setRiverRowExtras({
    summary: () => (state.flowNormal ? rowSummary(data && data.manifest, { loading: !data && !failed, error: failed && !data }) : null),
    mark: () => (state.flowNormal && ready() ? bar("sl-bar mini") : null),
    body: rowBody,
    act: (name, el) => {
      if (name === "flow-normal") { setFlowNormal(el.checked); return true; }
      if (name === "flow-normal-info") { openAbout(); return true; }
      return false;
    },
  });
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initFlowStatus() {
  registerExtras();
  const fromUrl = readFlowNormalParam(location.hash);
  if (fromUrl !== null) state.flowNormal = fromUrl;
  if (!state.mapOk || !map) return;
  map.on("sourcedata", (e) => { if (e.sourceId === NET.source && state.flowNormal) schedule(); });
  map.on("moveend", () => { if (state.flowNormal) schedule(); });
  // A basemap change replaces the style: every feature-state goes with it.
  map.on("style.load", () => { applied = new Set(); sync(); });
  window.addEventListener("hashchange", () => {
    const want = readFlowNormalParam(location.hash);
    const next = want === null ? false : want;
    if (next !== Boolean(state.flowNormal)) setFlowNormal(next, { write: false });
  });
  if (state.flowNormal) setFlowNormal(true, { write: false });
}
