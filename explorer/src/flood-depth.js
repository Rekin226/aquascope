// Flood depth where floods are forecast (#554): the RAS Mapper view of the
// forecast, keyless. When Floods ahead (#546) says a reach will pass its 10-,
// 25-, 50- or 100-year flow, the JRC CEMS-GloFAS depth map of the nearest
// return period at or below it is drawn around the reach, in blues, labelled
// "may flood in the next 15 days, model estimate".
//
// The map is cut into half-degree cells. For each cell in view (from zoom 7,
// where 90 m pixels start to mean something) a worker reads the depth windows
// it needs with byte ranges and paints them (flood-depth-worker.js with
// flood-depth-core.js); each cell is an image on the map, under the rivers.
// The time bar (core.js setTime) picks the forecast day, so playing the 15
// days steps the depth up and down with the forecast. A click on the depth
// opens the map card with the reach, the return period and the depth there.

import { $, actions, clickLayers, escapeHtml, onTime, state } from "./core.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { holdSettle, map } from "./map.js?v=__BUILD__";
import { openCard } from "./map-card.js?v=__BUILD__";
import { addDays, dayIndex, shortDay } from "./floods-ahead-core.js?v=__BUILD__";
import { floodsAheadData } from "./floods-ahead.js?v=__BUILD__";
import {
  CELL_PX, DEPTH_CREDIT, DEPTH_LABEL, DEPTH_MINZOOM, MAX_CELLS, RAMP, activeReaches, cellLines, cellParts,
  cellSignature, cellsFor, cellsInView, depthFacts, depthLegendLine, diskBox, probeCell, rampColor, rampCss, reachAt,
} from "./flood-depth-core.js?v=__BUILD__";

export { DEPTH_CREDIT };

const HIT = "fd-hit";
const PREFIX = "fd-cell-";
const KEEP = 48;              // painted cells kept in memory, so panning back is instant

let data = null;              // { manifest, features } from Floods ahead
let visible = true;
let day = -1;                 // the forecast day on show, -1 for the 15-day peak
let reaches = [];             // the reaches with a depth map on that day
let cells = new Map();        // key -> cell (its box and its reaches)
const onMap = new Map();      // key -> { sig, painted } of the cells drawn now
const pictures = new Map();   // `${key}|${sig}` -> { url, painted } (most recent last)
const loading = new Map();    // `${key}|${sig}` -> promise
let worker = null;
let reqId = 0;
const waiting = new Map();
let busy = 0;
let failed = false;
let legend = null;
let hitBound = false;

const vis = () => (visible ? "visible" : "none");

// ── reading a cell ──────────────────────────────────────────────────────────

function inWorker(cell, parts, lines) {
  if (!worker) {
    worker = new Worker(new URL("./flood-depth-worker.js?v=__BUILD__", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const job = waiting.get(e.data.id);
      if (!job) return;
      waiting.delete(e.data.id);
      if (e.data.error) job.reject(new Error(e.data.error)); else job.resolve(e.data);
    };
    worker.onerror = (e) => {
      console.warn("flood depth worker:", e && e.message);
      for (const job of waiting.values()) job.reject(new Error("worker unavailable"));
      waiting.clear();
      worker = false;
    };
  }
  if (worker === false) return Promise.reject(new Error("no worker"));
  const id = ++reqId;
  // Only what the worker needs: the box and the circles, not the reaches' forecast properties.
  const slim = { key: cell.key, box: cell.box, reaches: cell.reaches.map(({ id: rid, lon, lat, r, rp, order }) => ({ id: rid, lon, lat, r, rp, order })) };
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, cell: slim, parts, size: CELL_PX, lines: { act: lines.act, other: lines.other } });
  });
}

function keep(k, pic) {
  pictures.set(k, pic);
  for (const [old, p] of pictures) {
    if (pictures.size <= KEEP) break;
    const key = old.split("|")[0];
    if (onMap.has(key) && onMap.get(key).sig === old.split("|")[1]) continue;
    if (p.url) URL.revokeObjectURL(p.url);
    pictures.delete(old);
  }
  return pic;
}

function picture(cell, sig, lines) {
  const k = `${cell.key}|${sig}`;
  if (pictures.has(k)) {
    const p = pictures.get(k);
    pictures.delete(k);
    pictures.set(k, p);
    return Promise.resolve(p);
  }
  if (loading.has(k)) return loading.get(k);
  const p = inWorker(cell, cellParts(cell), lines)
    .then((res) => keep(k, { url: res.png ? URL.createObjectURL(res.png) : null, painted: res }))
    .finally(() => loading.delete(k));
  loading.set(k, p);
  return p;
}

// ── the layers ──────────────────────────────────────────────────────────────

// With Floods ahead, which chooses where it is drawn: over its soft glow and under its crisp class line, so the
// reach forecast to flood stays drawn on top of its flood plain. Without it, under the rivers and the gauges.
// Always over the river status and Floods past (#543).
function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const fa = layers.find((l) => l.id === "river-fa-casing" || l.id === "river-fa-line");
  if (fa) return fa.id;
  const hit = layers.find((l) => /^river-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

function ensureHit() {
  if (map.getSource(HIT)) return;
  map.addSource(HIT, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  // Invisible rectangles over the drawn cells: the pictures cannot be queried, these can.
  map.addLayer({ id: HIT, type: "fill", source: HIT, minzoom: DEPTH_MINZOOM, layout: { visibility: vis() },
    paint: { "fill-color": "#000", "fill-opacity": 0 } }, beforeId());
  if (hitBound) return;
  hitBound = true;   // a basemap change re-adds the layer; the handler stays bound to its id
  clickLayers.add(HIT);
  map.on("click", HIT, onClick);
}

// The drawn cells as rectangles: a click inside one is checked against the picture (probeCell); a dry pixel goes
// on to the map's own answer.
function syncHit() {
  const src = map.getSource(HIT);
  if (!src) return;
  src.setData({ type: "FeatureCollection", features: [...onMap.values()].map(({ cell }) => {
    const [w, s, e, n] = cell.box;
    return { type: "Feature", properties: { key: cell.key }, geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } };
  }) });
}

function drawCell(cell, pic) {
  const id = PREFIX + cell.key;
  const [w, s, e, n] = cell.box;
  const src = map.getSource(id);
  if (!pic.url) { removeCell(cell.key); return; }
  // One image in flight per source: a second updateImage would abort the first, so a busy one is replaced.
  if (src && src.updateImage && src.loaded()) {
    if (src.url !== pic.url) src.updateImage({ url: pic.url });
  } else {
    if (src) removeCell(cell.key);
    map.addSource(id, { type: "image", url: pic.url, coordinates: [[w, n], [e, n], [e, s], [w, s]] });
    map.addLayer({
      id, type: "raster", source: id, layout: { visibility: vis() },
      paint: {
        "raster-opacity": ["interpolate", ["linear"], ["zoom"], DEPTH_MINZOOM, 0.55, DEPTH_MINZOOM + 1, 1],
        "raster-fade-duration": 0, "raster-resampling": "linear",
      },
    }, beforeId());
  }
}

function removeCell(key) {
  const id = PREFIX + key;
  if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(id)) map.removeSource(id);
  if (onMap.delete(key)) syncHit();
}

// ── what to draw ────────────────────────────────────────────────────────────

function viewBounds() {
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
}

// The rivers in view from the stream tiles already loaded for the rivers and Floods ahead: riverId -> { order,
// segs, box }. Only rivers of order 5 and up and the forecast reaches themselves, which is what cellLines uses.
const NET_SOURCES = ["river-fa-net", "river-net"];
function riverLines(ids) {
  const out = new Map();
  for (const src of NET_SOURCES) {
    if (!map.getSource(src)) continue;
    let feats = [];
    try {
      feats = map.querySourceFeatures(src, { sourceLayer: "streams",
        filter: ["any", [">=", ["coalesce", ["get", "strahlerOrder"], 0], 5], ["in", ["to-number", ["get", "riverId"]], ["literal", ids]]] });
    } catch { continue; }
    for (const f of feats) {
      const id = Number(f.properties.riverId);
      const g = f.geometry;
      const parts = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : [];
      if (!parts.length) continue;
      let e = out.get(id);
      if (!e) { e = { order: Number(f.properties.strahlerOrder) || 0, segs: [], box: [Infinity, Infinity, -Infinity, -Infinity] }; out.set(id, e); }
      for (const line of parts) {
        for (let k = 0; k + 1 < line.length; k++) {
          const [x1, y1] = line[k], [x2, y2] = line[k + 1];
          e.segs.push(x1, y1, x2, y2);
          e.box[0] = Math.min(e.box[0], x1, x2); e.box[1] = Math.min(e.box[1], y1, y2);
          e.box[2] = Math.max(e.box[2], x1, x2); e.box[3] = Math.max(e.box[3], y1, y2);
        }
      }
    }
    if (out.size) break;   // one network is enough
  }
  return out;
}

let wanted = [];
let wantedSig = new Map();
function update() {
  renderLegend();
  if (!state.mapOk || !map || !data) return;
  if (!visible || map.getZoom() < DEPTH_MINZOOM) {
    for (const key of [...onMap.keys()]) removeCell(key);
    wanted = [];
    renderLegend();
    return;
  }
  ensureHit();
  const c = map.getCenter();
  wanted = cellsInView(cells, viewBounds(), [c.lng, c.lat], MAX_CELLS);
  const keys = new Set(wanted.map((x) => x.key));
  for (const key of [...onMap.keys()]) if (!keys.has(key)) removeCell(key);
  const lines = riverLines([...new Set(wanted.flatMap((x) => x.reaches.map((r) => r.id)))]);
  wantedSig = new Map();
  const jobs = [];
  for (const cell of wanted) {
    const cl = cellLines(cell, lines);
    const sig = cellSignature(cell, cl.sig);
    wantedSig.set(cell.key, sig);
    const now = onMap.get(cell.key);
    if (now && now.sig === sig) continue;
    busy++;
    const job = picture(cell, sig, cl).then((pic) => {
      if (!visible || wantedSig.get(cell.key) !== sig) return;
      drawCell(cell, pic);
      if (pic.url) { onMap.set(cell.key, { sig, painted: pic.painted, cell }); syncHit(); }
      failed = false;
    }).catch((err) => {
      console.info(`flood depth ${cell.key}:`, err && err.message);
      failed = true;
    }).finally(() => { busy--; renderLegend(); });
    jobs.push(job);
  }
  if (jobs.length) holdSettle(Promise.all(jobs));
  renderLegend();
}

function setDay(next) {
  day = next;
  reaches = data && !data.manifest.missing ? activeReaches(data.features, day) : [];
  cells = cellsFor(reaches);
  update();
}

// ── the click ───────────────────────────────────────────────────────────────

function onClick(e) {
  // Anything else that takes a click here (a gauge, a Floods ahead reach, a flood cell) answers for itself.
  const others = ["points", "clusters", ...[...clickLayers].filter((id) => id !== HIT)].filter((id) => map.getLayer(id));
  if (others.length && map.queryRenderedFeatures(e.point, { layers: others }).length) return;
  const { lng, lat } = e.lngLat;
  let hit = null;
  for (const { cell, painted } of onMap.values()) {
    hit = probeCell(cell, painted, lng, lat);
    if (hit) break;
  }
  if (!hit) { actions.selectPoint(lat, lng); return; }   // dry here: the map's own answer
  const reach = reachAt(reaches, lng, lat, hit.rp) || reachAt(reaches, lng, lat);
  if (!reach) { actions.selectPoint(lat, lng); return; }
  const issue = data.manifest.issue_date;
  const when = day >= 0 ? `on ${shortDay(addDays(issue, day))}` : (reach.props.day ? `(peak on ${shortDay(reach.props.day)})` : "");
  const f = depthFacts(reach, hit.depth, hit.rp, { day, when });
  const c = rampColor(hit.depth);
  openCard({
    id: `depth:${reach.id}:${hit.rp}:${lng.toFixed(3)},${lat.toFixed(3)}`, lngLat: [lng, lat], lift: 8,
    what: "Flood depth", title: f.title, sub: f.sub,
    status: { text: f.status, color: c ? `rgb(${c[0]},${c[1]},${c[2]})` : null },
    figure: f.figure, note: f.note,
    credit: "JRC CEMS-GloFAS hazard map v2.1.2, © European Union, CC BY 4.0. Forecast: GEOGLOWS, CC BY 4.0.",
    details: () => actions.selectPoint(reach.lat, reach.lon, { tab: "now" }),
    buttons: [{ id: "about", label: "About", title: "How this map is chosen, and what it is not", onClick: about }],
  });
}

// ── the legend ──────────────────────────────────────────────────────────────
// Where the page has the shared "On the map" legend (#map-legend, map-legend.js), the depth is one row in it.
// Until then it is a small card in the legend stack (#map-legends), under Floods ahead. Both show the same key.

let legendRow = null;     // map-legend.js's refreshLegend, once the depth is a row there
// The shared legend's element: optional on purpose (the page may not have it yet), so it is looked up by name.
const SHARED_LEGEND = "map-legend";

function viewCounts() {
  const zoom = state.mapOk && map ? map.getZoom() : 0;
  const inView = zoom >= DEPTH_MINZOOM ? new Set(wanted.flatMap((c) => c.reaches.map((r) => r.id))).size : 0;
  return { zoom, inView };
}

function act(name) {
  if (name === "hide") setDepthVisible(false);
  else if (name === "about") about();
  else if (name === "go") goToOne();
  else if (name === "min" && legend) { userFold = !folded(); renderLegend(); }
}

// The key: the honest label, the ramp, what is drawn, where it comes from.
function keyHtml(withAbout = false) {
  const { zoom, inView } = viewCounts();
  const line = depthLegendLine({ n: reaches.length, inView, zoom });
  const ticks = RAMP.map((s) => `<span style="left:${Math.round(Math.sqrt(s.depth_m / 10) * 100)}%">` +
    `${s.depth_m < 1 ? "0" : s.depth_m}${s.depth_m >= 10 ? "+ m" : ""}</span>`).join("");
  const go = reaches.length && (zoom < DEPTH_MINZOOM || !inView)
    ? ' <button type="button" class="fd-go" data-act="go">Show one</button>' : "";
  return `<p class="fd-tag">${escapeHtml(DEPTH_LABEL[0].toUpperCase() + DEPTH_LABEL.slice(1))}</p>` +
    `<div class="fd-ramp" role="img" aria-label="Water depth from 0 to over 10 metres, light to deep blue" style="background:${rampCss()}"></div>` +
    `<div class="fd-ticks" aria-hidden="true">${ticks}</div>` +
    `<p class="fd-line">${escapeHtml(line)}${go}</p>` +
    (failed ? '<p class="fd-err">Some depth tiles could not be read.</p>' : "") +
    `<p class="fd-src">JRC GloFAS hazard maps v2.1.2, © EU, CC BY 4.0${withAbout ? ' · <button type="button" class="fd-go" data-act="about">About</button>' : ""}</p>`;
}

// The row's definition for "On the map" (map-legend.js registerLegendRow).
const ROW = {
  id: "flood-depth",
  title: "Flood depth",
  mark: () => `<i class="fd-mark" style="background:${rampCss()}"></i>`,
  summary: () => {
    if (busy > 0) return "reading…";
    const { zoom, inView } = viewCounts();
    return zoom < DEPTH_MINZOOM || !inView ? "model estimate, zoom in" : "model estimate";
  },
  on: () => visible,
  empty: () => !data || Boolean(data.manifest.missing) || !reaches.length,
  toggle: (on) => setDepthVisible(on),
  body: () => keyHtml(true),
  act: (name) => act(name),
};

function buildLegend() {
  if (document.getElementById(SHARED_LEGEND)) {
    import("./map-legend.js?v=__BUILD__").then((m) => {
      m.registerLegendRow(ROW);
      legendRow = () => m.refreshLegend(ROW.id);
    }).catch((err) => console.info("flood depth legend row:", err && err.message));
    return;
  }
  const stack = document.getElementById("map-legends");
  if (!stack || legend) return;
  legend = document.createElement("section");
  legend.className = "fd-legend";
  legend.setAttribute("aria-label", "Flood depth legend");
  // Under Floods ahead, which chooses where it is drawn.
  const fa = stack.querySelector(".fa-legend");
  if (fa) fa.after(legend); else stack.prepend(legend);
  legend.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-act]");
    if (btn) act(btn.dataset.act);
  });
}

// Folded to one line on a phone and while the depth cannot be seen (under zoom 7), unless the reader chose.
let userFold = null;
function folded() {
  if (userFold !== null) return userFold;
  const phone = Boolean(globalThis.matchMedia && globalThis.matchMedia("(max-width: 860px)").matches);
  return phone || viewCounts().zoom < DEPTH_MINZOOM;
}

function renderLegend() {
  if (legendRow) { legendRow(); return; }
  if (!legend) return;
  // Before the first issue Floods ahead's own legend says so; this one stays out of the way.
  legend.hidden = !visible || !data || Boolean(data.manifest.missing);
  if (legend.hidden) return;
  const min = folded();
  legend.classList.toggle("min", min);
  const reading = busy > 0 ? '<span class="fd-busy" aria-hidden="true"></span>' : "";
  legend.innerHTML =
    `<header><b>Flood depth</b>${reading}` +
    (min ? `<span class="fd-mini" style="background:${rampCss()}" aria-hidden="true"></span><span class="fd-est">model estimate</span>` : "") +
    '<button class="fd-btn info" type="button" data-act="about" aria-label="About the flood depth map" title="About this map">i</button>' +
    `<button class="fd-btn" type="button" data-act="min" aria-expanded="${min ? "false" : "true"}" ` +
    `aria-label="${min ? "Show" : "Fold"} the flood depth legend" title="${min ? "Show" : "Fold"}">${min ? "+" : "–"}</button>` +
    '<button class="fd-btn" type="button" data-act="hide" aria-label="Hide the flood depth map" title="Hide">×</button></header>' +
    (min ? "" : keyHtml());
}

// The strongest reach of the day: the deepest map, then the largest peak against its 2-year flow.
function goToOne() {
  if (!reaches.length) return;
  const ratio = (r) => Number(r.props.peak) / Math.max(1e-9, Number(r.props.q2));
  const best = [...reaches].sort((a, b) => b.rp - a.rp || ratio(b) - ratio(a))[0];
  map.flyTo({ center: [best.lon, best.lat], zoom: Math.max(map.getZoom(), 10.5), duration: 1600 });
}

function about() {
  import("./shell.js?v=__BUILD__").then(({ openModal }) => openModal("Flood depth where floods are forecast", `
    <p>When Floods ahead says a river reach will pass its 10-, 25-, 50- or 100-year flow in the next 15 days, the map
    shows the JRC flood depth map for the nearest return period at or below it (10, 20, 50 or 100 years; JRC has no
    2- or 5-year maps), within a few kilometres of the reach: 3 km on a Strahler order 5 river, 1.5 km more for each
    order up, fading at the edge. Near a confluence the circle also takes in the other river, which may not be
    forecast to flood. Move the time bar through the forecast, or press Play the 15 days, and the depth
    steps up and down with the forecast.</p>
    <p><strong>What it is not.</strong> A precomputed hazard map chosen by a forecast, not a flood simulation of
    this event: a model estimate twice over. The forecast (GEOGLOWS) and the hazard map (JRC, made with LISFLOOD
    and LISFLOOD-FP) are different models, and their return periods are not the same floods. The maps cover large
    rivers and leave out permanent water; JRC warns that depths over 10 m on small channels can be artefacts. For
    warnings, follow your national hydrological or meteorological service.</p>
    <p class="muted">Depth: JRC CEMS-GloFAS global river flood hazard maps v2.1.2, 3 arc-seconds (about 90 m),
    read in your browser with byte ranges from the <a href="https://source.coop/nlebovits/jrc-glofas" target="_blank" rel="noopener">Source Cooperative mirror</a>
    by geotiff.js (MIT). Licence: CC BY 4.0. JRC's <a href="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/copyright.txt" target="_blank" rel="noopener">copyright notice</a>
    licenses the dataset under CC BY 4.0, credit given and changes indicated; its README says "no restrictions, free
    and open Copernicus product". &copy; European Union. Citation: Baugh et al. (2024), European Commission, JRC.
    The same in Python: <code>aquascope layers depth RIVER_ID</code>.</p>`));
}

// ── on and off ──────────────────────────────────────────────────────────────

export function setDepthVisible(on) {
  visible = Boolean(on);
  state.depthOn = visible;
  const toggle = $("toggle-depth");
  if (toggle) toggle.checked = visible;
  renderCredits();
  if (state.mapOk && map && map.getLayer(HIT)) map.setLayoutProperty(HIT, "visibility", vis());
  update();
}

export function initFloodDepth() {
  if (!state.mapOk || !map) return;
  buildLegend();
  state.depthOn = visible;
  const toggle = $("toggle-depth");
  if (toggle) {
    toggle.checked = visible;
    toggle.addEventListener("change", (e) => setDepthVisible(e.target.checked));
  }
  onTime((t) => {
    if (t.date === t.prev.date || !data || data.manifest.missing) return;
    const next = dayIndex(data.manifest.issue_date, t.date);
    if (next !== day) setDay(next);
  });
  let t = 0;
  const later = (ms) => { clearTimeout(t); t = setTimeout(update, ms); };
  map.on("moveend", () => later(120));
  // The stream tiles shape the cells (cellLines): when more of them arrive, look again.
  map.on("sourcedata", (e) => {
    if (NET_SOURCES.includes(e.sourceId) && e.isSourceLoaded && visible && map.getZoom() >= DEPTH_MINZOOM) later(400);
  });
  // A basemap change replaces the style: our sources go with it, so draw the cells again.
  map.on("style.load", () => { onMap.clear(); setTimeout(() => setDay(day), 0); });
  floodsAheadData().then((d) => {
    data = d || { manifest: { missing: true }, features: [] };
    setDay(data.manifest.missing ? -1 : dayIndex(data.manifest.issue_date, state.date));
  });
  renderLegend();
}
