// Floods past (#547): where floods were reported in the news and seen by
// radar, on the globe, following the time bar. On from the start, showing
// the twelve months up to the map date (or the latest twelve on record);
// playing the time bar replays them month by month, and a range shows its
// months together. Clicking a cell lists its events, from Python
// (aquascope.context.floods_past, the same function as the CLI and the MCP
// tool) in a light worker.
//
// The months are small gzipped JSON files in the Archive, one per month,
// written by the mirror-context workflow. The drawing rules and the words are
// in floods-past-core.js, which node tests.

import { $, actions, clickLayers, datedExtras, escapeHtml, onTime, setTime, state, timeState } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { syncTimeBar } from "./time-ui.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { callLight } from "./worker-client.js?v=__BUILD__";
import {
  CELLS, CELL_MINZOOM, DARK_BASEMAPS, FLOODS_PAST_BASE, HEAT_FADE, HEAT_MAXZOOM, MAX_MONTHS, NEWS_COLOR, NEWS_STROKE, POINTS, RADAR_COLOR,
  RADAR_PERIOD, cellBbox, cellsGeoJSON, eventDates, fmtCount, legendLines, monthLabel, monthsBetween,
  monthsOnRecord, newsFill, newsHeat, newsRadius, newsStroke, newsWeight, placeLabel, radarFill, radarHeat, radarWeight,
  readFloodsParam, shortWhen, standoutFilters, standoutThresholds, windowFor, windowLabel, areaLabel,
} from "./floods-past-core.js?v=__BUILD__";

const SLOTS = ["a", "b"];
const LAYER_IDS = [...SLOTS.flatMap((s) => [`fp-heat-radar-${s}`, `fp-heat-news-${s}`, `fp-radar-${s}`, `fp-news-${s}`]),
  "fp-sel-line"];
const OPACITY = { news: 1, stroke: 1, newsHeat: 0.9, radarHeat: 0.85 };
// The heat hands over to the marks between zoom 6 and 7; the slot's crossfade factor k scales both.
const heatOpacity = (kind, k) => ["interpolate", ["linear"], ["zoom"], HEAT_FADE[0], OPACITY[`${kind}Heat`] * k, HEAT_FADE[1], 0];
const markOpacity = (o, k) => ["interpolate", ["linear"], ["zoom"], CELL_MINZOOM, 0, CELL_MINZOOM + 0.6, o * k];
const EMPTY = { type: "FeatureCollection", features: [] };

let index;                 // undefined: not read yet; null: not published; else index.json
let indexLoading = null;
const monthFiles = new Map();   // month -> Promise<cells>
let front = "a";
let drawn = "";            // the months on screen, joined
let drawToken = 0;
let current = { months: [], mode: "window", latest: false };
let shown = null;          // the GeoJSON on the front slot, re-added after a basemap change
let popup = null;
let popupRun = 0;

const narrow = () => Boolean(globalThis.matchMedia && matchMedia("(max-width: 520px)").matches);
const still = () => Boolean(globalThis.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
const fadeMs = () => (still() ? 0 : 480);

// ── data ────────────────────────────────────────────────────────────────────

async function gunzipJson(resp) {
  if (typeof DecompressionStream === "undefined") throw new Error("this browser cannot read gzip");
  const stream = resp.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

function loadIndex() {
  if (index !== undefined) return Promise.resolve(index);
  if (!indexLoading) {
    indexLoading = fetch(`${FLOODS_PAST_BASE}index.json`)
      .then((r) => (r.ok ? r.json() : null))
      .catch((err) => { console.info("floods past unavailable:", err && err.message); return null; })
      .then((v) => { index = v; return v; });
  }
  return indexLoading;
}

function loadMonth(month) {
  if (!monthFiles.has(month)) {
    const p = fetch(`${FLOODS_PAST_BASE}months/${month}.json.gz`)
      .then((r) => (r.ok ? gunzipJson(r) : { cells: [] }))
      .then((j) => j.cells || [])
      .catch((err) => { monthFiles.delete(month); throw err; });
    monthFiles.set(month, p);
  }
  return monthFiles.get(month);
}

function prefetch(range) {
  if (!index || !state.floodsPast || !range) return;
  const months = monthsOnRecord(index, monthsBetween(range.from, range.to)).slice(-MAX_MONTHS);
  for (const m of months) loadMonth(m).catch(() => {});
}

// ── layers ──────────────────────────────────────────────────────────────────

// Under the rivers and Floods ahead (their ids start "river-") and the gauges: the order on the map is
// basemap, river status, Floods past, rivers, Floods ahead, gauges (#543).
function beforeGauges() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => /^river-/.test(l.id) || ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

// The basemap's first label layer: the radar cells go under it, so place names stay crisp on top.
function belowLabels() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => l.type === "symbol" && !/^(points|selected|cluster-count|river-|study-|fp-)/.test(l.id));
  return hit ? hit.id : undefined;
}

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getSource("fp-a") && map.getSource("fp-b")) return true;
  const before = beforeGauges();
  const t = { duration: fadeMs(), delay: 0 };
  for (const s of SLOTS) {
    if (!map.getSource(`fp-${s}`)) map.addSource(`fp-${s}`, { type: "geojson", data: s === front && shown ? shown : EMPTY });
    const k = s === front && shown ? 1 : 0;
    // The world view is a heat map, so twelve months of reports read as hotspots rather than confetti;
    // closer in, radar shades its half-degree cells and news is a circle per cell, and a click lands on either.
    map.addLayer({
      id: `fp-radar-${s}`, type: "fill", source: `fp-${s}`, minzoom: CELL_MINZOOM, filter: CELLS,
      paint: { "fill-color": radarFill(DARK_BASEMAPS.has(state.basemap)), "fill-opacity": markOpacity(1, k), "fill-opacity-transition": t, "fill-antialias": false },
    }, belowLabels() || before);
    // News under radar at the world view: the reports are everywhere people are, the radar glow is where
    // the water was, and it should not be buried under the reports.
    for (const [kind, heat] of [["news", newsHeat(heatScale)], ["radar", radarHeat(heatScale)]]) {
      map.addLayer({
        id: `fp-heat-${kind}-${s}`, type: "heatmap", source: `fp-${s}`, maxzoom: HEAT_MAXZOOM,
        filter: ["all", POINTS, [">", ["get", kind], 0]],
        paint: { ...heat, "heatmap-opacity": heatOpacity(kind, k), "heatmap-opacity-transition": t },
      }, before);
    }
    map.addLayer({
      id: `fp-news-${s}`, type: "circle", source: `fp-${s}`, minzoom: CELL_MINZOOM,
      filter: ["all", POINTS, [">", ["get", "news"], 0]],
      layout: { "circle-sort-key": ["-", 0, ["get", "news"]] },
      paint: {
        "circle-color": newsFill(), "circle-radius": newsRadius(),
        "circle-opacity": markOpacity(OPACITY.news, k), "circle-opacity-transition": t,
        "circle-stroke-color": newsStroke(), "circle-stroke-width": 0.6,
        "circle-stroke-opacity": markOpacity(OPACITY.stroke, k), "circle-stroke-opacity-transition": t,
      },
    }, before);
  }
  // The clicked cell, outlined while its popup is open.
  if (!map.getSource("fp-sel")) {
    map.addSource("fp-sel", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "fp-sel-line", type: "line", source: "fp-sel",
      paint: { "line-color": "#10222f", "line-width": 1.6, "line-opacity": 0.85 } }, before);
  }
  for (const s of SLOTS) for (const kind of ["radar", "news"]) clickLayers.add(`fp-${kind}-${s}`);
  setVisible(state.floodsPast);
  return true;
}

function setVisible(on) {
  if (!state.mapOk || !map) return;
  for (const id of LAYER_IDS) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
}

function setSlotOpacity(slot, on) {
  const k = on ? 1 : 0;
  map.setPaintProperty(`fp-heat-radar-${slot}`, "heatmap-opacity", heatOpacity("radar", k));
  map.setPaintProperty(`fp-heat-news-${slot}`, "heatmap-opacity", heatOpacity("news", k));
  map.setPaintProperty(`fp-radar-${slot}`, "fill-opacity", markOpacity(1, k));
  map.setPaintProperty(`fp-news-${slot}`, "circle-opacity", markOpacity(OPACITY.news, k));
  map.setPaintProperty(`fp-news-${slot}`, "circle-stroke-opacity", markOpacity(OPACITY.stroke, k));
}

let heatScale = 12;
function setHeatScale(n) {
  const months = Math.max(1, n || 1);
  if (months === heatScale || !ensureLayers()) return;
  heatScale = months;
  for (const s of SLOTS) {
    map.setPaintProperty(`fp-heat-news-${s}`, "heatmap-weight", newsWeight(months));
    map.setPaintProperty(`fp-heat-radar-${s}`, "heatmap-weight", radarWeight(months));
  }
}

// ── only what stands out (floods-past-core.js standoutThresholds) ──────────

let shownCells = [];
function cellList(fc) {
  return ((fc && fc.features) || []).filter((f) => f.geometry.type === "Point")
    .map((f) => ({ lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1], news: f.properties.news, radar: f.properties.radar }));
}

// The region is what is on screen; on the world view, the whole world.
function viewBox() {
  if (!map || map.getZoom() < 3) return null;
  const b = map.getBounds();
  return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v, i) => (i % 2 ? v : ((v + 540) % 360) - 180));
}

function applyStandout(slot = front) {
  if (!state.mapOk || !map || !map.getLayer(`fp-news-${slot}`)) return;
  const f = standoutFilters(standoutThresholds(shownCells, viewBox()));
  map.setFilter(`fp-news-${slot}`, f.news);
  map.setFilter(`fp-heat-news-${slot}`, f.news);
  map.setFilter(`fp-heat-radar-${slot}`, f.radarHeat);
  map.setFilter(`fp-radar-${slot}`, f.radarCells);
}

// Wait until the back slot has taken its new data, so the fade shows it rather than an empty frame.
function whenLoaded(sourceId, ms = 700) {
  return new Promise((resolve) => {
    let timer = null;
    const done = () => { clearTimeout(timer); map.off("sourcedata", onData); resolve(); };
    const onData = (e) => { if (e.sourceId === sourceId && e.isSourceLoaded) done(); };
    timer = setTimeout(done, ms);
    map.on("sourcedata", onData);
  });
}

// Crossfade to the new months: the data goes into the hidden slot, then the two swap opacity.
async function show(fc) {
  if (!ensureLayers()) return;
  const back = front === "a" ? "b" : "a";
  map.getSource(`fp-${back}`).setData(fc);
  shown = fc;
  shownCells = cellList(fc);
  applyStandout(back);
  await whenLoaded(`fp-${back}`);
  setSlotOpacity(back, true);
  setSlotOpacity(front, false);
  front = back;
}

// ── following the time bar ──────────────────────────────────────────────────

async function redraw() {
  if (!state.floodsPast) { renderLegend(); return; }
  const idx = await loadIndex();
  if (!state.floodsPast) return;
  const win = windowFor(timeState(), idx ? (idx.news && idx.news.last ? idx.news.last.slice(0, 7) : idx.last) : null);
  current = win;
  syncDated(idx);
  renderLegend();
  if (!idx) return;
  const months = monthsOnRecord(idx, win.months);
  const key = months.join(",");
  if (key === drawn && map && map.getSource(`fp-${front}`)) return;
  const my = ++drawToken;
  setBusy(true);
  try {
    const cells = await Promise.all(months.map(loadMonth));
    if (my !== drawToken) return;
    drawn = key;
    setHeatScale(months.length);
    await show(cellsGeoJSON(cells, idx.deg || 0.5));
  } catch (err) {
    console.warn("floods past: could not read a month", err);
    if (my === drawToken) setNote("Could not read this month's floods. Try again in a moment.");
  } finally {
    if (my === drawToken) setBusy(false);
  }
}

// The time bar shows itself for a dated layer; Floods past is one while it is on.
// A date after the record shows the latest twelve months, which the legend says,
// so the bar's "only" note is kept for when the layer really has nothing to show.
function syncDated(idx) {
  const had = datedExtras.get("floods-past");
  if (state.floodsPast && idx) {
    datedExtras.set("floods-past", {
      id: "floods-past", label: "Floods past", time: true, monthly: true,
      since: `${idx.first}-01`, until: current.latest ? undefined : `${idx.last}-01`,
    });
  } else {
    datedExtras.delete("floods-past");
  }
  const now = datedExtras.get("floods-past");
  if (Boolean(had) !== Boolean(now)) syncTimeBar({ layersChanged: true });
  else if (now && had.until !== now.until) syncTimeBar();
}

export function setFloodsPast(on, { write = true } = {}) {
  state.floodsPast = Boolean(on);
  const toggle = $("toggle-floods-past");
  if (toggle) toggle.checked = state.floodsPast;
  setVisible(state.floodsPast);
  if (!state.floodsPast) {
    closePopup();
    syncDated(index);
  }
  void redraw();
  if (write) writeUrl();
  renderCredits();
}

// ── the row in "On the map" (map-legend.js) ─────────────────────────────────

let noteText = "";
let busy = false;
function setNote(text) { noteText = text || ""; renderLegend(); }
function setBusy(on) { if (busy !== Boolean(on)) { busy = Boolean(on); renderLegend(); } }

const dot = `<svg class="fp-key" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="4.6" fill="${NEWS_COLOR}" stroke="${NEWS_STROKE}" stroke-opacity=".6" stroke-width="1"/></svg>`;
// Radar is a shaded cell on the map, so its key is a square.
const glow = `<svg class="fp-key" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="2" ` +
  `fill="${RADAR_COLOR}" fill-opacity=".55" stroke="${RADAR_COLOR}" stroke-opacity=".8"/></svg>`;
const pair = `<svg class="fp-key" viewBox="0 0 22 16" aria-hidden="true"><rect x="9" y="3" width="11" height="11" rx="2" ` +
  `fill="${RADAR_COLOR}" fill-opacity=".5"/><circle cx="7" cy="8" r="4.4" fill="${NEWS_COLOR}" stroke="#fff" stroke-width="1.2"/></svg>`;

function renderLegend() {
  refreshLegend("floods-past");
  if (index === null) {
    // Until the mirror-context workflow has published the grid there is nothing to draw; the rail says so too.
    const label = $("toggle-floods-past") && $("toggle-floods-past").parentElement.querySelector(".rail-label");
    if (label && !label.querySelector(".muted")) label.insertAdjacentHTML("beforeend", ' <span class="muted">(not published yet)</span>');
  }
}

function rowSummary() {
  if (index === undefined) return "loading";
  if (index === null) return "not published yet";
  return shortWhen(current) + (busy ? ", loading" : "");
}

function rowBody() {
  const l = legendLines(index, current);
  if (!l) return "";
  const playing = state.playing && current.mode === "frame";
  return `<p class="ml-when" aria-live="polite">${escapeHtml(l.when)}</p>` +
    `<div class="fp-row" title="Flood events extracted from news articles: somewhere a flood was reported">${dot}` +
    `<span class="fp-what">Reported in the news</span><span class="fp-n">${escapeHtml(l.news)}</span></div>` +
    `<div class="fp-row" title="Sentinel-1 radar pixels classified as flood water: water seen from space">${glow}` +
    `<span class="fp-what">Seen by radar</span><span class="fp-n">${escapeHtml(l.radar)}</span></div>` +
    `<p class="ml-src">Drawn where a place stands out from its region. ` +
    `Groundsource, CC BY 4.0; Microsoft, MIT.</p>` +
    (noteText ? `<p class="fp-note">${escapeHtml(noteText)}</p>` : "") +
    `<div class="ml-actions"><button type="button" class="ml-btn" data-act="play" aria-pressed="${playing ? "true" : "false"}">` +
    `${playing ? "Pause" : "Replay month by month"}</button>` +
    `<button type="button" class="ml-btn quiet" data-act="about">About</button></div>`;
}

function registerRow() {
  registerLegendRow({
    id: "floods-past", title: "Floods past",
    mark: () => pair,
    summary: rowSummary,
    on: () => Boolean(state.floodsPast && state.mapOk),
    empty: () => index === null || index === undefined,
    toggle: (on) => setFloodsPast(on),
    body: rowBody,
    act: (name) => { if (name === "play") replay(); else if (name === "about") aboutModal(); },
  });
}

// Replay the months on screen through the time bar: a month step, the window as the range, then play.
function replay() {
  const tbPlay = $("tb-play");
  if (state.playing) { if (tbPlay) tbPlay.click(); return; }
  const months = current.months.length ? current.months : [];
  if (!months.length || !tbPlay) return;
  setTime({ step: "month", range: { from: `${months[0]}-01`, to: `${months[months.length - 1]}-01` },
    date: `${months[0]}-01` }, { source: "bar" });
  tbPlay.click();
}

function aboutModal() {
  const radar = `${monthLabel(RADAR_PERIOD[0])} to ${monthLabel(RADAR_PERIOD[1])}`;
  const news = index && index.news ? `${monthLabel(index.news.first)} to ${monthLabel(index.news.last)}` : "";
  openModal("Floods past", `
    <p><b style="color:${NEWS_COLOR}">Reported in the news</b>: flood events that Google's Groundsource extracted
      from news articles (${escapeHtml(news)}). Each is counted once, in the month it began, at the centre of the
      area it affected. A report says a flood happened; it does not measure it, and places with more news
      coverage have more reports.</p>
    <p><b style="color:${RADAR_COLOR}">Seen by radar</b>: 20 m Sentinel-1 pixels the Microsoft AI for Good Lab
      classified as flood water, after the dataset's own filters against false alarms, ${escapeHtml(radar)} only.
      Radar sees water through cloud, but not under forest or between buildings.</p>
    <p>Both are summed per half-degree cell (about 55 km) and month. The circles grow with the logarithm of the
      count. The time bar picks the months: the twelve up to the map date, the month being played, or the range
      you set. Click a circle for its events.</p>
    <p class="muted">Groundsource: Mayo, R. et al. (2026), doi:10.5281/zenodo.18647054, CC BY 4.0.
      Microsoft: Misra, A. et al. (2025), Nature Communications 16, 5762, MIT licence.
      Rolled up by the AquaScope Archive (context/floods/monthly).</p>`);
}

// ── a click on a cell ───────────────────────────────────────────────────────

function outlineCell(bbox) {
  const src = state.mapOk && map && map.getSource("fp-sel");
  if (!src) return;
  if (!bbox) { src.setData(EMPTY); return; }
  const [w, s, e, n] = bbox;
  src.setData({ type: "FeatureCollection", features: [{ type: "Feature", properties: {},
    geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] } }] });
}

function closePopup() {
  popupRun++;
  if (popup) { const p = popup; popup = null; p.remove(); }
  outlineCell(null);
}

function cellCard(p, months, body) {
  const [w, s] = cellBbox(p.row, p.col, (index && index.deg) || 0.5);
  const d = (index && index.deg) || 0.5;
  const lat = s + d / 2, lon = w + d / 2;
  return `<div class="fp-pop">` +
    `<div class="fp-pop-head"><strong>${escapeHtml(windowLabel(months))}</strong>` +
    `<span class="muted">${escapeHtml(placeLabel(lat, lon))}</span></div>` +
    `<div class="fp-pop-counts">` +
    `<span>${dot}${p.news ? `${fmtCount(p.news)} in the news` : "none in the news"}</span>` +
    `<span>${glow}${p.radar ? `${fmtCount(p.radar)} radar detections` : "no radar detections"}</span></div>` +
    body + `</div>`;
}

function eventsHtml(res, months) {
  if (!res || res.error) return `<p class="fp-pop-note muted">${escapeHtml((res && res.error) || "Could not read the events.")}</p>`;
  const news = res.news || {};
  const events = news.events || [];
  let html = "";
  if (events.length) {
    const more = (news.events_found || 0) - events.length;
    html += `<ul class="fp-events">${events.map((e) => `<li><span>${escapeHtml(eventDates(e.start, e.end))}</span>` +
      `<span class="muted">${areaLabel(e.area_km2)}</span></li>`).join("")}</ul>`;
    if (more > 0) html += `<p class="fp-pop-note muted">and ${fmtCount(more)} more news events</p>`;
  }
  const byMonth = (res.radar && res.radar.by_month) || {};
  const radarMonths = Object.entries(byMonth).filter(([, n]) => n > 0);
  if (radarMonths.length) {
    html += `<p class="fp-pop-note">Radar saw flooding in ${radarMonths.length === months.length && months.length > 1 ? "every month" :
      radarMonths.map(([m]) => monthLabel(m)).slice(-6).join(", ")}${radarMonths.length > 6 && radarMonths.length !== months.length ? " and more" : ""}.</p>`;
  }
  html += `<p class="fp-pop-src muted">News: Groundsource (CC BY 4.0). Radar: Microsoft Sentinel-1 (MIT).</p>`;
  return html;
}

let lastClick = null;
async function onCellClick(e) {
  // A cell with both marks hears one click twice (once per layer): answer it once.
  if (e.originalEvent && e.originalEvent === lastClick) return;
  lastClick = e.originalEvent || null;
  // A gauge under the click wins: it has its own handler.
  if (map.queryRenderedFeatures(e.point, { layers: ["points", "clusters"].filter((id) => map.getLayer(id)) }).length) return;
  const f = (e.features || [])[0];
  if (!f) return;
  const p = f.properties || {};
  const months = monthsOnRecord(index, current.months);
  if (!months.length) return;
  closePopup();
  const my = ++popupRun;
  const deg = (index && index.deg) || 0.5;
  const [w0, s0] = cellBbox(p.row, p.col, deg);
  const coords = [w0 + deg / 2, s0 + deg / 2];
  // On a phone the map is a strip above the sheet: bring the cell up near its top and open the card below it.
  const phone = narrow();
  if (phone) map.panBy([0, e.point.y - 18], { duration: still() ? 0 : 300 });
  popup = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: phone ? "min(300px, calc(100vw - 32px))" : "300px",
    className: "fp-popup", offset: 6, ...(phone ? { anchor: "top" } : {}) })
    .setLngLat(coords)
    .setHTML(cellCard(p, months, '<p class="fp-pop-note muted" role="status">Reading the events…</p>'))
    .addTo(map);
  const opened = popup;
  document.body.classList.add("fp-pop-open");
  popup.on("close", () => {
    document.body.classList.remove("fp-pop-open");
    if (popup === opened) { popup = null; popupRun++; outlineCell(null); }
  });
  const bbox = cellBbox(p.row, p.col, (index && index.deg) || 0.5);
  outlineCell(bbox);
  let res;
  try {
    res = await callLight("context", { op: "floods_month", start: months[0], end: months[months.length - 1], bbox });
  } catch (err) {
    res = { error: `Could not read the events: ${err.message}` };
  }
  if (my !== popupRun || !popup) return;
  popup.setHTML(cellCard(p, months, eventsHtml(res, months) +
    '<button class="btn tiny fp-open" type="button">Open this place</button>'));
  const btn = popup.getElement() && popup.getElement().querySelector(".fp-open");
  if (btn) btn.addEventListener("click", () => { closePopup(); actions.selectPoint(coords[1], coords[0], { fly: false }); });
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initFloodsPast() {
  registerRow();
  const fromUrl = readFloodsParam(location.hash);
  if (fromUrl !== null) state.floodsPast = fromUrl;
  const toggle = $("toggle-floods-past");
  if (toggle) {
    toggle.checked = state.floodsPast;
    toggle.addEventListener("change", (e) => setFloodsPast(e.target.checked));
  }
  if (!state.mapOk || !map) return;
  // A basemap change replaces the style and drops our layers: put them back with what they showed.
  map.on("style.load", () => { if (ensureLayers()) { setVisible(state.floodsPast); applyStandout(); } });
  // A new region on screen has its own standouts.
  map.on("moveend", () => { if (state.floodsPast && shownCells.length) applyStandout(); });
  for (const s of SLOTS) {
    for (const kind of ["radar", "news"]) {
      const id = `fp-${kind}-${s}`;
      map.on("click", id, (e) => { if (s === front) void onCellClick(e); });
      map.on("mouseenter", id, () => { if (s === front) map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
  }
  onTime((t) => {
    const moved = t.date !== t.prev.date || t.playing !== t.prev.playing || t.step !== t.prev.step ||
      JSON.stringify(t.range) !== JSON.stringify(t.prev.range);
    // A range is what a play or a GIF walks: fetch its months now, so each frame is drawn before it is shown.
    if (t.range && JSON.stringify(t.range) !== JSON.stringify(t.prev.range)) prefetch(t.range);
    if (moved) { closePopupIfStale(); void redraw(); }
  });
  window.addEventListener("hashchange", () => {
    const want = readFloodsParam(location.hash);
    const next = want === null ? true : want;
    if (next !== state.floodsPast) setFloodsPast(next, { write: false });
  });
  ensureLayers();
  void redraw();
}

// A popup belongs to the months it was opened for.
function closePopupIfStale() {
  if (!popup) return;
  closePopup();
}

export const floodsPastWindow = () => ({ ...current, label: windowLabel(current.months) });
