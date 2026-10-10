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
import { callLight } from "./worker-client.js?v=__BUILD__";
import {
  CELLS, CELL_MINZOOM, FLOODS_PAST_BASE, HEAT_MAXZOOM, MAX_MONTHS, NEWS_COLOR, NEWS_STROKE, POINTS, RADAR_COLOR,
  RADAR_PERIOD, cellBbox, cellsGeoJSON, eventDates, fmtCount, legendLines, monthLabel, monthsBetween,
  monthsOnRecord, newsHeat, newsRadius, newsWeight, placeLabel, radarFill, radarHeat, radarWeight,
  readFloodsParam, windowFor, windowLabel,
} from "./floods-past-core.js?v=__BUILD__";

const SLOTS = ["a", "b"];
const LAYER_IDS = [...SLOTS.flatMap((s) => [`fp-heat-radar-${s}`, `fp-heat-news-${s}`, `fp-radar-${s}`, `fp-news-${s}`]),
  "fp-sel-line"];
const OPACITY = { news: 0.85, stroke: 0.6, newsHeat: 0.85, radarHeat: 0.8 };
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

function beforeGauges() {
  for (const id of ["catchment-fill", "gauge-heat", "clusters", "points"]) if (map.getLayer(id)) return id;
  return undefined;
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
      paint: { "fill-color": radarFill(), "fill-opacity": k, "fill-opacity-transition": t, "fill-antialias": false },
    }, belowLabels() || before);
    // News under radar at the world view: the reports are everywhere people are, the radar glow is where
    // the water was, and it should not be buried under the reports.
    for (const [kind, heat] of [["news", newsHeat(heatScale)], ["radar", radarHeat(heatScale)]]) {
      map.addLayer({
        id: `fp-heat-${kind}-${s}`, type: "heatmap", source: `fp-${s}`, maxzoom: HEAT_MAXZOOM,
        filter: ["all", POINTS, [">", ["get", kind], 0]],
        paint: { ...heat, "heatmap-opacity": OPACITY[`${kind}Heat`] * k, "heatmap-opacity-transition": t },
      }, before);
    }
    map.addLayer({
      id: `fp-news-${s}`, type: "circle", source: `fp-${s}`, minzoom: CELL_MINZOOM,
      filter: ["all", POINTS, [">", ["get", "news"], 0]],
      layout: { "circle-sort-key": ["-", 0, ["get", "news"]] },
      paint: {
        "circle-color": NEWS_COLOR, "circle-radius": newsRadius(),
        "circle-opacity": OPACITY.news * k, "circle-opacity-transition": t,
        "circle-stroke-color": NEWS_STROKE, "circle-stroke-width": 0.8,
        "circle-stroke-opacity": OPACITY.stroke * k, "circle-stroke-opacity-transition": t,
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
  map.setPaintProperty(`fp-heat-radar-${slot}`, "heatmap-opacity", OPACITY.radarHeat * k);
  map.setPaintProperty(`fp-heat-news-${slot}`, "heatmap-opacity", OPACITY.newsHeat * k);
  map.setPaintProperty(`fp-radar-${slot}`, "fill-opacity", k);
  map.setPaintProperty(`fp-news-${slot}`, "circle-opacity", OPACITY.news * k);
  map.setPaintProperty(`fp-news-${slot}`, "circle-stroke-opacity", OPACITY.stroke * k);
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
  syncDated(idx);
  const win = windowFor(timeState(), idx ? (idx.news && idx.news.last ? idx.news.last.slice(0, 7) : idx.last) : null);
  current = win;
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
function syncDated(idx) {
  const had = datedExtras.has("floods-past");
  if (state.floodsPast && idx) {
    datedExtras.set("floods-past", {
      id: "floods-past", label: "Floods past", time: true, monthly: true,
      since: `${idx.first}-01`, until: `${idx.last}-01`,
    });
  } else {
    datedExtras.delete("floods-past");
  }
  if (had !== datedExtras.has("floods-past")) syncTimeBar({ layersChanged: true });
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

// ── the legend ──────────────────────────────────────────────────────────────

const legendHost = () => $("fp-legend");

// Folded, the legend is one line (the two marks and the months): how it starts on a phone, where the map is small.
let folded = Boolean(globalThis.matchMedia && matchMedia("(max-width: 520px)").matches);
let noteText = "";
function setNote(text) { noteText = text || ""; renderLegend(); }
function setBusy(on) { const c = $("fp-legend"); if (c) c.classList.toggle("busy", Boolean(on)); }

const dot = `<svg class="fp-key" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="4.6" fill="${NEWS_COLOR}" stroke="${NEWS_STROKE}" stroke-opacity=".6" stroke-width="1"/></svg>`;
// Radar is a shaded cell on the map, so its key is a square.
const glow = `<svg class="fp-key" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="2" ` +
  `fill="${RADAR_COLOR}" fill-opacity=".55" stroke="${RADAR_COLOR}" stroke-opacity=".8"/></svg>`;

function renderLegend() {
  const card = legendHost();
  if (!card) return;
  card.hidden = !state.floodsPast || !state.mapOk;
  if (card.hidden) return;
  if (index === undefined) {
    card.innerHTML = `<div class="fp-head"><h3>Floods past</h3></div><p class="fp-sub muted">Loading…</p>`;
    return;
  }
  if (index === null) {
    // Until the mirror-context workflow has published the grid there is nothing to draw: no card on the map,
    // and the rail says why.
    card.hidden = true;
    const label = $("toggle-floods-past") && $("toggle-floods-past").parentElement.querySelector(".rail-label");
    if (label && !label.querySelector(".muted")) label.insertAdjacentHTML("beforeend", ' <span class="muted">(not published yet)</span>');
    return;
  }
  const l = legendLines(index, current);
  const playing = state.playing && current.mode === "frame";
  card.classList.toggle("folded", folded);
  card.innerHTML =
    `<div class="fp-head"><h3>Floods past</h3>` +
    `<button class="fp-btn fp-play" type="button" aria-pressed="${playing ? "true" : "false"}" ` +
    `title="${playing ? "Pause" : "Replay these months one by one"}" aria-label="${playing ? "Pause the replay" : "Replay month by month"}">` +
    (playing ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.6v14H7zM13.4 5H17v14h-3.6z" fill="currentColor"/></svg>'
      : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg>') +
    `</button><button class="fp-btn fp-about" type="button" title="What the colours mean" aria-label="About Floods past">` +
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>' +
    '<path d="M12 11v6M12 7.5v.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg></button>' +
    `<button class="fp-btn fp-fold" type="button" aria-expanded="${folded ? "false" : "true"}" ` +
    `title="${folded ? "Show the counts and sources" : "Fold the legend"}" aria-label="${folded ? "Show the legend" : "Fold the legend"}">` +
    `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${folded ? "M6 9l6 6 6-6" : "M6 15l6-6 6 6"}" fill="none" ` +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></button>' +
    `${closeBtn()}</div>` +
    `<p class="fp-mini">${dot}<span>News</span>${glow}<span>Radar</span><span class="fp-mini-when">${escapeHtml(l.when)}</span></p>` +
    `<p class="fp-sub" aria-live="polite">${escapeHtml(l.when)}</p>` +
    `<div class="fp-row" title="Flood events extracted from news articles: somewhere a flood was reported">${dot}` +
    `<span class="fp-what">Reported in the news</span><span class="fp-n">${escapeHtml(l.news)}</span></div>` +
    `<div class="fp-row" title="Sentinel-1 radar pixels classified as flood water: water seen from space">${glow}` +
    `<span class="fp-what">Seen by radar</span><span class="fp-n">${escapeHtml(l.radar)}</span></div>` +
    (noteText ? `<p class="fp-note">${escapeHtml(noteText)}</p>` : "") +
    `<p class="fp-foot muted">Groundsource, CC BY 4.0 · Microsoft, MIT</p>`;
  wireLegend(card);
}

const closeBtn = () => '<button class="fp-btn fp-close" type="button" title="Hide Floods past" aria-label="Hide Floods past">' +
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button>';

function wireLegend(card) {
  const close = card.querySelector(".fp-close");
  if (close) close.addEventListener("click", () => setFloodsPast(false));
  const play = card.querySelector(".fp-play");
  if (play) play.addEventListener("click", replay);
  const fold = card.querySelector(".fp-fold");
  if (fold) fold.addEventListener("click", () => { folded = !folded; renderLegend(); card.querySelector(".fp-fold")?.focus(); });
  const about = card.querySelector(".fp-about");
  if (about) about.addEventListener("click", aboutModal);
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
      `<span class="muted">${e.area_km2 ? `${fmtCount(Math.round(e.area_km2))} km²` : ""}</span></li>`).join("")}</ul>`;
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

async function onCellClick(e) {
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
  popup = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: "300px", className: "fp-popup", offset: 6 })
    .setLngLat(coords)
    .setHTML(cellCard(p, months, '<p class="fp-pop-note muted" role="status">Reading the events…</p>'))
    .addTo(map);
  const opened = popup;
  popup.on("close", () => { if (popup === opened) { popup = null; popupRun++; outlineCell(null); } });
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
  const fromUrl = readFloodsParam(location.hash);
  if (fromUrl !== null) state.floodsPast = fromUrl;
  const toggle = $("toggle-floods-past");
  if (toggle) {
    toggle.checked = state.floodsPast;
    toggle.addEventListener("change", (e) => setFloodsPast(e.target.checked));
  }
  if (!state.mapOk || !map) return;
  // A basemap change replaces the style and drops our layers: put them back with what they showed.
  map.on("style.load", () => { if (ensureLayers()) setVisible(state.floodsPast); });
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
