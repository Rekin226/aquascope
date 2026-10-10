// World river status on the map (#544): GEOGLOWS's monthly HydroSOS map of
// every river basin, drawn under the gauges, on by default at the newest month
// and replayed month by month through the time bar.
//
// Each month is one GeoTIFF of the whole world. A worker decodes its red band
// and paints the classes on a Web Mercator square (status-worker.js with
// status-core.js), and MapLibre stretches that picture over the world, globe
// or flat, as an image source.
// The date is the time bar's (state.date): this module only listens.

import { $, actions, escapeHtml, onTime, setTime, state, trace } from "./core.js?v=__BUILD__";
import { registerDatedLayer } from "./layers.js?v=__BUILD__";
import { holdSettle, map } from "./map.js?v=__BUILD__";
import { STATUS_CLASSES } from "./now-core.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { openLegendRow, refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { syncTimeBar } from "./time-ui.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";
import { defaultRange, frameDates, nextFrame, normaliseRange } from "./timeline.js?v=__BUILD__";
import {
  GEOTIFF_MODULE, STATUS_CORNERS, STATUS_CREDIT, STATUS_LIST_URL, decodeStatus, gridToPng, latestDay, missingMonths,
  CLASS_ALPHA, focusPalette, hexRgb, monthLabel, parseListing, statusDatedLayer, statusHeadline, statusMonthFor,
  statusUrl,
} from "./status-core.js?v=__BUILD__";

const SOURCE_ID = "status-src";
const LAYER_ID = "status-layer";
const DEFAULT_OPACITY = 0.85;
// Months kept ready to show, as PNGs of a few hundred kB: the one on screen,
// the next one while playing, and a few either side for stepping back.
const KEEP = 8;

let months = null;          // every month the bucket has, oldest first, once listed
let listing = null;
let shown = null;           // the month on the map
let wanted = null;          // the month the date asks for
let failed = null;          // { month } when the last one could not be read
let side = 2048;            // the picture's width and height
let worker = null;          // a Worker, or false once it has failed
let reqId = 0;
const waiting = new Map();  // worker request id -> { resolve, reject }
const pictures = new Map(); // month (with the focus, #561) -> object URL of its PNG (most recent last)
// Only some classes painted (#561, "where are rivers much above normal"): ids from STATUS_CLASSES; [] is all.
let focus = [];
let shownFocus = "";        // the focus the picture on the map was painted with
const focusKey = () => focus.join(",");
const keyOf = (month, f = focusKey()) => (f ? `${month}|${f}` : month);
const loading = new Map();  // month -> promise of an object URL
const headlines = new Map(); // month -> the caption's one line (status-core.js statusHeadline)

// A power-of-two square, smaller where memory is short.
function pictureSize() {
  const small = Math.min(screen.width || 1024, screen.height || 768) < 700 ||
    (navigator.deviceMemory && navigator.deviceMemory <= 4);
  return small ? 1024 : 2048;
}

// ── which months exist ──────────────────────────────────────────────────────

// One listing of the bucket (about 140 kB, CORS *), for the newest month and the gaps.
function listMonths() {
  if (listing) return listing;
  listing = (async () => {
    const out = [];
    let token = null;
    for (let page = 0; page < 10; page++) {
      const url = token ? `${STATUS_LIST_URL}&continuation-token=${encodeURIComponent(token)}` : STATUS_LIST_URL;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`listing HTTP ${res.status}`);
      const parsed = parseListing(await res.text());
      out.push(...parsed.months);
      token = parsed.token;
      if (!token) break;
    }
    months = [...new Set(out)].sort();
    trace(`status: ${months.length} months, ${months[0]} to ${months[months.length - 1]}`);
    return months;
  })();
  listing.catch((err) => { console.warn("status listing:", err && err.message); listing = null; });
  return listing;
}

// ── making a month's picture ────────────────────────────────────────────────

function inWorker(url, classes) {
  if (!worker) {
    worker = new Worker(new URL("./status-worker.js?v=__BUILD__", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const job = waiting.get(e.data.id);
      if (!job) return;
      waiting.delete(e.data.id);
      if (e.data.noCanvas) worker = false;   // no OffscreenCanvas: make them on the page from now on
      if (e.data.error) job.reject(new Error(e.data.error)); else job.resolve({ png: e.data.png, regions: e.data.regions });
    };
    worker.onerror = (e) => {
      console.warn("status worker:", e && e.message);
      worker = false;
      for (const job of waiting.values()) job.reject(new Error("worker unavailable"));
      waiting.clear();
    };
  }
  const id = ++reqId;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, url, width: side, height: side, focus: classes });
  });
}

async function onPage(url, classes) {
  const geotiff = await import(GEOTIFF_MODULE);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const { grid, regions } = await decodeStatus(geotiff, await res.arrayBuffer(), side, side, { withRegions: true });
  const png = await gridToPng(grid, side, side, (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h }),
    focusPalette(classes));
  return { png, regions };
}

function keep(key, objectUrl) {
  pictures.set(key, objectUrl);
  for (const old of [...pictures.keys()]) {
    if (pictures.size <= KEEP) break;
    if (old === keyOf(shown, shownFocus) || old === keyOf(wanted)) continue;
    URL.revokeObjectURL(pictures.get(old));
    pictures.delete(old);
  }
  return objectUrl;
}

function load(month) {
  const key = keyOf(month), classes = focus.slice();
  if (pictures.has(key)) {
    const u = pictures.get(key);
    pictures.delete(key);
    pictures.set(key, u);    // most recent last
    return Promise.resolve(u);
  }
  if (loading.has(key)) return loading.get(key);
  const url = statusUrl(month);
  const make = worker === false ? onPage(url, classes)
    : inWorker(url, classes).catch((err) => { if (worker === false) return onPage(url, classes); throw err; });
  // The caption reads every class, whatever the focus paints (#561): the regions come from the whole grid.
  const p = make.then(({ png, regions }) => {
    if (regions) headlines.set(month, statusHeadline(month, regions));
    return keep(key, URL.createObjectURL(png));
  }).finally(() => loading.delete(key));
  loading.set(key, p);
  return p;
}

// ── the map layer ───────────────────────────────────────────────────────────

// Under the basemap's water where it has one, so the coast stays crisp and the
// labels stay on top, and under every layer of ours (hillshade, overlays, floods
// past, rivers, catchments, gauges): whichever of these comes first in the style.
function beforeId() {
  const layers = (map.getStyle() && map.getStyle().layers) || [];
  const hit = layers.find((l) => l.id === "water" || l.type === "symbol" || l.id === "hillshade" ||
    /^(ov-|fp-|river-|basins|study-)/.test(l.id) ||
    ["catchment-fill", "gauge-heat", "clusters", "points"].includes(l.id));
  return hit ? hit.id : undefined;
}

// Put a month's picture on the map: a new source the first time (and after a
// basemap change), a new image after that. The old picture stays until the
// new one has loaded, and the layer has no fade, so a step never flashes.
let queued = null;          // a picture waiting for the one before it to land
function draw(objectUrl) {
  if (!map) return;
  const src = map.getSource(SOURCE_ID);
  if (src && src.updateImage) {
    // One image in flight at a time: a second updateImage would abort the first, which MapLibre reports as an error.
    if (src.url !== objectUrl) {
      if (src.loaded()) src.updateImage({ url: objectUrl });
      else queued = objectUrl;
    }
  } else {
    map.addSource(SOURCE_ID, { type: "image", url: objectUrl, coordinates: STATUS_CORNERS });
    map.addLayer({
      id: LAYER_ID, type: "raster", source: SOURCE_ID,
      paint: {
        "raster-opacity": opacityByZoom(),
        "raster-fade-duration": 0,
        "raster-resampling": "linear",
      },
    }, beforeId());
  }
  if (map.getLayer(LAYER_ID)) map.setLayoutProperty(LAYER_ID, "visibility", "visible");
}

// Full strength on the world view, fading as you zoom in, where the basemap,
// the rivers and the gauges are what you came for and a basin is the whole screen.
function opacityByZoom() {
  const o = state.opacity.status ?? DEFAULT_OPACITY;
  return ["interpolate", ["linear"], ["zoom"], 3, o, 6, o * 0.55, 9, o * 0.3];
}

function hide() {
  if (map && map.getLayer(LAYER_ID)) map.setLayoutProperty(LAYER_ID, "visibility", "none");
}

function removeLayer() {
  if (!map) return;
  if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID);
  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);
}

// Draw the month the map date asks for. With no map for that month (before
// 1990, a gap, after the newest), nothing is drawn and the legend says why.
function update() {
  if (!state.status || !state.mapOk || !months) { renderLegend(); return; }
  const month = statusMonthFor(state.date, months);
  wanted = month;
  failed = null;
  if (!month) {
    shown = null;
    hide();
    renderLegend();
    caption();
    return;
  }
  if (month === shown && shownFocus === focusKey() && map.getSource(SOURCE_ID)) { renderLegend(); preload(); return; }
  renderLegend();
  const painted = focusKey();
  const job = load(month).then((objectUrl) => {
    if (wanted !== month || !state.status || painted !== focusKey()) return;
    shown = month;
    shownFocus = painted;
    draw(objectUrl);
    renderLegend();
    caption();
    preload();
  }).catch((err) => {
    console.warn(`status ${month}:`, err && err.message);
    if (wanted === month) { failed = { month }; renderLegend(); }
  });
  holdSettle(job);   // a play waits for the frame (map.js whenSettled)
}

// While playing, start reading the frame after this one, so it is ready when the bar moves.
function preload() {
  if (!state.playing || !months) return;
  const range = normaliseRange(state.timeRange) || defaultRange(state.date, state.timeStep);
  const { dates } = frameDates(range.from, range.to, state.timeStep);
  const next = statusMonthFor(nextFrame(dates, state.date), months);
  if (next && next !== shown) load(next).catch(() => {});
}

// ── the row in "On the map" (map-legend.js) ─────────────────────────────────

// A swatch as the map draws it: the class colour at the strength it is painted with.
function swatchColor(c, i) {
  const [r, g, b] = hexRgb(c.color);
  const a = (CLASS_ALPHA[i + 1] / 255) * (state.opacity.status ?? DEFAULT_OPACITY);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

// The five classes, much below to much above: the row's mark (small) and its key (wide).
// A class left off by a focus (#561) is a faint swatch.
const statusBar = (cls) => `<span class="${cls}">${STATUS_CLASSES.map((c, i) =>
  `<i style="--c:${swatchColor(c, i)}" title="${escapeHtml(c.label)}"${focus.length && !focus.includes(c.id) ? ' class="off"' : ""}></i>`).join("")}</span>`;

function rowMonth() {
  return wanted || shown;
}

// The few words beside the name: the month on the map, or why there is none.
function rowSummary() {
  if (!months) return "loading";
  const latest = months[months.length - 1];
  const month = rowMonth();
  if (month) {
    const busy = month !== shown && !failed;
    const only = focus.length ? `, only ${STATUS_CLASSES.filter((c) => focus.includes(c.id)).map((c) => c.label.toLowerCase()).join(" or ")}` : "";
    return `${monthLabel(month)}${only}${busy ? ", loading" : ""}` +
      `${failed && failed.month === month ? ", could not read" : ""}`;
  }
  const ym = String(state.date || "").slice(0, 7);
  const gap = months.length && ym > months[0] && ym < latest && missingMonths(months).includes(ym);
  return gap ? `no map for ${monthLabel(ym)}` : `${monthLabel(months[0])} to ${monthLabel(latest)} only`;
}

// Only some classes painted (Ask the map, #561): which, and a way to put the rest back.
function focusLine() {
  if (!focus.length) return "";
  const names = STATUS_CLASSES.filter((c) => focus.includes(c.id)).map((c) => c.label).join(" or ");
  return `<p class="sl-focus">Only ${escapeHtml(names)} ` +
    '<button type="button" class="ml-link" data-act="all">show all</button></p>';
}

function rowBody() {
  return statusBar("sl-bar") +
    '<div class="sl-ends"><span>much below</span><span>normal</span><span>much above</span></div>' + focusLine() +
    '<p class="ml-src">Each basin\'s monthly flow against its normal. Modelled: GEOGLOWS, CC BY 4.0. ' +
    '<button type="button" class="ml-link" data-act="info">About</button></p>';
}

function renderLegend() {
  refreshLegend("status");
}

// The first view's one line, over the globe (#map-caption, beside the click hint): where the rivers are low
// or high in the month on the map, made from the file itself. Empty while the layer is off or has no month.
function caption() {
  const el = $("map-caption");
  if (!el) return;
  const text = state.status && shown ? headlines.get(shown) || "" : "";
  // "River status, September 2026:" leads, quietly bold; the rest is the news.
  const cut = text.indexOf(": ");
  el.innerHTML = cut > 0 ? `<b>${escapeHtml(text.slice(0, cut + 1))}</b> ${escapeHtml(text.slice(cut + 2))}` : escapeHtml(text);
  el.hidden = !text;
}

function registerRow() {
  registerLegendRow({
    id: "status", title: "River status",
    mark: () => statusBar("sl-bar mini"),
    summary: rowSummary,
    on: () => Boolean(state.status && state.mapOk),
    empty: () => Boolean(months && !rowMonth()),
    toggle: (on) => chooseStatus(on),
    body: rowBody,
    act: (name) => {
      if (name === "info") openAbout();
      else if (name === "all") setStatusFocus([]);
    },
  });
}

function openAbout() {
  const latest = months && months[months.length - 1];
  const gaps = months ? missingMonths(months) : [];
  const swatches = STATUS_CLASSES.map((c) =>
    `<span class="sw"><i style="background:${c.color}"></i>${escapeHtml(c.label)}</span>`).join("");
  openModal("World river status", `
    <p>Each river basin is coloured by how its flow that month compares with the same month in other years:
    below the 10th percentile is much below normal, 10th to 25th below, 25th to 75th normal, 75th to 90th above,
    and over the 90th much above.</p>
    <div class="swatches">${swatches}</div>
    <p>The flows are modelled, not measured: GEOGLOWS v2's retrospective simulation, the month's mean flow summed
    at the outlets of each HydroBASINS level-4 basin, against that basin's percentiles for the calendar month
    (<code>hydrosos/thresholds.parquet</code>).
    One colour per basin, so a small river inside a big basin can differ.</p>
    <p>${months ? `${months.length} months, ${escapeHtml(monthLabel(months[0]))} to ${escapeHtml(monthLabel(latest))}` +
      (gaps.length ? `; no map for ${gaps.map((g) => escapeHtml(monthLabel(g))).join(", ")}` : "") + "." : ""}
    Move the time bar, or play it with the step set to a month, to watch the months go by.</p>
    <p class="muted">GEOGLOWS draws these classes in the WMO HydroSOS colours, red to blue. AquaScope uses its own
    brown to teal (safe for colour-blind readers), the same as the gauges' "Today vs normal", so a dot and the
    basin around it mean the same thing.</p>
    <p class="muted">${STATUS_CREDIT.attribution}. Licence: ${escapeHtml(STATUS_CREDIT.licence)}.
    How it is made: <a href="https://github.com/geoglows/rfs-v2-retrospective-update" target="_blank" rel="noopener">
    monthly_products.py</a>. Read in the browser with geotiff.js (MIT).
    The same in Python: <code>aquascope layers status</code>.</p>`);
}

// ── the rail row ────────────────────────────────────────────────────────────

function buildRailRow() {
  const check = $("toggle-status");
  if (!check) return;
  check.addEventListener("change", (e) => chooseStatus(e.target.checked));
  $("status-opacity").addEventListener("input", (e) => {
    state.opacity.status = Number(e.target.value);
    if (map && map.getLayer(LAYER_ID)) map.setPaintProperty(LAYER_ID, "raster-opacity", opacityByZoom());
    renderLegend();
  });
  $("status-info").addEventListener("click", (e) => { e.preventDefault(); openAbout(); });
  syncRailRow();
}

function syncRailRow() {
  const check = $("toggle-status");
  if (!check) return;
  check.checked = Boolean(state.status);
  check.closest(".overlay-row").querySelector(".overlay-controls").hidden = !state.status;
  $("status-opacity").value = state.opacity.status ?? DEFAULT_OPACITY;
}

// ── on and off ──────────────────────────────────────────────────────────────

/** Show or hide the layer to match `on` (from the URL, Back, or the rail). */
export function setStatusVisible(on) {
  state.status = Boolean(on);
  if (!state.mapOk || !map) return;
  if (state.status) {
    if (shown && pictures.has(keyOf(shown, shownFocus))) draw(pictures.get(keyOf(shown, shownFocus)));
    listMonths().then(() => { syncTimeBar({ layersChanged: true }); update(); }).catch(() => renderLegend());
  } else {
    removeLayer();
  }
  caption();
  syncRailRow();
  renderLegend();
}

// The reader's own choice, from the rail or the legend: also the credits, the bar and the link.
function chooseStatus(on) {
  setStatusVisible(on);
  // Monthly maps want a monthly step: a day step would replay the same map twelve times.
  if (on && state.timeStep === "day") setTime({ step: "month" }, { source: "layer" });
  renderCredits();
  syncTimeBar({ layersChanged: true });
  writeUrl();
}

/**
 * Paint only some classes (ids from STATUS_CLASSES, such as ["much_above"]); [] paints them all again (#561).
 * The map answers "where are rivers much above normal" by itself; the legend says what is left out.
 */
export function setStatusFocus(classes = []) {
  const ids = new Set(STATUS_CLASSES.map((c) => c.id));
  focus = STATUS_CLASSES.map((c) => c.id).filter((id) => (classes || []).includes(id) && ids.has(id));
  update();
  renderLegend();
  if (focus.length) openLegendRow("status", true);   // the key says what is left out, with "show all"
  return focus.slice();
}

export const getStatusFocus = () => focus.slice();

export function initStatusLayer(url = {}) {
  side = pictureSize();
  actions.setStatus = setStatusVisible;
  registerDatedLayer(() => (state.status ? statusDatedLayer(months) : null));
  buildRailRow();
  registerRow();
  // The picture that waited for the one before it (draw()).
  map.on("sourcedata", (e) => {
    if (e.sourceId !== SOURCE_ID || !queued || !e.source || !map.getSource(SOURCE_ID).loaded()) return;
    const next = queued;
    queued = null;
    draw(next);
  });
  // A basemap change replaces the whole style; put the layer back on the new one.
  map.on("style.load", () => {
    const key = shown && keyOf(shown, shownFocus);
    if (state.status && key && pictures.has(key)) draw(pictures.get(key));
  });
  onTime((t) => {
    if (t.date !== t.prev.date || t.playing !== t.prev.playing) update();
  });
  // Open on the newest month: with no date in the link, move the map date into it,
  // and step by a month so play walks the months.
  const bootDate = state.date;
  listMonths().then((list) => {
    if (state.status && !url.date && state.date === bootDate && list.length) {
      setTime({ date: latestDay(list), step: url.step ? undefined : "month" }, { source: "layer" });
    }
    syncTimeBar({ layersChanged: true });
    update();
  }).catch(() => renderLegend());
  renderLegend();
}

