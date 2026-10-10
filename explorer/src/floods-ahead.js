// Floods ahead (#546): the river reaches the GEOGLOWS forecast expects to reach
// their 2-year flow in the next 15 days, on the globe and on by default.
//
// The daily job (flood-warnings.yml, aquascope.archive.warnings) publishes
// forecasts/warnings/latest.geojson (a point per reach with its class, peak
// day and a class for each of the 15 days) and manifest.json. Far out, each
// reach is a soft glow in its class colour; closer in, the river itself lights
// up along the GEOGLOWS stream tiles, and the Archive gauges on those reaches
// pulse gently. The map's date (the time bar, core.js setTime) picks the day:
// inside the forecast's 15 days the reaches show their class on that day,
// otherwise the 15-day peak. Clicking a reach opens a small card on the map.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, clickLayers, escapeHtml, onTime, setTime, sourceStyle, state } from "./core.js?v=__BUILD__";
import { renderCredits } from "./layer-ui.js?v=__BUILD__";
import { refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { STREAMS_PMTILES } from "./river-core.js?v=__BUILD__";
import {
  FLOODS_CREDIT, FLOOD_CLASSES, FORECAST_DAYS, NONE, addDays, classColor, countsFor, dayIndex, gaugesFor, idsByClass,
  issueLine, legendLine, lineColorExpr, lineFilterExpr, pointsFor, reachFacts, shortDay,
} from "./floods-ahead-core.js?v=__BUILD__";

export { FLOODS_CREDIT };

const BASE = `${CONFIG.forecastsBase}warnings/`;
const L = {
  net: "river-fa-net", pts: "river-fa-pts", gauges: "river-fa-gauges",
  glow: "river-fa-glow", casing: "river-fa-casing", line: "river-fa-line", halo: "river-fa-halo", dot: "river-fa-dot",
  ringCasing: "river-fa-ring-casing", ring: "river-fa-ring", pulse: "river-fa-pulse",
};
const ALL = [L.glow, L.casing, L.line, L.halo, L.dot, L.ringCasing, L.ring, L.pulse];
const CLICKABLE = [L.dot, L.halo, L.line];
// Past this zoom the river lines carry the class and the glow dots step back.
const LINE_ZOOM = 5;
// The gauges stop clustering past zoom 6 (map.js, clusterMaxZoom); before that a ring would sit on a cluster
// bubble or on empty map, so the rings start here.
const RING_ZOOM = 7;
// A thin dark edge under the class colour: the pale 2-year yellow is the commonest class and needs it on a light map.
const CASING = "rgba(24,28,36,0.45)";

let data = null;          // { manifest, features, byId }
let loading = null;
let visible = true;
let day = -1;             // the forecast day on show, -1 for the 15-day peak
let popup = null;
let pulseFrame = 0;
let playToken = 0;
let gaugeRetry = 0;
let gaugeCoords = [];     // [lon, lat] of the gauges pulsing now

const still = () => Boolean(globalThis.matchMedia && globalThis.matchMedia("(prefers-reduced-motion: reduce)").matches);

function beforeGauges() {
  for (const id of ["catchment-fill", "gauge-heat", "clusters", "points"]) if (map.getLayer(id)) return id;
  return undefined;
}

// ── data ────────────────────────────────────────────────────────────────────

function load() {
  if (loading) return loading;
  loading = (async () => {
    const res = await fetch(`${BASE}manifest.json`, { cache: "no-cache" });
    if (!res.ok) { data = { manifest: { missing: true }, features: [], byId: new Map() }; return; }
    const manifest = await res.json();
    const geo = await fetch(`${BASE}latest.geojson`, { cache: "no-cache" });
    const fc = geo.ok ? await geo.json() : { features: [] };
    const features = fc.features || [];
    data = { manifest, features, byId: new Map(features.map((f) => [Number(f.properties.river_id), f])) };
  })().catch((err) => {
    console.info("floods ahead:", err && err.message);
    data = { manifest: { missing: true }, features: [], byId: new Map() };
  });
  return loading;
}

// ── layers ──────────────────────────────────────────────────────────────────

const vis = () => (visible ? "visible" : "none");

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getLayer(L.dot)) return true;
  try {
    const before = beforeGauges();
    if (globalThis.pmtiles && !maplibregl.__aqPmtiles) {
      maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
      maplibregl.__aqPmtiles = true;
    }
    const lines = Boolean(globalThis.pmtiles);
    if (lines && !map.getSource(L.net)) map.addSource(L.net, { type: "vector", url: `pmtiles://${STREAMS_PMTILES}` });
    if (!map.getSource(L.pts)) map.addSource(L.pts, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    if (!map.getSource(L.gauges)) map.addSource(L.gauges, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    const rank = ["coalesce", ["get", "c"], 0];
    if (lines) {
      // The river itself: a wide soft glow under a crisp line, both in the class colour.
      map.addLayer({
        id: L.glow, type: "line", source: L.net, "source-layer": "streams", minzoom: 3, filter: ["boolean", false],
        layout: { visibility: vis(), "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": NONE, "line-blur": ["interpolate", ["linear"], ["zoom"], 3, 3, 10, 9],
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 6, 6, 11, 10, 22],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 3, 0, LINE_ZOOM - 1, 0.4, 8, 0.6],
        },
      }, before);
      map.addLayer({
        id: L.casing, type: "line", source: L.net, "source-layer": "streams", minzoom: 3, filter: ["boolean", false],
        layout: { visibility: vis(), "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": CASING,
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 2.6, 6, 4, 10, 6.6],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 3, 0, LINE_ZOOM - 1, 0.8],
        },
      }, before);
      map.addLayer({
        id: L.line, type: "line", source: L.net, "source-layer": "streams", minzoom: 3, filter: ["boolean", false],
        layout: { visibility: vis(), "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": NONE,
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1.4, 6, 2.6, 10, 5],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 3, 0, LINE_ZOOM - 1, 0.95],
        },
      }, before);
    }
    // Far out: a soft glow per reach, bigger for a bigger class, fading as the lines take over.
    map.addLayer({
      id: L.halo, type: "circle", source: L.pts,
      layout: { visibility: vis(), "circle-sort-key": rank },
      paint: {
        "circle-color": ["match", ["get", "c"], ...FLOOD_CLASSES.flatMap((c) => [c.rp, c.color]), NONE],
        // A soft glow, not a blot (#543 design pass): small on the globe, so the river status reads through it.
        "circle-radius": ["interpolate", ["linear"], ["zoom"],
          1, ["interpolate", ["linear"], rank, 2, 4.5, 10, 6.5, 100, 10],
          6, ["interpolate", ["linear"], rank, 2, 9, 10, 12, 100, 18]],
        "circle-blur": 0.9,
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], 1, 0.4, LINE_ZOOM - 0.5, 0.32, 7, 0],
      },
    }, before);
    map.addLayer({
      id: L.dot, type: "circle", source: L.pts,
      layout: { visibility: vis(), "circle-sort-key": rank },
      paint: {
        "circle-color": ["match", ["get", "c"], ...FLOOD_CLASSES.flatMap((c) => [c.rp, c.color]), NONE],
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, ["interpolate", ["linear"], rank, 2, 1.8, 100, 3.4],
          6, ["interpolate", ["linear"], rank, 2, 2.8, 100, 5]],
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], LINE_ZOOM, 1, 7.5, 0],
      },
    }, before);
    // The gauges on a flooded reach: a still ring in the reach's class colour (all a reader who prefers less
    // motion sees), and over it a second ring that breathes outwards.
    const ringColor = ["match", ["get", "c"], ...FLOOD_CLASSES.flatMap((c) => [c.rp, c.color]), NONE];
    const ringRadius = ["interpolate", ["linear"], ["zoom"], RING_ZOOM, 7, 10, 10];
    map.addLayer({
      id: L.ringCasing, type: "circle", source: L.gauges, minzoom: RING_ZOOM,
      layout: { visibility: vis() },
      paint: {
        "circle-color": NONE, "circle-radius": ringRadius,
        "circle-stroke-color": CASING, "circle-stroke-width": 4.2, "circle-stroke-opacity": 0.8,
      },
    });
    map.addLayer({
      id: L.ring, type: "circle", source: L.gauges, minzoom: RING_ZOOM,
      layout: { visibility: vis() },
      paint: {
        "circle-color": NONE, "circle-radius": ringRadius,
        "circle-stroke-color": ringColor, "circle-stroke-width": 2.2, "circle-stroke-opacity": 0.95,
      },
    });
    map.addLayer({
      id: L.pulse, type: "circle", source: L.gauges, minzoom: RING_ZOOM,
      layout: { visibility: vis() },
      paint: {
        "circle-color": NONE, "circle-radius": 9,
        "circle-stroke-color": ringColor, "circle-stroke-width": 2, "circle-stroke-opacity": 0,
      },
    });
    for (const id of CLICKABLE) {
      if (!map.getLayer(id)) continue;
      clickLayers.add(id);   // its own handler answers; the map-wide one leaves it alone
      map.on("click", id, onClick);
      map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
    return true;
  } catch (err) {
    console.info("floods ahead layers unavailable:", err && err.message);
    return false;
  }
}

// Draw the reaches for the map's date: the class on that forecast day, or the 15-day peak.
function draw() {
  if (!data || !ensureLayers()) return;
  const groups = idsByClass(data.features, day);
  if (map.getLayer(L.line)) {
    const color = lineColorExpr(groups), filter = lineFilterExpr(groups);
    for (const id of [L.glow, L.casing, L.line]) {
      map.setFilter(id, filter);
      if (id !== L.casing) map.setPaintProperty(id, "line-color", color);
    }
  }
  map.getSource(L.pts).setData(pointsFor(data.features, day));
  drawGauges();
  renderLegend();
}

function drawGauges() {
  if (!data || !map.getSource(L.gauges)) return;
  const keys = gaugesFor(data.features, day);
  if (keys.size && !state.byKey.size && gaugeRetry++ < 20) {   // the catalogue is still loading
    setTimeout(drawGauges, 1500);
    return;
  }
  const feats = [];
  for (const [key, c] of keys) {
    const r = state.byKey.get(key);
    if (r && Number.isFinite(r.lat) && Number.isFinite(r.lon)) {
      feats.push({ type: "Feature", properties: { key, c }, geometry: { type: "Point", coordinates: [r.lon, r.lat] } });
    }
  }
  map.getSource(L.gauges).setData({ type: "FeatureCollection", features: feats });
  const had = gaugeCoords.length;
  gaugeCoords = feats.map((f) => f.geometry.coordinates);
  if (Boolean(had) !== Boolean(gaugeCoords.length)) pulse();
}

// ── the gentle motion ───────────────────────────────────────────────────────

// Only while a pulsing gauge is on screen: a ring that breathes costs a repaint per frame, and a page that is
// open all day should not repaint a map with nothing moving on it.
function pulse() {
  cancelAnimationFrame(pulseFrame);
  if (map && map.getLayer(L.pulse)) map.setPaintProperty(L.pulse, "circle-stroke-opacity", 0);
  if (still() || !visible || !gaugeCoords.length) return;
  let last = 0;
  let idle = false;
  const tick = (now) => {
    pulseFrame = requestAnimationFrame(tick);
    if (document.hidden || now - last < 66) return;       // about 15 frames a second is plenty for a breath
    last = now;
    if (!map.getLayer(L.pulse)) return;
    const box = map.getBounds();
    const inView = map.getZoom() >= RING_ZOOM && gaugeCoords.some((c) => box.contains(c));
    if (!inView) {
      if (!idle) { map.setPaintProperty(L.pulse, "circle-stroke-opacity", 0); idle = true; }
      return;
    }
    idle = false;
    const p = (now % 2400) / 2400;                         // one breath every 2.4 s
    map.setPaintProperty(L.pulse, "circle-radius", 9 + 14 * p);
    map.setPaintProperty(L.pulse, "circle-stroke-opacity", 0.8 * (1 - p) ** 1.4);
  };
  pulseFrame = requestAnimationFrame(tick);
}

// ── the row in "On the map" (map-legend.js) ─────────────────────────────────

const classesMark = () => `<span class="fa-mini">${FLOOD_CLASSES.map((c) => `<i style="background:${c.color}"></i>`).join("")}</span>`;

function reachCount() {
  if (!data || data.manifest.missing) return 0;
  return [...countsFor(data.features, day).values()].reduce((a, b) => a + b, 0);
}

function rowSummary() {
  if (!data) return "loading";
  if (data.manifest.missing) return "nothing published yet";
  const n = reachCount();
  const reaches = `${n.toLocaleString("en-GB")} reach${n === 1 ? "" : "es"}`;
  if (day >= 0) return `${n ? n.toLocaleString("en-GB") : "none"} on ${shortDay(addDays(data.manifest.issue_date, day))}`;
  return n ? reaches : "none in 15 days";
}

function rowBody() {
  const m = data.manifest;
  const counts = countsFor(data.features, day);
  const n = reachCount();
  const chips = FLOOD_CLASSES.map((c) => {
    const k = counts.get(c.rp) || 0;
    return `<span class="fa-chip${k ? "" : " zero"}" title="${k.toLocaleString("en-GB")} reach${k === 1 ? "" : "es"} at or above the ${c.label} flow">` +
      `<i style="background:${c.color}"></i>${c.rp}</span>`;
  }).join("");
  const playing = playToken > 0;
  return `<div class="fa-scale" role="img" aria-label="Return-period classes, 2 to 100 years">${chips}<span class="fa-unit">year flow</span></div>` +
    `<p class="ml-when">${escapeHtml(legendLine(m, day, n))}</p>` +
    `<p class="ml-src">${escapeHtml(issueLine(m))}. Model forecast, not an official warning.</p>` +
    `<div class="ml-actions">` +
    `<button type="button" class="ml-btn" data-act="${playing ? "stop" : "play"}">${playing ? "Stop" : "Play the 15 days"}</button>` +
    `<button type="button" class="ml-btn quiet" data-act="about">About</button></div>`;
}

function renderLegend() {
  refreshLegend("floods-ahead");
}

function registerRow() {
  registerLegendRow({
    id: "floods-ahead", title: "Floods ahead",
    mark: classesMark,
    summary: rowSummary,
    on: () => visible,
    empty: () => !data || data.manifest.missing || !reachCount(),
    toggle: (on) => setFloodsVisible(on),
    body: rowBody,
    act: (name) => {
      if (name === "play") play();
      else if (name === "stop") stopPlay();
      else if (name === "about") about();
    },
  });
}

function about() {
  const m = (data && data.manifest) || {};
  import("./shell.js?v=__BUILD__").then(({ openModal }) => openModal("Floods ahead", `
    <p>${escapeHtml(m.method || "")}</p>
    <p><strong>What it is not.</strong> ${escapeHtml(m.not || "")}</p>
    <p class="muted">Forecast run ${escapeHtml(m.run || "")}, made ${escapeHtml(m.made || "")}; ${Number(m.checked || 0).toLocaleString("en-GB")} reaches checked.</p>
    <p class="muted">${FLOODS_CREDIT.attribution}. ${escapeHtml(FLOODS_CREDIT.licence)}.</p>`));
}

// ── the map's date ──────────────────────────────────────────────────────────

function dayFor(date) {
  return data && !data.manifest.missing ? dayIndex(data.manifest.issue_date, date) : -1;
}

function stopPlay() {
  if (!playToken) return;
  playToken = 0;
  renderLegend();
}

// Walk the shared map date through the forecast's 15 days and back to where it was. It moves the one date
// everything follows (the time bar shows it), at the time bar's own pace.
async function play() {
  if (!data || data.manifest.missing) return;
  const token = ++playToken;
  const back = state.date;
  renderLegend();
  for (let i = 0; i < FORECAST_DAYS && token === playToken; i++) {
    setTime({ date: addDays(data.manifest.issue_date, i) }, { source: "floods" });
    await new Promise((r) => setTimeout(r, still() ? 1100 : 650));
  }
  if (token === playToken) {
    playToken = 0;
    setTime({ date: back }, { source: "floods" });
  }
  renderLegend();
}

// ── the card on the map ─────────────────────────────────────────────────────

function onClick(e) {
  if (!data) return;
  // A gauge under the pointer is what the reader meant: its own handler opens it.
  const gaugeHit = ["points", "clusters"].filter((id) => map.getLayer(id));
  if (gaugeHit.length && map.queryRenderedFeatures(e.point, { layers: gaugeHit }).length) return;
  const f = e.features && e.features[0];
  if (!f) return;
  const id = Number(f.properties.river_id ?? f.properties.riverId);
  const hit = data.byId.get(id);
  if (!hit) return;
  openCard(hit, e.lngLat);
}

function openCard(f, lngLat) {
  const p = f.properties;
  const facts = reachFacts(p, day, (data.manifest && data.manifest.members) || 51);
  const [lon, lat] = f.geometry.coordinates;
  const gauges = String(p.gauges || "").split(";").filter(Boolean).map((key) => {
    const r = state.byKey.get(key);
    const name = r ? r.name || key.split("/")[1] : key;
    const who = r ? sourceStyle(r.source).label : key.split("/")[0];
    return `<button type="button" class="fa-gauge" data-key="${escapeHtml(key)}">${escapeHtml(name)} <span class="muted">${escapeHtml(who)}</span></button>`;
  });
  const html = `<div class="fa-card">
    <div class="fa-card-head"><i style="background:${classColor(p.rp)}"></i><strong>${escapeHtml(facts.title)}</strong></div>
    <p>${escapeHtml(facts.peak)}</p>
    ${facts.today ? `<p>${escapeHtml(facts.today)}</p>` : ""}
    ${facts.agree ? `<p class="muted">${escapeHtml(facts.agree)}</p>` : ""}
    ${gauges.length ? `<div class="fa-gauges"><span class="muted">Gauge${gauges.length > 1 ? "s" : ""} here</span>${gauges.join("")}</div>` : ""}
    <div class="fa-card-actions"><button type="button" class="btn tiny primary fa-open">The 15-day forecast</button></div>
    <p class="fa-foot muted">${escapeHtml(facts.reach)} GEOGLOWS model forecast, not an official warning.</p>
  </div>`;
  if (popup) popup.remove();
  popup = new maplibregl.Popup({ closeButton: true, closeOnClick: true, maxWidth: "300px", offset: 10, className: "fa-popup" })
    .setLngLat(lngLat || [lon, lat]).setHTML(html).addTo(map);
  const el = popup.getElement();
  el.querySelector(".fa-open").addEventListener("click", () => {
    popup.remove();
    actions.selectPoint(lat, lon, { tab: "now" });
  });
  for (const b of el.querySelectorAll(".fa-gauge")) {
    b.addEventListener("click", () => { popup.remove(); actions.selectStation(b.dataset.key, { fly: false }); });
  }
}

// ── on and off ──────────────────────────────────────────────────────────────

export function floodsVisible() { return visible; }

/** The issue and its reaches once read ({ manifest, features }), for the layers drawn from it: flood depth (#554). */
export function floodsAheadData() { return load().then(() => data); }

export function setFloodsVisible(on) {
  visible = Boolean(on);
  state.floodsOn = visible;
  renderCredits();
  const toggle = $("toggle-floods");
  if (toggle) toggle.checked = visible;
  if (!visible) { stopPlay(); if (popup) popup.remove(); }
  if (state.mapOk && map) {
    for (const id of ALL) {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis());
    }
  }
  if (visible) void load().then(draw);
  pulse();
  renderLegend();
}

export function initFloodsAhead() {
  if (!state.mapOk || !map) return;
  registerRow();
  state.floodsOn = visible;
  renderCredits();
  const toggle = $("toggle-floods");
  if (toggle) {
    toggle.checked = visible;
    toggle.addEventListener("change", (e) => setFloodsVisible(e.target.checked));
  }
  onTime((t) => {
    if (t.source !== "floods" && playToken) stopPlay();   // the reader took the date back
    if (t.date === t.prev.date) return;
    const next = dayFor(t.date);
    if (next !== day) { day = next; draw(); }
    else renderLegend();
  });
  // A basemap swap carries our layers over but loses the paint we animate; put the motion back.
  map.on("style.load", () => { if (visible) setTimeout(pulse, 0); });
  void load().then(() => {
    day = dayFor(state.date);
    if (visible) { draw(); pulse(); }
    renderLegend();
  });
}
