// The FEWS view (#556): forecast points in threshold colours, and one click
// giving the ensemble plume in the map card, as Delft-FEWS and the GloFAS/EFAS
// viewers show them.
//
// Two kinds of forecast point. The Floods ahead reaches (floods-ahead.js) keep
// their own layer; a click on one opens the map card here, and a light worker
// reads the reach's 51 members (aquascope.nownext.plume) against the layer's own
// return-period flows, so the card and the map agree. The Archive's forecast
// gauges (the daily forecast job's latest issued file, read with DuckDB) are
// classed by aquascope.nownext.forecast_points against each gauge's own
// return-period flows, ringed on the map in the same colours, and their card
// shows the plume corrected to the gauge with the gauge's own record before it.
// Both follow the Floods ahead switch and the map's date.

import { CONFIG } from "../config.js?v=__BUILD__";
import { actions, clickLayers, onTime, sourceStyle, state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { callLight } from "./worker-client.js?v=__BUILD__";
import { classColor, reachFacts } from "./floods-ahead-core.js?v=__BUILD__";
import {
  BELOW, FEWS_CLASSES, dayLabel, dayOf, fewsColor, obsBefore, plumeHead, plumeLabel, pointsGeoJSON, pointsLine,
} from "./fews-core.js?v=__BUILD__";

const L = { src: "fews-pts", casing: "fews-ring-casing", ring: "fews-ring", dot: "fews-dot" };
const ALL = [L.casing, L.ring, L.dot];
// The gauges stop clustering past zoom 6 (map.js), so a ring around one starts at 7; before that only a gauge
// whose forecast reaches a threshold shows, as a small solid mark.
const RING_ZOOM = 7;
const KEY_ZOOM = 6;
const CASING = "rgba(24,28,36,0.5)";
const WARNINGS = `${CONFIG.forecastsBase}warnings/`;
const REACH_CREDIT = "GEOGLOWS v2 forecast (CC BY 4.0) and return periods (CC BY-NC-SA 4.0). Model output, not an " +
  "official warning.";

let visible = true;
let points = null;       // aquascope.nownext.forecast_points: { issue_date, points, ... }
let byKey = new Map();
let loading = null;
let day = -1;
let key = null;          // the legend line
let retry = 0;
let shown = [];          // [lon, lat] of the forecast gauges on the map now

// ── the forecast gauges ─────────────────────────────────────────────────────

const plain = (row) => {
  const out = {};
  for (const [k, v] of Object.entries(row)) out[k] = typeof v === "bigint" ? Number(v) : v;
  return out;
};

/** The forecast gauges, loaded once: the latest issued file's GEOGLOWS rows, classed in a light worker. */
export function forecastPoints() {
  if (loading) return loading;
  loading = (async () => {
    const res = await fetch(`${CONFIG.forecastsBase}manifest.json`, { cache: "no-cache" });
    if (!res.ok) return null;
    const manifest = await res.json();
    const issue = (manifest.issues || []).filter((i) => i && i.file).at(-1);
    if (!issue) return null;
    const url = `${CONFIG.forecastsBase}${String(issue.file).replace(/^forecasts\//, "")}`;
    const { conn } = await duck();
    const table = await conn.query(`SELECT * REPLACE (CAST(issue_date AS VARCHAR) AS issue_date,
      CAST(valid_date AS VARCHAR) AS valid_date, CAST(init_date AS VARCHAR) AS init_date)
      FROM read_parquet('${url}') WHERE model = 'geoglows'`);
    const rows = table.toArray().map((r) => plain(r.toJSON()));
    const got = await callLight("now", { op: "points", args: { rows } });
    points = got;
    byKey = new Map((got.points || []).map((p) => [p.key, p]));
    draw();
    return got;
  })().catch((err) => {
    console.info("forecast gauges:", err && err.message);
    loading = null;   // a later look tries again
    return null;
  });
  return loading;
}

/** The map card's plume for a forecast gauge (with its record before the run), or null for any other gauge. */
export function gaugePlume(stationKey, series = null) {
  const pt = byKey.get(stationKey);
  if (!pt || !(pt.date || []).length) return null;
  const skill = pt.corrected && Number.isFinite(pt.kge_corrected)
    ? `Corrected to this gauge's record: KGE ${pt.kge_corrected.toFixed(2)} on the hindcast` +
      (Number.isFinite(pt.kge_raw) ? `, raw ${pt.kge_raw.toFixed(2)}.` : ".")
    : "Raw GEOGLOWS: there was no correction to this gauge's record that day.";
  return {
    data: { ...pt, observed: series ? obsBefore(series, pt.date[0]) : null },
    head: plumeHead(pt), color: pt.classed ? fewsColor(pt.rp) : null,
    label: plumeLabel(pt, { corrected: pt.corrected }),
    line: `${skill} GEOGLOWS v2, CC BY 4.0.`,
  };
}

// ── the map ─────────────────────────────────────────────────────────────────

const vis = () => (visible ? "visible" : "none");

function ensureLayers() {
  if (!state.mapOk || !map) return false;
  if (map.getLayer(L.ring)) return true;
  try {
    map.addSource(L.src, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    const color = ["match", ["get", "c"], ...FEWS_CLASSES.flatMap((c) => [c.rp, c.color]), BELOW.color];
    const up = [">", ["get", "c"], 0];
    // Far out: only a gauge expected to pass a threshold, a small solid mark with a dark edge.
    map.addLayer({
      id: L.dot, type: "circle", source: L.src, maxzoom: RING_ZOOM, filter: up,
      layout: { visibility: vis(), "circle-sort-key": ["get", "c"] },
      paint: {
        "circle-color": color, "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 3.2, 6, 4.6],
        "circle-stroke-color": CASING, "circle-stroke-width": 1.2,
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], 2.5, 0, 3.5, 1],
        "circle-stroke-opacity": ["interpolate", ["linear"], ["zoom"], 2.5, 0, 3.5, 1],
      },
    });
    // Close in: a ring around the gauge, solid in the class colour, or a thin slate one below the 2-year flow.
    const radius = ["interpolate", ["linear"], ["zoom"], RING_ZOOM, 9, 11, 12];
    map.addLayer({
      id: L.casing, type: "circle", source: L.src, minzoom: RING_ZOOM, filter: up,
      layout: { visibility: vis() },
      paint: { "circle-color": "rgba(0,0,0,0)", "circle-radius": radius, "circle-stroke-color": CASING,
        "circle-stroke-width": 4.4, "circle-stroke-opacity": 0.75 },
    });
    map.addLayer({
      id: L.ring, type: "circle", source: L.src, minzoom: RING_ZOOM,
      layout: { visibility: vis(), "circle-sort-key": ["get", "c"] },
      paint: {
        "circle-color": "rgba(0,0,0,0)", "circle-radius": radius, "circle-stroke-color": color,
        "circle-stroke-width": ["case", up, 2.4, 1.5], "circle-stroke-opacity": ["case", up, 1, 0.9],
      },
    });
    for (const id of [L.dot, L.ring]) {
      clickLayers.add(id);
      map.on("click", id, onClick);
      map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
    map.on("moveend", renderKey);
    return true;
  } catch (err) {
    console.info("forecast gauge layers unavailable:", err && err.message);
    return false;
  }
}

function draw() {
  if (!points || !ensureLayers()) return;
  if (!state.byKey.size && retry++ < 20) {   // the catalogue is still loading
    setTimeout(draw, 1500);
    return;
  }
  const d = (points.points || []).length ? dayOf(points.points[0].date, state.date) : -1;
  day = d;
  const fc = pointsGeoJSON(points.points, (k) => state.byKey.get(k), { day });
  shown = fc.features.map((f) => f.geometry.coordinates);
  map.getSource(L.src).setData(fc);
  renderKey();
}

function onClick(e) {
  const f = e.features && e.features[0];
  if (!f || !f.properties.key) return;
  actions.selectStation(f.properties.key, { fly: false });
}

// One quiet line in the legend stack, only while forecast gauges' rings are in sight.
function renderKey() {
  const stack = document.getElementById("map-legends");
  if (!stack) return;
  if (!key) {
    key = document.createElement("p");
    key.className = "fews-key";
    stack.appendChild(key);
  }
  const line = pointsLine(points);
  let near = false;
  if (map && state.mapOk && map.getZoom() >= KEY_ZOOM) {
    const box = map.getBounds();
    near = shown.some((c) => box.contains(c));
  }
  key.hidden = !visible || !line || !near;
  if (key.hidden) return;
  key.innerHTML = `<i class="fews-ring-sw" aria-hidden="true"></i><span>${line.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</span>`;
  key.title = "Archive gauges with a forecast that day, corrected to each gauge's record and classed against its own " +
    "return-period flows. Solid ring: the class colour reached; thin slate ring: below the 2-year flow.";
}

export function setFewsVisible(on) {
  visible = Boolean(on);
  if (state.mapOk && map) for (const id of ALL) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis());
  renderKey();
  if (visible) void forecastPoints();
}

// ── a Floods ahead reach in the map card ────────────────────────────────────

const swatch = (color) => `<svg width="11" height="11" viewBox="0 0 11 11" aria-hidden="true"><circle cx="5.5" cy="5.5" r="4.6" fill="${color}" stroke="rgba(0,0,0,.25)"/></svg>`;

// The reach's return-period flows as the layer has them; a file written before #556 carries only the 2-year one
// in the browser's copy, so the others are read from the issue's parquet.
async function reachThresholds(p) {
  const q = { q2: p.q2, q5: p.q5, q10: p.q10, q25: p.q25, q50: p.q50, q100: p.q100 };
  if (p.q5 === undefined) {
    try {
      const { conn } = await duck();
      const t = await conn.query(`SELECT q2, q5, q10, q25, q50, q100 FROM read_parquet('${WARNINGS}latest.parquet')
        WHERE river_id = ${Number(p.river_id)} LIMIT 1`);
      const row = t.toArray().map((r) => plain(r.toJSON()))[0];
      if (row) Object.assign(q, row);
    } catch (err) {
      console.info("Floods ahead thresholds:", err && err.message);
    }
  }
  return q;
}

/** Open the map card on a Floods ahead reach: the layer's facts at once, then the ensemble plume. */
export function openReachCard(f, lngLat, { day: d = -1, manifest = {} } = {}) {
  if (!actions.openMapCard) return null;
  const p = f.properties;
  const [lon, lat] = f.geometry.coordinates;
  const facts = reachFacts(p, d, manifest.members || 51);
  const at = lngLat ? [Number(lngLat.lng ?? lngLat[0]), Number(lngLat.lat ?? lngLat[1])] : [lon, lat];
  const gauges = String(p.gauges || "").split(";").filter(Boolean).slice(0, 2).map((k, i) => {
    const r = state.byKey.get(k);
    const name = r ? r.name || k.split("/")[1] : k;
    return { id: `fa-gauge-${i}`, label: name.length > 22 ? `${name.slice(0, 21)}…` : name,
      title: `The gauge on this reach (${r ? sourceStyle(r.source).label : k.split("/")[0]}): its record and forecast`,
      onClick: () => actions.selectStation(k, { fly: false }) };
  });
  const id = `fa:${p.river_id}`;
  const handle = actions.openMapCard({
    id, lngLat: at, lift: 8, what: "Floods ahead", whatIcon: swatch(classColor(p.rp)),
    title: facts.title, sub: `River reach ${p.river_id}${p.order ? ` · stream order ${p.order}` : ""}`,
    status: { text: [facts.peak, facts.today].filter(Boolean).join(" "), color: classColor(p.rp) },
    plume: { pending: "Reading the 51 forecast members…" },
    credit: REACH_CREDIT,
    details: () => actions.selectPoint(lat, lon, { tab: "now" }),
    buttons: gauges,
  });
  void (async () => {
    const thresholds = await reachThresholds(p);
    // The same run the map shows (manifest.issue_date), so the card and the colours agree.
    const pl = await callLight("now", { op: "plume", args: { river_id: Number(p.river_id), run: manifest.issue_date || null, thresholds: {
      ...thresholds, source: `the Floods ahead issue of ${manifest.issue_date || "today"}`,
      licence: "CC BY-NC-SA 4.0 (GEOGLOWS v2 return periods)" } } }, { priority: 1 });
    if (pl.error) throw new Error(pl.error);
    // Should that run not answer, the newest one does, and the card says so rather than quietly disagree.
    const newer = manifest.issue_date && pl.issued && pl.issued !== manifest.issue_date
      ? ` The map's colours are from the run of ${dayLabel(manifest.issue_date)}.` : "";
    handle.update({ plume: { data: pl, label: plumeLabel(pl), line: `${pl.members_line || ""}${newer}`.trim() } });
  })().catch((err) => {
    handle.update({ plume: { empty: `The members did not answer this time (${err.message}). Details has the forecast.` } });
  });
  return handle;
}

// ── wiring ──────────────────────────────────────────────────────────────────

export function initFews() {
  if (!state.mapOk || !map) return;
  visible = state.floodsOn !== false;
  onTime((t) => {
    if (t.date === t.prev.date || !points) return;
    const next = (points.points || []).length ? dayOf(points.points[0].date, t.date) : -1;
    if (next !== day) draw();
  });
  map.on("style.load", () => { if (points) setTimeout(draw, 0); });
  // After the first paint: DuckDB and a light worker are both busy with the catalogue and Python just then.
  setTimeout(() => { if (visible) void forecastPoints(); }, 2500);
}
