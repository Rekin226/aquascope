// AquaScope Explorer, main thread: catalog -> map -> click -> worker (Pyodide) -> charts.
// No build step. Everything static; the only server is a CDN and the agencies' own APIs.

import { CONFIG } from "./config.js?v=__BUILD__";

const SOURCE_STYLE = {
  usgs: { label: "USGS (US)", color: "#1565c0" },
  uk_ea: { label: "Environment Agency (UK)", color: "#2e7d32" },
  hubeau_hydrometrie: { label: "Hub'Eau (FR)", color: "#c62828" },
  pegelonline: { label: "PEGELONLINE (DE)", color: "#ef6c00" },
  ireland_opw: { label: "OPW (IE)", color: "#6a1b9a" },
  taiwan_cwa: { label: "CWA (TW)", color: "#00838f" },
};
const FALLBACK_COLOR = "#546e7a";
const VAR_LABEL = {
  discharge: "discharge", water_level: "water level", precipitation: "rainfall",
  groundwater_level: "groundwater", climate: "climate", water_quality: "water quality",
};

const $ = (id) => document.getElementById(id);
const EMPTY_FC = { type: "FeatureCollection", features: [] };
// Debug hooks (harmless in production): window.__aq.state, window.__aq.log
const dbg = (window.__aq = { log: [], state: null });
const trace = (msg) => { dbg.log.push(`${new Date().toISOString().slice(11, 19)} ${msg}`); };
const state = { stations: [], byKey: new Map(), hidden: new Set(), selected: null, result: null, workerReady: false, pending: new Map(), reqId: 0, mapOk: false, marker: null, point: null, ask: { running: false, catalogSent: false, markdown: null, id: null } };
dbg.state = state;

// ── catalog ─────────────────────────────────────────────────────────────────

// One DuckDB-WASM instance for the page: the catalog at boot, the basin
// topology and attributes on demand (all read in place over HTTPS ranges).
let duckPromise = null;
function duck() {
  if (duckPromise) return duckPromise;
  duckPromise = (async () => {
    trace("duckdb: import");
    const duckdb = await import(CONFIG.duckdbModule);
    const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
    trace(`duckdb: bundle ${bundle.mainModule}`);
    const workerUrl = URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }));
    const worker = new Worker(workerUrl);
    const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(workerUrl);
    trace("duckdb: instantiated");
    const conn = await db.connect();
    return { db, conn };
  })();
  duckPromise.catch(() => { duckPromise = null; });
  return duckPromise;
}

async function loadCatalogDuckDB() {
  const { conn } = await duck();
  const sql = `SELECT source, station_id, name, latitude, longitude, variables,
                      CAST(period_start AS VARCHAR) AS period_start, CAST(period_end AS VARCHAR) AS period_end, url
               FROM read_parquet('${CONFIG.stationsParquet}')`;
  const table = await conn.query(sql);
  trace(`duckdb: query returned ${table.numRows} rows`);
  const rows = table.toArray().map((r) => r.toJSON());
  return rows.map((r) => ({
    source: r.source, station_id: r.station_id, name: r.name ?? null,
    lat: Number(r.latitude), lon: Number(r.longitude),
    variables: Array.isArray(r.variables) ? r.variables : (r.variables?.toArray?.() ?? []),
    period_start: r.period_start ? String(r.period_start).slice(0, 10) : null,
    period_end: r.period_end ? String(r.period_end).slice(0, 10) : null,
    url: r.url ?? null,
  }));
}

async function loadCatalogGeoJSON() {
  const res = await fetch(CONFIG.stationsGeoJSON);
  if (!res.ok) throw new Error(`GeoJSON ${res.status}`);
  const gj = await res.json();
  return gj.features.map((f) => ({
    source: f.properties.source, station_id: f.properties.station_id, name: f.properties.name ?? null,
    lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1],
    variables: f.properties.variables ?? [], period_start: f.properties.period_start ?? null,
    period_end: f.properties.period_end ?? null, url: f.properties.url ?? null,
  }));
}

async function loadCatalog() {
  $("count").textContent = "loading catalog…";
  let rows;
  try {
    rows = await loadCatalogDuckDB();
    console.info(`catalog via DuckDB-WASM: ${rows.length} stations`);
  } catch (err) {
    console.warn("DuckDB-WASM path failed, falling back to GeoJSON:", err);
    trace(`duckdb failed: ${err && err.message}; geojson fallback`);
    rows = await loadCatalogGeoJSON();
    console.info(`catalog via GeoJSON: ${rows.length} stations`);
  }
  state.stations = rows;
  state.byKey = new Map(rows.map((r) => [`${r.source}/${r.station_id}`, r]));
  return rows;
}

function toFeatureCollection(rows) {
  return {
    type: "FeatureCollection",
    features: rows.filter((r) => !state.hidden.has(r.source)).map((r) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [r.lon, r.lat] },
      properties: { key: `${r.source}/${r.station_id}`, source: r.source, name: r.name ?? "", color: (SOURCE_STYLE[r.source] || {}).color || FALLBACK_COLOR },
    })),
  };
}

// ── map ─────────────────────────────────────────────────────────────────────

let map;
function initMap() {
  try {
    return initMapUnsafe();
  } catch (err) {
    console.warn("map unavailable:", err && err.message);
    map = null;
    return Promise.resolve(false);
  }
}

function initMapUnsafe() {
  map = dbg.map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      sources: {
        carto: {
          type: "raster", tileSize: 256,
          tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png", "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"],
          attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
        },
      },
      layers: [{ id: "carto", type: "raster", source: "carto" }],
    },
    center: [0, 30], zoom: 1.6, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.ScaleControl(), "bottom-left");
  // Resolve true when the style loaded, false if WebGL is missing or the map
  // errors out / stalls: the catalog, search and analysis panel still work.
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), 12000);
    map.once("load", () => { clearTimeout(timer); resolve(true); });
    map.once("error", (e) => { console.warn("map error", e && e.error); clearTimeout(timer); resolve(false); });
  });
}

function addStationLayers(fc) {
  map.addSource("catchment", { type: "geojson", data: EMPTY_FC });
  map.addLayer({ id: "catchment-fill", type: "fill", source: "catchment", paint: { "fill-color": "#1565c0", "fill-opacity": 0.14 } });
  map.addLayer({ id: "catchment-line", type: "line", source: "catchment", paint: { "line-color": "#0d47a1", "line-width": 1.6, "line-dasharray": [2, 1.5] } });
  map.addSource("stations", { type: "geojson", data: fc, cluster: true, clusterMaxZoom: 9, clusterRadius: 38 });
  map.addLayer({
    id: "clusters", type: "circle", source: "stations", filter: ["has", "point_count"],
    paint: {
      "circle-color": "#1565c0", "circle-opacity": 0.75, "circle-stroke-color": "#fff", "circle-stroke-width": 1.5,
      "circle-radius": ["step", ["get", "point_count"], 14, 50, 18, 250, 24, 1000, 30],
    },
  });
  map.addLayer({
    id: "cluster-count", type: "symbol", source: "stations", filter: ["has", "point_count"],
    layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 11, "text-font": ["Open Sans Semibold"] },
    paint: { "text-color": "#fff" },
  });
  map.addLayer({
    id: "points", type: "circle", source: "stations", filter: ["!", ["has", "point_count"]],
    paint: { "circle-color": ["get", "color"], "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 6, 14, 9], "circle-stroke-color": "#fff", "circle-stroke-width": 1 },
  });
  map.addLayer({
    id: "selected", type: "circle", source: "stations", filter: ["==", ["get", "key"], "__none__"],
    paint: { "circle-color": "#ffd600", "circle-radius": 11, "circle-stroke-color": "#212121", "circle-stroke-width": 2 },
  });

  map.on("click", "clusters", async (e) => {
    const f = map.queryRenderedFeatures(e.point, { layers: ["clusters"] })[0];
    const zoom = await map.getSource("stations").getClusterExpansionZoom(f.properties.cluster_id);
    map.easeTo({ center: f.geometry.coordinates, zoom });
  });
  map.on("click", "points", (e) => {
    const f = e.features[0];
    selectStation(f.properties.key, { fly: false });
  });
  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
  map.on("mouseenter", "points", (e) => {
    map.getCanvas().style.cursor = "pointer";
    const p = e.features[0].properties;
    popup.setLngLat(e.features[0].geometry.coordinates).setHTML(`<strong>${escapeHtml(p.name || p.key.split("/")[1])}</strong><br><span class="muted">${(SOURCE_STYLE[p.source] || {}).label || p.source}</span>`).addTo(map);
  });
  map.on("mouseleave", "points", () => { map.getCanvas().style.cursor = ""; popup.remove(); });
  map.on("click", (e) => {
    const hit = map.queryRenderedFeatures(e.point, { layers: ["points", "clusters"] });
    if (hit.length) return; // handled by the layer handlers
    selectPoint(e.lngLat.lat, e.lngLat.lng);
  });
  map.on("mouseenter", "clusters", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "clusters", () => (map.getCanvas().style.cursor = ""));
}

function refreshMapData() {
  if (state.mapOk) map.getSource("stations").setData(toFeatureCollection(state.stations));
  const visible = state.stations.filter((r) => !state.hidden.has(r.source)).length;
  $("count").textContent = `${visible.toLocaleString()} stations${state.mapOk ? "" : " (map unavailable here: WebGL is off; search still works)"}`;
}

// ── legend / search ─────────────────────────────────────────────────────────

function buildLegend() {
  const counts = {};
  for (const r of state.stations) counts[r.source] = (counts[r.source] || 0) + 1;
  const el = $("legend");
  el.innerHTML = "";
  for (const src of Object.keys(counts).sort((a, b) => counts[b] - counts[a])) {
    const st = SOURCE_STYLE[src] || { label: src, color: FALLBACK_COLOR };
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.innerHTML = `<i style="background:${st.color}"></i>${escapeHtml(st.label)} <em>${counts[src].toLocaleString()}</em>`;
    chip.title = `Toggle ${st.label}`;
    chip.addEventListener("click", () => {
      if (state.hidden.has(src)) state.hidden.delete(src); else state.hidden.add(src);
      chip.classList.toggle("off", state.hidden.has(src));
      refreshMapData();
    });
    el.appendChild(chip);
  }
  const bchip = document.createElement("button");
  bchip.className = "chip basins-toggle off"; bchip.id = "chip-basins";
  bchip.innerHTML = `<i style="background:#6a1b9a"></i>Basins`;
  bchip.title = "Toggle BasinATLAS sub-basin outlines (HydroATLAS, CC BY 4.0)";
  bchip.addEventListener("click", () => setBasinsVisible(!basinsLayersOn));
  el.appendChild(bchip);
}

function initSearch() {
  const input = $("search"), box = $("search-results");
  let t;
  input.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      const q = input.value.trim().toLowerCase();
      if (q.length < 2) { box.hidden = true; return; }
      const hits = [];
      for (const r of state.stations) {
        if (state.hidden.has(r.source)) continue;
        if ((r.name && r.name.toLowerCase().includes(q)) || r.station_id.toLowerCase().includes(q)) {
          hits.push(r);
          if (hits.length >= 25) break;
        }
      }
      box.innerHTML = hits.length ? "" : `<div class="hit muted">no match</div>`;
      for (const r of hits) {
        const d = document.createElement("div");
        d.className = "hit";
        d.innerHTML = `<i style="background:${(SOURCE_STYLE[r.source] || {}).color || FALLBACK_COLOR}"></i>${escapeHtml(r.name || r.station_id)} <span class="muted">${escapeHtml(r.station_id)}</span>`;
        d.addEventListener("click", () => { box.hidden = true; input.value = ""; selectStation(`${r.source}/${r.station_id}`, { fly: true }); });
        box.appendChild(d);
      }
      box.hidden = false;
    }, 120);
  });
  document.addEventListener("click", (e) => { if (!box.contains(e.target) && e.target !== input) box.hidden = true; });
}

// ── selection + panel ───────────────────────────────────────────────────────

function selectStation(key, { fly } = { fly: false }) {
  const r = state.byKey.get(key);
  if (!r) return;
  state.selected = r;
  state.result = null;
  if (state.mapOk) {
    map.setFilter("selected", ["==", ["get", "key"], key]);
    if (fly) map.flyTo({ center: [r.lon, r.lat], zoom: Math.max(map.getZoom(), 9) });
  }
  history.replaceState(null, "", `#s=${encodeURIComponent(key)}`);

  $("panel-empty").hidden = true;
  $("panel-point").hidden = true;
  $("panel-ask").hidden = true;
  $("panel-station").hidden = false;
  if (state.marker) { state.marker.remove(); state.marker = null; }
  const st = SOURCE_STYLE[r.source] || { label: r.source, color: FALLBACK_COLOR };
  const badge = $("st-source");
  badge.textContent = st.label; badge.style.background = st.color;
  $("st-name").textContent = r.name || r.station_id;
  $("st-id").textContent = r.station_id;
  $("st-vars").textContent = (r.variables || []).map((v) => VAR_LABEL[v] || v).join(", ") || "—";
  $("st-period").textContent = r.period_start ? ` · ${r.period_start} → ${r.period_end || "present"}` : "";
  const agency = $("st-agency");
  if (r.url) { agency.href = r.url; agency.hidden = false; } else agency.hidden = true;
  $("btn-csv").disabled = true;
  $("btn-ci").disabled = false;
  for (const id of ["sec-summary", "sec-hydro", "sec-ffa", "sec-fdc", "sec-trend", "sec-gr4j", "sec-notes", "sec-methods"]) $(id).hidden = true;
  resetGr4j();
  $("panel").scrollTop = 0;
  clearCatchment();
  requestAnalysis(r);
  requestCatchment({ station: r });
  requestBasin(r.lat, r.lon, "st");
}

function setStatus(text, kind = "info") {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${kind}`;
  el.hidden = !text;
}

// ── click anywhere ──────────────────────────────────────────────────────────

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371, d2r = Math.PI / 180;
  const dLat = (lat2 - lat1) * d2r, dLon = (lon2 - lon1) * d2r;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * d2r) * Math.cos(lat2 * d2r) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function nearestStations(lat, lon, n = 6) {
  const out = [];
  for (const r of state.stations) {
    if (state.hidden.has(r.source)) continue;
    const d = haversineKm(lat, lon, r.lat, r.lon);
    if (out.length < n) { out.push([d, r]); out.sort((a, b) => a[0] - b[0]); }
    else if (d < out[n - 1][0]) { out[n - 1] = [d, r]; out.sort((a, b) => a[0] - b[0]); }
  }
  return out;
}

async function selectPoint(lat, lon) {
  lat = Math.round(lat * 1e4) / 1e4; lon = Math.round(lon * 1e4) / 1e4;
  state.selected = null; state.result = null; state.point = { lat, lon };
  if (state.mapOk) {
    map.setFilter("selected", ["==", ["get", "key"], "__none__"]);
    if (state.marker) state.marker.remove();
    state.marker = new maplibregl.Marker({ color: "#455a64" }).setLngLat([lon, lat]).addTo(map);
  }
  history.replaceState(null, "", `#p=${lat},${lon}`);
  $("panel-empty").hidden = true; $("panel-station").hidden = true; $("panel-ask").hidden = true; $("panel-point").hidden = false;
  $("pt-title").textContent = `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`;
  $("pt-coords").textContent = `lat ${lat}, lon ${lon}`;
  for (const id of ["pt-sec-climate", "pt-sec-glofas", "pt-sec-notes", "pt-sec-methods"]) $(id).hidden = true;
  clearCatchment();
  requestCatchment({ point: { lat, lon } });
  requestBasin(lat, lon, "pt");
  const near = nearestStations(lat, lon);
  $("pt-nearest").innerHTML = near.map(([d, r]) => `<li data-key="${escapeHtml(`${r.source}/${r.station_id}`)}"><i style="background:${(SOURCE_STYLE[r.source] || {}).color || FALLBACK_COLOR}"></i>${escapeHtml(r.name || r.station_id)} <span class="muted">${escapeHtml((SOURCE_STYLE[r.source] || {}).label || r.source)}</span><span class="dist">${d < 10 ? d.toFixed(1) : Math.round(d)} km</span></li>`).join("") || `<li class="muted">no gauges in the catalog</li>`;
  for (const li of $("pt-nearest").querySelectorAll("li[data-key]")) li.addEventListener("click", () => selectStation(li.dataset.key, { fly: true }));
  $("panel").scrollTop = 0;
  const statusEl = $("pt-status");
  const setPt = (t, k = "info") => { statusEl.textContent = t; statusEl.className = `status ${k}`; statusEl.hidden = !t; };
  setPt(state.workerReady ? "Asking Open-Meteo about this point…" : "Loading Python in your browser (once, ~15 MB)…");
  try {
    const res = await call("anywhere", { lat, lon, years: 10 });
    if (!state.point || state.point.lat !== lat || state.point.lon !== lon) return;
    setPt("");
    renderPoint(res);
  } catch (err) {
    setPt(`Could not describe this point: ${err.message}`, "error");
  }
}

function renderPoint(res) {
  const c = res.climate;
  if (c) {
    $("pt-sec-climate").hidden = false;
    $("pt-kpis").innerHTML = [
      ["rainfall", `${fmt(c.precipitation_mm_per_year, 0)} mm/yr`, "ERA5, 10-yr mean"],
      ["reference ET0", `${fmt(c.et0_mm_per_year, 0)} mm/yr`, "FAO-56"],
      ["aridity", `${fmt(c.aridity_index, 2)}`, c.aridity_class || "P / ET0"],
      ["temperature", `${fmt(c.temperature_mean_c, 1)} °C`, `wettest day ${fmt(c.wettest_day_mm, 0)} mm`],
    ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
    // Distinct labels: Plotly merges categorical x values with the same name, so
    // one-letter months collapsed May/Jun/Jul/Aug into Mar/Jan/Apr and drew 8 bars.
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    Plotly.react("plot-climate", [
      { x: months, y: c.monthly_precipitation_mm, type: "bar", name: "rain (mm)", marker: { color: "#1565c0" } },
      { x: months, y: c.monthly_et0_mm, type: "scatter", mode: "lines+markers", name: "ET0 (mm)", line: { color: "#ef6c00" } },
    ], { ...PLOT_LAYOUT, height: 220, yaxis: { title: { text: "mm / month" } }, legend: { orientation: "h", y: 1.15 }, barmode: "group" }, PLOT_CONFIG);
  }
  const g = res.glofas;
  if (g && g.n) {
    $("pt-sec-glofas").hidden = false;
    $("pt-glofas-kpis").innerHTML = [
      ["mean", `${fmt(g.stats.mean)} m³/s`, `${g.start} → ${g.end}`],
      ["max", `${fmt(g.stats.max)} m³/s`, "modelled daily"],
    ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
    if (g.ffa && g.ffa.fits) {
      const rps = g.ffa.return_periods, gv = g.ffa.fits.gev_lmoments || {}, lp = g.ffa.fits.lp3 || {};
      $("pt-ffa-table").innerHTML = `<thead><tr><th>T (yr)</th><th>GEV L-moments</th><th>LP3 (90 % CI)</th></tr></thead><tbody>` +
        rps.map((rp, i) => `<tr><td>${rp}</td><td>${gv.q ? fmt(gv.q[i]) : "—"}</td><td>${lp.q ? `${fmt(lp.q[i])} <span class="ci">[${fmt(lp.ci[i][0])}, ${fmt(lp.ci[i][1])}]</span>` : "—"}</td></tr>`).join("") +
        `</tbody><tfoot><tr><td colspan="3" class="muted">Indicative only: GloFAS grid-cell discharge in m³/s, ${g.ffa.n_years} modelled years.</td></tr></tfoot>`;
    } else {
      $("pt-ffa-table").innerHTML = "";
    }
  }
  const notes = res.notes || [];
  $("pt-notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  $("pt-sec-notes").hidden = notes.length === 0;
  const ms = res.methods || [];
  $("pt-methods").innerHTML = ms.map((m) => `<li><strong>${escapeHtml(m.name)}.</strong> ${escapeHtml(m.text)}<br><span class="cite">${escapeHtml(m.citation)}</span></li>`).join("");
  $("pt-attribution").textContent = res.attribution ? `Data: ${res.attribution}` : "";
  $("pt-sec-methods").hidden = ms.length === 0;
  if (!$("pt-catchment").hidden) addMethodOnce("pt-methods", "pt-sec-methods", NLDI_METHOD);
  if (!$("pt-basin").hidden) addMethodOnce("pt-methods", "pt-sec-methods", BASIN_METHOD);
  if (!$("pt-similar").hidden) addMethodOnce("pt-methods", "pt-sec-methods", SIMILAR_METHOD);
}

// ── catchments (USGS NLDI, public domain) ───────────────────────────────────
// Upstream drainage basins come from the agency that publishes them under an
// open licence: the USGS Network Linked Data Index traces NHDPlus V2 catchments
// for any NWIS gauge and for any point in the US (nearest flowline). Nothing is
// stored on our side; HydroBASINS was ruled out because its licence forbids
// stand-alone redistribution.

const NLDI = "https://api.water.usgs.gov/nldi/linked-data";
const NLDI_METHOD = {
  name: "Upstream catchment (USGS NLDI)",
  text: "Drainage basin upstream of the gauge (or of the NHDPlus V2 flowline nearest to a clicked point), traced by the USGS Network Linked Data Index over NHDPlus V2 catchments; simplified geometry, area computed on the sphere. US only.",
  citation: "U.S. Geological Survey, Network Linked Data Index (NLDI) API, https://api.water.usgs.gov/nldi/; NHDPlus Version 2 (U.S. EPA and USGS). Public domain.",
};
let catchmentReq = 0;

function inNldiExtent(lat, lon) { return lat > 17 && lat < 72 && lon > -170 && lon < -64; }

function ringAreaKm2(ring) {
  // spherical excess (Chamberlain & Duquette 2007), as turf.js does; ring = [[lon, lat], ...]
  const R = 6371.0088, d2r = Math.PI / 180;
  let sum = 0;
  const n = ring.length;
  if (n < 3) return 0;
  for (let i = 0; i < n; i++) {
    const p1 = ring[i], p2 = ring[(i + 1) % n], p3 = ring[(i + 2) % n];
    sum += (p3[0] * d2r - p1[0] * d2r) * Math.sin(p2[1] * d2r);
  }
  return Math.abs(sum * R * R / 2);
}
function geojsonAreaKm2(geom) {
  if (!geom) return 0;
  const polys = geom.type === "Polygon" ? [geom.coordinates] : geom.type === "MultiPolygon" ? geom.coordinates : [];
  let a = 0;
  for (const poly of polys) { poly.forEach((ring, k) => { a += (k === 0 ? 1 : -1) * ringAreaKm2(ring); }); }
  return a;
}
function bboxOf(geom) {
  let w = 180, s = 90, e = -180, n = -90;
  const walk = (c) => { if (typeof c[0] === "number") { w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]); } else c.forEach(walk); };
  walk(geom.coordinates);
  return [[w, s], [e, n]];
}

function clearCatchment() {
  catchmentReq++;
  basinReq++;
  if (state.mapOk && map.getSource("catchment")) map.getSource("catchment").setData(EMPTY_FC);
  for (const id of ["st-catchment", "pt-catchment", "st-basin", "pt-basin", "st-similar", "pt-similar"]) { const el = $(id); if (el) { el.textContent = ""; el.hidden = true; } }
  highlightBasins([]);
}

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function requestCatchment({ station, point }) {
  const my = ++catchmentReq;
  let url = null, label = "";
  if (station) {
    if (station.source !== "usgs" || !/^USGS-\d+$/.test(station.station_id)) return;
    url = `${NLDI}/nwissite/${station.station_id}/basin?simplified=true&splitCatchment=false`;
    label = "st-catchment";
  } else if (point) {
    if (!inNldiExtent(point.lat, point.lon)) return;
    label = "pt-catchment";
  } else return;
  try {
    if (point) {
      const hl = await fetchJson(`${NLDI}/hydrolocation?coords=POINT(${point.lon} ${point.lat})`);
      const comid = (hl.features || []).map((f) => f.properties && f.properties.comid).find(Boolean);
      if (!comid || my !== catchmentReq) return;
      url = `${NLDI}/comid/${comid}/basin?simplified=true`;
    }
    const gj = await fetchJson(url);
    if (my !== catchmentReq) return; // the user moved on
    const feat = (gj.features || [])[0];
    if (!feat || !feat.geometry) return;
    const km2 = geojsonAreaKm2(feat.geometry);
    if (state.mapOk && map.getSource("catchment")) {
      map.getSource("catchment").setData({ type: "FeatureCollection", features: [feat] });
      if (km2 > 5) map.fitBounds(bboxOf(feat.geometry), { padding: 40, maxZoom: 11, duration: 700 });
    }
    const el = $(label);
    if (el) {
      const areaTxt = km2 >= 100 ? Math.round(km2).toLocaleString() : km2.toFixed(1);
      el.innerHTML = `${station ? "Upstream catchment" : "Catchment upstream of the nearest stream reach"}: <strong>${areaTxt} km²</strong> <span class="muted">(USGS NLDI, drawn on the map)</span>`;
      el.hidden = false;
    }
    addMethodOnce(station ? "methods" : "pt-methods", station ? "sec-methods" : "pt-sec-methods", NLDI_METHOD);
  } catch (err) {
    console.info("catchment unavailable:", err && err.message);
  }
}

function addMethodOnce(listId, sectionId, m) {
  const ol = $(listId);
  if (!ol || [...ol.querySelectorAll("li strong")].some((el) => el.textContent.startsWith(m.name))) return;
  const li = document.createElement("li");
  li.innerHTML = `<strong>${escapeHtml(m.name)}.</strong> ${escapeHtml(m.text)}<br><span class="cite">${escapeHtml(m.citation)}</span>`;
  ol.appendChild(li);
  $(sectionId).hidden = false;
}

// ── BasinATLAS: catchments everywhere (HydroATLAS v1.0, CC BY 4.0) ─────────
// The Archive publishes the level-12 sub-basins as PMTiles (drawn here), an
// indexed FlatGeobuf (which sub-basin contains a point: a few range reads),
// a topology parquet (upstream walk in DuckDB-WASM) and the attributes
// parquet (one row per sub-basin, BasinATLAS's own upstream-aggregated fields).

const BASIN_METHOD = {
  name: "Catchment attributes from BasinATLAS (HydroATLAS v1.0)",
  text: "Level-12 HydroBASINS sub-basin containing the point, its upstream sub-basins traced through NEXT_DOWN, and BasinATLAS's upstream-aggregated attributes (climate, land cover, soils, population, regulation) read for the outlet sub-basin.",
  citation: "Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6. CC BY 4.0.",
};
// [key, label, upstream field, sub-basin field, unit, divisor]
const BASIN_FIELDS = [
  ["elev", "mean elevation", "ele_mt_uav", "ele_mt_sav", "m", 1],
  ["slope", "mean slope", "slp_dg_uav", "slp_dg_sav", "°", 10],
  ["pre", "precipitation", "pre_mm_uyr", "pre_mm_syr", "mm/yr", 1],
  ["pet", "potential ET", "pet_mm_uyr", "pet_mm_syr", "mm/yr", 1],
  ["ari", "aridity P/PET", "ari_ix_uav", "ari_ix_sav", "", 100],
  ["tmp", "air temperature", "tmp_dc_uyr", "tmp_dc_syr", "°C", 10],
  ["snw", "snow cover", "snw_pc_uyr", "snw_pc_syr", "%", 1],
  ["dis", "natural discharge", "dis_m3_pyr", "dis_m3_pyr", "m³/s", 1],
  ["for", "forest", "for_pc_use", "for_pc_sse", "%", 1],
  ["crp", "cropland", "crp_pc_use", "crp_pc_sse", "%", 1],
  ["urb", "urban", "urb_pc_use", "urb_pc_sse", "%", 1],
  ["gla", "glacier", "gla_pc_use", "gla_pc_sse", "%", 1],
  ["lka", "lakes", "lka_pc_use", "lka_pc_sse", "%", 10],
  ["kar", "karst", "kar_pc_use", "kar_pc_sse", "%", 1],
  ["cly", "clay in soil", "cly_pc_uav", "cly_pc_sav", "%", 1],
  ["snd", "sand in soil", "snd_pc_uav", "snd_pc_sav", "%", 1],
  ["ppd", "population density", "ppd_pk_uav", "ppd_pk_sav", "/km²", 1],
  ["pop", "population", "pop_ct_usu", "pop_ct_ssu", "k people", 1],
  ["dor", "regulation by dams", "dor_pc_pva", "dor_pc_pva", "%", 10],
  ["hft", "human footprint", "hft_ix_u09", "hft_ix_s09", "/50", 10],
];
let basinReq = 0;
let basinsLayersOn = false;
let basinsLayersAdded = false;
let topoLoaded = null;

function basinsUrl(name) { return new URL(name, new URL(CONFIG.basinsBase, location.href)).href; } // absolute: DuckDB and pmtiles want full URLs

function ensureBasinsLayers() {
  if (basinsLayersAdded || !state.mapOk || !globalThis.pmtiles) return;
  try {
    if (!maplibregl.__aqPmtiles) { maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile); maplibregl.__aqPmtiles = true; }
    map.addSource("basins6", { type: "vector", url: `pmtiles://${basinsUrl("lev06.pmtiles")}` });
    map.addSource("basins12", { type: "vector", url: `pmtiles://${basinsUrl("lev12.pmtiles")}` });
    const before = map.getLayer("catchment-fill") ? "catchment-fill" : undefined;
    map.addLayer({ id: "basins6-line", type: "line", source: "basins6", "source-layer": "basins6", minzoom: 1, maxzoom: 7,
      layout: { visibility: "none" }, paint: { "line-color": "#6a1b9a", "line-opacity": 0.35, "line-width": 0.6 } }, before);
    map.addLayer({ id: "basins12-line", type: "line", source: "basins12", "source-layer": "basins", minzoom: 6,
      layout: { visibility: "none" }, paint: { "line-color": "#6a1b9a", "line-opacity": 0.4, "line-width": 0.5 } }, before);
    map.addLayer({ id: "basins12-up", type: "fill", source: "basins12", "source-layer": "basins", minzoom: 4,
      filter: ["in", ["get", "HYBAS_ID"], ["literal", []]],
      paint: { "fill-color": "#6a1b9a", "fill-opacity": 0.18, "fill-outline-color": "#4a148c" } }, before);
    basinsLayersAdded = true;
  } catch (err) {
    console.info("basins layers unavailable:", err && err.message);
  }
}

function setBasinsVisible(on) {
  basinsLayersOn = on;
  ensureBasinsLayers();
  if (!basinsLayersAdded) return;
  for (const id of ["basins6-line", "basins12-line"]) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  const chip = document.getElementById("chip-basins");
  if (chip) chip.classList.toggle("off", !on);
}

function highlightBasins(ids) {
  if (!basinsLayersAdded || !state.mapOk) return;
  // HYBAS_ID is stored as a number in the tiles; keep the list bounded for the expression evaluator
  const list = ids.slice(0, 20000).map(Number);
  map.setFilter("basins12-up", ["in", ["get", "HYBAS_ID"], ["literal", list]]);
}

function pointInRing(pt, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if (((yi > pt[1]) !== (yj > pt[1])) && (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}
function pointInGeom(pt, geom) {
  const polys = geom.type === "Polygon" ? [geom.coordinates] : geom.type === "MultiPolygon" ? geom.coordinates : [];
  return polys.some((poly) => pointInRing(pt, poly[0]) && !poly.slice(1).some((hole) => pointInRing(pt, hole)));
}

async function subBasinAt(lat, lon) {
  if (!globalThis.flatgeobuf) return null;
  const d = 0.02;
  const rect = { minX: lon - d, minY: lat - d, maxX: lon + d, maxY: lat + d };
  let best = null;
  for await (const f of flatgeobuf.deserialize(basinsUrl("lev12.fgb"), rect)) {
    if (f && f.geometry && pointInGeom([lon, lat], f.geometry)) { best = f; break; }
    if (f && !best) best = { ...f, __near: true };
  }
  if (!best) return null;
  const p = best.properties || {};
  const num = (k) => (p[k] === undefined || p[k] === null ? null : Number(p[k]));
  return { hybas_id: num("hybas_id") ?? num("HYBAS_ID"), next_down: num("next_down") ?? num("NEXT_DOWN"), main_bas: num("main_bas") ?? num("MAIN_BAS"),
    sub_area: num("sub_area") ?? num("SUB_AREA"), up_area: num("up_area") ?? num("UP_AREA"), pfaf_id: num("pfaf_id") ?? num("PFAF_ID"),
    geometry: best.geometry, approximate: Boolean(best.__near) };
}

async function ensureTopology() {
  if (topoLoaded) return topoLoaded;
  topoLoaded = (async () => {
    const { conn } = await duck();
    await conn.query(`CREATE TABLE IF NOT EXISTS topo AS SELECT hybas_id, next_down, up_area, sub_area FROM read_parquet('${basinsUrl("lev12_topology.parquet")}')`);
    return true;
  })();
  topoLoaded.catch(() => { topoLoaded = null; });
  return topoLoaded;
}

async function upstreamIds(hybasId, limit = 20000) {
  await ensureTopology();
  const { conn } = await duck();
  const res = await conn.query(`WITH RECURSIVE up AS (
      SELECT hybas_id FROM topo WHERE hybas_id = ${Number(hybasId)}
      UNION ALL SELECT t.hybas_id FROM topo t JOIN up ON t.next_down = up.hybas_id)
    SELECT hybas_id FROM up LIMIT ${limit + 1}`);
  return res.toArray().map((r) => Number(r.hybas_id));
}

async function basinAttributes(hybasId) {
  const { conn } = await duck();
  const res = await conn.query(`SELECT * FROM read_parquet('${basinsUrl("lev12_attributes.parquet")}') WHERE hybas_id = ${Number(hybasId)} LIMIT 1`);
  const rows = res.toArray().map((r) => r.toJSON());
  if (!rows.length) return null;
  const row = rows[0];
  const out = {};
  for (const [key, label, uf, sf, unit, div] of BASIN_FIELDS) {
    const raw = row[uf] ?? row[sf];
    if (raw === undefined || raw === null) continue;
    const v = Number(raw) / div;
    if (!Number.isFinite(v) || Number(raw) <= -9999) continue; // -9999 = no data in BasinATLAS
    out[key] = { label, value: v, unit, field: row[uf] !== undefined && row[uf] !== null ? uf : sf };
  }
  return out;
}

async function requestBasin(lat, lon, target) {
  const my = ++basinReq;
  const el = $(`${target}-basin`);
  if (!el) return;
  try {
    const sb = await subBasinAt(lat, lon);
    if (my !== basinReq) return;
    if (!sb || sb.hybas_id === null) return;
    const upArea = sb.up_area;
    el.innerHTML = `Catchment (BasinATLAS): <strong>${fmt(upArea, 0)} km²</strong> upstream · sub-basin ${sb.hybas_id}${sb.approximate ? ' <span class="muted">(nearest)</span>' : ""} <span class="muted">· loading attributes…</span>`;
    el.hidden = false;
    setBasinsVisible(true);
    const [ids, attrs] = await Promise.all([
      upstreamIds(sb.hybas_id).catch((e) => { console.info("upstream walk unavailable:", e && e.message); return [sb.hybas_id]; }),
      basinAttributes(sb.hybas_id).catch((e) => { console.info("basin attributes unavailable:", e && e.message); return null; }),
    ]);
    if (my !== basinReq) return;
    highlightBasins(ids);
    const capped = ids.length > 20000;
    const kv = attrs ? Object.values(attrs).map((a) => `<div><span>${escapeHtml(a.label)}</span><b>${fmt(a.value, a.unit === "%" ? 0 : 1)}${a.unit ? " " + a.unit : ""}</b></div>`).join("") : "";
    el.innerHTML = `Catchment (BasinATLAS level 12): <strong>${fmt(upArea, 0)} km²</strong> upstream, ${capped ? "over 20,000" : ids.length.toLocaleString()} sub-basins${sb.approximate ? ' <span class="muted">(nearest sub-basin)</span>' : ""}` +
      (kv ? `<div class="kv">${kv}</div>` : `<div class="muted">attributes unavailable</div>`) +
      `<div class="basin-foot muted">HydroATLAS v1.0, CC BY 4.0 (Linke et al. 2019). Upstream sub-basins highlighted on the map at zoom 4+.</div>`;
    addMethodOnce(target === "st" ? "methods" : "pt-methods", target === "st" ? "sec-methods" : "pt-sec-methods", BASIN_METHOD);
    if (attrs) requestSimilar({ lat, lon, upArea, attrs, target, my });
  } catch (err) {
    console.info("basin lookup unavailable:", err && err.message);
  }
}

// ── similar gauged basins (PUB-lite) ────────────────────────────────────────
// The harvest publishes basins/station_catchments.parquet: every catalog
// station with the BasinATLAS attributes of its catchment. Standardise the
// feature set over that table, measure the weighted distance to the target's
// catchment (plus distance on the ground: "combined"), list the closest gauges
// that measure discharge. Same features and weights as aquascope.archive.similar.

const SIMILAR_METHOD = {
  name: "Similar gauged basins (physical similarity / spatial proximity)",
  text: "Gauged stations ranked by weighted Euclidean distance in standardised BasinATLAS catchment attribute space (area, relief, climate, land cover, soils, human pressure) combined with great-circle distance (in units of 500 km); the donor-selection step of regionalisation.",
  citation: "Blöschl, G., Sivapalan, M., Wagener, T., Viglione, A., Savenije, H. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University Press; Oudin, L. et al. (2008). Spatial proximity, physical similarity, regression and ungaged catchments. Water Resour. Res. 44, W03413.",
};
// [table column, basin-card key, transform, weight]
const SIMILAR_FEATURES = [
  ["area_km2", "__area", "log10", 1.5], ["elevation_m", "elev", null, 1.0], ["slope_deg", "slope", "log1p", 1.0],
  ["precipitation_mm_yr", "pre", null, 1.5], ["aridity_index", "ari", "log1p", 1.5], ["temperature_c", "tmp", null, 1.0],
  ["snow_cover_pct", "snw", null, 1.0], ["forest_pct", "for", null, 0.7], ["cropland_pct", "crp", null, 0.7],
  ["urban_pct", "urb", "log1p", 0.7], ["clay_pct", "cly", null, 0.5], ["sand_pct", "snd", null, 0.5],
  ["population_density", "ppd", "log1p", 0.5], ["degree_of_regulation_pct", "dor", "log1p", 0.7],
];
let similarTable = null;  // { rows: [...], stats: {col: {mu, sd}} }
const tf = (v, how) => (how === "log10" ? Math.log10(Math.max(v, 1e-3)) : how === "log1p" ? Math.log1p(Math.max(v, 0)) : v);

async function ensureSimilarTable() {
  if (similarTable) return similarTable;
  const { conn } = await duck();
  const res = await conn.query(`SELECT * FROM read_parquet('${basinsUrl("station_catchments.parquet")}')`);
  const rows = res.toArray().map((r) => r.toJSON());
  const stats = {};
  for (const [col, , how] of SIMILAR_FEATURES) {
    const xs = rows.map((r) => (r[col] === null || r[col] === undefined ? NaN : tf(Number(r[col]), how))).filter((x) => Number.isFinite(x));
    const mu = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(xs.length, 1)) || 1;
    stats[col] = { mu, sd };
  }
  similarTable = { rows, stats };
  return similarTable;
}

async function requestSimilar({ lat, lon, upArea, attrs, target, my }) {
  const el = $(`${target}-similar`);
  if (!el) return;
  try {
    const { rows, stats } = await ensureSimilarTable();
    if (my !== basinReq) return;
    const selfKey = target === "st" && state.selected ? `${state.selected.source}/${state.selected.station_id}` : null;
    // A station's own table row is the better target: it carries the agency's area and the attribute scope
    // (a small creek inside a big river's sub-basin must not be described by the big river).
    const selfRow = selfKey ? rows.find((r) => `${r.source}/${r.station_id}` === selfKey) : null;
    const tvals = {};
    for (const [col, key] of SIMILAR_FEATURES) {
      let v;
      if (selfRow && selfRow[col] !== null && selfRow[col] !== undefined) v = selfRow[col];
      else v = key === "__area" ? upArea : (attrs[key] && attrs[key].value);
      if (v !== undefined && v !== null && Number.isFinite(Number(v))) tvals[col] = Number(v);
    }
    const used = SIMILAR_FEATURES.filter(([col]) => tvals[col] !== undefined);
    if (!used.length) return;
    const wsum = used.reduce((a, [, , , w]) => a + w * w, 0);
    const scored = [];
    for (const r of rows) {
      const key = `${r.source}/${r.station_id}`;
      if (key === selfKey) continue;
      const st = state.byKey.get(key);
      if (!st || !(st.variables || []).includes("discharge")) continue;
      let acc = 0;
      for (const [col, , how, w] of used) {
        const { mu, sd } = stats[col];
        const zt = (tf(tvals[col], how) - mu) / sd;
        const v = r[col];
        const z = v === null || v === undefined ? NaN : (tf(Number(v), how) - mu) / sd;
        const d = Number.isFinite(z) ? Math.abs(z - zt) : 3.0;
        acc += (d * w) ** 2;
      }
      const dAttr = Math.sqrt(acc / wsum);
      const dKm = haversineKm(lat, lon, st.lat, st.lon);
      scored.push({ st, dAttr, dKm, score: Math.sqrt(dAttr ** 2 + (dKm / 500) ** 2), area: Number(r.area_km2 ?? r.up_area) });
    }
    scored.sort((a, b) => a.score - b.score);
    const top = scored.slice(0, 8);
    if (!top.length) return;
    el.innerHTML = `<h3>Similar gauged basins <span class="muted">catchment attributes + distance</span></h3><ul class="nearest similar"></ul>` +
      `<div class="basin-foot muted">Donor candidates for an ungauged site: ${scored.length.toLocaleString()} gauged catchments compared on ${used.length} standardised BasinATLAS attributes and distance. Click one to open it.</div>`;
    const ul = el.querySelector("ul");
    for (const s of top) {
      const li = document.createElement("li");
      li.dataset.key = `${s.st.source}/${s.st.station_id}`;
      li.innerHTML = `<i style="background:${(SOURCE_STYLE[s.st.source] || {}).color || FALLBACK_COLOR}"></i>${escapeHtml(s.st.name || s.st.station_id)} <span class="muted">${escapeHtml((SOURCE_STYLE[s.st.source] || {}).label || s.st.source)} · ${fmt(s.area, 0)} km²</span><span class="dist">${s.dKm < 10 ? s.dKm.toFixed(1) : Math.round(s.dKm).toLocaleString()} km · Δ${s.dAttr.toFixed(2)}</span>`;
      li.addEventListener("click", () => selectStation(li.dataset.key, { fly: true }));
      ul.appendChild(li);
    }
    el.hidden = false;
    addMethodOnce(target === "st" ? "methods" : "pt-methods", target === "st" ? "sec-methods" : "pt-sec-methods", SIMILAR_METHOD);
    renderRegime(el, target, scored, my).catch((err) => console.info("flow regime unavailable:", err && err.message));
  } catch (err) {
    console.info("similar basins unavailable:", err && err.message);
  }
}

// ── estimated flow regime (regionalisation) ─────────────────────────────────
// The signatures of every gauged donor (basins/station_signatures.parquet, built
// weekly from the archived discharge) transferred to the target as a
// similarity-weighted mean over the k most similar donors (geometric mean for
// the mm/d magnitudes, weights 1/(distance + 0.05)), with one weighted standard
// deviation as the band and the published leave-one-out skill next to each
// number. Same arithmetic as aquascope.archive.regionalize (method "similarity").

const REGIME_METHOD = {
  name: "Regionalisation of flow signatures from similar gauged basins",
  text: "Flow signatures of the gauged donors (mean, median, Q95 and Q05 daily flow in mm/d, mean annual maximum, baseflow index, seasonality, flashiness) transferred as an inverse-distance-weighted average over the 10 most similar catchments in standardised BasinATLAS attribute space (geometric mean for magnitudes); the band is one weighted standard deviation of the donors; the skill is leave-one-out over all donors.",
  citation: "Blöschl, G. et al. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University Press; Oudin, L. et al. (2008). Water Resour. Res. 44, W03413; Addor, N. et al. (2018). A ranking of hydrological signatures based on their predictability in space. Water Resour. Res. 54, 8792-8812.",
};
// [column, label, unit, log?, lower bound, upper bound]
const REGIME_ROWS = [
  ["q_mean_mm", "Mean flow", "mm/d", true], ["q95_mm", "Low flow (Q95)", "mm/d", true], ["q05_mm", "High flow (Q05)", "mm/d", true],
  ["q_annual_max_mm", "Mean annual max", "mm/d", true], ["baseflow_index", "Baseflow index", "", false, 0, 1],
  ["seasonality_index", "Seasonality", "", false, 0, 1], ["flashiness_index", "Flashiness", "", false, 0],
];
let regimeData = null;  // { sig: Map key -> row, skill: {...} | null }
async function ensureRegimeData() {
  if (regimeData) return regimeData;
  const { conn } = await duck();
  const res = await conn.query(`SELECT * FROM read_parquet('${basinsUrl("station_signatures.parquet")}')`);
  const sig = new Map();
  for (const r of res.toArray().map((x) => x.toJSON())) sig.set(`${r.source}/${r.station_id}`, r);
  let skill = null;
  try { const rs = await fetch(basinsUrl("regionalization_skill.json")); if (rs.ok) skill = await rs.json(); } catch (_) { /* optional */ }
  regimeData = { sig, skill };
  return regimeData;
}

async function renderRegime(el, target, scored, my) {
  const { sig, skill } = await ensureRegimeData();
  if (my !== basinReq) return;
  const donors = scored.filter((s) => sig.has(`${s.st.source}/${s.st.station_id}`)).sort((a, b) => a.dAttr - b.dAttr).slice(0, 10);
  if (donors.length < 3) return;
  const w = donors.map((d) => 1 / (d.dAttr + 0.05));
  const per = ((skill || {}).methods || {}).similarity || {};
  const rows = [];
  for (const [col, label, unit, isLog, lo, hi] of REGIME_ROWS) {
    const vals = [], ws = [];
    donors.forEach((d, i) => { const v = sig.get(`${d.st.source}/${d.st.station_id}`)[col]; if (v !== null && v !== undefined && Number.isFinite(Number(v))) { vals.push(Number(v)); ws.push(w[i]); } });
    if (!vals.length) continue;
    const wsum = ws.reduce((a, b) => a + b, 0);
    const y = vals.map((v) => (isLog ? Math.log(Math.max(v, 1e-3)) : v));
    const mean = y.reduce((a, v, i) => a + (ws[i] / wsum) * v, 0);
    const sd = Math.sqrt(y.reduce((a, v, i) => a + (ws[i] / wsum) * (v - mean) ** 2, 0));
    const back = (v) => { let out = isLog ? Math.exp(v) : v; if (lo !== undefined) out = Math.max(lo, out); if (hi !== undefined) out = Math.min(hi, out); return out; };
    const sk = per[col];
    rows.push({ label, unit, value: back(mean), low: back(mean - sd), high: back(mean + sd), n: vals.length, sk });
  }
  if (!rows.length) return;
  const digits = (v) => (v >= 100 ? 0 : v >= 10 ? 1 : v >= 1 ? 2 : 3);
  const box = document.createElement("div");
  box.className = "regime";
  box.innerHTML = `<h3>Estimated flow regime <span class="muted">${donors.length} most similar donors, weighted</span></h3>` +
    `<table class="ffa regime"><thead><tr><th>Signature</th><th>Estimate</th><th>Band</th><th>LOO</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${escapeHtml(r.label)}</td><td>${fmt(r.value, digits(r.value))}${r.unit ? " " + r.unit : ""}</td>` +
      `<td class="ci">${fmt(r.low, digits(r.low))} to ${fmt(r.high, digits(r.high))}</td>` +
      `<td class="ci">${r.sk && r.sk.nse !== null && r.sk.nse !== undefined ? "NSE " + Number(r.sk.nse).toFixed(2) + (r.sk.median_ape !== null && r.sk.median_ape !== undefined ? ", ±" + Math.round(Number(r.sk.median_ape) * 100) + " %" : "") : "–"}</td></tr>`).join("") +
    `</tbody></table><div class="basin-foot muted">Prediction in ungauged basins: what the ${donors.length} closest donors in attribute space would suggest here (geometric mean for mm/d; band = one weighted standard deviation of the donors; LOO = leave-one-out skill over ${(skill && skill.n_stations) ? skill.n_stations.toLocaleString() : "all"} donors, NSE and median error). Not a measurement.</div>`;
  el.appendChild(box);
  addMethodOnce(target === "st" ? "methods" : "pt-methods", target === "st" ? "sec-methods" : "pt-sec-methods", REGIME_METHOD);
}

// ── worker (Pyodide) ────────────────────────────────────────────────────────

let worker;
function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(`./worker.js?v=${CONFIG.build}`);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { if (state.selected && !state.result) setStatus(m.text, "info"); return; }
    if (m.type === "ask_progress") { askLog(m.text); return; }
    if (m.type === "ready") { state.workerReady = true; return; }
    const pending = state.pending.get(m.id);
    if (!pending) return;
    state.pending.delete(m.id);
    if (m.type === "error") pending.reject(new Error(m.message)); else pending.resolve(m.result);
  };
  worker.onerror = (e) => { console.error(e); setStatus(`Worker error: ${e.message}`, "error"); };
  worker.postMessage({ type: "init", pyodideIndexURL: CONFIG.pyodideIndexURL, wheelsJson: new URL(CONFIG.wheelsJson, location.href).href, build: CONFIG.build });
  return worker;
}

function call(type, payload) {
  ensureWorker();
  const id = ++state.reqId;
  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...payload });
  });
}

async function requestAnalysis(r) {
  const key = `${r.source}/${r.station_id}`;
  setStatus(state.workerReady ? "Fetching the record from the agency…" : "Loading Python in your browser (once, ~15 MB)…", "info");
  try {
    const result = await call("analyze", { source: r.source, station_id: r.station_id, years: CONFIG.years });
    if (!state.selected || `${state.selected.source}/${state.selected.station_id}` !== key) return; // user moved on
    state.result = result;
    render(result, r);
  } catch (err) {
    if (!state.selected || `${state.selected.source}/${state.selected.station_id}` !== key) return;
    setStatus(`Could not analyse this station: ${err.message}`, "error");
  }
}

// ── rendering ───────────────────────────────────────────────────────────────

const PLOT_LAYOUT = { margin: { l: 48, r: 12, t: 8, b: 36 }, height: 240, font: { family: "system-ui, sans-serif", size: 11 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" };
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function fmt(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const ax = Math.abs(x);
  if (ax >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (ax >= 10) return x.toLocaleString(undefined, { maximumFractionDigits: digits });
  return x.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function render(res, r) {
  const st = SOURCE_STYLE[r.source] || { color: FALLBACK_COLOR };
  if (res.error || !res.n) {
    setStatus(res.error || "No observations returned.", "warn");
    renderNotes(res); renderMethods(res);
    return;
  }
  setStatus("", "info");
  $("btn-csv").disabled = false;
  const unit = res.unit || "";
  const varLabel = VAR_LABEL[res.variable] || res.variable;

  // KPIs
  const k = res.stats || {};
  $("kpis").innerHTML = [
    ["record", `${res.start} → ${res.end}`, `${res.years} yr · ${res.n.toLocaleString()} obs`],
    ["mean", `${fmt(k.mean)} ${unit}`, varLabel],
    ["max", `${fmt(k.max)} ${unit}`, "observed"],
    ["min", `${fmt(k.min)} ${unit}`, "observed"],
  ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  $("sec-summary").hidden = false;

  // hydrograph
  $("sec-hydro").hidden = false;
  const traces = [{ x: res.series.t, y: res.series.v, mode: "lines", line: { width: 1, color: st.color }, name: varLabel, hovertemplate: "%{x}<br>%{y:.3~f} " + unit + "<extra></extra>" }];
  if (res.annual_max && res.annual_max.year.length > 1) {
    traces.push({ x: res.annual_max.year.map((y) => `${y}-07-01`), y: res.annual_max.v, mode: "markers", marker: { color: "#e53935", size: 6 }, name: "annual max", hovertemplate: "%{x|%Y} annual max<br>%{y:.3~f} " + unit + "<extra></extra>" });
  }
  Plotly.react("plot-hydro", traces, { ...PLOT_LAYOUT, yaxis: { title: { text: unit }, rangemode: "tozero" }, showlegend: false }, PLOT_CONFIG);

  // FFA
  if (res.ffa && res.ffa.fits) {
    $("sec-ffa").hidden = false;
    $("ffa-years").textContent = `(${res.ffa.n_years} annual maxima)`;
    renderFfaTable(res.ffa, unit);
    renderFfaPlot(res.ffa, unit, st.color);
  }

  // FDC
  if (res.fdc) {
    $("sec-fdc").hidden = false;
    Plotly.react("plot-fdc", [{ x: res.fdc.exceedance, y: res.fdc.q, mode: "lines", line: { color: st.color, width: 2 }, hovertemplate: "%{x:.1f} % exceedance<br>%{y:.3~f} " + unit + "<extra></extra>" }],
      { ...PLOT_LAYOUT, xaxis: { title: { text: "% of time exceeded" }, range: [0, 100] }, yaxis: { title: { text: unit }, type: "log" }, showlegend: false,
        annotations: [{ x: 95, y: Math.log10(res.fdc.q95 || 1), text: `Q95 ${fmt(res.fdc.q95)}`, showarrow: true, arrowhead: 2, ax: -40, ay: -30 }, { x: 10, y: Math.log10(res.fdc.q10 || 1), text: `Q10 ${fmt(res.fdc.q10)}`, showarrow: true, arrowhead: 2, ax: 40, ay: -30 }] }, PLOT_CONFIG);
  }

  // GR4J: discharge records in m3/s can be modelled (needs a catchment area and Open-Meteo forcing; on demand)
  if (res.variable === "discharge" && /m3\/s|m³\/s/.test(unit) && res.series && res.series.t.length > 365 * 4 && window.GR4J) {
    $("sec-gr4j").hidden = false;
  }

  // trend
  if (res.trend) {
    $("sec-trend").hidden = false;
    const t = res.trend;
    const dir = t.trend === "no trend" ? "no significant trend" : `a ${t.trend} trend`;
    $("trend-text").innerHTML = `Mann-Kendall on ${t.n_years} annual means: <strong>${dir}</strong> (p = ${fmt(t.p_value, 3)}, τ = ${fmt(t.tau, 2)}). Sen's slope ${fmt(t.sens_slope_per_year, 3)} ${unit}/yr.`;
  }

  renderNotes(res); renderMethods(res);
}

function renderNotes(res) {
  const notes = [...(res.notes || [])];
  if (res.fetch_note) notes.unshift(res.fetch_note);
  $("notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  $("sec-notes").hidden = notes.length === 0;
}

function renderMethods(res) {
  const ms = res.methods || [];
  $("methods").innerHTML = ms.map((m) => `<li><strong>${escapeHtml(m.name)}.</strong> ${escapeHtml(m.text)}<br><span class="cite">${escapeHtml(m.citation)}</span></li>`).join("");
  $("attribution").textContent = res.attribution ? `Data: ${res.attribution}. Licence: ${res.license}. Computed with aquascope in your browser.` : "";
  $("sec-methods").hidden = ms.length === 0 && !res.attribution;
  if (!$("st-catchment").hidden) addMethodOnce("methods", "sec-methods", NLDI_METHOD);
  if (!$("st-basin").hidden) addMethodOnce("methods", "sec-methods", BASIN_METHOD);
  if (!$("st-similar").hidden) addMethodOnce("methods", "sec-methods", SIMILAR_METHOD);
  if (!$("gr4j-out").hidden) for (const m of GR4J_METHODS) addMethodOnce("methods", "sec-methods", m);
}

// ── GR4J in the page ────────────────────────────────────────────────────────
// explorer/gr4j.js is a straight port of aquascope.models.rainfall_runoff.GR4J
// (checked against it to round-off in the test suite) plus a small differential
// evolution. Forcing: Open-Meteo's ERA5-Land/ERA5 blend at the gauge point
// (daily precipitation, FAO-56 ET0), the same recipe as the Caravan exporter and
// HydroGym; area: the station's catchment row (agency, else BasinATLAS).

const GR4J_METHODS = [
  { name: "GR4J rainfall-runoff model, calibrated by differential evolution",
    text: "Four-parameter daily lumped model (production store X1, groundwater exchange X2, routing store X3, unit-hydrograph time base X4) run on ERA5-Land/ERA5 rainfall and FAO-56 ET0 at the gauge; parameters found by differential evolution (best/1/bin, population 20, 40 generations) maximising KGE on the first 65 % of the record after a one-year warm-up; the last 35 % is validation only.",
    citation: "Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. J. Hydrol. 279, 275-289; Storn, R., Price, K. (1997). Differential evolution. J. Global Optim. 11, 341-359; Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE performance criteria. J. Hydrol. 377, 80-91." },
  { name: "Forcing at the gauge point",
    text: "Daily precipitation and FAO-56 Penman-Monteith reference ET0 from Open-Meteo's historical weather API (ERA5-Land where available, ERA5 elsewhere), at the gauge coordinates, not catchment-averaged.",
    citation: "Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst. Sci. Data 13, 4349-4383; Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999-2049; Open-Meteo (CC BY 4.0)." },
];
let gr4jRun = 0;

function resetGr4j() {
  gr4jRun++;
  const out = $("gr4j-out"); if (out) out.hidden = true;
  const st = $("gr4j-status"); if (st) st.textContent = "";
  const btn = $("btn-gr4j"); if (btn) { btn.disabled = false; btn.textContent = "Calibrate GR4J"; }
}

async function stationArea(key) {
  try {
    const { rows } = await ensureSimilarTable();
    const row = rows.find((r) => `${r.source}/${r.station_id}` === key);
    if (row && row.area_km2) return { area: Number(row.area_km2), source: row.area_source === "agency" ? "agency" : "BasinATLAS upstream area" };
  } catch (err) { console.info("catchment table unavailable:", err && err.message); }
  return null;
}

async function runGr4j() {
  const r = state.selected, res = state.result;
  if (!r || !res || !res.series) return;
  const key = `${r.source}/${r.station_id}`;
  const my = ++gr4jRun;
  const btn = $("btn-gr4j"), status = $("gr4j-status");
  btn.disabled = true; btn.textContent = "Calibrating…";
  const say = (t) => { if (my === gr4jRun) status.textContent = t; };
  try {
    say("Looking up the catchment area…");
    const area = await stationArea(key);
    if (my !== gr4jRun) return;
    if (!area) { say("No catchment area for this gauge (not in the station catchments table), so flow cannot be expressed in mm/d."); btn.disabled = false; btn.textContent = "Calibrate GR4J"; return; }
    const t = res.series.t, v = res.series.v;
    let start = t[0], end = t[t.length - 1];
    if (start < "1940-01-02") start = "1940-01-02";
    say("Fetching ERA5-Land/ERA5 rainfall and ET0 at the gauge (Open-Meteo)…");
    const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${r.lat}&longitude=${r.lon}&start_date=${start}&end_date=${end}&daily=precipitation_sum,et0_fao_evapotranspiration&models=best_match&timezone=UTC`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Open-Meteo ${resp.status}${resp.status === 429 ? " (rate limit, try again in a minute)" : ""}`);
    const om = await resp.json();
    if (my !== gr4jRun) return;
    const days = om.daily.time, P = om.daily.precipitation_sum, E = om.daily.et0_fao_evapotranspiration;
    const obsByDay = new Map(); for (let i = 0; i < t.length; i++) obsByDay.set(t[i], v[i]);
    const n = days.length;
    const precip = new Float64Array(n), pet = new Float64Array(n), obs = new Array(n).fill(NaN);
    let lastE = 2.0;
    for (let i = 0; i < n; i++) {
      precip[i] = Number.isFinite(P[i]) && P[i] > 0 ? P[i] : 0;
      if (Number.isFinite(E[i])) lastE = Math.max(0, E[i]);
      pet[i] = lastE;
      const o = obsByDay.get(days[i]);
      if (o !== undefined && o !== null && Number.isFinite(o) && o >= 0) obs[i] = o * 86.4 / area.area;  // m3/s -> mm/d
    }
    const nObs = obs.filter(Number.isFinite).length;
    if (nObs < 365 * 3) { say(`Only ${nObs} days of overlap between the record and the forcing; not enough to calibrate.`); btn.disabled = false; btn.textContent = "Calibrate GR4J"; return; }
    const calEnd = Math.floor(n * 0.65);
    const t0 = performance.now();
    const fit = await GR4J.calibrate(precip, pet, obs, { objective: "kge", warmup: 365, calEnd, popsize: 20, generations: 40, seed: 1,
      onProgress: (g, best, G) => say(`Differential evolution: generation ${g}/${G}, best KGE ${best.toFixed(3)}…`) });
    if (my !== gr4jRun) return;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    say("");
    const c = fit.calibration, val = fit.validation || {};
    const f3 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(3));
    const f1 = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(1));
    $("gr4j-table").innerHTML = `<thead><tr><th>Parameter</th><th>Value</th><th>Metric</th><th>Calibration<br><span class="ci">${days[365]} to ${days[calEnd - 1]}</span></th><th>Validation<br><span class="ci">${days[calEnd]} to ${days[n - 1]}</span></th></tr></thead><tbody>` +
      `<tr><td>X1 production store</td><td>${fit.params.X1.toFixed(0)} mm</td><td>KGE</td><td>${f3(c.kge)}</td><td>${f3(val.kge)}</td></tr>` +
      `<tr><td>X2 exchange</td><td>${fit.params.X2.toFixed(2)} mm/d</td><td>NSE</td><td>${f3(c.nse)}</td><td>${f3(val.nse)}</td></tr>` +
      `<tr><td>X3 routing store</td><td>${fit.params.X3.toFixed(0)} mm</td><td>log-NSE</td><td>${f3(c.log_nse)}</td><td>${f3(val.log_nse)}</td></tr>` +
      `<tr><td>X4 UH time base</td><td>${fit.params.X4.toFixed(2)} d</td><td>PBIAS</td><td>${f1(c.pbias)} %</td><td>${f1(val.pbias)} %</td></tr></tbody>`;
    // plot: last 6 years, observed vs simulated, in the station's unit
    const from = Math.max(0, n - 365 * 6);
    const toUnit = (mm) => mm * area.area / 86.4;
    const st = SOURCE_STYLE[r.source] || { color: FALLBACK_COLOR };
    Plotly.react("plot-gr4j", [
      { x: days.slice(from), y: Array.from(obs.slice(from), (o) => (Number.isFinite(o) ? toUnit(o) : null)), mode: "lines", line: { width: 1, color: st.color }, name: "observed", hovertemplate: "%{x}<br>obs %{y:.3~f} " + res.unit + "<extra></extra>" },
      { x: days.slice(from), y: Array.from(fit.sim.slice(from), toUnit), mode: "lines", line: { width: 1, color: "#e53935" }, name: "GR4J", hovertemplate: "%{x}<br>sim %{y:.3~f} " + res.unit + "<extra></extra>" },
    ], { ...PLOT_LAYOUT, yaxis: { title: { text: res.unit }, rangemode: "tozero" }, legend: { orientation: "h", y: 1.15 }, shapes: calEnd > from ? [{ type: "line", x0: days[calEnd], x1: days[calEnd], y0: 0, y1: 1, yref: "paper", line: { color: "#999", dash: "dot" } }] : [] }, PLOT_CONFIG);
    $("gr4j-foot").textContent = `Last six years shown (dotted line: start of the validation period). Area ${area.area.toLocaleString(undefined, { maximumFractionDigits: 0 })} km² (${area.source}); forcing at the gauge point, not catchment-averaged, so wet mountainous catchments will under-run. ${fit.simulations} simulations in ${secs} s. Point values, not a forecast.`;
    $("gr4j-out").hidden = false;
    btn.textContent = "Recalibrate";
    btn.disabled = false;
    for (const m of GR4J_METHODS) addMethodOnce("methods", "sec-methods", m);
  } catch (err) {
    if (my !== gr4jRun) return;
    say(`GR4J failed: ${err.message}`);
    btn.disabled = false; btn.textContent = "Calibrate GR4J";
  }
}

function renderFfaTable(ffa, unit) {
  const rps = ffa.return_periods;
  const g = ffa.fits.gev_lmoments || {}, l = ffa.fits.lp3 || {}, b = ffa.fits.gev_bootstrap;
  const head = `<tr><th>T (yr)</th><th>GEV L-moments</th><th>LP3 (90 % CI)</th>${b ? "<th>GEV bootstrap (90 % CI)</th>" : ""}</tr>`;
  const rows = rps.map((rp, i) => {
    const gq = g.q ? fmt(g.q[i]) : (g.error ? "n/a" : "—");
    const lq = l.q ? `${fmt(l.q[i])} <span class="ci">${l.ci && l.ci[i] ? `[${fmt(l.ci[i][0])}, ${fmt(l.ci[i][1])}]` : ""}</span>` : (l.error ? "n/a" : "—");
    const bq = b ? `${fmt(b.q[i])} <span class="ci">${b.ci && b.ci[i] ? `[${fmt(b.ci[i][0])}, ${fmt(b.ci[i][1])}]` : ""}</span>` : "";
    return `<tr><td>${rp}</td><td>${gq}</td><td>${lq}</td>${b ? `<td>${bq}</td>` : ""}</tr>`;
  }).join("");
  $("ffa-table").innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody><tfoot><tr><td colspan="${b ? 4 : 3}" class="muted">Return levels in ${unit}. T = return period.</td></tr></tfoot>`;
}

function renderFfaPlot(ffa, unit, color) {
  const rps = ffa.return_periods;
  const traces = [];
  if (ffa.fits.gev_lmoments && ffa.fits.gev_lmoments.q) traces.push({ x: rps, y: ffa.fits.gev_lmoments.q, mode: "lines+markers", name: "GEV (L-moments)", line: { color } });
  if (ffa.fits.lp3 && ffa.fits.lp3.q) {
    traces.push({ x: rps, y: ffa.fits.lp3.q, mode: "lines+markers", name: "LP3", line: { color: "#8e24aa" } });
    if (ffa.fits.lp3.ci) {
      traces.push({ x: [...rps, ...rps.slice().reverse()], y: [...ffa.fits.lp3.ci.map((c) => c[1]), ...ffa.fits.lp3.ci.map((c) => c[0]).reverse()], fill: "toself", fillcolor: "rgba(142,36,170,0.12)", line: { color: "transparent" }, name: "LP3 90 % CI", hoverinfo: "skip" });
    }
  }
  if (ffa.fits.gev_bootstrap) {
    const b = ffa.fits.gev_bootstrap;
    traces.push({ x: rps, y: b.q, mode: "lines+markers", name: "GEV (MLE)", line: { color: "#f57c00", dash: "dot" } });
    traces.push({ x: [...rps, ...rps.slice().reverse()], y: [...b.ci.map((c) => c[1]), ...b.ci.map((c) => c[0]).reverse()], fill: "toself", fillcolor: "rgba(245,124,0,0.12)", line: { color: "transparent" }, name: "GEV 90 % CI", hoverinfo: "skip" });
  }
  Plotly.react("plot-ffa", traces, { ...PLOT_LAYOUT, height: 260, xaxis: { title: { text: "return period (years)" }, type: "log" }, yaxis: { title: { text: unit } }, legend: { orientation: "h", y: 1.15 } }, PLOT_CONFIG);
}

// ── Ask (the Analyst in the browser) ────────────────────────────────────────
// aquascope.ai_engine.analyst.ask runs inside the Pyodide worker: the model
// call goes from the worker straight to the provider (bring your own key), the
// tools run on the catalog held here plus the agencies' APIs, and the report
// ends with Data and Methods sections assembled from tool results.

const ASK_PROVIDERS = {
  groq: { base_url: "https://api.groq.com/openai/v1", model: "openai/gpt-oss-120b" },
  huggingface: { base_url: "https://router.huggingface.co/v1", model: "Qwen/Qwen2.5-72B-Instruct" },
  openai: { base_url: "https://api.openai.com/v1", model: "gpt-4o-mini" },
  mistral: { base_url: "https://api.mistral.ai/v1", model: "mistral-small-latest" },
  openrouter: { base_url: "https://openrouter.ai/api/v1", model: "openai/gpt-4o-mini" },
  custom: { base_url: "", model: "" },
};
const ASK_EXAMPLES = [
  "What is the 100-year flood of the Thames at Kingston, and how sure can we be?",
  "Compare the low flows (Q95) of the Seine at Paris and the Loire at Blois.",
  "Is the Potomac at Little Falls getting drier? Use the annual-mean trend.",
  "How wet is Taipei compared with London, and what is the aridity class of each?",
  "Which UK boreholes near Cambridge have the longest groundwater records?",
];
const ASK_STORE = "aquascope.ask.settings";

function askSettings() {
  try { return JSON.parse(localStorage.getItem(ASK_STORE) || sessionStorage.getItem(ASK_STORE) || "{}"); } catch { return {}; }
}
function saveAskSettings() {
  const remember = $("ask-remember").checked;
  const data = { provider: $("ask-provider").value, model: $("ask-model").value, base_url: $("ask-base-url").value, remember };
  const withKey = { ...data, key: $("ask-key").value };
  try {
    if (remember) { localStorage.setItem(ASK_STORE, JSON.stringify(withKey)); sessionStorage.removeItem(ASK_STORE); }
    else { sessionStorage.setItem(ASK_STORE, JSON.stringify(withKey)); localStorage.setItem(ASK_STORE, JSON.stringify(data)); }
  } catch { /* storage blocked: fine, the tab keeps the values */ }
}

function initAsk() {
  const provider = $("ask-provider"), model = $("ask-model"), baseRow = $("ask-base-url-row"), base = $("ask-base-url");
  const saved = askSettings();
  if (saved.provider && ASK_PROVIDERS[saved.provider]) provider.value = saved.provider;
  const applyProvider = (keepModel) => {
    const p = ASK_PROVIDERS[provider.value];
    baseRow.hidden = provider.value !== "custom";
    if (!keepModel) model.value = p.model;
    if (provider.value !== "custom") base.value = p.base_url;
    model.placeholder = p.model || "model id";
  };
  applyProvider(Boolean(saved.model));
  if (saved.model) model.value = saved.model;
  if (saved.base_url && provider.value === "custom") base.value = saved.base_url;
  if (saved.key) $("ask-key").value = saved.key;
  $("ask-remember").checked = Boolean(saved.remember);
  $("ask-settings").open = !saved.key;
  provider.addEventListener("change", () => applyProvider(false));
  for (const el of [provider, model, base, $("ask-key"), $("ask-remember")]) el.addEventListener("change", saveAskSettings);

  const ex = $("ask-examples");
  for (const q of ASK_EXAMPLES) {
    const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => { $("ask-question").value = q; $("ask-question").focus(); });
    ex.appendChild(b);
  }
  $("btn-ask").addEventListener("click", openAsk);
  $("ask-run").addEventListener("click", runAsk);
  $("ask-question").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAsk(); });
  $("ask-copy").addEventListener("click", async () => {
    if (!state.ask.markdown) return;
    try { await navigator.clipboard.writeText(state.ask.markdown); $("ask-copy").textContent = "Copied!"; setTimeout(() => ($("ask-copy").textContent = "Copy Markdown"), 1500); }
    catch { prompt("Copy the report", state.ask.markdown); }
  });
  $("ask-download").addEventListener("click", () => {
    if (!state.ask.markdown) return;
    const blob = new Blob([state.ask.markdown], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "aquascope-answer.md";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
}

function openAsk() {
  $("panel-empty").hidden = true; $("panel-station").hidden = true; $("panel-point").hidden = true;
  $("panel-ask").hidden = false;
  const q = $("ask-question");
  if (state.selected && !q.value.trim()) {
    const r = state.selected;
    q.value = `Summarise the record of ${r.name || r.station_id} (${r.source} / ${r.station_id}): period, mean, trend, and the flood frequency if the record allows it.`;
  }
  const chip = document.getElementById("ask-this-station");
  if (chip) chip.remove();
  if (state.selected) {
    const r = state.selected;
    const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.id = "ask-this-station";
    b.textContent = `About ${r.name || r.station_id}`;
    b.addEventListener("click", () => { q.value = `Summarise the record of ${r.name || r.station_id} (${r.source} / ${r.station_id}): period, mean, trend, and the flood frequency if the record allows it.`; q.focus(); });
    $("ask-examples").prepend(b);
  }
  $("panel").scrollTop = 0;
  q.focus();
}

function askStatus(text, kind = "info") {
  const el = $("ask-status");
  el.textContent = text; el.className = `status ${kind}`; el.hidden = !text;
}
function askLog(text) {
  const log = $("ask-log");
  log.hidden = false;
  const li = document.createElement("li");
  const m = String(text).match(/^tool (\w+)\((.*)\)$/);
  li.innerHTML = m ? `<code>${escapeHtml(m[1])}</code>(${escapeHtml(m[2]).slice(0, 160)})` : escapeHtml(text);
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
}

async function ensureCatalogInWorker() {
  if (state.ask.catalogSent) return;
  const rows = state.stations.map((r) => ({
    source: r.source, station_id: r.station_id, name: r.name, latitude: r.lat, longitude: r.lon,
    variables: r.variables || [], period_start: r.period_start, period_end: r.period_end, url: r.url,
    agency: (SOURCE_STYLE[r.source] || {}).label || r.source,
  }));
  await call("catalog", { rows });
  state.ask.catalogSent = true;
}

async function runAsk() {
  if (state.ask.running) return;
  const question = $("ask-question").value.trim();
  const provider = $("ask-provider").value, key = $("ask-key").value.trim();
  const model = $("ask-model").value.trim() || ASK_PROVIDERS[provider].model;
  const base_url = provider === "custom" ? $("ask-base-url").value.trim() : ASK_PROVIDERS[provider].base_url;
  if (!question) { askStatus("Type a question first.", "warn"); return; }
  if (!key && provider !== "custom") { askStatus("Paste an API key (Groq and Hugging Face give free ones).", "warn"); $("ask-settings").open = true; return; }
  if (provider === "custom" && !base_url) { askStatus("A custom endpoint needs its base URL (ending in /v1).", "warn"); return; }
  saveAskSettings();
  state.ask.running = true; state.ask.markdown = null;
  $("ask-run").disabled = true; $("ask-run").textContent = "Working…";
  $("ask-copy").hidden = true; $("ask-download").hidden = true;
  $("ask-result").hidden = true; $("ask-stations").hidden = true;
  $("ask-log").innerHTML = ""; $("ask-log").hidden = true;
  askStatus(state.workerReady ? "Preparing…" : "Loading Python in your browser (once, ~15 MB)…");
  try {
    await ensureCatalogInWorker();
    askStatus(`Asking ${model} via ${provider}…`);
    const res = await call("ask", { question, provider: provider === "custom" ? "custom" : provider, model, api_key: key || (provider === "custom" ? "none" : null), base_url, max_steps: 8 });
    askStatus("");
    state.ask.markdown = res.markdown;
    renderAsk(res);
  } catch (err) {
    askStatus(`The analyst could not finish: ${err.message}`, "error");
  } finally {
    state.ask.running = false;
    $("ask-run").disabled = false; $("ask-run").textContent = "Ask";
  }
}

function renderAsk(res) {
  const out = $("ask-result");
  out.innerHTML = mdToHtml(res.markdown);
  out.hidden = false;
  $("ask-copy").hidden = false; $("ask-download").hidden = false;
  // chips for every station the tools touched: click to open it on the map
  const chips = [];
  for (const d of res.data_used || []) {
    const m = String(d.label || "").match(/^(\S+) \/ (.+)$/);
    if (!m) continue;
    const key = `${m[1]}/${m[2]}`;
    const r = state.byKey.get(key);
    if (!r) continue;
    const b = document.createElement("button"); b.className = "chip"; b.type = "button";
    b.innerHTML = `<i style="background:${(SOURCE_STYLE[r.source] || {}).color || FALLBACK_COLOR}"></i>${escapeHtml(r.name || r.station_id)}`;
    b.title = "Open this station on the map";
    b.addEventListener("click", () => selectStation(key, { fly: true }));
    chips.push(b);
  }
  const box = $("ask-stations");
  box.innerHTML = "";
  if (chips.length) { const l = document.createElement("span"); l.className = "muted"; l.textContent = "Stations used: "; box.appendChild(l); for (const c of chips) box.appendChild(c); }
  box.hidden = chips.length === 0;
}

// A small Markdown renderer for the analyst's reports (headings, lists,
// tables, code, emphasis, links). Input is escaped first; only our own tags
// come out.
function mdToHtml(md) {
  const lines = String(md || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let i = 0, list = null, para = [];
  const flushPara = () => { if (para.length) { html.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  const inline = (t) => escapeHtml(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flushPara(); closeList();
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      html.push(`<pre><code>${escapeHtml(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); closeList(); html.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flushPara(); closeList(); html.push("<hr>"); i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(); closeList();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      html.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/), ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { closeList(); html.push(`<${kind}>`); list = kind; }
      html.push(`<li>${inline((ul || ol)[1])}</li>`); i++;
      continue;
    }
    if (!line.trim()) { flushPara(); closeList(); i++; continue; }
    para.push(line.trim()); i++;
  }
  flushPara(); closeList();
  return html.join("\n").replace(/<p>(Produced by aquascope[^<]*)<\/p>/, '<p class="foot">$1</p>');
}

// ── buttons ─────────────────────────────────────────────────────────────────

function initButtons() {
  $("btn-share-pt").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); $("btn-share-pt").textContent = "Copied!"; setTimeout(() => ($("btn-share-pt").textContent = "Copy link"), 1500); }
    catch { prompt("Copy this link", location.href); }
  });
  $("btn-share").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); $("btn-share").textContent = "Copied!"; setTimeout(() => ($("btn-share").textContent = "Copy link"), 1500); }
    catch { prompt("Copy this link", location.href); }
  });
  $("btn-csv").addEventListener("click", async () => {
    if (!state.result || !state.selected) return;
    const csv = await call("csv", {});
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.selected.source}_${state.selected.station_id}.csv`.replace(/[^\w.-]+/g, "_");
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  $("btn-gr4j").addEventListener("click", () => runGr4j());
  $("btn-ci").addEventListener("click", async () => {
    if (!state.result || !state.result.ffa) return;
    const btn = $("btn-ci");
    btn.disabled = true; btn.textContent = "Bootstrapping 1,000 GEV fits…";
    try {
      const ci = await call("flood_ci", {});
      state.result.ffa.fits.gev_bootstrap = ci;
      if (!state.result.methods.some((m) => m.name === ci.method.name)) state.result.methods.push(ci.method);
      renderFfaTable(state.result.ffa, state.result.unit);
      renderFfaPlot(state.result.ffa, state.result.unit, (SOURCE_STYLE[state.selected.source] || {}).color || FALLBACK_COLOR);
      renderMethods(state.result);
      btn.textContent = "Bootstrap CI added";
    } catch (err) {
      btn.disabled = false; btn.textContent = "Add bootstrap 90 % CI (GEV, slow)";
      setStatus(`Bootstrap failed: ${err.message}`, "error");
    }
  });
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ── boot ────────────────────────────────────────────────────────────────────

(async function boot() {
  trace("boot");
  initButtons();
  initSearch();
  initAsk();
  const mapReady = initMap();
  trace("map init called");
  const catalogReady = loadCatalog().then(() => true).catch((err) => {
    console.error(err);
    $("count").textContent = "catalog unavailable";
    setStatus(`Could not load the station catalog: ${err.message}`, "error");
    return false;
  });
  const [mapOk, catalogOk] = await Promise.all([mapReady, catalogReady]);
  trace(`ready: map=${mapOk} catalog=${catalogOk} stations=${state.stations.length}`);
  state.mapOk = Boolean(mapOk && map);
  if (!catalogOk) return;
  if (mapOk) {
    addStationLayers(toFeatureCollection(state.stations));
    map.fitBounds([[-128, 12], [128, 62]], { padding: 12, animate: false }); // US west coast to Taiwan
  }
  buildLegend();
  refreshMapData();
  ensureWorker(); // warm Python in the background so the first click is quicker
  const m = location.hash.match(/#s=(.+)$/);
  if (m) selectStation(decodeURIComponent(m[1]), { fly: true });
  const pm = location.hash.match(/#p=(-?[\d.]+),(-?[\d.]+)$/);
  if (pm) {
    const lat = Number(pm[1]), lon = Number(pm[2]);
    if (state.mapOk) map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 7) });
    selectPoint(lat, lon);
  }
})();
