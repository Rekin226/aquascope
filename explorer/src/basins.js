// Catchments, catchment attributes, similar gauged basins and the regionalised
// flow regime. Same arithmetic as before (and as aquascope.archive.similar /
// .regionalize); what changed is that every card now reports its own state, so
// "no catchment here" and "the lookup failed" stop looking identical.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, escapeHtml, fmt, haversineKm, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { fitBoundsTo, map, setCatchmentGeometry } from "./map.js?v=__BUILD__";
import { BASIN_METHOD, NLDI_METHOD, REGIME_METHOD, SIMILAR_METHOD, addMethodOnce } from "./methods.js?v=__BUILD__";
import { hideCard, setCard, setTab } from "./shell.js?v=__BUILD__";

const NLDI = "https://api.water.usgs.gov/nldi/linked-data";

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

// [table column, basin-card key, transform, weight]
const SIMILAR_FEATURES = [
  ["area_km2", "__area", "log10", 1.5], ["elevation_m", "elev", null, 1.0], ["slope_deg", "slope", "log1p", 1.0],
  ["precipitation_mm_yr", "pre", null, 1.5], ["aridity_index", "ari", "log1p", 1.5], ["temperature_c", "tmp", null, 1.0],
  ["snow_cover_pct", "snw", null, 1.0], ["forest_pct", "for", null, 0.7], ["cropland_pct", "crp", null, 0.7],
  ["urban_pct", "urb", "log1p", 0.7], ["clay_pct", "cly", null, 0.5], ["sand_pct", "snd", null, 0.5],
  ["population_density", "ppd", "log1p", 0.5], ["degree_of_regulation_pct", "dor", "log1p", 0.7],
];

// [column, label, unit, log?, lower bound, upper bound]
const REGIME_ROWS = [
  ["q_mean_mm", "Mean flow", "mm/d", true], ["q95_mm", "Low flow (Q95)", "mm/d", true], ["q05_mm", "High flow (Q05)", "mm/d", true],
  ["q_annual_max_mm", "Mean annual max", "mm/d", true], ["baseflow_index", "Baseflow index", "", false, 0, 1],
  ["seasonality_index", "Seasonality", "", false, 0, 1], ["flashiness_index", "Flashiness", "", false, 0],
];

const tf = (v, how) => (how === "log10" ? Math.log10(Math.max(v, 1e-3)) : how === "log1p" ? Math.log1p(Math.max(v, 0)) : v);

let catchmentReq = 0;
let basinReq = 0;
let basinsLayersAdded = false;
let topoLoaded = null;
let similarTable = null;
let regimeData = null;

export function basinsUrl(name) {
  return new URL(name, new URL(CONFIG.basinsBase, location.href)).href; // absolute: DuckDB and pmtiles want full URLs
}

const methodsList = (target) => (target === "st" ? "methods" : "pt-methods");
const root = (target) => $(target === "st" ? "panel-station" : "panel-point");

// ── geometry helpers ────────────────────────────────────────────────────────

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
  for (const poly of polys) poly.forEach((ring, k) => { a += (k === 0 ? 1 : -1) * ringAreaKm2(ring); });
  return a;
}

function bboxOf(geom) {
  let w = 180, s = 90, e = -180, n = -90;
  const walk = (c) => { if (typeof c[0] === "number") { w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]); } else c.forEach(walk); };
  walk(geom.coordinates);
  return [[w, s], [e, n]];
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

const inNldiExtent = (lat, lon) => lat > 17 && lat < 72 && lon > -170 && lon < -64;

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// ── reset ───────────────────────────────────────────────────────────────────

export function clearCatchment() {
  catchmentReq++;
  basinReq++;
  setCatchmentGeometry(null);
  for (const id of ["st-catchment", "pt-catchment", "st-basin", "pt-basin", "st-similar", "pt-similar", "st-regime", "pt-regime"]) hideCard($(id));
  highlightBasins([]);
}

// ── NLDI catchment (US, public domain) ──────────────────────────────────────

export async function requestCatchment({ station, point, target }) {
  const my = ++catchmentReq;
  const el = $(`${target}-catchment`);
  const r = root(target);
  let url = null;
  if (station) {
    if (station.source !== "usgs" || !/^USGS-\d+$/.test(station.station_id)) {
      hideCard(el);
      return;
    }
    url = `${NLDI}/nwissite/${station.station_id}/basin?simplified=true&splitCatchment=false`;
  } else if (point) {
    if (!inNldiExtent(point.lat, point.lon)) { hideCard(el); return; }
  } else return;

  setCard(el, "loading", { message: "Tracing the upstream catchment (USGS NLDI)…" });
  setTab(r, "catchment", { enabled: true });
  try {
    if (point) {
      const hl = await fetchJson(`${NLDI}/hydrolocation?coords=POINT(${point.lon} ${point.lat})`);
      const comid = (hl.features || []).map((f) => f.properties && f.properties.comid).find(Boolean);
      if (my !== catchmentReq) return;
      if (!comid) { setCard(el, "empty", { message: "No NHDPlus flowline near this point." }); return; }
      url = `${NLDI}/comid/${comid}/basin?simplified=true`;
    }
    const gj = await fetchJson(url);
    if (my !== catchmentReq) return; // the user moved on
    const feat = (gj.features || [])[0];
    if (!feat || !feat.geometry) { setCard(el, "empty", { message: "The NLDI has no basin for this gauge." }); return; }
    const km2 = geojsonAreaKm2(feat.geometry);
    setCatchmentGeometry(feat);
    if (km2 > 5) fitBoundsTo(bboxOf(feat.geometry));
    const areaTxt = km2 >= 100 ? Math.round(km2).toLocaleString() : km2.toFixed(1);
    el.querySelector(".card-body").innerHTML =
      `${station ? "Upstream catchment" : "Catchment upstream of the nearest stream reach"}: <strong>${areaTxt} km²</strong> ` +
      `<span class="muted">(USGS NLDI, drawn on the map)</span>`;
    setCard(el, "ready");
    addMethodOnce(methodsList(target), NLDI_METHOD);
  } catch (err) {
    if (my !== catchmentReq) return;
    setCard(el, "error", {
      message: `The USGS NLDI did not answer (${err.message}).`,
      retry: () => requestCatchment({ station, point, target }),
    });
  }
}

// ── BasinATLAS layers on the map ────────────────────────────────────────────

export function ensureBasinsLayers() {
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

export function setBasinsVisible(on) {
  state.basinsOn = on;
  ensureBasinsLayers();
  if (!basinsLayersAdded) return;
  for (const id of ["basins6-line", "basins12-line"]) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  const toggle = $("toggle-basins");
  if (toggle) toggle.checked = on;
}

function highlightBasins(ids) {
  if (!basinsLayersAdded || !state.mapOk) return;
  // HYBAS_ID is stored as a number in the tiles; keep the list bounded for the expression evaluator
  map.setFilter("basins12-up", ["in", ["get", "HYBAS_ID"], ["literal", ids.slice(0, 20000).map(Number)]]);
}

// ── BasinATLAS lookups ──────────────────────────────────────────────────────

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
  return {
    hybas_id: num("hybas_id") ?? num("HYBAS_ID"), next_down: num("next_down") ?? num("NEXT_DOWN"),
    main_bas: num("main_bas") ?? num("MAIN_BAS"), sub_area: num("sub_area") ?? num("SUB_AREA"),
    up_area: num("up_area") ?? num("UP_AREA"), pfaf_id: num("pfaf_id") ?? num("PFAF_ID"),
    geometry: best.geometry, approximate: Boolean(best.__near),
  };
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

export async function requestBasin(lat, lon, target) {
  const my = ++basinReq;
  const el = $(`${target}-basin`);
  const r = root(target);
  if (!el) return;
  setCard(el, "loading", { message: "Finding the sub-basin (BasinATLAS)…" });
  setTab(r, "catchment", { enabled: true });
  try {
    const sb = await subBasinAt(lat, lon);
    if (my !== basinReq) return;
    if (!sb || sb.hybas_id === null) {
      setCard(el, "empty", { message: "No BasinATLAS sub-basin here (open sea, or outside the level-12 layer)." });
      return;
    }
    const body = el.querySelector(".card-body");
    body.innerHTML = `Catchment (BasinATLAS): <strong>${fmt(sb.up_area, 0)} km²</strong> upstream · sub-basin ${sb.hybas_id}` +
      `${sb.approximate ? ' <span class="muted">(nearest)</span>' : ""} <span class="muted">· reading attributes…</span>`;
    setCard(el, "ready");
    setBasinsVisible(true);
    const [ids, attrs] = await Promise.all([
      upstreamIds(sb.hybas_id).catch((e) => { console.info("upstream walk unavailable:", e && e.message); return [sb.hybas_id]; }),
      basinAttributes(sb.hybas_id).catch((e) => { console.info("basin attributes unavailable:", e && e.message); return null; }),
    ]);
    if (my !== basinReq) return;
    highlightBasins(ids);
    const capped = ids.length > 20000;
    const kv = attrs ? Object.values(attrs).map((a) =>
      `<div><span>${escapeHtml(a.label)}</span><b>${fmt(a.value, a.unit === "%" ? 0 : 1)}${a.unit ? " " + a.unit : ""}</b></div>`).join("") : "";
    body.innerHTML = `Catchment (BasinATLAS level 12): <strong>${fmt(sb.up_area, 0)} km²</strong> upstream, ` +
      `${capped ? "over 20,000" : ids.length.toLocaleString()} sub-basins${sb.approximate ? ' <span class="muted">(nearest sub-basin)</span>' : ""}` +
      (kv ? `<div class="kv">${kv}</div>` : `<div class="muted">attributes unavailable for this sub-basin</div>`) +
      `<div class="basin-foot muted">HydroATLAS v1.0, CC BY 4.0 (Linke et al. 2019). Upstream sub-basins highlighted on the map at zoom 4+.</div>`;
    addMethodOnce(methodsList(target), BASIN_METHOD);
    if (attrs) requestSimilar({ lat, lon, upArea: sb.up_area, attrs, target, my });
    else setCard($(`${target}-similar`), "empty", { message: "Similar basins need the catchment attributes, which are missing here." });
  } catch (err) {
    if (my !== basinReq) return;
    setCard(el, "error", {
      message: `Could not read the BasinATLAS layer (${err.message}).`,
      retry: () => requestBasin(lat, lon, target),
    });
  }
}

// ── similar gauged basins ───────────────────────────────────────────────────

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

export async function stationArea(key) {
  try {
    const { rows } = await ensureSimilarTable();
    const row = rows.find((r) => `${r.source}/${r.station_id}` === key);
    if (row && row.area_km2) return { area: Number(row.area_km2), source: row.area_source === "agency" ? "agency" : "BasinATLAS upstream area" };
  } catch (err) {
    console.info("catchment table unavailable:", err && err.message);
  }
  return null;
}

async function requestSimilar({ lat, lon, upArea, attrs, target, my }) {
  const el = $(`${target}-similar`);
  const r = root(target);
  if (!el) return;
  setCard(el, "loading", { message: "Comparing catchments…" });
  try {
    const { rows, stats } = await ensureSimilarTable();
    if (my !== basinReq) return;
    const selfKey = target === "st" && state.selected ? stationKey(state.selected) : null;
    // A station's own table row is the better target: it carries the agency's area and the attribute scope
    // (a small creek inside a big river's sub-basin must not be described by the big river).
    const selfRow = selfKey ? rows.find((row) => `${row.source}/${row.station_id}` === selfKey) : null;
    const tvals = {};
    for (const [col, key] of SIMILAR_FEATURES) {
      let v;
      if (selfRow && selfRow[col] !== null && selfRow[col] !== undefined) v = selfRow[col];
      else v = key === "__area" ? upArea : (attrs[key] && attrs[key].value);
      if (v !== undefined && v !== null && Number.isFinite(Number(v))) tvals[col] = Number(v);
    }
    const used = SIMILAR_FEATURES.filter(([col]) => tvals[col] !== undefined);
    if (!used.length) { setCard(el, "empty", { message: "No comparable attributes for this catchment." }); return; }
    const wsum = used.reduce((a, [, , , w]) => a + w * w, 0);
    const scored = [];
    for (const row of rows) {
      const key = `${row.source}/${row.station_id}`;
      if (key === selfKey) continue;
      const st = state.byKey.get(key);
      if (!st || !(st.variables || []).includes("discharge")) continue;
      let acc = 0;
      for (const [col, , how, w] of used) {
        const { mu, sd } = stats[col];
        const zt = (tf(tvals[col], how) - mu) / sd;
        const v = row[col];
        const z = v === null || v === undefined ? NaN : (tf(Number(v), how) - mu) / sd;
        const d = Number.isFinite(z) ? Math.abs(z - zt) : 3.0;
        acc += (d * w) ** 2;
      }
      const dAttr = Math.sqrt(acc / wsum);
      const dKm = haversineKm(lat, lon, st.lat, st.lon);
      scored.push({ st, dAttr, dKm, score: Math.sqrt(dAttr ** 2 + (dKm / 500) ** 2), area: Number(row.area_km2 ?? row.up_area) });
    }
    scored.sort((a, b) => a.score - b.score);
    const top = scored.slice(0, 8);
    if (!top.length) { setCard(el, "empty", { message: "No gauged discharge basins to compare with." }); return; }
    const body = el.querySelector(".card-body");
    body.innerHTML = `<ul class="nearest similar"></ul>` +
      `<div class="basin-foot muted">Donor candidates for an ungauged site: ${scored.length.toLocaleString()} gauged catchments compared on ` +
      `${used.length} standardised BasinATLAS attributes and distance. Click one to open it.</div>`;
    const ul = body.querySelector("ul");
    for (const s of top) {
      const li = document.createElement("li");
      li.tabIndex = 0;
      li.setAttribute("role", "button");
      li.dataset.key = stationKey(s.st);
      const sst = sourceStyle(s.st.source);
      li.innerHTML = `<i style="background:${sst.color}"></i>${escapeHtml(s.st.name || s.st.station_id)} ` +
        `<span class="muted">${escapeHtml(sst.label)} · ${fmt(s.area, 0)} km²</span>` +
        `<span class="dist">${s.dKm < 10 ? s.dKm.toFixed(1) : Math.round(s.dKm).toLocaleString()} km · Δ${s.dAttr.toFixed(2)}</span>`;
      const open = () => actions.selectStation(li.dataset.key, { fly: true });
      li.addEventListener("click", open);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      ul.appendChild(li);
    }
    setCard(el, "ready");
    setTab(r, "similar", { enabled: true, count: top.length });
    addMethodOnce(methodsList(target), SIMILAR_METHOD);
    renderRegime(target, scored, my).catch((err) => {
      setCard($(`${target}-regime`), "error", { message: `Flow regime unavailable (${err.message}).` });
    });
  } catch (err) {
    if (my !== basinReq) return;
    setCard(el, "error", { message: `Could not compare catchments (${err.message}).` });
  }
}

// ── regionalised flow regime ────────────────────────────────────────────────

async function ensureRegimeData() {
  if (regimeData) return regimeData;
  const { conn } = await duck();
  const res = await conn.query(`SELECT * FROM read_parquet('${basinsUrl("station_signatures.parquet")}')`);
  const sig = new Map();
  for (const r of res.toArray().map((x) => x.toJSON())) sig.set(`${r.source}/${r.station_id}`, r);
  let skill = null;
  try {
    const rs = await fetch(basinsUrl("regionalization_skill.json"));
    if (rs.ok) skill = await rs.json();
  } catch { /* optional */ }
  regimeData = { sig, skill };
  return regimeData;
}

async function renderRegime(target, scored, my) {
  const el = $(`${target}-regime`);
  if (!el) return;
  setCard(el, "loading", { message: "Transferring flow signatures from the donors…" });
  const { sig, skill } = await ensureRegimeData();
  if (my !== basinReq) return;
  const donors = scored.filter((s) => sig.has(stationKey(s.st))).sort((a, b) => a.dAttr - b.dAttr).slice(0, 10);
  if (donors.length < 3) {
    setCard(el, "empty", { message: `Only ${donors.length} donors with archived signatures near here; three are needed.` });
    return;
  }
  const w = donors.map((d) => 1 / (d.dAttr + 0.05));
  const per = ((skill || {}).methods || {}).similarity || {};
  const rows = [];
  for (const [col, label, unit, isLog, lo, hi] of REGIME_ROWS) {
    const vals = [], ws = [];
    donors.forEach((d, i) => {
      const v = sig.get(stationKey(d.st))[col];
      if (v !== null && v !== undefined && Number.isFinite(Number(v))) { vals.push(Number(v)); ws.push(w[i]); }
    });
    if (!vals.length) continue;
    const wsum = ws.reduce((a, b) => a + b, 0);
    const y = vals.map((v) => (isLog ? Math.log(Math.max(v, 1e-3)) : v));
    const mean = y.reduce((a, v, i) => a + (ws[i] / wsum) * v, 0);
    const sd = Math.sqrt(y.reduce((a, v, i) => a + (ws[i] / wsum) * (v - mean) ** 2, 0));
    const back = (v) => { let out = isLog ? Math.exp(v) : v; if (lo !== undefined) out = Math.max(lo, out); if (hi !== undefined) out = Math.min(hi, out); return out; };
    rows.push({ label, unit, value: back(mean), low: back(mean - sd), high: back(mean + sd), n: vals.length, sk: per[col] });
  }
  if (!rows.length) { setCard(el, "empty", { message: "The donors have no usable signatures." }); return; }
  const digits = (v) => (v >= 100 ? 0 : v >= 10 ? 1 : v >= 1 ? 2 : 3);
  el.querySelector(".card-body").innerHTML =
    `<div class="card-title"><h4>Estimated flow regime <span class="muted">${donors.length} most similar donors, weighted</span></h4></div>` +
    `<table class="ffa regime" id="${target}-regime-table"><thead><tr><th>Signature</th><th>Estimate</th><th>Band</th><th>LOO</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${escapeHtml(r.label)}</td><td>${fmt(r.value, digits(r.value))}${r.unit ? " " + r.unit : ""}</td>` +
      `<td class="ci">${fmt(r.low, digits(r.low))} to ${fmt(r.high, digits(r.high))}</td>` +
      `<td class="ci">${r.sk && r.sk.nse !== null && r.sk.nse !== undefined
        ? "NSE " + Number(r.sk.nse).toFixed(2) + (r.sk.median_ape !== null && r.sk.median_ape !== undefined ? ", ±" + Math.round(Number(r.sk.median_ape) * 100) + " %" : "")
        : "–"}</td></tr>`).join("") +
    `</tbody></table><div class="basin-foot muted">Prediction in ungauged basins: what the ${donors.length} closest donors in attribute space would suggest here ` +
    `(geometric mean for mm/d; band = one weighted standard deviation of the donors; LOO = leave-one-out skill over ` +
    `${(skill && skill.n_stations) ? skill.n_stations.toLocaleString() : "all"} donors, NSE and median error). Not a measurement.</div>`;
  setCard(el, "ready");
  addMethodOnce(methodsList(target), REGIME_METHOD);
}
