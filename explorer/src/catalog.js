// The station catalog and the one DuckDB-WASM instance the page shares:
// the catalog at boot, the basin topology and attributes on demand, all read
// in place over HTTPS range requests (no server, no download).

import { CONFIG } from "../config.js?v=__BUILD__";
import { sourceStyle, state, stationKey, trace } from "./core.js?v=__BUILD__";
import { RECENT_BREAKS, RECORD_BREAKS, breakColor, recordYears, yearsSinceLast } from "./layers.js?v=__BUILD__";

let duckPromise = null;

export function duck() {
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

export async function loadCatalog() {
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
  state.byKey = new Map(rows.map((r) => [stationKey(r), r]));
  for (const r of rows) r._fold = null; // filled lazily by the search index
  return rows;
}

// The map gets one colour per style mode, computed here so the paint
// expression stays a plain ["get", ...] and the legend and the dots cannot
// drift apart.
export function toFeatureCollection(rows) {
  const now = new Date();
  return {
    type: "FeatureCollection",
    features: rows.filter((r) => !state.hidden.has(r.source)).map((r) => {
      const years = recordYears(r, now);
      const stale = yearsSinceLast(r, now);
      return {
        type: "Feature",
        geometry: { type: "Point", coordinates: [r.lon, r.lat] },
        properties: {
          key: stationKey(r), source: r.source, name: r.name ?? "",
          color: sourceStyle(r.source).color,
          colorRecord: breakColor(RECORD_BREAKS, years),
          colorRecent: breakColor(RECENT_BREAKS, stale),
          years: years === null ? -1 : Math.round(years * 10) / 10,
        },
      };
    }),
  };
}

export function sourceCounts() {
  const counts = {};
  for (const r of state.stations) counts[r.source] = (counts[r.source] || 0) + 1;
  return counts;
}
