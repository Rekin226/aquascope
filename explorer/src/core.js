// Shared state, DOM helpers and formatting for the Explorer.
// Every other module imports from here; `actions` is the seam that lets a
// module (the map, the Ask drawer, a similar-basins list) open a station
// without importing the panel that owns it.

export const $ = (id) => document.getElementById(id);
export const EMPTY_FC = { type: "FeatureCollection", features: [] };

export const SOURCE_STYLE = {
  usgs: { label: "USGS (US)", color: "#1565c0" },
  uk_ea: { label: "Environment Agency (UK)", color: "#2e7d32" },
  hubeau_hydrometrie: { label: "Hub'Eau (FR)", color: "#c62828" },
  pegelonline: { label: "PEGELONLINE (DE)", color: "#ef6c00" },
  ireland_opw: { label: "OPW (IE)", color: "#6a1b9a" },
  taiwan_cwa: { label: "CWA (TW)", color: "#00838f" },
};
export const FALLBACK_COLOR = "#546e7a";
export const VAR_LABEL = {
  discharge: "discharge", water_level: "water level", precipitation: "rainfall",
  groundwater_level: "groundwater", climate: "climate", water_quality: "water quality",
};

export const sourceStyle = (src) => SOURCE_STYLE[src] || { label: src, color: FALLBACK_COLOR };
export const stationKey = (r) => `${r.source}/${r.station_id}`;

// Debug hooks (harmless in production): window.__aq.state, window.__aq.log.
// globalThis rather than window so the pure helpers below can be imported and
// tested under node (tests/test_explorer/test_explorer_assets.py).
export const dbg = (globalThis.__aq = { log: [], state: null, map: null });
export const trace = (msg) => { dbg.log.push(`${new Date().toISOString().slice(11, 19)} ${msg}`); };

export const state = {
  stations: [], byKey: new Map(), hidden: new Set(),
  selected: null, result: null, point: null,
  workerReady: false, booting: true, pending: new Map(), reqId: 0,
  mapOk: false, marker: null, basinsOn: false,
  // layers (#232)
  basemap: "light", overlays: new Set(), opacity: {}, date: null,
  terrain: false, hillshade: false, globe: false,
  gaugeStyle: "source", heat: false,
  ask: { running: false, catalogSent: false, markdown: null, run: 0 },
};
dbg.state = state;

// Filled in by the modules that own each behaviour (breaks import cycles).
export const actions = {
  selectStation: () => {},
  selectPoint: () => {},
  openAsk: () => {},
  applyUrl: () => {},
  refreshMapData: () => {},
};

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Numbers. With no `digits` the precision adapts to the magnitude, which is
// what most call sites want (0.414 m³/s keeps three decimals, 14,603 obs none).
// With `digits` given it is honoured at every magnitude: it used to be ignored
// below 10, so fmt(area, 0) printed "303.412 km²" instead of "303 km²".
export function fmt(x, digits) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  const ax = Math.abs(n);
  const d = digits === undefined ? (ax >= 1000 ? 0 : ax >= 10 ? 1 : 3) : Math.max(0, digits);
  return n.toLocaleString(undefined, { maximumFractionDigits: d });
}

// "a increasing trend" was wrong; pick the article from the first sound.
export const article = (word) => (/^[aeiou]/i.test(String(word)) ? "an" : "a");

export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371, d2r = Math.PI / 180;
  const dLat = (lat2 - lat1) * d2r, dLon = (lon2 - lon1) * d2r;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * d2r) * Math.cos(lat2 * d2r) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// Accent- and case-insensitive key for search ("Rhône" matches "rhone").
export function foldText(s) {
  return String(s || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

export function downloadBlob(name, text, type = "text/plain") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name.replace(/[^\w.-]+/g, "_");
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

export async function copyText(text, btn, done = "Copied!") {
  const label = btn ? btn.textContent : null;
  try {
    await navigator.clipboard.writeText(text);
    if (btn) { btn.textContent = done; setTimeout(() => { btn.textContent = label; }, 1500); }
  } catch {
    prompt("Copy this", text);
  }
}

// Rows of [label, value] to CSV, for the "download this table" buttons.
export function toCsv(header, rows) {
  const cell = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header, ...rows].map((r) => r.map(cell).join(",")).join("\n") + "\n";
}
