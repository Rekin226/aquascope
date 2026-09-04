// Shared state, DOM helpers and formatting for the Explorer.
// Every other module imports from here; `actions` is the seam that lets a
// module (the map, the Ask drawer, a similar-basins list) open a station
// without importing the panel that owns it.

export const $ = (id) => document.getElementById(id);
export const EMPTY_FC = { type: "FeatureCollection", features: [] };

// Colour *and* shape, because colour alone does not survive colour-vision
// deficiency: Hub'Eau's red and the Environment Agency's green are ΔE 4.2 apart
// under deuteranopia, and they are the two largest European sources. Six hues
// cannot pass an all-pairs check however they are chosen, so identity carries a
// second channel instead of a new palette (#283). Shapes are assigned so that
// the pairs which collapse on hue are the least alike in outline.
export const SOURCE_STYLE = {
  usgs: { label: "USGS (US)", color: "#1565c0", shape: "circle" },
  uk_ea: { label: "Environment Agency (England)", color: "#2e7d32", shape: "triangle" },
  hubeau_hydrometrie: { label: "Hub'Eau (FR)", color: "#c62828", shape: "square" },
  pegelonline: { label: "PEGELONLINE (DE)", color: "#ef6c00", shape: "diamond" },
  ireland_opw: { label: "OPW (IE)", color: "#6a1b9a", shape: "pentagon" },
  taiwan_cwa: { label: "CWA (TW)", color: "#00838f", shape: "cross" },
};
export const FALLBACK_COLOR = "#546e7a";
export const FALLBACK_SHAPE = "circle";
export const VAR_LABEL = {
  discharge: "discharge", water_level: "water level", precipitation: "rainfall",
  groundwater_level: "groundwater", climate: "climate", water_quality: "water quality",
};

export const sourceStyle = (src) => SOURCE_STYLE[src] || { label: src, color: FALLBACK_COLOR, shape: FALLBACK_SHAPE };
export const stationKey = (r) => `${r.source}/${r.station_id}`;

// Debug hooks (harmless in production): window.__aq.state, window.__aq.log.
// globalThis rather than window so the pure helpers below can be imported and
// tested under node (tests/test_explorer/test_explorer_assets.py).
export const dbg = (globalThis.__aq = { log: [], state: null, map: null });
export const trace = (msg) => { dbg.log.push(`${new Date().toISOString().slice(11, 19)} ${msg}`); };

// How the map looks before anyone touches it. Named here rather than spread
// through the modules because the URL writer needs it too: a shared link only
// carries what differs from these, and it has to be able to say "globe off",
// which it cannot do by leaving the parameter out.
export const LAYER_DEFAULTS = {
  basemap: "light",
  terrain: false,
  // Relief under the basemap: the light style on its own is near-white land on
  // pale water, which reads as an empty page rather than as a map.
  hillshade: true,
  // Sparse worldwide coverage, so the world view is a globe (#281).
  globe: true,
  gaugeStyle: "source",
  heat: false,
};

export const state = {
  stations: [], byKey: new Map(), hidden: new Set(),
  selected: null, result: null, point: null,
  workerReady: false, booting: true, pending: new Map(), reqId: 0,
  mapOk: false, marker: null, basinsOn: false,
  // layers (#232)
  overlays: new Set(), opacity: {}, date: null,
  ...LAYER_DEFAULTS,
  ask: { running: false, catalogSent: false, markdown: null, run: 0 },
  // One drawer, two modes (Ask, Solve); the Solve chip is part of the URL.
  drawerOpen: false, drawerMode: "ask",
  solve: { playbook: null, running: false },
};
dbg.state = state;

// Filled in by the modules that own each behaviour (breaks import cycles).
export const actions = {
  selectStation: () => {},
  selectPoint: () => {},
  openAsk: () => {},
  openSolve: () => {},
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

// P-values. A test with p = 0.000192 is not "p = 0", which is mathematically
// impossible for most tests and misleads readers; report small values as "< 0.001".
export function fmtP(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  const n = Number(p);
  if (!Number.isFinite(n)) return "—";
  if (n < 0.001) return "< 0.001";
  return n.toFixed(3);
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
