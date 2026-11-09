// Paint the globe (#657), the pure half: the questions, which gauges a view reads, the colours and the
// legend, the card's words and the figure's text. No DOM and no map, so node imports it
// (explorer/tests/paint-core.test.mjs). Every number painted comes from aquascope.paint in the worker; the
// question list mirrors aquascope.paint.questions() (a test holds both to explorer/tests/fixtures/paint-questions.json).

export const QUESTIONS = [
  {
    id: "flood_trend", label: "Trend in annual floods", short: "Flood trend", kind: "trend", needs: "daily",
    unit: "% per decade", breaks: [-10, -5, -2, 2, 5, 10], diverging: true,
    describe: "Sen's slope of the annual maximum daily flow, in per cent of its mean per decade, over complete " +
      "hydrological years; Mann-Kendall p-value.",
    year_start: { north: 10, south: 4 },
  },
  {
    id: "low_flow_trend", label: "Trend in low flows", short: "Low-flow trend", kind: "trend", needs: "daily",
    unit: "% per decade", breaks: [-10, -5, -2, 2, 5, 10], diverging: true,
    describe: "Sen's slope of the annual minimum 7-day mean flow, in per cent of its mean per decade, over " +
      "complete climate years; Mann-Kendall p-value.",
    year_start: { north: 4, south: 10 },
  },
  {
    id: "baseflow_index", label: "Baseflow index", short: "Baseflow index", kind: "value", needs: "signatures",
    unit: "", breaks: [0.2, 0.35, 0.5, 0.65, 0.8], diverging: false,
    describe: "The share of the flow that is baseflow, from the Lyne-Hollick filter (alpha 0.925, three passes) " +
      "the Archive runs on every mirrored record.",
    year_start: { north: 10, south: 4 },
  },
];

export const questionById = (id) => QUESTIONS.find((q) => q.id === id) || null;

// The defaults of aquascope.paint (MAX_GAUGES, MIN_YEARS, ALPHA_GLOBAL). A phone reads fewer gauges a view.
export const MAX_GAUGES = 300;
export const MAX_GAUGES_PHONE = 120;
export const MIN_YEARS = 20;
export const ALPHA_GLOBAL = 0.05;
export const YEAR_CHOICES = [10, 20, 30, 50];
export const PERIODS = [
  { id: "all", label: "Whole record", start_year: null },
  { id: "1960", label: "Since 1960", start_year: 1960 },
  { id: "1980", label: "Since 1980", start_year: 1980 },
];

// Red where the flow falls, blue where it rises (ColorBrewer RdBu, 7 classes). The middle class, "no change", is a
// warm light grey rather than RdBu's white, so a painted gauge never reads as an empty ring.
export const DIVERGING = ["#b2182b", "#ef8a62", "#fddbc7", "#ddd8cc", "#d1e5f0", "#67a9cf", "#2166ac"];
// Baseflow index, low to high (ColorBrewer YlGnBu, 6 classes).
export const SEQUENTIAL = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"];

// The gauges that are not painted, said on the map rather than left out.
// Small and quiet, so the answers stand out, but on the map: the map says where it has no answer.
export const STATUS_STYLE = {
  too_short: { label: "Record too short", color: "#8a96a3", ring: "#8a96a3", hollow: false },
  no_archive: { label: "No record in the Archive", color: "#7b8794", ring: "#7b8794", hollow: true },
  over_cap: { label: "Not read yet (over the cap)", color: "#b3bdc7", ring: "#b3bdc7", hollow: false },
  failed: { label: "Could not be read", color: "#c2410c", ring: "#c2410c", hollow: true },
  queued: { label: "Reading", color: "#9fb3c8", ring: "#ffffff", hollow: false },
};

export const palette = (q) => (q && q.diverging ? DIVERGING : SEQUENTIAL);

const isNum = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

/** The class (0 to breaks.length) a value falls in, as aquascope.paint.class_of (numpy searchsorted, right). */
export function classOf(q, value) {
  if (!isNum(value)) return null;
  const v = Number(value);
  let i = 0;
  while (i < q.breaks.length && v >= q.breaks[i]) i++;
  return i;
}

const num = (x) => {
  const v = Number(x);
  const s = Number.isInteger(v) ? String(Math.abs(v)) : String(Math.abs(v));
  return v < 0 ? `−${s}` : s;
};

/** The words for class i: "below −10", "−10 to −5", ..., "10 and above" (with the unit on the ends). */
export function classLabel(q, i) {
  const b = q.breaks;
  const unit = q.unit ? ` ${q.unit}` : "";
  if (i <= 0) return `below ${num(b[0])}${unit}`;
  if (i >= b.length) return `${num(b[b.length - 1])} and above${unit}`;
  return `${num(b[i - 1])} to ${num(b[i])}`;
}

// ── which gauges (the rules of aquascope.paint.plan, on the same fixture) ───

export function inBox(lat, lon, bbox) {
  const [w, s, e, n] = bbox;
  if (!(lat >= s && lat <= n)) return false;
  return w <= e ? lon >= w && lon <= e : lon >= w || lon <= e;
}

const keyOf = (r) => `${r.source}/${r.station_id}`;
const yearsOf = (sig) => {
  if (!sig) return -1;
  const v = Number(sig.record_years);
  return sig.record_years === null || sig.record_years === undefined || !Number.isFinite(v) ? -1 : v;
};

/**
 * Which gauges a painting reads, and why the others are not painted (aquascope.paint.plan).
 * `stations`: catalog rows ({source, station_id, site_id, name, lat, lon, variables}); `sig`: a Map of
 * "source/station_id" -> {record_years}, the Archive's signatures.parquet. Returns { read, rows, counts }.
 */
export function planGauges(stations, sig, { bbox = null, minYears = MIN_YEARS, maxGauges = MAX_GAUGES } = {}) {
  const bySite = new Map();
  for (const s of stations || []) {
    const lat = s.latitude ?? s.lat, lon = s.longitude ?? s.lon;
    if (lat === null || lat === undefined || lon === null || lon === undefined) continue;
    const la = Number(lat), lo = Number(lon);
    if (bbox && !inBox(la, lo, bbox)) continue;
    const vars = s.variables;
    if (vars && vars.length && !vars.includes("discharge")) continue;
    const key = keyOf(s);
    const has = sig.has(key);
    const row = {
      key, source: String(s.source), station_id: String(s.station_id), name: s.name || "", latitude: la, longitude: lo,
      record_years: has ? roundSig(yearsOf(sig.get(key)), 4) : null,
    };
    const site = `${row.source}\u0000${s.site_id || s.station_id}`;
    const have = bySite.get(site);
    if (!have) { bySite.set(site, row); continue; }
    const yNew = yearsOf(sig.get(key)), yOld = yearsOf(sig.get(have.key));
    if (yNew > yOld || (yNew === yOld && row.station_id < have.station_id)) bySite.set(site, row);
  }
  const rows = [...bySite.values()].sort((a, b) => {
    const d = yearsOf(sig.get(b.key)) - yearsOf(sig.get(a.key));
    return d !== 0 ? d : (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
  });
  const read = [];
  for (const r of rows) {
    const y = yearsOf(sig.get(r.key));
    if (!sig.has(r.key)) Object.assign(r, { status: "no_archive", note: "not in the AquaScope Archive" });
    else if (y < minYears) Object.assign(r, { status: "too_short", note: `${y.toFixed(1)} years in the Archive; the question needs ${minYears}` });
    else if (read.length >= maxGauges) Object.assign(r, { status: "over_cap", note: `past the cap of ${maxGauges} gauges a view; zoom in` });
    else { Object.assign(r, { status: "queued", note: "" }); read.push(r); }
  }
  const counts = { queued: 0, too_short: 0, no_archive: 0, over_cap: 0 };
  for (const r of rows) counts[r.status] += 1;
  return { read, rows, counts };
}

// Python's _num: `digits` significant figures.
function roundSig(x, digits) {
  if (!Number.isFinite(x)) return null;
  if (x === 0) return 0;
  const mag = Math.max(0, digits - 1 - Math.floor(Math.log10(Math.abs(x))));
  const f = 10 ** mag;
  return Math.round(x * f) / f;
}

/** The Archive's daily file for a gauge, next to stations.parquet (aquascope.archive.observations.archive_series_url). */
export function archiveFileUrl(stationsUrl, source, stationId) {
  const base = String(stationsUrl).replace(/stations\.parquet(\?.*)?$/, "");
  const safe = String(stationId).replace(/\//g, "_");
  return `${base}obs/discharge/${encodeURIComponent(source)}/${encodeURIComponent(safe)}.csv.gz`;
}

export function batchesOf(list, n) {
  const out = [];
  for (let i = 0; i < list.length; i += n) out.push(list.slice(i, i + n));
  return out;
}

/** The view as [west, south, east, north], clamped, two decimals (a globe's bounds can pass ±180). */
export function viewBbox(b) {
  const w = Math.max(-180, b.west), e = Math.min(180, b.east);
  const s = Math.max(-85, b.south), n = Math.min(85, b.north);
  const r = (v) => Math.round(v * 100) / 100;
  return (b.east - b.west) >= 360 ? [-180, r(s), 180, r(n)] : [r(w), r(s), r(e), r(n)];
}

// ── the map's paint ─────────────────────────────────────────────────────────

/** MapLibre paint properties for the painted gauges: the class colour, size and ring by significance and status. */
export function paintProps(q) {
  const colors = palette(q);
  const match = ["match", ["coalesce", ["get", "cls"], -1]];
  colors.forEach((c, i) => match.push(i, c));
  match.push("#c9d2da");
  const statusColor = ["match", ["get", "status"]];
  for (const [k, v] of Object.entries(STATUS_STYLE)) statusColor.push(k, v.color);
  statusColor.push("#c9d2da");
  const statusRing = ["match", ["get", "status"]];
  for (const [k, v] of Object.entries(STATUS_STYLE)) statusRing.push(k, v.ring);
  statusRing.push("#ffffff");
  const painted = ["==", ["get", "status"], "painted"];
  const sig = ["==", ["get", "fdr_significant"], true];
  const queued = ["==", ["get", "status"], "queued"];
  const hollow = ["in", ["get", "status"], ["literal", Object.entries(STATUS_STYLE).filter(([, v]) => v.hollow).map(([k]) => k)]];
  const tested = q.kind === "trend";
  const size = (z, s, n, v, other) => [z, ["case", painted, tested ? ["case", sig, s, n] : v, queued, other * 1.4, other]];
  return {
    "circle-radius": ["interpolate", ["linear"], ["zoom"],
      ...size(2, 4.2, 3.2, 3.6, 1.1), ...size(6, 7, 5.2, 5.8, 1.9), ...size(10, 10, 7.5, 8.5, 3.2)],
    "circle-color": ["case", painted, match, statusColor],
    "circle-opacity": ["case", hollow, 0, painted, tested ? ["case", sig, 1, 0.85] : 0.95, queued, 0.9, 0.55],
    "circle-stroke-color": ["case", painted, tested ? ["case", sig, "#111827", "#5b6673"] : "#33414f", statusRing],
    "circle-stroke-width": ["case", painted, tested ? ["case", sig, 1.8, 0.7] : 0.8, hollow, 0.8, 0],
    "circle-stroke-opacity": ["case", painted, tested ? ["case", sig, 1, 0.75] : 0.8, 0.6],
  };
}

/** Painted gauges draw over the unread ones, the significant on top. */
export function paintLayout(q) {
  const painted = ["==", ["get", "status"], "painted"];
  const sig = ["==", ["get", "fdr_significant"], true];
  return { "circle-sort-key": ["case", painted, q.kind === "trend" ? ["case", sig, 3, 2] : 2, 1] };
}

/** The legend: the classes with their counts, then each unpainted status with its count. */
export function legendRows(q, rows) {
  const colors = palette(q);
  const painted = (rows || []).filter((r) => r.status === "painted" && isNum(r.value));
  const classes = colors.map((color, i) => ({
    color, label: classLabel(q, i), n: painted.filter((r) => classOf(q, r.value) === i).length,
  }));
  const status = Object.entries(STATUS_STYLE).filter(([k]) => k !== "queued").map(([k, v]) => ({
    status: k, color: v.color, hollow: v.hollow, label: v.label, n: (rows || []).filter((r) => r.status === k).length,
  })).filter((s) => s.n > 0);
  const reading = (rows || []).filter((r) => r.status === "queued").length;
  return { classes, status, reading };
}

/** "Reading 48 of 212 gauges · 3 could not be read". */
export function progressWords({ done = 0, total = 0, failed = 0, phase = "read" } = {}) {
  if (phase === "plan") return "Finding the gauges in view…";
  if (phase === "boot") return "Loading Python in your browser (about 30 MB, once)…";
  if (phase === "field") return "Testing the field…";
  const base = total ? `Painted ${done} of ${total} gauge${total === 1 ? "" : "s"}` : "No gauge to read";
  return failed ? `${base} · ${failed} could not be read` : base;
}

const fmtNum = (v, d = 1) => (isNum(v) ? Number(v).toLocaleString("en-GB", { maximumFractionDigits: d }) : "—");
const signed = (v, d = 1) => (isNum(v) ? `${Number(v) > 0 ? "+" : Number(v) < 0 ? "−" : ""}${fmtNum(Math.abs(Number(v)), d)}` : "—");
export const fmtPValue = (p) => (!isNum(p) ? "—" : Number(p) < 0.001 ? "< 0.001" : Number(p).toFixed(3));

/** What a painted gauge's card says: { what, title, sub, status: {text, color}, figure, note }. */
export function cardFor(q, row) {
  const title = row.name || row.station_id;
  const years = row.first_year && row.last_year ? `${row.first_year} to ${row.last_year}` : "";
  const sub = [row.source, years].filter(Boolean).join(" · ");
  const what = `Painted: ${q.short}`;
  if (row.status !== "painted") {
    const st = STATUS_STYLE[row.status] || STATUS_STYLE.failed;
    return { what, title, sub, status: { text: st.label, color: st.color }, figure: null, note: row.note || "" };
  }
  const color = palette(q)[classOf(q, row.value) ?? 0];
  if (q.kind !== "trend") {
    return { what, title, sub, status: { text: classLabel(q, classOf(q, row.value)), color },
      figure: { value: fmtNum(row.value, 2), unit: "", label: q.label }, note: row.note || "" };
  }
  const sig = row.fdr_significant === true;
  const dir = Number(row.tau) > 0 ? "rising" : Number(row.tau) < 0 ? "falling" : "flat";
  const text = sig ? `Significant after the false discovery rate: ${dir}`
    : row.fdr_significant === false ? `Not significant after the false discovery rate (${dir})` : `Field test pending (${dir})`;
  const missing = Number(row.missing_years) > 0 ? `, ${row.missing_years} missing` : "";
  const note = `Mann-Kendall p = ${fmtPValue(row.p)} over ${row.n_years} complete years${missing}; ` +
    `Sen's slope ${signed(row.slope_per_year, 3)} m³/s a year on a mean of ${fmtNum(row.mean, 2)} m³/s.`;
  return { what, title, sub, status: { text, color: sig ? color : "#6b7785" },
    figure: { value: signed(row.value, 1), unit: q.unit, label: q.id === "flood_trend" ? "Annual floods" : "Annual 7-day low flow" },
    note };
}

/** The legend row's few words: "212 painted · field significant" (or not). */
export function summaryWords(q, res) {
  if (!res) return "";
  const c = res.counts || {};
  const parts = [`${c.painted || 0} painted`];
  if (q.kind === "trend" && res.field && res.field.n_tested) {
    parts.push(res.field.field_significant ? "field significant" : "field not significant");
  }
  return parts.join(" · ");
}

export function fileStem(q, bbox) {
  const b = bbox ? `_${bbox.map((v) => v.toFixed(1)).join("_")}` : "";
  return `aquascope-paint-${q.id}${b}`.replace(/[^\w.-]+/g, "_");
}

/** Greedy word wrap for the figure's text, by a measure function (canvas measureText in the page). */
export function wrapText(text, maxWidth, measure) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (line && measure(next) > maxWidth) { lines.push(line); line = w; } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

/** The figure's title and subtitle lines. */
export function figureTitle(q, res) {
  const p = (res && res.params) || {};
  const years = p.start_year ? `, years from ${p.start_year}` : "";
  const sub = q.kind === "trend"
    ? `Sen's slope, ${q.unit}; filled with a dark ring: significant after the false discovery rate ` +
      `(alpha_FDR ${(res && res.field && res.field.alpha_fdr) || 2 * ALPHA_GLOBAL}); at least ${p.min_years || MIN_YEARS} complete years${years}`
    : `at least ${p.min_years || MIN_YEARS} years of record`;
  return { title: `${q.label}: ${(res && res.counts && res.counts.painted) || 0} gauges`, sub };
}
