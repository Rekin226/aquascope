// Replay an event (#659): the pieces of the player that do no drawing, so node can test them. The story itself
// (the stops, their cameras, dates, layers and notes) is written by the package (aquascope.replay.story); this file
// checks it, times it, fills the two notes the page computes itself (Flood it on the GPU and what is under it, from
// the package's templates), draws the small charts beside a note and reads and writes the share link.

export const STORY_SCHEMA = "aquascope.story";
export const STORY_VERSION = 1;

/** The stops of a story, or an error when it is not a story this player knows. */
export function checkStory(story) {
  if (!story || story.schema !== STORY_SCHEMA) throw new Error("not an AquaScope story");
  if (story.version !== STORY_VERSION) throw new Error(`a story of version ${story.version}; this player reads ${STORY_VERSION}`);
  if (!Array.isArray(story.stops) || !story.stops.length) throw new Error("the story has no stops");
  for (const s of story.stops) {
    const c = s.camera || {};
    if (!Array.isArray(c.center) || c.center.length !== 2 || !Number.isFinite(c.zoom)) {
      throw new Error(`stop ${s.id} has no camera`);
    }
  }
  return story.stops;
}

/** How long a stop stays on screen while playing (ms): long enough to read its note, at most 11 s. */
export function dwellMs(stop, { still = false } = {}) {
  const words = String((stop && stop.note) || "").split(/\s+/).filter(Boolean).length;
  const base = stop && (stop.kind === "wave" || stop.kind === "flood") ? 6500 : 4200;
  return still ? Math.min(11000, base + words * 170) : Math.min(11000, base + words * 140);
}

/** The camera for a stop on this screen: a phone's map is a strip above the sheet, so it steps back a little. */
export function cameraFor(stop, { narrow = false } = {}) {
  const c = stop.camera;
  return {
    center: [Number(c.center[0]), Number(c.center[1])],
    zoom: Number(c.zoom) - (narrow ? 0.8 : 0),
    pitch: Math.max(0, Math.min(70, Number(c.pitch || 0))),
    bearing: Number(c.bearing || 0),
  };
}

/** A template from the package with its {fields} filled; a field missing stays visible rather than vanishing. */
export function fillTemplate(template, fill) {
  return String(template || "").replace(/\{(\w+)\}/g, (m, k) => (fill && fill[k] !== undefined && fill[k] !== null ? String(fill[k]) : m));
}

const n1 = (v) => Number(v).toLocaleString("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const n0 = (v) => Math.round(Number(v)).toLocaleString("en-GB");

/** The flood stop's fields from the page's own run (Flood it's summary), as the package formats them. */
export function floodFill(stats, base = {}) {
  return { ...base, wet_km2: n1(stats.wet_km2), max_depth_m: n1(stats.max_m ?? stats.max_depth_m) };
}

/** The exposure stop's fields from the flood views' count (compare_exposure's "after"), or null without one. */
export function exposureFill(res) {
  if (!res || !res.buildings) return null;
  const after = (k) => (res[k] && typeof res[k] === "object" ? res[k].after : res[k]);
  if (after("buildings") === undefined || after("buildings") === null) return null;
  return { buildings: n0(after("buildings")), roads_km: n1(after("roads_km") || 0), people: n0(after("people") || 0) };
}

/**
 * The share link's value: the reach and the window, or the place and the month Floods past is asked about, and for a
 * Floods past event its number of news reports, so the link replays the same story (the event's first note says it).
 */
export function encodeReplay(ev) {
  if (!ev) return "";
  const parts = [ev.river_id ?? "", ev.start || "", ev.end || ""];
  if (Number.isFinite(Number(ev.lat)) && Number.isFinite(Number(ev.lon))) {
    parts.push(Number(ev.lat).toFixed(4), Number(ev.lon).toFixed(4));
    if (ev.reports !== null && ev.reports !== undefined && Number.isInteger(Number(ev.reports)) && Number(ev.reports) > 0) {
      parts.push(String(Number(ev.reports)));
    }
  }
  return parts.join(",");
}

const ISO = /^\d{4}-\d{2}-\d{2}$/;

/** `rp=river_id,start,end[,lat,lon[,reports]]` from a hash, or null. With reports it is a Floods past event again. */
export function decodeReplay(hash) {
  const q = new URLSearchParams(String(hash || "").replace(/^#/, ""));
  if (!q.has("rp")) return null;
  const [rid, start, end, lat, lon, reports] = String(q.get("rp")).split(",");
  if (!ISO.test(start || "") || (end && !ISO.test(end))) return null;
  const out = { river_id: /^\d{9}$/.test(rid || "") ? Number(rid) : null, start, end: end || start };
  if (lat !== undefined && lon !== undefined && Number.isFinite(Number(lat)) && Number.isFinite(Number(lon))) {
    out.lat = Number(lat);
    out.lon = Number(lon);
  }
  if (out.river_id === null && out.lat === undefined) return null;
  if (out.lat !== undefined && /^\d+$/.test(reports || "") && Number(reports) > 0) {
    out.reports = Number(reports);
    out.event = { start: out.start, end: out.end, lat: out.lat, lon: out.lon, reports: out.reports };
  }
  return out;
}

// What the assembling list says for each piece, in the order the package reads them.
export const PIECES = [
  ["gauges", "The gauges against GEOGLOWS"],
  ["rain", "The rain over the basin (IMERG)"],
  ["forecast", "What the forecast said"],
  ["wave", "The wave downstream (Route it)"],
  ["flood", "The flood at the peak (Flood it)"],
];

/** The assembling list: each piece's state ("ok", "missing", "pending", "waiting") and its line. */
export function pieceRows(record) {
  const P = (record && record.pieces) || {};
  return PIECES.map(([k, label]) => {
    const p = P[k];
    if (!p) return { piece: k, label, state: "waiting", text: "" };
    if (p.pending) return { piece: k, label, state: "pending", text: k === "flood" ? "runs on your GPU when the story reaches it" : "still reading" };
    if (p.available) return { piece: k, label, state: "ok", text: "" };
    return { piece: k, label, state: "missing", text: String(p.note || "not available") };
  });
}

/** The scrubber's segments: each stop's share of the whole play, so a long stop is a longer segment. */
export function scrubSegments(stops) {
  const d = stops.map((s) => dwellMs(s));
  const total = d.reduce((a, b) => a + b, 0) || 1;
  let at = 0;
  return d.map((ms, i) => { const seg = { i, from: at / total, width: ms / total }; at += ms; return seg; });
}

// ── the small charts beside a note ──────────────────────────────────────────

function scale(lo, hi, a, b) {
  const k = hi > lo ? (b - a) / (hi - lo) : 0;
  return (v) => a + (v - lo) * k;
}

/** A small SVG chart for a stop that carries a series: flow (observed and modelled) or rain (daily bars). Plain
 * marks, no axes but the peak's value; the numbers stay in the note. */
export function sparkline(series, { w = 168, h = 64, marker = null } = {}) {
  if (!series) return "";
  const pad = 4;
  if (series.kind === "rain") {
    const v = (series.values || []).map((r) => Number(r[1]));
    if (!v.length) return "";
    const hi = Math.max(...v, 1);
    const bw = (w - 2 * pad) / v.length;
    const y = scale(0, hi, h - pad, pad + 10);
    const bars = v.map((x, i) => `<rect x="${(pad + i * bw).toFixed(1)}" y="${y(x).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" ` +
      `height="${(h - pad - y(x)).toFixed(1)}" class="rp-bar${x === hi ? " pk" : ""}"/>`).join("");
    return `<svg class="rp-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Daily basin rain">` +
      `${bars}<text x="${w - pad}" y="10" text-anchor="end" class="rp-sv">${n0(hi)} mm</text></svg>`;
  }
  const obs = (series.observed || []).map((r) => [r[0], Number(r[1])]);
  const mod = (series.modelled || []).map((r) => [r[0], Number(r[1])]);
  const days = [...new Set([...obs, ...mod].map((r) => r[0]))].sort();
  if (days.length < 2) return "";
  const all = [...obs, ...mod].map((r) => r[1]);
  const hi = Math.max(...all, 1e-6);
  const x = (d) => pad + (days.indexOf(d) / (days.length - 1)) * (w - 2 * pad);
  const y = scale(0, hi, h - pad, pad + 10);
  const path = (rows) => rows.map((r, i) => `${i ? "L" : "M"}${x(r[0]).toFixed(1)},${y(r[1]).toFixed(1)}`).join("");
  let mk = "";
  if (marker && days.includes(marker)) mk = `<line x1="${x(marker).toFixed(1)}" x2="${x(marker).toFixed(1)}" y1="${pad}" y2="${h - pad}" class="rp-mk"/>`;
  return `<svg class="rp-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Daily flow, observed and modelled">` +
    mk + (mod.length ? `<path d="${path(mod)}" class="rp-mod"/>` : "") + (obs.length ? `<path d="${path(obs)}" class="rp-obs"/>` : "") +
    `<text x="${w - pad}" y="10" text-anchor="end" class="rp-sv">${n0(hi)} m³/s</text>` +
    `<text x="${pad}" y="10" class="rp-sv">${obs.length ? "─ gauge" : ""}${obs.length && mod.length ? "  " : ""}${mod.length ? "- - GEOGLOWS" : ""}</text></svg>`;
}

/** The forecast stop's chart: the members' range for the peak day as a bar, the median and the observed peak. */
export function forecastBar(run, observed, { w = 168, h = 64 } = {}) {
  if (!run || run.median_cms === undefined) return "";
  const pad = 6;
  const vals = [run.low_cms, run.high_cms, run.median_cms, observed].filter((v) => Number.isFinite(Number(v))).map(Number);
  const hi = Math.max(...vals) * 1.08;
  const x = scale(0, hi, pad, w - pad);
  const yb = h / 2;
  const at = (v) => (x(v) > w * 0.7 ? "end" : x(v) < w * 0.3 ? "start" : "middle");
  let out = `<line x1="${pad}" x2="${w - pad}" y1="${yb + 14}" y2="${yb + 14}" class="rp-mk"/>` +
    `<text x="${pad}" y="10" class="rp-sv">members, median ┃</text>` +
    `<rect x="${x(run.low_cms).toFixed(1)}" y="${yb - 7}" width="${Math.max(2, x(run.high_cms) - x(run.low_cms)).toFixed(1)}" height="14" class="rp-range"/>` +
    `<line x1="${x(run.median_cms).toFixed(1)}" x2="${x(run.median_cms).toFixed(1)}" y1="${yb - 11}" y2="${yb + 11}" class="rp-med"/>`;
  if (Number.isFinite(Number(observed))) {
    out += `<path d="M${x(observed).toFixed(1)},${yb - 15}l-5,-8h10z" class="rp-obsmk"/>` +
      `<text x="${x(observed).toFixed(1)}" y="${h - 4}" text-anchor="${at(observed)}" class="rp-sv">gauge ${n0(observed)} m³/s</text>`;
  }
  return `<svg class="rp-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="The forecast's members for the peak day against the gauge">${out}</svg>`;
}
