// Floods past (#547), the pure part: no DOM, no map, so node can test it.
//
// The numbers come from Python: aquascope.context.floods_past rolls the
// Archive's Groundsource news events and Microsoft Sentinel-1 radar detections
// into one small file per month on a half-degree grid (the mirror-context
// workflow publishes it), and answers a clicked cell with its events. This
// module only decides which months the time bar is pointing at, how a cell is
// drawn, and how the legend says it.

export const FLOODS_PAST_BASE =
  "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/context/floods/monthly/";

// Two sources, two colours and two marks, so neither colour nor shape alone
// carries the difference: news is a crisp orange dot, radar a soft violet
// glow. Orange and violet stay apart under the common colour-vision
// deficiencies (they fall on the yellow and blue sides), and both read on the
// light and the dark basemap.
export const NEWS_COLOR = "#e8820c";
export const NEWS_STROKE = "#7a3c00";
export const RADAR_COLOR = "#8b5cf6";

export const NEWS_CREDIT = {
  label: "Floods past: news",
  attribution: 'Flood events from news: <a href="https://doi.org/10.5281/zenodo.18647054">Groundsource</a> (Mayo et al. 2026, Google)',
  licence: "CC BY 4.0",
};
export const RADAR_CREDIT = {
  label: "Floods past: radar",
  attribution: 'Sentinel-1 flood detections: <a href="https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset">Microsoft AI for Good Lab</a> (Misra et al. 2025)',
  licence: "MIT",
};
export const RADAR_PERIOD = ["2014-10", "2024-09"];

// How many months a window may span (one small file each), and the default.
export const MAX_MONTHS = 60;
export const DEFAULT_WINDOW = 12;

const MONTH = /^(\d{4})-(\d{2})/;
const NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** The month of a day or a month ("2024-07-14" gives "2024-07"); null for anything else. */
export function monthOf(value) {
  const m = MONTH.exec(String(value || ""));
  return m && +m[2] >= 1 && +m[2] <= 12 ? `${m[1]}-${m[2]}` : null;
}

export function addMonths(month, n) {
  const m = MONTH.exec(month);
  const total = +m[1] * 12 + (+m[2] - 1) + n;
  const y = Math.floor(total / 12);
  return `${String(y).padStart(4, "0")}-${String(total - y * 12 + 1).padStart(2, "0")}`;
}

export function monthsBetween(a, b) {
  let [from, to] = [monthOf(a), monthOf(b)];
  if (!from || !to) return [];
  if (to < from) [from, to] = [to, from];
  const out = [from];
  while (out[out.length - 1] < to) out.push(addMonths(out[out.length - 1], 1));
  return out;
}

export function monthLabel(month) {
  const m = MONTH.exec(month || "");
  return m ? `${NAMES[+m[2] - 1]} ${m[1]}` : "";
}

export function windowLabel(months) {
  if (!months || !months.length) return "";
  return months.length === 1 ? monthLabel(months[0])
    : `${monthLabel(months[0])} to ${monthLabel(months[months.length - 1])}`;
}

/**
 * The months the layer shows for the time bar's state, given the last month on
 * record. Playing, or stepping by month: the month of the date, so a season
 * replays (or is walked) month by month. A range set: its months (the last 60
 * at most). Otherwise the 12 months that end at the map date, or at the last
 * month on record when the date is later, which is what the page opens on
 * (also when stepping by month, but not while playing).
 */
export function windowFor({ date, range, playing, step } = {}, last) {
  const at = monthOf(date);
  // Stepping by month to a date after the record (the world river status opens
  // the map on its newest month, #544) falls through to the latest twelve, so
  // the layer is not blank on the first view; playing still shows that month.
  const afterRecord = Boolean(!playing && last && at && at > last);
  if ((playing || step === "month") && at && !afterRecord) return { months: [at], mode: "frame", latest: false };
  const r = range && monthOf(range.from) && monthOf(range.to) ? monthsBetween(range.from, range.to) : null;
  if (r && r.length) {
    const months = r.slice(-MAX_MONTHS);
    return { months, mode: "range", latest: false, truncated: r.length > months.length };
  }
  let end = at || last;
  const latest = Boolean(last && end && end > last);
  if (latest) end = last;
  if (!end) return { months: [], mode: "window", latest: false };
  return { months: monthsBetween(addMonths(end, -(DEFAULT_WINDOW - 1)), end), mode: "window", latest };
}

/** The months of a window that the published index has a file for. */
export function monthsOnRecord(index, months) {
  const have = new Set(((index && index.months) || []).map((m) => m.month));
  return (months || []).filter((m) => have.has(m));
}

/** Totals for the legend, from the index (the per-month sums Python wrote). */
export function windowTotals(index, months) {
  const want = new Set(months || []);
  let news = 0, radar = 0;
  for (const m of (index && index.months) || []) {
    if (want.has(m.month)) { news += m.news || 0; radar += m.radar || 0; }
  }
  return { news, radar };
}

/** Whether any month of the window falls inside the radar's period. */
export function radarCovers(months) {
  return (months || []).some((m) => m >= RADAR_PERIOD[0] && m <= RADAR_PERIOD[1]);
}

// ── drawing ─────────────────────────────────────────────────────────────────

/** The bounds of a half-degree cell: [west, south, east, north]. */
export function cellBbox(row, col, deg = 0.5) {
  const s = -90 + row * deg, w = -180 + col * deg;
  return [w, s, w + deg, s + deg].map((v) => Math.round(v * 1e6) / 1e6);
}

/**
 * Several months' [row, col, news, radar] cells as one GeoJSON point per cell,
 * at its centre, with the window's sums (the same sum Python's merge_cells
 * makes, so the circle and the clicked cell's count agree).
 */
export function cellsGeoJSON(monthsCells, deg = 0.5) {
  const acc = new Map();
  for (const cells of monthsCells || []) {
    for (const [row, col, news, radar] of cells || []) {
      const key = row * 10000 + col;
      const a = acc.get(key);
      if (a) { a[2] += news; a[3] += radar; } else acc.set(key, [row, col, news, radar]);
    }
  }
  const features = [];
  const half = deg / 2;
  for (const [row, col, news, radar] of acc.values()) {
    const lon = -180 + (col + 0.5) * deg, lat = -90 + (row + 0.5) * deg;
    features.push({
      type: "Feature",
      properties: { row, col, news, radar },
      geometry: { type: "Point", coordinates: [lon, lat] },
    });
    // Radar is drawn as the cell itself once zoomed in: water seen over an area, not a report at a point.
    if (radar > 0) {
      features.push({
        type: "Feature",
        properties: { row, col, news, radar, cell: 1 },
        geometry: { type: "Polygon", coordinates: [[
          [lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
          [lon - half, lat - half]]] },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

// The world view is a heat map of the cells, weighted by the logarithm of the
// count (a cell with 1,000 news events is not 1,000 times hotter than one with
// a single event, or the map would be one orange blot over Jakarta). Closer in,
// radar is the half-degree cells themselves, shaded by their count, and news a
// circle per cell, both of which a click can land on.
export const HEAT_MAXZOOM = 5;
export const CELL_MINZOOM = 4.5;

export const POINTS = ["==", ["geometry-type"], "Point"];
export const CELLS = ["==", ["geometry-type"], "Polygon"];

const ln1p = (field) => ["ln", ["+", 1, ["get", field]]];
// ln(1 + n) / ln(1 + full): a cell with `full` or more counts weighs 1.
const weight = (field, full) => ["min", 1, ["/", ln1p(field), Math.log(1 + full)]];
// Radar detections below `floor` weigh nothing at the world view: a few stray 20 m pixels in a year are
// everywhere and would tint whole continents. The cells and the click still show them.
const rampFrom = (field, floor, full) => ["max", 0, ["min", 1,
  ["/", ["-", ln1p(field), Math.log(1 + floor)], Math.log(1 + full) - Math.log(1 + floor)]]];

// The heat is scaled to the length of the window, so a single month being
// replayed reads as clearly as the twelve months the page opens on: a cell
// "full" of news has 25 events a month, of radar 80,000 detections a month,
// and radar below 80 a month is left out of the heat.
export function newsWeight(months = 12) {
  return weight("news", 25 * Math.max(1, months));
}

export function radarWeight(months = 12) {
  const m = Math.max(1, months);
  return rampFrom("radar", 80 * m, 80000 * m);
}

export function newsHeat(months = 12) {
  return {
    "heatmap-weight": newsWeight(months),
    "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.55, 3, 0.9, 5.5, 1.3],
    "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 2.5, 2, 5, 4, 11, 5.5, 18],
    "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
      0, "rgba(253,200,120,0)", 0.12, "rgba(253,190,105,0.32)", 0.35, "rgba(244,150,40,0.62)",
      0.65, "rgba(222,110,8,0.82)", 1, "rgba(150,58,0,0.92)"],
  };
}

export function radarHeat(months = 12) {
  return {
    "heatmap-weight": radarWeight(months),
    "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.55, 3, 0.9, 5.5, 1.3],
    "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 3, 2, 6, 4, 13, 5.5, 20],
    "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
      0, "rgba(167,139,250,0)", 0.12, "rgba(160,130,250,0.35)", 0.35, "rgba(139,92,246,0.6)",
      0.65, "rgba(112,50,225,0.8)", 1, "rgba(76,29,149,0.92)"],
  };
}

export function newsRadius() {
  return ["interpolate", ["linear"], ["zoom"],
    4, ["+", 2, ["*", 0.85, ln1p("news")]],
    8, ["+", 4, ["*", 1.6, ln1p("news")]],
  ];
}

// The radar cell's colour: clear below 200 detections (a few hectares of 20 m pixels, which three cells in
// four have in any month and which would otherwise tint whole countries), deepening to violet by two million
// (some 800 km² of water). On a dark basemap the ramp climbs to a lighter violet instead, so the wettest
// cells stand out rather than sink. The fill's own opacity is left free for the crossfade between months.
export const DARK_BASEMAPS = new Set(["dark", "satellite", "satellite-recent", "daily"]);

export function radarFill(dark = false) {
  const stops = dark
    ? ["rgba(196,181,253,0)", "rgba(167,139,250,0.12)", "rgba(167,139,250,0.24)", "rgba(180,158,252,0.4)",
      "rgba(205,190,254,0.55)"]
    : ["rgba(167,139,250,0)", "rgba(167,139,250,0.16)", "rgba(139,92,246,0.32)", "rgba(109,40,217,0.5)",
      "rgba(76,29,149,0.68)"];
  const at = [200, 2000, 20000, 200000, 2000000];
  return ["interpolate", ["linear"], ln1p("radar"), ...at.flatMap((n, i) => [Math.log(1 + n), stops[i]])];
}

// A news circle fades with a small count, so a cell with one report is a whisper and a cell with fifty is
// solid: closer in, every inhabited cell has a report or two, and equal dots would be confetti. The colour
// carries this, not circle-opacity, which the crossfade between months uses.
export function newsFill() {
  return ["interpolate", ["linear"], ln1p("news"),
    Math.log(2), "rgba(232,130,12,0.28)", Math.log(1 + 10), "rgba(232,130,12,0.7)", Math.log(1 + 50), "rgba(232,130,12,1)"];
}

export function newsStroke() {
  return ["interpolate", ["linear"], ln1p("news"),
    Math.log(2), "rgba(122,60,0,0.15)", Math.log(1 + 10), "rgba(122,60,0,0.5)", Math.log(1 + 50), "rgba(122,60,0,0.8)"];
}

export function fmtCount(n) {
  const v = Number(n) || 0;
  if (v >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)} M`;
  if (v >= 1e4) return `${Math.round(v / 1000)}k`;
  return v.toLocaleString("en-GB");
}

/** The legend's two lines, in as few words as will do. */
export function legendLines(index, win) {
  if (!index) return null;
  const months = win.months || [];
  const totals = windowTotals(index, months);
  const newsLast = monthOf((index.news && index.news.last) || index.last);
  const newsFirst = monthOf((index.news && index.news.first) || index.first);
  const covered = radarCovers(months);
  const newsNone = months.length && months.every((m) => m > newsLast || m < newsFirst);
  return {
    when: windowLabel(months) + (win.latest ? " (latest on record)" : "") + (win.truncated ? " (last 60 months)" : ""),
    news: newsNone ? `${newsFirst.slice(0, 4)} to ${newsLast.slice(0, 4)} only` : `${fmtCount(totals.news)} events`,
    radar: covered ? `${fmtCount(totals.radar)} detections` : `${RADAR_PERIOD[0].slice(0, 4)} to ${RADAR_PERIOD[1].slice(0, 4)} only`,
    newsCount: totals.news,
    radarCount: covered ? totals.radar : null,
  };
}

/** An event's affected area: "120 km²", "under 1 km²", or nothing when the source has none. */
export function areaLabel(km2) {
  const v = Number(km2);
  if (!Number.isFinite(v) || v <= 0) return "";
  return v < 1 ? "under 1 km²" : `${fmtCount(Math.round(v))} km²`;
}

/** "29 Jul 2021" or "29 to 31 Jul 2021" for a news event. */
export function eventDates(start, end) {
  const d = (s) => {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s || "");
    return m ? { y: m[1], m: NAMES[+m[2] - 1], d: String(+m[3]) } : null;
  };
  const a = d(start), b = d(end);
  if (!a) return "";
  if (!b || (b.y === a.y && b.m === a.m && b.d === a.d)) return `${a.d} ${a.m} ${a.y}`;
  if (b.y === a.y && b.m === a.m) return `${a.d} to ${b.d} ${a.m} ${a.y}`;
  if (b.y === a.y) return `${a.d} ${a.m} to ${b.d} ${b.m} ${a.y}`;
  return `${a.d} ${a.m} ${a.y} to ${b.d} ${b.m} ${b.y}`;
}

/** A cell centre as "26.25° N, 85.75° E". */
export function placeLabel(lat, lon) {
  const ns = lat >= 0 ? "N" : "S", ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}° ${ns}, ${Math.abs(lon).toFixed(2)}° ${ew}`;
}

// The URL flag: the layer is on by default, so a link only carries fp=0.
export function readFloodsParam(hash) {
  const q = new URLSearchParams(String(hash || "").replace(/^#/, ""));
  return q.has("fp") ? q.get("fp") !== "0" : null;
}
