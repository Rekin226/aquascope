// Floods ahead (#546), the pure part: no DOM, no map, so node can test it.
// Which reaches are expected to flood, by how much and on which day all come
// from Python (aquascope.archive.warnings, run daily by flood-warnings.yml);
// this module only decides how that is drawn and said: the class colours, the
// class on the map's date, the MapLibre expressions and the short sentences.

// Return-period classes, yellow to deep purple. Lightness falls steadily with
// the class, so the order still reads under any colour-vision deficiency. The
// soft glow under each line keeps the dark end visible on a dark map, and a thin
// dark casing keeps the pale yellow end visible on a light one.
export const FLOOD_CLASSES = [
  { rp: 2, label: "2-year", color: "#f2c94c" },
  { rp: 5, label: "5-year", color: "#f2994a" },
  { rp: 10, label: "10-year", color: "#e2553f" },
  { rp: 25, label: "25-year", color: "#c2185b" },
  { rp: 50, label: "50-year", color: "#8e24aa" },
  { rp: 100, label: "100-year", color: "#4a148c" },
];
// One character per day in a reach's `daily`: the index into this list (aquascope.archive.warnings.DAILY_CODES).
export const DAILY_CODES = [0, 2, 5, 10, 25, 50, 100];
export const FORECAST_DAYS = 15;
export const NONE = "rgba(0,0,0,0)";

export const FLOODS_CREDIT = {
  label: "Floods ahead",
  attribution: 'GEOGLOWS v2 forecast and return periods (<a href="https://registry.opendata.aws/geoglows-v2/">GEOGloWS ECMWF Streamflow Service</a>), classed daily by AquaScope',
  licence: "forecast CC BY 4.0, return periods CC BY-NC-SA 4.0; model output, not an official warning",
};

export const classColor = (rp) => (FLOOD_CLASSES.find((c) => c.rp === Number(rp)) || {}).color || NONE;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const ISO = /^(\d{4})-(\d{2})-(\d{2})$/;
const utc = (iso) => {
  const m = ISO.exec(String(iso || ""));
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3]) : null;
};

/** "Mon 12 Oct" for an ISO day. */
export function shortDay(iso) {
  const t = utc(iso);
  if (t === null) return "";
  const d = new Date(t);
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

/** The ISO day `n` days after `iso`. */
export function addDays(iso, n) {
  const t = utc(iso);
  return t === null ? null : new Date(t + n * 86400000).toISOString().slice(0, 10);
}

/** Which forecast day the map's date is (0 for the run's start), or -1 outside the 15 days. */
export function dayIndex(issueDate, date) {
  const a = utc(issueDate), b = utc(date);
  if (a === null || b === null) return -1;
  const i = Math.round((b - a) / 86400000);
  return i >= 0 && i < FORECAST_DAYS ? i : -1;
}

/** A reach's class on forecast day `i` from its `daily` string; -1 means the 15-day peak class. */
export function classOn(props, i) {
  if (!props) return 0;
  if (i < 0) return Number(props.rp) || 0;
  const code = Number(String(props.daily || "").charAt(i));
  return DAILY_CODES[code] || 0;
}

/** River ids by class for one view (a day, or the peak with -1), highest class first. */
export function idsByClass(features, i) {
  const out = new Map(FLOOD_CLASSES.map((c) => [c.rp, []]).reverse());
  for (const f of features || []) {
    const rp = classOn(f.properties, i);
    if (rp && out.has(rp)) out.get(rp).push(Number(f.properties.river_id));
  }
  return out;
}

/** The line colour for the stream tiles: each flooded riverId in its class colour, everything else transparent. */
export function lineColorExpr(groups) {
  const branches = [];
  for (const [rp, ids] of groups) if (ids.length) branches.push(ids, classColor(rp));
  return branches.length ? ["match", ["to-number", ["get", "riverId"]], ...branches, NONE] : NONE;
}

/** Only the flooded reaches of the stream tiles; nothing at all when there are none. */
export function lineFilterExpr(groups) {
  const ids = [];
  for (const list of groups.values()) ids.push(...list);
  return ids.length ? ["match", ["to-number", ["get", "riverId"]], ids, true, false] : ["boolean", false];
}

/** The points again with `c`, the class shown now, and only those flooded in this view. */
export function pointsFor(features, i) {
  const out = [];
  for (const f of features || []) {
    const c = classOn(f.properties, i);
    if (c) out.push({ ...f, properties: { ...f.properties, c } });
  }
  return { type: "FeatureCollection", features: out };
}

/** The Archive gauges on reaches flooded in this view, with the class to pulse in. */
export function gaugesFor(features, i) {
  const out = new Map();
  for (const f of features || []) {
    const c = classOn(f.properties, i);
    if (!c || !f.properties.gauges) continue;
    for (const key of String(f.properties.gauges).split(";")) {
      if (key && (out.get(key) || 0) < c) out.set(key, c);
    }
  }
  return out;
}

export function countsFor(features, i) {
  const counts = new Map(FLOOD_CLASSES.map((c) => [c.rp, 0]));
  for (const f of features || []) {
    const c = classOn(f.properties, i);
    if (c) counts.set(c, (counts.get(c) || 0) + 1);
  }
  return counts;
}

/** The legend's one line: what is drawn, for which day, from which run. */
export function legendLine(manifest, i, n) {
  if (!manifest || manifest.missing) return "Nothing published yet. It appears after the first daily run.";
  const start = manifest.issue_date;
  const reaches = `${Number(n || 0).toLocaleString("en-GB")} river reach${n === 1 ? "" : "es"}`;
  if (i >= 0) return `${shortDay(addDays(start, i))}, day ${i + 1} of ${FORECAST_DAYS}: ${reaches} at or above the 2-year flow.`;
  const by = shortDay(manifest.valid_to || addDays(start, FORECAST_DAYS - 1));
  // The browser file keeps the biggest reaches when there are very many; say so rather than undercount.
  const of = manifest.geojson_truncated && manifest.n > n ? ` (the largest of ${Number(manifest.n).toLocaleString("en-GB")})` : "";
  return `${reaches}${of} expected to reach the 2-year flow by ${by}.`;
}

export function issueLine(manifest) {
  if (!manifest || manifest.missing) return "";
  return `GEOGLOWS forecast of ${shortDay(manifest.issue_date)}${manifest.smoke ? " (smoke sample)" : coverageNote(manifest)}`;
}

/** " (partial: 87% of rivers read)" when the daily job ran out of time or lost chunks, else "". A gap in the
 * reading is a gap on the map, so the legend says so rather than look complete. */
export function coverageNote(manifest) {
  const c = (manifest && manifest.chunks) || {};
  const needed = Number(c.needed), read = Number(c.read);
  if (!(needed > 0) || !Number.isFinite(read) || read >= needed) return "";
  return ` (partial: ${Math.floor((100 * read) / needed)}% of rivers read)`;
}

const fmtFlow = (x) => {
  const n = Number(x);
  if (!Number.isFinite(n)) return "?";
  return n >= 100 ? Math.round(n).toLocaleString("en-GB") : Number(n.toPrecision(2)).toLocaleString("en-GB");
};

/** The card's facts for one reach, as plain sentences (the page escapes and lays them out). */
export function reachFacts(props, i = -1, members = 51) {
  const rp = Number(props.rp) || 0;
  const now = classOn(props, i);
  const ratio = Number(props.peak) / Number(props.q2);
  const share = Number(props.share);
  const facts = {
    title: rp ? `${rp}-year flow expected` : "Below the 2-year flow",
    peak: `Peak ${fmtFlow(props.peak)} m³/s on ${shortDay(props.day)}` +
      (Number.isFinite(ratio) ? `, ${ratio.toFixed(1)} times the 2-year flow (${fmtFlow(props.q2)} m³/s).` : "."),
    agree: Number.isFinite(share) ? `${Math.round(share * members)} of the ${members} forecast members reach the 2-year flow.` : "",
    reach: `River reach ${props.river_id}${props.order ? `, stream order ${props.order}` : ""}.`,
    today: i >= 0 ? (now ? `On the map's date: ${now}-year flow.` : "On the map's date: below the 2-year flow.") : "",
  };
  return facts;
}
