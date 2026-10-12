// The map card (#548), the pure part: no DOM, no map, so node can test it.
// Every number on the card comes from somewhere else: the daily status
// snapshot (forecasts/status/latest.parquet, written by aquascope.nownext in
// the forecast job), the record aquascope.explore returned, or the forecast
// aquascope.nownext returned. This module only decides how they are said and
// drawn: the snapshot's sentence, the last twelve months of a record, the
// forecast's peak, and the sparkline's paths.

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
const SHORT = MONTHS.map((m) => m.slice(0, 3));

// The words aquascope.nownext.flow_status uses, so a snapshot row reads exactly like the Now tab.
const SUBJECT = { discharge: "Flow", water_level: "Water level", groundwater_level: "Groundwater level" };
const LABEL = {
  much_below: "much below normal", below: "below normal", normal: "normal", above: "above normal",
  much_above: "much above normal",
};

export function ordinal(n) {
  if (n === null || n === undefined || n === "") return "";
  const k = Math.round(Number(n));
  if (!Number.isFinite(k)) return "";
  if (k % 100 >= 10 && k % 100 <= 20) return `${k}th`;
  return `${k}${{ 1: "st", 2: "nd", 3: "rd" }[k % 10] || "th"}`;
}

function parseDay(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ""));
  return m ? { y: Number(m[1]), m: Number(m[2]), d: Number(m[3]) } : null;
}

const dayNumber = (p) => Date.UTC(p.y, p.m - 1, p.d) / 86400000;

// "8 October" or, with the year, "8 October 2026"; `short` gives "8 Oct".
export function dayMonth(iso, { year = false, short = false } = {}) {
  const p = parseDay(iso);
  if (!p) return "";
  return `${p.d} ${(short ? SHORT : MONTHS)[p.m - 1]}${year ? ` ${p.y}` : ""}`;
}

// The sentence for one row of the daily status snapshot ({cls, pct, date, n_years}), in
// flow_status's own form: "Flow is above normal for 8 October (82nd percentile of 46 years)." when the
// day is at most two days before `today`, else "Flow was ... on 8 October 2026 (...)." (flow_status adds
// ", the latest day in the record", which the card leaves out for room). null when the row has no class.
// tests/test_explorer/test_map_card.py holds this to aquascope.nownext.flow_status word for word.
export function snapshotSentence(row, { variable = "discharge", today = null } = {}) {
  if (!row || !LABEL[row.cls]) return null;
  const word = SUBJECT[variable] || "The value";
  const pct = Number(row.pct);
  const years = Number(row.n_years);
  const of = Number.isFinite(pct)
    ? ` (${ordinal(pct)} percentile${Number.isFinite(years) && years > 0 ? ` of ${years} years` : ""})`
    : "";
  const day = parseDay(row.date);
  const now = parseDay(today || new Date().toISOString());
  const age = day && now ? dayNumber(now) - dayNumber(day) : Infinity;
  if (age <= 2) return `${word} is ${LABEL[row.cls]} for ${dayMonth(row.date)}${of}.`;
  return `${word} was ${LABEL[row.cls]} on ${dayMonth(row.date, { year: true })}${of}.`;
}

const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

// The last `days` days of a daily record ({t, v}), counted back from its own last day, so a record that
// ended in 1995 still draws its last year rather than nothing.
export function lastDays(series, days = 365) {
  const t = (series && series.t) || [];
  const v = (series && series.v) || [];
  if (!t.length) return { t: [], v: [] };
  const end = parseDay(t[t.length - 1]);
  if (!end) return { t: [], v: [] };
  const from = dayNumber(end) - days + 1;
  let i = t.length - 1;
  while (i > 0) {
    const p = parseDay(t[i - 1]);
    if (!p || dayNumber(p) < from) break;
    i--;
  }
  return { t: t.slice(i), v: v.slice(i).map((x) => (finite(x) ? Number(x) : null)) };
}

// The record's latest value and its day: the last finite one.
export function latestValue(series) {
  const t = (series && series.t) || [];
  const v = (series && series.v) || [];
  for (let i = v.length - 1; i >= 0; i--) if (finite(v[i])) return { value: Number(v[i]), date: t[i] };
  return null;
}

// The forecast's highest daily ensemble mean and its day ({date, mean, ...} from nownext.forecast).
export function forecastPeak(part) {
  const t = (part && part.date) || [];
  const v = (part && part.mean) || [];
  let best = null;
  for (let i = 0; i < v.length; i++) {
    if (finite(v[i]) && (!best || Number(v[i]) > best.value)) best = { value: Number(v[i]), date: t[i] };
  }
  return best;
}

// A value as the card prints it: the precision follows the magnitude, as nownext's _fmt_q does.
export function cardNumber(x) {
  if (!finite(x)) return "";
  const n = Number(x);
  const a = Math.abs(n);
  if (a >= 100) return Math.round(n).toLocaleString("en-US");
  if (a >= 10) return n.toFixed(1);
  if (a === 0) return "0";
  return String(Number(n.toPrecision(3)));
}

export const prettyUnit = (u) => ({ "m3/s": "m³/s", "ft3/s": "ft³/s" }[u] || u || "");

// The sparkline as SVG path data, in a `w` x `h` box with `pad` pixels kept clear at the top and bottom.
// `line` is the series (gaps break it), `band` an optional {lo, hi} pair drawn as one filled shape, and
// `area` the same line closed down to the bottom (for a record, which has no band). A record is scaled from
// zero when it never goes below it, so a dry spell sits low rather than filling the box. A forecast
// (`fromZero: false`) is scaled to its own range, so its rise or fall shows rather than a flat line near the
// top; the range is kept at least a fifth of the largest value, so a steady river is not drawn as a flood.
export function sparkPaths({ v = [], band = null }, { w = 240, h = 44, pad = 3, fromZero = true } = {}) {
  const vals = v.map((x) => (finite(x) ? Number(x) : null));
  const all = vals.filter((x) => x !== null);
  if (band) for (const k of ["lo", "hi"]) for (const x of band[k] || []) if (finite(x)) all.push(Number(x));
  if (all.length < 2) return null;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (fromZero && lo >= 0) lo = 0;
  if (!fromZero) {
    const least = Math.abs(hi) * 0.2;
    if (hi - lo < least) {
      const mid = (hi + lo) / 2;
      lo = mid - least / 2;
      hi = mid + least / 2;
      if (lo < 0 && Math.min(...all) >= 0) { hi -= lo; lo = 0; }
    }
  }
  const span = hi - lo || 1;
  const n = vals.length;
  const x = (i) => (n === 1 ? w / 2 : (i * w) / (n - 1));
  const y = (val) => pad + (h - 2 * pad) * (1 - (val - lo) / span);
  const r = (k) => Math.round(k * 10) / 10;
  let line = "";
  let pen = false;
  for (let i = 0; i < n; i++) {
    if (vals[i] === null) { pen = false; continue; }
    line += `${pen ? "L" : "M"}${r(x(i))},${r(y(vals[i]))}`;
    pen = true;
  }
  let area = "";
  const firstI = vals.findIndex((q) => q !== null);
  let lastI = -1;
  for (let i = n - 1; i >= 0; i--) if (vals[i] !== null) { lastI = i; break; }
  if (!band && firstI >= 0 && lastI > firstI) {
    area = `M${r(x(firstI))},${r(y(lo))}`;
    for (let i = firstI; i <= lastI; i++) if (vals[i] !== null) area += `L${r(x(i))},${r(y(vals[i]))}`;
    area += `L${r(x(lastI))},${r(y(lo))}Z`;
  }
  let fill = "";
  if (band && band.lo && band.hi) {
    const top = [];
    const bottom = [];
    for (let i = 0; i < n; i++) {
      if (finite(band.hi[i]) && finite(band.lo[i])) {
        top.push(`${r(x(i))},${r(y(Number(band.hi[i])))}`);
        bottom.unshift(`${r(x(i))},${r(y(Number(band.lo[i])))}`);
      }
    }
    if (top.length > 1) fill = `M${top.join("L")}L${bottom.join("L")}Z`;
  }
  // The last point, for a dot that says "this end is the latest".
  const end = lastI >= 0 ? { x: r(x(lastI)), y: r(y(vals[lastI])) } : null;
  return { line, area, band: fill, end };
}

// Where the card sits relative to what was clicked, in the map container's pixels. Above the anchor when
// it fits, else below; always inside the container with `margin` to spare. `lift` is the marker's height
// above its anchor (a pin reaches up, a gauge dot barely does). The tail's x is relative to the card.
export function placeCard({ ax, ay, cw, ch, W, H, lift = 12, gap = 10, margin = 10, reserveRight = 0 }) {
  const right = W - reserveRight;
  const inside = ax >= 0 && ay >= 0 && ax <= right && ay <= H;
  let left = Math.round(ax - cw / 2);
  left = Math.max(margin, Math.min(left, right - cw - margin));
  let side = "above";
  let top = Math.round(ay - lift - gap - ch);
  if (top < margin) {
    side = "below";
    top = Math.round(ay + gap + 6);
  }
  top = Math.max(margin, Math.min(top, H - ch - margin));
  const tail = Math.max(16, Math.min(cw - 16, Math.round(ax - left)));
  return { left, top, side, tail, inside };
}

export const WHAT = { gauge: "Gauge", reach: "River", place: "Place" };
