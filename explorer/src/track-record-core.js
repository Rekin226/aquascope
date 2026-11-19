// The forecast track record (#785), the pure part: no DOM, so node can test it.
// Every number comes from the Archive's daily verification (verify/, written by
// aquascope.archive.verify): each gauge's own record over the newest 30 days in
// verify/forecast_skill_gauges.parquet, every gauge pooled in summary.json. The
// trust word (good, fair, poor, unknown) is in the table too. This module only
// picks the rows and says them, in the words of aquascope.forecast_skill
// (tests/test_explorer/test_track_record.py holds the two together).

// The bands of aquascope.archive.verify.TRUST_BANDS: half the forecasts were within this factor of the gauge.
export const TRUST_BANDS = [
  { trust: "good", maxFactor: 2, words: "within a factor of 2 of the gauge" },
  { trust: "fair", maxFactor: 4, words: "within a factor of 2 to 4" },
  { trust: "poor", maxFactor: null, words: "off by more than a factor of 4" },
];
export const UNKNOWN = "unknown";
export const MIN_PAIRS = 30;
export const TRACK_DAYS = 30;
export const TRACK_LEADS = [1, 3];
// Ordinal and colour-blind safe (the evidence ladder's ramp), and always with the word, so colour is never the
// only encoding.
export const TRUST_COLORS = { good: "#1a9850", fair: "#fdae61", poor: "#d73027", unknown: "#b0bec5" };
export const LABELS = { geoglows: "GEOGLOWS", glofas: "GloFAS" };
const MODEL_ORDER = ["geoglows", "glofas"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export const trustColor = (t) => TRUST_COLORS[t] || TRUST_COLORS.unknown;

/** verify/ sits in the same dataset as stations.parquet. */
export function verifyBase(stationsUrl) {
  return String(stationsUrl).replace(/stations\.parquet(\?.*)?$/, "verify/");
}

const sqlString = (s) => `'${String(s).replace(/'/g, "''")}'`;

/** The card's lead window for every gauge: a couple of thousand rows, read once. */
export function windowSql(url, leads = TRACK_LEADS) {
  return `SELECT * FROM read_parquet(${sqlString(url)}) WHERE lead_min = ${Number(leads[0])} AND lead_max = ${Number(leads[1])}`;
}

const num = (x) => (x === null || x === undefined || x === "" || !Number.isFinite(Number(x)) ? null : Number(x));

/** DuckDB rows to plain JSON: BigInt counts become numbers, dates become ISO days. */
export function plainRow(row) {
  const out = {};
  for (const [k, v] of Object.entries(row || {})) {
    out[k] = typeof v === "bigint" ? Number(v) : v;
  }
  return out;
}

/** {byGauge: Map "source/station_id" -> rows, byReach: Map river_id -> rows}. */
export function indexRows(rows) {
  const byGauge = new Map();
  const byReach = new Map();
  for (const r of rows || []) {
    if (!r) continue;
    const key = `${r.source}/${r.station_id}`;
    if (!byGauge.has(key)) byGauge.set(key, []);
    byGauge.get(key).push(r);
    const rid = num(r.river_id);
    if (rid !== null) {
      if (!byReach.has(rid)) byReach.set(rid, []);
      byReach.get(rid).push(r);
    }
  }
  return { byGauge, byReach };
}

function shortDay(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ""));
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}` : "";
}

// One decimal, rounded half up the same way in Python and here, so the two never disagree on a tie.
const oneDecimal = (x) => (Math.floor(Number(x) * 10 + 0.5) / 10).toFixed(1);

const label = (r) => {
  const name = LABELS[r.model] || String(r.model || "model");
  return r.variant === "corrected" ? `${name}, corrected` : name;
};

function leadWords(lo, hi) {
  lo = Number(lo); hi = Number(hi);
  if (lo === hi) return lo === 0 ? "the same day" : `${lo} day${lo === 1 ? "" : "s"} ahead`;
  return `${lo} to ${hi} days ahead`;
}

/** "Forecast track record here, last 30 days, 1 to 3 days ahead:" */
export function trackHead(windowDays = TRACK_DAYS, leads = TRACK_LEADS) {
  return `Forecast track record here, last ${Number(windowDays)} days, ${leadWords(leads[0], leads[1])}:`;
}

/** One model's line, as aquascope.forecast_skill.track_line says it. */
export function trackLine(r, { minPairs = MIN_PAIRS } = {}) {
  const n = Math.trunc(num(r.n_pairs) || 0);
  const factor = num(r.factor);
  if (!r.enough_pairs || factor === null) return `${label(r)}: not enough pairs yet (${n} of ${Math.trunc(minPairs)}).`;
  const span = r.first_valid && r.last_valid ? `, ${shortDay(r.first_valid)} to ${shortDay(r.last_valid)}` : "";
  return `${label(r)}: within a factor of ${oneDecimal(factor)} (${n} pair${n === 1 ? "" : "s"}${span}).`;
}

const inWindow = (r, leads) => num(r.lead_min) === leads[0] && num(r.lead_max) === leads[1];

/** One row per model for the card: the lead window, the asked variant when the model has it, else raw. */
export function pick(rows, { variant = "corrected", leads = TRACK_LEADS } = {}) {
  const out = [];
  for (const m of MODEL_ORDER) {
    const mine = (rows || []).filter((r) => r && r.model === m && inWindow(r, leads));
    const hit = mine.find((r) => r.variant === variant) || mine.find((r) => r.variant === "raw");
    if (hit) out.push(hit);
  }
  return out;
}

const trustOf = (r) => (r.enough_pairs ? String(r.trust || UNKNOWN) : UNKNOWN);

/** The card's block, as aquascope.forecast_skill.track_lines builds it, or null (the card then shows nothing). */
export function trackLines(rows, { variant = "corrected", leads = TRACK_LEADS, minPairs = MIN_PAIRS } = {}) {
  const chosen = pick(rows, { variant, leads });
  if (!chosen.length) return null;
  const days = Math.trunc(num(chosen[0].window_days) || TRACK_DAYS);
  return {
    head: trackHead(days, leads),
    lines: chosen.map((r) => ({ model: r.model, variant: r.variant, text: trackLine(r, { minPairs }), trust: trustOf(r) })),
  };
}

/**
 * Floods ahead's badge for one reach: the raw GEOGLOWS record of the gauge on it with the most pairs (Floods
 * ahead classes the raw forecast), as {head, trust, text}. With no gauge on the reach the trust is unknown.
 */
export function reachTrust(rows, { leads = TRACK_LEADS, minPairs = MIN_PAIRS } = {}) {
  const mine = (rows || []).filter((r) => r && r.model === "geoglows" && r.variant === "raw" && inWindow(r, leads))
    .sort((a, b) => (num(b.n_pairs) || 0) - (num(a.n_pairs) || 0));
  const head = trackHead(TRACK_DAYS, leads);
  if (!mine.length) return { head, trust: UNKNOWN, text: "GEOGLOWS: no gauge on this reach has a verified forecast yet." };
  return { head: trackHead(Math.trunc(num(mine[0].window_days) || TRACK_DAYS), leads), trust: trustOf(mine[0]),
    text: trackLine(mine[0], { minPairs }) };
}

/**
 * The evidence ladder's ranking (aquascope.forecast_skill.rank_models, the `forecasts` key of model_skill) as
 * list items {trust, text}: the ranked models first, then those without enough pairs; null when there is none.
 */
export function rankingRows(f) {
  if (!f) return null;
  const rows = [...(f.ranked || []).map((x) => ({ ...x, enough_pairs: true })),
    ...(f.not_enough || []).map((x) => ({ ...x, enough_pairs: false }))];
  if (!rows.length) return null;
  return rows.map((x) => ({ trust: x.enough_pairs ? String(x.trust || UNKNOWN) : UNKNOWN, text: trackLine(x) }));
}

/** The legend's key: the four words with what each means. */
export function trustLegend() {
  return [...TRUST_BANDS.map((b) => ({ trust: b.trust, color: trustColor(b.trust), label: `${b.trust}: ${b.words}` })),
    { trust: UNKNOWN, color: trustColor(UNKNOWN), label: `unknown: fewer than ${MIN_PAIRS} pairs, or no gauge here` }];
}

/** Every gauge pooled, one line per model and variant at the summary's headline lead (lead 1). */
export function networkLines(summary) {
  const head = (summary && summary.headline) || [];
  return head.filter((r) => r && num(r.factor) !== null).map((r) => {
    const n = Math.trunc(num(r.n_pairs) || 0);
    return `${label(r)}: within a factor of ${oneDecimal(r.factor)} (${n.toLocaleString("en-GB")} pairs at ` +
      `${Math.trunc(num(r.n_gauges) || 0).toLocaleString("en-GB")} gauges)`;
  });
}

/** The legend row's few words: the window it covers, or why it is empty. */
export function legendSummary(summary) {
  const g = summary && summary.gauges;
  if (!g) return "every gauge pooled";
  return `last ${Number(g.window_days || TRACK_DAYS)} days, to ${shortDay(g.last_valid)}`;
}
