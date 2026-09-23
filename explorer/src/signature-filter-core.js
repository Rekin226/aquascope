// The signature filter's pure half: the filter spec, the SQL it becomes and the
// words it is described in. The rules live in aquascope.archive.signatures
// (normalize_filter, filter_signatures, describe_filter); this is a thin mirror
// of them so the map can filter in DuckDB-WASM without waking Python. No DOM;
// node-importable (explorer/tests/signature-filter.test.mjs).

export const TRENDS = ["rising", "falling", "none"];

const num = (x) => {
  if (x === null || x === undefined || x === "") return null;
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
};

const TREND_ALIASES = { increasing: "rising", up: "rising", decreasing: "falling", down: "falling", "no trend": "none", stationary: "none" };

/** Keep the known fields with valid values, as normalize_filter does. */
export function normalizeFilter(spec) {
  const s = spec || {};
  const out = {};
  const y = num(s.min_years);
  if (y !== null && y > 0) out.min_years = Math.round(y * 10) / 10;
  let t = String(s.flood_trend ?? "").trim().toLowerCase();
  t = TREND_ALIASES[t] || t;
  if (TRENDS.includes(t)) out.flood_trend = t;
  const lo = num(s.bfi_min);
  const hi = num(s.bfi_max);
  if (lo !== null && lo > 0 && lo <= 1) out.bfi_min = Math.round(lo * 1000) / 1000;
  if (hi !== null && hi >= 0 && hi < 1) out.bfi_max = Math.round(hi * 1000) / 1000;
  if (out.bfi_min !== undefined && out.bfi_max !== undefined && out.bfi_min > out.bfi_max) {
    [out.bfi_min, out.bfi_max] = [out.bfi_max, out.bfi_min];
  }
  return out;
}

export const isEmptyFilter = (spec) => Object.keys(normalizeFilter(spec)).length === 0;

/** The WHERE clause for signatures.parquet. Values are numbers or one of TRENDS, so nothing is interpolated raw. */
export function filterWhere(spec) {
  const s = normalizeFilter(spec);
  const parts = [];
  if (s.min_years !== undefined) parts.push(`data_years >= ${s.min_years}`);
  if (s.flood_trend !== undefined) parts.push(`amax_trend = '${s.flood_trend}'`);
  if (s.bfi_min !== undefined) parts.push(`bfi >= ${s.bfi_min}`);
  if (s.bfi_max !== undefined) parts.push(`bfi <= ${s.bfi_max}`);
  return parts.length ? parts.join(" AND ") : "TRUE";
}

export function filterSql(url, spec) {
  const safe = String(url).replace(/'/g, "''");
  return `SELECT source, station_id FROM read_parquet('${safe}') WHERE ${filterWhere(spec)}`;
}

/** The same words as describe_filter in Python. */
export function describeFilter(spec) {
  const s = normalizeFilter(spec);
  const parts = [];
  if (s.min_years !== undefined) parts.push(`${s.min_years}+ years of data`);
  if (s.flood_trend !== undefined) {
    parts.push({ rising: "rising flood trend", falling: "falling flood trend", none: "no flood trend" }[s.flood_trend]);
  }
  if (s.bfi_min !== undefined && s.bfi_max !== undefined) parts.push(`BFI ${s.bfi_min} to ${s.bfi_max}`);
  else if (s.bfi_min !== undefined) parts.push(`BFI ${s.bfi_min} and up`);
  else if (s.bfi_max !== undefined) parts.push(`BFI up to ${s.bfi_max}`);
  return parts.join(", ") || "no filter";
}

export function matchLine(nMatch, spec) {
  if (isEmptyFilter(spec)) return "";
  const n = Number(nMatch) || 0;
  return `${n.toLocaleString("en-US")} gauge${n === 1 ? "" : "s"} match`;
}

/** signatures.parquet sits next to stations.parquet in the dataset. */
export function signaturesUrl(stationsUrl) {
  return String(stationsUrl).replace(/stations\.parquet(\?.*)?$/, "signatures.parquet");
}

/**
 * The filter an Ask run asked for: the last successful filter_gauges call.
 * Returns {spec} for structured arguments, {question} when only words were
 * given (Python parses those), or null when the run never filtered.
 */
export function filterFromToolCalls(toolCalls) {
  const calls = (toolCalls || []).filter((c) => c && c.name === "filter_gauges" && c.ok !== false);
  if (!calls.length) return null;
  const args = calls[calls.length - 1].arguments || {};
  const spec = normalizeFilter(args);
  if (!isEmptyFilter(spec)) return { spec, question: args.question || null };
  if (args.question) return { spec: null, question: String(args.question) };
  return null;
}
