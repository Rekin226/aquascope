// The signature filter bar in the rail: "years of data", "flood trend" and a
// baseflow index range, applied to the gauge layer from signatures.parquet
// (built weekly by `aquascope harvest signatures`) through DuckDB-WASM. When the
// dataset has no signatures.parquet yet, the bar stays hidden.
//
// The Ask drawer and an in-browser agent (WebMCP) set it through
// actions.setSignatureFilter / actions.setSignatureFilterFromQuestion; words are
// read by aquascope.archive.signatures.parse_filter_question in the worker, so
// the page and the Python tools agree on what "50+ years, rising floods" means.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, state, trace } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import {
  describeFilter, filterFromToolCalls, filterSql, isEmptyFilter, matchLine, normalizeFilter, signaturesUrl,
} from "./signature-filter-core.js?v=__BUILD__";

const URL_ = CONFIG.signaturesParquet || signaturesUrl(CONFIG.stationsParquet);
let available = null;       // null = not probed, then true / false
let current = {};
let runId = 0;

// The rendered form, or null before the probe has found the table.
function body() {
  const f = $("sigfilter")?.querySelector(".sigfilter-body");
  return f && f.querySelector('[name="min_years"]') ? f : null;
}

function formHtml() {
  return `
    <label class="rail-field">Years of data, at least
      <input type="number" name="min_years" min="0" max="200" step="5" placeholder="any" inputmode="numeric" />
    </label>
    <label class="rail-field">Flood trend (annual maxima)
      <select name="flood_trend">
        <option value="">any</option><option value="rising">rising</option>
        <option value="falling">falling</option><option value="none">no trend</option>
      </select>
    </label>
    <div class="rail-field">Baseflow index
      <span class="sigfilter-range">
        <input type="number" name="bfi_min" min="0" max="1" step="0.05" placeholder="0" aria-label="Baseflow index from" />
        <span>to</span>
        <input type="number" name="bfi_max" min="0" max="1" step="0.05" placeholder="1" aria-label="Baseflow index up to" />
      </span>
    </div>
    <div class="sigfilter-foot">
      <span class="sigfilter-count muted" aria-live="polite"></span>
      <button type="button" class="btn small sigfilter-clear" hidden>Clear</button>
    </div>
    <p class="rail-note muted">From the mirrored daily discharge: Mann-Kendall on annual maxima, Lyne-Hollick baseflow.</p>`;
}

function readForm() {
  const f = body();
  const v = (name) => f.querySelector(`[name="${name}"]`).value;
  return normalizeFilter({ min_years: v("min_years"), flood_trend: v("flood_trend"), bfi_min: v("bfi_min"), bfi_max: v("bfi_max") });
}

function writeForm(spec) {
  const f = body();
  if (!f) return;
  for (const name of ["min_years", "flood_trend", "bfi_min", "bfi_max"]) {
    f.querySelector(`[name="${name}"]`).value = spec[name] ?? "";
  }
}

function showCount(n, spec) {
  const f = body();
  if (!f) return;
  f.querySelector(".sigfilter-count").textContent = matchLine(n, spec);
  f.querySelector(".sigfilter-clear").hidden = isEmptyFilter(spec);
}

/**
 * Filter the gauge layer to the stations whose signatures meet ``spec``.
 * Returns {filter, description, n_match} (or {error}).
 */
export async function applySignatureFilter(spec) {
  const clean = normalizeFilter(spec);
  current = clean;
  const mine = ++runId;
  if (available === false) return { error: "This dataset has no signatures table yet.", filter: clean };
  writeForm(clean);
  if (isEmptyFilter(clean)) {
    state.sigMatch = null;
    actions.refreshMapData();
    showCount(0, clean);
    return { filter: clean, description: describeFilter(clean), n_match: null };
  }
  try {
    const { conn } = await duck();
    const table = await conn.query(filterSql(URL_, clean));
    if (mine !== runId) return { filter: clean, superseded: true };
    const keys = new Set(table.toArray().map((r) => `${r.source}/${r.station_id}`));
    state.sigMatch = keys;
    actions.refreshMapData();
    showCount(keys.size, clean);
    announce(`${matchLine(keys.size, clean)}: ${describeFilter(clean)}`);
    return { filter: clean, description: describeFilter(clean), n_match: keys.size };
  } catch (err) {
    trace(`signature filter failed: ${err && err.message}`);
    return { error: `The filter could not run: ${err && err.message}`, filter: clean };
  }
}

/** Words to a filter via the same Python rules the MCP tool uses, then onto the map. */
export async function setFilterFromQuestion(question, fields = {}) {
  let spec = normalizeFilter(fields);
  try {
    const res = await call("tool", { name: "filter_gauges", arguments: { question, ...spec, spec_only: true } });
    if (res && res.filter) spec = res.filter;
    if (res && res.note && isEmptyFilter(spec)) return { error: res.note, filter: spec };
  } catch (err) {
    if (isEmptyFilter(spec)) return { error: `Could not read a filter from the question: ${err && err.message}` };
  }
  if ($("sigfilter")) $("sigfilter").open = true;
  return applySignatureFilter(spec);
}

/** After an Ask run: if the analyst called filter_gauges, show that filter on the map too. */
export function applyFromAskResult(res) {
  const found = filterFromToolCalls(res && res.tool_calls);
  if (!found || available === false) return;
  if (found.question) setFilterFromQuestion(found.question, found.spec || {});
  else applySignatureFilter(found.spec);
}

async function probe() {
  try {
    const { conn } = await duck();
    const t = await conn.query(`SELECT count(*) AS n FROM read_parquet('${URL_.replace(/'/g, "''")}')`);
    const n = Number(t.toArray()[0].n);
    return n > 0;
  } catch (err) {
    trace(`signatures.parquet not available: ${err && err.message}`);
    return false;
  }
}

export async function initSignatureFilter() {
  actions.setSignatureFilter = applySignatureFilter;
  actions.setSignatureFilterFromQuestion = setFilterFromQuestion;
  actions.applyAskFilter = applyFromAskResult;
  const box = $("sigfilter");
  if (!box) return false;
  available = await probe();
  if (!available) { box.hidden = true; return false; }
  const f = box.querySelector(".sigfilter-body");
  f.innerHTML = formHtml();
  let t;
  f.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => applySignatureFilter(readForm()), 300);
  });
  f.querySelector(".sigfilter-clear").addEventListener("click", () => applySignatureFilter({}));
  box.hidden = false;
  if (!isEmptyFilter(current)) applySignatureFilter(current);
  trace(`signature filter ready (${URL_})`);
  return true;
}
