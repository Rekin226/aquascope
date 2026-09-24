// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  describeFilter, filterFromToolCalls, filterSql, filterWhere, isEmptyFilter, matchLine, normalizeFilter, signaturesUrl,
} from "../src/signature-filter-core.js";

test("normalizeFilter keeps valid fields and drops the rest, like the Python normalize_filter", () => {
  assert.deepEqual(normalizeFilter({ min_years: "50", flood_trend: "Increasing", bfi_min: "", bfi_max: 0.6 }),
    { min_years: 50, flood_trend: "rising", bfi_max: 0.6 });
  assert.deepEqual(normalizeFilter({ min_years: -1, flood_trend: "sideways", bfi_min: 3 }), {});
  assert.deepEqual(normalizeFilter({ bfi_min: 0.7, bfi_max: 0.2 }), { bfi_min: 0.2, bfi_max: 0.7 });
  assert.equal(isEmptyFilter(null), true);
});

test("the WHERE clause only ever carries numbers and the three trend words", () => {
  assert.equal(filterWhere({}), "TRUE");
  assert.equal(filterWhere({ min_years: 40, flood_trend: "rising", bfi_min: 0.3, bfi_max: 0.5 }),
    "data_years >= 40 AND amax_trend = 'rising' AND bfi >= 0.3 AND bfi <= 0.5");
  assert.equal(filterWhere({ flood_trend: "x' OR 1=1 --" }), "TRUE");
  assert.match(filterSql("https://h/x'y/signatures.parquet", { min_years: 10 }),
    /read_parquet\('https:\/\/h\/x''y\/signatures.parquet'\) WHERE data_years >= 10$/);
});

test("the words match describe_filter in Python", () => {
  assert.equal(describeFilter({ min_years: 50, flood_trend: "rising" }), "50+ years of data, rising flood trend");
  assert.equal(describeFilter({ bfi_min: 0.3, bfi_max: 0.5 }), "BFI 0.3 to 0.5");
  assert.equal(describeFilter({ bfi_max: 0.3 }), "BFI up to 0.3");
  assert.equal(describeFilter({}), "no filter");
  assert.equal(matchLine(1, { min_years: 5 }), "1 gauge match");
  assert.equal(matchLine(1234, { min_years: 5 }), "1,234 gauges match");
  assert.equal(matchLine(3, {}), "");
});

test("signatures.parquet sits next to stations.parquet", () => {
  assert.equal(signaturesUrl("https://huggingface.co/datasets/R/g/resolve/main/stations.parquet"),
    "https://huggingface.co/datasets/R/g/resolve/main/signatures.parquet");
});

test("an Ask run's last filter_gauges call becomes the map filter", () => {
  assert.equal(filterFromToolCalls([{ name: "find_stations", arguments: {}, ok: true }]), null);
  assert.equal(filterFromToolCalls(null), null);
  assert.deepEqual(filterFromToolCalls([
    { name: "filter_gauges", arguments: { min_years: 20 }, ok: true },
    { name: "filter_gauges", arguments: { flood_trend: "falling" }, ok: true },
    { name: "filter_gauges", arguments: { min_years: 99 }, ok: false },
  ]), { spec: { flood_trend: "falling" }, question: null });
  assert.deepEqual(filterFromToolCalls([{ name: "filter_gauges", arguments: { question: "50+ years" }, ok: true }]),
    { spec: null, question: "50+ years" });
});
