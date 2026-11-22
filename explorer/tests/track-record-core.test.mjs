// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  MIN_PAIRS, TRUST_BANDS, TRUST_COLORS, indexRows, legendSummary, networkLines, pick, plainRow, rankingRows,
  reachTrust, trackHead, trackLine, trackLines, trustColor, trustLegend, verifyBase, windowSql,
} from "../src/track-record-core.js";

const row = (model, variant, factor, n, extra = {}) => ({
  source: "usgs", station_id: "A", river_id: 7, model, variant, lead_min: 1, lead_max: 3, window_days: 30,
  n_pairs: n, first_valid: "2026-10-19", last_valid: "2026-11-17", enough_pairs: n >= 30, factor,
  trust: n < 30 ? "unknown" : factor <= 2 ? "good" : factor <= 4 ? "fair" : "poor", ...extra,
});

const ROWS = [
  row("geoglows", "raw", 4.8, 84), row("geoglows", "corrected", 1.84, 84), row("glofas", "raw", 2.25, 41),
  row("glofas", "raw", 1.3, 12, { lead_min: 2, lead_max: 2 }),
];

test("verify/ sits in the same dataset as the catalog, and the window query escapes quotes", () => {
  assert.equal(verifyBase("https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet"),
    "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/verify/");
  assert.equal(windowSql("https://x/it's.parquet"),
    "SELECT * FROM read_parquet('https://x/it''s.parquet') WHERE lead_min = 1 AND lead_max = 3");
});

test("the card says each model once, corrected when it can, with the pairs and the dates", () => {
  const b = trackLines(ROWS);
  assert.equal(b.head, "Forecast track record here, last 30 days, 1 to 3 days ahead:");
  assert.deepEqual(b.lines.map((l) => l.text), [
    "GEOGLOWS, corrected: within a factor of 1.8 (84 pairs, 19 Oct to 17 Nov).",
    "GloFAS: within a factor of 2.3 (41 pairs, 19 Oct to 17 Nov).",
  ]);
  assert.deepEqual(b.lines.map((l) => l.trust), ["good", "fair"]);
  assert.match(trackLines(ROWS, { variant: "raw" }).lines[0].text, /^GEOGLOWS: within a factor of 4\.8/);
  assert.deepEqual(pick(ROWS).map((r) => r.variant), ["corrected", "raw"]);
});

test("too few pairs says so; nothing published says nothing", () => {
  assert.equal(trackLine(row("glofas", "raw", 1.2, 12)), "GloFAS: not enough pairs yet (12 of 30).");
  assert.equal(trackLines([]), null);
  assert.equal(trackLines(null), null);
  assert.equal(trackLines([row("glofas", "raw", 1.2, 40, { lead_min: 2, lead_max: 2 })]), null);
  assert.equal(trackLines([row("glofas", "raw", 1.2, 12)]).lines[0].trust, "unknown");
});

test("Floods ahead's badge takes the raw GEOGLOWS record of the reach's best-sampled gauge, else unknown", () => {
  const two = [row("geoglows", "raw", 3, 40, { station_id: "B" }), row("geoglows", "raw", 1.5, 60),
    row("geoglows", "corrected", 1.1, 90)];
  const t = reachTrust(two);
  assert.equal(t.trust, "good");
  assert.match(t.text, /^GEOGLOWS: within a factor of 1\.5 \(60 pairs/);
  assert.equal(t.head, trackHead());
  const none = reachTrust([]);
  assert.equal(none.trust, "unknown");
  assert.match(none.text, /no gauge on this reach/);
});

test("rows index by gauge and by reach; BigInt counts become numbers", () => {
  const { byGauge, byReach } = indexRows([...ROWS, row("glofas", "raw", 2, 50, { station_id: "B", river_id: null })]);
  assert.equal(byGauge.get("usgs/A").length, 4);
  assert.equal(byGauge.get("usgs/B").length, 1);
  assert.equal(byReach.get(7).length, 4);
  assert.deepEqual(plainRow({ n_pairs: 84n, factor: 1.8 }), { n_pairs: 84, factor: 1.8 });
});

test("the legend names the four words and what each means; colour is never alone", () => {
  const key = trustLegend();
  assert.deepEqual(key.map((k) => k.trust), ["good", "fair", "poor", "unknown"]);
  for (const k of key) assert.ok(k.label.startsWith(`${k.trust}: `));
  assert.match(key[3].label, new RegExp(`fewer than ${MIN_PAIRS} pairs`));
  assert.equal(trustColor("nope"), TRUST_COLORS.unknown);
  assert.deepEqual(TRUST_BANDS.map((b) => b.maxFactor), [2, 4, null]);
});

test("the pooled headline and the window read from summary.json", () => {
  const summary = { headline: [{ model: "geoglows", variant: "raw", factor: 4.23, n_pairs: 5240, n_gauges: 767 },
    { model: "glofas", variant: "raw", factor: null, n_pairs: 0, n_gauges: 0 }],
  gauges: { window_days: 30, last_valid: "2026-11-17" } };
  assert.deepEqual(networkLines(summary), ["GEOGLOWS: within a factor of 4.2 (5,240 pairs at 767 gauges)"]);
  assert.equal(legendSummary(summary), "last 30 days, to 17 Nov");
  assert.deepEqual(networkLines(null), []);
});

test("the evidence ladder's ranking reads as list items, ranked first", () => {
  const items = rankingRows({ ranked: [{ model: "geoglows", variant: "corrected", factor: 1.84, n_pairs: 84, trust: "good",
    first_valid: "2026-10-19", last_valid: "2026-11-17" }],
  not_enough: [{ model: "glofas", variant: "raw", factor: 1.2, n_pairs: 12, trust: "unknown" }] });
  assert.deepEqual(items, [
    { trust: "good", text: "GEOGLOWS, corrected: within a factor of 1.8 (84 pairs, 19 Oct to 17 Nov)." },
    { trust: "unknown", text: "GloFAS: not enough pairs yet (12 of 30)." },
  ]);
  assert.equal(rankingRows(null), null);
  assert.equal(rankingRows({ ranked: [], not_enough: [] }), null);
});
