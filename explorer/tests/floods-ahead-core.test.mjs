// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  DAILY_CODES, FLOOD_CLASSES, FORECAST_DAYS, NONE, addDays, classColor, classOn, countsFor, dayIndex, gaugesFor,
  idsByClass, issueLine, legendLine, lineColorExpr, lineFilterExpr, pointsFor, reachFacts, shortDay,
} from "../src/floods-ahead-core.js";

const reach = (id, rp, daily, gauges = "") => ({
  type: "Feature", geometry: { type: "Point", coordinates: [100, 10] },
  properties: { river_id: id, rp, daily, gauges, peak: 2500, q2: 900, day: "2026-10-12", share: 0.71, order: 7 },
});
const FEATURES = [
  reach(1, 10, "001330000000000", "usgs/A"),
  reach(2, 2, "000001000000000", "usgs/A;uk_ea/B"),
  reach(3, 100, "000000066600000"),
];
const MANIFEST = { issue_date: "2026-10-09", valid_to: "2026-10-23" };

test("the classes are the return periods aquascope.archive.warnings uses, light to dark", () => {
  assert.deepEqual(FLOOD_CLASSES.map((c) => c.rp), [2, 5, 10, 25, 50, 100]);
  assert.deepEqual(DAILY_CODES, [0, 2, 5, 10, 25, 50, 100]);
  // Lightness falls with the class, so the order survives any colour-vision deficiency.
  const lum = (hex) => {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const l = FLOOD_CLASSES.map((c) => lum(c.color));
  for (let i = 1; i < l.length; i++) assert.ok(l[i] < l[i - 1], `${FLOOD_CLASSES[i].label} is darker`);
  assert.equal(classColor(25), "#c2185b");
  assert.equal(classColor(3), NONE);
});

test("the map's date picks the forecast day, and outside the 15 days the peak", () => {
  assert.equal(dayIndex("2026-10-09", "2026-10-09"), 0);
  assert.equal(dayIndex("2026-10-09", "2026-10-23"), FORECAST_DAYS - 1);
  assert.equal(dayIndex("2026-10-09", "2026-10-24"), -1);
  assert.equal(dayIndex("2026-10-09", "2026-10-02"), -1);
  assert.equal(dayIndex(null, "2026-10-09"), -1);
  const p = FEATURES[0].properties;
  assert.equal(classOn(p, -1), 10);
  assert.equal(classOn(p, 2), 2);
  assert.equal(classOn(p, 3), 10);
  assert.equal(classOn(p, 0), 0);
  assert.equal(classOn({ rp: 5, daily: "" }, 4), 0);
});

test("ids by class, highest first, feed the stream tiles' colour and filter", () => {
  const peak = idsByClass(FEATURES, -1);
  assert.deepEqual([...peak.keys()], [100, 50, 25, 10, 5, 2]);
  assert.deepEqual(peak.get(100), [3]);
  assert.deepEqual(peak.get(10), [1]);
  const color = lineColorExpr(peak);
  assert.equal(color[0], "match");
  assert.deepEqual(color.slice(2, 4), [[3], classColor(100)]);
  assert.equal(color.at(-1), NONE);
  assert.deepEqual(lineFilterExpr(peak)[2].sort(), [1, 2, 3]);
  const none = idsByClass([], -1);
  assert.equal(lineColorExpr(none), NONE);
  assert.deepEqual(lineFilterExpr(none), ["boolean", false]);
});

test("on one day only that day's flooded reaches are drawn, and their gauges pulse", () => {
  const day5 = pointsFor(FEATURES, 5);
  assert.deepEqual(day5.features.map((f) => [f.properties.river_id, f.properties.c]), [[2, 2]]);
  assert.deepEqual([...gaugesFor(FEATURES, 5)], [["usgs/A", 2], ["uk_ea/B", 2]]);
  // On the peak view a gauge on two reaches pulses in the bigger class.
  assert.equal(gaugesFor(FEATURES, -1).get("usgs/A"), 10);
  assert.deepEqual([...countsFor(FEATURES, -1)].filter(([, n]) => n), [[2, 1], [10, 1], [100, 1]]);
});

test("the legend says the day, the count and where it is from, and degrades before the first publish", () => {
  assert.equal(legendLine({ missing: true }), "Nothing published yet. It appears after the first daily run.");
  assert.equal(legendLine({ ...MANIFEST, n: 25000, geojson_truncated: true }, -1, 10000),
    "10,000 river reaches (the largest of 25,000) expected to reach the 2-year flow by Fri 23 Oct.");
  assert.equal(legendLine(MANIFEST, -1, 3), "3 river reaches expected to reach the 2-year flow by Fri 23 Oct.");
  assert.equal(legendLine(MANIFEST, 3, 1), "Mon 12 Oct, day 4 of 15: 1 river reach at or above the 2-year flow.");
  assert.equal(issueLine(MANIFEST), "GEOGLOWS forecast of Fri 9 Oct");
  assert.equal(issueLine({ ...MANIFEST, smoke: true }), "GEOGLOWS forecast of Fri 9 Oct (smoke sample)");
  const partial = { ...MANIFEST, chunks: { needed: 3458, read: 3000, failed: 0, skipped_time: 458 } };
  assert.equal(issueLine(partial), "GEOGLOWS forecast of Fri 9 Oct (partial: 86% of rivers read)");
  assert.equal(issueLine({ ...MANIFEST, chunks: { needed: 3458, read: 3458 } }), "GEOGLOWS forecast of Fri 9 Oct");
  assert.equal(issueLine({ ...partial, smoke: true }), "GEOGLOWS forecast of Fri 9 Oct (smoke sample)");
  assert.equal(shortDay("2026-10-12"), "Mon 12 Oct");
  assert.equal(addDays("2026-10-30", 3), "2026-11-02");
});

test("the card's facts are plain sentences with the ratio and the members", () => {
  const f = reachFacts(FEATURES[0].properties, -1, 51);
  assert.equal(f.title, "10-year flow expected");
  assert.equal(f.peak, "Peak 2,500 m³/s on Mon 12 Oct, 2.8 times the 2-year flow (900 m³/s).");
  assert.equal(f.agree, "36 of the 51 forecast members reach the 2-year flow.");
  assert.equal(f.reach, "River reach 1, stream order 7.");
  assert.equal(f.today, "");
  assert.equal(reachFacts(FEATURES[0].properties, 0).today, "On the map's date: below the 2-year flow.");
  for (const text of [...Object.values(f), legendLine(MANIFEST, -1, 3)]) assert.ok(!text.includes("\u2014"));
});
