// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  COVER_COLORS, START, USES, areaFeatures, areaLabel, checkRing, compareRows, coverBar, flow, headline, landuseArgs,
  logLabel, pctWords, ringBounds, routeArgs, soilWords, useShort, useTone,
} from "../src/landuse-core.js";
import { slotOf } from "../src/map-actions-core.js";

const FIELD = [[4.955, 45.7], [4.97, 45.7], [4.97, 45.708], [4.955, 45.708]];
// A small answer shaped like aquascope.sketch.landuse.landuse_change's.
const RES = {
  ok: true, area_km2: 1.04, use: "residential_1_4", rain_mm: 100, distribution: "II", soils: { C: 0.8, D: 0.2 },
  cover: [{ code: 40, label: "Cropland", share: 0.6, cn: 85 }, { code: 10, label: "Tree cover", share: 0.3, cn: 70 },
    { code: 80, label: "Permanent water bodies", share: 0.1, cn: null }],
  before: { cn: 80.0, q_mm: 47.6, peak_cms: 6.421, tc_h: 1.12 }, after: { cn: 83.0, q_mm: 55.0, peak_cms: 9.84, tc_h: 1.03 },
  change: { peak_cms: 3.419, peak_pct: 53.2, volume_m3: 7700 }, route: { peak_cms: 3.419, duration_h: 1.25 },
  reach: { river_id: 210139527, lat: 45.7214, lon: 4.9832, from: [4.9622, 45.708] },
};

test("the inputs, checked", () => {
  const a = landuseArgs({ polygon: [...FIELD, FIELD[0]], use: "paved", rain_mm: "80", distribution: "IA", soil_group: "B" });
  assert.deepEqual(a, { polygon: FIELD, use: "paved", rain_mm: 80, distribution: "IA", soil_group: "B" });
  assert.equal(landuseArgs({ polygon: FIELD, use: "impervious", rain_mm: 50, impervious_pct: 40 }).impervious_pct, 40);
  assert.equal(landuseArgs({ polygon: FIELD, use: "paved", rain_mm: 50, distribution: "X" }).distribution, "II");
  assert.throws(() => landuseArgs({ polygon: FIELD, use: "castle", rain_mm: 50 }));
  assert.throws(() => landuseArgs({ polygon: FIELD, use: "paved", rain_mm: 0 }));
  assert.throws(() => landuseArgs({ polygon: FIELD, use: "impervious", rain_mm: 50, impervious_pct: "" }));
  assert.throws(() => landuseArgs({ polygon: FIELD, use: "paved", rain_mm: 50, soil_group: "E" }));
  assert.throws(() => checkRing(FIELD.slice(0, 2)));
  assert.equal(START.rain_mm, 100);
  assert.equal(USES.length, 14);
});

test("the words", () => {
  assert.equal(useShort("commercial"), "Commercial and business");
  assert.equal(pctWords(53.2), "+53%");
  assert.equal(pctWords(-3.4), "-3%");
  assert.equal(flow(9.84), "9.84");
  assert.equal(flow(0.4211), "0.421");
  assert.equal(flow(1250.4), "1,250");
  assert.equal(areaLabel(RES), "Residential, 1/4 acre lots\n+53% peak");
  assert.equal(headline(RES), "1.04 km² as residential, 1/4 acre lots: runoff 48 to 55 mm from 100 mm of rain, " +
    "the area's peak 6.42 to 9.84 m³/s (+53%).");
  assert.deepEqual(compareRows(RES)[0], ["Curve number", "80", "83"]);
  assert.deepEqual(compareRows(RES)[3], ["Peak, m³/s", "6.42", "9.84"]);
  assert.equal(soilWords(RES.soils), "C 80%, D 20%");
  assert.equal(logLabel({ polygon: FIELD, result: RES }), "Residential, 1/4 acre lots on 1.0 km²: peak +53%");
  assert.equal(logLabel({ polygon: FIELD }), "Draw an area for a land-use change");
  assert.equal(logLabel(null), "Take the land-use change off");
  assert.deepEqual(routeArgs(RES), { peak_cms: 3.419, duration_h: 1.25, shape: "triangle" });
  assert.equal(routeArgs({ ...RES, route: null }), null);
  assert.equal(slotOf({ type: "change_landuse" }), "landuse");
});

test("the cover bar and the map", () => {
  const bar = coverBar(RES.cover);
  assert.deepEqual(bar.map((b) => b.color), [COVER_COLORS[40], COVER_COLORS[10], COVER_COLORS[80]]);
  const fc = areaFeatures({ polygon: FIELD, result: RES });
  assert.deepEqual(fc.features.map((f) => f.properties.kind), ["area", "path", "outlet"]);
  assert.equal(fc.features[0].properties.color, useTone("residential_1_4"));
  assert.deepEqual(fc.features[0].geometry.coordinates[0].at(-1), FIELD[0]);
  assert.deepEqual(fc.features[1].geometry.coordinates, [RES.reach.from, [RES.reach.lon, RES.reach.lat]]);
  assert.equal(areaFeatures(null).features.length, 0);
  assert.equal(areaFeatures({ polygon: FIELD }).features.length, 1);
  assert.notEqual(useTone("woods_good"), useTone("paved"));
  const box = ringBounds(FIELD, RES);
  assert.deepEqual(box, [[4.955, 45.7], [4.9832, 45.7214]]);
});
