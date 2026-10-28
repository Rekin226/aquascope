// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  STORM_START, areaLogLabel, checkRing, hydroChart, ringBounds, stormArgs, stormFacts, stormFeatures, stormHeadline,
  stormLabel, volumeWords,
} from "../src/storm-core.js";
import { logLabel } from "../src/route-core.js";
import { slotOf } from "../src/map-actions-core.js";

const RING = [[-3.75, 52.35], [-3.2, 52.35], [-3.2, 52.65], [-3.75, 52.65]];

// A small answer shaped like aquascope.routing.route_storm's.
const RESULT = {
  ok: true, hours: 185,
  storm: { rain_mm: 50, duration_h: 6, runoff_coefficient: 0.3, n_reaches: 75, area_km2: 1284.6, runoff_m3: 19300000,
    out_m3: 19300000, out_pct: 100, rule: "0.3 of the rain runs off: 15 mm over each unit catchment under the storm." },
  biggest: { river_id: 230463893, peak_cms: 140, peak_h: 15 },
  hydrographs: [
    { role: "biggest", river_id: 230463893, peak_cms: 140, peak_h: 15, dt_h: 1, cms: [0, 40, 140, 90, 30, 0] },
    { role: "outlet", river_id: 230461752, peak_cms: 100, peak_h: 40, dt_h: 1, cms: [0, 0, 20, 60, 100, 50] },
  ],
};

test("an area is checked: three points or more, a closed ring opened, numbers rounded", () => {
  assert.deepEqual(checkRing([...RING, RING[0]]), RING);
  assert.throws(() => checkRing(RING.slice(0, 2)), /3 to 200/);
  assert.throws(() => checkRing([[0, 0], [1, 0], [1, 95]]), /3 to 200/);
  assert.equal(checkRing([[1.123456789, 2], [3, 4], [5, 6]])[0][0], 1.12346);
});

test("the worker's arguments are checked like the package checks them", () => {
  const a = stormArgs({ polygon: RING, ...STORM_START });
  assert.deepEqual(a, { polygon: RING, rain_mm: 50, duration_h: 6, runoff_coefficient: 0.3 });
  assert.throws(() => stormArgs({ polygon: RING, rain_mm: 0, duration_h: 6, runoff_coefficient: 0.3 }), /rain/);
  assert.throws(() => stormArgs({ polygon: RING, rain_mm: 50, duration_h: 6, runoff_coefficient: 1.2 }), /runoff coefficient/);
  assert.throws(() => stormArgs({ polygon: RING, rain_mm: 50, duration_h: 900, runoff_coefficient: 0.3 }), /hours/);
});

test("the area is drawn closed, with its rain as its label", () => {
  const fc = stormFeatures({ polygon: RING, rain_mm: 50, duration_h: 6 }, { live: true });
  const ring = fc.features[0].geometry.coordinates[0];
  assert.equal(ring.length, 5);
  assert.deepEqual(ring[0], ring[4]);
  assert.equal(fc.features[1].properties.label, "50 mm in 6 h");
  assert.deepEqual(fc.features[1].geometry.coordinates, [-3.475, 52.5]);
  assert.equal(fc.features[0].properties.live, 1);
  assert.equal(stormFeatures(null).features.length, 0);
  assert.equal(stormLabel({ rain_mm: 120, duration_h: 48 }), "120 mm in 2 days");
  assert.deepEqual(ringBounds(RING), [[-3.75, 52.35], [-3.2, 52.65]]);
});

test("the chart draws the package's hydrographs on one scale", () => {
  const c = hydroChart(RESULT.hydrographs, { w: 300, h: 96 });
  assert.equal(c.paths.length, 2);
  assert.equal(c.top, 140);
  assert.equal(c.hours, 5);
  assert.match(c.paths[0].d, /^M34 78/);         // zero at the bottom of the plot
  assert.ok(c.paths[0].d.includes(` ${c.yTop}`));  // the biggest peak reaches the top
  assert.equal(hydroChart([]), null);
});

test("the words come from the package's numbers", () => {
  assert.equal(stormHeadline(RESULT),
    "50 mm in 6 h over 1,285 km²: the rivers below carry up to 140 m³/s more, 15 h after the rain starts.");
  const facts = stormFacts(RESULT);
  assert.deepEqual(facts.map((f) => f.label), ["Runs off", "Under the storm", "Runoff", "At the main outlet", "Left by the outlets"]);
  assert.equal(facts[0].value, "0.3 of the rain");
  assert.equal(facts[2].value, "19.3 million m³");
  assert.equal(facts[4].value, "100 % of the runoff in 7 d 17 h");
  assert.equal(volumeWords(840000), "840,000 m³");
  assert.equal(volumeWords(2.5e9), "2.5 km³");
});

test("the area and the run each have their slot in the action log", () => {
  assert.equal(slotOf({ type: "draw_storm", area: "drawn" }, 4), "storm");
  assert.equal(slotOf({ type: "route", river_id: 1, inflow: "series" }, 5), "route");
  assert.equal(logLabel(RESULT), "Route a storm of 50 mm in 6 h");
  assert.equal(areaLogLabel({ polygon: RING }), "Draw the storm area (4 points)");
  assert.equal(areaLogLabel({ polygon: RING }, { moved: true }), "Move the storm area (4 points)");
});
