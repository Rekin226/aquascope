// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  RESERVOIR_START, damHydrographs, lakeFeatures, lakeLogLabel, reservoirArgs, reservoirFacts, reservoirHeadline, waveFlood,
  waveWords,
} from "../src/reservoir-core.js";
import { hydroChart } from "../src/storm-core.js";
import { logLabel } from "../src/route-core.js";
import { slotOf } from "../src/map-actions-core.js";

const RING = [[-3.4, 52.505], [-3.34, 52.505], [-3.34, 52.53], [-3.4, 52.53]];
const DAM = { river_id: 230458457, order: 3, lon: -3.348684, lat: 52.506335, downstream_id: 230463896 };

// A small answer shaped like aquascope.sketch.reservoir.route_reservoir's.
const RESULT = {
  ok: true, river_id: 230458457, inflow: { peak_cms: 64.1, duration_h: 24, shape: "triangle", return_period: 100 },
  reservoir: {
    dam: DAM, crest_m: 10, top_m: 15, width_m: 40, coefficient: 1.7049, dt_h: 1,
    inflow_cms: [0, 20, 64.1, 40, 10, 0], outflow_cms: [0, 1, 8, 20, 24.8, 22],
    peak_in_cms: 64.1, peak_out_cms: 24.8, peak_in_h: 2, peak_out_h: 4, reduction_pct: 61.2, delay_h: 10,
    flood_storage_m3: 2510000, max_depth_m: 10.49, overtopped: false, what: "the 100-year flow of this reach over 24 h",
    without: { peak_at_dam_cms: 64.1, furthest_river_id: 230461752, furthest_peak_cms: 37.7, furthest_peak_with_cms: 20.2 },
    flood: { return_period: 100, duration_h: 24 },
  },
};

test("the inputs are checked: the top above the crest, a flood", () => {
  assert.deepEqual(reservoirArgs({ ...RESERVOIR_START, kind: "rp" }),
    { crest_m: 10, top_m: 15, width_m: 30, duration_h: 24, return_period: 100 });
  assert.deepEqual(reservoirArgs({ ...RESERVOIR_START, kind: "peak", peak_cms: 300 }),
    { crest_m: 10, top_m: 15, width_m: 30, duration_h: 24, peak_cms: 300 });
  assert.throws(() => reservoirArgs({ ...RESERVOIR_START, top_m: 8 }), /top of the dam/);
  assert.throws(() => reservoirArgs({ ...RESERVOIR_START, width_m: 0 }), /width/);
  assert.throws(() => reservoirArgs({ ...RESERVOIR_START, kind: "wave", wave: null }), /no flood wave/);
});

test("the wave on the map is the flood it stands for", () => {
  const storm = { result: { ok: true, storm: { rain_mm: 50, duration_h: 6, runoff_coefficient: 0.3 }, polygon: RING } };
  assert.deepEqual(waveFlood(storm), { storm: { polygon: RING, rain_mm: 50, duration_h: 6, runoff_coefficient: 0.3 } });
  const flood = { result: { ok: true, river_id: 230442136, inflow: { peak_cms: 500, duration_h: 12, shape: "triangle", return_period: null } } };
  assert.deepEqual(waveFlood(flood), { from_river_id: 230442136, duration_h: 12, peak_cms: 500 });
  const replay = { result: { ok: true, river_id: 1, inflow: { peak_cms: 400, duration_h: 90, shape: "series" } } };
  assert.equal(waveFlood(replay), null);
  assert.deepEqual(waveFlood({ result: RESULT }), { return_period: 100, duration_h: 24 });
  assert.deepEqual(reservoirArgs({ ...RESERVOIR_START, kind: "wave", wave: flood }),
    { crest_m: 10, top_m: 15, width_m: 30, from_river_id: 230442136, duration_h: 12, peak_cms: 500 });
  assert.equal(waveWords(flood), "500 m³/s routed on the map, from reach 230442136");
  assert.equal(waveWords(storm), "the storm on the map (50 mm in 6 h)");
  assert.equal(waveWords({ result: RESULT }), "the 100-year flood at the dam, as before");
});

test("the lake and the dam are drawn", () => {
  const fc = lakeFeatures({ polygon: RING, dam: DAM }, { live: true });
  assert.equal(fc.features.length, 2);
  assert.equal(fc.features[0].geometry.coordinates[0].length, 5);
  assert.equal(fc.features[0].properties.live, 1);
  assert.deepEqual(fc.features[1].geometry.coordinates, [DAM.lon, DAM.lat]);
  assert.equal(lakeFeatures(null).features.length, 0);
});

test("the card says the package's numbers", () => {
  assert.equal(reservoirHeadline(RESULT), "The reservoir takes the flood's peak at the dam from 64 to 25 m³/s, 61.2 % lower and 10 h later.");
  const facts = reservoirFacts(RESULT);
  assert.deepEqual(facts.map((f) => f.label), ["The flood", "Highest water", "Held at most", "The dam", "Furthest down"]);
  assert.equal(facts[2].value, "2.5 million m³ above the crest");
  assert.equal(facts[4].value, "20 m³/s, without it 38 m³/s");
  const c = hydroChart(damHydrographs(RESULT));
  assert.deepEqual(c.paths.map((p) => p.role), ["inflow", "outflow"]);
  assert.equal(c.top, 64.1);
});

test("the lake and the run each have their slot in the log", () => {
  assert.equal(slotOf({ type: "draw_reservoir", area: "drawn" }, 3), "reservoir");
  assert.equal(logLabel(RESULT), "A flood through a reservoir, 61.2 % lower past the dam");
  assert.equal(lakeLogLabel({ polygon: RING, dam: DAM }), "Draw a reservoir on reach 230458457");
});
