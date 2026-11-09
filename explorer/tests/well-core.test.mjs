// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULTS, LEVEL_COLORS, SPANS, addWell, contourFeatures, corners, daysWords, distWords, editWell, fillPixels,
  fromSlider, headline, levelColor, logLabel, mapArgs, metres, modflowCellAt, modflowWords, peacemanRadius, rateWords,
  sWords, toSlider, wellsValue,
} from "../src/well-core.js";
import { slotOf } from "../src/map-actions-core.js";

test("the words", () => {
  assert.equal(rateWords(1000), "1,000 m³/day (11.6 L/s)");
  assert.equal(rateWords(1728), "1,728 m³/day (20.0 L/s)");
  assert.equal(metres(3.7291), "3.73");
  assert.equal(metres(12.44), "12.4");
  assert.equal(distWords(850), "850 m");
  assert.equal(distWords(21822), "21.8 km");
  assert.equal(daysWords(30), "30 days");
  assert.equal(daysWords(365), "1 year");
  assert.equal(daysWords(0.25), "6 hours");
  assert.equal(sWords(1e-4), "1e-4");
  assert.equal(sWords(2.5e-5), "2.5e-5");
  assert.equal(sWords(0.0051), "0.0051");
});

test("the sliders round-trip the defaults", () => {
  assert.equal(fromSlider(toSlider(DEFAULTS.T, SPANS.T), SPANS.T), 500);
  assert.equal(fromSlider(toSlider(DEFAULTS.S, SPANS.S), SPANS.S), 1e-4);
  assert.equal(fromSlider(toSlider(DEFAULTS.days, SPANS.days), SPANS.days), 30);
  assert.equal(fromSlider(0, SPANS.T), 1);
  assert.equal(fromSlider(1000, SPANS.T), 100000);
});

test("the wells as one undoable value", () => {
  let v = addWell(null, { lon: 2.35, lat: 48.85 });
  assert.deepEqual(v, { wells: [{ id: 1, lon: 2.35, lat: 48.85, rate: 1000 }], T: 500, S: 1e-4, days: 30 });
  v = addWell(v, { lon: 2.37, lat: 48.86, rate: 3000 });
  assert.equal(v.wells[1].id, 2);
  const moved = editWell(v, 2, { lon: 2.38, lat: 48.87 });
  assert.equal(logLabel(v, moved), "Move well 2");
  assert.equal(logLabel(v, editWell(v, 1, { rate: 500 })), "Well 1 pumps 500 m³/day (5.8 L/s)");
  assert.equal(logLabel(v, editWell(v, 1, null)), "Remove a well (1 left)");
  assert.equal(logLabel(null, addWell(null, { lon: 0, lat: 0 })), "Drop well 1, 1,000 m³/day (11.6 L/s)");
  assert.equal(logLabel(v, null), "Take the wells off the map");
  assert.equal(logLabel(v, wellsValue({ ...v, days: 365 })), "Read the cone after 1 year");
  // ids stay with their wells after one is removed, so a new one does not take a number in use
  const after = addWell(editWell(v, 1, null), { lon: 1, lat: 1 });
  assert.deepEqual(after.wells.map((w) => w.id), [2, 3]);
  assert.throws(() => wellsValue({ wells: [{ lon: 0, lat: 95 }] }));
  assert.throws(() => wellsValue({ wells: [{ lon: 0, lat: 0, rate: -1 }] }));
  assert.throws(() => wellsValue({ wells: Array.from({ length: 11 }, () => ({ lon: 0, lat: 0 })) }));
  assert.equal(slotOf({ type: "drop_well" }), "wells");
});

test("the worker's arguments are the package's", () => {
  const v = addWell(addWell(null, { lon: 1, lat: 2, rate: 800 }), { lon: 3, lat: 4 });
  assert.deepEqual(mapArgs(v, { frames: true }), {
    wells: [{ lon: 1, lat: 2, rate_m3_day: 800, name: "Well 1" }, { lon: 3, lat: 4, rate_m3_day: 1000, name: "Well 2" }],
    days: 30, transmissivity: 500, storativity: 1e-4, frames: true,
  });
});

test("the cone on the map", () => {
  const fc = contourFeatures([{ level: 0.05, label: "0.05 m", lines: [[[0, 0], [1, 1]]] },
    { level: 1, label: "1 m", lines: [[[0, 0], [1, 0]]] }]);
  assert.equal(fc.features.length, 2);
  assert.equal(fc.features[0].properties.edge, 1);
  assert.equal(fc.features[1].properties.edge, 0);
  assert.equal(fc.features[1].properties.color, levelColor(1));
  assert.equal(levelColor(0.07), LEVEL_COLORS[0][1]);
  assert.equal(levelColor(0.5), LEVEL_COLORS[3][1]);
  const { px, width, height } = fillPixels({ nx: 2, ny: 1, values: [0.01, 2.0] });
  assert.equal(width * height * 4, px.length);
  assert.equal(px[3], 0);          // under 5 cm: clear
  assert.ok(px[7] > 100);          // 2 m: well filled
  assert.deepEqual(corners([0, 1, 2, 3]), [[0, 3], [2, 3], [2, 1], [0, 1]]);
  assert.match(headline({ days: 30, reach_m: 21822, wells: [{ name: "Well 1", at_well_m: 3.729 }] }),
    /^After 30 days, the well draws the water down 3\.73 m at its face; the 5 cm line reaches 21\.8 km out\.$/);
});

test("a MODFLOW cell under a well, and Peaceman's radius", () => {
  // two square cells of 0.01 degree at the equator, side by side
  const rings = [0, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0.01, 0.01, 0.02, 0.01, 0.02, 0, 0.01, 0];
  const offsets = [0, 4, 8];
  const c = modflowCellAt(rings, offsets, 0.015, 0.005);
  assert.equal(c.cell, 1);
  assert.ok(Math.abs(c.dx_m - 1111.95) < 1 && Math.abs(c.dy_m - 1111.95) < 1);
  assert.equal(modflowCellAt(rings, offsets, 0.05, 0.005), null);
  assert.ok(Math.abs(peacemanRadius(c) - 222.4) < 0.5);
  const words = modflowWords({ name: "m.hds", layer: 1, layers: [1, 2, 3] }, c, 0.84);
  assert.match(words, /cell 2, 1\.1 km by 1\.1 km, layer 1 of 3\. Its well cell would read about 0\.84 m, the drawdown 222 m from the well/);
});
