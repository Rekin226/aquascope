// The flood views of a model result (#619, #620): the pure part (colours, pictures, the card's words).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  HAZARD_COLOURS, HAZARD_SHORT, LENSES, areaWords, hazardPacked, hoursWords, lensFacts, lensLine, lensScale, lensTicks,
  lensesFor, limitWords, pack, packedRamp, paintGrid, ARRIVAL_STOPS,
} from "../src/mesh-views-core.js";

const CLASSES = [
  { code: "H1", dv: 0.3, depth: 0.3, velocity: 2, words: "Generally safe for vehicles, people and buildings." },
  { code: "H2", dv: 0.6, depth: 0.5, velocity: 2, words: "Unsafe for small vehicles." },
  { code: "H3", dv: 0.6, depth: 1.2, velocity: 2, words: "Unsafe for vehicles, children and the elderly." },
  { code: "H4", dv: 1, depth: 2, velocity: 2, words: "Unsafe for vehicles and people." },
  { code: "H5", dv: 4, depth: 4, velocity: 4, words: "Unsafe for vehicles and people. All building types vulnerable to structural damage." },
  { code: "H6", dv: null, depth: null, velocity: null, words: "Unsafe for vehicles and people. All building types considered vulnerable to failure." },
];
const META = {
  available: ["depth", "hazard", "max_depth", "max_speed", "arrival"],
  hazard: { classes: CLASSES, from: "step", rule: "the highest class reached at any output step",
    source: { short: "AIDR Guideline 7-3 (2014), Tables 1 and 2" } },
  arrival: { depth_m: 0.1, rule: "the first output step with at least 0.1 m of water", last_h: 16 },
  units: { length: "m", velocity: "m/s" },
};
const TIMES = ["2026-03-15T00:00:00", "2026-03-15T01:00:00", "2026-03-15T02:00:00"];
const VIEWS = {
  meta: META,
  arrays: {
    hazard: Int8Array.from([0, 4, 6]), hazard_dv: Float32Array.from([NaN, 0.8, 10.9]),
    hazard_depth: Float32Array.from([NaN, 1.6, 4.8]), hazard_speed: Float32Array.from([NaN, 0.5, 2.28]),
    hazard_step: Int32Array.from([-1, 2, 1]), arrival_step: Int32Array.from([-1, 0, 2]),
    arrival_h: Float32Array.from([NaN, 0, 2]), max_depth: Float32Array.from([NaN, 1.6, 4.8]),
    max_depth_step: Int32Array.from([-1, 2, 1]), max_speed: Float32Array.from([NaN, 0.5, 2.28]),
    max_speed_step: Int32Array.from([-1, 2, 1]),
  },
};
const UNITS = { depth: "m", velocity: "m/s" };

test("the lenses on offer follow what the package computed, in the legend's order", () => {
  assert.deepEqual(LENSES.map((l) => l.id), ["depth", "hazard", "arrival", "max_depth", "max_speed"]);
  assert.deepEqual(lensesFor({ available: ["depth", "max_depth", "max_speed"] }).map((l) => l.id), ["depth", "max_depth", "max_speed"]);
  assert.deepEqual(lensesFor(null).map((l) => l.id), ["depth"]);
});

test("six hazard colours, light to dark, with the dry class transparent", () => {
  assert.equal(HAZARD_COLOURS.length, 6);
  assert.equal(HAZARD_SHORT.length, 6);
  const p = hazardPacked();
  assert.equal(p.length, 7);
  assert.equal(p[0], 0);
  assert.equal(p[1], pack(0xf3, 0xdf, 0x70, 0.88));
});

test("a grid is painted through its ramp, NaN left clear", () => {
  const ramp = packedRamp(ARRIVAL_STOPS, 4);
  const out = paintGrid(new Uint32Array(4), Float32Array.from([NaN, 0, 8, 99]), { lo: 0, hi: 16, sqrt: false }, ramp);
  assert.equal(out[0], 0);
  assert.equal(out[1], ramp[0]);
  assert.equal(out[2], ramp[2]);
  assert.equal(out[3], ramp[3]);   // above the scale: the top colour
});

test("each lens has its own scale", () => {
  const depth = { lo: 0, hi: 5, sqrt: true };
  assert.deepEqual(lensScale("arrival", META, depth, 2), { lo: 0, hi: 16, sqrt: false });
  assert.deepEqual(lensScale("max_speed", META, depth, 2.28), { lo: 0, hi: 2.5, sqrt: false });
  assert.equal(lensScale("max_depth", META, depth, 0), depth);
  assert.deepEqual(lensTicks("arrival", { lo: 0, hi: 16 }).map((t) => t.label), ["0 h", "8 h", "16 h"]);
});

test("hours, areas and limits in words", () => {
  assert.equal(hoursWords(0), "at the start");
  assert.equal(hoursWords(2.5), "2 h 30 min");
  assert.equal(hoursWords(30), "1 day 6 h");
  assert.equal(areaWords(4800), "4,800 m²");
  assert.equal(areaWords(185000), "18.5 ha");
  assert.equal(areaWords(2.3e6), "2.3 km²");
  assert.match(limitWords(CLASSES[3]), /^H4 means depth × velocity up to 1 m²\/s, depth up to 2 m and speed up to 2 m\/s\.$/);
  assert.match(limitWords(CLASSES[5]), /past an H5 limit: depth × velocity over 4 m²\/s, depth over 4 m or speed over 4 m\/s/);
});

test("the card explains the hazard at a cell with the step and the numbers that put it there", () => {
  const f = lensFacts("hazard", VIEWS, 2, TIMES, UNITS);
  assert.equal(f.status, `H6: ${CLASSES[5].words}`);
  assert.equal(f.figure.value, "H6");
  assert.match(f.note, /Worst at 15 Mar 2026 01:00, with 4\.8 m of water moving at 2\.28 m\/s \(depth × velocity 10\.9 m²\/s\)/);
  assert.match(f.note, /AIDR Guideline 7-3/);
  assert.equal(f.color, HAZARD_COLOURS[5]);
  assert.match(lensFacts("hazard", VIEWS, 0, TIMES, UNITS).status, /^Dry all through the run/);
});

test("the card says when a place floods, or that it was wet from the start", () => {
  assert.match(lensFacts("arrival", VIEWS, 2, TIMES, UNITS).status, /^Floods at 15 Mar 2026 02:00, 2 h after the start$/);
  assert.match(lensFacts("arrival", VIEWS, 1, TIMES, UNITS).status, /^Wet from the start of the run/);
  assert.match(lensFacts("arrival", VIEWS, 0, TIMES, UNITS).status, /^Never 0\.1 m deep/);
  assert.equal(lensFacts("depth", VIEWS, 1, TIMES, UNITS), null);
});

test("the card gives the largest depth and speed with the step they were reached", () => {
  const d = lensFacts("max_depth", VIEWS, 2, TIMES, UNITS);
  assert.equal(d.status, "Largest depth 4.8 m over the run");
  assert.match(d.note, /01:00/);
  assert.equal(lensFacts("max_speed", VIEWS, 1, TIMES, UNITS).status, "Largest speed 0.5 m/s over the run");
});

test("each lens has its line in the key", () => {
  assert.match(lensLine("hazard", META), /from depth and speed at every step$/);
  assert.equal(lensLine("arrival", META), "When each place first has 0.1 m of water, after the start of the run");
  assert.equal(lensLine("depth", META, { step: 1, times: TIMES }), "Depth at 15 Mar 2026 01:00");
});

test("a file with only the water surface calls its largest value a water surface, not a depth", () => {
  const ws = { ...VIEWS, meta: { ...META, variable: "water_surface" } };
  assert.equal(lensLine("max_depth", ws.meta), "Highest water surface in each place over the run");
  assert.equal(lensFacts("max_depth", ws, 2, TIMES, { water_surface: "m" }).status, "Largest water surface 4.8 m over the run");
});
