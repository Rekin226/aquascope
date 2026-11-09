// The flood views' pure part (#621, #622): how the section, the hydrograph and the counts are drawn and said.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildingWords, clock, tickDigits, countsAt, exposureRows, exposureSummary, hoursWords, hydroKey, hydrographSvg, lengthWords, niceTicks,
  peopleWords, sectionHeading, sectionIndexAt, sectionLevels, sectionNote, sectionReadout, sectionSvg, sourceLines, wetAt, wetFilter,
} from "../src/flood-views-core.js";

const TIMES = ["2026-03-15T00:00:00", "2026-03-15T01:00:00", "2026-03-15T02:00:00", "2026-03-15T03:00:00"];

// a valley 300 m across: ground 10 at the banks, 6 in the middle; water at 8 on step 2, 9 at the peak
const SEC = {
  distance_m: [0, 100, 200, 300], cell: [0, 1, 2, -1], ground: [10, 6, 7, null],
  levels: [[null, null, null, null], [null, 7, null, null], [null, 8, 8, null], [null, 7.5, 7.5, null]],
  max_level: [null, 8, 8, null], times: TIMES, facts: { length_m: 300, widest_wet_m: 200, widest_step: 2, deepest: 2, deepest_at_m: 100, deepest_step: 2 },
  ground_is: "the lowest ground in each cell (HEC-RAS Cells Minimum Elevation)", notes: ["25% of the line lies off the mesh."],
};

test("axis ticks are round and cover the range", () => {
  assert.deepEqual(niceTicks(0, 10, 5), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(niceTicks(257.3, 263.1, 4), [258, 260, 262]);
  assert.deepEqual(niceTicks(1, 1), [1]);
});

test("axis labels carry the decimals their step needs", () => {
  assert.equal(tickDigits([0, 0.05, 0.1]), 2);
  assert.equal(tickDigits([0, 0.5, 1]), 1);
  assert.equal(tickDigits([0, 0.25, 0.5]), 2);
  assert.equal(tickDigits([0, 2.5, 5]), 1);
  assert.equal(tickDigits([258, 260, 262]), 0);
  const svg = hydrographSvg({ ...HY, depth: [null, 0.04, 0.11, 0.08, null, null], peak: { ...HY.peak, depth: 0.11 } });
  const labels = [...svg.matchAll(/class="fv-tick" x="29"[^>]*>([^<]+)</g)].map((m) => m[1]);
  assert.ok(labels.length >= 3);
  assert.equal(new Set(labels).size, labels.length);   // no two ticks read the same
});

test("words for lengths, hours and clock times", () => {
  assert.equal(lengthWords(850), "850 m");
  assert.equal(lengthWords(1670.6), "1.67 km");
  assert.equal(hoursWords(0.5), "30 min");
  assert.equal(hoursWords(13), "13 h");
  assert.equal(hoursWords(54), "2 d 6 h");
  assert.equal(clock("2026-03-15T14:00:00"), "14:00");
  assert.equal(clock("2026-03-15T14:00:00", { day: true }), "15 Mar 14:00");
});

test("the section draws the ground, the water at the step and the highest water dashed", () => {
  const svg = sectionSvg(SEC, { step: 1, w: 400, h: 200 });
  assert.match(svg, /class="fv-ground"/);
  assert.equal((svg.match(/class="fv-water"/g) || []).length, 1);   // one wet stretch at step 1
  assert.match(svg, /class="fv-max"/);
  assert.match(svg, />A</);
  assert.match(svg, />A′</);
  assert.doesNotMatch(svg, /NaN/);
  const max = sectionSvg(SEC, { view: "max", w: 400, h: 200 });
  assert.doesNotMatch(max, /class="fv-max"/);   // the highest water is what is drawn
  assert.deepEqual(sectionLevels(SEC, { view: "max" }), SEC.max_level);
  assert.deepEqual(sectionLevels(SEC, { step: 9 }), SEC.levels[3]);   // clamped to the run
});

test("the section's readout, heading and note", () => {
  assert.equal(sectionReadout(SEC, 1, { step: 2 }), "100 m · ground 6 · water 8 · 2 m deep");
  assert.equal(sectionReadout(SEC, 0, { step: 2 }), "0 m · ground 10 · dry");
  assert.equal(sectionReadout(SEC, 3, { step: 2 }), "300 m · off the mesh");
  assert.equal(sectionHeading(SEC, { step: 2 }), "300 m · at 15 Mar 02:00");
  assert.equal(sectionHeading(SEC, { view: "max" }), "300 m · highest water of the run");
  const note = sectionNote(SEC);
  assert.match(note, /widest water 200 m at 02:00/);
  assert.match(note, /Cells Minimum Elevation/);
  assert.match(note, /off the mesh/);
});

test("the pointer finds the nearest point of the section", () => {
  const w = 400, h = 200;
  assert.equal(sectionIndexAt(SEC, 50 + ((400 - 50 - 14) * 2) / 3, { w, h }), 2);
  assert.equal(sectionIndexAt(SEC, 5, { w, h }), -1);
});

const HY = {
  hours: [0, 1, 2, 3, 4, 5], times: ["2026-03-15T06:00:00", "2026-03-15T07:00:00", "2026-03-15T08:00:00",
    "2026-03-15T09:00:00", "2026-03-15T10:00:00", "2026-03-15T11:00:00"],
  depth: [null, 0.4, 1.2, 0.8, null, null], level: [null, 100.4, 101.2, 100.8, null, null], ground: 100, threshold: 0.01,
  wet: true, arrival: { step: 1, hours: 1, time: "2026-03-15T07:00:00" }, peak: { step: 2, hours: 2, depth: 1.2, level: 101.2, time: "2026-03-15T08:00:00" },
  recession: { step: 4, hours: 4, time: "2026-03-15T10:00:00" }, duration_h: 3, still_wet: false,
};

test("the hydrograph marks the arrival, the peak, the time under water and the step on show", () => {
  const svg = hydrographSvg(HY, { step: 3 });
  assert.match(svg, /class="fv-arrive"/);
  assert.match(svg, /class="fv-peak"/);
  assert.match(svg, /3 h under water/);
  assert.match(svg, /class="fv-now"/);
  assert.match(svg, />1\.2 m</);
  assert.match(svg, />level</);   // the right axis, the water level
  assert.doesNotMatch(svg, /NaN/);
  assert.equal(hydroKey(HY), "Arrives 15 Mar 07:00 · peak 1.2 m deep (level 101.2) at 15 Mar 08:00 · under water 3 h, receding by 15 Mar 10:00.");
  const still = { ...HY, recession: null, still_wet: true, duration_h: 4, arrival: { ...HY.arrival, step: 0 } };
  assert.match(hydroKey(still), /^Wet from the run's start/);
  assert.match(hydroKey(still), /still wet at the end, 4 h so far/);
  assert.match(hydrographSvg(still), /4 h\+ under water/);
  assert.equal(hydroKey({ wet: false, note: "The water never got deeper than 0.01 here." }), "The water never got deeper than 0.01 here.");
});

const RES = {
  times: TIMES, threshold: 0.01, units: { depth: "m" },
  buildings: { per_step: [0, 12, 30, 25], peak: { value: 30, step: 2 }, ever: 34, on_mesh: 120 },
  roads: { per_step_km: [0, 0.4, 2.25, 1.5], peak: { value: 2.25, step: 2 }, ever_km: 2.5, on_mesh_km: 9 },
  population: { per_step: [0, 41, 1234.5, 900], peak: { value: 1234.5, step: 2 }, ever: 1300, on_mesh: 4000 },
  sources: {
    buildings: { credit: "© OpenStreetMap contributors, Overture Maps Foundation", licence: "ODbL 1.0", link: "https://b", release: "2026-10-21.0" },
    roads: { credit: "© OpenStreetMap contributors, Overture Maps Foundation", licence: "ODbL 1.0", link: "https://r", release: "2026-10-21.0" },
    population: { credit: "European Commission, JRC", licence: "CC BY 4.0", link: "https://p" },
  },
};

test("the counts at a step, over the run, and their words", () => {
  assert.deepEqual(countsAt(RES, 1), { buildings: 12, roadsKm: 0.4, people: 41 });
  assert.deepEqual(countsAt(RES, 0, "max"), { buildings: 34, roadsKm: 2.5, people: 1300 });
  const rows = exposureRows(RES, 2);
  assert.deepEqual(rows.map((r) => r.value), ["30 buildings", "2.3 km of road", "~1,200 people"]);
  assert.match(rows[0].sub, /of 120 · peak 30 at 02:00/);
  assert.match(exposureRows(RES, 0, "max")[0].sub, /at some point/);
  assert.equal(exposureSummary(RES, 1), "12 bldg · 400 m road");
});

test("people are rounded to what a grid can say", () => {
  assert.equal(peopleWords(0.2), "nobody");
  assert.equal(peopleWords(3), "a few people");
  assert.equal(peopleWords(37), "about 40 people");
  assert.equal(peopleWords(1234.5), "about 1,200 people");
  assert.equal(peopleWords(16584), "about 17,000 people");
});

test("a feature is lit at the steps its string says", () => {
  assert.deepEqual(wetFilter(2), ["==", ["slice", ["get", "w"], 2, 3], "1"]);
  assert.deepEqual(wetFilter(2, "max"), ["in", "1", ["get", "w"]]);
  assert.equal(wetAt("0110", 1), true);
  assert.equal(wetAt("0110", 3), false);
  assert.equal(wetAt("0000", 0, "max"), false);
});

test("sources are credited once per credit and licence", () => {
  const lines = sourceLines(RES);
  assert.equal(lines.length, 2);
  assert.match(lines[0].text, /^buildings and roads: © OpenStreetMap contributors, Overture Maps Foundation, release 2026-10-21.0, ODbL 1.0$/);
  assert.match(lines[1].text, /CC BY 4.0/);
});

test("a clicked building says when it was under water", () => {
  const w = buildingWords({ w: "0110", d: 0.84, c: "residential" }, TIMES, { step: 1 });
  assert.equal(w.title, "A residential");
  assert.equal(w.status, "Flooded at 01:00. Under water from 15 Mar 01:00 to 15 Mar 02:00; up to 0.84 m deep in a cell under it.");
  assert.equal(buildingWords({ w: "0000" }, TIMES).status, "Stays dry over the whole run.");
  assert.match(buildingWords({ w: "1100" }, TIMES).note, /Already wet/);
});
