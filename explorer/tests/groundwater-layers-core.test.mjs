// Groundwater views (#625, #626): the 3D prisms, the cross-section, the hydrograph and the budget chart.
import test from "node:test";
import assert from "node:assert/strict";
import {
  budgetRows, budgetStep, budgetSvg, dateWords, distWords, everWet, exaggerationChoices, hydrographSvg, hydrographWords,
  layerLine, niceTicks, prismFeatures, rateWords, sectionAt, sectionRange, sectionSvg, sectionValue, termColor, veWords,
} from "../src/groundwater-layers-core.js";
import { isModflowDrop, sortModelFiles } from "../src/groundwater-core.js";

// Two cells side by side, two layers, two times (time x layer x cell).
const rings = new Float64Array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 1, 2, 0, 1, 0]);
const offsets = new Int32Array([0, 4, 8]);
const heads = new Float32Array([50, NaN, 45, 44, 51, NaN, 46, NaN]);
const shape = [2, 2, 2];
const geo = { nlay: 2, top: new Float32Array([60, 60]), botm: new Float32Array([40, 40, 0, NaN]), base: 0, metres_per_unit: 1 };

test("prisms stand between each layer's bottom and top, exaggerated, coloured by head", () => {
  const model = { rings, offsets, heads, shape, layers: [1, 2] };
  const wet = everWet(heads, shape);
  assert.deepEqual([...wet], [1, 0, 1, 1]);
  const fc = prismFeatures(geo, model, 0, { exaggeration: 5, range: [40, 60], wet });
  // layer 1 cell 2 never has a head (inactive): left out; layer 2 cell 2 has no bottom: left out
  assert.deepEqual(fc.features.map((f) => [f.properties.l, f.properties.c]), [[0, 0], [1, 0]]);
  const [top, deep] = fc.features;
  assert.equal(top.properties.base, 200);
  assert.equal(top.properties.height, 300);
  assert.equal(deep.properties.base, 0);
  assert.equal(deep.properties.height, 200);
  assert.match(top.properties.color, /^rgb\(/);
  assert.equal(top.geometry.coordinates[0].length, 5);
  // time 1: layer 2 cell 2 is dry now but was wet: grey; pulled apart, the upper layer rises by the gap
  const later = prismFeatures({ ...geo, botm: new Float32Array([40, 40, 0, 0]) }, model, 1, { wet, gap: 50 });
  const dry = later.features.find((f) => f.properties.l === 1 && f.properties.c === 1);
  assert.equal(dry.properties.color, "#b9c0c7");
  assert.equal(later.features.find((f) => f.properties.l === 0).properties.base, 40 + 50);
  // only the layers shown
  assert.equal(prismFeatures(geo, model, 0, { shown: new Set([1]), wet }).features.length, 1);
  // a confining bed (MODFLOW-2005 LAYCBD) between the layers: layer 2 starts at its own top, not at layer 1's bottom
  const bed = prismFeatures({ ...geo, tops: new Float32Array([60, 60, 30, 30]) }, model, 0, { wet });
  assert.equal(bed.features.find((f) => f.properties.l === 1).properties.height, 30);
});

const sec = {
  segments: [{ cell: 0, row: 1, col: 1, from_m: 0, to_m: 100 }, { cell: 1, row: 1, col: 2, from_m: 100, to_m: 200 },
    { cell: 5, row: 1, col: 6, from_m: 400, to_m: 500 }],
  layers: [1, 2], length_m: 600, vertices_m: [0, 300, 600], metres_per_unit: 1,
  top: new Float32Array([60, 58, 55, 40, 39, 38]),
  bottom: new Float32Array([40, 39, 38, 0, 0, 0]),
  heads: new Float32Array([52, 51, NaN, 50, 49, NaN, 53, 52, NaN, 51, 50, NaN]),
  water_table: new Float32Array([52, 51, NaN, 53, 52, NaN]),
};

test("the section's values and its range over every time", () => {
  assert.equal(sectionValue(sec, "top", 0, 1, 2), 38);
  assert.equal(sectionValue(sec, "head", 1, 1, 1), 50);
  assert.equal(sectionValue(sec, "head", 0, 0, 2), null);
  assert.equal(sectionValue(sec, "wt", 1, 0, 0), 53);
  const [lo, hi] = sectionRange(sec);
  assert.ok(lo < 0 && hi > 60);
});

test("the section draws bands, the water table, deeper heads, the gap and states its exaggeration", () => {
  const d = sectionSvg(sec, 0, { width: 600, height: 200, unit: "m" });
  assert.match(d.svg, /^<svg class="gwl-svg"/);
  assert.equal((d.svg.match(/class="gwl-band"/g) || []).length, 4);   // two layers, two runs (a gap at 200-400 m)
  assert.match(d.svg, /class="gwl-wt"/);
  assert.match(d.svg, /Layer 2 head/);
  assert.match(d.svg, /Water table/);
  assert.match(d.svg, /class="gwl-nohead"/);                           // the last cell has no head in any layer
  assert.ok(d.noHead);
  assert.match(d.svg, /class="gwl-bend"/);
  assert.match(d.svg, />A</);
  assert.match(d.svg, />A′</);
  // 600 m over 476 px against ~70 m over 156 px: the drawing stretches the vertical about 4.5 times
  assert.ok(d.ve > 3 && d.ve < 7, String(d.ve));
  assert.equal(veWords(d.ve), `x${Math.round(d.ve)}`);
  assert.equal(veWords(37), "x35");
  assert.equal(veWords(1.2), "x1.2");
});

test("the hover reads the cell and each layer at a distance", () => {
  const at = sectionAt(sec, 1, 150);
  assert.equal(at.col, 2);
  assert.deepEqual(at.layers.map((l) => l.head), [52, 50]);
  assert.equal(at.water_table, 52);
  assert.equal(sectionAt(sec, 0, 300), null);
});

const hyd = {
  times: [31, 59, 90], dates: ["2024-01-31", "2024-02-29", "2024-03-31"],
  series: [
    { layer: 1, head: [10, 9, 9.5], min: 9, max: 10, min_at: "2024-02-29" },
    { layer: 2, head: [8, null, 7], min: 7, max: 8, min_at: "2024-03-31" },
    { layer: 3, head: [null, null, null], min: null, max: null, min_at: null },
  ],
};

test("the hydrograph: one line per layer with a head, the time marked", () => {
  const h = hydrographSvg(hyd, 1, { unit: "m" });
  assert.equal((h.svg.match(/class="gwl-hline"/g) || []).length, 2);
  assert.match(h.svg, /class="gwl-now"/);
  assert.match(h.svg, /Jan 2024/);
  assert.match(h.key, /Layer 1/);
  assert.doesNotMatch(h.key, /Layer 3/);
  assert.equal(hydrographSvg({ times: [1], series: [{ layer: 1, head: [null], min: null }] }, 0), null);
  assert.deepEqual(hydrographWords(hyd, 1, "m"), [
    "Layer 1: 9.00 m (lowest 9.00 m, 29 Feb 2024)", "Layer 2: no head now (lowest 7.00 m, 31 Mar 2024)"]);
  assert.equal(layerLine(0), layerLine(8));
});

const budget = {
  kstpkper: [[1, 1], [1, 2]], times: [30, 60], dates: ["2024-01-31", "2024-03-01"], unit: "m³/day",
  terms: [{ label: "Recharge", in: [100, 80], out: [0, 0] }, { label: "Wells", in: [0, 0], out: [60, 60] },
    { label: "Storage", in: [0, 10], out: [40, 30] }],
  total_in: [100, 90], total_out: [100, 90], discrepancy_pct: [0, 0],
};

test("the budget chart stacks terms in above zero and out below, and its table", () => {
  const d = budgetSvg(budget, 1, { width: 500, height: 180 });
  assert.equal(d.n, 2);
  assert.match(d.svg, /class="gwl-col"/);
  assert.match(d.svg, /class="gwl-zero"/);
  assert.match(d.svg, /in ↑/);
  // 2 times: recharge in twice, wells out twice, storage out twice and in once
  assert.equal((d.svg.match(/<rect x=/g) || []).length, 7);
  assert.match(d.key, /Recharge/);
  const r = budgetRows(budget, 1);
  assert.deepEqual(r.rows.map((x) => x.net), [80, -60, -20]);
  assert.equal(r.when, "1 Mar 2024");
  assert.equal(budgetStep(budget, [1, 2]), 1);
  assert.equal(budgetStep(budget, [9, 9]), -1);
  assert.equal(termColor("Recharge"), termColor("Recharge", 3));
  assert.notEqual(termColor("Something new", 0), termColor("Something new", 1));
});

test("words and ticks", () => {
  assert.equal(distWords(350), "350 m");
  assert.equal(distWords(4200), "4.2 km");
  assert.equal(distWords(12000), "12 km");
  assert.equal(rateWords(2500), "2,500");
  assert.equal(rateWords(12.44), "12.4");
  assert.equal(rateWords(0.0312), "0.031");
  assert.equal(rateWords(null), "–");
  assert.equal(dateWords("2025-03-02T03:00"), "Mar 2025");
  assert.equal(dateWords("2025-03-02T03:00", { day: true }), "2 Mar 2025");
  assert.equal(dateWords(30), "t 30");
  assert.deepEqual(niceTicks(0, 100, 4), [0, 25, 50, 75, 100]);
  assert.deepEqual(exaggerationChoices(5), [1, 2, 5, 10, 20]);
  assert.deepEqual(exaggerationChoices(1), [1, 2, 5]);
});

test("the budget file comes with the model's files", () => {
  const f = (name) => ({ name });
  const s = sortModelFiles([f("m.hds"), f("m.dis"), f("m.cbc"), f("m.nam")]);
  assert.equal(s.cbc.name, "m.cbc");
  assert.equal(sortModelFiles([f("m.hds"), f("m.bud")]).cbc.name, "m.bud");
  assert.equal(isModflowDrop([f("m.cbc")]), true);
});
