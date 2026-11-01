// Levees and breaches (#663): the drawn lines become the package's edits in a fixed order, the run's words say
// what was drawn, the crest control's profile is drawn from levee_profile's answer, and the before-and-after rows
// and features come straight from compare_exposure's.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  checkSketch, crestHeights, editsKey, exposureChangeFeatures, exposureChangeRows, profilePaths, sketchEdits,
  sketchFeatures, sketchWords,
} from "../src/floodsim-sketch-core.js";
import { createCpuSim } from "../src/floodsim-core.js";

const line = [[-2.2, 52.05], [-2.19, 52.06]];

test("a sketch is checked and rounded; a levee needs its crest", () => {
  assert.deepEqual(checkSketch({ id: 3, kind: "levee", line: [[-2.2000001, 52.05], [-2.19, 52.06]], crest: 14.237 }),
    { id: 3, kind: "levee", line, crest: 14.24 });
  assert.deepEqual(checkSketch({ id: 4, kind: "breach", line }), { id: 4, kind: "breach", line });
  assert.equal(checkSketch({ id: 4, kind: "breach", line, invert: "9.5" }).invert, 9.5);
  assert.throws(() => checkSketch({ id: 1, kind: "levee", line }));
  assert.throws(() => checkSketch({ id: 1, kind: "wall", line, crest: 3 }));
  assert.throws(() => checkSketch({ id: 1, kind: "breach", line: [[0, 0]] }));
});

test("the edits: levees first, then breaches, each in the order drawn; the key tells runs apart", () => {
  const s = [{ id: 5, kind: "breach", line }, { id: 2, kind: "levee", line, crest: 14 }, { id: 9, kind: "levee", line, crest: 15 }];
  assert.deepEqual(sketchEdits(s).map((e) => [e.kind, e.id]), [["levee", 2], ["levee", 9], ["breach", 5]]);
  assert.deepEqual(sketchEdits(s)[2], { id: 5, kind: "breach", line, invert: null });
  assert.equal(editsKey(sketchEdits(s)), editsKey(sketchEdits([...s].reverse())));
  assert.notEqual(editsKey(sketchEdits(s)), editsKey(sketchEdits(s.slice(1))));
  assert.equal(editsKey([]), "[]");
  assert.equal(sketchWords(sketchEdits(s)), "with 2 levees and a breach");
  assert.equal(sketchWords([{ kind: "levee", crest: 14.24 }]), "with a levee at 14.2 m");
  assert.equal(sketchWords([]), "");
});

test("the lines on the map, the selected one and the one being drawn", () => {
  const fc = sketchFeatures([{ id: 2, kind: "levee", line, crest: 14 }], { selected: 2, draft: [[-2.1, 52]] });
  assert.equal(fc.features.length, 2);
  assert.deepEqual(fc.features[0].properties, { id: 2, kind: "levee", selected: 1, label: "14.0 m" });
  assert.equal(fc.features[1].geometry.type, "Point");
});

test("the crest control's profile: ground, flood, the crest, all inside the box", () => {
  const p = { distance_m: [0, 25, 50, 75], ground_m: [10, 9.5, 9.8, 11], terrain_m: [10, 9.5, 9.8, 11],
    water_m: [null, 10.4, 10.4, null] };
  const d = profilePaths(p, 11.2, { w: 200, h: 80 });
  assert.match(d.ground, /^M4 /);
  assert.ok(d.ground.endsWith("Z"));
  assert.equal(d.now, "");                    // no levee there yet
  assert.equal((d.water.match(/M/g) || []).length, 1);
  assert.match(d.crest, /^M4 [\d.]+L196 /);
  for (const v of [...p.ground_m, 10.4, 11.2]) assert.ok(d.y(v) > 0 && d.y(v) < 80, v);
  assert.ok(d.y(11.2) < d.y(10.4));          // the crest above the flood
  assert.deepEqual(crestHeights(p, 11.2), [0.2, 1.7]);
  assert.notEqual(profilePaths({ ...p, terrain_m: [10, 12, 12, 11] }, null).now, "");
});

test("before and after: the rows and the features from compare_exposure's answer", () => {
  const c = { buildings: { before: 120, after: 87, change: -33 }, roads_km: { before: 4.21, after: 3.9, change: -0.31 },
    people: { before: 310.4, after: 250, change: -60.4 },
    kept_dry: [{ id: "a", ring: [[0, 0], [1, 0], [1, 1], [0, 1]] }], newly_flooded: [{ id: "b", ring: [[0, 0], [1, 0], [1, 1], [0, 0]] }],
    roads_changed: [{ coords: [[0, 0], [1, 1]], change: "kept_dry", length_m: 30 }] };
  const rows = exposureChangeRows(c);
  assert.deepEqual(rows.map((r) => [r.label, r.before, r.after, r.change]),
    [["Buildings", "120", "87", "−33"], ["Roads", "4.2", "3.9", "−0.3"], ["People", "310", "250", "−60"]]);
  const fc = exposureChangeFeatures(c);
  assert.equal(fc.features.length, 3);
  assert.equal(fc.features[0].geometry.coordinates[0].length, 5);   // closed
  assert.equal(fc.features[1].geometry.coordinates[0].length, 4);   // already closed
  assert.equal(fc.features[2].properties.change, "kept_dry");
});

test("the JS solver matches numpy with a levee and a breach in place", () => {
  const fx = JSON.parse(readFileSync(new URL("../../tests/fixtures/floodsim_twin_levee.json", import.meta.url), "utf8"));
  const ny = fx.z.length, nx = fx.z[0].length;
  const edges = Object.fromEntries(Object.entries(fx.edges).map(([k, v]) => [k, v.map((x) => (x === null ? NaN : x))]));
  const sim = createCpuSim({ z: fx.z.flat(), n: fx.n.flat(), nx, ny, dx: fx.dx, edges }, { sources: fx.sources, tableDtS: fx.table_dt_s });
  sim.advance(fx.duration_s);
  assert.equal(sim.state.steps, fx.steps);
  let worst = 0;
  const h = fx.h.flat();
  for (let i = 0; i < h.length; i++) worst = Math.max(worst, Math.abs(sim.h[i] - h[i]));
  assert.ok(worst < 1e-6, `largest difference ${worst}`);
  const v = sim.volume();
  assert.ok(Math.abs(v.error_m3) / v.inflow_m3 < 1e-12);
  // The levee holds: the cells behind it stay dry here as in numpy.
  let wet = 0;
  for (const x of sim.h) if (x > 0.01) wet++;
  assert.equal(wet, fx.wet);
});

test("each drawn line is its own slot in the undo log, kept across its edits", async () => {
  const { slotOf } = await import("../src/map-actions-core.js");
  assert.equal(slotOf({ type: "draw_levee", sketch: 4 }, 99), "sketch:4");
  assert.equal(slotOf({ type: "draw_breach" }, 12), "sketch:12");
});

test("before and after is counted where the two runs differ", async () => {
  const { changeBox } = await import("../src/floodsim-sketch-core.js");
  const nx = 10, ny = 8, a = new Float32Array(nx * ny), b = new Float32Array(nx * ny);
  a.fill(1); b.fill(1);
  b[3 * nx + 4] = 0; b[5 * nx + 6] = 1.2;          // one cell kept dry, one deeper
  b[1 * nx + 1] = 1.03;                            // within the tolerance
  assert.deepEqual(changeBox(a, b, nx, ny, { pad: 1 }), { r0: 2, r1: 6, c0: 3, c1: 7 });
  assert.deepEqual(changeBox(a, b, nx, ny, { pad: 50 }), { r0: 0, r1: 7, c0: 0, c1: 9 });
  assert.equal(changeBox(a, a, nx, ny), null);
});
