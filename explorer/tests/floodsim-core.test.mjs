// Flood it (#658): the JS twin of aquascope.floodsim runs the same grids as the Python fixtures and must match them,
// the run's clock (frames, checks, the steady state) follows the package's, and the UGRID file is well formed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  FLOODSIM_WGSL, NC_TYPES, areaVolume, cellVelocity, corners, createCpuSim, edgeCodes, flowAt, frameTimes, keptCells,
  mergeSources, packFrame, pixelSizeM, runLoop, steadyCheck, stopTimes, terrariumElevation, ugridFile, unpackDepth,
  wetAtWall, wetBox, windowTiles, worldPx,
} from "../src/floodsim-core.js";

const fx = JSON.parse(readFileSync(new URL("../../tests/fixtures/floodsim_twin.json", import.meta.url), "utf8"));
const fe = JSON.parse(readFileSync(new URL("../../tests/fixtures/floodsim_twin_edges.json", import.meta.url), "utf8"));

function worst(a, b) {
  let w = 0;
  for (let i = 0; i < b.length; i++) w = Math.max(w, Math.abs(a[i] - b[i]));
  return w;
}

test("the JS solver matches the numpy solver step for step (free edges, one inflow)", () => {
  const ny = fx.z.length, nx = fx.z[0].length;
  const sim = createCpuSim({ z: fx.z.flat(), n: fx.n.flat(), nx, ny, dx: fx.dx }, { sources: [fx.source] });
  sim.advance(fx.duration_s);
  assert.equal(sim.state.steps, fx.steps);
  // Round-off differs (pow, the order of sums), and a wet/dry face can amplify it a little: within a micrometre.
  assert.ok(worst(sim.h, fx.h.flat()) < 1e-6);
  const v = sim.volume();
  assert.ok(Math.abs(v.error_m3) / v.inflow_m3 < 1e-12);
  assert.ok(Math.abs(v.outflow_m3 - fx.volume.outflow_m3) / fx.volume.outflow_m3 < 1e-8);
});

test("walls, a normal-depth outfall, a free outfall and a table inflow match numpy too", () => {
  const ny = fe.z.length, nx = fe.z[0].length;
  const edges = Object.fromEntries(Object.entries(fe.edges).map(([k, v]) => [k, v.map((x) => (x === null ? NaN : x))]));
  const sim = createCpuSim({ z: fe.z.flat(), n: fe.n.flat(), nx, ny, dx: fe.dx, edges }, { sources: fe.sources, tableDtS: fe.table_dt_s });
  sim.advance(fe.duration_s);
  assert.equal(sim.state.steps, fe.steps);
  assert.ok(worst(sim.h, fe.h.flat()) < 1e-6, `largest difference ${worst(sim.h, fe.h.flat())}`);
  const v = sim.volume();
  assert.ok(Math.abs(v.inflow_m3 - fe.volume.inflow_m3) / fe.volume.inflow_m3 < 1e-9);
  assert.ok(Math.abs(v.outflow_m3 - fe.volume.outflow_m3) / fe.volume.outflow_m3 < 1e-8);
  assert.ok(Math.abs(v.error_m3) / v.inflow_m3 < 1e-12);
});

test("still water stays still in JS too", () => {
  const nx = 12, ny = 10, z = [], h0 = [];
  for (let i = 0; i < nx * ny; i++) { const zi = (i * 7919) % 13 / 10; z.push(zi); h0.push(Math.max(2 - zi, 0)); }
  for (let c = 0; c < nx; c++) { z[c] = 5; h0[c] = 0; z[(ny - 1) * nx + c] = 5; h0[(ny - 1) * nx + c] = 0; }
  for (let r = 0; r < ny; r++) { z[r * nx] = 5; h0[r * nx] = 0; z[r * nx + nx - 1] = 5; h0[r * nx + nx - 1] = 0; }
  const sim = createCpuSim({ z, n: 0.03, nx, ny, dx: 10, h0, edges: { n: 0, s: 0, w: 0, e: 0 } });
  sim.advance(300);
  for (let i = 0; i < h0.length; i++) assert.ok(Math.abs(sim.h[i] - h0[i]) < 1e-12);
});

test("flows from tables, edge codes and merged inflows", () => {
  assert.equal(flowAt(5, 99, 3600), 5);
  assert.equal(flowAt([0, 10, 4], 1800, 3600), 5);
  assert.equal(flowAt([0, 10, 4], 9e9, 3600), 4);
  assert.equal(flowAt([0, 10, 4], -5, 3600), 0);
  const c = edgeCodes(3, 2, { n: 0, e: [null, 0.01] });
  assert.deepEqual([...c.n], [0, 0, 0]);
  assert.ok(Number.isNaN(c.s[0]) && Number.isNaN(c.e[0]));
  assert.equal(c.e[1], 0.01);
  assert.deepEqual(mergeSources([[1, 2, 3], [1, 2, [1, 2]], [0, 0, 7]]), [[1, 2, [4, 5]], [0, 0, [7]]]);
});

test("the clock: frames, checks and the near steady state, as aquascope.floodsim", async () => {
  assert.deepEqual(frameTimes(5), [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9900, 10800, 14400, 18000]);
  assert.deepEqual(stopTimes(2, true), [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200]);
  const s = steadyCheck({ wet_m2: 100, volume_m3: 1000 }, { wet_m2: 100.5, volume_m3: 1005 }, 1);
  assert.ok(s.steady && s.d_area < 0.01);
  assert.ok(!steadyCheck({ wet_m2: 100, volume_m3: 1000 }, { wet_m2: 103, volume_m3: 1000 }, 1).steady);
  assert.deepEqual(areaVolume([0, 0.04, 0.06, 1], 2), { wet_m2: 8, volume_m3: 4.4 });
  // A small closed valley with an outfall settles; the loop stops at the first check that meets the rule.
  const nx = 20, ny = 8, z = [];
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) z.push(-0.01 * c * 20 + 0.3 * Math.abs(r - 4));
  const edges = { n: 0, s: 0, w: 0, e: [0, 0, 0, 0.01, 0.01, 0.01, 0, 0] };
  const sim = createCpuSim({ z, nx, ny, dx: 20, n: 0.05, edges }, { sources: [[4, 1, 2]] });
  const engine = { advance: (t) => sim.advance(t), depth: () => new Float32Array(sim.h), faces: () => sim.faces() };
  const res = await runLoop(engine, { nx, ny, dx: 20, hours: 24, steady: true });
  assert.ok(res.steadyAt !== null && res.steadyAt <= 24 * 3600 && res.steadyAt % 3600 === 0);
  assert.ok(res.checks.at(-1).steady);
  assert.equal(res.frames.at(-1).t, res.steadyAt);
  const v = sim.volume();
  assert.ok(v.outflow_m3 > 0 && Math.abs(v.error_m3) / v.inflow_m3 < 1e-12);
});

test("velocities, packed frames, boxes and walls", () => {
  const nx = 3, ny = 2;
  const h = Float32Array.from([1, 0.005, 2, 0, 0.5, 0]);
  const qx = new Float64Array(ny * (nx + 1)).fill(1), qy = new Float64Array((ny + 1) * nx).fill(-2);
  const { u, v } = cellVelocity(h, qx, qy, nx, ny);
  assert.equal(u[0], 1); assert.equal(v[0], 2); assert.ok(Number.isNaN(u[1]));
  const f = packFrame(60, h, u, v);
  assert.deepEqual([...f.idx], [0, 2, 4]);
  assert.deepEqual([...unpackDepth(f, 6)], [1, 0, 2, 0, 0.5, 0]);
  assert.deepEqual(wetBox(h, nx, ny), { r0: 0, r1: 1, c0: 0, c1: 2 });
  assert.equal(wetAtWall(h, nx, ny, { n: 0, s: 0, w: 0, e: 0 }), true);
  assert.equal(wetAtWall(h, nx, ny, { n: NaN, s: NaN, w: NaN, e: NaN }), false);
  assert.deepEqual([...keptCells(Float32Array.from([0, 0, 0, 0, 1, 0, 0, 0, 0]), 3, 3, 1)], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
});

test("terrarium, tiles and corners", () => {
  assert.deepEqual([...terrariumElevation([128, 0, 0, 255, 128, 10, 128, 255])], [0, 10.5]);
  assert.ok(Math.abs(pixelSizeM(0, 0) - 40075016.686 / 256) < 1e-6);
  const [x, y] = worldPx(-2.2, 52.06, 12);
  const win = { x0: Math.floor(x) - 100, y0: Math.floor(y) - 50, nx: 300, ny: 520, zoom: 12 };
  const t = windowTiles(win);
  assert.equal(t.tiles.length, t.cols * t.rows);
  assert.match(t.tiles[0].url, /^https:\/\/elevation-tiles-prod\.s3\.amazonaws\.com\/terrarium\/12\/\d+\/\d+\.png$/);
  const [nw, ne, se] = corners(win);
  assert.ok(nw[0] < -2.2 && ne[0] > -2.2 && se[1] < 52.06 && nw[1] > 52.06);
});

test("the UGRID file is NetCDF-3 classic with the mesh, the bed, the levels and the velocities", () => {
  const nx = 4, ny = 3, z = Array.from({ length: 12 }, (_, i) => i);
  const h = new Float32Array(12); h[5] = 1.5;
  const f0 = packFrame(0, new Float32Array(12), null, null), f1 = packFrame(3600, h, Float32Array.from(h).fill(0.2), Float32Array.from(h).fill(-0.1));
  const [x, y] = worldPx(-2.2, 52.06, 12);
  const { bytes, cells } = ugridFile({ win: { x0: Math.floor(x), y0: Math.floor(y), nx, ny, zoom: 12 }, bed: z, hMax: h, frames: [f0, f1], start: "2026-10-27" });
  assert.equal(cells, 12);   // one wet cell and its ring of three covers this small grid
  assert.deepEqual([...bytes.slice(0, 4)], [0x43, 0x44, 0x46, 0x01]);
  const text = new TextDecoder().decode(bytes);
  for (const name of ["mesh_topology", "mesh2d_face_nodes", "mesh2d_flowelem_bl", "mesh2d_s1", "mesh2d_ucx", "seconds since 2026-10-27"]) {
    assert.ok(text.includes(name), name);
  }
  assert.equal(NC_TYPES.FLOAT, 5);
});

test("the WGSL has every pass, the edge codes and the inflow tables", () => {
  for (const fn of ["fn stepDt", "fn faces", "fn limit", "fn scale", "fn depth", "fn inflow", "fn edge", "pow(hf, 7.0 / 3.0)",
    "pow(d, 5.0 / 3.0)", "atomicMax"]) {
    assert.ok(FLOODSIM_WGSL.includes(fn), fn);
  }
  // At most eight storage buffers a stage (the WebGPU default limit).
  assert.ok((FLOODSIM_WGSL.match(/var<storage/g) || []).length <= 8);
});

test("the run's words: the flow once, a steady run's clock, the wall said only when true (#664 follow-up)", async () => {
  const { flowLine, clockLabel, clockSummary, depthTag, theLabel, wallContact, wallWords } = await import("../src/floodsim-core.js");
  const q = (v) => `${Math.round(v)} m³/s`;
  const g = { click: { flow_cms: 954, return_period: 2154 }, sources: [1, 2] };
  const typed = flowLine({ words: "954 m³/s", choice: { flow_cms: 954 }, g }, q);
  assert.equal(`${typed.words}${typed.rest}`.match(/954 m³\/s/g).length, 1);
  const rp = flowLine({ words: "the 100-year flow", choice: { return_period: 100 }, g: { click: { flow_cms: 343, return_period: 100 }, sources: [1] } }, q);
  assert.match(rp.rest, /\(343 m³\/s\)/);
  assert.equal(theLabel("954 m³/s"), "954 m³/s");
  assert.equal(theLabel("100-year flow"), "the 100-year flow");
  assert.equal(clockLabel(5 * 3600, { steady: true, steadyAt: 27 * 3600 }), "5 h into a steady run");
  assert.equal(clockLabel(27 * 3600, { steady: true, steadyAt: 27 * 3600, last: true }), "steady after 27 h");
  assert.equal(clockLabel(5 * 3600, { steady: false }), "+5 h");
  assert.equal(clockSummary({ steady: true }), "Flood it: hours since the water was turned on.");
  assert.equal(clockSummary({ steady: false }), "Flood it: hours from the hydrograph's start.");
  assert.equal(depthTag("100-year flow", 3600, { steady: true }), "Depth after 1 h of a steady run at the 100-year flow, a screening flood");
  // The wall: a 12 x 10 grid walled all round but an outfall on the east; water against the north wall far from the inflow.
  const nx = 12, ny = 10, h = new Float64Array(nx * ny);
  const edges = { n: 0, s: 0, w: 0, e: Array.from({ length: ny }, (_, r) => (r >= 3 && r <= 6 ? 0.001 : 0)) };
  const rule = { near: 2 };
  for (let c = 6; c < 10; c++) h[c] = 0.4;                  // north edge, 4 faces, deep
  let w = wallContact(h, nx, ny, edges, [[5, 0]], rule);
  assert.equal(w.held, true); assert.equal(w.sides.n, 4);
  // the same words as tests/test_floodsim.py's WALL_WORDS_N
  assert.equal(wallWords(w, 30), "The water reaches the grid's north edge along 120 m. The valley goes on past it, and the " +
    "edge holds the water like a wall, so the flood there is deeper and narrower than it would be. Flood a reach nearer that " +
    "edge to see it.");
  h.fill(0);
  for (let c = 6; c < 10; c++) h[c] = 0.06;                 // a film under 10 cm: not held
  assert.equal(wallContact(h, nx, ny, edges, [[5, 0]], rule).held, false);
  h.fill(0);
  for (let r = 2; r < 8; r++) h[r * nx + nx - 1] = 1;       // at the outfall and the walls beside it: not held
  assert.equal(wallContact(h, nx, ny, edges, [[5, 0]], rule).held, false);
  h.fill(0);
  for (let r = 3; r < 7; r++) h[r * nx] = 1;                // by the inflow on the west edge: not held
  assert.equal(wallContact(h, nx, ny, edges, [[5, 0]], rule).held, false);
  assert.equal(wallWords({ held: false }, 30), "");
});
