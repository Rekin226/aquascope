// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  CELL_DEG, DEPTH_LABEL, DEPTH_MINZOOM, DEPTH_TILES, MAX_CELLS, activeReaches, cellBox, cellLines, cellMask,
  cellParts, cellSignature, cellsFor, cellsInView, depthFacts, depthLegendLine, depthReturnPeriod, depthUrl,
  paintCell, probeCell, rampColor, rampCss, reachAt, reachRadiusKm, rowLats, tileFor,
} from "../src/flood-depth-core.js";

const reach = (id, lon, lat, rp, daily, order = 6) => ({
  type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] },
  properties: { river_id: id, rp, daily, order, peak: 1400, q2: 247, day: "2026-10-12", share: 0.9, gauges: "" },
});
// One reach at its 100-year flow on most days, one at 25 years, one that never passes 5 years.
const FEATURES = [
  reach(1, 121.206, 60.6537, 100, "466666665432110"),
  reach(2, 121.30, 60.80, 25, "444444444444444", 5),
  reach(3, 121.50, 60.90, 5, "222222222222222"),
];

// A depth window over the cell [121, 60.5, 121.5, 61] at 0.005 degrees: every pixel `v` metres deep.
function grid(v, box = [121, 60.5, 121.5, 61], d = 0.005) {
  const width = Math.round((box[2] - box[0]) / d), height = Math.round((box[3] - box[1]) / d);
  return { data: new Float32Array(width * height).fill(v), x0: box[0], y0: box[3], dx: d, dy: -d, width, height };
}

test("the map is the largest return period not above the forecast class; no 2- or 5-year maps", () => {
  assert.deepEqual([2, 5, 10, 25, 50, 100].map(depthReturnPeriod), [null, null, 10, 20, 50, 100]);
  assert.equal(depthReturnPeriod(undefined), null);
  assert.equal(DEPTH_LABEL, "may flood in the next 15 days, model estimate");
});

test("tiles, URLs and the radius rule", () => {
  assert.equal(DEPTH_TILES.length, 271);
  assert.equal(tileFor(60.65, 121.2), "ID226_N70_E120");
  assert.equal(tileFor(0, -150), null);
  assert.equal(depthUrl("ID226_N70_E120", 50),
    "https://data.source.coop/nlebovits/jrc-glofas/depth-rp50/ID226_N70_E120/ID226_N70_E120_RP50_depth.tif");
  assert.deepEqual([3, 5, 6, 7, 12, null].map(reachRadiusKm), [2.5, 3, 4.5, 6, 10, 3]);
});

test("the reaches with a depth map follow the forecast day", () => {
  const peak = activeReaches(FEATURES, -1);
  assert.deepEqual(peak.map((r) => [r.id, r.rp]), [[1, 100], [2, 20]]);
  const day0 = activeReaches(FEATURES, 0);
  assert.deepEqual(day0.map((r) => [r.id, r.cls, r.rp]), [[1, 25, 20], [2, 25, 20]]);
  const day11 = activeReaches(FEATURES, 11);   // reach 1 is down to 5 years, reach 2 still at 25
  assert.deepEqual(day11.map((r) => r.id), [2]);
  assert.equal(activeReaches([], -1).length, 0);
});

test("cells: the half-degree cells each circle touches, what they read and their signature", () => {
  const cells = cellsFor(activeReaches(FEATURES, -1));
  assert.ok(cells.has("242:121"), "the cell [121, 60.5, 121.5, 61] holds both reaches");
  const c = cells.get("242:121");
  assert.deepEqual(cellBox("242:121"), [121, 60.5, 121.5, 61]);
  assert.deepEqual(c.reaches.map((r) => r.id).sort(), [1, 2]);
  assert.deepEqual(cellParts(c).map((p) => p.rp), [20, 100]);
  assert.ok(cellParts(c)[0].url.endsWith("/depth-rp20/ID226_N70_E120/ID226_N70_E120_RP20_depth.tif"));
  const day = cellsFor(activeReaches(FEATURES, 0)).get("242:121");
  assert.notEqual(cellSignature(c), cellSignature(day), "a new day with other maps is a new picture");
  assert.equal(cellSignature(c), cellSignature({ reaches: [...c.reaches].reverse() }));
  const inView = cellsInView(cells, [120, 60, 122, 62], [121.2, 60.65], 1);
  assert.equal(inView.length, 1);
  assert.equal(inView[0].key, "242:121", "the nearest cell to the middle first");
  assert.equal(cellsInView(cells, [0, 0, 1, 1], [0.5, 0.5]).length, 0);
  assert.ok(MAX_CELLS >= 8 && CELL_DEG === 0.5 && DEPTH_MINZOOM >= 6);
});

test("the ramp: dry below 5 cm, light to deep blue, the legend gradient", () => {
  assert.equal(rampColor(0.01), null);
  assert.equal(rampColor(-9999), null);
  assert.equal(rampColor(NaN), null);
  assert.deepEqual(rampColor(1), [106, 174, 232, 0.74]);
  assert.deepEqual(rampColor(50), [23, 63, 153, 0.93]);
  const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  assert.ok(lum(rampColor(0.2)) > lum(rampColor(2)) && lum(rampColor(2)) > lum(rampColor(8)), "deeper is darker");
  assert.match(rampCss(), /^linear-gradient\(90deg, rgba\(191,224,247,0\.55\) 7%/);
});

test("Mercator rows: north to south, unevenly", () => {
  const lats = rowLats(60.5, 61, 4);
  assert.ok(lats[0] < 61 && lats[0] > lats[1] && lats[3] > 60.5);
});

test("a cell is painted inside the circles only, and the click reads it back", () => {
  const cell = cellsFor(activeReaches(FEATURES, -1)).get("242:121");
  const grids = new Map([[100, grid(2.5)], [20, grid(0.8)]]);
  const w = 100, h = 100;
  const p = paintCell(cell, grids, w, h);
  assert.ok(p.wet > 0 && p.max === 2.5);
  const at1 = probeCell(cell, p, 121.206, 60.6537, w, h);
  assert.deepEqual(at1, { depth: 2.5, rp: 100 });
  const at2 = probeCell(cell, p, 121.30, 60.80, w, h);
  assert.deepEqual(at2, { depth: 0.8, rp: 20 });
  assert.equal(probeCell(cell, p, 121.45, 60.55, w, h), null, "far from both reaches");
  assert.equal(probeCell(cell, p, 122, 60.6, w, h), null, "outside the cell");
  // The rim fades: a pixel near the edge of a circle is fainter than one at the middle.
  const alpha = (lon, lat) => {
    const x = Math.floor(((lon - 121) / 0.5) * w);
    const lats = rowLats(60.5, 61, h);
    let y = 0;
    while (y < h - 1 && lats[y] > lat) y++;
    return p.rgba[(y * w + x) * 4 + 3];
  };
  assert.ok(alpha(121.206, 60.6537) > alpha(121.206 + 0.052, 60.6537));
});

test("dry pixels stay clear, and a missing window draws nothing", () => {
  const cell = cellsFor(activeReaches(FEATURES, -1)).get("242:121");
  const p = paintCell(cell, new Map([[100, grid(-9999)]]), 50, 50);
  assert.equal(p.wet, 0);
  assert.equal(probeCell(cell, p, 121.206, 60.6537, 50, 50), null);
});

test("a larger river nearby keeps its own flood plain", () => {
  const cell = cellsFor(activeReaches([FEATURES[0]], -1)).get("242:121");
  const own = { order: 6, segs: [121.18, 60.66, 121.23, 60.65], box: [121.18, 60.65, 121.23, 60.66] };
  const lena = { order: 9, segs: [121.0, 60.63, 121.5, 60.63], box: [121.0, 60.63, 121.5, 60.63] };   // 2.6 km south
  const lines = cellLines(cell, new Map([[1, own], [99, lena]]));
  assert.equal(lines.act[0].segs, own.segs);
  assert.deepEqual(lines.other, lena.segs);
  const small = cellLines(cell, new Map([[1, own], [98, { ...lena, order: 4 }]]));
  assert.deepEqual(small.other, [], "a smaller river does not cut");
  assert.notEqual(lines.sig, small.sig);
  const p = paintCell(cell, new Map([[100, grid(3)]]), 200, 200, lines);
  assert.ok(probeCell(cell, p, 121.206, 60.665, 200, 200), "up the reach's own valley");
  assert.equal(probeCell(cell, p, 121.206, 60.633, 200, 200), null, "by the larger river");
  const plain = paintCell(cell, new Map([[100, grid(3)]]), 200, 200);
  assert.ok(probeCell(cell, plain, 121.206, 60.633, 200, 200), "without lines it is the circle");
  const mask = cellMask(cell, lines, 200, 200);
  assert.equal(mask.gw, 100);
});

test("the reach a click belongs to, and the card's words", () => {
  const rs = activeReaches(FEATURES, -1);
  assert.equal(reachAt(rs, 121.21, 60.655).id, 1);
  assert.equal(reachAt(rs, 121.21, 60.655, 20), null, "only reaches drawn from that map");
  assert.equal(reachAt(rs, 125, 60), null);
  const f = depthFacts(rs[1], 0.84, 20, { day: 3, when: "on Mon 12 Oct" });
  assert.equal(f.title, "River reach 2");
  assert.equal(f.sub, "May flood in the next 15 days, model estimate");
  assert.equal(f.status, "Forecast to pass its 25-year flow on Mon 12 Oct. Shown: the 20-year depth map, the nearest at or below it.");
  assert.deepEqual(f.figure, { value: "0.8", unit: "m", label: "deep here, 20-year map" });
  assert.equal(f.note, "");
  const peak = depthFacts(rs[0], 12.4, 100);
  assert.equal(peak.figure.value, "12");
  assert.match(peak.status, /the 100-year depth map\.$/);
  assert.match(peak.note, /15-day peak/);
});

test("the legend line", () => {
  assert.equal(depthLegendLine({ n: 0, inView: 0, zoom: 9 }), "No river is forecast to pass its 10-year flow on this day.");
  assert.equal(depthLegendLine({ n: 1165, inView: 0, zoom: 4 }),
    "1,165 reaches forecast to pass the 10-year flow. Zoom in on one to see the depth.");
  assert.equal(depthLegendLine({ n: 1165, inView: 30, zoom: 9 }), "30 in view, of 1,165 reaches forecast to pass the 10-year flow.");
  assert.equal(depthLegendLine({ n: 1, inView: 0, zoom: 9 }), "None in view, of 1 reach forecast to pass the 10-year flow.");
});
