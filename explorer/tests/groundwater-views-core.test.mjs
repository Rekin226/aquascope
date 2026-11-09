// What a hydrogeologist reads off MODFLOW heads (#623, #624): the smooth surface, the features, the words.
import test from "node:test";
import assert from "node:assert/strict";
import {
  VIEW_MODES, captureFeatures, changeSince, contourFeatures, gradientWords, levelLabel, modeOf, paintSmooth, rateWords,
  smoothScale, viewFeatures, viewLines, wellCardOf, wellColor, wellFeatures,
} from "../src/groundwater-views-core.js";
import { HEAD_RAMP, rampColor } from "../src/groundwater-core.js";

const v = {
  kind: "modflow_views", time_index: 3, layer: 1, length_unit: "meters",
  head: { step: 2, range: [38, 56], levels: [{ level: 40, lines: [[[2.3, 48.85], [2.31, 48.86]]] }, { level: 42, lines: [[[2.3, 48.86], [2.31, 48.87]]] }] },
  flow: { arrows: [{ lon: 2.3, lat: 48.86, bearing: 116.6, size: 1 }], gradient: 0.0045, bearing: 116.6, towards: "east-south-east" },
  drawdown: { reference: 0, range: [-1.8, 1.8], step: 0.5, levels: [{ level: 0.5, lines: [[[2.3, 48.85], [2.31, 48.86]]] }, { level: -0.5, lines: [[[2.3, 48.85], [2.31, 48.86]]] }], suggested_reference: 21 },
  wells: [{ id: "W1", row: 15, col: 23, layer: 1, model_row: 15, model_col: 23, lon: 2.32, lat: 48.86, pumping: true, in_layer: true, q: null, head: 47.1, drawdown: 2.3, capture_km2: 1.24, from: "found" },
    { id: "W2", row: 3, col: 4, layer: 2, model_row: 3, model_col: 4, lon: 2.3, lat: 48.87, pumping: false, in_layer: false, q: -500, head: 50, from: "wel" }],
  capture: [{ well: "W1", area_km2: 1.24, rings: [[[2.31, 48.85], [2.33, 48.85], [2.33, 48.87], [2.31, 48.85]]], paths: [[[2.3, 48.86], [2.32, 48.86]]] }],
  wells_how: "found in the heads: ...", wells_from: "found", limits: "A head file has no conductivity.",
};

test("three modes, each asking Python only for what it draws", () => {
  assert.deepEqual(VIEW_MODES.map((m) => m.id), ["head", "drawdown", "capture"]);
  assert.deepEqual(modeOf("capture").show, ["head", "capture"]);
  assert.equal(modeOf("nope").id, "head");
});

test("the smooth surface interpolates between centres and leaves inactive cells clear", () => {
  const vals = new Float32Array([0, 10, NaN, 10]);           // 2 x 2, the bottom-left cell inactive
  const { px, width, height } = paintSmooth(vals, 2, 2, [0, 10], "head", 4);
  assert.equal(width, 8); assert.equal(height, 8);
  const alpha = (x, y) => px[(y * width + x) * 4 + 3];
  assert.equal(alpha(1, 6), 0);                               // inside the inactive cell
  assert.ok(alpha(1, 1) > 0 && alpha(6, 6) > 0);
  // the top row runs from the first cell's colour to the second's, smoothly
  const red = (x) => px[(1 * width + x) * 4];
  const lo = rampColor(HEAD_RAMP, 0)[0], hi = rampColor(HEAD_RAMP, 1)[0];
  assert.ok(red(0) === lo && red(7) === hi);
  assert.ok(red(3) < lo && red(3) > hi);                      // a colour between them (the ramp darkens)
  assert.equal(smoothScale(30, 40), 16);
  assert.equal(smoothScale(400, 400), 4);
});

test("drawdown paints the change since the reference in the diverging ramp, faint near no change", () => {
  const ch = changeSince(new Float32Array([9, 10, NaN]), new Float32Array([10, 10, 10]));
  assert.equal(ch[0], -1); assert.equal(ch[1], 0); assert.ok(Number.isNaN(ch[2]));
  const { px } = paintSmooth(new Float32Array([-1, 0, 0, 0]), 2, 2, [-1, 1], "drawdown", 2);
  assert.ok(px[3] > px[(1 * 4 + 3) * 4 + 3]);                  // the fall is stronger than no change
});

test("contour labels and index lines", () => {
  assert.equal(levelLabel(40, 2, "m"), "40 m");
  assert.equal(levelLabel(0.5, 0.5), "0.5");
  assert.equal(levelLabel(0.25, 0.25), "0.25");
  const f = contourFeatures(v.head, "m");
  assert.equal(f.length, 2);
  assert.equal(f[0].properties.label, "40 m");
  assert.equal(f[0].properties.index, true);                  // 40 m is the 20th interval: an index line
  assert.equal(f[1].properties.index, false);
  assert.equal(contourFeatures({ step: 2, levels: [{ level: 50, lines: [] }] })[0].properties.index, true);
  const d = contourFeatures(v.drawdown, "m", { drawdown: true });
  assert.equal(d[0].properties.label, "0.5 m");
  assert.equal(d[1].properties.label, "+0.5 m");
  assert.equal(d[1].properties.rise, true);
});

test("each mode draws its own features", () => {
  const head = viewFeatures(v, "head", "m");
  assert.equal(head.arrows.features.length, 1);
  assert.equal(head.zones.features.length, 0);
  assert.equal(head.wells.features.length, 2);
  const cap = viewFeatures(v, "capture", "m");
  assert.equal(cap.arrows.features.length, 0);
  assert.equal(cap.zones.features.length, 1);
  assert.equal(cap.paths.features.length, 1);
  const ring = cap.zones.features[0].geometry.coordinates[0];
  assert.deepEqual(ring[0], ring[ring.length - 1]);
  assert.equal(cap.zones.features[0].properties.color, wellColor(1));
  assert.equal(viewFeatures(v, "drawdown").contours.features[0].properties.label, "0.5");
  assert.equal(viewFeatures({ error: "x" }, "head").contours.features.length, 0);
  const w = wellFeatures(v.wells);
  assert.equal(w[0].properties.pumping, true);
  assert.equal(w[1].properties.here, false);
  assert.equal(captureFeatures([{ well: "W2", rings: [[[0, 0], [1, 0], [1, 1], [0, 1]]], paths: [] }]).zones[0].geometry.coordinates[0].length, 5);
});

test("the legend says what is drawn, briefly", () => {
  const head = viewLines(v, "head", { unit: "m" });
  assert.match(head[0], /^Contours every 2 m, heavier every 10 m\.$/);
  assert.match(head[1], /east-south-east, about 1 in 220/);
  const dd = viewLines(v, "drawdown", { unit: "m", stepLabel: (t) => `month ${t + 1}` });
  assert.match(dd[0], /^Change since month 1: brown where the water table is lower now/);
  const later = viewLines({ ...v, time_index: 3, drawdown: { ...v.drawdown, reference: 21 } }, "drawdown", { stepLabel: (t) => `month ${t + 1}` });
  assert.match(later[0], /^Change against month 22, a later step: brown/);
  assert.match(dd[1], /Deepest at W1: 2\.30 m down/);
  const cap = viewLines(v, "capture", { unit: "m" });
  assert.match(cap[1], /W1 1\.2 km²/);
  assert.match(cap[2], /found in the heads/);
  assert.match(viewLines({ ...v, wells: [] }, "capture")[0], /No well pumps/);
  assert.deepEqual(viewLines(null, "head"), ["Drawing…"]);
  for (const line of [...head, ...dd, ...cap]) assert.ok(!line.includes("\u2014"), "no em dashes");
});

test("gradients and rates in a reader's words", () => {
  assert.equal(gradientWords(0.0045), "about 1 in 220");
  assert.equal(gradientWords(0.0001), "about 1 in 10,000");
  assert.equal(gradientWords(0), "");
  assert.equal(rateWords(-3791.85, "m", "days"), "pumping 3,792 m³/d");
  assert.equal(rateWords(12.5, "", "days"), "injecting 12.5 (model units)");
  assert.equal(rateWords(0, "m", "days"), "not pumping");
});

test("the well card", () => {
  const c = wellCardOf(v, v.wells[0], { unit: "m", when: "2024-05-01" });
  assert.equal(c.title, "Well W1");
  assert.match(c.sub, /row 15, column 23, layer 1 · 2024-05-01/);
  assert.equal(c.figure.label, "drawdown");
  assert.match(c.note, /Capture zone about 1\.2 km²/);
  const other = wellCardOf(v, v.wells[1], { unit: "m" });
  assert.match(other.status.text, /Screened in layer 2/);
  assert.equal(other.figure.label, "head");
});
