// Flow vs normal (#559): the compact file, the lookup, the colours and the words.
import { test } from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";

import {
  FS_UNCLASSED, classOf, decodeFlowStatus, flowColorExpr, fsClass, isGzip, issueLine, readFlowNormalParam,
  rowSummary, sharesLine,
} from "../src/flowstatus-core.js";
import { STATUS_CLASSES } from "../src/now-core.js";
import { FS_WIDER, riverCasingOpacity, riverCasingWidth, riverOpacityFlow, riverThemeKind, riverWidth, riverWidthFlow } from "../src/river-core.js";

// The same bytes aquascope.flowstatus.encode writes for ids [5, 127, 128, 16384, 820425136, 2147483647] and
// classes [1, 3, 2, 4, 5, 3] (tests/test_flowstatus.py), before gzip.
function file(ids, classes) {
  const out = [0x41, 0x51, 0x46, 0x53, 1, ids.length & 255, (ids.length >> 8) & 255, (ids.length >> 16) & 255, 0];
  let prev = 0;
  for (const id of ids) {
    let v = id - prev;
    prev = id;
    while (v >= 128) { out.push((v % 128) | 128); v = Math.floor(v / 128); }
    out.push(v);
  }
  return new Uint8Array([...out, ...classes]);
}

test("the file decodes to sorted ids and their classes, big ids included", () => {
  const ids = [5, 127, 128, 16384, 820425136, 2147483647];
  const data = decodeFlowStatus(file(ids, [1, 3, 2, 4, 5, 3]));
  assert.deepEqual([...data.ids], ids);
  assert.deepEqual([...data.classes], [1, 3, 2, 4, 5, 3]);
  assert.equal(classOf(data, 820425136), 5);
  assert.equal(classOf(data, "16384"), 4);
  assert.equal(classOf(data, 6), 0);
  assert.equal(classOf(null, 5), 0);
  assert.throws(() => decodeFlowStatus(new Uint8Array([1, 2, 3])));
  assert.throws(() => decodeFlowStatus(file(ids, [1, 2])));
  assert.ok(isGzip(gzipSync(Buffer.from("x"))) && !isGzip(file(ids, [1, 1, 1, 1, 1, 1])));
});

test("a reach looks its class up fast among many", () => {
  const n = 50000;
  const ids = Array.from({ length: n }, (_, i) => 1000 + i * 7);
  const classes = ids.map((_, i) => (i % 5) + 1);
  const data = decodeFlowStatus(file(ids, classes));
  for (const i of [0, 1, 777, n - 1]) assert.equal(classOf(data, ids[i]), classes[i]);
  assert.equal(classOf(data, 1001), 0);
});

test("the colours are the river status layer's five, grey where there is no class", () => {
  const expr = flowColorExpr("light");
  assert.equal(expr[0], "match");
  STATUS_CLASSES.forEach((c, i) => {
    assert.equal(expr[2 + i * 2], i + 1);
    assert.equal(expr[3 + i * 2], c.color);
    assert.equal(fsClass(i + 1).id, c.id);
  });
  assert.equal(expr[expr.length - 1], FS_UNCLASSED.light);
  assert.equal(flowColorExpr("dark").at(-1), FS_UNCLASSED.dark);
  assert.equal(fsClass(0), null);
  assert.equal(riverThemeKind("satellite"), "imagery");
  const op = riverOpacityFlow(0.35);
  assert.equal(op[0], "interpolate");                     // zoom on the outside, as MapLibre needs
  assert.equal(op[4][0], "case");
  // the casing only under a classed reach, and never where the line itself has no width
  const cw = riverCasingWidth(1.6);
  assert.equal(cw[0], "interpolate");
  assert.deepEqual(cw[4].slice(-4), [10, 5.5, 12, 6.4]);    // the classed line's width plus the casing's
  assert.equal(cw[4][4], 0);
  // a classed reach is drawn wider than the plain network, an unclassed one keeps its width
  const wf = riverWidthFlow(), w = riverWidth();
  assert.equal(wf[0], "interpolate");
  assert.equal(wf[4][0], "case");
  assert.deepEqual(wf[4][3], w[4]);
  assert.deepEqual(wf[4][2].slice(-2), [12, Number((4 + FS_WIDER).toFixed(2))]);
  assert.equal(wf[4][2][4], 0);                             // still nothing where the network draws nothing
  const co = riverCasingOpacity();
  assert.equal(co[4][0], "case");
  assert.equal(co[4][3], 0);
});

const M = { issue_date: "2026-10-11", month: 10, base_period: "1950 to 2019",
  counts: { much_below: 100, below: 150, normal: 500, above: 150, much_above: 100 } };

test("the words take every number from the manifest", () => {
  assert.equal(rowSummary(M), "flow vs normal, Sun 11 Oct");
  assert.equal(rowSummary({ ...M, smoke: true }), "flow vs normal, Sun 11 Oct (sample)");
  assert.equal(rowSummary({ missing: true }), "flow vs normal, not published yet");
  assert.equal(rowSummary(null, { loading: true }), "flow vs normal, reading");
  assert.equal(sharesLine(M), "Of 1,000 reaches, 25% above normal and 25% below.");
  assert.equal(sharesLine({}), "");
  assert.match(issueLine(M), /forecast for Sun 11 Oct against each reach's October normal \(1950 to 2019\)/);
  assert.match(issueLine({ missing: true }), /Not published yet/);
});

test("a link turns the colouring on or off, or says nothing", () => {
  assert.equal(readFlowNormalParam("#fn=1&v=3/10/20"), true);
  assert.equal(readFlowNormalParam("#fn=0"), false);
  assert.equal(readFlowNormalParam("#v=3/10/20"), null);
});
