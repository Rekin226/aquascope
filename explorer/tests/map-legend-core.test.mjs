// "On the map" (#543 design pass): one legend for every layer.
import { test } from "node:test";
import assert from "node:assert/strict";

import { ROW_ORDER, chipLabel, rowRank, rowState, sortRows, startsFolded, startsOpen } from "../src/map-legend-core.js";

test("the rows follow the layers on the map, top first, and unknown rows go last in the order they came", () => {
  const rows = [{ id: "status" }, { id: "news-x" }, { id: "gauges" }, { id: "floods-past" }, { id: "action-log" },
    { id: "floods-ahead" }, { id: "rivers" }];
  assert.deepEqual(sortRows(rows).map((r) => r.id),
    ["gauges", "floods-ahead", "rivers", "floods-past", "status", "news-x", "action-log"]);
  assert.ok(rowRank("river-lit") < rowRank("gauges"));
  assert.ok(rowRank("forecast-points") > rowRank("gauges") && rowRank("forecast-points") < rowRank("floods-ahead"));
  assert.equal(new Set(ROW_ORDER).size, ROW_ORDER.length);
});

test("a row is on, off, or one muted line when its layer has nothing to draw", () => {
  assert.equal(rowState({ on: true }), "on");
  assert.equal(rowState({ on: false, empty: true }), "off");
  assert.equal(rowState({ on: true, empty: true }), "empty");
});

test("the folded chip counts only the layers being drawn", () => {
  assert.equal(chipLabel(["on", "off", "empty", "on"]), "On the map · 2");
  assert.equal(chipLabel(["off"]), "On the map");
  assert.equal(chipLabel(undefined), "On the map");
});

test("a phone starts with the chip, a wide screen with the river status's colours open", () => {
  assert.equal(startsFolded(390), true);
  assert.equal(startsFolded(1440), false);
  assert.equal(startsFolded(0), false);
  assert.equal(startsOpen("status", 1440), true);
  assert.equal(startsOpen("status", 390), false);
  assert.equal(startsOpen("gauges", 1440), false);
});
