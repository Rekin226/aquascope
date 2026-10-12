// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  BEFORE, CLASSES, NEUTRAL, basinRows, beforeSet, classOf, colorOf, lineColorExpr, lineFilterExpr, logLabel, outletLines,
  paintedIds, pct, rangeWords, reachCard, resultBounds, scenarioWords, sliderValues,
} from "../src/whatif-core.js";
import { slotOf } from "../src/map-actions-core.js";

// A small answer shaped like aquascope.whatif_map.whatif's.
const RESULT = {
  ok: true,
  scenario: { dp_pct: -20, dt_c: 2, dpet_pct: null, demand_cms: 0, label: "20% less rain, 2 °C warmer" },
  reach: { river_id: 230000002, lat: 47.0, lon: 8.0 },
  basin: { area_km2: 500, centre: [47.01, 8.02], ids: [230000002, 230000001] },
  outlet: {
    area_km2: 500,
    mean: { label: "mean flow", change_pct: -38.3, range: [-39.0, -37.3] },
    low: { label: "low flow (Q95, exceeded 95% of days)", change_pct: -38.5, range: [-39.5, -37.5] },
    flood: { label: "10-year flood", change_pct: -46.0, range: [-46.3, -44.5] },
  },
  reaches: [
    { river_id: 230000002, area_km2: 500, share: 1, lon: 8.0, lat: 47.0, mean_change_pct: -38.3, mean_range: [-39, -37.3],
      q95_change_pct: -38.5, q95_range: [-39.5, -37.5], flood_change_pct: -46, flood_range: [-46.3, -44.5], confluence: null },
    { river_id: 230000003, area_km2: 950, share: 0.5263, lon: 8.1, lat: 47.05, mean_change_pct: -20.16, mean_range: [-20.5, -19.6],
      q95_change_pct: -20.3, q95_range: null, flood_change_pct: -24.2, flood_range: null, confluence: { n: 1, tributary_km2: 400 } },
  ],
};

test("a change is said with a true minus and its range", () => {
  assert.equal(pct(-38.3), "−38%");
  assert.equal(pct(4.4), "+4%");
  assert.equal(pct(0.2), "0%");
  assert.equal(pct(null), "n/a");
  assert.equal(rangeWords([-39.0, -37.3]), "−39 to −37%");
  assert.equal(rangeWords(null), "");
});

test("the classes run drier brown to wetter teal, quiet grey within ±3%", () => {
  assert.equal(colorOf(1), NEUTRAL);
  assert.equal(colorOf(-60), CLASSES[0].color);
  assert.equal(colorOf(80), CLASSES[CLASSES.length - 1].color);
  assert.equal(classOf(-10), 2);
  assert.equal(classOf(-9.9), 3);
  assert.equal(classOf(undefined), 4);
});

test("the paint is a match on riverId; reaches left of the swipe are drawn as before", () => {
  const expr = lineColorExpr(RESULT.reaches, "mean");
  assert.equal(expr[0], "match");
  assert.deepEqual(expr[2], [230000002]);
  assert.equal(expr[3], colorOf(-38.3));
  const swiped = lineColorExpr(RESULT.reaches, "mean", new Set([230000002]));
  assert.ok(swiped.includes(BEFORE) && !swiped.includes(colorOf(-38.3)));
  assert.deepEqual(lineFilterExpr([1, "2", 2]), ["match", ["to-number", ["get", "riverId"]], [1, 2], true, false]);
  assert.deepEqual(lineFilterExpr([]), ["boolean", false]);
  assert.equal(lineColorExpr([], "mean"), "rgba(0,0,0,0)");
});

test("the basin's reaches carry the outlet's change, the outlet itself is on the way down", () => {
  const rows = basinRows(RESULT);
  assert.deepEqual(rows.map((r) => r.river_id), [230000001]);
  assert.equal(rows[0].mean_change_pct, -38.3);
  assert.deepEqual(paintedIds(RESULT), [230000002, 230000003, 230000001]);
});

test("the swipe splits by longitude", () => {
  const where = (id) => ({ 230000002: 8.0, 230000003: 8.1, 230000001: 7.9 }[id]);
  assert.deepEqual([...beforeSet(paintedIds(RESULT), where, 8.05)].sort(), [230000001, 230000002]);
  assert.equal(beforeSet(paintedIds(RESULT), where, null).size, 0);
});

test("a painted reach's card says the three changes, its share and the confluence", () => {
  const c = reachCard(RESULT, 230000003);
  assert.equal(c.status, "Mean flow −20% (−21 to −20%)");
  assert.equal(c.figure.value, "53");
  assert.match(c.note, /Major confluence 1: 400 km² join here/);
  assert.match(c.note, /10-year flood −24%/);
  assert.equal(c.figure.label, "of the water is the basin's");
  assert.equal(reachCard(RESULT, 230000002).figure.label, "drain to the outlet");
  assert.match(reachCard(RESULT, 230000001).note, /lumped/);
  assert.equal(reachCard(RESULT, 999), null);
});

test("the outlet lines, the log label and the frame", () => {
  const lines = outletLines(RESULT);
  assert.deepEqual(lines.map((l) => l.key), ["mean", "low", "flood"]);
  assert.equal(lines[0].text, "−38% (−39 to −37%)");
  assert.equal(logLabel(RESULT), "What if 20% less rain, 2 °C warmer at river reach 230000002");
  const box = resultBounds(RESULT);
  assert.ok(box[0] < 8.0 && box[2] > 8.1 && box[1] < 47.0 && box[3] > 47.05);
  assert.equal(resultBounds({}), null);
});

test("the sliders take a scenario in their words and within their ends", () => {
  assert.equal(scenarioWords({ dp_pct: -20, dt_c: 2, demand_cms: 0 }), "20% less rain, 2 °C warmer");
  assert.equal(scenarioWords({}), "no change");
  assert.deepEqual(sliderValues({ dp_pct: -80, dt_c: 9, demand_cms: 3 }), { dp_pct: -50, dt_c: 6, demand_cms: 3 });
});

test("a what-if run has its own slot in the action log", () => {
  assert.equal(slotOf({ type: "what_if" }), "whatif");
});
