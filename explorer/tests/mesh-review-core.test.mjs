// Review my model results (#564): the pin a finding becomes, the cells it flags while its pin stays, the words.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CHECK_LABELS, MAX_FLAG_CELLS, SEVERITY, checkRows, flaggedCells, gaugesInBox, reviewLine, reviewPin, reviewTarget, reviewWho,
  runSpan,
} from "../src/mesh-review-core.js";
import { ActionLog, checkPin } from "../src/map-actions-core.js";

const finding = (over = {}) => ({
  id: "r1", rank: 1, check: "negative", label: "Negative depths", severity: "problem", lat: 41.05, lon: -77.6,
  title: "Negative depth: -0.500 m", reason: "The water surface falls below the bed here.",
  facts: [{ id: "f1", label: "Lowest depth", value: -0.5, unit: "m", shown: "-0.500", when: "" },
    { id: "f2", label: "Cells with a negative depth", value: 1, unit: "cells", shown: "1", when: "" }],
  source: "Your model file, read by aquascope.io.mesh", cells: [12, 13], ...over,
});

test("a finding becomes a review pin that keeps its severity, its cells and its mesh through checkPin", () => {
  const pin = checkPin(reviewPin(finding(), { mesh: "run.p01.hdf" }));
  assert.equal(pin.kind, "review");
  assert.equal(pin.severity, "problem");
  assert.equal(pin.rank, 1);
  assert.equal(pin.check, "Negative depths");
  assert.equal(pin.mesh, "run.p01.hdf");
  assert.deepEqual(pin.cells, [12, 13]);
  // The numbers shown are the package's formatting, not the raw value.
  assert.deepEqual(pin.facts[0], { label: "Lowest depth", value: "-0.500", unit: "m" });
  assert.equal(pin.text, "The water surface falls below the bed here.");
  const odd = checkPin(reviewPin(finding({ severity: "dreadful", cells: [1, -2, 2.5, 3] }), { mesh: "m" }));
  assert.equal(odd.severity, "info");
  assert.deepEqual(odd.cells, [1, 3]);
  assert.equal(reviewPin(finding({ cells: Array.from({ length: MAX_FLAG_CELLS + 50 }, (_, i) => i) })).cells.length, MAX_FLAG_CELLS);
});

test("the flagged cells follow the pins in the log: undo a pin and its flags go", () => {
  const log = new ActionLog();
  const add = (f) => {
    const p = checkPin(reviewPin(f, { mesh: "run" }));
    return log.add({ action: { type: "add_pin", ...p }, slot: `pin:${log.peekId()}`, after: p }).id;
  };
  const a = add(finding({ severity: "info", cells: [1, 2] }));
  const b = add(finding({ severity: "warning", cells: [2, 3] }));
  let cells = flaggedCells(log.pins(), "run");
  assert.deepEqual([...cells].sort(), [[1, "info"], [2, "warning"], [3, "warning"]]);
  assert.equal(flaggedCells(log.pins(), "another-run").size, 0);
  log.undo(b);
  cells = flaggedCells(log.pins(), "run");
  assert.deepEqual([...cells].sort(), [[1, "info"], [2, "info"]]);
  log.undo(a);
  assert.equal(flaggedCells(log.pins(), "run").size, 0);
});

test("the severities, the checks and the words", () => {
  assert.deepEqual(Object.keys(SEVERITY), ["problem", "warning", "info"]);
  assert.equal(Object.keys(CHECK_LABELS).length, 9);
  assert.deepEqual(reviewWho("rules"), ["Rules", "checks by code, no model"]);
  assert.equal(reviewWho("key", "claude via anthropic")[0], "claude via anthropic");
  assert.match(reviewWho("device")[1], /picked on this device; checks and words by code/);
  // Asked, and its reply refused or timed out: said, not "no model" (#590).
  assert.deepEqual(reviewWho("rules", "", "Chrome's built-in model"),
    ["Rules", "Chrome's built-in model asked, its choice not used; checks and words by code"]);
  assert.equal(reviewLine({ summary: "Nothing flagged; 5 of 9 checks ran." }, 3),
    "Nothing flagged; 5 of 9 checks ran. 3 pins, most severe first.");
  assert.equal(reviewLine({ summary: "Nothing flagged." }, 0), "Nothing flagged. No pin to drop.");
  const rows = checkRows([{ check: "gauges", label: "Gauges in the mesh", status: "ok", line: "No Archive gauge." },
    { check: "tiny", status: "skipped", line: "No depth." }, { check: "negative", status: "flagged", line: "1 cell." }]);
  assert.deepEqual(rows.map((r) => r.mark), ["✓", "–", "!"]);
  assert.equal(rows[1].label, "Very shallow water");
});

test("the gauges handed to the review are the catalogue's inside the mesh's box, nearest the middle first", () => {
  const st = [
    { source: "usgs", station_id: "a", name: "Edge", lat: 41.0, lon: -77.7 },
    { source: "usgs", station_id: "b", name: "Middle", lat: 41.05, lon: -77.55 },
    { source: "usgs", station_id: "c", name: "Out", lat: 45, lon: -77.55 },
    { source: "usgs", station_id: "d", lat: NaN, lon: -77.55 },
  ];
  const rows = gaugesInBox(st, [-77.76, 40.96, -77.33, 41.18]);
  assert.deepEqual(rows.map((r) => r.station_id), ["b", "a"]);
  assert.deepEqual(Object.keys(rows[0]).sort(), ["lat", "lon", "name", "source", "station_id"]);
  assert.equal(gaugesInBox(st, [-77.76, 40.96, -77.33, 41.18], 1).length, 1);
});

test("the run's days for the gauge records", () => {
  assert.deepEqual(runSpan(["1999-01-01T12:00:00", "1999-01-04T12:00:00"]), { start: "1999-01-01", end: "1999-01-04" });
  assert.equal(runSpan([]), null);
});

// ── which model the review is about (#590) ──────────────────────────────────

const PORTSMOUTH = { name: "Portsmouth.p01.hdf", bbox: [-76.4, 36.8, -76.2, 36.9], openedAt: 1, visible: true };
const FREYBERG = { name: "freyberg.hds", bbox: [-98.1, 30.0, -97.9, 30.2], openedAt: 2, visible: true };
const TEXAS = [-99, 29, -97, 31];
const VIRGINIA = [-77, 36, -76, 37];

test("the review is of the model on screen, and names it", () => {
  // The MODFLOW sample on screen in Texas, an older HEC-RAS mesh off screen in Virginia: not reviewed out of sight.
  const t = reviewTarget({ mesh: PORTSMOUTH, modflow: FREYBERG, view: TEXAS });
  assert.equal(t.review, false);
  assert.match(t.line, /^freyberg\.hds \(MODFLOW heads\) is the model on screen/);
  assert.match(t.line, /Portsmouth\.p01\.hdf is loaded but not in view/);
  assert.deepEqual(reviewTarget({ mesh: PORTSMOUTH, modflow: FREYBERG, view: VIRGINIA }), { review: true, name: "Portsmouth.p01.hdf" });
  // Both in view: the one shown last.
  const both = [-100, 25, -70, 40];
  assert.equal(reviewTarget({ mesh: PORTSMOUTH, modflow: FREYBERG, view: both }).review, false);
  assert.equal(reviewTarget({ mesh: { ...PORTSMOUTH, openedAt: 3 }, modflow: FREYBERG, view: both }).review, true);
  // A mesh alone but out of view is said; with no view (the legend row) it is reviewed.
  assert.match(reviewTarget({ mesh: PORTSMOUTH, view: TEXAS }).line, /not in view/);
  assert.equal(reviewTarget({ mesh: PORTSMOUTH }).review, true);
  assert.match(reviewTarget({}).line, /^No model result on the map/);
  // The final line names the file.
  assert.equal(reviewLine({ summary: "1 problem." }, 2, "Portsmouth.p01.hdf"), "Portsmouth.p01.hdf: 1 problem. 2 pins, most severe first.");
  // A model picked the pins (#590, round 3): its order, and how many it left out; the counts stay the rules'.
  assert.equal(reviewLine({ summary: "3 things worth a look; 9 of 9 checks ran." }, 6, "Flood_model.p01.hdf", { picked: true, leftOut: 2 }),
    "Flood_model.p01.hdf: 3 things worth a look; 9 of 9 checks ran. 6 pins, in the model's order. 2 findings not pinned "
    + "(the model left them out; the checks list has every one).");
});
