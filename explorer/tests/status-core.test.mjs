// World river status (#544): the months, the classes and the Mercator layout.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CLASS_ALPHA, FILE_RGB, MERC_MAX_LAT, STATUS_CORNERS, classGrid, classOfRed, hexRgb, latestDay, mercatorRowLat,
  missingMonths, monthLabel, monthsToPeriods, paintGrid, parseListing, statusDatedLayer, statusMonthFor,
  statusPalette, statusUrl,
} from "../src/status-core.js";
import { STATUS_CLASSES } from "../src/now-core.js";
import { datedLayersOn, imageFor, registerDatedLayer } from "../src/layers.js";
import { layerCovers, missingNote } from "../src/timeline.js";

const LISTING = `<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>
<Contents><Key>hydrosos/cogs/2026-01.tif</Key></Contents><Contents><Key>hydrosos/cogs/2025-12.tif</Key></Contents>
<Contents><Key>hydrosos/cogs/2026-02.tif</Key></Contents><Contents><Key>hydrosos/cogs/2026-04.tif</Key></Contents>
<Contents><Key>hydrosos/cogs/notes.txt</Key></Contents><IsTruncated>false</IsTruncated></ListBucketResult>`;

test("the bucket listing gives the months in order and the gaps between them", () => {
  const { months, token } = parseListing(LISTING);
  assert.deepEqual(months, ["2025-12", "2026-01", "2026-02", "2026-04"]);
  assert.equal(token, null);
  assert.deepEqual(missingMonths(months), ["2026-03"]);
  assert.deepEqual(monthsToPeriods(months), ["2025-12-01/2026-02-01/P1M", "2026-04-01/2026-04-01/P1M"]);
  const next = parseListing("<NextContinuationToken>abc/=</NextContinuationToken>");
  assert.equal(next.token, "abc/=");
  assert.deepEqual(parseListing("").months, []);
});

test("a map date shows its own month, or nothing when there is no map for it", () => {
  const months = ["2025-12", "2026-01", "2026-02", "2026-04"];
  assert.equal(statusMonthFor("2026-01-31", months), "2026-01");
  assert.equal(statusMonthFor("2026-03-15", months), null, "a gap draws nothing");
  assert.equal(statusMonthFor("2026-10-03", months), null, "after the newest month");
  assert.equal(statusMonthFor("nonsense", months), null);
  assert.equal(latestDay(months), "2026-04-15");
  assert.equal(monthLabel("2026-09"), "Sep 2026");
  assert.equal(statusUrl("1990-01"), "https://geoglows-v2.s3.us-west-2.amazonaws.com/hydrosos/cogs/1990-01.tif");
});

test("the time bar sees the layer's span and its gaps", () => {
  const layer = statusDatedLayer(["2025-12", "2026-01", "2026-02", "2026-04"]);
  assert.equal(layer.since, "2025-12-01");
  assert.equal(layer.until, "2026-04-01");
  assert.equal(imageFor(layer, "2026-03-20"), null);
  assert.ok(layerCovers(layer, "2026-01-15", "2026-10-10"));
  assert.ok(!layerCovers(layer, "2026-03-15", "2026-10-10"), "the gap month is not covered");
  assert.ok(!layerCovers(layer, "2026-06-15", "2026-10-10"), "nor a month after the newest");
  assert.match(missingNote(layer, "2026-03-15", "2026-10-10"), /no image for Mar 2026/);
  // before the listing arrives the layer runs from 1990 with an open end
  const early = statusDatedLayer(null);
  assert.equal(early.since, "1990-01-01");
  assert.equal(early.until, null);
});

test("a registered layer joins the dated layers while it is on", () => {
  let on = true;
  const off = registerDatedLayer(() => (on ? statusDatedLayer(["2026-01"]) : null));
  try {
    assert.deepEqual(datedLayersOn("light", []).map((l) => l.id), ["status"]);
    on = false;
    assert.deepEqual(datedLayersOn("light", []), []);
  } finally {
    off();
  }
  assert.deepEqual(datedLayersOn("light", []), []);
});

test("the red band alone tells the five classes apart", () => {
  assert.equal(new Set(FILE_RGB.map((c) => c[0])).size, 5);
  FILE_RGB.forEach(([r], i) => assert.equal(classOfRed(r), i + 1));
  assert.equal(classOfRed(0), 0, "nodata");
  assert.equal(classOfRed(17), 0, "a colour that is not a class draws nothing");
});

test("Mercator rows run from about 85 N to 85 S, and the corners match", () => {
  assert.ok(Math.abs(mercatorRowLat(0, 2048) - MERC_MAX_LAT) < 0.1);
  assert.ok(Math.abs(mercatorRowLat(2047, 2048) + MERC_MAX_LAT) < 0.1);
  assert.ok(Math.abs(mercatorRowLat(1023.5, 2048)) < 1e-9, "the middle row is the equator");
  assert.deepEqual(STATUS_CORNERS[0], [-180, MERC_MAX_LAT]);
  assert.deepEqual(STATUS_CORNERS[2], [180, -MERC_MAX_LAT]);
});

test("classGrid samples the right cell: a 4 x 2 world, north half wet, south half dry", () => {
  // srcW 4 (90 degrees each), srcH 2 (north, south); class 5 north, 1 south, nodata in the far west
  const red = new Uint8Array([0, 44, 44, 44, 0, 205, 205, 205]);
  const grid = classGrid(red, 4, 2, 8, 8);
  assert.equal(grid.length, 64);
  assert.equal(grid[0 * 8 + 0], 0, "far west is nodata");
  assert.equal(grid[0 * 8 + 4], 5, "north is much above");
  assert.equal(grid[7 * 8 + 4], 1, "south is much below");
});

test("the palette is the gauges' five colours, normal painted lightest", () => {
  const p = statusPalette();
  assert.equal(p.length, 6);
  assert.deepEqual(p[0], [0, 0, 0, 0]);
  STATUS_CLASSES.forEach((c, i) => assert.deepEqual(p[i + 1].slice(0, 3), hexRgb(c.color)));
  assert.ok(CLASS_ALPHA[3] < Math.min(CLASS_ALPHA[1], CLASS_ALPHA[2], CLASS_ALPHA[4], CLASS_ALPHA[5]));
  const rgba = paintGrid(new Uint8Array([0, 1, 5]), new Uint8ClampedArray(12), p);
  assert.deepEqual([...rgba.slice(0, 4)], [0, 0, 0, 0]);
  assert.deepEqual([...rgba.slice(4, 8)], p[1]);
  assert.deepEqual([...rgba.slice(8, 12)], p[5]);
});
