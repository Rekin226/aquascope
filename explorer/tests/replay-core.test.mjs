// Replay an event (#659): the player's pure parts.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  STORY_SCHEMA, STORY_VERSION, cameraFor, checkStory, decodeReplay, dwellMs, encodeReplay, exposureFill, fillTemplate,
  floodFill, forecastBar, pieceRows, scrubSegments, sparkline,
} from "../src/replay-core.js";

const stop = (id, extra = {}) => ({ id, kind: id, title: id, note: "A short note of six words.", date: "2024-11-26",
  camera: { center: [-2.2, 52.05], zoom: 11.2, pitch: 50, bearing: -20 }, layers: [], ...extra });

test("a story is checked: schema, version, stops with cameras", () => {
  const s = { schema: STORY_SCHEMA, version: STORY_VERSION, stops: [stop("rain"), stop("gauge")] };
  assert.equal(checkStory(s).length, 2);
  assert.throws(() => checkStory({ ...s, schema: "other" }), /not an AquaScope story/);
  assert.throws(() => checkStory({ ...s, version: 2 }), /version 2/);
  assert.throws(() => checkStory({ ...s, stops: [] }), /no stops/);
  assert.throws(() => checkStory({ ...s, stops: [{ id: "x", camera: { center: [1] } }] }), /no camera/);
});

test("a phone steps the camera back; pitch stays within what MapLibre draws", () => {
  const c = cameraFor(stop("gauge", { camera: { center: [1, 2], zoom: 10, pitch: 90, bearing: 5 } }), { narrow: true });
  assert.deepEqual(c, { center: [1, 2], zoom: 9.2, pitch: 70, bearing: 5 });
  assert.equal(cameraFor(stop("gauge")).zoom, 11.2);
});

test("a stop stays long enough to read, never more than 11 s, the flood longer", () => {
  assert.ok(dwellMs(stop("gauge")) >= 4200 && dwellMs(stop("gauge")) < 6000);
  assert.ok(dwellMs(stop("flood")) > dwellMs(stop("gauge")));
  assert.equal(dwellMs({ kind: "x", note: "word ".repeat(400) }), 11000);
  const segs = scrubSegments([stop("a"), stop("flood")]);
  assert.equal(segs.length, 2);
  assert.ok(Math.abs(segs[0].width + segs[1].width - 1) < 1e-9 && segs[1].from === segs[0].width);
});

test("the page's flood numbers fill the package's template; a missing field stays visible", () => {
  const t = "At {flow} ({flow_source}), Flood it puts {wet_km2} km² under water, {max_depth_m} m at the deepest.";
  const f = floodFill({ wet_km2: 41.234, max_m: 6.04 }, { flow: "433 m³/s", flow_source: "the gauge" });
  assert.equal(fillTemplate(t, f), "At 433 m³/s (the gauge), Flood it puts 41.2 km² under water, 6.0 m at the deepest.");
  assert.equal(fillTemplate("{a} and {b}", { a: 1 }), "1 and {b}");
  assert.deepEqual(exposureFill({ buildings: { after: 1234 }, roads_km: { after: 45.67 }, people: { after: 2050.4 } }),
    { buildings: "1,234", roads_km: "45.7", people: "2,050" });
  assert.equal(exposureFill(null), null);
});

test("the share link carries the reach, the window and the place, and reads back", () => {
  const v = encodeReplay({ river_id: 230567275, start: "2024-11-19", end: "2024-11-29", lat: 52.0496, lon: -2.2005 });
  assert.equal(v, "230567275,2024-11-19,2024-11-29,52.0496,-2.2005");
  assert.deepEqual(decodeReplay(`#v=5/1/2&rp=${v}`), { river_id: 230567275, start: "2024-11-19", end: "2024-11-29", lat: 52.0496, lon: -2.2005 });
  assert.deepEqual(decodeReplay("#rp=,2024-09-27,,35.58,-82.56"), { river_id: null, start: "2024-09-27", end: "2024-09-27", lat: 35.58, lon: -82.56 });
  assert.equal(decodeReplay("#rp=12,nope"), null);
  assert.equal(decodeReplay("#rp=,2024-09-27"), null);
  assert.equal(decodeReplay("#s=usgs/1"), null);
  // A Floods past event keeps its reports, so a fresh page replays the same story (#664 follow-up).
  const e = encodeReplay({ river_id: 230564005, start: "2024-11-19", end: "2024-11-29", lat: 52.04958, lon: -2.20047, reports: 102 });
  assert.equal(e, "230564005,2024-11-19,2024-11-29,52.0496,-2.2005,102");
  assert.deepEqual(decodeReplay(`#rp=${e}`), { river_id: 230564005, start: "2024-11-19", end: "2024-11-29", lat: 52.0496, lon: -2.2005,
    reports: 102, event: { start: "2024-11-19", end: "2024-11-29", lat: 52.0496, lon: -2.2005, reports: 102 } });
  assert.equal(encodeReplay({ river_id: 230564005, start: "2024-11-19", end: "2024-11-29", reports: 102 }), "230564005,2024-11-19,2024-11-29");
});

test("the assembling list says each piece's state", () => {
  const rows = pieceRows({ pieces: { gauges: { available: true }, rain: { pending: true }, forecast: { available: false, note: "before July 2024" },
    flood: { pending: true } } });
  assert.deepEqual(rows.map((r) => r.state), ["ok", "pending", "missing", "waiting", "pending"]);
  assert.equal(rows[2].text, "before July 2024");
});

test("the small charts draw the series they are given and nothing without one", () => {
  const flow = sparkline({ kind: "flow", observed: [["2024-11-25", 400], ["2024-11-26", 433]], modelled: [["2024-11-25", 100], ["2024-11-26", 112]] },
    { marker: "2024-11-26" });
  assert.match(flow, /class="rp-obs"/);
  assert.match(flow, /class="rp-mod"/);
  assert.match(flow, /433 m³\/s/);
  assert.match(sparkline({ kind: "rain", values: [["a", 2], ["b", 42]] }), /42 mm/);
  assert.equal(sparkline(null), "");
  assert.match(forecastBar({ median_cms: 63, low_cms: 48, high_cms: 111 }, 433), /gauge 433/);
  assert.equal(forecastBar({ lead_days: 7 }, 433), "");
});
