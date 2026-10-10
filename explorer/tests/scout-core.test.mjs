// Scout (#563): the view the page hands the package, the pin a finding becomes, and the line under the bar.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  GLOBE_RADIUS_KM, KIND_LABELS, PIN_GAP_PX, SCOUT_MAX, scoutLine, scoutMonth, scoutPin, scoutView, scoutWho,
} from "../src/scout-core.js";
import { checkPin } from "../src/map-actions-core.js";

test("the globe zoomed out is a circle around the centre, never the far side", () => {
  const v = scoutView({ globe: true, zoom: 1.6, center: [-30, 25], width: 1440, height: 844 });
  assert.deepEqual(v.center, [25, -30]);
  assert.equal(v.radius_km, GLOBE_RADIUS_KM);
  // Two pins closer than a pin's height on screen cannot be told apart.
  const mpp = (40075016.686 * Math.cos((25 * Math.PI) / 180)) / (512 * 2 ** 1.6);
  assert.equal(v.min_km, Math.round((mpp * PIN_GAP_PX) / 1000));
  const near = scoutView({ globe: true, zoom: 3.5, center: [10, 50], width: 400, height: 300 });
  assert.ok(near.radius_km < GLOBE_RADIUS_KM && near.radius_km >= 50);
});

test("closer in, the outline of what is on screen; then the bounds; the whole world is null", () => {
  const outline = [[-10, 35], [30, 35], [30, 60], [-10, 60]];
  assert.deepEqual(scoutView({ globe: true, zoom: 5, center: [10, 48], outline }).polygon, outline);
  assert.deepEqual(scoutView({ zoom: 5, center: [10, 48], bounds: [-10.12345, 35, 30, 60] }).bbox, [-10.123, 35, 30, 60]);
  assert.deepEqual(scoutView({ zoom: 5, center: [180, 0], bounds: [170, -5, 190, 5] }).bbox, [170, -5, -170, 5]);
  assert.equal(scoutView({ zoom: 0.5, center: [0, 0], bounds: [-200, -85, 200, 85] }), null);
  assert.equal(scoutView({ zoom: 0.5, center: [0, 0], outline: [[-180, 0], [180, 0], [180, 10]] }), null);
  assert.equal(scoutView({ zoom: 5 }), null);
});

test("the month of the map date", () => {
  assert.equal(scoutMonth("2026-09-15"), "2026-09");
  assert.equal(scoutMonth(null), null);
});

test("a finding becomes a numbered pin with its reason and where it came from", () => {
  const f = {
    lat: -24.03, lon: -50.69, title: "Paraná, Brazil: 25-year flow forecast", reason: "GEOGLOWS forecast …",
    facts: [{ label: "Forecast peak", value: 2840, unit: "m³/s" }, { label: "a" }, 1, 2, 3, 4],
    source: "GEOGLOWS v2 forecast, CC BY 4.0", rank: 1, kind: "floods_ahead", placed_by: "photon",
  };
  const pin = scoutPin(f, { mode: "daily", made: "2026-10-10T15:00:00Z" });
  assert.equal(pin.kind, "Floods ahead");
  assert.equal(pin.text, f.reason);
  assert.equal(pin.facts.length, 5);
  assert.match(pin.source, /scout file of 2026-10-10; place name: Photon/);
  const checked = checkPin(pin);
  assert.equal(checked.rank, 1);
  assert.equal(checked.kind, "Floods ahead");
  assert.equal(checked.facts.length, 1);   // the bad facts are dropped by the log's own check
  assert.match(scoutPin({ ...f, placed_by: undefined }).source, /scanned in your browser\)$/);
  assert.equal(checkPin({ lat: 0, lon: 0, title: "x", rank: 0 }).rank, undefined);
  assert.equal(checkPin({ lat: 0, lon: 0, title: "x", rank: 2.5 }).rank, undefined);
});

test("who found them, and the line", () => {
  assert.deepEqual(scoutWho("rules"), ["Rules", "no model"]);
  assert.equal(scoutWho("device", "Gemini Nano")[0], "Gemini Nano");
  assert.match(scoutWho("key", "m via p")[1], /numbers by code/);
  assert.equal(scoutLine({ mode: "daily" }, 10), "10 pins, most worth a look first, from today's scout file");
  assert.equal(scoutLine({ mode: "live" }, 1), "1 pin, most worth a look first, scanned now");
  assert.match(scoutLine({ notes: ["no Floods ahead issue could be read"] }, 0), /^Nothing stands out in this view \(no Floods/);
  assert.equal(SCOUT_MAX, 10);
  assert.deepEqual(Object.keys(KIND_LABELS), ["floods_ahead", "status", "gauges_today", "floods_past", "models_disagree"]);
});
