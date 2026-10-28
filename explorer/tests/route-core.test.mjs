// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  RAMP, busiestFrame, cardPlaces, flatterWords, flowWords, frameFeatureStates, frameLabel, frameStates, headline, hoursWords, logLabel,
  pathBounds, pathFilterExpr, peakShares, placeFeatures, reachedBounds, routeArgs, townsFrom, waveColorExpr,
  waveWidthExpr,
} from "../src/route-core.js";
import { slotOf } from "../src/map-actions-core.js";

// A small answer shaped like aquascope.routing.route's.
const RESULT = {
  ok: true, river_id: 230301988, hours: 60,
  inflow: { peak_cms: 2000, duration_h: 24, shape: "triangle", return_period: null },
  reaches: [
    { river_id: 1, peak_cms: 2000, peak_h: 8, front_h: 4, attenuation_pct: 0, reached: true, along_km: 1 },
    { river_id: 2, peak_cms: 1500, peak_h: 30, front_h: 22, attenuation_pct: 25, reached: true, along_km: 60 },
    { river_id: 3, peak_cms: 900, peak_h: null, front_h: 50, attenuation_pct: 55, reached: true, along_km: 140 },
  ],
  furthest: { river_id: 2, index: 1, peak_cms: 1500, peak_h: 30, front_h: 22, attenuation_pct: 25, along_km: 60 },
  frames: { t_h: [0, 10, 20, 30], ids: [1, 2, 3], share: [[0, 0, 0], [90, 10, 0], [40, 60, 5], [5, 75, 30]] },
  geometry: { type: "LineString", coordinates: [[7.6, 47.5], [7.6, 47.9], [7.7, 48.5], [7.8, 49.5]] },
  places: [
    { kind: "gauge", name: "Rheinweiler", lat: 47.7, lon: 7.53, along_km: 22, peak_h: 19, peak_cms: 1730, attenuation_pct: 13, reached: true },
    { kind: "town", name: "Strasbourg", lat: 48.58, lon: 7.75, along_km: 130, peak_h: 60, peak_cms: 1440, attenuation_pct: 28, reached: true },
    { kind: "town", name: "Far", lat: 50, lon: 8, along_km: 400, peak_h: null, peak_cms: 10, attenuation_pct: 99, reached: false },
  ],
};

test("the wave is coloured and widened by its share, from feature-state", () => {
  const c = waveColorExpr();
  assert.equal(c[0], "interpolate");
  assert.deepEqual(c[2], ["coalesce", ["feature-state", "rw"], 0]);
  assert.equal(c.at(-1), RAMP.at(-1)[1]);
  assert.equal(waveWidthExpr()[0], "interpolate");
  assert.deepEqual(pathFilterExpr([3, 1, 3]), ["match", ["to-number", ["get", "riverId"]], [3, 1], true, false]);
  assert.deepEqual(pathFilterExpr([]), ["boolean", false]);
});

test("hours, flows and attenuation are said plainly", () => {
  assert.equal(hoursWords(6.4), "6 h");
  assert.equal(hoursWords(52), "2 d 4 h");
  assert.equal(hoursWords(72), "3 days");
  assert.equal(hoursWords(340), "14.2 days");
  assert.equal(hoursWords(null), "n/a");
  assert.equal(frameLabel(30), "+30 h");
  assert.equal(flowWords(1234.4), "1,234");
  assert.equal(flowWords(38.2), "38");
  assert.equal(flowWords(4.04), "4");
  assert.equal(flatterWords(44.6), "−45%");
  assert.equal(flatterWords(0), "0%");
});

test("frames: each reach's share at a step, the peak everywhere, the busiest step", () => {
  assert.deepEqual(frameStates(RESULT.frames, 2), [[1, 40], [2, 60], [3, 5]]);
  assert.deepEqual(peakShares(RESULT.frames), [90, 75, 30]);
  assert.deepEqual(frameStates(RESULT.frames, 0, { peak: true }), [[1, 90], [2, 75], [3, 30]]);
  assert.equal(busiestFrame(RESULT.frames), 3);
  assert.deepEqual(frameFeatureStates(RESULT.frames, 2)[1], [2, { rw: 60, rp: 75 }]);
});

test("the places: labels on the map and the card's list", () => {
  const fc = placeFeatures(RESULT.places);
  assert.equal(fc.features.length, 3);
  assert.equal(fc.features[1].properties.text, "Strasbourg · peak 2 d 12 h · 1,440 m³/s");
  assert.equal(fc.features[2].properties.text, "Far · too small to matter");
  assert.deepEqual(cardPlaces(RESULT.places).map((p) => p.name), ["Rheinweiler", "Strasbourg"]);
  // Spread along the stretch, a city before a town before a gauge where they stand close together.
  const many = [0, 1, 2, 50, 51, 100, 150, 200, 201, 300].map((km, i) => ({ kind: i === 1 ? "city" : i % 2 ? "town" : "gauge",
    name: `P${i}`, along_km: km, peak_h: km, peak_cms: 1, reached: true, lat: 0, lon: 0 }));
  const picked = cardPlaces(many, 4).map((p) => p.name);
  assert.equal(picked.length, 4);
  assert.equal(picked[0], "P1");
  const fc2 = placeFeatures(many, cardPlaces(many, 4));
  assert.equal(fc2.features.filter((f) => f.properties.text).length, 4);
});

test("the headline and the log line come from the package's numbers", () => {
  assert.equal(headline(RESULT), "A flood peaking at 2,000 m³/s over 24 h. 30 h later its peak is 60 km down at 1,500 m³/s, −25% on the start.");
  assert.equal(logLabel(RESULT), "Route 2,000 m³/s");
  assert.equal(logLabel({ inflow: { return_period: 10 } }), "Route the 10-year flood");
  assert.equal(slotOf({ type: "route" }), "route");
});

test("framing: the whole path, or the stretch the peak reached", () => {
  assert.deepEqual(pathBounds(RESULT), [[7.6, 47.5], [7.8, 49.5]]);
  const part = reachedBounds(RESULT);
  assert.ok(part[1][1] < 49.5 && part[1][1] >= 47.9);
});

test("the panel's inputs are checked before anything runs", () => {
  assert.deepEqual(routeArgs({ kind: "peak", peak: "500", duration: "24", block: true }),
    { duration_h: 24, shape: "block", peak_cms: 500 });
  assert.deepEqual(routeArgs({ kind: "rp", rp: "100", duration: "12", block: false }),
    { duration_h: 12, shape: "triangle", return_period: 100 });
  assert.throws(() => routeArgs({ kind: "peak", peak: "", duration: "24" }), /peak flow/);
  assert.throws(() => routeArgs({ kind: "rp", rp: "10", duration: "0" }), /duration/);
});

test("towns come from the basemap's place labels, cities and towns only, once each", () => {
  const f = (name, cls, lon, lat) => ({ properties: { name, class: cls }, geometry: { type: "Point", coordinates: [lon, lat] } });
  const got = townsFrom([f("Köln", "city", 6.96, 50.94), f("Köln", "city", 6.96, 50.94), f("Hamlet", "hamlet", 7, 50),
    { properties: { name: "Bonn", class: "town", "name:latin": "Bonn" }, geometry: { type: "Point", coordinates: [7.1, 50.73] } }]);
  assert.deepEqual(got.map((t) => t.name), ["Köln", "Bonn"]);
  assert.equal(got[0].kind, "city");
  assert.equal(got[1].kind, "town");
});
