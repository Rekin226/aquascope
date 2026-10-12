import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_WATCH, WATCH_KEY, _resetMemory, addItem, applyDigest, areaBox, areaName, digestRequest, findItem, isWatched, makeItem,
  normaliseThreshold, readWatch, removeItem, setThreshold, shouldGreet, thresholdChoice, thresholdLabel, toggleItem,
  watchId,
} from "../src/watch-core.js?v=__BUILD__";

function fakeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    raw: m,
  };
}

// Storage that refuses everything, as in some private windows.
const denied = {
  getItem() { throw new Error("SecurityError"); },
  setItem() { throw new Error("SecurityError"); },
};

const gauge = () => makeItem("gauge", { source: "usgs", station_id: "USGS-01646500", name: "Potomac", lat: 38.94971,
  lon: -77.12753, river_id: 760021611 }, "2026-10-01");
const reach = () => makeItem("reach", { river_id: 760021611, lat: 38.9, lon: -77.1 }, "2026-10-01");
const area = () => makeItem("area", { bbox: [-77.5, 38.1, -76.8, 39.0] }, "2026-10-01");

test("ids are the ones aquascope.watch.parse_item gives", () => {
  assert.equal(gauge().id, "usgs/USGS-01646500");
  assert.equal(reach().id, "river:760021611");
  assert.equal(area().id, "area:-77.5,38.1,-76.8,39");
  assert.equal(watchId({ kind: "area", bbox: [-122.41941, 37.0, -0.00001, 38.123456] }), "area:-122.4194,37,0,38.1235");
  assert.equal(watchId({ kind: "lake" }), null);
  assert.equal(gauge().lat, 38.9497);
  assert.equal(reach().name, "River reach 760021611");
  assert.equal(area().name, areaName([-77.5, 38.1, -76.8, 39.0]));
});

test("watch, persist, dedupe, toggle and unwatch", () => {
  _resetMemory();
  const s = fakeStorage();
  assert.deepEqual(readWatch(s).items, []);
  addItem(gauge(), s);
  addItem(reach(), s);
  addItem(gauge(), s);
  assert.equal(readWatch(s).items.length, 2);
  assert.equal(readWatch(s).items[0].id, "river:760021611", "newest first");
  assert.ok(isWatched("usgs/USGS-01646500", s));
  assert.equal(JSON.parse(s.raw.get(WATCH_KEY)).items.length, 2);
  toggleItem(area(), s);
  assert.ok(isWatched(area().id, s));
  toggleItem(area(), s);
  assert.ok(!isWatched(area().id, s));
  removeItem("usgs/USGS-01646500", s);
  assert.deepEqual(readWatch(s).items.map((i) => i.id), ["river:760021611"]);
});

test("the page keeps working when storage is blocked", () => {
  _resetMemory();
  assert.deepEqual(readWatch(denied).items, []);
  addItem(gauge(), denied);
  assert.ok(isWatched("usgs/USGS-01646500", denied), "kept in memory");
  setThreshold("usgs/USGS-01646500", { return_period: 10 }, denied);
  assert.deepEqual(findItem("usgs/USGS-01646500", denied).threshold, { return_period: 10 });
  assert.deepEqual(readWatch(null).items.length, 1, "no storage at all");
});

test("a write that fails keeps the memory copy, and corrupt storage is ignored", () => {
  _resetMemory();
  const s = fakeStorage();
  addItem(gauge(), s);
  const full = { getItem: s.getItem, setItem() { throw new Error("QuotaExceededError"); } };
  addItem(reach(), full);
  assert.equal(readWatch(full).items.length, 2);
  _resetMemory();
  addItem(gauge(), fakeStorage());
  const bad = fakeStorage();
  bad.setItem(WATCH_KEY, "{not json");
  assert.equal(readWatch(bad).items.length, 1, "the memory copy, not a crash");
  _resetMemory();
  const odd = fakeStorage();
  odd.setItem(WATCH_KEY, JSON.stringify({ items: [{ kind: "gauge", id: "x" }, { kind: "reach", river_id: 5, id: "river:5" }] }));
  assert.deepEqual(readWatch(odd).items.map((i) => i.id), ["river:5"], "items whose id does not match are dropped");
});

test("the list is capped", () => {
  _resetMemory();
  const s = fakeStorage();
  for (let i = 0; i < MAX_WATCH + 5; i++) addItem(makeItem("reach", { river_id: i }), s);
  assert.equal(readWatch(s).items.length, MAX_WATCH);
});

test("thresholds", () => {
  assert.deepEqual(normaliseThreshold({ return_period: "10" }), { return_period: 10 });
  assert.deepEqual(normaliseThreshold({ value: "300.5" }), { value: 300.5 });
  assert.equal(normaliseThreshold({ value: "" }), null);
  assert.equal(normaliseThreshold({ return_period: 0.5 }), null);
  assert.equal(normaliseThreshold(null), null);
  assert.equal(thresholdChoice(null), "rp:2");
  assert.equal(thresholdChoice({ return_period: 25 }), "rp:25");
  assert.equal(thresholdChoice({ value: 3 }), "value");
  assert.equal(thresholdLabel(null), "the 2-year flow");
  assert.equal(thresholdLabel({ return_period: 10 }), "the 10-year flow");
  assert.equal(thresholdLabel({ value: 300 }), "300 m³/s");
  _resetMemory();
  const s = fakeStorage();
  addItem(gauge(), s);
  setThreshold("usgs/USGS-01646500", { value: 300 }, s);
  assert.deepEqual(findItem("usgs/USGS-01646500", s).threshold, { value: 300 });
  setThreshold("usgs/USGS-01646500", null, s);
  assert.equal(findItem("usgs/USGS-01646500", s).threshold, null);
});

test("the digest request says when each item was last seen", () => {
  const st = { v: 1, lastVisit: "2026-10-03", seen: { "river:760021611": { date: "2026-10-05", class: "above" } },
    items: [gauge(), reach(), { ...area(), added: "2026-10-06" }, { ...makeItem("reach", { river_id: 9 }), added: "2026-10-08" }] };
  const req = digestRequest(st, "2026-10-08");
  assert.deepEqual(req.last_seen["usgs/USGS-01646500"], { date: "2026-10-03" });
  assert.deepEqual(req.last_seen["river:760021611"], { date: "2026-10-05", class: "above" });
  assert.deepEqual(req.last_seen[area().id], { date: "2026-10-06" }, "added after the last visit");
  assert.deepEqual(req.last_seen["river:9"], { date: "2026-10-08" }, "added today");
  assert.equal(req.today, "2026-10-08");
  const g = req.items.find((i) => i.kind === "gauge");
  assert.deepEqual(Object.keys(g).sort(), ["id", "kind", "lat", "lon", "name", "river_id", "source", "station_id", "threshold"]);
  assert.deepEqual(req.items.find((i) => i.kind === "area").bbox, [-77.5, 38.1, -76.8, 39]);
});

test("a digest's seen states are kept, and the visit counts once every item answered", () => {
  const st = { v: 1, lastVisit: "2026-10-01", seen: {}, items: [gauge(), reach()] };
  const part = applyDigest(st, [{ id: "usgs/USGS-01646500", seen: { date: "2026-10-08", class: "normal" } },
    { id: "river:760021611", error: "boom", seen: { date: "x" } }, { id: "gone", seen: { date: "y" } }],
  { complete: false, today: "2026-10-08" });
  assert.deepEqual(part.seen, { "usgs/USGS-01646500": { date: "2026-10-08", class: "normal" } });
  assert.equal(part.lastVisit, "2026-10-01");
  assert.equal(applyDigest(part, [], { today: "2026-10-08" }).lastVisit, "2026-10-08");
  assert.equal(st.seen["usgs/USGS-01646500"], undefined, "the input is not changed");
});

test("the greeting shows once a day, and only with something watched", () => {
  assert.equal(shouldGreet({ items: [], lastVisit: null }, "2026-10-08"), false);
  assert.equal(shouldGreet({ items: [gauge()], lastVisit: "2026-10-07" }, "2026-10-08"), true);
  assert.equal(shouldGreet({ items: [gauge()], lastVisit: "2026-10-08" }, "2026-10-08"), false);
});

test("a box drawn on a wrapped map is brought back to -180..180", () => {
  assert.deepEqual(areaBox([200, 10, 210, 20]), [-160, 10, -150, 20]);
  assert.deepEqual(areaBox([-77.5, 38.1, -76.8, 39]), [-77.5, 38.1, -76.8, 39]);
  assert.deepEqual(areaBox([170, 0, 190, 5]), [170, 0, -170, 5]);
  assert.deepEqual(areaBox([-200, 0, 200, 5]), [-180, 0, 180, 5]);
  assert.equal(makeItem("area", { bbox: [-400, 0, -390, 5] }).id, "area:-40,0,-30,5");
});
