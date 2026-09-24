import test from "node:test";
import assert from "node:assert/strict";
import {
  PLACES_KEY, MAX_PLACES, _resetMemory, addPlace, hasPlace, placeFromStation, readPlaces, removePlace, togglePlace,
} from "../src/places.js?v=__BUILD__";
import { COMPARE_COLORS, COMPARE_DASHES, MAX_COMPARE, MIN_COMPARE, compareRequest } from "../src/compare.js?v=__BUILD__";

function fakeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    raw: m,
  };
}

// Storage that refuses everything, as in some private windows.
const deniedStorage = {
  getItem() { throw new Error("SecurityError"); },
  setItem() { throw new Error("SecurityError"); },
};

const usgs = { source: "usgs", station_id: "USGS-01013500", name: "Fish River", lat: 47.2, lon: -68.6, period_start: "1903-01-01" };
const ea = { source: "uk_ea", station_id: "39001", name: "Thames at Kingston", lat: 51.4, lon: -0.3 };

test("add, persist, dedupe and remove", () => {
  _resetMemory();
  const s = fakeStorage();
  assert.deepEqual(readPlaces(s), []);
  addPlace(placeFromStation(usgs), s);
  addPlace(placeFromStation(ea), s);
  addPlace(placeFromStation(usgs), s);
  const list = readPlaces(s);
  assert.equal(list.length, 2);
  assert.equal(list[0].station_id, "39001", "newest first");
  assert.ok(hasPlace("usgs/USGS-01013500", s));
  assert.equal(JSON.parse(s.raw.get(PLACES_KEY)).length, 2);
  removePlace("usgs/USGS-01013500", s);
  assert.ok(!hasPlace("usgs/USGS-01013500", s));
  togglePlace(placeFromStation(ea), s);
  assert.deepEqual(readPlaces(s), []);
});

test("a place keeps only what the list needs", () => {
  const p = placeFromStation({ ...usgs, variables: ["discharge"], url: "https://x" });
  assert.deepEqual(Object.keys(p).sort(), ["added", "lat", "lon", "name", "period_start", "source", "station_id"]);
});

test("the page works when storage is denied", () => {
  _resetMemory();
  addPlace(placeFromStation(usgs), deniedStorage);
  assert.ok(hasPlace("usgs/USGS-01013500", deniedStorage));
  assert.equal(readPlaces(null).length, 1);
});

test("corrupt or foreign storage does not break the list", () => {
  _resetMemory();
  const s = fakeStorage();
  s.setItem(PLACES_KEY, "{not json");
  assert.deepEqual(readPlaces(s), []);
  s.setItem(PLACES_KEY, JSON.stringify([{ source: "usgs" }, 7, placeFromStation(ea)]));
  assert.deepEqual(readPlaces(s).map((p) => p.station_id), ["39001"]);
});

test("another tab clearing the list is followed", () => {
  _resetMemory();
  const s = fakeStorage();
  addPlace(placeFromStation(usgs), s);
  s.removeItem(PLACES_KEY);
  assert.deepEqual(readPlaces(s), []);
});

test("the list is capped", () => {
  _resetMemory();
  const s = fakeStorage();
  for (let i = 0; i < MAX_PLACES + 5; i++) addPlace({ source: "usgs", station_id: String(i), name: String(i) }, s);
  assert.equal(readPlaces(s).length, MAX_PLACES);
});

test("the compare request carries the areas the page knows", () => {
  assert.equal(MIN_COMPARE, 2);
  assert.equal(MAX_COMPARE, 5);
  assert.ok(COMPARE_COLORS.length >= MAX_COMPARE && COMPARE_DASHES.length >= MAX_COMPARE);
  const areas = new Map([["usgs/USGS-01013500", { area: 2261, source: "agency" }], ["uk_ea/39001", null]]);
  const req = compareRequest([placeFromStation(usgs), placeFromStation(ea)], areas);
  assert.deepEqual(req[0], {
    source: "usgs", station_id: "USGS-01013500", label: "Fish River", period_start: "1903-01-01", area_km2: 2261,
  });
  assert.equal(req[1].area_km2, null);
  assert.equal(req[1].period_start, null);
});
