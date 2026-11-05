import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { lookupWaterway, nearestWaterway } from "../src/waterway-name.js";

// Synthetic geometries, not a recorded Overpass response.
const way = (name, type, lat, extra = {}) => ({ type: "way", tags: { name, waterway: type, ...extra },
  geometry: [{ lat, lon: -0.01 }, { lat, lon: 0.01 }] });

test("prefer rivers, then choose the closest line within a type", () => {
  const elements = [way("canal", "canal", 0.00001), way("far river", "river", 0.002),
    way("near river", "river", 0.0001), way("lake", "lake", 0)];
  assert.equal(nearestWaterway(elements, 0, 0), "near river");
  assert.equal(nearestWaterway([way("Rhein", "river", 0.0001, { "name:en": "Rhine" })], 0, 0), "Rhine");
});

test("ignore non-waterways, unnamed or distant lines and malformed geometry", () => {
  assert.equal(nearestWaterway([way("pond", "lake", 0), way("distant", "river", 0.02),
    { type: "way", tags: { waterway: "river" }, geometry: [] }], 0, 0), null);
});

test("recorded Basel Overpass and Cologne OSM data name the snapped Rhine reaches", () => {
  // OSM data © OpenStreetMap contributors, ODbL. Basel: response from overpass.osm.ch,
  // 2026-11-05, around 47.560169,7.589742. Cologne: named ways extracted from an OSM
  // map API bbox, not an Overpass response; the fixture records its source and limitations.
  const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));
  const basel = load("overpass-721-basel.json"), cologne = load("osm-721-cologne.json");
  assert.equal(nearestWaterway(basel.elements, 47.560169, 7.589742), "Rhine");
  assert.equal(nearestWaterway(basel.elements, 47.5596, 7.5886), "Birsig", "the click can be closer to a tributary");
  assert.equal(nearestWaterway(cologne.elements, 50.937552, 6.965579), "Rhine");
});

test("use geometry response and fall back on HTTP failure or abort", async () => {
  let seen;
  const fetcher = async (url) => { seen = new URL(url); return { ok: true, json: async () => ({ elements: [way("Rhine", "river", 0)] }) }; };
  assert.equal(await lookupWaterway(0, 0, { fetcher }), "Rhine");
  assert.match(seen.searchParams.get("data"), /out geom/);
  assert.equal(await lookupWaterway(0, 0, { fetcher: async () => ({ ok: false }) }), null);
  const hangs = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new Error("aborted"))));
  assert.equal(await lookupWaterway(0, 0, { timeout: 10, fetcher: hangs }), null);
});
