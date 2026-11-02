import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  PLACE_CREDIT, WATER_TAGS, isWater, normalizePlace, photonReverseUrl, photonSearchUrl, placeScore, placesFirst,
  pointLabel, rankPlaces, riverLabel, riverName,
} from "../src/place-search-core.js";

// Real Photon answers (2026-10-12), trimmed: see the fixture's _note.
const P = JSON.parse(readFileSync(new URL("./fixtures/photon-2026-10-12.json", import.meta.url), "utf8"));
// The #637 cases: a pond nearer than the Rhine, a lake with no river near, and a fountain at Basel.
const R = JSON.parse(readFileSync(new URL("./fixtures/photon-637-river-card.json", import.meta.url), "utf8"));

test("search URLs: anything, or water only", () => {
  const any = new URL(photonSearchUrl(" Jamuna "));
  assert.equal(any.origin + any.pathname, "https://photon.komoot.io/api/");
  assert.equal(any.searchParams.get("q"), "Jamuna");
  assert.equal(any.searchParams.get("lang"), "en");
  assert.deepEqual(any.searchParams.getAll("osm_tag"), []);
  const water = new URL(photonSearchUrl("Jamuna", { water: true, limit: 8 }));
  assert.deepEqual(water.searchParams.getAll("osm_tag"), WATER_TAGS);
  assert.equal(water.searchParams.get("limit"), "8");
});

test("reverse URL rounds the point and carries the radius", () => {
  const u = new URL(photonReverseUrl(24.351234, 89.80006, { water: true, radiusKm: 5 }));
  assert.equal(u.origin + u.pathname, "https://photon.komoot.io/reverse");
  assert.equal(u.searchParams.get("lat"), "24.3512");
  assert.equal(u.searchParams.get("lon"), "89.8001");
  assert.equal(u.searchParams.get("radius"), "5");
  assert.deepEqual(u.searchParams.getAll("osm_tag"), WATER_TAGS);
  assert.deepEqual(new URL(photonReverseUrl(1, 2)).searchParams.getAll("osm_tag"), []);
});

test("water is a waterway or natural=water, not a place called after one", () => {
  assert.equal(isWater({ osm_key: "waterway", osm_value: "river" }), true);
  assert.equal(isWater({ osm_key: "natural", osm_value: "water" }), true);
  assert.equal(isWater({ osm_key: "place", osm_value: "village" }), false);
  assert.equal(isWater({ osm_key: "boundary", osm_value: "administrative" }), false);
  assert.equal(isWater(null), false);
});

test("Jamuna: the river in Bangladesh comes before the districts and villages of that name", () => {
  const ranked = rankPlaces([...P.jamuna_any, ...P.jamuna_water], "Jamuna");
  assert.equal(ranked[0].name, "Jamuna River");
  assert.equal(ranked[0].country, "Bangladesh");
  assert.equal(ranked[0].river, true);
  assert.equal(ranked[0].whole, true, "the whole river (the OSM relation), not one stretch");
  assert.deepEqual([ranked[0].lat, ranked[0].lon], [24.5025354, 89.7193122]);
  assert.equal(ranked[0].detail, "Rajshahi Division, Bangladesh");
  // One result per name and country: the Bangladesh stretches fold into the whole river.
  assert.equal(ranked.filter((p) => p.name === "Jamuna River" && p.country === "Bangladesh").length, 1);
  // The other rivers of that name come next, ahead of every village and district.
  const firstLand = ranked.findIndex((p) => !p.water);
  const lastWater = ranked.map((p) => p.water).lastIndexOf(true);
  assert.ok(firstLand === -1 || lastWater < firstLand);
});

test("Jamuna, the general search alone: the river still wins over the villages", () => {
  const ranked = rankPlaces(P.jamuna_any, "jamuna", 3);
  assert.equal(ranked.length, 3);
  assert.equal(ranked[0].name, "Jamuna River");
});

test("Rhine: the river, then the canals, then the state", () => {
  const ranked = rankPlaces([...P.rhine_any, ...P.rhine_water], "Rhine");
  assert.equal(ranked[0].name, "Rhine");
  assert.equal(ranked[0].country, "Germany");
  assert.equal(ranked[0].whole, true);
  assert.ok(ranked.findIndex((p) => p.name === "Rhineland-Palatinate") > ranked.findIndex((p) => p.name === "Rhine Canal"));
});

test("a town of the name asked beats a river that only contains it", () => {
  const town = { properties: { osm_key: "place", osm_value: "city", name: "Thames", country: "New Zealand" }, geometry: { coordinates: [175.54, -37.14] } };
  const canal = { properties: { osm_key: "waterway", osm_value: "canal", osm_type: "W", name: "Thames and Medway Canal", country: "United Kingdom" }, geometry: { coordinates: [0.46, 51.43] } };
  const ranked = rankPlaces([canal, town], "Thames");
  assert.equal(ranked[0].name, "Thames");
  assert.ok(placeScore(normalizePlace(town), "Thames") > placeScore(normalizePlace(canal), "Thames"));
});

test("features without a name or a point are dropped; limit holds", () => {
  assert.equal(normalizePlace({ properties: { osm_key: "waterway" }, geometry: { coordinates: [1, 2] } }), null);
  assert.equal(normalizePlace({ properties: { name: "X" }, geometry: { coordinates: [] } }), null);
  assert.deepEqual(rankPlaces(null, "x"), []);
  assert.equal(rankPlaces([...P.rhine_any, ...P.rhine_water], "Rhine", 2).length, 2);
});

test("a point named by the nearest water, else by the nearest town", () => {
  assert.deepEqual(pointLabel(P["reverse_water_24.35_89.8"], P["reverse_any_24.35_89.8"]),
    { name: "Jamuna River", detail: "Sirajganj, Rajshahi Division, Bangladesh", water: true });
  assert.deepEqual(pointLabel([], P["reverse_any_24.35_89.8"]),
    { name: "Near Elenga", detail: "Dhaka Division, Bangladesh", water: false });
  assert.equal(pointLabel([], []), null);
  assert.equal(pointLabel(null, null), null);
});

test("a name with a digit (a postcode, a road number) is not a name", () => {
  const road = [{ properties: { osm_key: "waterway", osm_value: "canal", name: "Canal 12" } }];
  const any = [{ properties: { name: "KT1 4AA", city: "London", state: "England", country: "United Kingdom" } }];
  assert.deepEqual(pointLabel(road, any), { name: "Near London", detail: "England, United Kingdom", water: false });
});

test("the credit is the package's", () => {
  assert.equal(PLACE_CREDIT, "Places: Photon by komoot, data © OpenStreetMap contributors (ODbL)");
});

test("a pond nearer the point names the place; the river behind it names the reach (#637)", () => {
  const water = R.rhine_galgenbergsee_water;
  // The place card keeps the nearest named water's name...
  assert.deepEqual(pointLabel(water, []),
    { name: "Galgenbergsee", detail: "Baden-Württemberg, Germany", water: true });
  // ...and the river is read separately, for the River card and Route it's title.
  assert.deepEqual(riverLabel(water),
    { name: "Rhine", detail: "Baden-Württemberg, Germany", water: true, river: true });
  assert.equal(riverName({ ...pointLabel(water, []), river: riverLabel(water) }), "Rhine");
  assert.equal(riverName(pointLabel(water, [])), "", "a label without a river names no reach");
});

test("a lake with no river near it is not a river (#637)", () => {
  const lake = R.bodensee_water;
  assert.equal(riverLabel(lake), null);
  // The nearest water is still the place's name — a lake is not relabelled a river.
  assert.deepEqual(pointLabel(lake, []),
    { name: "Lake Constance", detail: "Baden-Württemberg, Germany", water: true });
  assert.equal(riverName({ ...pointLabel(lake, []), river: riverLabel(lake) }), "");
});

test("a river alone still names both the place and the reach (#637)", () => {
  assert.deepEqual(pointLabel(R.rhine_alone_water, []),
    { name: "Rhine", detail: "Baden-Württemberg, Germany", water: true });
  assert.equal(riverLabel(R.rhine_alone_water).name, "Rhine");
});

test("a fountain is never a river, though it can name the place (#637)", () => {
  const label = pointLabel([], R.basel_fountain_any);
  assert.deepEqual(label, { name: "Near Fischmarktbrunnen", detail: "Basel-Stadt, Switzerland", water: false });
  assert.equal(riverLabel(R.basel_fountain_any), null, "an amenity=fountain is not a waterway");
  assert.equal(riverName({ ...label, river: riverLabel(R.basel_fountain_any) }), "");
  // A canal or a stream is riverish too, a pond or a fountain is not.
  assert.equal(riverLabel([{ properties: { osm_key: "waterway", osm_value: "stream", name: "Bächle" } }]).name, "Bächle");
  assert.equal(riverLabel([{ properties: { osm_key: "natural", osm_value: "water", name: "Weiher" } }]), null);
  assert.equal(riverName(null), "");
});

test("Rhine goes above gauges that only contain the word; a gauge on the river keeps the gauges first", () => {
  const rhine = rankPlaces([...P.rhine_any, ...P.rhine_water], "Rhine")[0];
  assert.equal(placesFirst("Rhine", ["PELICAN RIVER NEAR RHINELANDER, WI", "LEE VINING C BL RHINEDOLLAR DAM"], rhine), true);
  assert.equal(placesFirst("Rhine", ["Rhein at Basel", "RHINE AT LOBITH"], rhine), false);
  assert.equal(placesFirst("rhine", [], rhine), true);
  // A place that is not named as asked does not jump the queue.
  const canal = rankPlaces(P.rhine_water, "Rhine Canal")[0];
  assert.equal(placesFirst("Rhin", [], canal), false);
  assert.equal(placesFirst("Rhine", [], null), false);
  const jamuna = rankPlaces(P.jamuna_water, "Jamuna")[0];
  assert.equal(placesFirst("Jamuna", [], jamuna), true, "'Jamuna' asks for the Jamuna River");
});
