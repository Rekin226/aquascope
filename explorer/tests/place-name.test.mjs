import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// The #637 cases: a pond nearer than the Rhine, and a fountain at Basel with nothing watery near.
const R = JSON.parse(readFileSync(new URL("./fixtures/photon-637-river-card.json", import.meta.url), "utf8"));

// Photon's reverse answers, offline: the water request (any osm_tag) gets `water`, the plain one `any`.
const asked = [];
let water = R.rhine_galgenbergsee_water;
globalThis.fetch = async (url) => {
  const u = new URL(String(url));
  asked.push(u);
  const isWaterRequest = u.searchParams.getAll("osm_tag").length > 0;
  return { ok: true, json: async () => ({ features: isWaterRequest ? water : R.basel_fountain_any }) };
};

const { namePoint } = await import("../src/place-name.js");

test("a point's label keeps the lake's name and carries the river behind it (#637)", async () => {
  water = R.rhine_galgenbergsee_water;
  const label = await namePoint(47.6601, 8.2011);
  assert.equal(label.name, "Galgenbergsee", "the place is still named by the nearest water");
  assert.equal(label.water, true);
  assert.deepEqual(label.river, { name: "Rhine", detail: "Baden-Württemberg, Germany", water: true, river: true });
  assert.equal(asked.length, 1, "one reverse request when a named water answers");
  const u = asked[0];
  assert.equal(u.searchParams.get("radius"), "5");
  assert.equal(u.searchParams.get("limit"), "5", "room for the river behind a nearer pond");
  assert.deepEqual(u.searchParams.getAll("osm_tag"), ["waterway:river", "waterway:canal", "natural:water"]);
});

test("with no water near, a fountain names the place and no reach is named after it (#637)", async () => {
  water = [];
  const label = await namePoint(47.5584, 7.5916);
  assert.equal(label.name, "Near Fischmarktbrunnen");
  assert.equal(label.river, null, "the River card and Route it keep saying \"River reach N\"");
  assert.equal(asked.length, 3, "the water request and the plain one for each of the two points");
});
