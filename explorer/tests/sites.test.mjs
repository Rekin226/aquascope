import test from "node:test";
import assert from "node:assert/strict";
import { colocatedOffsets, siteKey } from "../src/sites.js";
import { toFeatureCollection } from "../src/catalog.js?v=__BUILD__";
import { state } from "../src/core.js?v=__BUILD__";

const pair = [
  { source: "hubeau_hydrometrie", station_id: "A891030101", site_id: "A8910301", lat: 48, lon: 7 },
  { source: "hubeau_hydrometrie", station_id: "A891030102", site_id: "A8910301", lat: 48, lon: 7 },
];

test("site identity includes the source and defaults to the station ID", () => {
  assert.equal(siteKey(pair[0]), siteKey(pair[1]));
  assert.notEqual(siteKey(pair[0]), siteKey({ ...pair[0], source: "other" }));
  assert.equal(siteKey({ source: "x", station_id: "one" }), siteKey({ source: "x", station_id: "one", site_id: "" }));
});

test("offsets separate overlapping icons deterministically without changing coordinates", () => {
  const before = structuredClone(pair);
  const first = colocatedOffsets(pair);
  assert.deepEqual(first, colocatedOffsets([...pair].reverse()));
  const [a, b] = [...first.values()];
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) >= 24 - 1e-8);
  assert.deepEqual(pair, before);
  assert.deepEqual([...colocatedOffsets([pair[0]]).values()], [[0, 0]]);
});

test("larger coordinate groups retain enough spacing for every icon", () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({ ...pair[0], station_id: String(i) }));
  const offsets = [...colocatedOffsets(rows).values()];
  for (let i = 0; i < offsets.length; i++) {
    for (let j = i + 1; j < offsets.length; j++) {
      assert.ok(Math.hypot(offsets[i][0] - offsets[j][0], offsets[i][1] - offsets[j][1]) >= 24 - 1e-8);
    }
  }
});

test("map features retain real coordinates and recompute offsets for visible sources", () => {
  const rows = [pair[0], { ...pair[1], source: "other" }];
  state.hidden.clear();
  const features = toFeatureCollection(rows).features;
  assert.equal(features.length, 2);
  assert.deepEqual(features[0].geometry.coordinates, [7, 48]);
  assert.notDeepEqual(features[0].properties.offset, features[1].properties.offset);
  state.hidden.add("other");
  const visible = toFeatureCollection(rows).features;
  assert.equal(visible.length, 1);
  assert.deepEqual(visible[0].properties.offset, [0, 0]);
  state.hidden.clear();
});
