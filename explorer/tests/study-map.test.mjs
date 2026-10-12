// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  boundsOf, drawnOnMap, featuresFromArtifact, featuresFromWorkspace, highlightFilters, stepsOnMap, stepsOnMapHtml,
  studyLayers,
} from "../src/study-map-data.js";

const pt = (lon, lat, props) => ({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: props });
const box = (w, s, e, n, props) => ({
  type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }, properties: props,
});

const WS = {
  site: { lat: 51.415, lon: -0.308 },
  run: { results: [
    { id: "s1", tool: "describe_catchment", map: { type: "FeatureCollection", features: [pt(-0.308, 51.415, { role: "catchment", label: "Catchment", step_id: "s1" })] } },
    { id: "s2", tool: "analyze_station", result: {} },
    { id: "s3", tool: "anywhere", map: { type: "FeatureCollection", features: [
      box(-0.375, 51.375, -0.125, 51.625, { role: "grid_cell", label: "ERA5 cell", step_id: "s3" }),
      box(-0.35, 51.4, -0.3, 51.45, { role: "grid_cell", label: "GloFAS cell", step_id: "s3" }),
      { type: "Feature", properties: {} },
    ] } },
    { id: "s4", tool: "similar_basins", map: { features: [pt(-1, 52, { role: "donor", step_id: "s4" }), pt(-2, 53, { role: "donor", step_id: "s4" })] } },
  ] },
};

test("the workspace's study map is the site, then each step's features as the engine placed them", () => {
  const fc = featuresFromWorkspace(WS);
  assert.equal(fc.type, "FeatureCollection");
  assert.deepEqual(fc.features.map((f) => f.properties.role), ["site", "catchment", "grid_cell", "grid_cell", "donor", "donor"]);
  assert.deepEqual(fc.features[0].geometry.coordinates, [-0.308, 51.415]);
  assert.deepEqual(featuresFromWorkspace({ site: WS.site }).features, [], "no step placed anything: nothing, not a lone site");
  assert.deepEqual(featuresFromWorkspace(null).features, []);
});

test("a streamed study_map.geojson artifact decodes to its features; anything else is not the map", () => {
  const fc = { type: "FeatureCollection", features: [pt(1, 2, { role: "gauge", step_id: "s2", label: "Kingstön" }), null] };
  const data = Buffer.from(JSON.stringify(fc), "utf-8").toString("base64");
  const decode = (b64) => Buffer.from(b64, "base64").toString("latin1");
  const got = featuresFromArtifact({ id: "study-map", data }, decode);
  assert.equal(got.features.length, 1);
  assert.equal(got.features[0].properties.label, "Kingstön", "UTF-8 survives the base64 round trip");
  assert.equal(featuresFromArtifact({ id: "fig-s1-series", data }, decode), null);
  assert.equal(featuresFromArtifact({ id: "study-map" }, decode), null, "no bytes, no map");
  assert.equal(featuresFromArtifact({ id: "study-map", data: "not json" }, decode), null);
});

test("the bounds of the study, and of one step", () => {
  const fc = featuresFromWorkspace(WS);
  assert.deepEqual(boundsOf(fc.features), [-2, 51.375, -0.125, 53]);
  assert.deepEqual(boundsOf(fc.features, "s3"), [-0.375, 51.375, -0.125, 51.625]);
  assert.deepEqual(boundsOf(fc.features, "s1"), [-0.308, 51.415, -0.308, 51.415], "a point is its own bounds");
  assert.equal(boundsOf(fc.features, "s9"), null);
  assert.equal(boundsOf([]), null);
});

test("the site and donor scatter figures are left to the map; the others stay", () => {
  assert.equal(drawnOnMap({ id: "fig-s1-site_map" }), true);
  assert.equal(drawnOnMap({ id: "fig-s5-donors_map-svg" }), true);
  assert.equal(drawnOnMap({ id: "fig-s3-frequency_curve" }), false);
  assert.equal(drawnOnMap({ id: "x", meta: { kind: "site_map" } }), true);
  assert.equal(drawnOnMap({ id: "fig-s1-site_map", meta: { kind: "series" } }), false, "the kind the artifact states wins");
  assert.equal(drawnOnMap(null), false);
});

test("the layers are all study-* on one source, and the highlight follows one step", () => {
  const layers = studyLayers("study-map");
  assert.deepEqual(layers.map((l) => l.id), ["study-fill", "study-line", "study-points", "study-hl-line", "study-hl-points"]);
  assert.ok(layers.every((l) => l.source === "study-map"));
  assert.deepEqual(highlightFilters("s3").line[2], ["==", ["get", "step_id"], "s3"]);
  assert.deepEqual(highlightFilters(null).points[2], ["==", ["get", "step_id"], ""], "nothing picked, nothing lit");
});

test("the finished board lists the steps on the map, escaped, with what each placed", () => {
  assert.deepEqual(stepsOnMap(WS), [
    { id: "s1", tool: "describe_catchment", words: "catchment" },
    { id: "s3", tool: "anywhere", words: "2 grid cells" },
    { id: "s4", tool: "similar_basins", words: "2 donors" },
  ]);
  const html = stepsOnMapHtml(WS, (t) => `<${t}>`);
  assert.equal((html.match(/data-map-step=/g) || []).length, 3);
  assert.match(html, /data-map-step="s3" title="2 grid cells">s3 &lt;anywhere&gt;<\/button>/);
  assert.match(html, /On the map:/);
  assert.equal(stepsOnMapHtml({ run: { results: [] } }), "");
  assert.doesNotMatch(html, /[—–]/);
});
