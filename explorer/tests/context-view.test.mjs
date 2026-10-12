import test from "node:test";
import assert from "node:assert/strict";
import {
  CONTEXT_LAYERS, boxProblem, contextLine, credits, fetchOrder, floodPoints, floodYears, normaliseBox, pickChart, rainYears,
  titleCase,
} from "../src/context-view.js";

const FLOODS = {
  layer: "flood_history", ok: true, summary: "2 flood events in the news within 25 km, latest 2021-05-01.",
  news: { available: true, n_events: 2, by_year: { 2021: 1, 2019: 1 },
    recent: [{ start: "2021-05-01", lat: 45.1, lon: 5.1 }, { start: "2019-11-20", lat: 45.2, lon: 5.0 }] },
  radar: { available: true, months: { "2018-03": 40, "2018-04": 2, "2021-05": 12 } },
  sources: [{ key: "groundsource", label: "G", attribution: "Groundsource, CC BY 4.0", licence: "CC-BY-4.0" }],
};
const RAIN = {
  layer: "rain_gauge", ok: true, summary: "Nearest rain gauge: X.", station: { id: "RAIN1", name: "Grenoble" },
  record: { annual_mm: { 2022: 900.5, 2021: 1010.2 } },
  sources: [{ key: "rain_gauge", label: "R", attribution: "NOAA GHCN-Daily", licence: "CC0-1.0" }],
};

test("the layers are listed once each, in the engine's names", () => {
  const ids = CONTEXT_LAYERS.map((l) => l.id);
  assert.deepEqual(ids, ["flood_history", "surface_water", "flood_hazard", "dams", "rain_gauge", "actual_et", "soil"]);
});

test("every layer is asked for once, quickest first, with Soil started early when workers run side by side", () => {
  const ids = CONTEXT_LAYERS.map((l) => l.id).sort();
  for (const n of [1, 3]) assert.deepEqual(fetchOrder(n).slice().sort(), ids);
  assert.equal(fetchOrder(1)[0], "dams");
  assert.equal(fetchOrder(1).at(-1), "soil");
  assert.deepEqual(fetchOrder(3).slice(0, 3), ["dams", "soil", "rain_gauge"]);
});

test("a line says loading, the summary, an error, or a quiet not-yet", () => {
  assert.equal(contextLine("dams", undefined).state, "loading");
  assert.deepEqual(contextLine("flood_history", FLOODS), { id: "flood_history", label: "Flood history", state: "ok", text: FLOODS.summary });
  assert.equal(contextLine("soil", { error: "worker died" }).state, "error");
  assert.equal(contextLine("soil", { ok: false, summary: "Could not read this layer: 404" }).text, "Could not read this layer: 404");
  const unpublished = { ok: true, available: false, summary: "Dams are not available yet (the mirror is not published)." };
  assert.equal(contextLine("dams", unpublished).state, "empty");
});

test("the chart prefers flood events by year, then radar months, then the rain gauge", () => {
  assert.deepEqual(floodYears(FLOODS), { x: ["2019", "2021"], y: [1, 1], what: "flood events in the news", unit: "events" });
  const radarOnly = { ...FLOODS, news: { available: false } };
  assert.deepEqual(floodYears(radarOnly).y, [2, 1]);
  assert.equal(floodYears({ news: { available: false }, radar: { available: false } }), null);
  assert.deepEqual(rainYears(RAIN).x, ["2021", "2022"]);
  assert.equal(pickChart({ flood_history: FLOODS, rain_gauge: RAIN }).unit, "events");
  assert.equal(pickChart({ flood_history: { news: { available: false } }, rain_gauge: RAIN }).unit, "mm");
  assert.equal(pickChart({}), null);
});

test("flood events become map points, from a point's list or an area's", () => {
  const fc = floodPoints(FLOODS);
  assert.equal(fc.features.length, 2);
  assert.deepEqual(fc.features[0].geometry.coordinates, [5.1, 45.1]);
  const area = floodPoints({ points: [[45, 5, "2020-01-01"], [null, 5, "bad"]] });
  assert.equal(area.features.length, 1);
  assert.equal(floodPoints(null).features.length, 0);
});

test("credits list each source once", () => {
  const c = credits({ flood_history: FLOODS, rain_gauge: RAIN, again: FLOODS, broken: { error: "x" } });
  assert.deepEqual(c.map((x) => x.licence), ["CC-BY-4.0", "CC0-1.0"]);
});

test("a drawn box is folded onto the globe and checked", () => {
  assert.deepEqual(normaliseBox({ west: 365, south: 44, east: 366, north: 46 }), { west: 5, south: 44, east: 6, north: 46 });
  const across = normaliseBox({ west: 170, south: -10, east: 190, north: 10 });
  assert.match(boxProblem(across), /date line/);
  assert.match(boxProblem({ west: 0, south: 0, east: 30, north: 10 }), /smaller box/);
  assert.equal(boxProblem({ west: 4, south: 44, east: 6, north: 46 }), null);
  assert.deepEqual(normaliseBox({ west: -200, south: -95, east: 200, north: 95 }), { west: -180, south: -90, east: 180, north: 90 });
});

test("GHCN's upper-case station names read as the summary line writes them", () => {
  assert.equal(titleCase("NIJMEGEN"), "Nijmegen");
  assert.equal(titleCase("DE BILT"), "De Bilt");
  assert.equal(titleCase("SAO PAULO-MIRANTE"), "Sao Paulo-Mirante");
  assert.equal(titleCase("Already Mixed"), "Already Mixed");
  const station = { ...RAIN, station: { id: "NLE00101989", name: "NIJMEGEN" } };
  assert.equal(rainYears(station).what, "yearly rainfall at Nijmegen");
});
