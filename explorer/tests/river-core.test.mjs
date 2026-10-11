// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  FLOW_PERIOD, FLOW_STEPS, HL, RIVER_THEMES, cumulativeKm, damFacts, damName, damsGeoJSON, flowDash, flowOpacity,
  highlightColor, highlightOpacity, highlightWidth, lineBounds, lineUpTo, networkStates, networkSummary, notableDams,
  RIVERS_ATTRIBUTION, riverOpacity, riverTheme, riverWidth, snapLine, STREAMS_PMTILES,
} from "../src/river-core.js";

test("snapLine says where the click landed, or that no stream is near", () => {
  assert.equal(snapLine({ snapped: true, river_id: 230260670, distance_m: 199.8, strahler_order: 5 }),
    "Snapped 200 m to river reach 230260670, stream order 5.");
  assert.equal(snapLine({ snapped: true, river_id: 1, distance_m: 1500 }, { gauge: true }),
    "This gauge is 1.5 km from river reach 1.");
  assert.equal(snapLine({ snapped: false, max_distance_m: 1000, nearest: { river_id: 2, distance_m: 1432 } }),
    "No stream within 1 km. The nearest mapped reach is 1.4 km away.");
  assert.equal(snapLine({ snapped: false, max_distance_m: 200, searched_m: 3000, nearest: null }),
    "No stream within 3 km of this point.");
  assert.equal(snapLine(null), "");
});

test("snapLine says when the main channel won over a nearer stream, and names a larger river further off", () => {
  const near = { river_id: 9, distance_m: 60, strahler_order: 2 };
  assert.equal(snapLine({ snapped: true, river_id: 3, distance_m: 380, strahler_order: 9, choice: "main_channel", nearer: near }),
    "Snapped 380 m to the main channel (order 9); a smaller stream is 60 m away.");
  assert.equal(snapLine({ snapped: true, river_id: 3, distance_m: 380, strahler_order: 9, choice: "area", nearer: near }, { gauge: true }),
    "This gauge is 380 m from river reach 3, stream order 9, the one whose upstream area matches the catchment (the nearest line is 60 m away).");
  assert.equal(snapLine({ snapped: true, river_id: 9, distance_m: 300, strahler_order: 2, choice: "nearest",
    larger: { river_id: 3, distance_m: 1340, strahler_order: 8 } }),
  "Snapped 300 m to river reach 9, stream order 2. A larger river (order 8) is 1.3 km away.");
  assert.equal(snapLine({ snapped: false, max_distance_m: 1000, nearest: { river_id: 2, distance_m: 1432 },
    larger: { river_id: 3, distance_m: 2100, strahler_order: 8 } }),
  "No stream within 1 km. The nearest mapped reach is 1.4 km away. A larger river (order 8) is 2.1 km away.");
});

test("lineUpTo grows the trace by distance, not by vertex", () => {
  const line = [[0, 0], [0, 1], [0, 1.001], [0, 1.002], [0, 2]];
  assert.deepEqual(lineUpTo(line, 0), [[0, 0], [0, 0]]);
  assert.deepEqual(lineUpTo(line, 1), line);
  const half = lineUpTo(line, 0.5);
  const end = half[half.length - 1];
  assert.ok(Math.abs(end[1] - 1.0) < 0.002, `half way is near lat 1, got ${end[1]}`);
  assert.ok(half.length <= 3);
  assert.deepEqual(lineUpTo([[1, 1]], 0.5), [[1, 1]]);
});

test("cumulativeKm and lineBounds", () => {
  const c = cumulativeKm([[0, 0], [0, 1], [1, 1]]);
  assert.equal(c.length, 3);
  assert.ok(Math.abs(c[1] - 111.19) < 0.1);
  assert.deepEqual(lineBounds([[3, -1], [-2, 4], [0, 0]]), [-2, -1, 3, 4]);
  assert.equal(lineBounds([]), null);
});

test("the stream layer is styled by Strahler order and points at the GEOGLOWS bucket", () => {
  const w = riverWidth();
  assert.equal(w[0], "interpolate");
  assert.ok(JSON.stringify(w).includes("strahlerOrder"));
  assert.match(STREAMS_PMTILES, /^https:\/\/geoglows-v2\.s3\.us-west-2\.amazonaws\.com\/.+streams\.pmtiles$/);
});

test("a dam is named by GDW, else by its reservoir, else plainly", () => {
  assert.equal(damName({ name: "Muehleberg" }), "Muehleberg");
  assert.equal(damName({ name: "unnamed", reservoir: "Luzern" }), "Luzern dam");
  assert.equal(damName({ name: null, reservoir: null }), "Unnamed dam");
  assert.equal(damName(null), "");
});

test("damFacts says the storage and the main use in a few words", () => {
  assert.equal(damFacts({ capacity_mcm: 2550, purpose: "Hydroelectricity" }), "2,550 million m³, hydroelectricity");
  assert.equal(damFacts({ capacity_mcm: 9.62 }), "9.6 million m³");
  assert.equal(damFacts({ capacity_mcm: null, purpose: null }), "");
  assert.equal(damFacts({ capacity_mcm: 0, purpose: "Irrigation" }), "irrigation");
});

test("damsGeoJSON keeps each dam's place in the list and skips the unplaced", () => {
  const fc = damsGeoJSON([{ name: "A", lat: 46.9, lon: 7.4 }, { name: "B", lat: null, lon: 7 }, { name: "C", lat: 47, lon: 8 }]);
  assert.equal(fc.type, "FeatureCollection");
  assert.deepEqual(fc.features.map((f) => f.properties.i), [0, 2]);
  assert.deepEqual(fc.features[1].geometry.coordinates, [8, 47]);
  assert.equal(damsGeoJSON(undefined).features.length, 0);
});

test("notableDams lists the named or storing dams and counts the unnamed weirs", () => {
  const dams = [{ name: "unnamed" }, { name: "Muehleberg", capacity_mcm: 25 }, { name: "unnamed", capacity_mcm: 9.6 },
    { name: "unnamed", reservoir: "Luzern" }, { name: "unnamed", capacity_mcm: null }];
  const { notable, others } = notableDams(dams);
  assert.equal(notable.length, 3);
  assert.equal(others, 2);
  assert.deepEqual(notableDams(undefined), { notable: [], others: 0 });
});

// ── living rivers (#545) ──────────────────────────────────────────────────────

// The value an interpolate expression gives at an input, for the plain numeric stops these tables use.
function at(expr, x) {
  const stops = expr.slice(3);
  for (let i = 0; i < stops.length; i += 2) {
    if (x <= stops[i]) {
      if (i === 0) return stops[1];
      const [x0, y0, x1, y1] = [stops[i - 2], stops[i - 1], stops[i], stops[i + 1]];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
  }
  return stops[stops.length - 1];
}
const zoomStop = (expr, z) => { const s = expr.slice(3); return s[s.indexOf(z) + 1]; };

test("the great rivers show on the globe and the small streams only close up", () => {
  const op = riverOpacity(), w = riverWidth();
  assert.equal(at(zoomStop(op, 1), 6), 0);            // order 6 hidden on the globe
  assert.ok(at(zoomStop(op, 1), 7) < at(zoomStop(op, 1), 8));
  assert.ok(at(zoomStop(op, 1), 9) > 0.8);            // order 9 clear
  assert.ok(at(zoomStop(w, 1), 10) > at(zoomStop(w, 1), 8));
  assert.ok(at(zoomStop(op, 8), 2) > 0.3);            // headwaters by zoom 8
  assert.ok(at(zoomStop(w, 12), 10) > at(zoomStop(w, 3), 10));
  const dim = riverOpacity(0.5);
  assert.equal(at(zoomStop(dim, 1), 10), at(zoomStop(op, 1), 10) / 2);
  assert.ok(JSON.stringify(flowOpacity()).includes("strahlerOrder"));
});

test("the flow dash keeps its length, walks one way and repeats every FLOW_STEPS", () => {
  const len = (a) => a.reduce((s, v) => s + v, 0);
  const dashLen = (a) => a[0] + a[2];
  for (let k = 0; k < FLOW_STEPS; k++) {
    const d = flowDash(k);
    assert.equal(d.length, 4);
    assert.ok(Math.abs(len(d) - FLOW_PERIOD) < 1e-6, `period at step ${k}`);
    assert.ok(Math.abs(dashLen(d) - 1.2) < 1e-6, `dash at step ${k}`);
    assert.ok(d.every((v) => v >= 0));
  }
  assert.deepEqual(flowDash(FLOW_STEPS + 3), flowDash(3));
  assert.deepEqual(flowDash(-1), flowDash(FLOW_STEPS - 1));
  // TDX-Hydro lines start downstream, so the dash's start moves towards 0 as the steps go on.
  assert.ok(flowDash(6)[1] < flowDash(5)[1] && flowDash(5)[0] === 0);
  assert.equal(new Set(Array.from({ length: 100 }, (_, k) => JSON.stringify(flowDash(k)))).size, FLOW_STEPS);
});

test("a lit network gives each reach one role, the clicked reach first", () => {
  const m = networkStates(5, [5, 4, 3], [5, 6, 7]);
  assert.equal(m.get(5), HL.here);
  assert.equal(m.get(4), HL.up);
  assert.equal(m.get(6), HL.down);
  assert.equal(m.size, 5);
  assert.equal(networkStates(1).size, 1);
});

test("the lit lines are styled by feature-state, blue upstream and orange to the sea", () => {
  const th = riverTheme("light");
  assert.ok(JSON.stringify(highlightWidth()).includes("feature-state"));
  assert.deepEqual(highlightOpacity(0.8), ["case", [">", ["coalesce", ["feature-state", "hl"], 0], 0], 0.8, 0]);
  const c = highlightColor(th);
  assert.equal(c[c.length - 1], th.up);
  assert.ok(c.includes(th.down));
  assert.equal(riverTheme("dark"), RIVER_THEMES.dark);
  assert.equal(riverTheme("satellite-recent"), RIVER_THEMES.imagery);
  assert.equal(riverTheme(undefined), RIVER_THEMES.light);
  for (const t of Object.values(RIVER_THEMES)) assert.notEqual(t.up, t.down);
});

test("networkSummary says Python's numbers in a few words", () => {
  const s = networkSummary({
    upstream: { n_upstream: 123896, upstream_area_km2: 1775541.6, truncated: true, min_area_km2: 1183.2 },
    downstream: { n_ids: 323, truncated: false },
  });
  assert.equal(s.up, "123,896 reaches, 1.78 million km²");
  assert.equal(s.down, "323 reaches");
  assert.equal(s.cut, "lit: reaches draining over 1,183 km²");
  const small = networkSummary({ upstream: { n_upstream: 1, upstream_area_km2: 12.4 }, downstream: { n_ids: 5000, truncated: true } });
  assert.equal(small.up, "1 reach, 12 km²");
  assert.equal(small.down, "5,000 reaches and on");
  assert.equal(small.cut, "");
  assert.deepEqual(networkSummary(null), { up: "", down: "", cut: "" });
});

test("the map's own attribution line credits the river network and its licence", () => {
  assert.match(RIVERS_ATTRIBUTION, /TDX-Hydro/);
  assert.match(RIVERS_ATTRIBUTION, /CC BY-SA 4\.0/);
  assert.match(RIVERS_ATTRIBUTION, /href="https:\/\/registry\.opendata\.aws\/geoglows-v2\/"/);
});
