// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { FLOOD_CLASSES } from "../src/floods-ahead-core.js";
import {
  BELOW, FEWS_CLASSES, classWords, colourDistances, dayOf, fewsColor, flowText, labOf, obsBefore, plumeAlt,
  plumeHead, plumeKey, plumeLabel, plumeScale, plumeSvg, pointsGeoJSON, pointsLine,
} from "../src/fews-core.js";

// What aquascope.nownext.plume returns, cut down.
const PLUME = {
  river_id: 760716396, issued: "2026-10-10",
  date: ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13"],
  median: [2213, 2000, 1755, 1578], p25: [2200, 1990, 1700, 1500], p75: [2220, 2010, 1800, 1650],
  min: [2190, 1980, 1600, 1300], max: [2240, 2050, 1900, 1900], mean: [2213, 2001, 1760, 1580],
  thresholds: { return_periods: [2, 5, 10, 25, 50, 100], q: [1290, 2030, 2520, 3140, 3590, 4050] },
  class_daily: [5, 2, 2, 2], rp: 5, peak: 2213, peak_date: "2026-10-10", first_date: "2026-10-10",
};

test("the classes are Floods ahead's, with a quiet slate below the 2-year flow", () => {
  assert.deepEqual(FEWS_CLASSES.map((c) => c.rp), [0, 2, 5, 10, 25, 50, 100]);
  assert.deepEqual(FEWS_CLASSES.slice(1), FLOOD_CLASSES);
  assert.equal(fewsColor(5), FLOOD_CLASSES[1].color);
  assert.equal(fewsColor(0), BELOW.color);
  assert.equal(fewsColor(undefined), BELOW.color);
  assert.equal(classWords(10), "the 10-year flow");
  assert.equal(classWords(0), "below the 2-year flow");
});

test("the colours stay apart under every simulated colour-vision deficiency", () => {
  const d = colourDistances();
  for (const [kind, gaps] of Object.entries(d)) {
    for (const g of gaps) assert.ok(g >= 10, `${kind}: neighbouring classes ${g.toFixed(1)} apart`);
  }
  // Lightness falls with the class for normal vision.
  const L = FLOOD_CLASSES.map((c) => labOf(c.color)[0]);
  for (let i = 1; i < L.length; i++) assert.ok(L[i] < L[i - 1]);
  // The slate is far from every class, whatever the vision.
  for (const kind of [null, "protanopia", "deuteranopia", "tritanopia"]) {
    const b = labOf(BELOW.color, kind);
    for (const c of FLOOD_CLASSES) {
      const l = labOf(c.color, kind);
      assert.ok(Math.hypot(b[0] - l[0], b[1] - l[1], b[2] - l[2]) >= 20, `${kind}: slate vs ${c.label}`);
    }
  }
});

test("the threshold lines drawn are the ones in range, plus the next when it is near", () => {
  const s = plumeScale(PLUME);
  assert.deepEqual(s.lines.map((l) => l.rp), [2, 5, 10]);   // 2,520 is within 2.2 times 2,240
  assert.equal(s.above, null);
  assert.ok(s.ymax > 2520);
  // A gauge far below its 2-year flow: the line would flatten the plume, so it is named instead.
  const low = plumeScale({ ...PLUME, median: [1, 2], p75: [1, 2], max: [1, 3], mean: [1, 2],
    thresholds: { return_periods: [2, 5], q: [40, 60] } });
  assert.deepEqual(low.lines, []);
  assert.deepEqual(low.above, { rp: 2, q: 40 });
  assert.equal(plumeScale({ ...PLUME, thresholds: null }).lines.length, 0);
  // One wet member far above the rest: the axis follows the bulk and names the range's real top.
  const wet = plumeScale({ ...PLUME, max: [2240, 2050, 1900, 30000] });
  assert.equal(wet.clipped, 30000);
  assert.ok(wet.ymax < 30000 && wet.ymax >= 2520);
  assert.match(plumeSvg({ ...PLUME, max: [2240, 2050, 1900, 30000] }), /range runs to 30,000 ↑/);
});

test("the plume draws its bands, median, lines, run mark, class strip and the gauge's record", () => {
  const svg = plumeSvg({ ...PLUME, observed: { t: ["2026-10-01", "2026-10-09"], v: [800, 2100] } },
    { w: 300, h: 132, mapDate: "2026-10-12" });
  assert.match(svg, /^<svg width="300" height="132"/);
  for (const cls of ["pl-all", "pl-mid", "pl-med", "pl-thr", "pl-run", "pl-cls", "pl-obs", "pl-today"]) {
    assert.ok(svg.includes(`class="${cls}"`), cls);
  }
  assert.equal((svg.match(/class="pl-obs-dot"/g) || []).length, 2);
  assert.ok(svg.includes(">5-yr<") && svg.includes('pl-ax pl-ax-run" x="') && svg.includes(">10 Oct<"));
  assert.equal((svg.match(/class="pl-cls"/g) || []).length, 4);   // every day is classed here
  assert.ok(!svg.includes("NaN"));
  assert.equal(plumeSvg({ date: ["2026-10-10"] }), "");
  // A map date outside the drawn days marks nothing.
  assert.ok(!plumeSvg(PLUME, { mapDate: "2027-01-01" }).includes("pl-today"));
});

test("the words around the plume", () => {
  assert.equal(plumeHead(PLUME), "Forecast: the 5-year flow at its highest, from Sat 10 Oct.");
  assert.equal(plumeHead({ ...PLUME, rp: 0 }), "Forecast: below the 2-year flow on every day.");
  assert.match(plumeHead({ ...PLUME, thresholds: null }), /no return-period flows/);
  assert.equal(plumeLabel(PLUME), "Next 4 days, modelled, run of 10 Oct");
  assert.equal(plumeLabel({ ...PLUME, issued: null }, { corrected: true }), "Next 4 days, corrected to this gauge");
  assert.ok(plumeKey(PLUME).includes("class by day"));
  assert.ok(!plumeKey({ ...PLUME, class_daily: [0, 0] }).includes("class by day"));
  assert.match(plumeAlt(PLUME), /^Forecast flow for the 4 days from 10 Oct: median, middle half and full range/);
  assert.match(plumeAlt(PLUME), /2,213 m³\/s on 10 Oct/);
  assert.ok(plumeKey({ observed: { t: ["2026-10-01"] } }).includes("observed"));
  assert.ok(!plumeKey(PLUME).includes("observed"));
  assert.equal(flowText(2213.4), "2,213");
  assert.equal(flowText(37.97), "38.0");
  assert.equal(flowText(0.1375), "0.14");
});

test("forecast gauges on the map: classed ones only, with the day's class or the peak", () => {
  const pts = [
    { key: "usgs/A", classed: true, rp: 5, class_daily: [0, 2, 5] },
    { key: "usgs/B", classed: true, rp: 0, class_daily: [0, 0, 0] },
    { key: "usgs/C", classed: false, rp: 0 },
    { key: "usgs/D", classed: true, rp: 2, class_daily: [2, 0, 0] },
  ];
  const at = (k) => ({ "usgs/A": { lat: 1, lon: 2 }, "usgs/B": { lat: 3, lon: 4 }, "usgs/C": { lat: 5, lon: 6 } }[k]);
  const fc = pointsGeoJSON(pts, at);
  assert.deepEqual(fc.features.map((f) => [f.properties.key, f.properties.c]), [["usgs/B", 0], ["usgs/A", 5]]);
  assert.deepEqual(pointsGeoJSON(pts, at, { day: 1 }).features.map((f) => f.properties.c), [0, 2]);
  assert.equal(dayOf(["2026-10-09", "2026-10-10"], "2026-10-10"), 1);
  assert.equal(dayOf(["2026-10-09", "2026-10-10"], "2026-10-11"), -1);
  assert.equal(pointsLine({ issue_date: "2026-10-09", points: pts }), "3 forecast gauges, Fri 9 Oct: 2 expected to reach the 2-year flow or more.");
  assert.equal(pointsLine({ issue_date: "2026-10-09", points: [pts[1]] }), "1 forecast gauge, Fri 9 Oct: none expected to reach the 2-year flow.");
  assert.equal(pointsLine({ points: [pts[2]] }), "");
});

test("the gauge's record before the run", () => {
  const s = { t: ["2026-09-01", "2026-09-20", "2026-10-08", "2026-10-09"], v: [1, 2, null, 4] };
  assert.deepEqual(obsBefore(s, "2026-10-09"), { t: ["2026-09-20", "2026-10-09"], v: [2, 4] });
  assert.equal(obsBefore({ t: [], v: [] }, "2026-10-09"), null);
  assert.equal(obsBefore(s, null), null);
});
