// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  NO_STATUS_COLOR, STATUS_CLASSES, forecastTraces, markerShapes, nowColor, plotRange, skillText, snapshotLine,
  statusClass, thresholdsToShow,
} from "../src/now-core.js";

test("the five classes are the ones aquascope.nownext uses, in order", () => {
  assert.deepEqual(STATUS_CLASSES.map((c) => c.id), ["much_below", "below", "normal", "above", "much_above"]);
  assert.equal(statusClass("above").label, "above normal");
  assert.equal(statusClass("nonsense"), null);
});

test("nowColor keeps the agency colour until a snapshot has loaded, then greys the gauges it lacks", () => {
  assert.equal(nowColor(null, "usgs/1", "#1565c0"), "#1565c0");
  const map = new Map([["usgs/1", { cls: "much_above" }]]);
  assert.equal(nowColor(map, "usgs/1", "#1565c0"), STATUS_CLASSES[4].color);
  assert.equal(nowColor(map, "usgs/2", "#1565c0"), NO_STATUS_COLOR);
  // the map's default view (#544): a gauge the snapshot lacks keeps its agency colour
  assert.equal(nowColor(map, "usgs/2", "#1565c0", "#1565c0"), "#1565c0");
  assert.equal(nowColor(map, "usgs/1", "#1565c0", "#1565c0"), STATUS_CLASSES[4].color);
});

test("snapshotLine says when and from where, or that there is none yet", () => {
  assert.match(snapshotLine({ missing: true }), /No status snapshot yet/);
  const line = snapshotLine({ made: "2026-10-08T06:52:10Z", n: 312, sources: ["usgs", "uk_ea"] }, (s) => s.toUpperCase());
  assert.equal(line, "Daily snapshot of 8 Oct 2026, 06:52 UTC: 312 gauges with a fresh record, from USGS, UK_EA.");
});

test("thresholdsToShow keeps the lowest line and the ones near the forecast", () => {
  const thr = { return_periods: [2, 5, 10, 25, 50, 100], q: [100, 150, 180, 220, 250, 280] };
  assert.deepEqual(thresholdsToShow(thr, 30).map((t) => t.T), [2]);
  assert.deepEqual(thresholdsToShow(thr, 130).map((t) => t.T), [2, 5, 10]);
  assert.deepEqual(thresholdsToShow({ return_periods: [2], q: [null] }, 10), []);
  assert.deepEqual(thresholdsToShow(null, 10), []);
});

test("thresholdsToShow leaves out a lowest line far above the forecast, so the lines are not pressed flat", () => {
  // The Potomac on 8 October 2026: the corrected forecast near 60 m³/s, the gauge's 2-year flow 2,876 m³/s.
  const thr = { return_periods: [2, 5, 10], q: [2876, 3900, 4600] };
  assert.deepEqual(thresholdsToShow(thr, 374), []);
  assert.deepEqual(thresholdsToShow(thr, 720).map((t) => t.T), [2]);
  // No forecast value to compare with: the lowest line is still drawn.
  assert.deepEqual(thresholdsToShow(thr, null).map((t) => t.T), [2]);
});

test("the marker shows only when the map date is inside the plot", () => {
  const range = ["2026-09-08", "2026-10-22"];
  assert.equal(markerShapes("2026-10-01", range).length, 1);
  assert.equal(markerShapes("2026-10-01", range)[0].x0, "2026-10-01");
  assert.deepEqual(markerShapes("2026-01-01", range), []);
  assert.deepEqual(markerShapes(null, range), []);
});

const FC = {
  geoglows: { date: ["2026-10-07", "2026-10-08"], mean: [1, 2], p25: [0.8, 1.5], p75: [1.2, 2.5], min: [0.5, 1], max: [1.5, 3] },
  glofas: { date: ["2026-10-08", "2026-10-09"], mean: [3, 4] },
  thresholds: { return_periods: [2, 5], q: [10, 20] },
  gauge_thresholds: { return_periods: [2, 5], q: [4, 8] },
};

test("forecastTraces draws the raw bands without a gauge and the corrected ones with one", () => {
  const raw = forecastTraces(FC);
  assert.ok(raw.some((t) => t.name === "GEOGLOWS ensemble mean"));
  assert.ok(raw.some((t) => t.name === "GloFAS mean"));
  const lines = raw.filter((t) => t.text);
  assert.deepEqual(lines.map((t) => t.y[0]), [10]);

  const corrected = { ...FC, correction: { forecast: { date: FC.geoglows.date, mean: [2, 3], p25: [1, 2], p75: [3, 4] } } };
  const traces = forecastTraces(corrected, { recent: { t: ["2026-10-01", "2026-10-06"], v: [2, 6] } });
  const names = traces.map((t) => t.name);
  assert.ok(names.includes("GEOGLOWS, corrected to the gauge"));
  assert.ok(names.includes("GEOGLOWS raw"));
  assert.ok(names.includes("observed"));
  assert.deepEqual(traces.filter((t) => t.text).map((t) => t.y[0]), [4, 8]);  // the gauge's own, up to 1.5 x 6
  assert.deepEqual(plotRange(corrected, { t: ["2026-10-01"] }), ["2026-10-01", "2026-10-09"]);
});

test("forecastTraces is empty when nothing answered", () => {
  assert.deepEqual(forecastTraces({ geoglows: { error: "x" }, glofas: { error: "y" } }), []);
  assert.deepEqual(forecastTraces(null), []);
});

test("skillText quotes the skill line, or says why there is no correction", () => {
  assert.equal(skillText({}), "");
  assert.equal(skillText({ correction: { error: "too short" } }), "Not corrected to the gauge: too short");
  assert.equal(skillText({ correction: { skill_line: "Corrected forecast: KGE 0.71 on the 2008-2022 hindcast, raw 0.38.",
    skill: {} } }), "Corrected forecast: KGE 0.71 on the 2008-2022 hindcast, raw 0.38.");
  assert.equal(skillText({ correction: { skill_line: "KGE.", skill_detail: "Bias +2 %.", skill: { note: "Worse." } },
    reach_check: { note: "Far." } }), "KGE. Bias +2 %. Worse. Far.");
});
