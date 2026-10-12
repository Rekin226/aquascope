import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_SPAN, addStep, annualMaxPoints, chartDate, clampDate, defaultRange, frameDates, gifSize, isIsoDate, layerCovers,
  layersMissing, missingNote, nextFrame, normaliseRange, readTimeParams, shortDate, spanLabel, writeTimeParams,
} from "../src/timeline.js";
import { OVERLAYS, basemapById, datedLayersOn, imageFor, layerDate, layerImages, overlayById } from "../src/layers.js";
import { onTime, setTime, state } from "../src/core.js";
import { clickedDay } from "../src/charts.js";

test("only real calendar days count as dates", () => {
  assert.ok(isIsoDate("2024-02-29"));
  assert.ok(!isIsoDate("2023-02-29"));
  assert.ok(!isIsoDate("2024-13-01"));
  assert.ok(!isIsoDate("2024-1-01"));
  assert.ok(!isIsoDate(null));
});

test("a step moves by a day, a week or a calendar month", () => {
  assert.equal(addStep("2024-12-31", "day", 1), "2025-01-01");
  assert.equal(addStep("2024-03-01", "day", -1), "2024-02-29");
  assert.equal(addStep("2024-01-01", "week", 2), "2024-01-15");
  assert.equal(addStep("2024-01-31", "month", 1), "2024-02-29", "the day clamps to the end of a short month");
  assert.equal(addStep("2023-01-31", "month", 1), "2023-02-28");
  assert.equal(addStep("2024-03-15", "month", -3), "2023-12-15");
  assert.equal(addStep("2024-05-10", "month", -12), "2023-05-10");
  assert.equal(addStep("not a date", "day", 1), "not a date");
});

test("a date is clamped into a range, never past today", () => {
  assert.equal(clampDate("2030-01-01", null, "2026-10-08"), "2026-10-08");
  assert.equal(clampDate("1990-01-01", "2000-06-01", null), "2000-06-01");
  assert.equal(clampDate("2010-01-01", "2000-06-01", "2026-10-08"), "2010-01-01");
});

test("frames run from start to end at the step, both ends included, with a cap", () => {
  assert.deepEqual(frameDates("2024-01-01", "2024-01-03", "day").dates, ["2024-01-01", "2024-01-02", "2024-01-03"]);
  assert.deepEqual(frameDates("2024-01-31", "2024-04-30", "month").dates,
    ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"], "month steps count from the start, so they do not drift");
  assert.deepEqual(frameDates("2024-01-03", "2024-01-01", "day").dates, ["2024-01-01", "2024-01-02", "2024-01-03"],
    "a reversed range is read the right way round");
  const capped = frameDates("2000-01-01", "2024-01-01", "day", 40);
  assert.equal(capped.dates.length, 40);
  assert.equal(capped.truncated, true);
  assert.equal(frameDates("2024-01-01", "2024-01-01", "week").dates.length, 1);
  assert.deepEqual(frameDates("bad", "2024-01-01").dates, []);
});

test("the default range is a dozen steps ending at the date, never in the future", () => {
  assert.deepEqual(defaultRange("2024-06-30", "day", "2026-10-08"), { from: "2024-06-19", to: "2024-06-30" });
  assert.deepEqual(defaultRange("2024-12-15", "month", "2026-10-08"), { from: "2024-01-15", to: "2024-12-15" });
  assert.equal(defaultRange("2030-01-01", "day", "2026-10-08").to, "2026-10-08");
  assert.equal(frameDates(...Object.values(defaultRange("2024-06-30", "week", "2026-10-08")), "week").dates.length,
    DEFAULT_SPAN);
});

test("play wraps round and picks up from the nearest frame", () => {
  const dates = ["2024-01-01", "2024-01-02", "2024-01-03"];
  assert.equal(nextFrame(dates, "2024-01-01"), "2024-01-02");
  assert.equal(nextFrame(dates, "2024-01-03"), "2024-01-01");
  assert.equal(nextFrame(dates, "2024-01-01T"), "2024-01-02");
  assert.equal(nextFrame(dates, "2025-01-01"), "2024-01-01");
  assert.equal(nextFrame([], "2024-01-01"), null);
  assert.deepEqual(normaliseRange({ from: "2024-02-01", to: "2024-01-01" }), { from: "2024-01-01", to: "2024-02-01" });
  assert.equal(normaliseRange({ from: "x", to: "2024-01-01" }), null);
});

test("each dated layer knows the days it can show", () => {
  const storage = overlayById("storage");
  const soil = overlayById("soil");
  assert.equal(storage.until, "2022-07-01", "GIBS has no GRACE month after July 2022");
  assert.ok(layerCovers(storage, "2010-05-01", "2026-10-08"));
  assert.ok(!layerCovers(storage, "2024-05-01", "2026-10-08"));
  assert.ok(!layerCovers(soil, "2014-01-01", "2026-10-08"), "SMAP starts on 31 March 2015");
  assert.ok(layerCovers(overlayById("landcover"), "1990-01-01"), "an undated layer is never out of range");
  const missing = layersMissing([storage, soil, overlayById("precip")], "2024-05-01", "2026-10-08");
  assert.deepEqual(missing.map((l) => l.id), ["storage"]);
  assert.equal(spanLabel(storage), "Apr 2002 to Jul 2022");
  assert.ok(layerCovers(storage, "2022-07-20", "2026-10-08"), "the last month is covered to its end");
  assert.ok(!layerCovers(storage, "2022-08-01", "2026-10-08"));
  assert.equal(spanLabel(soil), "31 Mar 2015 onwards");
  for (const o of OVERLAYS.filter((l) => l.time)) assert.ok(isIsoDate(o.since), `${o.id} has a first day`);
  assert.ok(isIsoDate(basemapById("daily").since));
});

test("a monthly layer asks for the first of the month, or its own first day", () => {
  const storage = overlayById("storage");
  assert.equal(layerDate(storage, "2010-05-17"), "2010-05-01");
  assert.equal(layerDate(storage, "2002-04-20"), "2002-04-04", "April 2002 starts on the 4th");
  assert.equal(layerDate(overlayById("precip"), "2010-05-17"), "2010-05-17");
});

test("GRACE asks for a day GIBS has an image for, and knows its gaps", () => {
  const storage = overlayById("storage");
  // Checked against GIBS tiles on 2026-10-08: 2004-02-01 is a 404, 2004-02-04 is the February image.
  assert.equal(layerDate(storage, "2004-02-15"), "2004-02-04");
  assert.equal(layerDate(storage, "2004-02-01"), "2004-02-04", "the month's image, though it starts later");
  assert.equal(layerDate(storage, "2011-02-20"), "2011-02-08");
  assert.equal(layerDate(storage, "2004-01-20"), "2004-01-01", "January 2004's image covers 13 days only");
  assert.equal(layerDate(storage, "2016-01-15"), "2016-01-04");
  assert.equal(layerDate(storage, "2016-01-30"), "2016-01-29", "two images in January 2016");
  assert.equal(imageFor(storage, "2002-06-15"), null, "June 2002 is a gap");
  assert.equal(imageFor(storage, "2017-09-01"), null, "the GRACE to GRACE-FO gap");
  assert.equal(imageFor(storage, "2022-08-01"), null, "nothing after July 2022");
  assert.ok(layerImages(storage).every((img) => img.start <= "2022-07-01"));
  assert.ok(!layerCovers(storage, "2017-09-01", "2026-10-08"));
  assert.ok(layerCovers(storage, "2004-02-01", "2026-10-08"));
  assert.equal(missingNote(storage, "2017-09-01", "2026-10-08"), "Water storage anomaly: no image for Sep 2017.");
  assert.equal(missingNote(storage, "2024-05-01", "2026-10-08"), "Water storage anomaly: Apr 2002 to Jul 2022 only.");
});

test("the dated layers on screen are the dated basemap and the dated overlays", () => {
  assert.deepEqual(datedLayersOn("daily", ["precip", "landcover"]).map((l) => l.id), ["daily", "precip"]);
  assert.deepEqual(datedLayersOn("light", new Set(["landcover"])), []);
});

test("a chart's x value becomes a day, and anything else is ignored", () => {
  assert.equal(chartDate("2021-07-14"), "2021-07-14");
  assert.equal(chartDate("2021-07-14 06:30"), "2021-07-14");
  assert.equal(chartDate("2021-07-14T06:30:00Z"), "2021-07-14");
  assert.equal(chartDate("2021-07-14T00:00:00+00:00"), "2021-07-14", "a pandas isoformat");
  assert.equal(chartDate(new Date(Date.UTC(2021, 6, 14))), "2021-07-14");
  assert.equal(chartDate("Jul"), null);
  assert.equal(chartDate(1998), null);
  assert.equal(chartDate("2021-02-30"), null);
  assert.equal(shortDate("2021-07-14"), "14 Jul 2021");
});

test("a click on an annual-maximum marker does not move the map", () => {
  assert.equal(clickedDay([{ x: "2019-07-01", data: { meta: { mapDate: false } } }]), null);
  assert.equal(clickedDay([{ x: "2019-07-01", data: { meta: { mapDate: false } } }, { x: "2019-03-12", data: {} }]),
    "2019-03-12");
  assert.equal(clickedDay([{ x: "Mar", data: {} }]), null);
  assert.equal(clickedDay(undefined), null);
});

test("the time state round-trips through the URL, and junk is ignored", () => {
  const q = writeTimeParams(new URLSearchParams(), {
    date: "2024-05-01", step: "week", range: { from: "2024-06-30", to: "2024-01-01" },
    compare: { date: "2023-05-01", layer: "precip" },
  }, { dated: true });
  assert.equal(q.get("d"), "2024-05-01");
  assert.equal(q.get("ts"), "week");
  assert.equal(q.get("r"), "2024-01-01..2024-06-30");
  assert.equal(q.get("cmp"), "2023-05-01");
  assert.equal(q.get("cl"), "precip");
  assert.deepEqual(readTimeParams(q), {
    date: "2024-05-01", step: "week", range: { from: "2024-01-01", to: "2024-06-30" },
    compare: { date: "2023-05-01", layer: "precip" },
  });
  // The defaults stay out of the link, and a date with nothing dated on screen is not worth carrying.
  const quiet = writeTimeParams(new URLSearchParams(), { date: "2024-05-01", step: "day" }, { dated: false });
  assert.equal(quiet.toString(), "");
  const bad = readTimeParams(new URLSearchParams("d=2024-02-30&ts=hour&r=2024-01-01..soon&cmp=x&cl=<b>"));
  assert.deepEqual(bad, {});
  assert.deepEqual(readTimeParams(new URLSearchParams("cmp=2023-01-01&cl=<b>")), { compare: { date: "2023-01-01", layer: null } });
});

test("setTime tells every subscriber once per real change, with who moved it", () => {
  state.date = "2024-01-01";
  const seen = [];
  const off = onTime((t) => seen.push([t.date, t.prev.date, t.source]));
  assert.equal(setTime({ date: "2024-01-02" }, { source: "chart" }), true);
  assert.equal(setTime({ date: "2024-01-02" }, { source: "chart" }), false, "no change, no event");
  setTime({ range: { from: "2024-01-01", to: "2024-01-05" } });
  assert.equal(setTime({ range: { from: "2024-01-01", to: "2024-01-05" } }), false);
  off();
  setTime({ date: "2024-02-01" });
  assert.deepEqual(seen, [["2024-01-02", "2024-01-01", "chart"], ["2024-01-02", "2024-01-02", "bar"]]);
  assert.equal(state.date, "2024-02-01");
});

test("a GIF frame is at most 640 wide, never upscaled, in even pixels", () => {
  assert.deepEqual(gifSize(2560, 1440), { width: 640, height: 360 });
  assert.deepEqual(gifSize(375, 501), { width: 374, height: 500 });
  assert.deepEqual(gifSize(1280, 721, 640), { width: 640, height: 360 });
});

test("annual maxima sit on the day of the peak when the analysis gives it, else at 1 July", () => {
  const am = { year: [1995, 1996], v: [10, 12], date: ["1995-03-17", "1996-01-21"] };
  assert.deepEqual(annualMaxPoints(am), { x: ["1995-03-17", "1996-01-21"], onDay: true });
  assert.deepEqual(annualMaxPoints({ year: [1995, 1996], v: [10, 12] }), { x: ["1995-07-01", "1996-07-01"], onDay: false });
  assert.equal(annualMaxPoints({ year: [1995], v: [1], date: [null] }).onDay, false);
  assert.deepEqual(annualMaxPoints(null), { x: [], onDay: false });
});
