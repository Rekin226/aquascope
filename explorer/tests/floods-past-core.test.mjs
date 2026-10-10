// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  FLOODS_PAST_BASE, MAX_MONTHS, NEWS_CREDIT, RADAR_CREDIT, addMonths, cellBbox, cellsGeoJSON, eventDates, fmtCount,
  legendLines, monthLabel, monthOf, monthsBetween, monthsOnRecord, newsWeight, placeLabel, radarCovers, radarWeight,
  readFloodsParam, windowFor, windowLabel, windowTotals, areaLabel, radarFill, newsFill,
  CELL_MINZOOM, HEAT_FADE, HEAT_MAXZOOM, STANDOUT_FLOOR, STANDOUT_SHARE, inBox, shortWhen, standoutFilters,
  standoutThresholds,
} from "../src/floods-past-core.js";

const INDEX = {
  deg: 0.5, first: "2000-01", last: "2026-02",
  news: { first: "2000-01-01", last: "2026-02-03" },
  months: [
    { month: "2019-07", news: 18000, radar: 40000000 },
    { month: "2019-08", news: 20000, radar: 50000000 },
    { month: "2025-12", news: 27103, radar: 0 },
    { month: "2026-01", news: 26711, radar: 0 },
    { month: "2026-02", news: 12, radar: 0 },
  ],
};

test("months step, span and print the same way as the Python side", () => {
  assert.equal(monthOf("2024-07-14"), "2024-07");
  assert.equal(monthOf("2024-13-01"), null);
  assert.equal(addMonths("2024-11", 3), "2025-02");
  assert.equal(addMonths("2024-01", -1), "2023-12");
  assert.deepEqual(monthsBetween("2025-01-09", "2024-11-30"), ["2024-11", "2024-12", "2025-01"]);
  assert.equal(monthLabel("2019-07"), "Jul 2019");
  assert.equal(windowLabel(["2019-07", "2019-08"]), "Jul 2019 to Aug 2019");
});

test("the window follows the time bar: twelve months, a played or stepped month, or a range", () => {
  // the page opens after the last month on record: the latest twelve, said so
  assert.deepEqual(windowFor({ date: "2026-10-03" }, "2026-02"),
    { months: monthsBetween("2025-03", "2026-02"), mode: "window", latest: true });
  const back = windowFor({ date: "2019-08-15" }, "2026-02");
  assert.equal(back.months[0], "2018-09");
  assert.equal(back.months.at(-1), "2019-08");
  assert.equal(back.latest, false);
  // playing, or stepping a month at a time, shows the month of the frame
  assert.deepEqual(windowFor({ date: "2019-07-01", playing: true, range: { from: "2019-01-01", to: "2019-12-01" } }, "2026-02").months, ["2019-07"]);
  assert.deepEqual(windowFor({ date: "2019-07-01", step: "month" }, "2026-02").months, ["2019-07"]);
  // stepping by month after the record (the river status opens on its newest month): the latest twelve
  assert.deepEqual(windowFor({ date: "2026-09-15", step: "month" }, "2026-02"),
    { months: monthsBetween("2025-03", "2026-02"), mode: "window", latest: true });
  // playing past the record shows that (empty) month
  assert.deepEqual(windowFor({ date: "2026-09-15", step: "month", playing: true }, "2026-02").months, ["2026-09"]);
  // a range set and not playing: all its months, at most sixty
  const r = windowFor({ date: "2019-07-01", range: { from: "2019-06-01", to: "2019-08-20" } }, "2026-02");
  assert.deepEqual(r.months, ["2019-06", "2019-07", "2019-08"]);
  const long = windowFor({ date: "2019-07-01", range: { from: "2000-01-01", to: "2019-12-01" } }, "2026-02");
  assert.equal(long.months.length, MAX_MONTHS);
  assert.equal(long.months.at(-1), "2019-12");
  assert.equal(long.truncated, true);
  assert.deepEqual(windowFor({}, null).months, []);
});

test("only the months the index has are fetched, and the legend sums them", () => {
  assert.deepEqual(monthsOnRecord(INDEX, ["2019-06", "2019-07", "2019-08"]), ["2019-07", "2019-08"]);
  assert.deepEqual(windowTotals(INDEX, ["2019-07", "2019-08"]), { news: 38000, radar: 90000000 });
  assert.equal(radarCovers(["2024-09"]), true);
  assert.equal(radarCovers(["2024-10", "2025-01"]), false);
});

test("the legend says the months, the counts, and when radar has nothing to say", () => {
  const latest = legendLines(INDEX, windowFor({ date: "2026-10-03" }, "2026-02"));
  assert.equal(latest.when, "Mar 2025 to Feb 2026 (latest on record)");
  assert.equal(latest.news, "54k events");
  assert.equal(latest.radar, "2014 to 2024 only");
  assert.equal(latest.radarCount, null);
  const monsoon = legendLines(INDEX, { months: ["2019-07"], mode: "frame" });
  assert.equal(monsoon.when, "Jul 2019");
  assert.equal(monsoon.news, "18k events");
  assert.equal(monsoon.radar, "40 M detections");
  const before = legendLines(INDEX, { months: ["1995-01"], mode: "frame" });
  assert.equal(before.news, "2000 to 2026 only");
  assert.equal(legendLines(null, { months: [] }), null);
});

test("cells sum over the months into one point per cell, plus the radar cell as a square", () => {
  const fc = cellsGeoJSON([[[281, 371, 3, 150], [10, 20, 1, 0]], [[281, 371, 1, 50]]]);
  const points = fc.features.filter((f) => f.geometry.type === "Point");
  const squares = fc.features.filter((f) => f.geometry.type === "Polygon");
  assert.equal(points.length, 2);
  assert.deepEqual(points[0].properties, { row: 281, col: 371, news: 4, radar: 200 });
  assert.deepEqual(points[0].geometry.coordinates, [5.75, 50.75]);
  assert.equal(squares.length, 1);   // a cell with no radar has no square
  assert.deepEqual(squares[0].geometry.coordinates[0][0], [5.5, 50.5]);
  assert.deepEqual(squares[0].geometry.coordinates[0][2], [6, 51]);
  assert.deepEqual(cellBbox(281, 371), [5.5, 50.5, 6, 51]);
});

test("heat weights scale with the window, so a replayed month reads like the year", () => {
  // ["min", 1, ["/", ln(1 + news), ln(1 + full)]]: full is 25 events a month
  assert.equal(newsWeight(12)[2][2], Math.log(1 + 300));
  assert.equal(newsWeight(1)[2][2], Math.log(1 + 25));
  const year = JSON.stringify(radarWeight(12));
  assert.ok(year.includes(String(Math.log(1 + 960))));
});

test("dates, places and counts are said plainly", () => {
  assert.equal(eventDates("2021-07-14", "2021-07-14"), "14 Jul 2021");
  assert.equal(eventDates("2021-07-14", "2021-07-16"), "14 to 16 Jul 2021");
  assert.equal(eventDates("2021-07-30", "2021-08-02"), "30 Jul to 2 Aug 2021");
  assert.equal(eventDates("2021-12-30", "2022-01-02"), "30 Dec 2021 to 2 Jan 2022");
  assert.equal(eventDates("2021-07-14", null), "14 Jul 2021");
  assert.equal(placeLabel(-6.25, 106.75), "6.25° S, 106.75° E");
  assert.equal(fmtCount(9999), "9,999");
  assert.equal(fmtCount(378567), "379k");
  assert.equal(fmtCount(30882548), "31 M");
  assert.equal(fmtCount(1500000), "1.5 M");
});

test("the layer is on unless the link says fp=0, and credits its two sources with their licences", () => {
  assert.equal(readFloodsParam("#fp=0&d=2019-07-01"), false);
  assert.equal(readFloodsParam("#fp=1"), true);
  assert.equal(readFloodsParam("#d=2019-07-01"), null);
  assert.equal(NEWS_CREDIT.licence, "CC BY 4.0");
  assert.equal(RADAR_CREDIT.licence, "MIT");
  assert.ok(FLOODS_PAST_BASE.startsWith("https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/context/"));
});

test("small counts stay quiet: radar cells clear below 200 detections, a lone news report faint", () => {
  for (const dark of [false, true]) {
    const ramp = radarFill(dark);
    assert.equal(ramp[3], Math.log(201));
    assert.match(ramp[4], /,0\)$/);
    assert.match(ramp[ramp.length - 1], dark ? /^rgba\(205,190,254/ : /^rgba\(76,29,149/);
  }
  // translucent all the way up (#543 design pass): a report never hides the basins or the rivers under it
  const news = newsFill();
  assert.match(news[4], /0\.38\)$/);
  const alphas = news.filter((v) => typeof v === "string" && v.startsWith("rgba")).map((c) => Number(/,([\d.]+)\)$/.exec(c)[1]));
  assert.ok(alphas.every((a) => a < 0.8) && alphas[0] < alphas[alphas.length - 1]);
});

// ── what stands out (#543 design pass) ───────────────────────────────────────

test("the legend row says the window in a few words", () => {
  assert.equal(shortWhen({ months: ["2024-07"] }), "Jul 2024");
  assert.equal(shortWhen({ months: monthsBetween("2025-03", "2026-02") }), "year to Feb 2026");
  assert.equal(shortWhen({ months: monthsBetween("2024-03", "2024-07") }), "5 months to Jul 2024");
  assert.equal(shortWhen({ months: [] }), "");
});

test("a box across the antimeridian still holds its cells", () => {
  assert.ok(inBox(10, 5, [0, 0, 20, 10]));
  assert.ok(!inBox(30, 5, [0, 0, 20, 10]));
  assert.ok(inBox(179, 5, [170, 0, -170, 10]) && inBox(-175, 5, [170, 0, -170, 10]));
  assert.ok(!inBox(0, 5, [170, 0, -170, 10]));
  assert.ok(inBox(0, 0, null));
});

test("only the cells that stand out from the region on screen are drawn", () => {
  // a region where every cell has reports (the Ganges plain): 100 cells with 1 to 100 news events
  const cells = Array.from({ length: 100 }, (_, i) => ({ lon: 80 + (i % 10) * 0.5, lat: 24 + Math.floor(i / 10) * 0.5,
    news: i + 1, radar: i < 50 ? 0 : (i + 1) * 1000 }));
  const t = standoutThresholds(cells, [79, 23, 86, 30]);
  const drawn = cells.filter((c) => c.news >= t.news).length;
  assert.ok(drawn <= Math.ceil(STANDOUT_SHARE * 100) && drawn >= 10, `drew ${drawn} of 100`);
  assert.ok(cells.filter((c) => c.radar >= t.radar).length <= Math.ceil(STANDOUT_SHARE * 50));
  // the same cells elsewhere on the map are not the region on screen
  assert.deepEqual(standoutThresholds(cells, [0, 0, 10, 10]), { ...STANDOUT_FLOOR });
  // a quiet month: a few single reports stay under the floor
  const quiet = [{ lon: 1, lat: 1, news: 1, radar: 0 }, { lon: 2, lat: 1, news: 1, radar: 300 }];
  const q = standoutThresholds(quiet);
  assert.equal(q.news, STANDOUT_FLOOR.news);
  assert.equal(q.radar, STANDOUT_FLOOR.radar);
  const f = standoutFilters(t);
  assert.deepEqual(f.news[2], [">=", ["get", "news"], t.news]);
  assert.deepEqual(f.radarCells[2], [">=", ["get", "radar"], t.radar]);
});

test("a soft heat up to the regional view hands over to small marks close in", () => {
  assert.ok(CELL_MINZOOM <= HEAT_FADE[0] && HEAT_FADE[1] <= HEAT_MAXZOOM);
  assert.ok(CELL_MINZOOM >= 6, "Bangladesh at zoom 5.5 is a density, not a grid of dots");
});

test("an event's area reads plainly, small ones included", () => {
  assert.equal(areaLabel(120.4), "120 km²");
  assert.equal(areaLabel(0.72), "under 1 km²");
  assert.equal(areaLabel(null), "");
  assert.equal(areaLabel(""), "");
});
