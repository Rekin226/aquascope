// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  DIVERGING, QUESTIONS, STATUS_STYLE, archiveFileUrl, batchesOf, cardFor, classLabel, classOf, figureTitle, fileStem,
  legendRows, paintProps, planGauges, progressWords, questionById, summaryWords, viewBbox, wrapText,
} from "../src/paint-core.js";

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));
const flood = questionById("flood_trend");

test("the questions are aquascope.paint.questions() (the fixture pytest also checks)", () => {
  assert.deepEqual(QUESTIONS, fixture("paint-questions.json"));
});

test("planGauges follows aquascope.paint.plan on the shared fixture", () => {
  const fix = fixture("paint-plan.json");
  const sig = new Map(fix.signatures.map((r) => [`${r.source}/${r.station_id}`, r]));
  const p = fix.params;
  const out = planGauges(fix.stations, sig, { bbox: p.bbox, minYears: p.min_years, maxGauges: p.max_gauges });
  assert.deepEqual(out.read.map((r) => r.key), fix.expected.read);
  assert.deepEqual(out.rows.map((r) => ({ key: r.key, status: r.status, record_years: r.record_years, note: r.note })),
    fix.expected.rows);
  assert.deepEqual(out.counts, fix.expected.counts);
});

test("classes follow numpy's searchsorted (right), and say their ranges", () => {
  assert.equal(classOf(flood, -12), 0);
  assert.equal(classOf(flood, -10), 1);
  assert.equal(classOf(flood, 0), 3);
  assert.equal(classOf(flood, 2), 4);
  assert.equal(classOf(flood, 10), 6);
  assert.equal(classOf(flood, null), null);
  assert.equal(classLabel(flood, 0), "below −10 % per decade");
  assert.equal(classLabel(flood, 3), "−2 to 2");
  assert.equal(classLabel(flood, 6), "10 and above % per decade");
  assert.equal(classLabel(questionById("baseflow_index"), 2), "0.35 to 0.5");
});

test("the paint marks significance by size and ring, and unpainted gauges by status", () => {
  const p = paintProps(flood);
  const colors = p["circle-color"][2];
  assert.deepEqual(colors.slice(2, 2 + 2 * DIVERGING.length).filter((_, i) => i % 2 === 1), DIVERGING);
  assert.match(JSON.stringify(p["circle-stroke-color"]), /fdr_significant/);
  assert.match(JSON.stringify(p["circle-opacity"]), /no_archive/);
  const bfi = paintProps(questionById("baseflow_index"));
  assert.doesNotMatch(JSON.stringify(bfi["circle-stroke-color"]), /fdr_significant/);
});

test("the legend counts classes and lists only the statuses present", () => {
  const rows = [
    { status: "painted", value: 12 }, { status: "painted", value: 1 }, { status: "painted", value: -3 },
    { status: "too_short" }, { status: "no_archive" }, { status: "no_archive" }, { status: "queued" },
  ];
  const L = legendRows(flood, rows);
  assert.deepEqual(L.classes.map((c) => c.n), [0, 0, 1, 1, 0, 0, 1]);
  assert.deepEqual(L.status.map((s) => [s.status, s.n]), [["too_short", 1], ["no_archive", 2]]);
  assert.equal(L.reading, 1);
  assert.ok(STATUS_STYLE.no_archive.hollow && !STATUS_STYLE.too_short.hollow);
});

test("a painted gauge's card says its number, its p-value and whether the field test keeps it", () => {
  const row = { key: "usgs/x", name: "Delaware at Trenton", source: "usgs", status: "painted", value: 6.4, p: 0.0004,
    tau: 0.21, slope_per_year: 12.5, mean: 1950, n_years: 92, first_year: 1913, last_year: 2024, missing_years: 2,
    fdr_significant: true };
  const c = cardFor(flood, row);
  assert.equal(c.figure.value, "+6.4");
  assert.equal(c.figure.unit, "% per decade");
  assert.match(c.status.text, /^Significant after the false discovery rate: rising/);
  assert.match(c.note, /p = < 0.001 over 92 complete years, 2 missing/);
  assert.match(cardFor(flood, { ...row, fdr_significant: false, tau: -0.1, value: -3 }).status.text, /Not significant.*falling/);
  assert.equal(cardFor(flood, { status: "no_archive", station_id: "D", note: "not in the AquaScope Archive" }).status.text,
    "No record in the Archive");
});

test("words, files and the view", () => {
  assert.equal(progressWords({ done: 48, total: 212, failed: 3 }), "Painted 48 of 212 gauges · 3 could not be read");
  assert.equal(progressWords({ phase: "plan" }), "Finding the gauges in view…");
  assert.equal(summaryWords(flood, { counts: { painted: 212 }, field: { n_tested: 200, field_significant: true } }),
    "212 painted · field significant");
  assert.equal(fileStem(flood, [-8, 49.8, 2, 59]), "aquascope-paint-flood_trend_-8.0_49.8_2.0_59.0");
  assert.deepEqual(viewBbox({ west: -200, south: -90, east: 200, north: 90 }), [-180, -85, 180, 85]);
  assert.deepEqual(viewBbox({ west: -8.123, south: 49.81, east: 2.004, north: 59.0 }), [-8.12, 49.81, 2, 59]);
  assert.deepEqual(batchesOf([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.equal(archiveFileUrl("https://huggingface.co/datasets/R/a/resolve/main/stations.parquet", "usgs", "USGS-0101"),
    "https://huggingface.co/datasets/R/a/resolve/main/obs/discharge/usgs/USGS-0101.csv.gz");
  assert.match(archiveFileUrl("https://h/stations.parquet", "x", "a/b"), /obs\/discharge\/x\/a_b\.csv\.gz$/);
  assert.deepEqual(wrapText("one two three four", 9, (t) => t.length), ["one two", "three", "four"]);
  const t = figureTitle(flood, { counts: { painted: 12 }, params: { min_years: 30, start_year: 1960 }, field: { alpha_fdr: 0.1 } });
  assert.equal(t.title, "Trend in annual floods: 12 gauges");
  assert.match(t.sub, /at least 30 complete years, years from 1960/);
});
