// Water balance on the map (#557): ramps, frames, words, the card's bars and the link.
import test from "node:test";
import assert from "node:assert/strict";
import {
  NO_VALUE, VARIABLES, balanceBars, basinCard, basinSentence, colorFor, fieldOf, fillColorExpression, frameValues,
  hasNormals, monthIndex, rampCss, rampEnds, rampStops, readWbParam, rowOf, rowSummary, runoffNote, variableNote,
  wbDatedLayer, wbMonthFor,
} from "../src/waterbalance-core.js";

// A year file as aquascope.waterbalance.year_files writes it: two months, three subbasins.
const doc = {
  year: 2026, months: ["2026-08", "2026-09"], era5t: [false, true], n: 3,
  p_mm: [[120, 30, null], [84.4, 10, 5]], et_mm: [[70, 20, 4], [61.2, 25, 4]], q_mm: [[30, null, -3], [12, null, -1]],
  closure_mm: [[20, null, 2], [11.2, null, 2]],
  p_anom: [[10, null, null], [-20, null, null]], et_anom: [[null, null, null], [null, null, null]],
  q_anom: [[null, null, null], [null, null, null]], closure_anom: [[null, null, null], [null, null, null]],
};
const manifest = { first_month: "1990-01", last_month: "2026-09", years: ["1990", "2025", "2026"],
  era5_final_until: "2026-06-30" };

test("every ramp rises, and the anomaly ramps are centred on zero", () => {
  for (const v of VARIABLES) {
    for (const mode of ["amount", "anom"]) {
      const s = rampStops(v.id, mode);
      for (let i = 1; i < s.length; i++) assert.ok(s[i][0] > s[i - 1][0], `${v.id} ${mode}`);
      if (mode === "anom") assert.equal(s[Math.floor(s.length / 2)][0], 0);
    }
  }
  assert.equal(rampStops("snow")[0][1], rampStops("p")[0][1]);     // an unknown variable falls back to rain
});

test("colours interpolate between stops, clamp at the ends, and grey a missing value", () => {
  const s = [[0, "#000000"], [10, "#ffffff"]];
  assert.equal(colorFor(5, s), "#808080");
  assert.equal(colorFor(-3, s), "#000000");
  assert.equal(colorFor(99, s), "#ffffff");
  assert.equal(colorFor(null, s), NO_VALUE);
  assert.equal(colorFor(NaN, s), NO_VALUE);
});

test("the map's fill expression greys features without a number", () => {
  const e = fillColorExpression("q", "amount");
  assert.equal(e[0], "case");
  assert.deepEqual(e[1], ["==", ["typeof", ["feature-state", "v"]], "number"]);
  assert.equal(e[2][0], "interpolate");
  assert.equal(e[3], NO_VALUE);
});

test("a frame is the month's array for the variable and mode", () => {
  assert.equal(fieldOf("closure", "anom"), "closure_anom");
  assert.equal(monthIndex(doc, "2026-09"), 1);
  assert.deepEqual(frameValues(doc, "2026-09", "q", "amount"), [12, null, -1]);
  assert.deepEqual(frameValues(doc, "2026-08", "p", "anom"), [10, null, null]);
  assert.equal(frameValues(doc, "2026-01", "p", "amount"), null);
});

test("the map date picks a month the table has", () => {
  assert.equal(wbMonthFor("2026-09-15", manifest), "2026-09");
  assert.equal(wbMonthFor("2026-10-11", manifest), null);           // after the newest month
  assert.equal(wbMonthFor("1989-12-01", manifest), null);
  assert.equal(wbMonthFor("2000-05-01", manifest), null);           // a year not computed yet
  assert.equal(wbMonthFor("2026-09-15", null), null);
  const d = wbDatedLayer(manifest);
  assert.equal(d.monthly, true);
  assert.equal(d.since, "1990-01-01");
  assert.equal(d.until, "2026-09-01");
});

test("the row says what is painted, or why nothing is", () => {
  assert.equal(rowSummary({ manifest, month: "2026-09", variable: "q", mode: "anom" }), "Runoff vs normal, Sep 2026");
  assert.equal(rowSummary({ manifest, month: null, variable: "p", mode: "amount", date: "2026-10-11" }),
    "Jan 1990 to Sep 2026 only; none for Oct 2026");
  assert.equal(rowSummary({ manifest: null }), "loading");
  assert.deepEqual(rampEnds("p", "amount"), ["0", "50", "400 mm"]);
  assert.deepEqual(rampEnds("closure", "amount"), ["−100", "0", "100 mm"]);
  assert.deepEqual(rampEnds("et", "anom"), ["−40", "0", "+40 mm"]);
  assert.match(rampCss("p", "amount"), /^linear-gradient\(90deg, #f7fbff 0%/);
  assert.match(variableNote("closure", "amount"), /storage/);
  assert.match(variableNote("p", "anom", { normals: false }), /No 1991-2020 normal yet/);
});

test("the sentence carries the file's numbers, and a losing river is said so", () => {
  const r = rowOf(doc, 1, 0);
  assert.equal(r.era5t, true);
  assert.equal(basinSentence(r), "September 2026: 84 mm of rain, 61 mm evaporated, 12 mm ran off; P − ET − Q leaves +11 mm. " +
    "Rain was 20 mm below the 1991-2020 normal.");
  assert.match(basinSentence(rowOf(doc, 0, 2)), /the river lost 3 mm crossing it/);
  assert.match(basinSentence(rowOf(doc, 0, 1)), /no runoff, so no closure/);
});

test("the bars draw rain beside evaporation and runoff, a loss below the line, the month shaded", () => {
  const rows = doc.months.map((_, i) => rowOf(doc, i, 2));
  const svg = balanceBars(rows, { month: "2026-09" });
  assert.match(svg, /^<svg /);
  assert.ok(svg.includes("#a6611a"));                              // the river loses water
  assert.ok(svg.includes("rgba(31,41,51,0.07)"));                  // the month on the map
  assert.equal((svg.match(/<text /g) || []).length, 13);          // twelve initials and the scale
  const withNormal = balanceBars(doc.months.map((_, i) => rowOf(doc, i, 0)), { month: "2026-09", normalP: Array(12).fill(50) });
  assert.equal((withNormal.match(/stroke-width="1.4"/g) || []).length, 2);   // a tick per month drawn
});

test("the card: title, sentence, bars, and the honest notes", () => {
  const c = basinCard(doc, 0, "2026-09", { id: "1040006850", a: 56800, r: 1 }, { variable: "p" });
  assert.equal(c.title, "Level-4 subbasin");
  assert.equal(c.sub, "HydroBASINS 1040006850 · 56,800 km²");
  assert.match(c.status.text, /^September 2026: 84 mm of rain/);
  assert.match(c.chart.svg, /^<svg/);
  assert.match(c.note, /ERA5T, preliminary/);
  const none = basinCard(doc, 1, "2026-09", { id: "1", a: 10, r: 0 });
  assert.match(none.note, /No runoff here/);
  assert.match(basinCard(doc, 0, "2026-01", { id: "1" }).status.text, /No water balance for January 2026/);
  const full = { ...doc, months: Array.from({ length: 12 }, (_, i) => `2025-${String(i + 1).padStart(2, "0")}`), year: 2025,
    era5t: Array(12).fill(false) };
  for (const k of ["p_mm", "et_mm", "q_mm", "closure_mm", "p_anom", "et_anom", "q_anom", "closure_anom"]) {
    full[k] = Array.from({ length: 12 }, () => [10, null, null]);
  }
  assert.match(basinCard(full, 0, "2025-03", { id: "1" }).note, /Over 2025, P − ET − Q adds up to \+120 mm/);
});

test("the link: wb=1, the variable and the mode; nothing when the hash says nothing", () => {
  assert.equal(readWbParam("#v=2/0/0"), null);
  assert.deepEqual(readWbParam("#wb=1&wbv=closure&wbm=anom"), { on: true, variable: "closure", mode: "anom" });
  assert.deepEqual(readWbParam("#wb=0&wbv=snow"), { on: false, variable: "p", mode: "amount" });
});

test("normals count only when some subbasin has one", () => {
  assert.equal(hasNormals(null), false);
  assert.equal(hasNormals({ p: [[null, null], [null, null]] }), false);
  assert.equal(hasNormals({ p: [[null, 3.2]] }), true);
});

test("a subbasin without runoff says why: no reach, or which check withheld it", () => {
  assert.equal(runoffNote({ r: 1 }), null);
  assert.match(runoffNote({ r: 0 }), /no outlet reach/);
  assert.match(runoffNote({ r: 0, c: "through" }), /^Runoff withheld: the river brings in more than a metre/);
  assert.match(runoffNote({ r: 0, c: "outlet" }), /less than half/);
  assert.match(basinCard(doc, 0, "2026-09", { id: "1", a: 10, r: 0, c: "upstream" }).note, /fails these checks/);
});

test("the normal rain tick is the calendar month's, also in a year that starts late", () => {
  const late = { ...doc, months: ["2026-08", "2026-09"] };
  const rows = late.months.map((_, i) => rowOf(late, i, 2));     // 5 mm of rain at most
  const normalP = Array(12).fill(null);
  normalP[8] = 400;                                               // September's normal, far above the bars
  const svg = balanceBars(rows, { month: "2026-09", normalP });
  assert.match(svg, />400 mm</);                                  // the scale makes room for September's tick
});
