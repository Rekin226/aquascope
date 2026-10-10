// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  cardNumber, dayMonth, forecastPeak, lastDays, latestValue, ordinal, placeCard, prettyUnit, snapshotSentence,
  sparkPaths,
} from "../src/map-card-core.js";

test("ordinals follow English, teens included", () => {
  assert.deepEqual([1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 100, 101].map(ordinal),
    ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "100th", "101st"]);
  assert.equal(ordinal(82.4), "82nd");
  assert.equal(ordinal(null), "");
});

test("days are said the way the Now tab says them", () => {
  assert.equal(dayMonth("2026-10-08"), "8 October");
  assert.equal(dayMonth("2026-10-08", { year: true }), "8 October 2026");
  assert.equal(dayMonth("2026-10-08T00:00:00", { short: true, year: true }), "8 Oct 2026");
  assert.equal(dayMonth("nonsense"), "");
});

test("a fresh snapshot row reads as today, an older one names its day", () => {
  const row = { cls: "above", pct: 82.2, date: "2026-10-08", n_years: 46 };
  assert.equal(snapshotSentence(row, { today: "2026-10-09" }),
    "Flow is above normal for 8 October (82nd percentile of 46 years).");
  assert.equal(snapshotSentence(row, { today: "2026-10-20" }),
    "Flow was above normal on 8 October 2026 (82nd percentile of 46 years).");
  assert.equal(snapshotSentence({ ...row, cls: "much_below", pct: 3.4 }, { today: "2026-10-08", variable: "water_level" }),
    "Water level is much below normal for 8 October (3rd percentile of 46 years).");
  assert.equal(snapshotSentence({ cls: "nonsense" }), null);
  assert.equal(snapshotSentence(null), null);
});

test("the last twelve months count back from the record's own end", () => {
  const t = [], v = [];
  for (let i = 0; i < 800; i++) {
    const d = new Date(Date.UTC(1993, 0, 1) + i * 86400000);
    t.push(d.toISOString().slice(0, 10));
    v.push(i % 50 === 0 ? null : i);
  }
  const y = lastDays({ t, v }, 365);
  assert.equal(y.t.length, 365);
  assert.equal(y.t[y.t.length - 1], t[t.length - 1]);
  assert.equal(y.v[y.v.length - 1], 799);
  assert.deepEqual(lastDays({ t: [], v: [] }), { t: [], v: [] });
  assert.deepEqual(latestValue({ t: ["2026-01-01", "2026-01-02"], v: [3, null] }), { value: 3, date: "2026-01-01" });
  assert.equal(latestValue({ t: [], v: [] }), null);
});

test("the forecast's peak is its highest daily mean, with the day", () => {
  assert.deepEqual(forecastPeak({ date: ["2026-10-10", "2026-10-11", "2026-10-12"], mean: [5, 9.5, null] }),
    { value: 9.5, date: "2026-10-11" });
  assert.equal(forecastPeak({ date: [], mean: [] }), null);
  assert.equal(forecastPeak(null), null);
});

test("numbers keep the precision nownext gives them", () => {
  assert.equal(cardNumber(1234.6), "1,235");
  assert.equal(cardNumber(59.401), "59.4");
  assert.equal(cardNumber(1.906), "1.91");
  assert.equal(cardNumber(0.000412), "0.000412");
  assert.equal(cardNumber(0), "0");
  assert.equal(cardNumber(null), "");
  assert.equal(prettyUnit("m3/s"), "m³/s");
  assert.equal(prettyUnit("m"), "m");
});

test("the sparkline starts at zero for a positive record, breaks at gaps, and ends on a dot", () => {
  const p = sparkPaths({ v: [1, 2, null, 4] }, { w: 30, h: 20, pad: 0 });
  assert.equal(p.line, "M0,15L10,10M30,0");
  assert.ok(p.area.startsWith("M0,20"));
  assert.deepEqual(p.end, { x: 30, y: 0 });
  assert.equal(p.band, "");
  const b = sparkPaths({ v: [2, 3], band: { lo: [1, 2], hi: [3, 4] } }, { w: 10, h: 10, pad: 0 });
  assert.ok(b.band.startsWith("M0,"));
  assert.equal(b.area, "");
  assert.equal(sparkPaths({ v: [null, 1] }), null);
  // A record below zero (a level against a datum) is scaled to its own range.
  const neg = sparkPaths({ v: [-2, -1] }, { w: 10, h: 10, pad: 0 });
  assert.equal(neg.line, "M0,10L10,0");
});

test("a forecast is scaled to its own range, never less than a fifth of its peak", () => {
  // From zero, 90 to 100 is a flat line near the top; on its own range the rise fills the box.
  const rec = sparkPaths({ v: [90, 100] }, { w: 10, h: 100, pad: 0 });
  assert.equal(rec.line, "M0,10L10,0");
  const fc = sparkPaths({ v: [90, 100] }, { w: 10, h: 100, pad: 0, fromZero: false });
  assert.equal(fc.line, "M0,75L10,25");
  // A steady river stays steady: 100 to 101 moves by a twentieth of the box at most.
  const flat = sparkPaths({ v: [100, 101] }, { w: 10, h: 100, pad: 0, fromZero: false });
  const [y0, y1] = flat.line.match(/,([\d.]+)/g).map((x) => Number(x.slice(1)));
  assert.ok(Math.abs(y0 - y1) <= 5, flat.line);
  // A wide range is drawn edge to edge.
  const wide = sparkPaths({ v: [10, 100] }, { w: 10, h: 100, pad: 0, fromZero: false });
  assert.equal(wide.line, "M0,100L10,0");
});

test("the card sits above what was clicked, flips below near the top, and stays inside the map", () => {
  const above = placeCard({ ax: 500, ay: 500, cw: 300, ch: 200, W: 1400, H: 800, lift: 10, gap: 10 });
  assert.deepEqual([above.left, above.top, above.side, above.tail, above.inside], [350, 280, "above", 150, true]);
  const below = placeCard({ ax: 500, ay: 60, cw: 300, ch: 200, W: 1400, H: 800 });
  assert.equal(below.side, "below");
  assert.ok(below.top > 60);
  const edge = placeCard({ ax: 1390, ay: 500, cw: 300, ch: 200, W: 1400, H: 800, reserveRight: 400 });
  assert.equal(edge.left, 1400 - 400 - 300 - 10);
  assert.equal(edge.inside, false);
  assert.ok(edge.tail <= 300 - 16);
});
