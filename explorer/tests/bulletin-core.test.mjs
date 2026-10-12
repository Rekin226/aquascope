// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { NO_BULLETIN_TEXT, bulletinLegendLine, bulletinPaths, latestEntry, monthLabel } from "../src/bulletin-core.js";

test("latestEntry picks the month the index names, else the newest listed, else nothing", () => {
  const idx = { latest: "2026-09", months: [{ month: "2026-09", classed: 1659 }, { month: "2026-08" }] };
  assert.equal(latestEntry(idx).classed, 1659);
  assert.equal(latestEntry({ months: [{ month: "2026-07" }, { month: "2026-08" }] }).month, "2026-08");
  assert.deepEqual(latestEntry({ latest: "2026-09" }), { month: "2026-09" });
  assert.equal(latestEntry(null), null);
  assert.equal(latestEntry({ latest: "../../etc", months: [{ month: "x" }] }), null);
});

test("bulletinPaths are the files the workflow writes under bulletins/<month>/", () => {
  const p = bulletinPaths("https://hf.example/bulletins/", "2026-09");
  assert.equal(p.html, "https://hf.example/bulletins/2026-09/bulletin.html");
  assert.equal(p.markdown, "https://hf.example/bulletins/2026-09/bulletin.md");
  assert.equal(p.status, "https://hf.example/bulletins/2026-09/status.parquet");
});

test("the legend line says which month, or that there is no bulletin yet", () => {
  assert.equal(monthLabel("2026-09"), "September 2026");
  assert.equal(monthLabel("nope"), "");
  assert.match(bulletinLegendLine({ missing: true }), /No bulletin yet/);
  assert.match(bulletinLegendLine(null), /agency colours/);
  assert.equal(bulletinLegendLine({ month: "2026-09", classed: 1659 }),
    "September 2026: 1,659 gauges classed against the same month in other years.");
  assert.match(NO_BULLETIN_TEXT, /3rd of next month/);
});
