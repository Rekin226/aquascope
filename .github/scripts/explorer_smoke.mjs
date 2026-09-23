#!/usr/bin/env node
// Real browser -> live catalog -> observed record -> downloaded CSV.
// AQ_BASE_URL can point at a built preview. Failures retain per-case diagnostics.
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export const CASES = [
  { id: "fish-river-us", source: "usgs", station: "USGS-01013500" },
  { id: "kingston-uk", source: "uk_ea", station: "8496ce69-482c-406a-a2f0-ac418ef8f099" },
  { id: "seine-fr", source: "hubeau_hydrometrie", station: "F700000103" },
];

export async function recordToExport(page, baseURL, entry, { output, phase = "cold" } = {}) {
  const started = performance.now();
  const errors = [];
  const onError = e => errors.push(String(e.message));
  page.on("pageerror", onError);
  const result = { case: entry.id, phase, status: "failed", timings_ms: {} };
  try {
    if (phase === "warm") await page.reload({ waitUntil: "domcontentloaded", timeout: 60000 });
    else await page.goto(`${baseURL.replace(/\/$/, "")}/#s=${entry.source}/${entry.station}&tab=overview`, { waitUntil: "domcontentloaded", timeout: 60000 });
    result.timings_ms.dom = Math.round(performance.now() - started);
    await page.waitForFunction(() => globalThis.__aq?.state?.stations?.length > 0, null, { timeout: 90000 });
    result.timings_ms.catalog = Math.round(performance.now() - started);
    await page.waitForFunction(() => !document.querySelector("#btn-csv")?.disabled || document.querySelector("#status.error"), null, { timeout: 180000 });
    assert.equal(await page.locator("#btn-csv").isEnabled(), true, await page.locator("#status").textContent());
    const evidence = await page.evaluate(() => {
      const r = globalThis.__aq.state.result;
      return { n: r.n, variable: r.variable, unit: r.unit, start: r.start, end: r.end,
        snapshot: r.data_snapshot, eligibility: r.eligibility };
    });
    assert.ok(evidence.n > 365, "representative workflow needs more than one year of actual observations");
    assert.equal(evidence.variable, "discharge");
    assert.equal(evidence.unit, "m3/s");
    assert.match(evidence.snapshot, /^sha256:[a-f0-9]{64}$/);
    result.evidence = evidence;
    result.timings_ms.record = Math.round(performance.now() - started);
    const [download] = await Promise.all([page.waitForEvent("download", { timeout: 30000 }), page.locator("#btn-csv").click()]);
    assert.equal(await download.failure(), null);
    const csv = await readFile(await download.path(), "utf8");
    assert.equal(csv.trim().split(/\r?\n/).length - 1, evidence.n,
      "download must preserve every analyzed observation, not a reduced plotting series");
    assert.match(csv.split(/\r?\n/)[0], /date|time/i);
    if (output) await download.saveAs(`${output}/${entry.id}-${phase}.csv`);
    result.timings_ms.export = Math.round(performance.now() - started);
    assert.deepEqual(errors, [], "uncaught browser errors");
    result.status = "passed";
  } catch (error) {
    result.error = String(error.message);
    result.visible_status = await page.locator("#status").textContent().catch(() => null);
    result.page_errors = errors;
  } finally { page.off("pageerror", onError); }
  return result;
}

export function percentiles(values) {
  const sorted = values.filter(Number.isFinite).sort((a,b) => a-b);
  if (sorted.length < 5) return { n: sorted.length, note: "At least five repetitions required; no percentile claim." };
  const at = p => sorted[Math.ceil(sorted.length * p) - 1];
  return { n: sorted.length, p50: at(.5), p75: at(.75) };
}

async function main() {
  const modulePath = process.env.AQ_PLAYWRIGHT_MODULE;
  const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : "playwright");
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const baseURL = process.env.AQ_BASE_URL || "https://rekin226-aquascope-explorer.static.hf.space";
  const output = process.env.AQ_OUTPUT || "browser-smoke-output";
  const repetitions = Math.max(1, Math.min(20, Number(process.env.AQ_REPETITIONS || 1)));
  await mkdir(output, { recursive: true });
  const results = [];
  try {
    for (const entry of CASES) for (let i=0; i<repetitions; i++) {
      // Cold = fresh browser context; warm = reload in that same context.
      // This does not flush OS/CDN caches. Each phase starts a new Python worker.
      const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
      const page = await context.newPage();
      for (const phase of ["cold", "warm"]) {
        const result = await recordToExport(page, baseURL, entry, { output, phase });
        results.push({ repetition: i+1, ...result });
        console.log(JSON.stringify(result));
        await writeFile(`${output}/results.json`, JSON.stringify({ baseURL, results }, null, 2));
      }
      await context.close();
    }
  } finally { await browser.close(); }
  const timing_summary = CASES.flatMap(entry => ["cold", "warm"].map(phase => ({
    case: entry.id, phase, export_ms: percentiles(results.filter(r => r.case===entry.id && r.phase===phase && r.status==="passed").map(r => r.timings_ms.export)),
  })));
  await writeFile(`${output}/results.json`, JSON.stringify({ baseURL, measured_at: new Date().toISOString(),
    environment: { node: process.version, platform: process.platform },
    protocol: "Fresh browser context vs reload in same context; new Python worker in both; OS and CDN caches uncontrolled.",
    results, timing_summary }, null, 2));
  if (results.some(r => r.status !== "passed")) process.exitCode = 1;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
