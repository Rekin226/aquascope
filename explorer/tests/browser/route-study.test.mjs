// Exercise the route form and Study intake together. The worker fixture replaces GEOGLOWS and Pyodide.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

test("Study receives the flood wave the user ran on the map", async t => {
  const modulePath = process.env.AQ_PLAYWRIGHT_MODULE;
  const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : "playwright");
  const browser = await chromium.launch({ executablePath: process.env.AQ_CHROMIUM_EXECUTABLE || undefined });
  t.after(() => browser.close());
  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const root = new URL("../../", import.meta.url);
  const html = (await readFile(new URL("index.html", root), "utf8"))
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "").replace(/<link\b[^>]*>/g, "");
  const origin = "http://explorer.test";
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (/^\/(?:config|src\/[\w-]+)\.js$/.test(url.pathname)) {
      const body = await readFile(new URL(url.pathname.slice(1), root), "utf8");
      return route.fulfill({ contentType: "text/javascript", body });
    }
    return route.abort();
  });
  await page.goto(origin);
  await page.evaluate(async () => {
    window.requests = [];
    window.Worker = class {
      postMessage(msg) {
        if (msg.type === "init") return;
        window.requests.push(msg);
        let result;
        if (msg.type === "route") {
          const { river_id, peak_cms, duration_h, shape } = msg.args;
          result = { ok: true, river_id, hours: 48, reaches: [], places: [],
            inflow: { peak_cms, duration_h, shape, source: "specified wave" },
            furthest: { river_id, index: 0, peak_cms, peak_h: 1, attenuation_pct: 0 } };
        } else if (msg.type === "studio" && msg.op === "start") {
          const step = { id: "s1", tool: "route_flood", arguments: msg.intake.route_flood };
          result = { workspace: { status: "review", messages: [], study: { steps: [step] } },
            reply: { kind: "plan", payload: {} } };
        } else if (msg.type === "studio" && msg.op === "approve") {
          result = { workspace: { ...msg.workspace, status: "done" }, reply: { kind: "report", payload: {} } };
        } else { throw new Error(`Unexpected worker request ${msg.type}:${msg.op}`); }
        queueMicrotask(() => this.onmessage({ data: { id: msg.id, type: "result", result } }));
      }
    };
    const route = await import("/src/route-map.js?v=__BUILD__");
    const studio = await import("/src/studio.js?v=__BUILD__");
    route.initRoute();
    studio.initStudy();
    route.openRoute({ river_id: 230301988, lat: 47.56, lon: 7.59 });
  });
  const chip = page.getByRole("button", { name: "Study this routed flood wave" });
  assert.equal(await page.locator('[data-act="study-route"]').count(), 0);
  await page.locator("#rt-peak").fill("537.5");
  await page.locator("#rt-block").check();
  await page.locator("#rt-run").click();
  await page.locator("#rt-result").waitFor({ state: "visible" });
  await page.evaluate(async () => (await import("/src/studio.js?v=__BUILD__")).openStudy());
  await chip.click();
  await page.locator('#study-board[data-status="review"]').waitFor({ state: "visible" });
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await page.locator('#study-board[data-status="done"]').waitFor({ state: "visible" });
  const requests = await page.evaluate(() => window.requests);
  assert.deepEqual(requests.map(({ type, op }) => [type, op]), [
    ["route", "flood"], ["studio", "start"], ["studio", "approve"],
  ]);
  const { lat, lon, intake } = requests[1];
  assert.deepEqual({ lat, lon, intake }, { lat: 47.56, lon: 7.59, intake: { route_flood: {
    river_id: 230301988, peak_cms: 537.5, duration_h: 24, shape: "block",
  } } });
  assert.deepEqual(errors, []);
});
