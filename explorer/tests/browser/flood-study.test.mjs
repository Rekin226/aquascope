// Real Flood it and Study UI events, with offline terrain and worker fixtures.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

test("Study receives the selected Flood it scenario", async t => {
  const modulePath = process.env.AQ_PLAYWRIGHT_MODULE;
  const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : "playwright");
  const browser = await chromium.launch({ executablePath: process.env.AQ_CHROMIUM_EXECUTABLE || undefined });
  t.after(() => browser.close());
  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  const root = new URL("../../", import.meta.url);
  const html = (await readFile(new URL("index.html", root), "utf8"))
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "").replace(/<link\b[^>]*>/g, "");
  const origin = "http://explorer.test";
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: html });
    if (/^\/(?:config|src\/[\w-]+)\.js$/.test(url.pathname)) {
      return route.fulfill({ contentType: "text/javascript", body: await readFile(new URL(url.pathname.slice(1), root), "utf8") });
    }
    return route.abort();
  });
  await page.goto(origin);
  await page.evaluate(async () => {
    window.requests = [];
    Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true });
    window.OffscreenCanvas = class {
      getContext() { return { drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(16).fill(128) }) }; }
    };
    window.fetch = async () => ({ ok: true, blob: async () => new Blob() });
    window.createImageBitmap = async () => ({});
    window.Worker = class {
      constructor(url) { this.sim = String(url).includes("floodsim-worker"); }
      terminate() {}
      postMessage(msg) {
        if (this.sim) {
          queueMicrotask(() => {
            this.onmessage({ data: { frame: { t: 3600 }, h: new Float32Array([0.2, 0.2, 0.2, 0.2]) } });
            this.onmessage({ data: { done: { steps: 1, hMax: new Float32Array([0.2, 0.2, 0.2, 0.2]) } } });
          });
          return;
        }
        if (msg.type === "init") return;
        window.requests.push(msg);
        const win = { x0: 0, y0: 0, nx: 2, ny: 2, zoom: 1 };
        let result;
        if (msg.type === "catalog") result = {};
        else if (msg.type === "floodsim" && msg.op === "plan") result = { ok: true, river_id: 230301988, window: win, reaches: {} };
        else if (msg.type === "floodsim" && msg.op === "prepare") result = { ok: true, nx: 2, ny: 2, dx: 100, window: win, z: new Float32Array(4), ground: new Float32Array(4), n: new Float32Array(4), edges: {}, sources: [], click: { flow_cms: 500, return_period: 100 }, coarse: true };
        else if (msg.type === "studio" && msg.op === "start") result = { workspace: { status: "review", messages: [], study: { steps: [{ id: "s1", tool: "flood_inundation", arguments: msg.intake.flood_inundation }] } }, reply: { kind: "plan", payload: {} } };
        else throw new Error(`Unexpected worker request ${msg.type}:${msg.op}`);
        queueMicrotask(() => this.onmessage({ data: { id: msg.id, type: "result", result } }));
      }
    };
    const flood = await import("/src/floodsim-map.js?v=__BUILD__");
    const studio = await import("/src/studio.js?v=__BUILD__");
    flood.initFloodIt();
    studio.initStudy();
    await flood.floodIt({ river_id: 230301988, lat: 47.56, lon: 7.59 });
  });
  await page.locator('#floodit.done').waitFor({ state: "visible" });
  await page.locator('#floodit [data-rp="50"]').click();
  await page.locator('#floodit [data-rp="50"][aria-pressed="true"]').waitFor();
  await page.locator('#floodit.done').waitFor();
  await page.evaluate(async () => (await import("/src/studio.js?v=__BUILD__")).openStudy());
  await page.getByRole("button", { name: "Study this screening flood" }).click();
  await page.locator('#study-board[data-status="review"]').waitFor({ state: "visible" });
  const requests = await page.evaluate(() => window.requests);
  const start = requests.find(r => r.type === "studio" && r.op === "start");
  assert.deepEqual({ lat: start.lat, lon: start.lon, scenario: start.intake.flood_inundation }, {
    lat: 47.56, lon: 7.59, scenario: { river_id: 230301988, lat: 47.56, lon: 7.59, return_period: 50, hours: 48, coarse: true },
  });
  assert.deepEqual(errors, []);
});
