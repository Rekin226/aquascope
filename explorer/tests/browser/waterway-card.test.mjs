import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const origin = "http://aquascope.test";
const provider = "https://waterways.example.test/api/interpreter";
const root = new URL("../../", import.meta.url);
const fixture = async (name) => JSON.parse(await readFile(new URL(`../fixtures/${name}`, import.meta.url), "utf8"));

test("the Explorer names a snapped reach without blocking or overwriting the next card", async () => {
  const modulePath = process.env.AQ_PLAYWRIGHT_MODULE;
  const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : "playwright");
  const browser = await chromium.launch({ executablePath: process.env.AQ_CHROMIUM_EXECUTABLE || undefined });
  const page = await browser.newPage();
  const requests = new EventEmitter();
  const pending = new Set();
  const seen = [];
  const cologne = { lat: 50.9375, lon: 6.97, snap_lat: 50.937552, snap_lon: 6.965579, river_id: 101 };
  const basel = { lat: 47.5596, lon: 7.5886, snap_lat: 47.560169, snap_lon: 7.589742, river_id: 202 };
  try {
    page.setDefaultTimeout(5000);
    await page.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.href === provider) {
        seen.push(route.request());
        const response = await new Promise((reply) => {
          pending.add(reply);
          requests.emit("request", (result) => {
            pending.delete(reply);
            reply(result);
          });
        });
        return route.fulfill({ ...response, headers: { "Access-Control-Allow-Origin": origin } });
      }
      if (url.origin !== origin) return route.abort();
      if (url.pathname === "/") {
        const html = await readFile(new URL("index.html", root), "utf8");
        return route.fulfill({ contentType: "text/html", body: html
          .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "").replace(/<link\b[^>]*>/g, "") });
      }
      if (url.pathname === "/config.js" || /^\/src\/[\w-]+\.js$/.test(url.pathname)) {
        return route.fulfill({ path: fileURLToPath(new URL(url.pathname.slice(1), root)), contentType: "text/javascript" });
      }
      if (/^\/(?:light-)?worker\.js$/.test(url.pathname)) {
        return route.fulfill({ contentType: "text/javascript", body: "self.onmessage = () => {};" });
      }
      return route.abort();
    });
    await page.goto(origin);
    await page.evaluate(async (endpoint) => {
      const { CONFIG } = await import("/config.js?v=__BUILD__");
      CONFIG.waterwayOverpassURL = endpoint;
      const { actions, state } = await import("/src/core.js?v=__BUILD__");
      const { initMapCard } = await import("/src/map-card.js?v=__BUILD__");
      document.body.classList.add("panel-collapsed");
      const panel = document.getElementById("panel-point");
      panel.hidden = false;
      actions.openRoute = (target) => { globalThis.lastRoute = target; };
      initMapCard();
      globalThis.snapCard = (point, select = true) => {
        if (select) {
          state.point = { lat: point.lat, lon: point.lon };
          state.pointLabel = null;
          document.dispatchEvent(new CustomEvent("aq:surface", { detail: { id: "panel-point" } }));
        }
        panel.dispatchEvent(new CustomEvent("riversnap", { detail: { ...point, snapped: true } }));
        return document.getElementById("mc-title").textContent;
      };
    }, provider);
    const start = async (point) => {
      const [[reply], fallback] = await Promise.all([
        once(requests, "request", { signal: AbortSignal.timeout(5000) }),
        page.evaluate((coords) => globalThis.snapCard(coords), point),
      ]);
      assert.equal(fallback, `River reach ${point.river_id}`);
      return reply;
    };
    const respond = async (reply, point, response) => {
      const matching = page.waitForResponse((entry) => entry.url() === provider &&
        new URLSearchParams(entry.request().postData()).get("data")?.includes(`${point.snap_lat},${point.snap_lon}`));
      reply(response);
      await (await matching).finished();
      await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
    };
    const title = page.locator("#mc-title");
    const credit = page.locator('#map-card a[href*="openstreetmap.org/copyright"]');

    const olderReply = await start(cologne);
    const newerReply = await start(basel);
    assert.equal(seen[0].method(), "POST");
    assert.match(new URLSearchParams(seen[0].postData()).get("data"), /50\.937552,6\.965579/);
    await respond(newerReply, basel, { json: await fixture("overpass-721-basel.json") });
    assert.equal(await title.textContent(), "Rhine");
    assert.match(await credit.textContent(), /OpenStreetMap contributors/);
    await page.locator('#map-card [data-act="route"]').click();
    assert.equal(await page.evaluate(() => globalThis.lastRoute.name), "Rhine");
    assert.equal(await page.evaluate((point) => globalThis.snapCard(point, false), basel), "Rhine");

    const older = await fixture("osm-721-cologne.json");
    for (const element of older.elements) {
      element.tags = { ...element.tags, name: "The previous reach", "name:en": "The previous reach" };
    }
    await respond(olderReply, cologne, { json: older });
    assert.equal(await title.textContent(), "Rhine");
    assert.equal(seen.length, 2);

    const other = { lat: 0, lon: 0, snap_lat: 0.001, snap_lon: 0.001, river_id: 303 };
    const failed = await start(other);
    await respond(failed, other, { status: 503, body: "Unavailable" });
    assert.equal(await title.textContent(), "River reach 303");
    assert.equal(await credit.count(), 0);

    const slow = { lat: 0.2, lon: 0.2, snap_lat: 0.201, snap_lon: 0.201, river_id: 404 };
    const [aborted] = await Promise.all([
      page.waitForEvent("requestfailed", { predicate: (request) => request.url() === provider, timeout: 8000 }),
      start(slow),
    ]);
    assert.equal(aborted.failure().errorText, "net::ERR_ABORTED");
    assert.equal(await title.textContent(), "River reach 404");
    assert.equal(await credit.count(), 0);
  } finally {
    for (const reply of pending) reply({ status: 503, body: "Unavailable" });
    await page.unrouteAll({ behavior: "ignoreErrors" });
    await browser.close();
  }
});
