// Offline layout regression for #611. Use the shipped HTML/CSS, without
// loading MapLibre, the catalog or Python; the live smoke covers the handoff.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

test("station actions stay clickable inside the scrolling inspector", async t => {
  const modulePath = process.env.AQ_PLAYWRIGHT_MODULE;
  const { chromium } = await import(modulePath ? pathToFileURL(modulePath).href : "playwright");
  const browser = await chromium.launch({
    executablePath: process.env.AQ_CHROMIUM_EXECUTABLE || undefined,
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const html = (await readFile(new URL("../../index.html", import.meta.url), "utf8"))
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
    .replace(/<link\b[^>]*>/g, "");
  const css = await readFile(new URL("../../style.css", import.meta.url), "utf8");
  try {
    // The daily smoke viewport, both responsive breakpoints, a phone, and a
    // short panel that must scroll. Optional exports/units change wrapping.
    for (const [width, height] of [[1280, 900], [1440, 900], [1100, 900], [900, 900],
      [860, 900], [768, 900], [390, 844], [320, 568], [1280, 420]]) {
      for (const optionalActions of [false, true]) {
        const page = await browser.newPage({ viewport: { width, height } });
        try {
          page.setDefaultTimeout(3000);
          await page.route("**/*", route => route.abort());
          await page.setContent(html);
          await page.addStyleTag({ content: css });
          await page.evaluate(optional => {
            const $ = id => document.getElementById(id);
            document.body.classList.remove("panel-collapsed");
            $("panel-empty").hidden = true;
            $("panel-station").hidden = false;
            $("st-name").textContent = "Fish River near Fort Kent, Maine";
            $("btn-csv").disabled = false;
            $("st-export").hidden = $("btn-unit").hidden = $("st-feed").hidden = !optional;
            $("st-export").innerHTML = '<option>Export for a model…</option>';
            const canvas = document.createElement("canvas");
            canvas.className = "maplibregl-canvas";
            canvas.style.cssText = "width:100%;height:100%";
            $("map").append(canvas);
            globalThis.mapClicks = 0;
            canvas.addEventListener("click", () => globalThis.mapClicks++);
            // Isolate the inspector, its toggle and the map underneath it.
            // Other map overlays have their own layout and service state.
            document.querySelector(".map-wrap").replaceChildren($("map"));
            // Witness real pointer delivery without invoking data services.
            globalThis.menuClicks = [];
            document.querySelector(".more-menu").addEventListener("click", event => {
              event.preventDefault();
              globalThis.menuClicks.push(event.target.id);
            });
          }, optionalActions);
          const menu = page.locator("details.more");
          t.diagnostic(`${width}x${height}, optional=${optionalActions}`);
          await menu.locator("summary").click();
          const label = `${width}x${height}, optional=${optionalActions}`;
          const items = menu.locator(".more-menu > :visible");
          for (const item of await items.all()) {
            await item.scrollIntoViewIfNeeded();
            const bounds = await item.evaluate(el => {
              const r = el.getBoundingClientRect();
              const panel = document.querySelector("#panel").getBoundingClientRect();
              return { left: r.left, right: r.right, panelLeft: panel.left, panelRight: panel.right,
                hit: el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)) };
            });
            assert.ok(bounds.left >= bounds.panelLeft && bounds.right <= bounds.panelRight,
              `${label}: menu must fit horizontally: ${JSON.stringify(bounds)}`);
            assert.ok(bounds.hit, `${label}: menu item must receive pointer events`);
            const id = await item.getAttribute("id");
            await item.click({ timeout: 3000 });
            assert.equal(await page.evaluate(() => globalThis.menuClicks.at(-1)), id, label);
          }
          await menu.locator("summary").click();
          assert.equal(await menu.getAttribute("open"), null, label);
          await page.locator("canvas").click({ position: { x: 20, y: 10 } });
          assert.equal(await page.evaluate(() => globalThis.mapClicks), 1, label);
        } finally { await page.close(); }
      }
    }
  } finally { await browser.close(); }
});
