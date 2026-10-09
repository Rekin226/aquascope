// Records the Explorer's release demo: a scripted walk through the live app,
// captured by Playwright, then cut and encoded by ffmpeg into an MP4 that
// LinkedIn and Reddit accept as is (H.264, yuv420p, faststart, no audio).
//
//   npm i --no-save playwright          (or use a global install)
//   node explorer/demo/record_demo.mjs --url https://rekin226-aquascope-explorer.static.hf.space/
//   node explorer/demo/record_demo.mjs --url http://localhost:8000/ --out /tmp/demo
//
// The waits for Python, agency data and model fits are cut out of the final
// video (a one-second "working" beat is kept), so its length does not depend
// on how fast the network was. A scene that fails is logged and skipped rather
// than ending the take. Stills of each scene are saved next to the video.
//
// Nothing here is shipped by build.py; it drives the deployed page from outside.

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
  (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]?.startsWith("--") ? true : all[i + 1] ?? true]] : acc), []));
const URL_ = args.url || "https://rekin226-aquascope-explorer.static.hf.space/";
const OUT = resolve(args.out || "explorer-demo");
const W = Number(args.width || 1920), H = Number(args.height || 1080);
const VERSION = args.version ||
  readFileSync(join(HERE, "../../aquascope/__init__.py"), "utf8").match(/__version__ = "([^"]+)"/)?.[1] || "";
// The Thames at Kingston: the gauge behind the recorded flood study, so the scenes and the study agree.
const STATION = args.station || "uk_ea/63b6ae06-60fc-4e92-8337-aa0e5fdd7e79";
const QUERY = args.query || "Kingston";
const STUDY = args.study || "kingston-flood";

let chromium;
try { ({ chromium } = await import("playwright")); }
catch {
  // ES modules ignore NODE_PATH, so a global install has to be found by hand.
  try {
    const root = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
    ({ chromium } = await import(pathToFileURL(join(root, "playwright", "index.mjs")).href));
  } catch { console.error("playwright is not installed: npm i --no-save playwright"); process.exit(2); }
}

mkdirSync(OUT, { recursive: true });

// --- the overlay: captions, title cards and a visible cursor -----------------------------------------
// Injected before the app's own scripts. Pointer events pass through, so it never changes what a click hits.
const OVERLAY = () => {
  const css = `
  #demo-cursor{position:fixed;z-index:2147483647;width:26px;height:26px;margin:-4px 0 0 -4px;pointer-events:none;
    transition:transform .12s;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 2l7 19 2.6-7.4L20 11z' fill='%23111' stroke='%23fff' stroke-width='1.5' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat;}
  #demo-cursor.down{transform:scale(.82)}
  .demo-ripple{position:fixed;z-index:2147483646;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;
    border:3px solid #0a84ff;pointer-events:none;animation:demo-rip .55s ease-out forwards}
  @keyframes demo-rip{from{opacity:.9;transform:scale(.3)}to{opacity:0;transform:scale(1.4)}}
  #demo-caption{position:fixed;z-index:2147483645;left:50%;bottom:44px;transform:translateX(-50%) translateY(20px);
    max-width:min(1200px,80vw);padding:16px 28px;border-radius:14px;background:rgba(10,22,40,.88);color:#fff;
    font:600 30px/1.3 system-ui,-apple-system,"Segoe UI",sans-serif;text-align:center;opacity:0;pointer-events:none;
    transition:opacity .35s,transform .35s;box-shadow:0 10px 40px rgba(0,0,0,.35)}
  #demo-caption.on{opacity:1;transform:translateX(-50%) translateY(0)}
  #demo-caption small{display:block;margin-top:6px;font-weight:400;font-size:21px;opacity:.82}
  #demo-card{position:fixed;inset:0;z-index:2147483644;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:18px;background:linear-gradient(135deg,#06243f 0%,#0b4f7c 55%,#1488b5 100%);color:#fff;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;text-align:center;opacity:0;pointer-events:none;
    transition:opacity .6s}
  #demo-card.on{opacity:1}
  #demo-card .k{font-size:28px;letter-spacing:.18em;text-transform:uppercase;opacity:.8}
  #demo-card h1{margin:0;font-size:92px;font-weight:800;letter-spacing:-.02em}
  #demo-card p{margin:0;max-width:1300px;font-size:38px;line-height:1.35;opacity:.95}
  #demo-card code{font-size:34px;padding:10px 22px;border-radius:10px;background:rgba(255,255,255,.14)}
  #demo-card .u{font-size:30px;opacity:.9}`;
  const boot = () => {
    if (document.getElementById("demo-cursor")) return;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
    for (const id of ["demo-cursor", "demo-caption", "demo-card"]) {
      const el = document.createElement("div");
      el.id = id;
      document.body.appendChild(el);
    }
    const cur = document.getElementById("demo-cursor");
    cur.style.left = "-100px";
    addEventListener("mousemove", (e) => { cur.style.left = `${e.clientX}px`; cur.style.top = `${e.clientY}px`; }, true);
    addEventListener("mousedown", (e) => {
      cur.classList.add("down");
      const r = document.createElement("div");
      r.className = "demo-ripple";
      r.style.left = `${e.clientX}px`;
      r.style.top = `${e.clientY}px`;
      document.body.appendChild(r);
      setTimeout(() => r.remove(), 600);
    }, true);
    addEventListener("mouseup", () => cur.classList.remove("down"), true);
  };
  if (document.body) boot(); else addEventListener("DOMContentLoaded", boot);
  window.__demo = {
    caption(html) {
      const c = document.getElementById("demo-caption");
      if (!html) { c.classList.remove("on"); return; }
      c.innerHTML = html;
      c.classList.add("on");
    },
    card(html) {
      const c = document.getElementById("demo-card");
      if (!html) { c.classList.remove("on"); return; }
      c.innerHTML = html;
      c.classList.add("on");
    },
  };
};

// --- the edit list ----------------------------------------------------------------------------------
// Every stretch of the recording is kept unless it falls inside a cut(). Times are seconds since the
// page (and so the video) was created.
const t0 = Date.now();
const now = () => (Date.now() - t0) / 1000;
const cuts = [];
const log = (...a) => console.log(`[${now().toFixed(1).padStart(6)}s]`, ...a);

async function waiting(label, fn, { keep = 1.0, timeout = 120000 } = {}) {
  const start = now();
  try {
    await fn(timeout);
    return true;
  } catch (e) {
    log(`  ${label}: gave up (${e.message.split("\n")[0]})`);
    return false;
  } finally {
    const end = now();
    if (end - start > keep + 0.5) cuts.push([start + keep, end - 0.2]);
    log(`  ${label}: ${(end - start).toFixed(1)}s`);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  // A software GL is enough for MapLibre; the flag keeps WebGL on in headless containers.
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const context = await browser.newContext({
  viewport: { width: W, height: H },
  deviceScaleFactor: 1,
  recordVideo: { dir: join(OUT, "raw"), size: { width: W, height: H } },
  colorScheme: "light",
});
await context.addInitScript(OVERLAY);
const page = await context.newPage();
page.on("pageerror", (e) => log("page error:", e.message));

const caption = (title, sub = "") =>
  page.evaluate(([t, s]) => window.__demo.caption(t ? `${t}${s ? `<small>${s}</small>` : ""}` : ""), [title, sub]);
const card = (html) => page.evaluate((h) => window.__demo.card(h), html);
const still = (name) => page.screenshot({ path: join(OUT, `${name}.png`) }).catch(() => {});

async function clickEl(locator, { pause = 450 } = {}) {
  await locator.scrollIntoViewIfNeeded().catch(() => {});
  const box = await locator.boundingBox();
  if (!box) throw new Error("not on screen");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 28 });
  await sleep(pause);
  await page.mouse.down();
  await sleep(90);
  await page.mouse.up();
  await locator.click({ trial: true }).catch(() => {});
  await sleep(250);
}

// A tab in the station panel: click it, then wait for its card to fill.
async function showTab(tab, cardSel, title, sub, { hold = 6000, ready } = {}) {
  log(`scene: ${tab}`);
  await clickEl(page.locator(`#t-${tab}`));
  await caption(title, sub);
  await waiting(`${tab} ready`, (t) => (ready
    ? page.waitForFunction(ready, null, { timeout: t })
    : page.locator(cardSel).waitFor({ state: "visible", timeout: t })));
  await sleep(800);
  await still(`scene-${tab}`);
  await sleep(hold);
}

async function scene(name, fn) {
  try { await fn(); } catch (e) { log(`scene ${name} skipped: ${e.message.split("\n")[0]}`); await caption(""); }
}

// --- the take ---------------------------------------------------------------------------------------
log(`recording ${URL_} at ${W}x${H} into ${OUT}`);
await page.goto(URL_, { waitUntil: "domcontentloaded" });

await scene("title", async () => {
  await card(`<div class="k">AquaScope${VERSION ? ` v${VERSION}` : ""}</div><h1>AquaScope Explorer</h1>` +
    `<p>Every public water gauge we can reach, on one map.<br>Analysed in your browser. Nothing to install.</p>`);
  await sleep(3800);
});

await scene("map", async () => {
  await waiting("catalog", (t) => page.waitForFunction(() => {
    const c = document.getElementById("count");
    return c && !/loading/i.test(c.textContent) && /\d/.test(c.textContent);
  }, null, { timeout: t }));
  await card("");
  await sleep(700);
  const count = (await page.locator("#count").textContent().catch(() => "")).trim();
  await caption("Every public river gauge, on one map", count ? `${count} · open data from agencies worldwide` : "");
  await page.mouse.move(W * 0.45, H * 0.5, { steps: 20 });
  await sleep(2500);
  await still("scene-map");
  // A slow drift towards the gauge, so the map reads as alive before the search.
  await page.evaluate(() => window.__aq?.map?.flyTo({ center: [-0.31, 51.41], zoom: 6.2, duration: 5000, essential: true }));
  await sleep(5600);
});

await scene("search", async () => {
  log("scene: search");
  await caption("Search any gauge, river or station id");
  const box = page.locator("#search");
  await clickEl(box);
  await box.pressSequentially(QUERY, { delay: 110 });
  const hit = page.locator("#search-results .hit:not(.muted)").first();
  await waiting("search hits", (t) => hit.waitFor({ state: "visible", timeout: t }), { keep: 0.6 });
  await sleep(1300);
  // Prefer the Thames gauge when it is listed, otherwise the first hit.
  const thames = page.locator("#search-results .hit", { hasText: /thames/i }).first();
  await clickEl((await thames.count()) ? thames : hit);
});

await scene("station", async () => {
  // If the search found nothing usable, open the gauge by its link instead.
  const opened = await page.locator("#panel-station").isVisible().catch(() => false);
  if (!opened) await page.evaluate((s) => { location.hash = `s=${s}`; }, STATION);
  await caption("The observed record, straight from the agency",
    "Fetched and analysed by the aquascope Python package, running in your browser");
  await waiting("hydrograph", (t) => page.locator("#st-hydro-card").waitFor({ state: "visible", timeout: t }), { timeout: 180000 });
  await sleep(1200);
  await still("scene-record");
  await page.mouse.move(W * 0.78, H * 0.5, { steps: 30 });
  await sleep(5500);
});

await scene("floods", () => showTab("floods", "#st-ffa-card", "Flood frequency with confidence limits",
  "LP3 (Bulletin 17C) and GEV fits, return-period table, one click", {
    ready: () => document.querySelectorAll("#ffa-table tr").length > 2,
    hold: 7000,
  }));

await scene("flows", () => showTab("flows", "#st-fdc-card", "Flow duration and trend",
  "Low-flow indices and a Mann-Kendall trend on the same record"));

await scene("model", async () => {
  log("scene: model");
  await clickEl(page.locator("#t-model"));
  await caption("Calibrate a GR4J rainfall-runoff model", "Gauge record plus reanalysis climate, fitted in the tab");
  await sleep(1200);
  await clickEl(page.locator("#btn-gr4j"));
  await waiting("gr4j", (t) => page.waitForFunction(() => document.querySelectorAll("#gr4j-table tr").length > 1,
    null, { timeout: t }), { timeout: 240000 });
  await sleep(1000);
  await still("scene-model");
  await sleep(6000);
});

await scene("study", async () => {
  log("scene: study");
  await caption("");
  await page.evaluate((id) => { location.hash = `study=${id}`; }, STUDY);
  await caption("Ask a question, get a study", "Brief, plan, report and figures from real gauges, with every step checkable");
  await waiting("study", (t) => page.waitForFunction(() => {
    const d = document.getElementById("drawer");
    return d && !d.hidden && d.getBoundingClientRect().width > 100 && d.textContent.trim().length > 400;
  }, null, { timeout: t }));
  await sleep(1500);
  await still("scene-study");
  // Scroll the drawer slowly so the report reads as a document, not a screenshot.
  const drawer = page.locator("#drawer");
  const box = await drawer.boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.55, { steps: 25 });
    for (let i = 0; i < 14; i++) { await page.mouse.wheel(0, 140); await sleep(450); }
  }
  await sleep(2500);
});

await scene("end", async () => {
  await caption("");
  await card(`<h1>AquaScope</h1><p>Open-source hydrology: one Python engine,<br>a web Explorer, a CLI and an MCP server.</p>` +
    `<code>pip install aquascope</code>` +
    `<div class="u">rekin226-aquascope-explorer.static.hf.space<br>github.com/Rekin226/aquascope</div>`);
  await sleep(5500);
});

const video = page.video();
await context.close();   // flushes the recording
await browser.close();
const raw = await video.path();
log(`raw take: ${raw}`);

// --- the cut ----------------------------------------------------------------------------------------
const total = now();
cuts.sort((a, b) => a[0] - b[0]);
const keep = [];
let at = 0;
for (const [a, b] of cuts) {
  if (a > at) keep.push([at, a]);
  at = Math.max(at, b);
}
keep.push([at, total + 5]);
writeFileSync(join(OUT, "edit.json"), JSON.stringify({ url: URL_, total, cuts, keep }, null, 1));

const parts = keep.map(([a, b], i) =>
  `[0:v]trim=start=${a.toFixed(2)}:end=${b.toFixed(2)},setpts=PTS-STARTPTS[v${i}]`);
const filter = `${parts.join(";")};${keep.map((_, i) => `[v${i}]`).join("")}concat=n=${keep.length}:v=1:a=0,fps=30,format=yuv420p[out]`;
const mp4 = join(OUT, `aquascope-explorer-demo${VERSION ? `-v${VERSION}` : ""}.mp4`);
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", raw, "-filter_complex", filter, "-map", "[out]",
  "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-movflags", "+faststart", mp4], { stdio: "inherit" });
log(`done: ${mp4}`);
