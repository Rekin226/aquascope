// Exercise the station panel's real run/reset functions with an offline forcing stub.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const panel = await readFile(new URL("../src/panel-station.js", import.meta.url), "utf8");
const helper = await readFile(new URL("../src/gr4j-area.js", import.meta.url), "utf8");
const functions = panel.slice(panel.indexOf("function resetGr4j()"), panel.indexOf("// ── wiring"));

function harness(area, controls = {}) {
  const elements = Object.fromEntries(["gr4j-out", "gr4j-status", "btn-gr4j", "btn-gr4j-stop",
    "gr4j-area-warning", "gr4j-area-message", "btn-gr4j-continue"].map(id => [id, { hidden: true }]));
  let requests = 0;
  const context = vm.createContext({
    elements,
    $: id => elements[id],
    state: { selected: { lat: 1, lon: 2 }, result: { series: { t: ["2020-01-01"], v: [1] } } },
    stationKey: () => "station", stationArea: controls.stationArea || (async () => ({ area })),
    setStatusEl: (el, text) => { el.textContent = text; },
    fetch: async () => {
      requests++;
      if (controls.fetch) return controls.fetch();
      return { ok: true, json: async () => ({ daily: { time: [], precipitation_sum: [], et0_fao_evapotranspiration: [] } }) };
    },
    Cancelled: class Cancelled extends Error {},
  });
  vm.runInContext(`let gr4jRun = 0, gr4jCancel = null;\n${helper.replaceAll("export ", "")}\n${functions}\n` +
    "globalThis.run = runGr4j; globalThis.reset = resetGr4j; globalThis.stop = () => gr4jCancel?.();", context);
  return { ...context, requests: () => requests };
}

test("large basin does not fetch forcing until the user clicks Run anyway", async () => {
  const h = harness(10001), pending = h.run();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.requests(), 0);
  assert.equal(h.elements["gr4j-area-warning"].hidden, false);
  assert.equal(h.elements["gr4j-status"].textContent, "Review the catchment area before continuing.");
  h.elements["btn-gr4j-continue"].onclick();
  await pending;
  assert.equal(h.requests(), 1);
  assert.equal(h.elements["btn-gr4j"].disabled, false);
});

test("Stop restores the controls while area lookup or forcing download is in flight", async () => {
  for (const stage of ["stationArea", "fetch"]) {
    let release;
    const waiting = new Promise(resolve => { release = resolve; });
    const controls = stage === "stationArea"
      ? { stationArea: () => waiting }
      : { fetch: () => waiting };
    const h = harness(10, controls), pending = h.run();
    await new Promise(resolve => setImmediate(resolve));
    h.stop();
    release(stage === "stationArea" ? { area: 10 } : {
      ok: true, json: async () => ({ daily: { time: [], precipitation_sum: [], et0_fao_evapotranspiration: [] } }),
    });
    await pending;
    assert.equal(h.elements["btn-gr4j"].disabled, false, stage);
    assert.equal(h.elements["btn-gr4j-stop"].hidden, true, stage);
    assert.equal(h.elements["gr4j-status"].textContent, "Calibration stopped.", stage);
  }
});

test("threshold, Stop and station reset do not fetch forcing without consent", async () => {
  const small = harness(10000);
  await small.run();
  assert.equal(small.requests(), 1);
  for (const action of ["stop", "reset"]) {
    const h = harness(10001), pending = h.run();
    await new Promise(resolve => setImmediate(resolve));
    h[action]();
    await pending;
    assert.equal(h.requests(), 0, action);
    assert.equal(h.elements["gr4j-area-warning"].hidden, true, action);
    assert.equal(h.elements["btn-gr4j"].disabled, false, action);
  }
});
