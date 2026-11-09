// Scout (#563): the view the page hands the package, the pin a finding becomes, and the line under the bar.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  GLOBE_RADIUS_KM, KIND_LABELS, PIN_GAP_PX, SCOUT_MAX, askedFailed, keepVisible, refusedNote, scoutLine, scoutMonth, scoutPin, scoutView,
  scoutWho, wordingLine,
} from "../src/scout-core.js";
import { checkPin } from "../src/map-actions-core.js";

test("the globe zoomed out is a circle around the centre, never the far side", () => {
  const v = scoutView({ globe: true, zoom: 1.6, center: [-30, 25], width: 1440, height: 844 });
  assert.deepEqual(v.center, [25, -30]);
  assert.equal(v.radius_km, GLOBE_RADIUS_KM);
  // Two pins closer than a pin's height on screen cannot be told apart.
  const mpp = (40075016.686 * Math.cos((25 * Math.PI) / 180)) / (512 * 2 ** 1.6);
  assert.equal(v.min_km, Math.round((mpp * PIN_GAP_PX) / 1000));
  const near = scoutView({ globe: true, zoom: 3.5, center: [10, 50], width: 400, height: 300 });
  assert.ok(near.radius_km < GLOBE_RADIUS_KM && near.radius_km >= 50);
});

test("closer in, the outline of what is on screen; then the bounds; the whole world is null", () => {
  const outline = [[-10, 35], [30, 35], [30, 60], [-10, 60]];
  assert.deepEqual(scoutView({ globe: true, zoom: 5, center: [10, 48], outline }).polygon, outline);
  assert.deepEqual(scoutView({ zoom: 5, center: [10, 48], bounds: [-10.12345, 35, 30, 60] }).bbox, [-10.123, 35, 30, 60]);
  assert.deepEqual(scoutView({ zoom: 5, center: [180, 0], bounds: [170, -5, 190, 5] }).bbox, [170, -5, -170, 5]);
  assert.equal(scoutView({ zoom: 0.5, center: [0, 0], bounds: [-200, -85, 200, 85] }), null);
  assert.equal(scoutView({ zoom: 0.5, center: [0, 0], outline: [[-180, 0], [180, 0], [180, 10]] }), null);
  assert.equal(scoutView({ zoom: 5 }), null);
});

test("the month of the map date", () => {
  assert.equal(scoutMonth("2026-09-15"), "2026-09");
  assert.equal(scoutMonth(null), null);
});

test("a finding becomes a numbered pin with its reason and where it came from", () => {
  const f = {
    lat: -24.03, lon: -50.69, title: "Paraná, Brazil: 25-year flow forecast", reason: "GEOGLOWS forecast …",
    facts: [{ label: "Forecast peak", value: 2840, unit: "m³/s" }, { label: "a" }, 1, 2, 3, 4],
    source: "GEOGLOWS v2 forecast, CC BY 4.0", rank: 1, kind: "floods_ahead", placed_by: "photon",
  };
  const pin = scoutPin(f, { mode: "daily", made: "2026-10-10T15:00:00Z" });
  assert.equal(pin.kind, "Floods ahead");
  assert.equal(pin.text, f.reason);
  assert.equal(pin.facts.length, 5);
  assert.match(pin.source, /scout file of 2026-10-10; place name: Photon/);
  const checked = checkPin(pin);
  assert.equal(checked.rank, 1);
  assert.equal(checked.kind, "Floods ahead");
  assert.equal(checked.facts.length, 1);   // the bad facts are dropped by the log's own check
  assert.match(scoutPin({ ...f, placed_by: undefined }).source, /scanned in your browser\)$/);
  assert.equal(checkPin({ lat: 0, lon: 0, title: "x", rank: 0 }).rank, undefined);
  assert.equal(checkPin({ lat: 0, lon: 0, title: "x", rank: 2.5 }).rank, undefined);
});

test("who found them, and the line", () => {
  assert.deepEqual(scoutWho("rules"), ["Rules", "no model"]);
  assert.equal(scoutWho("device", "Gemini Nano")[0], "Gemini Nano");
  assert.match(scoutWho("key", "m via p")[1], /words by code/);
  assert.equal(scoutLine({ mode: "daily" }, 10), "10 pins, most worth a look first, from today's scout file");
  assert.equal(scoutLine({ mode: "live" }, 1), "1 pin, most worth a look first, scanned now");
  assert.match(scoutLine({ notes: ["no Floods ahead issue could be read"] }, 0), /^Nothing stands out in this view \(no Floods/);
  assert.equal(SCOUT_MAX, 10);
  assert.deepEqual(Object.keys(KIND_LABELS), ["floods_ahead", "status", "gauges_today", "floods_past", "models_disagree"]);
});

test("pins under a bar are left out and the rest renumbered, a model's extras kept after them", () => {
  const cands = Array.from({ length: 14 }, (_, i) => ({ id: `f${i}`, rank: i + 1, lat: i, lon: 0 }));
  const res = { mode: "live", picks: cands.slice(0, 10), candidates: cands };
  const hide = new Set(["f0", "f3"]);
  const out = keepVisible(res, (f) => !hide.has(f.id), 10, 0);
  assert.equal(out.picks.length, 10);
  assert.deepEqual(out.picks.map((f) => f.id), ["f1", "f2", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]);
  assert.deepEqual(out.picks.map((f) => f.rank), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(out.candidates.length, 10);
  assert.equal(out.hidden, 2);
  assert.equal(out.mode, "live");
  const withModel = keepVisible(res, (f) => !hide.has(f.id), 10, 6);
  assert.equal(withModel.candidates.length, 12);
  // A check that throws keeps the finding (better a pin under a bar than none).
  assert.equal(keepVisible(res, () => { throw new Error("no map"); }, 10, 0).picks[0].id, "f0");
});

test("the line names who ordered the pins and gives a refusal's reason (#590)", () => {
  const rules = (id) => ({ id, by: "rules" });
  const device = (id) => ({ id, by: "device" });
  // A reply outside the schema (Europe, 2026-10-25: a mangled id "2026-09-1") is refused whole, said in one line.
  let line = wordingLine({ by: "device", model: "Chrome's built-in model", picks: [rules("a"), rules("b"), rules("c")],
    refused: ["the model's reply was refused: reply.pins[0].id '2026-09-1' is not one of a, b, c"] });
  assert.equal(line.who, "Rules");
  assert.equal(line.how, "Chrome's built-in model asked, its choice not used");
  assert.deepEqual(line.notes, ["the model's reply was refused: reply.pins[0].id '2026-09-1' is not one of a, b, c; rules kept"]);
  // An id that is no pin here (the off-screen gauge of the recheck) is said, not lost.
  line = wordingLine({ by: "device", model: "Chrome's built-in model", picks: [rules("a")], unknown: ["USGS-01646500"] });
  assert.deepEqual(line.notes, ["the model wrote about USGS-01646500, not a pin in this view; rules kept"]);
  // Two of three picked on the device; the third follows in the rules' order.
  line = wordingLine({ by: "device", model: "Gemini Nano", picks: [device("a"), device("b"), rules("c")] });
  assert.equal(line.who, "Gemini Nano");
  assert.equal(line.how, "2 of 3 ordered on this device; words by code");
  assert.deepEqual(line.notes, []);
  // A pick whose variant the finding cannot take keeps the rules' wording, and says so.
  line = wordingLine({ by: "device", model: "Gemini Nano", picks: [device("a")],
    refused: ["status:much_below:2026-09:1: variant c cannot be written for it, the rules' wording is kept"] });
  assert.deepEqual(line.notes, ["1 pick not used, rules kept: variant c cannot be written for it, the rules' wording is kept"]);
  // All picked; no model.
  assert.equal(wordingLine({ by: "key", model: "m via p", picks: [{ by: "key" }] }).how, "ordered with your key; words by code");
  assert.deepEqual(wordingLine({ picks: [rules("a")] }), { who: "Rules", how: "no model", notes: [] });
  line = wordingLine({ failed: "the model could not be asked (no key); the rules' wording is kept", picks: [rules("a")] });
  assert.equal(line.how, "the model could not be asked");
  assert.equal(refusedNote([]), "");
});

test("after a timeout the line says the model was asked and timed out (#590)", () => {
  // Europe, 2026-10-26: the device ran to the 40 s timeout and the line said "Rules, no model".
  const abort = Object.assign(new Error("signal is aborted without reason"), { name: "AbortError" });
  const failed = askedFailed(abort, 40000);
  assert.equal(failed, "it was asked and timed out after 40 s; the rules' order and wording are kept");
  assert.equal(askedFailed(new Error("the on-device model took more than 40 s"), 40000), failed);
  assert.match(askedFailed(new Error("kErrorUnknown")), /asked and gave no answer \(kErrorUnknown\)/);
  const line = wordingLine({ by: "device", model: "Chrome's built-in model", picks: [{ id: "a", by: "rules" }], failed });
  assert.equal(line.who, "Rules");
  assert.equal(line.how, "Chrome's built-in model asked, no answer");
  assert.deepEqual(line.notes, [failed]);
});
