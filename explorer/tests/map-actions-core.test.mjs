// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { ActionLog, BY_LABEL, LAYER_CONTROLS, agoWords, checkPin, slotOf } from "../src/map-actions-core.js";
import { focusPalette, statusPalette } from "../src/status-core.js";

const fly = (to) => ({ type: "fly_to", bbox: to });

test("each action has the slot it changes; pins and areas each their own", () => {
  assert.equal(slotOf(fly([0, 0, 1, 1])), "view");
  assert.equal(slotOf({ type: "set_time", date: "2023-09-15" }), "time");
  assert.equal(slotOf({ type: "set_layer", layer: "status", on: true }), "layer:status");
  assert.equal(slotOf({ type: "focus_status", classes: [] }), "status-focus");
  assert.equal(slotOf({ type: "set_basemap", basemap: "dark" }), "basemap");
  assert.equal(slotOf({ type: "highlight_river" }), "river");
  assert.equal(slotOf({ type: "add_pin" }, 7), "pin:7");
  assert.equal(slotOf({ type: "draw_area" }, 8), "area:8");
  assert.equal(slotOf({ type: "launch" }), null);
});

test("undoing the newest action puts back what it replaced", () => {
  const log = new ActionLog();
  const e = log.add({ action: { type: "set_layer", layer: "status", on: false }, before: true, after: false, label: "Turn off World river status" });
  assert.equal(e.slot, "layer:status");
  assert.deepEqual(log.undo(e.id).restore, { slot: "layer:status", value: true });
  assert.equal(log.live().length, 0);
  assert.equal(log.undo(e.id), null, "an undone action cannot be undone twice");
});

test("undoing an older action leaves the screen alone and hands its 'before' on", () => {
  const log = new ActionLog();
  const a = log.add({ action: fly([1, 1, 2, 2]), before: "world", after: "A" });
  const b = log.add({ action: fly([3, 3, 4, 4]), before: "A", after: "B" });
  const res = log.undo(a.id);
  assert.equal(res.restore, null, "the view now belongs to the later flight");
  assert.equal(log.get(b.id).before, "world");
  assert.deepEqual(log.undo(b.id).restore, { slot: "view", value: "world" }, "and undoing that one goes all the way back");
});

test("undo all puts every slot back as it was before the first action touched it", () => {
  const log = new ActionLog();
  log.add({ action: fly([1, 1, 2, 2]), before: "world", after: "A" });
  log.add({ action: { type: "set_time", date: "2023-09-15" }, before: { date: "2026-09-15" }, after: { date: "2023-09-15" } });
  log.add({ action: fly([3, 3, 4, 4]), before: "A", after: "B" });
  const id = log.peekId();
  log.add({ action: { type: "add_pin", title: "x" }, before: null, after: { lat: 1, lon: 2, title: "x" } });
  const restores = log.undoAll();
  assert.deepEqual(restores, [
    { slot: "view", value: "world" },
    { slot: "time", value: { date: "2026-09-15" } },
    { slot: `pin:${id}`, value: null },
  ]);
  assert.equal(log.live().length, 0);
  assert.deepEqual(log.undoAll(), [], "nothing left to undo");
});

test("pins are listed with who dropped them, and leave the list when undone", () => {
  let now = 1000;
  const log = new ActionLog({ now: () => now });
  const pin = checkPin({ lat: 23.7123456, lon: 90.4, title: "  Dhaka  ", text: "check the gauges",
    facts: [{ label: "flow", value: 12.5, unit: "m³/s" }, { label: "no value" }], source: "GEOGLOWS v2, CC BY 4.0" });
  assert.deepEqual(pin, { lat: 23.71235, lon: 90.4, title: "Dhaka", text: "check the gauges",
    facts: [{ label: "flow", value: 12.5, unit: "m³/s" }], source: "GEOGLOWS v2, CC BY 4.0" });
  const e = log.add({ action: { type: "add_pin", ...pin }, before: null, after: pin, by: "api", label: "Pin: Dhaka" });
  log.add({ action: fly([0, 0, 1, 1]), before: "world", after: "A" });
  assert.deepEqual(log.pins(), [{ id: e.id, ...pin, by: "api", at: 1000 }]);
  log.undo(e.id);
  assert.deepEqual(log.pins(), []);
});

test("a pin without a place or a title is refused", () => {
  assert.throws(() => checkPin({ lat: 95, lon: 0, title: "x" }), /lat/);
  assert.throws(() => checkPin({ lat: 10, lon: 0, title: " " }), /title/);
  assert.throws(() => checkPin(null), /lat/);
});

test("listeners hear every change, and the oldest entries fall off past the cap", () => {
  const log = new ActionLog({ max: 3 });
  let heard = 0;
  log.subscribe(() => { heard += 1; });
  for (let i = 0; i < 5; i++) log.add({ action: fly([0, 0, 1, 1]), before: i, after: i + 1 });
  assert.equal(heard, 5);
  assert.deepEqual(log.entries.map((e) => e.id), [3, 4, 5]);
  log.undo(5);
  assert.equal(heard, 6);
});

test("every layer the grammar names has a control on the page, and every 'by' a word", () => {
  for (const id of Object.values(LAYER_CONTROLS)) assert.match(id, /^(toggle-|ov-|btn-)/);
  assert.equal(BY_LABEL.device, "on device");
  assert.equal(agoWords(0, 20_000), "just now");
  assert.equal(agoWords(0, 5 * 60_000), "5 min ago");
});

test("the river status can be painted for some classes only", () => {
  const all = statusPalette();
  assert.deepEqual(focusPalette([]), all);
  assert.deepEqual(focusPalette(null), all);
  const only = focusPalette(["much_above"]);
  assert.equal(only[5][3], all[5][3], "much above keeps its strength");
  for (const i of [1, 2, 3, 4]) assert.equal(only[i][3], 0, "the others are left off the map");
  assert.deepEqual(only[0], all[0]);
});
