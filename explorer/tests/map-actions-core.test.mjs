// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  ActionLog, BY_LABEL, LAYER_CONTROLS, NOTE_SIDES, agoWords, checkPin, explainWho, inkOn, noteBody, noteRect, placeNotes,
  slotOf,
} from "../src/map-actions-core.js";
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

// ── Explain what is on screen (#562): notes ────────────────────────────────

test("a note keeps what its callout and card need, checked", () => {
  const p = checkPin({
    lat: 24.5, lon: 94, title: "Below normal", text: "Tan: flow in the 10th to 25th percentile.", kind: "note", n: 2,
    tone: "#dfc27d", place: "Manipur, India", when: "September 2026", credit: "GEOGLOWS v2, CC BY 4.0", group: "explain",
    facts: [{ label: "Largest area below normal in view", value: "41%", when: "September 2026" }],
  });
  assert.equal(p.kind, "note");
  assert.equal(p.n, 2);
  assert.equal(p.tone, "#dfc27d");
  assert.equal(p.place, "Manipur, India");
  assert.equal(p.group, "explain");
  assert.deepEqual(p.facts[0], { label: "Largest area below normal in view", value: "41%", when: "September 2026" });
  const bad = checkPin({ lat: 1, lon: 1, title: "x", kind: "note", tone: "red;background:url(x)", n: 1.5 });
  assert.equal(bad.tone, "", "only a plain hex colour");
  assert.equal(bad.n, 0);
  assert.equal(checkPin({ lat: 1, lon: 1, title: "x" }).kind, undefined, "a plain pin stays a pin");
});

test("a note's card gives its words, then the facts they leave out", () => {
  const pin = {
    text: "Dark teal: flow above the 90th percentile for September, on 23% of the mapped land in view.",
    when: "September 2026",
    facts: [
      { label: "Largest area much above normal in view", value: "23%", when: "September 2026" },
      { label: "Its size", value: "660,000 km²", when: "September 2026" },
      { label: "Gauges in view much above normal today", value: "5", when: "on 9 Oct" },
    ],
  };
  assert.equal(noteBody(pin), `${pin.text} · Its size: 660,000 km² · Gauges in view much above normal today: 5 (on 9 Oct)`);
});

test("the number on a dot is dark on the pale classes and white on the deep ones", () => {
  for (const pale of ["#dfc27d", "#80cdc1", "#f2c94c"]) assert.equal(inkOn(pale), "#0f1c26", pale);
  for (const deep of ["#a6611a", "#018571", "#8b5cf6", "#4a148c", "#33475a"]) assert.equal(inkOn(deep), "#ffffff", deep);
  assert.equal(inkOn("not a colour"), "#ffffff");
});

test("note cards go on the side that keeps them off each other and off the map's furniture", () => {
  const size = { w: 200, h: 60 };
  assert.deepEqual(noteRect({ x: 100, y: 100 }, size, "right"), { x: 114, y: 70, w: 200, h: 60 });
  assert.deepEqual(noteRect({ x: 100, y: 100 }, size, "top"), { x: 0, y: 26, w: 200, h: 60 });
  const frame = { w: 1000, h: 700 };
  // Alone in the middle: to the right.
  assert.deepEqual(placeNotes([{ x: 500, y: 300 }], [size], frame), ["right"]);
  // Near the right edge: to the left, inside the frame.
  assert.deepEqual(placeNotes([{ x: 950, y: 300 }], [size], frame), ["left"]);
  // Two side by side: neither card covers the other's point, so they open away from each other.
  assert.deepEqual(placeNotes([{ x: 400, y: 300 }, { x: 520, y: 300 }], [size, size], frame), ["left", "right"]);
  // A legend on the right of a point pushes its card elsewhere.
  const legend = { x: 510, y: 200, w: 300, h: 200 };
  assert.notEqual(placeNotes([{ x: 500, y: 300 }], [size], frame, [legend])[0], "right");
  assert.deepEqual(NOTE_SIDES, ["right", "left", "top", "bottom"]);
});

test("when the model's notes are refused, the line says so, not 'no model' (#590)", () => {
  assert.deepEqual(explainWho({ model: "Chrome's built-in model", notes: [], refused: ["the model's reply was refused: reply.notes[0].note 'f4' is not one of n1, n2"] }),
    ["Rules", "Chrome's built-in model's choice was refused"]);
  assert.deepEqual(explainWho({ model: "Chrome's built-in model", error: "the on-device model took more than 30 s" }),
    ["Rules", "Chrome's built-in model did not answer (the on-device model took more than 30 s)"]);
  assert.deepEqual(explainWho({ model: "claude via anthropic", notes: [] }), ["Rules", "claude via anthropic chose no note"]);
});
