import test from "node:test";
import assert from "node:assert/strict";
import { state, actions } from "../src/core.js?v=__BUILD__";
import { initSearch, searchStations } from "../src/search.js";
import { renderNearestStations } from "../src/panel-point.js";

// Minimal DOM and worker doubles exercise the panel's controls and async boundary.
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.value = ""; }
  set innerHTML(value) { this.html = value; this.children = []; }
  get innerHTML() { return this.html || ""; }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); }
  replaceChildren(...nodes) { this.children = nodes; this.html = ""; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  add(option) { this.children.push(option); }
}
const list = new Element("ul");
globalThis.document = { getElementById: () => list, createElement: (tag) => new Element(tag) };
globalThis.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
globalThis.location = { href: "http://localhost/" };
const messages = [];
globalThis.Worker = class {
  constructor() {
    messages.push("worker started");
    throw new Error("Local search must not start Pyodide");
  }
};
const records = ["A891030101", "A891030102"].map((station_id) => ({
  source: "hubeau_hydrometrie", station_id, site_id: "A8910301", name: "Example gauge",
  latitude: 48, longitude: 7, period_start: "2000-01-01", period_end: null,
}));

function reset() {
  messages.length = 0;
  state.stations = records.map((r) => ({ ...r, lat: r.latitude, lon: r.longitude }));
  state.ask.catalogSent = false;
  state.hidden.clear();
  state.point = { lat: 48, lon: 7 };
  list.replaceChildren();
}

test("search groups synchronously without starting a worker", () => {
  reset();
  const hits = searchStations("Example");
  assert.equal(hits.length, 1);
  assert.equal(hits[0].record_count, 2);
  assert.equal(messages.length, 0);
  assert.equal(searchStations("A891030102")[0].station_id, "A891030102");
});

test("nearest panel has one site row and can open either member", async () => {
  reset();
  const opened = [];
  actions.selectStation = (key) => opened.push(key);
  renderNearestStations(48, 7);
  assert.equal(list.children.length, 1);
  const [button, label] = list.children[0].children;
  assert.equal(label.children[0], "2 records at this site");
  button.events.click();
  const selector = label.children[1];
  selector.value = selector.children[2].value;
  selector.events.change();
  assert.deepEqual(opened, records.map((r) => `${r.source}/${r.station_id}`));
  assert.equal(selector.value, "");
  assert.equal(messages.length, 0);
});

test("all sources hidden shows an empty result without an unfiltered search", async () => {
  reset();
  state.hidden.add("hubeau_hydrometrie");
  await renderNearestStations(48, 7);
  assert.match(list.innerHTML, /no gauges/);
  assert.equal(messages.length, 0);
});

test("an old point response cannot replace the current view", async () => {
  reset();
  state.point = { lat: 49, lon: 8 };
  await renderNearestStations(48, 7);
  assert.equal(list.children.length, 0);
});

test("failed nearest search offers a working retry", async () => {
  reset();
  state.stations = null;
  await renderNearestStations(48, 7);
  assert.match(list.children[0].textContent, /Could not load/);
  state.stations = records.map((r) => ({ ...r, lat: r.latitude, lon: r.longitude }));
  await list.children[0].children[0].events.click();
  assert.equal(list.children[0].className, "nearest-site");
});

test("closing search cancels the pending debounce", async () => {
  reset();
  const input = new Element("input"), box = new Element("div");
  input.value = "Ex";
  input.blur = () => {};
  const original = globalThis.document;
  globalThis.document = {
    ...original, getElementById: (id) => id === "search" ? input : box, addEventListener: () => {},
  };
  try {
    initSearch();
    input.events.input();
    input.events.keydown({ key: "Escape" });
    await new Promise((resolve) => setTimeout(resolve, 180));
    assert.equal(box.hidden, true);
    assert.equal(messages.length, 0);
  } finally {
    globalThis.document = original;
  }
});
