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
let response;
let beforeResult = () => {};
globalThis.Worker = class {
  postMessage(message) {
    messages.push(message);
    if (message.type === "init") return;
    queueMicrotask(() => {
      if (message.type === "tool") beforeResult();
      this.onmessage({ data: { id: message.id, type: "result", result: message.type === "catalog" ? {} : response } });
    });
  }
};
const records = ["A891030101", "A891030102"].map((station_id) => ({
  source: "hubeau_hydrometrie", station_id, site_id: "A8910301", name: "Example gauge",
  latitude: 48, longitude: 7, period_start: "2000-01-01", period_end: null,
}));
const site = { ...records[0], record_count: 2, records };

function reset() {
  response = { stations: [site] };
  messages.length = 0;
  state.stations = records.map((r) => ({ ...r, lat: r.latitude, lon: r.longitude }));
  state.ask.catalogSent = false;
  state.hidden.clear();
  state.point = { lat: 48, lon: 7 };
  beforeResult = () => {};
  list.replaceChildren();
}

test("search sends site IDs to the shared engine and keeps grouped results", async () => {
  reset();
  assert.deepEqual(await searchStations("Example"), [site]);
  assert.equal(messages.find((m) => m.type === "catalog").rows[0].site_id, "A8910301");
  const request = messages.find((m) => m.type === "tool");
  assert.equal(request.name, "find_stations");
  assert.equal(request.arguments.query, "Example");
});

test("nearest panel has one site row and can open either member", async () => {
  reset();
  const opened = [];
  actions.selectStation = (key) => opened.push(key);
  await renderNearestStations(48, 7);
  assert.equal(list.children.length, 1);
  const [button, label] = list.children[0].children;
  assert.equal(label.children[0], "2 records at this site");
  button.events.click();
  const selector = label.children[1];
  selector.value = selector.children[2].value;
  selector.events.change();
  assert.deepEqual(opened, records.map((r) => `${r.source}/${r.station_id}`));
  assert.equal(selector.value, "");
  assert.equal(messages.find((m) => m.type === "tool").arguments.limit, 6);
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
  beforeResult = () => { state.point = { lat: 49, lon: 8 }; list.innerHTML = "new point"; };
  await renderNearestStations(48, 7);
  assert.equal(list.innerHTML, "new point");
});

test("failed nearest search offers a working retry", async () => {
  reset();
  response = { error: "test failure" };
  await renderNearestStations(48, 7);
  assert.match(list.children[0].textContent, /test failure/);
  response = { stations: [site] };
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
