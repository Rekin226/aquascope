// node --test explorer/tests/
//
// studio.js is not pure: it reaches for the DOM through $ and talks to the
// Pyodide worker through worker-client.js. This stubs just enough of both to
// exercise the waiting board, the done board and addFiles under node:
//
//   - a generic element stand-in for document.getElementById/createElement
//     (every DOM method it offers is a no-op; every property just stores and
//     reads back whatever studio.js itself sets, so it behaves like a real
//     element without knowing in advance which ids studio.js will ask for);
//   - a fake Worker whose postMessage resolves through a responder each test
//     sets, standing in for the worker's postMessage/onmessage round trip
//     (worker-client.js's callCancelable/call run unmodified against it).
//
// Both are module-level, set after the static imports below run (none of
// which touches document, Worker or location at their own top level, only
// inside functions this file calls later), so they are in place before any
// test does.

import test from "node:test";
import assert from "node:assert/strict";

import { S, BOARDS, addFiles, boardStatus } from "../src/studio.js";
import { state } from "../src/core.js?v=__BUILD__";

const METHODS = new Set([
  "addEventListener", "removeEventListener", "focus", "blur", "click", "scrollIntoView",
  "insertAdjacentHTML", "remove", "setAttribute", "removeAttribute", "appendChild", "removeChild",
  "dispatchEvent",
]);
const QUERIES = new Set(["querySelector", "closest"]);

function makeEl() {
  const target = {
    dataset: {}, style: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  };
  return new Proxy(target, {
    get(t, prop) {
      if (prop in t) return t[prop];
      if (prop === "querySelectorAll") return () => [];
      if (QUERIES.has(prop)) return () => null;
      if (METHODS.has(prop)) return () => undefined;
      return undefined;
    },
    set(t, prop, value) { t[prop] = value; return true; },
  });
}

const elements = new Map();
globalThis.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeEl());
    return elements.get(id);
  },
  createElement() { return makeEl(); },
  addEventListener() {}, removeEventListener() {},
  body: makeEl(),
};
globalThis.location = { href: "http://localhost/" };

const posted = [];
let responder = null;

class FakeWorker {
  constructor() { this.onmessage = null; this.onerror = null; }
  postMessage(msg) {
    posted.push(msg);
    if (msg.type === "init") return;   // the boot message: no reply needed for these tests
    queueMicrotask(() => {
      const reply = responder ? responder(msg) : null;
      if (this.onmessage) {
        this.onmessage({ data: reply || { type: "error", id: msg.id, message: `no stub for ${msg.type} ${msg.op || ""}` } });
      }
    });
  }
  terminate() {}
}
globalThis.Worker = FakeWorker;

// No station catalog is loaded in this file: skip the round trip a real Ask/Study session would make first.
state.ask.catalogSent = true;

function resetState() {
  S.ws = null; S.site = null; S.busy = false; S.writing = false; S.declined = null; S.files = [];
  S.catchment = null; S.area = null; S.donors = null;
  posted.length = 0;
  responder = null;
}

// ── boardStatus ──────────────────────────────────────────────────────────────

test("boardStatus gives waiting its own board instead of folding it into running", () => {
  resetState();
  assert.equal(boardStatus(), "intake", "no workspace yet");
  S.ws = { status: "waiting" };
  assert.equal(boardStatus(), "waiting");
  // Waiting is new; the boards it used to fold into still map to themselves.
  for (const st of ["intake", "review", "done", "declined"]) {
    S.ws = { status: st };
    assert.equal(boardStatus(), st);
  }
  S.ws = { status: "waiting" };
  S.busy = true;
  assert.equal(boardStatus(), "running", "a call in flight still wins, waiting or not");
});

// ── the waiting board ────────────────────────────────────────────────────────

test("the waiting board shows the request, and Continue without only when the crew allows it", () => {
  resetState();
  S.ws = {
    status: "waiting", tables: {},
    pending_request: {
      what: "abstraction (pumping) records for the wells around the site",
      why: "a decline can only be attributed to pumping, drought or land use with the abstraction history",
      effect: "with them the cause can be tested; without them the study stops at what the levels show",
      can_continue: true,
    },
  };
  let html = BOARDS.waiting();
  assert.match(html, /abstraction \(pumping\) records for the wells/);
  assert.match(html, /Why: a decline can only be attributed/);
  assert.match(html, /What it changes: with them the cause can be tested/);
  assert.match(html, /data-act="continue"/);
  assert.match(html, /Continue without/);
  assert.match(html, /id="study-file"/, "the file drop area is offered while waiting");

  S.ws.pending_request.can_continue = false;
  html = BOARDS.waiting();
  assert.doesNotMatch(html, /data-act="continue"/, "no continuation the crew allows: no button");
  assert.match(html, /cannot answer this brief/);
});

// ── the done board ───────────────────────────────────────────────────────────

test("the done board shows the grade badge, the decision bullets and the findings", () => {
  resetState();
  S.ws = {
    status: "done", tables: {}, artifacts: [],
    report: {
      answer: "Design flow: 12.3 m3/s.",
      grade: "indicative",
      decision: {
        answer: "Design flow: 12.3 m3/s (indicative).",
        conditions: ["the annual maxima are independent"],
        limitations: ["step s2 did not pass a gate"],
        what_would_change_it: ["a longer record for gev_lmoments"],
      },
      data_requests: [{ what: "abstraction records", why: "attribute the cause", effect_on_grade: "moves to established" }],
      findings: [{ id: "f1", claim: "Q5: 12.3 m3/s", basis: ["s3.ffa.fits.gev_lmoments.q.5"], grade: "indicative" }],
      key_numbers: [], not_established: [],
    },
  };
  const html = BOARDS.done();
  assert.match(html, /class="study-grade grade-indicative"/);
  assert.match(html, /title="indicative: a fallback/);
  assert.match(html, />indicative</);
  assert.match(html, /Design flow: 12\.3 m3\/s \(indicative\)\./);
  assert.match(html, /Holds if:/);
  assert.match(html, /Holds if:<\/p><ul><li>the annual maxima are independent<\/li><\/ul>/);
  assert.match(html, /Limitations and unresolved checks:<\/p><ul><li>step s2 did not pass a gate/);
  assert.match(html, /step s2 did not pass a gate/);
  assert.match(html, /Would change it:/);
  assert.match(html, /a longer record for gev_lmoments/);
  assert.match(html, /Additional evidence needed:/);
  assert.match(html, /abstraction records: moves to established/);
  assert.match(html, /<details class="study-findings"><summary>Findings<\/summary>/);
  assert.match(html, /\[indicative\] Q5: 12\.3 m3\/s/);
  assert.match(html, /s3\.ffa\.fits\.gev_lmoments\.q\.5/);
  // The existing answer article is still there, unmoved.
  assert.match(html, /<article class="study-answer/);
});

test("the done board renders nothing new for a report that predates the findings (#419)", () => {
  resetState();
  S.ws = {
    status: "done", tables: {}, artifacts: [],
    report: { answer: "An older recorded answer.", key_numbers: [], not_established: [] },
  };
  const html = BOARDS.done();
  assert.doesNotMatch(html, /study-grade/);
  assert.doesNotMatch(html, /study-decision/);
  assert.doesNotMatch(html, /study-findings/);
  assert.match(html, /An older recorded answer\./);
});

// ── addFiles at any status ──────────────────────────────────────────────────

test("addFiles, once a study exists, attaches the table to it instead of staging it", async () => {
  resetState();
  S.ws = { id: "ws1", status: "review", tables: {}, brief: {}, messages: [] };
  let called = null;
  responder = (msg) => {
    if (msg.type === "studio" && msg.op === "add_table") {
      called = msg;
      return {
        type: "result", id: msg.id,
        result: {
          reply: { kind: "answer", text: "flows.csv is attached; the crew will use it.", payload: {} },
          workspace: { ...msg.workspace, tables: { ...msg.workspace.tables, [msg.name]: msg.csv } },
          status: "review",
        },
      };
    }
    return { type: "error", id: msg.id, message: `unexpected op ${msg.op}` };
  };
  await addFiles([{ name: "flows.csv", text: async () => "date,value\n2020-01-01,1\n" }]);
  assert.ok(called, "the worker's add_table op was called");
  assert.equal(called.workspace.id, "ws1");
  assert.equal(called.name, "flows.csv");
  assert.equal(called.csv, "date,value\n2020-01-01,1\n");
  assert.deepEqual(S.files, [], "the table went straight to the workspace, not the pre-study staging list");
  assert.equal(S.ws.tables["flows.csv"], "date,value\n2020-01-01,1\n", "the reply was applied");
});

test("addFiles, before a study starts, still only stages the table (unchanged)", async () => {
  resetState();
  await addFiles([{ name: "flows.csv", text: async () => "date,value\n2020-01-01,1\n" }]);
  assert.equal(posted.filter((m) => m.type === "studio").length, 0, "no worker call yet: there is no study to attach to");
  assert.deepEqual(S.files, [{ id: "flows.csv", csv: "date,value\n2020-01-01,1\n" }]);
});
