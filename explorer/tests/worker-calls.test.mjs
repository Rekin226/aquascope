import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const SOURCE = readFileSync(new URL("../worker.js", import.meta.url), "utf8");

// worker.js in a sandbox, with a stand-in for Pyodide. Its runPythonAsync yields first, as the real one does
// while it loads the packages the code imports, then reads the arguments the way the Python does: through
// `json.loads(__aqCall("<key>", "args"))`, or through a bare `json.loads(__aqSomething)` global as the worker
// once did. A null there is the TypeError Python raises on JsNull. The result is the arguments, echoed.
function sandbox() {
  const posted = [];
  const ctx = vm.createContext({ console: { error() {}, warn() {}, log() {} }, setTimeout });
  ctx.self = ctx;
  ctx.postMessage = (m) => posted.push(m);
  vm.runInContext(SOURCE, ctx);
  const delays = [];
  ctx.__fake = {
    async runPythonAsync(code) {
      await new Promise((resolve) => setTimeout(resolve, delays.shift() ?? 0));
      const m = code.match(/json\.loads\((__aq\w+)(?:\(([^)]*)\))?\)/);
      if (!m) return "null";
      const value = m[2] === undefined ? ctx[m[1]] : ctx[m[1]](...JSON.parse(`[${m[2]}]`));
      if (value === null || value === undefined) {
        throw new Error("TypeError: the JSON object must be str, bytes or bytearray, not JsNull");
      }
      return value;
    },
  };
  vm.runInContext("pyodide = __fake; ready = Promise.resolve();", ctx);
  return { ctx, posted, delays };
}

// Two calls of each type, the first slower to start than the second: the order that cleared the first one's
// arguments when they travelled through one global per type (several gauges selected quickly).
const CASES = [
  ["assess", "lat", { lat: 10, lon: 1 }, { lat: 20, lon: 2 }],
  ["engineering", "tool", { op: "export", tool: "hecras" }, { op: "export", tool: "swmm" }],
  ["compare", "years", { stations: [], years: 10 }, { stations: [], years: 30 }],
  ["context", "name", { op: "point", name: "dams", lat: 1, lon: 1 }, { op: "point", name: "soil", lat: 1, lon: 1 }],
  ["watch", "today", { op: "digest", today: "2026-10-01" }, { op: "digest", today: "2026-10-02" }],
  ["workbench", "analysis", { analysis: "trend" }, { analysis: "gev" }],
  ["tool", "name", { name: "find_stations" }, { name: "flood_frequency" }],
  ["ask", "question", { question: "first" }, { question: "second" }],
  ["solve_plan", "problem", { problem: "flood", lat: 1, lon: 1 }, { problem: "drought", lat: 1, lon: 1 }],
  ["coerce_intake", "playbook", { playbook: "flood" }, { playbook: "drought" }],
  ["solve_run", "study", { study: { a: 1 } }, { study: { a: 2 } }],
  ["ingest", "filename", { text: "x", filename: "a.csv" }, { text: "y", filename: "b.csv" }],
  ["area_study", "question", { op: "run", question: "one" }, { op: "run", question: "two" }],
];

for (const [type, field, first, second] of CASES) {
  test(`two ${type} calls in flight each read their own arguments`, async () => {
    const { ctx, posted, delays } = sandbox();
    delays.push(20, 0);
    await Promise.all([
      ctx.onmessage({ data: { type, id: 1, ...first } }),
      ctx.onmessage({ data: { type, id: 2, ...second } }),
    ]);
    assert.deepEqual(posted.filter((m) => m.type === "error"), []);
    const results = Object.fromEntries(posted.filter((m) => m.type === "result").map((m) => [m.id, m.result]));
    // compared as JSON: the results are the sandbox's objects, with its own prototypes
    assert.equal(JSON.stringify(results[1][field]), JSON.stringify(first[field]));
    assert.equal(JSON.stringify(results[2][field]), JSON.stringify(second[field]));
    assert.equal(vm.runInContext("calls.size", ctx), 0, "every call drops its slots when it ends");
  });
}

test("a call that fails still drops its slots", async () => {
  const { ctx, posted } = sandbox();
  ctx.__fake.runPythonAsync = async () => { throw new Error("ValueError: boom"); };
  await ctx.onmessage({ data: { type: "assess", id: 7, lat: 1, lon: 1 } });
  assert.deepEqual(posted.filter((m) => m.type === "error").map((m) => m.id), [7]);
  assert.equal(vm.runInContext("calls.size", ctx), 0);
});
