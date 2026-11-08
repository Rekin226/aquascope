import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const SOURCE = readFileSync(new URL("../worker.js", import.meta.url), "utf8");

// worker.js in a sandbox, with a stand-in for Pyodide. Its runPythonAsync yields first, as the real one does
// while it loads the packages the code imports, then reads the arguments the way the Python does: through
// `json.loads(__aqCall("<key>", "args"))`, or the ways the worker once did, a bare `json.loads(__aqSomething)` JS
// global or a `json.loads(_xx_args)` Python global set with pyodide.globals.set. A missing value there is the
// TypeError Python raises on JsNull. The result is the arguments, echoed.
function sandbox() {
  const posted = [];
  const ctx = vm.createContext({ console: { error() {}, warn() {}, log() {} }, setTimeout });
  ctx.self = ctx;
  ctx.postMessage = (m) => posted.push(m);
  vm.runInContext(SOURCE, ctx);
  const delays = [];
  const pyGlobals = new Map();
  ctx.__fake = {
    globals: { set: (name, value) => pyGlobals.set(name, value), get: (name) => pyGlobals.get(name) },
    async loadPackage() {},
    runPython() {},
    async runPythonAsync(code) {
      await new Promise((resolve) => setTimeout(resolve, delays.shift() ?? 0));
      const m = code.match(/json\.loads\((__aq\w+|_\w+_args)(?:\(([^)]*)\))?\)/);
      if (!m) return "null";
      let value;
      if (m[1].startsWith("__aq")) value = m[2] === undefined ? ctx[m[1]] : ctx[m[1]](...JSON.parse(`[${m[2]}]`));
      else value = pyGlobals.get(m[1]);
      if (value === null || value === undefined) {
        throw new Error("TypeError: the JSON object must be str, bytes or bytearray, not JsNull");
      }
      // floodsim once left its JSON in a Python global (_fs_out) rather than returning it
      const out = code.match(/^(_\w+_out) = json\.dumps/m);
      if (out) pyGlobals.set(out[1], value);
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
  ["device_ask", "words", { op: "plan", words: "first" }, { op: "plan", words: "second" }],
  ["paint", "question", { op: "start", question: "trend" }, { op: "start", question: "status" }],
  ["map_command", "text", { op: "parse", text: "zoom to Paris" }, { op: "parse", text: "zoom to Lyon" }],
  ["scout", "context", { op: "prompt", context: "one" }, { op: "prompt", context: "two" }],
  ["whatif_words", "text", { op: "parse", text: "double the rain" }, { op: "parse", text: "halve the rain" }],
  ["mesh_review", "question", { op: "floods", question: "one" }, { op: "floods", question: "two" }],
  ["flood_exposure", "bbox", { bbox: [0, 0, 1, 1] }, { bbox: [2, 2, 3, 3] }],
  ["map_explain", "context", { op: "check", context: "one" }, { op: "check", context: "two" }],
  ["groundwater", "month", { op: "surface", month: "2026-01" }, { op: "surface", month: "2026-02" }],
  ["groundwater", "t", { op: "views", t: 1 }, { op: "views", t: 2 }],
  ["gw_layers", "cell", { op: "hydrograph", cell: 5 }, { op: "hydrograph", cell: 9 }],
  ["reservoir", "args", { op: "plan", args: { polygon: 1 } }, { op: "plan", args: { polygon: 2 } }],
  ["floodsim", "args", { op: "forecast", args: { river_id: 1 } }, { op: "forecast", args: { river_id: 2 } }],
  ["replay", "args", { op: "candidates", args: { lat: 1 } }, { op: "candidates", args: { lat: 2 } }],
  ["well", "args", { op: "map", args: { wells: [1] } }, { op: "map", args: { wells: [2] } }],
  ["landuse", "args", { op: "change", args: { use: "forest" } }, { op: "change", args: { use: "urban" } }],
];

for (const [type, field, first, second] of CASES) {
  test(`two ${type} ${first.op || ""} calls in flight each read their own arguments`, async () => {
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

test("a missing or null slot reaches Python as None, not jsnull", () => {
  const { ctx } = sandbox();
  vm.runInContext('calls.set("k", { bytes: null })', ctx);
  assert.equal(ctx.__aqCall("k", "bytes"), undefined);
  assert.equal(ctx.__aqCall("k", "nothing"), undefined);
  assert.equal(ctx.__aqCall("gone", "args"), undefined);
});
