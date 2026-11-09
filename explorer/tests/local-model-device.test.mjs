// The Prompt API engine (explorer/src/local-model.js) with a stubbed globalThis.LanguageModel, as Chrome 155
// offers it: each step's prompt gets the package's schema as responseConstraint and a signal, and a prompt that
// never replies is ended by the step's time limit or by Stop (#590, the fourth check: one prompt ran 176 s).
import { test } from "node:test";
import assert from "node:assert/strict";

const log = { prompts: [] };
let mode = "hang";
function session() {
  return {
    prompt(text, opts = {}) {
      log.prompts.push({ text, opts });
      if (mode === "analyse") {
        mode = "hang";
        return Promise.resolve('{"tool": "analyze_station", "arguments": {"station_id": "USGS-01646500"}}');
      }
      return new Promise((_, reject) => {
        opts.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
    },
    async clone() { return session(); },
    destroy() {},
  };
}
globalThis.LanguageModel = { async availability() { return "available"; }, async create() { return session(); } };

const { askLocally } = await import("../src/local-model.js");

// A closed step schema as the package builds it (device_ask.step_schema; its rules are in pytest).
const SCHEMA = {
  type: "object", additionalProperties: false, required: ["tool", "arguments"],
  properties: {
    tool: { type: "string", enum: ["analyze_station"] },
    arguments: { type: "object", additionalProperties: false,
      properties: { station_id: { type: "string", enum: ["USGS-01646500"] }, source: { type: "string", enum: ["usgs"] } } },
  },
};
const pkg = {
  plan: async ({ step }) => ({ mode: step === 1 ? "tool" : "either", budget: 3, prompt: `step ${step}`, schema: SCHEMA }),
  callTool: async () => ({ ok: true, name: "analyze_station", arguments: { source: "usgs", station_id: "USGS-01646500" },
    compact: { station_id: "USGS-01646500" } }),
  render: async (reply, results, words, why) => ({ ok: false, refused: [why], facts: results.length ? ["The mean is 321 m3/s."] : [], notes: [] }),
};

test("each prompt gets the package's closed schema and a signal; a hanging one ends at the time limit", async () => {
  log.prompts.length = 0;
  mode = "analyse";
  const res = await askLocally("Summarise this gauge", { ...pkg, stepMs: 80 });
  assert.equal(log.prompts.length, 2);
  for (const p of log.prompts) {
    assert.deepEqual(p.opts.responseConstraint, SCHEMA);
    assert.ok(p.opts.signal instanceof AbortSignal);
  }
  assert.equal(log.prompts[1].opts.signal.aborted, true);
  assert.match(res.stopped, /no reply within 0 s at step 2/);
  assert.deepEqual(res.lock.facts, ["The mean is 321 m3/s."]);
});

test("Stop aborts the prompt the device is writing", async () => {
  log.prompts.length = 0;
  mode = "hang";
  const stop = new AbortController();
  setTimeout(() => stop.abort(), 30);
  const res = await askLocally("Summarise this gauge", { ...pkg, signal: stop.signal });
  assert.equal(res.stopped, "you stopped it at step 1");
  assert.equal(log.prompts[0].opts.signal.aborted, true);
});
