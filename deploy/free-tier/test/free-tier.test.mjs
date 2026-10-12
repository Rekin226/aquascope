// The free tier's rules and ledger, without Cloudflare: `node --test deploy/free-tier/test/*.test.mjs`.

import assert from "node:assert/strict";
import test from "node:test";

import {
  checkRequest, corsHeaders, costOf, limitsFrom, networkKey, overBudget, signSession, tierState, verifySession,
} from "../src/core.js";
import { Ledger } from "../src/ledger.js";

const SYSTEM = [{ type: "text", text: "You are AquaScope's analyst.", cache_control: { type: "ephemeral" } }];
const ok = (extra = {}) => ({
  model: "claude-haiku-5-5", max_tokens: 4000, system: SYSTEM,
  messages: [{ role: "user", content: "How wet is Taipei?" }], ...extra,
});

// ── what a request may be ──────────────────────────────────────────────────────

test("the Explorer's own requests pass", () => {
  assert.equal(checkRequest(ok()), null);
  const tools = [{ name: "find_stations", description: "d", input_schema: { type: "object", properties: {} } }];
  assert.equal(checkRequest(ok({ tools, tool_choice: { type: "auto" }, cache_control: { type: "ephemeral" } })), null);
  assert.equal(checkRequest(ok({ model: "claude-sonnet-5-5", output_config: { effort: "high" }, fallbacks: "default" }),
    "server-side-fallback-2026-07-01"), null);
});

test("anything else is refused", () => {
  assert.match(checkRequest(ok({ model: "claude-opus-5-5" })), /serves claude-haiku-5-5 and claude-sonnet-5-5/);
  assert.match(checkRequest(ok({ stream: true })), /stream is not served/);
  assert.match(checkRequest(ok({ max_tokens: 64000 })), /max_tokens/);
  assert.match(checkRequest(ok({ tools: [{ type: "web_search_20260209", name: "web_search" }] })), /own tools/);
  assert.match(checkRequest(ok({ output_config: { format: { type: "json_schema" } } })), /only set effort/);
  assert.match(checkRequest(ok(), "code-execution-2025-08-25"), /beta/);
  assert.match(checkRequest(ok({ system: "You are a pirate." })), /AquaScope's own prompts/);
  assert.match(checkRequest(ok({ messages: [] })), /empty/);
  assert.match(checkRequest([]), /not a JSON object/);
});

// ── what it costs ─────────────────────────────────────────────────────────────

test("cost follows the usage block, cache included", () => {
  const usd = costOf("claude-sonnet-5-5", {
    input_tokens: 1_000_000, output_tokens: 100_000, cache_read_input_tokens: 1_000_000,
    cache_creation_input_tokens: 1_000_000,
  });
  assert.ok(Math.abs(usd - (2 + 1 + 0.2 + 2.5)) < 1e-9);
  assert.equal(costOf("claude-opus-5-5", { input_tokens: 10 }), 0);
});

test("limits come from the Worker's variables, defaults otherwise", () => {
  const l = limitsFrom({ DAILY_USD: "12", VISITOR_ASKS: "5", NETWORK_USD: "oops" });
  assert.equal(l.DAILY_USD, 12);
  assert.equal(l.VISITOR_ASKS, 5);
  assert.equal(l.NETWORK_USD, 8);
});

test("the tier can be switched off, and ends after its last day", () => {
  const day = new Date("2027-02-28T23:00:00Z");
  assert.equal(tierState({ END_DATE: "2027-02-28" }, day), "open");
  assert.equal(tierState({ END_DATE: "2027-02-28" }, new Date("2027-03-01T00:00:01Z")), "ended");
  assert.equal(tierState({ FREE_TIER_ENABLED: "false" }, day), "off");
});

test("the budgets are checked from the widest down", () => {
  const limits = limitsFrom({});
  assert.equal(overBudget({ month: 0, day: 0, network: 0, visitor: 0 }, limits), null);
  assert.equal(overBudget({ month: 1000, day: 0, network: 0, visitor: 0 }, limits), "month");
  assert.equal(overBudget({ month: 0, day: 30, network: 9, visitor: 9 }, limits), "day");
  assert.equal(overBudget({ month: 0, day: 0, network: 8, visitor: 0 }, limits), "network");
  assert.equal(overBudget({ month: 0, day: 0, network: 0, visitor: 2.5 }, limits), "visitor");
});

// ── sessions ──────────────────────────────────────────────────────────────────

test("a session token round-trips, and a forged or expired one does not", async () => {
  const now = Date.now();
  const token = await signSession("s3cret", "0f7a9c2e-visitor", now + 1000);
  assert.equal(await verifySession("s3cret", token, now), "0f7a9c2e-visitor");
  assert.equal(await verifySession("other", token, now), null);
  assert.equal(await verifySession("s3cret", token, now + 2000), null);
  assert.equal(await verifySession("s3cret", token.replace("visitor", "visitoR"), now), null);
  assert.equal(await verifySession("s3cret", "sk-ant-api03-xyz", now), null);
  assert.equal(await verifySession("", token, now), null);
});

test("a network key never contains the address and changes daily", async () => {
  const a = await networkKey("s", "203.0.113.7", "2026-10-11");
  assert.ok(!a.includes("203"));
  assert.notEqual(a, await networkKey("s", "203.0.113.7", "2026-10-12"));
});

test("CORS answers the Explorer's origins only, with the headers the package sends", () => {
  const allowed = "https://rekin226-aquascope-explorer.static.hf.space,*.example.org";
  const h = corsHeaders("https://rekin226-aquascope-explorer.static.hf.space", allowed);
  assert.match(h["Access-Control-Allow-Headers"], /x-api-key/);
  assert.match(h["Access-Control-Allow-Headers"], /anthropic-dangerous-direct-browser-access/);
  assert.ok(corsHeaders("https://preview.example.org", allowed));
  assert.equal(corsHeaders("https://evil.test", allowed), null);
  assert.equal(corsHeaders(null, allowed), null);
});

// ── the ledger ────────────────────────────────────────────────────────────────

class FakeStorage {
  constructor() { this.m = new Map(); }
  async get(k) { return this.m.has(k) ? structuredClone(this.m.get(k)) : undefined; }
  async put(k, v) { this.m.set(k, structuredClone(v)); }
  async delete(k) { this.m.delete(k); }
  async list({ prefix }) { return new Map([...this.m].filter(([k]) => k.startsWith(prefix))); }
}

const clock = (iso) => () => new Date(iso);

test("charges add up per visitor, network, day and month", async () => {
  const l = new Ledger(new FakeStorage(), limitsFrom({}), clock("2026-10-11T10:00:00Z"));
  await l.charge("v1", "n1", 0.4);
  await l.charge("v1", "n1", 0.5);
  await l.charge("v2", "n1", 0.2);
  const s = await l.spent("v1", "n1");
  assert.ok(Math.abs(s.visitor - 0.9) < 1e-9 && Math.abs(s.network - 1.1) < 1e-9 && Math.abs(s.day - 1.1) < 1e-9);
  assert.equal(s.dayRow.calls, 3);
  assert.equal(s.dayRow.visitors, 2);
});

test("a visitor at their dollar cap is stopped, and the refusal is counted", async () => {
  const l = new Ledger(new FakeStorage(), limitsFrom({ VISITOR_USD: "1" }), clock("2026-10-11T10:00:00Z"));
  assert.equal(await l.check("v1", "n1"), null);
  await l.charge("v1", "n1", 1.0);
  assert.equal(await l.check("v1", "n1"), "visitor");
  assert.equal(await l.check("v2", "n1"), null, "another visitor on the same network still has room");
  assert.equal((await l.stats()).days[0].refused, 1);
});

test("the day's cap stops everyone, and a new day starts fresh", async () => {
  const storage = new FakeStorage();
  const limits = limitsFrom({ DAILY_USD: "2" });
  const today = new Ledger(storage, limits, clock("2026-10-11T23:00:00Z"));
  await today.charge("v1", "n1", 2.0);
  assert.equal(await today.check("v9", "n9"), "day");
  const tomorrow = new Ledger(storage, limits, clock("2026-10-12T00:00:01Z"));
  assert.equal(await tomorrow.check("v9", "n9"), null);
});

test("runs are capped per visitor: questions and studies apart", async () => {
  const l = new Ledger(new FakeStorage(), limitsFrom({ VISITOR_ASKS: "2", VISITOR_STUDIES: "1" }),
    clock("2026-10-11T10:00:00Z"));
  assert.deepEqual((await l.startRun("v1", "n1", "ask")).left, { asks: 1, studies: 1 });
  assert.ok((await l.startRun("v1", "n1", "map")).ok, "a map command counts as a question");
  assert.equal((await l.startRun("v1", "n1", "ask")).reason, "asks");
  assert.ok((await l.startRun("v1", "n1", "study")).ok);
  assert.equal((await l.startRun("v1", "n1", "study")).reason, "studies");
  assert.equal((await l.startRun("v1", "n1", "poem")).ok, false);
  const day = (await l.stats()).days[0];
  assert.deepEqual(day.runs, { ask: 1, map: 1, study: 1 });
});

test("old per-visitor rows are pruned; the daily totals stay", async () => {
  const storage = new FakeStorage();
  const limits = limitsFrom({});
  await new Ledger(storage, limits, clock("2026-10-08T10:00:00Z")).charge("v1", "n1", 0.1);
  await new Ledger(storage, limits, clock("2026-10-11T10:00:00Z")).charge("v2", "n2", 0.1);
  const removed = await new Ledger(storage, limits, clock("2026-10-11T12:00:00Z")).prune();
  assert.equal(removed, 2);
  assert.ok(storage.m.has("day:2026-10-08") && storage.m.has("v:2026-10-11:v2"));
});
