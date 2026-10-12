// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  FREE_MODELS, allowanceLine, freeModelConfig, freeTierNote, liveSession, proxyMessage,
} from "../src/free-tier-core.js";

const NOW = Date.parse("2026-10-11T10:00:00Z");
const SESSION = { token: "aqf_v.1.sig", expires: NOW + 3600_000 };

test("a session is live until a minute before it expires, and only an aqf_ token counts", () => {
  assert.deepEqual(liveSession(JSON.stringify(SESSION), NOW), SESSION);
  assert.equal(liveSession({ ...SESSION, expires: NOW + 30_000 }, NOW), null);
  assert.equal(liveSession({ token: "sk-ant-api03-x", expires: NOW + 3600_000 }, NOW), null);
  assert.equal(liveSession("not json", NOW), null);
  assert.equal(liveSession(null, NOW), null);
});

test("questions go to Haiku 5.5, studies to Sonnet 5.5, through the proxy with the session as the key", () => {
  const ask = freeModelConfig("ask", SESSION, "https://free.example.workers.dev/");
  assert.equal(ask.model, FREE_MODELS.ask);
  assert.equal(ask.model, "claude-haiku-5-5");
  assert.equal(ask.provider, "anthropic");
  assert.equal(ask.base_url, "https://free.example.workers.dev");
  assert.equal(ask.api_key, SESSION.token);
  assert.equal(ask.free, true);
  assert.match(ask.label, /Claude Haiku 5\.5 \(free, sponsored\)/);
  assert.equal(freeModelConfig("study", SESSION, "https://x.dev").model, "claude-sonnet-5-5");
  assert.equal(freeModelConfig("map", SESSION, "https://x.dev").model, "claude-haiku-5-5");
  assert.equal(freeModelConfig("ask", null, "https://x.dev"), null);
  assert.equal(freeModelConfig("ask", SESSION, ""), null);
});

test("the tier's line says what a visitor gets, until when, and who pays; nothing when closed", () => {
  const line = freeTierNote({
    state: "open", end_date: "2027-02-28", sponsor: "Sponsored by Anthropic's AI for Science program",
    per_visitor_per_day: { asks: 20, studies: 2 },
  });
  assert.match(line, /20 questions and 2 studies a day/);
  assert.match(line, /Until 2027-02-28/);
  assert.match(line, /AI for Science/);
  assert.equal(freeTierNote({ state: "ended" }), "");
  assert.equal(freeTierNote(null), "");
});

test("allowance and proxy errors read as sentences", () => {
  assert.equal(allowanceLine({ asks: 1, studies: 2 }), "Left today: 1 question, 2 studies.");
  assert.equal(allowanceLine({ asks: 0, studies: 1 }), "Left today: 0 questions, 1 study.");
  assert.equal(allowanceLine(null), "");
  assert.equal(proxyMessage({ error: { message: "You have used today's free studies." } }, 402),
    "You have used today's free studies.");
  assert.match(proxyMessage({}, 401), /expired/);
  assert.match(proxyMessage(null, 500), /HTTP 500/);
});
