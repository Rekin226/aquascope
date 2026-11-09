// On-device Ask (#590): a fresh conversation per step, the package's step schema passed to the model, the
// package's sentences (never the model's) on the page, and the sampling options Chrome 155 accepts. The model
// replies are the ones Gemini Nano wrote on 2026-10-26 (tests/fixtures/ondevice_ask_check3.json); the package's
// plan, check and writer are stubs here (their rules, and the replay of every recorded reply, are in pytest:
// tests/test_ai_engine/test_device_ask_pick.py).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  askLoop, createSession, deviceAnswerHtml, deviceAnswerMarkdown, parseReply, samplingNote, samplingOptions, sessionPool,
  inView, selectedGaugeContext, withCatalogueName,
} from "../src/ask-local-core.js";
import { numbersIn, parseBriefReply } from "../src/intake.js";

// An engine whose conversations remember what they were told, and the schema each prompt was held to.
function recordingEngine(replies) {
  const convs = [];
  return {
    convs,
    open() {
      const conv = { prompts: [], schemas: [], closed: false };
      convs.push(conv);
      return {
        async generate(prompt, schema) {
          conv.prompts.push(prompt);
          conv.schemas.push(schema);
          return replies.shift();
        },
        close() { conv.closed = true; },
      };
    },
  };
}

const P = "USGS-01646500";
const SENTENCES = {
  [`${P}.mean`]: "The mean of the daily values at POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA (usgs USGS-01646500) is 321 m3/s.",
  [`${P}.trend.mean`]: "At USGS-01646500 there is no significant trend in the annual mean (Mann-Kendall p = 0.953; Sen's slope -0.0365 m3/s per year).",
};
const SUMMARY_FACTS = Object.values(SENTENCES);

// The package's three steps, as stubs with its rules in small: the first step may only call a tool, the last (or
// after a repeat) may only answer, an answer is the package's sentences for the keys picked, and anything else is
// refused with the facts gathered.
function packageStub({ budget = 3 } = {}) {
  const plans = [];
  return {
    plans,
    plan: async (a) => {
      plans.push({ ...a, results: a.results.length, notes: [...a.notes] });
      const keys = a.results.length ? Object.keys(SENTENCES) : [];
      const mode = !keys.length ? "tool" : a.force || a.step >= budget ? "answer" : "either";
      const props = {};
      if (mode !== "answer") props.tool = { type: "string", enum: ["analyze_station"] };
      if (mode !== "tool") props.answer = { type: "array", items: { type: "string", enum: keys } };
      return { mode, budget, prompt: `step ${a.step}: ${a.words}`, schema: { type: "object", properties: props } };
    },
    callTool: async (reply, results) => {
      const args = reply.arguments || {};
      if (args.station_id !== P) return { ok: false, refused: `the gauge id '${args.station_id}' is in neither your words nor a result` };
      if (results.some((r) => r.arguments.station_id === P)) return { ok: false, repeat: true, refused: "analyze_station already ran with these arguments" };
      return { ok: true, name: "analyze_station", arguments: { source: "usgs", station_id: P }, compact: { station_id: P } };
    },
    render: async (reply, results, words, why) => {
      const facts = results.length ? SUMMARY_FACTS : [];
      if (why) return { ok: false, refused: [why], facts, notes: [] };
      const picks = reply && reply.answer;
      if (typeof picks === "string") return { ok: false, refused: ["the model wrote free text instead of picking facts by key"], facts, notes: [] };
      if (!Array.isArray(picks) || !picks.every((k) => k in SENTENCES)) return { ok: false, refused: ["not among the facts"], facts, notes: [] };
      const sentences = picks.map((k) => SENTENCES[k]);
      return { ok: true, refused: [], sentences, text: sentences.join(" "), facts, notes: [] };
    },
  };
}

const ANALYSE = `{"tool": "analyze_station", "arguments": {"source": "usgs", "station_id": "${P}"}}`;
const SUMMARY = "Summarise the record at this gauge: period, mean, trend, and the flood frequency";

test("each step opens its own conversation, is held to the package's schema, and closes", async () => {
  const eng = recordingEngine([ANALYSE, JSON.stringify({ answer: [`${P}.mean`, `${P}.trend.mean`] })]);
  const pkg = packageStub();
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...pkg });
  assert.equal(eng.convs.length, 2);
  assert.ok(eng.convs.every((c) => c.closed && c.prompts.length === 1), "one prompt per conversation");
  assert.deepEqual(Object.keys(eng.convs[0].schemas[0].properties), ["tool"], "the first step may only call a tool");
  assert.deepEqual(eng.convs[1].schemas[0].properties.answer.items.enum, Object.keys(SENTENCES));
  assert.equal(res.lock.ok, true);
  assert.equal(res.answer, SUMMARY_FACTS.join(" "));
  const html = deviceAnswerHtml({ ...res, model: "Chrome built-in (Gemini Nano)" });
  assert.ok(html.includes("no significant trend in the annual mean (Mann-Kendall p = 0.953"));
  assert.match(html, /The model chose the tools and picked the facts; AquaScope wrote every sentence/);
});

test("check3's loop: a repeated call is not run, the next step may only answer, and the page shows the facts", async () => {
  // Recorded (ask-summary-1, ask-100yr-2, ask-50yr): analyze_station three times with the same arguments.
  const eng = recordingEngine([ANALYSE, ANALYSE, ANALYSE]);
  const pkg = packageStub();
  let ran = 0;
  const callTool = async (reply, results) => { const out = await pkg.callTool(reply, results); if (out.ok) ran += 1; return out; };
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...pkg, callTool });
  assert.equal(ran, 1, "the tool ran once");
  assert.equal(pkg.plans[2].force, true, "after the repeat the step is told to answer");
  assert.deepEqual(Object.keys(eng.convs[2].schemas[0].properties), ["answer"]);
  assert.ok(res.lock, "the lock is never null");
  assert.equal(res.lock.ok, false);
  const html = deviceAnswerHtml({ ...res, model: "m" });
  for (const f of SUMMARY_FACTS) assert.ok(html.includes(`<li>${f.replace(/'/g, "&#39;")}</li>`), f);
  assert.ok(!html.includes("used its steps without reaching an answer"));
});

test("a free-text answer never reaches the page: refused, with the package's facts", async () => {
  // Recorded (ask-anywhere's escape, in the Potomac's words): free text with a significance claim.
  const bad = "significant trend in annual maxima with a p = 0.0689";
  const eng = recordingEngine([ANALYSE, JSON.stringify({ answer: bad })]);
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...packageStub() });
  const html = deviceAnswerHtml({ ...res, model: "m" });
  assert.ok(!html.includes(bad));
  assert.match(html, /The model&#39;s reply was refused: the model wrote free text instead of picking facts by key\. What the tools returned:/);
  assert.match(deviceAnswerMarkdown("Q", res), /refused: the model wrote free text.*\n\n- The mean of the daily values/);
});

test("a call the words do not hold is not run, and the next step is told why", async () => {
  // Recorded (ask-compare): analyze_station with uk_ea / "abc", the system prompt's old example id.
  const eng = recordingEngine([ANALYSE, '{"tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": "abc"}}',
    JSON.stringify({ answer: [`${P}.mean`] })]);
  const pkg = packageStub({ budget: 5 });
  const res = await askLoop("Compare the Potomac with the Thames", { open: () => eng.open(), ...pkg });
  assert.equal(res.results.length, 1);
  assert.match(pkg.plans[2].notes[0], /'abc' is in neither your words nor a result/);
  assert.ok(res.toolCalls.some((c) => c.refused));
  assert.equal(res.lock.ok, true);
});

test("a list of gauges the model picked reads as a list", () => {
  const lock = { ok: true, picked: ["search.count", "E02102A.where", "3400TH.where"], notes: [],
    sentences: ["The catalogue search for \u201cKingston\u201d returned 2 gauges.", "KINGSTON VALE (uk_ea E02102A), at 51.43, -0.26.",
      "HOGSMILL AT EWELL (uk_ea 3400TH), at 51.35, -0.25."] };
  assert.match(deviceAnswerHtml({ lock }), /<ul class="ask-device-facts"><li>The catalogue search/);
  assert.match(deviceAnswerMarkdown("Q", { lock }), /\n- KINGSTON VALE \(uk_ea E02102A\), at 51\.43, -0\.26\.\n- HOGSMILL/);
  assert.doesNotMatch(deviceAnswerHtml({ lock: { ...lock, picked: ["E02102A.where"] } }), /<ul/);
});

test("nothing to look up, the steps used, and a broken writer all end with one line, never a blank", async () => {
  const none = await askLoop("What is a flood?", {
    open: () => { throw new Error("no prompt"); }, plan: async () => ({ mode: "none", budget: 3 }),
    callTool: async () => ({}), render: async (reply, results, words, why) => ({ ok: false, refused: [why], facts: [] }) });
  assert.match(none.stopped, /nothing in the question to look up/);
  assert.match(deviceAnswerHtml(none), /No answer: nothing in the question to look up.*No tool ran for this question/);
  const eng = recordingEngine([JSON.stringify({ answer: [`${P}.mean`] })]);
  const broken = await askLoop(SUMMARY, { open: () => eng.open(), plan: async () => ({ mode: "answer", budget: 1 }),
    callTool: async () => ({}), render: async () => { throw new Error("worker gone"); } });
  assert.equal(broken.lock.ok, false);
  assert.match(broken.lock.refused[0], /the answer could not be written \(worker gone\)/);
});

test("parseReply tolerates prose around the JSON", () => {
  assert.deepEqual(parseReply('Sure: {"answer": "x"} done'), { answer: "x" });
  assert.deepEqual(parseReply("no json"), { answer: "no json" });
});

test("sampling: Chrome 155 on the web has no params(), so the page asks for samplingMode and says what it got", async () => {
  // Chrome 155 on a web page: LanguageModel.params is undefined (recorded on 2026-10-25).
  const web = { create: async (o) => ({ samplingMode: o.samplingMode }) };
  assert.deepEqual(await samplingOptions(web, 0.1), { samplingMode: "most-predictable" });
  assert.equal(samplingNote(await createSession(web, await samplingOptions(web, 0.1))), "sampling most-predictable");
  // Without the origin trial the member is ignored: the note says Chrome's own sampling, not 0.1.
  assert.equal(samplingNote({}), "Chrome's default sampling");
  // An extension still has params(): topK and temperature together.
  const ext = { params: async () => ({ defaultTopK: 3, maxTemperature: 2, defaultTemperature: 1 }) };
  assert.deepEqual(await samplingOptions(ext, 0.1), { temperature: 0.1, topK: 3 });
  // A Chrome that refuses the option still gets a session, without it.
  const seen = [];
  const strict = { create: async (o) => { seen.push(o); if ("samplingMode" in o) throw new TypeError("no"); return {}; } };
  await createSession(strict, { initialPrompts: [], samplingMode: "most-predictable" });
  assert.equal(seen.length, 2);
  assert.ok(!("samplingMode" in seen[1]) && "initialPrompts" in seen[1]);
});

test("Studio brief: a return period or timescale the sentence does not hold is dropped (recorded replies)", () => {
  const winter = "Will the road by the pump station flood this winter?";
  assert.deepEqual(parseBriefReply('{"return_period": 1, "quantities": ["this winter"]}', winter),
    { brief: { quantities: ["this winter"] }, intake: {} });
  const school = "Is this site safe from a big flood for a new school?";
  assert.equal(parseBriefReply('{"return_period": 100}', school), null);   // a guess, even in range
  const bridge = "I need the design flood for a footbridge here, something like the 50-year peak, using data since 1980";
  assert.deepEqual(parseBriefReply('{"return_period": 50, "decision": "Design flood calculation"}', bridge),
    { brief: { decision: "Design flood calculation" }, intake: { return_period: 50 } });
  assert.deepEqual(parseBriefReply('{"return_period": 100}', "what is the hundred-year flood here?").intake, { return_period: 100 });
  assert.deepEqual(parseBriefReply('{"timescales": [3, 12, 6]}', "drought at 3 and twelve months").intake, { timescales: [3, 12] });
  assert.deepEqual(parseBriefReply('{"return_period": 200}').intake, { return_period: 200 }, "no sentence, no check");
  assert.deepEqual([...numbersIn("a 1 in 10,000 year event, two hundred years, twenty-five")].sort((a, b) => a - b),
    [1, 25, 200, 10000]);
});

// ── one prompt per session (recheck, 2026-10-26) ─────────────────────────────
// In Chrome 155 a second prompt in a session failed with "kErrorUnknown" in 11 of 13 trials after a first prompt
// with a responseConstraint; the same text as the only prompt of a fresh clone ran 4 of 4. This engine fails the
// same way: a conversation's second prompt throws.
function chromeLikeEngine(replies, { failFirst = 0 } = {}) {
  const convs = [];
  let failures = failFirst;
  return {
    convs,
    open() {
      const conv = { prompts: [], closed: false };
      convs.push(conv);
      return {
        async generate(prompt) {
          conv.prompts.push(prompt);
          if (conv.prompts.length > 1) throw new Error("kErrorUnknown");
          if (failures > 0) { failures -= 1; throw new Error("kErrorUnknown"); }
          return replies.shift();
        },
        close() { conv.closed = true; },
      };
    },
  };
}

test("a step that fails is tried once more in a fresh session", async () => {
  const eng = chromeLikeEngine([ANALYSE, JSON.stringify({ answer: [`${P}.mean`] })], { failFirst: 1 });
  const events = [];
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...packageStub(), onEvent: (m) => events.push(m) });
  assert.equal(res.answer, SENTENCES[`${P}.mean`]);
  assert.equal(eng.convs.length, 3);
  assert.ok(eng.convs.every((c) => c.closed));
  assert.ok(events.some((e) => /Step 1 failed \(kErrorUnknown\); trying it once more/.test(e)));
});

test("a step that fails twice after a tool ran shows the tools' facts and the reason, not an error", async () => {
  const eng = chromeLikeEngine([ANALYSE]);
  let calls = 0;
  const open = () => {
    calls += 1;
    if (calls === 1) return eng.open();
    return { async generate() { throw new Error("kErrorUnknown"); }, close() {} };
  };
  const res = await askLoop(SUMMARY, { open, ...packageStub() });
  assert.equal(calls, 3);
  assert.equal(res.lock.ok, false);
  assert.match(res.stopped, /stopped at step 2, twice \(kErrorUnknown\)/);
  const html = deviceAnswerHtml({ ...res, model: "Chrome built-in (Gemini Nano)" });
  assert.match(html, /No answer: the on-device model stopped at step 2, twice \(kErrorUnknown\)\. What the tools returned:/);
  assert.match(html, /<li>The mean of the daily values at POTOMAC/);
  assert.match(deviceAnswerMarkdown(SUMMARY, res), /No answer: the on-device model stopped at step 2/);
});

test("a first step that fails twice is an error the page reports, as before", async () => {
  const open = () => ({ async generate() { throw new Error("kErrorUnknown"); }, close() {} });
  await assert.rejects(askLoop(SUMMARY, { open, ...packageStub() }), /kErrorUnknown/);
});

// ── warm sessions for the one-off calls (recheck item 5) ─────────────────────
// A stub LanguageModel that counts creates, clones and destroys. This checks only that the pool is correct;
// whether a clone of a warm base is quicker than a create needs Chrome's model on a device.
function stubLM({ cloneFails = false } = {}) {
  const log = { create: [], clone: 0, destroyed: [] };
  let n = 0;
  const make = (system, kind) => {
    const id = `${kind}${++n}`;
    return {
      id, system,
      async clone() { if (cloneFails) throw new Error("clone failed"); log.clone += 1; return make(system, "clone"); },
      async prompt() { return "{}"; },
      destroy() { log.destroyed.push(id); },
    };
  };
  return { log, async create(options) { const sys = options.initialPrompts[0].content; log.create.push(sys); return make(sys, "base"); } };
}

test("one warm base per system prompt: later calls clone it rather than create a session", async () => {
  const LM = stubLM();
  const pool = sessionPool(LM);
  const a = await pool.open("map system", { samplingMode: "most-predictable" });
  const b = await pool.open("map system", { samplingMode: "most-predictable" });
  assert.equal(LM.log.create.length, 1, "the base is created once");
  assert.equal(LM.log.clone, 2);
  assert.ok(a.id.startsWith("clone") && b.id.startsWith("clone") && a !== b, "each call gets its own clone");
  await pool.open("explain system");
  assert.equal(LM.log.create.length, 2, "another system prompt gets its own base");
  assert.equal(pool.size(), 2);
  a.destroy();
  assert.deepEqual(LM.log.destroyed, [a.id], "destroying a clone leaves the base warm");
});

test("the pool keeps at most a few bases, destroying the least recently used", async () => {
  const LM = stubLM();
  const pool = sessionPool(LM, { max: 2 });
  await pool.open("one");
  await pool.open("two");
  await pool.open("one");      // touch "one"
  await pool.open("three");    // evicts "two"
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(pool.size(), 2);
  assert.equal(LM.log.destroyed.length, 1);
  await pool.open("one");
  assert.equal(LM.log.create.filter((s) => s === "one").length, 1, "the used base stayed warm");
  await pool.open("two");
  assert.equal(LM.log.create.filter((s) => s === "two").length, 2, "the evicted base is created again");
});

test("a clone that fails falls back to a fresh session", async () => {
  const LM = stubLM({ cloneFails: true });
  const pool = sessionPool(LM);
  const s = await pool.open("sys");
  assert.ok(s.id.startsWith("base"));
  assert.equal(LM.log.create.length, 2, "the base, then a fresh session for this call");
});

// ── the context line (recheck item 9) ─────────────────────────────────────────
const POTOMAC_ROW = { source: "usgs", station_id: "01646500", name: "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
  lat: 38.9497, lon: -77.1275 };
const EUROPE = { west: -12, south: 35, east: 32, north: 62 };
const VIRGINIA = { west: -78, south: 38.5, east: -76.5, north: 39.3 };

test("a selected gauge off screen is left out for the map's roles and said as off screen for Ask", () => {
  assert.match(selectedGaugeContext(POTOMAC_ROW, VIRGINIA), /^the gauge POTOMAC RIVER .* id 01646500, 38\.950, -77\.127\)$/);
  // Recorded: Scout in Europe ordered USGS-01646500, from the context line, which named it though it was off screen.
  assert.equal(selectedGaugeContext(POTOMAC_ROW, EUROPE), "");
  assert.match(selectedGaugeContext(POTOMAC_ROW, EUROPE, { offScreen: "say" }), /, selected earlier and now off screen$/);
  assert.match(selectedGaugeContext(POTOMAC_ROW, null), /^the gauge POTOMAC/, "no map yet: said as before");
});

test("inView wraps across the antimeridian and takes a whole-world view", () => {
  assert.ok(inView({ west: 170, south: -50, east: 190, north: -30 }, -41, -175));
  assert.ok(!inView({ west: 170, south: -50, east: 190, north: -30 }, -41, 160));
  assert.ok(inView({ west: -540, south: -85, east: 540, north: 85 }, 10, 100));
  assert.ok(!inView(EUROPE, 70, 10));
});

test("an analysis carries the catalogue's name and river, so two gauges can be told apart", () => {
  const byKey = new Map([["usgs/01646500", { ...POTOMAC_ROW, river: "Potomac River" }]]);
  const out = withCatalogueName({ ok: true, compact: { source: "usgs", station_id: "01646500", unit: "m3/s" } },
    { source: "usgs", station_id: "01646500" }, byKey);
  assert.equal(out.compact.name, POTOMAC_ROW.name);
  assert.equal(out.compact.river, "Potomac River");
  const err = { ok: false, compact: { error: "no data" } };
  assert.equal(withCatalogueName(err, {}, byKey), err);
  const named = { ok: true, compact: { source: "usgs", station_id: "01646500", name: "Little Falls" } };
  assert.equal(withCatalogueName(named, {}, byKey).compact.name, "Little Falls");
});

// ── the fourth check (2026-10-26): a time limit, a Stop, and the right reason ─────────────────────────────────

// A conversation whose prompt never replies until its signal aborts, as the 176 s prompt of the fourth check
// ("List the gauges on the Thames near Kingston", QuotaExceededError at the end).
function hangingEngine(before = []) {
  const signals = [];
  return {
    signals,
    open() {
      return {
        generate(prompt, schema, signal) {
          if (before.length) return Promise.resolve(before.shift());
          signals.push(signal);
          return new Promise((_, reject) => {
            if (signal) signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
          });
        },
        close() {},
      };
    },
  };
}

test("a prompt that hangs is ended by the step's time limit, on the facts gathered so far", async () => {
  const eng = hangingEngine([ANALYSE]);
  const t0 = Date.now();
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...packageStub(), stepMs: 60 });
  assert.ok(Date.now() - t0 < 2000);
  assert.equal(eng.signals.length, 1, "the hanging step got a signal");
  assert.equal(eng.signals[0].aborted, true, "and it was aborted, so the device stops writing");
  assert.match(res.stopped, /gave no reply within 0 s at step 2, so it was stopped/);
  const html = deviceAnswerHtml({ ...res, model: "m" });
  assert.match(html, /No answer: the on-device model gave no reply/);
  for (const f of SUMMARY_FACTS) assert.ok(html.includes(`<li>${f.replace(/'/g, "&#39;")}</li>`), f);
  assert.doesNotMatch(html, /Answered on your device/);
});

test("Stop ends a hanging prompt at once, with the facts gathered and one line why", async () => {
  const eng = hangingEngine([ANALYSE]);
  const stop = new AbortController();
  const events = [];
  const onEvent = (m) => { events.push(m); if (m.startsWith("Thinking… (step 2")) setTimeout(() => stop.abort(), 20); };
  const res = await askLoop(SUMMARY, { open: () => eng.open(), ...packageStub(), onEvent, signal: stop.signal, stepMs: 60000 });
  assert.equal(res.stopped, "you stopped it at step 2");
  assert.equal(eng.signals[0].aborted, true);
  assert.equal(res.results.length, 1);
  assert.match(deviceAnswerHtml(res), /No answer: you stopped it at step 2\. What the tools returned:/);
  // Stopped before the first step: nothing is prompted.
  const early = await askLoop(SUMMARY, { open: () => eng.open(), ...packageStub(), signal: AbortSignal.abort() });
  assert.equal(early.stopped, "you stopped it at step 1");
});

test("calls that were all refused say so, not 'nothing in the question to look up' (check4, Kingston)", async () => {
  const replies = [
    '{"tool": "find_stations", "arguments": {"query": "Kingston upon Thames", "near": [51.0, -0.3], "limit__": 5}}',
    '{"tool": "find_stations", "arguments": {"query": "Kingston upon Thames", "limit": 5, "near": [51.47, 0.50]}}',
  ];
  const eng = recordingEngine([...replies]);
  let step = 0;
  const res = await askLoop("Find the five gauges nearest to Kingston upon Thames", {
    open: () => eng.open(),
    plan: async () => ({ mode: ++step <= 2 ? "tool" : "none", budget: 5, prompt: "p", schema: {} }),
    callTool: async () => ({ ok: false, refused: "the search point is in neither your words nor a result" }),
    render: async (reply, results, words, why) => ({ ok: false, refused: [why], facts: [], notes: [] }),
  });
  assert.equal(res.stopped, "all 2 of the model's calls were refused (find_stations: the search point is in neither your words nor a result)");
  assert.doesNotMatch(deviceAnswerHtml(res), /nothing in the question|Answered on your device/);
});

test("the 'Answered on your device' header is over answers only", () => {
  const ok = { lock: { ok: true, sentences: ["The mean is 321 m3/s."], text: "", notes: [] }, model: "m" };
  assert.match(deviceAnswerHtml(ok), /^<div class="showcase-note">Answered on your device by m/);
  const refused = { lock: { ok: false, refused: ["AquaScope's own check stopped the answer"], facts: ["x"], notes: [] }, model: "m" };
  const html = deviceAnswerHtml(refused);
  assert.doesNotMatch(html, /Answered/);
  assert.match(html, /^<p class="ask-refused">The model&#39;s reply was refused/);
  assert.match(html, /Ran on your device by m, with no key/);
});
