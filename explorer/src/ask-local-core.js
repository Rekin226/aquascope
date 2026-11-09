// On-device Ask (#590), the parts with no browser in them, so node can run them with the model's recorded replies:
// the loop over one fresh conversation per step, the sampling options Chrome accepts, and the answer as the page
// shows it.
//
// Round 3, "the model picks, the code writes": three on-device checks held the model's free text to the results
// with patterns, and each round Gemini Nano found a new way past them. Now the model writes nothing the reader
// sees. Each step's reply is a tool call or a list of fact keys, held by a responseConstraint schema whose values
// are enums the package builds from this question (aquascope.ai_engine.device_ask.step_schema): the tools still
// open, the gauge ids and places the words and results hold, the keys of the facts the results allow. The
// package checks the call (check_call) and writes the answer from the keys (render_answer). Each step opens its
// own conversation and closes it (a second prompt in a session failed with "kErrorUnknown" in the recheck).

/** Is a point inside a view {west, south, east, north} (degrees)? A view across the antimeridian wraps. */
export function inView(view, lat, lon) {
  if (!view || !Number.isFinite(lat) || !Number.isFinite(lon)) return false;
  const { west, south, east, north } = view;
  if (!(lat >= south && lat <= north)) return false;
  if (east - west >= 360) return true;
  const wrap = (x) => ((((x + 180) % 360) + 360) % 360) - 180;
  const w = wrap(west), e = wrap(east), x = wrap(lon);
  return w <= e ? x >= w && x <= e : x >= w || x <= e;
}

/**
 * The selected gauge as the context line says it. In view, as it was. Off screen (the recheck found Scout in
 * Europe ordering the Potomac gauge, selected earlier, because the line named it): `offScreen: "say"` keeps it
 * and says so (Ask, where "this gauge" still means it), `"omit"` leaves it out (the map's roles, which are
 * about the view).
 */
export function selectedGaugeContext(r, view, { offScreen = "omit" } = {}) {
  if (!r) return "";
  const what = `the gauge ${r.name || r.station_id} (source ${r.source}, id ${r.station_id}, ` +
    `${Number(r.lat).toFixed(3)}, ${Number(r.lon).toFixed(3)})`;
  if (!view || inView(view, r.lat, r.lon)) return what;
  return offScreen === "say" ? `${what}, selected earlier and now off screen` : "";
}

/** The model's reply as an object: a tool call, or {answer}. Prose around the JSON is tolerated. */
export function parseReply(text) {
  const raw = String(text || "").trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return { answer: raw };
  try {
    return JSON.parse(raw.slice(start, end + 1));
  } catch {
    return { answer: raw };
  }
}

/**
 * The sampling options `LanguageModel.create()` takes, feature-detected (checked against
 * developer.chrome.com/docs/ai/prompt-api on 2026-10-25).
 *
 * In a Chrome extension, `LanguageModel.params()` gives the default topK and the temperature range, and a
 * session takes `topK` and `temperature`, both or neither. On the web there is no `params()` (it is
 * undefined in Chrome 155, which is why the 0.1 asked for was never set) and no raw numbers: the sampling
 * origin trial adds `samplingMode`, one of seven words, "most-predictable" through "most-creative". Without the trial
 * the member is ignored and Chrome keeps its own sampling; `samplingNote()` says which happened.
 */
export async function samplingOptions(LM, temperature) {
  if (LM && typeof LM.params === "function") {
    try {
      const p = await LM.params();
      if (p && Number.isFinite(p.defaultTopK)) {
        return { temperature: Math.min(temperature, Number.isFinite(p.maxTemperature) ? p.maxTemperature : temperature),
          topK: p.defaultTopK };
      }
    } catch { /* fall through to the web's option */ }
    return {};
  }
  const mode = temperature <= 0.2 ? "most-predictable" : temperature <= 0.5 ? "predictable" : "balanced";
  return { samplingMode: mode };
}

/** `LanguageModel.create(options)`, retried without the sampling option if this Chrome refuses it. */
export async function createSession(LM, options) {
  try {
    return await LM.create(options);
  } catch (err) {
    if (!("samplingMode" in options) && !("temperature" in options)) throw err;
    const rest = { ...options };
    for (const k of ["samplingMode", "temperature", "topK"]) delete rest[k];
    return LM.create(rest);
  }
}

/**
 * Warm sessions for one-off structured calls (Ask the map, Explain, Scout, What-if, the review): one base
 * session per system prompt, created once and never prompted, and a clone of it for each call, which the
 * caller destroys. `LanguageModel.create` took 20.6 to 21.1 s on 7 of 21 calls in the recheck (2026-10-26);
 * a call that clones a warm base does not create one. Whether a clone is quicker than a create on the device
 * is not measured here (Ask's clones also stalled now and then): that needs Chrome's model. At most `max`
 * bases are kept, the least recently used destroyed first. A base or a clone that fails falls back to a
 * fresh session, as before.
 */
export function sessionPool(LM, { max = 4 } = {}) {
  const bases = new Map();   // key -> Promise<session>
  const drop = (p) => { p.then((b) => { try { b.destroy(); } catch { /* fine */ } }, () => {}); };
  function base(key, options) {
    if (bases.has(key)) {
      const p = bases.get(key);
      bases.delete(key);
      bases.set(key, p);     // most recently used last
      return p;
    }
    const p = createSession(LM, options);
    bases.set(key, p);
    p.catch(() => { if (bases.get(key) === p) bases.delete(key); });
    while (bases.size > max) {
      const [oldest, old] = bases.entries().next().value;
      bases.delete(oldest);
      drop(old);
    }
    return p;
  }
  return {
    async open(system, sampling = {}) {
      const options = { initialPrompts: [{ role: "system", content: system }], ...sampling };
      const key = JSON.stringify([system, sampling]);
      let session = null;
      try {
        const b = await base(key, options);
        session = typeof b.clone === "function" ? await b.clone() : null;
      } catch {
        session = null;
      }
      return session || createSession(LM, options);
    },
    size: () => bases.size,
    clear() { for (const p of bases.values()) drop(p); bases.clear(); },
  };
}

/** What sampling a session actually runs with, in words: what was set, never what was only asked for. */
export function samplingNote(session) {
  if (session && typeof session.samplingMode === "string" && session.samplingMode) return `sampling ${session.samplingMode}`;
  if (session && Number.isFinite(session.temperature)) return `temperature ${session.temperature}`;
  return "Chrome's default sampling";
}

/**
 * A tool's `{ok, compact}` with the catalogue row's name and river added to an analysis (analyze_station
 * carries the id and no name). `byKey` is the page's catalogue map, keyed "source/station_id".
 */
export function withCatalogueName(out, args, byKey) {
  const c = out && out.compact;
  if (!c || typeof c !== "object" || c.error || !byKey || typeof byKey.get !== "function") return out;
  const source = c.source || (args && args.source);
  const id = c.station_id || (args && args.station_id);
  const row = source && id ? byKey.get(`${source}/${id}`) : null;
  if (!row) return out;
  const extra = {};
  if (!c.name && row.name) extra.name = row.name;
  if (!c.river && row.river) extra.river = row.river;
  return Object.keys(extra).length ? { ...out, compact: { ...c, ...extra } } : out;
}

/** How long one step's prompt may run on the device, as for the map's one-off calls (generateJsonLocally). The
 * fourth check's prompts took 0.6 to 5.3 s; one wrote a number for 176 s until Chrome's output limit. */
export const STEP_MS = 25000;

/** A step that ended before the model replied: `kind` "timeout" (the time limit) or "stop" (the reader). */
function cut(kind, ms) {
  const err = new Error(kind === "stop" ? "stopped" : `no reply in ${Math.round(ms / 1000)} s`);
  err.kind = kind;
  return err;
}

/**
 * One question on the device. The package does every check and writes every sentence; the model only picks.
 *
 * - `open()` gives a fresh conversation `{generate(prompt, schema, signal), close()}`;
 * - `plan({results, words, step, budget, force, notes})` is the package's step (device_ask.step_schema):
 *   `{mode, schema, prompt, budget}`, mode "tool" (the first step), "either", "answer" (the last step, or after a
 *   repeated call) or "none" (nothing in the words to look up);
 * - `callTool(reply, results, words)` checks the call (device_ask.check_call) and runs it in the worker:
 *   `{ok, name, arguments, compact}`, or `{refused}` (not run), or `{repeat: true}` (already run: not run again);
 * - `render(reply, results, words, why)` writes the answer from the keys (device_ask.render_answer):
 *   `{ok, text, sentences, refused, facts, notes}`.
 * - `signal` is the Stop button's; `stepMs` the time limit of one step's prompt. Either one ends the question
 *   on the facts gathered so far, with one line saying why (the fourth check: a prompt ran 176 s, with no Stop).
 *
 * Returns `{answer, toolCalls, results, lock, canned, stopped}`; `lock` is render's verdict, never null, so the
 * page always shows either the package's sentences or the facts gathered so far with one line why.
 */
export async function askLoop(question, { open, plan, callTool, render, onEvent = () => {}, context = "", maxSteps = 8,
                                          signal = null, stepMs = STEP_MS } = {}) {
  const words = context ? `${question}\n\nContext: ${context}` : question;
  const toolCalls = [];
  const results = [];
  const notes = [];
  let force = false;
  let budget = null;
  // One prompt in its own conversation, ended by the time limit or the Stop button, whichever comes first. The
  // abort reaches the session (Chrome's prompt takes a signal); the race ends the step even if it does not.
  const once = async (prompt, schema) => {
    if (signal && signal.aborted) throw cut("stop");
    const ctl = new AbortController();
    let conv = null;
    let done = false;
    let timer = null;
    let onStop = null;
    const ended = new Promise((_, reject) => {
      timer = setTimeout(() => { reject(cut("timeout", stepMs)); ctl.abort(); }, stepMs);
      onStop = () => { reject(cut("stop")); ctl.abort(); };
      if (signal) signal.addEventListener("abort", onStop, { once: true });
    });
    const run = (async () => {
      const c = await open();
      if (done) { try { c.close(); } catch { /* fine */ } throw cut("stop"); }
      conv = c;
      return c.generate(prompt, schema, ctl.signal);
    })();
    run.catch(() => {});
    ended.catch(() => {});
    try {
      return await Promise.race([run, ended]);
    } finally {
      done = true;
      clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onStop);
      try { if (conv) conv.close(); } catch { /* fine */ }
    }
  };
  const finish = async (reply, why = null, stopped = null) => {
    let verdict;
    try {
      verdict = await render(reply, results, words, why);
    } catch (err) {
      verdict = { ok: false, refused: [`the answer could not be written (${err.message})`], facts: [], notes: [] };
    }
    verdict = verdict || { ok: false, refused: ["no answer"], facts: [], notes: [] };
    return { answer: verdict.ok ? verdict.text : "", toolCalls, results, lock: verdict, canned: !verdict.ok,
      stopped: stopped || why };
  };
  // Why a question ends with no tool run: the calls the model made were refused (check4: four searches, each at a
  // made-up point, then "nothing in the question to look up"), or there was nothing to call.
  const noneWhy = () => {
    const refused = toolCalls.filter((c) => c.refused);
    if (refused.length) {
      const n = refused.length;
      return `${n === 1 ? "the model's one call was" : `all ${n} of the model's calls were`} refused ` +
        `(${refused[n - 1].name}: ${refused[n - 1].refused})`;
    }
    return results.length ? "the tools have nothing more to give for this question"
      : "nothing in the question to look up: name the gauge (its id) or the place (lat, lon)";
  };
  const cutWhy = (err, step) => (err.kind === "stop"
    ? `you stopped it at step ${step}`
    : `the on-device model gave no reply within ${Math.round(stepMs / 1000)} s at step ${step}, so it was stopped`);
  for (let step = 1; step <= maxSteps; step++) {
    if (signal && signal.aborted) return finish(null, cutWhy(cut("stop"), step));
    const p = await plan({ results, words, step, budget, force, notes });
    budget = p.budget || budget || maxSteps;
    if (p.mode === "none") return finish(null, noneWhy());
    // A small model takes a few seconds a step; without a running commentary that reads as a hang.
    onEvent(`Thinking… (step ${step} of ${budget})`);
    let text;
    try {
      text = await once(p.prompt, p.schema);
    } catch (first) {
      if (first.kind) return finish(null, cutWhy(first, step));
      onEvent(`Step ${step} failed (${first.message}); trying it once more in a fresh session…`);
      try {
        text = await once(p.prompt, p.schema);
      } catch (err) {
        if (err.kind) return finish(null, cutWhy(err, step));
        if (!results.length) throw err;
        return finish(null, `the on-device model stopped at step ${step}, twice (${err.message})`);
      }
    }
    if (signal && signal.aborted) return finish(null, cutWhy(cut("stop"), step));
    const reply = parseReply(text);
    const isTool = reply && typeof reply.tool === "string" && reply.tool;
    if (p.mode === "answer" || !isTool) return finish(reply);
    onEvent(`tool ${reply.tool}(${JSON.stringify(reply.arguments || {}).slice(0, 120)})`);
    let out;
    try {
      out = await callTool(reply, results, words);
    } catch (err) {
      out = { ok: false, name: reply.tool, arguments: reply.arguments || {}, compact: { error: err.message } };
    }
    out = out || { refused: "the tool did not answer" };
    if (out.repeat) {
      // The same call again (#590: analyze_station three times, then no answer): not run; the next step answers.
      force = true;
      notes.push(out.refused || `${reply.tool} already ran with these arguments`);
      toolCalls.push({ name: reply.tool, arguments: out.arguments || reply.arguments || {}, ok: false, repeat: true });
      onEvent(`${reply.tool} already ran with these arguments: answering from its result`);
      continue;
    }
    if (out.refused && !out.compact) {
      notes.push(out.refused);
      toolCalls.push({ name: reply.tool, arguments: reply.arguments || {}, ok: false, refused: out.refused });
      onEvent(`call refused: ${out.refused}`);
      continue;
    }
    onEvent(`Ran ${out.name || reply.tool}.`);
    const ok = Boolean(out.ok);
    results.push({ name: out.name || reply.tool, arguments: out.arguments || {}, compact: out.compact || {}, ok });
    toolCalls.push({ name: out.name || reply.tool, arguments: out.arguments || {}, ok });
  }
  return finish(null, results.length || !toolCalls.some((c) => c.refused)
    ? `the model used its ${budget} steps without picking its facts` : noneWhy());
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/**
 * The answer block. When the package wrote the answer: its sentences, with a line saying the model only picked
 * them. Otherwise the facts the tools gave (the package's summary lines) under one short line saying why.
 */
export function deviceAnswerHtml(result) {
  const head = `<div class="showcase-note">Answered on your device by ${esc(result.model || "a local model")}, ` +
    `with a reduced tool set and no key. The model chose the tools and picked the facts; AquaScope wrote every ` +
    `sentence from the tool results. For a hard question, bring a key above.</div>`;
  // Under a refusal or a stop nothing was answered, so no "Answered" header (check4): the facts are the tools',
  // in AquaScope's words, and a line below says where it ran.
  const foot = `<div class="showcase-note">Ran on your device by ${esc(result.model || "a local model")}, with no ` +
    `key; the lines above are AquaScope's, from the tool results. For a hard question, bring a key above.</div>`;
  const lock = result.lock || { ok: false, refused: ["no answer"], facts: [], notes: [] };
  const notes = Array.isArray(lock.notes) && lock.notes.length
    ? `<p class="ask-device-note">${lock.notes.map(esc).join(" ")}</p>` : "";
  if (lock.ok) {
    const lines = Array.isArray(lock.sentences) && lock.sentences.length ? lock.sentences : [lock.text || result.answer];
    const body = isList(lock) ? `<ul class="ask-device-facts">${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`
      : `<p>${lines.map(esc).join(" ")}</p>`;
    return `${head}${body}${notes}`;
  }
  const facts = Array.isArray(lock.facts) && lock.facts.length
    ? `<ul class="ask-device-facts">${lock.facts.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`
    : "<p>No tool ran for this question, so there are no results to show. Ask again, naming the gauge or the place.</p>";
  return `<p class="ask-refused">${esc(leadLine(result))}. What the tools returned:</p>` + facts + notes + foot;
}

// A list of gauges (two or more search rows picked) reads as a list, one gauge a line.
const isList = (lock) => Array.isArray(lock.picked) && lock.picked.filter((k) => String(k).endsWith(".where")).length >= 2;

function leadLine(result) {
  const lock = result.lock || {};
  const why = (lock.refused || []).join("; ") || "no answer";
  return result.stopped ? `No answer: ${result.stopped}` : `The model's reply was refused: ${why}`;
}

/** The same, as Markdown for Copy and Download. */
export function deviceAnswerMarkdown(question, result) {
  const lock = result.lock || { ok: false, refused: ["no answer"], facts: [] };
  const notes = Array.isArray(lock.notes) && lock.notes.length ? `\n${lock.notes.join(" ")}\n` : "";
  if (lock.ok) {
    const lines = Array.isArray(lock.sentences) && lock.sentences.length ? lock.sentences : [lock.text || result.answer];
    return `# ${question}\n\n${isList(lock) ? lines.map((l) => `- ${l}`).join("\n") : lines.join(" ")}\n${notes}`;
  }
  const facts = (lock.facts || []).map((f) => `- ${f}`).join("\n") || "No tool ran for this question.";
  return `# ${question}\n\n${leadLine(result)}. What the tools returned:\n\n${facts}\n${notes}`;
}
