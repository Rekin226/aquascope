// A model that runs on the reader's own machine, so Ask works with no key.
//
// Two routes, in order of preference:
//
//   1. Chrome's built-in Prompt API (`LanguageModel`, Gemini Nano), which
//      shipped to web pages in Chrome 148. Nothing to download from us.
//   2. WebLLM over WebGPU: a 2-4 GB model, downloaded once and cached by the
//      browser. The reader chooses to start that download; we never begin it
//      on our own.
//
// Neither does native tool calling reliably at this size, so the loop here is
// deliberately smaller than the Python one: the model picks one tool at a time
// from a short list by replying with JSON, we run it in the worker, and then it
// picks the facts of its answer by key. It never writes a sentence the reader
// sees: AquaScope writes each one from the tool results (#590, round 3). The UI
// says "on your device, reduced tool set" rather than pretending it is the same
// Analyst.

import {
  askLoop, createSession, deviceAnswerHtml, samplingNote, samplingOptions, sessionPool,
} from "./ask-local-core.js?v=__BUILD__";

// No example arguments: check3 found the model copying the example id "abc" into a real call. Which ids and
// places a call may take is in each step's message and schema, built from the question.
export const LOCAL_TOOLS = [
  { name: "find_stations", description: "Search the world gauge catalog: query (words from the question) or near [lat, lon]; limit up to 10." },
  { name: "analyze_station", description: "One gauge's record, flow duration, flood frequency and trend: source and station_id." },
  { name: "anywhere", description: "Climate and modelled discharge at a point with no gauge: lat, lon." },
  { name: "describe_catchment", description: "The catchment of a point: area, climate, land cover, population: lat, lon." },
];

const SYSTEM = `You are AquaScope's assistant, running on the reader's own machine. You do not write the answer: you
choose. AquaScope writes every sentence the reader sees from the tool results.
Tools:
${LOCAL_TOOLS.map((t) => `- ${t.name}: ${t.description}`).join("\n")}

Each message gives the question, the tools you may call now with the gauge ids and points they may take, and the
facts you may answer with, each by its key. Reply with ONE JSON object and nothing else, either
  {"tool": "<name>", "arguments": {...}}
to run a tool, or
  {"answer": ["<fact key>", ...]}
with the keys of the facts that answer the question, in the order to show them. Use only the keys, ids and points
the message lists. Do not call a tool again with the same arguments: its result is already in the facts.
For a list of gauges pick the search's gauges; for a trend pick the trend fact; for a flood pick the return level
asked for.`;

let engine = null;      // { kind: "prompt-api" | "webllm", generate(messages) }

export function webgpuAvailable() {
  return Boolean(navigator.gpu);
}

export function promptApiAvailable() {
  return typeof globalThis.LanguageModel !== "undefined";
}

export function localModelPossible() {
  return promptApiAvailable() || webgpuAvailable();
}

export function describeLocal() {
  if (promptApiAvailable()) return "Chrome's built-in model";
  if (webgpuAvailable()) return "a small open model, about 2 GB, downloaded once";
  return "not available in this browser (needs Chrome 138+, or WebGPU)";
}

// ── loading ─────────────────────────────────────────────────────────────────

export async function loadLocalModel(onProgress = () => {}) {
  if (engine) return engine;
  if (promptApiAvailable()) {
    const availability = await globalThis.LanguageModel.availability();
    if (availability === "unavailable") throw new Error("Chrome's built-in model is not available on this device.");
    // Chrome fires downloadprogress even when the model is already on the
    // machine, and it arrives as loaded: 1. Reporting that verbatim put
    // "Downloading the model: 100 %" on screen for the whole run, which reads
    // as a download stuck at the finish line rather than a model thinking.
    // Only availability says whether anything is actually being fetched.
    const willDownload = availability === "downloadable" || availability === "downloading";
    onProgress(willDownload
      ? "Downloading Chrome's built-in model (once)…"
      : "Starting Chrome's built-in model…");
    // One base session holds the model warm and is never prompted; every step of every question gets a clone
    // of it (the system prompt and nothing else), prompts it once and destroys it, so no question can see
    // another's tool results (#590: a later answer quoted an earlier question's Q10 and record start), and no
    // session takes a second prompt (which failed with "kErrorUnknown" in 11 of 13 trials on the device). A
    // clone that fails falls back to a new session.
    const LM = globalThis.LanguageModel;
    const sampling = await samplingOptions(LM, 0.2);
    const fresh = () => createSession(LM, { initialPrompts: [{ role: "system", content: SYSTEM }], ...sampling });
    const base = await createSession(LM, {
      initialPrompts: [{ role: "system", content: SYSTEM }],
      ...sampling,
      monitor(m) {
        m.addEventListener("downloadprogress", (e) => {
          if (!willDownload) return;
          const pct = Math.round((e.loaded || 0) * 100);
          onProgress(pct >= 100 ? "Model downloaded, starting it…" : `Downloading the model: ${pct} %`);
        });
      },
    });
    engine = {
      kind: "prompt-api",
      label: "Chrome built-in (Gemini Nano)",
      sampling: samplingNote(base),
      async open() {
        let session = null;
        try { session = typeof base.clone === "function" ? await base.clone() : null; } catch { session = null; }
        if (!session) session = await fresh();
        return {
          // The step's schema: enums built by the package from this question's words and results. The signal is
          // the step's time limit and the Stop button (askLoop), as generateJsonLocally's timeout is for the map.
          generate: (prompt, schema, signal) => session.prompt(prompt, {
            ...(schema ? { responseConstraint: schema } : {}), ...(signal ? { signal } : {}) }),
          close() { try { session.destroy(); } catch { /* fine */ } },
        };
      },
    };
    return engine;
  }
  if (!webgpuAvailable()) throw new Error("This browser has no WebGPU, so a local model cannot run here.");

  onProgress("Loading the WebLLM runtime…");
  const webllm = await import("https://esm.run/@mlc-ai/web-llm");
  // Pick from the runtime's own list rather than hard-coding an id that may be
  // renamed: a small, quantised, instruction-tuned model.
  const wanted = /(Qwen3-4B|Qwen2\.5-3B|Llama-3\.2-3B|Qwen3-1\.7B|gemma-2-2b)/i;
  const models = (webllm.prebuiltAppConfig && webllm.prebuiltAppConfig.model_list) || [];
  const pick = models.find((m) => wanted.test(m.model_id) && /q4f16/i.test(m.model_id))
    || models.find((m) => wanted.test(m.model_id))
    || models.find((m) => /q4f16/i.test(m.model_id));
  if (!pick) throw new Error("No suitable small model in the WebLLM catalogue.");
  onProgress(`Downloading ${pick.model_id} (once, then cached)…`);
  const mlc = await webllm.CreateMLCEngine(pick.model_id, {
    initProgressCallback: (p) => onProgress(p.text || `Loading ${pick.model_id}…`),
  });
  engine = {
    kind: "webllm",
    label: pick.model_id,
    mlc,
    sampling: "temperature 0.2",
    // WebLLM keeps no conversation: each call is the system prompt and this one prompt.
    async open() {
      return {
        async generate(prompt, schema, signal) {
          if (signal) signal.addEventListener("abort", () => { try { mlc.interruptGenerate(); } catch { /* fine */ } }, { once: true });
          const res = await mlc.chat.completions.create({
            messages: [{ role: "system", content: SYSTEM }, { role: "user", content: prompt }],
            temperature: 0.2,
            response_format: schema ? { type: "json_object", schema: JSON.stringify(schema) } : { type: "json_object" },
          });
          return res.choices[0].message.content || "";
        },
        close() {},
      };
    },
  };
  return engine;
}

export function localModelLabel() {
  return engine ? engine.label : null;
}

// The model a structured call would run on, named for a status line.
export function localReaderLabel() {
  if (promptApiAvailable()) return "Chrome's built-in model";
  return engine ? engine.label : null;
}

// ── one small structured call ───────────────────────────────────────────────
// Study reads the first sentence into a brief with a single call: its own
// system prompt, a schema, a low temperature, a timeout. It must never start a
// download on its own (a sentence typed into Study is not a choice to fetch
// gigabytes): it runs when Chrome's model is already on the machine, or when a
// WebLLM engine is already loaded in this tab.

const READY = { ok: false };  // cached: Chrome's model, already on this machine

export async function localModelReady() {
  if (engine) return true;
  if (READY.ok) return true;
  if (!promptApiAvailable()) return false;
  try {
    READY.ok = (await globalThis.LanguageModel.availability()) === "available";
  } catch {
    READY.ok = false;
  }
  return READY.ok;
}

function withTimeout(promise, ms, controller) {
  let timer;
  const clock = new Promise((_, reject) => {
    timer = setTimeout(() => {
      try { controller.abort(); } catch { /* fine */ }
      reject(new Error(`the on-device model took more than ${Math.round(ms / 1000)} s`));
    }, ms);
  });
  return Promise.race([promise, clock]).finally(() => clearTimeout(timer));
}

/**
 * One reply from the on-device model as text: `system` and `prompt`, a JSON
 * `schema` the reply is constrained to where the runtime allows, `temperature`
 * where the runtime takes one (see samplingOptions: a web page in Chrome 155 cannot set it), and `timeoutMs` after which the caller gets an error and
 * falls back. Chrome's model gets a session of its own for this one call, cloned
 * from a warm base session kept per system prompt (sessionPool; Ask's session,
 * with its tool prompt, is not touched); WebLLM is used only when Ask has already
 * loaded it. Check `localModelReady()` first.
 */
let pool = null;

export async function generateJsonLocally({ system, prompt, schema = null, temperature = 0.1, timeoutMs = 25000,
                                            onProgress = () => {} } = {}) {
  const controller = new AbortController();
  if (promptApiAvailable()) {
    onProgress("Reading your words on your device...");
    // Sampling as this Chrome accepts it (samplingOptions): topK and temperature in an extension, the
    // samplingMode word where the origin trial allows it, else Chrome's own.
    const LM = globalThis.LanguageModel;
    if (!pool) pool = sessionPool(LM);
    const session = await pool.open(system, await samplingOptions(LM, temperature));
    try {
      const opts = { signal: controller.signal };
      if (schema) opts.responseConstraint = schema;
      return await withTimeout(session.prompt(prompt, opts), timeoutMs, controller);
    } finally {
      try { session.destroy(); } catch { /* fine */ }
    }
  }
  if (!engine || engine.kind !== "webllm" || !engine.mlc) throw new Error("no on-device model is loaded in this tab");
  onProgress(`Reading your words on your device (${engine.label})...`);
  const req = engine.mlc.chat.completions.create({
    messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
    temperature,
    response_format: schema ? { type: "json_object", schema: JSON.stringify(schema) } : { type: "json_object" },
  });
  const res = await withTimeout(req, timeoutMs, controller);
  return res.choices[0].message.content || "";
}

// ── the reduced loop ────────────────────────────────────────────────────────

/**
 * Ask the local model. `plan`, `callTool` and `render` are the package's steps in the worker
 * (aquascope.ai_engine.device_ask: step_schema, check_call then the tool, render_answer): the model picks, the
 * package checks and writes. One fresh conversation per step, each prompt ended by the time limit (STEP_MS,
 * 25 s) or by `signal` (the Stop button).
 * Returns { answer, toolCalls, results, lock, model } and reports each step through onEvent.
 */
export async function askLocally(question, { plan, callTool, render, onEvent = () => {}, context = "", signal = null,
                                            stepMs } = {}) {
  const eng = await loadLocalModel(onEvent);
  const out = await askLoop(question, { open: () => eng.open(), plan, callTool, render, onEvent, context, signal,
    ...(stepMs ? { stepMs } : {}) });
  return { ...out, model: eng.label, sampling: eng.sampling };
}

export function localAnswerHtml(result) {
  return deviceAnswerHtml(result);
}
