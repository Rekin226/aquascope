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
// from a short list by replying with JSON, we run it in the worker, and after a
// few steps it writes the answer. That is honest about what a 4B model can do,
// and the UI says "on your device, reduced tool set" rather than pretending it
// is the same Analyst.

import { escapeHtml } from "./core.js?v=__BUILD__";

export const LOCAL_TOOLS = [
  {
    name: "find_stations",
    description: "Search the world gauge catalog. query for a name, near=[lat, lon] for the nearest, variable to filter.",
    example: '{"tool": "find_stations", "arguments": {"query": "Kingston", "limit": 5}}',
  },
  {
    name: "analyze_station",
    description: "Fetch and analyse one station: record, flood frequency, flow duration, trend.",
    example: '{"tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": "abc"}}',
  },
  {
    name: "anywhere",
    description: "Climate and modelled discharge at a point with no gauge (lat, lon).",
    example: '{"tool": "anywhere", "arguments": {"lat": 25.05, "lon": 121.55}}',
  },
  {
    name: "describe_catchment",
    description: "The catchment of a point: area, climate, land cover, population.",
    example: '{"tool": "describe_catchment", "arguments": {"lat": 47.0, "lon": -68.6}}',
  },
];

const SYSTEM = `You are AquaScope's assistant, running on the reader's own machine.
Answer only from tool results. You have these tools:
${LOCAL_TOOLS.map((t) => `- ${t.name}: ${t.description}\n  example: ${t.example}`).join("\n")}

Reply with ONE JSON object and nothing else, either
  {"tool": "<name>", "arguments": {...}}
to call a tool, or
  {"answer": "<two or three sentences, with units and the station name>"}
when you have enough. Never invent numbers: every figure must appear in a tool result.`;

const JSON_SCHEMA = {
  type: "object",
  properties: {
    tool: { type: "string", enum: LOCAL_TOOLS.map((t) => t.name) },
    arguments: { type: "object" },
    answer: { type: "string" },
  },
};

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
  if (promptApiAvailable()) return "Chrome's built-in model (nothing to download)";
  if (webgpuAvailable()) return "a small open model, about 2 GB, downloaded once and cached";
  return "not available in this browser (needs Chrome 148+, or WebGPU)";
}

// ── loading ─────────────────────────────────────────────────────────────────

export async function loadLocalModel(onProgress = () => {}) {
  if (engine) return engine;
  if (promptApiAvailable()) {
    onProgress("Starting Chrome's built-in model…");
    const availability = await globalThis.LanguageModel.availability();
    if (availability === "unavailable") throw new Error("Chrome's built-in model is not available on this device.");
    const session = await globalThis.LanguageModel.create({
      initialPrompts: [{ role: "system", content: SYSTEM }],
      monitor(m) {
        m.addEventListener("downloadprogress", (e) => onProgress(`Downloading the model: ${Math.round(e.loaded * 100)} %`));
      },
    });
    engine = {
      kind: "prompt-api",
      label: "Chrome built-in (Gemini Nano)",
      async generate(prompt) {
        return session.prompt(prompt, { responseConstraint: JSON_SCHEMA });
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
    async generate(prompt) {
      const res = await mlc.chat.completions.create({
        messages: [{ role: "system", content: SYSTEM }, { role: "user", content: prompt }],
        temperature: 0.2,
        response_format: { type: "json_object" },
      });
      return res.choices[0].message.content || "";
    },
  };
  return engine;
}

export function localModelLabel() {
  return engine ? engine.label : null;
}

// ── the reduced loop ────────────────────────────────────────────────────────

function parseReply(text) {
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
 * Ask the local model, running tools through `callTool` (the worker).
 * Returns { answer, toolCalls } and reports each step through onEvent.
 */
export async function askLocally(question, { callTool, onEvent = () => {}, maxSteps = 3, context = "" } = {}) {
  const eng = await loadLocalModel(onEvent);
  const toolCalls = [];
  let prompt = context ? `${question}\n\nContext: ${context}` : question;
  for (let step = 1; step <= maxSteps; step++) {
    const reply = parseReply(await eng.generate(prompt));
    if (reply.answer || !reply.tool) {
      return { answer: reply.answer || "The local model did not produce an answer.", toolCalls, model: eng.label };
    }
    onEvent(`tool ${reply.tool}(${JSON.stringify(reply.arguments || {}).slice(0, 120)})`);
    let payload;
    try {
      payload = await callTool(reply.tool, reply.arguments || {});
    } catch (err) {
      payload = { error: err.message };
    }
    toolCalls.push({ name: reply.tool, arguments: reply.arguments || {}, ok: !(payload && payload.error) });
    const trimmed = JSON.stringify(payload).slice(0, 3000);
    prompt = `${question}\n\nResult of ${reply.tool}: ${trimmed}\n\n` +
      "Now reply with either another tool call or the final answer, as one JSON object.";
  }
  return {
    answer: "The local model used its steps without reaching an answer. A larger model (bring your own key) will do better on this question.",
    toolCalls,
    model: eng.label,
  };
}

export function localAnswerHtml(result) {
  return `<div class="showcase-note">Answered on your device by ${escapeHtml(result.model || "a local model")}, ` +
    `with a reduced tool set and no key. Small models are weaker at multi-step work: for a hard question, ` +
    `bring a key above.</div><p>${escapeHtml(result.answer)}</p>`;
}
