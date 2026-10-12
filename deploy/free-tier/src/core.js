// The free tier's rules, kept apart from the Worker so they can be tested with `node --test`.
//
// A visitor with no API key gets Claude through this proxy for a while: the key stays here, the request is
// checked against what the Explorer actually sends, and every call is charged against three budgets (the
// visitor's, the visitor's network's, and the day's). Nothing here reads or keeps prompts or replies; the
// Worker records token counts and dollars only.

/** USD per million tokens (input, output) for the models the free tier serves. Cache reads cost a tenth of the
 * input price and five-minute cache writes 1.25 times it, as on Anthropic's list (October 2026). */
export const PRICES = {
  "claude-haiku-5-5": [0.10, 0.50],
  "claude-sonnet-5-5": [2.0, 10.0],
};

/** What a request may carry. Everything else (server tools, other betas, streaming, sampling) is refused, so the
 * proxy serves the Explorer's calls and not general-purpose ones. */
export const ALLOWED_FIELDS = new Set([
  "model", "max_tokens", "messages", "system", "tools", "tool_choice", "output_config", "cache_control",
  "fallbacks", "metadata",
]);
export const ALLOWED_BETAS = new Set(["server-side-fallback-2026-07-01"]);
export const MAX_OUTPUT_TOKENS = 16000;
export const MAX_BODY_BYTES = 600_000;

/** The defaults; every one can be overridden by a Worker variable of the same name. */
export const DEFAULT_LIMITS = {
  DAILY_USD: 30,            // the whole free tier, per UTC day
  MONTHLY_USD: 1000,        // a backstop for the month
  VISITOR_USD: 2.5,         // one visitor (one session), per day: two Sonnet studies and the questions
  NETWORK_USD: 8,           // one IP address, per day (a campus shares one)
  VISITOR_ASKS: 20,         // Ask questions (and map commands), per visitor per day
  VISITOR_STUDIES: 2,       // Studio studies, per visitor per day
  SESSION_HOURS: 24,
};

export const RUN_KINDS = { ask: "VISITOR_ASKS", map: "VISITOR_ASKS", study: "VISITOR_STUDIES" };

export function limitsFrom(env = {}) {
  const out = { ...DEFAULT_LIMITS };
  for (const k of Object.keys(out)) {
    const v = Number(env[k]);
    if (env[k] !== undefined && env[k] !== "" && Number.isFinite(v) && v >= 0) out[k] = v;
  }
  return out;
}

/** The dollars one reply cost, from the usage block the API returned. */
export function costOf(model, usage = {}) {
  const rate = PRICES[model];
  if (!rate) return 0;
  const n = (k) => (Number.isFinite(usage[k]) ? usage[k] : 0);
  const usd = n("input_tokens") * rate[0] + n("output_tokens") * rate[1]
    + n("cache_read_input_tokens") * rate[0] * 0.1 + n("cache_creation_input_tokens") * rate[0] * 1.25;
  return usd / 1e6;
}

/** A refusal reason for a request body, or null when it is one the Explorer would send. */
export function checkRequest(body, betaHeader = "") {
  if (!body || typeof body !== "object" || Array.isArray(body)) return "the body is not a JSON object";
  for (const k of Object.keys(body)) {
    if (!ALLOWED_FIELDS.has(k)) return `the field ${k} is not served by the free tier`;
  }
  if (!PRICES[body.model]) return `the free tier serves ${Object.keys(PRICES).join(" and ")} only`;
  if (!Number.isInteger(body.max_tokens) || body.max_tokens < 1 || body.max_tokens > MAX_OUTPUT_TOKENS) {
    return `max_tokens must be between 1 and ${MAX_OUTPUT_TOKENS}`;
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) return "messages is empty";
  for (const t of body.tools || []) {
    // Only the Explorer's own tools: a server tool (web search, code execution) has a type and costs extra.
    if (!t || typeof t !== "object" || t.type || !t.name || !t.input_schema) {
      return "only the Explorer's own tools are served";
    }
  }
  const of = body.output_config;
  if (of !== undefined && (typeof of !== "object" || Object.keys(of).some((k) => k !== "effort"))) {
    return "output_config may only set effort";
  }
  if (body.fallbacks !== undefined && body.fallbacks !== "default") return "fallbacks may only be \"default\"";
  const betas = String(betaHeader || "").split(",").map((s) => s.trim()).filter(Boolean);
  for (const b of betas) if (!ALLOWED_BETAS.has(b)) return `the beta ${b} is not served`;
  if (!systemText(body.system).toLowerCase().includes("aquascope")) {
    // A speed bump, not a lock: the budgets are what bound misuse. Every prompt the package sends names itself.
    return "the free tier serves AquaScope's own prompts";
  }
  return null;
}

export function systemText(system) {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) return system.map((b) => (b && typeof b.text === "string" ? b.text : "")).join("\n");
  return "";
}

/** Which budget a spend would break, or null. ``spent`` holds today's and this month's dollars. */
export function overBudget(spent, limits) {
  if (spent.month >= limits.MONTHLY_USD) return "month";
  if (spent.day >= limits.DAILY_USD) return "day";
  if (spent.network >= limits.NETWORK_USD) return "network";
  if (spent.visitor >= limits.VISITOR_USD) return "visitor";
  return null;
}

export const BUDGET_MESSAGES = {
  month: "The free tier's budget for this month is used up.",
  day: "The free tier's budget for today is used up; it resets at 00:00 UTC.",
  network: "Your network has used today's free allowance; it resets at 00:00 UTC.",
  visitor: "You have used today's free allowance; it resets at 00:00 UTC.",
  ended: "The free tier has ended. Ask ✨ still works with your own key, on your device, or from the recorded runs.",
  off: "The free tier is switched off. Ask ✨ still works with your own key, on your device, or from the recorded runs.",
};

export function utcDay(now = new Date()) {
  return now.toISOString().slice(0, 10);
}

export function utcMonth(now = new Date()) {
  return now.toISOString().slice(0, 7);
}

/** Whether the tier is open: switched on, and the end date (YYYY-MM-DD, inclusive) not passed. */
export function tierState(env = {}, now = new Date()) {
  if (String(env.FREE_TIER_ENABLED ?? "true").toLowerCase() !== "true") return "off";
  if (env.END_DATE && utcDay(now) > env.END_DATE) return "ended";
  return "open";
}

// ── sessions: an HMAC-signed token, so the Worker needs no store to recognise a visitor ─────────────────────

const enc = new TextEncoder();

function b64url(bytes) {
  let s = "";
  for (const b of new Uint8Array(bytes)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function hmac(secret, data) {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return b64url(await crypto.subtle.sign("HMAC", key, enc.encode(data)));
}

export async function signSession(secret, visitor, expiresMs) {
  const body = `${visitor}.${expiresMs}`;
  return `aqf_${body}.${await hmac(secret, body)}`;
}

/** The visitor id of a valid, unexpired token, or null. */
export async function verifySession(secret, token, nowMs = Date.now()) {
  if (!secret || typeof token !== "string" || !token.startsWith("aqf_")) return null;
  const parts = token.slice(4).split(".");
  if (parts.length !== 3) return null;
  const [visitor, exp, sig] = parts;
  if (!/^[A-Za-z0-9-]{8,64}$/.test(visitor) || !/^\d+$/.test(exp)) return null;
  const expected = await hmac(secret, `${visitor}.${exp}`);
  if (expected.length !== sig.length) return null;
  let diff = 0;
  for (let i = 0; i < sig.length; i++) diff |= expected.charCodeAt(i) ^ sig.charCodeAt(i);
  if (diff !== 0 || Number(exp) < nowMs) return null;
  return visitor;
}

/** A short, salted hash of an IP address, so the network budget never stores the address itself. */
export async function networkKey(secret, ip, day) {
  return (await hmac(secret, `net:${ip || "unknown"}:${day}`)).slice(0, 22);
}

// ── CORS ────────────────────────────────────────────────────────────────────

export function corsHeaders(origin, allowedOrigins) {
  const allowed = String(allowedOrigins || "").split(",").map((s) => s.trim()).filter(Boolean);
  const ok = origin && allowed.some((a) => a === origin || (a.startsWith("*.") && origin.endsWith(a.slice(1))));
  if (!ok) return null;
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": [
      "content-type", "x-api-key", "anthropic-version", "anthropic-beta",
      "anthropic-dangerous-direct-browser-access", "user-agent", "accept",
    ].join(", "),
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}
