// The free tier's rules on the page, with no DOM and no network, so `node --test` can check them.
//
// A visitor with no key can use Claude through AquaScope's proxy for a limited time (deploy/free-tier/). The
// proxy holds the key and the budgets; the page holds a session token for the day and asks the proxy before each
// Ask, map command or study. A key of the visitor's own always comes first.

/** The model each kind of run uses on the free tier: the small one for questions, the stronger one for studies. */
export const FREE_MODELS = { ask: "claude-haiku-5-5", map: "claude-haiku-5-5", study: "claude-sonnet-5-5" };

const MODEL_NAMES = { "claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5" };

export const SESSION_STORE = "aquascope.free.session";

/** A session that is still good for at least a minute, from what storage holds, or null. */
export function liveSession(raw, nowMs = Date.now()) {
  let s = raw;
  if (typeof raw === "string") {
    try { s = JSON.parse(raw); } catch { return null; }
  }
  if (!s || typeof s.token !== "string" || !s.token.startsWith("aqf_")) return null;
  if (!Number.isFinite(s.expires) || s.expires < nowMs + 60_000) return null;
  return s;
}

/** The model settings a run hands the worker on the free tier, or null without a live session or a proxy URL. */
export function freeModelConfig(purpose, session, url) {
  if (!session || !url) return null;
  const model = FREE_MODELS[purpose] || FREE_MODELS.ask;
  return {
    provider: "anthropic",
    model,
    api_key: session.token,
    base_url: url.replace(/\/+$/, ""),
    label: `${MODEL_NAMES[model] || model} (free, sponsored)`,
    free: true,
  };
}

/** One line for the tier: what it is, what a visitor gets, until when, and who pays. */
export function freeTierNote(status) {
  if (!status || status.state !== "open") return "";
  const per = status.per_visitor_per_day || {};
  const bits = ["Claude, free for a limited time."];
  if (per.asks || per.studies) bits.push(`${per.asks || 0} questions and ${per.studies || 0} studies a day.`);
  if (status.end_date) bits.push(`Until ${status.end_date}.`);
  if (status.sponsor) bits.push(`${status.sponsor}.`);
  return bits.join(" ");
}

/** What is left today, in words. */
export function allowanceLine(left) {
  if (!left) return "";
  const q = Number(left.asks) || 0;
  const s = Number(left.studies) || 0;
  return `Left today: ${q} question${q === 1 ? "" : "s"}, ${s} stud${s === 1 ? "y" : "ies"}.`;
}

/** The proxy's error body as one sentence for the visitor. */
export function proxyMessage(body, status) {
  const msg = body && body.error && body.error.message;
  if (msg) return msg;
  if (status === 401) return "The free session expired; start it again.";
  if (status === 402) return "The free allowance is used up for today.";
  return `The free tier could not answer (HTTP ${status}).`;
}
