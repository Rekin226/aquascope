// AquaScope's free tier: Claude for visitors without a key, for a limited time, paid from the AI for Science
// credits. The key lives here as a Worker secret; the Explorer stays a static page.
//
//   GET  /status        is the tier open, until when, and what a visitor gets
//   POST /session       a Turnstile token in, a signed session token out (24 h)
//   POST /run           the start of an Ask, a map command or a study, counted against the visitor's runs
//   POST /v1/messages   the Messages API, checked, charged and passed on (the session token is the x-api-key)
//   GET  /stats         daily totals for the stop-rule review (Authorization: Bearer ADMIN_TOKEN)
//
// Logs carry counts and dollars only. Prompts and replies pass through and are never stored.

import { DurableObject } from "cloudflare:workers";

import {
  BUDGET_MESSAGES, PRICES, checkRequest, corsHeaders, costOf, limitsFrom, MAX_BODY_BYTES, networkKey,
  signSession, tierState, utcDay, verifySession,
} from "./core.js";
import { Ledger } from "./ledger.js";

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

export class Meter extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ledger = new Ledger(ctx.storage, limitsFrom(env));
  }

  async check(visitor, network) { return this.ledger.check(visitor, network); }
  async charge(visitor, network, usd) { return this.ledger.charge(visitor, network, usd); }
  async startRun(visitor, network, kind) { return this.ledger.startRun(visitor, network, kind); }
  async allowance(visitor) { return this.ledger.allowance(visitor); }
  async stats(days) { return this.ledger.stats(days); }

  async alarm() {
    await this.ledger.prune();
    await this.ctx.storage.setAlarm(Date.now() + 6 * 3600 * 1000);
  }

  async ensureAlarm() {
    if ((await this.ctx.storage.getAlarm()) === null) await this.ctx.storage.setAlarm(Date.now() + 3600 * 1000);
  }
}

function json(data, status, cors) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", ...(cors || {}) },
  });
}

// Errors in the Messages API's own shape, so the package's client reads them like any other provider's. A
// spent budget is a 402, never a 429: a 429 would be retried, and waiting does not refill a budget.
function apiError(status, type, message, cors) {
  return json({ type: "error", error: { type, message } }, status, cors);
}

async function visitorOf(request, env) {
  return verifySession(env.SESSION_SECRET, request.headers.get("x-api-key") || "");
}

function meter(env) {
  return env.METER.get(env.METER.idFromName("global"));
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const cors = corsHeaders(request.headers.get("Origin"), env.ALLOWED_ORIGINS);
    if (request.method === "OPTIONS") {
      return cors ? new Response(null, { status: 204, headers: cors }) : new Response(null, { status: 403 });
    }
    if (request.headers.get("Origin") && !cors) return apiError(403, "permission_error", "origin not allowed");

    const state = tierState(env);
    const limits = limitsFrom(env);

    if (url.pathname === "/status" && request.method === "GET") {
      return json({
        state, end_date: env.END_DATE || null, models: Object.keys(PRICES),
        per_visitor_per_day: { asks: limits.VISITOR_ASKS, studies: limits.VISITOR_STUDIES },
        sponsor: env.SPONSOR_LINE || null, site_key: env.TURNSTILE_SITE_KEY || null,
      }, 200, cors);
    }

    if (url.pathname === "/stats" && request.method === "GET") {
      if (!env.ADMIN_TOKEN || request.headers.get("Authorization") !== `Bearer ${env.ADMIN_TOKEN}`) {
        return apiError(401, "authentication_error", "admin token required", cors);
      }
      return json(await meter(env).stats(Number(url.searchParams.get("days") || 14)), 200, cors);
    }

    if (state !== "open") return apiError(402, "free_tier_closed", BUDGET_MESSAGES[state], cors);

    if (url.pathname === "/session" && request.method === "POST") {
      const body = await request.json().catch(() => ({}));
      const form = new FormData();
      form.append("secret", env.TURNSTILE_SECRET || "");
      form.append("response", String(body.turnstile || ""));
      const ip = request.headers.get("CF-Connecting-IP");
      if (ip) form.append("remoteip", ip);
      const verdict = await fetch(TURNSTILE_URL, { method: "POST", body: form }).then((r) => r.json()).catch(() => ({}));
      if (!verdict.success) return apiError(403, "permission_error", "the browser check did not pass; reload and try again", cors);
      const visitor = crypto.randomUUID();
      const expires = Date.now() + limits.SESSION_HOURS * 3600 * 1000;
      const token = await signSession(env.SESSION_SECRET, visitor, expires);
      const m = meter(env);
      ctx.waitUntil(m.ensureAlarm());
      return json({ token, expires, left: await m.allowance(visitor), end_date: env.END_DATE || null }, 200, cors);
    }

    const visitor = await visitorOf(request, env);
    if (!visitor) return apiError(401, "authentication_error", "the free session expired; start it again", cors);
    const network = await networkKey(env.SESSION_SECRET, request.headers.get("CF-Connecting-IP"), utcDay());

    if (url.pathname === "/run" && request.method === "POST") {
      const body = await request.json().catch(() => ({}));
      const res = await meter(env).startRun(visitor, network, String(body.kind || ""));
      if (res.ok) return json(res, 200, cors);
      const message = res.reason === "asks" ? "You have used today's free questions; they reset at 00:00 UTC."
        : res.reason === "studies" ? "You have used today's free studies; they reset at 00:00 UTC."
          : BUDGET_MESSAGES[res.reason] || res.message;
      return apiError(402, "free_tier_limit", message, cors);
    }

    if (url.pathname === "/v1/messages" && request.method === "POST") {
      const raw = await request.text();
      if (raw.length > MAX_BODY_BYTES) return apiError(413, "request_too_large", "request too large", cors);
      let body;
      try { body = JSON.parse(raw); } catch { return apiError(400, "invalid_request_error", "invalid JSON", cors); }
      const beta = request.headers.get("anthropic-beta") || "";
      const refusal = checkRequest(body, beta);
      if (refusal) return apiError(400, "invalid_request_error", refusal, cors);

      const m = meter(env);
      const over = await m.check(visitor, network);
      if (over) return apiError(402, "free_tier_limit", BUDGET_MESSAGES[over], cors);

      const headers = {
        "content-type": "application/json",
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
      };
      if (beta) headers["anthropic-beta"] = beta;
      if (env.ANTHROPIC_WORKSPACE_ID) headers["anthropic-workspace-id"] = env.ANTHROPIC_WORKSPACE_ID;
      const upstream = await fetch(ANTHROPIC_URL, { method: "POST", headers, body: raw });
      const text = await upstream.text();
      if (upstream.ok) {
        let usd = 0;
        try { usd = costOf(body.model, JSON.parse(text).usage || {}); } catch { /* charged nothing it cannot read */ }
        ctx.waitUntil(m.charge(visitor, network, usd));
      }
      // A 429 from Anthropic is passed on as a 429: that one does clear with time.
      return new Response(text, { status: upstream.status, headers: { "content-type": "application/json", ...(cors || {}) } });
    }

    return apiError(404, "not_found_error", "no such endpoint", cors);
  },
};
