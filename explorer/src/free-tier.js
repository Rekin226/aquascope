// The free tier in the page: the proxy's status, the browser check that opens a session, and the count of runs.
// The rules are in free-tier-core.js; this module only talks to the proxy and to Cloudflare Turnstile.

import { CONFIG } from "../config.js?v=__BUILD__";
import {
  SESSION_STORE, allowanceLine, freeModelConfig, freeTierNote, liveSession, proxyMessage,
} from "./free-tier-core.js?v=__BUILD__";

const TURNSTILE_SRC = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

let statusPromise = null;
let lastLeft = null;

export function freeTierUrl() {
  return String(CONFIG.freeTierUrl || "").replace(/\/+$/, "");
}

/** The proxy's /status, fetched once per page; null when the tier is not configured or cannot be reached. */
export function freeTierStatus() {
  const url = freeTierUrl();
  if (!url) return Promise.resolve(null);
  if (!statusPromise) {
    statusPromise = fetch(`${url}/status`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return statusPromise;
}

export async function freeTierOpen() {
  const s = await freeTierStatus();
  return Boolean(s && s.state === "open" && s.site_key);
}

export async function freeTierLine() {
  return freeTierNote(await freeTierStatus());
}

export function currentSession() {
  try { return liveSession(sessionStorage.getItem(SESSION_STORE)); } catch { return null; }
}

function saveSession(s) {
  try { sessionStorage.setItem(SESSION_STORE, JSON.stringify(s)); } catch { /* the tab keeps it in memory */ }
}

export function endSession() {
  try { sessionStorage.removeItem(SESSION_STORE); } catch { /* fine */ }
}

/** The model settings for ``purpose`` (ask, map or study) on the free tier, or null without a session. */
export function freeConfig(purpose) {
  return freeModelConfig(purpose, currentSession(), freeTierUrl());
}

export function allowanceText() {
  return allowanceLine(lastLeft);
}

function loadTurnstile() {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = TURNSTILE_SRC;
    s.async = true;
    s.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error("the browser check did not load")));
    s.onerror = () => reject(new Error("the browser check did not load"));
    document.head.appendChild(s);
  });
}

/** A live session: the one this tab holds, or a new one after the browser check in ``container``. */
export async function ensureSession(container) {
  const have = currentSession();
  if (have) return have;
  const status = await freeTierStatus();
  if (!status || status.state !== "open" || !status.site_key) throw new Error("The free tier is not available right now.");
  const turnstile = await loadTurnstile();
  container.hidden = false;
  container.innerHTML = "";
  const token = await new Promise((resolve, reject) => {
    turnstile.render(container, {
      sitekey: status.site_key,
      theme: "auto",
      size: "flexible",
      callback: resolve,
      "error-callback": () => reject(new Error("the browser check failed; reload and try again")),
      "expired-callback": () => reject(new Error("the browser check expired; try again")),
    });
  });
  const res = await fetch(`${freeTierUrl()}/session`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ turnstile: token }),
  });
  const body = await res.json().catch(() => ({}));
  container.hidden = true;
  container.innerHTML = "";
  if (!res.ok) throw new Error(proxyMessage(body, res.status));
  const session = { token: body.token, expires: body.expires };
  saveSession(session);
  lastLeft = body.left || null;
  return session;
}

/** Count the start of a run (ask, map or study); throws the proxy's sentence when the allowance is spent. */
export async function startRun(kind) {
  const s = currentSession();
  if (!s) throw new Error("The free session expired; start it again.");
  const res = await fetch(`${freeTierUrl()}/run`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": s.token },
    body: JSON.stringify({ kind }),
  });
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) endSession();
  if (!res.ok) throw new Error(proxyMessage(body, res.status));
  lastLeft = body.left || null;
  return lastLeft;
}
