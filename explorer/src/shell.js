// The app shell: which surface is showing (empty / station / point), the
// tabbed inspector, the Ask drawer, per-card states and the boot progress bar.
// Nothing here knows about hydrology; the panels call into it.

import { $, escapeHtml, state } from "./core.js?v=__BUILD__";
import { announce, captureFocus } from "./a11y.js?v=__BUILD__";

const SURFACES = ["panel-empty", "panel-station", "panel-point", "panel-workbench"];

export function showSurface(id) {
  for (const s of SURFACES) { const el = $(s); if (el) el.hidden = s !== id; }
  const panel = $("panel");
  if (panel) panel.scrollTop = 0;
}

export function isSurface(id) {
  const el = $(id);
  return Boolean(el && !el.hidden);
}

// ── tabs ────────────────────────────────────────────────────────────────────
// One tablist per surface. Tabs whose pane has nothing to show are disabled
// with the reason on the button, so "not available here" never looks like
// "still loading".

export function initTabs(root) {
  const list = root.querySelector('[role="tablist"]');
  if (!list) return;
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  const select = (tab, focus = true) => {
    if (!tab || tab.disabled) return;
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      const pane = root.querySelector(`#${t.getAttribute("aria-controls")}`);
      if (pane) pane.hidden = !on;
    }
    if (focus) tab.focus();
    root.dispatchEvent(new CustomEvent("tabchange", { detail: { tab: tab.dataset.tab }, bubbles: true }));
    const scroller = root.querySelector(".tabs");
    if (scroller) tab.scrollIntoView({ block: "nearest", inline: "nearest" });
  };
  list.addEventListener("click", (e) => {
    const tab = e.target.closest('[role="tab"]');
    if (tab) { root.__wantTab = null; select(tab, false); }
  });
  list.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    root.__wantTab = null;
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (step) {
      e.preventDefault();
      const n = tabs.length;
      for (let k = 1; k <= n; k++) {
        const t = tabs[(((i + step * k) % n) + n) % n];
        if (!t.disabled) { select(t); break; }
      }
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      const pool = e.key === "Home" ? tabs : [...tabs].reverse();
      select(pool.find((t) => !t.disabled));
    }
  });
  root.__selectTab = (name, focus = false) => select(tabs.find((t) => t.dataset.tab === name), focus);
  root.__tabs = tabs;
}

export function activeTab(root) {
  const t = root && root.querySelector('[role="tab"][aria-selected="true"]');
  return t ? t.dataset.tab : null;
}

// Ask for a tab that may not be fillable yet (a deep link opens on Floods
// before the record has arrived). It is applied as soon as that tab is
// enabled, unless the reader picks another one first.
export function selectTab(root, name, focus = false) {
  if (!root) return;
  const tab = root.querySelector(`[role="tab"][data-tab="${name}"]`);
  if (tab && tab.disabled) { root.__wantTab = name; return; }
  root.__wantTab = null;
  if (root.__selectTab) root.__selectTab(name, focus);
}

// Enable/disable a tab and say why. `count` shows a small badge (e.g. 8 donors).
export function setTab(root, name, { enabled = true, reason = "", count = null } = {}) {
  const tab = root && root.querySelector(`[role="tab"][data-tab="${name}"]`);
  if (!tab) return;
  tab.disabled = !enabled;
  tab.title = enabled ? tab.dataset.title || "" : reason || "Not available for this selection";
  const badge = tab.querySelector(".tab-badge");
  if (badge) {
    badge.textContent = count === null || count === undefined ? "" : String(count);
    badge.hidden = count === null || count === undefined;
  }
  if (!enabled && tab.getAttribute("aria-selected") === "true") {
    const next = root.__tabs && root.__tabs.find((t) => !t.disabled);
    if (next) selectTab(root, next.dataset.tab);
  }
  // A tab someone asked for (deep link, Back) opens as soon as it can be filled.
  if (enabled && root.__wantTab === name) {
    root.__wantTab = null;
    if (root.__selectTab) root.__selectTab(name, false);
  }
}

// ── card states ─────────────────────────────────────────────────────────────
// setCard(el, "loading" | "ready" | "empty" | "error", {...}) replaces the old
// silent console.info: a card that cannot be filled says whether that is
// because it does not apply here or because something failed, with a retry.

export function setCard(el, kind, { message = "", retry = null, title = "" } = {}) {
  if (!el) return;
  el.dataset.state = kind;
  el.hidden = false;
  const body = el.querySelector(".card-body") || el;
  const note = el.querySelector(".card-note") || (() => {
    const d = document.createElement("div");
    d.className = "card-note";
    el.appendChild(d);
    return d;
  })();
  const hasOwnBody = body !== el;
  if (kind === "ready") {
    note.hidden = true; note.innerHTML = "";
    if (hasOwnBody) body.hidden = false;
    return;
  }
  if (hasOwnBody) body.hidden = true;
  note.hidden = false;
  if (kind === "loading") {
    note.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${escapeHtml(message || "Working…")}</span>`;
    return;
  }
  const icon = kind === "error" ? "!" : "·";
  const text = message || (kind === "empty" ? "Not available here." : "Something went wrong.");
  note.innerHTML = `<span class="note-icon ${kind}">${icon}</span><span>${escapeHtml(text)}</span>`;
  if (kind === "error") {
    // A card that fails says so on screen; say it to a reader too. "empty" is
    // not announced: several cards are legitimately empty on most stations.
    const heading = el.querySelector("h3, h2");
    announce(heading ? `${heading.textContent.trim()}: ${text}` : text);
  }
  if (retry) {
    const b = document.createElement("button");
    b.className = "btn tiny";
    b.textContent = "Retry";
    b.addEventListener("click", () => { setCard(el, "loading", { message: "Retrying…" }); retry(); });
    note.appendChild(b);
  }
  if (title) el.title = title;
}

export function hideCard(el) {
  if (!el) return;
  el.hidden = true;
  el.dataset.state = "";
  const note = el.querySelector(".card-note");
  if (note) { note.hidden = true; note.innerHTML = ""; }
}

// ── status lines ────────────────────────────────────────────────────────────

export function setStatusEl(el, text, kind = "info") {
  if (!el) return;
  el.textContent = text || "";
  el.className = `status ${kind}`;
  el.hidden = !text;
  if (text && (kind === "error" || kind === "warn")) announce(text);
}

// ── boot progress ───────────────────────────────────────────────────────────
// The Pyodide download is ~15 MB once. The worker already reports each stage;
// the bar shows them wherever the user is (it used to be dropped unless a
// station happened to be selected).

const BOOT_STEPS = [
  [/pyodide|python runtime/i, 0.25],
  [/numpy|scipy|pandas/i, 0.55],
  [/installing aquascope/i, 0.8],
];

export function bootProgress(text) {
  const bar = $("boot"), fill = $("boot-fill"), label = $("boot-label");
  if (!bar) return;
  if (!text) { bar.hidden = true; return; }
  bar.hidden = false;
  label.textContent = text;
  let pct = 0.15;
  for (const [re, p] of BOOT_STEPS) if (re.test(text)) pct = p;
  fill.style.width = `${Math.round(pct * 100)}%`;
}

export function bootDone() {
  const bar = $("boot"), fill = $("boot-fill");
  if (!bar || bar.hidden) return;
  fill.style.width = "100%";
  setTimeout(() => { bar.hidden = true; fill.style.width = "0%"; }, 400);
}

// ── drawer (Ask) ────────────────────────────────────────────────────────────
// The drawer sits beside the inspector instead of replacing it, so the station
// stays on screen while the Analyst works.

let releaseDrawer = null;

export function openDrawer() {
  const d = $("drawer");
  if (!d) return;
  d.hidden = false;
  document.body.classList.add("drawer-open");
  $("btn-ask").setAttribute("aria-expanded", "true");
  // Not trapped: on a wide screen the drawer sits beside the inspector, and
  // tabbing out to the map is the right behaviour.
  releaseDrawer = captureFocus(d, { onEscape: closeDrawer, restoreTo: $("btn-ask") });
  announce("Ask panel opened");
}

export function closeDrawer() {
  const d = $("drawer");
  if (!d) return;
  const wasOpen = !d.hidden;
  d.hidden = true;
  document.body.classList.remove("drawer-open");
  $("btn-ask").setAttribute("aria-expanded", "false");
  if (releaseDrawer) { releaseDrawer({ restore: wasOpen }); releaseDrawer = null; }
}

export function toggleDrawer() {
  const d = $("drawer");
  if (d && d.hidden) openDrawer(); else closeDrawer();
}

export function drawerOpen() {
  const d = $("drawer");
  return Boolean(d && !d.hidden);
}

// ── left rail (mobile) ──────────────────────────────────────────────────────

let releaseRail = null;

export function toggleRail(force) {
  const open = force === undefined ? !document.body.classList.contains("rail-open") : force;
  document.body.classList.toggle("rail-open", open);
  const btn = $("btn-rail");
  if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
  // The rail only overlays the map on a narrow screen; that is also the only
  // width where the toggle exists, so focus follows it there and nowhere else.
  if (open && btn && btn.offsetParent !== null) {
    releaseRail = captureFocus($("rail"), { onEscape: () => toggleRail(false), restoreTo: btn });
  } else if (releaseRail) {
    releaseRail({ restore: true });
    releaseRail = null;
  }
}

// ── modal ───────────────────────────────────────────────────────────────────

export function openModal(title, html) {
  const m = $("modal");
  $("modal-title").textContent = title;
  $("modal-body").innerHTML = html;
  m.hidden = false;
  // aria-modal="true" is a promise to the reader that the rest of the page is
  // out of reach, so Tab has to stay inside for it to be true.
  m.__release = captureFocus(m, { onEscape: closeModal, trap: true });
  announce(title);
}

export function closeModal() {
  const m = $("modal");
  if (!m || m.hidden) return;
  m.hidden = true;
  if (m.__release) { m.__release(); m.__release = null; }
}

export function initShell() {
  $("modal-close").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  $("btn-rail").addEventListener("click", () => toggleRail());
  $("drawer-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      $("search").focus();
      $("search").select();
    }
  });
  // Reflect state for tests and debugging.
  state.shellReady = true;
}
