// Focus and announcements: the parts of the shell a keyboard or a screen reader
// needs and a mouse does not.
//
// Three surfaces open over or beside the page (the Ask drawer, the modal, and
// the left rail on a phone). Opening one used to leave focus where it was, so a
// keyboard user had to tab through the whole page to reach it and Escape did
// nothing. Closing one used to drop focus to the top of the document instead of
// back to the control that opened it.
//
// The modal is genuinely modal, so focus is trapped inside it. The drawer and
// the rail are not: on a wide screen they sit beside the page and you should be
// able to tab out to the map. They get focus, Escape and restore, without a
// trap. Trapping there would be worse than doing nothing.

const FOCUSABLE = [
  "a[href]", "button:not([disabled])", "input:not([disabled])", "select:not([disabled])",
  "textarea:not([disabled])", "summary", "[tabindex]:not([tabindex='-1'])",
].join(",");

export function focusableWithin(container) {
  if (!container) return [];
  return [...container.querySelectorAll(FOCUSABLE)].filter(
    (el) => !el.hidden && el.offsetParent !== null && !el.closest("[hidden]"),
  );
}

/** Move focus into `container`, preferring an element it marks as the first stop. */
export function focusFirst(container) {
  if (!container) return null;
  const preferred = container.querySelector("[data-autofocus]");
  const target = (preferred && !preferred.disabled) ? preferred : focusableWithin(container)[0];
  if (target) target.focus();
  return target || null;
}

/**
 * Open a surface for keyboard use and return the function that closes it again.
 *
 * `trap` keeps Tab inside the container (for a real modal). Without it, Tab
 * leaves normally, which is what a side panel should do.
 */
export function captureFocus(container, { onEscape = null, trap = false, restoreTo = null } = {}) {
  const previous = restoreTo || document.activeElement;
  focusFirst(container);

  const onKey = (e) => {
    if (e.key === "Escape" && onEscape) {
      // A trapped surface owns every key. An untrapped one only owns Escape
      // while focus is inside it: the search box and the area-select tool have
      // their own meaning for Escape, and the open drawer must not eat theirs.
      const where = e.target || document.activeElement;
      if (!trap && where && !container.contains(where)) return;
      e.preventDefault();
      onEscape();
      return;
    }
    if (!trap || e.key !== "Tab") return;
    const items = focusableWithin(container);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    } else if (!container.contains(document.activeElement)) {
      e.preventDefault();
      first.focus();
    }
  };

  document.addEventListener("keydown", onKey, true);
  return function release({ restore = true } = {}) {
    document.removeEventListener("keydown", onKey, true);
    if (restore && previous && typeof previous.focus === "function" && document.contains(previous)) {
      previous.focus();
    }
  };
}

// ── announcements ───────────────────────────────────────────────────────────
// One polite live region for the whole app. Cards already say "loading",
// "not available here" and "failed, retry" on screen; without this they say it
// only on screen.

let region = null;

function liveRegion() {
  if (region && document.contains(region)) return region;
  region = document.getElementById("a11y-live");
  if (!region) {
    region = document.createElement("div");
    region.id = "a11y-live";
    region.className = "visually-hidden";
    region.setAttribute("role", "status");
    region.setAttribute("aria-live", "polite");
    region.setAttribute("aria-atomic", "true");
    document.body.appendChild(region);
  }
  return region;
}

let last = "";

/** Say something once, politely. Repeats are re-announced by nudging the text. */
export function announce(message) {
  const text = String(message || "").trim();
  if (!text) return;
  const el = liveRegion();
  el.textContent = text === last ? `${text} ` : text;
  last = text;
}
