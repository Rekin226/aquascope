// The action log (#561), the pure part: what an AI (or the reader through "Ask the map") did to the map,
// in order, each with what it replaced, so any one of them can be undone. No DOM, no map: node imports this
// directly (explorer/tests/map-actions-core.test.mjs). map-actions.js applies the actions and draws the log.
//
// Every action changes one "slot" of the map: the view, the map date, one layer, the basemap, the river
// status focus, the lit river, or one pin or drawn area of its own. An entry keeps the slot's value before
// and after. Undoing the newest entry on a slot puts its "before" back; undoing an older one changes
// nothing on screen (a later action owns the slot now) but hands its "before" to the next entry on that
// slot, so undoing that one later goes all the way back. Undo all puts every slot back as it was before
// the first live entry touched it.

// The control behind each layer id of aquascope.map_commands.LAYERS (a test keeps the two lists equal).
// The action flips it exactly as the reader's click would, so the URL, the credits and the legends follow.
export const LAYER_CONTROLS = {
  status: "toggle-status",
  floods_past: "toggle-floods-past",
  groundwater: "toggle-groundwater",
  water_balance: "toggle-water-balance",
  floods_ahead: "toggle-floods",
  flood_depth: "toggle-depth",
  rivers: "toggle-rivers",
  flow: "toggle-flow",
  basins: "toggle-basins",
  precip: "ov-precip",
  soil: "ov-soil",
  snow: "ov-snow",
  lst: "ov-lst",
  storage: "ov-storage",
  "surface-water": "ov-surface-water",
  landcover: "ov-landcover",
  hillshade: "toggle-hillshade",
  terrain: "toggle-terrain",
  heat: "toggle-heat",
  globe: "btn-globe",
};

// Who answered, as the log says it.
export const BY_LABEL = { rules: "rules", device: "on device", key: "your key", agent: "agent", api: "note" };

export const MAX_ENTRIES = 60;

/** The slot an action changes. Pins and areas each get their own, named by the entry's id. */
export function slotOf(action, id) {
  switch (action && action.type) {
    case "fly_to": return "view";
    case "set_time": return "time";
    case "set_layer": return `layer:${action.layer}`;
    case "focus_status": return "status-focus";
    case "set_basemap": return "basemap";
    case "highlight_river": return "river";
    case "draw_area": return `area:${id}`;
    case "add_pin": return `pin:${id}`;
    case "what_if": return "whatif";   // one what-if painted at a time (#565); a new run replaces it
    default: return null;
  }
}

/** A pin as the API takes it ({lat, lon, title, text, facts, source}), checked; throws on a bad one. */
export function checkPin(p) {
  const num = (v) => (v === null || v === undefined || v === "" || typeof v === "boolean" ? NaN : Number(v));
  const lat = num(p && p.lat), lon = num(p && p.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    throw new Error("a pin needs a lat (-90 to 90) and a lon (-180 to 180)");
  }
  const title = String((p && p.title) || "").replace(/\s+/g, " ").trim().slice(0, 80);
  if (!title) throw new Error("a pin needs a title");
  const facts = Array.isArray(p.facts) ? p.facts.slice(0, 8)
    .filter((f) => f && f.label !== undefined && f.value !== undefined && f.value !== null)
    .map((f) => ({
      label: String(f.label).slice(0, 60), value: f.value,
      ...(f.unit ? { unit: String(f.unit).slice(0, 20) } : {}),
      ...(f.when ? { when: String(f.when).slice(0, 40) } : {}),
    }))
    : [];
  // A rank (Scout, #563) is drawn in the pin; a kind says which layer the note is about.
  const rank = Number(p.rank);
  const out = {
    lat: Math.round(lat * 1e5) / 1e5, lon: Math.round(lon * 1e5) / 1e5, title,
    text: p.text ? String(p.text).replace(/\s+/g, " ").trim().slice(0, 600) : "",
    facts,
    source: p.source ? String(p.source).slice(0, 320) : "",
    ...(Number.isInteger(rank) && rank >= 1 && rank <= 99 ? { rank } : {}),
    ...(p.kind ? { kind: String(p.kind).slice(0, 40) } : {}),
  };
  // A note (Explain, #562) is drawn as a short callout on the map rather than a pin you must open.
  if (p.kind === "note") {
    out.kind = "note";
    out.n = Number.isInteger(p.n) && p.n > 0 && p.n < 100 ? p.n : 0;
    out.tone = /^#[0-9a-f]{6}$/i.test(String(p.tone || "")) ? String(p.tone) : "";
    out.place = p.place ? String(p.place).replace(/\s+/g, " ").trim().slice(0, 80) : "";
    out.when = p.when ? String(p.when).slice(0, 40) : "";
    out.credit = p.credit ? String(p.credit).slice(0, 400) : "";
    out.group = p.group ? String(p.group).slice(0, 40) : "";
  }
  // A review finding (#564) flags cells of the user's model: which mesh, which cells, how severe.
  if (p.kind === "review") {
    out.kind = "review";
    out.severity = ["problem", "warning", "info"].includes(p.severity) ? p.severity : "info";
    out.check = p.check ? String(p.check).slice(0, 40) : "";
    out.mesh = p.mesh ? String(p.mesh).slice(0, 200) : "";
    out.cells = Array.isArray(p.cells) ? p.cells.slice(0, 5000).map(Number).filter((c) => Number.isInteger(c) && c >= 0) : [];
  }
  return out;
}

/**
 * What a note's map card says under its title (#562): the note's words, then each fact it cites that the title
 * and words do not already give, as "label: value", with its date only when it differs from the note's.
 */
export function noteBody(pin) {
  const text = String((pin && pin.text) || "");
  const said = `${(pin && pin.title) || ""} ${text}`;
  const extra = ((pin && pin.facts) || []).filter((f) => !said.toLowerCase().includes(String(f.value).toLowerCase()))
    .map((f) => `${f.label}: ${f.value}${f.unit ? ` ${f.unit}` : ""}${f.when && f.when !== pin.when ? ` (${f.when})` : ""}`);
  return [text, ...extra].filter(Boolean).join(" · ");
}

// ── where each note's card goes (#562) ─────────────────────────────────────

/** The ink for a number on a dot of colour `hex`: dark on the pale classes (tan, light teal, the 2-year yellow). */
export function inkOn(hex) {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(String(hex || ""));
  if (!m) return "#ffffff";
  const lin = (v) => { const c = parseInt(v, 16) / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * lin(m[1]) + 0.7152 * lin(m[2]) + 0.0722 * lin(m[3]);
  // Whichever of white and the page's ink has more contrast on it.
  return (L + 0.05) / (0.0137 + 0.05) > 1.05 / (L + 0.05) ? "#0f1c26" : "#ffffff";
}

export const NOTE_SIDES = ["right", "left", "top", "bottom"];

/** The box a note's card takes on `side` of its point, `gap` px from it. */
export function noteRect(pt, size, side, gap = 14) {
  const { w, h } = size;
  if (side === "left") return { x: pt.x - gap - w, y: pt.y - h / 2, w, h };
  if (side === "top") return { x: pt.x - w / 2, y: pt.y - gap - h, w, h };
  if (side === "bottom") return { x: pt.x - w / 2, y: pt.y + gap, w, h };
  return { x: pt.x + gap, y: pt.y - h / 2, w, h };
}

const overlap = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
  Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));

/**
 * Which side of its point each note's card goes on, so the cards cover neither each other, nor another note's
 * point, nor the map's own furniture (`avoid`: the legend, the time bar, the box), and stay inside `frame`
 * ({w, h}). `points` are [{x, y}] on screen, `sizes` [{w, h}] in the same order; earlier notes choose first.
 * Pure, so node tests it. Returns one of NOTE_SIDES per note.
 */
export function placeNotes(points, sizes, frame, avoid = [], { gap = 14, dot = 12 } = {}) {
  const placed = [];
  const dots = points.map((p) => ({ x: p.x - dot, y: p.y - dot, w: 2 * dot, h: 2 * dot }));
  return points.map((pt, i) => {
    let best = NOTE_SIDES[0], bestScore = Infinity;
    for (const side of NOTE_SIDES) {
      const r = noteRect(pt, sizes[i], side, gap);
      const inside = Math.max(0, Math.min(r.x + r.w, frame.w) - Math.max(r.x, 0)) *
        Math.max(0, Math.min(r.y + r.h, frame.h) - Math.max(r.y, 0));
      let score = 3 * (r.w * r.h - inside);
      for (const o of placed) score += 2 * overlap(r, o);
      for (const [j, d] of dots.entries()) if (j !== i) score += 2 * overlap(r, d);
      for (const a of avoid) score += overlap(r, a);
      // Right reads first, then left: a small nudge, so ties keep the cards beside their points.
      score += NOTE_SIDES.indexOf(side) * 0.5;
      if (score < bestScore) { best = side; bestScore = score; }
    }
    placed.push(noteRect(pt, sizes[i], best, gap));
    return best;
  });
}

export class ActionLog {
  constructor({ max = MAX_ENTRIES, now = () => Date.now() } = {}) {
    this.entries = [];
    this.nextId = 1;
    this.max = max;
    this.now = now;
    this.listeners = new Set();
  }

  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  emit() { for (const fn of this.listeners) { try { fn(this); } catch (err) { console.error("action log listener", err); } } }

  /** The entries still in force, oldest first. */
  live() { return this.entries.filter((e) => !e.undone); }

  get(id) { return this.entries.find((e) => e.id === id) || null; }

  /** Reserve the next id (a pin's or area's slot is named by it before the entry exists). */
  peekId() { return this.nextId; }

  /**
   * Record an action that has been applied: `{ action, slot, before, after, label, by, detail, command }`.
   * Returns the entry. The oldest entries fall off past `max` (they can no longer be undone).
   */
  add({ action, slot, before = null, after = null, label = "", by = "rules", detail = "", command = "" }) {
    const id = this.nextId++;
    const entry = {
      id, action, slot: slot || slotOf(action, id), before, after, label, by, detail, command,
      at: this.now(), undone: false,
    };
    this.entries.push(entry);
    while (this.entries.length > this.max) this.entries.shift();
    this.emit();
    return entry;
  }

  /**
   * Undo one entry. Returns `{ entry, restore }`, where `restore` is `{ slot, value }` to put back on the map,
   * or null when a later entry owns the slot (its "before" is updated instead). Null for an unknown or
   * already undone id.
   */
  undo(id) {
    const entry = this.entries.find((e) => e.id === id && !e.undone);
    if (!entry) return null;
    entry.undone = true;
    const later = this.entries.find((e) => !e.undone && e.slot === entry.slot && e.id > entry.id);
    let restore = null;
    if (later) later.before = entry.before;
    else restore = { slot: entry.slot, value: entry.before };
    this.emit();
    return { entry, restore };
  }

  /** Undo every live entry: one restore per slot, the value from before the first entry touched it. */
  undoAll() {
    const first = new Map();
    for (const e of this.entries) {
      if (e.undone) continue;
      if (!first.has(e.slot)) first.set(e.slot, e.before);
      e.undone = true;
    }
    this.emit();
    return [...first].map(([slot, value]) => ({ slot, value }));
  }

  /** The pins in force, oldest first: { id, ...pin, by, at }. */
  pins() {
    return this.live().filter((e) => e.action && e.action.type === "add_pin")
      .map((e) => ({ id: e.id, ...e.after, by: e.by, at: e.at }));
  }
}

/** How long ago, in a few words, for a log row's tooltip. */
export function agoWords(at, now = Date.now()) {
  const s = Math.max(0, Math.round((now - at) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  return `${h} h ago`;
}
