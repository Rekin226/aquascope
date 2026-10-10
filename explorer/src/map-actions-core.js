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
    .map((f) => ({ label: String(f.label).slice(0, 60), value: f.value, ...(f.unit ? { unit: String(f.unit).slice(0, 20) } : {}) }))
    : [];
  return {
    lat: Math.round(lat * 1e5) / 1e5, lon: Math.round(lon * 1e5) / 1e5, title,
    text: p.text ? String(p.text).replace(/\s+/g, " ").trim().slice(0, 600) : "",
    facts,
    source: p.source ? String(p.source).slice(0, 200) : "",
  };
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
