// Scout (#563), the pure part: no DOM, no map, so node can test it (explorer/tests/scout-core.test.mjs).
//
// The findings themselves are aquascope.map_scout's, made in the worker (or read from the daily file the
// flood-warnings workflow publishes under scout/): every number in them is formatted by the package. This
// module only says which part of the globe is on screen, turns a finding into the pin map-actions.js draws,
// and says in a few words who found them.

export const SCOUT_MAX = 10;
// How many more candidates a model may choose among, past the rules' ten.
export const SCOUT_EXTRA = 6;

// The kinds as aquascope.map_scout.KINDS labels them (a test keeps the two lists equal).
export const KIND_LABELS = {
  floods_ahead: "Floods ahead",
  status: "World river status",
  gauges_today: "Gauges today",
  floods_past: "Floods past",
  models_disagree: "Models and gauges",
};

const EARTH_M = 40075016.686;
// The globe's far side starts 10,007 km from the centre; past about 7,500 km a pin sits on the limb.
export const GLOBE_RADIUS_KM = 7500;

/**
 * The part of the globe on screen, as aquascope.map_scout.check_view takes it. On the globe zoomed out it is
 * a circle around the centre (a box would hold the far side too); otherwise the outline of what is on screen
 * (`outline`, [lon, lat] points around the frame: on a curved map that is not a box), else the map's bounds, or
 * null when they span the whole world. `center` is [lon, lat] (MapLibre's order), `bounds` [w, s, e, n].
 */
export function scoutView(opts = {}) {
  const v = viewShape(opts);
  if (!v) return v;
  // Two pins closer than about a pin's height on screen cannot be told apart: the package keeps them apart.
  const { zoom = 2, center = [0, 0] } = opts;
  const mpp = (EARTH_M * Math.cos((center[1] * Math.PI) / 180)) / (512 * 2 ** zoom);
  return { ...v, min_km: Math.round(Math.min(5000, Math.max(1, (mpp * PIN_GAP_PX) / 1000))) };
}

// About a numbered pin's height, in pixels.
export const PIN_GAP_PX = 34;

function viewShape({ globe = false, zoom = 2, center = [0, 0], bounds = null, outline = null, width = 1200,
  height = 800 } = {}) {
  const [lon, lat] = center;
  if (globe && zoom < 4) {
    const mpp = (EARTH_M * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** zoom);
    const halfDiagKm = (mpp * Math.hypot(width, height)) / 2 / 1000;
    return { center: [round(lat, 3), round(wrap(lon), 3)], radius_km: Math.round(Math.min(GLOBE_RADIUS_KM, Math.max(50, halfDiagKm))) };
  }
  if (outline && outline.length >= 3 && outline.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
    const lons = outline.map((p) => p[0]);
    if (Math.max(...lons) - Math.min(...lons) >= 359) return null;
    // Longitudes stay continuous (past 180 across the antimeridian); the package shifts points to match.
    return { polygon: outline.map(([x, y]) => [round(x, 3), round(Math.max(-90, Math.min(90, y)), 3)]) };
  }
  if (!bounds) return null;
  let [w, s, e, n] = bounds;
  if (e - w >= 360) return null;
  s = Math.max(-90, s);
  n = Math.min(90, n);
  return { bbox: [round(wrap(w), 3), round(s, 3), round(wrap(e), 3), round(n, 3)] };
}

const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;
function wrap(lon) {
  let x = ((((lon + 180) % 360) + 360) % 360) - 180;
  if (x === -180 && lon > 0) x = 180;
  return x;
}

/**
 * The findings a reader can see: the package ranks the part of the globe on screen (a circle on the globe zoomed
 * out, whose corners reach under the bars), so the page keeps those whose pin stands clear of the bars and the
 * time bar (`visible(f)`), in the package's order, renumbered: `max` picks, then `extra` more for a model.
 */
export function keepVisible(res, visible, max = SCOUT_MAX, extra = 0) {
  const all = (res && (res.candidates || res.picks)) || [];
  const seen = all.filter((f) => {
    try { return Boolean(visible(f)); } catch { return true; }
  });
  const cands = seen.slice(0, max + Math.max(0, extra)).map((f, i) => ({ ...f, rank: i + 1 }));
  return { ...res, candidates: cands, picks: cands.slice(0, max), hidden: all.length - seen.length };
}

/** The map date's month ("2026-09"), the month the status and Floods past findings are for. */
export function scoutMonth(date) {
  const m = /^(\d{4})-(\d{2})/.exec(String(date || ""));
  return m ? `${m[1]}-${m[2]}` : null;
}

/** The pin map-actions.js's addPin takes, from one finding: its reason, its facts and where they came from. */
export function scoutPin(f, { mode = "live", made = "" } = {}) {
  const when = mode === "daily" ? `scout file of ${String(made).slice(0, 10)}` : "scanned in your browser";
  const place = f.placed_by === "photon" ? "; place name: Photon, OpenStreetMap (ODbL)" : "";
  return {
    lat: f.lat, lon: f.lon, title: f.title, text: f.reason, facts: (f.facts || []).slice(0, 5),
    source: `${f.source} (${when}${place})`, rank: f.rank, kind: KIND_LABELS[f.kind] || f.kind,
  };
}

/** Who found and ordered the pins, for the line under the bar: [who, how]. The words are always the package's. */
export function scoutWho(by, model = "") {
  if (by === "device") return [model || "On-device model", "ordered on this device; words by code"];
  if (by === "key") return [model || "Your model", "ordered with your key; words by code"];
  return ["Rules", "no model"];
}

/**
 * Why a model that was asked gave nothing (#590): a timeout says it timed out, and after how long. `err` is what
 * generateJsonLocally threw (its own timeout, or the Prompt API's AbortError when the timeout aborts it first).
 */
export function askedFailed(err, waitMs = 40000) {
  const msg = String((err && err.message) || err || "");
  if ((err && err.name === "AbortError") || /abort|took more than|timed? ?out/i.test(msg)) {
    return `it was asked and timed out after ${Math.round(waitMs / 1000)} s; the rules' order and wording are kept`;
  }
  return `it was asked and gave no answer (${msg || "no reply"}); the rules' order and wording are kept`;
}

/**
 * One short note on refused picks (#590): the whole reply, or how many, and the first one's reason. `refused` is the package's
 * list ("<id>: <why>"); an id may hold a colon ("models_disagree:x/1") but never a colon and a space.
 */
export function refusedNote(refused = []) {
  if (!refused || !refused.length) return "";
  const first = String(refused[0]);
  // The whole reply refused by the package's schema check (#590, round 3): one line, the rules stand.
  if (refused.length === 1 && /^the model's reply was refused/.test(first)) return `${first}; rules kept`;
  const why = first.includes(": ") ? first.slice(first.indexOf(": ") + 2) : first;
  const more = refused.length > 1 ? ` (and ${refused.length - 1} more)` : "";
  return `${refused.length} pick${refused.length === 1 ? "" : "s"} not used, rules kept: ${why}${more}`;
}

/**
 * Who ordered the pins, and the note under the line (#590): a model that was asked but whose choice is on no
 * pin is not said to have ordered them. `by` and `model` are what the model path returned, `picks` the pins
 * (each with its own `by`), `refused` and `unknown` aquascope.map_scout.apply_wording's, `failed` why the
 * model could not be asked. Returns { who, how, notes }.
 */
export function wordingLine({ by = "rules", model = "", picks = [], refused = [], unknown = [], failed = "" } = {}) {
  const asked = by === "rules" ? "" : (model || (by === "device" ? "the on-device model" : "your model"));
  const used = asked ? picks.filter((f) => f.by === by).length : 0;
  const notes = [];
  if (failed) notes.push(failed);
  const r = refusedNote(refused);
  if (r) notes.push(r);
  if (!asked) return { who: "Rules", how: failed ? "the model could not be asked" : "no model", notes };
  if (!used) {
    if (!refused.length && !failed) {
      notes.push(unknown.length
        ? `the model wrote about ${unknown.join(", ")}, not a pin in this view; rules kept`
        : "the model picked none of these pins; rules kept");
    }
    return { who: "Rules", how: failed ? `${asked} asked, no answer` : `${asked} asked, its choice not used`, notes };
  }
  const [who, how] = scoutWho(by, model);
  return { who, how: used < picks.length ? `${used} of ${picks.length} ${how}` : how, notes };
}

/** The line itself: how many pins, and from what. */
export function scoutLine(res, n) {
  if (!n) {
    const why = (res && res.notes && res.notes.length) ? ` (${res.notes.join("; ")})` : "";
    return `Nothing stands out in this view${why}. Zoom out or move the map, then scout again.`;
  }
  const from = res && res.mode === "daily" ? "from today's scout file" : "scanned now";
  return `${n} pin${n === 1 ? "" : "s"}, most worth a look first, ${from}`;
}
