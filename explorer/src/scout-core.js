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

/** Who found and worded the pins, for the line under the bar: [who, how]. */
export function scoutWho(by, model = "") {
  if (by === "device") return [model || "On-device model", "worded on this device; numbers by code"];
  if (by === "key") return [model || "Your model", "worded with your key; numbers by code"];
  return ["Rules", "no model"];
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
