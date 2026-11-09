// Places by name in the search box, and a name for a point (#591). Pure, for node:test; search.js and
// place-name.js do the fetching.
//
// The gazetteer is Photon (komoot), the one the package's map commands use (aquascope/map_commands.py,
// resolve_place): keyless, CORS open, OpenStreetMap data. It ranks a river below a village of the same
// name ("Jamuna" gave eight districts and villages before the river), so the search asks twice, once for
// anything and once for water only, and ranks the two lists together here: a river or a lake whose name is
// the one asked comes first, the whole river (an OSM relation, with the widest extent) before one of its
// stretches. The GEOGLOWS reaches the River tab reads have no names, so a river is found by its OSM name and
// the map snaps the place to the reach underneath (river.js).
//
// This runs per keystroke, before Python has loaded in the browser, so it stays here rather than in the
// worker; the rules follow resolve_place's (the same noise words, the same extent order).

export const PHOTON_API = "https://photon.komoot.io/api/";
export const PHOTON_REVERSE = "https://photon.komoot.io/reverse";
// The same words as aquascope.map_commands.PLACE_CREDIT and map_explain.PLACE_CREDIT.
export const PLACE_CREDIT = "Places: Photon by komoot, data © OpenStreetMap contributors (ODbL)";

// Water in OSM: rivers and canals are waterways, lakes and reservoirs are natural=water.
export const WATER_TAGS = ["waterway:river", "waterway:canal", "natural:water"];
const RIVERISH = new Set(["river", "canal", "stream"]);

// Words a river's name carries that a reader does not type ("Jamuna" for "Jamuna River").
const NOISE = new Set("the river rio riviere fleuve fluss lake of de la le du des".split(" "));

export function fold(s) {
  return String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

const words = (s) => (fold(s).match(/[\p{L}\p{N}]+/gu) || []);
const core = (s) => words(s).filter((w) => !NOISE.has(w)).join(" ");

/** The search URL: anything (`water` false) or water only (`water` true). */
export function photonSearchUrl(query, { water = false, limit = 6, lang = "en" } = {}) {
  const q = new URLSearchParams({ q: String(query || "").trim(), limit: String(limit), lang });
  if (water) for (const t of WATER_TAGS) q.append("osm_tag", t);
  return `${PHOTON_API}?${q.toString()}`;
}

/** The reverse URL: the nearest water within `radiusKm` (`water` true), or the nearest anything. */
export function photonReverseUrl(lat, lon, { water = false, radiusKm = 5, limit = 1, lang = "en" } = {}) {
  const q = new URLSearchParams({ lat: String(Math.round(lat * 1e4) / 1e4), lon: String(Math.round(lon * 1e4) / 1e4),
    radius: String(radiusKm), limit: String(limit), lang });
  if (water) for (const t of WATER_TAGS) q.append("osm_tag", t);
  return `${PHOTON_REVERSE}?${q.toString()}`;
}

/** Is a Photon feature's properties water: a river, canal or stream, or a lake. */
export function isWater(p) {
  if (!p) return false;
  if (p.osm_key === "waterway") return true;
  return p.osm_key === "natural" && p.osm_value === "water";
}

const isRiver = (p) => Boolean(p) && p.osm_key === "waterway" && RIVERISH.has(p.osm_value);

// Photon's extent is [minLon, maxLat, maxLon, minLat]; its size in degrees, or 0.
function span(ext) {
  if (!Array.isArray(ext) || ext.length !== 4 || !ext.every(Number.isFinite)) return 0;
  return Math.abs(ext[2] - ext[0]) + Math.abs(ext[1] - ext[3]);
}

/** One Photon feature as a search result, or null when it has no name or no point. */
export function normalizePlace(f) {
  const p = (f && f.properties) || {};
  const c = (f && f.geometry && f.geometry.coordinates) || [];
  const lon = Number(c[0]), lat = Number(c[1]);
  if (!p.name || !Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const detail = [p.city, p.state, p.country].filter((x) => x && x !== p.name);
  return {
    name: String(p.name),
    detail: [...new Set(detail)].join(", "),
    lat, lon,
    kind: p.osm_value || p.type || "place",
    water: isWater(p),
    river: isRiver(p),
    whole: p.osm_type === "R",
    span: span(p.extent),
    country: p.country || "",
  };
}

/** How well a place answers `query`; higher is better. */
export function placeScore(place, query) {
  const q = fold(query).trim();
  const n = fold(place.name);
  const qc = core(query), nc = core(place.name);
  let s = 0;
  if (n === q || (qc && nc === qc)) s += 100;                       // the name, give or take "River"
  else if (n.startsWith(q) || (qc && nc.startsWith(qc))) s += 40;
  else if (n.includes(q) || (qc && nc.includes(qc))) s += 20;
  if (place.river) s += 70;                                          // the reason someone types "Jamuna" here
  else if (place.water) s += 50;
  if (place.water && place.whole) s += 15;                           // the whole river, not one stretch of it
  if (place.water) s += Math.min(20, 10 * Math.log10(1 + 10 * place.span));   // the Rhine before a brook called Rhine
  if (!place.water) {
    if (["city", "town"].includes(place.kind)) s += 30;
    else if (["village", "state", "country", "county", "region"].includes(place.kind)) s += 10;
  }
  return s;
}

/**
 * Photon's answers to the general and the water-only search, merged, ranked and cut to `limit`. One result
 * per name and country (a river comes back as many stretches): the best-scored one stays.
 */
export function rankPlaces(features, query, limit = 5) {
  const best = new Map();
  for (const f of features || []) {
    const p = normalizePlace(f);
    if (!p) continue;
    p.score = placeScore(p, query);
    const key = `${fold(p.name)}|${fold(p.country)}|${p.water ? "w" : "l"}`;
    const had = best.get(key);
    if (!had || p.score > had.score) best.set(key, p);
  }
  return [...best.values()].sort((a, b) => b.score - a.score || b.span - a.span).slice(0, Math.max(0, limit));
}

// A name a reader can use: not one with a digit (a postcode, a road number), as place_name does in the package.
const clean = (v) => (v && !/\d/.test(String(v)) ? String(v) : "");

/**
 * A name for a point that came without one (a link, a click): the nearest named water Photon knows within
 * the reverse radius, else the nearest place. `{ name, detail, water }`, or null when there is neither.
 */
export function pointLabel(waterFeatures, anyFeatures) {
  const w = (waterFeatures || []).map((f) => (f && f.properties) || {}).find((p) => isWater(p) && clean(p.name));
  if (w) {
    const where = [clean(w.city) || clean(w.county), clean(w.state), clean(w.country)].filter(Boolean);
    return { name: clean(w.name), detail: [...new Set(where)].join(", "), water: true };
  }
  const a = (anyFeatures || []).map((f) => (f && f.properties) || {})[0];
  if (!a) return null;
  const town = clean(a.city) || clean(a.name) || clean(a.county) || clean(a.district);
  const region = [clean(a.state), clean(a.country)].filter((x) => x && x !== town);
  if (!town && !region.length) return null;
  return town
    ? { name: `Near ${town}`, detail: [...new Set(region)].join(", "), water: false }
    : { name: region[0], detail: region.slice(1).join(", "), water: false };
}

/**
 * The river at a point (#637): the nearest named waterway in the same reverse answer that is a river, a canal
 * or a stream, as `{ name, detail, water: true, river: true }`, or null. A lake, a pond or a fountain is not a
 * river, so a River card or a routing panel's title falls back to its reach instead of taking that name.
 */
export function riverLabel(waterFeatures) {
  const r = (waterFeatures || []).map((f) => (f && f.properties) || {}).find((p) => isRiver(p) && clean(p.name));
  if (!r) return null;
  const where = [clean(r.city) || clean(r.county), clean(r.state), clean(r.country)].filter(Boolean);
  return { name: clean(r.name), detail: [...new Set(where)].join(", "), water: true, river: true };
}

/** The river's name in a point's label (#637): "" when the label is a lake, a fountain or a town. */
export function riverName(label) {
  return (label && label.river && label.river.name) || "";
}

/**
 * Whether the places go above the gauges in the search list: the best place is named what was asked
 * ("Rhine" is the river Rhine) and no gauge's name has the asked words as words of its own (a gauge "near
 * Rhinelander" only contains them). A gauge on the river ("JAMUNA AT BAHADURABAD") keeps the gauges first.
 */
export function placesFirst(query, gaugeNames, best) {
  if (!best) return false;
  const q = core(query);
  if (!q || core(best.name) !== q) return false;
  const asked = q.split(" ");
  return !(gaugeNames || []).some((n) => {
    const w = new Set(words(n));
    return asked.every((x) => w.has(x));
  });
}
