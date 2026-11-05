// A name for a point that came without one: a pasted link (#p=24.35,89.8&tab=river) or a click on the
// map (#591). The nearest named water in OpenStreetMap within 5 km, else the nearest town, from Photon's
// reverse lookup; the rules are place-search-core.js's. A point found by name in the search keeps that
// name instead, and nothing is asked.

import { photonReverseUrl, pointLabel, riverLabel } from "./place-search-core.js?v=__BUILD__";

const cache = new Map();
const WATER_KM = 5;
// Room for the river behind a lake or a fountain nearer the point (#637): the first is the label, the river
// among them names the River card and the routing panel.
const WATER_LIMIT = 5;
const TOWN_KM = 10;

async function features(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.features || [];
}

/** `{ name, detail, water, river }` for a point, or null (nothing named near, or Photon unreachable). */
export async function namePoint(lat, lon) {
  const key = `${lat},${lon}`;
  if (cache.has(key)) return cache.get(key);
  const ask = (async () => {
    try {
      const water = await features(photonReverseUrl(lat, lon, { water: true, radiusKm: WATER_KM, limit: WATER_LIMIT }));
      const river = riverLabel(water);
      const label = pointLabel(water, []);
      if (label) return { ...label, river };
      const any = pointLabel([], await features(photonReverseUrl(lat, lon, { radiusKm: TOWN_KM })));
      return any ? { ...any, river } : null;
    } catch (err) {
      console.info("place name unavailable:", err.message);
      cache.delete(key);   // a later look may reach it
      return null;
    }
  })();
  cache.set(key, ask);
  return ask;
}
