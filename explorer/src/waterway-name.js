// Find a named waterway near a snapped reach. The response includes line geometry so that
// distance is measured to the way, not to its first node or the middle of its bounding box.
const API = "https://overpass-api.de/api/interpreter";
const RADIUS_M = 500;
const TIMEOUT_MS = 4000;
const TYPES = new Map([["river", 0], ["canal", 1], ["stream", 2]]);

function segmentDistance(a, b, lat, lon) {
  const scaleX = 111320 * Math.cos(lat * Math.PI / 180), scaleY = 111320;
  const project = (p) => ({ x: ((((p.lon - lon) % 360) + 540) % 360 - 180) * scaleX, y: (p.lat - lat) * scaleY });
  const p = project(a), q = project(b), dx = q.x - p.x, dy = q.y - p.y;
  const length = dx * dx + dy * dy;
  const t = length ? Math.max(0, Math.min(1, -(p.x * dx + p.y * dy) / length)) : 0;
  return Math.hypot(p.x + t * dx, p.y + t * dy);
}

export function nearestWaterway(elements, lat, lon) {
  let best = null;
  for (const way of Array.isArray(elements) ? elements : []) {
    const tags = way?.tags || {}, rank = TYPES.get(tags.waterway);
    const name = (typeof tags["name:en"] === "string" && tags["name:en"].trim()) ||
      (typeof tags.name === "string" && tags.name.trim()) || "";
    if (way?.type !== "way" || rank === undefined || !name || !Array.isArray(way.geometry)) continue;
    const points = way.geometry;
    if (points.length < 2) continue;
    let distance = Infinity;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      if ([a?.lat, a?.lon, b?.lat, b?.lon].every(Number.isFinite)) distance = Math.min(distance, segmentDistance(a, b, lat, lon));
    }
    if (distance > RADIUS_M) continue;
    if (!best || rank < best.rank || (rank === best.rank && distance < best.distance)) best = { name, rank, distance };
  }
  return best?.name || null;
}

export async function lookupWaterway(lat, lon, { timeout = TIMEOUT_MS, fetcher = fetch } = {}) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  const query = `[out:json][timeout:3];way(around:${RADIUS_M},${lat},${lon})["waterway"~"^(river|canal|stream)$"]["name"];out geom;`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetcher(`${API}?${new URLSearchParams({ data: query })}`, { signal: controller.signal });
    if (!response.ok) return null;
    const data = await response.json();
    return nearestWaterway(data.elements, lat, lon);
  } catch { return null; }
  finally { clearTimeout(timer); }
}

// Reopening the same reach should reuse one lookup, including while it is in flight.
// Do not retain failures: a later attempt can recover when the service is available.
export function createWaterwayLookup({ request = lookupWaterway, now = Date.now, ttl = 300000, limit = 100 } = {}) {
  const cache = new Map();
  return (lat, lon) => {
    const key = `${lat},${lon}`;
    const previous = cache.get(key);
    if (previous && (previous.pending || previous.expires > now())) {
      cache.delete(key);
      cache.set(key, previous);
      return previous.promise;
    }
    cache.delete(key);
    const entry = { pending: true, expires: 0, promise: null };
    entry.promise = Promise.resolve().then(() => request(lat, lon)).catch(() => null).then((name) => {
      if (cache.get(key) === entry) {
        if (!name) cache.delete(key);
        else { entry.pending = false; entry.expires = now() + ttl; }
      }
      return name;
    });
    cache.set(key, entry);
    while (cache.size > limit) cache.delete(cache.keys().next().value);
    return entry.promise;
  };
}
