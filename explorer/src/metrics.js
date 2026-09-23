// Optional local pilot counters. No network, identifier, URL, question, location,
// file name, record, key or error text enters this store. Sharing is a file export.
const KEY = "aquascope.pilot-metrics.v1";
const EVENTS = new Set(["visit", "usable_record", "table_loaded", "study_ready", "export_handoff", "operation_ok", "operation_error"]);
const KINDS = new Set(["station", "table", "study", "csv", "json", "document", "other"]);
const USEFUL = new Set(["usable_record", "table_loaded", "study_ready", "export_handoff"]);

export function createMetrics(storage, now = () => new Date()) {
  let data = null;
  try {
    const saved = JSON.parse(storage?.getItem(KEY) || "null");
    if (saved?.version === 1 && saved.enabled === true && Array.isArray(saved.days)) data = saved;
  } catch { /* denied or unavailable storage means off */ }
  const persist = () => { try { storage?.setItem(KEY, JSON.stringify(data)); } catch { /* session only */ } };
  const fresh = () => ({ version: 1, enabled: true, days: [] });
  function record(event, { kind = "other", durationMs, runtime } = {}) {
    if (!data || !EVENTS.has(event)) return;
    const today = now().toISOString().slice(0, 10);
    const cutoff = new Date(now().getTime() - 27 * 86400000).toISOString().slice(0, 10);
    data.days = data.days.filter(d => d.day >= cutoff && d.day <= today);
    let day = data.days.find(d => d.day === today);
    if (!day) { day = { day: today, counts: {}, durations: {}, useful: false }; data.days.push(day); }
    const bucket = `${event}:${KINDS.has(kind) ? kind : "other"}`;
    day.counts[bucket] = Math.min(1000000, (day.counts[bucket] || 0) + 1);
    if (USEFUL.has(event)) day.useful = true;
    if (Number.isFinite(durationMs) && durationMs >= 0 && ["cold", "warm"].includes(runtime)) {
      const key = `${bucket}:${runtime}`;
      const samples = day.durations[key] || [];
      samples.push(Math.min(3600000, Math.round(durationMs)));
      day.durations[key] = samples.slice(-100);
    }
    persist();
  }
  return {
    record,
    enabled: () => Boolean(data),
    enable() { data = fresh(); persist(); record("visit"); },
    clear() { data = null; try { storage?.removeItem(KEY); } catch { /* unavailable */ } },
    snapshot() {
      if (!data) return { version: 1, enabled: false, days: [] };
      const out = JSON.parse(JSON.stringify(data));
      out.useful_days = out.days.filter(d => d.useful).length;
      out.disclosure = "Opt-in local counters, last 28 calendar days; no visitor identifier. Export handoff is not confirmation of a saved or used file. Cold/warm describes the Python worker, not the HTTP cache.";
      return out;
    },
  };
}
let storage;
try { storage = globalThis.localStorage; } catch { /* privacy mode */ }
export const metrics = createMetrics(storage);
