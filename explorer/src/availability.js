// Presentation of archive metadata. A catalog span is never an observation span.
const ARCHIVE = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/";
let capabilities = null;
let manifestRequest;

export async function loadAvailability() {
  try {
    const response = await fetch(new URL("../source-capabilities.json", import.meta.url));
    if (response.ok) capabilities = await response.json();
  } catch { /* unknown capability is not a claim that records are available */ }
}

export function catalogOnly(source) { return Array.isArray(capabilities?.record_sources) && !capabilities.record_sources.includes(source); }

export function availabilityLabel(source) {
  return catalogOnly(source) ? "Catalog only · use agency link" : "Observation coverage checked on opening";
}

export async function observationMetadata(source, station, variables = []) {
  if (!manifestRequest) {
    manifestRequest = fetch(`${ARCHIVE}obs/manifest.json`, { credentials: "omit" })
      .then((r) => { if (!r.ok) throw new Error("unavailable"); return r.json(); })
      .catch(() => null);
  }
  const manifest = await manifestRequest;
  if (!manifest) return "Archive observation refresh metadata is unavailable.";
  const records = variables.map((v) => [v, manifest.sources?.[`${source}/${v}`]?.stations?.[station]])
    .filter(([, entry]) => entry?.n > 0);
  if (!records.length) return "No mirrored observations listed. Any live agency record is checked separately.";
  return records.map(([variable, entry]) => `${variable}: archive ${entry.first}–${entry.last}; ` +
    `last successful observation update ${entry.harvested_at || "unknown"}` +
    (entry.last_attempt_status && entry.last_attempt_status !== "ok" ? `; latest refresh ${entry.last_attempt_status}, previous data retained` : ""))
    .join(". ");
}
