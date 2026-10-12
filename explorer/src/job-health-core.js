// Job health (#593), the pure part: no DOM, so node can test it.
//
// The scheduled jobs publish health/jobs/index.json (aquascope/archive/health_jobs.py): per job its last run,
// last success, last failure and the age after which its data is older than expected (stale_after_hours, from
// the package's one cadence table). This file reads it the way health_jobs.assess() does: the Explorer is
// given the limit, it only compares an age with it.

export const HEALTH_INDEX_URL = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/health/jobs/index.json";

const HOUR = 3600e3;

function parse(t) {
  const ms = Date.parse(String(t || ""));
  return Number.isFinite(ms) ? ms : null;
}

/**
 * How a job reads at `now` (ms): { state, ageHours }. state is "ok", "stale" (the last success is older than
 * stale_after_hours, or there never was one), "failing" (the last run failed, the data still on time),
 * "on demand" (no cadence) or "unknown" (no run recorded).
 */
export function jobState(entry, now = Date.now()) {
  const e = entry || {};
  const okAt = parse(e.last_success && e.last_success.finished_at);
  const last = e.last_run || null;
  const ageHours = okAt === null ? null : Math.round(((now - okAt) / HOUR) * 10) / 10;
  const limit = e.stale_after_hours;
  let state;
  if (!last && okAt === null) state = "unknown";
  else if (limit !== null && limit !== undefined && (ageHours === null || ageHours > Number(limit))) state = "stale";
  else if (last && last.status === "failure") state = "failing";
  else if (limit === null || limit === undefined) state = "on demand";
  else state = "ok";
  return { state, ageHours };
}

/** An age in words: "40 min", "9 h", "3 d". */
export function ageWords(hours) {
  if (hours === null || hours === undefined || !Number.isFinite(Number(hours))) return "";
  const h = Math.max(0, Number(hours));
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min`;
  if (h < 48) return `${Math.round(h)} h`;
  return `${Math.round(h / 24)} d`;
}

function jobsOf(index) {
  return index && Array.isArray(index.jobs) ? index.jobs.filter((e) => e && e.job) : [];
}

/**
 * The legend's notes: { rowId: { text, title } } for each map layer whose job's data is older than expected.
 * The row is narrow, so the text is the age ("3 d old"); the title says the rest ("Older than expected: the
 * Forecast archive last updated it 3 d ago; it runs daily."). A row fed by a job that is on time gets nothing.
 */
export function legendNotes(index, now = Date.now()) {
  const notes = {};
  for (const e of jobsOf(index)) {
    const { state, ageHours } = jobState(e, now);
    if (state !== "stale") continue;
    const age = ageWords(ageHours);
    const text = age ? `${age} old` : "not updated";
    const when = age ? `last updated it ${age} ago` : "has no successful run recorded";
    const title = `Older than expected: the ${e.label || e.job} ${when}; it runs ${e.cadence || "on a schedule"}.`;
    for (const row of e.layers || []) notes[row] = notes[row] ? { text, title: `${notes[row].title} ${title}` } : { text, title };
  }
  return notes;
}

const STATE_WORDS = { ok: "on time", stale: "late", failing: "last run failed", "on demand": "when dispatched",
  unknown: "no record yet" };

/** One item per job for Tools > Data status: { job, label, state, text, title }. */
export function statusItems(index, now = Date.now()) {
  return jobsOf(index).map((e) => {
    const { state, ageHours } = jobState(e, now);
    const age = ageWords(ageHours);
    const words = STATE_WORDS[state] || state;
    const text = age ? `${words}, ${age} ago` : words;
    const data = e.last_success && e.last_success.data_date ? ` Data of ${e.last_success.data_date}.` : "";
    const failed = e.last_failure && e.last_failure.finished_at
      ? ` Last failure ${String(e.last_failure.finished_at).slice(0, 10)}.` : "";
    const title = `${e.label || e.job}: ${e.what || ""} Runs ${e.cadence || "when dispatched"}.${data}${failed}`;
    return { job: e.job, label: e.label || e.job, state, text, title: title.replace(/\s+/g, " ").trim() };
  });
}

/** The line's head: "Data status: all 11 on time" or "Data status: 2 of 11 late or failing". */
export function statusHeadline(items) {
  const list = (items || []).filter((i) => i.state !== "on demand" && i.state !== "unknown");
  const bad = list.filter((i) => i.state === "stale" || i.state === "failing").length;
  if (!list.length) return "Data status: no record yet";
  return bad ? `Data status: ${bad} of ${list.length} late or failing` : `Data status: all ${list.length} on time`;
}
