// The same claim identity exported in reports, inspectable beside the browser result.
const esc = value => String(value ?? "not recorded").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function claimEvidenceHtml(evidence) {
  if (!evidence?.result_id) return "";
  const d = evidence.dataset || {};
  const interval = evidence.interval;
  const own = interval?.result_id === evidence.result_id && Array.isArray(interval.bounds);
  const rows = [
    ["Record", `${d.source || "source not recorded"} / ${d.station_id || "station not recorded"}`],
    ["Actual observations", `${d.start || "unknown"} to ${d.end || "unknown"}; ${d.observations ?? "unknown"} values`],
    ["Variable / unit", `${d.variable || "unknown"} / ${d.unit || "unknown"}`],
    ["Aggregation", evidence.aggregation], ["Estimator", evidence.estimator],
    ["Uncertainty for this estimate", own
      ? `${interval.bounds.join(" to ")} ${d.unit || ""}; ${interval.method || "method not recorded"}; confidence level ${interval.level ?? "not recorded"}`
      : "No interval recorded for this estimator."],
    ["Input fingerprint", d.snapshot], ["Software", `${d.software_version || "unknown"}; revision ${d.software_revision || "not recorded"}`],
    ["Archive revision", d.archive_revision],
  ];
  return `<details class="study-findings"><summary>Input and method for this result</summary><dl>` +
    rows.map(([label, value]) => `<dt>${esc(label)}</dt><dd style="overflow-wrap:anywhere">${esc(value)}</dd>`).join("") +
    `</dl></details>`;
}
