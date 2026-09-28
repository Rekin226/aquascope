// Presentation only: retain original report text in the workspace and exports.
// Unknown check names stay visible so a new engine check is never hidden.
const labels = {
  min_years: "record length", max_return_period_factor: "extrapolation limit",
  ci_finite: "finite confidence intervals", spread_within: "agreement between estimates",
  nse_min: "Nash-Sutcliffe efficiency", kge_min: "Kling-Gupta efficiency",
  not_empty: "available data", unit_present: "documented units", max_area_km2: "catchment size",
  min_donors: "number of reference catchments", status_is: "required status",
  min_samples: "sample count", fit_envelopes_max: "fit to the observed maximum",
  sampling_density: "sampling frequency", trend_on_series: "trend in the analyzed series",
  cross_check_ratio: "comparison ratio",
};

export function studyText(value) {
  return String(value)
    .replace(/\bgate ([a-z_]+)\b/g, (text, name) => labels[name] ? `${labels[name]} check` : text)
    .replace(/\bskipped: skipped:/g, "skipped:")
    .replace(/\bthe playbook's plan\b/g, "the built-in analysis plan")
    .replace(/\bplaybooks\b/g, "analysis workflows")
    .replace(/\bgates passed\b/g, "checks passed")
    .replace(/\bno model\b/g, "no AI model used")
    .replace(/\bthe validator\b/g, "the plan checks");
}

const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function studyWarningHtml(value) {
  const original = String(value), readable = studyText(original);
  return esc(readable) + (readable === original ? "" :
    `<details class="step-raw"><summary>Technical details</summary>${esc(original)}</details>`);
}
