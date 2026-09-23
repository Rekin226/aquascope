// Compare: two to five of My places on one axis. aquascope.compare does the
// work in the worker (fetch, align, normalise, flow duration, GEV flood
// curves); this module only asks for the catchment areas the page already
// holds and draws what comes back.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, escapeHtml, fmt } from "./core.js?v=__BUILD__";
import { addTableDownload, plot } from "./charts.js?v=__BUILD__";
import { stationArea } from "./basins.js?v=__BUILD__";
import { renderMethodList } from "./methods.js?v=__BUILD__";
import { selectTab, setCard, setStatusEl, setTab } from "./shell.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";

export const MIN_COMPARE = 2;
export const MAX_COMPARE = 5;

// Five places at most, so five lines: distinct hues, and a dash per line so
// the overlay still reads in greyscale or with a colour-vision deficiency.
export const COMPARE_COLORS = ["#0b6bb8", "#d95f02", "#1b9e77", "#7570b3", "#c2185b"];
export const COMPARE_DASHES = ["solid", "dash", "dot", "dashdot", "longdash"];

const TABS = ["hydro", "fdc", "ffa"];
let run = 0;

const root = () => $("panel-places");
const say = (text, kind = "info") => setStatusEl($("places-status"), text, kind);

/** The request the worker gets: one entry per place, with the area when the page knows it. */
export function compareRequest(places, areas = new Map()) {
  return places.map((p) => {
    const key = `${p.source}/${p.station_id}`;
    const a = areas.get(key);
    return {
      source: p.source, station_id: p.station_id, label: p.name || p.station_id,
      period_start: p.period_start || null,
      area_km2: a && Number.isFinite(Number(a.area)) ? Number(a.area) : null,
    };
  });
}

function styleFor(i) {
  return { color: COMPARE_COLORS[i % COMPARE_COLORS.length], dash: COMPARE_DASHES[i % COMPARE_DASHES.length] };
}

export async function runCompare(places) {
  if (places.length < MIN_COMPARE || places.length > MAX_COMPARE) {
    say(`Pick ${MIN_COMPARE} to ${MAX_COMPARE} places to compare.`, "warn");
    return;
  }
  const my = ++run;
  const btn = $("btn-compare");
  btn.disabled = true;
  for (const t of TABS) setTab(root(), t, { enabled: false, reason: "Comparing…" });
  say("Looking up catchment areas…");
  try {
    const areas = new Map();
    for (const p of places) {
      const key = `${p.source}/${p.station_id}`;
      areas.set(key, await stationArea(key));
    }
    if (my !== run) return;
    say(`Fetching ${places.length} records and lining them up. Long records take a while.`);
    const res = await call("compare", { stations: compareRequest(places, areas), years: CONFIG.years });
    if (my !== run) return;
    if (res.error) { say(res.error, "error"); return; }
    say("");
    renderCompare(res);
  } catch (err) {
    if (my !== run) return;
    say(`Compare failed: ${err.message}`, "error");
  } finally {
    if (my === run) btn.disabled = false;
  }
}

function renderCompare(res) {
  const compared = res.places.filter((p) => p.compared);
  const style = new Map(compared.map((p, i) => [p.key, styleFor(i)]));
  const unit = res.unit || "";
  const slug = compared.map((p) => p.station_id).join("-").slice(0, 60) || "compare";

  $("cmp-basis").textContent = res.basis || "";
  renderNotes(res);
  renderMethodList("cmp-methods", res.methods || []);

  // Hydrographs
  const h = res.hydrograph;
  if (h && compared.length) {
    const traces = compared.map((p) => ({
      x: h.t, y: h.series[p.key], mode: "lines", name: p.label, connectgaps: false,
      line: { width: 1, color: style.get(p.key).color, dash: style.get(p.key).dash },
      hovertemplate: `${escapeHtml(p.label)}<br>%{x}<br>%{y:.3~f} ${unit}<extra></extra>`,
    }));
    setCard($("cmp-hydro-card"), "ready");
    plot("plot-cmp-hydro", traces, {
      height: 280, yaxis: { title: { text: unit }, rangemode: "tozero" }, legend: { orientation: "h", y: 1.18 },
    }, `compare-${slug}-hydrographs`);
    $("cmp-window").textContent = `${h.window.start} to ${h.window.end}` +
      (h.window.overlap ? " (shared window)" : " (the records do not overlap)") +
      (h.bin_days > 1 ? `, ${h.bin_days}-day means` : "");
    setTab(root(), "hydro", { enabled: true });
  } else {
    setCard($("cmp-hydro-card"), "empty", { message: "No record came back to compare." });
    setTab(root(), "hydro", { enabled: true });
  }
  renderTable(res, compared, unit, slug);

  // Flow duration
  const fdcKeys = compared.filter((p) => res.fdc && res.fdc[p.key]);
  if (fdcKeys.length) {
    plot("plot-cmp-fdc", fdcKeys.map((p) => ({
      x: res.fdc[p.key].exceedance, y: res.fdc[p.key].q, mode: "lines", name: p.label,
      line: { width: 2, color: style.get(p.key).color, dash: style.get(p.key).dash },
      hovertemplate: `${escapeHtml(p.label)}<br>%{x:.1f} % exceedance<br>%{y:.3~f} ${unit}<extra></extra>`,
    })), {
      height: 280, xaxis: { title: { text: "% of time exceeded" }, range: [0, 100] },
      yaxis: { title: { text: unit }, type: "log" }, legend: { orientation: "h", y: 1.18 },
    }, `compare-${slug}-flow-duration`);
    setCard($("cmp-fdc-card"), "ready");
    setTab(root(), "fdc", { enabled: true });
  } else {
    setTab(root(), "fdc", { enabled: false, reason: "Flow duration curves are drawn for discharge only." });
  }

  // Flood frequency: the fitted GEV line and the observed maxima, one colour per place.
  const ffaKeys = compared.filter((p) => res.ffa && res.ffa[p.key] && res.ffa[p.key].q);
  if (ffaKeys.length) {
    const traces = [];
    for (const p of ffaKeys) {
      const f = res.ffa[p.key], s = style.get(p.key);
      traces.push({
        x: f.return_periods, y: f.q, mode: "lines", name: `${p.label} (GEV)`, legendgroup: p.key,
        line: { width: 2, color: s.color, dash: s.dash },
        hovertemplate: `${escapeHtml(p.label)}<br>T = %{x} yr<br>%{y:.3~f} ${unit}<extra></extra>`,
      });
      traces.push({
        x: f.empirical.T, y: f.empirical.q, mode: "markers", name: `${p.label} (annual maxima)`,
        legendgroup: p.key, showlegend: false, marker: { color: s.color, size: 5, opacity: 0.7 },
        hovertemplate: `${escapeHtml(p.label)}<br>observed, T ≈ %{x:.1f} yr<br>%{y:.3~f} ${unit}<extra></extra>`,
      });
    }
    plot("plot-cmp-ffa", traces, {
      height: 280, xaxis: { title: { text: "return period (years)" }, type: "log" },
      yaxis: { title: { text: unit } }, legend: { orientation: "h", y: 1.18 },
    }, `compare-${slug}-flood-frequency`);
    setCard($("cmp-ffa-card"), "ready");
    setTab(root(), "ffa", { enabled: true });
  } else {
    setTab(root(), "ffa", {
      enabled: false, reason: "Flood curves need ten or more complete years of daily discharge.",
    });
  }
  selectTab(root(), "hydro");
}

function renderTable(res, compared, unit, slug) {
  const head = `<tr><th>Place</th><th>Record</th><th>Area (km²)</th><th>Mean</th><th>Q95</th><th>Q10</th>` +
    `<th>100-yr flood</th></tr>`;
  const rows = res.places.map((p) => {
    const s = p.compared ? styleFor(compared.indexOf(p)) : null;
    const swatch = s ? `<span class="cmp-swatch" style="background:${s.color}"></span>` : "";
    if (!p.compared) {
      return `<tr class="muted"><td>${escapeHtml(p.label)}</td><td colspan="6">Left out: ${escapeHtml(p.error || "not comparable")}</td></tr>`;
    }
    return `<tr><td>${swatch}${escapeHtml(p.label)}</td><td>${escapeHtml(String(p.start).slice(0, 4))}–` +
      `${escapeHtml(String(p.end).slice(0, 4))}</td><td>${p.area_km2 ? fmt(p.area_km2) : "—"}</td>` +
      `<td>${fmt(p.mean)}</td><td>${fmt(p.q95)}</td><td>${fmt(p.q10)}</td><td>${fmt(p.q100)}</td></tr>`;
  }).join("");
  const table = $("cmp-table");
  table.innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>` +
    `<tfoot><tr><td colspan="7" class="muted">Values in ${escapeHtml(unit)}. Q95 is the flow exceeded 95 % of the time.</td></tr></tfoot>`;
  addTableDownload($("cmp-actions"), table, `compare-${slug}.csv`);
  setCard($("cmp-summary-card"), "ready");
}

function renderNotes(res) {
  const notes = res.notes || [];
  $("cmp-notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
  $("cmp-notes").hidden = !notes.length;
}
