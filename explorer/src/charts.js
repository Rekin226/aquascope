// Plotly defaults and the export helpers. Every figure keeps its PNG button
// (the mode bar used to be off entirely, so no chart on the page could be
// saved), and every table on the page can be downloaded as CSV.

import { downloadBlob, toCsv } from "./core.js?v=__BUILD__";

export const PLOT_LAYOUT = {
  margin: { l: 48, r: 12, t: 8, b: 36 },
  height: 240,
  font: { family: "system-ui, sans-serif", size: 11 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
};

export const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: "hover",
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"],
  toImageButtonOptions: { format: "png", scale: 2, filename: "aquascope-figure" },
};

// A figure that can name its own PNG download.
export function plot(id, traces, layout, filename) {
  const config = filename
    ? { ...PLOT_CONFIG, toImageButtonOptions: { ...PLOT_CONFIG.toImageButtonOptions, filename } }
    : PLOT_CONFIG;
  return Plotly.react(id, traces, { ...PLOT_LAYOUT, ...layout }, config);
}

// Read a rendered <table> back out as CSV so the download always matches what
// the page shows (superscripts and CI columns included).
export function tableToCsv(table) {
  const rows = [...table.querySelectorAll("tr")].map((tr) =>
    [...tr.querySelectorAll("th,td")].map((td) => td.textContent.replace(/\s+/g, " ").trim()));
  return rows.length ? toCsv(rows[0], rows.slice(1)) : "";
}

export function downloadTable(table, filename) {
  const csv = tableToCsv(table);
  if (csv) downloadBlob(filename, csv, "text/csv");
}

// Adds a small "CSV" button next to a table, once.
export function addTableDownload(container, table, filename) {
  if (!container || !table || container.querySelector(".table-dl")) return;
  const b = document.createElement("button");
  b.className = "btn tiny table-dl";
  b.type = "button";
  b.textContent = "CSV";
  b.title = "Download this table";
  b.addEventListener("click", () => downloadTable(table, filename));
  container.appendChild(b);
}
