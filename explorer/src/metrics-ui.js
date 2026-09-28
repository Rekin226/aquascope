import { $, downloadBlob, escapeHtml } from "./core.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";
import { metrics } from "./metrics.js?v=__BUILD__";

function openUsage() {
  const on = metrics.enabled();
  openModal("Optional usage log", `<p>Help evaluate whether AquaScope is useful by keeping a small log in this browser. It is ${on ? "on" : "off"}. Nothing is sent automatically.</p>
    <p>When enabled, it keeps daily action counts and operation durations for 28 days. It does not record questions, coordinates, uploaded records, filenames, keys, URLs, or a visitor identifier. The export lets you inspect and voluntarily share the log during a pilot.</p>
    <p>Download counts mean a file was handed to your browser; they do not prove it was saved or used. This opt-in sample cannot measure all visitors or cross-device retention.</p>
    <div class="row-actions"><button class="btn" data-usage="${on ? "clear" : "enable"}">${on ? "Turn off and delete log" : "Enable local log"}</button>
    ${on ? '<button class="btn" data-usage="export">Export usage log</button>' : ""}</div>
    ${on ? `<details><summary>Inspect stored counters</summary><pre>${escapeHtml(JSON.stringify(metrics.snapshot(), null, 2))}</pre></details>` : ""}`);
  $("modal-body").querySelectorAll("[data-usage]").forEach(button => button.addEventListener("click", () => {
    if (button.dataset.usage === "enable") { metrics.enable(); openUsage(); }
    else if (button.dataset.usage === "clear") { metrics.clear(); openUsage(); }
    else downloadBlob("aquascope-usage-log.json", JSON.stringify(metrics.snapshot(), null, 2), "application/json");
  }));
}
export function initMetrics() {
  metrics.record("visit");
  $("btn-usage").addEventListener("click", openUsage);
}
