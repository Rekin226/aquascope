// The sketchpad's tools menu (#664): one button in the map tools opens a short list of things to draw on the map
// (a storm, a reservoir, a well, a land use), each added by its own module with registerSketchTool. The menu only
// lists and opens them; each tool draws, runs and logs itself.

import { $ } from "./core.js?v=__BUILD__";

const tools = [];

/**
 * Add a tool to the menu: { id, label, hint, icon (inner SVG markup, 24 x 24), run () }. The menu shows them in the
 * order added.
 */
export function registerSketchTool(tool) {
  if (!tool || !tool.id || tools.some((t) => t.id === tool.id)) return;
  tools.push(tool);
  render();
}

function render() {
  const menu = $("sketch-menu");
  const btn = $("btn-sketch");
  if (!menu || !btn) return;
  btn.hidden = !tools.length;
  menu.innerHTML = tools.map((t) => `<button type="button" role="menuitem" class="sk-item" data-tool="${t.id}">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" ` +
    `stroke-linejoin="round" aria-hidden="true">${t.icon || ""}</svg><span><b>${t.label}</b>` +
    `${t.hint ? `<small>${t.hint}</small>` : ""}</span></button>`).join("");
}

function setOpen(open) {
  const menu = $("sketch-menu"), btn = $("btn-sketch");
  if (!menu || !btn) return;
  menu.hidden = !open;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) { const first = menu.querySelector(".sk-item"); if (first) first.focus({ preventScroll: true }); }
}

export function initSketchMenu() {
  const menu = $("sketch-menu"), btn = $("btn-sketch");
  if (!menu || !btn) return;
  btn.addEventListener("click", () => setOpen(menu.hidden));
  menu.addEventListener("click", (e) => {
    const item = e.target.closest(".sk-item");
    if (!item) return;
    setOpen(false);
    const t = tools.find((x) => x.id === item.dataset.tool);
    if (t) t.run();
  });
  menu.addEventListener("keydown", (e) => {
    const items = [...menu.querySelectorAll(".sk-item")];
    const i = items.indexOf(document.activeElement);
    if (e.key === "Escape") { e.preventDefault(); setOpen(false); btn.focus(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
  document.addEventListener("pointerdown", (e) => {
    if (!menu.hidden && !menu.contains(e.target) && !btn.contains(e.target)) setOpen(false);
  });
  render();
}
