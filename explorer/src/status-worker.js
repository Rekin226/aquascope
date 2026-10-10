// World river status (#544): reads one month's GeoTIFF off the main thread, so
// the globe keeps turning while a frame is made. A module worker; it answers
// { id, png, regions } (the month painted on a Web Mercator square, and the named
// regions' shares for the caption) or { id, error }.

import { GEOTIFF_MODULE, decodeStatus, focusPalette, gridToPng } from "./status-core.js?v=__BUILD__";

let geotiff = null;

self.onmessage = async (event) => {
  const { id, url, width, height, focus } = event.data || {};
  try {
    if (typeof OffscreenCanvas === "undefined") throw new Error("no OffscreenCanvas");
    geotiff = geotiff || await import(GEOTIFF_MODULE);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { grid, regions } = await decodeStatus(geotiff, await res.arrayBuffer(), width, height, { withRegions: true });
    const png = await gridToPng(grid, width, height, (w, h) => new OffscreenCanvas(w, h), focusPalette(focus));
    self.postMessage({ id, png, regions });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err), noCanvas: typeof OffscreenCanvas === "undefined" });
  }
};
