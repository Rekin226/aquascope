// World river status (#544): reads one month's GeoTIFF off the main thread, so
// the globe keeps turning while a frame is made. A module worker; it answers
// { id, png } (the month painted on a Web Mercator square) or { id, error }.

import { GEOTIFF_MODULE, decodeStatus, gridToPng } from "./status-core.js?v=__BUILD__";

let geotiff = null;

self.onmessage = async (event) => {
  const { id, url, width, height } = event.data || {};
  try {
    if (typeof OffscreenCanvas === "undefined") throw new Error("no OffscreenCanvas");
    geotiff = geotiff || await import(GEOTIFF_MODULE);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const grid = await decodeStatus(geotiff, await res.arrayBuffer(), width, height);
    const png = await gridToPng(grid, width, height, (w, h) => new OffscreenCanvas(w, h));
    self.postMessage({ id, png });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err), noCanvas: typeof OffscreenCanvas === "undefined" });
  }
};
