// Flood depth (#554): reads the JRC depth windows of a cell and paints them off
// the main thread. A module worker. It answers { id, png, depthCm, rpAt, wet, max }
// for a cell, or { id, error }.
//
// geotiff.js reads each half-degree window with byte ranges (a cell is a few
// hundred kB of a 17 to 180 MB file), and the windows are kept, so stepping
// the time bar repaints from memory.

import { GEOTIFF_MODULE, paintCell } from "./flood-depth-core.js?v=__BUILD__";

let geotiff = null;
const files = new Map();      // url -> promise of the full-resolution image
const windows = new Map();    // `${url}|${cell}` -> promise of a grid (most recent last)
const KEEP_BYTES = 96 << 20;  // about 66 windows of 600 x 600 floats
let kept = 0;

function image(url) {
  if (!files.has(url)) {
    const p = geotiff.fromUrl(url, { allowFullFile: false }).then((t) => t.getImage(0));
    p.catch(() => files.delete(url));
    files.set(url, p);
  }
  return files.get(url);
}

async function readWindow(url, box) {
  const img = await image(url);
  const [ox, oy] = img.getOrigin();
  const [dx, dy] = img.getResolution();
  const [w, s, e, n] = box;
  const c0 = Math.max(0, Math.floor((w - ox) / dx)), c1 = Math.min(img.getWidth(), Math.ceil((e - ox) / dx));
  const r0 = Math.max(0, Math.floor((n - oy) / dy)), r1 = Math.min(img.getHeight(), Math.ceil((s - oy) / dy));
  if (c1 <= c0 || r1 <= r0) return null;
  const [data] = await img.readRasters({ window: [c0, r0, c1, r1], samples: [0] });
  return { data, x0: ox + c0 * dx, y0: oy + r0 * dy, dx, dy, width: c1 - c0, height: r1 - r0 };
}

function grid(url, cell) {
  const key = `${url}|${cell.key}`;
  if (windows.has(key)) {
    const p = windows.get(key);
    windows.delete(key);
    windows.set(key, p);
    return p;
  }
  const p = readWindow(url, cell.box).then((g) => {
    kept += g ? g.data.byteLength : 0;
    for (const [k, old] of windows) {
      if (kept <= KEEP_BYTES || k === key) break;
      windows.delete(k);
      old.then((o) => { kept -= o ? o.data.byteLength : 0; }, () => {});
    }
    return g;
  });
  p.catch(() => windows.delete(key));
  windows.set(key, p);
  return p;
}

self.onmessage = async (event) => {
  const { id, cell, parts, size, lines } = event.data || {};
  try {
    if (typeof OffscreenCanvas === "undefined") throw new Error("no OffscreenCanvas");
    geotiff = geotiff || await import(GEOTIFF_MODULE);
    const grids = new Map();
    await Promise.all(parts.map(async (p) => {
      const g = await grid(p.url, cell);
      if (g) grids.set(p.rp, g);
    }));
    const painted = paintCell(cell, grids, size, size, lines);
    let png = null;
    if (painted.wet) {
      const canvas = new OffscreenCanvas(size, size);
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(size, size);
      img.data.set(painted.rgba);
      ctx.putImageData(img, 0, 0);
      png = await canvas.convertToBlob({ type: "image/png" });
    }
    self.postMessage({ id, png, depthCm: painted.depthCm, rpAt: painted.rpAt, wet: painted.wet, max: painted.max },
      [painted.depthCm.buffer, painted.rpAt.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
