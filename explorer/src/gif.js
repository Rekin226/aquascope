// A GIF of a played range (#522), made in the browser: step the map through
// the dates, wait for each day's tiles, copy the map canvas, stamp the date and
// the credit on it, and encode with gifenc (MIT, 9 kB, loaded only when someone
// asks for a GIF). Nothing leaves the page; the file is a download.

import { downloadBlob } from "./core.js?v=__BUILD__";
import { captureMap, whenSettled } from "./map.js?v=__BUILD__";
import { gifSize, shortDate } from "./timeline.js?v=__BUILD__";

const GIFENC = "https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js";
const MAX_WIDTH = 640;

function stamp(ctx, w, h, date, label, by = "NASA GIBS") {
  const pad = Math.round(w / 64);
  const big = Math.max(13, Math.round(w / 30));
  const small = Math.max(10, Math.round(big * 0.62));
  ctx.font = `600 ${big}px system-ui, -apple-system, sans-serif`;
  const text = shortDate(date);
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(10, 22, 32, .72)";
  ctx.fillRect(pad, pad, tw + pad * 1.6, big + pad * 1.1);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "top";
  ctx.fillText(text, pad * 1.8, pad * 1.55);
  // The credit the layers ask for, on every frame, because a GIF travels alone.
  ctx.font = `${small}px system-ui, -apple-system, sans-serif`;
  const credit = `${label ? `${label} · ` : ""}${by} · AquaScope Explorer`;
  const cw = ctx.measureText(credit).width;
  ctx.fillStyle = "rgba(10, 22, 32, .6)";
  ctx.fillRect(w - cw - pad * 2, h - small - pad * 1.4, cw + pad * 1.6, small + pad);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(credit, w - cw - pad * 1.2, h - small - pad * 0.9);
}

/**
 * Walk `dates`, capture a frame for each and download the GIF.
 * Resolves true when a file was saved, false when it was stopped.
 */
export async function makeGif({ dates, label, credit, step, setDate, isCancelled, onProgress }) {
  const { GIFEncoder, quantize, applyPalette } = await import(GIFENC);
  const gif = GIFEncoder();
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const delay = step === "day" ? 500 : 800;
  let size = null;
  for (let i = 0; i < dates.length; i++) {
    if (isCancelled()) return false;
    onProgress(i + 1, dates.length);
    setDate(dates[i]);
    await whenSettled(10000);
    if (isCancelled()) return false;
    const ok = await captureMap((src) => {
      if (!size) {
        size = gifSize(src.width, src.height, MAX_WIDTH);
        canvas.width = size.width;
        canvas.height = size.height;
      }
      ctx.drawImage(src, 0, 0, size.width, size.height);
    });
    if (!ok) throw new Error("the map could not be read");
    stamp(ctx, size.width, size.height, dates[i], label, credit);
    const { data } = ctx.getImageData(0, 0, size.width, size.height);
    const palette = quantize(data, 256);
    gif.writeFrame(applyPalette(data, palette), size.width, size.height, { palette, delay });
    // Let the page breathe between frames: quantising a 640 px frame takes a moment.
    await new Promise((r) => setTimeout(r, 0));
  }
  gif.finish();
  const name = `aquascope-${dates[0]}_${dates[dates.length - 1]}.gif`;
  downloadBlob(name, gif.bytes(), "image/gif");
  return true;
}
