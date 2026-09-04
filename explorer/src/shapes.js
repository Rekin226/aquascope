// A shape per source, so which agency a gauge belongs to is never carried by
// colour alone (#283).
//
// The six agency colours fail an all-pairs colour-vision check in both themes:
// Hub'Eau's red and the Environment Agency's green are ΔE 4.2 apart under
// deuteranopia, and those are the two largest European sources, so to a
// red-green colourblind reader most of western Europe is one undifferentiated
// wash. Six hues cannot be made to pass all-pairs on hue alone -- that is a
// property of the test, not of these particular six -- so the fix is a second
// channel rather than a new palette. The colours are unchanged.
//
// The map draws these as SDF icons, which is what lets `icon-color` stay
// data-driven: "colour by record length" still repaints the same symbols, and
// the shape goes on saying which agency underneath.

/** Draw a unit shape centred on (0,0) with radius 1 into a 2D context. */
const PATHS = {
  circle(ctx) {
    ctx.arc(0, 0, 1, 0, Math.PI * 2);
  },
  square(ctx) {
    const a = 0.82;
    ctx.rect(-a, -a, a * 2, a * 2);
  },
  triangle(ctx) {
    // Sat on its centroid rather than its bounding box, or it looks dropped.
    const r = 1.18;
    for (let i = 0; i < 3; i++) {
      const t = -Math.PI / 2 + (i * 2 * Math.PI) / 3;
      const x = Math.cos(t) * r, y = Math.sin(t) * r + 0.12;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
  },
  diamond(ctx) {
    const r = 1.2;
    ctx.moveTo(0, -r); ctx.lineTo(r, 0); ctx.lineTo(0, r); ctx.lineTo(-r, 0);
    ctx.closePath();
  },
  cross(ctx) {
    const a = 1.15, b = 0.42;
    ctx.rect(-a, -b, a * 2, b * 2);
    ctx.rect(-b, -a, b * 2, a * 2);
  },
  pentagon(ctx) {
    const r = 1.15;
    for (let i = 0; i < 5; i++) {
      const t = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      const x = Math.cos(t) * r, y = Math.sin(t) * r + 0.06;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
  },
};

export const SHAPE_NAMES = Object.keys(PATHS);

function fillShape(ctx, name, size, radius) {
  ctx.save();
  ctx.translate(size / 2, size / 2);
  ctx.scale(radius, radius);
  ctx.beginPath();
  (PATHS[name] || PATHS.circle)(ctx);
  ctx.restore();
  ctx.fill();
}

/**
 * A signed distance field for one shape, as MapLibre's `addImage` wants it.
 *
 * An SDF icon is the only kind whose colour can come from the data, and the
 * gauge colour has to stay data-driven for the "colour by" control. MapLibre
 * reads the alpha channel as distance: 0.5 is the edge, and the spec's own
 * cutoff is 8 pixels of range either side.
 */
export function shapeSdf(name, size = 40, radius = 12) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  fillShape(ctx, name, size, radius);
  const src = ctx.getImageData(0, 0, size, size).data;

  // Inside/outside from the drawn alpha, then a brute-force distance transform.
  // 32x32 over a 9 px window is about 300k comparisons per shape and runs once,
  // which is cheaper than shipping six pre-baked images and keeping them in
  // step with the palette.
  const inside = new Uint8Array(size * size);
  for (let i = 0; i < inside.length; i++) inside[i] = src[i * 4 + 3] > 127 ? 1 : 0;

  const SPREAD = 8;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const self = inside[i];
      let best = SPREAD;
      for (let dy = -SPREAD; dy <= SPREAD; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= size) continue;
        for (let dx = -SPREAD; dx <= SPREAD; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= size) continue;
          if (inside[yy * size + xx] === self) continue;
          const d = Math.hypot(dx, dy);
          if (d < best) best = d;
        }
      }
      // 0.5 at the edge, rising inwards, falling outwards.
      const signed = self ? best : -best;
      const alpha = Math.round((signed / SPREAD) * 0.5 * 255 + 127.5);
      out[i * 4] = 255; out[i * 4 + 1] = 255; out[i * 4 + 2] = 255;
      out[i * 4 + 3] = alpha;
    }
  }
  return { width: size, height: size, data: out };
}

/**
 * The same shape as an inline SVG, for the swatches in the rail and the search
 * results. A legend that only showed colour would be describing half of what
 * the map draws.
 */
export function shapeSvg(name, color, size = 11) {
  const r = size / 2 - 0.6;
  const pt = (t, rad, dy = 0) => `${(size / 2 + Math.cos(t) * rad).toFixed(2)},${(size / 2 + Math.sin(t) * rad + dy).toFixed(2)}`;
  const poly = (n, rad, dy) =>
    Array.from({ length: n }, (_, i) => pt(-Math.PI / 2 + (i * 2 * Math.PI) / n, rad, dy)).join(" ");
  let body;
  if (name === "square") {
    const a = r * 0.82;
    body = `<rect x="${size / 2 - a}" y="${size / 2 - a}" width="${a * 2}" height="${a * 2}" rx="1" fill="${color}"/>`;
  } else if (name === "triangle") {
    body = `<polygon points="${poly(3, r * 1.18, r * 0.12)}" fill="${color}"/>`;
  } else if (name === "diamond") {
    body = `<polygon points="${poly(4, r * 1.2, 0)}" fill="${color}"/>`;
  } else if (name === "pentagon") {
    body = `<polygon points="${poly(5, r * 1.15, r * 0.06)}" fill="${color}"/>`;
  } else if (name === "cross") {
    const a = r * 1.1, b = r * 0.4;
    body = `<rect x="${size / 2 - a}" y="${size / 2 - b}" width="${a * 2}" height="${b * 2}" fill="${color}"/>` +
      `<rect x="${size / 2 - b}" y="${size / 2 - a}" width="${b * 2}" height="${a * 2}" fill="${color}"/>`;
  } else {
    body = `<circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="${color}"/>`;
  }
  return `<svg class="shape-swatch" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" ` +
    `aria-hidden="true" focusable="false">${body}</svg>`;
}
