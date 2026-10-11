// Your model results on the globe (#555): the pure part of the mesh view.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  MAX_FILE_BYTES, advect, arrowSize, bearing, binOf, buildIndex, canvasFrame, cellAt, cellFacts, checkFile, colorAt,
  NO_GEOMETRY, friendlyError, heading, legendLine, pickModelFiles, metresPer, modelShort, modelTime, packedBins, paintPixels, particleTimeScale, rangeWords,
  rasterizeIds, scaleFor, shortName, thinArrows, ticks,
} from "../src/mesh-core.js";

// Two unit squares side by side, in degrees: cell 0 over x 0..1, cell 1 over x 1..2.
const MESH = {
  ring_offsets: Int32Array.from([0, 4, 8]),
  ring_lon: Float64Array.from([0, 1, 1, 0, 1, 2, 2, 1]),
  ring_lat: Float64Array.from([0, 0, 1, 1, 0, 0, 1, 1]),
};

test("files the viewer reads, and the ones it turns away in plain words", () => {
  assert.equal(checkFile({ name: "Muncie.p04.hdf", size: 1e6 }), null);
  assert.equal(checkFile({ name: "map.nc", size: 10 }), null);
  assert.match(checkFile({ name: "results.csv", size: 10 }), /HEC-RAS plan file \(\.p01\.hdf\) or a UGRID NetCDF/);
  assert.match(checkFile({ name: "Muncie.g05.hdf", size: 10 }), /geometry file, with no results/);
  assert.match(checkFile({ name: "huge.hdf", size: MAX_FILE_BYTES + 1 }), /up to 2 GB/);
  assert.match(checkFile({ name: "empty.nc", size: 0 }), /empty/);
});

test("a stripped plan file and its geometry file dropped together are paired", () => {
  const plan = { name: "Muncie.p04.timeseries.hdf", size: 5e6 };
  const geom = { name: "Muncie.g05.hdf", size: 1e6 };
  assert.deepEqual(pickModelFiles([geom, plan]), { file: plan, geometry: geom });
  assert.deepEqual(pickModelFiles([plan]), { file: plan, geometry: null });
  // A geometry file alone has no results; a drop with nothing readable says why.
  assert.match(pickModelFiles([geom]).why, /geometry file, with no results/);
  assert.match(pickModelFiles([{ name: "notes.txt", size: 3 }]).why, /not a model result/);
  assert.equal(pickModelFiles([]).why, "No file.");
  // The first readable file is the result: a stray text file does not stop the drop.
  assert.deepEqual(pickModelFiles([{ name: "notes.txt", size: 3 }, plan]), { file: plan, geometry: null });
});

test("a plan file without its mesh is told what to drop next; other errors stay the package's words", () => {
  const py = `Muncie.p04.timeseries.hdf ${NO_GEOMETRY}: Geometry/2D Flow Areas is missing. It looks like a stripped ` +
    "plan file. Open the full plan file (.p01.hdf), or give the geometry file the plan was run with (.g01.hdf) too.";
  const said = friendlyError(py, "Muncie.p04.timeseries.hdf");
  assert.equal(said, "Muncie.p04.timeseries.hdf has the model results but not the mesh. Drop the full plan file " +
    "(.p01.hdf), or this file together with its geometry file (.g01.hdf).");
  // A missing 2D area is named in words, not as its HDF5 path.
  const area = friendlyError(py.replace("Geometry/2D Flow Areas is", "Geometry/2D Flow Areas/BaldEagleCr/Faces FacePoint Indexes is"), "p.hdf");
  assert.match(area, /^p\.hdf has the model results but not the mesh \(no mesh for the 2D area BaldEagleCr\)\. Drop/);
  assert.doesNotMatch(area, /Geometry\//);
  assert.equal(friendlyError("x.hdf has no 2D flow areas."), "x.hdf has no 2D flow areas.");
});

test("the depth scale starts at zero and is square-root; the water surface scale is linear", () => {
  const depth = scaleFor(Float32Array.from([0.1, 0.5, 2.0, 3.7, NaN]), "depth");
  assert.equal(depth.lo, 0);
  assert.equal(depth.sqrt, true);
  assert.ok(depth.hi >= 3.7 && depth.hi <= 4);
  const ws = scaleFor([537.2, 600, 700, 765.2], "water_surface");
  assert.equal(ws.sqrt, false);
  assert.ok(ws.lo <= 537.2 && ws.hi >= 765.2);
  assert.equal(binOf(NaN, depth), -1);
  assert.equal(binOf(depth.hi * 10, depth, 32), 31);
  assert.equal(colorAt(NaN, depth), null);
  const shallow = colorAt(0.05, depth), deep = colorAt(depth.hi, depth);
  assert.ok(shallow[2] > deep[2] && shallow[3] < deep[3]);   // lighter and more see-through when shallow
  assert.deepEqual(ticks({ lo: 0, hi: 4, sqrt: true }).map((t) => t.at), [0, 50, 100]);
});

test("a click finds its cell, and outside the mesh finds none", () => {
  const index = buildIndex(MESH, 1);
  assert.equal(cellAt(index, 0.5, 0.5), 0);
  assert.equal(cellAt(index, 1.5, 0.2), 1);
  assert.equal(cellAt(index, 2.5, 0.5), -1);
  assert.equal(cellAt(index, 0.5, 1.5), -1);
});

test("pixel picture: each pixel centre takes the cell over it, no seam and no gap", () => {
  const fr = canvasFrame([0, 0, 2, 1], 2, 64);
  assert.ok(fr.width > fr.height);
  const n = MESH.ring_lon.length;
  const px = new Float32Array(n), py = new Float32Array(n);
  for (let k = 0; k < n; k++) { px[k] = fr.x(MESH.ring_lon[k]); py[k] = fr.y(MESH.ring_lat[k]); }
  const ids = rasterizeIds(MESH.ring_offsets, px, py, fr.width, fr.height);
  const mid = Math.floor(fr.height / 2) * fr.width;
  const row = Array.from(ids.slice(mid, mid + fr.width));
  const first0 = row.indexOf(0), last0 = row.lastIndexOf(0), first1 = row.indexOf(1);
  assert.ok(first0 >= 0 && first1 === last0 + 1, "cell 1 starts right where cell 0 ends");
  assert.ok(row.filter((v) => v === 0).length > 0.4 * fr.width);
  const colours = Uint32Array.from([0x11223344, 0x55667788]);
  const out = paintPixels(new Uint32Array(ids.length), ids, colours);
  assert.equal(out[mid + first0], 0x11223344);
  assert.equal(out[mid + first1], 0x55667788);
  const packed = packedBins(4).map((c) => c >>> 24);   // alpha, the top byte: deeper is more opaque
  assert.equal(packed.length, 4);
  assert.ok(packed.every((a, k) => k === 0 || a > packed[k - 1]));
});

test("arrows: one per screen cell, nearest the centre; size grows with speed", () => {
  const keep = thinArrows([{ i: 1, x: 5, y: 5 }, { i: 2, x: 16, y: 17 }, { i: 3, x: 50, y: 10 }], 34);
  assert.deepEqual(keep.sort(), [2, 3]);
  assert.equal(arrowSize(0, 2), 0.5);
  assert.equal(arrowSize(2, 2), 1);
  assert.ok(arrowSize(0.5, 2) < arrowSize(1.5, 2));
  assert.equal(Math.round(bearing(1, 0)), 90);
  assert.equal(Math.round(bearing(0, -1)), 180);
  assert.equal(heading(1, 1), "north-east");
  assert.equal(heading(0, 0), "");
});

test("particles move with the current, in metres or feet", () => {
  const [lon, lat] = advect(0, 0, 0, 1, 1000, 1);   // 1 m/s north for 1000 s
  assert.equal(lon, 0);
  assert.ok(Math.abs(lat - 1000 / 111195) < 1e-5);
  assert.equal(metresPer("ft/s"), 0.3048);
  assert.equal(metresPer("m s-1"), 1);
  assert.equal(particleTimeScale(0, 10), 0);
  assert.equal(particleTimeScale(2, 10, 1, 60), 300);
});

test("words: model time, the range, the legend line and the card", () => {
  assert.equal(modelTime("1999-01-02T14:00:00"), "2 Jan 1999 14:00");
  assert.equal(modelTime("1999-01-02T14:00:00", { year: false }), "2 Jan 14:00");
  assert.equal(rangeWords(["1999-01-01T12:00:00", "1999-01-02T12:00:00", "1999-01-04T12:00:00"]),
    "3 steps, 1 Jan 1999 12:00 to 4 Jan 12:00");
  assert.equal(rangeWords(["1999-12-31T00:00:00", "2000-01-01T00:00:00"]), "2 steps, 31 Dec 1999 00:00 to 1 Jan 2000 00:00");
  assert.equal(shortName("Grevelingen-FM_0000_map_long_name.nc", 20).length, 20);
  assert.equal(modelShort("HEC-RAS 6.5 February 2024"), "HEC-RAS 6.5");
  const meta = { variable: "depth", cells: 4425, units: { depth: "ft", velocity: "ft/s" },
    areas: [{ name: "Main", cells: 3359, start: 0 }, { name: "Upper", cells: 1066, start: 3359 }],
    times: ["1999-01-01T12:00:00", "1999-01-01T14:00:00"] };
  assert.equal(legendLine(meta), "Largest depth in each cell over the run · 4,425 cells");
  assert.equal(legendLine(meta, { step: 1 }), "Depth at 1 Jan 1999 14:00 · step 2 of 2");
  const now = cellFacts(meta, 3360, { at: { value: 2.345, u: 0, v: 1.5 }, largest: { value: 3.1, speed: 2 }, when: meta.times[1] });
  assert.equal(now.title, "Upper, cell 2");
  assert.equal(now.status, "Depth 2.35 ft at 1 Jan 1999 14:00");
  assert.equal(now.note, "Velocity 1.5 ft/s, towards the north.");
  assert.deepEqual(now.figure, { value: "3.1", unit: "ft", label: "largest depth" });
  const dry = cellFacts(meta, 0, { at: { value: NaN, u: NaN, v: NaN }, largest: { value: NaN }, when: meta.times[0] });
  assert.equal(dry.status, "Dry at 1 Jan 1999 12:00");
  assert.equal(dry.figure, null);
  const still = cellFacts(meta, 0, { at: { value: 1, u: 0, v: 0 }, largest: { value: 1 }, when: meta.times[0] });
  assert.equal(still.note, "Still water at this step.");
  const largest = cellFacts({ ...meta, variable: "water_surface", units: { water_surface: "ft" } }, 5, { largest: { value: 653.2 } });
  assert.equal(largest.status, "Largest water surface 653 ft over the run");
});
