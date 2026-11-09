// Groundwater on the map (#558): colours, painting, cards and the MODFLOW helpers.
import test from "node:test";
import assert from "node:assert/strict";
import {
  APPROX_LABEL, MODFLOW_ACCEPT, approxWords, isApproximate, readApproxChoice,
  CHANGE_ALPHA_MIN, CHANGE_RAMP, HEAD_RAMP, SURFACE_ALPHA, cellsGeoJSON, headWord, isModflowDrop, lengthUnit, surfaceAlpha, changeWord, fmtM, frameOf, gridCellAt, gridCorners, keyLines,
  maskScale, modelOutline, monthOf, paintMask, paintSurface, rampColor, rangeT, readGwParam, readPlaceForm, readingsWindow,
  rowSummary, sortModelFiles, stepForDate, surfaceCard, timeLabel, viewTouches, wellCard, wellsNear,
} from "../src/groundwater-core.js";

const grid = { west: -1, south: 51, east: 1, north: 53, nx: 2, ny: 2, values: [10, null, 20, 30], status: [2, 1, 2, 0] };
const res = {
  mode: "level", label: "July 2026", radius_km: 20, min_wells: 3, range: [10, 30], n_wells_in_view: 3,
  rule: "Drawn where at least 3 wells lie within 20 km", notes: [], contours: { step: 5, levels: [] }, grid,
  wells: [{ id: "a", name: "Well A", lon: -0.5, lat: 52.6, value: 12.3, how: "month mean" }],
};

test("ramps run from their first to their last stop", () => {
  assert.deepEqual(rampColor(HEAD_RAMP, 0), [255, 255, 217]);
  assert.deepEqual(rampColor(HEAD_RAMP, 1), [12, 44, 132]);
  assert.deepEqual(rampColor(CHANGE_RAMP, 0.5), [245, 245, 245]);
  assert.deepEqual(rampColor(HEAD_RAMP, 7), rampColor(HEAD_RAMP, 1));
  assert.equal(rangeT(20, [10, 30]), 0.5);
});

test("the surface paints drawn cells only, the mask hatches the sparse ones", () => {
  const px = paintSurface(grid, [10, 30], "level");
  assert.equal(px[3], SURFACE_ALPHA);
  assert.equal(px[7], 0);       // too few wells: not painted
  assert.equal(px[11], SURFACE_ALPHA);
  assert.equal(px[15], 0);       // no well near: not painted
  const m = paintMask(grid, 4);
  assert.equal(m.width, 8);
  const alpha = (x, y) => m.px[(y * m.width + x) * 4 + 3];
  assert.equal(alpha(0, 0), 0);                     // a drawn cell: no hatch
  assert.ok(alpha(7, 0) === 150 || alpha(7, 0) === 22);  // the sparse cell: hatched
  assert.equal(alpha(5, 6), 0);                     // a cell with no well near: nothing
  assert.ok(maskScale({ nx: 220, ny: 150 }) >= 2);
});

test("a point finds its cell on the Mercator grid", () => {
  assert.deepEqual(gridCorners(grid), [[-1, 53], [1, 53], [1, 51], [-1, 51]]);
  assert.equal(gridCellAt(grid, -0.5, 52.6).k, 0);
  assert.equal(gridCellAt(grid, 0.5, 51.2).k, 3);
  assert.equal(gridCellAt(grid, 5, 52), null);
});

test("cards say what is there and how sure", () => {
  const near = wellsNear(res.wells, -0.5, 52.5, 20);
  assert.equal(near.length, 1);
  const c = surfaceCard(res, gridCellAt(grid, -0.5, 52.6), near);
  assert.equal(c.title, "10 m above Ordnance Datum");
  assert.match(c.sub, /interpolated from 1 well within 20 km/);
  const sparse = surfaceCard(res, gridCellAt(grid, 0.5, 52.6), near);
  assert.equal(sparse.title, "Too few wells here");
  assert.match(sparse.status.text, /needs 3/);
  const ch = { ...res, mode: "change", label: "July 2025 to July 2026" };
  assert.equal(surfaceCard(ch, { status: 2, value: -0.42 }, near).title, "-0.42 m, falling");
  const w = wellCard(ch, { ...res.wells[0], value: 0.31, level: 12.3, level_before: 11.99 });
  assert.equal(w.figure.value, "+0.31");
  assert.match(w.status.text, /12\.0 to 12\.3 m AOD, rising/);
  assert.equal(changeWord(0.01), "about the same");
  assert.equal(fmtM(123.456), "123");
});

test("the legend's words", () => {
  assert.equal(rowSummary(null), "England's wells");
  assert.equal(rowSummary(res, { busy: true }), "reading…");
  assert.equal(rowSummary(res), "July 2026");
  assert.equal(rowSummary({ ...res, mode: "change", label: "a to b" }), "change, a to b");
  const k = keyLines(res);
  assert.match(k.head, /above Ordnance Datum/);
  assert.deepEqual(k.range, { lo: "10", hi: "30" });
  assert.ok(k.lines.some((l) => /3 wells in view · lines every 5 m/.test(l)));
  assert.match(k.mask, /fewer than 3 wells within 20 km/);
});

test("months, windows, the view and the link", () => {
  assert.equal(monthOf("2026-07-15"), "2026-07");
  assert.deepEqual(readingsWindow("2026-07"), { from: "2024-05-01", to: "2026-09-01" });
  assert.ok(viewTouches([-2, 51, 0, 52]));
  assert.ok(!viewTouches([10, 40, 20, 45]));
  assert.equal(readGwParam("#gw=1&v=5/52/-1"), true);
  assert.equal(readGwParam("#v=5/52/-1"), null);
});

test("dropped files are sorted by kind", () => {
  const f = (name) => ({ name });
  const s = sortModelFiles([f("model.dis.grb"), f("Model.HDS"), f("model.nam"), f("notes.txt"), f("model.tdis")]);
  assert.equal(s.hds.name, "Model.HDS");
  assert.equal(s.grid.name, "model.dis.grb");
  assert.deepEqual(s.sidecars.map((x) => x.name), ["model.nam", "model.tdis"]);
  assert.deepEqual(s.other.map((x) => x.name), ["notes.txt"]);
  assert.equal(sortModelFiles([f("a.disv")]).grid.name, "a.disv");
});

test("the place form is read and checked", () => {
  assert.deepEqual(readPlaceForm({ x: "500000", y: "4000000", rotation: "15", crs: "27700", start: "2015-01-01" }),
    { origin: [500000, 4000000], rotation: 15, crs: "EPSG:27700", start_date: "2015-01-01" });
  assert.match(readPlaceForm({ x: "1" }).error, /both/);
  assert.match(readPlaceForm({ crs: "UTM" }).error, /EPSG/);
  assert.match(readPlaceForm({}, { needCell: true }).error, /cell size/);
  assert.equal(readPlaceForm({ cell: "250", crs: "epsg:32614" }).crs, "EPSG:32614");
});

test("model cells and frames", () => {
  const rings = new Float64Array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 1, 2, 0, 1, 0]);
  const fc = cellsGeoJSON(rings, new Int32Array([0, 4, 8]));
  assert.equal(fc.features.length, 2);
  assert.equal(fc.features[1].id, 1);
  assert.deepEqual(fc.features[0].geometry.coordinates[0][4], [0, 1]);
  const heads = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);   // 2 times x 2 layers x 2 cells
  assert.deepEqual([...frameOf(heads, [2, 2, 2], 1, 0)], [5, 6]);
  assert.equal(stepForDate(["2015-01-02", "2015-02-01", "2015-03-01"], "2015-02-15"), 1);
  assert.equal(stepForDate(["2015-01-02"], "2014-01-01"), 0);
  assert.equal(timeLabel({ times: [31], time_units: "days" }, 0), "t = 31 days");
  assert.equal(timeLabel({ times: [31], dates: ["2015-02-01"] }, 0), "2015-02-01");
  // Two cells side by side: the outline runs round both.
  assert.deepEqual(modelOutline({ type: "dis", drawn_nrow: 1, drawn_ncol: 2 }, rings), [[0, 1], [2, 1], [2, 0], [0, 0], [0, 1]]);
  assert.equal(modelOutline({ type: "disv" }, rings), null);
});

test("the change fades toward no change; the head is one even wash", () => {
  assert.equal(surfaceAlpha(0.5, "level"), SURFACE_ALPHA);
  assert.equal(surfaceAlpha(0, "level"), SURFACE_ALPHA);
  assert.equal(surfaceAlpha(0.5, "change"), CHANGE_ALPHA_MIN);
  assert.equal(surfaceAlpha(0, "change"), SURFACE_ALPHA);
  assert.equal(surfaceAlpha(1, "change"), SURFACE_ALPHA);
  assert.ok(surfaceAlpha(0.4, "change") > CHANGE_ALPHA_MIN && surfaceAlpha(0.4, "change") < SURFACE_ALPHA);
  const px = paintSurface({ nx: 2, ny: 1, values: [0, 5], status: [2, 2] }, [-5, 5], "change");
  assert.equal(px[3], CHANGE_ALPHA_MIN);
  assert.equal(px[7], SURFACE_ALPHA);
});

test("a model's heads are labelled in its own length unit, and a drawdown file as drawdown", () => {
  assert.equal(lengthUnit({ grid: { length_units: "meters" } }), "m");
  assert.equal(lengthUnit({ grid: { length_units: "feet" } }), "ft");
  assert.equal(lengthUnit({ grid: { length_units: null } }), "");
  assert.equal(headWord({ text: "HEAD" }), "Head");
  assert.equal(headWord({ text: "DRAWDOWN" }), "Drawdown");
});

test("one drop zone: MODFLOW files go to the heads reader, mesh results do not", () => {
  assert.equal(isModflowDrop([{ name: "freyberg.hds" }, { name: "freyberg.dis" }]), true);
  assert.equal(isModflowDrop([{ name: "model.dis.grb" }]), true);
  assert.equal(isModflowDrop([{ name: "Muncie.p04.hdf" }]), false);
  assert.equal(isModflowDrop([{ name: "sandiego_map.nc" }]), false);
  assert.equal(isModflowDrop([]), false);
});

test("a model with no georeference is placed in one click, approximately, and says so (#591)", () => {
  assert.deepEqual(readApproxChoice([10.123456789, 50.5]), { approximate_at: [10.123457, 50.5] });
  assert.deepEqual(readApproxChoice([0, 89.9]).approximate_at, [0, 84]);   // a globe turned to a pole still places it
  assert.match(readApproxChoice(null).error, /centre/);
  assert.match(readApproxChoice([0, 0], {}, { needCell: true }).error, /cell size/);
  assert.deepEqual(readApproxChoice([1, 2], { cell: "250" }, { needCell: true }), { approximate_at: [1, 2], cell_size: 250 });
  assert.match(readApproxChoice([1, 2], { cell: "-3" }).error, /positive/);
  assert.equal(isApproximate({ grid: { approximate: true } }), true);
  assert.equal(isApproximate({ grid: { approximate: false, placed_by: "the model's files" } }), false);
  assert.equal(isApproximate(null), false);
  assert.equal(APPROX_LABEL, "approximate placement");
  assert.match(approxWords({ length_units: "feet" }), /true size in feet/);
  assert.match(approxWords({ length_units: null }), /as metres/);
});

test("the file picker offers every MODFLOW file the drop takes", () => {
  for (const ext of MODFLOW_ACCEPT.split(",")) assert.equal(isModflowDrop([{ name: `model${ext}` }]), true, ext);
});
