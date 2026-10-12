// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  GRADE_COLORS, NO_GRADE_COLOR, badge, bestByGauge, bestSql, fmtPct, gaugeSql, gradeColor, plainRow, skillLegend,
  skillTableRows, skillUrl,
} from "../src/evidence-core.js";

test("the skill table sits in the same dataset as the catalog", () => {
  assert.equal(skillUrl("https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet"),
    "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/skill/model_skill.parquet");
});

test("the gauge query escapes quotes, so a station id cannot break out of the string", () => {
  const sql = gaugeSql("https://x/skill.parquet", "uk_ea", "a'b");
  assert.match(sql, /station_id = 'a''b'/);
  assert.match(sql, /source = 'uk_ea'/);
  assert.match(bestSql("https://x/it's.parquet"), /read_parquet\('https:\/\/x\/it''s.parquet'\) WHERE is_best/);
});

test("grades map to their colours, and anything else is grey", () => {
  assert.equal(gradeColor("A"), GRADE_COLORS.A);
  assert.equal(gradeColor("D"), GRADE_COLORS.D);
  assert.equal(gradeColor(null), NO_GRADE_COLOR);
  assert.equal(gradeColor("Z"), NO_GRADE_COLOR);
});

test("the legend lists four grades and 'not scored', or says why it is empty", () => {
  const ready = skillLegend("ready");
  assert.equal(ready.length, 5);
  assert.match(ready[0].label, /^A: KGE 0\.75/);
  assert.equal(ready[4].label, "not scored");
  assert.match(skillLegend("missing")[0].label, /not published yet/);
  assert.equal(skillLegend("missing").length, 1);
});

test("best rows become a key -> grade map, ungraded rows dropped, BigInt counts made numbers", () => {
  const m = bestByGauge([
    { source: "usgs", station_id: "USGS-1", grade: "B", model: "nwm", kge: 0.61 },
    { source: "uk_ea", station_id: "x", grade: null, model: "geoglows", kge: 0.1 },
  ]);
  assert.equal(m.size, 1);
  assert.deepEqual(m.get("usgs/USGS-1"), { grade: "B", model: "nwm", kge: 0.61 });
  assert.deepEqual(plainRow({ n_days: 4000n, kge: 0.5 }), { n_days: 4000, kge: 0.5 });
});

test("percent errors carry their sign and a dash when missing", () => {
  assert.equal(fmtPct(38.4), "+38 %");
  assert.equal(fmtPct(-12), "−12 %");
  assert.equal(fmtPct(null), "—");
});

test("table rows keep the reason a model was not scored, and mark the monthly ones", () => {
  const rows = skillTableRows([
    { model: "geoglows", label: "GEOGLOWS v2", grade: "C", kge: 0.31, r: 0.6, alpha: 0.8, beta: 0.9, nse: 0.2, pbias: -10,
      q100_error_pct: 38, why: "KGE 0.31" },
    { model: "glofas", label: "GloFAS v4", grade: null, why: "the gauge record ends before GloFAS begins (1984)" },
    { model: "nwm", label: "NWM v3.0", grade: "A", kge: 0.8, from_table: true },
  ]);
  assert.equal(rows[0].scored, true);
  assert.equal(rows[0].kge, "0.31");
  assert.equal(rows[0].q100, "+38 %");
  assert.equal(rows[1].scored, false);
  assert.match(rows[1].why, /1984/);
  assert.equal(rows[2].published, true);
});

test("the header badge says the grade, or nothing without one", () => {
  assert.equal(badge(null), null);
  assert.equal(badge({ grade: null }), null);
  const b = badge({ grade: "B" });
  assert.equal(b.text, "Models B");
  assert.equal(b.color, GRADE_COLORS.B);
});
