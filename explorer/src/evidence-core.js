// The evidence ladder's pure half (#518): grade colours and words, the SQL for
// the published skill table, and the rows the Evidence tab shows. The scoring
// itself is aquascope.evidence in the worker; nothing here computes a metric.
// No DOM; node-importable (explorer/tests/evidence-core.test.mjs).

// Ordinal and colour-blind safe (the same ramp as "Last observation"), and
// always given as a legend, so colour is never the only encoding.
export const GRADE_COLORS = { A: "#1a9850", B: "#91cf60", C: "#fdae61", D: "#d73027" };
export const NO_GRADE_COLOR = "#b0bec5";
export const GRADES = ["A", "B", "C", "D"];

// The bands of aquascope.evidence.GRADING, in words for the legend and the tab.
export const GRADE_WORDS = {
  A: "KGE 0.75 and up",
  B: "KGE 0.5 to 0.75",
  C: "KGE above -0.41",
  D: "no better than the mean flow",
};

// One hue per model line; the gauge itself is drawn bold in its agency colour.
export const MODEL_COLORS = { geoglows: "#8e24aa", glofas: "#ef6c00", nwm: "#00897b", grrr: "#3949ab" };

export const gradeColor = (g) => GRADE_COLORS[g] || NO_GRADE_COLOR;

/** The legend for the "Best model skill" colouring: the four grades and "not scored", or why it is empty. */
export function skillLegend(status) {
  if (status === "missing") {
    return [{ color: NO_GRADE_COLOR, label: "not published yet: the monthly skill run has not filled the table" }];
  }
  if (status === "loading") return [{ color: NO_GRADE_COLOR, label: "loading the skill table…" }];
  return [...GRADES.map((g) => ({ color: GRADE_COLORS[g], label: `${g}: ${GRADE_WORDS[g]}` })),
    { color: NO_GRADE_COLOR, label: "not scored" }];
}

/** skill/model_skill.parquet sits in the same dataset as stations.parquet. */
export function skillUrl(stationsUrl) {
  return String(stationsUrl).replace(/stations\.parquet(\?.*)?$/, "skill/model_skill.parquet");
}

const sqlString = (s) => `'${String(s).replace(/'/g, "''")}'`;

/** The best graded model per gauge, for the map. */
export function bestSql(url) {
  return `SELECT source, station_id, grade, model, kge FROM read_parquet(${sqlString(url)}) WHERE is_best AND grade IS NOT NULL`;
}

/** Every published row for one gauge (the NWM and GRRR scores the page cannot compute). */
export function gaugeSql(url, source, stationId) {
  return `SELECT * FROM read_parquet(${sqlString(url)}) WHERE source = ${sqlString(source)} AND station_id = ${sqlString(stationId)}`;
}

/** DuckDB rows to plain JSON: BigInt counts become numbers. */
export function plainRow(row) {
  const out = {};
  for (const [k, v] of Object.entries(row || {})) out[k] = typeof v === "bigint" ? Number(v) : v;
  return out;
}

/** "source/station_id" -> {grade, model, kge}. */
export function bestByGauge(rows) {
  const m = new Map();
  for (const r of rows || []) {
    if (!r || !r.grade) continue;
    m.set(`${r.source}/${r.station_id}`, { grade: String(r.grade), model: r.model, kge: r.kge === null ? null : Number(r.kge) });
  }
  return m;
}

const num = (x) => (x === null || x === undefined || x === "" || !Number.isFinite(Number(x)) ? null : Number(x));

export function fmt2(x) {
  const v = num(x);
  return v === null ? "—" : v.toFixed(2);
}

export function fmtPct(x) {
  const v = num(x);
  if (v === null) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(0)} %`;
}

/** The skill table's rows, in the order the ladder lists the models; a row without a score keeps its reason. */
export function skillTableRows(models) {
  return (models || []).map((r) => ({
    model: r.model,
    label: r.label || r.model,
    grade: r.grade || null,
    scored: num(r.kge) !== null,
    kge: fmt2(r.kge), r: fmt2(r.r), alpha: fmt2(r.alpha), beta: fmt2(r.beta), nse: fmt2(r.nse),
    pbias: fmtPct(r.pbias),
    q2: fmtPct(r.q2_error_pct), q10: fmtPct(r.q10_error_pct), q100: fmtPct(r.q100_error_pct),
    why: r.why || "",
    published: Boolean(r.from_table),
  }));
}

/** The small badge in the gauge header: "Models: B" and what it means, or null. */
export function badge(best) {
  if (!best || !best.grade) return null;
  return {
    text: `Models ${best.grade}`,
    color: gradeColor(best.grade),
    title: `Best global model at this gauge: grade ${best.grade} (${GRADE_WORDS[best.grade] || ""}). Open Evidence for the table.`,
  };
}
