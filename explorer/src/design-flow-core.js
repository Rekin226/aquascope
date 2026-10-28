// The one design flow every tool reads (#664 follow-up): a reach's T-year flow on its curve, Log-Pearson III on
// GEOGLOWS v2's annual maxima (aquascope.floodsim.design_flow, flow_for and flow_words). The package fits the
// curve and computes every flow; this module only reads a flow off a curve the worker sent, for the picker's chips,
// and words the method and the source the same way on every card (Flood it, Route it, Draw a reservoir, Replay).

/** A flow on a curve { T, q }: linear in log(T) between the fitted points, held at the ends (floodsim.flow_for). */
export function flowFor(curve, T) {
  if (!curve || !Array.isArray(curve.q) || !Array.isArray(curve.T) || !curve.q.length) return null;
  const lt = curve.T.map(Math.log), x = Math.log(Math.min(Math.max(Number(T), curve.T[0]), curve.T.at(-1)));
  for (let k = 0; k < lt.length - 1; k++) {
    if (x <= lt[k + 1]) return curve.q[k] + ((curve.q[k + 1] - curve.q[k]) * (x - lt[k])) / (lt[k + 1] - lt[k]);
  }
  return curve.q.at(-1);
}

/**
 * The method and the source, in the package's words (floodsim.flow_words): "Log-Pearson III on GEOGLOWS v2's annual
 * maxima of simulated daily flow, 1940 to 2025". From a curve or a design flow ({ method, first_year, last_year }).
 */
export function designFlowWords(d) {
  if (!d || !d.method) return "";
  const span = d.first_year && d.last_year ? `, ${d.first_year} to ${d.last_year}` : "";
  return `${d.method} on GEOGLOWS v2's annual maxima of simulated daily flow${span}`;
}

/** The card's line for a design flow: "Flow: the 100-year flow, Log-Pearson III on ...". Empty for a flow set by hand. */
export function designFlowLine(design) {
  if (!design || !design.method) return "";
  const t = Number(design.return_period);
  return `Flow: the ${Number.isFinite(t) ? `${t}-year flow` : "design flow"}, ${designFlowWords(design)}.`;
}
