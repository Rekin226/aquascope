"""AquaScope's Bulletin 17C against the published Bulletin 17C examples.

Engineers trust a flood-frequency number when it matches the program they already use.
The reference cases here are the worked examples of Bulletin 17C itself (England et al.,
2018, Appendix 10), whose results were computed with USGS PeakFQ v7.1. HEC's own
documentation of the same data sets (HEC-SSP Examples, "Bulletin 17C | Expected Moments
Algorithm Examples") states that HEC-SSP produces the same results as PeakFQ for the first
six of them, so a match here is a match with HEC-SSP too.

Every number in :data:`CASES` was copied from the publication, table by table; nothing is
estimated. :func:`run` fits AquaScope's :func:`~aquascope.hydrology.flood_frequency.expected_moments_algorithm`
to each case's peaks and reports the differences, and :func:`markdown_table` renders them
for the docs page (a test checks the page carries exactly this table).

Sources:

- England, J. F. Jr., Cohn, T. A., Faber, B. A., Stedinger, J. R., Thomas, W. O. Jr., Veilleux,
  A. G., Kiang, J. E., and Mason, R. R. Jr. (2018). Guidelines for determining flood flow
  frequency, Bulletin 17C. U.S. Geological Survey Techniques and Methods, book 4, chap. B5.
  https://doi.org/10.3133/tm4B5
- HEC-SSP Examples 2.3, Bulletin 17C Examples:
  https://www.hec.usace.army.mil/confluence/sspdocs/sspexamples/latest/bulletin-17c-examples
"""

from __future__ import annotations

from typing import Any

import numpy as np

B17C_DOI = "https://doi.org/10.3133/tm4B5"
HEC_SSP_EXAMPLES = "https://www.hec.usace.army.mil/confluence/sspdocs/sspexamples/latest/bulletin-17c-examples"

#: The published cases. ``peaks`` in water-year order, ft3/s; ``published`` from the tables named in ``table``.
CASES: list[dict[str, Any]] = [
    {
        "id": "moose",
        "example": "Bulletin 17C Example 1 (systematic record)",
        "station": "USGS 01134500 Moose River at Victory, Vermont",
        "table": "Tables 10-2 (peaks) and 10-5 (quantiles); moments in the text of Example 1",
        "first_year": 1947,
        "peaks": [2080, 1670, 1480, 2940, 1560, 2380, 2720, 2860, 2620, 1710, 1370, 2180, 1160, 2780, 1580, 2110,
                  2160, 2750, 1190, 1560, 1800, 1600, 2400, 3010, 1490, 2920, 4940, 2550, 1250, 2670, 2020, 1460,
                  1620, 1460, 1570, 2890, 1840, 2950, 1380, 2350, 4180, 1700, 2200, 3430, 2270, 2180, 1900, 2760,
                  4536, 2160, 1860, 2680, 1540, 2110, 2950, 2410, 2230, 1980, 1610, 2640, 1930, 1940, 1810, 1900,
                  3140, 1370, 2180, 4250],
        "regional_skew": 0.44,
        "regional_skew_mse": 0.078,
        "published": {"mean": 3.3286, "std": 0.1403, "skew": 0.421, "skew_kind": "weighted",
                      "quantiles": {0.1: 3261, 0.04: 3911, 0.02: 4422, 0.01: 4957, 0.005: 5519, 0.002: 6313}},
    },
    {
        "id": "orestimba",
        "example": "Bulletin 17C Example 2 (potentially influential low floods)",
        "station": "USGS 11274500 Orestimba Creek near Newman, California",
        "table": "Tables 10-6 (peaks) and 10-9 (quantiles); moments and PILF threshold in the text of Example 2",
        # The station skew is negative: the text prints "-0.929 (station skew)" and figure 10-5 "-0.929 = skew (G)".
        "first_year": 1932,
        "peaks": [4260, 345, 516, 1320, 1200, 2180, 3230, 115, 3440, 3070, 1880, 6450, 1290, 5970, 782, 0, 0, 335,
                  175, 2920, 3660, 147, 0, 16, 5620, 1440, 10200, 5380, 448, 0, 1740, 8300, 156, 560, 128, 4200, 0,
                  5080, 1010, 584, 0, 1510, 922, 1010, 0, 0, 4360, 1270, 5210, 1130, 5550, 6360, 991, 50, 6990, 112,
                  0, 0, 4, 1260, 888, 4190, 12, 12000, 3130, 3320, 9470, 833, 2550, 958, 425, 2790, 2990, 1820, 1630,
                  0, 2110, 310, 4400, 4440, 0, 6250],
        "regional_skew": None,
        "regional_skew_mse": None,
        "published": {"mean": 3.0227, "std": 0.6821, "skew": -0.929, "skew_kind": "station",
                      "low_outlier_threshold": 782, "n_censored": 30,
                      "quantiles": {0.5: 1339, 0.2: 4026, 0.1: 6328, 0.04: 9426, 0.02: 11690, 0.01: 13820,
                                    0.005: 15800, 0.002: 18150}},
    },
]

#: Bulletin 17C examples not run here, and why.
NOT_RUN: list[dict[str, str]] = [
    {"example": "Example 3 (broken record), Back Creek near Jones Springs, WV",
     "why": "needs perception thresholds for the missing years; AquaScope's EMA does not take flow intervals yet"},
    {"example": "Example 4 (historical data), Arkansas River at Pueblo, CO",
     "why": "needs historical peaks and perception thresholds"},
    {"example": "Example 5 (crest-stage censored data), Bear Creek at Ottumwa, IA",
     "why": "needs censored flow intervals"},
    {"example": "Example 6 (historic data and low outliers), Santa Cruz River at Lochiel, AZ",
     "why": "needs historical peaks and perception thresholds"},
    {"example": "Example 7 (paleoflood data), American River at Fair Oaks, CA",
     "why": "needs paleoflood intervals"},
]


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Fit AquaScope's EMA to one case and compare it with the published values."""
    from aquascope.hydrology.flood_frequency import expected_moments_algorithm

    pub = case["published"]
    aeps = sorted(pub["quantiles"], reverse=True)
    rps = [1 / p for p in aeps]
    kw: dict[str, Any] = {}
    if case.get("regional_skew") is not None:
        kw = {"regional_skew": case["regional_skew"], "regional_skew_mse": case["regional_skew_mse"]}
    res = expected_moments_algorithm(np.asarray(case["peaks"], dtype=float),
                                     return_periods=[round(r) for r in rps], **kw)
    skew, mean, std = res.params
    rows = []
    for p, t in zip(aeps, rps):
        ours = float(res.return_periods[round(t)])
        ref = float(pub["quantiles"][p])
        rows.append({"aep": p, "published": ref, "aquascope": round(ours), "diff_pct": round(100 * (ours / ref - 1), 1)})
    return {
        "id": case["id"], "example": case["example"], "station": case["station"], "table": case["table"],
        "n_peaks": len(case["peaks"]),
        "moments": {"published": {"mean": pub["mean"], "std": pub["std"], "skew": pub["skew"]},
                    "aquascope": {"mean": round(float(mean), 4), "std": round(float(std), 4),
                                  "skew": round(float(skew), 3)}},
        "skew_kind": pub["skew_kind"],
        "low_outliers": {"published_threshold": pub.get("low_outlier_threshold"),
                         "published_censored": pub.get("n_censored"),
                         "aquascope_threshold": (None if res.low_outlier_threshold is None
                                                 else round(float(res.low_outlier_threshold), 1)),
                         "aquascope_censored": int(res.n_censored)},
        "quantiles": rows,
        "max_abs_diff_pct": max(abs(r["diff_pct"]) for r in rows),
    }


def run(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every reference case, plus the examples not run and why: a JSON-able report."""
    return {"source": B17C_DOI, "hec_ssp": HEC_SSP_EXAMPLES,
            "cases": [run_case(c) for c in (cases or CASES)], "not_run": NOT_RUN}


def markdown_table(report: dict[str, Any] | None = None) -> str:
    """The comparison as the Markdown tables the docs page carries."""
    report = report or run()
    out = []
    for c in report["cases"]:
        m = c["moments"]
        lo = c["low_outliers"]
        out += [f"**{c['example']}**: {c['station']}, {c['n_peaks']} peaks. Source: {c['table']}.", "",
                "| | Mean of log10 | Std of log10 | Skew |", "|---|---|---|---|",
                f"| Published ({c['skew_kind']} skew) | {m['published']['mean']} | {m['published']['std']} | "
                f"{m['published']['skew']} |",
                f"| AquaScope | {m['aquascope']['mean']} | {m['aquascope']['std']} | {m['aquascope']['skew']} |", ""]
        if lo["published_threshold"] is not None:
            out += [f"Low outliers: published threshold {lo['published_threshold']} ft3/s with "
                    f"{lo['published_censored']} peaks censored; AquaScope "
                    + (f"threshold {lo['aquascope_threshold']} ft3/s with {lo['aquascope_censored']} censored."
                       if lo["aquascope_threshold"] is not None
                       else f"found none ({lo['aquascope_censored']} zeros censored)."), ""]
        out += ["| AEP | Published (ft3/s) | AquaScope (ft3/s) | Difference |", "|---|---|---|---|"]
        out += [f"| {r['aep']} | {r['published']:,.0f} | {r['aquascope']:,.0f} | {r['diff_pct']:+.1f}% |"
                for r in c["quantiles"]]
        out.append("")
    return "\n".join(out)
