# Potomac flood frequency (Bulletin 17C)

A full Bulletin 17C flood-frequency analysis on **USGS gauge 01646500** (Potomac River near Washington, DC, at the Little Falls Pump Station), validated against the FEMA Flood Insurance Study.

Runtime: ~3 minutes (cold cache). Source notebook: [`Rekin226/aquascope-demos`](https://github.com/Rekin226/aquascope-demos/tree/main/01_potomac_flood_frequency).

---

## The scenario

USGS gauge **01646500** has one of the longest continuous peak-flow records in the eastern United States. Annual peaks go back to **1930**, including the historic **March 1936 flood** (≈ 484,000 cfs). It's a textbook reference site for **Bulletin 17C flood frequency analysis**, the federal standard used by FEMA, USACE, and state DOTs to set design floods for bridges, levees, and floodplain maps.

This case fits a **Log-Pearson Type III (LP3)** distribution to the annual peak series and a **GEV** for comparison, reports 10 / 50 / 100 / 500-year flood estimates with bootstrap confidence intervals, and validates the result against the FEMA Flood Insurance Study estimates for the same gauge.

---

## What this shows about AquaScope

- **`flood_analysis(method="lp3")`** runs the federal-standard LP3 fit (method-of-moments + weighted regional skew per Bulletin 17C § 5.2.4) and returns return-period estimates with variance-of-estimate confidence intervals.
- **`flood_analysis(method="gev_lmoments")`** fits the GEV alternative via L-moments. Same call signature, swap one string. L-moments are a standard estimator for GEV on annual maxima and are stable on records with historical outliers (e.g. the 1936 peak). Since 0.10.0 `method="gev"` also seeds its maximum-likelihood fit from L-moments and constrains the shape parameter, so the two agree far more closely than they used to.
- **Q-Q and P-P diagnostics** assemble directly from `FloodFreqResult.params` and `annual_max`. These are the plots reviewers expect on any flood-frequency paper.

---

## Run it

```python
from dataretrieval import nwis            # USGS's official client
from aquascope.api import flood_analysis
from aquascope.viz import qq_diagnostic_plot

# 1. Fetch the annual peak series (1931–2025, n = 80)
peaks = nwis.get_record(sites="01646500", service="peaks")
annual_max = peaks["peak_va"].dropna()

# 2. Bulletin 17C LP3 fit with weighted regional skew
lp3 = flood_analysis(
    annual_max,
    method="lp3",
    return_periods=[10, 50, 100, 500],
    regional_skew=0.4,            # weighted regional skew per Bulletin 17C
    ci_level=0.90,
)

# 3. GEV via L-moments (robust comparison fit)
gev = flood_analysis(annual_max, method="gev_lmoments", return_periods=[10, 50, 100, 500])

# 4. Save results
lp3.return_levels.to_csv("outputs/return_periods.csv", index=False)
qq_diagnostic_plot(lp3).savefig("outputs/qq_diagnostic.png", dpi=150)
```

Data is fetched via the `dataretrieval` package (USGS's official Python client) because AquaScope's `USGSCollector` targets the new OGC API, which does not yet expose historical annual peaks.

---

## Verified results

Run on 2026-08-12, n = 95 annual peaks (1931–2025):

| Return period |   LP3 (Bulletin 17C) |   LP3 90 % CI  | GEV (L-moments) |
| :-----------: | -------------------: | :------------: | --------------: |
|     10 yr     |          236,820 cfs | [207k, 271k]   |     221,373 cfs |
|     50 yr     |          374,509 cfs | [312k, 449k]   |     358,301 cfs |
|  **100 yr**   |      **443,154 cfs** | **[363k, 542k]** | **431,987 cfs** |
|    500 yr     |          629,636 cfs | [494k, 803k]   |     649,585 cfs |

The 1936 St. Patrick's Day flood (484,000 cfs) sits within the LP3 100-year 90 % confidence interval, consistent with the historic event being approximately a 100-year flood.

---

## Validation against FEMA

**Reference:** FEMA Flood Insurance Study, District of Columbia, Washington D.C., FIS Number 110001V000A, revised 2010-09-27 ([msc.fema.gov FIS](https://map1.msc.fema.gov/data/11/S/PDF/110001V000A.pdf) · also at [ncpc.gov](https://www.ncpc.gov/docs/DC_Flood_Insurance_Study_Pre-17th_Street_Levee.pdf)), **Table 4, page 15**: the 2005 USACE Bulletin 17B / LP3 update of the effective 1985 hydrology for gauge 01646500.

The FIS used Bulletin 17B / HEC-FFA with data through 2003 (~73 annual peaks plus the 1889 historical peak), station skew 0.3, no regional weighting. This case extends the record through 2025 (80 annual peaks) and uses Bulletin 17C-style weighted regional skew (0.4). We consider the result correct if every return period (10, 50, 100, 500 yr) falls within ±10 % of the FIS value.

| Return period | LP3 here (1931–2025) | GEV here (1931–2025) | FIS 2010 (1931–2003) | LP3 Δ % | GEV Δ % | Both within ±10 % |
| :-----------: | -------------------: | -------------------: | -------------------: | :-----: | :-----: | :---------------: |
|     10 yr     |          236,820 cfs |          221,373 cfs |          240,000 cfs |  -1.3 % |  -7.8 % |         ✅        |
|     50 yr     |          374,509 cfs |          358,301 cfs |          395,000 cfs |  -5.2 % |  -9.3 % |         ✅        |
|  **100 yr**   |      **443,154 cfs** |      **431,987 cfs** |      **475,000 cfs** | **-6.7 %** | **-9.1 %** |      ✅        |
|    500 yr     |          629,636 cfs |          649,585 cfs |          698,000 cfs |  -9.8 % |  -6.9 % |         ✅        |

**Overall: PASS.** The small systematic underestimate at long return periods is expected. The FIS censors the 1889 historical peak into the analysis (raising the upper tail), while this run uses only the 1931–2025 systematic record with weighted regional skew.

Both estimators are checked against the reference on purpose. Before 0.10.0 only LP3 was validated here, and a sign error in the GEV L-moment estimator left that column reading 54 % below the FIS at the 500-year level for nine releases without anything failing.

---

## Outputs

In the demo case repo at [`Rekin226/aquascope-demos`](https://github.com/Rekin226/aquascope-demos/tree/main/01_potomac_flood_frequency/outputs):

- `flood_frequency_curve.png`: return level vs. return period, LP3 with 90 % CI band, GEV overlaid.
- `qq_diagnostic.png`: Q-Q and probability plots for the LP3 fit.
- `return_periods.csv`: numeric table of return period, LP3 estimate, GEV estimate, lower/upper CI.
- `annual_peaks.csv`: the cleaned annual-peak series fetched from USGS NWIS (provenance-traceable).

---

## Next steps

- **[Try it yourself](https://github.com/Rekin226/aquascope-demos/tree/main/01_potomac_flood_frequency)**: full Jupyter notebook with plots.
- **[Theory guide](../theory.md)**: equations, decision tree, and DOI references for Bulletin 17C and the GEV / LP3 / Gumbel alternatives.
- **[Methodology matrix](../methodology_matrix.md)**: when to pick LP3 vs. GEV vs. non-stationary GEV vs. EMA.
- **[`flood_analysis` API reference](../api.md#aquascope.api)**: every parameter, return type, and option.
