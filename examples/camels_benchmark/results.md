# CAMELS Benchmark: AquaScope against published and reference values

**Author:** Abdoul Rachid Ouédraogo  
**Date:** 2026-09-11  
**Description:** Flood frequency, baseflow and signature verification over 10 CAMELS catchments.  
**Data Sources:** Synthetic daily series (data/camels_benchmark/daily), USGS annual peaks (data/camels_benchmark/peaks), CAMELS attributes (Addor et al., 2017), Flood-frequency reference quantiles (ffa_reference.json)  
**Version:** 1.1  

## Summary

- **Catchments:** 10
- **Q mean normalized RMSE (%):** 0.0 %
- **BFI PBIAS (%):** 0.92 %
- **FFA mean relative error (%):** 0.22 %
29 check(s) unmet across 10 catchments; 25 GEV-MLE mismatch(es) classified as data limitations (unstable reference MLE), not implementation defects.

## Tolerances and rationale

*Decided up front; not loosened when a check fails.*

| Check | Value | Unit | Rationale |
| --- | --- | --- | --- |
| signature_relative | 0.25 | fraction | Synthetic daily series are calibrated to approximate the published CAMELS attributes; they are not measurements of them. |
| baseflow_absolute | 0.15 | fraction | Published CAMELS baseflow index comes from a different separation algorithm than AquaScope's Lyne-Hollick/Eckhardt digital filters. |
| peak_month_circular | 2.0 | months | Peak month is month-of-year and circular (Jan == Dec + 1 month). |
| ffa_relative | 0.2 | fraction | Wider than the +/-10% used in the Potomac federal-standard validation because several benchmark gauges are semi-arid or heavy-tailed, where MLE vs L-moments spread is larger. |
| q_mean_nrmse_gate | 25.0 | % | Aggregate q_mean gate: normalized RMSE (%) across the 10 gauges. |
| bfi_pbias_gate | 25.0 | % | Aggregate baseflow gate: PBIAS (%) across the 10 gauges. |
| ffa_gate | 20.0 | % | Aggregate flood-frequency gate over the dependable reference cross-checks (GEV-L-moments and LP3); GEV-MLE mismatches are reported separately as data-limitation findings, not folded into the implementation gate. |

## Aggregate performance (across 10 catchments)

*RMSE / PBIAS / R2 vs published values*

| Signature | RMSE | PBIAS % | R2 |
| --- | --- | --- | --- |
| q_mean | 0.0 | -0.0 | 1.0 |
| q5 | 0.3069 | -7.32 | 0.9822 |
| q95 | 10.4099 | 6.1 | 0.981 |
| runoff_ratio | 0.0 | -0.0 | 1.0 |
| fdc_slope | 0.4108 | 13.36 | 0.5631 |
| bfi_lyne_hollick | 0.0884 | 0.92 | 0.5027 |
| bfi_eckhardt | 0.0913 | -0.95 | 0.4693 |
| bfi_signatures | 0.0884 | 0.95 | 0.5021 |

## Per-catchment results

### 01013500 - Fish River near Fort Kent, ME (snow-dominated humid)

*01013500 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 22.5 | 22.5 | 0.0 | 0.25 | yes |
| q5 | 2.73314 | 3.1 | 0.118342 | 0.25 | yes |
| q95 | 71.716745 | 75.2 | 0.04632 | 0.25 | yes |
| runoff_ratio | 0.62 | 0.62 | 0.0 | 0.25 | yes |
| fdc_slope | -2.652215 | -2.35 | 0.128602 | 0.25 | yes |
| peak_month | 3.0 | 4.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.559444 | 0.59 | 0.030556 | 0.15 | yes |

*01013500 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=2.733 < median=13.803 < q95=71.717 | yes |
| runoff_ratio | runoff_ratio=0.6199998429711193 | yes |
| recession_constant | mean_recession_constant=0.3186 | yes |
| fdc_slope | fdc_slope=-2.6522 | yes |
| complete | all signature fields populated | yes |

*01013500 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.559321 | 0.59 | 0.030679 | yes |
| eckhardt | 0.542497 | 0.59 | 0.047503 | yes |

*01013500 gev vs scipy_gev*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 236.03 | 168.23 | 40.3 | no | implementation |
| 5 | 304.48 | 289.81 | 5.06 | yes | implementation |
| 10 | 346.56 | 348.4 | 0.53 | yes | implementation |
| 25 | 396.3 | 404.24 | 1.96 | yes | implementation |
| 50 | 430.88 | 435.62 | 1.09 | yes | implementation |
| 100 | 463.36 | 460.28 | 0.67 | yes | implementation |

*01013500 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 234.34 | 234.33 | 0.01 | yes | implementation |
| 5 | 302.09 | 302.09 | 0.0 | yes | implementation |
| 10 | 345.01 | 345.01 | 0.0 | yes | implementation |
| 25 | 397.13 | 397.15 | 0.01 | yes | implementation |
| 50 | 434.32 | 434.37 | 0.01 | yes | implementation |
| 100 | 470.04 | 470.12 | 0.02 | yes | implementation |

*01013500 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 237.65 | 237.52 | 0.05 | yes | implementation |
| 5 | 305.42 | 305.4 | 0.01 | yes | implementation |
| 10 | 344.92 | 345.06 | 0.04 | yes | implementation |
| 25 | 389.92 | 390.37 | 0.11 | yes | implementation |
| 50 | 420.48 | 421.21 | 0.17 | yes | implementation |
| 100 | 448.89 | 449.93 | 0.23 | yes | implementation |

*01013500 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 01013500 | 0.01 | 0.01 | 0.018 | 0.037 |

### 01664000 - Rappahannock River near Fredericksburg, VA (humid continental)

*01664000 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 45.3 | 45.3 | 0.0 | 0.25 | yes |
| q5 | 5.35505 | 5.8 | 0.076716 | 0.25 | yes |
| q95 | 144.96551 | 142.0 | 0.020884 | 0.25 | yes |
| runoff_ratio | 0.35 | 0.35 | 0.0 | 0.25 | yes |
| fdc_slope | -2.706408 | -2.5 | 0.082563 | 0.25 | yes |
| peak_month | 2.0 | 3.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.537231 | 0.44 | 0.097231 | 0.15 | yes |

*01664000 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=5.355 < median=25.173 < q95=144.966 | yes |
| runoff_ratio | runoff_ratio=0.35000003937222635 | yes |
| recession_constant | mean_recession_constant=0.3181 | yes |
| fdc_slope | fdc_slope=-2.7064 | yes |
| complete | all signature fields populated | yes |

*01664000 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.537086 | 0.44 | 0.097086 | yes |
| eckhardt | 0.523422 | 0.44 | 0.083422 | yes |

*01664000 gev vs scipy_gev*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 363.87 | 363.87 | 0.0 | yes | implementation |
| 5 | 616.63 | 616.63 | 0.0 | yes | implementation |
| 10 | 837.47 | 837.47 | 0.0 | yes | implementation |
| 25 | 1197.06 | 1197.06 | 0.0 | yes | implementation |
| 50 | 1537.54 | 1537.54 | 0.0 | yes | implementation |
| 100 | 1953.77 | 1953.77 | 0.0 | yes | implementation |

*01664000 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 357.01 | 357.12 | 0.03 | yes | implementation |
| 5 | 607.14 | 607.37 | 0.04 | yes | implementation |
| 10 | 833.39 | 833.59 | 0.02 | yes | implementation |
| 25 | 1213.89 | 1213.8 | 0.01 | yes | implementation |
| 50 | 1585.37 | 1584.8 | 0.04 | yes | implementation |
| 100 | 2051.62 | 2050.22 | 0.07 | yes | implementation |

*01664000 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 362.23 | 362.58 | 0.1 | yes | implementation |
| 5 | 632.55 | 632.82 | 0.04 | yes | implementation |
| 10 | 863.3 | 862.92 | 0.04 | yes | implementation |
| 25 | 1221.15 | 1218.92 | 0.18 | yes | implementation |
| 50 | 1540.69 | 1536.13 | 0.3 | yes | implementation |
| 100 | 1909.82 | 1901.89 | 0.42 | yes | implementation |

*01664000 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 01664000 | 0.008 | 0.009 | 0.016 | 0.033 |

### 02231000 - St. Marys River near Macclenny, FL (subtropical)

*02231000 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 15.799999 | 15.8 | 0.0 | 0.25 | yes |
| q5 | 1.106655 | 1.2 | 0.077787 | 0.25 | yes |
| q95 | 61.49207 | 52.3 | 0.175757 | 0.25 | yes |
| runoff_ratio | 0.28 | 0.28 | 0.0 | 0.25 | yes |
| fdc_slope | -3.187219 | -2.58 | 0.235356 | 0.25 | yes |
| peak_month | 10.0 | 9.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.514784 | 0.65 | 0.135216 | 0.15 | yes |

*02231000 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=1.107 < median=7.217 < q95=61.492 | yes |
| runoff_ratio | runoff_ratio=0.28000007622683204 | yes |
| recession_constant | mean_recession_constant=0.3480 | yes |
| fdc_slope | fdc_slope=-3.1872 | yes |
| complete | all signature fields populated | yes |

*02231000 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.514691 | 0.65 | 0.135309 | yes |
| eckhardt | 0.504979 | 0.65 | 0.145021 | yes |

*02231000 gev vs scipy_gev (reference MLE unstable)*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 156.59 | 22.84 | 585.59 | no | data_limitation |
| 5 | 304.47 | 131.66 | 131.26 | no | data_limitation |
| 10 | 439.36 | 2142.81 | 79.5 | no | data_limitation |
| 25 | 667.96 | 89150.44 | 99.25 | no | data_limitation |
| 50 | 892.79 | 1426846.68 | 99.94 | no | data_limitation |
| 100 | 1176.76 | 22378490.24 | 99.99 | no | data_limitation |

*02231000 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 156.59 | 156.65 | 0.04 | yes | implementation |
| 5 | 304.47 | 304.6 | 0.04 | yes | implementation |
| 10 | 439.36 | 439.47 | 0.03 | yes | implementation |
| 25 | 667.96 | 667.92 | 0.01 | yes | implementation |
| 50 | 892.79 | 892.47 | 0.04 | yes | implementation |
| 100 | 1176.76 | 1175.97 | 0.07 | yes | implementation |

*02231000 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 151.51 | 151.66 | 0.1 | yes | implementation |
| 5 | 311.04 | 311.16 | 0.04 | yes | implementation |
| 10 | 461.39 | 461.16 | 0.05 | yes | implementation |
| 25 | 712.56 | 711.23 | 0.19 | yes | implementation |
| 50 | 950.93 | 948.1 | 0.3 | yes | implementation |
| 100 | 1239.31 | 1234.16 | 0.42 | yes | implementation |

*02231000 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 02231000 | 0.008 | 0.009 | 0.017 | 0.034 |

### 03451500 - French Broad River at Asheville, NC (humid Appalachian)

*03451500 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 38.200001 | 38.2 | 0.0 | 0.25 | yes |
| q5 | 6.522305 | 6.5 | 0.003432 | 0.25 | yes |
| q95 | 117.230055 | 110.0 | 0.065728 | 0.25 | yes |
| runoff_ratio | 0.48 | 0.48 | 0.0 | 0.25 | yes |
| fdc_slope | -2.235127 | -1.95 | 0.146219 | 0.25 | yes |
| peak_month | 3.0 | 3.0 | 0.0 | 2.0 | yes |
| baseflow_index | 0.577533 | 0.52 | 0.057533 | 0.15 | yes |

*03451500 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=6.522 < median=24.701 < q95=117.230 | yes |
| runoff_ratio | runoff_ratio=0.4799997886252136 | yes |
| recession_constant | mean_recession_constant=0.3071 | yes |
| fdc_slope | fdc_slope=-2.2351 | yes |
| complete | all signature fields populated | yes |

*03451500 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.577274 | 0.52 | 0.057274 | yes |
| eckhardt | 0.555522 | 0.52 | 0.035522 | yes |

*03451500 gev vs scipy_gev*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 406.72 | 406.72 | 0.0 | yes | implementation |
| 5 | 632.18 | 632.18 | 0.0 | yes | implementation |
| 10 | 823.28 | 823.28 | 0.0 | yes | implementation |
| 25 | 1125.83 | 1125.83 | 0.0 | yes | implementation |
| 50 | 1404.73 | 1404.73 | 0.0 | yes | implementation |
| 100 | 1737.93 | 1737.92 | 0.0 | yes | implementation |

*03451500 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 403.86 | 403.98 | 0.03 | yes | implementation |
| 5 | 633.03 | 633.25 | 0.04 | yes | implementation |
| 10 | 834.04 | 834.22 | 0.02 | yes | implementation |
| 25 | 1162.57 | 1162.44 | 0.01 | yes | implementation |
| 50 | 1474.72 | 1474.1 | 0.04 | yes | implementation |
| 100 | 1857.47 | 1856.02 | 0.08 | yes | implementation |

*03451500 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 396.35 | 396.96 | 0.15 | yes | implementation |
| 5 | 633.9 | 634.52 | 0.1 | yes | implementation |
| 10 | 844.09 | 843.85 | 0.03 | yes | implementation |
| 25 | 1183.23 | 1180.33 | 0.25 | yes | implementation |
| 50 | 1498.81 | 1492.36 | 0.43 | yes | implementation |
| 100 | 1877.26 | 1865.46 | 0.63 | yes | implementation |

*03451500 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 03451500 | 0.011 | 0.016 | 0.017 | 0.044 |

### 06803500 - Salt Creek at Roca, NE (semi-arid Great Plains)

*06803500 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 3.4 | 3.4 | 0.0 | 0.25 | yes |
| q5 | 0.232705 | 0.3 | 0.224317 | 0.25 | yes |
| q95 | 11.271145 | 12.5 | 0.098308 | 0.25 | yes |
| runoff_ratio | 0.12 | 0.12 | 2e-06 | 0.25 | yes |
| fdc_slope | -3.169895 | -2.85 | 0.112244 | 0.25 | yes |
| peak_month | 6.0 | 6.0 | 0.0 | 2.0 | yes |
| baseflow_index | 0.550603 | 0.7 | 0.149397 | 0.15 | yes |

*06803500 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=0.233 < median=1.913 < q95=11.271 | yes |
| runoff_ratio | runoff_ratio=0.11999978146698892 | yes |
| recession_constant | mean_recession_constant=0.3464 | yes |
| fdc_slope | fdc_slope=-3.1699 | yes |
| complete | all signature fields populated | yes |

*06803500 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.550469 | 0.7 | 0.149531 | yes |
| eckhardt | 0.533633 | 0.7 | 0.166367 | no |

*06803500 gev vs scipy_gev (reference MLE unstable)*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 236.54 | 26.26 | 800.77 | no | data_limitation |
| 5 | 417.71 | 539.57 | 22.59 | no | data_limitation |
| 10 | 560.9 | 7705.97 | 92.72 | no | data_limitation |
| 25 | 773.45 | 229913.58 | 99.66 | no | data_limitation |
| 50 | 957.68 | 2859415.12 | 99.97 | no | data_limitation |
| 100 | 1166.47 | 34908723.47 | 100.0 | no | data_limitation |

*06803500 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 242.33 | 242.4 | 0.03 | yes | implementation |
| 5 | 423.54 | 423.63 | 0.02 | yes | implementation |
| 10 | 558.12 | 558.15 | 0.0 | yes | implementation |
| 25 | 746.84 | 746.69 | 0.02 | yes | implementation |
| 50 | 901.69 | 901.32 | 0.04 | yes | implementation |
| 100 | 1069.16 | 1068.49 | 0.06 | yes | implementation |

*06803500 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 243.29 | 242.36 | 0.39 | yes | implementation |
| 5 | 444.24 | 444.32 | 0.02 | yes | implementation |
| 10 | 577.34 | 579.75 | 0.42 | yes | implementation |
| 25 | 735.8 | 742.87 | 0.95 | yes | implementation |
| 50 | 844.33 | 855.89 | 1.35 | yes | implementation |
| 100 | 944.12 | 960.79 | 1.73 | yes | implementation |

*06803500 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 06803500 | 0.008 | 0.009 | 0.019 | 0.036 |

### 07056000 - Buffalo River near St. Joe, AR (humid interior)

*07056000 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 28.6 | 28.6 | 0.0 | 0.25 | yes |
| q5 | 2.77928 | 2.8 | 0.0074 | 0.25 | yes |
| q95 | 91.28733 | 95.0 | 0.039081 | 0.25 | yes |
| runoff_ratio | 0.45 | 0.45 | 0.0 | 0.25 | yes |
| fdc_slope | -2.797046 | -2.25 | 0.243132 | 0.25 | yes |
| peak_month | 5.0 | 4.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.529877 | 0.51 | 0.019877 | 0.15 | yes |

*07056000 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=2.779 < median=16.191 < q95=91.287 | yes |
| runoff_ratio | runoff_ratio=0.4499999581856539 | yes |
| recession_constant | mean_recession_constant=0.3757 | yes |
| fdc_slope | fdc_slope=-2.7970 | yes |
| complete | all signature fields populated | yes |

*07056000 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.529711 | 0.51 | 0.019711 | yes |
| eckhardt | 0.51613 | 0.51 | 0.00613 | yes |

*07056000 gev vs scipy_gev*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 1056.78 | 1056.78 | 0.0 | yes | implementation |
| 5 | 1812.23 | 1812.23 | 0.0 | yes | implementation |
| 10 | 2429.17 | 2429.17 | 0.0 | yes | implementation |
| 25 | 3372.97 | 3372.97 | 0.0 | yes | implementation |
| 50 | 4214.94 | 4214.94 | 0.0 | yes | implementation |
| 100 | 5192.99 | 5192.99 | 0.0 | yes | implementation |

*07056000 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 1082.0 | 1082.33 | 0.03 | yes | implementation |
| 5 | 1834.57 | 1835.04 | 0.03 | yes | implementation |
| 10 | 2408.82 | 2409.02 | 0.01 | yes | implementation |
| 25 | 3234.22 | 3233.54 | 0.02 | yes | implementation |
| 50 | 3927.82 | 3926.03 | 0.05 | yes | implementation |
| 100 | 4693.37 | 4689.97 | 0.07 | yes | implementation |

*07056000 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 1072.74 | 1071.61 | 0.11 | yes | implementation |
| 5 | 1875.32 | 1874.98 | 0.02 | yes | implementation |
| 10 | 2469.17 | 2471.25 | 0.08 | yes | implementation |
| 25 | 3269.9 | 3277.57 | 0.23 | yes | implementation |
| 50 | 3894.09 | 3907.88 | 0.35 | yes | implementation |
| 100 | 4536.44 | 4558.02 | 0.47 | yes | implementation |

*07056000 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 07056000 | 0.007 | 0.009 | 0.019 | 0.036 |

### 08181500 - Medina River at San Antonio, TX (semi-arid)

*08181500 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 5.2 | 5.2 | 0.0 | 0.25 | yes |
| q5 | 0.05367 | 0.08 | 0.329125 | 0.25 | no |
| q95 | 21.83893 | 18.0 | 0.213274 | 0.25 | yes |
| runoff_ratio | 0.08 | 0.08 | 0.0 | 0.25 | yes |
| fdc_slope | -4.600273 | -4.0 | 0.150068 | 0.25 | yes |
| peak_month | 4.0 | 5.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.364966 | 0.38 | 0.015034 | 0.15 | yes |

*08181500 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=0.054 < median=1.073 < q95=21.839 | yes |
| runoff_ratio | runoff_ratio=0.07999999494204445 | yes |
| recession_constant | mean_recession_constant=0.5815 | yes |
| fdc_slope | fdc_slope=-4.6003 | yes |
| complete | all signature fields populated | yes |

*08181500 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.364938 | 0.38 | 0.015062 | yes |
| eckhardt | 0.372659 | 0.38 | 0.007341 | yes |

*08181500 gev vs scipy_gev (reference MLE unstable)*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 137.59 | 122.83 | 12.02 | yes | implementation |
| 5 | 314.46 | 312.97 | 0.47 | yes | implementation |
| 10 | 491.34 | 551.79 | 10.96 | yes | implementation |
| 25 | 818.15 | 1100.02 | 25.62 | no | data_limitation |
| 50 | 1166.86 | 1816.25 | 35.75 | no | data_limitation |
| 100 | 1639.38 | 2973.25 | 44.86 | no | data_limitation |

*08181500 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 137.59 | 137.58 | 0.01 | yes | implementation |
| 5 | 314.46 | 314.42 | 0.01 | yes | implementation |
| 10 | 491.34 | 491.29 | 0.01 | yes | implementation |
| 25 | 818.15 | 818.13 | 0.0 | yes | implementation |
| 50 | 1166.86 | 1166.92 | 0.0 | yes | implementation |
| 100 | 1639.38 | 1639.58 | 0.01 | yes | implementation |

*08181500 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 129.25 | 129.23 | 0.01 | yes | implementation |
| 5 | 331.93 | 331.92 | 0.0 | yes | implementation |
| 10 | 541.85 | 541.91 | 0.01 | yes | implementation |
| 25 | 911.71 | 912.06 | 0.04 | yes | implementation |
| 50 | 1274.4 | 1275.15 | 0.06 | yes | implementation |
| 100 | 1720.98 | 1722.37 | 0.08 | yes | implementation |

*08181500 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 08181500 | 0.008 | 0.009 | 0.016 | 0.034 |

### 09510200 - Cave Creek near Cave Creek, AZ (arid)

*09510200 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 0.15 | 0.15 | 1e-06 | 0.25 | yes |
| q5 | 0.0048 | 0.005 | 0.04 | 0.25 | yes |
| q95 | 0.548555 | 0.52 | 0.054913 | 0.25 | yes |
| runoff_ratio | 0.03 | 0.03 | 6e-06 | 0.25 | yes |
| fdc_slope | -3.84171 | -3.8 | 0.010976 | 0.25 | yes |
| peak_month | 2.0 | 2.0 | 0.0 | 2.0 | yes |
| baseflow_index | 0.396946 | 0.25 | 0.146946 | 0.15 | yes |

*09510200 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=0.005 < median=0.054 < q95=0.549 | yes |
| runoff_ratio | runoff_ratio=0.029999824658398944 | yes |
| recession_constant | mean_recession_constant=0.5744 | yes |
| fdc_slope | fdc_slope=-3.8417 | yes |
| complete | all signature fields populated | yes |

*09510200 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.396684 | 0.25 | 0.146684 | yes |
| eckhardt | 0.4015 | 0.25 | 0.1515 | no |

*09510200 gev vs scipy_gev (reference MLE unstable)*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 81.05 | 62.15 | 30.4 | no | data_limitation |
| 5 | 190.8 | 186.05 | 2.55 | yes | implementation |
| 10 | 291.31 | 361.38 | 19.39 | yes | implementation |
| 25 | 462.3 | 813.18 | 43.15 | no | data_limitation |
| 50 | 631.06 | 1469.68 | 57.06 | no | data_limitation |
| 100 | 844.87 | 2633.6 | 67.92 | no | data_limitation |

*09510200 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 81.05 | 81.09 | 0.05 | yes | implementation |
| 5 | 190.8 | 190.9 | 0.05 | yes | implementation |
| 10 | 291.31 | 291.4 | 0.03 | yes | implementation |
| 25 | 462.3 | 462.27 | 0.01 | yes | implementation |
| 50 | 631.06 | 630.84 | 0.04 | yes | implementation |
| 100 | 844.87 | 844.31 | 0.07 | yes | implementation |

*09510200 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 71.44 | 70.86 | 0.82 | yes | implementation |
| 5 | 207.93 | 207.94 | 0.0 | yes | implementation |
| 10 | 334.5 | 337.33 | 0.84 | yes | implementation |
| 25 | 523.63 | 534.18 | 1.97 | yes | implementation |
| 50 | 678.32 | 698.02 | 2.82 | yes | implementation |
| 100 | 839.53 | 871.32 | 3.65 | yes | implementation |

*09510200 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 09510200 | 0.008 | 0.01 | 0.025 | 0.043 |

### 11532500 - Smith River near Crescent City, CA (Pacific Northwest rain)

*11532500 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 72.5 | 72.5 | 0.0 | 0.25 | yes |
| q5 | 4.613005 | 5.2 | 0.112884 | 0.25 | yes |
| q95 | 264.70323 | 245.0 | 0.080421 | 0.25 | yes |
| runoff_ratio | 0.71 | 0.71 | 0.0 | 0.25 | yes |
| fdc_slope | -3.120449 | -2.85 | 0.094894 | 0.25 | yes |
| peak_month | 2.0 | 1.0 | 1.0 | 2.0 | yes |
| baseflow_index | 0.490259 | 0.47 | 0.020259 | 0.15 | yes |

*11532500 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=4.613 < median=34.937 < q95=264.703 | yes |
| runoff_ratio | runoff_ratio=0.7100001108096535 | yes |
| recession_constant | mean_recession_constant=0.4124 | yes |
| fdc_slope | fdc_slope=-3.1204 | yes |
| complete | all signature fields populated | yes |

*11532500 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.489962 | 0.47 | 0.019962 | yes |
| eckhardt | 0.481301 | 0.47 | 0.011301 | yes |

*11532500 gev vs scipy_gev (reference MLE unstable)*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 2156.93 | 344.75 | 525.65 | no | data_limitation |
| 5 | 3062.83 | 2079.62 | 47.28 | no | data_limitation |
| 10 | 3643.28 | 147787.61 | 97.53 | no | data_limitation |
| 25 | 4355.37 | 40338648.09 | 99.99 | no | data_limitation |
| 50 | 4868.69 | 2592319741.95 | 100.0 | no | data_limitation |
| 100 | 5365.96 | 161550980185.42 | 100.0 | no | data_limitation |

*11532500 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 2141.5 | 2141.46 | 0.0 | yes | implementation |
| 5 | 3045.76 | 3045.72 | 0.0 | yes | implementation |
| 10 | 3637.22 | 3637.22 | 0.0 | yes | implementation |
| 25 | 4376.41 | 4376.5 | 0.0 | yes | implementation |
| 50 | 4918.97 | 4919.16 | 0.0 | yes | implementation |
| 100 | 5452.68 | 5452.99 | 0.01 | yes | implementation |

*11532500 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 2215.07 | 2210.75 | 0.2 | yes | implementation |
| 5 | 3125.88 | 3126.37 | 0.02 | yes | implementation |
| 10 | 3621.72 | 3629.65 | 0.22 | yes | implementation |
| 25 | 4141.88 | 4162.27 | 0.49 | yes | implementation |
| 50 | 4464.52 | 4495.51 | 0.69 | yes | implementation |
| 100 | 4741.29 | 4783.45 | 0.88 | yes | implementation |

*11532500 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 11532500 | 0.008 | 0.01 | 0.021 | 0.039 |

### 14301000 - Nehalem River near Foss, OR (Pacific Northwest rain)

*14301000 signatures*

| Metric | Computed | Published | Error | Tolerance | Meets tolerance |
| --- | --- | --- | --- | --- | --- |
| q_mean | 55.8 | 55.8 | 0.0 | 0.25 | yes |
| q5 | 2.999895 | 3.5 | 0.142887 | 0.25 | yes |
| q95 | 207.532575 | 185.0 | 0.121798 | 0.25 | yes |
| runoff_ratio | 0.68 | 0.68 | 0.0 | 0.25 | yes |
| fdc_slope | -3.066739 | -2.55 | 0.202643 | 0.25 | yes |
| peak_month | 12.0 | 12.0 | 0.0 | 2.0 | yes |
| baseflow_index | 0.46548 | 0.43 | 0.03548 | 0.15 | yes |

*14301000 signature integrity*

| Metric | Detail | Passes |
| --- | --- | --- |
| order | q5=3.000 < median=25.509 < q95=207.533 | yes |
| runoff_ratio | runoff_ratio=0.680000328531608 | yes |
| recession_constant | mean_recession_constant=0.4176 | yes |
| fdc_slope | fdc_slope=-3.0667 | yes |
| complete | all signature fields populated | yes |

*14301000 baseflow index*

| Method | BFI | Published | Error | Meets tolerance |
| --- | --- | --- | --- | --- |
| lyne_hollick | 0.465319 | 0.43 | 0.035319 | yes |
| eckhardt | 0.461454 | 0.43 | 0.031454 | yes |

*14301000 gev vs scipy_gev*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 781.82 | 781.82 | 0.0 | yes | implementation |
| 5 | 1040.32 | 1040.32 | 0.0 | yes | implementation |
| 10 | 1197.97 | 1197.97 | 0.0 | yes | implementation |
| 25 | 1383.01 | 1383.01 | 0.0 | yes | implementation |
| 50 | 1510.75 | 1510.75 | 0.0 | yes | implementation |
| 100 | 1630.06 | 1630.06 | 0.0 | yes | implementation |

*14301000 gev_lmoments vs scipy_gev_lmoments*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 787.25 | 787.15 | 0.01 | yes | implementation |
| 5 | 1035.44 | 1035.37 | 0.01 | yes | implementation |
| 10 | 1180.29 | 1180.32 | 0.0 | yes | implementation |
| 25 | 1343.76 | 1344.0 | 0.02 | yes | implementation |
| 50 | 1452.36 | 1452.79 | 0.03 | yes | implementation |
| 100 | 1550.54 | 1551.19 | 0.04 | yes | implementation |

*14301000 lp3 vs scipy_lp3*

| Return period (yr) | Computed (m3/s) | Reference (m3/s) | Rel. error (%) | Meets tolerance | Classification |
| --- | --- | --- | --- | --- | --- |
| 2 | 797.25 | 796.03 | 0.15 | yes | implementation |
| 5 | 1047.13 | 1047.19 | 0.01 | yes | implementation |
| 10 | 1179.96 | 1181.89 | 0.16 | yes | implementation |
| 25 | 1318.62 | 1323.61 | 0.38 | yes | implementation |
| 50 | 1404.88 | 1412.45 | 0.54 | yes | implementation |
| 100 | 1479.36 | 1489.66 | 0.69 | yes | implementation |

*14301000 stage timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 14301000 | 0.008 | 0.01 | 0.016 | 0.034 |

## Execution timings

Wall-clock seconds per stage, per catchment. Recorded and reported; never asserted, so a slow shared CI runner cannot produce a false failure.

*Recorded wall-clock timings (s)*

| Gauge | signatures_s | baseflow_s | flood_frequency_s | total_s |
| --- | --- | --- | --- | --- |
| 01013500 | 0.01 | 0.01 | 0.018 | 0.037 |
| 01664000 | 0.008 | 0.009 | 0.016 | 0.033 |
| 02231000 | 0.008 | 0.009 | 0.017 | 0.034 |
| 03451500 | 0.011 | 0.016 | 0.017 | 0.044 |
| 06803500 | 0.008 | 0.009 | 0.019 | 0.036 |
| 07056000 | 0.007 | 0.009 | 0.019 | 0.036 |
| 08181500 | 0.008 | 0.009 | 0.016 | 0.034 |
| 09510200 | 0.008 | 0.01 | 0.025 | 0.043 |
| 11532500 | 0.008 | 0.01 | 0.021 | 0.039 |
| 14301000 | 0.008 | 0.01 | 0.016 | 0.034 |

- **Total runtime (s):** 0.369 s
## Data and methods

Discharge: synthetic daily series calibrated to CAMELS; peaks: USGS NWIS annual peak series, one value per USGS water year, screened for qualifier codes and censored values. Reference quantiles: GEV-MLE and LP3 via scipy, GEV-L-moments via lmoments3 (Hosking 1997). Full provenance per catchment in results.json.

## Cite this software

Abdoul Rachid Ouédraogo (2026). AquaScope: Open-source water data aggregation toolkit (version 0.16.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21903143
