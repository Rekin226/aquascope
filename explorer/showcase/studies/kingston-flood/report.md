# 100-Year Design Discharge for the Thames at Kingston Road Bridge

**Author:** AquaScope Studio  
**Date:** 2026-09-14  
**Description:** set the 100-year design discharge (with uncertainty bounds) for the new road bridge over the Thames at Kingston  
**Data Sources:** BasinATLAS (HydroATLAS v1.0), ERA5 via Open-Meteo, similar_basins, uk_ea  
**Version:** 1.0  

**Site:** 51.4150 N, 0.3080 W

**Answer.** Notice: the Critic's fix requests on limitations were not all resolved; read the report with the list of what this study does not establish.

The 100-year design discharge for the new road bridge over the Thames at Kingston is set at 652.5 m3/s (GEV, L-moments), with a 90% uncertainty band of 565.4 to 723.1 m3/s -- grade: established. This comes from the UK Environment Agency gauge at Kingston (station 8496ce69-482c-406a-a2f0-ac418ef8f099), a 142.9-year daily discharge record (1883-10-01 to 2026-09-09). An independent Log-Pearson III fit on the same record gives 624.0 m3/s (90% CI 573.2 to 679.4 m3/s), 4% below the GEV value, corroborating it. The record's largest observed flood, 800 m3/s in 1894, sits 18% above the GEV curve at its empirical return period, within the allowed envelope but a reminder that the tail is data-limited.

*Key numbers*

| Quantity | Value | Unit | Step |
| --- | --- | --- | --- |
| Upstream area | 9991.0 | km2 | s1 |
| Record length | 142.9 | years | s3 |
| Mean of the record | 65.52 | m3/s | s2 |
| 100-year return level, GEV (L-moments) | 652.5 | m3/s | s3 |
| 100-year return level, Log-Pearson III | 624.0 | m3/s | s3 |
| 100-year LP3 90 % interval, low | 573.2 | m3/s | s3 |
| 100-year LP3 90 % interval, high | 679.4 | m3/s | s3 |
| Q95 (exceeded 95 % of days) | 7.52 | m3/s | s2 |
| Q50 (median flow) | 39.9 | m3/s | s2 |
| Q10 | 162.0 | m3/s | s2 |
| Mann-Kendall p-value (annual mean) | 0.9411 |  | s2 |
| Sen's slope | -0.0029 | m3/s per year | s2 |
| 2-year return level, GEV (L-moments) | 307.9 | m3/s | s2 |
| 2-year return level, Log-Pearson III | 311.8 | m3/s | s2 |
| 5-year return level, GEV (L-moments) | 403.0 | m3/s | s2 |
| 5-year return level, Log-Pearson III | 407.1 | m3/s | s2 |
| 10-year return level, GEV (L-moments) | 464.8 | m3/s | s2 |
| 10-year return level, Log-Pearson III | 464.7 | m3/s | s2 |
| 25-year return level, GEV (L-moments) | 541.6 | m3/s | s2 |
| 25-year return level, Log-Pearson III | 532.2 | m3/s | s2 |
| 50-year return level, GEV (L-moments) | 597.6 | m3/s | s2 |
| 50-year return level, Log-Pearson III | 579.3 | m3/s | s2 |
| 100-year GEV bootstrap 90 % interval, low | 565.4 | m3/s | s3 |
| 100-year GEV bootstrap 90 % interval, high | 723.1 | m3/s | s3 |
| ERA5 precipitation | 651.8 | mm per year | s4 |
| ERA5 reference evapotranspiration | 708.7 | mm per year | s4 |
| Aridity index | 0.9198 |  | s4 |
| GloFAS mean discharge (cell) | 0.8471 | m3/s | s4 |
| mean daily flow | 0.5752 | mm/d | s5 |
| mean daily flow band, low | 0.3158 | mm/d | s5 |
| mean daily flow band, high | 1.048 | mm/d | s5 |
| low flow: exceeded 95 % of days | 0.1608 | mm/d | s5 |
| low flow: exceeded 95 % of days band, low | 0.0706 | mm/d | s5 |
| low flow: exceeded 95 % of days band, high | 0.3663 | mm/d | s5 |
| high flow: exceeded 5 % of days | 1.512 | mm/d | s5 |
| high flow: exceeded 5 % of days band, low | 0.872 | mm/d | s5 |
| high flow: exceeded 5 % of days band, high | 2.621 | mm/d | s5 |
| mean annual daily maximum | 3.049 | mm/d | s5 |
| mean annual daily maximum band, low | 1.325 | mm/d | s5 |
| mean annual daily maximum band, high | 7.02 | mm/d | s5 |
| mean flow / BasinATLAS precipitation | 0.349 | - | s5 |
| mean flow / BasinATLAS precipitation band, low | 0.2185 | - | s5 |
| mean flow / BasinATLAS precipitation band, high | 0.4795 | - | s5 |
| baseflow / total flow | 0.8078 | - | s5 |
| baseflow / total flow band, low | 0.7251 | - | s5 |
| baseflow / total flow band, high | 0.8904 | - | s5 |

## Summary

The design flow is fixed by an at-site flood-frequency analysis on the UK EA Kingston gauge (8496ce69-482c-406a-a2f0-ac418ef8f099), 142.9 years of daily discharge (1883-10-01 to 2026-09-09, n=51947). The GEV (L-moments) 100-year return level is 652.5 m3/s, with a 90% bootstrap CI of 565.4 to 723.1 m3/s. Log-Pearson III gives 624.0 m3/s (90% CI 573.2 to 679.4 m3/s), a 4% spread from GEV, well inside the 25% disagreement threshold, so the two fits corroborate rather than contradict. Mann-Kendall tests on the annual mean (p=0.9411) and annual maxima (p=0.1784) show no trend, supporting the stationarity assumption. A GloFAS cross-check at the same coordinates failed its gate (ratio 0.02, far outside the allowed factor 1.50) but its mean discharge of 0.85 m3/s versus the gauge's 65.52 m3/s indicates a catchment-representation mismatch in the model grid cell, not evidence against the at-site fit. Regionalised flow signatures from 10 similar gauges corroborate the general flow regime qualitatively but were not designed to produce a T=100 quantile.

## The decision

Decide the 100-year design discharge using the GEV (L-moments) fit on the Kingston gauge record: 652.5 m3/s, band 565.4 to 723.1 m3/s (90% bootstrap CI), grade established. This is conditioned on: the 142.9-year record being effectively at-site; no upstream regulation (BasinATLAS degree-of-regulation 0%, reservoir volume 0 million m3); confirmed stationarity (no trend in annual mean or maxima); and agreement between GEV and Log-Pearson III (624.0 m3/s) within 4%, well under the 25% disagreement threshold, so GEV is taken as central rather than averaging the two. The GloFAS cross-check is excluded from the decision because it fails its gate for an evident scale-mismatch reason, not a real conflict. What would change this: resolving the GloFAS grid-cell mismatch to obtain a genuine independent check; evidence of upstream regulation or urbanisation not captured in the static BasinATLAS field; or a non-stationary/climate-adjusted re-fit, which was not run here and is only a stated overlay caveat.

## Findings

Finding f1 (established): an independent Log-Pearson III fit on the Kingston gauge record gives a 100-year value of 624.0 m3/s, 4% below the GEV (L-moments) estimate of 652.5 m3/s -- both from station 8496ce69-482c-406a-a2f0-ac418ef8f099, well inside the 25% spread tolerance, so the two distributional families agree (established). Consistency check: the record maximum of 800 m3/s (1894) exceeds the GEV fit's value at that empirical return period (679.3 m3/s) by 18%, inside the 25% envelope but on the high side (established, with caution on tail reliance). Consistency check: the GloFAS cross-check 100-year value (14.77 m3/s) disagrees with the at-site estimate by two orders of magnitude and fails its gate; this is attributed to a catchment-representation mismatch in the GloFAS grid cell (mean discharge 0.85 m3/s versus the gauge's 65.52 m3/s), so it is set aside as not established rather than treated as contradicting the gauge record.

## Problem and decision

The task is to set the 100-year design discharge, with its uncertainty band, for a new road bridge over the Thames at Kingston (51.415N, -0.308E), so that the structure and its waterway opening are sized against a defensible flood estimate rather than a single point value.

## Site and data

The catchment upstream of the site (BasinATLAS, HydroATLAS v1.0) covers 9990.7 to 9991.0 km2, mean elevation 109.0 m, mean slope 2.0 degrees, mean annual precipitation 684.0 mm/yr, PET 695.0 mm/yr, aridity index 0.98, mean temperature 9.6 degC. Land cover is 44% cropland, 22% urban, 16% pasture, 2% forest. Degree of regulation by reservoirs is 0.0% and reservoir volume upstream is 0.0 million m3, confirming no upstream dams to adjust for. Mean annual natural discharge at the outlet is given as 84.65 m3/s. The gauge used, UK EA Kingston (8496ce69-482c-406a-a2f0-ac418ef8f099), is essentially at this point.

## Methodology

The design flow follows the flood_risk playbook in five steps. First, the catchment upstream of the site is characterised via BasinATLAS to confirm its size (9990.7 km2, 79 level-12 sub-basins) and the absence of impounding dams (degree of regulation 0.0%, reservoir volume 0.0 million m3), so the flood record can be treated as natural. Second, a Mann-Kendall trend pre-test is run on both the annual mean and the annual maxima of the Kingston gauge record to confirm stationarity before any flood-frequency model is fitted. Third, GEV (L-moments) and Log-Pearson III distributions are fitted to the 140 years of annual maxima at the Kingston gauge, with a maximum-likelihood GEV re-fit and 1,000-resample bootstrap confidence interval, and the T=100 year quantile is quoted from both fits together with their spread; a spread above 25% would be reported as disagreement rather than averaged away. Fourth, the at-site 100-year estimate is cross-checked against an independent GloFAS-modelled discharge frequency fit for the same coordinates, treated as indicative context rather than a design input. Fifth, the flow regime is corroborated against regionalised flow signatures transferred from the 10 most similar gauged catchments (of 1,155 available donors), used only as a secondary, non-substituting check on the general regime.

## Results: step s1

Catchment characterisation (BasinATLAS, HydroATLAS v1.0) for the point 51.415N, -0.308E: upstream area 9990.7 km2 (79 level-12 sub-basins), mean elevation 109.0 m, mean slope 2.0 degrees, mean annual precipitation 684.0 mm/yr, PET 695.0 mm/yr, actual ET 518.0 mm/yr, aridity index 0.98, mean temperature 9.6 degC, snow cover 8.0%. Land cover: 2% forest, 44% cropland, 16% pasture, 22% urban, 0% irrigated, 0% glacier, 0% wetland, 0.5% lake. Degree of regulation by reservoirs is 0.0% and reservoir volume upstream is 0.0 million m3, confirming no dams to bias the flood record. Mean annual natural discharge at the outlet is 84.65 m3/s; annual runoff 310.48 mm/yr.

![The site, in longitude and latitude (no basemap); no catalogue station was listed with it.](figures/s1_site_map.png)
*The site, in longitude and latitude (no basemap); no catalogue station was listed with it.*

*Catchment attributes from BasinATLAS for the site at 51.41 N, 0.31 W.*

| attribute | label | value | unit | source | note |
| --- | --- | --- | --- | --- | --- |
| n_sub_basins |  | 79.0 |  |  |  |
| area_km2 |  | 9990.8 |  |  |  |
| outlet_hybas_id |  | 2120392310.0 |  |  |  |
| upstream_area_km2 |  | 9990.7 |  |  |  |
| elevation_m | mean elevation | 109.0 | m | basinatlas_upstream |  |
| slope_deg | mean slope | 2.0 | degrees | basinatlas_upstream |  |
| precipitation_mm_yr | annual precipitation (WorldClim) | 684.0 | mm/yr | basinatlas_upstream |  |
| pet_mm_yr | annual potential evapotranspiration | 695.0 | mm/yr | basinatlas_upstream |  |
| aet_mm_yr | annual actual evapotranspiration | 518.0 | mm/yr | basinatlas_upstream |  |
| aridity_index | aridity index (P/PET) | 0.98 | P/PET | basinatlas_upstream |  |
| temperature_c | mean annual air temperature | 9.6 | °C | basinatlas_upstream |  |
| snow_cover_pct | annual snow cover extent | 8.0 | % | basinatlas_upstream |  |
| runoff_mm_yr | annual land-surface runoff | 310.48 | mm/yr | area_weighted_mean |  |
| discharge_m3s | mean annual natural discharge at the outlet | 84.65 | m3/s | basinatlas_upstream |  |
| forest_pct | forest cover | 2.0 | % | basinatlas_upstream |  |
| cropland_pct | cropland | 44.0 | % | basinatlas_upstream |  |
| pasture_pct | pasture | 16.0 | % | basinatlas_upstream |  |
| urban_pct | urban extent | 22.0 | % | basinatlas_upstream |  |
| irrigated_pct | irrigated area | 0.0 | % | basinatlas_upstream |  |
| glacier_pct | glacier extent | 0.0 | % | basinatlas_upstream |  |
| wetland_pct | wetlands (all classes) | 0.0 | % | basinatlas_upstream |  |
| lake_pct | lake area | 0.5 | % | basinatlas_upstream |  |
| karst_pct | karst extent | 48.0 | % | basinatlas_upstream |  |
| clay_pct | clay fraction in soil | 19.0 | % | basinatlas_upstream |  |
| silt_pct | silt fraction in soil | 37.0 | % | basinatlas_upstream |  |
| sand_pct | sand fraction in soil | 44.0 | % | basinatlas_upstream |  |
| soil_organic_carbon_t_ha | soil organic carbon | 45.0 | t/ha | basinatlas_upstream |  |
| soil_water_pct | annual soil water content | 81.0 | % | basinatlas_upstream |  |
| groundwater_table_cm | groundwater table depth | 151.86 | cm | area_weighted_mean |  |
| population_density | population density | 531.99 | people/km2 | basinatlas_upstream |  |
| population | population count | 5290300.78 | people | basinatlas_upstream |  |
| degree_of_regulation_pct | degree of regulation by reservoirs | 0.0 | % | basinatlas_upstream |  |
| human_footprint_2009 | human footprint (2009) | 30.7 | index 0-50 | basinatlas_upstream |  |
| reservoir_volume_mcm | reservoir volume upstream | 0.0 | million m3 | basinatlas_upstream |  |

## Results: step s2

Station analysis of the UK EA Kingston gauge (8496ce69-482c-406a-a2f0-ac418ef8f099): daily discharge record, 1883-10-01 to 2026-09-09 (142.9 years, n=51947, inferred daily resolution at 363.52 observations per year). Mean flow 65.52 m3/s, median 40.03 m3/s, min 0.01 m3/s, max 800.0 m3/s. Flow-duration curve: Q95 7.52 m3/s, Q50 39.9 m3/s, Q10 162.0 m3/s. Mann-Kendall trend test on the annual mean: p=0.9411, tau=-0.0043, Sen's slope -0.0029 m3/s per year, no trend. Mann-Kendall on the annual maxima: p=0.1784, tau=0.0769, Sen's slope 0.2906 m3/s per year, no trend. These pre-tests support the stationarity assumption behind the flood-frequency fit. The record maximum of 800.0 m3/s (1894) has an empirical return period of about 141 years (Weibull plotting position) against a GEV fit value there of 679.3 m3/s.

![Discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099), 1883 to 2026, with the annual maxima marked.](figures/s2_series.png)
*Discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099), 1883 to 2026, with the annual maxima marked.*

![Annual mean discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099) with the Sen slope line; the Mann-Kendall test finds no trend (p = 0.941, 140 years).](figures/s2_trend.png)
*Annual mean discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099) with the Sen slope line; the Mann-Kendall test finds no trend (p = 0.941, 140 years).*

*The record (17316 rows) is in the workbook (`workbook.xlsx`, sheet `s2_series`) and the notebook, not printed here.*

*Summary of the record at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099).*

| item | value |
| --- | --- |
| source | uk_ea |
| station_id | 8496ce69-482c-406a-a2f0-ac418ef8f099 |
| variable | discharge |
| unit | m3/s |
| n | 51947 |
| start | 1883-10-01 |
| end | 2026-09-09 |
| years | 142.9 |
| stats.mean | 65.5171 |
| stats.median | 40.027 |
| stats.min | 0.01 |
| stats.max | 800.0 |

*Annual maxima at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099).*

| year | value |
| --- | --- |
| 1884 | 227.0 |
| 1885 | 240.0 |
| 1886 | 236.0 |
| 1887 | 279.0 |
| 1888 | 206.0 |
| 1889 | 233.0 |
| 1890 | 200.0 |
| 1891 | 334.0 |
| 1892 | 231.0 |
| 1893 | 295.0 |
| 1894 | 800.0 |
| 1895 | 299.0 |
| 1896 | 226.0 |
| 1897 | 346.0 |
| 1898 | 183.0 |
| 1899 | 257.0 |
| 1900 | 527.0 |
| 1901 | 196.0 |
| 1902 | 151.0 |
| 1903 | 377.0 |
| 1904 | 510.0 |
| 1905 | 227.0 |
| 1906 | 245.0 |
| 1907 | 371.0 |
| 1908 | 330.0 |
| 1909 | 221.0 |
| 1910 | 425.0 |
| 1911 | 267.0 |
| 1912 | 360.0 |
| 1913 | 247.0 |
| 1914 | 298.0 |
| 1915 | 581.0 |
| 1916 | 362.0 |
| 1917 | 230.0 |
| 1918 | 347.0 |
| 1919 | 327.0 |
| 1920 | 247.0 |
| 1921 | 231.0 |
| 1922 | 193.0 |
| 1923 | 221.0 |
| 1924 | 334.0 |
| 1925 | 514.0 |
| 1926 | 364.0 |
| 1927 | 450.0 |
| 1928 | 522.0 |
| 1929 | 547.0 |
| 1930 | 314.0 |
| 1931 | 218.0 |
| 1932 | 268.0 |
| 1933 | 468.0 |

*Return levels at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099) by return period, with the confidence band.*

| T | GEV | LP3 | lower | upper |
| --- | --- | --- | --- | --- |
| 2.0 | 307.917 | 311.75 | 297.7184 | 326.4429 |
| 5.0 | 403.0441 | 407.1086 | 385.6949 | 429.7112 |
| 10.0 | 464.8468 | 464.6854 | 436.7122 | 494.4503 |
| 25.0 | 541.616 | 532.2235 | 495.2459 | 571.9622 |
| 50.0 | 597.6318 | 579.3046 | 535.4148 | 626.7922 |
| 100.0 | 652.4577 | 624.0012 | 573.1555 | 679.3576 |

*Flow-duration percentiles at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099).*

| exceedance_pct | value |
| --- | --- |
| 10.0 | 162.0 |
| 50.0 | 39.9 |
| 95.0 | 7.52 |

*Mann-Kendall trend test and Sen slope at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099).*

| item | value |
| --- | --- |
| on | annual mean |
| p_value | 0.9411 |
| tau | -0.0043 |
| trend | no trend |
| sens_slope_per_year | -0.0029 |
| n_years | 140 |

## Results: step s3

Dedicated flood-frequency analysis at T=100 years for the Kingston gauge: GEV (L-moments) gives 652.5 m3/s; Log-Pearson III gives 624.0 m3/s with an analytical 90% confidence interval of 573.2 to 679.4 m3/s. A GEV fit re-estimated by maximum likelihood (seeded from L-moments) with 1,000 bootstrap resamples gives a point estimate of 646.2 m3/s and a 90% bootstrap confidence interval of 565.4 to 723.1 m3/s. All quality gates passed: the minimum 20-year requirement is met (142.9 years available); T=100 is well within the cap of about 3x record length (429 years); the bootstrap confidence interval is finite; the GEV-LP3 spread is 4%, within the 25% disagreement threshold; and the record maximum (800 m3/s) sits at 1.18x the GEV fit's value at its empirical return period, within the 1.25x envelope allowed.

![Return levels of annual maximum discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099): GEV (L-moments) and Log-Pearson III fits with the GEV bootstrap 90 % band.](figures/s3_frequency_curve.png)
*Return levels of annual maximum discharge at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099): GEV (L-moments) and Log-Pearson III fits with the GEV bootstrap 90 % band.*

*Return levels at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099) by return period, with the confidence band.*

| T | GEV | LP3 | lower | upper |
| --- | --- | --- | --- | --- |
| 100.0 | 652.4577 | 624.0012 | 565.4195 | 723.1096 |

*Spread between the GEV and Log-Pearson III return levels at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099).*

| T | GEV | LP3 | spread_pct |
| --- | --- | --- | --- |
| 100.0 | 652.4577 | 624.0012 | 4.5 |

## Results: step s4

An independent GloFAS-modelled discharge series for the grid cell at 51.415N, -0.308E (1997-01-02 to 2026-09-07, 29.7 years, n=10841, daily) gives a mean discharge of 0.85 m3/s and a GEV (L-moments) 100-year return level of 14.77 m3/s (Log-Pearson III: 13.95 m3/s). This cross-check failed its gate: a ratio of 0.02 against the at-site reference of 652.5 m3/s, against an allowed tolerance factor of 1.50. The GloFAS cell's mean discharge (0.85 m3/s) is about 77 times smaller than the gauge's mean (65.52 m3/s), indicating the model grid cell does not represent the same effective catchment area as the gauge; the cross-check is therefore set aside as an artifact of catchment-representation mismatch, not used to corroborate or challenge the at-site estimate. ERA5-derived climate for the same cell: precipitation 651.8 mm/yr, reference evapotranspiration 708.7 mm/yr, aridity index 0.92 (humid), mean temperature 11.0 degC.

![Mean monthly precipitation (bars) and FAO-56 reference evapotranspiration (line) for the ERA5 cell at the site at 51.41 N, 0.31 W, 40 years ending 2026-09-07.](figures/s4_monthly_climate.png)
*Mean monthly precipitation (bars) and FAO-56 reference evapotranspiration (line) for the ERA5 cell at the site at 51.41 N, 0.31 W, 40 years ending 2026-09-07.*

![Annual maxima of the modelled discharge from GloFAS v4 (Open-Meteo) for the grid cell at the site at 51.41 N, 0.31 W, 1997 to 2026: a model output, indicative only, not a gauge reading.](figures/s4_glofas_series.png)
*Annual maxima of the modelled discharge from GloFAS v4 (Open-Meteo) for the grid cell at the site at 51.41 N, 0.31 W, 1997 to 2026: a model output, indicative only, not a gauge reading.*

*Mean monthly precipitation and reference evapotranspiration for the ERA5 cell at the site at 51.41 N, 0.31 W.*

| month | precipitation_mm | et0_mm |
| --- | --- | --- |
| 1 | 57.4408 | 18.0718 |
| 2 | 52.6396 | 27.9911 |
| 3 | 44.1331 | 45.1587 |
| 4 | 47.9887 | 70.6386 |
| 5 | 50.2874 | 92.6599 |
| 6 | 56.4205 | 106.592 |
| 7 | 54.846 | 110.5912 |
| 8 | 54.603 | 95.2519 |
| 9 | 49.6798 | 66.5506 |
| 10 | 63.3397 | 40.1351 |
| 11 | 63.7972 | 21.4795 |
| 12 | 56.4662 | 16.2437 |

*GloFAS modelled discharge for the grid cell at the site at 51.41 N, 0.31 W (indicative).*

| item | value |
| --- | --- |
| variable | discharge |
| unit | m3/s |
| n | 10841 |
| start | 1997-01-02 |
| end | 2026-09-07 |
| years | 29.7 |
| stats.mean | 0.8471 |
| stats.median | 0.55 |
| stats.min | 0.16 |
| stats.max | 13.96 |
| sampling.n | 10841 |
| sampling.span_years | 29.7 |
| sampling.per_year | 365.02 |
| sampling.inferred_resolution | daily |
| trend.on | annual mean |
| trend.p_value | 0.3881 |
| trend.tau | -0.1158 |
| trend.trend | no trend |
| trend.sens_slope_per_year | -0.0021 |
| trend.n_years | 29 |
| source | GloFAS v4 (modelled) via Open-Meteo |
| modelled | True |
| return_level_T2_gev | 5.7499 |
| return_level_T5_gev | 7.4492 |
| return_level_T10_gev | 8.8155 |
| return_level_T25_gev | 10.876 |
| return_level_T50_gev | 12.6892 |
| return_level_T100_gev | 14.7708 |
| q10 | 1.84 |
| q50 | 0.55 |
| q95 | 0.22 |

## Results: step s5

Regionalised flow signatures from the 10 most similar gauged catchments (by BasinATLAS attribute similarity, out of 1,155 available donors) corroborate the general flow regime at Kingston: mean daily flow 0.5752 mm/d (band 0.3158 to 1.0475 mm/d), median daily flow 0.4138 mm/d, low flow exceeded 95% of days 0.1608 mm/d (band 0.0706 to 0.3663 mm/d), high flow exceeded 5% of days 1.5118 mm/d (band 0.872 to 2.6211 mm/d), mean annual daily maximum 3.0492 mm/d (band 1.3245 to 7.0196 mm/d), runoff ratio 0.349 (band 0.2185 to 0.4795), baseflow index 0.8078 (band 0.7251 to 0.8904). These signatures are indicative of a humid, baseflow-dominated lowland regime consistent with the Kingston gauge's own flow-duration curve, but they are expressed in mm/d over donor catchments of varying size and are not intended to produce a T=100 discharge quantile; they serve only as qualitative corroboration of the flow regime, not a substitute for the at-site fit.

![Flow signatures transferred to the site from 10 donor catchments, with the one-standard-deviation band across donors as error bars and the leave-one-out skill (NSE) where published.](figures/s5_signatures_band.png)
*Flow signatures transferred to the site from 10 donor catchments, with the one-standard-deviation band across donors as error bars and the leave-one-out skill (NSE) where published.*

*Flow signatures at the site at 51.41 N, 0.31 W.*

| signature | label | value | low | high | unit | n_donors | nse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_mean_mm | mean daily flow | 0.5752 | 0.3158 | 1.0475 | mm/d | 10 |  |
| q_median_mm | median daily flow | 0.4138 | 0.2095 | 0.8174 | mm/d | 10 |  |
| q95_mm | low flow: exceeded 95 % of days | 0.1608 | 0.0706 | 0.3663 | mm/d | 10 |  |
| q05_mm | high flow: exceeded 5 % of days | 1.5118 | 0.872 | 2.6211 | mm/d | 10 |  |
| q_annual_max_mm | mean annual daily maximum | 3.0492 | 1.3245 | 7.0196 | mm/d | 10 |  |
| runoff_ratio | mean flow / BasinATLAS precipitation | 0.349 | 0.2185 | 0.4795 | - | 10 | -0.027 |
| baseflow_index | baseflow / total flow | 0.8078 | 0.7251 | 0.8904 | - | 10 | 0.33 |
| fdc_slope | slope of the flow-duration curve (log space, 33-66 %) | 1.8585 | 1.3266 | 2.3903 | - | 10 | 0.169 |
| high_flow_frequency | days above 3 x median per year | 29.188 | 8.4341 | 49.942 | days/yr | 10 | 0.263 |
| low_flow_frequency | days below 0.2 x median per year | 0.6454 | 0.0 | 1.4887 | days/yr | 10 | 0.269 |
| zero_flow_fraction | fraction of zero-flow days | 0.0 | 0.0 | 0.0 | - | 10 | -0.087 |
| seasonality_index | Markham seasonality of monthly flow | 0.2782 | 0.1884 | 0.3679 | - | 10 | 0.332 |
| flashiness_index | Richards-Baker flashiness | 0.1735 | 0.1063 | 0.2408 | - | 10 | 0.421 |

*Donor gauges selected for the site at 51.41 N, 0.31 W.*

| source | station_id | name | latitude | longitude | distance_km | score | similarity_distance | up_area_km2 | period_start | period_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hubeau_hydrometrie | E366060002 | La Lawe à Bruay-la-Buissière | 50.486717047 | 2.54446616 |  | 0.1068 | 0.1068 | 210.0 | 2011-01-01 |  |
| hubeau_hydrometrie | E366601001 | La Lawe à Béthune | 50.539490667 | 2.637101185 |  | 0.1075 | 0.1075 | 210.0 | 2009-01-01 |  |
| hubeau_hydrometrie | E364621001 | La Clarence à Robecq | 50.594520299 | 2.565962262 |  | 0.133 | 0.133 | 177.3 | 1969-01-01 |  |
| uk_ea | 67b89572-9b04-42a9-a629-fd8c7e302495 | Chart Leacon | 51.1451 | 0.846935 |  | 0.1521 | 0.1521 | 159.4 | 1979-12-31 |  |
| uk_ea | 54d2b82b-cd8a-4fae-bde4-c11d9fb80af7 | Wye | 51.185494 | 0.9303 |  | 0.1976 | 0.1976 | 282.0 | 1962-10-01 |  |
| uk_ea | 09aaf1dd-28ea-4230-9670-7709ee7aa970 | Stanley | 51.455677 | -2.06563 |  | 0.2009 | 0.2009 | 110.5 | 1970-01-01 |  |
| uk_ea | 84c3ec95-a8f2-4fb6-8aa9-94f91f65b5a8 | Barford Bridge | 52.438642 | -0.734983 |  | 0.2046 | 0.2046 | 218.5 | 1988-11-08 |  |
| uk_ea | 678aa360-300c-4c54-a3b1-47604cb60971 | Bourne End Hedsor | 51.571866 | -0.70858 |  | 0.2047 | 0.2047 | 6716.8 | 1964-11-28 |  |
| uk_ea | 3db09b08-5c7e-4f20-9363-2691eba109f8 | Willen | 52.058687 | -0.714748 |  | 0.2059 | 0.2059 | 401.2 | 1962-01-01 |  |
| hubeau_hydrometrie | E176601001 | La Rhonelle à Aulnoy-lez-Valenciennes | 50.329978539 | 3.531707092 |  | 0.2091 | 0.2091 | 110.0 | 1963-01-01 |  |

## Limitations and what this study does not establish

Design-flood guidance under climate change is immature (Wasko et al. 2024, HESS): the 652.5 m3/s estimate is stationary, and any climate scenario is an overlay on it, not a nonstationary fit. Rare quantiles move with the distribution and the estimator: GEV (L-moments) and Log-Pearson III are quoted together with their intervals and the 4% spread between them rather than averaged; a spread above 25% would have been reported as disagreement. GloFAS and the regionalised signatures are indicative cross-checks at coarser resolution than the gauge and may disagree with the at-site fit, as the GloFAS check did here for reasons of catchment-scale mismatch rather than genuine hydrological disagreement. The fit is stationary; any future climate scenario is an overlay caveat, not part of the quoted 100-year estimate.

## What this study does not establish

- Step s4, gate cross_check_ratio: 14.77 against the reference 652.5 at T = 100 years: ratio 0.02, outside the allowed factor of 1.50: the cross-check disagrees

## Caveats

- Design-flood guidance under climate change is immature (Wasko et al. 2024, HESS): the estimate here is stationary, and any climate scenario is an overlay on it, not a nonstationary fit.
- Rare quantiles move with the distribution and the estimator. Two fits (GEV by L-moments and Log-Pearson III) are quoted with their intervals and the spread between them; a spread above 25 percent is reported as disagreement, not averaged away.

## Recommendations

Adopt 652.5 m3/s as the 100-year design discharge, with a 90% uncertainty band of 565.4 to 723.1 m3/s, for sizing the new road bridge's waterway opening and freeboard. Before finalising, obtain the effective upstream area and routing represented by the GloFAS grid cell used in the cross-check, to determine whether the cross-check can be made usable rather than set aside. Confirm with the Environment Agency whether any lock or weir operation upstream of Kingston affects flood routing in a way not captured by BasinATLAS's static degree-of-regulation field, since the no-regulation assumption underlies the stationarity treatment of the record. Consider a sensitivity run that down-weights the 1894 record maximum (800 m3/s), since it sits at the edge of the fit's allowed envelope and could be pulling the GEV shape parameter. Treat any climate-change allowance as an explicit overlay above this stationary 100-year value, not as a revision of the value itself.

## References

1. Mann, H. B. (1945). Nonparametric tests against trend. Econometrica, 13, 245-259
2. Kendall (1975)
3. Sen, P. K. (1968). J. Am. Stat. Assoc., 63, 1379-1389.
4. England, J. F. et al. (2019). Guidelines for determining flood flow frequency, Bulletin 17C. USGS Techniques and Methods 4-B5.
5. Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. J. R. Stat. Soc. B, 52(1), 105-124.
6. Harrigan, S. et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979-present. Earth Syst. Sci. Data, 12, 2043-2060.
7. HydroATLAS v1.0 (BasinATLAS), CC BY 4.0. Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6
8. Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves I: new interpretation and confidence intervals. J. Water Resour. Plann. Manage., 120(4), 485-504.
9. England, J. F. Jr. et al. (2018). Guidelines for determining flood flow frequency, Bulletin 17C. USGS Techniques and Methods 4-B5.
10. Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.
11. Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., 146, 1999-2049
12. Open-Meteo.com (CC BY 4.0).
13. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56.
14. Bloeschl, G. et al. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University Press
15. Oudin, L. et al. (2008). Spatial proximity, physical similarity, regression and ungaged catchments. Water Resour. Res. 44, W03413.
16. Addor, N. et al. (2018). A ranking of hydrological signatures based on their predictability in space. Water Resour. Res. 54, 8792-8812.
17. Wasko, C. et al. (2024). A systematic review of climate change science for flood and design guidance. Hydrol. Earth Syst. Sci. 28, 1251-1285. doi:10.5194/hess-28-1251-2024
18. Nonstationary flood frequency estimates are parameter-fragile: Stoch. Environ. Res. Risk Assess. (2024), doi:10.1007/s00477-024-02680-9
19. Multi-approach cross-checks in infrastructure flood practice: J. Hydrol. (2024), doi:10.1016/j.jhydrol.2024.130698
20. Wasko et al. 2024, HESS
21. Rekin226 and contributors (2026). AquaScope: Open-source water data aggregation toolkit (version 0.16.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21903143

## Appendix: reproducibility

Re-run the same steps with no model: `aquascope run study.yaml`. Resume the workspace: `aquascope studio --resume workspace.json`.

Model: claude-sonnet-5 via anthropic; ledger: consultant 1 call(s), 4272 tokens, methodologist 1 call(s), 16527 tokens, analyst 1 call(s), 6874 tokens, interpreter 1 call(s), 31156 tokens, author 1 call(s), 32197 tokens, critic 1 call(s), 34626 tokens. aquascope 0.16.0.

```yaml
# An AquaScope study (version 3): the plan behind an answer, its gates, and what happened.
#   aquascope run study.yaml
version: 3
title: "Establish a defensible 100-year design discharge, with its u: 51.415, -0.308"
question: "Design flow for a new road bridge over the Thames at Kingston: the 100-year flood with its uncertainty band."
created: "2026-09-14T21:11:49+00:00"
aquascope_version: "0.16.0"
author: "methodologist"
model: "claude-sonnet-5"
problem:
  kind: "flood_risk"
  site: {"lat": 51.415, "lon": -0.308}
  params: {"return_period": 100, "decision": "design flow"}
  text: "Design flow for a new road bridge over the Thames at Kingston: the 100-year flood with its uncertainty band."
plan:
  author: "methodologist"
  playbook: "flood_risk"
  objective: "Establish a defensible 100-year design discharge, with its uncertainty band, for the new road bridge over the Thames at Kingston."
  decision: "set the 100-year design discharge (with uncertainty bounds) for the new road bridge over the Thames at Kingston"
  methodology: ["Characterise the catchment upstream of the site to confirm its size and the absence of dams or regulation that would bias the flood record.", "Run a trend pre-test on the Kingston gauge record (annual maxima and mean) to confirm stationarity before fitting a flood-frequency model.", "Fit GEV (L-moments) and Log-Pearson III distributions to the annual maxima at the Kingston gauge, with a bootstrap confidence interval on the GEV fit, and quote the T=100 year quantile from both with their spread.", "Cross-check the at-site 100-year estimate against an independent GloFAS-modelled discharge frequency fit for the same point.", "Corroborate the at-site flow regime with regionalised flow signatures from similar gauged catchments, as a secondary check, not a substitute for the at-site fit."]
  assumptions: ["daily resolution assumed for the discharge record, as the catalog does not state it", "at-site flood frequency analysis is appropriate given the 143-year gauge record essentially at the site (uk_ea station 8496ce69-482c-406a-a2f0-ac418ef8f099)", "GloFAS cross-check and regional signature methods used only to corroborate, not replace, the at-site analysis", "no upstream regulation or dams to adjust for, per BasinATLAS", "no upstream regulation or dams to adjust for, per BasinATLAS (0 dams)"]
  alternatives: [{"method": "similar_basins", "why_not": "marked marginal in the sufficiency table: meant for an ungauged point, while a 143-year gauge sits 0.1 km from the site"}, {"method": "nonstationary flood frequency fit", "why_not": "design-flood guidance under climate change is immature; a trend, if found, is reported as a caveat on the stationary estimate rather than used to fit a nonstationary model"}]
  limitations_expected: ["rare quantiles (T=100) are sensitive to the distribution and estimator chosen; GEV and LP3 are quoted together with their spread rather than averaged", "GloFAS and regionalised signatures are indicative cross-checks at coarser resolution than the gauge and may disagree with the at-site fit", "the fit is stationary; any future climate scenario is an overlay caveat, not part of the quoted 100-year estimate"]
  citations: ["England, J. F. et al. (2019). Guidelines for determining flood flow frequency, Bulletin 17C. USGS Techniques and Methods 4-B5.", "Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. J. R. Stat. Soc. B 52, 105-124.", "Wasko, C. et al. (2024). A systematic review of climate change science for flood and design guidance. Hydrol. Earth Syst. Sci. 28, 1251-1285. doi:10.5194/hess-28-1251-2024", "Nonstationary flood frequency estimates are parameter-fragile: Stoch. Environ. Res. Risk Assess. (2024), doi:10.1007/s00477-024-02680-9", "Multi-approach cross-checks in infrastructure flood practice: J. Hydrol. (2024), doi:10.1016/j.jhydrol.2024.130698", "Oudin, L. et al. (2008). Spatial proximity, physical similarity, regression and ungaged catchments. Water Resour. Res. 44, W03413.", "Harrigan, S. et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979-present. Earth Syst. Sci. Data 12, 2043-2060.", "Wasko et al. 2024, HESS"]
  caveats: ["Design-flood guidance under climate change is immature (Wasko et al. 2024, HESS): the estimate here is stationary, and any climate scenario is an overlay on it, not a nonstationary fit.", "Rare quantiles move with the distribution and the estimator. Two fits (GEV by L-moments and Log-Pearson III) are quoted with their intervals and the spread between them; a spread above 25 percent is reported as disagreement, not averaged away."]
  rationale: "Establish a defensible 100-year design discharge, with its uncertainty band, for the new road bridge over the Thames at Kingston."
  recon_notes: ["Record resolution is not in the catalog; daily is assumed for every variable.", "10 donor gauges from a pool of 34,786 gauged catchments.", "ERA5 temperature and forcing and GloFAS discharge are assumed reachable for any point on land (Open-Meteo); not checked here.", "CMIP6 change factors need model output you supply (aquascope.climate works on downloaded data); not counted."]
steps:
  - tool: "describe_catchment"
    id: "s1"
    rationale: "Catchment size, elevation and regulation by dams frame the estimate and confirm the maxima are natural."
    arguments:
      lat: 51.415
      lon: -0.308
    outputs: [{"kind": "figure", "id": "s1_site_map", "caption": "site map from describe_catchment"}, {"kind": "table", "id": "s1_catchment_attributes", "caption": "catchment attributes from describe_catchment"}]
  - tool: "analyze_station"
    id: "s2"
    rationale: "Mann-Kendall trend pre-test on annual maxima, the series a stationary flood fit assumes has no trend."
    method: "trend_mann_kendall"
    arguments:
      source: "uk_ea"
      station_id: "8496ce69-482c-406a-a2f0-ac418ef8f099"
    expects:
      - {"check": "min_years", "value": 20, "path": "years"}
      - {"check": "not_empty", "path": "trend"}
      - {"check": "unit_present", "path": "unit"}
      - {"check": "sampling_density", "value": "daily", "path": "sampling"}
      - {"check": "trend_on_series", "value": 0.05, "path": "ffa.amax_trend"}
    outputs: [{"kind": "figure", "id": "s2_annual_maxima", "caption": "annual maxima from analyze_station"}, {"kind": "figure", "id": "s2_trend", "caption": "trend from analyze_station"}, {"kind": "table", "id": "s2_summary", "caption": "summary from analyze_station"}, {"kind": "table", "id": "s2_trend", "caption": "trend table from analyze_station"}]
  - tool: "flood_frequency"
    id: "s3"
    rationale: "Return levels from GEV (L-moments) and Log-Pearson III with a bootstrap band give the 100-year design discharge and the spread between fits."
    method: "at_site_flood_frequency"
    arguments:
      source: "uk_ea"
      station_id: "8496ce69-482c-406a-a2f0-ac418ef8f099"
      bootstrap_ci: true
      return_periods: [100]
    expects:
      - {"check": "min_years", "value": 20, "path": "years"}
      - {"check": "max_return_period_factor", "value": 3, "path": "years", "return_period": 100}
      - {"check": "ci_finite", "path": "ffa.fits.gev_bootstrap.ci", "return_period": 100}
      - {"check": "spread_within", "value": 0.25, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"], "return_period": 100}
      - {"check": "fit_envelopes_max", "value": 0.25, "path": "ffa"}
    fallback: {"step": {"tool": "similar_basins", "arguments": {"source": "uk_ea", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "k": 5}, "rationale": "If the at-site fit fails its gates, donor gauges from similar catchments give a regional cross-check to quote instead.", "expects": []}}
    depends_on: ["s2"]
    outputs: [{"kind": "figure", "id": "s3_frequency_curve", "caption": "frequency curve from flood_frequency"}, {"kind": "table", "id": "s3_return_levels", "caption": "return levels from flood_frequency"}, {"kind": "table", "id": "s3_fit_spread", "caption": "fit spread from flood_frequency"}]
  - tool: "anywhere"
    id: "s4"
    rationale: "GloFAS modelled discharge for the cell gives an independent cross-check on the at-site 100-year quantile."
    method: "glofas_cross_check"
    arguments:
      lat: 51.415
      lon: -0.308
      years: 40
    expects:
      - {"check": "not_empty", "path": "climate", "repaired_from": "glofas"}
      - {"check": "cross_check_ratio", "value": 0.5, "path": "glofas.ffa.fits.gev_lmoments.q_by_T", "reference": "{{ result.s3.ffa.fits.gev_lmoments.q_by_T }}", "return_period": 100}
    depends_on: ["s3"]
    outputs: [{"kind": "figure", "id": "s4_glofas_series", "caption": "glofas series from anywhere"}, {"kind": "table", "id": "s4_glofas_summary", "caption": "glofas summary from anywhere"}]
  - tool: "regionalize_signatures"
    id: "s5"
    rationale: "Regionalised flow signatures from 10 donor catchments give a secondary corroboration of the flow regime, not a replacement for the at-site fit."
    method: "regionalize_signatures"
    arguments:
      lat: 51.415
      lon: -0.308
      k: 10
    expects:
      - {"check": "not_empty", "path": "estimates"}
      - {"check": "not_empty", "path": "skill"}
    depends_on: ["s3"]
    outputs: [{"kind": "table", "id": "s5_signatures", "caption": "regionalised flow signatures"}]
results:
  s1: {"ok": true, "gates": [], "summary": "latitude=51.415, longitude=-0.308, license=CC-BY-4.0, attribution=HydroATLAS v1.0 (BasinATLAS), CC BY 4.0. Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-ba", "fallback_used": false, "sha256": "2bcacbbfd6fa6878"}
  s2: {"ok": true, "gates": [{"check": "min_years", "passed": true, "detail": "142.9 years of record, 20 needed"}, {"check": "not_empty", "passed": true, "detail": "'trend' is present"}, {"check": "unit_present", "passed": true, "detail": "unit m3/s"}, {"check": "sampling_density", "passed": true, "detail": "51947 observations in 142.9 years: 363.52 a year, about daily; daily claimed"}, {"check": "trend_on_series", "passed": true, "detail": "Mann-Kendall on the annual maxima: p = 0.178, tau = 0.08: no trend at the 0.05 level"}], "summary": "source=uk_ea, station_id=8496ce69-482c-406a-a2f0-ac418ef8f099, variable=discharge, unit=m3/s, years=142.9, start=1883-10-01, end=2026-09-09", "fallback_used": false, "sha256": "d493bbea2e615604"}
  s3: {"ok": true, "gates": [{"check": "min_years", "passed": true, "detail": "142.9 years of record, 20 needed"}, {"check": "max_return_period_factor", "passed": true, "detail": "T = 100 years against a cap of about 429 years (3 times 142.9 years of record)"}, {"check": "ci_finite", "passed": true, "detail": "finite interval [565.4, 723.1] at T = 100 years"}, {"check": "spread_within", "passed": true, "detail": "spread 4% between 652.5, 624 (25% allowed) at T = 100 years"}, {"check": "fit_envelopes_max", "passed": true, "detail": "record maximum 800 (1894, T about 141 years) against the gev_lmoments fit's 679.3 there: ratio 1.18 (1.25 allowed)"}], "summary": "source=uk_ea, station_id=8496ce69-482c-406a-a2f0-ac418ef8f099, unit=m3/s, years=142.9, start=1883-10-01, end=2026-09-09", "fallback_used": false, "sha256": "7d1558a152ba79b3"}
  s4: {"ok": true, "gates": [{"check": "not_empty", "passed": true, "detail": "'climate' is present"}, {"check": "cross_check_ratio", "passed": false, "detail": "14.77 against the reference 652.5 at T = 100 years: ratio 0.02, outside the allowed factor of 1.50: the cross-check disagrees"}], "summary": "years=40, start=1986-09-07, end=2026-09-07", "fallback_used": false, "sha256": "374a4a32f3f979a3", "failed_reason": "gate failed: cross_check_ratio (14.77 against the reference 652.5 at T = 100 years: ratio 0.02, outside the allowed factor of 1.50: the cross-check disagrees)"}
  s5: {"ok": true, "gates": [{"check": "not_empty", "passed": true, "detail": "'estimates' is present"}, {"check": "not_empty", "passed": true, "detail": "'skill' is present"}], "summary": "method=similarity", "fallback_used": false, "sha256": "35c0557fb1532893"}
```

## Cite this software

AquaScope Studio (2026). AquaScope: Open-source water data aggregation toolkit (version 0.16.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21903143


---

*{'model': 'claude-sonnet-5', 'provider': 'anthropic', 'prose': 'model', 'tokens': {'consultant': {'calls': 1, 'prompt_tokens': 3299, 'completion_tokens': 973, 'cost_usd': 0.016328}, 'methodologist': {'calls': 1, 'prompt_tokens': 12771, 'completion_tokens': 3756, 'cost_usd': 0.063102}, 'analyst': {'calls': 1, 'prompt_tokens': 5978, 'completion_tokens': 896, 'cost_usd': 0.020916}, 'interpreter': {'calls': 1, 'prompt_tokens': 24765, 'completion_tokens': 6391, 'cost_usd': 0.11344}, 'author': {'calls': 2, 'prompt_tokens': 55142, 'completion_tokens': 15864, 'cost_usd': 0.268924}, 'critic': {'calls': 1, 'prompt_tokens': 25574, 'completion_tokens': 9052, 'cost_usd': 0.141668}}, 'total_tokens': 164461, 'total_usd': 0.624378, 'budget': None, 'dropped': 1, 'aquascope_version': '0.16.0', 'date': '2026-09-14 21:17 UTC', 'workspace': 'e131698dc948', 'plan_author': 'methodologist', 'written_by': {'answer': 'model', 'summary': 'model', 'decision': 'model', 'findings': 'model', 'problem': 'model', 'site_data': 'model', 'methodology': 'model', 'results-s1': 'model', 'results-s2': 'model', 'results-s3': 'model', 'results-s4': 'model', 'results-s5': 'model', 'limitations': 'model', 'recommendations': 'model', 'references': 'template', 'appendix': 'template'}}*
