# Advanced studies

A basic study answers one question: what is the 100-year flow here, assuming the
past is a fair sample of the future. The advanced studies answer the questions
that come after it. Each is a playbook the Studio plans without a model, built
from steps in `aquascope.advanced`. The same steps run in the Explorer, in
`aquascope studio`, in the Analyst and over MCP.

| Ask something like | Playbook | What runs |
| :--- | :--- | :--- |
| "Is the 100-year flood getting worse?" | `flood_change` | change points, the stationary fit, a GEV with a trend, peaks over threshold, the region's gauges |
| "How will climate change affect the flow and floods here by 2050?" | `climate_change` | the catchment, GR4J calibrated and validated, seven CMIP6 models through it |
| "What if rainfall drops 10% and it gets 2 degrees warmer?" | `catchment_response` | the catchment, GR4J calibrated and validated, the scenarios you name |

In the Explorer, pick a gauge, open **Study** and use one of the **Try a study**
questions, or type your own. In a terminal:

```bash
aquascope studio "Is the 100-year flood getting worse?" --at USGS-01013500
aquascope studio "What if rainfall drops 10% and it gets 2 degrees warmer?" --at USGS-01013500
```

## Is the flood changing? (`flood_change`)

1. **`change_points`** runs Pettitt's test for a step change, PELT for the
   segments and Mann-Kendall with Sen's slope for a trend, on the annual
   maxima. `stationary: false` means the record is not one sample.
2. **`flood_frequency`** gives the stationary reference: GEV by L-moments and
   Log-Pearson III.
3. **`nonstationary_flood`** fits a GEV whose location moves linearly with
   time, beside the stationary GEV fitted by the same likelihood. The
   likelihood-ratio test and AIC decide which model is preferred. The T-year
   level is reported at the first year, the last year and a horizon, with a
   bootstrap interval on the last year.
4. **`pot_flood`** takes every independent peak over a threshold (declustered
   by a separation in days) and fits a Generalised Pareto. It is a second
   estimate of the tail that keeps the second-largest flood of a wet year.
5. **`regional_flood`** studies the gauges within 75 km together: whether
   their trends are field-significant, and a pooled growth curve.

The answer leads with the level from the preferred model. The stationary
estimate stays the reference. The nonstationary fit is a sensitivity beside
it, because design guidance under a changing climate is not settled (Wasko et
al. 2024). A detected change says the record is not one sample; it does not
say why. Dams, a new rating curve, land-use change and climate all leave the
same mark.

## Climate change and the river (`climate_change`)

1. **`describe_catchment`** gives the upstream area, which turns flow into a
   depth the model can balance against rainfall.
2. **`catchment_model`** calibrates GR4J on the first half of the last 20
   years of ERA5 rainfall and FAO-56 reference evaporation, and validates it
   on the second half. It adds a degree-day snow store when at least a tenth
   of the precipitation falls below freezing. A 90 % band from the validation
   residuals is scored by how often the observations fall inside it. The
   validation KGE must reach 0.5.
3. **`climate_projection`** takes seven CMIP6 HighResMIP models from the
   Open-Meteo Climate API, bias-corrected onto ERA5-Land. Each model is
   compared with its own 1985-2014 baseline, for 2020-2049. Rainfall and
   temperature (Oudin evaporation, snow) drive the calibrated GR4J, model by
   model. The result is the change in mean flow, low flow (Q95) and the
   T-year flood, per model and as a spread.

If the catchment model fails its validation gate, the flow changes are not
quoted, and the step falls back to the climate change factors alone. With no
gauge nearby, the study is climate-only from the start.

What the numbers cannot say:

- **One emission pathway.** The HighResMIP future runs are close to SSP5-8.5
  and end in 2050.
- **The spread is the answer.** When the models disagree on the sign, the
  direction of change is not established, and the report says so.
- **A projection is heavy for the free tier.** By Open-Meteo's own weighting
  it counts as about 2,400 of the 10,000 free calls a day from one address.
  The Studio makes each request once per session. A refusal says the
  allowance is used up rather than failing silently.

## What if the rain or the temperature changes? (`catchment_response`)

The same calibrated, validated GR4J runs the observed days again with the
rainfall scaled and the air warmer. Warming raises reference evaporation by an
assumed 3 % per degree and warms the snow store. The result is the change in
mean flow, low flow, high flow and the median annual maximum, with the
validation skill quoted beside every number. It shows how sensitive the
catchment is, not a projection of its future.

## What each method assumes

Every method in the registry (`aquascope.methods`) now carries `assumes` and
`sensitive_to` (#376). If change points are known inside the record, methods
that assume stationarity turn marginal, with the year named. A regulated
catchment does the same for methods sensitive to regulation: a BasinATLAS
degree of regulation of at least 10 %, a threshold aquascope chose to flag
substantial regulation. So does a snowy one, at least 10 % mean snow cover.
Nothing is blocked. A stationary fit across a shift answers a narrower
question, and the point is to say which.

The plain design-flow study applies the same idea. Its flood fit runs Pettitt's
test on the annual maxima it already holds. When that finds a significant shift,
the answer is graded indicative and names the year.

## References

- Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. *J. R. Stat. Soc. C*, 28(2), 126-135.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA*, 107(500), 1590-1598.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Lang, M., Ouarda, T. B. M. J., & Bobée, B. (1999). Towards operational guidelines for over-threshold modeling. *J. Hydrol.*, 225, 103-117.
- Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *J. Hydrol.*, 279, 275-289.
- Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrol. Sci. J.*, 31(1), 13-24.
- Hock, R. (2003). Temperature index melt modelling in mountain areas. *J. Hydrol.*, 282, 104-115.
- Oudin, L. et al. (2005). Which potential evapotranspiration input for a lumped rainfall-runoff model? Part 2. *J. Hydrol.*, 303, 290-306.
- Haarsma, R. J. et al. (2016). High Resolution Model Intercomparison Project (HighResMIP v1.0) for CMIP6. *Geosci. Model Dev.*, 9, 4185-4208.
- Hosking, J. R. M., & Wallis, J. R. (1997). *Regional Frequency Analysis*. Cambridge University Press.
- Wasko, C. et al. (2024). A systematic review of climate change science relevant to Australian design flood estimation. *HESS*, 28, 1251-1285.
