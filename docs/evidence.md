# The evidence ladder: how well each model knows this river

A gauge measures the river. A global model guesses it. Where the two meet, the gauge can say how good the
guess is. The evidence ladder puts a gauge's own record beside every global model on the same river, scores each
model, gives it a grade from A to D, and says in one sentence where they disagree:

> GloFAS v4 fits this gauge best (KGE 0.58, grade B). GEOGLOWS v2's 100-year flow is 73 % below the gauge's.

It is one function, `aquascope.evidence.model_skill`, behind three faces:

- **Explorer**: the **Evidence** tab on a gauge, a small grade badge in the gauge's header, and "Best model
  skill" in the rail's gauge colouring.
- **CLI**: `aquascope evidence skill SOURCE STATION_ID` (or `--csv your.csv --at LAT LON` for your own record),
  and `aquascope evidence near LAT LON` for which model to lean on at an ungauged site.
- **MCP**: the tools `model_skill` and `model_to_lean_on`.

## The models

| Model | What is compared | Where it is scored | Licence |
|---|---|---|---|
| GEOGLOWS v2 | the river reach near the gauge whose upstream area matches the gauge's catchment | live, in the page too | CC BY 4.0 |
| GloFAS v4 (via Open-Meteo) | the 0.05 degree cell around the gauge whose mean flow is closest to the gauge's | live, in the page too | CC BY 4.0 |
| NWM v3.0 retrospective | the NHDPlus reach NWM links to the USGS gauge (else the NLDI's COMID) | monthly, in CI (US only) | public domain |
| Google GRRR (Flood Hub reanalysis) | the outlet of the gauge's HydroBASINS level-12 sub-basin | monthly, in CI | CC BY 4.0 |

Two notes on picking the model's site, because a model of the wrong river says nothing about this one:

- **GEOGLOWS**: every reach within 2 km of the gauge is a candidate, and the one whose upstream area is closest to
  the gauge's catchment area wins, so a gauge on a main stem is not compared with a tributary that passes nearer.
  The ratio of the two areas is recorded. Without a catchment area the nearest reach is taken, and the row says so.
- **GloFAS**: Open-Meteo gives no upstream area, so the cell is chosen by mean flow. That choice flatters the
  model's bias (beta); read its correlation (r) and variability (alpha) first.

A site whose upstream area is more than twice or less than half the gauge's is scored but **not graded**.

Why NWM and GRRR are only in the monthly table: NWM's store is chunked so that one reach's 44 years touch about
570 chunks, too much for a browser, and Google's bucket sends no CORS headers, so a page cannot read it. A
GitHub Actions job reads both once a month (`.github/workflows/model-skill.yml`) and publishes the scores to the
Archive as `skill/model_skill.parquet`. GloFAS is the other way round: Open-Meteo's free tier counts a request
longer than two weeks as several calls, so it is scored live for the gauge you open, not for every gauge each
month.

## The scores

Over the days the gauge and the model share (the last 30 years of the gauge record by default):

- **KGE**, the Kling-Gupta efficiency (Gupta et al. 2009), and its three parts:
  - **r**, the correlation: does the model rise and fall when the river does?
  - **alpha**, the ratio of standard deviations: are its swings the right size?
  - **beta**, the ratio of means: is its volume right?
- **NSE**, the Nash-Sutcliffe efficiency, and **PBIAS**, the percent bias.
- **The flood flows**: the 2-, 10- and 100-year flows of the gauge and of the model, each from its own GEV fit
  (L-moments) to the annual maxima of the same years, and the model's error in percent. This is the number a
  designer would take from the model. It needs ten shared years of annual maxima.

## The grades

| Grade | KGE | In words |
|---|---|---|
| A | 0.75 and up | tracks the gauge closely |
| B | 0.5 to 0.75 | usable with care |
| C | above -0.41 | better than the gauge's mean flow, but not by much |
| D | -0.41 or below | no better than the gauge's mean flow |

Then one check on floods: when the model's 100-year flow (the 10-year when there is no 100-year comparison)
differs from the gauge's by more than 50 %, the grade drops one letter, to D at most. A model with a fine KGE and
a 100-year flow twice the gauge's is not a model to design from.

Where the thresholds come from:

- **-0.41** is not a choice. Knoben et al. (2019) showed that KGE = 1 - sqrt(2), about -0.41, is what the mean
  flow itself scores, so below it a model does no better than a flat line at the gauge's mean.
- **0.75 and 0.5** are AquaScope's choice of round thresholds, not a published standard. They are fixed in
  `aquascope.evidence.GRADING`, so a grade means the same thing at every gauge.

No grade is given when the gauge and the model share fewer than three years of daily data.

## Which model to lean on near an ungauged site

`aquascope evidence near LAT LON` (the MCP tool `model_to_lean_on`) reads the published table, takes up to
eight graded gauges within 150 km, and names the model with the best median KGE over them (at least two gauges
each). The Studio's Scout reads the same thing for the site of a study, and when the study quotes a modelled
number (the GEOGLOWS reach record or the GloFAS cross-check), the Interpreter adds that sentence to the
conditions and limitations of the answer. It is local evidence about which model has done well around here, not
a guarantee at the site.

## The published table

`skill/model_skill.parquet` in the [Archive dataset](https://huggingface.co/datasets/Rekin226/aquascope-gauges)
has one row per gauge and model: the scores above, the grade and why, the model's site and how it was matched,
the area ratio, the disagreement sentence, the licence and when it was computed. `is_best` marks the best graded
model at each gauge, which is what the map colours by. `skill/manifest.json` says how many gauges were scored,
which were left for next month (each run takes the gauges never scored first, then the oldest), and what failed.

Until the first monthly run publishes the table, the Evidence tab still scores GEOGLOWS and GloFAS live, and the
map colouring says the table is not published yet.

## Limits

- Models are modelled, not measured. A grade is about this gauge's record over these years.
- Agency records carry their own errors: a rating curve extrapolated in a flood, a regulated river, a gauge moved.
  A low grade can be the gauge's fault as well as the model's.
- A model that was calibrated or trained on this gauge will score well on it. The published table does not say
  which gauges each model saw in training.

## References

- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and
  NSE performance criteria: implications for improving hydrological modelling. J. Hydrol. 377, 80-91.
- Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing
  Nash-Sutcliffe and Kling-Gupta efficiency scores. Hydrol. Earth Syst. Sci. 23, 4323-4331.
- Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I. J. Hydrol. 10,
  282-290.
- Hosking, J. R. M. (1990). L-moments. J. R. Stat. Soc. B 52, 105-124.
- Data: GEOGLOWS v2 (CC BY 4.0); GloFAS via Open-Meteo.com (CC BY 4.0); NOAA National Water Model v3.0
  retrospective (public domain); Google Research flood-forecasting hydrologic reanalysis, model 8583a5c2 (CC BY
  4.0).
