# Did the forecast come true? Scoring issued forecasts against the gauge

Since 2026-10-08 the Archive has kept every day's forecast for its live gauges (`forecasts/issued/<date>.parquet`)
and every day's gauge value (`forecasts/status/<date>.parquet`), see [Archive layout](archive-layout.md). Read
together, they are a verification record: what GEOGLOWS and GloFAS said a river would do, and what the gauge
then measured. `aquascope.archive.verify` scores that record (#785). The [evidence ladder](evidence.md) asks how
well a model reproduces a gauge's past; this asks how well its forecasts did, day by day, at each lead time.

```bash
aquascope evidence verify --out build --days 14          # the newest 14 issues, read from the Hub
aquascope evidence verify --out build --archive ./gauges  # a local copy of the dataset
python -m aquascope.archive.verify --out build --since 2026-10-08
```

It writes `build/verify/forecast_skill.parquet` (one row per model, variant, lead day and month),
`build/verify/forecast_skill_gauges.parquet` (each gauge's own record, below), `build/verify/summary.json`
(pair counts, the 1-day-lead headline, error by lead, the per-gauge window and trust counts) and
`build/verify/manifest.json` (what was read). Everything is computed from the published files by code: no key,
no model call, and the same files give the same numbers.

The daily `forecast-archive` job runs the same scoring right after each issue, on every issue so far, and
publishes these files as the Archive's `verify/` prefix ([Archive layout](archive-layout.md)). The
scorecard has 10 minutes of its own (`--verify-budget-min`); when some input could not be read in that time, the
day's files are kept in the run's artifact but not published, and the job's health record says so.

## Pairing

An issued row (gauge, model, valid date, lead day) is matched with the gauge's value for the same day from the
status snapshots. A snapshot carries each gauge's newest value, usually yesterday's, so the match is on the
value's own date, not the snapshot's; when several snapshots carry the same day, the newest one wins. Only
positive flows are scored, on both sides, because the errors are in log flow.

Raw and corrected-to-the-gauge values are separate rows (`variant` = `raw` or `corrected`), each with its own
pair count. Both models are issued corrected to the gauge: GEOGLOWS from its retrospective simulation, GloFAS
(since #790) from its reanalysis at the gauge's cell. A gauge's GloFAS rows stay raw until its reanalysis record
has been read, which the daily job does within a call budget, so GloFAS corrected pairs ramp in over the first
weeks and its corrected rows rest on fewer gauges than the raw ones until then.

## The scores

| Column | What it is |
| --- | --- |
| `median_abs_log10_error` | median of `abs(log10(forecast) - log10(observed))`, the forecast being the ensemble mean (the number the river card and Floods ahead show) |
| `factor` | `10 ** median_abs_log10_error`: half the forecasts were within this factor of the gauge |
| `median_log10_error` | the signed median: below zero, the model forecasts too little water |
| `crps_log10` | mean CRPS of the stored quantiles in log10 flow (below) |
| `hits`, `misses`, `false_alarms`, `correct_negatives` | exceedance of the gauge's own 2-year flow (`gauge_q2`), forecast mean against observed |
| `hit_rate`, `false_alarm_ratio` | `hits / (hits + misses)` and `false_alarms / (hits + false_alarms)`; empty when the denominator is zero |
| `n_pairs`, `n_gauges`, `first_valid`, `last_valid` | what the score stands on |
| `enough_pairs` | false below `--min-pairs` (30 by default): a card should say "not enough pairs yet" |

`month` is the valid date's month, plus `all` pooling every month.

**CRPS from three quantiles.** The issued files keep the ensemble's 25th, 50th and 75th percentiles, not the
members. The continuous ranked probability score is twice the integral of the quantile (pinball) loss over all
probability levels (Gneiting and Raftery 2007), so with levels 0.25, 0.5 and 0.75 it is approximated as

    CRPS ~ 2/3 * sum over tau of QS_tau(q_tau, y),   QS_tau(q, y) = (1{y < q} - tau) * (q - y)

computed on log10 flow, so a brook and a big river weigh the same and a perfect point forecast scores its
absolute log10 error. Three quantiles make it coarse and blind to the tails beyond the quartiles; compare it
between models and leads, not with a CRPS computed from full ensembles elsewhere.

## Each gauge's track record

`forecast_skill_gauges.parquet` scores every gauge on its own, over the newest 30 days of valid dates in the
record: one row per gauge, model, variant and lead window, the window being 1 to 3 days ahead (what the map
quotes) and each single lead day. It has the same scores as the pooled table (pair count, dates, factor, signed
median, CRPS, exceedance counts), the GEOGLOWS `river_id` the gauge was issued on, its position from
`forecasts/reaches.parquet`, and a trust word from the factor:

| Trust | Factor (half the forecasts were within it) |
| --- | --- |
| good | 2 or less: half to double the gauge's flow |
| fair | more than 2, up to 4 |
| poor | more than 4 |
| unknown | fewer than 30 pairs in the window, or no score |

The bands are round numbers chosen by AquaScope, not a published standard, and fixed in
`aquascope.archive.verify.TRUST_BANDS` so a word means the same at every gauge. A gauge is issued on only some
days (the daily job takes up to 250 gauges in turn), so 30 days usually give it fewer than 90 pairs, and many
gauges are still `unknown`.

## On the map and in the tools

`aquascope.forecast_skill` reads the published rows back and says them; nothing in it computes a score.

- **The river card.** A gauge's card lists, per model, "within a factor of X (N pairs, dates)" for 1 to 3 days
  ahead, corrected to the gauge where it was issued so, or "not enough pairs yet". A river reach's card does the
  same, raw, from the gauges on that reach. Each line carries its trust badge.
- **Floods ahead.** A flooded reach's card shows the trust badge of the raw GEOGLOWS forecast (what Floods ahead
  classes) at the best-sampled gauge on the reach, or `unknown` when no gauge on it has been verified.
- **The legend.** "Forecast track record" in On the map explains the four words and gives every gauge pooled at
  1 day ahead; its switch hides the lines and badges.
- **The evidence ladder.** `model_skill` (and the Explorer's Evidence tab) adds `forecasts`: the models and
  variants ranked by their verified factor at 1 to 3 days, where at least 30 pairs exist, then those without.
- **MCP.** `forecast_skill(station_id or lat/lon, model, lead_days, variant)` returns the gauge's rows, the
  ranking, the pooled rows for the same leads and the trust bands.

When `verify/` is not in the Archive, or a gauge has no row, the card, the badge and the legend row show
nothing.

## Limits

- **Sample size.** The record starts on 2026-10-08. A month and lead with a few hundred pairs is a few weeks of a
  few hundred gauges, so the scores move as it grows, and `enough_pairs` only guards against the thinnest rows.
- **The gauge mix differs.** Each model, variant and lead is scored on the gauges it has pairs for. GEOGLOWS
  needs a river reach, the corrected value needs a long enough record, and long leads only have pairs from older
  issues. A difference between two rows is partly a difference in rivers, not only in skill.
- **Floods are rare.** The hit rate needs observed floods above the 2-year flow. The first weeks of autumn 2026 had
  a handful, so the rates are mostly empty or rest on single digits; read the counts, not the ratio. The raw value
  is compared with the gauge's 2-year flow too, which is what a reader holding the raw number would do, and
  explains why raw GEOGLOWS, which forecasts too little water at most gauges, rarely raises an alarm.
- **The observation is provisional.** It is the agency's newest daily value when the snapshot was made; later
  revisions are not picked up.
- **A month is short.** A gauge's 30-day window holds a few dozen pairs at best, so its trust word can change from
  one week to the next. The card always gives the pair count and the dates next to the factor.
- **GloFAS corrected is young.** Its corrected values started with #790 (November 2026), so its corrected rows
  cover fewer days and gauges than GEOGLOWS corrected; compare them on matching months and leads.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. J. Am. Stat.
  Assoc. 102, 359-378.
- Laio, F., & Tamea, S. (2007). Verification tools for probabilistic forecasts of continuous hydrological
  variables. Hydrol. Earth Syst. Sci. 11, 1267-1277.
