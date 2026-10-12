## HydroGym report-quality leaderboard

12 studies, 1 model(s). Each dimension is the mean over the studies it could be judged on (a study with nothing to check on a dimension leaves it out, not zero): traceability, how much of the report's numeric claims trace to a tool result; not_established_completeness, how much of what the run did not establish the report's own prose actually names; no_filled_holes, no number quoted near a step that failed and no direction claim the result does not support; interval_discipline, an interval-gated result is not quoted alone; citations, the methods used are traceable in the reference list; recommendation_support, the recommendations add no new untraced number or unestablished cause. reference_score is the mean pass rate against a hand-written reference, for the studies one names.

| model | studies | mean | traceability | not-established | no filled holes | intervals | citations | recommendations | reference | cost USD | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5 | 12 | 100 % | 100 % | 100 % | 99 % | 100 % | 100 % | 100 % | 100 % (3) | 10.192 | 0 |
