# Report quality after the engineer-in-the-middle arc (2026-09-15)

The twelve recorded studies were run again on Claude Sonnet 5 with the crew of #411 (a failed gate fails its
step, the reviewer's-eye gates, the Interpreter with its findings and grades, the data requests), 10.19 USD for
the twelve. Scored with `aquascope gym reports bench` against the same checks as the 2026-09-14 row, and against
the three references rewritten for the new recordings.

| study | grade | decision | mean | notes |
| --- | --- | --- | --- | --- |
| kingston-flood | established | Q100 652.5 m3/s, band 565 to 723 | 1.00 | the GloFAS cross-check ran and its ratio gate failed (the Open-Meteo cell drains a tributary, 14.8 against 652.5); the report says so and quotes no cross-check as if it agreed |
| cambridge-groundwater | not established | Sen's slope +0.166 mAOD/yr on the full record | 0.97 | the ten-year window the brief singled out holds 9.7 years and failed its gate; the report answers "rising, and the recent comparison is not established" and names the abstraction records the cause question would need |
| toulouse-irrigation | indicative | peak demand met on 96.5 % of growing-season days | 1.00 | the 116-year record carries a significant decreasing trend (p < 0.001) and the report says the historical reliability is optimistic for it |
| potomac-supply | established | Q95 50.1 m3/s against an 8 m3/s abstraction | 1.00 | |
| sintra-supply | not established | demand met 83.2 % of days, from donors | 1.00 | the flow-duration step found no record and its fallback failed; the regional screening stands, labelled |
| wagga-flood | screening | no number answers the levee question | 1.00 | the regional transfer gives mean flow and Q95, not a 100-year flood, and the decision says so instead of quoting the mean flow |
| bamako-ungauged | screening | mean flow 0.78 mm/d from donors | 1.00 | |
| nairobi-drought | screening | SPEI-3 -1.40, moderately dry | 1.00 | |
| orleans-drought | indicative | baseflow index 0.79 with SPEI | 1.00 | |
| own-table-flood | indicative | Q50 166.7 m3/s on the client's table | 1.00 | the quality step failed its gate; the fit stands on the table |
| taichung-drought | indicative | SPI-3 0.87, near normal | 1.00 | the 130-year catalog span served ten usable years; the gate said so and the fallback carried the index |
| little-falls-water-quality | established | CCME WQI 100 of 100 | 1.00 | |

What the row shows that the 2026-09-14 one could not: every study ran every planned step or skipped it for a
stated reason; every answer carries a grade computed from the run; the three studies whose data fell short
(Cambridge's window, Sintra's record, Wagga's regional path) say so in the decision instead of in a footnote.
The two dimensions below 1.00 are the scorer's conservative reading of conditional sentences ("if that trend
were negative, the decision would shift toward decline") as direction claims; the references pass.

The 2026-09-14 row (mean 0.99 on the 2026-09-07 recordings) stays as the baseline: it caught two hand-converted
units in the Sintra study and a "downward trend" claimed on a p = 0.302 test in the Toulouse study that the
Critic had flagged as a note and never forced fixed. Both are gone: a failed check is now a fix, and the
Toulouse trend is significant on the longer record the new run fetched.
