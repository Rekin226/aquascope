# The AI roles on the device (#590)

**Question.** With no key, do the Explorer's AI roles (Ask the map, Explain this view,
Scout, Review my model results, What if, the Ask panel and the Studio brief) give answers a
reader can trust when Chrome's built-in model writes them?

**Answer: mostly, where a claim lock exists.** The locks in `map_explain.claim_lock`,
`map_scout.check_wording` and `whatif_map.parse_scenario_reply` held: no number outside the
facts reached a pin, a note or a scenario. The Ask panel had no lock, and it is where the
worst answers came from: numbers from an earlier question, a trend never computed, gauge
ids swapped between sites. Seventeen defects were found; the table at the end says where
each one was fixed. A recheck on the device after those fixes, and the second round of fixes it
led to, follow it; then a third check, and the change of design it led to: the model picks, the code
writes (Round 3, at the end).

The other half of #590, the same roles with a real key (the reader's own model through
Ask's provider list), is **still to be run by the maintainer**. Nothing on this page covers it.

## How it was run

- Date: 2026-10-25.
- Browser: the maintainer's Chrome 155 on a Mac, window 865 x 447 CSS px.
- Model: Gemini Nano through Chrome's Prompt API (`LanguageModel.availability()` was
  "available"). No key stored.
- Explorer: the live build at <https://rekin226-aquascope-explorer.static.hf.space/>
  (repo `main` at 49b00c1).
- WebLLM: not tested. When the Prompt API is present `loadLocalModel` uses it and the UI
  offers no WebLLM choice.

Every request went through the page's own UI (the box, the buttons, the panels). To see
what the model was sent and what it replied, `LanguageModel.create` was wrapped in the page
to log each prompt, reply and time. Map moves between requests and "Undo all" were done
from the console. One What-if line was submitted with `requestSubmit()` because the panel's
input was clipped at this window height. Three extra Studio brief checks called the Prompt
API directly with the page's own system prompt and schema (marked below).

Times are from submit to the final line on screen; "model" is the Prompt API call alone.

## Ask the map

| Request | Engine | Reply | Time (model) | Numbers in the facts? |
|---|---|---|---|---|
| show floods ahead in Bangladesh | device | fly to Bangladesh, Floods ahead on | 19.8 s (1.3 s) | yes (no numbers) |
| fly to the Rhine and show river status | device | fly to Rhine (bbox of the river), status on | 2.7 s (1.0 s) | yes |
| take me to the Mekong delta and turn on groundwater | rules | fly to Mekong Delta, groundwater on | 0.2 s | yes |
| where is it driest in Africa right now? zoom there | device | fly to bbox [15, 20, 30, 40] the model made up; status, surface water and land cover on; 2 actions refused and said | 4.3 s (2.7 s) | no: the bbox is the model's own |
| what did the Indus look like during the big Pakistan floods? | device | Floods past on, fly to "Indus, Sindh" (a town) at zoom 8; date left at 2026-09-15 | 2.2 s (1.2 s) | yes, but the question (the 2022 floods) was not answered |

## Explain this view

| View | Engine | Notes | Time (model) | Numbers in the facts? |
|---|---|---|---|---|
| Bangladesh, zoom 4 | rules (device tried) | 3 rules notes. The device's one note was refused, but the line says "Rules, no model" | 5.0 s (1.2 s) | the lock caught 20 and 75/90 under the wrong fact |
| Europe, zoom 4 | device | 2 device notes and 1 rules; 3 refused (no place) | 8.1 s (5.9 s) | yes. Wording: "96% ... significantly below normal" (the fact is "below or much below"); values said twice |
| US mid-Atlantic, zoom 6.5 | device | 1 device note and 1 rules; 1 refused | 5.8 s (3.8 s) | the lock caught "59" (from another fact). A passing note reads "54% (54%) ... 41,000 km² (41,000 km²)" |
| Africa, zoom 3 | device | 2 device and 1 rules; 1 refused | 8.5 s (4.2 s) | yes. "6,000,000 km² km²", values repeated in brackets |
| Australia, zoom 4 | device | 3 device notes; 2 refused | 9.1 s (6.0 s) | yes. Each note ends "This is located at a point on the map." (copied from the prompt) |

## Scout

| View | Engine | Result | Time (model) | Numbers in the facts? |
|---|---|---|---|---|
| world, zoom 1.2 | device | 7 pins; 3 device wordings, 4 refused | 21.8 s (13.9 s) | yes |
| Europe, zoom 4 | device | 2 pins from 4 findings: the model mangled two ids ("2026-09-1" for "2026-09:1") and those findings were dropped | 17.9 s (8.0 s) | yes; one reason adds advice ("worth validating") |
| US, zoom 4 | rules (device tried) | 4 pins, all 4 wordings refused: each wrote "January 2026" or "September 2026" outside the slots | 13.8 s (6.5 s) | the lock refused correct years |
| South Asia, zoom 4 | rules (device tried) | 1 pin, wording refused (a literal "January 2026") | 8.8 s (2.2 s) | the lock refused a correct year |
| South America, zoom 3 | rules (device tried) | 1 pin, wording refused (a literal "September 2026") | 2.7 s (1.9 s) | the lock refused a correct year |

Pin counts are low partly because this window is short and Scout leaves out pins that would
sit under a bar.

## Review my model results

Sample: HEC-RAS 2D, Portsmouth, Virginia (6,272 cells, 109 steps). 9 of 9 checks ran each time.

| Request | Engine | Result | Time (model) | Numbers in the facts? |
|---|---|---|---|---|
| review my model results | device | 8 pins; 6 device wordings, 2 refused | 29.0 s (7.6 s) | yes. Meaning flipped: the rules say "worth a look if it is not a channel or a pit you expect", the device "investigate if it's expected for the area" |
| is anything wrong with my flood model? | device (Ask the map) | not routed to the review: the model turned Floods past on | 1.5 s (0.8 s) | n/a |
| check my model | device | 1 device wording, 7 refused (reasons not shown) | 7.5 s (7.1 s) | numbers yes, claim no: "8 cells ... (< 0.0269 m)"; the rules say 8 cells are under 0.03 m and 0.0269 m is one cell's maximum |
| review my model results (MODFLOW Freyberg sample now on screen in Texas) | device | reviewed the older HEC-RAS mesh, pins off screen in Virginia; the final line does not name the file | 31.3 s (11.2 s) | yes |
| legend "Review these results" | device | 8 pins, 1 refused | 8.1 s (7.4 s) | yes |

## What if

Reach: a stream east of Siegburg (reach 230428063, 1,379 km²). The Rhine at Cologne
(143,851 km²) offered "What if..." but could not run: it is above the 100,000 km² limit.

| Line | Engine | Read as | Time (model) | Numbers in the words? |
|---|---|---|---|---|
| a hotter and drier climate | device, refused | model said rain -10 %, +1 °C; both refused; the line suggests Chrome's model, which was just used | about 21.5 s, model called after 20.3 s (1.2 s) | the lock caught invented numbers |
| what if the town pumps 300 litres per second out of the river | device, refused | model said demand 0.3 m³/s (correct); refused | model called after 20.6 s (1.2 s) | false refusal: "litres per second" is not read as l/s |
| summers get much hotter, maybe three degrees | device, refused | model said +3 °C (correct); refused | model called after 0.4 s (0.8 s) | false refusal: "three" is not counted as a number |
| rain falls by a fifth and evaporation goes up 10 percent | rules | "20% less rain"; the evaporation clause dropped without a word | 33.8 s (GR4J calibration) | yes, but half the line ignored |
| drought: precipitation down by a quarter with 1.5 C of warming | device | "25% less rain, 1.5 °C warmer"; mean flow -47 % (-48 to -46 %) | 23.0 s, model called after 20.6 s (1.2 s) | yes |

The result numbers (flow change, KGE) are the code's; the model only fills the four parameters.

## The Ask panel on the device

Gauge: USGS 01646500, Potomac at Little Falls. "Tell it what I am looking at" on.

| Question | Engine | Reply (short) | Time (model steps) | Numbers in the tool results? |
|---|---|---|---|---|
| Summarise the record ... period, mean, trend, and the flood frequency | device + analyze_station | mean 320.7738 m3/s, 1930 to 2026, min 3.43, max 12100, "trend ... is increasing", flood frequency "from the flow-duration curve" | 30.4 s (1.2 s, 3.8 s) | numbers yes. The trend and the GEV never reached the model: the payload was cut at 3,000 characters, inside a series |
| What is the 100-year flood at this gauge? | device, no tool call | "The 100-year flood (Q10) ... not directly available ... record from 1930-03-01" | 2.2 s (2.1 s) | no: Q10 and 1930-03-01 come from the previous question's tool result, still in the session; Q10 is not the 100-year flood |
| Find the five gauges nearest to Kingston upon Thames | device + find_stations | failed at step 2: "kErrorUnknown" | 2.9 s | n/a; the model passed near = [51.1, 0.0], about 35 km from Kingston |
| What is the climate and modelled discharge at 25.05, 121.55? | device + anywhere | 1733.4111 mm, 22.7201 °C, aridity 1.4838 (humid), GloFAS mean 5.6336 m3/s 2006 to 2026, "annual" min 1.35 and max 214.28 | 31.4 s (22.8 s, 4.4 s) | numbers yes; the min and max are daily, not annual |
| Is the Potomac at Little Falls getting drier? Use the annual-mean trend. | device, no tool call | "an increasing trend ... Mann-Kendall ... indicates an upward trend" | 1.6 s (1.4 s) | no trend result was ever seen. The recorded key answer gives Sen's slope +0.059 m³/s per year, p = 0.82: no trend |
| (retry) Find the gauges nearest to Kingston upon Thames | device + find_stations | five Kingston gauges with ids | 6.0 s (1.0 s, 4.5 s) | ids in the payload but swapped between sites; "nearest" is not what the tool returned |

## The Studio brief on the device

| Sentence | Engine | Brief read | Time | Numbers in the sentence? |
|---|---|---|---|---|
| I need the design flood for a footbridge here, something like the 50-year peak, using data since 1980 | device | return_period 50, "Design flood calculation"; the Consultant: 2 known, 0 to ask | 0.8 s | yes ("since 1980" not captured) |
| Will the road by the pump station flood this winter? | device | return_period 1 (not in the sentence; dropped by the 2 to 100,000 range), quantities ["this winter"] | 0.7 s | no, caught by the range only |
| Is this site safe from a big flood for a new school? (direct call, same prompt) | device | return_period 0 (dropped) | 0.7 s | no, caught by the range only |
| How low does the river get in a dry summer? We plan a water intake. (direct call) | device | return_period 0 (dropped) | 0.8 s | no, caught by the range only |
| Has the river been getting more extreme since the dam was built? (direct call) | device | return_period 0 (dropped), quantities ["none"] kept | 0.7 s | no, caught by the range only |

The on-device plan (at review) and prose (after the run) were not reached in this session.

## What got past the locks

1. **The Ask panel had no lock.** Numbers from a previous question ("Q10", "1930-03-01"),
   a trend direction never computed, gauge ids swapped between sites, invented search
   coordinates.
2. **Right numbers, wrong claim.** The review note "8 cells ... (< 0.0269 m)" and the
   Explain note "96% ... significantly below normal". Each number is in the cited facts,
   but they are put together wrongly. A presence check cannot catch this.
3. **Ask the map: a bounding box the model made up** for "where is it driest in Africa",
   applied as the answer.
4. **Studio brief: no check that a return period or timescale is in the sentence;** only
   the range check stopped the invented 1 and 0.

Console: no errors from the AI roles; two MapLibre missing-sprite warnings.

## The defects and where they were fixed

Two pull requests fix them: **A** is #639 on `fix/ondevice-ask-590` (the Ask panel, Ask the map,
the sampling and the Studio brief), **B** is #640 on `fix/ondevice-roles-590` (Scout, Explain, What
if and the review). Neither was checked on a real on-device model after the
fix: headless Chrome has none, so each fix is covered by tests that use the recorded model
replies above.

| # | Where | Defect | Fixed |
|---|---|---|---|
| 1 | `explorer/src/local-model.js` `loadLocalModel`, `askLocally` | one Prompt API session served every question, so earlier tool results answered later questions | A: one base session, never prompted; each question gets a clone of it (or a new session) and destroys it after |
| 2 | `local-model.js` `askLocally` | the tool result was cut at 3,000 characters, so the trend and flood frequency never reached the model | A: `aquascope.ai_engine.device_ask.compact_result` keeps the record, daily statistics, FDC points, return levels with their interval and the trend with its reading in words, and drops series and citations |
| 3 | `explorer/src/ask.js` `runLocally`, `local-model.js` `localAnswerHtml` | no claim lock on the on-device answer | A: `device_ask.lock_answer` holds every number, date, Q label and gauge id to this question's results (rounding allowed), checks that an id sits next to its own gauge's name and that trend words agree with the Mann-Kendall result; a refused answer is replaced by the tools' facts and one line saying why |
| 4 | `aquascope/map_scout.py` `apply_wording` | findings the model left out of its order were dropped | B: they follow in the rules' order up to `max_pins`, and an id written with other punctuation ("2026-09-1" for "2026-09:1") still finds its finding when only one reads that way |
| 5 | `map_scout.py` `check_wording` | a correct year written out ("September 2026") was refused | B: `slots_back` puts a slot's exact value back into its `{slot}` before the checks; another month or year is still refused, and so is a sign against the written value |
| 6 | `explorer/src/map-explain.js` `explainView` | when every device note was refused, the line said "Rules, no model" | B: the line says the model's wording was refused and lists the refusals (`map-actions-core.js` `explainWho`) |
| 7 | `aquascope/map_explain.py` | values said twice, prompt text copied into notes | B: `drop_repeats` writes a value once ("54% (54%)", "km² km²"; Scout and the review fill through it too); the prompt no longer offers "a point on the map", and a sentence that only copies the prompt is dropped |
| 8 | `aquascope/whatif_map.py` | "litres per second" and number words not read | B: litres or liters per second, l/s, L/s, cubic metres per second and number words are read ("one" only before a unit); Nano's 0.3 m³/s and +3 °C pass, and the digit in "m3/s" no longer counts as a number said |
| 9 | `explorer/src/whatif-map.js` `readLine` | the device's refusal not shown; the hint suggests the model just used | B: the line names the model and its reasons, and the hint names only a way not yet tried (`whatif-core.js` `notReadLine`) |
| 10 | `whatif-map.js`, `whatif_map.parse_scenario` | a partial rules match dropped a clause in silence | B: the clause in the run was read by the rules but dropped by the sliders (evaporation has none); it now goes with the run and is said. When the rules read only part of a line and no model reads it, the line names the words not read and nothing runs |
| 11 | `whatif-map.js` `readLine` | about 20 s before the device is called | B, in part: the rules' reading and the device prompt are one worker call, and a quick call goes to the main worker when every light worker is busy reading the network and the main worker is idle. Headless with a stub, the one light worker held by a slow host: 20.6 s before, 4 ms after. With the main worker busy as well the wait remains, and if the live 20 s was Chrome reloading Nano, this does not help; neither could be told apart without the device |
| 12 | `aquascope/map_commands.py` | "is anything wrong with my flood model?" did not reach the review | A: the review and explain phrases are wider, and a model may answer with `{"control": "review"}` or `{"control": "explain"}` |
| 13 | `explorer/src/map-command.js` `byDevice`, `map_commands` reply check | a bbox the model made up was applied as the answer | A: `map_commands.ground_places` keeps a box or point only when the words hold its numbers; otherwise the place the model named, the extent of a region the words name ("Africa"), or the model's label as a place the words say; failing those, refused with "name a place or a region" |
| 14 | `local-model.js` `promptOptions` | `LanguageModel.params` is undefined in Chrome 155, so the 0.1 temperature was never set | A: feature-detected. An extension (with `params()`) gets `topK` and `temperature`; a web page gets `samplingMode`, which only the sampling origin trial honours (checked against the Prompt API docs on 2026-10-25), else Chrome's own sampling; a Chrome that refuses the option gets a session without it |
| 15 | `explorer/src/intake.js` `parseBriefReply` | a return period not in the sentence was kept when in range | A: a return period or timescale is kept only when the sentence states it, in digits or words ("50-year", "hundred-year", "twelve months") |
| 16 | `explorer/src/mesh-review.js` `reviewModel` | the review did not name its file, and reviewed an older mesh off screen | B: from Ask the map the review is of the model on screen; a MODFLOW model there, or the mesh out of view, is said instead (`mesh-review-core.js` `reviewTarget`). The legend row and WebMCP still review the loaded mesh, and the final line names the file |
| 17 | `explorer/src/map-card.js` | "What if..." offered on reaches above the 100,000 km² limit | B, in the What if panel rather than the card: it checks the basin as it opens (`whatif_map._basin_check`, main worker) and says the limit with Run off. The Rhine at Cologne, headless: 143,881 km², said after 16 s. The card still offers the button |

"Right numbers, wrong claim" (escape 2), fixed in B for the two cases seen, by checks that look
past the presence of a number:

- Explain "96% ... significantly below normal": `map_explain.class_claims` refuses a claim of the
  stronger class ("much", "significantly", "well below normal") unless a cited fact is of that
  class, and a side of normal no cited fact is on. It reads a note as a whole, so a note citing
  both the share and a much-below area would pass.
- Review "8 cells ... (< 0.0269 m)": the film check's threshold is a fact of its own ("A film is
  shallower than", 0.03 m) and the count names it; a sentence that puts the count under another
  value is refused. Only the film check's count carries a bound so far.

Other ways to put right numbers together wrongly are not caught. The real-key run of #590 is
still to be done by the maintainer.

## Recheck after #639 and #640

Date: 2026-10-26.
Browser: the maintainer's Chrome 155 on a Mac, window 865 x 447 CSS px, a new tab, hard-reloaded.
Model: Gemini Nano through the Prompt API (`LanguageModel.availability()` = "available"). No key stored (Ask's "forget key" control hidden); no key read or entered.
Explorer: the live build at rekin226-aquascope-explorer.static.hf.space, asset version `c406699`, which loads `src/ask-local-core.js`.

How it was run. As in the first run: every request through the page's own UI or its own entry points (`__aq.mapAsk.run`, `__aq.mapExplain.run`, `__aq.scout.run`, the What-if form), with `LanguageModel.create` wrapped to log prompts, replies and times. Map moves were done from the console. Before the Scout runs the selected gauge was cleared from the console (see Scout). Marked "direct" below: Prompt API calls made from the console with the page's own system prompt and schema, and answers sent to the page's own lock in the worker (`device_ask`, op `lock`).

Times are from submit to the final line; "create" is `LanguageModel.create`, "model" is the prompt call.

### Before and after

| Baseline defect | Now |
|---|---|
| Ask: one session for every question, earlier results answer later questions | Gone. Each question gets a clone, destroyed after. "And what is its 50-year flood?" called the tool itself |
| Ask: tool payload cut at 3,000 characters, trend and GEV never seen | Gone. The step-2 prompt is 1,693 characters with stats, return levels and both trends |
| Ask: no lock | Lock in place. Replayed baseline answers are refused: "Q10 ... 1930-03-01" with no tool run (date and Q10 refused), "an increasing trend" (trend wording refused) |
| Ask: "kErrorUnknown" once | Worse: 7 of 12 UI runs stopped at step 2 with "kErrorUnknown" (see Ask) |
| Ask the map: model's own bbox applied for "driest in Africa" | Gone. "refused the model's own box, replaced by the extent of Africa" |
| Ask the map: "is anything wrong with my flood model?" not routed to the review | Gone. The review ran |
| Explain: "Rules, no model" when every device note was refused | Gone. The line names the model and gives each refusal's reason |
| Explain: "54% (54%)", "km² km²", "This is located at a point on the map" | Not seen. New copies: "is pinned, unnamed", "on on Sat 7 Nov" |
| Scout: findings with mangled ids dropped | Gone. Europe: 3 findings, 3 pins, though Nano again wrote "2026-09-1" and "2026-01-275-407" |
| Scout: written-out years refused | Gone. "September 2026" and "January 2026" pass |
| What-if: "300 litres per second" and "three degrees" refused | Gone. Read as 0.3 m³/s and +3 °C |
| What-if: refusal not shown, hint suggests Chrome's model | Gone. "Chrome's built-in model: refused dp_pct 10: that number is not in the words" |
| What-if: evaporation clause dropped | Gone. Rules read "20% less rain, 10% more evaporation" |
| What-if: about 20 s before the device call | The wait before the call is now 3 to 62 ms, but `LanguageModel.create` itself took 20.6 and 21.1 s in 2 of 4 device reads |
| What-if: Rhine at Cologne offered a run it could not do | Gone. The panel says "143,881 km2 drain to this reach ... the limit here is 100,000 km2" and Run is disabled |
| Review: MODFLOW on screen, older HEC-RAS mesh reviewed off screen | Gone. The line says the MODFLOW file is on screen and not covered, and names the loaded mesh that is out of view |
| Review: "8 cells ... (< 0.0269 m)" | Not seen |
| Review: meaning flipped ("investigate if it's an expected feature") | Still there, on 5 pins |
| Temperature 0.1 never set | samplingMode "most-predictable" is passed and `create` does not throw, but the session reports no `samplingMode`, so the page says "Chrome's default sampling" (samplingNote). It does not claim a setting it cannot see |

### Ask panel

Gauge USGS 01646500 selected, "Tell it what I am looking at" on.

| Question | Result | Steps (model) | Lock |
|---|---|---|---|
| Summarise the record at this gauge: period, mean, trend, and the flood frequency (4 runs) | 4 of 4 "The on-device model could not run: ... kErrorUnknown" after analyze_station | 0.9 to 21.1 s, then error in 0.8 s | not reached |
| What is the 100-year flood at this gauge? (2 runs) | 1 error; 1 answer: 10795.57 m3/s | 0.93 s, 2.6 s; total 9.3 s | passed. The value is the LP3 100-year; the GEV is 11247.39 and the method is not named |
| Is the Potomac at Little Falls getting drier? Use the annual-mean trend. (2 runs) | 2 of 2 kErrorUnknown | 1.3 and 21.9 s, then error | not reached |
| Find the five gauges nearest to Kingston upon Thames | analysed one gauge (Kingston, 8496ce69-..., id right) instead of listing five | 21.7, 1.5, 3.7 s; total 29.7 s | passed |
| What is the climate and modelled discharge at 25.05, 121.55? | 5.63 m3/s, 22.72 °C, 1733.41 mm; min and max no longer called annual | 0.86, 2.95 s; total 8.1 s (was 31.4 s) | passed |
| New: compare the mean flow of the Potomac (USGS-01646500) with the Thames at Kingston | means, periods and maxima right; "the Potomac's ... q95 of 3.43 m3/s" is wrong (3.43 is its daily minimum; its Q95 is 35.7) | 0.96, 1.7, 5.7 s; total 14.6 s | passed: escape |
| New: And what is its 50-year flood? | 9150.06 m3/s with its 90 % interval, from a fresh tool call | 0.92, 1.3, 2.3 s | passed |

Claim-lock pass rate. UI: 5 answers reached the lock, 5 passed. Direct (page's prompt and compact result as a single first prompt, then the page's lock): 9 answers, 9 passed. Of these 14 passing answers, 2 show a wrong number as fact: the q95 above, and "the return period for a flood of magnitude 2 m3/s is 2 years, and ... 10 m3/s is 25 years" (direct). The trend answers were right in 3 of 3 direct runs ("no significant trend ... Sen's slope -0.0365 m3/s per year").

The kErrorUnknown failure, from direct trials on the same prompts:
- a second prompt in one session, after a first prompt with `responseConstraint`, failed in 11 of 13 trials, with and without `samplingMode` and on clones and fresh sessions alike;
- with no constraint on the first prompt, 2 of 2 passed;
- the same step-2 text as the only prompt of a fresh clone passed 4 of 4.

Per-step time with fresh sessions: step 1 took 0.86 to 1.25 s in 9 of 12 runs and 21.1 to 21.9 s in 3; later steps took 1.3 to 5.7 s. The first question also waited 16.5 s for the base session.

### Ask the map

| Request | Engine | Result | Time (create, model) |
|---|---|---|---|
| where is it driest in Africa right now? zoom there | device | bbox refused and said, Africa's extent used; it also started an unasked playback, September to October 2026 by month. "Driest" not answered | 23.7 s (20.9 s, 2.7 s) |
| is anything wrong with my flood model? (HEC-RAS Portsmouth on screen) | rules, then device wording | the review ran: 8 pins, final line names Flood_model.p01_BC12_S6_R6.hdf, 1 wording refused | 32.8 s (20.8 s, 9.2 s) |
| New: does this simulation look plausible to you? | device | `{"actions": []}`, "Not understood"; not routed to the review | 1.8 s (1.2 s, 0.4 s) |
| New: tell me in plain words what this map is showing me | device | flew to the selected gauge and turned river status on; not routed to Explain | 3.2 s (0.8 s, 1.2 s) |
| (review) are the depths in my flood model realistic? | device | flew to and lit up the Nile, the example in the system prompt; 2 actions applied | 10.3 s (0.7 s, 1.1 s) |

### Explain

| View | Result | Time (create, model) | Passing device notes |
|---|---|---|---|
| Bangladesh, zoom 4 (run 1) | 2 device + 1 rules; 3 refused, reasons on the line | 13.1 s (1.2 s, 9.5 s) | readable; one gives the largest area's 20 % as "20% of the mapped land is below normal" (27 % is below or much below) |
| Bangladesh, zoom 4 (run 2) | 1 device + 2 rules; 4 refused | 5.6 s (0.8 s, 4.7 s) | escape: "The river is mapped at 21% above normal". The 21 % is the share of the view covered by the largest above-normal area, not a flow |
| Europe, zoom 4 | 1 device + 2 rules; 0 refused | 24.9 s (21.1 s, 1.8 s) | right: "92% of the view ... much below normal ... 1,400,000 km²" |
| New: Australia, zoom 4 | 1 device + 2 rules; 3 refused (90, 10/25, 2.0 not in cited facts) | 10.3 s (1.0 s, 5.8 s) | right (news counts) |
| New: Africa, zoom 3 | 4 device; 1 refused | 12.4 s (1.1 s, 7.6 s) | escapes: "flow at least 81% below normal" (81 % is the land below or much below normal), "flow above normal at 6%" (6 % of land). Also "is pinned, unnamed" copied from the prompt and "on on Sat 7 Nov" |

Refusal reasons list "²" as a number when a cited fact has no km² ("(760000, ²)").

### Scout

| View | Result | Time (create, model) |
|---|---|---|
| Europe, zoom 4, gauge USGS 01646500 still selected | 3 of 3 pins, all in rules wording: Nano ordered only "USGS-01646500", from the context line, which names the selected gauge though it is off screen. The line still says "worded on this device" and reports no refusal | 45.8 s (20.8 s, 1.9 s) |
| Europe, zoom 4, selection cleared (2 runs) | 3 of 3 pins each time; 3 device wordings, then 2 device + 1 refused. One passing reason reverses the finding: "poor agreement with a gauge's water flow, raising questions about the gauge's measurement" (the rules say the models miss this river) | 22.3 s (0.3 s, 5.2 s); 20.7 s (0.3 s, 5.0 s) |
| South America, zoom 3 | 1 pin, device wording passed ("September 2026" written out) | 3.3 s (0.5 s, 1.8 s) |
| New: US, zoom 4 | 4 pins, 4 wordings refused. Right refusals: Nano wrote a slot that does not exist, `{year}` | 25.5 s (0.3 s, 6.1 s) |
| New: South Asia, zoom 4 | 2 pins, 2 device wordings; both add interpretation ("suggesting substantial water flow") | 21.2 s (0.3 s, 3.8 s) |

No number outside the slots reached a pin. The line still gives a count of refusals, not reasons.

### What if

Reach 230428063, east of Siegburg (1,379 km², validation KGE 0.65, fair, 17 of 24 parameter sets kept).

| Line | Read as | Wait before device call | Create, model | Total |
|---|---|---|---|---|
| what if the town pumps 300 litres per second out of the river | device: 0.3 m³/s taken; mean flow −1 % | 62 ms | 0.6 s, 0.8 s | 96.7 s (95 s cold GR4J calibration) |
| summers get much hotter, maybe three degrees | device: 3 °C warmer, applied to the whole year without saying so; mean flow −6 % | 19 ms | 20.6 s, 1.2 s | 27.1 s |
| rain falls by a fifth and evaporation goes up 10 percent | rules: 20 % less rain, 10 % more evaporation; mean flow −41 % | no device call | | 5.4 s |
| Rhine at Cologne, reach 230515098 | panel says the 100,000 km² limit, Run disabled | | | |
| New: a much wetter future | device said rain +10 %; refused and said, with the model's name | 3 ms | 21.1 s, 1.1 s | 22.4 s |
| New: what if the factory takes twenty litres a second | device: 0.02 m³/s (right), but the status line and the map log say "0 m³/s taken" | 6 ms | 0.4 s, 0.8 s | 2.5 s |

### Review my model results

| Request | Result | Time (create, model) |
|---|---|---|
| review my model results (MODFLOW Freyberg on screen in Texas, HEC-RAS mesh loaded in Virginia) | "freyberg.hds (MODFLOW heads) is the model on screen: the review reads HEC-RAS 2D and UGRID meshes ... Flood_model.p01_BC12_S6_R6.hdf is loaded but not in view" | 0.4 s, no model call |
| is anything wrong with my flood model? (mesh on screen) | 8 pins, 7 device wordings, 1 refused ("Two cells exhibit a wet/dry cycle 5 times"). The 5 deepest-water pins say "investigate if it's an expected feature"; the rules say worth a look if it is not | 32.8 s (20.8 s, 9.2 s) |
| New: check my model (mesh on screen) | 8 pins, 8 device wordings, 0 refused; "check if expected (e.g. channel or pit)"; Nano put the 8.97 m pin before the 17.1 m one under "most severe first" | 29.8 s (21.1 s, 8.5 s) |

### Session start time

Across the 21 calls the page made for Ask the map, Explain, Scout, What-if and the review, `LanguageModel.create` took 20.6 to 21.1 s on 7 and 0.28 to 1.2 s on 14. This is now where the "20 s wait" sits. In Ask, the first prompt on a clone stalled the same way in 3 of 12 runs.

### Console

No errors from the AI roles. Two MapLibre exceptions, "Cannot read properties of undefined (reading 'key')", at one moment during the session.

### Remaining defects

Blocking for 1.0 (a claim-lock escape, or a wrong number shown as fact):

1. Explain: a share of land written as a flow ("mapped at 21% above normal", "flow at least 81% below normal", "flow above normal at 6%") passes. `aquascope/map_explain.py` `class_claims` / `check_model_notes`: refuse a "N% above/below normal" phrase when the cited fact is a share of the view, or fill that fact as "N% of the mapped land in view" and refuse literal values.
2. Ask: right number, wrong label passes ("the Potomac's q95 of 3.43 m3/s"; "a flood of magnitude 2 m3/s ... 2 years"). `aquascope/ai_engine/device_ask.py` `lock_answer`: tie a Q label to that gauge's flow-duration value, and a return period to its own return level.
3. What-if: "0 m³/s taken" for a 0.02 m³/s run, in the status line and the map log. `aquascope/whatif_map.py` `_g` (used by `scenario_label`) and `explorer/src/whatif-core.js` `g`: keep significant figures for small values.

Not blocking:

4. Ask: "kErrorUnknown" on step 2 in 7 of 12 runs, so summary and trend questions do not answer. `explorer/src/ask-local-core.js` `askLoop`: run each step in a fresh clone (the step prompt already restates the question; carry all results so far), or retry a failed step once in a fresh clone.
5. Session start of about 21 s in a third of calls. `explorer/src/local-model.js` `generateJsonLocally` creates a session per call. A warm base session per role might help, but Ask's clones also stalled, so this is not shown to work.
6. Review: the meaning flip persists ("investigate if it's an expected feature"). The wording check that `explorer/src/mesh-review.js` `wordWithModel` calls looks at numbers only; the package function was not opened in this recheck.
7. Scout: a reason that reverses the finding passes ("questions about the gauge's measurement"). `aquascope/map_scout.py` `check_wording` checks numbers and slots only.
8. Scout: the line says "worded on this device" when no device wording was used, and gives counts, not reasons. `explorer/src/scout.js` `runScout`.
9. Context: the selected gauge is named in the context line when it is off screen, and Nano answers about it (Scout in Europe, Ask the map). The page's `currentContext()`.
10. Ask the map: actions on a place the words do not name (the Nile, from the prompt's example); review and Explain requests in other words do not reach those controls. `aquascope/map_commands.py` reply check, `explorer/src/map-command.js` `byDevice`: refuse a place that is not in the words, as #639 does for boxes.
11. Explain: prompt text copied ("is pinned, unnamed"), "on on Sat 7 Nov", and "²" listed as a number in refusal reasons. `aquascope/map_explain.py` `notes_prompt`, `drop_repeats`, `claim_lock`.
12. What-if: "summers ... three degrees" runs as a year-round warming without saying so. `explorer/src/whatif-map.js` `readLine` (`heldWords`).
13. Ask: "five gauges nearest to Kingston" analysed one gauge; the 100-year answer gives the LP3 value without naming it while the page's facts use GEV. `aquascope/ai_engine/device_ask.py` prompt and facts.

### Round 2 fixes

Two more pull requests take the remaining defects: **C** is #642 on `fix/ondevice-ask-labels-590`
(the Ask panel, Ask the map, the context line, the device sessions), **D** is #641 on
`fix/ondevice-roles-round2-590` (Explain, Scout, What if, the review). Each was tested with the
recorded replies above as fixtures, in pytest and node, and C also in headless Chrome with a stubbed
`LanguageModel`; neither was rerun on the device.

| # | Defect | Fixed in |
|---|---|---|
| 1 | Explain: a share of land written as a flow | D |
| 2 | Ask: right number, wrong label ("q95 of 3.43", "2 m3/s ... 2 years") | C: `device_ask.lock_answer` ties each labelled number to its own value for the gauge its clause is about: a Q label to that gauge's flow-duration point, an N-year flood to its return level for N, a mean, minimum, maximum or record dates to their own fields. A number written in m3/s must be a flow, not a return period. The recorded comparison is refused on its Q95 alone; with the Potomac's Q95 it passes. The page adds the catalogue's name and river to an analysis, so the lock can tell two gauges apart |
| 3 | What if: "0 m³/s taken" for 0.02 m³/s | D |
| 4 | Ask: "kErrorUnknown" at step 2 | C: `ask-local-core.js` `askLoop` runs each step as the only prompt of a fresh clone, carrying the question and every tool result so far, and tries a failed step once more in another. A step that fails twice after tools ran shows what they returned, with the reason. A stub that fails a session's second prompt, as Chrome 155 did, now reaches the answer |
| 5 | About 21 s session starts | C, not shown to help: `generateJsonLocally` clones a warm base session kept per system prompt (`ask-local-core.js` `sessionPool`, at most four), and Ask the map sends its context with the words, so its system prompt and warm base stay the same. Tested with a stub for correctness only (one create, then clones); whether a clone avoids the 21 s needs the device, and Ask's clones also stalled |
| 6 | Review: the meaning flip | D |
| 7 | Scout: a reason that reverses the finding | D |
| 8 | Scout: "worded on this device" with no device wording | D |
| 9 | Context: the selected gauge named when off screen | C: `currentContext` leaves an off-screen gauge out for the map's roles and says it is "selected earlier and now off screen" for Ask, where "this gauge" still means it |
| 10 | Ask the map: a place the words do not name; review and Explain in other words | C: `map_commands.ground_places` refuses a place the words do not name ("Nile, a place the words do not name"); the prompt's examples say `<river>`, `<place>` and `<region>` instead of the Nile, the Rhine and South Asia; "realistic", "plausible to you", "are the depths right" reach the review and "what this map is showing me" the notes on the view, by the rules, with no model |
| 11 | Explain: prompt text copied, "on on", "²" as a number | D |
| 12 | What if: "summers" run as a whole year | D |
| 13 | Ask: five gauges asked, one analysed; the 100-year flood without its method | C: a question that asks for several gauges is refused when the answer names fewer than the search found, and the page shows the search's list; the device prompt asks for the list. The compact result says which return level is GEV (the page's figures) and which LP3, and the lock refuses an LP3 value given without saying LP3 |

## Third check after #641 and #642

Date: 2026-10-26.
Browser: the maintainer's Chrome 155 on a Mac, window 865 x 447 CSS px, a new tab, hard-reloaded.
Model: Gemini Nano through the Prompt API (`LanguageModel.availability()` = "available"). No key read, entered or stored.
Explorer: the live build at rekin226-aquascope-explorer.static.hf.space, asset version `15e7d48`.

How it was run. As in the recheck: each request through the page's own entry points (the Ask panel's box and button, `__aq.mapAsk.run`, `__aq.mapExplain.run`, `__aq.scout.run`, `__aq.review.run`, `__aq.whatIf.read`), with `LanguageModel.create`, `clone` and `prompt` wrapped to log prompts, raw replies and times. Map moves were done from the console. The HEC-RAS Portsmouth sample was opened from the sample chooser for the review requests. Gauge USGS 01646500 stayed selected all session, so it was off screen for every map role. One Explain run is left out of the counts: Floods past was still on from an Ask the map action. The raw replies are in `ondevice-raw-replies-3.json` (48 requests, 66 prompts).

### Recheck defects 1 to 13

| # | Defect | Now |
|---|---|---|
| 1 | Explain: a share of land written as a flow passes | Gone as recorded: no "N% above/below normal" written as a flow passed in 24 device notes, and the check refused one. A new form passes (escape 3 below) |
| 2 | Ask: right number, wrong label passes | The recorded q95 case was not reproducible: the comparison never reached an answer. An LP3 value given as GEV was refused. The label check now refuses right answers (see refusals) |
| 3 | What-if: "0 m³/s taken" for 0.02 m³/s | Gone. Status line and map log both say 0.02 m³/s |
| 4 | Ask: kErrorUnknown | Gone: none in 14 questions (33 prompts). A new failure takes its place: 5 of 14 questions used all three steps on tool calls and gave no answer |
| 5 | Session starts of about 21 s | Not seen. 0 of 20 `create` calls took 20 s or more; see Session start time |
| 6 | Review: "investigate if it's an expected feature" passes | Gone. In 40 device wordings the 5 "Check if this depth is expected" ones were refused; none passed |
| 7 | Scout: a reason that reverses the finding passes | Gone. The models-disagree reason was refused in 2 of 2 Europe runs (once wrongly, see refusals) |
| 8 | Scout: line claims device wording it did not use | Gone: "2 of 3 worded on this device", "asked, its wording not used", first reason given. After a 40 s timeout the line said "Rules, no model" without saying the model was asked |
| 9 | Context names an off-screen gauge | Gone. Ask the map and Scout prompts left out the selected gauge (it was in Virginia, the maps elsewhere) |
| 10 | Ask the map: a place the words do not name; review and Explain phrases | Mostly gone. No Nile; a coordinate the words did not name was refused. "does this simulation look plausible", "are the depths ... realistic", "is anything wrong", "check my model" reached the review and "tell me in plain words ..." the notes. "are there any numerical instabilities in this run?" did not; it turned on Floods past and river status unasked |
| 11 | Explain: prompt copies, "on on", "²" in reasons | Gone as recorded. New copies: an unclosed placeholder on a pin ("Peak against the 2-year flow is 2.5 times when {f15."), "on Sun 25 Oct when on Sun 25 Oct", "2.8 times when 2.8 times" |
| 12 | What-if: "summers" applied year-round in silence | Gone. "“summers” not applied: the change runs over the whole year"; "winters" the same |
| 13 | Ask: list of gauges, flood method | Method: gone. List: not reproducible. For "five gauges nearest to Kingston upon Thames" the model called find_stations twice and then describe_catchment, and the page showed no gauges |

### New claim-lock escapes

Shown as fact, all from passing device replies:

1. Ask: "significant trend in annual maxima with a p = 0.0689" (25.05, 121.55). The result says `significant: false`, "no significant trend in the annual maxima". The same sentence also says "no significant trend" for the annual mean, and the trend check judges a sentence by one phrase.
2. What-if: "a brewery upstream draws 7.5 l/s ..." was read and run as 7.5 m³/s taken: mean flow −33 %, low flow (Q95) −100 %. The lock accepts the number either as written or divided by 1000.
3. Explain: the largest single area's share written as the share of all land. South America: "14% of the mapped land is much below normal" and "17% of the mapped land is below normal"; a second run of the view had 58 % of the land below or much below normal and wrote both "58% ... below normal" and "15% of the mapped land is below normal". The prompt tells the model to write these facts as "{fN} of the mapped land".
4. Explain: right numbers, wrong claim. "A smaller area of 23%" (the other area was 11 %); "Another significant area (340,000 km² of the mapped land)" (340,000 km² is the size of the same area); "980,000 km² of the mapped land is below normal".
5. Scout: "Severe Flooding Forecast in Northern Territory ... with a substantial chance of exceeding it" for a 2-year flow that 13 of 51 ensemble members reach.

Also seen, not counted: five review pins each saying "The deepest water is X m", where the rules say the deepest "in this part of the mesh".

### Refusals per role

| Role | Device replies checked | Refused | Right refusals | Correct replies refused |
|---|---|---|---|---|
| Ask panel | 9 answers | 6 | 1 (LP3 value without "LP3") | 5: "100-year" read as a gauge id (2); Sen's slope or a p-value read as the daily mean (3, one loosely worded) |
| Explain | 24 notes | 8 | 7 (5 with no place, 2 with a number not in their facts) | 1: "({f10}) ... covering 3% of the mapped land" refused as a share without "of the mapped land" |
| Scout | 19 wordings | 8 | 1 (`{year}`, no such slot) | 6, and 1 borderline: a fair models-disagree paraphrase; "10th percentile" copied from the rules' own text (3); "Oct 26th", "Oct 25th" (2) |
| Review | 40 wordings | 6 | 5 (the meaning flip) | 1: "Two cells" (the rules say 2 cells flicker) |
| What-if | 8 values | 2 | 1 ("a much wetter future", +10 %) | 1: "one and a half degrees" |
| Ask the map | 4 action replies | 2 actions | 2 (a made-up box, a coordinate) | 0 |

Rates of correct replies wrongly refused: Ask 5 of 9, Scout 6 of 19, What-if 1 of 8, Explain 1 of 24, review 1 of 40, Ask the map 0 of 4.

### Ask answer rate

14 questions, gauge USGS 01646500 selected, context on.

| Result | Count | Questions |
|---|---|---|
| Answered | 3 | Q95 and median (right); a two-sentence report summary (right); climate and modelled discharge at 25.05, 121.55 (escape 1) |
| Refused | 6 | summary x2, 100-year flood, annual-mean trend x2, "have the annual floods become bigger" |
| No answer (steps used) | 5 | summary, 100-year flood, 50-year flood, five gauges near Kingston, Potomac vs Thames |
| Error | 0 | |

Answer rate 3 of 14. Of the summary and trend questions (summary x3, report summary, annual-mean trend x2, floods bigger), 1 of 7 answered. In all 5 no-answer runs the model repeated a tool call with the same arguments (`analyze_station` three times in 3 of them); for the comparison it called `analyze_station` with `uk_ea` / `abc`, the example in the system prompt. When the steps run out the page shows the stock line only, not what the tools returned. Fact lines show "1.125e+04 m3/s".

### Session start time

Across 152 Prompt API calls (20 `create`, 66 `clone`, 66 `prompt`), 1 took 20 s or more: a Scout prompt in Europe that ran to the 40 s timeout. `create` took 0.32 to 1.2 s 19 times and 13.5 s once (Ask's first base). Clones took 0 to 8 ms. Of 33 map-role prompts, 19 began with a new base (Explain and Scout put the view's facts in the system prompt, so the pool key changes with the view) and 14 cloned a warm one. The 21 s starts of the recheck did not occur in this session; this run cannot say whether the pool is the reason.

### Remaining defects

Blocking for 1.0 (a claim-lock escape, or a wrong number shown as fact):

1. Ask: a significance claim per clause. `aquascope/ai_engine/device_ask.py` `_trend_check` reads one significance phrase per sentence; split at clause breaks (`_CLAUSE_BREAK`) and check each claim against its own test.
2. What-if: "7.5 l/s" runs as 7.5 m³/s. `aquascope/whatif_map.py` `_numbers_in` (used by `parse_scenario_reply`): when the words say l/s, a `demand_cms` must match the value divided by 1000, not the value as written.
3. Explain: a largest-area share written as the share of all land. `aquascope/map_explain.py` `notes_prompt` (the "write {fN} of the mapped land" hint on largest-area facts) and `share_claims`: give largest-area facts their own wording ("the largest area, {fN} of the view, ...") and refuse them as a whole-view share.
4. Explain: right numbers, wrong comparison or role ("A smaller area of 23%", "Another ... area" for the same area's size, "{size} of the mapped land"). `aquascope/map_explain.py` `check_model_notes` / `share_claims`: refuse comparison words between facts and "of the mapped land" after a size fact.
5. Scout: severity added to a forecast ("Severe Flooding", "substantial chance" for 13 of 51 members at the 2-year flow). `aquascope/map_scout.py` `CLAIMS["floods_ahead"]` has no `not` list and `check_wording` does not count severity words; refuse severity and likelihood words the rules do not use.

Not blocking:

6. Ask: tool-call loops and no answer in 5 of 14. `explorer/src/ask-local-core.js` `askLoop`: stop on a repeated identical call and ask for the answer from the results; on the last step allow only an answer. `explorer/src/local-model.js` `LOCAL_TOOLS`: the example id "abc" gets copied.
7. Ask: steps used shows no tool results. `askLoop` returns `lock: null` with `STEPS_USED`; run the lock with an empty answer as `stoppedResult` does, so `deviceAnswerHtml` lists the facts.
8. Ask: correct answers refused. `device_ask.py` `lock_answer` (`_ALNUM_ID` takes "100-year" for a gauge id) and `_label_check` (a number in "m3/s per year" counts as a flow, and "p-value for the annual mean is 0.953" ties 0.953 to the mean).
9. Ask: "1.125e+04 m3/s" in the fact lines. `device_ask.py` `_fmt`.
10. Explain: an unclosed placeholder reaches a pin ("when {f15."), and "when" repeats a value. `map_explain.py` `_fill` / `check_model_notes` (refuse any brace left after filling) and `drop_repeats`.
11. Scout: fair wordings refused ("Models have difficulty accurately predicting flow ..."; "10th percentile" copied from the rules; "Oct 26th"). `map_scout.py` `CLAIMS["models_disagree"]` and `check_wording`.
12. Scout: after a timeout the line says "Rules, no model". `explorer/src/scout-core.js` `wordingLine` (the `failed` case).
13. Review: "Two cells" refused although the rules say 2 cells. `aquascope/mesh_review.py` `_locked`.
14. What-if: "one and a half degrees" refused. `whatif_map.py` `_numbers_in` / `_FRACTIONS`.
15. Ask the map: "are there any numerical instabilities in this run?" does not reach the review and adds layers. `aquascope/map_commands.py` `_REVIEW`. "which rivers in Spain are running low this month?" reads the place as "Spain are running low" and stops without the device. `map_commands.py` `_place_phrase`.

### Round 3: the model picks, the code writes

Three checks on the device, and the escapes did not converge. The first run found the Ask panel with no lock and
the map roles' locks holding. Round 1 added a lock to Ask; the recheck found right numbers under wrong labels,
shares written as flows and meanings turned round. Round 2 tightened the patterns; the third check found new forms
(a significance claim per clause, litres per second read as m³/s, the largest area's share written as the whole
view's, severity added to a forecast) and the locks themselves refusing right answers (Ask: 5 of 9). Each lock is a
pattern over free text. A small model has more ways to write a sentence than a pattern can list, and the stricter
the pattern the more right answers it refuses.

So on 2026-10-26 the maintainer changed the design for every on-device role, and for the reader's own key where
the code path is shared. The model no longer writes a number, a date, an id, a unit, a place name or a qualitative
claim that reaches the page. Its reply is a JSON object held by a `responseConstraint` schema whose values are
enums built from the facts at hand: tool names, gauge ids, places, layers, fact keys and sentence ids. The code
writes every sentence the reader sees, with wording that lives in the package. Anything outside the enums is
refused before display by a schema check in the package, the refusal is said in one short line, and the rules'
wording is shown. The pattern locks stay as a second line of defence. The keyless rules path is unchanged and
stays the default when no model is ready.

Two pull requests: **F**, #644 on `fix/ask-pick-not-write-590` (the Ask panel and Ask the map), and **E**, #645 on
`fix/map-roles-pick-not-write-590` (Explain, Scout, the review and What if).

**The Ask panel** (`aquascope.ai_engine.device_ask`, `explorer/src/ask-local-core.js`, `local-model.js`, `ask.js`).
Each step's reply is `{"tool": ..., "arguments": {...}}` or `{"answer": [fact keys]}`; the package builds the
step's schema and message (`step_schema`), checks a call (`check_call`) and writes the answer (`render_answer`).

| The model decides | The model cannot |
|---|---|
| which tool to run next, from those still open | run a tool twice with the same arguments: a repeat is not run, and the next step may only answer |
| which gauge, from the ids in the reader's words, the context line or an earlier result | name any other id ("abc", the old example, is gone from the prompt and refused) |
| which point, from a pair the words or a result give | search near a point it chose itself |
| which search phrase, from runs of the question's words | search for words the reader did not write |
| which facts answer the question, and their order, by key: `USGS-01646500.trend.mean`, `USGS-01646500.fdc.q95`, `USGS-01646500.gev.100`, `USGS-01646500.record`, `search.count`, ... | write a sentence: each one is the package's, naming the gauge, the label, the value with its unit and the method (GEV or LP3, Mann-Kendall), with numbers in the Explorer's format ("11,247 m3/s", never "1.125e+04") |
| a few connecting sentences by id (`say.floods`, `say.trend`, ...), which say how the facts were made | state a trend: the trend sentence is the package's reading of the test ("no significant trend in the annual maxima (Mann-Kendall p = 0.0689; ...)") |

The first step may only call a tool (there are no facts yet), the last may only answer, and a question about one
gauge or place gets three replies (a comparison or a list five). When the steps run out, or the model stops, the
page shows the facts gathered so far under one line saying why; the verdict is never empty.

**Ask the map** (`aquascope.map_commands`, `explorer/src/map-command.js`). Its replies were already structured. Now
the device's schema for a request (`reply_schema`) makes the place a model may name an enum of runs of the
request's words, and the layers it may switch an enum of those the words ask for, and leaves out the free-text
fields. With the words, `parse_model_reply` refuses a layer switch the words do not ask for and drops a model's own
titles and notes; the package titles a pin with its place. The same check runs on a reply from the reader's key.

**The check3 items in F (#644)**

| Check3 item | Now |
|---|---|
| Blocking 1: "significant trend in annual maxima with a p = 0.0689" | Fixed by construction: no key writes a significance claim the test does not make; the replayed intent renders "no significant trend in the annual maxima (Mann-Kendall p = 0.0689; ...)" |
| 6: tool-call loops, no answer in 5 of 14; the example id "abc" copied | Fixed: a repeated call is not run and the next step may only answer; the last step may only answer; ids come from an enum of the words and results, and "abc" is gone from the prompt |
| 7: steps used, no tool results shown | Fixed: the page shows the facts gathered with the reason |
| 8: correct answers refused ("100-year" as an id, a p-value or Sen's slope as the mean) | Fixed by construction (no free text to read), and in the second-line lock: the 5 recorded right answers it refused now pass it |
| 9: "1.125e+04 m3/s" | Fixed: `device_ask.display`, whole numbers with commas from 100, three significant figures below, never an exponent |
| 13 (recheck): five gauges asked, one analysed | Fixed: the search's gauges are facts by key (`E02102A.where`, ...) the model picks |
| 15: "numerical instabilities" not reaching the review, layers turned on unasked; "rivers in Spain are running low" read as a place | Fixed: the instabilities question and close phrasings reach the review by the rules; a layer the words do not ask for is refused; "which rivers in Spain are running low this month?" is the river status, below normal, in Spain |

**Replaying the recorded replies.** The 14 Ask questions of the third check
(`tests/fixtures/ondevice_ask_check3.json`) through the new path, with the tools as fixtures holding the recorded
figures (`tests/test_ai_engine/test_device_ask_pick.py`): the replies are in the old shape, so all 9 free-text
answers are refused by the schema check, and the 5 loops end at the forced answer step. 0 of 14 put model text on
the page; 14 of 14 show the package's facts under a one-line reason (the old page showed a stock line with nothing
under it for 5). The calls hold: 14 of 14 first calls pass `check_call`, the 4 repeats are not run, and the 2
"uk_ea / abc" calls are refused. Replayed as the new schema allows, the intent of each recorded question (11
questions: summaries, the 100-year and 50-year floods, both trends, the climate at 25.05, 121.55, Q95 and median,
the report summary, the five Kingston gauges, the Potomac and Thames comparison) gives a correct answer in 11 of
11, each sentence the package's own.

**What needs the device.** Whether Chrome's Prompt API accepts these schemas (string enums of up to about 60
values, an integer range, nested objects) and whether Gemini Nano picks sensible keys: the answer rate on the
device is not known until it is run. The step messages now list the facts with their sentences, about 3,000 to
4,000 characters for one gauge, so a step may take longer than before.

## Fourth check after #644 and #645

Date: 2026-10-26.
Browser: the maintainer's Chrome 155 on a Mac, window 865 x 447 CSS px, a new tab, hard-reloaded.
Model: Gemini Nano through the Prompt API (`LanguageModel.availability()` = "available"). No key read, entered or stored.
Explorer: the live build at rekin226-aquascope-explorer.static.hf.space, asset version `337cacf` (main after #644 and #645).

How it was run. As in the third check: each request through the page's own entry points (the Ask panel's box and button with the on-device tier, `__aq.mapAsk.run`, `__aq.mapExplain.run`, `__aq.scout.run`, `__aq.review.run`, `__aq.whatIf.read`), with `LanguageModel.create`, `clone` and `prompt` wrapped to log each prompt's `responseConstraint`, the raw reply and its time. Map moves were done from the console. Gauge USGS 01646500 stayed selected all session. The What-if lines ran on GEOGLOWS reach 110888926 (Somali Region, Ethiopia). The review ran on the HEC-RAS Portsmouth and MODFLOW Freyberg samples from the sample chooser. The raw replies are in `ondevice-raw-replies-4.json` (58 requests, 77 prompts).

The design held. Every reply the device gave was ids, keys, variants, tool names or number tokens, and every sentence on the page was the package's. No wrong number and no reversed claim reached the page.

### Schema acceptance

Chrome accepted every schema. No error on `create`, no schema-too-large error, no refusal of a `responseConstraint`.

| Role | Prompts | Largest schema | Largest enum | Errors |
|---|---|---|---|---|
| Ask panel | 36 | 2,032 chars | 55 values | 1: `QuotaExceededError` ("the response exceeded output limits") after 176 s |
| Ask the map | 6 | 1,405 chars | 8 | 0 |
| Explain | 10 | 367 chars | 10 | 0 |
| Scout | 9 | 477 chars | 5 | 0 |
| What if | 8 | 677 chars | 11 | 0 |
| Review | 8 | 350 chars | 8 | 0 |

Accepted is not the same as enforced. The Ask step schema leaves `arguments` open (no `additionalProperties: false`) and gives `lat`, `lon` and `near` as free numbers. Nano used both: it added keys (`limit__`, `limittypicode`), made up search points ([51.0, -0.3], [51.47, 0.50], ...), and once ran a number until the output limit (the 176 s error). The map roles' schemas are closed, and their replies stayed inside them.

### Escapes

0 in all roles. Checked: every Ask sentence against the tool results (11,247 m3/s GEV 100-year, 9,305 m3/s 50-year, p = 0.953 and 0.96 with "no significant trend"); every Explain and Scout pin against the facts listed under it; every What-if reading against the words; every review pin against the rules' wording.

The third check's blocking items did not recur: no trend claim but the package's; "draws 7.5 l/s" ran as 0.0075 m³/s; the largest area reads "The largest area much below normal in view covers 33% of the mapped land in view (12,000,000 km²)", with the whole-view share as its own sentence; no comparison between areas; no severity or likelihood word in any Scout pin ("31 of 51 ensemble members reach it").

### Ask answer rate

14 questions, gauge USGS 01646500 selected, context on. 10 from the third check, two repeats, two new.

| Result | Count | Questions |
|---|---|---|
| Answered, on point | 6 | 100-year flood x2 (GEV, 11,247 m3/s; the second run added the GloFAS cell's 100-year, 87.1 m3/s, labelled modelled), 50-year flood, annual-mean trend, Q95 and median, climate and modelled discharge at 25.05, 121.55 |
| Answered, off the question | 5 | summary (record, mean, both trends; no flood frequency), "have the annual floods become bigger" (ten return levels, not the annual-maxima trend), two-sentence report summary (ten facts), GEV or LP3 100-year (the 2-year values picked), Potomac vs Thames (the Potomac only; the search found the Kingston gauge but the model never analysed it) |
| Refused by the second-line lock | 2 | summary (right picks, refused for the package's own `say.trend` sentence); list of Thames gauges near Kingston (`search.count` only, refused as "names 1 of 2 gauges"; the catalogue search returned ten US rivers "near Kingston") |
| No answer | 1 | five gauges nearest Kingston upon Thames: four calls, each with a made-up search point, all refused; the page said "nothing in the question to look up" |

Answer rate 11 of 14 (third check: 3 of 14), 6 of 14 on point. By kind: summary 1 of 2 answered (neither complete), trend 2 of 2 answered (1 with the wrong facts), comparison 1 answered with one side, list 0 of 2. Repeated calls were not run (3 times), and every refusal listed the facts gathered.

### Pick quality

- **Ask:** the first call was right in 13 of 14 questions. Single-fact questions get the right key. Questions that need the right two or three keys go wrong: the model picks neighbouring keys (2-year for 100-year, return levels for a trend) or fills the ten allowed picks.
- **Ask the map:** places came only from the words; two made-up points were refused and "rivers in Spain running low" went to the rules (4 actions). The picks are thin: "driest in Africa" moved the map but turned on no status layer; "the Danube compared with normal" kept only the normal class (as in the third check); "floods ahead along the Mekong" turned the layer on without moving there; "my MODFLOW model" was tried as a place.
- **Explain:** 10 of 10 picks valid. In 5 of 10 the model picked the first three or four notes in order, all variant a; Africa skipped the status notes for the forecasts. Sensible, close to the rules' order.
- **Scout:** 9 of 9 valid and reasonable orders (status and forecasts ahead of past floods). In South America it picked the one finding three times with three variants; it was shown once.
- **What if:** 8 of 8 lines read right by the model: "twenty" l/s, "7.5" l/s, "one and a half" degC, "a fifth" as a fraction, seasons noted and not applied, "a much wetter future" left with no number. One right reading was refused by the package ("half a cubic metre per second", below).
- **Review:** all 8 HEC-RAS phrasings reached the review, "are there any numerical instabilities in this run?" included. The model always chose variant a and in 4 of 8 left out the two wet-dry flicker findings, including for "numerical instabilities" and "is the model stable?", where they are the answer. The line says so ("2 findings not pinned"). On the Freyberg sample the review said, by the rules, that it does not read MODFLOW heads.

### Timings

183 Prompt API calls (29 `create`, 77 `clone`, 77 `prompt`). 1 took 20 s or more: the Ask prompt that ran 176 s to `QuotaExceededError`. `create` took 0.31 to 1.14 s 28 times and 13.3 s once (Ask's first base). The other 76 prompts took 0.59 to 5.3 s: Ask 0.59 to 5.3 s, Ask the map 0.85 to 1.7 s, Explain 0.92 to 1.5 s, Scout 0.70 to 2.4 s, What-if 0.80 to 1.4 s, review 1.5 to 2.1 s. A Scout run takes 17 to 27 s on the page, almost all of it the scan; the first What-if line took 120 s, the GR4J calibration.

During the 176 s prompt the Ask panel showed "Thinking… (step 1 of 5)" and "Working…" with no Stop button.

### Usefulness

- **Ask panel:** worth having: with no key it now answers most single-fact questions in about 10 s in the package's words, though summaries, comparisons and lists still need a key.
- **Ask the map:** marginal: the rules already handle the clear requests, and the model's extra moves are thin.
- **Explain:** marginal: the picks are valid but mostly the rules' own first three notes.
- **Scout:** marginal: a sensible order, but the rules' order is as good and the scan, not the model, sets the wait.
- **What if:** worth having: it reads spelled-out numbers, units and fractions the rules leave unread, in about a second, and the code holds the values.
- **Review:** not worth it yet: the rules' pins are the same words, and the model drops the findings a stability question is about.

### Remaining defects

Blocking for 1.0 (an escape, or a role that cannot run because the schema is refused): none.

Not blocking:

1. Ask: open step schema. `aquascope/ai_engine/device_ask.py` `step_schema`: `lat`, `lon` and `near` are free numbers and `arguments` takes any key, so Nano made up search points (Kingston failed 4 of 4 calls), added keys, and once ran a number for 176 s to `QuotaExceededError`. Make the points an enum of the known pairs (or leave them out when there are none) and set `additionalProperties: false`.
2. Ask: no time limit and no Stop on the device path. `explorer/src/local-model.js` (the Prompt API engine's `open().generate`, which passes no signal, unlike `generateJsonLocally`'s 25 s) and `explorer/src/ask.js` `runLocally` (`ask-stop` stays hidden).
3. Ask: the second-line lock refuses the package's own connective. `device_ask.py` `_trend_check` (via `lock_answer`) reads `CONNECTIVES["say.trend"]` ("a p-value below 0.05 is called significant") as a significance claim and refuses a right answer.
4. Ask: wrong reason after refused calls. `explorer/src/ask-local-core.js` `askLoop` says "nothing in the question to look up" when the question named a place and the calls were refused; say which call was refused and why.
5. Ask: weak fact picks. `device_ask.py` `_step_prompt` / `step_schema`: tie keys to the words ("100-year" to `.100` keys, "become bigger" to `.trend.max`), allow fewer picks for "two sentences", and for a comparison do not offer the answer step until each side has a fact. `_known` also offers search phrases such as "List", "USGS-" and "Potomac USGS-".
6. Ask: the header "Answered on your device ... AquaScope wrote every sentence" sits above a refusal. `explorer/src/ask.js` `deviceAnswerHtml`.
7. Ask the map: thin picks. `aquascope/map_commands.py` `reply_schema` / `parse_model_reply`: a tool name ("MODFLOW") is offered as a place; `set_time` passes when the words do not ask for a date; "compared with normal" gives `focus_status` ["normal"].
8. Review: the flicker findings left out for stability questions. `aquascope/mesh_review.py` `review_schema` / `review_variants`: when the words ask about stability or instabilities, require the wet-dry findings.
9. What if: "half a cubic metre per second" refused ("the words say fraction, not m3/s"). `aquascope/whatif_map.py` `number_tokens` / `_token_value`: "half a" before a unit is 0.5 in that unit, not a share.
10. What if: after "Not read" the panel keeps the previous line's slider values (+3 °C) on screen. `explorer/src/whatif-map.js` `readLine` (the not-read branch).
11. Scout: the forecast pin lists "2-year flow: 6.43" twice, and "at or above the 10-year flow of 4,220 m³/s; the 2-year flow is 3,090 m³/s. All 51 ensemble members reach it." can read as the 10-year flow (the count is for the 2-year). `aquascope/map_scout.py` floods-ahead facts and wording.
12. Catalogue search: "Thames near Kingston" returns ten US gauges "near Kingston" (Etowah GA, Grand WI, Silver Creek KY, Sevier UT, ...). The `find_stations` ranking, not the device.

Also seen, not counted: Explain on a view centred on England said "covers 100% of the mapped land in view (170,000 km²)", with the largest area in Lower Saxony, next to a gauge note "18 gauges in view are much above normal". The figures match their facts; they come from different sources and dates (September modelled status, 23 October gauges), and the page does not say so side by side.

Not blocking items: #646. Fixed here: items 1 to 6 of the Ask list.
