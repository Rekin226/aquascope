# Practitioner pilot and independent review

This is a protocol ready to run, not evidence of users, endorsement or validation.
Keep identifiable participant details outside the public repository. Participation,
usage-log sharing and permission to publish a case study are separate choices.

## Recruitment and consent

Invite ten practitioners who have an actual river-data task: aim for a mix of
hydrology researchers, consulting engineers, GIS analysts and instructors, across
at least three supported regions. Avoid counting maintainers or contributors as
independent adopters. Record recruitment source and previous AquaScope experience
so a convenience sample is not presented as representative of all visitors.

Explain that records, coordinates, questions and files need not be shared. The
optional local usage log has no identifier and is never transmitted automatically.
Participants may decline it or delete it. Use an agreed participant code only in
the private pilot sheet; do not put names or project locations in public logs.

## Five independent comprehension sessions

Run five sessions before a broad launch, with no coaching during the tasks.
Start at Explorer on the participant's usual device. Record success, elapsed time,
errors and assistance separately; stop timing on an actionable failure.

1. Find a relevant gauge. Explain whether its catalog period is the same as its
   accessible observations, and identify the variable, units and actual period.
2. Open a worked analysis. Explain which estimate has an interval, what its grade
   covers, and one unresolved limitation. Distinguish daily mean maxima from
   instantaneous flood peaks.
3. Import the provided synthetic sample CSV. Identify its synthetic status,
   date/value mapping and coverage, then run a suitable analysis.
4. Download an observation CSV and inspect it in the tool they normally use.
   Count completion only after they confirm usable rows and units in that tool.
5. Save a completed study, reopen it, and explain the difference between that file
   and a plan link that reruns against current data. Identify what sharing reveals.

Proposed gate: four of five participants complete their chosen useful workflow
without intervention and correctly explain its main limitation. Report the task
breakdown and failures; a combined success count must not hide a science misunderstanding.
This is a usability gate for this small sample, not a population conversion estimate.

## Real-project follow-up

Over 28 days, seek three independent uses in actual work. An export or a promise
to use the tool does not qualify. Ask for a brief description of what the artifact
helped them do, which downstream tool accepted it, and what remained difficult.
A redacted artifact or reproducible steps can be sufficient; private project data
are not required. Obtain separate written permission before publishing a name,
quote, screenshot, project description or endorsement.

Follow up around day 7 and day 28 only through the participant's agreed channel.
Record return to another useful action and continued project use separately.
A researcher may use the same result throughout a project without daily visits.
Voluntary logs support the account; they are not proof of a downloaded file being used.

## Independent scientific review

Ask a hydrologist who did not implement the change to review each promoted example:

- Input provenance, units, observation period, missingness and year-selection rule.
- Daily maxima versus instantaneous peaks and suitability for the stated task.
- Estimator, result identity, interval procedure and uncertainty pairing.
- Trend quantity, record length, extrapolation and fit-comparison checks.
- Catchment comparability for model/gauge comparisons; skipped checks are visible.
- Reproduction from retained data and exact software revision in another environment.
- Conclusions, limitations and suitability of any design or regulatory wording.

Record reviewer, date, example/input hashes, software revision, findings, fixes and
recheck outcome. A review cannot be transferred to changed inputs or scientific
code without considering whether re-review is needed. Until signed off, label
examples “maintainer reproduction; independent domain review pending”.

## Ownership and evidence

| Responsibility | Acceptance evidence | Current owner |
| --- | --- | --- |
| Scientific review | Signed review against immutable inputs/revision | Unassigned; independent reviewer required |
| Archive operations | Published revision and source statuses for two consecutive scheduled runs | Maintainer to nominate |
| Explorer release | Browser checks, accessibility/mobile inspection and artifact round trip | Merge/release approver |
| Pilot facilitation | Consent, uncoached task results and follow-up records | Maintainer to nominate |
| Documentation/example review | Reproduction in a clean environment | Contributor reviewer |

Use existing contributor pathways for implementation and review. These roles are
proposed responsibilities, not assignments to people who have not agreed.

The public results sheet should contain only aggregate counts and redacted themes.
Maintain a private log with columns: participant code, role, region, prior experience,
consent, device, task, start, end, successful, assistance, failure category,
downstream confirmation, day-7 use, day-28 use, project-use evidence, publication permission.
