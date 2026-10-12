# Follow-up on the September Explorer launch

Prepared 23 September 2026. These are drafts and a request audit; nothing has been
posted or sent by this implementation task.

The [existing Hydrology thread](https://www.reddit.com/r/Hydrology/comments/1w4fo6d/i_got_tired_of_writing_the_same_agency_download/)
contains concrete requests. The visible discussion records CFS and England-label
follow-ups, demand for Canada and Greek/Polish coverage, missing western Colorado
observations, and uncertainty about appropriate uses. Poland feedback now points
to [issue 456](https://github.com/Rekin226/aquascope/issues/456), which remains open.

| Request | Evidence/status | Next acceptance gate |
| --- | --- | --- |
| CFS display | Existing toggle and maintainer follow-up | Inspect conversion and exported units in usability sessions |
| England agency boundary | Label already corrected | Keep SEPA/DFI/NRW coverage claims separate |
| Poland daily dates | Collector exists; issue 456 requests independent comparison | Real gauge/day comparison, especially Nov/Dec |
| Greece | Hydroscope/OpenHi paths exist; availability and licences vary | Verify mirrored availability; unresolved licences stay unresolved |
| Canada/PCIC | Requested in launch comments | Check current issue ownership before promising an implementation |
| Western Colorado empty pins | User reported unavailable observations | Reproduce a named gauge; catalog size is not observation coverage |
| Appropriate uses and accuracy | User could explore but could not identify a use | Show an explicit screening task, its input limits and an export |

## Draft practitioner follow-up

AquaScope Hydrology is being updated around a shorter workflow: find a usable
record, check its actual coverage, then export it or save a reproducible study.
The changes clarify which estimator an uncertainty interval belongs to and keep
failed checks visible. They also distinguish a catalog pin from available data.

After the release and live checks pass, link to one relevant regional workflow,
its downloadable result and limitations. Invite a specific comparison against a
gauge the reader knows. Do not claim independent validation until that review is
complete, or present an export as evidence of a real-project use.

## Draft pilot invitation

Would you try AquaScope on a river-data task you already need to complete? The
session focuses on finding a record, judging its limits and using an export in
your normal tools. No account, API key or private project data are required for
the core workflow. Sharing the optional local usage log is voluntary. We would
ask about continued usefulness later and seek separate permission for any public
case study or quotation.

Use targeted hydrology/GIS or teaching channels where this task is relevant.
A follow-up should answer a prior request and offer one reproducible result;
repeat broad launch posts only after independently useful examples exist.
