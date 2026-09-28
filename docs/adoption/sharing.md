# Reopen and share a completed study

After a study finishes, choose **Save complete study**. The `.aqstudy.json` file
contains the brief, plan, input tables, results, findings and artifact bytes.
Use **Tools → Open saved study** to reopen it. Importing this file restores stored
results; it does not run the plan or refetch a gauge. Runtime packages may still
need to load from the network the first time this browser opens a study.

A checksum detects changed contents. It is not a signature, author authentication,
or independent verification of the science. Keep the original file if you need
to preserve its exact input and result identities. Complete files are limited to
50 MB. A lightweight `workspace.json` may lack artifact bytes; use the full bundle
or the original complete study if that happens.

**Copy plan link** shares instructions to run an analysis. It does not carry a
completed result. The recipient may get different observations from live agencies.
**Download bundle** contains reports, tables, figures, the notebook and workspace.

## Retained data and result tables

Station and flood-analysis steps retain their own full observation CSVs, including
subdaily timestamps. Plotting decimation does not change these inputs. Check the
step's input hash and observation count when reproducing a result.

Flood result CSVs use one row per estimator and return period. `lower`, `upper`,
`interval_method` and `confidence_level` belong to that row's `estimator` and
`estimate`. Empty bounds mean that estimator has no recorded interval; an interval
from another fit is not substituted. This replaces the older wide table with
generic GEV/LP3 columns and one ambiguous pair of bounds.

Workbook data tables over 100,000 rows continue in numbered sheets. Keep all parts
when importing them elsewhere. The CSV retains the original serialized precision;
spreadsheet applications impose their usual numeric precision limits.

## Rerun the notebook

The notebook requests current agency data through the same execution path as
Explorer. Its opening answer is labelled as the original answer; the run prints
a new answer and regenerates its findings, checks, full input tables and figures.
Uploaded tables are retained from `workspace.json`. Install the exact source
revision shown in the notebook when one is recorded.

The last cell saves the new artifacts under `rerun-<id>/`, preserving the original
bundle files. A failed step produces new limitations rather than reusing its old
tables or claims. For the original result without a new agency request, reopen
the complete study instead. Current-data reruns are distinct from reproducing
the original analysis against its retained inputs.

## Publish an explicit, versioned report

For a public result, review the complete file or self-contained HTML report for
private inputs first. Then publish an immutable version through your chosen
repository release or data repository. Include input hashes, source and agency
credits, software version and build revision, method/interval identity, check
outcomes and limitations. Name the release explicitly and retain its immutable
URL or DOI. A mutable “latest” page is not a reproducible snapshot.

AquaScope does not upload these files when you save or reopen them. Anyone to
whom you give the complete file can inspect the included inputs. Do not publish
private uploads, locations or a participant's results without their permission.
