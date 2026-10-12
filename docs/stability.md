# Stability: what 1.0 promises

From 1.0, AquaScope follows [semantic versioning](https://semver.org/). Within a major version (1.x), what you
import, the commands you type, the MCP tools your assistant calls and the Archive files you read keep working.
A minor release (1.1) adds things; a patch release (1.0.1) fixes things; only a major release (2.0) removes
anything, and only after a deprecation period.

This page says what is covered, what counts as breaking, how a deprecation warns and for how long, and what
"experimental" means. Before 1.0, main follows it through a quiet period with no breaking changes (#589).

## Four public surfaces

| Surface | Where it is listed | Checked by |
| --- | --- | --- |
| Python names | [`tests/fixtures/public_api.json`](https://github.com/Rekin226/aquascope/blob/main/tests/fixtures/public_api.json), `"python"` | `tests/test_public_api.py` |
| CLI commands and flags | the same file, `"cli"` | `tests/test_public_api.py` |
| MCP tools and arguments | the same file, `"mcp"` | `tests/test_public_api.py` |
| The Archive's paths and columns | [Archive layout](archive-layout.md) | `tests/test_public_api.py` (every published prefix is documented) |

### Python

Stable are the names the README and these docs show, read conservatively:

- the top level: `aquascope.collect`, `aquascope.recommend`, `aquascope.find_stations`, `aquascope.__version__`
  and the convenience names in `aquascope.__init__` (`flood_analysis`, `HydroAgent`, `GRACEProcessor`, ...);
- `aquascope.api`, every function in it;
- every name in the `__all__` of `aquascope.hydrology`, `agri`, `viz`, `collectors`, `ai_engine`, `io`,
  `registry`, `workbench`, `methods`, `playbooks`, `studio`, `advanced`, `area_study`, and of `archive` with
  `archive.basins`, `caravan`, `regionalize`, `signatures` and `similar`;
- the names the docs show from `explore`, `compare`, `ingest`, `evaluation`, `archive.catalog`,
  `schemas.water_data`, `schemas.station`, `collectors.base`, `io.interop`, `models.rainfall_runoff`,
  `pipelines.model_builder`, `analysis.eda`, `analysis.quality`, `ai_engine.analyst`, `ai_engine.agent`,
  `ai_engine.recommender`, `ai_engine.team`, `ai_engine.verify`, `ai_engine.providers`,
  `ai_engine.llm_transport`, `utils.http_client` (for collector authors), `utils.storage`, and the Study Desk's
  `studio.desk.comment`, `resolve` and `sensitivity` with `studio.prompts.as_json`;
- `aquascope.experimental` itself: its lists and `is_experimental` (the lists' contents change, their names do not).

The snapshot file lists every one of them. A name is public because it is
in that file, not because it lacks an underscore: a module that is not listed (for example `aquascope.cli`,
`aquascope.mcp_server`, `aquascope.dashboard`, `aquascope.maintenance`, the Studio's roles, or the
Archive's build steps such as `aquascope.archive.harvest`) is internal, and so is any name in a listed module
that the file does not name. Call the stable names, or the CLI and MCP faces, instead.

Records (`StreamflowReading`, `Station` and the other schema classes) are covered field by field: a field is
not removed or renamed, and its unit does not change.

### CLI

Every `aquascope` command in the snapshot keeps its name, its flags and its positional arguments. The
`--json` output of a stable command is covered like a Python result: keys are not removed or renamed.

Commands that exist for CI and maintainers carry no promise: `aquascope harvest`, `basins assign`,
`basins build`, `basins signatures`, `basins loo`, `evidence build`, `evidence publish` and
`studio-showcase`.

### MCP

Every tool `aquascope mcp` registers in the snapshot keeps its name and its arguments, and a new argument always
has a default, so a call that worked keeps working. The keys of a tool's result are covered like a Python
result.

### The Archive

[Archive layout](archive-layout.md) is the contract: paths, files, columns, types and units.

## What counts as breaking

A change is breaking when code, a script, an assistant or a query that used a stable name correctly before the
change fails, or silently means something else, after it. In particular:

- removing or renaming a stable module, function, class, constant, method or attribute;
- removing or renaming a parameter, making an optional parameter required, adding a required parameter, or
  moving a parameter that can be passed by position;
- removing a key from a returned dict, a field from a record, or a column from a returned table, or changing
  its type or unit;
- removing or renaming a CLI command or flag, changing what a flag means, or changing the number or order of
  positional arguments;
- removing or renaming an MCP tool or argument, or adding a required argument;
- renaming or removing an Archive path or column, or changing a column's type or unit;
- changing a stable function's default so that the same call returns a different kind of answer (for example
  a different distribution or unit), as opposed to a corrected number.

## What does not count as breaking

- New modules, functions, parameters with defaults, dict keys, record fields, CLI flags, MCP tools, Archive
  files and columns. Read results by key and columns by name.
- Different numbers from a fix. When a method is corrected (a bug, a revised standard such as FAO-56 Rev.1, a
  better fit), its results change; the CHANGELOG says so under "Changed". Numbers are the science and are
  meant to get better; names and shapes are the interface and stay.
- New and revised data: agencies revise records, new days arrive, sources come and go from the catalogue.
- Error messages, log lines, warnings, human-readable text output (without `--json`), progress bars, HTML
  and plot styling.
- Anything internal or experimental (below).
- Dependencies: raising a minimum version, or moving a dependency between extras, in a minor release. Python
  versions: support for a Python version is dropped only in a minor release, announced in the CHANGELOG one
  release ahead. Python 3.10, 3.11 and 3.12 are tested today.

## Deprecation

A stable name is never just removed. It is deprecated first:

1. **It keeps working.** A renamed function stays importable under its old name (`aquascope._deprecation.moved`
   in the old module's `__getattr__`); a function that should not be used any more stays in place
   (`aquascope._deprecation.deprecated`). A CLI verb or flag keeps being accepted, and an MCP tool keeps being
   registered.
2. **It warns.** Python names raise `AquaScopeDeprecationWarning`, a subclass of `FutureWarning`, so the warning
   shows by default in scripts and notebooks. The message says since when, what to use instead, and when it
   goes: "aquascope.x.old is deprecated since 1.2.0 and will be removed in 2.0; use aquascope.x.new instead."
   A deprecated CLI verb or flag prints the same line to stderr and then runs. A deprecated MCP tool's
   description starts with "Deprecated:" and its result carries a `"deprecated"` key with the message, so the
   assistant sees it. A deprecated Archive path is written alongside its replacement, and the prefix's
   `manifest.json` (or the dataset card) says which is going.
3. **It is registered.** The pull request adds an entry to `DEPRECATED` in `aquascope/_deprecation.py`
   (`since`, `use`, `remove_in`) and a CHANGELOG line under "Deprecated". The public-surface test then accepts
   the warning, and still fails if the name stops working before `remove_in`.
4. **It goes only in a major release,** and only after it has warned in released versions for at least six
   months. A name deprecated in 1.9 a month before 2.0 stays until 3.0.

## Experimental

Some parts of AquaScope are shipped, documented and tested, but too new to freeze. They are listed in
`aquascope.experimental` (`MODULES`, `NAMES`, `CLI`, `CLI_FLAGS`, `MCP_TOOLS`) and marked "experimental" in
[Archive layout](archive-layout.md). At 0.27.0 that is everything first released in 0.26.0 and 0.27.0, plus the
benchmark:

- **Python:** `aquascope.rivers`, `river_path`, `nownext`, `evidence`, `watch`, `bulletin`, `context`,
  `flood_depth`, `flowstatus`, `waterbalance`, `whatif_map`, `mesh_review`, the `map_*` modules
  (`map_commands`, `map_explain`, `map_layers`, `map_scout`, `map_time`), `io.mesh`, `io.swmm`,
  `io.engineering`, `groundwater.surface`, `hydrology.b17c_check`, and `aquascope.gym` (to be renamed, #440).
- **CLI:** `layers`, `map`, `whatif`, `mesh`, `swmm`, `bulletin`, `warnings`, `waterbalance`, `flowstatus`,
  `now`, `watch`, `river`, `evidence`, `context`, `modflow`, `export`, `gym`, and the `surface` flags of
  `groundwater`.
- **MCP:** the 36 tools those modules expose (`snap_to_river`, `reach_record`, `flow_forecast`, `map_command`,
  `mesh_review`, `water_balance`, ...; the full list is `aquascope.experimental.MCP_TOOLS`).

Experimental means:

- it may change or be removed in a minor release, without a deprecation period;
- every such change is in the CHANGELOG;
- it is held to the same bar as the rest: real tests, verified sources, no invented numbers;
- it does not warn when used (the Explorer uses most of it every day).

A part graduates when it has gone a few releases without a breaking change and its interface has settled: it
leaves `aquascope.experimental`, its names enter the snapshot, and from then on it is stable.
`aquascope.experimental.is_experimental("aquascope.rivers")` answers the question in code.

## Internal

Anything that is neither in the snapshot nor experimental is internal: underscore names, unlisted modules, the
maintainer commands above, the Explorer's JavaScript, the workflow files. The Explorer installs the wheel
built from the same commit, so a module only its worker imports (the Studio's steering, portable and bundle
helpers, for example) is internal too. Internal code changes whenever it needs
to. If you depend on something internal, open an issue: the answer is usually to make it public.

## How it is enforced

`tests/test_public_api.py` runs in CI on every pull request. It fails when a stable Python name, CLI command or
flag, or MCP tool or argument is missing without a removable `DEPRECATED` entry; when a deprecated name stops
warning; when a stable MCP tool gains a required argument; when a new command or tool is not classified as
stable, experimental or internal; when a name is added to a stable module's `__all__` but not to the snapshot;
and when a prefix the code publishes is missing from [Archive layout](archive-layout.md).

To make something new stable, add it and run `python tests/test_public_api.py --update`, which adds to the
snapshot and never removes. The diff of `tests/fixtures/public_api.json` is the review of the promise.
