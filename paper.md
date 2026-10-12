---
title: 'AquaScope: One Open Hydrology Engine for Python, the Browser and AI Assistants'
tags:
  - Python
  - hydrology
  - open data
  - flood frequency analysis
  - WebAssembly
  - large language models
  - reproducibility
authors:
  - given-names: Abdoul Rachid
    surname: Ouédraogo
    orcid: 0000-0002-4616-4153
    affiliation: 1
affiliations:
  - name: National Central University, Taiwan
    index: 1
    ror: 00944ve71
date: 12 October 2026
bibliography: paper.bib
---

<!-- This text describes release 0.27.0. Counts (sources, tests, cases) are as of that
     release; refresh them all together with the release that is submitted (#72, #595). -->

# Summary

AquaScope is an open-source (MIT) Python package for river and water data. It reads
records from 37 public data sources (national agencies in the Americas, Europe, Africa,
Asia and Australia, and global services) into one set of schemas, runs hydrological
analyses on them, and reports each result with the period, units, method and checks it
rests on. The same engine reaches users in three ways: as a Python library and
command-line tool, as a web application (the Explorer) that runs the package inside the
browser with nothing installed, and as tools for AI assistants through the Model Context
Protocol. A scheduled harvest also publishes an open Archive of station catalogs and
daily observations. Release 0.27.0 is on PyPI and archived on Zenodo (concept DOI
10.5281/zenodo.21903143); the Explorer runs at
<https://rekin226-aquascope-explorer.static.hf.space/>.

# Statement of need

A hydrologist asked "how big is the 100-year flood here, and is it changing?" must first
find a gauge, learn its agency's interface, check how long and complete the record
really is, choose a method the record can support, and then explain the answer and its
uncertainty to someone who will act on it. Outside the United States the first steps
alone often mean a bespoke script per agency. The last step is now also served by
language models, which write fluent answers but can quote numbers no analysis produced.

AquaScope targets researchers, consultants, students and agency staff who work across
countries and need defensible answers rather than a single data client or a single
model. Its purpose is to make the whole path (find, check, analyse, report) one
reproducible chain that a person, a script or an AI assistant can follow, and to make
the limits of each answer as visible as the answer.

# State of the field

Several mature tools cover parts of this path. HyRiver [@Chegini2021] gives Python
access to hydroclimate web services, mainly United States and North American federal
ones (NWIS, NHDPlus, 3DEP, Daymet), as analysis-ready `xarray` and `geopandas` objects.
NeuralHydrology [@Kratzert2022] is a library for training deep-learning rainfall-runoff
models. Caravan [@Kratzert2023] standardises large-sample datasets of forcing,
streamflow and catchment attributes for global studies. HydroShare [@Horsburgh2016] is a
repository for publishing and sharing hydrologic data and models. The GEOGloWS ECMWF
Streamflow Service [@Hales2025], GloFAS [@Alfieri2013; @Harrigan2020] and Google's Flood
Hub [@Nearing2024] provide global modelled discharge, forecasts and warnings.

AquaScope builds on these rather than replacing them: it reads GEOGloWS and GloFAS
forecasts and simulations, grades them (and the Flood Hub reanalysis) against gauge
records, writes Caravan-format datasets from its Archive, and exports records as inputs
for HEC-HMS, HEC-RAS, HEC-SSP, SWMM, MODFLOW 6, Delft-FEWS and Raven. What it adds is
the connection between them: one engine that runs unchanged in a notebook, in a browser
tab and behind an assistant, and that checks what each answer may claim. We chose to
build rather than extend an existing client because that requirement reaches every layer
(the HTTP transport must work under WebAssembly, the base install must stay small enough
for the browser, and every analysis must return a structured result an assistant can be
checked against), which is outside the scope those projects set themselves.

# Software design

**One engine, three faces.** Every capability is a plain function from inputs to a
dictionary in the package. The command-line interface, the MCP server (more than fifty
tools) and the Explorer are thin layers over the same functions, so a number in the
browser, a script and an assistant's answer come from the same code. Collectors subclass
one base class and normalise into Pydantic schemas; analysis uses pandas
[@McKinney2010], NumPy [@Harris2020] and SciPy [@Virtanen2020]; heavy dependencies are
optional extras.

**The browser runs the package.** The Explorer is a static page: MapLibre for the map,
DuckDB-WASM [@Raasveldt2019] to query the Archive's GeoParquet catalog in place, and
Pyodide to run the AquaScope wheel in a web worker. A click on a gauge fetches its
record and computes flood frequency (GEV by L-moments [@Hosking1997] and log-Pearson
III), flow duration and a Mann-Kendall trend [@Mann1945]. A click on a river snaps to
its GEOGloWS reach and shows its simulated record since 1940, its 15-day forecast and
its path to the sea. Users can also drop their own model results on the globe (HEC-RAS
2D and UGRID meshes, MODFLOW heads, SWMM networks); the files are read in the browser,
never uploaded, and reviewed by code checks pinned on the map. \autoref{fig:explorer}
shows the map with its keyless notes.

**What a method needs is data.** A registry of 26 methods records, with citations, the
record length below which each is marginal or not defensible, catchment-size ceilings
for lumped models such as GR4J [@Perrin2003], and return-period limits relative to
record length. Ten problem playbooks (flood risk, drought, groundwater decline, supply
reliability and others) and a catalogue of 43 plan steps read this registry, and every
step carries gates, small checks evaluated on its output (for example a minimum number
of years, a finite confidence interval, or a Nash-Sutcliffe [@Nash1970] or Kling-Gupta
[@Gupta2009] floor). A failed gate fails its step and is reported; a study's answer is
graded established, indicative, screening or not established.

**AI under verification.** The AI layer is optional and keyless by default: rules and
templates plan and write without a model, a small model can run on the device, and a
user's own key enables a hosted model. In every tier, models plan and word but do not
compute. A model-written plan must pass a validator that checks every step against the
catalogue and the registry; a sentence whose numbers are in no result is dropped; and on
the map a "claim lock" makes the model write each number as a placeholder that code
fills in, refusing wording with any other number or with a sign or digit attached to a
placeholder. HydroGym, a benchmark shipped with the package, scores agents against 25
reference plans written by a hydrologist at seven real sites, including cases where the
right answer is to decline.

![The Explorer over the Ganges and Brahmaputra basins (map date 15 September 2026). Four notes
from "Explain this view", written by rules with no model, each give a number read
by code from a named source: monthly river status (GEOGloWS), a 15-day forecast
peak, and reported flood events.\label{fig:explorer}](docs/img/globe/explain-bangladesh.png)

# Research impact statement

AquaScope has had 25 GitHub releases since May 2026 (the recent ones archived on Zenodo
under one concept DOI), and 32 people besides the maintainer appear on its
all-contributors board, most with merged code or tests. The Archive
(<https://huggingface.co/datasets/Rekin226/aquascope-gauges>) held 75,787 station
records at its last published harvest (5 October 2026), with daily observations mirrored
only for sources whose licences allow it. A GeoLibre plugin and a QGIS layer file read
it directly.

The benchmark already changed the software. In the first recorded run (7 September 2026)
of the plan-writing role with two Claude models, both scored about 0.9 on the solvable
cases but declined none of the seven cases that should be declined. After the engine was
changed to honour each playbook's decline rules before any model is asked and to let a
model decline, a rerun the next day scored 0.96 and 0.93 against 0.97 for the rule tree
alone, with every out-of-scope case declined. The rows,
plans and scoring code are in the repository with a deposit package for a DOI.
Peer-reviewed studies that use AquaScope are not yet known to us; documented use by
people outside the project is an open acceptance criterion for version 1.0.

# Limitations

The test suite (about 4,300 tests, Python 3.10 to 3.12) and the CAMELS-based regression
harness [@Addor2017] check the software, not every scientific use. The harness uses
synthetic daily series calibrated toward ten CAMELS catchments and observed USGS annual
peaks; the Bulletin 17C [@England2018] routines match a published example on a plain
record but not yet one with low outliers. HydroGym's references are one hydrologist's,
and its name collides with an unrelated reinforcement-learning platform, so a rename is
planned before its DOI. Agency interfaces change, some block browser requests, and the
weekly harvest has failed silently before; the Archive has no DOI of its own yet. Global
model outputs are simulations and are labelled so; the 15-day flood outlook on the map
is not an official warning. The public API is not frozen until 1.0.

# AI usage disclosure

Generative AI was used extensively. Much of the code, tests and documentation was
written with Anthropic's Claude models through Claude Code, under the maintainer's
direction: 551 of the 818 non-merge commits on the main branch at 0.27.0 carry a Claude
co-author trailer. Changes are merged through pull requests that must pass continuous
integration (lint, tests on three Python versions, JavaScript tests, the regression
harness), and the maintainer is responsible for every release. This paper was drafted
with Claude from the repository's documentation, and its final text is the author's
responsibility; every reference was resolved to its DOI through OpenAlex or Crossref,
and every count was read from the repository or the published Archive.

# Acknowledgements

We thank the contributors listed in the repository, the maintainers of the scientific
Python ecosystem, and the agencies and projects whose open data make AquaScope possible,
among them USGS, the UK Environment Agency, Hub'Eau, PEGELONLINE, Taiwan's MOENV and
WRA, ECMWF, GEOGloWS and the FAO.

# References
