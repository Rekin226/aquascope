"""The public surface that 1.0 promises (#588, ``docs/stability.md``).

``tests/fixtures/public_api.json`` is the snapshot: the stable Python names per module, the stable CLI commands
with their flags, and the stable MCP tools with their arguments. These tests fail when

* a stable name, command, flag, tool or argument disappears without a removable entry in
  ``aquascope._deprecation.DEPRECATED``;
* a deprecated name stops warning;
* a stable MCP tool gains a required argument;
* a new name, command or tool appears without being classified (stable in the snapshot, experimental in
  ``aquascope.experimental``, or internal in the snapshot's ``cli_internal``).

Adding to the stable surface is a reviewed change: ``python tests/test_public_api.py --update`` adds what is new
to the snapshot and never removes anything. Removing from it is a major-release change.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "fixtures" / "public_api.json"

if str(ROOT) not in sys.path:  # `python tests/test_public_api.py --update` from a checkout
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import aquascope  # noqa: E402
from aquascope import _deprecation, experimental  # noqa: E402

ALL = "__all__"  # take the module's own __all__
FUNCTIONS = "functions"  # every public function and class the module itself defines

# How the stable Python surface is chosen: what the README and the docs show, read conservatively. The Explorer's
# worker installs the wheel built from the same commit, so a name only the worker imports is internal (the
# Explorer tests guard it); every stable-module name the worker uses is in here anyway. A module's ``__all__``
# is its author's statement of what is public; elsewhere only the names the
# documentation shows are frozen. Experimental modules and names (aquascope.experimental) are left out.
STABLE_SPEC: dict[str, Any] = {
    "aquascope": ["__version__", "collect", "recommend", "HydroAgent", "ChallengePlanner", "ModelRecommender",
                  "plan_irrigation", "benchmark_aquastat", "estimate_wapor_productivity", "flood_analysis",
                  "baseflow_analysis", "flow_duration", "compute_all_signatures", "detect_changepoints", "fit_copula",
                  "bayesian_regression", "ensemble_forecast", "generate_report", "groundwater_analysis",
                  "climate_downscale", "climate_indices", "find_stations", "station_catalogs", "Station",
                  "GRACEProcessor", "CMIP6Processor"],
    "aquascope.api": FUNCTIONS,
    "aquascope.advanced": ALL,
    "aquascope.agri": ALL,
    "aquascope.agri.water_balance": ["SoilProperties", "SoilWaterBalance"],
    "aquascope.ai_engine": ALL,
    "aquascope.ai_engine.agent": ["HydroAgent"],
    "aquascope.ai_engine.analyst": ["AskResult", "ask"],
    "aquascope.ai_engine.llm_transport": ALL,
    "aquascope.ai_engine.providers": ALL,
    "aquascope.ai_engine.recommender": ["DatasetProfile", "Recommendation", "recommend"],
    "aquascope.ai_engine.team": ALL,
    "aquascope.ai_engine.verify": ALL,
    "aquascope.analysis.eda": ["generate_eda_report", "print_eda_report", "profile_dataset"],
    "aquascope.analysis.quality": ["assess_quality", "preprocess", "print_quality_report"],
    "aquascope.archive": ALL,
    "aquascope.archive.basins": ALL,
    "aquascope.archive.caravan": ALL,
    "aquascope.archive.catalog": ["catalog_url", "load_stations", "search_stations", "set_catalog"],
    "aquascope.archive.regionalize": ALL,
    "aquascope.archive.signatures": ALL,
    "aquascope.archive.similar": ALL,
    "aquascope.area_study": ALL,
    "aquascope.collectors": ALL,
    "aquascope.collectors.base": ["BaseCollector"],
    "aquascope.compare": ["compare_series", "compare_stations"],
    "aquascope.evaluation": ["find_studies", "score_path", "stats"],
    "aquascope.experimental": ["CLI", "CLI_FLAGS", "MCP_TOOLS", "MODULES", "NAMES", "is_experimental",
                               "is_experimental_command"],
    "aquascope.explore": ["analyze_series", "analyze_station", "anywhere", "assess_site", "fetch_series", "flood_ci",
                          "to_csv", "water_quality_samples"],
    "aquascope.hydrology": ALL,
    "aquascope.hydrology.flood_frequency": ["expected_moments_algorithm", "fit_nonstationary_gev"],
    "aquascope.ingest": ["ingest", "ingest_text"],
    "aquascope.io": ALL,
    "aquascope.io.interop": ["records_to_geodataframe", "records_to_xarray"],
    "aquascope.methods": ALL,
    "aquascope.models.rainfall_runoff": ["GR4J", "predict_quantiles"],
    "aquascope.pipelines.model_builder": ["list_available_pipelines", "run_pipeline"],
    "aquascope.playbooks": ALL,
    "aquascope.registry": ALL,
    "aquascope.schemas.station": ["Station", "VARIABLES", "in_bbox"],
    "aquascope.schemas.water_data": ["ClimateReading", "DataSource", "GeoLocation", "Quality", "StreamflowReading",
                                     "WaterLevelReading", "WaterQualitySample"],
    "aquascope.studio": ALL,
    "aquascope.studio.desk": ["comment", "resolve", "sensitivity"],
    "aquascope.studio.prompts": ["as_json"],
    "aquascope.utils.http_client": ["CachedHTTPClient", "RateLimiter"],
    "aquascope.utils.storage": ["export_netcdf", "save_records"],
    "aquascope.viz": ALL,
    "aquascope.workbench": ALL,
}


# ── reading the surface as it is now ────────────────────────────────────────


def python_surface() -> dict[str, list[str]]:
    """The stable Python names the package has now, per module, from :data:`STABLE_SPEC`."""
    out: dict[str, list[str]] = {}
    for mod_name, how in STABLE_SPEC.items():
        mod = importlib.import_module(mod_name)
        if how == ALL:
            names = list(mod.__all__)
        elif how == FUNCTIONS:
            names = [n for n, v in vars(mod).items() if not n.startswith("_")
                     and (inspect.isfunction(v) or inspect.isclass(v)) and v.__module__ == mod_name]
        else:
            names = list(how)
        skip = set(experimental.NAMES.get(mod_name, ()))
        out[mod_name] = sorted(n for n in names if n not in skip)
    return out


class _ParsedError(Exception):
    pass


def cli_parser() -> argparse.ArgumentParser:
    """The ``aquascope`` argument parser, captured from ``aquascope.cli.main`` without running a command."""
    from aquascope import cli

    seen: dict[str, argparse.ArgumentParser] = {}

    def capture(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        seen["parser"] = self
        raise _ParsedError

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture), \
            mock.patch.object(sys, "argv", ["aquascope"]):
        try:
            cli.main()
        except _ParsedError:
            pass
    return seen["parser"]


def cli_surface() -> dict[str, dict[str, Any]]:
    """Every command (``"river snap"``) with its long flags and its number of positional arguments."""
    out: dict[str, dict[str, Any]] = {}

    def walk(parser: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        flags: list[str] = []
        positionals = 0
        subs = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subs = action
            elif isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
                continue
            elif action.option_strings:
                flags.append(max(action.option_strings, key=len))
            else:
                positionals += 1
        if prefix:
            out[" ".join(prefix)] = {"flags": sorted(flags), "positionals": positionals}
        if subs is not None:
            done: set[int] = set()
            for name, sub in subs.choices.items():
                if id(sub) not in done:  # an alias shares its parser
                    done.add(id(sub))
                    walk(sub, (*prefix, name))

    walk(cli_parser(), ())
    return out


def mcp_surface() -> dict[str, dict[str, Any]]:
    """Every MCP tool ``aquascope mcp`` registers, with its arguments and which of them are required. Read by
    registering on a recorder, so the ``mcp`` package is not needed."""
    from aquascope import mcp_server

    tools: dict[str, dict[str, Any]] = {}

    class Recorder:
        def tool(self, *args: Any, **kwargs: Any):
            def register(fn):
                params = inspect.signature(fn).parameters.values()
                tools[fn.__name__] = {"args": [p.name for p in params],
                                      "required": [p.name for p in params if p.default is inspect.Parameter.empty]}
                return fn
            return register

        def resource(self, *args: Any, **kwargs: Any):
            return lambda fn: fn

    with mock.patch.object(mcp_server, "_server", Recorder):
        mcp_server.build_server()
    return tools


def load_snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def _version() -> str:
    return aquascope.__version__


def _gone_ok(key: str) -> bool:
    return _deprecation.removable(key, _version())


# ── the tests ───────────────────────────────────────────────────────────────


def test_stable_python_names_still_import():
    missing = []
    for mod_name, names in load_snapshot()["python"].items():
        for name in names:
            key = f"{mod_name}:{name}"
            if _gone_ok(key):
                continue
            try:
                mod = importlib.import_module(mod_name)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    obj = getattr(mod, name)
            except (ImportError, AttributeError):
                missing.append(key)
                continue
            if key in _deprecation.DEPRECATED:
                warned = any(issubclass(w.category, _deprecation.AquaScopeDeprecationWarning) for w in caught)
                assert warned or hasattr(obj, "__deprecated__"), f"{key} is deprecated but does not warn"
    assert not missing, (
        "Stable names are gone without a deprecation (docs/stability.md): " + ", ".join(missing)
        + ". Keep them working through aquascope._deprecation.moved or .deprecated and register them in DEPRECATED."
    )


def test_stable_cli_commands_and_flags_still_parse():
    now = cli_surface()
    problems = []
    for command, frozen in load_snapshot()["cli"].items():
        if _gone_ok(f"cli:{command}"):
            continue
        if command not in now:
            problems.append(f"aquascope {command} is gone")
            continue
        for flag in frozen["flags"]:
            if flag not in now[command]["flags"] and not _gone_ok(f"cli:{command} {flag}"):
                problems.append(f"aquascope {command} {flag} is gone")
        if now[command]["positionals"] != frozen["positionals"]:
            problems.append(f"aquascope {command} takes {now[command]['positionals']} positional arguments, "
                            f"not {frozen['positionals']}")
    assert not problems, "Breaking CLI changes (docs/stability.md): " + "; ".join(problems)


def test_stable_mcp_tools_and_arguments_still_exist():
    now = mcp_surface()
    problems = []
    for tool, frozen in load_snapshot()["mcp"].items():
        if _gone_ok(f"mcp:{tool}"):
            continue
        if tool not in now:
            problems.append(f"tool {tool} is gone")
            continue
        for arg in frozen["args"]:
            if arg not in now[tool]["args"] and not _gone_ok(f"mcp:{tool}.{arg}"):
                problems.append(f"{tool}({arg}) is gone")
        new_required = sorted(set(now[tool]["required"]) - set(frozen["required"]))
        if new_required:
            problems.append(f"{tool} now requires {', '.join(new_required)} (a new argument needs a default)")
    assert not problems, "Breaking MCP changes (docs/stability.md): " + "; ".join(problems)


def test_new_public_names_are_recorded():
    """A name added to a stable module's ``__all__`` (or to STABLE_SPEC) is a promise: it goes in the snapshot."""
    frozen = load_snapshot()["python"]
    new = {m: sorted(set(names) - set(frozen.get(m, []))) for m, names in python_surface().items()}
    new = {m: n for m, n in new.items() if n}
    assert not new, (f"New stable names not in the snapshot: {new}. Either run `python tests/test_public_api.py "
                     "--update`, or list them in aquascope.experimental.NAMES.")


def test_every_command_and_tool_is_classified():
    snap = load_snapshot()
    internal = snap["cli_internal"]
    loose = [c for c in cli_surface() if c not in snap["cli"] and not experimental.is_experimental_command(c)
             and not any(c == i or c.startswith(i + " ") for i in internal)]
    assert not loose, (f"Unclassified CLI commands: {loose}. Add them to aquascope.experimental.CLI, to the "
                       "snapshot's cli_internal, or (stable) with `python tests/test_public_api.py --update`.")
    tools = [t for t in mcp_surface() if t not in snap["mcp"] and t not in experimental.MCP_TOOLS]
    assert not tools, (f"Unclassified MCP tools: {tools}. Add them to aquascope.experimental.MCP_TOOLS, or "
                       "(stable) with `python tests/test_public_api.py --update`.")


def test_experimental_and_stable_do_not_overlap():
    snap = load_snapshot()
    assert not [m for m in snap["python"] if experimental.is_experimental(m)]
    assert not [c for c in snap["cli"] if experimental.is_experimental_command(c)]
    assert not set(snap["mcp"]) & set(experimental.MCP_TOOLS)
    for mod_name, names in experimental.NAMES.items():
        assert not set(names) & set(snap["python"].get(mod_name, []))
    for command, flags in experimental.CLI_FLAGS.items():
        assert not set(flags) & set(snap["cli"].get(command, {}).get("flags", []))


def test_experimental_entries_exist():
    """A stale experimental entry hides nothing, but it misleads: every one must name something real."""
    for mod_name in experimental.MODULES:
        importlib.import_module(mod_name)
    for mod_name, names in experimental.NAMES.items():
        mod = importlib.import_module(mod_name)
        assert all(hasattr(mod, n) for n in names), mod_name
    commands = cli_surface()
    assert all(c in commands for c in experimental.CLI)
    for command, flags in experimental.CLI_FLAGS.items():
        assert set(flags) <= set(commands[command]["flags"]), command
    assert set(experimental.MCP_TOOLS) <= set(mcp_surface())


def test_deprecation_register_is_well_formed():
    major = int(_version().split(".")[0])
    for key, entry in _deprecation.DEPRECATED.items():
        assert re.fullmatch(r"(cli:[a-z][\w -]*|mcp:\w+(\.\w+)?|aquascope(\.\w+)*:\w+)", key), key
        assert int(entry.remove_in.split(".")[0]) > int(entry.since.split(".")[0]), key
        assert int(entry.remove_in.split(".")[0]) >= major, key
        assert entry.use.strip(), key


def test_moved_name_keeps_working_and_warns():
    register = {"aquascope.old_home:old_name": _deprecation.Deprecation(since="1.1.0", use="aquascope.collect")}
    getattr_ = _deprecation.moved("aquascope.old_home", {"old_name": "aquascope:collect"}, register=register)
    with pytest.warns(_deprecation.AquaScopeDeprecationWarning, match="deprecated since 1.1.0.*removed in 2.0"):
        assert getattr_("old_name") is aquascope.collect
    with pytest.raises(AttributeError):
        getattr_("never_existed")


def test_deprecated_function_warns_when_called():
    register = {"aquascope.x:f": _deprecation.Deprecation(since="1.2.0", use="aquascope.x.g")}

    @_deprecation.deprecated("aquascope.x:f", register=register)
    def f(a: int) -> int:
        return a + 1

    with pytest.warns(FutureWarning, match="aquascope.x.f is deprecated"):
        assert f(1) == 2
    assert "use aquascope.x.g" in f.__deprecated__
    assert _deprecation.removable("aquascope.x:f", "2.0.0", register)
    assert not _deprecation.removable("aquascope.x:f", "1.9.0", register)
    assert "aquascope river snap is deprecated" in _deprecation.message(
        "cli:river snap", {"cli:river snap": _deprecation.Deprecation(since="1.1.0", use="aquascope reach")})


# The Archive's published prefixes, as the code names them: each must be documented in docs/archive-layout.md.
ARCHIVE_FOLDERS = [
    ("aquascope.archive.forecasts", "FOLDER"), ("aquascope.archive.warnings", "FOLDER"),
    ("aquascope.archive.feeds", "FOLDER"), ("aquascope.bulletin", "FOLDER"), ("aquascope.evidence", "SKILL_FOLDER"),
    ("aquascope.flowstatus", "FOLDER"), ("aquascope.flowstatus", "CLIM_FOLDER"), ("aquascope.map_scout", "FOLDER"),
    ("aquascope.waterbalance", "FOLDER"), ("aquascope.context._common", "MIRROR_FOLDER"),
    ("aquascope.context.floods_past", "FOLDER"), ("aquascope.archive.health_jobs", "FOLDER"),
    ("aquascope.archive.samples", "FOLDER"),
]


def test_archive_prefixes_are_documented():
    doc = (ROOT / "docs" / "archive-layout.md").read_text(encoding="utf-8")
    undocumented = []
    for mod_name, attr in ARCHIVE_FOLDERS:
        folder = getattr(importlib.import_module(mod_name), attr)
        if mod_name.startswith("aquascope.context.floods_past"):
            folder = f"context/{folder}"
        if f"`{folder}/" not in doc:
            undocumented.append(folder)
    for top in ("stations.parquet", "stations.geojson", "health.json", "signatures.parquet", "obs/", "basins/"):
        if f"`{top}" not in doc:
            undocumented.append(top)
    assert not undocumented, f"Archive paths missing from docs/archive-layout.md: {undocumented}"


def test_policy_pages_are_linked():
    for page in ("stability.md", "archive-layout.md"):
        assert (ROOT / "docs" / page).exists()
        assert page in (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), page
        assert f"docs/{page}" in (ROOT / "README.md").read_text(encoding="utf-8"), page


# ── updating the snapshot ───────────────────────────────────────────────────


def updated_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """``snap`` with every new stable name, command, flag, tool and argument added. Nothing is ever removed."""
    out = json.loads(json.dumps(snap))
    for mod_name, names in python_surface().items():
        out["python"][mod_name] = sorted(set(out["python"].get(mod_name, [])) | set(names))
    internal = out["cli_internal"]
    for command, now in cli_surface().items():
        if experimental.is_experimental_command(command) or any(
                command == i or command.startswith(i + " ") for i in internal):
            continue
        flags = set(now["flags"]) - set(experimental.CLI_FLAGS.get(command, ()))
        old = out["cli"].get(command, {"flags": []})
        out["cli"][command] = {"flags": sorted(set(old["flags"]) | flags), "positionals": now["positionals"]}
    for tool, now in mcp_surface().items():
        if tool in experimental.MCP_TOOLS:
            continue
        old = out["mcp"].get(tool, {"args": [], "required": now["required"]})
        out["mcp"][tool] = {"args": old["args"] + [a for a in now["args"] if a not in old["args"]],
                            "required": old["required"]}
    out["python"] = dict(sorted(out["python"].items()))
    out["cli"] = dict(sorted(out["cli"].items()))
    out["mcp"] = dict(sorted(out["mcp"].items()))
    return out


if __name__ == "__main__":  # python tests/test_public_api.py --update
    if sys.argv[1:] != ["--update"]:
        sys.exit("usage: python tests/test_public_api.py --update   (adds new stable names; never removes)")
    current = load_snapshot() if SNAPSHOT.exists() else {
        "about": "", "python": {}, "cli": {}, "cli_internal": [], "mcp": {}}
    SNAPSHOT.write_text(json.dumps(updated_snapshot(current), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)}")
