"""What in AquaScope is experimental: shipped, documented and tested, but not yet covered by the 1.0 promise.

An experimental module, CLI verb or MCP tool may change or go away in a minor release, without a deprecation
period. It is still held to the usual bar (real tests, real data, no invented numbers); what it lacks is a
promise that its names and arguments will stay. The CHANGELOG says when one changes, and when one becomes
stable it moves out of these lists and into the public-surface snapshot (``tests/fixtures/public_api.json``).

What is here and why (``docs/stability.md`` has the full policy):

* Everything first released in 0.26.0 and 0.27.0 (October 2026): rivers as objects, now and next, the evidence
  ladder, watch, the monthly bulletin, place context, engineering exports, and the model-results-on-the-map arc
  (``map_*``, flood depth, flow against normal, the water balance, mesh review, what if on the map, MODFLOW
  heads, SWMM). These modules changed in nearly every review of their first week.
* ``aquascope.gym`` and ``aquascope gym``: the benchmark is being renamed (#440). The old name will keep working
  as a deprecated alias, but the package path is not frozen before the rename.

Anything not listed here and not in the snapshot (for example ``aquascope.dashboard``, ``aquascope.maintenance``,
the Archive's build steps and every name that starts with an underscore) is internal: free to change at any time.

Use :func:`is_experimental` to check a module path.
"""

from __future__ import annotations

# Module paths. A package covers its submodules (``aquascope.context`` covers ``aquascope.context.floods_past``).
MODULES: tuple[str, ...] = (
    "aquascope.bulletin",
    "aquascope.context",
    "aquascope.evidence",
    "aquascope.flood_depth",
    "aquascope.flowstatus",
    "aquascope.groundwater.surface",
    "aquascope.gym",
    "aquascope.hydrology.b17c_check",
    "aquascope.io.engineering",
    "aquascope.io.mesh",
    "aquascope.io.swmm",
    "aquascope.map_commands",
    "aquascope.map_explain",
    "aquascope.map_layers",
    "aquascope.map_scout",
    "aquascope.map_time",
    "aquascope.mesh_review",
    "aquascope.nownext",
    "aquascope.river_path",
    "aquascope.rivers",
    "aquascope.watch",
    "aquascope.waterbalance",
    "aquascope.whatif_map",
)

# Experimental names re-exported from an otherwise stable module.
NAMES: dict[str, tuple[str, ...]] = {
    "aquascope.io": ("mesh_summary", "read_mesh", "read_ras_plan", "read_swmm_inp", "read_swmm_out", "read_ugrid",
                     "swmm_summary"),
}

# CLI commands, as typed after ``aquascope`` (a verb covers its subcommands).
CLI: tuple[str, ...] = (
    "bulletin",
    "context",
    "evidence",
    "export",
    "flowstatus",
    "gym",
    "layers",
    "map",
    "mesh",
    "modflow",
    "now",
    "river",
    "swmm",
    "warnings",
    "watch",
    "waterbalance",
    "whatif",
)

# Experimental flags on a stable verb: ``aquascope groundwater surface`` (MODFLOW and well surfaces, 0.27.0).
CLI_FLAGS: dict[str, tuple[str, ...]] = {
    "groundwater": ("--bbox", "--compare", "--geojson", "--json", "--mode", "--month"),
}

# MCP tool names (``aquascope mcp``).
MCP_TOOLS: tuple[str, ...] = (
    "area_context",
    "correct_to_gauge",
    "dated_layers",
    "downstream_ids",
    "engineering_export",
    "explain_view",
    "flood_depth_overlay",
    "flood_events_month",
    "flood_warnings",
    "flow_forecast",
    "flow_status",
    "flow_vs_normal",
    "forecast_plume",
    "groundwater_surface",
    "layer_frames",
    "map_command",
    "map_scout",
    "mesh_review",
    "mesh_summary",
    "model_skill",
    "model_to_lean_on",
    "modflow_heads",
    "place_context",
    "reach_record",
    "river_status_month",
    "river_status_summary",
    "snap_to_river",
    "status_bulletin",
    "swmm_summary",
    "trace_downstream",
    "upstream_area",
    "upstream_dams",
    "upstream_ids",
    "watch_digest",
    "water_balance",
    "whatif_map",
)


def is_experimental(module: str) -> bool:
    """True when ``module`` (a dotted path) is, or sits inside, an experimental module."""
    return any(module == m or module.startswith(m + ".") for m in MODULES)


def is_experimental_command(command: str) -> bool:
    """True when the CLI ``command`` (``"river snap"``) is, or sits under, an experimental verb."""
    return any(command == c or command.startswith(c + " ") for c in CLI)
