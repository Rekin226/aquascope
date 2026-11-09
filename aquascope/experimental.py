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
* The MODFLOW views added after 0.27.0 (#625, #626): ``aquascope.groundwater.layers`` (layers, cross-sections,
  a cell's heads over time) and ``aquascope.groundwater.budget`` (the cell-by-cell budget reader).
* The map that computes (#664): ``aquascope.routing``, ``aquascope route`` and the MCP tools ``route_flood`` and
  ``route_storm`` (#656), new in 1.1.
* Paint the globe (#657, after 1.0.0): ``aquascope.paint``, ``aquascope paint`` and the MCP tool ``paint_map``.
* Flood it (#658): ``aquascope.floodsim`` (a local-inertial screening flood of a river's valley), ``aquascope flood``
  and the MCP tool ``flood_reach``, new in 1.1.
* Replay an event (#659): ``aquascope.replay`` (a past flood assembled from its data and told as a story),
  ``aquascope replay`` and the MCP tools ``replay_candidates`` and ``replay_event``, new in 1.1.
* The sketchpad (#664): ``aquascope.sketch``. Drop a well (#660): ``aquascope.sketch.drawdown`` (Theis with
  superposition), ``aquascope well`` and the MCP tool ``drop_well``; change land use (#661):
  ``aquascope.sketch.landuse`` (the SCS curve number, TR-55), ``aquascope landuse`` and the MCP tool
  ``change_landuse``; a drawn reservoir (#662: ``aquascope.sketch.reservoir``, ``aquascope route reservoir``
  and the MCP tool ``route_reservoir``), new in 1.1.
* ``aquascope.gym`` and ``aquascope gym``: the benchmark, now called AquaScope Bench (#440). Only the name changed;
  the module path and the CLI verb stay, and the task suite is not frozen before its DOI (#175).

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
    "aquascope.flood_exposure",
    "aquascope.flood_section",
    "aquascope.floodsim",
    "aquascope.flowstatus",
    "aquascope.groundwater.budget",
    "aquascope.groundwater.layers",
    "aquascope.groundwater.surface",
    "aquascope.groundwater.views",
    "aquascope.gym",
    "aquascope.hydrology.b17c_check",
    "aquascope.io.engineering",
    "aquascope.io.mesh",
    "aquascope.io.swmm",
    "aquascope.map_commands",
    "aquascope.map_explain",
    "aquascope.map_layers",
    "aquascope.map_pick",
    "aquascope.map_scout",
    "aquascope.map_time",
    "aquascope.mesh_review",
    "aquascope.mesh_views",
    "aquascope.nownext",
    "aquascope.paint",
    "aquascope.replay",
    "aquascope.river_path",
    "aquascope.rivers",
    "aquascope.routing",
    "aquascope.sketch",
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
    "flood",
    "flowstatus",
    "gym",
    "landuse",
    "layers",
    "map",
    "mesh",
    "modflow",
    "now",
    "paint",
    "replay",
    "river",
    "route",
    "swmm",
    "warnings",
    "watch",
    "waterbalance",
    "well",
    "whatif",
)

# Experimental flags on a stable verb: ``aquascope groundwater surface`` (MODFLOW and well surfaces, 0.27.0).
CLI_FLAGS: dict[str, tuple[str, ...]] = {
    "groundwater": ("--bbox", "--compare", "--geojson", "--json", "--mode", "--month"),
}

# MCP tool names (``aquascope mcp``).
MCP_TOOLS: tuple[str, ...] = (
    "area_context",
    "change_landuse",
    "correct_to_gauge",
    "dated_layers",
    "downstream_ids",
    "drop_well",
    "engineering_export",
    "explain_view",
    "flood_depth_overlay",
    "flood_events_month",
    "flood_reach",
    "flood_warnings",
    "flow_forecast",
    "flow_status",
    "flow_vs_normal",
    "forecast_plume",
    "groundwater_surface",
    "layer_frames",
    "map_command",
    "map_scout",
    "mesh_lenses",
    "mesh_review",
    "mesh_summary",
    "model_skill",
    "model_to_lean_on",
    "modflow_heads",
    "paint_map",
    "place_context",
    "reach_record",
    "replay_candidates",
    "replay_event",
    "river_status_month",
    "river_status_summary",
    "route_flood",
    "route_reservoir",
    "route_storm",
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
