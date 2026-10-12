"""The Archive: harvest AquaScope's sources into open, cloud-native files.

Phase 0 (#188) harvests every station catalog into ``stations.parquet``
(GeoParquet), ``stations.geojson`` and a ``health.json`` status report, and
publishes the folder to a public Hugging Face dataset. Phase 1 adds daily
observations per station (``obs/<variable>/<source>/<id>.csv.gz``); Phase 2
adds more variables and one Parquet bundle per (variable, source). Only
sources whose terms allow it are ever mirrored; the registry's
``redistributable`` flag is the gate.

Requires the ``archive`` extra (``pip install "aquascope[archive]"``).
"""

from aquascope.archive.bundles import build_bundles, load_observations
from aquascope.archive.harvest import HarvestReport, harvest_stations, infer_site_ids, write_dataset_card
from aquascope.archive.observations import fetch_archived_series, harvest_observations
from aquascope.archive.publish import publish_folder

__all__ = [
    "HarvestReport",
    "build_bundles",
    "fetch_archived_series",
    "harvest_observations",
    "harvest_stations",
    "infer_site_ids",
    "load_observations",
    "publish_folder",
    "write_dataset_card",
]
