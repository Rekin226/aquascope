"""Data collectors for Taiwan and global water data sources."""

from aquascope.collectors.base import BaseCollector
from aquascope.collectors.sdg6 import SDG6Collector
from aquascope.collectors.taiwan_moenv import TaiwanMOENVCollector
from aquascope.collectors.taiwan_wra import (
    TaiwanWRAReservoirCollector,
    TaiwanWRAWaterLevelCollector,
)
from aquascope.collectors.usgs import USGSCollector

__all__ = [
    "BaseCollector",
    "TaiwanMOENVCollector",
    "TaiwanWRAWaterLevelCollector",
    "TaiwanWRAReservoirCollector",
    "USGSCollector",
    "SDG6Collector",
]
