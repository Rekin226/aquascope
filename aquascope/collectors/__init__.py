"""Data collectors for Taiwan and global water data sources."""

from aquascope.collectors.base import BaseCollector
from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.collectors.sdg6 import SDG6Collector
from aquascope.collectors.taiwan_civil_iot import TaiwanCivilIoTCollector
from aquascope.collectors.taiwan_moenv import TaiwanMOENVCollector
from aquascope.collectors.taiwan_wra import (
    TaiwanWRAReservoirCollector,
    TaiwanWRAWaterLevelCollector,
)
from aquascope.collectors.usgs import USGSCollector
from aquascope.collectors.wqp import WQPCollector

__all__ = [
    "BaseCollector",
    "GEMStatCollector",
    "SDG6Collector",
    "TaiwanCivilIoTCollector",
    "TaiwanMOENVCollector",
    "TaiwanWRAReservoirCollector",
    "TaiwanWRAWaterLevelCollector",
    "USGSCollector",
    "WQPCollector",
]
