"""Data collectors for Taiwan and global water data sources."""

from aquascope.collectors.aquastat import AquastatCollector
from aquascope.collectors.base import BaseCollector
from aquascope.collectors.copernicus import CopernicusCollector
from aquascope.collectors.eu_wfd import EUWFDCollector
from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.collectors.japan_mlit import JapanMLITCollector
from aquascope.collectors.korea_wamis import KoreaWAMISCollector
from aquascope.collectors.openmeteo import OpenMeteoCollector
from aquascope.collectors.sdg6 import SDG6Collector
from aquascope.collectors.taiwan_civil_iot import TaiwanCivilIoTCollector
from aquascope.collectors.taiwan_moenv import TaiwanMOENVCollector
from aquascope.collectors.taiwan_wra import (
    TaiwanWRAReservoirCollector,
    TaiwanWRAWaterLevelCollector,
)
from aquascope.collectors.usgs import USGSCollector
from aquascope.collectors.wapor import WaPORCollector
from aquascope.collectors.wqp import WQPCollector

__all__ = [
    "AquastatCollector",
    "BaseCollector",
    "CopernicusCollector",
    "EUWFDCollector",
    "GEMStatCollector",
    "JapanMLITCollector",
    "KoreaWAMISCollector",
    "OpenMeteoCollector",
    "SDG6Collector",
    "TaiwanCivilIoTCollector",
    "TaiwanMOENVCollector",
    "TaiwanWRAReservoirCollector",
    "TaiwanWRAWaterLevelCollector",
    "USGSCollector",
    "WaPORCollector",
    "WQPCollector",
]
