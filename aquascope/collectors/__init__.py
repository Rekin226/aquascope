"""Data collectors for Taiwan and global water data sources."""

from aquascope.collectors.ireland_opw import IrelandOPWCollector
from aquascope.collectors.aquastat import AquastatCollector
from aquascope.collectors.base import BaseCollector
from aquascope.collectors.camels_cl import CAMELSCLCollector
from aquascope.collectors.copernicus import CopernicusCollector
from aquascope.collectors.eu_wfd import EUWFDCollector
from aquascope.collectors.france_hubeau import HubeauHydrometrieCollector
from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.collectors.grdc import GRDCCollector
from aquascope.collectors.india_wris import IndiaWRISCollector
from aquascope.collectors.japan_mlit import JapanMLITCollector
from aquascope.collectors.korea_wamis import KoreaWAMISCollector
from aquascope.collectors.openmeteo import OpenMeteoCollector
from aquascope.collectors.sdg6 import SDG6Collector
from aquascope.collectors.taiwan_civil_iot import TaiwanCivilIoTCollector
from aquascope.collectors.taiwan_datagov import TaiwanDataGovCollector
from aquascope.collectors.taiwan_moenv import TaiwanMOENVCollector
from aquascope.collectors.taiwan_wra import (
    TaiwanWRAGroundwaterCollector,
    TaiwanWRAGroundwaterDailyCollector,
    TaiwanWRAReservoirCollector,
    TaiwanWRAWaterLevelCollector,
)
from aquascope.collectors.taiwan_wra_fhy import TaiwanWRAFhyCollector
from aquascope.collectors.taiwan_wra_iot import TaiwanWRAIoTCollector
from aquascope.collectors.usgs import USGSCollector
from aquascope.collectors.wapor import WaPORCollector
from aquascope.collectors.wqp import WQPCollector

__all__ = [
    "AquastatCollector",
    "BaseCollector",
    "CAMELSCLCollector",
    "CopernicusCollector",
    "EUWFDCollector",
    "GEMStatCollector",
    "GRDCCollector",
    "HubeauHydrometrieCollector",
    "IndiaWRISCollector",
    "JapanMLITCollector",
    "KoreaWAMISCollector",
    "OpenMeteoCollector",
    "SDG6Collector",
    "TaiwanCivilIoTCollector",
    "TaiwanDataGovCollector",
    "TaiwanMOENVCollector",
    "TaiwanWRAFhyCollector",
    "TaiwanWRAGroundwaterCollector",
    "TaiwanWRAGroundwaterDailyCollector",
    "TaiwanWRAIoTCollector",
    "TaiwanWRAReservoirCollector",
    "TaiwanWRAWaterLevelCollector",
    "USGSCollector",
    "WaPORCollector",
    "WQPCollector",
    "IrelandOPWCollector",
]
