"""Daily soil water balance model.

Implements the FAO-56 single crop coefficient approach for
tracking soil moisture depletion and irrigation scheduling.

The water balance equation::

    Dr,i = Dr,i-1 - (P - RO)i - Ii - CRi + ETc,i + DPi

where:
    Dr = root zone depletion (mm)
    P  = precipitation
    RO = runoff
    I  = irrigation
    CR = capillary rise (assumed 0)
    ETc = crop evapotranspiration
    DP = deep percolation

References
----------
Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
    Crop evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56, Chapter 8. Rome: FAO.
    ISBN 92-5-104219-5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from aquascope.schemas.agriculture import SoilWaterStatus

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SoilProperties:
    """Soil hydraulic properties.

    Parameters
    ----------
    field_capacity : float
        Volumetric water content at field capacity (m³/m³, e.g. 0.30).
    wilting_point : float
        Volumetric water content at permanent wilting point (m³/m³, e.g. 0.15).
    root_depth : float
        Effective rooting depth in metres (default 1.0).

    References
    ----------
    Allen et al. (1998), FAO-56 Table 19. ISBN 92-5-104219-5.
    """

    field_capacity: float
    wilting_point: float
    root_depth: float = 1.0

    @property
    def total_available_water(self) -> float:
        """Total available water in the root zone.

        FAO-56 Eq. 82::

            TAW = 1000 × (θ_FC − θ_WP) × Zr   (mm)

        Returns
        -------
        float
            TAW in mm.
        """
        return 1000.0 * (self.field_capacity - self.wilting_point) * self.root_depth

    def readily_available_water(self, p: float = 0.5) -> float:
        """Readily available water.

        FAO-56 Eq. 83::

            RAW = p × TAW

        Parameters
        ----------
        p : float
            Depletion fraction for no stress (default 0.5).

        Returns
        -------
        float
            RAW in mm.
        """
        return p * self.total_available_water


class SoilWaterBalance:
    """Daily soil water balance tracker.

    Parameters
    ----------
    soil : SoilProperties
        Soil hydraulic properties.
    depletion_fraction : float
        Fraction of TAW that can be depleted before stress (p, default 0.5).
    initial_depletion : float
        Starting depletion in mm (default 0.0 = field capacity).

    References
    ----------
    Allen et al. (1998), FAO-56 Ch. 8. ISBN 92-5-104219-5.
    """

    def __init__(
        self,
        soil: SoilProperties,
        depletion_fraction: float = 0.5,
        initial_depletion: float = 0.0,
    ) -> None:
        self.soil = soil
        self.p = depletion_fraction
        self.depletion = initial_depletion
        self.taw = soil.total_available_water
        self.raw = soil.readily_available_water(depletion_fraction)

    def step(
        self,
        etc: float,
        precipitation: float = 0.0,
        irrigation: float = 0.0,
        runoff: float = 0.0,
    ) -> SoilWaterStatus:
        """Advance the water balance by one day.

        Parameters
        ----------
        etc : float
            Crop evapotranspiration (mm/day).
        precipitation : float
            Daily precipitation (mm).
        irrigation : float
            Applied irrigation (mm).
        runoff : float
            Surface runoff (mm).

        Returns
        -------
        SoilWaterStatus
            Updated soil water status.
        """
        # Water balance: depletion increases with ETc, decreases with P and I
        self.depletion = self.depletion - (precipitation - runoff) - irrigation + etc

        # Deep percolation occurs when depletion goes negative (above FC)
        deep_percolation = 0.0
        if self.depletion < 0:
            deep_percolation = abs(self.depletion)
            self.depletion = 0.0

        # Cap depletion at TAW (cannot go below wilting point meaningfully
        # in this simplified model)
        self.depletion = min(self.depletion, self.taw)

        soil_moisture = self.taw - self.depletion
        trigger = self.depletion >= self.raw

        return SoilWaterStatus(
            date=date.today(),
            soil_moisture_mm=round(soil_moisture, 2),
            depletion_mm=round(self.depletion, 2),
            deep_percolation_mm=round(deep_percolation, 2),
            runoff_mm=round(runoff, 2),
            irrigation_trigger=trigger,
        )

    def run(
        self,
        etc_series: pd.Series,
        precip_series: pd.Series,
        irrigation_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run the water balance over a time series.

        Parameters
        ----------
        etc_series : pandas.Series
            Daily ETc (mm/day) with ``DatetimeIndex``.
        precip_series : pandas.Series
            Daily precipitation (mm) with ``DatetimeIndex``.
        irrigation_series : pandas.Series | None
            Daily irrigation (mm).  If *None*, no irrigation is applied.

        Returns
        -------
        pandas.DataFrame
            Daily soil water status records.
        """
        import pandas as pd

        rows: list[dict] = []
        for idx in etc_series.index:
            etc_val = float(etc_series.loc[idx])
            precip_val = float(precip_series.loc[idx]) if idx in precip_series.index else 0.0
            irr_val = float(irrigation_series.loc[idx]) if irrigation_series is not None and idx in irrigation_series.index else 0.0

            status = self.step(etc_val, precipitation=precip_val, irrigation=irr_val)
            rows.append({
                "date": idx,
                "soil_moisture_mm": status.soil_moisture_mm,
                "depletion_mm": status.depletion_mm,
                "deep_percolation_mm": status.deep_percolation_mm,
                "irrigation_trigger": status.irrigation_trigger,
            })

        return pd.DataFrame(rows)

    def auto_irrigate(
        self,
        etc_series: pd.Series,
        precip_series: pd.Series,
        efficiency: float = 0.7,
    ) -> pd.DataFrame:
        """Run balance with automatic irrigation when depletion exceeds RAW.

        Parameters
        ----------
        etc_series : pandas.Series
            Daily ETc (mm/day).
        precip_series : pandas.Series
            Daily precipitation (mm).
        efficiency : float
            Irrigation system efficiency (0–1).

        Returns
        -------
        pandas.DataFrame
            Daily water balance with ``irrigation_mm`` column.
        """
        import pandas as pd

        rows: list[dict] = []
        for idx in etc_series.index:
            etc_val = float(etc_series.loc[idx])
            precip_val = float(precip_series.loc[idx]) if idx in precip_series.index else 0.0

            # Auto-irrigate: if depletion would exceed RAW, apply enough water
            irrigation = 0.0
            if self.depletion >= self.raw:
                # Refill to field capacity
                net_needed = self.depletion
                irrigation = net_needed / efficiency if efficiency > 0 else net_needed

            status = self.step(etc_val, precipitation=precip_val, irrigation=irrigation)
            rows.append({
                "date": idx,
                "soil_moisture_mm": status.soil_moisture_mm,
                "depletion_mm": status.depletion_mm,
                "deep_percolation_mm": status.deep_percolation_mm,
                "irrigation_mm": round(irrigation, 2),
                "irrigation_trigger": status.irrigation_trigger,
            })

        return pd.DataFrame(rows)
