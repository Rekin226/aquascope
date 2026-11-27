"""Daily CemaNeige snow accounting in equal-area elevation bands (Valéry et al., 2014).

The classic two-parameter formulation tracks the water equivalent and thermal state
of the snowpack. The basin-average rain plus melt can be passed directly to GR4J.
Reference: https://doi.org/10.1016/j.jhydrol.2014.04.058
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CemaNeigeResult:
    """Daily basin rain plus melt (mm), and end-of-day diagnostics for each band.

    Band columns are median elevations in metres. Snowpack, snowfall and melt are
    in mm water equivalent over the band's area. Thermal state is in degrees C.
    """

    liquid: pd.Series
    snowpack: pd.DataFrame
    thermal_state: pd.DataFrame
    snowfall: pd.DataFrame
    melt: pd.DataFrame


class CemaNeige:
    """Prepare a fixed set of equal-area bands for repeated daily snow simulations.

    Supply consecutive daily basin precipitation (mm) and mean air temperature
    (degrees C), with matching DatetimeIndexes. Supply the median elevation of
    each band (m) and the forcing elevation (m) from the same vertical datum.
    The default is one band at the forcing elevation.

    Temperature decreases by `lapse_rate` degrees C per km (default 6.5, a
    configurable constant). Precipitation increases exponentially with elevation
    (`precip_gradient`, default 0.0004 per metre, capped at 4000 m), rescaled
    to conserve the basin input each day. Set the gradient to zero to use the
    same precipitation in every band.
    The mean-temperature USACE partition is all snow at or below -1 C and all
    rain at or above 3 C, interpolated in between. Band temperatures or snowfall
    generated with a different lapse rate can give different flows.

    `mean_annual_snowfall` is the long-term basin mean solid precipitation in
    mm/year. When omitted, it is estimated as 365.25 times the record's daily
    mean snowfall. Supply the calibration-period value for independent validation.
    """

    def __init__(
        self,
        precip: pd.Series,
        temperature: pd.Series,
        *,
        band_elevations: Sequence[float] | None = None,
        forcing_elevation: float = 0.0,
        lapse_rate: float = 6.5,
        precip_gradient: float = 0.0004,
        mean_annual_snowfall: float | None = None,
    ) -> None:
        index = precip.index
        if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
                or not index.equals(temperature.index)
                or not (np.diff(index.tz_localize(None).to_period("D").asi8) == 1).all()):
            raise ValueError("Precipitation and temperature need the same consecutive daily DatetimeIndex.")
        p = precip.to_numpy(dtype=float)
        temp = temperature.to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(temp).all():
            raise ValueError("Daily precipitation and temperature must be finite, with non-negative precipitation.")
        elevations = np.asarray([forcing_elevation] if band_elevations is None else band_elevations, dtype=float)
        if (elevations.ndim != 1 or elevations.size == 0 or not np.isfinite(elevations).all()
                or not np.isfinite(forcing_elevation) or not np.isfinite(lapse_rate) or lapse_rate < 0):
            raise ValueError("Supply finite band and forcing elevations and a non-negative temperature lapse rate.")
        if (np.diff(elevations) < 0).any():
            raise ValueError("Supply band elevations from low to high.")
        if not np.isfinite(precip_gradient) or precip_gradient < 0:
            raise ValueError("The precipitation gradient must be finite and non-negative.")
        # The precipitation gradient saturates at 4000 m. Scaling the weights to
        # mean one preserves the observed basin precipitation, including dry days.
        z = np.minimum(elevations, 4000.0)
        weights = np.exp(precip_gradient * (z - z.max()))
        band_precip = p[:, None] * (weights / weights.mean())
        self._temperature = temp[:, None] - (elevations - forcing_elevation) * lapse_rate / 1000.0
        solid_fraction = np.clip((3.0 - self._temperature) / 4.0, 0.0, 1.0)
        self._snowfall = band_precip * solid_fraction
        self._rain = (band_precip * (1.0 - solid_fraction)).mean(axis=1)
        # Preserve the exact input where every band receives only rain.
        rain_only = ~self._snowfall.any(axis=1)
        self._rain[rain_only] = p[rain_only]
        annual = float(365.25 * self._snowfall.mean()) if mean_annual_snowfall is None else mean_annual_snowfall
        if not np.isfinite(annual) or annual < 0:
            raise ValueError("Mean annual snowfall must be finite and non-negative.")
        self.mean_annual_snowfall = float(annual)
        self.index = index
        self.band_elevations = pd.Index(elevations, name="elevation_m")

    def simulate(self, ctg: float = 0.5, kf: float = 3.0) -> CemaNeigeResult:
        """Run Valéry et al. (2014) from an empty pack and a zero initial thermal state.

        CTG (0 to 1) is the weight of the previous day's thermal state; Kf is the
        melt factor in mm per degree C per day. Snow can melt only once its thermal
        state reaches zero. Fractional snow cover limits actual melt.
        """
        if not np.isfinite(ctg) or not 0 <= ctg <= 1 or not np.isfinite(kf) or kf < 0:
            raise ValueError("CTG must be between 0 and 1 and Kf must be finite and non-negative.")
        snowpack = np.empty_like(self._snowfall)
        thermal = np.empty_like(self._snowfall)
        melt = np.empty_like(self._snowfall)
        full_cover = 0.9 * self.mean_annual_snowfall
        for j in range(len(self.band_elevations)):
            pack, heat = 0.0, 0.0
            for i, (snow, temp) in enumerate(zip(self._snowfall[:, j], self._temperature[:, j])):
                pack += snow
                heat = min(0.0, ctg * heat + (1.0 - ctg) * temp)
                cover = min(1.0, pack / full_cover) if full_cover > 0 else 1.0
                potential = min(pack, kf * max(temp, 0.0)) if heat == 0 else 0.0
                melt[i, j] = potential * (0.1 + 0.9 * cover)
                pack -= melt[i, j]
                snowpack[i, j] = pack
                thermal[i, j] = heat

        def bands(values: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(values, index=self.index, columns=self.band_elevations)

        liquid = pd.Series(self._rain + melt.mean(axis=1), index=self.index, name="liquid_precip")
        return CemaNeigeResult(liquid, bands(snowpack), bands(thermal), bands(self._snowfall.copy()), bands(melt))
