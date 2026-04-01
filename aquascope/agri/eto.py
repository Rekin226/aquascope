"""FAO-56 Penman-Monteith reference evapotranspiration.

Implements the standardized FAO Penman-Monteith equation for computing
reference evapotranspiration (ET₀) from weather data.

The equation (Allen et al., 1998, Eq. 6)::

    ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (eₛ - eₐ)]
          / [Δ + γ(1 + 0.34 u₂)]

where:
    Rn = net radiation (MJ/m²/day)
    G  = soil heat flux (MJ/m²/day), ≈ 0 for daily
    T  = mean temperature (°C)
    u₂ = wind speed at 2m (m/s)
    eₛ = saturation vapour pressure (kPa)
    eₐ = actual vapour pressure (kPa)
    Δ  = slope of vapour pressure curve (kPa/°C)
    γ  = psychrometric constant (kPa/°C)

References
----------
Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
    Crop evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56. Rome: FAO.
    ISBN 92-5-104219-5
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions – each corresponds to a numbered equation in FAO-56
# ---------------------------------------------------------------------------


def saturation_vapor_pressure(t: float) -> float:
    """Saturation vapour pressure at temperature *t* (°C).

    FAO-56 Eq. 11::

        e°(T) = 0.6108 × exp(17.27 × T / (T + 237.3))

    Parameters
    ----------
    t : float
        Air temperature in °C.

    Returns
    -------
    float
        Saturation vapour pressure in kPa.

    References
    ----------
    Allen et al. (1998), Eq. 11. ISBN 92-5-104219-5.
    """
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))


def slope_vapor_pressure_curve(t: float) -> float:
    """Slope of the saturation vapour pressure curve at temperature *t*.

    FAO-56 Eq. 13::

        Δ = 4098 × e°(T) / (T + 237.3)²

    Parameters
    ----------
    t : float
        Air temperature in °C.

    Returns
    -------
    float
        Slope Δ in kPa/°C.

    References
    ----------
    Allen et al. (1998), Eq. 13. ISBN 92-5-104219-5.
    """
    es = saturation_vapor_pressure(t)
    return 4098.0 * es / (t + 237.3) ** 2


def atmospheric_pressure(elevation: float) -> float:
    """Atmospheric pressure at a given elevation.

    FAO-56 Eq. 7::

        P = 101.3 × ((293 − 0.0065 × z) / 293) ^ 5.26

    Parameters
    ----------
    elevation : float
        Elevation above sea level in metres.

    Returns
    -------
    float
        Atmospheric pressure in kPa.

    References
    ----------
    Allen et al. (1998), Eq. 7. ISBN 92-5-104219-5.
    """
    return 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26


def psychrometric_constant(elevation: float) -> float:
    """Psychrometric constant at a given elevation.

    FAO-56 Eq. 8::

        γ = 0.665 × 10⁻³ × P

    Parameters
    ----------
    elevation : float
        Elevation above sea level in metres.

    Returns
    -------
    float
        Psychrometric constant γ in kPa/°C.

    References
    ----------
    Allen et al. (1998), Eq. 8. ISBN 92-5-104219-5.
    """
    p = atmospheric_pressure(elevation)
    return 0.665e-3 * p


def extraterrestrial_radiation(latitude: float, doy: int) -> float:
    """Extraterrestrial radiation for a given latitude and day of year.

    FAO-56 Eq. 21, using Eqs. 23–25 for solar geometry.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees (negative for Southern Hemisphere).
    doy : int
        Day of the year (1–366).

    Returns
    -------
    float
        Extraterrestrial radiation Ra in MJ/m²/day.

    References
    ----------
    Allen et al. (1998), Eqs. 21–25. ISBN 92-5-104219-5.
    """
    phi = math.radians(latitude)
    # Eq. 24: solar declination
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)  # Eq. 23
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)  # Eq. 24
    # Eq. 25: sunset hour angle
    ws = math.acos(-math.tan(phi) * math.tan(delta))  # Eq. 25
    # Eq. 21
    gsc = 0.0820  # solar constant MJ/m²/min
    ra = (
        (24.0 * 60.0 / math.pi)
        * gsc
        * dr
        * (ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws))
    )
    return ra


def clear_sky_radiation(ra: float, elevation: float) -> float:
    """Clear-sky solar radiation.

    FAO-56 Eq. 37::

        Rso = (0.75 + 2 × 10⁻⁵ × z) × Ra

    Parameters
    ----------
    ra : float
        Extraterrestrial radiation in MJ/m²/day.
    elevation : float
        Elevation above sea level in metres.

    Returns
    -------
    float
        Clear-sky radiation Rso in MJ/m²/day.

    References
    ----------
    Allen et al. (1998), Eq. 37. ISBN 92-5-104219-5.
    """
    return (0.75 + 2.0e-5 * elevation) * ra


def net_shortwave_radiation(rs: float, albedo: float = 0.23) -> float:
    """Net shortwave radiation.

    FAO-56 Eq. 38::

        Rns = (1 − α) × Rs

    Parameters
    ----------
    rs : float
        Incoming solar radiation in MJ/m²/day.
    albedo : float
        Albedo coefficient (default 0.23 for reference grass).

    Returns
    -------
    float
        Net shortwave radiation Rns in MJ/m²/day.

    References
    ----------
    Allen et al. (1998), Eq. 38. ISBN 92-5-104219-5.
    """
    return (1.0 - albedo) * rs


def net_longwave_radiation(
    t_min: float,
    t_max: float,
    ea: float,
    rs: float,
    rso: float,
) -> float:
    """Net outgoing longwave radiation.

    FAO-56 Eq. 39 (Stefan-Boltzmann)::

        Rnl = σ [(T_max_K⁴ + T_min_K⁴)/2] (0.34 − 0.14√eₐ) (1.35 Rs/Rso − 0.35)

    Parameters
    ----------
    t_min : float
        Minimum daily temperature in °C.
    t_max : float
        Maximum daily temperature in °C.
    ea : float
        Actual vapour pressure in kPa.
    rs : float
        Measured solar radiation in MJ/m²/day.
    rso : float
        Clear-sky solar radiation in MJ/m²/day.

    Returns
    -------
    float
        Net longwave radiation Rnl in MJ/m²/day.

    References
    ----------
    Allen et al. (1998), Eq. 39. ISBN 92-5-104219-5.
    """
    sigma = 4.903e-9  # Stefan-Boltzmann constant MJ/K⁴/m²/day
    t_min_k = t_min + 273.16
    t_max_k = t_max + 273.16
    # Avoid division by zero when Rso is 0
    rs_rso_ratio = min(rs / rso, 1.0) if rso > 0 else 0.5
    rnl = (
        sigma
        * ((t_max_k**4 + t_min_k**4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(ea))
        * (1.35 * rs_rso_ratio - 0.35)
    )
    return rnl


def net_radiation(
    rs: float,
    rso: float,
    t_min: float,
    t_max: float,
    ea: float,
    elevation: float,
    albedo: float = 0.23,
) -> float:
    """Net radiation at the crop surface.

    FAO-56 Eq. 40::

        Rn = Rns − Rnl

    Parameters
    ----------
    rs : float
        Incoming solar radiation in MJ/m²/day.
    rso : float
        Clear-sky radiation in MJ/m²/day.
    t_min : float
        Minimum daily temperature in °C.
    t_max : float
        Maximum daily temperature in °C.
    ea : float
        Actual vapour pressure in kPa.
    elevation : float
        Elevation in metres (unused here but kept for API consistency).
    albedo : float
        Albedo coefficient (default 0.23).

    Returns
    -------
    float
        Net radiation Rn in MJ/m²/day.

    References
    ----------
    Allen et al. (1998), Eq. 40. ISBN 92-5-104219-5.
    """
    rns = net_shortwave_radiation(rs, albedo)
    rnl = net_longwave_radiation(t_min, t_max, ea, rs, rso)
    return rns - rnl


# ---------------------------------------------------------------------------
# Main ET₀ functions
# ---------------------------------------------------------------------------


def penman_monteith_daily(
    t_min: float,
    t_max: float,
    rh_min: float,
    rh_max: float,
    u2: float,
    rs: float,
    latitude: float,
    elevation: float,
    doy: int,
) -> float:
    """FAO-56 Penman-Monteith daily reference evapotranspiration.

    Parameters
    ----------
    t_min : float
        Minimum daily temperature (°C).
    t_max : float
        Maximum daily temperature (°C).
    rh_min : float
        Minimum relative humidity (%).
    rh_max : float
        Maximum relative humidity (%).
    u2 : float
        Wind speed at 2 m height (m/s).
    rs : float
        Incoming solar radiation (MJ/m²/day).
    latitude : float
        Latitude in decimal degrees.
    elevation : float
        Station elevation above sea level (m).
    doy : int
        Day of the year (1–366).

    Returns
    -------
    float
        Reference evapotranspiration ET₀ in mm/day.

    References
    ----------
    Allen et al. (1998), Eq. 6. ISBN 92-5-104219-5.
    """
    t_mean = (t_min + t_max) / 2.0

    # Vapour pressures
    es_min = saturation_vapor_pressure(t_min)
    es_max = saturation_vapor_pressure(t_max)
    es = (es_min + es_max) / 2.0  # Eq. 12
    ea = (es_min * rh_max / 100.0 + es_max * rh_min / 100.0) / 2.0  # Eq. 17

    # Slope of saturation vapour pressure curve
    delta = slope_vapor_pressure_curve(t_mean)

    # Psychrometric constant
    gamma = psychrometric_constant(elevation)

    # Radiation terms
    ra = extraterrestrial_radiation(latitude, doy)
    rso = clear_sky_radiation(ra, elevation)
    rn = net_radiation(rs, rso, t_min, t_max, ea, elevation)

    # Soil heat flux ≈ 0 for daily time step
    g = 0.0

    # FAO-56 Eq. 6
    numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (t_mean + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    eto = numerator / denominator

    return max(eto, 0.0)


def hargreaves(t_min: float, t_max: float, ra: float) -> float:
    """Hargreaves reference ET₀ estimate (temperature-based).

    A simpler alternative when only temperature data is available::

        ET₀ = 0.0023 × (T_mean + 17.8) × (T_max − T_min)^0.5 × Ra

    Parameters
    ----------
    t_min : float
        Minimum daily temperature (°C).
    t_max : float
        Maximum daily temperature (°C).
    ra : float
        Extraterrestrial radiation (MJ/m²/day).  Use ``extraterrestrial_radiation``
        to compute this.

    Returns
    -------
    float
        Reference evapotranspiration ET₀ in mm/day.

    References
    ----------
    Hargreaves, G. H. & Samani, Z. A. (1985). Reference crop
    evapotranspiration from temperature. *Applied Engineering in
    Agriculture*, 1(2), 96–99.
    """
    t_mean = (t_min + t_max) / 2.0
    td = t_max - t_min
    if td < 0:
        td = 0.0
    # Ra is in MJ/m²/day; multiply by 0.408 to convert to mm/day equivalent
    return 0.0023 * (t_mean + 17.8) * math.sqrt(td) * ra * 0.408


def penman_monteith_series(
    df: pandas.DataFrame,
    latitude: float,
    elevation: float,
) -> pandas.Series:
    """Apply Penman-Monteith ET₀ to each row of a weather DataFrame.

    The DataFrame must have a ``DatetimeIndex`` and the following columns:
    ``t_min``, ``t_max``, ``rh_min``, ``rh_max``, ``wind_speed``,
    ``solar_radiation``.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily weather observations.
    latitude : float
        Latitude in decimal degrees.
    elevation : float
        Station elevation in metres.

    Returns
    -------
    pandas.Series
        Daily ET₀ values in mm/day, indexed like *df*.

    References
    ----------
    Allen et al. (1998), FAO Irrigation and Drainage Paper 56.
    ISBN 92-5-104219-5.
    """
    import pandas as pd  # noqa: F811

    results: list[float] = []
    for idx, row in df.iterrows():
        doy = idx.timetuple().tm_yday  # type: ignore[union-attr]
        eto = penman_monteith_daily(
            t_min=row["t_min"],
            t_max=row["t_max"],
            rh_min=row["rh_min"],
            rh_max=row["rh_max"],
            u2=row["wind_speed"],
            rs=row["solar_radiation"],
            latitude=latitude,
            elevation=elevation,
            doy=doy,
        )
        results.append(eto)
    return pd.Series(results, index=df.index, name="eto")
