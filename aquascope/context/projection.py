"""Interrupted Goode Homolosine (ESRI:54052), the grid SoilGrids is published on.

A forward projection only, spherical as in PROJ's ``igh``: sinusoidal lobes
between about 40.7 degrees north and south, Mollweide lobes poleward of that,
two lobes in the north and four in the south. Pure Python so the Explorer's
worker can project a click without pyproj.
"""

from __future__ import annotations

import math

__all__ = ["homolosine"]

R = 6_378_137.0
_D = math.radians
#: The latitude where the sinusoidal and Mollweide lobes meet: 40 deg 44' 11.8".
PHI_BOUNDARY = _D(40 + 44 / 60 + 11.8 / 3600)

_SQRT2 = math.sqrt(2.0)


def _sinu(lam: float, phi: float) -> tuple[float, float]:
    return lam * math.cos(phi), phi


def _moll(lam: float, phi: float) -> tuple[float, float]:
    k = math.pi * math.sin(phi)
    theta = phi
    for _ in range(30):
        v = (theta + math.sin(theta) - k) / (1.0 + math.cos(theta))
        theta -= v
        if abs(v) < 1e-12:
            break
    theta /= 2.0
    if abs(abs(phi) - math.pi / 2) < 1e-12:
        theta = math.copysign(math.pi / 2, phi)
    return 2.0 * _SQRT2 / math.pi * lam * math.cos(theta), _SQRT2 * math.sin(theta)


# y offset of the Mollweide lobes, so they meet the sinusoidal ones at the boundary latitude
_DY0 = _sinu(0.0, PHI_BOUNDARY)[1] - _moll(0.0, PHI_BOUNDARY)[1]


def _zone(lam: float, phi: float) -> tuple[str, float, float]:
    """(projection, central meridian, y offset) of the lobe holding (lam, phi), radians."""
    if phi >= PHI_BOUNDARY:
        return ("moll", _D(-100), _DY0) if lam <= _D(-40) else ("moll", _D(30), _DY0)
    if phi >= 0:
        return ("sinu", _D(-100), 0.0) if lam <= _D(-40) else ("sinu", _D(30), 0.0)
    kind, dy = ("sinu", 0.0) if phi >= -PHI_BOUNDARY else ("moll", -_DY0)
    if lam <= _D(-100):
        return kind, _D(-160), dy
    if lam <= _D(-20):
        return kind, _D(-60), dy
    if lam <= _D(80):
        return kind, _D(20), dy
    return kind, _D(140), dy


def homolosine(lon: float, lat: float) -> tuple[float, float]:
    """(x, y) in metres on the Interrupted Goode Homolosine grid for a WGS84 longitude and latitude."""
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude {lat} is outside -90..90")
    lon = (lon + 180.0) % 360.0 - 180.0
    lam, phi = _D(lon), _D(lat)
    kind, lam0, y0 = _zone(lam, phi)
    x, y = (_sinu if kind == "sinu" else _moll)(lam - lam0, phi)
    return (x + lam0) * R, (y + y0) * R
