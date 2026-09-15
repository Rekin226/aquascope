"""Station catalog schema.

A ``Station`` is one monitoring location as advertised by a source's own
catalog (a gauge, a well, a reservoir, a climate station). It carries only
what every source can answer: where it is, what it measures, and how to get
back to the agency page. Observations themselves keep using the reading
models in ``aquascope.schemas.water_data``.

The controlled vocabulary for ``variables`` lives in ``VARIABLES`` and is
shared with ``aquascope.registry.SourceMeta.variables`` so a station catalog,
a source description, and a harvest partition all say the same word for the
same thing.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

# One word per physical quantity. Keep this short and stable: it names
# parquet partitions in the archive and filter values in the CLI.
VARIABLES: tuple[str, ...] = (
    "discharge",
    "water_level",
    "precipitation",
    "groundwater_level",
    "reservoir_storage",
    "water_quality",
    "evapotranspiration",
    "climate",
    "indicator",
)


class Station(BaseModel):
    """One monitoring location from a source's station catalog."""

    source: str = Field(..., description="Registry source key, e.g. 'usgs', 'uk_ea'")
    station_id: str = Field(..., description="The agency's own identifier for the location")
    site_id: str = Field(default="", description="Identifies a physical site within a source")
    name: str | None = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    variables: tuple[str, ...] = Field(default=(), description="Subset of VARIABLES measured here")
    period_start: date | None = None
    period_end: date | None = None
    url: str | None = Field(default=None, description="Deep link to the agency page for this station")
    river: str | None = None
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-3")
    extra: dict = Field(default_factory=dict, description="Source-specific attributes worth keeping")

    @model_validator(mode="after")
    def _default_site_id(self) -> Station:
        if not self.site_id:
            self.site_id = self.station_id
        return self

    @field_validator("variables")
    @classmethod
    def _known_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [v for v in value if v not in VARIABLES]
        if unknown:
            raise ValueError(f"Unknown variable(s) {unknown}; allowed: {list(VARIABLES)}")
        return tuple(value)


def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float] | None) -> bool:
    """True when (lat, lon) falls inside ``bbox`` = (west, south, east, north), or bbox is None."""
    if bbox is None:
        return True
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east
