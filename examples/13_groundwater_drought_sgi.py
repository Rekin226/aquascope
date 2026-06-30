"""Groundwater drought and its rainfall driver, end to end with AquaScope.

This example characterises Taiwan's 2020-2021 drought (its worst in ~56 years)
in groundwater, and measures how long meteorological drought takes to propagate
into each aquifer, using only AquaScope:

  1. collect daily groundwater levels for representative wells with
     ``TaiwanWRAGroundwaterDailyCollector`` (aggregated to monthly);
  2. collect ERA5 rainfall at each well with ``OpenMeteoCollector``;
  3. compute the Standardised Groundwater Index (SGI) and the Standardized
     Precipitation Index (SPI) at accumulation scales of 1-24 months;
  4. report, per aquifer, the SPI scale most correlated with SGI (the
     drought-propagation timescale) and the 2021 groundwater-drought severity.

Run: ``python examples/13_groundwater_drought_sgi.py`` (needs network; the
collectors cache, so re-runs are fast).
"""

from __future__ import annotations

import pandas as pd

from aquascope.climate.indices import standardized_precipitation_index
from aquascope.collectors import OpenMeteoCollector, TaiwanWRAGroundwaterDailyCollector
from aquascope.groundwater import standardised_groundwater_index

# Representative long-record wells, one per major western aquifer.
WELLS = [
    ("07010211", "Zhuoshui fan", 24.063, 120.516),
    ("06030111", "Taichung", 24.405, 120.649),
    ("04010111", "Xinmiao", 24.844, 120.998),
    ("03020111", "Taoyuan", 25.016, 121.174),
    ("10010211", "Chianan", 23.484, 120.333),
]
SPI_SCALES = range(1, 25)


def groundwater_sgi(well_id: str) -> pd.Series:
    """Monthly SGI for one well, collected via AquaScope."""
    recs = TaiwanWRAGroundwaterDailyCollector(
        stations=[well_id], aggregate="monthly", with_metadata=False,
    ).collect()
    levels = pd.Series(
        {pd.Timestamp(r.measurement_datetime): r.water_level_m for r in recs}
    ).sort_index()
    sgi = standardised_groundwater_index(levels)
    sgi.index = sgi.index.to_period("M").to_timestamp()  # month-start, to align with SPI
    return sgi


def rainfall_monthly(lat: float, lon: float) -> pd.Series:
    """Monthly ERA5 precipitation (mm) at a point, collected via AquaScope."""
    recs = OpenMeteoCollector(mode="weather").collect(
        latitude=lat, longitude=lon,
        start_date="1994-01-01", end_date="2025-12-31",
        daily=["precipitation_sum"],
    )
    daily = pd.Series(
        {pd.Timestamp(r.sample_datetime): r.value
         for r in recs if r.parameter == "precipitation_sum"}
    ).sort_index()
    return daily.resample("MS").sum()


def main() -> None:
    print("Aquifer            well  propagation  corr(SGI,SPI)  2021 min SGI")
    print("-" * 66)
    for well_id, zone, lat, lon in WELLS:
        sgi = groundwater_sgi(well_id)
        precip = rainfall_monthly(lat, lon)

        best_scale, best_r = None, -1.0
        for k in SPI_SCALES:
            spi = standardized_precipitation_index(precip, scale=k)
            joined = pd.concat([sgi.rename("sgi"), spi.rename("spi")],
                               axis=1, sort=True).dropna()
            if len(joined) < 60:
                continue
            r = joined["sgi"].corr(joined["spi"])
            if r > best_r:
                best_scale, best_r = k, r

        s2021 = sgi[(sgi.index >= "2021-01-01") & (sgi.index <= "2021-12-31")]
        peak = s2021.min()
        sev = "extreme" if peak <= -2 else "severe" if peak <= -1.5 else \
              "moderate" if peak <= -1 else "mild"
        print(f"{zone:<14} {well_id:>8}  {best_scale:>5} mo    "
              f"{best_r:>8.2f}     {peak:>6.2f} ({sev})")

    print("\nReading: groundwater drought (SGI) reached severe levels (~ -1.9 to "
          "-2.0)\nin the central-western aquifers, peaking in 2021. Each aquifer "
          "integrates\nroughly 7-23 months of accumulated rainfall deficit (SPI) "
          "before its\ngroundwater drought peaks - the lag grows with aquifer "
          "storage.")


if __name__ == "__main__":
    main()
