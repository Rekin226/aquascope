# CAMELS Benchmark Data

Synthetic daily streamflow series for 10 well-known USGS catchments, generated
to approximate published CAMELS catchment statistics.

## Purpose

These files serve as **regression tests** for AquaScope's hydrological
computation modules (`aquascope.hydrology.signatures`, `aquascope.hydrology.baseflow`,
`aquascope.hydrology.flow_duration`).  They are **not** real observational data
and must not be used for scientific analysis.

## Contents

| File | Description |
|------|-------------|
| `catchments.json` | Published attributes for 10 CAMELS catchments |
| `generate_synthetic.py` | Script that creates the CSV files below |
| `<gauge_id>.csv` | Synthetic daily discharge + precipitation (2000–2009) |

## Catchments

The 10 catchments span diverse US hydroclimates:

| Gauge ID | Name | Climate |
|----------|------|---------|
| 01013500 | Fish River near Fort Kent, ME | Snow-dominated humid |
| 01664000 | Rappahannock River near Fredericksburg, VA | Humid continental |
| 02231000 | St. Marys River near Macclenny, FL | Subtropical |
| 03451500 | French Broad River at Asheville, NC | Humid Appalachian |
| 06803500 | Salt Creek at Roca, NE | Semi-arid Great Plains |
| 07056000 | Buffalo River near St. Joe, AR | Humid interior |
| 08181500 | Medina River at San Antonio, TX | Semi-arid |
| 09510200 | Cave Creek near Cave Creek, AZ | Arid |
| 11532500 | Smith River near Crescent City, CA | Pacific Northwest rain |
| 14301000 | Nehalem River near Foss, OR | Pacific Northwest rain |

## Regeneration

```bash
python data/camels_benchmark/generate_synthetic.py
```

The script uses `np.random.default_rng(42)` so results are fully reproducible.

## References

- Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017).
  The CAMELS data set: catchment attributes and meteorology for
  large-sample studies. *Hydrol. Earth Syst. Sci.*, 21, 5293–5313.
  doi:[10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017)
- Newman, A. J., Clark, M. P., Sampson, K., et al. (2015).
  Development of a large-sample watershed-scale hydrometeorological
  data set for the contiguous USA. *Hydrol. Earth Syst. Sci.*, 19, 209–223.
  doi:[10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015)
