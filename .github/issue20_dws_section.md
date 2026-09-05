## South Africa DWS Verified Hydrology

- **Source type:** `south_africa_dws`
- **Coverage:** South African river gauges — verified daily mean discharge and point water level
- **Collector:** `aquascope.collectors.south_africa_dws.SouthAfricaDWSCollector`
- **Authentication:** None

DWS stations are addressed by the agency gauge code, for example `C1H001`.
The collector calls the deterministic `Verified/HyData.aspx` interface and
normalises daily mean discharge (`D_AVG_FR`) to `StreamflowReading` and point
water level (`COR_LEVEL`) to `WaterLevelReading`.

```python
from aquascope.collectors import SouthAfricaDWSCollector

collector = SouthAfricaDWSCollector()
flow = collector.collect(
    station_id="C1H001",
    variable="discharge",
    start_date="2026-01-20",
    end_date="2026-01-21",
)
levels = collector.collect(
    station_id="C1H001",
    variable="water_level",
    days=7,
)
```

From the CLI:

```bash
aquascope collect --source south_africa_dws --station C1H001 \
  --variable discharge --start-date 2026-01-20 --end-date 2026-01-21

aquascope collect --source south_africa_dws --station C1H001 \
  --variable water_level --days 7
```

**Provider availability boundary:** DWS can return an application-level
Kisters `ScriptServerODBC` failure inside an HTTP 200 response. The collector
checks the body and raises a clear `RuntimeError`; it never interprets that
error page as valid hydrological data. Caching is therefore disabled by
default for this collector. A successful live collection still depends on the
DWS/Kisters backend being healthy.

DWS reuse terms have not yet been verified, so the registry marks this source
as non-redistributable. AquaScope can collect it directly, but the open archive
must not mirror the observations until the terms are established.

