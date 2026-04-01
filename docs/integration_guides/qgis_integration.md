# QGIS Integration Guide

Export AquaScope station data to GeoJSON and load it in
[QGIS](https://qgis.org/) for spatial visualisation and analysis.

## Prerequisites

```bash
pip install aquascope geojson
```

QGIS 3.28+ is required on the desktop side.

## Export Stations to GeoJSON

```python
import json
from aquascope.collectors import TaiwanMOENVCollector

collector = TaiwanMOENVCollector(api_key="YOUR_KEY")
records = collector.collect()

features = []
for r in records:
    if r.latitude and r.longitude:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r.longitude, r.latitude],
            },
            "properties": {
                "station_id": r.station_id,
                "station_name": r.station_name,
                "timestamp": r.timestamp.isoformat(),
                "source": r.source.value,
                "parameters": {
                    k: v for k, v in r.parameters.items()
                } if hasattr(r, "parameters") else {},
            },
        })

geojson = {
    "type": "FeatureCollection",
    "features": features,
}
with open("aquascope_stations.geojson", "w") as f:
    json.dump(geojson, f, indent=2)

print(f"Exported {len(features)} stations")
```

## Loading in QGIS

1. Open QGIS and go to **Layer → Add Layer → Add Vector Layer**.
2. Select `aquascope_stations.geojson` as the source.
3. Click **Add** — stations appear as point features on the map.

## Styling by Water Quality

1. Right-click the layer → **Properties → Symbology**.
2. Choose **Graduated** classification.
3. Select a parameter field (e.g., `DO`) as the value column.
4. Pick a colour ramp (e.g., RdYlGn) and click **Classify**.
5. Stations are now colour-coded by dissolved oxygen level.

## Combining with Basemaps

```
# In the QGIS Python console:
from qgis.core import QgsRasterLayer

url = ("type=xyz&url=https://tile.openstreetmap.org/"
       "{z}/{x}/{y}.png")
basemap = QgsRasterLayer(url, "OpenStreetMap", "wms")
QgsProject.instance().addMapLayer(basemap)
```

## Batch Export — Multiple Sources

```python
from aquascope.collectors import USGSCollector, GEMStatCollector

for CollectorClass in [USGSCollector, GEMStatCollector]:
    collector = CollectorClass()
    records = collector.collect()
    # ... same GeoJSON export logic as above
```

## Tips

- Use QGIS **Processing → Buffer** to create catchment areas
  around monitoring stations.
- Enable the **TimeManager** plugin to animate water quality
  changes over time.
- Export styled maps to PDF with **Project → New Print Layout**.
- For large datasets, consider GeoPackage (`.gpkg`) format
  instead of GeoJSON for better performance.
