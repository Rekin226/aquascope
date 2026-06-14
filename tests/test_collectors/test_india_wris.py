from aquascope.collectors.india_wris import IndiaWRISCollector
from aquascope.schemas.water_data import DataSource


def test_normalise():
    raw = [
        {
            "stationCode": "Asga",
            "stationName": "Asga",
            "latitude": 16.9139,
            "longitude": 73.6,
            "dataValue": 0.62,
            "dataTime": "2017-06-01T08:30:00",
            "unit": "m",
            "description": "MANUAL-Water Level by Staff Gauge (0)",
        }
    ]

    collector = IndiaWRISCollector()
    records = collector.normalise(raw)

    assert len(records) == 1

    record = records[0]

    assert record.source == DataSource.INDIA_WRIS
    assert record.station_id == "Asga"
    assert record.water_level == 0.62
    assert record.unit == "m"
