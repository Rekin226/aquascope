from datetime import datetime, timezone
import pytest

from aquascope.collectors.ireland_opw import IrelandOPWCollector
from aquascope.schemas.water_data import DataSource


@pytest.fixture
def collector():
    return IrelandOPWCollector()


def test_normalize(collector):
    raw_record = {"datetime": "2026-07-22 09:15:00", "value": 1.234}
    metadata = {
        "geometry": {"coordinates": [-8.5, 53.2]},
        "properties": {"station_name": "River Shannon"}
    }
    
    reading = collector.normalise(raw_record, "25017", metadata)
    
    assert reading.source == DataSource.IRELAND_OPW
    assert reading.station_id == "25017"
    assert reading.water_level == 1.234
    assert reading.station_name == "River Shannon"
    assert reading.location.latitude == 53.2
    assert reading.location.longitude == -8.5
    assert reading.reading_datetime == datetime(2026, 7, 22, 9, 15, tzinfo=timezone.utc)