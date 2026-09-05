from __future__ import annotations

from datetime import datetime

import pytest

from aquascope.collectors.south_africa_dws import SouthAfricaDWSCollector
from aquascope.schemas.water_data import (
    DataSource,
    StreamflowReading,
    WaterLevelReading,
)


DAILY_HTML = """
<html><body><pre>
DWS verified daily values
DATE D_AVG_FR QUAL
20260120 12.34 1
20260121 - 9
20260122 14.50 2
https://www.dws.gov.za/
</pre></body></html>
"""

POINT_HTML = """
<html><body><pre>
DATE TIME COR_LEVEL COR_LEVEL_QUAL COR_FLOW COR_FLOW_QUAL
20251231 23:45 2.150 1 18.2 1
20260101 001500 2.175 2 18.5 2
</pre></body></html>
"""


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get_text(self, path, params=None, headers=None, use_cache=True):
        self.calls.append(
            {"path": path, "params": params, "headers": headers, "use_cache": use_cache}
        )
        return self.responses.pop(0)


def test_collect_daily_discharge_builds_contract_and_normalises():
    client = FakeClient([DAILY_HTML])
    collector = SouthAfricaDWSCollector(client=client)

    records = collector.collect(
        station_id="c1h001",
        start_date="2026-01-20",
        end_date="2026-01-22",
    )

    assert [record.discharge_cms for record in records] == [12.34, 14.5]
    assert all(isinstance(record, StreamflowReading) for record in records)
    assert all(record.source == DataSource.SOUTH_AFRICA_DWS for record in records)
    assert records[0].station_id == "C1H001"
    assert records[0].reading_datetime == datetime(2026, 1, 20)
    assert records[0].remark == "DWS quality code: 1"

    call = client.calls[0]
    assert call["path"] == "HyData.aspx"
    assert call["params"] == {
        "Station": "C1H001100.00",
        "DataType": "Daily",
        "StartDT": "2026-01-20",
        "EndDT": "2026-01-22",
        "SiteType": "RIV",
    }
    assert call["use_cache"] is False
    assert set(call["headers"]) == {"Accept", "User-Agent"}


def test_water_level_is_chunked_by_calendar_year_and_deduplicated():
    client = FakeClient([POINT_HTML, POINT_HTML])
    collector = SouthAfricaDWSCollector(client=client)

    records = collector.collect(
        station_id="C1H001100.00",
        variable="water_level",
        start_date="2025-12-31",
        end_date="2026-01-01",
    )

    assert len(client.calls) == 2
    assert client.calls[0]["params"]["Station"] == "C1H001100.00"
    assert client.calls[0]["params"]["DataType"] == "Point"
    assert client.calls[0]["params"]["StartDT"] == "2025-12-31"
    assert client.calls[0]["params"]["EndDT"] == "2025-12-31"
    assert client.calls[1]["params"]["StartDT"] == "2026-01-01"
    assert client.calls[1]["params"]["EndDT"] == "2026-01-01"

    assert len(records) == 2
    assert all(isinstance(record, WaterLevelReading) for record in records)
    assert records[0].water_level == 2.15
    assert records[0].reading_datetime == datetime(2025, 12, 31, 23, 45)
    assert records[1].reading_datetime == datetime(2026, 1, 1, 0, 15)


def test_daily_requests_use_twenty_calendar_year_chunks():
    client = FakeClient(
        [
            "<pre>No data for this period</pre>",
            "<pre>No data for this period</pre>",
            "<pre>No data for this period</pre>",
        ]
    )
    collector = SouthAfricaDWSCollector(client=client)

    assert collector.collect(
        station_id="C1H001",
        start_date="1980-06-01",
        end_date="2021-02-03",
    ) == []

    assert [call["params"]["StartDT"] for call in client.calls] == [
        "1980-06-01",
        "2000-01-01",
        "2020-01-01",
    ]
    assert [call["params"]["EndDT"] for call in client.calls] == [
        "1999-12-31",
        "2019-12-31",
        "2021-02-03",
    ]


def test_http_200_kisters_error_fails_closed():
    body = """ERROR [28000] [Kisters][ScriptServerODBC Driver]Client unable to establish connection.
Can't connect to ScriptServer at cenwhyd101:8085.
<html><body></body></html>"""
    collector = SouthAfricaDWSCollector(client=FakeClient([body]))

    with pytest.raises(RuntimeError, match="Kisters ScriptServer failure"):
        collector.fetch_raw(
            station_id="A2H076",
            start_date="2026-01-20",
            end_date="2026-01-21",
        )


def test_no_data_response_is_an_empty_collection():
    collector = SouthAfricaDWSCollector(
        client=FakeClient(["<html><body><pre>No data for this period</pre></body></html>"])
    )
    assert collector.collect(station_id="X3H023", days=1, end_date="2026-01-01") == []


def test_plain_text_and_pre_br_tables_are_supported():
    plain = "DATE D_AVG_FR QUAL\n20260120 1.25 1\n"
    with_br = "<html><body><pre>DATE D_AVG_FR QUAL<br>20260120 1.25 1</pre></body></html>"

    for body in (plain, with_br):
        collector = SouthAfricaDWSCollector(client=FakeClient([body]))
        records = collector.collect(
            station_id="C1H001",
            start_date="2026-01-20",
            end_date="2026-01-20",
        )
        assert len(records) == 1
        assert records[0].discharge_cms == 1.25


def test_unexpected_html_is_not_silently_treated_as_no_data():
    collector = SouthAfricaDWSCollector(client=FakeClient(["<html><title>Runtime Error</title></html>"]))
    with pytest.raises(ValueError, match="expected <pre> hydrology table"):
        collector.fetch_raw(station_id="C1H001", days=1, end_date="2026-01-01")


@pytest.mark.parametrize(
    "body, variable, message",
    [
        ("<pre>DATE LEVEL QUAL\n20260120 1.2 1</pre>", "discharge", "unexpected header"),
        ("<pre>DATE TIME FLOW QUAL\n20260120 1200 1.2 1</pre>", "water_level", "unexpected header"),
        ("<pre>garbage</pre>", "discharge", "missing its DATE header"),
    ],
)
def test_unexpected_table_contract_fails_closed(body, variable, message):
    collector = SouthAfricaDWSCollector(client=FakeClient([body]))
    with pytest.raises(ValueError, match=message):
        collector.fetch_raw(
            station_id="C1H001",
            variable=variable,
            start_date="2026-01-20",
            end_date="2026-01-20",
        )


def test_malformed_and_out_of_range_rows_are_skipped():
    body = """<pre>
DATE D_AVG_FR QUAL
20260119 9.0 1
20261340 8.0 1
20260120 7.0 2
</pre>"""
    collector = SouthAfricaDWSCollector(client=FakeClient([body]))
    records = collector.collect(
        station_id="C1H001",
        start_date="2026-01-20",
        end_date="2026-01-20",
    )
    assert [record.discharge_cms for record in records] == [7.0]


@pytest.mark.parametrize("station", [None, "", "bad/code", "A", "C1H001<script>"])
def test_invalid_station_ids_are_rejected(station):
    collector = SouthAfricaDWSCollector(client=FakeClient([]))
    with pytest.raises(ValueError, match="station_id"):
        collector.fetch_raw(station_id=station, days=1, end_date="2026-01-01")


def test_invalid_query_arguments_are_rejected():
    collector = SouthAfricaDWSCollector(client=FakeClient([]))

    with pytest.raises(ValueError, match="variable"):
        collector.fetch_raw(station_id="C1H001", variable="rainfall", days=1)
    with pytest.raises(ValueError, match="at least 1"):
        collector.fetch_raw(station_id="C1H001", days=0)
    with pytest.raises(ValueError, match="either start_date or days"):
        collector.fetch_raw(
            station_id="C1H001",
            start_date="2026-01-01",
            end_date="2026-01-02",
            days=2,
        )
    with pytest.raises(ValueError, match="after end_date"):
        collector.fetch_raw(
            station_id="C1H001",
            start_date="2026-01-02",
            end_date="2026-01-01",
        )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        collector.fetch_raw(station_id="C1H001", start_date="01/02/2026")


def test_conflicting_duplicate_observations_are_rejected():
    body = """<pre>
DATE D_AVG_FR QUAL
20260120 1.0 1
20260120 2.0 1
</pre>"""
    collector = SouthAfricaDWSCollector(client=FakeClient([body]))
    with pytest.raises(ValueError, match="conflicting observations"):
        collector.fetch_raw(
            station_id="C1H001",
            start_date="2026-01-20",
            end_date="2026-01-20",
        )


def test_malformed_raw_rows_are_skipped_during_normalisation():
    collector = SouthAfricaDWSCollector(client=FakeClient([]))
    records = collector.normalise(
        [
            {
                "station_id": "C1H001",
                "variable": "discharge",
                "reading_datetime": "2026-01-01T00:00:00",
                "value": "3.5",
                "quality": "1",
            },
            {
                "station_id": "C1H001",
                "variable": "discharge",
                "reading_datetime": "not-a-date",
                "value": "nan",
            },
        ]
    )
    assert len(records) == 1
    assert records[0].discharge_cms == 3.5
