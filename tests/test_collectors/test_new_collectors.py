"""Tests for the new v0.2.0 collectors: GEMStat, WQP, Taiwan Civil IoT."""


from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.collectors.taiwan_civil_iot import TaiwanCivilIoTCollector
from aquascope.collectors.wqp import WQPCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample


class TestGEMStatCollector:
    def test_init(self):
        collector = GEMStatCollector()
        assert collector.name == "gemstat"

    def test_parse_csv_valid(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit,Latitude,Longitude,Country Code\n"
            "GEM001,2023-01-15,pH,7.2,pH units,47.3,8.5,CH\n"
            "GEM002,2023-02-20,DO,8.1,mg/L,48.1,11.6,DE\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 2
        assert all(isinstance(s, WaterQualitySample) for s in samples)
        assert samples[0].source == DataSource.GEMSTAT
        assert samples[0].parameter == "pH"
        assert samples[0].value == 7.2

    def test_parse_csv_skips_nd_values(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
            "GEM001,2023-01-15,pH,ND,pH units\n"
            "GEM002,2023-02-20,DO,8.1,mg/L\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 1

    def test_parse_csv_empty(self):
        csv_content = "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 0

    def test_parse_csv_max_records(self):
        header = "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
        rows = "".join(f"GEM{i:03d},2023-01-{(i % 28) + 1:02d},pH,{7.0 + i * 0.01},pH units\n" for i in range(100))
        samples = GEMStatCollector.parse_gemstat_csv(header + rows, max_records=10)
        assert len(samples) == 10

    def test_parse_csv_with_location(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit,Latitude,Longitude\n"
            "GEM001,2023-06-01,DO,7.5,mg/L,25.033,121.565\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert samples[0].location is not None
        assert abs(samples[0].location.latitude - 25.033) < 0.001


class TestWQPCollector:
    def test_init(self):
        collector = WQPCollector()
        assert collector.name == "wqp"

    def test_normalise_valid(self):
        raw = [
            {
                "MonitoringLocationIdentifier": "USGS-01010000",
                "MonitoringLocationName": "Test Station",
                "LatitudeMeasure": "44.5",
                "LongitudeMeasure": "-67.5",
                "ActivityStartDate": "2023-05-15",
                "ActivityStartTime/Time": "10:30:00",
                "CharacteristicName": "Dissolved oxygen (DO)",
                "ResultMeasureValue": "8.5",
                "ResultMeasure/MeasureUnitCode": "mg/l",
            },
        ]
        collector = WQPCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].source == DataSource.WQP
        assert samples[0].value == 8.5

    def test_normalise_skips_empty_values(self):
        raw = [
            {
                "MonitoringLocationIdentifier": "USGS-01010000",
                "ActivityStartDate": "2023-05-15",
                "CharacteristicName": "pH",
                "ResultMeasureValue": "",
            },
            {
                "MonitoringLocationIdentifier": "USGS-01010000",
                "ActivityStartDate": "2023-05-15",
                "CharacteristicName": "pH",
                "ResultMeasureValue": "-",
            },
        ]
        collector = WQPCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0


class TestTaiwanCivilIoTCollector:
    def test_init(self):
        collector = TaiwanCivilIoTCollector()
        assert collector.name == "taiwan_civil_iot"

    def test_normalise_valid_datastream(self):
        raw = [
            {
                "name": "Water Level",
                "unitOfMeasurement": {"symbol": "m"},
                "Thing": {
                    "@iot.id": 1,
                    "name": "Station Alpha",
                    "Locations": [
                        {"location": {"coordinates": [121.5, 25.0]}}
                    ],
                },
                "Observations": [
                    {
                        "result": 3.45,
                        "phenomenonTime": "2024-03-15T10:00:00Z",
                    }
                ],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].source == DataSource.TAIWAN_CIVIL_IOT
        assert samples[0].value == 3.45
        assert samples[0].location is not None
        assert abs(samples[0].location.latitude - 25.0) < 0.01

    def test_normalise_skips_no_observations(self):
        raw = [
            {
                "name": "Flow Rate",
                "Thing": {"@iot.id": 2, "name": "Empty Station"},
                "Observations": [],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0

    def test_normalise_skips_null_result(self):
        raw = [
            {
                "name": "pH",
                "Thing": {"@iot.id": 3, "name": "Null Station"},
                "Observations": [{"result": None, "phenomenonTime": "2024-01-01T00:00:00Z"}],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0
