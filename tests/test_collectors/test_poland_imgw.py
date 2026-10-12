"""Poland IMGW-PIB collector.

Every archive line and live record below is a verbatim capture from
``danepubliczne.imgw.pl`` taken on 2026-09-14, one from each file format the
archive has shipped: comma-separated cp1250 with quoted fields and a leading
space in the id (monthly files, here ``codz_2020_12``), semicolon-separated
UTF-8 with a byte-order mark (``codz_2023``), every line wrapped in one more
layer of quotes (``codz_2024``), and comma-separated cp1250 again (``codz_2025``).
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.poland_imgw import (
    ARCHIVE_BASE,
    PolandIMGWCollector,
    archive_files,
    calendar_year,
    decode_archive,
    hydro_month,
    hydro_year,
    parse_archive,
)
from aquascope.schemas.water_data import (
    DataSource,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)

LIVE = [
    {"id_stacji": "149180020", "stacja": "Chałupki", "rzeka": "Odra", "wojewodztwo": "śląskie", "lon": "18.3278", "lat": "49.9219", "rok_zalozenia_stacji": "1891", "rzedna_zerawodowskazu": "192.727", "kilometr_biegu_rzeki": "725.82", "stan_alarmowy": "420", "stan_ostrzegawczy": "300", "stan_wody": "143", "stan_wody_data_pomiaru": "2026-09-14 02:50:00", "temperatura_wody": None, "temperatura_wody_data_pomiaru": None, "przeplyw": "8.32", "przeplyw_data": "2026-09-14 03:10:00", "zjawisko_lodowe": "0", "zjawisko_lodowe_data_pomiaru": "2026-04-23 06:00:00", "zjawisko_zarastania": "0", "zjawisko_zarastania_data_pomiaru": "2026-09-02 11:40:00"},  # noqa: E501
    {"id_stacji": "151140030", "stacja": "Przewoźniki", "rzeka": "Skroda", "wojewodztwo": "lubuskie", "lon": "14.8217", "lat": "51.5253", "rok_zalozenia_stacji": "1957", "rzedna_zerawodowskazu": "114.049", "kilometr_biegu_rzeki": "4.22", "stan_alarmowy": "340", "stan_ostrzegawczy": "300", "stan_wody": "225", "stan_wody_data_pomiaru": "2026-09-14 02:50:00", "temperatura_wody": None, "temperatura_wody_data_pomiaru": None, "przeplyw": "0.11", "przeplyw_data": "2026-02-18 09:50:00", "zjawisko_lodowe": "0", "zjawisko_lodowe_data_pomiaru": "2026-02-26 11:20:00", "zjawisko_zarastania": "0", "zjawisko_zarastania_data_pomiaru": "2026-08-31 11:00:00"},  # noqa: E501
    {"id_stacji": "152200030", "stacja": "Wyszogród", "rzeka": "Wisła", "wojewodztwo": "mazowieckie", "lon": "20.1931", "lat": "52.3847", "rok_zalozenia_stacji": "1946", "rzedna_zerawodowskazu": "60.355", "kilometr_biegu_rzeki": "351.19", "stan_alarmowy": "550", "stan_ostrzegawczy": "500", "stan_wody": "225", "stan_wody_data_pomiaru": "2026-09-14 02:50:00", "temperatura_wody": "17.4", "temperatura_wody_data_pomiaru": "2026-09-13 06:00:00", "przeplyw": "212", "przeplyw_data": "2026-09-14 03:10:00", "zjawisko_lodowe": "0", "zjawisko_lodowe_data_pomiaru": "2026-04-26 06:00:00", "zjawisko_zarastania": "0", "zjawisko_zarastania_data_pomiaru": "2026-05-28 10:20:00"},  # noqa: E501
    {"id_stacji": "154190080", "stacja": "Żukowo", "rzeka": "Jez. Drużno", "wojewodztwo": "warmińsko-mazurskie", "lon": "19.4508", "lat": "54.1028", "rok_zalozenia_stacji": "1975", "rzedna_zerawodowskazu": "-5.275", "kilometr_biegu_rzeki": "16.17", "stan_alarmowy": None, "stan_ostrzegawczy": None, "stan_wody": "568", "stan_wody_data_pomiaru": "2026-09-14 02:50:00", "temperatura_wody": "16.7", "temperatura_wody_data_pomiaru": "2026-09-13 06:00:00", "przeplyw": None, "przeplyw_data": None, "zjawisko_lodowe": "3", "zjawisko_lodowe_data_pomiaru": "2023-01-23 12:10:00", "zjawisko_zarastania": "0", "zjawisko_zarastania_data_pomiaru": "2026-09-11 08:40:00"},  # noqa: E501
]

# codz_2024.zip: the whole line wrapped in an extra layer of quotes, cp1250.
LINES_2024 = "\n".join([
    '"149180020,CHAŁUPKI,Odra (1),""2024"",""01"",""01"",113,25.400,,""11"""',
    '"149180020,CHAŁUPKI,Odra (1),""2024"",""02"",""01"",125,31.800,,""12"""',
    '"149180020,CHAŁUPKI,Odra (1),""2024"",""03"",""01"",198,82.900,,""01"""',
    '"150180030,KOŹLE,Odra (1),""2024"",""01"",""01"",283,,,""11"""',
    '"150170240,KRAPKOWICE,Odra (1),""2024"",""06"",""03"",236,,12.6,""04"""',
])
# codz_2023.zip: semicolons, UTF-8 with a byte-order mark, 99.9 for "no temperature".
LINES_2023 = "\n".join([
    "149180020;CHAŁUPKI;Odra (1);2023;01;01;88;14.200;99.9;11",
    "149180020;CHAŁUPKI;Odra (1);2023;01;02;89;13.700;99.9;11",
])
# codz_2020_12.zip: a monthly file, quoted fields, a leading space in the id, cp1250.
LINES_2020_12 = "\n".join([
    '" 149180020","CHAŁUPKI","Odra (1)","2020","12","01",292,172.000,99.9,"10"',
    '" 149190250","JABŁONKA","Piekielnik (82224)","2020","12","31",153,.770,99.9,"10"',
])
# codz_2025.zip: quoted fields, cp1250, an empty level.
LINES_2025 = "\n".join([
    '"149180020",CHAŁUPKI,Odra (1),"2025","01","01",165,21.400,,"11"',
    '"150170080",JARNOŁTÓWEK,Złoty Potok (117644),"2025","07","28",,0.480,,"05"',
])


def _zip(inner: str, text: str, encoding: str, bom: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(inner, (("\ufeff" if bom else "") + text + "\n").encode(encoding))
    return buf.getvalue()


ARCHIVE = {
    f"{ARCHIVE_BASE}/2020/codz_2020_12.zip": _zip("codz_2020_12.csv", LINES_2020_12, "cp1250"),
    f"{ARCHIVE_BASE}/2023/codz_2023.zip": _zip("codz_2023.csv", LINES_2023, "utf-8", bom=True),
    f"{ARCHIVE_BASE}/2024/codz_2024.zip": _zip("codz_2024.csv", LINES_2024, "cp1250"),
    f"{ARCHIVE_BASE}/2025/codz_2025.zip": _zip("codz_2025.csv", LINES_2025, "cp1250"),
}


class _Collector(PolandIMGWCollector):
    """The real collector over canned zips: a URL outside ARCHIVE is a 404."""

    def __init__(self, cache_dir: Path):
        client = MagicMock()
        client.get_json.return_value = LIVE
        super().__init__(client=client, cache_dir=cache_dir)
        self.downloads: list[str] = []

    def _download(self, url: str) -> bytes | None:
        self.downloads.append(url)
        return ARCHIVE.get(url)


# ── the hydrological calendar ────────────────────────────────────────────


def test_hydrological_calendar():
    assert hydro_year(date(2023, 10, 31)) == 2023 and hydro_year(date(2023, 11, 1)) == 2024
    assert hydro_month(date(2023, 11, 1)) == 1 and hydro_month(date(2024, 1, 1)) == 3
    assert hydro_month(date(2024, 10, 1)) == 12
    assert calendar_year(2024, 11) == 2023 and calendar_year(2024, 12) == 2023 and calendar_year(2024, 1) == 2024


def test_archive_files_monthly_then_yearly_and_clamped_to_1951():
    # September 2022 to January 2023: two monthly files of hydro year 2022, then the 2023 yearly file.
    assert archive_files(date(2022, 9, 15), date(2023, 1, 10)) == [
        (2022, "codz_2022_11.zip"), (2022, "codz_2022_12.zip"), (2023, "codz_2023.zip"),
    ]
    # Nothing before hydro year 1951 (November 1950); December 1951 is hydro month 02 of 1952.
    files = archive_files(date(1930, 1, 1), date(1951, 12, 31))
    assert files[0] == (1951, "codz_1951_01.zip") and len(files) == 14
    assert files[-2:] == [(1952, "codz_1952_01.zip"), (1952, "codz_1952_02.zip")]
    assert archive_files(date(2024, 5, 1), date(2024, 4, 1)) == []


# ── parsing every format the archive has shipped ────────────────────────


@pytest.mark.parametrize(
    ("raw", "first_date", "first_id"),
    [
        (_zip("a.csv", LINES_2024, "cp1250"), date(2023, 11, 1), "149180020"),
        (_zip("a.csv", LINES_2023, "utf-8", bom=True), date(2022, 11, 1), "149180020"),
        (_zip("a.csv", LINES_2020_12, "cp1250"), date(2020, 10, 1), "149180020"),
        (_zip("a.csv", LINES_2025, "cp1250"), date(2024, 11, 1), "149180020"),
    ],
    ids=["2024-wrapped-lines", "2023-semicolon-bom", "2020-cp1250-monthly", "2025-cp1250-yearly"],
)
def test_every_archive_format_parses_to_calendar_dates(raw, first_date, first_id):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        frame = parse_archive(decode_archive(z.read(z.namelist()[0])))
    assert str(frame["station_id"].iloc[0]) == first_id  # leading space stripped, BOM gone
    assert frame["date"].iloc[0].date() == first_date
    assert "CHAŁUPKI" in set(frame["name"])  # diacritics survive both encodings


def test_hydro_month_crossing_the_calendar_year_and_missing_values():
    frame = parse_archive(LINES_2024)
    chalupki = frame[frame["station_id"] == "149180020"].sort_values("date")
    assert [d.date() for d in chalupki["date"]] == [date(2023, 11, 1), date(2023, 12, 1), date(2024, 1, 1)]
    assert list(chalupki["discharge_cms"]) == pytest.approx([25.4, 31.8, 82.9])
    kozle = frame[frame["station_id"] == "150180030"].iloc[0]
    assert kozle["level_cm"] == 283 and kozle["discharge_cms"] != kozle["discharge_cms"]  # NaN
    krapkowice = frame[frame["station_id"] == "150170240"].iloc[0]
    assert krapkowice["date"].date() == date(2024, 4, 3) and float(krapkowice["temp_c"]) == pytest.approx(12.6)
    sentinel = parse_archive(LINES_2023)
    assert sentinel["temp_c"].isna().all()  # 99.9 is "not measured", never a temperature


# ── the collector ────────────────────────────────────────────────────────


def test_stations_from_the_live_api(tmp_path):
    c = _Collector(tmp_path)
    stations = {s.station_id: s for s in c.stations()}
    assert set(stations) == {"149180020", "151140030", "152200030", "154190080"}
    ch = stations["149180020"]
    assert (ch.latitude, ch.longitude) == (49.9219, 18.3278) and ch.river == "Odra" and ch.country == "POL"
    assert ch.variables == ("discharge", "water_level")
    assert ch.period_start == date(1891, 1, 1) and ch.period_end == date(2026, 9, 14)
    assert ch.extra["alarm_stage_cm"] == 420 and ch.extra["warning_stage_cm"] == 300
    assert stations["152200030"].variables == ("discharge", "water_level", "water_quality")
    assert stations["154190080"].variables == ("water_level", "water_quality")  # no discharge now
    assert stations["154190080"].extra["alarm_stage_cm"] is None
    # Filters: bbox around Warsaw keeps Wyszogród only; variable narrows to what is reporting now.
    assert [s.name for s in c.stations(bbox=(20.0, 52.0, 21.0, 53.0))] == ["Wyszogród"]
    assert {s.station_id for s in c.stations(variable="water_quality")} == {"152200030", "154190080"}
    assert c.stations(variable="reservoir_storage") == []
    assert len(c.stations(max_items=2)) == 2
    c.client.get_json.assert_called_once()  # one request for the whole network


def test_collect_discharge_level_and_temperature_from_the_archive(tmp_path):
    c = _Collector(tmp_path)
    flow = c.collect(variable="discharge", station_ids=["149180020"], start="2023-11-01", end="2024-01-31")
    assert all(isinstance(r, StreamflowReading) for r in flow) and len(flow) == 3
    assert flow[0].source is DataSource.POLAND_IMGW and flow[0].station_name == "CHAŁUPKI"
    assert [(r.reading_datetime.date(), r.discharge_cms) for r in flow][:2] == [
        (date(2023, 11, 1), pytest.approx(25.4)), (date(2023, 12, 1), pytest.approx(31.8)),
    ]
    level = c.collect(
        variable="water_level", station_ids=["149180020", "150180030"], start="2023-11-01", end="2023-11-30",
    )
    assert sorted((r.station_id, round(r.water_level, 2), r.unit) for r in level) == [
        ("149180020", 1.13, "m"), ("150180030", 2.83, "m"),
    ]
    assert all(isinstance(r, WaterLevelReading) for r in level)
    # Koźle has a level but no discharge on that day: present above, absent here.
    assert c.collect(variable="discharge", station_ids=["150180030"], start="2023-11-01", end="2023-11-30") == []
    temp = c.collect(variable="water_quality", station_ids=["150170240"], start="2024-04-01", end="2024-04-30")
    assert len(temp) == 1 and isinstance(temp[0], WaterQualitySample)
    assert temp[0].parameter == "water_temperature" and temp[0].value == pytest.approx(12.6) and temp[0].unit == "degC"
    # One download per file, then the parsed frame is reused: three collects, one 2024 zip.
    assert c.downloads.count(f"{ARCHIVE_BASE}/2024/codz_2024.zip") == 1


def test_window_across_file_formats_and_an_unpublished_year(tmp_path):
    c = _Collector(tmp_path)
    recs = c.collect(variable="discharge", station_ids=["149180020"], start="2020-10-01", end="2026-09-14")
    assert [r.reading_datetime.date() for r in recs] == [
        date(2020, 10, 1), date(2022, 11, 1), date(2022, 11, 2), date(2023, 11, 1),
        date(2023, 12, 1), date(2024, 1, 1), date(2024, 11, 1),
    ]
    # Hydro years 2021-2022 have no canned monthly files and 2026 is unpublished:
    # every one a 404, none fatal, none asked twice.
    misses = [u for u in c.downloads if u not in ARCHIVE]
    assert f"{ARCHIVE_BASE}/2026/codz_2026.zip" in misses and len(misses) == len(set(misses))


def test_zips_are_cached_on_disk_for_finished_years(tmp_path):
    first = _Collector(tmp_path)
    first.collect(variable="discharge", station_ids=["149180020"], start="2023-11-01", end="2023-11-30")
    assert (tmp_path / "codz_2024.zip").exists()
    second = _Collector(tmp_path)
    recs = second.collect(variable="discharge", station_ids=["149180020"], start="2023-11-01", end="2023-11-30")
    assert len(recs) == 1 and second.downloads == []  # served from disk, no request


def test_latest_only_uses_each_values_own_timestamp(tmp_path):
    c = _Collector(tmp_path)
    flow = c.collect(variable="discharge", station_ids=["151140030"], latest_only=True)
    level = c.collect(variable="water_level", station_ids=["151140030"], latest_only=True)
    assert flow[0].reading_datetime == datetime(2026, 2, 18, 9, 50) and flow[0].discharge_cms == pytest.approx(0.11)
    assert level[0].reading_datetime == datetime(2026, 9, 14, 2, 50) and level[0].water_level == pytest.approx(2.25)
    assert flow[0].location is not None and flow[0].location.latitude == pytest.approx(51.5253)
    assert c.collect(variable="discharge", station_ids=["154190080"], latest_only=True) == []  # no discharge now
    with pytest.raises(ValueError, match="Unknown variable"):
        c.collect(variable="precipitation", station_ids=["151140030"])


def test_explorer_fetch_path_reads_the_archive(monkeypatch, tmp_path):
    from aquascope import explore

    c = _Collector(tmp_path)
    monkeypatch.setattr(explore, "build_collector", lambda key: c)
    out = explore.fetch_series("poland_imgw", "149180020", prefer_archive=False, period_start="2023-11-01")
    assert out["variable"] == "discharge" and out["unit"] == "m3/s"
    assert out["series"].index.min().date() == date(2023, 11, 1)
    assert "hydrological-year" in out["note"] and "last October" in out["note"]


def test_a_second_collector_reuses_the_parsed_archive(tmp_path, monkeypatch):
    """The harvest builds a collector per station; each used to re-read and re-parse every zip (~3 min a station)."""
    import aquascope.collectors.poland_imgw as imgw

    first = _Collector(tmp_path)
    first.collect(variable="discharge", station_ids=["149180020"], start="2023-11-01", end="2024-01-31")
    parsed = []
    real = imgw.parse_archive
    monkeypatch.setattr(imgw, "parse_archive", lambda text: parsed.append(1) or real(text))
    second = _Collector(tmp_path)
    recs = second.collect(variable="water_level", station_ids=["150180030"], start="2023-11-01", end="2023-11-30")
    assert recs and parsed == [] and second.downloads == []


def test_a_recent_year_is_parsed_again_once_its_cache_has_expired(tmp_path, monkeypatch):
    import aquascope.collectors.poland_imgw as imgw

    c = _Collector(tmp_path)
    c.collect(variable="discharge", station_ids=["149180020"], start="2024-11-01", end="2025-01-31")
    key = (str(tmp_path), "codz_2025.zip")
    loaded_at, frame = imgw.PolandIMGWCollector._shared_frames[key]
    imgw.PolandIMGWCollector._shared_frames[key] = (loaded_at - imgw.RECENT_TTL_SECONDS - 1, frame)
    parsed = []
    real = imgw.parse_archive
    monkeypatch.setattr(imgw, "parse_archive", lambda text: parsed.append(1) or real(text))
    _Collector(tmp_path).collect(variable="discharge", station_ids=["149180020"], start="2024-11-01",
                                 end="2025-01-31")
    assert parsed  # the recent year was read again
