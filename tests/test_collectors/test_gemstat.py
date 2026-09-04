"""Tests for the GEMStat (UNEP) collector and Zenodo payload validation."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample


def _make_fake_gemstat_zip() -> bytes:
    """Build a minimal in-memory GEMStat ZIP archive with station metadata and pH records."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Station metadata CSV
        station_out = io.StringIO()
        writer = csv.writer(station_out)
        writer.writerow(
            [
                "GEMS Station Number",
                "Local Station Number",
                "Country Name",
                "Water Type",
                "Station Identifier",
                "Station Narrative",
                "Water Body Name",
                "Main Basin",
                "Upstream Basin Area",
                "Elevation",
                "Monitoring Type",
                "Date Station Opened",
                "Responsible Collection Agency",
                "Latitude",
                "Longitude",
                "River Width",
                "Discharge",
                "Max. Depth",
                "Lake Area",
                "Lake Volume",
                "Average Retention",
                "Area of Aquifer",
                "Depth of Impermeable Lining",
                "Production Zone",
                "Mean Abstraction Rate",
                "Mean Abstraction Level",
            ]
        )
        writer.writerow(
            [
                "IRL00001",
                "LOC001",
                "Ireland",
                "River station",
                "River Shannon",
                "",
                "",
                "Shannon",
                "",
                "10",
                "SURVEILLANCE",
                "",
                "EPA Ireland",
                "53.5",
                "-8.5",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "DEU00001",
                "LOC002",
                "Germany",
                "River station",
                "Rhine River",
                "",
                "",
                "Rhine",
                "",
                "20",
                "SURVEILLANCE",
                "",
                "UBA Germany",
                "50.5",
                "8.5",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        zf.writestr("GEMStat_station_metadata.csv", station_out.getvalue())

        # pH parameter CSV
        ph_out = io.StringIO()
        pwriter = csv.writer(ph_out)
        pwriter.writerow(
            [
                "GEMS Station Number",
                "Sample Date",
                "Sample Time",
                "Depth",
                "Parameter Code",
                "Analysis Method Code",
                "Value Flags",
                "Value",
                "Unit",
                "Data Quality",
                "Integrated Value",
                "Remark",
                "License Information",
            ]
        )
        pwriter.writerow(
            [
                "IRL00001",
                "2020-05-15",
                "10:00",
                "0.5",
                "pH",
                "M01",
                "",
                "7.8",
                "pH units",
                "Good",
                "",
                "Clean reading",
                "",
            ]
        )
        pwriter.writerow(
            [
                "IRL00001",
                "2020-06-15",
                "10:00",
                "0.5",
                "pH",
                "M01",
                "",
                "7.9",
                "pH units",
                "Good",
                "",
                "Summer reading",
                "",
            ]
        )
        pwriter.writerow(
            [
                "DEU00001",
                "2020-05-15",
                "11:00",
                "0.5",
                "pH",
                "M01",
                "",
                "8.1",
                "pH units",
                "Good",
                "",
                "German reading",
                "",
            ]
        )
        zf.writestr("pH.csv", ph_out.getvalue())

    return buf.getvalue()


class _MockStreamingResponse:
    """Mock context manager response for httpx.stream."""

    def __init__(self, chunks: list[bytes], status_code: int = 200, content_type: str = "application/zip"):
        self.chunks = chunks
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size: int = 65536):
        yield from self.chunks


class TestGEMStatCollector:
    """Tests verifying GEMStat download handling, error surfacing, and normalization."""

    def test_gemstat_init_and_attributes(self):
        c = GEMStatCollector()
        assert c.name == "gemstat"
        assert "pH" in c.DEFAULT_PARAMETERS
        assert "Temperature" in c.DEFAULT_PARAMETERS

    def test_gemstat_fetch_raw_html_response_raises_diagnostic_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify that an HTML response from Zenodo raises a diagnostic error rather than BadZipFile."""
        monkeypatch.chdir(tmp_path)
        c = GEMStatCollector()

        # Mock Zenodo record API
        c.client.get_json = mock.MagicMock(
            return_value={
                "files": [
                    {
                        "key": "GFQA_v3.zip",
                        "size": 1000,
                        "checksum": "fake_chk_1",
                        "links": {"content": "https://zenodo.org/fake/download"},
                    }
                ]
            }
        )

        html_body = b"<!DOCTYPE html><html><body>Blocked by Cloudflare</body></html>"
        mock_resp = _MockStreamingResponse([html_body], status_code=200, content_type="text/html; charset=utf-8")

        with mock.patch("httpx.stream", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Zenodo returned HTML instead of ZIP archive"):
                c.fetch_raw()

        # Ensure no corrupt .zip file was preserved in data/cache
        cached_zips = list(Path("data/cache").glob("*.zip"))
        assert len(cached_zips) == 0

    def test_gemstat_fetch_raw_non_pk_magic_bytes_raises_diagnostic_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify that non-ZIP binary payloads raise a diagnostic error naming the preview."""
        monkeypatch.chdir(tmp_path)
        c = GEMStatCollector()

        c.client.get_json = mock.MagicMock(
            return_value={
                "files": [
                    {
                        "key": "GFQA_v3.zip",
                        "size": 1000,
                        "checksum": "fake_chk_2",
                        "links": {"self": "https://zenodo.org/fake/download"},
                    }
                ]
            }
        )

        garbage_payload = b"RANDOM_NON_ZIP_BYTES_1234567890"
        mock_resp = _MockStreamingResponse([garbage_payload], status_code=200, content_type="application/octet-stream")

        with mock.patch("httpx.stream", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="downloaded payload is not a valid ZIP archive"):
                c.fetch_raw()

        cached_zips = list(Path("data/cache").glob("*.zip"))
        assert len(cached_zips) == 0

    def test_gemstat_fetch_raw_success_and_filtering(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify successful download, caching, station join, and country filtering."""
        monkeypatch.chdir(tmp_path)
        c = GEMStatCollector()

        zip_data = _make_fake_gemstat_zip()
        c.client.get_json = mock.MagicMock(
            return_value={
                "files": [
                    {
                        "key": "GFQA_v3.zip",
                        "size": len(zip_data),
                        "checksum": "fake_chk_3",
                        "links": {"content": "https://zenodo.org/fake/download"},
                    }
                ]
            }
        )

        mock_resp = _MockStreamingResponse([zip_data], status_code=200, content_type="application/zip")

        with mock.patch("httpx.stream", return_value=mock_resp):
            # Query Ireland
            rows_irl = c.fetch_raw(country="Ireland")
            assert len(rows_irl) == 2
            assert all(r["_country"] == "Ireland" for r in rows_irl)
            assert rows_irl[0]["GEMS Station Number"] == "IRL00001"
            assert rows_irl[0]["_station_name"] == "River Shannon"

            # Query Germany with max_records=1
            rows_deu = c.fetch_raw(country="Germany", max_records=1)
            assert len(rows_deu) == 1
            assert rows_deu[0]["_country"] == "Germany"

            # Query unknown country
            rows_unknown = c.fetch_raw(country="Atlantis")
            assert rows_unknown == []

    def test_gemstat_fetch_raw_recovers_from_corrupt_cached_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify that an existing corrupt cached file is detected, removed, and re-downloaded."""
        monkeypatch.chdir(tmp_path)
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        c = GEMStatCollector()
        zip_data = _make_fake_gemstat_zip()
        checksum = "fake_chk_recovery"

        import hashlib

        cache_key = hashlib.md5(checksum.encode()).hexdigest()
        corrupt_zip = cache_dir / f"gemstat_{cache_key}.zip"
        corrupt_zip.write_text("CORRUPTED EXISTING ARCHIVE")

        c.client.get_json = mock.MagicMock(
            return_value={
                "files": [
                    {
                        "key": "GFQA_v3.zip",
                        "size": len(zip_data),
                        "checksum": checksum,
                        "links": {"content": "https://zenodo.org/fake/download"},
                    }
                ]
            }
        )

        mock_resp = _MockStreamingResponse([zip_data], status_code=200, content_type="application/zip")

        with mock.patch("httpx.stream", return_value=mock_resp):
            rows = c.fetch_raw(country="Ireland")
            assert len(rows) == 2

    def test_gemstat_normalise(self):
        """Verify normalisation of raw dicts into WaterQualitySample records."""
        c = GEMStatCollector()
        raw = [
            {
                "GEMS Station Number": "IRL00001",
                "_station_name": "River Shannon",
                "_country": "Ireland",
                "_lat": "53.5",
                "_lon": "-8.5",
                "Sample Date": "2020-05-15",
                "Parameter Code": "pH",
                "Value": "7.8",
                "Unit": "pH units",
            },
            {
                # Missing or non-numeric value should be skipped safely
                "GEMS Station Number": "IRL00001",
                "Sample Date": "2020-05-16",
                "Parameter Code": "pH",
                "Value": "ND",
            },
        ]
        samples = c.normalise(raw)
        assert len(samples) == 1
        s = samples[0]
        assert isinstance(s, WaterQualitySample)
        assert s.source == DataSource.GEMSTAT
        assert s.station_id == "IRL00001"
        assert s.station_name == "River Shannon"
        assert s.location.latitude == 53.5
        assert s.location.longitude == -8.5
        assert s.parameter == "pH"
        assert s.value == 7.8
        assert s.unit == "pH units"
        assert s.county == "Ireland"
