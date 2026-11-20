"""WaPOR catalog periods, exact masked area means, and bounded public reads."""

import datetime as dt
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from aquascope.collectors import wapor
from aquascope.collectors.base import CollectorError
from aquascope.utils.cog import COGError


def _catalog_item(variable, unit, scale=0.1):
    mapset = f"L1-{variable}-D"
    code = f"WAPOR-3.{mapset}.2025-05-D2"
    return {
        "code": code,
        "downloadUrl": f"{wapor._BUCKET}/{mapset}/{code}.tif",
        "dimensions": [{"type": "TIME", "member": {"startDate": "2025-05-11", "endDate": "2025-05-20"}}],
        "measureUnit": unit,
        "scale": scale,
        "offset": 0.0,
        "noDataValue": -9999.0,
    }


def _cog(values):
    return Mock(
        transform=(0, 1, 0, 2, 0, -1),
        epsg=4326,
        width=2,
        height=2,
        scale=0.1,
        offset=0.0,
        nodata=-9999.0,
        read_window=Mock(return_value=np.array(values)),
    )


@pytest.mark.parametrize(
    "variable, unit, scale",
    [
        ("RET", "mm/day", 0.1),
        ("AETI", "mm/day", 0.1),
        ("NPP", "gC/m²/day", 0.001),
    ],
)
def test_catalog_observations_align_period_and_keep_native_unit(monkeypatch, variable, unit, scale):
    item = _catalog_item(variable, unit, scale)
    client = Mock(get_json=Mock(return_value={"response": {"items": [item], "hasNext": False}}))
    cog = _cog([[0, 30]])
    cog.scale = scale
    open_cog = Mock(return_value=cog)
    monkeypatch.setattr(wapor, "open_cog", open_cog)

    [result] = wapor.WaPORCollector(client).collect(
        bbox=(0.5, 1, 2, 2), start_date="2025-05-15", end_date="2025-05-15", variable=variable
    )

    assert client.get_json.call_args.args[0].endswith(f"/mapsets/L1-{variable}-D/rasters")
    assert client.get_json.call_args.kwargs["params"]["filter"] == "time:OVERLAPS:2025-05-15:2025-05-15"
    cog.read_window.assert_called_once_with(0, 0, 1, 0)
    assert (result.cube_code, result.unit, result.level) == (variable, unit, "L1")
    assert result.value == pytest.approx(20)
    assert (result.start_date, result.end_date) == (dt.date(2025, 5, 11), dt.date(2025, 5, 20))
    assert result.statistic == "area_weighted_mean"
    assert open_cog.call_args.args == (item["downloadUrl"],)


def test_box_mean_masks_nodata_and_weights_by_latitude():
    raster = _cog([[10, 20], [0, np.nan]])
    raster.transform = (0, 1, 0, 80, 0, -30)
    raster.width = raster.height = 2
    northern = np.sin(np.radians(80)) - np.sin(np.radians(50))
    southern = np.sin(np.radians(50)) - np.sin(np.radians(20))
    assert wapor._bbox_mean(raster, (0, 20, 2, 80)) == pytest.approx(30 * northern / (2 * northern + southern))


def test_box_with_no_valid_cells_returns_no_observation():
    assert wapor._bbox_mean(_cog([[np.nan]]), (0, 1, 1, 2)) is None


def test_start_date_alone_uses_today_as_end():
    client = Mock(get_json=Mock(return_value={"response": {"items": [], "hasNext": False}}))
    assert wapor.WaPORCollector(client).collect(bbox=(0, 0, 1, 1), start_date="2026-04-01") == []
    assert client.get_json.call_args.kwargs["params"]["filter"] == (f"time:OVERLAPS:2026-04-01:{dt.date.today()}")


def test_unverified_download_location_fails_before_raster_request(monkeypatch):
    item = _catalog_item("RET", "mm/day") | {"downloadUrl": "https://example.invalid/different.tif"}
    client = Mock(get_json=Mock(return_value={"response": {"items": [item], "hasNext": False}}))
    open_cog = Mock()
    monkeypatch.setattr(wapor, "open_cog", open_cog)
    with pytest.raises(CollectorError) as error:
        wapor.WaPORCollector(client).collect(bbox=(0, 0, 1, 1))
    assert "unexpected raster download location" in str(error.value.__cause__)
    open_cog.assert_not_called()


def test_range_fetcher_rejects_full_response_without_reading_it():
    class UnreadableBody(httpx.SyncByteStream):
        def __iter__(self):
            pytest.fail("a full-file response must never be read")

    def response(request):
        assert request.headers["Range"] == "bytes=0-2"
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, stream=UnreadableBody())

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(COGError, match="did not honour"):
            wapor._range_fetcher(client)("https://example.test/raster", 0, 2)


def test_range_fetcher_reads_only_the_confirmed_span():
    transport = httpx.MockTransport(
        lambda _: httpx.Response(206, headers={"Content-Range": "bytes 1-3/4"}, stream=httpx.ByteStream(b"abc"))
    )
    with httpx.Client(transport=transport) as client:
        assert wapor._range_fetcher(client)("https://example.test/raster", 1, 5) == b"abc"
