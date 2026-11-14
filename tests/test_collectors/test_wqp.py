"""WQP state-code normalisation (#754).

``fetch_raw`` used to forward ``state_code`` unchanged, so a postal abbreviation
reached WQP as ``statecode=RI`` and came back as a bare HTTP 400. These tests use
a fake client that records the params sent; no network access.
"""

from __future__ import annotations

import pytest

from aquascope.collectors.wqp import WQPCollector, _normalise_wqp_state


class FakeClient:
    """Records the params of the one request the collector makes.

    ``_client`` has no ``stream`` attribute, so the collector takes its
    buffered path and the fake only has to implement ``get_text``.
    """

    base_url = "https://wqp.test"
    _client = object()

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_text(self, url, params=None, **kwargs) -> str:  # noqa: ANN001, ANN002, ANN003, ANN201
        self.calls.append((url, params or {}))
        return ""


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("RI", "US:44"),
        ("ri", "US:44"),
        ("44", "US:44"),
        ("4", "US:04"),
        ("US:44", "US:44"),
        ("us:44", "US:44"),
        (" US:44 ", "US:44"),
    ],
)
def test_normalises_the_accepted_forms(given: str, expected: str) -> None:
    assert _normalise_wqp_state(given) == expected


@pytest.mark.parametrize("given", ["ZZ", "", "US:", "US:on", "Rhode Island"])
def test_rejects_anything_else_naming_the_accepted_forms(given: str) -> None:
    with pytest.raises(ValueError, match="US:44"):
        _normalise_wqp_state(given)


def test_fetch_raw_sends_the_fips_form(client: FakeClient) -> None:
    WQPCollector(client=client).fetch_raw(state_code="RI")
    assert client.calls[0][1]["statecode"] == "US:44"


def test_fetch_raw_rejects_a_bad_state_before_any_request(client: FakeClient) -> None:
    with pytest.raises(ValueError):
        WQPCollector(client=client).fetch_raw(state_code="ZZ")
    assert client.calls == []
