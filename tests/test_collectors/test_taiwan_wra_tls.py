"""TLS-configuration tests for the Taiwan WRA (taiwan_wra.py) collectors (#169).

``opendata.wra.gov.tw`` (``WRA_BASE``) chains to the same TWCA-issued cert
type as ``iot.wra.gov.tw``: certs missing the Subject Key Identifier
extension, which Python 3.13+ rejects under its default strict profile.
Confirmed via direct request against the live host on 2026-08-31 (`SSL:
CERTIFICATE_VERIFY_FAILED, Missing Subject Key Identifier`).

These collectors previously used ``verify=False`` against ``WRA_BASE``,
which disabled certificate verification outright — the exact anti-pattern
#169 rules out. They now use ``relax_strict_tls=True`` instead, which drops
only the strict-profile check; full chain and hostname verification stay on.

``gweb.wra.gov.tw`` (``GWEB_BASE``) is a different host whose chain verifies
cleanly under the strict profile (confirmed 2026-08-31: plain 403, no TLS
error) — its client needs neither ``relax_strict_tls`` nor ``verify=False``.
"""

from __future__ import annotations

import ssl
import sys

import pytest

from aquascope.collectors.taiwan_wra import (
    GWEB_BASE,
    WRA_BASE,
    TaiwanWRAGroundwaterCollector,
    TaiwanWRAGroundwaterDailyCollector,
    TaiwanWRAReservoirCollector,
    TaiwanWRAWaterLevelCollector,
)


def _ssl_context(collector):
    return collector.client._client._transport._pool._ssl_context


class TestWRABaseCollectorsRelaxStrictTLS:
    """WaterLevel, Reservoir, and Groundwater all point at WRA_BASE and must
    relax only the strict X.509 profile check, never disable verification."""

    @pytest.mark.parametrize(
        "collector_cls",
        [TaiwanWRAWaterLevelCollector, TaiwanWRAReservoirCollector, TaiwanWRAGroundwaterCollector],
    )
    def test_relaxes_only_strict_profile(self, collector_cls):
        collector = collector_cls()
        ctx = _ssl_context(collector)

        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True
        assert not ctx.verify_flags & ssl.VERIFY_X509_STRICT

    @pytest.mark.parametrize(
        "collector_cls",
        [TaiwanWRAWaterLevelCollector, TaiwanWRAReservoirCollector, TaiwanWRAGroundwaterCollector],
    )
    def test_points_at_wra_base(self, collector_cls):
        collector = collector_cls()
        assert collector.client.base_url == WRA_BASE

    @pytest.mark.skipif(
        sys.version_info < (3, 13),
        reason="VERIFY_X509_STRICT only defaults on in Python 3.13+; the bug "
        "this guards against can't reproduce on older interpreters.",
    )
    @pytest.mark.parametrize(
        "collector_cls",
        [TaiwanWRAWaterLevelCollector, TaiwanWRAReservoirCollector, TaiwanWRAGroundwaterCollector],
    )
    def test_never_disables_verification_outright(self, collector_cls):
        # Hard requirement from #169: verify=False is not an acceptable fix,
        # anywhere. These previously set verify=False directly.
        collector = collector_cls()
        ctx = _ssl_context(collector)
        assert ctx.verify_mode != ssl.CERT_NONE


class TestGwebBaseCollectorDoesNotNeedRelaxation:
    """GWEB_BASE's chain verifies fine under the strict profile — its client
    should keep full default verification, no relax_strict_tls needed."""

    def test_points_at_gweb_base(self):
        collector = TaiwanWRAGroundwaterDailyCollector()
        assert collector.client.base_url == GWEB_BASE

    @pytest.mark.skipif(
        sys.version_info < (3, 13),
        reason="VERIFY_X509_STRICT only defaults on in Python 3.13+; there's "
        "nothing to assert about it being 'kept on' on older interpreters.",
    )
    def test_keeps_default_strict_profile(self):
        collector = TaiwanWRAGroundwaterDailyCollector()
        ctx = _ssl_context(collector)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True
        # Unlike the WRA_BASE collectors, this one should NOT relax the
        # strict profile — gweb's chain doesn't need it.
        assert ctx.verify_flags & ssl.VERIFY_X509_STRICT

    @pytest.mark.skipif(
        sys.version_info < (3, 13),
        reason="VERIFY_X509_STRICT only defaults on in Python 3.13+.",
    )
    def test_never_disables_verification_outright(self):
        collector = TaiwanWRAGroundwaterDailyCollector()
        ctx = _ssl_context(collector)
        assert ctx.verify_mode != ssl.CERT_NONE
