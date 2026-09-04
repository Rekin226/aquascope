"""The sufficiency registry: what a record supports, said in plain words."""

from __future__ import annotations

from aquascope.methods import (
    DEFENSIBLE,
    MARGINAL,
    METHODS,
    NOT_DEFENSIBLE,
    SiteContext,
    assess_method,
    describe_preconditions,
    method_ids,
    sufficiency_table,
)


def _gauged(years: float, **kw) -> SiteContext:
    return SiteContext(years_by_variable={"discharge": years}, resolution_by_variable={"discharge": "daily"}, **kw)


def test_registry_is_data_and_every_entry_names_a_tool():
    assert len(METHODS) >= 15
    assert all(m.tool for m in METHODS.values())
    assert "at_site_flood_frequency" in method_ids("flood_risk")
    assert "spi" not in method_ids("flood_risk")
    rows = describe_preconditions()
    assert rows[0]["id"] and "min_years" in rows[0]


def test_flood_frequency_thresholds_and_return_period_cap():
    assert assess_method("at_site_flood_frequency", _gauged(39))["status"] == DEFENSIBLE
    marginal = assess_method("at_site_flood_frequency", _gauged(12))
    assert marginal["status"] == MARGINAL and "12 years" in marginal["reason"]
    short = assess_method("at_site_flood_frequency", _gauged(8))
    assert short["status"] == NOT_DEFENSIBLE and "floor" in short["reason"]
    capped = assess_method("at_site_flood_frequency", _gauged(25, return_period=200))
    assert capped["status"] == MARGINAL and "T = 200" in capped["reason"]
    assert assess_method("at_site_flood_frequency", _gauged(39, return_period=100))["status"] == DEFENSIBLE


def test_ungauged_site_routes_to_transfer_methods():
    ctx = SiteContext(donors=5, available={"glofas"})
    assert ctx.ungauged
    assert assess_method("at_site_flood_frequency", ctx)["status"] == NOT_DEFENSIBLE
    assert assess_method("regionalize_signatures", ctx)["status"] == DEFENSIBLE
    assert assess_method("glofas_cross_check", ctx)["status"] == DEFENSIBLE
    few = assess_method("regionalize_signatures", SiteContext(donors=2))
    assert few["status"] == NOT_DEFENSIBLE and "donor" in few["reason"]


def test_resolution_area_and_inputs_gate_methods():
    monthly = SiteContext(years_by_variable={"discharge": 30}, resolution_by_variable={"discharge": "monthly"})
    assert assess_method("at_site_flood_frequency", monthly)["status"] == NOT_DEFENSIBLE
    assert assess_method("trend_mann_kendall", monthly)["status"] == DEFENSIBLE
    big = _gauged(30, area_km2=101_033, available={"forcing"})
    gr4j = assess_method("gr4j_calibration", big)
    assert gr4j["status"] == NOT_DEFENSIBLE and "ceiling" in gr4j["reason"]
    assert assess_method("gr4j_calibration", _gauged(30, area_km2=900, available={"forcing"}))["status"] == DEFENSIBLE
    assert assess_method("gr4j_calibration", _gauged(30, area_km2=900))["status"] == NOT_DEFENSIBLE
    precip = SiteContext(years_by_variable={"precipitation": 35}, resolution_by_variable={"precipitation": "monthly"})
    assert assess_method("spi", precip)["status"] == DEFENSIBLE
    assert assess_method("spei", precip)["status"] == NOT_DEFENSIBLE
    precip.available.add("temperature")
    assert assess_method("spei", precip)["status"] == DEFENSIBLE


def test_sufficiency_table_orders_defensible_first():
    rows = sufficiency_table(_gauged(39, return_period=100), problem="flood_risk")
    assert rows[0]["status"] == DEFENSIBLE
    statuses = [r["status"] for r in rows]
    assert statuses == sorted(statuses, key=[DEFENSIBLE, MARGINAL, NOT_DEFENSIBLE].index)
    assert {r["method"] for r in rows} == set(method_ids("flood_risk"))
    assert all("label" in r and "reason" in r for r in rows)
