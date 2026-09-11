"""Pydantic models for ``benchmarks.camels_benchmark`` results.

These models are the single source of truth for the ``results.json`` shape:
``build_results`` validates every harness output through ``Results`` and returns
a canonical ``model_dump(mode="json")`` dict, and ``Results.model_json_schema()``
generates the committed ``benchmarks/results.schema.json`` that external
consumers validate against. Together they replace a hand-written JSON Schema:
the Python types are the contract, the schema file is a derived artifact.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.1"
SchemaVersion = Literal[SCHEMA_VERSION]

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
ReturnPeriod = Literal[2, 5, 10, 25, 50, 100]

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class _Base(BaseModel):
    """Every model tolerates exactly the declared fields."""

    model_config = ConfigDict(extra="forbid")


class Tolerance(_Base):
    value: float
    unit: str
    rationale: str


class BenchmarkMetadata(_Base):
    schema_version: SchemaVersion
    generated: date
    aquascope_version: str
    note: str
    timings_not_asserted: Literal[True]
    tolerances: dict[str, Tolerance]


class PublishedCheck(_Base):
    metric: str
    computed: float
    published: float
    error_type: Literal["relative", "absolute", "circular_months"]
    error: float
    tolerance: float
    check_passes: bool


class IntegrityCheck(_Base):
    metric: Literal["order", "runoff_ratio", "recession_constant", "fdc_slope", "complete"]
    detail: str
    check_passes: bool


class SignatureValues(_Base):
    q_mean: float
    q5: float
    q95: float
    median_flow: float
    runoff_ratio: float | None
    mean_recession_constant: float
    fdc_slope: float
    peak_month: int = Field(ge=1, le=12)
    baseflow_index: float


class Signatures(_Base):
    checks: list[PublishedCheck]
    values: SignatureValues
    integrity: list[IntegrityCheck]
    seconds: float = Field(ge=0)


class BaseflowMethod(_Base):
    bfi: float
    published_bfi: float
    seconds: float = Field(ge=0)
    check: PublishedCheck


class BaseflowMethods(_Base):
    lyne_hollick: BaseflowMethod
    eckhardt: BaseflowMethod


class Baseflow(_Base):
    methods: BaseflowMethods
    seconds: float = Field(ge=0)


class Fit(_Base):
    return_period: ReturnPeriod
    computed_m3s: float
    reference_m3s: float
    relative_error_pct: float
    check_passes: bool
    classification: Literal["implementation", "data_limitation"]


class FfaMethod(_Base):
    reference_key: str
    reference_mle_unstable: bool
    mean_relative_error_pct: float
    max_relative_error_pct: float
    warnings: list[str]
    seconds: float = Field(ge=0)
    fits: list[Fit]


class FfaMethods(_Base):
    gev: FfaMethod
    gev_lmoments: FfaMethod
    lp3: FfaMethod


class FloodFrequency(_Base):
    methods: FfaMethods
    seconds: float = Field(ge=0)


class CatchmentTimings(_Base):
    signatures_s: float = Field(ge=0)
    baseflow_s: float = Field(ge=0)
    flood_frequency_s: float = Field(ge=0)
    total_s: float = Field(ge=0)


class Catchment(_Base):
    name: str
    climate: str
    signatures: Signatures
    baseflow: Baseflow
    flood_frequency: FloodFrequency
    timings: CatchmentTimings


class Gates(_Base):
    q_mean_nrmse_pct: float
    q_mean_nrmse_tolerance_pct: float
    q_mean_gate_met: bool
    bfi_pbias_pct: float
    bfi_pbias_tolerance_pct: float
    bfi_gate_met: bool
    ffa_cross_method_mean_relative_error_pct: float
    ffa_tolerance_pct: float
    ffa_gate_met: bool


class Metrics(_Base):
    rmse: float | None
    pbias: float | None
    r2: float | None


class SummaryTimings(_Base):
    total_seconds: float = Field(ge=0)


class Summary(_Base):
    n_catchments: int = Field(ge=0)
    n_unmet: int = Field(ge=0)
    n_data_limitation_findings: int = Field(ge=0)
    n_integrity_failures: int = Field(ge=0)
    gates: Gates
    aggregate: dict[str, Metrics]
    timings: SummaryTimings


class SoftwareInfo(_Base):
    version: str
    author: str
    doi: str
    citation: str


class Results(_Base):
    """Top-level ``results.json`` document (the CAMELS benchmark contract)."""

    model_config = ConfigDict(
        extra="forbid",
        title="CAMELS benchmark results",
        description=(
            "Top-level output of benchmarks.camels_benchmark.build_results(). results.json "
            "is the single source of truth; results.md and results.html are renderings of "
            "it. This schema, generated from these Pydantic models, is the contract any "
            "consumer can validate against."
        ),
    )

    metadata: BenchmarkMetadata
    catchments: dict[str, Catchment]
    summary: Summary
    software: SoftwareInfo


def results_json_schema() -> dict:
    """The JSON Schema for ``Results``, with its dialect declared.

    Committed verbatim as ``benchmarks/results.schema.json``; a drift-guard test
    keeps the committed file identical to this output.
    """
    return {"$schema": JSON_SCHEMA_DIALECT, **Results.model_json_schema()}
