"""Country-scale agricultural water benchmarking using AQUASTAT data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkMetric:
    """Definition of a supported AQUASTAT benchmark metric."""

    id: str
    name: str
    numerator: str
    denominator: str
    description: str
    output_unit: str
    percent: bool = False
    convert_to_m3_per_ha: bool = False


@dataclass
class AgricultureBenchmarkResult:
    """Structured result from an AQUASTAT benchmarking workflow."""

    metric_id: str
    metric_name: str
    output_unit: str
    summary: str
    table: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "output_unit": self.output_unit,
            "summary": self.summary,
            "table": self.table.to_dict("records"),
        }


BENCHMARK_METRICS: dict[str, BenchmarkMetric] = {
    "agricultural_withdrawal_per_irrigated_area": BenchmarkMetric(
        id="agricultural_withdrawal_per_irrigated_area",
        name="Agricultural Withdrawal per Irrigated Area",
        numerator="Agricultural water withdrawal",
        denominator="Total area equipped for irrigation",
        description="Benchmark agricultural water withdrawal intensity using irrigated area as the denominator.",
        output_unit="m3/ha",
        convert_to_m3_per_ha=True,
    ),
    "agricultural_withdrawal_share_pct": BenchmarkMetric(
        id="agricultural_withdrawal_share_pct",
        name="Agricultural Withdrawal Share",
        numerator="Agricultural water withdrawal",
        denominator="Total water withdrawal",
        description="Measure the share of total withdrawals allocated to agriculture.",
        output_unit="%",
        percent=True,
    ),
    "withdrawal_pressure_on_renewable_resources_pct": BenchmarkMetric(
        id="withdrawal_pressure_on_renewable_resources_pct",
        name="Withdrawal Pressure on Renewable Resources",
        numerator="Total water withdrawal",
        denominator="Total renewable water resources",
        description="Measure overall withdrawal pressure relative to renewable water resources.",
        output_unit="%",
        percent=True,
    ),
}


def list_benchmark_metrics() -> list[str]:
    """Return supported benchmark metric IDs."""
    return sorted(BENCHMARK_METRICS)


def _first_non_empty(values: pd.Series) -> str:
    """Return the first non-empty string in a series."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _normalise_aquastat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw or saved AQUASTAT data into a standard table."""
    import pandas as pd

    rename_map = {
        "Area": "country",
        "Area Code": "country_code",
        "Year": "year",
        "Element": "variable",
        "Value": "value",
        "Unit": "unit",
    }
    data = df.rename(columns=rename_map).copy()

    required = {"country", "year", "variable", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"AQUASTAT data is missing required columns: {sorted(missing)}")

    if "country_code" not in data.columns:
        data["country_code"] = data["country"]
    if "unit" not in data.columns:
        data["unit"] = ""

    cols = ["country", "country_code", "year", "variable", "value", "unit"]
    data = data[cols].copy()
    data["country"] = data["country"].astype(str)
    data["country_code"] = data["country_code"].astype(str)
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["unit"] = data["unit"].fillna("").astype(str)
    data = data.dropna(subset=["year", "value"])
    data["year"] = data["year"].astype(int)
    return data


def _units_support_m3_per_ha_conversion(numerator_unit: str, denominator_unit: str) -> bool:
    """Check whether common AQUASTAT units can be converted to m3/ha."""
    num = numerator_unit.lower().replace(" ", "")
    den = denominator_unit.lower().replace(" ", "")
    return "10^9m3" in num and "1000ha" in den


def benchmark_aquastat(
    df: pd.DataFrame,
    metric_id: str,
    *,
    year: int | None = None,
    countries: list[str] | None = None,
    latest_only: bool = True,
    top_n: int | None = None,
) -> AgricultureBenchmarkResult:
    """Compute a country-scale benchmark from AQUASTAT data."""
    import pandas as pd

    metric = BENCHMARK_METRICS.get(metric_id)
    if metric is None:
        raise ValueError(f"Unknown benchmark metric {metric_id!r}. Available: {list_benchmark_metrics()}")

    data = _normalise_aquastat_dataframe(df)

    if countries:
        wanted = {country.strip().lower() for country in countries if country.strip()}
        data = data[
            data["country"].str.lower().isin(wanted) | data["country_code"].str.lower().isin(wanted)
        ].copy()

    if year is not None:
        data = data[data["year"] == year].copy()

    if data.empty:
        raise ValueError("No AQUASTAT records match the requested filters.")

    units = data.groupby("variable", dropna=False)["unit"].agg(_first_non_empty).to_dict()

    pivot = data.pivot_table(
        index=["country", "country_code", "year"],
        columns="variable",
        values="value",
        aggfunc="mean",
    ).reset_index()

    missing_cols = [column for column in (metric.numerator, metric.denominator) if column not in pivot.columns]
    if missing_cols:
        raise ValueError(f"AQUASTAT data does not contain variables needed for {metric_id!r}: {missing_cols}")

    table = pivot[["country", "country_code", "year", metric.numerator, metric.denominator]].copy()
    denominator = pd.to_numeric(table[metric.denominator], errors="coerce")
    denominator = denominator.where(denominator != 0)
    numerator = pd.to_numeric(table[metric.numerator], errors="coerce")
    metric_values = numerator / denominator

    output_unit = metric.output_unit
    if metric.percent:
        metric_values = metric_values * 100.0
    elif metric.convert_to_m3_per_ha and _units_support_m3_per_ha_conversion(
        units.get(metric.numerator, ""),
        units.get(metric.denominator, ""),
    ):
        metric_values = metric_values * 1_000_000.0
    elif metric.convert_to_m3_per_ha:
        output_unit = "ratio_from_source_units"

    table["metric_value"] = metric_values
    table = table.dropna(subset=["metric_value"]).copy()
    if table.empty:
        raise ValueError(f"No complete numerator/denominator pairs available for {metric_id!r}.")

    table["metric_value"] = table["metric_value"].round(4)
    table[metric.numerator] = pd.to_numeric(table[metric.numerator], errors="coerce").round(4)
    table[metric.denominator] = pd.to_numeric(table[metric.denominator], errors="coerce").round(4)

    if latest_only and year is None:
        latest = table.sort_values(["country_code", "year"]).groupby("country_code", dropna=False).tail(1)
        table = latest.copy()

    table = table.sort_values(["metric_value", "year", "country"], ascending=[False, False, True]).reset_index(drop=True)
    if top_n is not None:
        table = table.head(top_n).reset_index(drop=True)

    scope = f"year {year}" if year is not None else ("latest year per country" if latest_only else "all available years")
    summary = (
        f"Computed {metric.name.lower()} for {len(table)} records using {scope}. "
        f"Metric unit: {output_unit}."
    )

    return AgricultureBenchmarkResult(
        metric_id=metric.id,
        metric_name=metric.name,
        output_unit=output_unit,
        summary=summary,
        table=table,
    )
