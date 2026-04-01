"""WaPOR-based agricultural water productivity workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aquascope.agri.benchmark import AgricultureBenchmarkResult, benchmark_aquastat
from aquascope.collectors.wapor import WAPOR_VARIABLES

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_DATE_COLUMNS = ("date", "sample_datetime", "reading_datetime", "datetime", "time")
_START_COLUMNS = ("start_date", "startDate", "time_start", "_requested_start_date")
_END_COLUMNS = ("end_date", "endDate", "time_end", "_requested_end_date")
_VALUE_COLUMNS = ("value", "eto_mm", "mean", "summaryValue", "rasterValue")
_UNIT_COLUMNS = ("unit", "units", "measureUnit")
_BBOX_COLUMNS = ("bbox", "_bbox")
_AOI_COLUMNS = ("aoi_id", "aoiId")
_LEVEL_COLUMNS = ("level", "rasterLevel", "cube_level")
_CODE_COLUMNS = ("cube_code", "variable", "_cube_code", "code")
_LABEL_COLUMNS = ("cube_label", "label", "caption", "_cube_label")
_STAT_COLUMNS = ("statistic", "aggregation", "measure")
_MERGE_KEYS = ["date", "start_date", "end_date", "bbox", "aoi_id", "level"]
_DEFAULT_AQUASTAT_CONTEXT_METRICS = (
    "agricultural_withdrawal_share_pct",
    "agricultural_withdrawal_per_irrigated_area",
)


@dataclass(frozen=True)
class ProductivityMetric:
    """Definition of a supported WaPOR productivity metric."""

    id: str
    name: str
    numerator: str
    denominator: str
    description: str
    output_unit: str
    percent: bool = False
    convert_gm2_per_mm_to_kgm3: bool = False


@dataclass
class WaPORProductivityResult:
    """Structured result from a WaPOR productivity workflow."""

    metric_id: str
    metric_name: str
    output_unit: str
    aggregate_value: float
    summary: str
    table: pd.DataFrame
    aquastat_context: list[AgricultureBenchmarkResult] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "output_unit": self.output_unit,
            "aggregate_value": self.aggregate_value,
            "summary": self.summary,
            "table": self.table.to_dict("records"),
            "aquastat_context": [context.to_dict() for context in (self.aquastat_context or [])],
        }


PRODUCTIVITY_METRICS: dict[str, ProductivityMetric] = {
    "biomass_water_productivity": ProductivityMetric(
        id="biomass_water_productivity",
        name="Biomass Water Productivity",
        numerator="NPP",
        denominator="AETI",
        description="Estimate biomass production per unit of actual evapotranspiration from WaPOR outputs.",
        output_unit="kg/m3",
        convert_gm2_per_mm_to_kgm3=True,
    ),
    "relative_evapotranspiration_pct": ProductivityMetric(
        id="relative_evapotranspiration_pct",
        name="Relative Evapotranspiration",
        numerator="AETI",
        denominator="RET",
        description="Measure actual evapotranspiration relative to reference evapotranspiration.",
        output_unit="%",
        percent=True,
    ),
    "biomass_per_reference_et": ProductivityMetric(
        id="biomass_per_reference_et",
        name="Biomass per Reference ET",
        numerator="NPP",
        denominator="RET",
        description="Compare biomass production against reference evapotranspiration demand.",
        output_unit="kg/m3",
        convert_gm2_per_mm_to_kgm3=True,
    ),
}


def list_productivity_metrics() -> list[str]:
    """Return supported productivity metric IDs."""
    return sorted(PRODUCTIVITY_METRICS)


def _empty_series(length: int):
    import pandas as pd

    return pd.Series([None] * length, dtype="object")


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _coerce_iso_dates(values) -> pd.Series:
    import pandas as pd

    parsed = pd.to_datetime(values, errors="coerce")
    formatter = parsed.dt.strftime if hasattr(parsed, "dt") else parsed.strftime
    result = pd.Series(formatter("%Y-%m-%d"), index=getattr(values, "index", None), dtype="object")
    return result.where(parsed.notna(), None)


def _normalise_bbox(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            return None
        try:
            return ",".join(f"{float(part):g}" for part in value)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        return text
    try:
        return ",".join(f"{float(part):g}" for part in parts)
    except ValueError:
        return text


def _normalise_unit_text(unit: str | None) -> str:
    return str(unit or "").lower().replace(" ", "").replace("²", "2")


def _supports_kg_per_m3_conversion(numerator_unit: str | None, denominator_unit: str | None) -> bool:
    num = _normalise_unit_text(numerator_unit)
    den = _normalise_unit_text(denominator_unit)
    return any(token in num for token in ("g/m2", "gm-2", "g.m-2")) and "mm" in den


def _first_non_empty(values: pd.Series) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _normalise_wapor_dataframe(df: pd.DataFrame, *, cube_code: str) -> pd.DataFrame:
    """Normalize a saved WaPOR CSV/JSON file into a standard table."""
    import pandas as pd

    data = df.copy()
    if data.empty:
        raise ValueError(f"WaPOR {cube_code} data is empty.")

    value_col = _first_column(data, _VALUE_COLUMNS)
    if value_col is None:
        raise ValueError(f"Could not find a numeric value column for WaPOR {cube_code} data.")

    date_col = _first_column(data, _DATE_COLUMNS)
    start_col = _first_column(data, _START_COLUMNS)
    end_col = _first_column(data, _END_COLUMNS)
    label_col = _first_column(data, _LABEL_COLUMNS)
    unit_col = _first_column(data, _UNIT_COLUMNS)
    bbox_col = _first_column(data, _BBOX_COLUMNS)
    aoi_col = _first_column(data, _AOI_COLUMNS)
    level_col = _first_column(data, _LEVEL_COLUMNS)
    stat_col = _first_column(data, _STAT_COLUMNS)

    date_series = _coerce_iso_dates(data[date_col]) if date_col else _empty_series(len(data))
    start_series = _coerce_iso_dates(data[start_col]) if start_col else date_series.copy()
    end_series = _coerce_iso_dates(data[end_col]) if end_col else date_series.copy()

    normalised = pd.DataFrame(
        {
            "cube_code": cube_code,
            "cube_label": data[label_col].astype(str) if label_col else WAPOR_VARIABLES.get(cube_code, cube_code),
            "date": date_series,
            "start_date": start_series,
            "end_date": end_series,
            "bbox": data[bbox_col].map(_normalise_bbox) if bbox_col else _empty_series(len(data)),
            "aoi_id": data[aoi_col].astype(str) if aoi_col else _empty_series(len(data)),
            "level": data[level_col].astype(str) if level_col else _empty_series(len(data)),
            "value": pd.to_numeric(data[value_col], errors="coerce"),
            "unit": data[unit_col].astype(str) if unit_col else _empty_series(len(data)),
            "statistic": data[stat_col].astype(str) if stat_col else _empty_series(len(data)),
        }
    )
    normalised = normalised.dropna(subset=["value"]).reset_index(drop=True)
    if normalised.empty:
        raise ValueError(f"WaPOR {cube_code} data does not contain usable values.")
    return normalised


def _prepare_frames(
    *,
    aeti_df: pd.DataFrame | None,
    npp_df: pd.DataFrame | None,
    ret_df: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if aeti_df is not None:
        frames["AETI"] = _normalise_wapor_dataframe(aeti_df, cube_code="AETI")
    if npp_df is not None:
        frames["NPP"] = _normalise_wapor_dataframe(npp_df, cube_code="NPP")
    if ret_df is not None:
        frames["RET"] = _normalise_wapor_dataframe(ret_df, cube_code="RET")
    return frames


def _merge_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge normalized WaPOR frames on shared period and AOI keys."""

    prepared: list[tuple[str, pd.DataFrame]] = []
    for code, frame in frames.items():
        renamed = frame.copy()
        renamed = renamed.rename(
            columns={
                "value": f"{code}_value",
                "unit": f"{code}_unit",
                "cube_label": f"{code}_label",
                "statistic": f"{code}_statistic",
            }
        )
        prepared.append((code, renamed))

    join_keys = [key for key in _MERGE_KEYS if all(frame[key].notna().any() for _, frame in prepared)]
    if not join_keys:
        if all(len(frame) == 1 for _, frame in prepared):
            join_keys = ["_merge_key"]
            prepared = [(code, frame.assign(_merge_key=0)) for code, frame in prepared]
        else:
            raise ValueError("Could not align WaPOR files. Use files with matching dates or period columns.")

    merged: pd.DataFrame | None = None
    keep_columns = join_keys.copy()
    for code in frames:
        keep_columns.extend([f"{code}_value", f"{code}_unit", f"{code}_label", f"{code}_statistic"])

    for _, frame in prepared:
        frame_keep = frame[[column for column in keep_columns if column in frame.columns]].copy()
        if merged is None:
            merged = frame_keep
        else:
            merged = merged.merge(frame_keep, on=join_keys, how="inner")

    if merged is None or merged.empty:
        raise ValueError("No overlapping WaPOR periods were found across the provided files.")

    if "date" in merged.columns:
        merged = merged.sort_values("date", kind="stable")
    elif "start_date" in merged.columns:
        merged = merged.sort_values("start_date", kind="stable")
    return merged.reset_index(drop=True)


def _build_aquastat_context(
    aquastat_df: pd.DataFrame,
    *,
    metrics: list[str] | None,
    year: int | None,
    countries: list[str] | None,
    top_n: int | None,
) -> list[AgricultureBenchmarkResult]:
    """Build optional AQUASTAT benchmark tables for a productivity report."""
    requested_metrics = metrics if metrics is not None else list(_DEFAULT_AQUASTAT_CONTEXT_METRICS)
    explicit_metrics = metrics is not None

    context: list[AgricultureBenchmarkResult] = []
    for metric_id in requested_metrics:
        try:
            context.append(
                benchmark_aquastat(
                    aquastat_df,
                    metric_id,
                    year=year,
                    countries=countries,
                    latest_only=year is None,
                    top_n=top_n,
                )
            )
        except ValueError as exc:
            if explicit_metrics:
                raise
            logger.info("Skipping AQUASTAT context metric %s: %s", metric_id, exc)
    return context


def estimate_wapor_productivity(
    *,
    metric_id: str,
    aeti_df: pd.DataFrame | None = None,
    npp_df: pd.DataFrame | None = None,
    ret_df: pd.DataFrame | None = None,
    aquastat_df: pd.DataFrame | None = None,
    aquastat_metrics: list[str] | None = None,
    aquastat_year: int | None = None,
    aquastat_countries: list[str] | None = None,
    aquastat_top_n: int | None = 10,
) -> WaPORProductivityResult:
    """Compute a WaPOR productivity or ET performance metric."""
    import pandas as pd

    metric = PRODUCTIVITY_METRICS.get(metric_id)
    if metric is None:
        raise ValueError(f"Unknown productivity metric {metric_id!r}. Available: {list_productivity_metrics()}")

    frames = _prepare_frames(aeti_df=aeti_df, npp_df=npp_df, ret_df=ret_df)
    missing = [code for code in (metric.numerator, metric.denominator) if code not in frames]
    if missing:
        raise ValueError(
            f"Metric {metric_id!r} requires WaPOR files for {missing}. "
            "Provide the matching --aeti-file, --npp-file, or --ret-file inputs."
        )

    merged = _merge_frames({code: frames[code] for code in sorted({metric.numerator, metric.denominator})})
    numerator = pd.to_numeric(merged[f"{metric.numerator}_value"], errors="coerce")
    denominator = pd.to_numeric(merged[f"{metric.denominator}_value"], errors="coerce")
    denominator = denominator.where(denominator != 0)

    metric_values = numerator / denominator
    output_unit = metric.output_unit
    numerator_unit = _first_non_empty(merged[f"{metric.numerator}_unit"])
    denominator_unit = _first_non_empty(merged[f"{metric.denominator}_unit"])

    if metric.percent:
        metric_values = metric_values * 100.0
    elif metric.convert_gm2_per_mm_to_kgm3 and not _supports_kg_per_m3_conversion(numerator_unit, denominator_unit):
        output_unit = "ratio_from_source_units"

    table = merged.copy()
    table["metric_value"] = metric_values
    table = table.dropna(subset=["metric_value"]).reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No valid numerator/denominator pairs were available for {metric_id!r}.")

    aggregate_value = float(numerator.sum() / denominator.sum())
    if metric.percent:
        aggregate_value = aggregate_value * 100.0

    pretty_table = table.rename(
        columns={
            f"{metric.numerator}_value": metric.numerator,
            f"{metric.denominator}_value": metric.denominator,
            f"{metric.numerator}_unit": f"{metric.numerator}_unit",
            f"{metric.denominator}_unit": f"{metric.denominator}_unit",
        }
    )
    pretty_table[metric.numerator] = pd.to_numeric(pretty_table[metric.numerator], errors="coerce").round(4)
    pretty_table[metric.denominator] = pd.to_numeric(pretty_table[metric.denominator], errors="coerce").round(4)
    pretty_table["metric_value"] = pd.to_numeric(pretty_table["metric_value"], errors="coerce").round(4)

    aquastat_context = None
    if aquastat_df is not None:
        aquastat_context = _build_aquastat_context(
            aquastat_df,
            metrics=aquastat_metrics,
            year=aquastat_year,
            countries=aquastat_countries,
            top_n=aquastat_top_n,
        )

    summary = (
        f"Computed {metric.name.lower()} across {len(pretty_table)} aligned WaPOR periods. "
        f"Aggregate value: {aggregate_value:.4f} {output_unit}."
    )
    if aquastat_context:
        summary += f" Added {len(aquastat_context)} AQUASTAT benchmark table(s) for country context."

    return WaPORProductivityResult(
        metric_id=metric.id,
        metric_name=metric.name,
        output_unit=output_unit,
        aggregate_value=round(aggregate_value, 4),
        summary=summary,
        table=pretty_table,
        aquastat_context=aquastat_context,
    )
