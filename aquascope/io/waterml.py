"""WaterML 2.0 read/write support for AquaScope.

WaterML 2.0 is an OGC/ISO 19156-based standard for encoding hydrological
time-series data.  This module uses only the Python standard library
``xml.etree.ElementTree`` parser — no extra dependencies are required.

References
----------
- OGC WaterML 2.0: https://www.ogc.org/standards/waterml
- Schema: http://schemas.opengis.net/waterml/2.0/
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── XML namespace constants ──────────────────────────────────────────────────
WML2_NS = "http://www.opengis.net/waterml/2.0"
GML_NS = "http://www.opengis.net/gml/3.2"
OM_NS = "http://www.opengis.net/om/2.0"
SAMS_NS = "http://www.opengis.net/samplingSpatial/2.0"
SA_NS = "http://www.opengis.net/sampling/2.0"

_NS = {
    "wml2": WML2_NS,
    "gml": GML_NS,
    "om": OM_NS,
    "sams": SAMS_NS,
    "sa": SA_NS,
}


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class WaterMLTimeSeries:
    """Parsed WaterML 2.0 time series.

    Parameters
    ----------
    station_id : str
        Unique identifier for the monitoring station.
    station_name : str
        Human-readable station name.
    parameter : str
        Observed property (e.g. ``"Discharge"``).
    unit : str
        Unit of measurement (e.g. ``"m3/s"``).
    latitude : float | None
        WGS-84 latitude of the station.
    longitude : float | None
        WGS-84 longitude of the station.
    timestamps : list[datetime]
        Observation timestamps.
    values : list[float]
        Observed numeric values.
    quality_codes : list[str]
        Per-value quality flags (empty string when absent).
    metadata : dict[str, str]
        Arbitrary key/value metadata.
    """

    station_id: str
    station_name: str
    parameter: str
    unit: str
    latitude: float | None = None
    longitude: float | None = None
    timestamps: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    quality_codes: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


# ── Read ─────────────────────────────────────────────────────────────────────
def read_waterml(path: str | Path) -> list[WaterMLTimeSeries]:
    """Read a WaterML 2.0 XML file.

    Parses ``<wml2:MeasurementTimeseries>`` elements and extracts the
    sampling point, observed property, unit, time-value pairs, and quality
    flags.  A single WaterML file may contain multiple time series.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the WaterML 2.0 XML file.

    Returns
    -------
    list[WaterMLTimeSeries]
        One entry per ``<wml2:observationMember>`` in the document.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ET.ParseError
        If the XML is malformed.
    """
    path = Path(path)
    tree = ET.parse(path)  # noqa: S314
    root = tree.getroot()

    results: list[WaterMLTimeSeries] = []
    members = root.findall("wml2:observationMember", _NS)

    for member in members:
        obs = member.find("om:OM_Observation", _NS)
        if obs is None:
            continue

        # ── station info ──
        station_id, station_name, lat, lon = _parse_feature_of_interest(obs)

        # ── observed property ──
        prop_el = obs.find("om:observedProperty", _NS)
        parameter = (prop_el.get("title") or prop_el.get("href", "unknown")) if prop_el is not None else "unknown"

        # ── result timeseries ──
        mts = obs.find(".//wml2:MeasurementTimeseries", _NS)
        if mts is None:
            continue

        unit = _parse_default_unit(mts)
        timestamps, values, quality_codes = _parse_points(mts)

        # ── metadata ──
        metadata: dict[str, str] = {}
        desc = root.find("gml:description", _NS)
        if desc is not None and desc.text:
            metadata["description"] = desc.text

        results.append(
            WaterMLTimeSeries(
                station_id=station_id,
                station_name=station_name,
                parameter=parameter,
                unit=unit,
                latitude=lat,
                longitude=lon,
                timestamps=timestamps,
                values=values,
                quality_codes=quality_codes,
                metadata=metadata,
            )
        )

    logger.debug("read_waterml: parsed %d time series from %s", len(results), path)
    return results


def _parse_feature_of_interest(obs: ET.Element) -> tuple[str, str, float | None, float | None]:
    """Extract station ID, name, and coordinates from *om:featureOfInterest*."""
    station_id = "unknown"
    station_name = "unknown"
    lat: float | None = None
    lon: float | None = None

    foi = obs.find("om:featureOfInterest", _NS)
    if foi is None:
        return station_id, station_name, lat, lon

    sf = foi.find("sams:SF_SpatialSamplingFeature", _NS)
    if sf is None:
        return station_id, station_name, lat, lon

    gml_id = sf.get(f"{{{GML_NS}}}id")
    if gml_id:
        station_id = gml_id

    name_el = sf.find("gml:name", _NS)
    if name_el is not None and name_el.text:
        station_name = name_el.text

    pos_el = sf.find(".//gml:pos", _NS)
    if pos_el is not None and pos_el.text:
        parts = pos_el.text.strip().split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
            except ValueError:
                pass

    return station_id, station_name, lat, lon


def _parse_default_unit(mts: ET.Element) -> str:
    """Extract the default unit from *wml2:defaultPointMetadata*."""
    uom_el = mts.find(".//wml2:defaultPointMetadata/wml2:DefaultTVPMeasurementMetadata/wml2:uom", _NS)
    if uom_el is not None:
        return uom_el.get("code", uom_el.get("title", "unknown"))
    return "unknown"


def _parse_points(mts: ET.Element) -> tuple[list[datetime], list[float], list[str]]:
    """Parse ``wml2:point`` children into parallel timestamp / value / quality lists."""
    timestamps: list[datetime] = []
    values: list[float] = []
    quality_codes: list[str] = []

    for pt in mts.findall("wml2:point", _NS):
        tvp = pt.find("wml2:MeasurementTVP", _NS)
        if tvp is None:
            continue

        time_el = tvp.find("wml2:time", _NS)
        val_el = tvp.find("wml2:value", _NS)
        if time_el is None or val_el is None or time_el.text is None or val_el.text is None:
            continue

        try:
            ts = _parse_iso_datetime(time_el.text.strip())
            v = float(val_el.text.strip())
        except (ValueError, TypeError):
            logger.debug("skipping unparseable point: time=%s value=%s", time_el.text, val_el.text)
            continue

        qc = ""
        qual_el = tvp.find("wml2:metadata/wml2:TVPMeasurementMetadata/wml2:qualifier/wml2:Category/wml2:value", _NS)
        if qual_el is not None and qual_el.text:
            qc = qual_el.text.strip()

        timestamps.append(ts)
        values.append(v)
        quality_codes.append(qc)

    return timestamps, values, quality_codes


def _parse_iso_datetime(text: str) -> datetime:
    """Parse an ISO-8601 datetime string, tolerating the ``Z`` suffix."""
    text = text.replace("Z", "+00:00")
    return datetime.fromisoformat(text)


# ── Write ────────────────────────────────────────────────────────────────────
def write_waterml(
    timeseries: list[WaterMLTimeSeries] | WaterMLTimeSeries,
    path: str | Path,
    creator: str = "AquaScope",
) -> None:
    """Write time series to a WaterML 2.0 XML document.

    Creates a valid WaterML 2.0 document with:

    - ``wml2:Collection`` root element
    - ``gml:description`` metadata
    - ``wml2:observationMember`` for each time series
    - ``om:OM_Observation`` with sampling feature and observed property
    - ``wml2:MeasurementTimeseries`` with ``wml2:point`` elements

    Parameters
    ----------
    timeseries : list[WaterMLTimeSeries] | WaterMLTimeSeries
        One or more time series to serialise.
    path : str | Path
        Destination file path.
    creator : str
        Value written into ``gml:description``.
    """
    if isinstance(timeseries, WaterMLTimeSeries):
        timeseries = [timeseries]

    path = Path(path)

    # ── register namespaces so output uses short prefixes ──
    ET.register_namespace("wml2", WML2_NS)
    ET.register_namespace("gml", GML_NS)
    ET.register_namespace("om", OM_NS)
    ET.register_namespace("sams", SAMS_NS)
    ET.register_namespace("sa", SA_NS)

    root = ET.Element(f"{{{WML2_NS}}}Collection")
    root.set(f"{{{GML_NS}}}id", "aquascope-collection")

    desc = ET.SubElement(root, f"{{{GML_NS}}}description")
    desc.text = f"Generated by {creator}"

    for idx, ts in enumerate(timeseries):
        _build_observation_member(root, ts, idx)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")

    logger.debug("write_waterml: wrote %d time series to %s", len(timeseries), path)


def _build_observation_member(parent: ET.Element, ts: WaterMLTimeSeries, idx: int) -> None:
    """Append one ``<wml2:observationMember>`` subtree."""
    member = ET.SubElement(parent, f"{{{WML2_NS}}}observationMember")
    obs = ET.SubElement(member, f"{{{OM_NS}}}OM_Observation")
    obs.set(f"{{{GML_NS}}}id", f"obs-{idx}")

    # ── observed property ──
    prop = ET.SubElement(obs, f"{{{OM_NS}}}observedProperty")
    prop.set("title", ts.parameter)
    prop.set("href", f"urn:ogc:def:property:{ts.parameter}")

    # ── feature of interest ──
    foi = ET.SubElement(obs, f"{{{OM_NS}}}featureOfInterest")
    sf = ET.SubElement(foi, f"{{{SAMS_NS}}}SF_SpatialSamplingFeature")
    sf.set(f"{{{GML_NS}}}id", ts.station_id)

    name_el = ET.SubElement(sf, f"{{{GML_NS}}}name")
    name_el.text = ts.station_name

    if ts.latitude is not None and ts.longitude is not None:
        shape = ET.SubElement(sf, f"{{{SAMS_NS}}}shape")
        point = ET.SubElement(shape, f"{{{GML_NS}}}Point")
        point.set(f"{{{GML_NS}}}id", f"point-{ts.station_id}")
        pos = ET.SubElement(point, f"{{{GML_NS}}}pos")
        pos.set("srsName", "urn:ogc:def:crs:EPSG::4326")
        pos.text = f"{ts.latitude} {ts.longitude}"

    # ── result: MeasurementTimeseries ──
    result_el = ET.SubElement(obs, f"{{{OM_NS}}}result")
    mts = ET.SubElement(result_el, f"{{{WML2_NS}}}MeasurementTimeseries")
    mts.set(f"{{{GML_NS}}}id", f"ts-{idx}")

    # default point metadata (unit)
    dpm = ET.SubElement(mts, f"{{{WML2_NS}}}defaultPointMetadata")
    tvpm = ET.SubElement(dpm, f"{{{WML2_NS}}}DefaultTVPMeasurementMetadata")
    uom = ET.SubElement(tvpm, f"{{{WML2_NS}}}uom")
    uom.set("code", ts.unit)

    for i, (t, v) in enumerate(zip(ts.timestamps, ts.values)):
        pt = ET.SubElement(mts, f"{{{WML2_NS}}}point")
        tvp = ET.SubElement(pt, f"{{{WML2_NS}}}MeasurementTVP")

        time_el = ET.SubElement(tvp, f"{{{WML2_NS}}}time")
        time_el.text = t.isoformat()

        val_el = ET.SubElement(tvp, f"{{{WML2_NS}}}value")
        val_el.text = str(v)

        qc = ts.quality_codes[i] if i < len(ts.quality_codes) else ""
        if qc:
            meta = ET.SubElement(tvp, f"{{{WML2_NS}}}metadata")
            tvp_meta = ET.SubElement(meta, f"{{{WML2_NS}}}TVPMeasurementMetadata")
            qualifier = ET.SubElement(tvp_meta, f"{{{WML2_NS}}}qualifier")
            cat = ET.SubElement(qualifier, f"{{{WML2_NS}}}Category")
            cat_val = ET.SubElement(cat, f"{{{WML2_NS}}}value")
            cat_val.text = qc


# ── DataFrame conversions ────────────────────────────────────────────────────
def waterml_to_dataframe(timeseries: list[WaterMLTimeSeries]) -> pd.DataFrame:
    """Convert WaterML time series to a pandas DataFrame.

    Parameters
    ----------
    timeseries : list[WaterMLTimeSeries]
        Parsed time series objects.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``station_id``, ``parameter``, ``value``,
        ``unit``, ``quality_code``, ``latitude``, ``longitude``.
    """
    rows: list[dict[str, object]] = []
    for ts in timeseries:
        for i, (t, v) in enumerate(zip(ts.timestamps, ts.values)):
            qc = ts.quality_codes[i] if i < len(ts.quality_codes) else ""
            rows.append(
                {
                    "timestamp": t,
                    "station_id": ts.station_id,
                    "parameter": ts.parameter,
                    "value": v,
                    "unit": ts.unit,
                    "quality_code": qc,
                    "latitude": ts.latitude,
                    "longitude": ts.longitude,
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_waterml(
    df: pd.DataFrame,
    station_col: str = "station_id",
    param_col: str = "parameter",
    value_col: str = "value",
    time_col: str = "timestamp",
    unit_col: str = "unit",
) -> list[WaterMLTimeSeries]:
    """Convert a DataFrame to WaterML time series objects.

    Groups by *station_col* + *param_col* to create separate time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular data.
    station_col : str
        Column with station identifiers.
    param_col : str
        Column with parameter names.
    value_col : str
        Column with numeric measurement values.
    time_col : str
        Column with timestamps.
    unit_col : str
        Column with unit strings.

    Returns
    -------
    list[WaterMLTimeSeries]
        One ``WaterMLTimeSeries`` per unique station–parameter combination.
    """
    results: list[WaterMLTimeSeries] = []
    grouped = df.groupby([station_col, param_col])

    for (sid, param), group in grouped:
        unit = str(group[unit_col].iloc[0]) if unit_col in group.columns else "unknown"

        lat: float | None = None
        lon: float | None = None
        if "latitude" in group.columns:
            lat_val = group["latitude"].iloc[0]
            lat = float(lat_val) if pd.notna(lat_val) else None
        if "longitude" in group.columns:
            lon_val = group["longitude"].iloc[0]
            lon = float(lon_val) if pd.notna(lon_val) else None

        timestamps = [pd.Timestamp(t).to_pydatetime() for t in group[time_col]]
        quality_codes = list(group["quality_code"].astype(str)) if "quality_code" in group.columns else []

        station_name = str(group["station_name"].iloc[0]) if "station_name" in group.columns else str(sid)

        results.append(
            WaterMLTimeSeries(
                station_id=str(sid),
                station_name=station_name,
                parameter=str(param),
                unit=unit,
                latitude=lat,
                longitude=lon,
                timestamps=timestamps,
                values=list(group[value_col].astype(float)),
                quality_codes=quality_codes,
            )
        )

    return results
