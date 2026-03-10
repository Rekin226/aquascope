"""
Helpers for persisting collected data to CSV / Parquet / JSON.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def save_records(
    records: Sequence[BaseModel],
    dest_dir: str | Path = "data/raw",
    prefix: str = "water_data",
    fmt: str = "json",
) -> Path:
    """
    Persist a list of Pydantic model instances.

    Parameters
    ----------
    records : list[BaseModel]
        Data records (WaterQualitySample, WaterLevelReading, etc.).
    dest_dir : str | Path
        Directory to write to.
    prefix : str
        File name prefix.
    fmt : str
        ``"json"`` or ``"csv"``.

    Returns
    -------
    Path  — the file that was written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.{fmt}"
    filepath = dest / filename

    dicts = [r.model_dump(mode="json") for r in records]

    if fmt == "json":
        filepath.write_text(json.dumps(dicts, ensure_ascii=False, indent=2, default=str))
    elif fmt == "csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV export.  pip install pandas")
        df = pd.DataFrame(dicts)
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")

    logger.info("Saved %d records → %s", len(records), filepath)
    return filepath
