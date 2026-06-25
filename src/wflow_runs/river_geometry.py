from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_geometry(
    river_geometry,
    *,
    require_stream_geo: bool = True,
    require_variable: bool = True,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """QA river geometry before HydroMT-Wflow ``setup_rivers`` consumes it."""
    import geopandas as gpd

    rivers = gpd.read_file(river_geometry) if isinstance(river_geometry, (str, Path)) else river_geometry.copy()
    rows: list[dict] = []
    _append_numeric_geometry_check(rows, rivers, "rivwth", require_variable=require_variable)
    if "rivdph" in rivers:
        _append_numeric_geometry_check(rows, rivers, "rivdph", require_variable=require_variable)
    elif "qbankfull" in rivers:
        _append_numeric_geometry_check(rows, rivers, "qbankfull", require_variable=require_variable)
    else:
        rows.append({"check": "river_depth_source", "status": "failed", "message": "missing rivdph or qbankfull"})
    if require_stream_geo:
        source_columns = [col for col in ("rivwth_source", "rivdph_source", "qbankfull_source") if col in rivers]
        source_text = " ".join(rivers[col].fillna("").astype(str).str.cat(sep=" ") for col in source_columns)
        has_stream_geo = "STREAM-geo" in source_text
        fallback_only = bool(source_text) and all("fallback" in value.lower() for value in source_text.split() if value)
        status = "passed" if has_stream_geo and not fallback_only else "failed"
        rows.append(
            {
                "check": "stream_geo_source",
                "status": status,
                "message": f"source_columns={source_columns or 'none'}; has_stream_geo={has_stream_geo}",
            }
        )
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow river geometry QA failed: {details}")
    return report




def _append_numeric_geometry_check(rows: list[dict], rivers, column: str, *, require_variable: bool) -> None:
    if column not in rivers:
        rows.append({"check": column, "status": "failed", "message": f"missing {column}"})
        return
    values = pd.to_numeric(rivers[column], errors="coerce").to_numpy(dtype=float)
    valid = values[np.isfinite(values) & (values > 0)]
    unique = int(len(np.unique(np.round(valid, 4)))) if valid.size else 0
    status = "passed"
    if valid.size == 0:
        status = "failed"
    elif require_variable and unique <= 1:
        status = "review_required"
    rows.append({"check": column, "status": status, "message": f"valid={int(valid.size)}; unique={unique}"})
