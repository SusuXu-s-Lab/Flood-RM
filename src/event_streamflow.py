from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from paths import resolve_location_path


def streamflow_records_path(config: dict, location_root: str | Path) -> Path:
    """Return the reviewed streamflow records path from current or legacy config."""
    value = _driver_streamflow_value(config, "records", "records_file", "path")
    if value is None:
        value = (
            config.get("collection", {})
            .get("usgs_streamgages", {})
            .get("streamflow_records", "data/sources/usgs_streamgages/streamflow_records.csv")
        )
        if isinstance(value, dict):
            value = value.get("output", "data/sources/usgs_streamgages/streamflow_records.csv")
    return resolve_location_path(location_root, value or "data/sources/usgs_streamgages/streamflow_records.csv")


def streamflow_members_path(config: dict, location_root: str | Path) -> Path:
    """Return the streamflow member table path from current or legacy config."""
    value = ((config.get("event_catalog", {}) or {}).get("forcing_members", {}) or {}).get("streamflow")
    if value is None:
        value = _driver_streamflow_value(config, "members", "members_file")
    return resolve_location_path(location_root, value or "data/sources/usgs_streamgages/streamflow_members.csv")


def event_streamflow_records_path(
    config: dict,
    location_root: str | Path,
    event_id: str,
    member: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    settings = (((config.get("wflow", {}) or {}).get("streamflow_realization", {}) or {}))
    root = resolve_location_path(
        location_root,
        settings.get("event_records_root", "data/sources/usgs_streamgages/event_streamflow_iv"),
    )
    token = "_".join(
        [
            str(event_id),
            str(member["member_id"]),
            pd.Timestamp(start).strftime("%Y%m%dT%H%M%S"),
            pd.Timestamp(end).strftime("%Y%m%dT%H%M%S"),
        ]
    )
    return root / f"{safe_filename_token(token)}.csv"


def streamflow_member_metadata(config: dict, location_root: str | Path, row: pd.Series) -> dict:
    member_id = str(required_event_value(row, "streamflow_member_id"))
    site_no = str(row.get("streamflow_member_site_no") or member_id.split("_")[0])
    event_time = row.get("streamflow_member_time") or row.get("event_reference_time")
    peak_flow_cfs = finite_float(row.get("streamflow_template_value"))
    contributing: list[str] = []

    members_path = streamflow_members_path(config, location_root)
    if members_path.exists():
        members = pd.read_csv(members_path, dtype={"site_no": str})
        match = members[members["member_id"].astype(str) == member_id]
        if not match.empty:
            mrow = match.iloc[0]
            site_no = str(mrow.get("site_no") or site_no)
            event_time = mrow.get("event_time") or event_time
            peak_flow_cfs = finite_float(mrow.get("peak_flow_cfs")) or peak_flow_cfs
            contributing = split_site_list(mrow.get("contributing_site_nos"))

    return {
        "member_id": member_id,
        "site_no": site_no,
        "event_time": event_time,
        "peak_flow_cfs": peak_flow_cfs,
        "site_nos": list(dict.fromkeys([site_no, *contributing])),
    }


def required_event_value(row: pd.Series, key: str):
    value = row.get(key)
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Event Catalog row is missing required Wflow forcing field: {key}")
    return value


def split_site_list(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _driver_streamflow_value(config: dict, *keys: str):
    streamflow_cfg = (((config.get("event_catalog", {}) or {}).get("driver_records", {}) or {}).get("streamflow", {}) or {})
    if isinstance(streamflow_cfg, str):
        return streamflow_cfg if "path" in keys or "records" in keys else None
    for key in keys:
        value = streamflow_cfg.get(key)
        if value:
            return value
    return None
