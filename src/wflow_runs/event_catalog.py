from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import resolve_location_path
from wflow_runs.types import DesignEvent

CFS_TO_CMS = 0.028316846592


def event_paths(config: dict[str, Any], location_root: str | Path, event_id: str) -> dict[str, Path]:
    root = Path(location_root)
    events_root = resolve_location_path(root, (config.get("wflow", {}) or {}).get("events_root", "data/wflow/events"))
    event_root = events_root / str(event_id)
    return {
        "event_root": event_root,
        "discharge": event_root / "sfincs_discharge.nc",
        "qa_csv": event_root / "sfincs_discharge.qa.csv",
        "acceptance": event_root / "sfincs_discharge.acceptance.json",
        "amplification": event_root / "sfincs_discharge.amplification.json",
        "zero_rain_discharge": event_root / "_zero_rain" / "sfincs_discharge.nc",
    }


def legacy_dynamic_handoff_paths(config: dict[str, Any], location_root: str | Path, event_id: str) -> dict[str, Path]:
    """Return the notebook-visible dynamic handoff paths used by ``wflow_runs``."""
    root = Path(location_root)
    events_root = resolve_location_path(root, (config.get("wflow", {}) or {}).get("events_root", "data/wflow/events"))
    event_root = events_root / str(event_id)
    return {
        "event_root": event_root,
        "discharge": event_root / "sfincs_discharge.nc",
        "qa_csv": event_root / "sfincs_discharge.dynamic_handoff_qa.csv",
        "acceptance": event_root / "sfincs_discharge.dynamic_handoff.json",
        "zero_rain_discharge": event_root / "_zero_rain" / "sfincs_discharge.nc",
    }


def event_catalog_path(config: dict[str, Any], location_root: str | Path, catalog_path=None) -> Path:
    if catalog_path is not None:
        return resolve_location_path(location_root, catalog_path)
    configured = (((config.get("event_catalog", {}) or {}).get("catalog", {}) or {}).get("probability_catalog"))
    if configured:
        return resolve_location_path(location_root, configured)
    return resolve_location_path(location_root, "data/event_catalog/catalog/probability_catalog.csv")


def legacy_event_catalog_row(location_root: str | Path, event_id: str, catalog_path=None) -> pd.Series:
    """Return the old ``wflow_runs.replay`` catalog row contract."""
    root = Path(location_root)
    path = Path(catalog_path) if catalog_path else resolve_location_path(root, "data/event_catalog/catalog/probability_catalog.csv")
    if not path.is_absolute():
        path = root / path
    catalog = pd.read_csv(path)
    catalog["event_id"] = catalog["event_id"].astype(str)
    match = catalog[catalog["event_id"] == str(event_id)]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not in {path}")
    if "event_reference_time" not in match:
        raise ValueError(f"{path} has no event_reference_time column")
    return match.iloc[0]


def event_reference_time(location_root: str | Path, event_id: str, catalog_path=None) -> pd.Timestamp:
    return pd.Timestamp(legacy_event_catalog_row(location_root, event_id, catalog_path)["event_reference_time"])


def required_event_value(row: pd.Series, key: str):
    value = row.get(key)
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Event Catalog row is missing required Wflow forcing field: {key}")
    return value


def catalog_rainfall_start(row: pd.Series):
    reference = row.get("event_reference_time")
    offset = row.get("rainfall_start_offset_hours")
    if reference is not None and offset is not None and not pd.isna(reference) and not pd.isna(offset):
        return pd.Timestamp(reference) + pd.Timedelta(hours=float(offset))
    if reference is None or pd.isna(reference):
        return None
    return pd.Timestamp(reference)


def event_window(reference_time, *, pre_event_hours: float = 48.0, post_event_hours: float = 72.0, timestep_seconds: int = 3600) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.Timestamp(reference_time)
    if pd.isna(ref):
        raise ValueError(f"event_reference_time is not a valid timestamp: {reference_time!r}")
    step = pd.Timedelta(seconds=int(timestep_seconds))
    return (ref - pd.Timedelta(hours=float(pre_event_hours))).floor(step), (ref + pd.Timedelta(hours=float(post_event_hours))).ceil(step)


def read_event(config: dict[str, Any], location_root: str | Path, event_id: str, *, catalog_path=None) -> DesignEvent:
    root = Path(location_root)
    catalog_file = event_catalog_path(config, root, catalog_path)
    catalog = pd.read_csv(catalog_file, dtype={"event_id": str})
    match = catalog[catalog["event_id"].astype(str).eq(str(event_id))]
    if match.empty:
        raise ValueError(f"event_id {event_id!r} not found in {catalog_file}")
    row = match.iloc[0]
    reference_time = pd.Timestamp(_first(row, ["event_reference_time", "reference_time", "time"]))
    pre, post = configured_event_window_hours(config)
    start, end = event_window(reference_time, pre_event_hours=pre, post_event_hours=post)
    paths = event_paths(config, root, event_id)
    precip = _path_from_row(root, row, ["wflow_precip_path", "event_precip", "precip_path"], paths["event_root"] / "precip.nc")
    temp_pet = _path_from_row(root, row, ["wflow_temp_pet_path", "event_temp_pet", "temp_pet_path"], paths["event_root"] / "temp_pet.nc")
    q_target_cms = _target_cms(row)
    probability = _probability_metadata(row)
    return DesignEvent(
        event_id=str(event_id),
        reference_time=reference_time,
        window_start=start,
        window_end=end,
        precip_path=precip,
        temp_pet_path=temp_pet,
        rainfall_member_id=_string_or_none(_first(row, ["rainfall_member_id", "member_id"])),
        rainfall_member_file=_path_from_row(root, row, ["rainfall_member_file"], None),
        rainfall_scale_factor=float(_float_or_none(_first(row, ["rainfall_scale_factor", "rainfall_scale"])) or 1.0),
        probability=probability,
        q_target_cms=q_target_cms,
        attrs={"catalog_path": str(catalog_file)},
    )


def configured_event_window_hours(
    config: dict[str, Any],
    *,
    default_pre_event_hours: float = 48.0,
    default_post_event_hours: float = 72.0,
) -> tuple[float, float]:
    cfg = ((config.get("wflow", {}) or {}).get("event_window", {}) or {})
    timing = ((config.get("scenario_build", {}) or {}).get("timing", {}) or {})
    pre = _positive_float(cfg.get("pre_event_hours"), default=default_pre_event_hours)
    if "post_event_hours" in cfg:
        post = _positive_float(cfg.get("post_event_hours"), default=default_post_event_hours)
    else:
        drain_down = max(float(timing.get("drain_down_hours", 0.0) or 0.0), 0.0)
        post = float(default_post_event_hours) + drain_down
    return pre, post


def _probability_metadata(row) -> dict[str, float]:
    pairs = {
        "p_event": _float_or_none(_first(row, ["p_event", "probability", "event_probability"])),
        "aep": _float_or_none(_first(row, ["aep", "annual_exceedance_probability"])),
        "return_period_years": _float_or_none(_first(row, ["return_period_years", "sample_rp_years", "rp_years"])),
        "weight": _float_or_none(_first(row, ["weight", "catalog_weight"])),
    }
    return {key: value for key, value in pairs.items() if value is not None}


def _target_cms(row: pd.Series) -> float | None:
    for key, factor in [("streamflow_target_cms", 1.0), ("q_target_cms", 1.0), ("streamflow_target_cfs", CFS_TO_CMS), ("target_peak_cfs", CFS_TO_CMS)]:
        value = _float_or_none(row.get(key))
        if value and value > 0:
            return float(value) * factor
    return None


def _first(row: pd.Series, keys: list[str]):
    for key in keys:
        if key in row and not pd.isna(row.get(key)) and str(row.get(key)).strip() != "":
            return row.get(key)
    return None


def _path_from_row(root: Path, row: pd.Series, keys: list[str], default) -> Path | None:
    value = _first(row, keys)
    if value is None:
        return default
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _float_or_none(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _positive_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not (np.isfinite(out) and out > 0):
        return float(default)
    return out


def _string_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
