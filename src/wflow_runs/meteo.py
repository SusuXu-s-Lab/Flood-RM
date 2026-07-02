from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from collect_sources.derived.aorc_event_meteo import (
    aorc_wflow_temp_pet_variables,
    prepare_aorc_temp_pet_for_wflow,
)
from event_forcing import find_aorc_event_window, prepare_aorc_precip_for_sfincs
from paths import resolve_location_path, write_json
from wflow_runs.event_catalog import (
    catalog_rainfall_start,
    event_window,
    legacy_event_catalog_row,
    required_event_value,
)


def build_meteo(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
    overwrite: bool = False,
) -> dict:
    """Stage the per-event Wflow forcing files consumed by the replay update."""
    location_root = Path(location_root).resolve()
    wflow = config.get("wflow", {})
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    event_dir = events_root / str(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)

    row = legacy_event_catalog_row(location_root, event_id, catalog_path)
    start, end = event_window(
        row["event_reference_time"],
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )

    precip_path = event_dir / "precip.nc"
    temp_pet_path = event_dir / "temp_pet.nc"
    precip_provenance = event_dir / "precip_provenance.json"
    temp_pet_provenance = event_dir / "temp_pet_provenance.json"
    write_precip = overwrite or not precip_path.exists() or not _provenance_window_matches(precip_provenance, start, end)
    write_temp_pet = (
        overwrite
        or write_precip
        or not temp_pet_path.exists()
        or not _provenance_window_matches(temp_pet_provenance, start, end)
    )

    rainfall_source_nc = event_rainfall_source_nc(config, location_root, row)
    scale_factor = _positive_float(row.get("rainfall_scale_factor"), default=1.0)
    if write_temp_pet:
        require_event_meteo_variables(config, rainfall_source_nc, event_id=event_id)
    if write_precip:
        precip_cfg = (wflow.get("event_forcing", {}) or {}).get("precipitation", {}) or {}
        aorc_cfg = (config.get("collection", {}) or {}).get("aorc_sst", {}) or {}
        prepare_aorc_precip_for_sfincs(
            rainfall_source_nc,
            precip_path,
            t_start=start,
            t_stop=end,
            variable=str(precip_cfg.get("variable", aorc_cfg.get("variable", "APCP_surface"))),
            window_alignment=str(precip_cfg.get("window_alignment", "start")),
            precip_start=catalog_rainfall_start(row),
            scale_factor=scale_factor,
        )
        write_json(
            precip_provenance,
            {
                "source_nc": str(rainfall_source_nc),
                "output_nc": str(precip_path),
                "time_start": start.isoformat(),
                "time_stop": end.isoformat(),
                "rainfall_scale_factor": scale_factor,
                "hydromt_sfincs_contract": "SfincsPrecipitation.create(cumulative_input=True)",
            },
        )

    if write_temp_pet:
        source_time_start = catalog_rainfall_start(row) or start
        prepare_aorc_temp_pet_for_wflow(
            rainfall_source_nc,
            temp_pet_path,
            t_start=start,
            t_stop=end,
            precip_template=precip_path,
            variable_candidates=aorc_wflow_temp_pet_variables(config),
            source_time_start=source_time_start,
            provenance_path=temp_pet_provenance,
        )

    return {
        "event_id": str(event_id),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "rainfall_source_nc": str(rainfall_source_nc),
        "rainfall_scale_factor": scale_factor,
        "precip_path": str(precip_path),
        "temp_pet_path": str(temp_pet_path),
        "precip_provenance": str(precip_provenance),
        "temp_pet_provenance": str(temp_pet_provenance),
        "precip_written": bool(write_precip),
        "temp_pet_written": bool(write_temp_pet),
    }


def resolve_event_rainfall_source_nc(config: dict, location_root, event_id: str, *, catalog_path=None) -> Path:
    """Resolve the catalog-selected AORC event-window file used by Wflow replay."""
    location_root = Path(location_root).resolve()
    row = legacy_event_catalog_row(location_root, event_id, catalog_path)
    return event_rainfall_source_nc(config, location_root, row)


def event_rainfall_source_nc(config: dict, location_root: Path, row: pd.Series) -> Path:
    rainfall_member_file = required_event_value(row, "rainfall_member_file")
    rainfall_member_file = resolve_location_path(location_root, rainfall_member_file)
    precip_cfg = (
        (config.get("wflow", {}) or {})
        .get("event_forcing", {})
        .get("precipitation", {})
        or {}
    )
    event_windows_dir = precip_cfg.get("event_windows_dir") or (rainfall_member_file.parent / "event_windows")
    event_windows_dir = resolve_location_path(location_root, event_windows_dir)
    return find_aorc_event_window(
        event_windows_dir,
        member_id=str(required_event_value(row, "rainfall_member_id")),
        storm_start=required_event_value(row, "rainfall_member_time"),
    )


def require_event_meteo_variables(config: dict, source_nc: Path, *, event_id: str) -> None:
    import xarray as xr

    candidates_by_target = aorc_wflow_temp_pet_variables(config)
    missing: dict[str, list[str]] = {}
    with xr.open_dataset(source_nc) as ds:
        available = set(ds.data_vars)
        for target, candidates in candidates_by_target.items():
            if not any(candidate in available and _data_array_has_finite(ds[candidate]) for candidate in candidates):
                missing[target] = list(candidates)
    if not missing:
        return

    meteo_cfg = ((config.get("collection", {}) or {}).get("aorc_sst", {}) or {}).get("event_meteo", {}) or {}
    config_hint = ""
    if not bool(meteo_cfg.get("enabled", False)):
        config_hint = " Set collection.aorc_sst.event_meteo.enabled: true in the location Wflow config."
    missing_text = "; ".join(f"{target}: tried {candidates}" for target, candidates in missing.items())
    raise RuntimeError(
        f"Wflow event meteo forcing for {event_id} cannot be built because the selected AORC "
        f"event-window file is rainfall-only or stale: {source_nc}. Missing variables: {missing_text}."
        f"{config_hint} Rerun 02_flood/02_collect_sources.ipynb from the AORC SST Event Windows "
        "cell so event windows are regenerated with AORC event_meteo variables, then rerun the "
        "dynamic handoff notebook."
    )


def _data_array_has_finite(da) -> bool:
    return bool(np.isfinite(da).any().compute().item())


def _positive_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not (np.isfinite(out) and out > 0):
        return float(default)
    return out


def _provenance_window_matches(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not Path(path).exists():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        existing_start = pd.Timestamp(payload.get("time_start"))
        existing_end = pd.Timestamp(payload.get("time_stop"))
    except Exception:
        return False
    return existing_start == pd.Timestamp(start) and existing_end == pd.Timestamp(end)
