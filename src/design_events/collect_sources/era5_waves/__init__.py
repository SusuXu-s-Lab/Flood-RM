from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr

from design_events.collect_sources.era5_waves.cds import (
    era5_wave_variables,
    fetch_era5_waves,
)
from design_events.collect_sources.era5_waves.earthdatahub import (
    earthdatahub_wave_variables,
    fetch_era5_waves_from_earthdatahub,
)
from tqdm.auto import tqdm as iter_progress


era5_wave_short_variables = ["swh", "pp1d", "mwd", "wdw"]


def _timestamp(value):
    return None if value is None else pd.Timestamp(value).isoformat()


def _relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_artifact_path(paths, source, kind):
    return paths["source_artifacts_root"] / f"{source}_{kind}.json"


def read_source_artifact(paths, source, kind):
    path = source_artifact_path(paths, source, kind)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_artifact_covers(paths, source, kind, start, end):
    manifest = read_source_artifact(paths, source, kind)
    if manifest is None or manifest.get("status") != "complete":
        return False
    if manifest.get("metadata", {}).get("smoke") is True:
        return False
    artifact_start = manifest.get("start")
    artifact_end = manifest.get("end")
    if not artifact_start or not artifact_end:
        return False
    return pd.Timestamp(artifact_start) <= pd.Timestamp(start) and pd.Timestamp(artifact_end) >= pd.Timestamp(end)


def write_source_artifact(paths, source, kind, start=None, end=None, artifacts=None, metadata=None, status="complete"):
    manifest = {
        "study_location": paths["location_name"],
        "source": source,
        "kind": kind,
        "status": status,
        "start": _timestamp(start),
        "end": _timestamp(end),
        "artifacts": {key: _relative_path(value, paths["repo_root"]) for key, value in (artifacts or {}).items()},
        "metadata": metadata or {},
    }
    path = source_artifact_path(paths, source, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _repo_path(paths, value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"} and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return paths["repo_root"] / path


def _wave_output_path(paths, spec):
    return _repo_path(paths, spec.get("output_path")) or paths["era5_waves_nc"]


def _bbox(spec):
    value = spec.get("bbox_wgs84")
    if value is None:
        raise ValueError("era5_waves.bbox_wgs84 is required")
    if len(value) != 4:
        raise ValueError("era5_waves.bbox_wgs84 must have four values: W S E N")
    return tuple(float(x) for x in value)


def _collection_window(settings, smoke=False):
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    spec = settings.get("era5_waves", {})
    if spec.get("start") is not None:
        start = pd.Timestamp(spec["start"])
    if spec.get("end") is not None:
        end = pd.Timestamp(spec["end"])
    if smoke:
        start = pd.Timestamp(spec.get("smoke_start", start))
        end = pd.Timestamp(spec.get("smoke_end", min(end, start + pd.Timedelta(hours=23))))
    if end < start:
        raise ValueError("era5_waves end date must be on or after start date")
    return start, end


def validate_wave_dataset(path, required_short_variables=None):
    required = list(required_short_variables or era5_wave_short_variables)
    with xr.open_dataset(path) as ds:
        variables = list(ds.data_vars)
        missing = [name for name in required if name not in ds.data_vars]
        if missing:
            raise ValueError(f"ERA5 wave dataset missing variables: {missing}")
        time_dim = "valid_time" if "valid_time" in ds.sizes else "time"
        time_count = int(ds.sizes.get(time_dim, 0))
    return {"variables": variables, "time_count": time_count}


def wave_dataset_covers(path, start, end):
    if not Path(path).exists():
        return False
    with xr.open_dataset(path) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        if time_name not in ds.coords or int(ds.sizes.get(time_name, 0)) == 0:
            return False
        times = pd.to_datetime(ds[time_name].values)
        return pd.Timestamp(times.min()) <= pd.Timestamp(start) and pd.Timestamp(times.max()) >= pd.Timestamp(end)


def collect_era5_waves(settings, skip_existing=False, smoke=False, fetcher=None):
    paths = settings["paths"]
    spec = settings.get("era5_waves", {})
    output_path = _wave_output_path(paths, spec)
    start, end = _collection_window(settings, smoke=smoke)
    provider = spec.get("provider", "cds")
    variables = list(spec.get("variables", _provider_default_variables(provider)))
    fetcher = fetcher or _provider_fetcher(provider)

    stages = iter_progress(
        ["fetch", "validate", "manifest"],
        total=3,
        desc="ERA5 waves",
        unit="stage",
        dynamic_ncols=True,
    )
    can_reuse = (
        skip_existing
        and output_path.exists()
        and source_artifact_covers(paths, "era5", "snapwave_boundary_forcing", start, end)
        and wave_dataset_covers(output_path, start, end)
    )
    for stage in stages:
        if hasattr(stages, "set_postfix_str"):
            stages.set_postfix_str(stage, refresh=False)
        if stage == "fetch":
            if can_reuse:
                print(f"ERA5 waves: reusing complete production artifact {output_path}")
            else:
                print(
                    "ERA5 waves: fetching "
                    f"{provider} {start.isoformat()} to {end.isoformat()} -> {output_path}"
                )
                fetcher(
                    _bbox(spec),
                    (start, end),
                    output_path,
                    variables=variables,
                    force=not can_reuse,
                    **_provider_fetch_kwargs(paths, spec, provider),
                )
        elif stage == "validate":
            validation = validate_wave_dataset(
                output_path,
                required_short_variables=spec.get("short_variables", era5_wave_short_variables),
            )
            print(
                "ERA5 waves: validated "
                f"{validation['time_count']:,} time steps and {len(validation['variables'])} variables"
            )
        else:
            artifact_json = write_source_artifact(
                paths,
                source="era5",
                kind="snapwave_boundary_forcing",
                start=start,
                end=end,
                artifacts={"wave_netcdf": output_path},
                metadata={
                    "bbox_wgs84": list(_bbox(spec)),
                    "provider": provider,
                    "variables": variables,
                    "short_variables": era5_wave_short_variables,
                    "time_count": validation["time_count"],
                    "smoke": bool(smoke),
                },
            )

    return {
        "wave_netcdf": output_path,
        "source_artifact_json": artifact_json,
        "variables": validation["variables"],
        "time_count": validation["time_count"],
    }


def _provider_default_variables(provider):
    if provider == "cds":
        return era5_wave_variables
    if provider == "earthdatahub":
        return earthdatahub_wave_variables
    raise ValueError(f"unsupported era5_waves provider: {provider}")


def _provider_fetcher(provider):
    if provider == "cds":
        return fetch_era5_waves
    if provider == "earthdatahub":
        return fetch_era5_waves_from_earthdatahub
    raise ValueError(f"unsupported era5_waves provider: {provider}")


def _provider_fetch_kwargs(paths, spec, provider):
    if provider != "earthdatahub":
        return {}
    kwargs = {}
    if spec.get("url") is not None:
        kwargs["url"] = spec["url"]
    if spec.get("token") is not None:
        kwargs["token"] = spec["token"]
    if spec.get("auth_path") is not None:
        kwargs["auth_path"] = _repo_path(paths, spec["auth_path"])
    if spec.get("chunk_months") is not None:
        kwargs["chunk_months"] = spec["chunk_months"]
    return kwargs
