from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm.auto import tqdm as iter_progress

from collect_sources.audit import Artifact, covers, netcdf_covers, resolve, write_artifact
from derived.gridded import subset
from paths import location_or_repo_path_from_paths
from source_artifacts import source_artifact_covers, write_source_artifact

CDS_WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "peak_wave_period",
    "mean_wave_direction",
    "wave_spectral_directional_width",
]
EDH_URL = "https://api.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-ocean-v0.zarr"
SHORT_VARIABLES = ["swh", "pp1d", "mwd", "wdw"]


def cds_payload(bbox_wgs84, start, end, variables=None) -> dict:
    west, south, east, north = map(float, bbox_wgs84)
    hours = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).ceil("D"), freq="h", inclusive="left")
    return {
        "product_type": "reanalysis",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": list(variables or CDS_WAVE_VARIABLES),
        "year": sorted({f"{t.year}" for t in hours}),
        "month": sorted({f"{t.month:02d}" for t in hours}),
        "day": sorted({f"{t.day:02d}" for t in hours}),
        "time": sorted({f"{t.hour:02d}:00" for t in hours}),
        "area": [north, west, south, east],
    }


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    out = Path(paths.get("era5_waves_nc") or resolve(paths, spec.get("output_path", "data/sources/era5/waves.nc")))
    provider = spec.get("provider", "cds")
    bbox = tuple(float(x) for x in spec["bbox_wgs84"])
    if skip_existing and out.exists() and covers(paths, "era5", "waves", start, end) and netcdf_covers(out, start, end):
        return Artifact("era5", "waves", start, end, {"wave_netcdf": out}, {"provider": provider, "reused": True})
    out.parent.mkdir(parents=True, exist_ok=True)
    if provider == "cds":
        _fetch_cds(bbox, start, end, out, variables=spec.get("variables"), force=True)
    elif provider == "earthdatahub":
        _fetch_edh(bbox, start, end, out, variables=spec.get("variables", SHORT_VARIABLES), url=spec.get("url"))
    else:
        raise ValueError(f"unsupported ERA5 provider: {provider}")
    variables, time_count = _validate(out, spec.get("short_variables", SHORT_VARIABLES))
    artifact = Artifact("era5", "waves", start, end, {"wave_netcdf": out}, {"provider": provider, "bbox_wgs84": list(bbox), "variables": variables, "time_count": time_count})
    write_artifact(paths, artifact)
    return artifact


def _fetch_cds(bbox, start, end, out, *, variables=None, force=False):
    if out.exists() and not force:
        return out
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("Install cdsapi and configure ~/.cdsapirc to fetch ERA5 from CDS") from exc
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", cds_payload(bbox, start, end, variables), str(out))
    return out


def _fetch_edh(bbox, start, end, out, *, variables=None, url=None):
    with xr.open_dataset(url or EDH_URL, engine="zarr", chunks={}) as ds:
        subset(ds, variables=list(variables or SHORT_VARIABLES), bbox=bbox, start=start, end=end).to_netcdf(out)
    return out


def _validate(path: Path, required: list[str]):
    result = validate_wave_dataset(path, required_short_variables=required)
    return result["variables"], result["time_count"]


# Notebook/workflow-facing wave collection (formerly collect_sources.era5_waves).

era5_wave_short_variables = SHORT_VARIABLES


def _repo_path(paths, value):
    if value is None:
        return None
    return location_or_repo_path_from_paths(paths, value)


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
    # cds/earthdatahub import from this module, so resolve providers lazily.
    if provider == "cds":
        from collect_sources.cds import era5_wave_variables

        return era5_wave_variables
    if provider == "earthdatahub":
        from collect_sources.earthdatahub import earthdatahub_wave_variables

        return earthdatahub_wave_variables
    raise ValueError(f"unsupported era5_waves provider: {provider}")


def _provider_fetcher(provider):
    if provider == "cds":
        from collect_sources.cds import fetch_era5_waves

        return fetch_era5_waves
    if provider == "earthdatahub":
        from collect_sources.earthdatahub import fetch_era5_waves_from_earthdatahub

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
