from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from design_events.stochastic_boundary.audit import Artifact, covers, netcdf_covers, resolve, write_artifact
from design_events.stochastic_boundary.gridded import coord, subset

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
    with xr.open_dataset(path) as ds:
        variables = list(ds.data_vars)
        missing = [v for v in required if v not in variables]
        if missing:
            raise ValueError(f"ERA5 wave NetCDF missing variables: {missing}")
        t = coord(ds, ("time", "valid_time"))
        return variables, int(ds.sizes.get(t, 0))
