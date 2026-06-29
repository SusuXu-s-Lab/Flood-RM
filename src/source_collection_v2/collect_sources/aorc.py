from __future__ import annotations

from pathlib import Path

import pandas as pd

from source_collection_v2.stochastic_boundary.audit import Artifact, nonempty, write_artifact
from source_collection_v2.stochastic_boundary.gridded import open_zarr, subset
from source_collection_v2.stochastic_boundary.meteo import aorc_source_variables

AORC_ZARR = "s3://noaa-nws-aorc-v1-1-1km/{year}.zarr"


def open_year(year: int, spec: dict):
    """Open one AORC annual Zarr store."""
    return open_zarr(
        str(spec.get("zarr_year_pattern", AORC_ZARR)).format(year=int(year)),
        chunks=spec.get("chunks", {}),
        consolidated=spec.get("consolidated", True),
    )


def fetch_window(settings: dict, *, output=None) -> Path:
    """Download an AORC bbox/time subset to NetCDF; no event-selection science."""
    paths, spec = settings["paths"], settings.get("spec", settings.get("aorc", {}))
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    if end == end.floor("D"):
        end += pd.Timedelta(hours=23)
    output = Path(output or spec.get("output") or Path(paths["location_root"]) / "data/sources/aorc/aorc_subset.nc")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        with open_year(year, spec) as ds:
            variables = spec.get("variables") or aorc_source_variables(ds, spec, settings.get("config"))
            frames.append(subset(ds, variables=variables, bbox=spec["bbox_wgs84"], start=max(start, pd.Timestamp(f"{year}-01-01")), end=min(end, pd.Timestamp(f"{year}-12-31 23:00"))))
    import xarray as xr

    (xr.concat(frames, dim="time") if len(frames) > 1 else frames[0]).to_netcdf(output)
    return output


def collect(settings: dict, *, skip_existing=True) -> Artifact:
    paths, spec = settings["paths"], settings.get("spec", settings.get("aorc", {}))
    path = Path(spec.get("output") or Path(paths["location_root"]) / "data/sources/aorc/aorc_subset.nc")
    if not (skip_existing and nonempty(path)):
        path = fetch_window(settings, output=path)
    artifact = Artifact("aorc", "subset", settings["start"], settings["end"], {"netcdf": path}, {"variables": spec.get("variables")})
    write_artifact(paths, artifact)
    return artifact
