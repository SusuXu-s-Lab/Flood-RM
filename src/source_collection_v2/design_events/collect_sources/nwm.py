from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from design_events.stochastic_boundary.audit import Artifact, covers, resolve, write_artifact
from design_events.stochastic_boundary.gridded import open_zarr
from design_events.stochastic_boundary.hydrology import derive_soilsat_top, nwm_points


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    root = Path(paths.get("nwm_root") or resolve(paths, spec.get("output_dir", "data/sources/nwm")))
    stream_csv = Path(paths.get("nwm_streamflow_csv") or root / "streamflow.csv")
    soil_csv = Path(paths.get("nwm_soil_moisture_csv") or root / "soil_moisture.csv")
    if skip_existing and stream_csv.exists() and soil_csv.exists() and covers(paths, "nwm", "hydrologic_state", start, end):
        return Artifact("nwm", "hydrologic_state", start, end, {"streamflow_csv": stream_csv, "soil_moisture_csv": soil_csv}, {"reused": True})
    root.mkdir(parents=True, exist_ok=True)
    stream = _streamflow(spec.get("streamflow", {}), start, end, stream_csv)
    soil = _soil_moisture(spec.get("soil_moisture", {}), paths, start, end, soil_csv)
    artifact = Artifact("nwm", "hydrologic_state", start, end, {"streamflow_csv": stream_csv, "soil_moisture_csv": soil_csv}, {"streamflow_rows": len(stream), "soil_moisture_rows": len(soil), "version": spec.get("version")})
    write_artifact(paths, artifact)
    return artifact


def _open(spec: dict) -> xr.Dataset:
    return open_zarr(spec["zarr"], chunks=spec.get("chunks", {}))


def _streamflow(spec: dict, start, end, out: Path) -> pd.DataFrame:
    if spec.get("available") is False or not spec.get("feature_ids"):
        frame = pd.DataFrame(columns=["time", "feature_id", spec.get("variable", "streamflow")])
    else:
        with _open(spec).sel(time=slice(start, end)) as ds:
            variable, dim = spec.get("variable", "streamflow"), spec.get("feature_dim", "feature_id")
            frame = ds[variable].sel({dim: spec["feature_ids"]}).to_dataframe(name=variable).reset_index()
    out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out, index=False)
    return frame


def _soil_moisture(spec: dict, paths: dict, start, end, out: Path) -> pd.DataFrame:
    points = nwm_points(spec, paths)
    variables = [str(v) for v in spec.get("variables", ["SOIL_M"])]
    if not points:
        frame = pd.DataFrame(columns=["time", "point_id", *variables])
    else:
        import xarray as xr
        with _open(spec).sel(time=slice(start, end)) as ds:
            available = [v for v in variables if v in ds]
            if "SOILSAT_TOP" in variables and "SOILSAT_TOP" not in available and "SOIL_M" in ds:
                available.append("SOIL_M")
            x, y = spec.get("x", "x"), spec.get("y", "y")
            point_id = [str(p.get("id", i)) for i, p in enumerate(points)]
            xx = xr.DataArray([float(p[x]) for p in points], dims="point", coords={"point_id": ("point", point_id)})
            yy = xr.DataArray([float(p[y]) for p in points], dims="point", coords={"point_id": ("point", point_id)})
            frame = ds[available].sel({x: xx, y: yy}, method="nearest").to_dataframe().reset_index()
        frame = derive_soilsat_top(frame, layer_dim=spec.get("soil_layer_dim", "soil_layers_stag"), top_layers=spec.get("soilsat_top_layers", [0, 1]))
        if spec.get("aggregate_points", False) and "time" in frame:
            agg = {c: (c, "mean") for c in ["SOIL_M", "SOILSAT_TOP"] if c in frame}
            frame = frame.groupby("time", as_index=False).agg(**agg)
    out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out, index=False)
    return frame
