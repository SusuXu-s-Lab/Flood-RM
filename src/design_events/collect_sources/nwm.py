from __future__ import annotations

import pandas as pd

from design_events.progress import iter_progress
from design_events.source_artifacts import source_artifact_covers, write_source_artifact


def _configured(path_or_default, paths, key):
    return paths.get(key) or path_or_default


def _open_zarr(url, chunks=None):
    import xarray as xr

    kwargs = {"chunks": chunks or {}}
    if str(url).startswith("s3://"):
        kwargs["storage_options"] = {"anon": True}
    return xr.open_zarr(url, **kwargs)


def _time_slice(ds, start, end):
    if "time" not in ds.coords and "time" not in ds.dims:
        return ds
    return ds.sel(time=slice(start, end))


def _write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _empty(path, columns):
    return _write_frame(pd.DataFrame(columns=columns), path)


def soil_moisture_csv_has_variables(path, variables):
    variables = [str(value) for value in (variables or [])]
    if not variables:
        return True
    path = _as_path(path)
    if not path.exists():
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError:
        return False
    return set(variables).issubset(set(columns))


def repair_soil_moisture_csv(path, *, variables, spec):
    path = _as_path(path)
    variables = [str(value) for value in (variables or [])]
    if not variables or not path.exists():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        return False
    missing = [name for name in variables if name not in columns]
    if not missing:
        return False
    if missing != ["SOILSAT_TOP"] or "SOIL_M" not in columns:
        return False

    print("NWM soil moisture: deriving SOILSAT_TOP locally from existing SOIL_M CSV")
    frame = pd.read_csv(path)
    frame = _derive_soilsat_top(frame, requested=variables, spec=spec)
    _write_frame(frame, path)
    return True


def _as_path(path):
    from pathlib import Path

    return Path(path)


def collect_streamflow(settings, open_zarr=_open_zarr):
    paths = settings["paths"]
    nwm = settings["nwm"]
    spec = nwm.get("streamflow", {})
    output_csv = _configured(paths["nwm_root"] / "streamflow.csv", paths, "nwm_streamflow_csv")
    feature_ids = spec.get("feature_ids", [])
    if not feature_ids:
        print("NWM streamflow: no feature IDs configured; writing empty artifact")
        return _empty(output_csv, ["time", "feature_id", "streamflow"])
    print(f"NWM streamflow: opening {spec['zarr']}")
    ds = _time_slice(open_zarr(spec["zarr"], chunks=spec.get("chunks", {})), settings["start"], settings["end"])
    variable = spec.get("variable", "streamflow")
    feature_dim = spec.get("feature_dim", "feature_id")
    print(f"NWM streamflow: extracting {len(feature_ids)} feature IDs")
    frame = (
        ds[variable]
        .sel({feature_dim: feature_ids})
        .to_dataframe(name=variable)
        .reset_index()
    )
    print(f"NWM streamflow: writing {len(frame):,} rows to {output_csv}")
    return _write_frame(frame, output_csv)


def collect_soil_moisture(settings, open_zarr=_open_zarr):
    paths = settings["paths"]
    nwm = settings["nwm"]
    spec = nwm.get("soil_moisture", {})
    output_csv = _configured(paths["nwm_root"] / "soil_moisture.csv", paths, "nwm_soil_moisture_csv")
    points = spec.get("points", [])
    if not points:
        print("NWM soil moisture: no points configured; writing empty artifact")
        return _empty(output_csv, ["time", "point_id", "soil_moisture"])
    print(f"NWM soil moisture: opening {spec['zarr']}")
    ds = _time_slice(open_zarr(spec["zarr"], chunks=spec.get("chunks", {})), settings["start"], settings["end"])
    variables = [str(value) for value in spec.get("variables", ["soil_m"])]
    selected_variables = _available_soil_variables(ds, variables)
    missing_variables = sorted(set(variables) - set(ds.data_vars))
    if missing_variables:
        print(
            "NWM soil moisture: deriving or skipping missing variables: "
            + ", ".join(missing_variables)
        )
    x_name = spec.get("x", "x")
    y_name = spec.get("y", "y")
    frames = []
    point_iter = iter_progress(
        points,
        total=len(points),
        desc="NWM soil points",
        unit="point",
        dynamic_ncols=True,
    )
    for point in point_iter:
        point_id = point.get("id", f"{point[x_name]}_{point[y_name]}")
        if hasattr(point_iter, "set_postfix_str"):
            point_iter.set_postfix_str(str(point_id), refresh=False)
        selected = ds[selected_variables].sel({x_name: point[x_name], y_name: point[y_name]}, method="nearest")
        frame = selected.to_dataframe().reset_index()
        frame = _derive_soilsat_top(frame, requested=variables, spec=spec)
        frame["point_id"] = point_id
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    print(f"NWM soil moisture: writing {len(frame):,} rows to {output_csv}")
    return _write_frame(frame, output_csv)


def _available_soil_variables(ds, variables):
    available = [name for name in variables if name in ds.data_vars]
    if "SOILSAT_TOP" in variables and "SOILSAT_TOP" not in ds.data_vars and "SOIL_M" in ds.data_vars:
        if "SOIL_M" not in available:
            available.append("SOIL_M")
    missing = [name for name in variables if name not in ds.data_vars]
    unsupported = [name for name in missing if name != "SOILSAT_TOP"]
    if unsupported:
        raise KeyError(
            "NWM soil moisture dataset is missing requested variables: "
            + ", ".join(unsupported)
        )
    if not available:
        raise KeyError("No requested NWM soil moisture variables are available")
    return available


def _derive_soilsat_top(frame, *, requested, spec):
    if "SOILSAT_TOP" not in requested or "SOILSAT_TOP" in frame.columns:
        return frame
    if "SOIL_M" not in frame.columns:
        return frame

    layer_name = spec.get("soil_layer_dim", "soil_layers_stag")
    top_layers = [int(value) for value in spec.get("soilsat_top_layers", [0, 1])]
    if layer_name in frame.columns:
        group_columns = [
            name
            for name in frame.columns
            if name not in {layer_name, "SOIL_M"}
        ]
        top = frame[frame[layer_name].isin(top_layers)].copy()
        if top.empty:
            frame["SOILSAT_TOP"] = pd.NA
        else:
            derived = (
                top.groupby(group_columns, dropna=False)["SOIL_M"]
                .mean()
                .clip(lower=0.0, upper=1.0)
                .reset_index(name="SOILSAT_TOP")
            )
            frame = frame.merge(derived, on=group_columns, how="left")
        source = "derived_from_SOIL_M_layers_" + "_".join(str(value) for value in top_layers)
    else:
        frame["SOILSAT_TOP"] = frame["SOIL_M"].clip(lower=0.0, upper=1.0)
        source = "derived_from_SOIL_M"

    frame["SOILSAT_TOP_source"] = source
    return frame


def collect_nwm(settings, skip_existing=False, smoke=False):
    paths = settings["paths"]
    nwm = settings["nwm"]
    paths["nwm_root"].mkdir(parents=True, exist_ok=True)
    streamflow_csv = _configured(paths["nwm_root"] / "streamflow.csv", paths, "nwm_streamflow_csv")
    soil_csv = _configured(paths["nwm_root"] / "soil_moisture.csv", paths, "nwm_soil_moisture_csv")
    soil_spec = nwm.get("soil_moisture", {})
    soil_variables = soil_spec.get("variables", ["soil_m"])
    artifact_covers = source_artifact_covers(
        paths,
        "nwm",
        "retrospective_hydrologic_state",
        settings["start"],
        settings["end"],
    )
    if (
        skip_existing
        and streamflow_csv.exists()
        and soil_csv.exists()
        and artifact_covers
        and not soil_moisture_csv_has_variables(soil_csv, soil_variables)
    ):
        repair_soil_moisture_csv(
            soil_csv,
            variables=soil_variables,
            spec=soil_spec,
        )
    can_reuse = (
        skip_existing
        and streamflow_csv.exists()
        and soil_csv.exists()
        and soil_moisture_csv_has_variables(soil_csv, soil_variables)
        and artifact_covers
    )
    if can_reuse:
        print(f"NWM: reusing complete production artifacts in {paths['nwm_root']}")
        streamflow = pd.read_csv(streamflow_csv)
        soil = pd.read_csv(soil_csv)
    else:
        stages = iter_progress(
            ["streamflow", "soil moisture"],
            total=2,
            desc="NWM sources",
            unit="stage",
            dynamic_ncols=True,
        )
        for stage in stages:
            if stage == "streamflow":
                streamflow = collect_streamflow(settings)
            else:
                soil = collect_soil_moisture(settings)
    write_source_artifact(
        paths,
        source="nwm",
        kind="retrospective_hydrologic_state",
        start=settings["start"],
        end=settings["end"],
        artifacts={
            "streamflow_csv": streamflow_csv,
            "soil_moisture_csv": soil_csv,
        },
        metadata={
            "version": nwm.get("version", "2.1"),
            "bucket": nwm.get("bucket"),
            "streamflow_zarr": nwm.get("streamflow", {}).get("zarr"),
            "soil_moisture_zarr": nwm.get("soil_moisture", {}).get("zarr"),
            "soil_moisture_variables": nwm.get("soil_moisture", {}).get("variables", []),
            "smoke": bool(smoke),
        },
    )
    return {
        "reused": bool(can_reuse),
        "streamflow_rows": int(len(streamflow)),
        "soil_moisture_rows": int(len(soil)),
        "streamflow_csv": streamflow_csv,
        "soil_moisture_csv": soil_csv,
    }
