from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .domain import domain_submodels, model_crs, read_handoff_points
from .qa import read_acceptance, validate_event_boundary, write_acceptance
from .states import prepare_event_instate, validate_instates
from .types import BoundaryRun, DesignEvent
from paths import resolve_location_path, write_json
from wflow_runs.runner import clean_output_dir, run_solver, zero_event_forcing

CFS_TO_CMS = 0.028316846592
_GENERATED = "# GENERATED FILE - source: wflow_runs.event\n"
_LEGACY_REPLAY_NOTICE = (
    "# GENERATED FILE - do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


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


def _ensure_local_catalog(config: dict[str, Any], location_root: str | Path) -> Path:
    location_root = Path(location_root)
    wflow = config.get("wflow", {}) or {}
    path = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    if path.exists() and not bool(wflow.get("rewrite_data_catalog", False)):
        return path

    collection = config.get("collection", {}) or {}
    hydro = collection.get("national_hydrography", {}) or {}
    static = config.get("static_sources", {}) or {}
    project = config.get("project", {}) or {}
    crs = str(project.get("reference_crs", "EPSG:4326"))
    events_root = Path(wflow.get("events_root", "data/wflow/events"))

    def abs_uri(value):
        return str(resolve_location_path(location_root, value).resolve())

    catalog: dict[str, Any] = {
        "meta": {"roots": [".."], "source": "wflow_runs.event"},
        "event_precip": _raster_xarray(abs_uri(events_root / "<event_id>" / "precip.nc"), "event_forcing"),
        "event_temp_pet": _raster_xarray(abs_uri(events_root / "<event_id>" / "temp_pet.nc"), "event_forcing"),
    }
    if hydro.get("hydromt_basemap"):
        catalog["us_hydrography_basemap"] = _raster_xarray(abs_uri(hydro["hydromt_basemap"]), "hydrography")
    if hydro.get("river_geometry"):
        catalog["nhdplus_hr_river_geometry"] = _geodataframe(abs_uri(hydro["river_geometry"]), crs, "hydrography")
    reservoirs = (hydro.get("reservoirs", {}) or {}).get("output") or hydro.get("reservoirs_output")
    if reservoirs:
        catalog["nhdplus_hr_wflow_reservoirs"] = _geodataframe(abs_uri(reservoirs), crs, "hydrography")
    landcover = ((static.get("wflow_collection_extent", {}) or {}).get("landcover_output") or (static.get("landcover", {}) or {}).get("output"))
    if landcover:
        catalog["esa_worldcover"] = _rasterio(abs_uri(landcover), "landuse")
    soil = hydro.get("wflow_soil_parameters")
    if soil:
        name = (((wflow.get("source_strategy", {}) or {}).get("soils", {}) or {}).get("wflow_parameters") or "ssurgo_wflow_soil_parameters")
        catalog[name] = _raster_xarray(abs_uri(soil), "soils")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_GENERATED + yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def _bind_event_catalog_template(data_catalog: str | Path, event_root: str | Path, event_id: str) -> Path:
    text = Path(data_catalog).read_text(encoding="utf-8").replace("<event_id>", str(event_id))
    out = Path(event_root) / "_hydromt_data_catalog.yml"
    out.write_text(_GENERATED + text, encoding="utf-8")
    return out


def _write_event_update_workflow(update_config: str | Path, event_root: str | Path, start, end) -> Path:
    payload = yaml.safe_load(Path(update_config).read_text(encoding="utf-8")) or {}
    _set_update_times(payload, start, end)
    out = Path(event_root) / "_hydromt_update.yml"
    out.write_text(_GENERATED + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


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


def write_legacy_replay_data_catalog(data_catalog: str | Path, event_dir: str | Path, event_id: str) -> Path:
    """Materialise the old replay data catalog with runtime-local event URIs."""
    data_catalog = Path(data_catalog)
    event_dir = Path(event_dir)
    location_root = data_catalog.resolve().parents[2]
    marker = f"/locations/{location_root.name}/"
    text = data_catalog.read_text(encoding="utf-8").replace("<event_id>", str(event_id))

    def reroot(match: "re.Match[str]") -> str:
        uri = match.group("path").strip()
        if marker in uri:
            uri = str(location_root / uri.split(marker, 1)[1])
        elif not uri.startswith("/"):
            uri = str(location_root / uri)
        return f"{match.group('indent')}{uri}"

    text = re.sub(r"(?m)^(?P<indent>\s*uri:\s*)(?P<path>\S.*)$", reroot, text)
    out = event_dir / "_replay_data_catalog.yml"
    out.write_text(
        "# GENERATED FILE - do not edit by hand. Overwritten when the Wflow event replay step runs.\n"
        + text,
        encoding="utf-8",
    )
    return out


def write_legacy_replay_update_config(update_config: str | Path, event_dir: str | Path, start, end) -> Path:
    """Copy the old update-forcing config with replay window timestamps inserted."""
    workflow = yaml.safe_load(Path(update_config).read_text(encoding="utf-8")) or {}
    _set_update_times(workflow, start, end)
    out = Path(event_dir) / "_wflow_update_forcing.yml"
    out.write_text(
        _LEGACY_REPLAY_NOTICE.format(source="the Wflow event replay step")
        + yaml.safe_dump(workflow, sort_keys=False),
        encoding="utf-8",
    )
    return out


def clean_legacy_replay_submodel_output_dir(event_dir: str | Path, out_dir: str | Path) -> None:
    """Remove a generated event-submodel directory without touching event artifacts."""
    event_dir = Path(event_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == event_dir or event_dir not in out_dir.parents:
        raise ValueError(f"refusing to clean replay output outside event dir: {out_dir}")
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()


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


def require_discharge_window(
    discharge_nc: str | Path,
    *,
    expected_end,
    event_id: str | None = None,
) -> pd.Timestamp:
    """Require a Wflow-to-SFINCS discharge file to cover the event window."""
    path = Path(discharge_nc)
    with xr.open_dataset(path) as ds:
        if "time" not in ds:
            raise RuntimeError(f"Dynamic Wflow handoff discharge lacks a time coordinate: {path}")
        actual_end = pd.Timestamp(ds["time"].max().values)
    expected_end = pd.Timestamp(expected_end)
    if actual_end < expected_end:
        label = "event" if event_id is None else str(event_id)
        raise RuntimeError(
            f"Dynamic Wflow handoff for {label} is stale: {path} ends at "
            f"{actual_end.isoformat()}, but the current Wflow event window ends at {expected_end.isoformat()}. "
            "Rerun 04/b_prepare_wflow_dynamic_handoff.ipynb with rerun=True so meteo, Wflow discharge, "
            "zero-rain QA, and SFINCS staging use the extended window."
        )
    return actual_end


def _probability_metadata(row) -> dict[str, float]:
    pairs = {
        "p_event": _float_or_none(_first(row, ["p_event", "probability", "event_probability"])),
        "aep": _float_or_none(_first(row, ["aep", "annual_exceedance_probability"])),
        "return_period_years": _float_or_none(_first(row, ["return_period_years", "sample_rp_years", "rp_years"])),
        "weight": _float_or_none(_first(row, ["weight", "catalog_weight"])),
    }
    return {key: value for key, value in pairs.items() if value is not None}


def _set_update_times(payload: dict[str, Any], start, end) -> None:
    def update(node):
        data = node.setdefault("data", {})
        data["time.starttime"] = pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S")
        data["time.endtime"] = pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%S")

    if isinstance(payload.get("setup_config"), dict):
        update(payload["setup_config"])
    for step in payload.get("steps", []) or []:
        if isinstance(step, dict) and isinstance(step.get("setup_config"), dict):
            update(step["setup_config"])


def _raster_xarray(uri: str, category: str) -> dict[str, Any]:
    return {"data_type": "RasterDataset", "driver": {"name": "raster_xarray"}, "uri": uri, "metadata": {"category": category}}


def _rasterio(uri: str, category: str) -> dict[str, Any]:
    return {"data_type": "RasterDataset", "driver": {"name": "rasterio"}, "uri": uri, "metadata": {"category": category}}


def _geodataframe(uri: str, crs: str, category: str) -> dict[str, Any]:
    return {"data_type": "GeoDataFrame", "driver": {"name": "pyogrio"}, "uri": uri, "metadata": {"crs": crs, "category": category}}


def run_event_boundary(
    config: dict[str, Any],
    location_root: str | Path,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = True,
    force: bool = False,
    zero_rain: bool = False,
    model_cls=None,
) -> BoundaryRun:
    """Run one stochastic event through HydroMT-Wflow and write SFINCS discharge forcing."""
    root = Path(location_root)
    event = read_event(config, root, event_id, catalog_path=catalog_path)
    paths = event_paths(config, root, event_id)
    paths["event_root"].mkdir(parents=True, exist_ok=True)
    if execute:
        validate_instates(config, root, raise_on_error=True)
        _require_event_forcing(event)

    submodels = domain_submodels(config, root)
    if not submodels:
        raise RuntimeError("No Wflow domain submodels are configured or manifested")
    wflow = config.get("wflow", {}) or {}
    base_root = resolve_location_path(root, wflow.get("base_model_root", "data/wflow/base"))
    update_config = resolve_location_path(root, wflow.get("update_forcing_config", "wflow_update_forcing.yml"))
    base_catalog = _ensure_local_catalog(config, root)
    event_catalog = _bind_event_catalog_template(base_catalog, paths["event_root"], event.event_id)
    event_update = _write_event_update_workflow(update_config, paths["event_root"], event.window_start, event.window_end)
    update_steps = _workflow_steps(event_update)

    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for submodel in submodels:
        sid = str(submodel["wflow_submodel_id"])
        base_model = base_root / sid
        event_model = paths["event_root"] / sid
        if execute:
            if force and event_model.exists():
                shutil.rmtree(event_model)
            _update_model(base_model, event_model, steps=update_steps, data_libs=[str(event_catalog)], model_cls=model_cls)
            prepare_event_instate(event_model, base_model, model_cls=model_cls)
            clean_output_dir(event_model / "wflow_sbm.toml")
            run_solver(config, event_model / "wflow_sbm.toml", cwd=root)
        gauges = event_model / "staticgeoms" / "gauges_sfincs.geojson"
        if not gauges.exists():
            gauges = base_model / "staticgeoms" / "gauges_sfincs.geojson"
        outputs.append({"run_model_root": event_model, "run_output_dir": event_model / "run_event", "gauges_geojson": gauges, "submodel_id": sid})
        rows.append({"event_id": event.event_id, "wflow_submodel_id": sid, "event_model_root": str(event_model), "run_output_dir": str(event_model / "run_event"), "gauges_geojson": str(gauges), "status": "completed" if execute else "planned"})

    amplification = {"K": 1.0, "status": "not_run"}
    zero_path = None
    if execute:
        from coupling import amplification as coupling_amplification
        from coupling import discharge as coupling_discharge

        coupling_discharge.merge_submodel_discharge(
            outputs,
            model_crs=model_crs(config),
            out_path=paths["discharge"],
            handoff_points=coupling_discharge.sfincs_handoff_points(config, root, model_crs(config)),
        )
        amplification = coupling_amplification.apply_same_frequency_amplification(
            config,
            root,
            event.event_id,
            discharge_nc=paths["discharge"],
            submodel_runs=outputs,
            event=event,
            write_provenance=False,
        )
        write_json(paths["amplification"], amplification)
        if zero_rain:
            zero_path = run_zero_rain_control(config, root, event.event_id, execute=True, model_cls=model_cls)
        expected = {p.id for p in read_handoff_points(config, root, crs=model_crs(config))}
        qa = validate_event_boundary(paths["discharge"], expected_source_ids=expected, window=(event.window_start, event.window_end), zero_rain_discharge_nc=zero_path, raise_on_error=False)
        qa.to_csv(paths["qa_csv"], index=False)
        write_acceptance(paths["acceptance"], event=event, discharge_nc=paths["discharge"], qa_report=qa, amplification=amplification, metadata={"hydromt_update_workflow": str(event_update), "hydromt_data_catalog": str(event_catalog)})
        status = "accepted" if not qa["status"].isin(["failed", "review_required"]).any() else "failed"
    else:
        qa = pd.DataFrame()
        status = "planned"
    return BoundaryRun(event=event, discharge_nc=paths["discharge"], acceptance_json=paths["acceptance"], qa_csv=paths["qa_csv"], status=status, amplification=amplification, report=pd.DataFrame(rows))


def require_event_boundary(config: dict[str, Any], location_root: str | Path, event_id: str) -> pd.Series:
    paths = event_paths(config, location_root, event_id)
    payload = read_acceptance(paths["acceptance"])
    if payload.get("status") != "accepted":
        raise RuntimeError(f"Wflow event boundary is not accepted: {paths['acceptance']}")
    if not paths["discharge"].exists():
        raise FileNotFoundError(paths["discharge"])
    return pd.Series({"event_id": str(event_id), "status": "accepted", "sfincs_discharge": str(paths["discharge"]), "acceptance_json": str(paths["acceptance"])}, name="wflow_event_boundary_acceptance")


def run_zero_rain_control(config: dict[str, Any], location_root: str | Path, event_id: str, *, execute: bool = True, model_cls=None) -> Path:
    root = Path(location_root)
    paths = event_paths(config, root, event_id)
    zero_root = paths["event_root"] / "_zero_rain"
    outputs: list[dict[str, Any]] = []
    for submodel in domain_submodels(config, root):
        sid = str(submodel["wflow_submodel_id"])
        source = paths["event_root"] / sid
        target = zero_root / sid
        if execute:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            zero_event_forcing(target / "inmaps-event.nc")
            clean_output_dir(target / "wflow_sbm.toml")
            run_solver(config, target / "wflow_sbm.toml", cwd=root)
        outputs.append({"run_model_root": target, "run_output_dir": target / "run_event", "gauges_geojson": target / "staticgeoms" / "gauges_sfincs.geojson", "submodel_id": sid})
    if execute:
        from coupling import discharge as coupling_discharge

        coupling_discharge.merge_submodel_discharge(
            outputs,
            model_crs=model_crs(config),
            out_path=paths["zero_rain_discharge"],
            handoff_points=coupling_discharge.sfincs_handoff_points(config, root, model_crs(config)),
        )
    return paths["zero_rain_discharge"]


def _workflow_steps(workflow_file: str | Path) -> list[dict[str, Any]]:
    from hydromt.readers import read_workflow_yaml

    _, _, steps = read_workflow_yaml(str(workflow_file))
    return list(steps or [])


def _update_model(base_root: str | Path, out_root: str | Path, *, steps: list[dict[str, Any]], data_libs: list[str], model_cls=None):
    cls = _wflow_model_cls(model_cls)
    model = cls(root=str(base_root), mode="r", data_libs=data_libs)
    model.update(model_out=str(out_root), steps=steps)
    return model


def _read_model(root: str | Path, *, model_cls=None, mode: str = "r"):
    cls = _wflow_model_cls(model_cls)
    model = cls(root=str(root), mode=mode)
    model.read()
    return model


def _wflow_model_cls(model_cls=None):
    if model_cls is not None:
        return model_cls
    from hydromt_wflow import WflowSbmModel

    return WflowSbmModel


def _require_event_forcing(event: DesignEvent) -> None:
    missing = [str(path) for path in [event.precip_path, event.temp_pet_path] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Wflow event forcing is missing: " + "; ".join(missing))


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
