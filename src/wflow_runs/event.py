from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import shutil

import pandas as pd
import xarray as xr
import yaml

from .domain import domain_submodels, model_crs, read_handoff_points
from .event_catalog import (
    catalog_rainfall_start,
    configured_event_window_hours,
    event_catalog_path,
    event_paths,
    event_reference_time,
    event_window,
    legacy_dynamic_handoff_paths,
    legacy_event_catalog_row,
    read_event,
    required_event_value,
)
from .qa import read_acceptance, validate_event_boundary, write_acceptance
from .states import prepare_event_instate, validate_instates
from .types import BoundaryRun, DesignEvent
from paths import resolve_location_path, write_json
from wflow_runs.runner import clean_output_dir, run_solver, zero_event_forcing

_GENERATED = "# GENERATED FILE - source: wflow_runs.event\n"
_LEGACY_REPLAY_NOTICE = (
    "# GENERATED FILE - do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


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

