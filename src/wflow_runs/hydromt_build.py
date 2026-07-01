from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from paths import location_root_from_paths, resolve_location_path
from coupling.wflow_domain_set import plan_wflow_domain_set
from wflow_runs.domain import render_hydromt_build_steps
import wflow_runs.catalog as wflow_catalog
from wflow_runs.gauges import (
    observation_gauge_layer_matches,
    sfincs_gauge_layer_matches,
    write_wflow_observation_gauge_locations,
    write_wflow_sfincs_gauge_locations,
    write_wflow_sfincs_handoff_gauge_locations,
)
import wflow_runs.hydromt_recipe as hydromt_recipe
import wflow_runs.repairs as wflow_repairs
import wflow_runs.reservoirs as wflow_reservoirs
import wflow_runs.staticmaps_qa as staticmaps_qa
import wflow_runs.types as wflow_types


def build_wflow_build_plan(config, paths) -> wflow_types.WflowBuildPlan:
    """Return the notebook-facing HydroMT-Wflow build/update plan."""
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    plugin = str(wflow.get("plugin", "wflow_sbm"))
    base_model_root = resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base"))
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    data_catalog = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    build_config = resolve_location_path(location_root, wflow.get("build_config", "wflow_build.yml"))
    update_forcing_config = resolve_location_path(
        location_root,
        wflow.get("update_forcing_config", "wflow_update_forcing.yml"),
    )
    hydromt_recipe.ensure_model_recipe_file(config, "wflow_build", build_config)
    hydromt_recipe.ensure_model_recipe_file(config, "wflow_update_forcing", update_forcing_config)

    build_workflow = hydromt_recipe.read_workflow(build_config)
    update_workflow = hydromt_recipe.read_workflow(update_forcing_config)
    build_steps = hydromt_recipe.workflow_step_names(build_workflow)
    update_steps = hydromt_recipe.workflow_step_names(update_workflow)
    region_kind = hydromt_recipe.workflow_region_kind(build_workflow)
    domain_set = wflow.get("domain_set", {})
    domain_status = hydromt_recipe.workflow_domain_status(region_kind, domain_set)
    review_required = bool(domain_set.get("review_required", False)) or domain_status != "configured"

    return wflow_types.WflowBuildPlan(
        study_location=str(config.get("project", {}).get("name", paths.get("location_name", location_root.name))),
        plugin=plugin,
        base_model_root=base_model_root,
        events_root=events_root,
        data_catalog=data_catalog,
        build_config=build_config,
        update_forcing_config=update_forcing_config,
        build_steps=build_steps,
        update_steps=update_steps,
        region_kind=region_kind,
        review_required=review_required,
        domain_status=domain_status,
        build_command=(
            f"hydromt build {plugin} {base_model_root} "
            f"-i {build_config} -d {data_catalog} -vvv"
        ),
        update_command=(
            f"hydromt update {plugin} {base_model_root} "
            f"-i {update_forcing_config} -d {data_catalog} "
            f"-o {events_root / '<event_id>'} -vvv"
        ),
    )


def build_wflow_steps_for_submodel(
    build_config: Path,
    submodel: dict,
    *,
    gauges_fn=None,
    sfincs_snap_to_river: bool = True,
    sfincs_snap_uparea: bool = True,
    obs_gauges_fn=None,
    obs_snap_to_river: bool = True,
    obs_snap_uparea: bool = True,
    handoff_config: dict | None = None,
) -> list[dict]:
    """Return HydroMT-Wflow build steps with the reviewed submodel region."""
    submodel = dict(submodel)
    if gauges_fn is not None:
        submodel["gauges_fn"] = gauges_fn
    if obs_gauges_fn is not None:
        submodel["observation_gauges_fn"] = obs_gauges_fn
    config = {"wflow": {"handoff": dict(handoff_config or {})}}
    return render_hydromt_build_steps(
        config,
        Path(),
        build_config,
        submodel,
        sfincs_snap_to_river=sfincs_snap_to_river,
        sfincs_snap_uparea=sfincs_snap_uparea,
        obs_snap_to_river=obs_snap_to_river,
        obs_snap_uparea=obs_snap_uparea,
    )


def build_wflow_submodel(
    config,
    paths,
    *,
    submodel_id: str | None = None,
    model_cls=None,
    force: bool = False,
    write_catalog: bool = True,
) -> dict:
    """Build one reviewed HydroMT-Wflow submodel for a Location Workspace."""
    location_root = location_root_from_paths(paths)
    build_plan = build_wflow_build_plan(config, paths)
    domain_plan = plan_wflow_domain_set(config, paths)
    if domain_plan.status != "ready":
        raise RuntimeError(f"Wflow Domain Set plan is not ready: {domain_plan.status}: {domain_plan.issues}")
    submodel = _select_wflow_submodel(domain_plan.submodels, submodel_id)
    selected_id = str(submodel["wflow_submodel_id"])
    model_root = build_plan.base_model_root / selected_id
    apply_repairs = _legacy_wflow_repairs_enabled(config)
    if (
        _is_built_wflow_model(model_root)
        and not force
        and sfincs_gauge_layer_matches(model_root, submodel, config=config, paths=paths)
        and observation_gauge_layer_matches(model_root, submodel, config)
    ):
        wflow_reservoirs.assert_wflow_reservoir_staticmaps_current(config, model_root, selected_id)
        wflow_repairs.normalize_wflow_staticmaps_nodata(model_root)
        if apply_repairs:
            wflow_repairs.repair_wflow_river_width(model_root)
            wflow_repairs.repair_wflow_canopy_parameters(model_root)
            wflow_repairs.repair_wflow_gauge_map(model_root)
        qa = _wflow_staticmap_qa(model_root, config)
        catalog_path = wflow_catalog.build_wflow_data_catalog(config, paths) if write_catalog else build_plan.data_catalog
        model = staticmaps_qa.open_wflow_model(model_root, catalog_path, model_cls=model_cls, mode="r")
        return {
            "status": "reused",
            "wflow_submodel_id": selected_id,
            "base_model_root": model_root,
            "data_catalog": catalog_path,
            "built": True,
            "staticmap_qa_status": staticmaps_qa.qa_status(qa),
            "model": model,
        }
    if _is_built_wflow_model(model_root) and not force:
        force = True
    if force and model_root.exists():
        shutil.rmtree(model_root)

    wflow_repairs.ensure_wflow_hydrography_basemap_nodata(config, paths)
    catalog_path = wflow_catalog.build_wflow_data_catalog(config, paths) if write_catalog else build_plan.data_catalog
    missing_required = [
        row
        for row in wflow_catalog.wflow_catalog_source_readiness(catalog_path)
        if row["required_for_build"] and row["local_file"] and row["exists"] is False
    ]
    if missing_required:
        raise FileNotFoundError(
            "Missing required HydroMT-Wflow source files before build: "
            + json.dumps(
                [{"source": row["source"], "uri": row["uri"]} for row in missing_required],
                indent=2,
            )
        )

    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    gauge_summary = write_wflow_sfincs_handoff_gauge_locations(config, paths, submodel)
    if outlet_source in {"stream_boundary_crossings", "boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        obs_gauge_summary = (
            write_wflow_observation_gauge_locations(config, paths, submodel)
            if submodel.get("gauge_site_nos")
            else None
        )
    elif outlet_source == "encompassing_huc":
        obs_gauge_summary = (
            write_wflow_observation_gauge_locations(config, paths, submodel)
            if submodel.get("gauge_site_nos")
            else None
        )
    else:
        gauge_summary = write_wflow_sfincs_gauge_locations(config, paths, submodel)
        obs_gauge_summary = write_wflow_observation_gauge_locations(config, paths, submodel)
    steps = build_wflow_steps_for_submodel(
        build_plan.build_config,
        submodel,
        gauges_fn=gauge_summary["gauges_fn"],
        sfincs_snap_to_river=bool(gauge_summary.get("snap_to_river", True)),
        sfincs_snap_uparea=bool(gauge_summary.get("snap_uparea", True)),
        obs_gauges_fn=obs_gauge_summary["gauges_fn"] if obs_gauge_summary else None,
        obs_snap_to_river=obs_gauge_summary.get("snap_to_river", True) if obs_gauge_summary else True,
        obs_snap_uparea=obs_gauge_summary["snap_uparea"] if obs_gauge_summary else True,
        handoff_config=config.get("wflow", {}).get("handoff", {}),
    )
    model = staticmaps_qa.open_wflow_model(model_root, catalog_path, model_cls=model_cls, mode="w+")
    model.build(steps=steps)
    wflow_repairs.normalize_wflow_staticmaps_nodata(model_root)
    if apply_repairs:
        wflow_repairs.repair_wflow_river_width(model_root)
        wflow_repairs.repair_wflow_canopy_parameters(model_root)
    qa = _wflow_staticmap_qa(model_root, config)
    return {
        "status": "built",
        "wflow_submodel_id": selected_id,
        "base_model_root": model_root,
        "data_catalog": catalog_path,
        "gauges_fn": gauge_summary["gauges_fn"],
        "gauge_count": gauge_summary["gauge_count"],
        "observation_gauges_fn": obs_gauge_summary["gauges_fn"] if obs_gauge_summary else None,
        "observation_gauge_count": obs_gauge_summary["gauge_count"] if obs_gauge_summary else 0,
        "built": _is_built_wflow_model(model_root),
        "staticmap_qa_status": staticmaps_qa.qa_status(qa),
        "model": model,
    }


def _legacy_wflow_repairs_enabled(config: dict) -> bool:
    return bool((config.get("wflow", {}) or {}).get("apply_legacy_repairs", False))


def _wflow_staticmap_qa(model_root: Path, config: dict) -> pd.DataFrame:
    try:
        report = staticmaps_qa.validate_staticmaps(
            model_root,
            river_upa_km2=config.get("inland_coupling", {}).get("discharge_forcing", {}).get("river_upa_km2"),
            raise_on_error=False,
        )
        if wflow_reservoirs.wflow_reservoirs_enabled(config):
            reservoir_report = wflow_reservoirs.validate_wflow_reservoir_staticmaps(
                model_root,
                required=True,
                raise_on_error=False,
            )
            report = pd.concat([report, reservoir_report], ignore_index=True)
        return report
    except FileNotFoundError as exc:
        return pd.DataFrame(
            [{"check": "staticmaps", "status": "not_available", "message": str(exc)}]
        )


def _select_wflow_submodel(submodels: tuple[dict, ...], submodel_id: str | None) -> dict:
    if submodel_id is None:
        return submodels[0]
    for submodel in submodels:
        if str(submodel["wflow_submodel_id"]) == str(submodel_id):
            return submodel
    available = ", ".join(str(submodel["wflow_submodel_id"]) for submodel in submodels)
    raise ValueError(f"Wflow Submodel not found: {submodel_id}. Available submodels: {available}")


def _is_built_wflow_model(model_root: Path) -> bool:
    if not model_root.exists():
        return False
    files = {path.name for path in model_root.rglob("*") if path.is_file() and path.name != ".gitkeep"}
    return bool(files & {"wflow_sbm.toml", "staticmaps.nc", "staticgeoms.nc"})
