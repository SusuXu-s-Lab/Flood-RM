from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from .domain import domain_submodels, plan_domain
from .paths import location_path
from wflow_boundary_compat.catalog import ensure_local_catalog
from wflow_boundary_compat.hydromt_native import build_model, workflow_steps


DEFAULT_Q_STANDARD_NAME = "river_water__volume_flow_rate"


def build_base_models(
    config: dict[str, Any],
    location_root: str | Path,
    *,
    submodel_ids: list[str] | None = None,
    force: bool = False,
    execute: bool = True,
    model_cls=None,
) -> pd.DataFrame:
    """Build reviewed Wflow base submodels with native HydroMT-Wflow calls.

    The only project-specific work here is translating the reviewed SFINCS handoff
    manifest into native ``setup_basemaps`` + ``setup_gauges`` workflow steps. HydroMT-
    Wflow owns static maps, geoms, TOML writing, and component layout.
    """
    root = Path(location_root)
    wflow = config.get("wflow", {}) or {}
    base_root = location_path(root, wflow.get("base_model_root", "data/wflow/base"))
    build_config = location_path(root, wflow.get("build_config", "wflow_build.yml"))
    data_catalog = ensure_local_catalog(config, root)
    submodels = domain_submodels(config, root) or plan_domain(config, root, write=True)
    wanted = {str(v) for v in submodel_ids or []}
    if wanted:
        submodels = [s for s in submodels if str(s.get("wflow_submodel_id")) in wanted]
    rows: list[dict[str, Any]] = []
    for submodel in submodels:
        sid = str(submodel["wflow_submodel_id"])
        model_root = base_root / sid
        if force and model_root.exists() and execute:
            shutil.rmtree(model_root)
        reusable = _built(model_root) and not force
        if execute and not reusable:
            steps = render_build_steps(config, root, build_config, submodel)
            build_model(model_root, steps=steps, data_libs=[str(data_catalog)], model_cls=model_cls)
        status = "reused" if reusable else "built" if execute else "planned"
        qa = validate_staticmaps(model_root, raise_on_error=False) if _built(model_root) else pd.DataFrame()
        rows.append(
            {
                "wflow_submodel_id": sid,
                "status": status,
                "base_model_root": str(model_root),
                "data_catalog": str(data_catalog),
                "build_config": str(build_config),
                "staticmap_status": _qa_status(qa),
            }
        )
    return pd.DataFrame(rows)


def render_build_steps(config: dict[str, Any], location_root: str | Path, build_config: str | Path, submodel: dict[str, Any]) -> list[dict[str, Any]]:
    """Return native HydroMT-Wflow workflow steps for one submodel."""
    steps = deepcopy(workflow_steps(build_config))
    region = submodel.get("hydromt_region") or submodel.get("region")
    if not region:
        raise ValueError(f"{submodel.get('wflow_submodel_id')} has no HydroMT region")
    _replace_basemap_region(steps, region)

    handoff = (config.get("wflow", {}) or {}).get("handoff", {}) or {}
    q_standard_name = str(handoff.get("source_standard_name", DEFAULT_Q_STANDARD_NAME))
    gauges_fn = submodel.get("gauges_fn") or submodel.get("sfincs_gauges_fn")
    if gauges_fn:
        _set_setup_gauges(
            steps,
            location_path(location_root, gauges_fn),
            basename="sfincs",
            gauge_toml_param=q_standard_name,
            derive_subcatch=True,
            replace_existing_unnamed=True,
        )
    obs_fn = submodel.get("observation_gauges_fn")
    if obs_fn:
        _set_setup_gauges(
            steps,
            location_path(location_root, obs_fn),
            basename="usgs",
            gauge_toml_param=q_standard_name,
            derive_subcatch=False,
        )
    return steps


def validate_staticmaps(model_root: str | Path, *, raise_on_error: bool = True) -> pd.DataFrame:
    """Small stakeholder QA over HydroMT-Wflow native ``staticmaps.nc``."""
    path = Path(model_root) / "staticmaps.nc"
    rows: list[dict[str, str]] = []
    if not path.exists():
        rows.append({"check": "staticmaps", "status": "failed", "message": f"missing {path}"})
        return _report(rows, raise_on_error)
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        for name in ["subcatchment", "local_drain_direction", "river_mask"]:
            rows.append({"check": name, "status": "passed" if name in ds else "failed", "message": "present" if name in ds else "missing"})
        active = np.asarray(ds["river_mask"].values) > 0 if "river_mask" in ds else None
        for name in ["river_width", "river_depth"]:
            if name not in ds:
                rows.append({"check": name, "status": "review_required", "message": "missing; check setup_rivers recipe"})
                continue
            values = np.asarray(ds[name].values, dtype=float)
            mask = np.isfinite(values) & (values > 0)
            if active is not None and active.shape == values.shape:
                mask = mask & active
            count = int(mask.sum())
            rows.append({"check": name, "status": "passed" if count else "failed", "message": f"positive_river_cells={count}"})
    return _report(rows, raise_on_error)


def _replace_basemap_region(steps: list[dict[str, Any]], region: dict[str, Any]) -> None:
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("setup_basemaps"), dict):
            step["setup_basemaps"]["region"] = deepcopy(region)
            return
    steps.insert(0, {"setup_basemaps": {"region": deepcopy(region)}})


def _set_setup_gauges(
    steps: list[dict[str, Any]],
    gauges_fn: str | Path,
    *,
    basename: str,
    gauge_toml_param: str,
    derive_subcatch: bool,
    replace_existing_unnamed: bool = False,
) -> None:
    step = {
        "setup_gauges": {
            "gauges_fn": str(gauges_fn),
            "index_col": "index",
            "snap_to_river": True,
            "snap_uparea": False,
            "basename": basename,
            "gauge_toml_header": ["Q"],
            "gauge_toml_param": [gauge_toml_param],
            "derive_subcatch": bool(derive_subcatch),
        }
    }
    for i, existing in enumerate(steps):
        if not (isinstance(existing, dict) and isinstance(existing.get("setup_gauges"), dict)):
            continue
        current = existing["setup_gauges"].get("basename")
        if current == basename or (replace_existing_unnamed and current in (None, "", "gauges")):
            steps[i] = step
            return
    steps.append(step)


def _built(model_root: Path) -> bool:
    return (model_root / "wflow_sbm.toml").exists() and (model_root / "staticmaps.nc").exists()


def _qa_status(report: pd.DataFrame) -> str:
    if report.empty:
        return "not_run"
    if report["status"].isin(["failed"]).any():
        return "failed"
    if report["status"].isin(["review_required"]).any():
        return "review_required"
    return "passed"


def _report(rows: list[dict[str, Any]], raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow staticmap QA failed: {details}")
    return report
