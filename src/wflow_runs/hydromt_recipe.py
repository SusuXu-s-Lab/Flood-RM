from __future__ import annotations

from pathlib import Path

import yaml


GENERATED_NOTICE = (
    "# GENERATED FILE - do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


def ensure_model_recipe_file(config: dict, key: str, path: Path) -> Path:
    """Write an extracted HydroMT model recipe when the Location config carries one."""
    recipe = (config.get("_model_recipes") or {}).get(key)
    if recipe is None:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        GENERATED_NOTICE.format(source=f"the {key} model YAML extraction")
        + yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_workflow(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return normalize_workflow(workflow)


def normalize_workflow(workflow: dict) -> dict:
    """Return HydroMT workflow steps from either supported YAML shape."""
    if "steps" in workflow:
        return workflow

    steps = []
    passthrough = {}
    for name, options in workflow.items():
        if is_hydromt_step_name(name):
            steps.append({name: {} if options is None else options})
        else:
            passthrough[name] = options
    return {**passthrough, "steps": steps}


def is_hydromt_step_name(name) -> bool:
    text = str(name)
    return text.startswith("setup_") or text.startswith("write_") or "." in text


def wflow_build_uses_river_geometry(build_config: Path) -> bool:
    if not Path(build_config).exists():
        return True
    workflow = yaml.safe_load(Path(build_config).read_text(encoding="utf-8")) or {}
    steps = workflow.get("steps") if isinstance(workflow.get("steps"), list) else [workflow]
    for step in steps:
        if not isinstance(step, dict) or "setup_rivers" not in step:
            continue
        river_geom = (step.get("setup_rivers") or {}).get("river_geom_fn")
        return river_geom not in (None, "")
    return False


def workflow_step_names(workflow: dict) -> tuple[str, ...]:
    names = []
    for step in workflow.get("steps", []):
        if isinstance(step, dict) and step:
            names.append(str(next(iter(step))))
    return tuple(names)


def workflow_region_kind(workflow: dict) -> str:
    for step in workflow.get("steps", []):
        if not isinstance(step, dict):
            continue
        basemaps = step.get("setup_basemaps")
        if not isinstance(basemaps, dict):
            continue
        region = basemaps.get("region", {})
        if isinstance(region, dict) and region:
            return str(next(iter(region)))
    return "missing"


def workflow_domain_status(region_kind: str, domain_set: dict) -> str:
    submodels = domain_set.get("submodels") or []
    if region_kind == "bbox" and domain_set.get("allow_multiple_submodels") is False:
        return "configured"
    if region_kind == "bbox" and not submodels:
        return "review_required_bbox_placeholder"
    if region_kind in {"basin", "subbasin", "interbasin"} or submodels:
        return "configured"
    return "review_required_missing_region"
