from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pandas as pd
import yaml

from design_events.runtime import load_runtime as load_design_runtime
from sfincs_runs.config import load_runtime as load_sfincs_runtime
from study_location import define_location
from wflow_runs.build_plan import (
    build_wflow_build_plan,
    plan_wflow_domain_set,
    write_wflow_subbasin_fabric_from_nhdplus,
)


@dataclass(frozen=True)
class WflowNotebookContext:
    location_root: Path
    repo_root: Path
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    runtime_config: dict


@dataclass(frozen=True)
class WflowCoupledNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    design_paths: dict
    runtime_config: dict
    sfincs_config: dict
    wflow_config: dict
    sfincs_scenarios_root: Path
    scenario_catalog_path: Path
    probability_catalog_path: Path
    readiness_path: Path
    blocked_path: Path
    accepted_path: Path
    joint_worklist_path: Path
    incompatible_path: Path
    events_root: Path
    wflow_base_root: Path
    wflow_handoff_manifest: Path

    def resolve_location_path(self, value) -> Path:
        return resolve_location_path(self.location_root, value)

    def ensure_parent(self, value) -> Path:
        path = self.resolve_location_path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class WflowCalibrationNotebookRuntime(WflowCoupledNotebookRuntime):
    streamflow_records_path: Path
    event_streamflow_iv_root: Path
    audit_plots_dir: Path


def _load_wflow_sfincs_runtime(
    location_root,
    *,
    wflow_domain_review_required: bool | None = None,
) -> WflowCoupledNotebookRuntime:
    """Load derived paths for Wflow-coupled Flood Notebook Workflow stages.

    Generated artifact paths stay derived from the Location Workspace convention
    instead of being repeated in notebook cells or location YAML.
    """
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    config, paths = load_sfincs_runtime(location_root / "config.yaml")
    _, design_paths = load_design_runtime(location_root / "config.yaml")
    config = deepcopy(config)
    if wflow_domain_review_required is not None:
        config["wflow"]["domain_set"]["review_required"] = bool(wflow_domain_review_required)
    config.setdefault("scenario_run", {})

    location_name = location_root.name
    sfincs_scenarios_root = location_root / "data/sfincs/scenarios"
    wflow = config.get("wflow", {})
    return WflowCoupledNotebookRuntime(
        location_root=location_root,
        location_name=location_name,
        repo_root=repo_root,
        config=config,
        paths=paths,
        design_paths=design_paths,
        runtime_config=config,
        sfincs_config=config,
        wflow_config={"wflow": wflow},
        sfincs_scenarios_root=sfincs_scenarios_root,
        scenario_catalog_path=location_root / "data/event_catalog/catalog/scenario_catalog.csv",
        probability_catalog_path=location_root / "data/event_catalog/catalog/probability_catalog.csv",
        readiness_path=sfincs_scenarios_root / f"{location_name}_dynamic_handoff_readiness.csv",
        blocked_path=sfincs_scenarios_root / f"{location_name}_blocked_dynamic_handoffs.csv",
        accepted_path=sfincs_scenarios_root / f"{location_name}_accepted_dynamic_handoffs.csv",
        joint_worklist_path=sfincs_scenarios_root / f"{location_name}_joint_wflow_sfincs_worklist.csv",
        incompatible_path=sfincs_scenarios_root / f"{location_name}_incompatible_dynamic_handoffs.csv",
        events_root=resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events")),
        wflow_base_root=resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base")),
        wflow_handoff_manifest=resolve_location_path(
            location_root,
            wflow.get("handoff", {}).get("manifest", "data/wflow/domain_set_handoff.yaml"),
        ),
    )


def load_calibration_runtime(
    location_root,
    *,
    create_audit_dirs: bool = True,
) -> WflowCalibrationNotebookRuntime:
    """Load derived paths for the Wflow Readiness calibration notebook.

    When ``create_audit_dirs`` is true, the helper creates the calibration plot
    output directory used by the notebook.
    """
    base = _load_wflow_sfincs_runtime(location_root, wflow_domain_review_required=False)
    streamflow_records_path = base.location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    event_streamflow_iv_root = base.location_root / "data/sources/usgs_streamgages/event_streamflow_iv"
    audit_plots_dir = base.location_root / "data/wflow/audit/plots"
    if create_audit_dirs:
        audit_plots_dir.mkdir(parents=True, exist_ok=True)
    return WflowCalibrationNotebookRuntime(
        **base.__dict__,
        streamflow_records_path=streamflow_records_path,
        event_streamflow_iv_root=event_streamflow_iv_root,
        audit_plots_dir=audit_plots_dir,
    )


def load_runtime(location_root, *, workflow: str = "coupled", **kwargs):
    """Load the Wflow notebook runtime for a named workflow."""
    if workflow == "coupled":
        return _load_wflow_sfincs_runtime(location_root, **kwargs)
    if workflow == "calibration":
        return load_calibration_runtime(location_root, **kwargs)
    raise ValueError("workflow must be 'coupled' or 'calibration'")


def load_wflow_notebook_context(location_name: str | None = None, *, start: Path | None = None) -> WflowNotebookContext:
    location_root = find_location_root(location_name, start=start)
    repo_root = location_root.parents[1]
    config = _read_yaml(location_root / "config.yaml")
    includes = config.get("includes", {})
    grid_include = includes.get("smartds") or includes.get("grid")
    grid_config = _read_yaml(location_root / grid_include) if grid_include else {}
    data_sources = _read_yaml(location_root / includes["data_sources"]) if "data_sources" in includes else {}
    sfincs_config = _read_yaml(location_root / config["includes"]["sfincs"])
    wflow_include = includes.get("wflow")
    wflow_config = _read_yaml(location_root / wflow_include) if wflow_include else {}
    runtime_config = define_location(location_root / "config.yaml").config
    if not data_sources:
        data_sources = runtime_config
    if not wflow_config and "wflow" in runtime_config:
        wflow_config = {"wflow": runtime_config["wflow"]}
    return WflowNotebookContext(
        location_root=location_root,
        repo_root=repo_root,
        config=config,
        grid_config=grid_config,
        data_sources=data_sources,
        sfincs_config=sfincs_config,
        wflow_config=wflow_config,
        runtime_config=runtime_config,
    )


def find_location_root(location_name: str | None = None, *, start: Path | None = None) -> Path:
    # With no name, resolve the location from the working directory: the nearest
    # ancestor that holds a config.yaml and sits directly under locations/. A name
    # narrows the match to that specific location workspace.
    here = (start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        if (base / "config.yaml").exists() and (
            base.parent.name == "locations" if location_name is None else base.name == location_name
        ):
            return base
        if location_name is not None:
            candidate = base / "locations" / location_name
            if (candidate / "config.yaml").exists():
                return candidate
    if location_name is not None:
        fallback = Path("locations") / location_name
        if (fallback / "config.yaml").exists():
            return fallback.resolve()
    raise FileNotFoundError(
        "Could not locate a locations/<name>/config.yaml above the working directory"
        if location_name is None
        else f"Could not locate locations/{location_name}/config.yaml"
    )


def resolve_location_path(location_root: Path, relative_path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else location_root / path


def exists_table(location_root: Path, named_paths: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(resolve_location_path(location_root, relative_path)),
                "exists": resolve_location_path(location_root, relative_path).exists(),
            }
            for name, relative_path in named_paths.items()
        ]
    )


def domain_summary(config: dict, location_root: Path) -> tuple:
    build_plan = build_wflow_build_plan(config, {"location_root": location_root})
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    wflow_cfg = config.get("wflow", {}) or {}
    domain_set = wflow_cfg.get("domain_set", {}) or {}
    # Use the same defaults the rest of wflow_runs applies via .get(), so locations that
    # don't spell out every optional domain_set key in YAML (greensboro, austin) summarize
    # instead of KeyError'ing. Defaults mirror write_wflow_domain_set_manifest (build_plan).
    summary = pd.Series(
        {
            "allow_multiple_submodels": domain_set.get("allow_multiple_submodels", False),
            "review_required": build_plan.review_required,
            "domain_status": build_plan.domain_status,
            "reviewed_subbasin_plan_status": domain_plan.status,
            "hydromt_region_kind": build_plan.region_kind,
            "event_catalog_scope": domain_set.get("event_catalog_scope", "shared_across_domain_set"),
            "configured_submodel_count": len(domain_set.get("submodels", []) or []),
            "reviewed_submodel_count": domain_plan.submodel_count,
            "reviewed_handoff_count": domain_plan.handoff_count,
            "domain_set_manifest": wflow_cfg.get("domain_set_manifest", "data/wflow/domain_set.yaml"),
        }
    )
    return build_plan, domain_plan, summary


def subbasins(domain_plan) -> pd.DataFrame:
    if not domain_plan.submodels:
        return pd.DataFrame(
            [{"status": domain_plan.status, "issue": issue} for issue in domain_plan.issues]
        )
    rows = []
    for submodel in domain_plan.submodels:
        outlet_region = submodel.get("outlet_region", submodel["region"])
        outlet_xy = outlet_region.get("subbasin") if isinstance(outlet_region, dict) else None
        outlet_lon, outlet_lat = outlet_xy if outlet_xy else (None, None)
        rows.append(
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "hydromt_region_kind": submodel["region_kind"],
                "hydromt_region": submodel["region"],
                "handoff_outlet_lon": outlet_lon,
                "handoff_outlet_lat": outlet_lat,
                "sfincs_domain_ids": ", ".join(submodel["sfincs_domain_ids"]),
                "sfincs_handoff_ids": ", ".join(submodel["sfincs_handoff_ids"]),
                "gauge_site_nos": ", ".join(submodel["gauge_site_nos"]),
                "frequency_basis": ", ".join(submodel["frequency_basis"]),
            }
        )
    return pd.DataFrame(rows)




def wflow_event_replay_plan(config: dict, location_root: Path, event_id: str | None) -> pd.Series:
    """Return the reviewed HydroMT-Wflow event replay command for one catalog event."""
    build_plan = build_wflow_build_plan(config, {"location_root": location_root})
    command = build_plan.update_command.replace("<event_id>", str(event_id or "<event_id>"))
    resolved_command, runner_status, runner_issue = _describe_hydromt_command(command, location_root)
    event_dir = build_plan.events_root / str(event_id) if event_id else build_plan.events_root / "<event_id>"
    return pd.Series(
        {
            "event_id": event_id,
            "wflow_event_dir": str(event_dir),
            "wflow_discharge_forcing": str(event_dir / "sfincs_discharge.nc"),
            "hydromt_wflow_update_command": command,
            "resolved_hydromt_wflow_update_command": resolved_command,
            "hydromt_runner_status": runner_status,
            "hydromt_runner_issue": runner_issue,
        },
        name="wflow_event_replay_plan",
    )


def run_wflow_event_replay(
    config: dict,
    location_root: Path,
    event_id: str,
    *,
    execute: bool = False,
) -> pd.Series:
    """Run or dry-run the HydroMT-Wflow event replay command for one event."""
    plan = wflow_event_replay_plan(config, location_root, event_id)
    command = str(plan["hydromt_wflow_update_command"])
    resolved_command = str(plan["resolved_hydromt_wflow_update_command"])
    runner_status = str(plan["hydromt_runner_status"])
    runner_issue = str(plan["hydromt_runner_issue"])
    if execute:
        command_parts = _resolve_hydromt_command(command, location_root)
        try:
            subprocess.run(command_parts, cwd=Path(location_root), check=True, env=_hydromt_subprocess_env(location_root))
        except FileNotFoundError as exc:
            raise RuntimeError(_hydromt_missing_message(command, location_root)) from exc
        status = "completed"
    else:
        status = "dry_run"
    return pd.Series(
        {
            "event_id": event_id,
            "status": status,
            "command": command,
            "resolved_command": resolved_command,
            "hydromt_runner_status": runner_status,
            "hydromt_runner_issue": runner_issue,
            "wflow_event_dir": plan["wflow_event_dir"],
            "wflow_discharge_forcing": plan["wflow_discharge_forcing"],
        },
        name="wflow_event_replay",
    )


def prepare_wflow_subbasin_fabric(config: dict, location_root: Path, domain_plan) -> tuple:
    wflow = config["wflow"]
    data_sources = config["collection"]["national_hydrography"]
    inputs = exists_table(
        location_root,
        {
            "NHDPlus HR river geometry": data_sources["river_geometry"],
            "NHDPlus HR catchments": data_sources["catchments"],
        },
    )
    if domain_plan.status == "ready" and inputs["exists"].all():
        result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})
    else:
        subbasin_fabric_path = resolve_location_path(
            location_root,
            wflow["domain_set"].get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
        )
        result = {
            "subbasin_fabric": subbasin_fabric_path,
            "subbasin_geometry_files": tuple(sorted(subbasin_fabric_path.with_suffix("").glob("*.geojson"))),
            "diagnostics_csv": resolve_location_path(
                location_root,
                wflow["domain_set"].get(
                    "subbasin_fabric_diagnostics",
                    "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
                ),
            ),
            "submodel_count": 0,
            "catchment_count": 0,
            "statuses": ("missing_inputs_or_review_required",),
        }
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    summary = pd.Series(
        {
            "subbasin_fabric": str(result["subbasin_fabric"]),
            "subbasin_geometry_files": len(result.get("subbasin_geometry_files", ())),
            "diagnostics_csv": str(result["diagnostics_csv"]),
            "submodel_count": result["submodel_count"],
            "catchment_count": result["catchment_count"],
            "statuses": ", ".join(result["statuses"]),
            "coverage_status": result.get("coverage_status"),
            "coverage_catchment_count": result.get("coverage_catchment_count", 0),
            "evaluation_footprint_within_domain": result.get("evaluation_footprint_within_domain"),
            "evaluation_footprint_uncovered_km2": result.get("evaluation_footprint_uncovered_km2"),
            "power_extent_within_domain": result.get("power_extent_within_domain"),
            "power_extent_uncovered_km2": result.get("power_extent_uncovered_km2"),
            "replanned_status": domain_plan.status,
            "replanned_hydromt_region_kinds": ", ".join(
                sorted({submodel["region_kind"] for submodel in domain_plan.submodels})
            ),
        },
        name="nhdplus_subbasin_fabric_result",
    )
    return result, inputs, domain_plan, summary


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _describe_hydromt_command(command: str, location_root: Path) -> tuple[str, str, str]:
    try:
        command_parts = _resolve_hydromt_command(command, location_root)
    except RuntimeError as exc:
        return command, "missing", str(exc)
    runner_status = "configured"
    if command_parts:
        runner = Path(command_parts[0])
        if (runner.name.startswith("hydromt") or runner.name.startswith("python")) and ".venv" in runner.parts:
            runner_status = "project_venv"
        elif len(command_parts) >= 3 and command_parts[1:3] == ["-m", "hydromt.cli.main"]:
            runner_status = "active_python"
        elif command_parts[0] == "hydromt":
            runner_status = "path"
    return shlex.join(command_parts), runner_status, ""


def _resolve_hydromt_command(command: str, location_root: Path) -> list[str]:
    command_parts = shlex.split(command)
    if not command_parts:
        raise ValueError("Cannot run an empty HydroMT-Wflow command.")
    if command_parts[0] != "hydromt":
        return command_parts

    for candidate in _project_hydromt_candidates(location_root):
        if _valid_console_script(candidate):
            return [str(candidate), *command_parts[1:]]

    for candidate in _project_python_candidates(location_root):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return [str(candidate), "-m", "hydromt.cli.main", *command_parts[1:]]

    hydromt_on_path = shutil.which("hydromt")
    if hydromt_on_path:
        return [hydromt_on_path, *command_parts[1:]]

    if importlib.util.find_spec("hydromt.cli.main") is not None:
        return [sys.executable, "-m", "hydromt.cli.main", *command_parts[1:]]

    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python", "-m", "hydromt.cli.main", *command_parts[1:]]

    raise RuntimeError(_hydromt_missing_message(command, location_root))


def _valid_console_script(path: Path) -> bool:
    if not (path.exists() and os.access(path, os.X_OK)):
        return False
    try:
        first_line = path.open("rb").readline(512).decode("utf-8", errors="ignore").strip()
    except OSError:
        return False
    if not first_line.startswith("#!"):
        return True
    try:
        interpreter = shlex.split(first_line[2:].strip())[0]
    except (IndexError, ValueError):
        return False
    interpreter_path = Path(interpreter)
    if not interpreter_path.is_absolute():
        return True
    if not interpreter_path.exists():
        return False
    script_venv = _venv_root_for_script(path)
    interpreter_venv = _venv_root_for_script(interpreter_path)
    if script_venv is not None and interpreter_venv is not None:
        return script_venv == interpreter_venv
    return True


def _project_hydromt_candidates(location_root: Path) -> tuple[Path, ...]:
    script_name = "hydromt.exe" if os.name == "nt" else "hydromt"
    return tuple(path / script_name for path in _project_venv_bin_dirs(location_root))


def _project_python_candidates(location_root: Path) -> tuple[Path, ...]:
    script_name = "python.exe" if os.name == "nt" else "python"
    return tuple(path / script_name for path in _project_venv_bin_dirs(location_root))


def _project_venv_bin_dirs(location_root: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for start in (Path(location_root), Path.cwd()):
        resolved = start.resolve()
        roots.extend([resolved, *resolved.parents])

    unique_roots = []
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique_roots.append(root)

    script_dirs = []
    for root in unique_roots:
        if os.name == "nt":
            script_dir = root / ".venv" / "Scripts"
        else:
            script_dir = root / ".venv" / "bin"
        if script_dir.exists():
            script_dirs.append(script_dir)
    return tuple(script_dirs)


def _venv_root_for_script(path: Path) -> Path | None:
    parent = path.parent
    if parent.name not in {"bin", "Scripts"}:
        return None
    venv_root = parent.parent
    return venv_root if venv_root.name == ".venv" else None


def _hydromt_missing_message(command: str, location_root: Path) -> str:
    tried = [
        "PATH: hydromt",
        *[str(path) for path in _project_hydromt_candidates(location_root)],
        *[f"{path} -m hydromt.cli.main" for path in _project_python_candidates(location_root)],
        f"{sys.executable} -m hydromt.cli.main",
    ]
    return (
        "HydroMT CLI executable not found for Wflow event replay. "
        "The notebook generated a HydroMT command, but the active kernel cannot spawn it. "
        "Activate the project environment or install the HydroMT-Wflow CLI, then rerun the cell. "
        f"Generated command: {command}. Tried: {', '.join(tried)}"
    )


def _hydromt_subprocess_env(location_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    debug_value = env.get("DEBUG")
    if debug_value is not None and not str(debug_value).lstrip("-").isdigit():
        env["DEBUG"] = "0"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    venv_dirs = _project_venv_bin_dirs(location_root or Path.cwd())
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*(str(path) for path in venv_dirs), existing_path])
    return env


# Compact notebook-facing workflow verbs. These wrappers import lazily so the
# Wflow facade does not create circular imports with replay/handoff modules.
def collect_warmup(*args, **kwargs):
    from design_events.collect_sources.aorc_sst import collect_warmup

    return collect_warmup(*args, **kwargs)


def handoff_readiness(*args, **kwargs):
    from sfincs_runs.scenarios import handoff_readiness

    return handoff_readiness(*args, **kwargs)


def plan_example(*args, **kwargs):
    from sfincs_runs.scenarios import plan_example

    return plan_example(*args, **kwargs)


def validate_staticmaps(*args, **kwargs):
    from wflow_runs.build_plan import validate_staticmaps

    return validate_staticmaps(*args, **kwargs)


def plan_handoff(*args, **kwargs):
    from wflow_runs.dynamic_handoff import plan_handoff

    return plan_handoff(*args, **kwargs)


def plan_streamflow(*args, **kwargs):
    from wflow_runs.dynamic_handoff import plan_streamflow

    return plan_streamflow(*args, **kwargs)


def prepare_handoff(*args, **kwargs):
    from wflow_runs.dynamic_handoff import prepare_handoff

    return prepare_handoff(*args, **kwargs)


def require_handoff(*args, **kwargs):
    from wflow_runs.dynamic_handoff import require_handoff

    return require_handoff(*args, **kwargs)


def ensure_dynamic_handoff(*args, **kwargs):
    from wflow_runs.dynamic_handoff import ensure_dynamic_handoff

    return ensure_dynamic_handoff(*args, **kwargs)


def run_handoffs(*args, **kwargs):
    from wflow_runs.dynamic_handoff_batch import run_handoffs

    return run_handoffs(*args, **kwargs)


def build_meteo(*args, **kwargs):
    from wflow_runs.replay import build_meteo

    return build_meteo(*args, **kwargs)


def validate_geometry(*args, **kwargs):
    from wflow_runs.river_geometry import validate_geometry

    return validate_geometry(*args, **kwargs)


def plan_warmup(*args, **kwargs):
    from wflow_runs.states import plan_warmup

    return plan_warmup(*args, **kwargs)


def prepare_instates(*args, **kwargs):
    from wflow_runs.states import prepare_instates

    return prepare_instates(*args, **kwargs)


def validate_instates(*args, **kwargs):
    from wflow_runs.states import validate_instates

    return validate_instates(*args, **kwargs)
