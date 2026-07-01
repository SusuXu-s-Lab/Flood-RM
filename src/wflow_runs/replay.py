"""Domain-set Wflow event replay → merged SFINCS discharge forcing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shlex
import shutil
import subprocess

import pandas as pd

from coupling import amplification as coupling_amplification
from coupling import discharge as coupling_discharge
from paths import resolve_location_path
from wflow_runs.repairs import (
    normalize_wflow_staticmaps_nodata,
    repair_wflow_canopy_parameters,
    repair_wflow_gauge_map,
    repair_wflow_river_width,
)
from wflow_runs.staticmaps_qa import validate_staticmaps
from wflow_runs.notebook import (
    _describe_hydromt_command,
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
)
from wflow_runs.states import prepare_wflow_event_instate
from wflow_runs.event import (
    clean_legacy_replay_submodel_output_dir,
    event_reference_time,
    event_window,
    write_legacy_replay_data_catalog,
    write_legacy_replay_update_config,
)
from wflow_runs.domain import configured_or_manifest_submodels as _v2_configured_or_manifest_submodels
from wflow_runs.runner import clean_output_dir as _v2_clean_output_dir
from wflow_runs.runner import wflow_run_command as _v2_wflow_run_command
from wflow_runs.runner import zero_event_forcing as _zero_event_forcing


def _discharge_handoff_source(config: dict) -> str:
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    return str(forcing.get("source", "wflow_replay")).strip().lower()


# ─── orchestration (subprocess steps gated behind execute=True) ───────────────


@dataclass(frozen=True)
class ReplayStep:
    submodel_id: str
    update_command: str
    run_command: str
    run_output_dir: str
    gauges_geojson: str


def replay_inland_domain_set(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = False,
    pre_event_hours: float = 48.0,
    post_event_hours: float = 72.0,
) -> pd.DataFrame:
    """Replay every Wflow submodel for one event and write the merged SFINCS discharge.

    With ``execute=False`` (default) this plans the work — resolves the window, checks
    prerequisites, and returns the per-submodel HydroMT-update + Wflow-run commands —
    without spawning anything. With ``execute=True`` it runs the updates + Wflow engine
    and writes ``events/<event>/sfincs_discharge.nc``.
    """
    location_root = Path(location_root).resolve()
    wflow = config.get("wflow", {})
    base_root = resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base"))
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    update_cfg = resolve_location_path(location_root, wflow.get("update_forcing_config", "wflow_update_forcing.yml"))
    data_catalog = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    model_crs = wflow.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))

    reference_time = event_reference_time(location_root, event_id, catalog_path)
    start, end = event_window(reference_time, pre_event_hours=pre_event_hours, post_event_hours=post_event_hours)

    event_dir = events_root / event_id
    if execute:
        # Per-event meteo forcing is only consumed by the actual HydroMT update; a dry run
        # still plans (and writes the runnable -i/-d configs) without it.
        _require_event_forcing(event_dir)
    submodels = _domain_set_submodels(config, location_root)
    if not submodels:
        raise ValueError("wflow.domain_set has no submodels to replay")

    event_dir.mkdir(parents=True, exist_ok=True)
    per_event_catalog = _write_per_event_data_catalog(data_catalog, event_dir, event_id)
    per_event_update = _write_per_event_update_config(update_cfg, event_dir, start, end)
    run_command_template = _wflow_run_command(config)
    discharge_source = _discharge_handoff_source(config)

    steps: list[ReplayStep] = []
    apply_repairs = _wflow_replay_repairs_enabled(config)
    for submodel in submodels:
        submodel_id = str(submodel["wflow_submodel_id"])
        submodel_base = base_root / submodel_id
        if not (submodel_base / "wflow_sbm.toml").exists():
            raise FileNotFoundError(f"Wflow submodel base not built: {submodel_base}")
        if execute:
            normalize_wflow_staticmaps_nodata(submodel_base)
            repair_wflow_canopy_parameters(submodel_base)
        if execute and apply_repairs:
            repair_wflow_river_width(submodel_base)
            repair_wflow_gauge_map(submodel_base)
        out_dir = event_dir / submodel_id
        if execute:
            _prepare_replay_submodel_output_dir(event_dir, out_dir)
        update_command = (
            f"hydromt update wflow_sbm {submodel_base} "
            f"-i {per_event_update} -d {per_event_catalog} -o {out_dir} -vvv"
        )
        run_config = out_dir / "wflow_sbm.toml"
        run_command = run_command_template.format(run_config=run_config)
        steps.append(
            ReplayStep(
                submodel_id=submodel_id,
                update_command=update_command,
                run_command=run_command,
                run_output_dir=str(out_dir / "run_event"),
                gauges_geojson=str(submodel_base / "staticgeoms" / "gauges_sfincs.geojson"),
            )
        )

    rows = []
    for step in steps:
        status = "planned"
        resolved_update_command, hydromt_runner_status, hydromt_runner_issue = _describe_hydromt_command(
            step.update_command,
            location_root,
        )
        if execute:
            _run(_resolve_hydromt_command(step.update_command, location_root), cwd=location_root)
            prepare_wflow_event_instate(event_dir / step.submodel_id, base_root / step.submodel_id)
            # Event models run with rainfall + antecedent moisture only; frequency provenance
            # is applied as a single-K Same-Frequency Amplification on the merged output below.
            normalize_wflow_staticmaps_nodata(event_dir / step.submodel_id)
            repair_wflow_canopy_parameters(event_dir / step.submodel_id)
            if apply_repairs:
                repair_wflow_river_width(event_dir / step.submodel_id)
                repair_wflow_gauge_map(event_dir / step.submodel_id)
            validate_staticmaps(event_dir / step.submodel_id)
            _prepare_wflow_run_output_dir(event_dir / step.submodel_id / "wflow_sbm.toml")
            _run(shlex.split(step.run_command), cwd=location_root)
            status = "completed"
        rows.append(
            {
                "event_id": event_id,
                "submodel_id": step.submodel_id,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "update_command": step.update_command,
                "resolved_update_command": resolved_update_command,
                "hydromt_runner_status": hydromt_runner_status,
                "hydromt_runner_issue": hydromt_runner_issue,
                "run_command": step.run_command,
                "run_output_dir": step.run_output_dir,
                "status": status,
            }
        )

    discharge_path = event_dir / "sfincs_discharge.nc"
    if execute:
        if discharge_source in {"event_streamflow", "event_streamflow_timeseries", "catalog_streamflow"}:
            coupling_discharge.write_event_streamflow_handoff_discharge(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                model_crs=model_crs,
                out_path=discharge_path,
                start=start,
                end=end,
            )
        else:
            coupling_discharge.merge_submodel_discharge(
                [{"run_output_dir": s.run_output_dir, "gauges_geojson": s.gauges_geojson} for s in steps],
                model_crs=model_crs,
                out_path=discharge_path,
                handoff_points=coupling_discharge.sfincs_handoff_points(config, location_root, model_crs),
            )
            # One Same-Frequency Amplification K applied uniformly to the handoff hydrographs.
            # No-op (K=1) until the catalog provides a per-event target and a
            # primary_reference_gage is configured.
            coupling_amplification.apply_same_frequency_amplification(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                discharge_nc=discharge_path,
                submodel_runs=[{"run_output_dir": s.run_output_dir, "gauges_geojson": s.gauges_geojson} for s in steps],
            )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    report["sfincs_discharge_source"] = discharge_source
    return report


def run_zero_rain_control(
    config: dict,
    location_root,
    event_id: str,
    *,
    execute: bool = False,
) -> pd.DataFrame:
    """Run a Wflow startup/baseflow control with event rainfall and inflow set to zero.

    The control reuses the already materialised event Wflow model folders. It copies
    them below ``events/<event>/_zero_rain/<submodel>``, zeros the dynamic forcing
    variables in ``inmaps-event.nc``, runs Wflow, and writes
    ``events/<event>/_zero_rain/sfincs_discharge.nc`` for dynamic-handoff QA.
    """
    location_root = Path(location_root).resolve()
    wflow = config.get("wflow", {}) or {}
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    model_crs = wflow.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    event_dir = events_root / str(event_id)
    zero_root = event_dir / "_zero_rain"
    run_command_template = _wflow_run_command(config)
    rows = []
    outputs = []
    for submodel in _domain_set_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        source_model = event_dir / submodel_id
        control_model = zero_root / submodel_id
        if not (source_model / "wflow_sbm.toml").exists():
            raise FileNotFoundError(
                f"Zero-rain control requires the event Wflow model first: {source_model}"
            )
        status = "planned"
        if execute:
            if control_model.exists():
                shutil.rmtree(control_model)
            control_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_model, control_model)
            _zero_event_forcing(control_model / "inmaps-event.nc")
            _prepare_wflow_run_output_dir(control_model / "wflow_sbm.toml")
            _run(shlex.split(run_command_template.format(run_config=control_model / "wflow_sbm.toml")), cwd=location_root)
            status = "completed"
        run_output_dir = control_model / "run_event"
        gauges_geojson = control_model / "staticgeoms" / "gauges_sfincs.geojson"
        outputs.append({"run_output_dir": run_output_dir, "gauges_geojson": gauges_geojson})
        rows.append(
            {
                "event_id": str(event_id),
                "submodel_id": submodel_id,
                "control_model_root": str(control_model),
                "run_output_dir": str(run_output_dir),
                "status": status,
            }
        )
    discharge_path = zero_root / "sfincs_discharge.nc"
    if execute:
        coupling_discharge.merge_submodel_discharge(
            outputs,
            model_crs=model_crs,
            out_path=discharge_path,
            handoff_points=coupling_discharge.sfincs_handoff_points(config, location_root, model_crs),
        )
        (zero_root / "zero_rain_control.provenance.json").write_text(
            json.dumps(
                {
                    "event_id": str(event_id),
                    "control": "zero_event_forcing",
                    "zeroed_variables": ["precip"],
                    "purpose": "dynamic_handoff_startup_baseflow_qa",
                    "sfincs_discharge": str(discharge_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    report = pd.DataFrame(rows)
    report["sfincs_discharge_forcing"] = str(discharge_path)
    report["sfincs_discharge_written"] = bool(execute and discharge_path.exists())
    return report


def _require_event_forcing(event_dir: Path) -> None:
    missing = [name for name in ("precip.nc", "temp_pet.nc") if not (event_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "per-event Wflow forcing is not built (the event_precip/event_temp_pet contract): "
            + ", ".join(str(event_dir / name) for name in missing)
            + ". Run build_meteo before replaying."
        )


def _domain_set_submodels(config: dict, location_root: Path) -> list[dict]:
    return _v2_configured_or_manifest_submodels(config, location_root)


def _write_per_event_data_catalog(data_catalog: Path, event_dir: Path, event_id: str) -> Path:
    return write_legacy_replay_data_catalog(data_catalog, event_dir, event_id)


def _write_per_event_update_config(update_cfg: Path, event_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return write_legacy_replay_update_config(update_cfg, event_dir, start, end)


def _prepare_replay_submodel_output_dir(event_dir: Path, out_dir: Path) -> None:
    clean_legacy_replay_submodel_output_dir(event_dir, out_dir)


def _prepare_wflow_run_output_dir(run_config: Path) -> None:
    """Remove the generated Wflow run-output directory before solver execution."""
    _v2_clean_output_dir(run_config)


def _wflow_run_command(config: dict) -> str:
    """Resolve the Wflow engine command template (``{run_config}`` placeholder)."""
    return _v2_wflow_run_command(config)


def _wflow_replay_repairs_enabled(config: dict) -> bool:
    replay_cfg = (config.get("wflow", {}) or {}).get("replay", {}) or {}
    return bool(replay_cfg.get("apply_legacy_repairs", False))


def _run(command_parts, *, cwd) -> None:
    if not command_parts:
        raise ValueError("empty command")
    try:
        subprocess.run(command_parts, cwd=Path(cwd), check=True, env=_hydromt_subprocess_env(Path(cwd)))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"executable not found for replay step: {shlex.join([str(p) for p in command_parts])}"
        ) from exc
