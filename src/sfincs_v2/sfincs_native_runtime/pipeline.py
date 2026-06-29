from __future__ import annotations

from pathlib import Path
import subprocess

from .audit import audit_run_folder
from .forcing import stage_inland_event_forcing
from .io import copy_base_model
from .solver import run_sfincs


def run_inland_event_pipeline(
    *,
    event_id: str,
    base_root: str | Path,
    scenario_root: str | Path,
    storage_root: str | Path,
    wflow_discharge_nc: str | Path,
    sfincs_bin: str | None = None,
    wflow_command: list[str] | None = None,
    precip_nc: str | Path | None = None,
    direct_rainfall: bool = False,
    force: bool = False,
    keep_stage: bool = True,
) -> dict:
    """Atomic Wflow -> HydroMT-SFINCS -> SFINCS event run.

    ``wflow_command`` is intentionally outside this package's implementation. It
    can be a HydroMT-Wflow dynamic-replay command that produces
    ``wflow_discharge_nc``.  The SFINCS runtime only consumes the declared NetCDF
    contract.
    """
    if wflow_command is not None:
        proc = subprocess.run(wflow_command, check=False)
        if proc.returncode:
            raise RuntimeError(f"Wflow command failed with exit code {proc.returncode}: {wflow_command}")

    run_root = copy_base_model(Path(base_root), Path(scenario_root), force=force)
    manifest = stage_inland_event_forcing(
        run_root,
        event_id=event_id,
        wflow_discharge_nc=wflow_discharge_nc,
        precip_nc=precip_nc,
        direct_rainfall=direct_rainfall,
    )
    audit = audit_run_folder(run_root)
    if not audit.passed:
        raise RuntimeError(f"SFINCS staging audit failed for {event_id}: {audit.to_dict()}")
    result = run_sfincs(run_root, storage_dir=Path(storage_root) / str(event_id), sfincs_bin=sfincs_bin, keep_stage=keep_stage)
    return {**manifest.to_dict(), **result.to_dict(), "audit": audit.to_dict()}
