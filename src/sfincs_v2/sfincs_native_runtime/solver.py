from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .io import copy_base_model, copy_retained_outputs, remove_solver_outputs
from .schema import SfincsRunResult


def build_sfincs_command(
    *,
    sfincs_bin: str | None = None,
    sfincs_image: str = "deltares/sfincs-cpu:latest",
    allow_native: bool = True,
    model_root: Path | None = None,
) -> list[str]:
    """Resolve a SFINCS solver command.

    Preference: explicit binary/command, native ``sfincs`` on PATH, then Docker.
    """
    if sfincs_bin:
        path = Path(sfincs_bin)
        if path.exists():
            return [str(path)]
        return shlex.split(sfincs_bin)
    if allow_native:
        native = shutil.which("sfincs")
        if native:
            return [native]
    docker = shutil.which("docker")
    if docker:
        root = Path(model_root or ".").resolve()
        return [docker, "run", "--rm", "-v", f"{root}:/data", "-w", "/data", sfincs_image, "sfincs"]
    raise RuntimeError("No SFINCS runner found. Pass sfincs_bin, install sfincs, or install Docker.")


def sfincs_subprocess_env(*, threads: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("DEBUG") is not None and not str(env["DEBUG"]).lstrip("-").isdigit():
        env["DEBUG"] = "0"
    if threads is not None:
        if int(threads) < 1:
            raise ValueError("threads must be >= 1")
        env["OMP_NUM_THREADS"] = str(int(threads))
    return env


def run_sfincs(
    run_root: str | Path,
    *,
    storage_dir: str | Path | None = None,
    command: list[str] | str | None = None,
    sfincs_bin: str | None = None,
    threads: int | None = None,
    require_map: bool = True,
    keep_stage: bool = True,
) -> SfincsRunResult:
    """Run SFINCS in ``run_root`` and copy retained outputs to ``storage_dir``."""
    run_root = Path(run_root)
    if command is None:
        command_list = build_sfincs_command(sfincs_bin=sfincs_bin, model_root=run_root)
    elif isinstance(command, str):
        command_list = shlex.split(command)
    else:
        command_list = list(command)
    log_path = run_root / "sfincs_log.txt"
    event_id = run_root.name
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command_list,
            cwd=run_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=sfincs_subprocess_env(threads=threads),
        )
    duration = time.time() - t0
    map_path = run_root / "sfincs_map.nc"
    storage = Path(storage_dir) if storage_dir is not None else run_root
    metadata = {
        "event_id": event_id,
        "stage_dir": str(run_root),
        "storage_dir": str(storage),
        "runner_command": command_list,
        "returncode": int(process.returncode),
        "duration_sec": duration,
    }
    if storage_dir is not None:
        storage.mkdir(parents=True, exist_ok=True)
        copy_retained_outputs(run_root, storage)
        (storage / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path = storage / "sfincs_log.txt"
        map_path = storage / "sfincs_map.nc"
    else:
        (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"SFINCS failed for {event_id} with exit code {process.returncode}; see {log_path}")
    if require_map and not map_path.exists():
        raise RuntimeError(f"SFINCS completed for {event_id} but produced no sfincs_map.nc")
    return SfincsRunResult(event_id, run_root, run_root, storage, log_path, map_path, int(process.returncode), duration, "completed")


def stage_prepared_event(src_dir: Path, stage_dir: Path, *, force: bool = True) -> Path:
    """Copy/hardlink one prepared scenario to a mutable stage directory."""
    src_dir = Path(src_dir)
    stage_dir = Path(stage_dir)
    return copy_base_model(src_dir, stage_dir, force=force)


def scenario_units(scenarios_root: str | Path, *, scenario_catalog: str | Path | None = None, event_ids=None, limit=None) -> list[tuple[str, Path]]:
    """Resolve leaf scenario folders, including nested inland domain folders."""
    root = Path(scenarios_root)
    if scenario_catalog is not None:
        table = pd.read_csv(scenario_catalog)
        if "run_root" not in table:
            raise ValueError(f"scenario catalog missing run_root: {scenario_catalog}")
        if event_ids is not None:
            wanted = {str(e) for e in event_ids}
            table = table[table["event_id"].astype(str).isin(wanted)]
        if limit is not None:
            table = table.head(int(limit))
        units = []
        for _, row in table.iterrows():
            src = Path(str(row["run_root"]))
            if not src.is_absolute():
                src = root / src
            key = src.relative_to(root).as_posix() if src.is_relative_to(root) else os.path.relpath(src, root)
            units.append((key, src))
        return units

    selected = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if event_ids is not None:
        wanted = {str(e) for e in event_ids}
        selected = [path for path in selected if path.name in wanted]
    if limit is not None:
        selected = selected[: int(limit)]
    units: list[tuple[str, Path]] = []
    for path in selected:
        # If nested domains exist, use leaf folders with sfincs.inp.
        leaves = sorted(child for child in path.glob("*/sfincs.inp") if child.is_file())
        if leaves:
            units.extend((leaf.parent.relative_to(root).as_posix(), leaf.parent) for leaf in leaves)
        else:
            units.append((path.name, path))
    return units


def _run_one_unit(key: str, src_dir: Path, *, run_root: Path, storage_root: Path, command: list[str], force: bool, dry_run: bool, keep_stage: bool, threads: int | None) -> dict:
    stage_dir = run_root / key
    storage_dir = storage_root / key
    if (storage_dir / "sfincs_map.nc").exists() and not force:
        copied = copy_retained_outputs(src_dir, storage_dir, overwrite=False)
        return {"event_id": key, "status": "skipped", "retained_files_synced": copied}
    stage_prepared_event(src_dir, stage_dir, force=True)
    remove_solver_outputs(stage_dir)
    if dry_run:
        if not keep_stage:
            shutil.rmtree(stage_dir, ignore_errors=True)
        return {"event_id": key, "status": "dry-run"}
    result = run_sfincs(stage_dir, storage_dir=storage_dir, command=command, threads=threads)
    if not keep_stage:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return result.to_dict()


def run_prepared_events(
    scenarios_root: str | Path,
    *,
    storage_root: str | Path,
    run_root: str | Path,
    scenario_catalog: str | Path | None = None,
    event_ids=None,
    limit: int | None = None,
    sfincs_bin: str | None = None,
    command: list[str] | str | None = None,
    workers: int = 1,
    force_rerun: bool = False,
    dry_run: bool = False,
    keep_stage: bool = False,
    threads: int | None = None,
) -> pd.DataFrame:
    """Run prepared SFINCS event folders, preserving nested domain layout."""
    units = scenario_units(scenarios_root, scenario_catalog=scenario_catalog, event_ids=event_ids, limit=limit)
    if not units:
        raise RuntimeError("No selected SFINCS event folders")
    run_root = Path(run_root)
    storage_root = Path(storage_root)
    run_root.mkdir(parents=True, exist_ok=True)
    storage_root.mkdir(parents=True, exist_ok=True)
    if command is None:
        command_list = [] if dry_run else build_sfincs_command(sfincs_bin=sfincs_bin, model_root=run_root)
    elif isinstance(command, str):
        command_list = shlex.split(command)
    else:
        command_list = list(command)

    rows: list[dict] = []
    if workers <= 1 or len(units) == 1:
        for key, src in units:
            rows.append(
                _run_one_unit(
                    key,
                    src,
                    run_root=run_root,
                    storage_root=storage_root,
                    command=command_list,
                    force=force_rerun,
                    dry_run=dry_run,
                    keep_stage=keep_stage,
                    threads=threads,
                )
            )
        return pd.DataFrame(rows)

    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(
                _run_one_unit,
                key,
                src,
                run_root=run_root,
                storage_root=storage_root,
                command=command_list,
                force=force_rerun,
                dry_run=dry_run,
                keep_stage=keep_stage,
                threads=threads,
            ): key
            for key, src in units
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"event_id": key, "status": "failed", "message": str(exc)})
    return pd.DataFrame([*rows, *failures])
