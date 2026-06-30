from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pandas as pd

from sfincs_runs.config import load_runtime
from sfincs_runs.inland_coupled import (
    audit_inland_coupled_batch_readiness,
    stage_inland_coupled_scenario_forcing,
    stage_scenarios,
)
from wflow_runs.dynamic_handoff_batch import run_handoffs
from location_runtime import resolve_location_path


def run_inland_coupled_event(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    sfincs_bin: str,
    storage_dir=None,
    run_root=None,
    force_rerun: bool = False,
    keep_stage: bool = False,
    dry_run: bool = False,
    force_wflow: bool = False,
    overwrite_meteo: bool = False,
) -> dict:
    """Run one event as an atomic Wflow -> SFINCS coupled pipeline."""
    location_root = Path(location_root)
    scenarios_root = _location_path(location_root, config.get("paths", {}).get("scenarios_root", "data/sfincs/scenarios"))
    storage_dir = Path(storage_dir) if storage_dir is not None else _location_path(
        location_root, config.get("paths", {}).get("storage_root", "data/sfincs/run_outputs")
    )
    run_root = Path(run_root) if run_root is not None else _location_path(
        location_root, config.get("paths", {}).get("run_root", "data/sfincs/run_stage")
    )
    catalog_path = _location_path(location_root, catalog_path or "data/event_catalog/catalog/scenario_catalog.csv")
    event_id = str(event_id)
    t0 = time.time()

    wflow_report = run_handoffs(
        config,
        location_root,
        catalog_path=catalog_path,
        event_ids=[event_id],
        status="all",
        execute=not dry_run,
        force=force_wflow,
        overwrite_meteo=overwrite_meteo,
    )
    if wflow_report["status"].eq("failed").any():
        raise RuntimeError(f"Wflow dynamic handoff failed for {event_id}: {wflow_report.to_dict('records')}")

    if dry_run:
        return {
            "event_id": event_id,
            "status": "dry-run",
            "wflow_status": ",".join(wflow_report["status"].astype(str)),
            "duration_sec": time.time() - t0,
        }

    scenario_report = stage_scenarios(
        config,
        {"location_root": location_root},
        catalog_path=catalog_path,
        event_ids=[event_id],
        force=True,
        write_reports=False,
    )
    forcing_report = stage_inland_coupled_scenario_forcing(
        config,
        {"location_root": location_root},
        scenario_report=scenario_report,
        write_reports=False,
    )
    audit = audit_inland_coupled_batch_readiness(
        config,
        {"location_root": location_root},
        catalog_path=catalog_path,
        event_ids=[event_id],
        staged_catalog=scenario_report[["event_id", "run_root"]],
    )
    failed = audit[audit["status"].ne("passed")]
    if not failed.empty:
        raise RuntimeError(f"Staged SFINCS inputs are not ready for {event_id}: {failed.to_dict('records')}")

    event_catalog = _write_event_scenario_catalog(scenarios_root, scenario_report, event_id)
    command = [
        sys.executable,
        "-m",
        "sfincs_runs.run_events",
        "--config",
        str(location_root / "config.yaml"),
        "--scenarios-dir",
        str(scenarios_root),
        "--scenario-catalog",
        str(event_catalog),
        "--storage-dir",
        str(storage_dir),
        "--run-root",
        str(run_root),
        "--sfincs-bin",
        sfincs_bin,
        "--event-id",
        event_id,
    ]
    if force_rerun:
        command.append("--force-rerun")
    if keep_stage:
        command.append("--keep-stage")
    process = subprocess.run(command, cwd=location_root.parents[1], check=False)
    if process.returncode:
        raise RuntimeError(f"SFINCS run failed for {event_id}; command returned {process.returncode}: {shlex.join(command)}")
    return {
        "event_id": event_id,
        "status": "completed",
        "wflow_status": ",".join(wflow_report["status"].astype(str)),
        "sfincs_domains": int(len(forcing_report)),
        "scenario_catalog": str(event_catalog),
        "duration_sec": time.time() - t0,
    }


def _write_event_scenario_catalog(scenarios_root: Path, scenario_report: pd.DataFrame, event_id: str) -> Path:
    event_dir = scenarios_root / str(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)
    catalog = scenario_report[["event_id", "run_root"]].copy()
    catalog["run_root"] = [Path(value).relative_to(scenarios_root).as_posix() for value in catalog["run_root"]]
    path = event_dir / "_scenario_catalog.csv"
    catalog.to_csv(path, index=False)
    return path


def _events_from_worklist(path: Path, *, limit=None) -> list[str]:
    worklist = pd.read_csv(path)
    if "event_id" not in worklist:
        raise ValueError(f"Worklist is missing event_id: {path}")
    events = worklist["event_id"].astype(str).drop_duplicates().tolist()
    return events[: int(limit)] if limit is not None else events


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run coupled Wflow -> SFINCS inland events.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog-path", default="data/event_catalog/catalog/scenario_catalog.csv")
    parser.add_argument("--worklist", type=Path, default=None)
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sfincs-bin", required=True)
    parser.add_argument("--storage-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-wflow", action="store_true")
    parser.add_argument("--overwrite-meteo", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ.pop("DEBUG", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/flood-rm-matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    args = _parse_args(argv)
    config_path = Path(args.config).resolve()
    location_root = config_path.parent
    config, _paths = load_runtime(config_path)
    config["wflow"]["domain_set"]["review_required"] = False
    if args.event_ids:
        event_ids = [str(e) for e in args.event_ids]
    elif args.worklist is not None:
        event_ids = _events_from_worklist(Path(args.worklist), limit=args.limit)
    else:
        catalog_path = _location_path(location_root, args.catalog_path)
        event_ids = _events_from_worklist(catalog_path, limit=args.limit)

    rows = []
    for event_id in event_ids:
        try:
            rows.append(
                run_inland_coupled_event(
                    config,
                    location_root,
                    event_id,
                    catalog_path=args.catalog_path,
                    sfincs_bin=args.sfincs_bin,
                    storage_dir=args.storage_dir,
                    run_root=args.run_root,
                    force_rerun=args.force_rerun,
                    keep_stage=args.keep_stage,
                    dry_run=args.dry_run,
                    force_wflow=args.force_wflow,
                    overwrite_meteo=args.overwrite_meteo,
                )
            )
        except Exception as exc:
            rows.append({"event_id": event_id, "status": "failed", "message": str(exc)})
            break
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    out = args.out
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
        print(json.dumps({"report": str(out)}, indent=2))
    return 1 if report["status"].eq("failed").any() else 0


def _location_path(location_root: Path, value) -> Path:
    return resolve_location_path(location_root, value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
