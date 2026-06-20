import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sfincs_runs.config import build_paths, load_runtime

paths = build_paths()
default_scenarios_root = paths["scenarios_root"]
default_storage_root = paths["storage_root"]
default_run_root = paths["run_root"]
stale_run_files = {"sfincs_map.nc", "sfincs_his.nc", "sfincs_rst.nc", "sfincs.log", "sfincs_log.txt"}
retained_output_files = {
    "sfincs_map.nc",
    "sfincs_his.nc",
    "sfincs.log",
    "sfincs_log.txt",
    "sfincs.inp",
    "forcing_manifest.json",
    "sfincs.bnd",
    "sfincs.bzs",
    "sfincs.obs",
    "sfincs.weir",
    "sfincs.thd",
    "sfincs.rug",
    "sfincs.rug.obs",
    "snapwave.bnd",
    "snapwave.bhs",
    "snapwave.btp",
    "snapwave.bwd",
    "snapwave.bds",
}


def event_id(x):
    text = str(x).strip()
    if text.startswith("evt_") and text[4:].isdigit():
        return f"evt_{int(text[4:]):04d}"
    if text.isdigit() and int(text) > 0:
        return f"evt_{int(text):04d}"
    if text and "/" not in text and "\\" not in text and text not in {".", ".."}:
        return text
    raise ValueError(f"Bad event id: {x!r}")


def event_dirs(root, ids=None, limit=None):
    selected = sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("."))
    if ids:
        wanted = {event_id(x) for x in ids}
        selected = [p for p in selected if p.name in wanted]
        missing = wanted - {p.name for p in selected}
        if missing:
            raise FileNotFoundError(f"Missing events: {', '.join(sorted(missing))}")
    return selected[:limit] if limit is not None else selected


def units_from_catalog(catalog_path, scenarios_dir, ids=None, limit=None):
    """Resolve (key, src_dir) run units from a scenario_catalog.csv.

    Inland Wflow->SFINCS scenarios are nested per SFINCS domain
    (``scenarios/<event_id>/<domain_id>``), so the flat ``event_dirs`` walk does not
    reach the leaf run folders. The catalog's ``run_root`` column already points at each
    leaf; ``key`` is that run folder's path relative to ``scenarios_dir`` so storage and
    stage trees preserve the ``<event_id>/<domain_id>`` nesting. Coastal (flat) catalogs
    resolve to ``key == <event_id>``, matching the ``event_dirs`` behaviour.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    scenarios_dir = Path(scenarios_dir)
    rows = []
    with catalog_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_root = (row.get("run_root") or "").strip()
            if not run_root:
                continue
            src = Path(run_root)
            if not src.is_absolute():
                src = scenarios_dir / src
            rows.append((str(row.get("event_id", "")).strip(), src))
    if not rows:
        raise RuntimeError(f"No run_root entries in {catalog_path}")
    if ids:
        wanted = {event_id(x) for x in ids}
        rows = [(rid, src) for rid, src in rows if event_id(rid) in wanted]
        missing = wanted - {event_id(rid) for rid, _ in rows}
        if missing:
            raise FileNotFoundError(f"Missing events: {', '.join(sorted(missing))}")
    if limit is not None:
        rows = rows[: int(limit)]
    units = []
    for _rid, src in rows:
        try:
            key = src.relative_to(scenarios_dir).as_posix()
        except ValueError:
            key = Path(os.path.relpath(src, scenarios_dir)).as_posix()
        units.append((key, src))
    return units


def link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_event(src_dir, stage_dir):
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if src.is_file() and src.name not in stale_run_files:
            link_or_copy(src, stage_dir / src.name)
        elif src.is_dir():
            shutil.copytree(src, stage_dir / src.name, copy_function=link_or_copy)


def save_outputs(stage_dir, storage_dir, metadata):
    storage_dir.mkdir(parents=True, exist_ok=True)
    for name in retained_output_files:
        if (stage_dir / name).exists():
            shutil.copy2(stage_dir / name, storage_dir / name)
    (storage_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run prepared SFINCS event folders.")
    p.add_argument("--config", default=None, help="optional config overlay yaml")
    p.add_argument("--scenarios-dir", type=Path, default=None)
    p.add_argument(
        "--scenario-catalog",
        type=Path,
        default=None,
        help="run the run_root folders listed in this scenario_catalog.csv instead of walking "
        "flat top-level event folders (required for nested inland <event_id>/<domain_id> scenarios)",
    )
    p.add_argument("--storage-dir", type=Path, default=None)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--sfincs-bin", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--event-id", action="append", dest="event_ids")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--keep-stage", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    config, runtime_paths = load_runtime(args.config)
    run_cfg = config.get("scenario_run", {})
    sfincs_bin_env = str(run_cfg.get("sfincs_bin_env", "SFINCS_BIN"))
    args.scenarios_dir = args.scenarios_dir or runtime_paths["scenarios_root"]
    args.storage_dir = args.storage_dir or runtime_paths["storage_root"]
    args.run_root = args.run_root or runtime_paths["run_root"]
    args.sfincs_bin = (
        args.sfincs_bin
        or run_cfg.get("sfincs_bin")
        or os.environ.get(sfincs_bin_env, "")
    )
    return args


def run_one(key, src_dir, args, command_template):
    storage_dir = args.storage_dir / key
    stage_dir = args.run_root / key
    if (storage_dir / "sfincs_map.nc").exists() and not args.force_rerun:
        return {"event_id": key, "status": "skipped", "reason": "sfincs_map.nc exists"}

    stage_event(src_dir, stage_dir)
    if args.dry_run:
        if not args.keep_stage:
            shutil.rmtree(stage_dir, ignore_errors=True)
        return {"event_id": key, "status": "dry-run"}

    command = [part.format(stage_dir=str(stage_dir)) for part in command_template]
    t0 = time.time()
    with (stage_dir / "sfincs_log.txt").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=stage_dir, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    metadata = {
        "event_id": key,
        "source_scenario_dir": str(src_dir),
        "stage_dir": str(stage_dir),
        "storage_dir": str(storage_dir),
        "runner_command": command,
        "returncode": int(process.returncode),
        "duration_sec": time.time() - t0,
    }
    save_outputs(stage_dir, storage_dir, metadata)
    if not args.keep_stage:
        shutil.rmtree(stage_dir, ignore_errors=True)

    if process.returncode:
        raise RuntimeError(f"{key} failed. See {storage_dir / 'sfincs_log.txt'}.")
    if not (storage_dir / "sfincs_map.nc").exists():
        raise RuntimeError(f"{key} produced no sfincs_map.nc.")
    return {"event_id": key, "status": "completed", "duration_sec": metadata["duration_sec"]}


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not args.scenarios_dir.exists():
        raise FileNotFoundError(args.scenarios_dir)

    if args.scenario_catalog is not None:
        units = units_from_catalog(args.scenario_catalog, args.scenarios_dir, args.event_ids, args.limit)
    else:
        units = [(event_dir.name, event_dir) for event_dir in event_dirs(args.scenarios_dir, args.event_ids, args.limit)]
    if not units:
        raise RuntimeError("No selected event folders.")
    command_template = [] if args.dry_run else shlex.split(args.sfincs_bin)
    if not args.dry_run and not command_template:
        raise ValueError("Provide --sfincs-bin or set SFINCS_BIN.")

    args.storage_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    print(f"Scenarios: {args.scenarios_dir}")
    print(f"Storage: {args.storage_dir}")
    print(f"Run root: {args.run_root}")
    print(f"Events: {len(units)}")
    print("Mode: dry-run" if args.dry_run else f"Runner: {' '.join(command_template)}")

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, key, src_dir, args, command_template): key for key, src_dir in units}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{result['status']}] {name}")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                print(f"[failed] {name}: {exc}", file=sys.stderr)

    print(json.dumps({
        "event_count": len(units),
        "completed": sum(r["status"] == "completed" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "dry_run": sum(r["status"] == "dry-run" for r in results),
        "failed": len(failures),
    }, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
