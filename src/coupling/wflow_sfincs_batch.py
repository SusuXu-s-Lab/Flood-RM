from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

from sfincs_runs.config import load_runtime
from coupling.dynamic_handoff import (
    dynamic_handoff_paths,
    prepare_handoff,
    require_handoff,
)
from wflow_runs.event import configured_event_window_hours
from wflow_runs.meteo import build_meteo


def dynamic_handoff_batch_worklist(
    config: dict,
    location_root,
    *,
    catalog_path=None,
    event_ids=None,
    status: str = "blocked",
    limit=None,
) -> pd.DataFrame:
    """Build a Wflow dynamic handoff batch worklist."""
    from sfincs_runs.scenarios import handoff_readiness

    location_root = Path(location_root)
    readiness = handoff_readiness(
        config,
        location_root,
        catalog_path=catalog_path,
        event_ids=event_ids,
        limit=limit,
    )
    status = str(status).lower()
    if status in {"blocked", "accepted"}:
        out = readiness[readiness["status"].eq(status)].copy()
    elif status == "all":
        out = readiness.copy()
    else:
        raise ValueError("status must be one of: blocked, accepted, all")
    return out.reset_index(drop=True)


def run_handoffs(
    config: dict,
    location_root,
    *,
    catalog_path=None,
    event_ids=None,
    status: str = "blocked",
    limit=None,
    execute: bool = False,
    force: bool = False,
    overwrite_meteo: bool = False,
) -> pd.DataFrame:
    """Prepare Wflow dynamic handoffs for a batch of Wflow-SFINCS events."""
    location_root = Path(location_root)
    worklist = dynamic_handoff_batch_worklist(
        config,
        location_root,
        catalog_path=catalog_path,
        event_ids=event_ids,
        status=status,
        limit=limit,
    )
    rows = []
    for i, item in enumerate(worklist.to_dict("records"), start=1):
        event_id = str(item["event_id"])
        paths = dynamic_handoff_paths(config, location_root, event_id)
        t0 = time.time()
        if paths["acceptance"].exists() and not force:
            try:
                accepted = require_handoff(config, location_root, event_id, catalog_path=catalog_path)
                rows.append(
                    {
                        "event_id": event_id,
                        "status": "skipped_accepted",
                        "sfincs_discharge_forcing": accepted["sfincs_discharge_forcing"],
                        "dynamic_handoff_acceptance": accepted["dynamic_handoff_acceptance"],
                        "duration_sec": time.time() - t0,
                        "message": "",
                    }
                )
                continue
            except Exception:
                pass
        try:
            if execute:
                pre_event_hours, post_event_hours = configured_event_window_hours(config)
                build_meteo(
                    config,
                    location_root,
                    event_id,
                    catalog_path=catalog_path,
                    pre_event_hours=pre_event_hours,
                    post_event_hours=post_event_hours,
                    overwrite=overwrite_meteo,
                )
            report = prepare_handoff(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                execute=execute,
            )
            status_value = "accepted" if execute else "planned"
            if execute:
                require_handoff(config, location_root, event_id, catalog_path=catalog_path)
            rows.append(
                {
                    "event_id": event_id,
                    "status": status_value,
                    "sfincs_discharge_forcing": str(paths["discharge"]),
                    "dynamic_handoff_acceptance": str(paths["acceptance"]),
                    "duration_sec": time.time() - t0,
                    "message": f"{i}/{len(worklist)}",
                    "rows": int(len(report)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "event_id": event_id,
                    "status": "failed",
                    "sfincs_discharge_forcing": str(paths["discharge"]),
                    "dynamic_handoff_acceptance": str(paths["acceptance"]),
                    "duration_sec": time.time() - t0,
                    "message": str(exc),
                }
            )
            if execute:
                break
    return pd.DataFrame(rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch prepare dynamic Wflow-to-SFINCS handoffs.")
    parser.add_argument("--config", required=True, help="Location config.yaml")
    parser.add_argument("--catalog-path", default="data/event_catalog/catalog/scenario_catalog.csv")
    parser.add_argument("--event-id", action="append", dest="event_ids")
    parser.add_argument("--status", choices=["blocked", "accepted", "all"], default="blocked")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true", help="Run Wflow replay/QA; omit for dry planning.")
    parser.add_argument("--force", action="store_true", help="Reprocess already accepted events.")
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
    catalog_path = Path(args.catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = location_root / catalog_path
    report = run_handoffs(
        config,
        location_root,
        catalog_path=catalog_path,
        event_ids=args.event_ids,
        status=args.status,
        limit=args.limit,
        execute=args.execute,
        force=args.force,
        overwrite_meteo=args.overwrite_meteo,
    )
    print(report.to_string(index=False))
    out = args.out
    if out is None:
        out_dir = location_root / config.get("wflow", {}).get("events_root", "data/wflow/events")
        out = out_dir / ("dynamic_handoff_batch_execute.csv" if args.execute else "dynamic_handoff_batch_plan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(json.dumps({"report": str(out), "failed": int(report["status"].eq("failed").sum())}, indent=2))
    return 1 if "failed" in set(report.get("status", [])) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
