from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "locations").exists():
            return parent
    return Path.cwd()

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch USGS instantaneous (IV) streamflow hydrographs for Wflow calibration/validation "
            "(Wflow Readiness) over named historical event windows. ADR-0016: not runtime forcing."
        )
    )
    parser.add_argument("--location", required=True, help="Location name, e.g. greensboro or austin.")
    parser.add_argument("--worklist", help="CSV with event_id rows. Defaults to the joint Wflow-SFINCS worklist.")
    parser.add_argument("--catalog", help="Event catalog CSV. Defaults to data/event_catalog/catalog/scenario_catalog.csv.")
    parser.add_argument("--limit", type=int, help="Optional first-N event limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Refetch even when the event IV cache already exists.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit nonzero if any event has no cached/fetched IV records.")
    args = parser.parse_args()

    repo_root = _repo_root()
    src_root = repo_root / "src"
    if src_root.exists() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from study_location import define_location
    from wflow_runs.replay import _event_reference_time, resolve_event_window
    from wflow_runs.streamflow_realization import cache_wflow_event_instantaneous_streamflow

    location_root = repo_root / "locations" / args.location
    config = define_location(location_root / "config.yaml").config
    # Fetches observed USGS instantaneous (IV) hydrographs for calibration/validation of the
    # Wflow discharge generator, not as per-event runtime forcing. Point it at named
    # historical validation events (via --worklist / --catalog) to pull their IV windows.
    config.setdefault("wflow", {}).setdefault("streamflow_calibration", {}).setdefault(
        "event_records_root", "data/sources/usgs_streamgages/event_streamflow_iv"
    )
    catalog_path = Path(args.catalog) if args.catalog else location_root / "data/event_catalog/catalog/scenario_catalog.csv"
    if not catalog_path.is_absolute():
        catalog_path = location_root / catalog_path
    worklist_path = (
        Path(args.worklist)
        if args.worklist
        else location_root / "data/sfincs/scenarios" / f"{args.location}_joint_wflow_sfincs_worklist.csv"
    )
    if not worklist_path.is_absolute():
        worklist_path = location_root / worklist_path
    worklist = pd.read_csv(worklist_path, dtype={"event_id": str})
    if "event_id" not in worklist:
        raise ValueError(f"{worklist_path} lacks event_id column")
    event_ids = worklist["event_id"].dropna().astype(str).drop_duplicates().tolist()
    if args.limit is not None:
        event_ids = event_ids[: args.limit]
    rows = []
    for index, event_id in enumerate(event_ids, start=1):
        reference_time = _event_reference_time(location_root, event_id, catalog_path)
        start, end = resolve_event_window(reference_time)
        print(f"[{index}/{len(event_ids)}] {event_id}: {start.isoformat()} -> {end.isoformat()}", flush=True)
        rows.append(
            cache_wflow_event_instantaneous_streamflow(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                start=start,
                end=end,
                overwrite=args.overwrite,
            )
        )
    report = pd.DataFrame(rows)
    report_path = location_root / "data/sources/usgs_streamgages/event_streamflow_iv_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(report["status"].value_counts(dropna=False).to_string())
    print(f"report: {report_path}")
    missing = report[~report["status"].isin(["cached", "fetched"])]
    if args.fail_on_missing and not missing.empty:
        print(missing[["event_id", "member_id", "status"]].to_string(index=False), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())