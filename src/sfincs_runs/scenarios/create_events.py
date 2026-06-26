import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from sfincs_runs.config import build_paths, load_runtime
from sfincs_runs.scenarios import event_forcing
from sfincs_runs.scenarios.event_forcing import EventForcing
from sfincs_runs.scenarios.io import write_json
from sfincs_runs.scenarios.scenarios import (
    assert_event_catalog_audit,
    build_event_timeseries,
    ensure_clean_dir,
    events_dir,
    select_zsini_from_series,
)

paths = build_paths()
default_design_outputs = paths["design_outputs_root"]
default_base_model = paths["base_model_root"]
default_scenarios = paths["scenarios_root"]

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Create SFINCS event folders from design-event hydrographs.")
    p.add_argument("--config", default=None, help="optional config overlay yaml")
    p.add_argument("--design-outputs", type=Path, default=None)
    p.add_argument("--design-scenario", default="base")
    p.add_argument("--forcing-variable", choices=["auto", "surge", "surge_absolute", "water_level_total"], default="auto")
    p.add_argument("--base-dir", type=Path, default=None)
    p.add_argument("--scenarios-dir", type=Path, default=None)
    p.add_argument("--event-id", action="append", dest="event_ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", help="keep existing scenario folders and rebuild only missing/incomplete events")
    p.add_argument("--include-waves", action="store_true")
    p.add_argument("--include-precip", action="store_true")
    p.add_argument("--full-forcing", action="store_true", help="stage coastal water level, SnapWave, precipitation, and soil moisture")
    p.add_argument("--zsini-mode", choices=["dry", "boundary_t0"], default="dry")
    p.add_argument("--tref", default="2000-01-01 00:00:00")
    args = p.parse_args(argv)
    runtime_config, runtime_paths = load_runtime(args.config)
    if args.full_forcing:
        args.include_waves = True
        args.include_precip = True
    args.design_outputs = args.design_outputs or runtime_paths["design_outputs_root"]
    if args.base_dir is None and runtime_config.get("coastal_waves", False):
        wave_base = (
            runtime_config.get("coastal_wave_coupling", {})
            .get("quadtree", {})
            .get("base_model_root")
        )
        if wave_base:
            wave_base = Path(wave_base)
            args.base_dir = (
                wave_base
                if wave_base.is_absolute()
                else runtime_paths["location_root"] / wave_base
            )
    args.base_dir = args.base_dir or runtime_paths["base_model_root"]
    args.scenarios_dir = args.scenarios_dir or runtime_paths["scenarios_root"]
    args.default_scenarios_dir = runtime_paths["scenarios_root"]
    args.runtime_config = runtime_config
    args.runtime_paths = runtime_paths
    return args

def select_rows(df, event_ids=None, limit=None):
    if event_ids:
        wanted = {str(x).strip() for x in event_ids}
        df = df[df["event_id"].astype(str).isin(wanted)].copy()
        missing = wanted - set(df["event_id"].astype(str))
        if missing:
            raise FileNotFoundError(f"Missing event IDs: {', '.join(sorted(missing))}")
    df = df.head(int(limit)).copy() if limit is not None else df.copy()
    if df.empty:
        raise RuntimeError("No events selected.")
    return df.reset_index(drop=True)


def read_event_catalog_inputs(root, *, scenario="base"):
    root = Path(root)
    assert_event_catalog_audit(root)
    catalog_path = root / "catalog" / "event_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    df = pd.read_csv(catalog_path)
    if "event_id" not in df:
        raise RuntimeError(f"Missing event_id column in {catalog_path}")
    df["event_id"] = df["event_id"].astype(str)

    ev_dir = events_dir(root, scenario)
    member_path = ev_dir / "surge_event_members.nc"
    ds = xr.open_dataset(member_path).load()
    df["design_scenario"] = ds.attrs.get("scenario_name", scenario)
    df["design_slr_offset_m"] = ds.attrs.get("slr_offset_m", 0.0)
    return df, ds


def event_forcing_from_row(row, ds, args) -> EventForcing:
    ts = build_event_timeseries(row, surge_event_members=ds, forcing_variable=args.forcing_variable)
    h = ts["h"].reset_index(drop=True)
    t_start = pd.Timestamp(args.tref)
    return EventForcing(
        event_id=str(row.event_id),
        catalog=row.to_dict(),
        h=h,
        forcing_variable=str(ts["forcing_variable"]),
        t_start=t_start,
        t_stop=t_start + pd.Timedelta(hours=max(0, len(h) - 1)),
        zsini=select_zsini_from_series(h, mode=args.zsini_mode),
        design_scenario=str(row.get("design_scenario", args.design_scenario)),
        design_slr_offset_m=float(row.get("design_slr_offset_m", 0.0)),
        surge_dataset=str(events_dir(args.design_outputs, args.design_scenario) / "surge_event_members.nc"),
    )


def _build_precip_model(base_dir):
    import os

    os.environ.pop("DEBUG", None)
    from hydromt_sfincs import SfincsModel

    return SfincsModel(root=str(base_dir), mode="r+")


def _read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def required_scenario_files(*, include_waves=False, include_precip=False):
    names = {
        "sfincs.inp",
        "sfincs.bnd",
        "sfincs.bzs",
        "forcing_manifest.json",
    }
    if include_waves:
        names.update(
            {
                "snapwave.bnd",
                "snapwave.bhs",
                "snapwave.btp",
                "snapwave.bwd",
                "snapwave.bds",
            }
        )
    if include_precip:
        names.update(
            {
                "aorc_precip_for_sfincs.nc",
                "sfincs_netampr.nc",
            }
        )
    return names


def scenario_is_complete(event_dir, *, include_waves=False, include_precip=False):
    event_dir = Path(event_dir)
    missing = [
        name
        for name in required_scenario_files(
            include_waves=include_waves,
            include_precip=include_precip,
        )
        if not (event_dir / name).exists()
    ]
    if missing:
        return False
    manifest = _read_manifest(event_dir / "forcing_manifest.json")
    if include_waves and not bool(manifest.get("expected_has_waves")):
        return False
    if include_precip and not bool(manifest.get("expected_has_precip")):
        return False
    if include_precip and not _scenario_precipitation_is_valid(event_dir, manifest):
        return False
    return True


def _scenario_precipitation_is_valid(event_dir, manifest):
    event_dir = Path(event_dir)
    netampr = event_dir / str(manifest.get("netamprfile") or "sfincs_netampr.nc")
    if not netampr.exists():
        return False
    run_start = manifest.get("run_start")
    run_stop = manifest.get("run_stop")
    if not run_start or not run_stop:
        return False
    try:
        with xr.open_dataset(netampr) as ds:
            if "time" not in ds:
                return False
            times = pd.to_datetime(ds["time"].values)
            if len(times) == 0:
                return False
            if times[0] != pd.Timestamp(run_start) or times[-1] != pd.Timestamp(run_stop):
                return False
            for name in ds.data_vars:
                values = ds[name].values
                if np.issubdtype(values.dtype, np.number) and np.isfinite(values).any():
                    return True
            return False
    except Exception:
        return False


def _report_row(manifest, *, row, index, event_dir, args, status):
    return {
        **manifest,
        "source_event_index": index + 1,
        "run_root": str(event_dir),
        "scenario_status": status,
        "precip_mode": "netampr" if args.include_precip else "surgeonly",
        "design_outputs_root": str(args.design_outputs),
        "driver_h_magnitude": float(row.get("absolute_peak_m", row.get("peak_m", row.get("peak", float("nan"))))),
        "driver_p_magnitude": float(row.get("p", 0.0)),
        "zsini_mode": args.zsini_mode,
        "template_id": str(row.get("template_id", "")),
        "tail_morph_factor": float(row.get("tail_morph_factor", 1.0)),
    }


def _validate_snapwave_source_windows(df, *, paths):
    if not {"event_id", "snapwave_member_file", "snapwave_valid_start_time", "snapwave_valid_end_time"}.issubset(df.columns):
        return
    issues = []
    for wave_file_value, group in df.groupby("snapwave_member_file", dropna=True):
        wave_file = event_forcing._resolve_catalog_path(wave_file_value, paths=paths or {})
        with xr.open_dataset(wave_file) as ds:
            time_name = "time" if "time" in ds.coords else "valid_time" if "valid_time" in ds.coords else None
            if time_name is None or ds[time_name].size == 0:
                continue
            source_start = pd.Timestamp(ds[time_name].values[0])
            source_stop = pd.Timestamp(ds[time_name].values[-1])
        starts = pd.to_datetime(group["snapwave_valid_start_time"], errors="coerce")
        stops = pd.to_datetime(group["snapwave_valid_end_time"], errors="coerce")
        bad = group[(starts < source_start) | (stops > source_stop) | starts.isna() | stops.isna()]
        for row in bad.head(10).itertuples(index=False):
            issues.append(
                f"{row.event_id}: {row.snapwave_valid_start_time}..{row.snapwave_valid_end_time} "
                f"outside {source_start}..{source_stop}"
            )
    if issues:
        detail = "; ".join(issues)
        raise RuntimeError(f"Selected Event Catalog SnapWave windows are outside ERA5 wave coverage: {detail}")


def build_scenarios(args, *, config=None, runtime_paths=None, sf_model=None):
    config = args.runtime_config if config is None else config
    runtime_paths = args.runtime_paths if runtime_paths is None else runtime_paths
    if args.design_scenario != "base" and args.scenarios_dir == args.default_scenarios_dir:
        args.scenarios_dir = args.default_scenarios_dir.with_name(f"{args.default_scenarios_dir.name}_{args.design_scenario}")

    # 1. Read design-event inputs with the full forcing-pairing metadata.
    df, ds = read_event_catalog_inputs(args.design_outputs, scenario=args.design_scenario)
    df = select_rows(df, args.event_ids, args.limit)
    if args.include_waves:
        _validate_snapwave_source_windows(df, paths=runtime_paths)

    # 2. Set up clean output folder. stage_run hardlinks static base files
    # where possible, so a 500-event batch does not duplicate the quadtree grid.
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive.")
    if args.resume:
        root = Path(args.scenarios_dir)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = ensure_clean_dir(args.scenarios_dir, force=args.force)
    rows, wall_t0 = [], time.time()
    precip_model = sf_model

    for i, row in df.iterrows():
        # 3. Build one event folder.
        forcing = event_forcing_from_row(row, ds, args)
        event_dir = root / forcing.event_id
        if args.resume and scenario_is_complete(
            event_dir,
            include_waves=args.include_waves,
            include_precip=args.include_precip,
        ):
            manifest = _read_manifest(event_dir / "forcing_manifest.json")
            rows.append(
                _report_row(
                    manifest,
                    row=row,
                    index=i,
                    event_dir=event_dir,
                    args=args,
                    status="skipped_existing",
                )
            )
            continue

        staged = event_forcing.stage_run(
            args.base_dir,
            root,
            forcing,
            force=True,
            include_waves=args.include_waves,
            include_precip=args.include_precip,
            timing_config={"allow_legacy_inference": True},
            paths=runtime_paths,
            config=config,
        )
        if args.include_precip:
            if precip_model is None:
                precip_model = _build_precip_model(args.base_dir)
            event_forcing.stage_precip(
                precip_model,
                staged.run_root,
                forcing,
                paths=runtime_paths,
                config=config,
            )
            if not _scenario_precipitation_is_valid(
                staged.run_root,
                _read_manifest(staged.run_root / "forcing_manifest.json"),
            ):
                raise RuntimeError(
                    f"Invalid precipitation forcing written for {forcing.event_id}: "
                    f"{staged.run_root / 'sfincs_netampr.nc'}"
                )

        # 4. Keep small receipts for later batch runs/stats checks.
        manifest = _read_manifest(staged.run_root / "forcing_manifest.json")
        out = _report_row(
            manifest,
            row=row,
            index=i,
            event_dir=staged.run_root,
            args=args,
            status="written",
        )
        rows.append(out)

        if (i + 1) % 25 == 0 or i + 1 == len(df):
            written = sum(r.get("scenario_status") == "written" for r in rows)
            skipped = sum(r.get("scenario_status") == "skipped_existing" for r in rows)
            print(f"  {i + 1}/{len(df)} processed; {written} written, {skipped} skipped ({time.time() - wall_t0:.0f}s total)")

    # 5. Build catalogs used by SLURM and run statistics.
    report = pd.DataFrame(rows).sort_values("event_id")
    report[["event_id", "run_root"]].to_csv(root / "scenario_catalog.csv", index=False)
    report.to_csv(root / "scenario_build_report.csv", index=False)
    write_json(root / "scenario_summary.json", {
        "base_model_root": str(args.base_dir),
        "target_scenarios_dir": str(root),
        "event_count": len(report),
        "elapsed_seconds": time.time() - wall_t0,
        "design_outputs_root": str(args.design_outputs),
        "design_scenario": args.design_scenario,
        "surge_dataset": str(events_dir(args.design_outputs, args.design_scenario) / "surge_event_members.nc"),
        "forcing_variable": args.forcing_variable,
        "zsini_mode": args.zsini_mode,
        "include_waves": bool(args.include_waves),
        "include_precip": bool(args.include_precip),
        "static_file_strategy": "hardlink_then_copy",
        "resume": bool(args.resume),
        "written_count": int((report["scenario_status"] == "written").sum()) if "scenario_status" in report else len(report),
        "skipped_existing_count": int((report["scenario_status"] == "skipped_existing").sum()) if "scenario_status" in report else 0,
    })
    print(f"Wrote {len(report)} scenarios to {root}")
    return report


def main():
    args = parse_args()
    build_scenarios(args)

if __name__ == "__main__":
    main()
