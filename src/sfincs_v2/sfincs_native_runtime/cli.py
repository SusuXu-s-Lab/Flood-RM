from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit import audit_run_folder
from .coastal import build_coastal_hydrograph_from_analog, stage_coastal_event_forcing
from .forcing import stage_inland_event_forcing
from .hydrology import prepare_aorc_precip_for_sfincs
from .probability import annual_rate_table, catalog_depth_probability, completed_runs
from .runtime import load_config, native_source_config_from_dict, paths_from_config
from .schema import NativeSourceConfig
from .solver import run_prepared_events
from .sources import create_wflow_source_contract


def _add_common_runtime(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--location-root", type=Path, default=None)


def _runtime(args):
    config = load_config(args.config)
    location_root = args.location_root or args.config.parent
    return config, paths_from_config(config, location_root=location_root)


def create_sources_cmd(args) -> int:
    config, paths = _runtime(args)
    forcing_cfg = ((config.get("inland_coupling") or {}).get("discharge_forcing") or {})
    source_cfg = NativeSourceConfig(**native_source_config_from_dict({**forcing_cfg, **{k: v for k, v in vars(args).items() if v is not None}}))
    out = args.output or paths.source_contract
    src = create_wflow_source_contract(
        args.base_root or paths.base_model_root,
        sfincs_domain_id=args.sfincs_domain_id,
        output=out,
        source_config=source_cfg,
        data_libs=[str(paths.data_catalog)] if paths.data_catalog and paths.data_catalog.exists() else None,
        wflow_submodel_id=args.wflow_submodel_id or args.sfincs_domain_id,
    )
    print(src[["index", "name", "sfincs_domain_id", "wflow_submodel_id"]].to_string(index=False))
    print(f"source_contract={out}")
    return 0


def stage_inland_cmd(args) -> int:
    manifest = stage_inland_event_forcing(
        args.run_root,
        event_id=args.event_id,
        wflow_discharge_nc=args.wflow_discharge_nc,
        precip_nc=args.precip_nc,
        direct_rainfall=bool(args.precip_nc),
        initial_zsini_m=args.zsini,
        probability_weight=args.probability_weight,
        total_rate_per_year=args.total_rate_per_year,
        annual_rate=args.annual_rate,
        sfincs_domain_id=args.sfincs_domain_id or "",
    )
    print(pd.Series(manifest.to_dict()).to_string())
    return 0


def stage_coastal_cmd(args) -> int:
    components = pd.read_csv(args.components_csv, parse_dates=[args.time_column]).set_index(args.time_column)
    eta = build_coastal_hydrograph_from_analog(
        components,
        args.peak_time,
        args.scale_factor,
        window_hours=args.window_hours,
        msl_offset_m=args.msl_offset_m,
        return_absolute_time=True,
    )
    manifest = stage_coastal_event_forcing(
        args.run_root,
        event_id=args.event_id,
        eta=eta,
        precip_nc=args.precip_nc,
        include_precip=bool(args.precip_nc),
        initial_zsini_m=args.zsini,
        probability_weight=args.probability_weight,
        total_rate_per_year=args.total_rate_per_year,
        annual_rate=args.annual_rate,
        sfincs_domain_id=args.sfincs_domain_id or "",
        metadata={"coastal_analog_peak_time": str(pd.Timestamp(args.peak_time)), "coastal_water_level_scale_factor": float(args.scale_factor)},
    )
    print(pd.Series(manifest.to_dict()).to_string())
    return 0


def prepare_precip_cmd(args) -> int:
    out = prepare_aorc_precip_for_sfincs(
        args.source_nc,
        args.output_nc,
        t_start=args.t_start,
        t_stop=args.t_stop,
        variable=args.variable,
        freq=args.freq,
        window_alignment=args.window_alignment,
        precip_start=args.precip_start,
        scale_factor=args.scale_factor,
    )
    print(out)
    return 0


def run_cmd(args) -> int:
    report = run_prepared_events(
        args.scenarios_root,
        storage_root=args.storage_root,
        run_root=args.run_root,
        scenario_catalog=args.scenario_catalog,
        event_ids=args.event_id,
        limit=args.limit,
        sfincs_bin=args.sfincs_bin,
        workers=args.workers,
        force_rerun=args.force_rerun,
        dry_run=args.dry_run,
        keep_stage=args.keep_stage,
        threads=args.threads,
    )
    print(report.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
    return 1 if "status" in report and report["status"].eq("failed").any() else 0


def audit_cmd(args) -> int:
    report = audit_run_folder(args.run_root)
    print(pd.DataFrame([issue.__dict__ for issue in report.issues]).to_string(index=False) if report.issues else "passed")
    return 0 if report.passed else 1


def probability_cmd(args) -> int:
    weights = pd.read_csv(args.event_rates_csv)
    if "annual_rate" not in weights:
        if args.total_rate_per_year is None:
            raise ValueError("--total-rate-per-year is required unless event_rates_csv already has annual_rate")
        weights = annual_rate_table(weights, args.total_rate_per_year)
    runs = completed_runs(args.storage_root)
    ds = catalog_depth_probability(runs, weights, thresholds_ft=tuple(args.threshold_ft))
    args.output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output_nc)
    print(args.output_nc)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfincs-native-runtime", description="Minimal native HydroMT-SFINCS event runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-sources", help="Create SFINCS-native river inflow source contract for Wflow")
    _add_common_runtime(p)
    p.add_argument("--base-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--sfincs-domain-id", required=True)
    p.add_argument("--wflow-submodel-id", default=None)
    p.add_argument("--hydrography", default=None)
    p.add_argument("--river-upa-km2", type=float, default=None)
    p.add_argument("--river-len-m", type=float, default=None)
    p.add_argument("--buffer-m", type=float, default=None)
    p.add_argument("--river-width-m", type=float, default=None)
    p.set_defaults(func=create_sources_cmd)

    p = sub.add_parser("stage-inland", help="Stage Wflow discharge into one SFINCS event folder")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument("--wflow-discharge-nc", type=Path, required=True)
    p.add_argument("--precip-nc", type=Path, default=None)
    p.add_argument("--zsini", type=float, default=None)
    p.add_argument("--sfincs-domain-id", default="")
    p.add_argument("--probability-weight", type=float, default=None)
    p.add_argument("--total-rate-per-year", type=float, default=None)
    p.add_argument("--annual-rate", type=float, default=None)
    p.set_defaults(func=stage_inland_cmd)

    p = sub.add_parser("stage-coastal", help="Stage coastal water-level forcing into one SFINCS event folder")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--event-id", required=True)
    p.add_argument("--components-csv", type=Path, required=True)
    p.add_argument("--time-column", default="time")
    p.add_argument("--peak-time", required=True)
    p.add_argument("--scale-factor", type=float, required=True)
    p.add_argument("--window-hours", type=float, default=72.0)
    p.add_argument("--msl-offset-m", type=float, default=0.0)
    p.add_argument("--precip-nc", type=Path, default=None)
    p.add_argument("--zsini", type=float, default=None)
    p.add_argument("--sfincs-domain-id", default="")
    p.add_argument("--probability-weight", type=float, default=None)
    p.add_argument("--total-rate-per-year", type=float, default=None)
    p.add_argument("--annual-rate", type=float, default=None)
    p.set_defaults(func=stage_coastal_cmd)

    p = sub.add_parser("prepare-precip", help="Prepare AORC interval precipitation for HydroMT-SFINCS")
    p.add_argument("--source-nc", type=Path, required=True)
    p.add_argument("--output-nc", type=Path, required=True)
    p.add_argument("--t-start", required=True)
    p.add_argument("--t-stop", required=True)
    p.add_argument("--variable", default="APCP_surface")
    p.add_argument("--freq", default="1h")
    p.add_argument("--window-alignment", choices=("start", "wettest"), default="start")
    p.add_argument("--precip-start", default=None)
    p.add_argument("--scale-factor", type=float, default=1.0)
    p.set_defaults(func=prepare_precip_cmd)

    p = sub.add_parser("run", help="Run prepared SFINCS event folders")
    p.add_argument("--scenarios-root", type=Path, required=True)
    p.add_argument("--storage-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--scenario-catalog", type=Path, default=None)
    p.add_argument("--event-id", action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sfincs-bin", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-stage", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=run_cmd)

    p = sub.add_parser("audit", help="Audit a staged SFINCS event folder")
    p.add_argument("--run-root", type=Path, required=True)
    p.set_defaults(func=audit_cmd)

    p = sub.add_parser("probability", help="Build catalog-weighted flood-depth probability rasters")
    p.add_argument("--storage-root", type=Path, required=True)
    p.add_argument("--event-rates-csv", type=Path, required=True)
    p.add_argument("--total-rate-per-year", type=float, default=None)
    p.add_argument("--threshold-ft", type=float, action="append", default=[0.5, 1.0, 2.0])
    p.add_argument("--output-nc", type=Path, required=True)
    p.set_defaults(func=probability_cmd)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
