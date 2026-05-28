from __future__ import annotations
import argparse
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

# depreciated cora data pull in NOAA source, must use separate venv to collect CORA data to prevent breakage
cora_env = Path(__file__).resolve().parent / "collect_sources" / ".venv-cora-fast"
cora_python = cora_env / "bin" / "python"

def build_parser():
    parser = argparse.ArgumentParser(description="Run design-event steps.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    collect = subparsers.add_parser("collect_cora", help="collect the single-node Cora water-level series")
    collect.add_argument("--config", default=None, help="optional config overlay yaml")
    collect.add_argument("--start", default=None, help="optional start override yyyy-mm-dd")
    collect.add_argument("--end", default=None, help="optional end override yyyy-mm-dd")
    collect.add_argument("--skip-existing", action="store_true", help="reuse local outputs when present")
    collect.add_argument("--smoke", action="store_true", help="pull 2018-01 only for a fast check")
    sources = subparsers.add_parser("collect_sources", help="collect configured source artifacts")
    sources.add_argument("--config", default=None, help="optional config overlay yaml")
    sources.add_argument("--start", default=None, help="optional start override yyyy-mm-dd")
    sources.add_argument("--end", default=None, help="optional end override yyyy-mm-dd")
    sources.add_argument("--skip-existing", action="store_true", help="reuse local outputs when present")
    sources.add_argument("--smoke", action="store_true", help="pull 2018-01 only for a fast check")
    waves = subparsers.add_parser("collect_era5_waves", help="collect ERA5 SnapWave boundary forcing")
    waves.add_argument("--config", default=None, help="optional config overlay yaml")
    waves.add_argument("--start", default=None, help="optional start override yyyy-mm-dd")
    waves.add_argument("--end", default=None, help="optional end override yyyy-mm-dd")
    waves.add_argument("--skip-existing", action="store_true", help="reuse local outputs when present")
    waves.add_argument("--smoke", action="store_true", help="pull a small configured wave window")
    aorc_sst = subparsers.add_parser("collect_aorc_sst", help="collect direct AORC SST rainfall members")
    aorc_sst.add_argument("--config", default=None, help="optional config overlay yaml")
    aorc_sst.add_argument("--start", default=None, help="optional start override yyyy-mm-dd")
    aorc_sst.add_argument("--end", default=None, help="optional end override yyyy-mm-dd")
    aorc_sst.add_argument("--skip-existing", action="store_true", help="reuse local outputs when present")
    readiness = subparsers.add_parser("check_readiness", help="write data-acquisition readiness audit")
    readiness.add_argument("--config", default=None, help="optional config overlay yaml")
    catalog = subparsers.add_parser("build_catalog", help="extract historical surge peaks and fit the marginal model")
    catalog.add_argument("--config", default=None, help="optional config overlay yaml")
    sensitivity = subparsers.add_parser("sensitivity", help="write POT threshold/model sensitivity table")
    sensitivity.add_argument("--config", default=None, help="optional config overlay yaml")
    sample = subparsers.add_parser("sample_peaks", help="sample synthetic target peaks from the fitted marginal")
    sample.add_argument("--config", default=None, help="optional config overlay yaml")
    distribution = subparsers.add_parser("event_distribution", help="summarize mild-to-extreme event coverage")
    distribution.add_argument("--config", default=None, help="optional config overlay yaml")
    events = subparsers.add_parser("build_event_members", help="build the 2500 surge event members")
    events.add_argument("--config", default=None, help="optional config overlay yaml")
    events.add_argument("--scenario", default="base",
                        help="MSL-shift scenario name from config.yaml scenarios block")
    event_catalog = subparsers.add_parser("build_event_catalog", help="build the event recipe catalog")
    event_catalog.add_argument("--config", default=None, help="optional config overlay yaml")
    event_catalog.add_argument("--scenario", default="base",
                               help="MSL-shift scenario name from config.yaml scenarios block")
    stress_training = subparsers.add_parser(
        "select_resilience_stress_training",
        help="write the consequence-enriched SFINCS stress/training event subset",
    )
    stress_training.add_argument("--config", default=None, help="optional config overlay yaml")
    run_all = subparsers.add_parser("all", help="run collect_cora -> build_catalog -> sample_peaks -> build_event_members")
    run_all.add_argument("--config", default=None, help="optional config overlay yaml")
    run_all.add_argument("--start", default=None, help="optional start override yyyy-mm-dd")
    run_all.add_argument("--end", default=None, help="optional end override yyyy-mm-dd")
    run_all.add_argument("--skip-existing", action="store_true", help="reuse local Cora outputs when present")
    run_all.add_argument("--smoke", action="store_true", help="pull 2018-01 only for a fast check")
    run_all.add_argument("--scenario", default="base",
                         help="MSL-shift scenario name applied at build_event_members")
    return parser

def _collection_settings(config, paths, args):
    import pandas as pd

    collection = config.get("collection", {})
    start = pd.Timestamp(args.start or collection.get("start", "1979-01-01"))
    end = pd.Timestamp(args.end or collection.get("end", "2022-12-31"))
    if end < start:
        raise ValueError("end date must be on or after start date")
    return {
        "config": config,
        "paths": paths,
        "start": start,
        "end": end,
        "cora": collection.get("cora", {}),
    }

def _running_in_cora_venv():
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and Path(virtual_env).resolve() == cora_env.resolve():
        return True
    return Path(sys.prefix).resolve() == cora_env.resolve()

def _rerun_in_cora_venv(argv):
    if not cora_python.exists():
        raise FileNotFoundError(f"missing Cora Python: {cora_python}")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(cora_env)
    env["PATH"] = f"{cora_env / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    completed = subprocess.run(
        [str(cora_python), "-m", "design_events", *argv],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)

def _load_step(module_name):
    return import_module(f"design_events.{module_name}")

def _read_settings(config_path, scenario=None):
    from design_events.config import load_runtime
    return load_runtime(config_path, scenario=scenario)

def _show_written(paths, keys):
    for key in keys:
        print(f"wrote {paths[key]}")

def run_collection(args):
    collect_cora = _load_step("collect_sources.cora").collect_cora
    config, paths = _read_settings(args.config)
    settings = _collection_settings(config, paths, args)
    collect_cora(settings, skip_existing=args.skip_existing, smoke=args.smoke)

def run_era5_waves(args):
    era5_waves = _load_step("collect_sources.era5_waves")
    config, paths = _read_settings(args.config)
    collection = config.get("collection", {})
    settings = {
        "config": config,
        "paths": paths,
        "start": pd.Timestamp(args.start or collection.get("start", "1979-01-01")),
        "end": pd.Timestamp(args.end or collection.get("end", "2022-12-31")),
        "era5_waves": collection.get("era5_waves", {}),
    }
    result = era5_waves.collect_era5_waves(
        settings,
        skip_existing=args.skip_existing,
        smoke=args.smoke,
    )
    print(f"wrote {result['wave_netcdf']}")


def run_aorc_sst(args):
    aorc_sst = _load_step("collect_sources.aorc_sst")
    config, paths = _read_settings(args.config)
    collection = config.get("collection", {})
    settings = {
        "config": config,
        "paths": paths,
        "start": pd.Timestamp(args.start or collection.get("aorc_sst", {}).get("start_date") or collection.get("start", "1979-01-01")),
        "end": pd.Timestamp(args.end or collection.get("aorc_sst", {}).get("end_date") or collection.get("end", "2022-12-31")),
        "aorc_sst": collection.get("aorc_sst", {}),
    }
    result = aorc_sst.collect_aorc_sst(settings, skip_existing=args.skip_existing)
    print(f"wrote {result['ranked_storms_csv']} ({result['ranked_rows']} ranked storms)")
    print(f"wrote {paths['aorc_sst_rainfall_members_csv']} ({result['rainfall_member_rows']} members)")


def run_readiness(args):
    readiness = _load_step("readiness")
    config, paths = _read_settings(args.config)
    audit = readiness.write_data_acquisition_readiness(config, paths)
    print(f"wrote {paths['data_acquisition_readiness_json']} (passed={audit['passed']})")

def run_source_collection(args):
    collect_all_sources = _load_step("collect_sources.all_sources").collect_all_sources
    config, paths = _read_settings(args.config)
    result = collect_all_sources(
        config,
        paths,
        start=args.start,
        end=args.end,
        skip_existing=args.skip_existing,
        smoke=args.smoke,
    )
    print(f"cora rows: {result['cora_rows']}")
    if result["aorc_sst"] is not None:
        print(f"wrote {paths['aorc_sst_rainfall_members_csv']}")
    if result["era5_waves"] is not None:
        print(f"wrote {result['era5_waves']['wave_netcdf']}")

def run_catalog(args):
    build_catalog = _load_step("fit_history.peaks").build_catalog
    config, paths = _read_settings(args.config)
    artifacts = build_catalog(config, paths)
    _show_written(paths, ["historical_peaks_csv", "marginal_params_csv", "marginal_rps_csv"])
    if artifacts["figure"] is not None:
        print(f"wrote {artifacts['figure']}")

def run_sampler(args):
    build_sampled_peaks = _load_step("build_events.sample_peaks").build_sampled_peaks
    config, paths = _read_settings(args.config)
    df = build_sampled_peaks(config, paths)
    print(f"wrote {paths['sampled_peaks_csv']} ({len(df)} events)")

def run_event_distribution(args):
    write_event_distribution_artifacts = _load_step(
        "build_events.event_distribution"
    ).write_event_distribution_artifacts
    config, paths = _read_settings(args.config)
    summary = write_event_distribution_artifacts(config, paths)
    print(
        f"wrote {paths['event_distribution_summary_csv']} "
        f"and {paths['event_distribution_plot_png']} ({summary['event_count']} events)"
    )

def run_sensitivity(args):
    build_threshold_model_sensitivity = _load_step("fit_history.peaks").build_threshold_model_sensitivity
    config, paths = _read_settings(args.config)
    df = build_threshold_model_sensitivity(config, paths)
    print(f"wrote {paths['sensitivity_csv']} ({len(df)} rows)")

def run_event_members(args):
    hydrographs = _load_step("build_events.hydrographs")
    build_surge_event_artifacts = hydrographs.build_surge_event_artifacts
    write_event_artifacts = hydrographs.write_event_artifacts
    scenario = getattr(args, "scenario", "base")
    config, paths = _read_settings(args.config, scenario=scenario)
    print(
        f"[scenario] {paths['scenario']['name']} "
        f"(slr_offset_m={paths['scenario']['slr_offset_m']:+.3f})"
    )
    artifacts = build_surge_event_artifacts(config, paths)
    write_event_artifacts(paths, artifacts)
    _show_written(
        paths,
        [
            "template_bank_nc",
            "event_members_nc",
            "event_summary_csv",
            "event_acceptance_json",
            "lagtimes_csv",
            "event_overview_png",
        ],
    )

def run_event_catalog(args):
    build_event_catalog = _load_step("build_events.event_catalog").build_event_catalog
    scenario = getattr(args, "scenario", "base")
    config, paths = _read_settings(args.config, scenario=scenario)
    df = build_event_catalog(config, paths)
    print(f"wrote {paths['event_catalog_csv']} ({len(df)} events)")

def run_resilience_stress_training(args):
    write_resilience_stress_training_artifacts = _load_step(
        "build_events.event_selection"
    ).write_resilience_stress_training_artifacts
    config, paths = _read_settings(args.config)
    df = write_resilience_stress_training_artifacts(config, paths)
    print(f"wrote {paths['resilience_stress_training_catalog_csv']} ({len(df)} events)")

def run_all(args):
    run_collection(args)
    run_catalog(args)
    run_sampler(args)
    run_event_members(args)
    run_event_catalog(args)

stage_handlers = {
    "collect_cora": run_collection,
    "collect_sources": run_source_collection,
    "collect_era5_waves": run_era5_waves,
    "collect_aorc_sst": run_aorc_sst,
    "check_readiness": run_readiness,
    "build_catalog": run_catalog,
    "sensitivity": run_sensitivity,
    "sample_peaks": run_sampler,
    "event_distribution": run_event_distribution,
    "build_event_members": run_event_members,
    "build_event_catalog": run_event_catalog,
    "select_resilience_stress_training": run_resilience_stress_training,
    "all": run_all,
}

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage.startswith("collect_cora") and not _running_in_cora_venv():
        _rerun_in_cora_venv(argv)
    handler = stage_handlers.get(args.stage)
    if handler is None:
        parser.error(f"unknown stage {args.stage!r}")
    handler(args)

if __name__ == "__main__":
    main()
