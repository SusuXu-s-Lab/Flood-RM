from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Point

from sfincs_runs.hydrology import (
    find_aorc_event_window,
    prepare_aorc_precip_for_sfincs,
    summarize_soil_moisture,
)
from sfincs_runs.scenarios.io import count_nonempty_lines, parse_sfincs_inp, write_bzs, write_json, write_sfincs_inp
from sfincs_runs.scenarios.scenarios import (
    assert_event_catalog_audit,
    build_event_timeseries,
    events_dir,
    select_zsini_from_series,
)
from sfincs_runs.scenarios.timing import plan_event_forcing_support_window
from sfincs_runs.snapwave_setup import era5_spectra_to_snapwave_timeseries


@dataclass(frozen=True)
class EventForcing:
    event_id: str
    catalog: dict
    h: pd.Series
    forcing_variable: str
    t_start: pd.Timestamp
    t_stop: pd.Timestamp
    zsini: float
    design_scenario: str
    design_slr_offset_m: float
    surge_dataset: str


@dataclass(frozen=True)
class StagedEventRun:
    run_root: Path
    manifest: dict


@dataclass(frozen=True)
class SfincsRunResult:
    run_root: Path
    log_path: Path
    map_path: Path
    returncode: int


@dataclass(frozen=True)
class SingleUseEvent:
    plan: object
    forcing: EventForcing
    base_model_root: Path


stale_run_files = (
    "sfincs_map.nc",
    "sfincs_his.nc",
    "sfincs_rst.nc",
    "sfincs.log",
    "sfincs_log.txt",
    "sfincs.precip",
    "sfincs.bzs",
    "forcing_manifest.json",
)

mutable_staged_files = frozenset(
    {
        "sfincs.inp",
        "sfincs.bzs",
        "sfincs.seff",
        "sfincs.precip",
        "sfincs_netampr.nc",
        "aorc_precip_for_sfincs.nc",
        "forcing_manifest.json",
        "snapwave.bhs",
        "snapwave.btp",
        "snapwave.bwd",
        "snapwave.bds",
    }
)


_missing_config_value = object()


def _link_static_or_copy_mutable(src, dst):
    if Path(src).name in mutable_staged_files:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _snapshot_sfincs_config(config_component, keys) -> dict:
    return {key: _sfincs_config_get(config_component, key) for key in keys}


def _sfincs_config_get(config_component, key):
    getter = getattr(config_component, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            pass
    data = getattr(config_component, "data", None)
    if isinstance(data, dict):
        return data.get(key, _missing_config_value)
    return _missing_config_value


def _restore_sfincs_config(config_component, snapshot: dict) -> None:
    setter = getattr(config_component, "set", None)
    if not callable(setter):
        return
    for key, value in snapshot.items():
        if value is _missing_config_value:
            continue
        setter(key, value)


def load_event_forcing(
    design_outputs_root,
    *,
    event_id,
    design_scenario="base",
    forcing_variable="auto",
    tref="2000-01-01 00:00:00",
    zsini_mode="dry",
):
    root = Path(design_outputs_root)
    assert_event_catalog_audit(root)
    catalog = pd.read_csv(root / "catalog" / "event_catalog.csv")
    if "event_id" not in catalog:
        raise RuntimeError(f"Missing event_id column in {root / 'catalog' / 'event_catalog.csv'}")

    event_id = str(event_id).strip()
    selected = catalog[catalog["event_id"].astype(str) == event_id]
    if selected.empty:
        raise FileNotFoundError(f"Missing event ID in Event Catalog: {event_id}")
    if len(selected) > 1:
        raise RuntimeError(f"Event Catalog contains duplicate event_id rows: {event_id}")

    row = selected.iloc[0]
    surge_dataset = events_dir(root, design_scenario) / "surge_event_members.nc"
    ds = xr.open_dataset(surge_dataset).load()
    try:
        ts = build_event_timeseries(row, surge_event_members=ds, forcing_variable=forcing_variable)
    finally:
        ds.close()

    h = ts["h"].reset_index(drop=True)
    t_start = pd.Timestamp(tref)
    t_stop = t_start + pd.Timedelta(hours=max(0, len(h) - 1))
    return EventForcing(
        event_id=event_id,
        catalog=row.to_dict(),
        h=h,
        forcing_variable=str(ts["forcing_variable"]),
        t_start=t_start,
        t_stop=t_stop,
        zsini=select_zsini_from_series(h, mode=zsini_mode),
        design_scenario=design_scenario,
        design_slr_offset_m=float(ds.attrs.get("slr_offset_m", 0.0)),
        surge_dataset=str(surge_dataset),
    )


def build_single_use_event(
    config,
    paths,
    *,
    event_id=None,
    base_model_root=None,
    design_scenario="base",
    forcing_variable="auto",
    tref="2000-01-01 00:00:00",
    zsini_mode="dry",
):
    from sfincs_runs.single_use_case import build_single_use_case_plan

    plan = build_single_use_case_plan(
        config,
        paths,
        event_id=event_id,
        base_model_root=base_model_root,
    )
    forcing = load_event_forcing(
        plan.design_outputs_root,
        event_id=plan.event_id,
        design_scenario=design_scenario,
        forcing_variable=forcing_variable,
        tref=tref,
        zsini_mode=zsini_mode,
    )
    return SingleUseEvent(
        plan=plan,
        forcing=forcing,
        base_model_root=Path(plan.base_model_root),
    )


def stage_event_run(
    base_model_root,
    run_stage_root,
    forcing: EventForcing,
    *,
    force=False,
    include_waves=False,
    include_precip=False,
    extra_overrides=None,
    timing_config=None,
    paths=None,
    config=None,
):
    base_model_root = Path(base_model_root)
    event_root = Path(run_stage_root) / forcing.event_id
    if event_root.exists():
        if not force:
            raise FileExistsError(f"{event_root} exists. Use force=True to replace it.")
        shutil.rmtree(event_root)
    event_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_model_root, event_root, copy_function=_link_static_or_copy_mutable)
    for name in stale_run_files:
        path = event_root / name
        if path.exists():
            path.unlink()

    n_bnd = count_nonempty_lines(base_model_root / "sfincs.bnd")
    if n_bnd <= 0:
        raise RuntimeError(f"Empty boundary file: {base_model_root / 'sfincs.bnd'}")

    timing_cfg = dict(timing_config or {})
    support_window = plan_event_forcing_support_window(
        forcing.catalog,
        model_start_time=forcing.t_start,
        coastal_sample_count=len(forcing.h),
        allow_legacy_inference=bool(timing_cfg.get("allow_legacy_inference", True)),
        spinup_hours=float(timing_cfg.get("spinup_hours", 0)),
        drain_down_hours=float(timing_cfg.get("drain_down_hours", 0)),
        min_run_hours=timing_cfg.get("min_run_hours"),
        max_run_hours=timing_cfg.get("max_run_hours"),
    )

    overrides = {
        "tref": support_window.run_start.strftime("%Y%m%d %H%M%S"),
        "tstart": support_window.run_start.strftime("%Y%m%d %H%M%S"),
        "tstop": support_window.run_stop.strftime("%Y%m%d %H%M%S"),
        "zsini": f"{forcing.zsini:.6f}",
        "bzsfile": "sfincs.bzs",
        "storevel": "1",
    }
    if include_precip:
        overrides.update(
            {
                "netamprfile": "sfincs_netampr.nc",
                "storecumprcp": "1",
                "storemeteo": "1",
            }
        )
    if extra_overrides:
        overrides.update({str(k).lower(): str(v) for k, v in extra_overrides.items()})

    template = (base_model_root / "sfincs.inp").read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()
    write_sfincs_inp(
        template,
        event_root / "sfincs.inp",
        overrides=overrides,
        remove_keys={"srcfile", "disfile", "precipfile"},
    )
    write_bzs(
        event_root / "sfincs.bzs",
        forcing.h,
        n_bnd=n_bnd,
        start_seconds=_driver_start_seconds(support_window, "coastal"),
    )

    wave_manifest = {}
    if include_waves:
        wave_manifest = stage_event_snapwave(
            event_root,
            forcing,
            paths=paths or {"location_root": base_model_root.parent},
            config=config or {},
        )

    manifest = _event_manifest(
        base_model_root,
        forcing,
        n_bnd=n_bnd,
        include_precip=include_precip,
        include_waves=include_waves,
        support_window=support_window,
    )
    manifest.update(wave_manifest)
    write_json(event_root / "forcing_manifest.json", manifest)
    return StagedEventRun(run_root=event_root, manifest=manifest)


def resolve_event_hydrology_inputs(forcing: EventForcing, *, paths, config):
    catalog = forcing.catalog
    rainfall_member_id = _required_catalog_value(catalog, "rainfall_member_id")
    rainfall_member_time = pd.Timestamp(_required_catalog_value(catalog, "rainfall_member_time"))
    rainfall_member_file = _resolve_catalog_path(
        _required_catalog_value(catalog, "rainfall_member_file"),
        paths=paths,
    )
    hydrology_cfg = (config.get("coastal_wave_coupling") or {}).get("hydrology") or {}
    soil_cfg = hydrology_cfg.get("soil_moisture") or {}
    rainfall_windows_dir = (
        hydrology_cfg.get("precipitation", {}).get("event_windows_dir")
        or rainfall_member_file.parent / "event_windows"
    )
    rainfall_source_nc = find_aorc_event_window(
        _resolve_catalog_path(rainfall_windows_dir, paths=paths),
        member_id=rainfall_member_id,
        storm_start=rainfall_member_time,
    )

    soil_summary = None
    soil_member_file = catalog.get("soil_moisture_member_file")
    soil_member_time = catalog.get("soil_moisture_member_time")
    if soil_member_file is not None and not pd.isna(soil_member_file):
        soil_summary = summarize_soil_moisture(
            _resolve_catalog_path(soil_member_file, paths=paths),
            at_time=soil_member_time if soil_member_time is not None else rainfall_member_time,
            lookback_hours=float(soil_cfg.get("lookback_hours", 24)),
        )

    return {
        "rainfall_source_nc": str(rainfall_source_nc),
        "rainfall_member_id": str(rainfall_member_id),
        "rainfall_member_time": str(_required_catalog_value(catalog, "rainfall_member_time")),
        "rainfall_storm_start": rainfall_member_time.strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture_summary": soil_summary,
    }


def stage_event_precipitation(sf, run_root, forcing: EventForcing, *, paths, config):
    run_start, run_stop = _staged_run_window(Path(run_root), forcing)
    hydrology = resolve_event_hydrology_inputs(forcing, paths=paths, config=config)
    hydrology_cfg = (config.get("coastal_wave_coupling") or {}).get("hydrology") or {}
    precip_cfg = hydrology_cfg.get("precipitation") or {}
    window_alignment = str(precip_cfg.get("window_alignment", "wettest"))
    rainfall_scale_factor = _nullable_float(forcing.catalog.get("rainfall_scale_factor"))
    if rainfall_scale_factor is None or not (rainfall_scale_factor > 0):
        rainfall_scale_factor = 1.0
    prepared_precip = prepare_aorc_precip_for_sfincs(
        hydrology["rainfall_source_nc"],
        Path(run_root) / "aorc_precip_for_sfincs.nc",
        t_start=run_start,
        t_stop=run_stop,
        variable=str(precip_cfg.get("variable", "APCP_surface")),
        align_start_to_run=True,
        window_alignment=window_alignment,
        precip_start=_catalog_rainfall_start(forcing.catalog),
        scale_factor=rainfall_scale_factor,
    )
    soil_manifest = _stage_event_soil_moisture(Path(run_root), hydrology)
    sf.data_catalog.from_dict(
        {
            "event_precip": {
                "uri": str(prepared_precip),
                "data_type": "RasterDataset",
                "driver": {"name": "raster_xarray"},
                "metadata": {"crs": 4326},
            }
        }
    )
    old_root, old_mode = sf.root.path, sf.root.mode
    config_snapshot = _snapshot_sfincs_config(
        sf.config,
        ("tref", "tstart", "tstop", "precipfile", "netamprfile"),
    )
    sf.root.set(Path(run_root), mode="r+")
    try:
        sf.config.set("tref", run_start.strftime("%Y%m%d %H%M%S"))
        sf.config.set("tstart", run_start.strftime("%Y%m%d %H%M%S"))
        sf.config.set("tstop", run_stop.strftime("%Y%m%d %H%M%S"))
        # HydroMT-SFINCS reads configured meteo files before appending new data
        # in r+ mode; clear stale pointers until create/write has produced them.
        sf.config.set("precipfile", None)
        sf.config.set("netamprfile", None)
        sf.precipitation.create(
            precip="event_precip",
            buffer=float(precip_cfg.get("buffer_m", 30000.0)),
            cumulative_input=bool(precip_cfg.get("cumulative_input", True)),
            time_label=str(precip_cfg.get("time_label", "right")),
            aggregate=False,
        )
        sf.precipitation.write()
    finally:
        _restore_sfincs_config(sf.config, config_snapshot)
        sf.root.set(old_root, mode=old_mode)

    manifest = {
        **hydrology,
        **soil_manifest,
        "prepared_precip": str(prepared_precip),
        "netamprfile": "sfincs_netampr.nc",
        "rainfall_scale_factor": rainfall_scale_factor,
        "rainfall_window_alignment": window_alignment,
        "rainfall_start_offset_hours": _nullable_float(forcing.catalog.get("rainfall_start_offset_hours")),
    }
    manifest_path = Path(run_root) / "forcing_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing.update(manifest)
        write_json(manifest_path, existing)
    return manifest


def _staged_run_window(run_root: Path, forcing: EventForcing) -> tuple[pd.Timestamp, pd.Timestamp]:
    manifest_path = Path(run_root) / "forcing_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_start = manifest.get("run_start")
        run_stop = manifest.get("run_stop")
        if run_start and run_stop:
            return pd.Timestamp(run_start), pd.Timestamp(run_stop)
    return pd.Timestamp(forcing.t_start), pd.Timestamp(forcing.t_stop)


def _driver_start_seconds(support_window, driver: str) -> float:
    for window in support_window.driver_windows:
        if window.driver == driver:
            start = window.start_time(support_window.event_reference_time)
            return float((start - support_window.run_start) / pd.Timedelta(seconds=1))
    return 0.0


def _manifest_driver_start_seconds(run_root: Path, driver: str) -> float:
    manifest_path = Path(run_root) / "forcing_manifest.json"
    if not manifest_path.exists():
        return 0.0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_start = manifest.get("run_start")
    reference = manifest.get("event_reference_time")
    if not run_start or not reference:
        return 0.0
    for window in manifest.get("driver_windows", []):
        if window.get("driver") != driver:
            continue
        start = pd.Timestamp(reference) + pd.Timedelta(hours=float(window["start_offset_hours"]))
        return float((start - pd.Timestamp(run_start)) / pd.Timedelta(seconds=1))
    return 0.0


def stage_event_snapwave(run_root, forcing: EventForcing, *, paths=None, config=None) -> dict:
    run_root = Path(run_root)
    catalog = forcing.catalog
    wave_file = _resolve_catalog_path(
        _required_catalog_value(catalog, "snapwave_member_file"),
        paths=paths or {},
    )
    start = pd.Timestamp(_required_catalog_value(catalog, "snapwave_valid_start_time"))
    stop = pd.Timestamp(_required_catalog_value(catalog, "snapwave_valid_end_time"))
    points = _read_snapwave_boundary_points(run_root)
    with xr.open_dataset(wave_file) as ds:
        timeseries = era5_spectra_to_snapwave_timeseries(
            ds,
            points,
            event_window=(start, stop),
        )
    if not timeseries or timeseries["bhs"].empty:
        raise RuntimeError(f"No ERA5 SnapWave records for {start}..{stop} in {wave_file}")

    written = {}
    start_seconds = _manifest_driver_start_seconds(run_root, "wave")
    for key, frame in timeseries.items():
        values = frame.astype(float).copy()
        if key == "bds":
            values = _directional_spread_degrees(values)
        out_path = run_root / f"snapwave.{key}"
        _write_snapwave_timeseries(out_path, values, start, start_seconds=start_seconds)
        written[f"snapwave_{key}file"] = out_path.name
    written["snapwave_member_file"] = str(wave_file)
    written["snapwave_valid_start_time"] = start.strftime("%Y-%m-%dT%H:%M:%S")
    written["snapwave_valid_end_time"] = stop.strftime("%Y-%m-%dT%H:%M:%S")
    return written


def _stage_event_soil_moisture(run_root: Path, hydrology: dict) -> dict:
    summary = hydrology.get("soil_moisture_summary")
    if not summary:
        return {"sefffile": "", "initial_soil_moisture_fraction": None}
    smax_path = run_root / "sfincs.smax"
    if not smax_path.exists():
        raise FileNotFoundError(f"Cannot stage event soil moisture without {smax_path}")
    smax = np.fromfile(smax_path, dtype="<f4")
    fraction = float(summary["mean_soil_moisture"])
    fraction = float(np.clip(fraction / 100.0 if fraction > 1.0 else fraction, 0.0, 1.0))
    seff = (smax * fraction).astype("<f4")
    seff_path = run_root / "sfincs.seff"
    seff.tofile(seff_path)
    return {
        "sefffile": seff_path.name,
        "initial_soil_moisture_fraction": fraction,
    }


def _read_snapwave_boundary_points(run_root: Path) -> gpd.GeoDataFrame:
    bnd_path = run_root / "snapwave.bnd"
    if not bnd_path.exists():
        raise FileNotFoundError(bnd_path)
    rows = []
    for index, line in enumerate(bnd_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append({"name": f"{index:04d}", "x": float(parts[0]), "y": float(parts[1])})
    if not rows:
        raise RuntimeError(f"No SnapWave boundary points found in {bnd_path}")

    epsg = int(parse_sfincs_inp(run_root / "sfincs.inp").get("epsg", 4326))
    if epsg != 4326:
        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
        for row in rows:
            row["x"], row["y"] = transformer.transform(row["x"], row["y"])
    return gpd.GeoDataFrame(
        {"name": [row["name"] for row in rows]},
        geometry=[Point(row["x"], row["y"]) for row in rows],
        crs="EPSG:4326",
    )


def _directional_spread_degrees(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(np.abs(finite))) <= (2 * np.pi + 1e-6):
        return frame.apply(np.rad2deg)
    return frame


def _write_snapwave_timeseries(
    path: Path,
    frame: pd.DataFrame,
    start: pd.Timestamp,
    *,
    start_seconds=0.0,
) -> None:
    index = pd.to_datetime(frame.index)
    seconds = float(start_seconds) + ((index - start) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    values = np.column_stack([seconds, frame.to_numpy(dtype=float)])
    lines = [" ".join(f"{value:.3f}" for value in row) for row in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sfincs_runner(
    model_root,
    *,
    sfincs_bin=None,
    sfincs_image=None,
    allow_native=None,
    config=None,
):
    run_cfg = (config or {}).get("scenario_run", {})
    sfincs_bin_env = str(run_cfg.get("sfincs_bin_env", "SFINCS_BIN"))
    if sfincs_bin is None:
        sfincs_bin = run_cfg.get("sfincs_bin") or os.environ.get(sfincs_bin_env, "")
    sfincs_image = (
        run_cfg.get("sfincs_image")
        or os.environ.get("SFINCS_IMAGE", "deltares/sfincs-cpu:sfincs-v2.3.0-mt-Faber-Release")
        if sfincs_image is None
        else sfincs_image
    )
    allow_native = (
        os.environ.get("SFINCS_ALLOW_NATIVE", "").strip() == "1"
        if allow_native is None
        else bool(allow_native)
    )
    if str(sfincs_bin).strip():
        default_bin = Path(str(sfincs_bin))
        if default_bin.exists():
            return [str(default_bin)]
        return shlex.split(str(sfincs_bin))
    if allow_native:
        native = shutil.which("sfincs")
        if native:
            return [native]
    docker = shutil.which("docker")
    if docker:
        return [
            docker,
            "run",
            "--rm",
            "-v",
            f"{Path(model_root)}:/data",
            "-w",
            "/data",
            sfincs_image,
            "sfincs",
        ]
    raise RuntimeError(f"No SFINCS runner found. Set {sfincs_bin_env} or install docker/sfincs.")


def run_sfincs_model(run_root, *, runner=None, require_map=True, config=None):
    run_root = Path(run_root)
    runner = runner or build_sfincs_runner(run_root, config=config)
    log_path = run_root / "sfincs_log.txt"
    with log_path.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            runner,
            cwd=run_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"SFINCS failed (exit {proc.returncode}). See {log_path}.")
    map_path = run_root / "sfincs_map.nc"
    if require_map and not map_path.exists():
        raise RuntimeError(f"SFINCS completed without producing {map_path.name}.")
    return SfincsRunResult(
        run_root=run_root,
        log_path=log_path,
        map_path=map_path,
        returncode=int(proc.returncode),
    )


def _event_manifest(base_model_root, forcing, *, n_bnd, include_precip, include_waves, support_window):
    catalog = forcing.catalog
    return {
        "event_id": forcing.event_id,
        "t_start": forcing.t_start.strftime("%Y-%m-%d %H:%M:%S"),
        "t_stop": forcing.t_stop.strftime("%Y-%m-%d %H:%M:%S"),
        "event_reference_time": support_window.event_reference_time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_start": support_window.run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "run_stop": support_window.run_stop.strftime("%Y-%m-%d %H:%M:%S"),
        "run_duration_hours": support_window.duration_hours,
        "timing_policy": support_window.timing_policy,
        "driver_windows": [
            {
                "driver": window.driver,
                "start_offset_hours": float(window.start_offset_hours),
                "peak_offset_hours": None
                if window.peak_offset_hours is None
                else float(window.peak_offset_hours),
                "end_offset_hours": float(window.end_offset_hours),
            }
            for window in support_window.driver_windows
        ],
        "boundary_points": int(n_bnd),
        "base_model_root": str(base_model_root),
        "design_scenario": forcing.design_scenario,
        "design_slr_offset_m": forcing.design_slr_offset_m,
        "forcing_variable": forcing.forcing_variable,
        "expected_zsini_m": float(forcing.zsini),
        "expected_bzs_t0_mean_m": float(forcing.h.iloc[0]),
        "expected_bzs_peak_max_m": float(forcing.h.max()),
        "expected_has_precip": bool(include_precip),
        "expected_has_waves": bool(include_waves),
        "rainfall_member_id": _nullable_str(catalog.get("rainfall_member_id")),
        "rainfall_member_file": _nullable_str(catalog.get("rainfall_member_file")),
        "rainfall_member_time": _nullable_str(catalog.get("rainfall_member_time")),
        "soil_moisture_member_id": _nullable_str(catalog.get("soil_moisture_member_id")),
        "soil_moisture_member_file": _nullable_str(catalog.get("soil_moisture_member_file")),
        "soil_moisture_member_time": _nullable_str(catalog.get("soil_moisture_member_time")),
        "snapwave_member_id": _nullable_str(catalog.get("snapwave_member_id")),
        "snapwave_member_file": _nullable_str(catalog.get("snapwave_member_file")),
        "snapwave_pairing_policy": _nullable_str(catalog.get("snapwave_pairing_policy")),
        "probability_weight": _nullable_float(catalog.get("probability_weight")),
        "sample_rp_years": _nullable_float(catalog.get("sample_rp_years")),
        "surge_dataset": forcing.surge_dataset,
    }


def _nullable_str(value):
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _nullable_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _catalog_rainfall_start(catalog):
    if catalog is None:
        return None
    reference = catalog.get("event_reference_time") or catalog.get("coastal_template_peak_time")
    offset = catalog.get("rainfall_start_offset_hours")
    if reference is None or offset is None or pd.isna(reference) or pd.isna(offset):
        return None
    return pd.Timestamp(reference) + pd.Timedelta(hours=float(offset))


def _required_catalog_value(catalog, key):
    value = catalog.get(key)
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise RuntimeError(f"Event Catalog row is missing required hydrology field: {key}")
    return value


def _resolve_catalog_path(value, *, paths):
    path = Path(str(value))
    if path.is_absolute():
        return path
    location_root = Path(paths.get("location_root", "."))
    return location_root / path
