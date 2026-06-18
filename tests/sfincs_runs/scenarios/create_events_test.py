from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sfincs_runs.scenarios import create_events
from sfincs_runs.scenarios.create_events import parse_args
from sfincs_runs.scenarios.joint_handoff import write_joint_catalog_sfincs_handoff


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_create_events_accepts_location_config_for_defaults():
    args = parse_args(["--config", "locations/marshfield/config.yaml"])

    assert suffix(args.design_outputs) == "locations/marshfield/data/event_catalog"
    assert suffix(args.base_dir) == "locations/marshfield/data/sfincs/base_quadtree_snapwave"
    assert suffix(args.scenarios_dir) == "locations/marshfield/data/sfincs/scenarios"


def test_precip_model_builder_clears_debug_env_before_hydromt_import(monkeypatch):
    captured = {}

    class FakeSfincsModel:
        def __init__(self, *, root, mode):
            captured["root"] = root
            captured["mode"] = mode
            captured["debug"] = os.environ.get("DEBUG")

    monkeypatch.setenv("DEBUG", "release")
    monkeypatch.setitem(__import__("sys").modules, "hydromt_sfincs", type("M", (), {"SfincsModel": FakeSfincsModel})())

    model = create_events._build_precip_model("/tmp/base")

    assert isinstance(model, FakeSfincsModel)
    assert captured == {"root": "/tmp/base", "mode": "r+", "debug": None}


def test_build_scenarios_can_stage_full_wave_coupled_event_folders_efficiently(tmp_path, monkeypatch):
    design_root = tmp_path / "event_catalog"
    catalog_dir = design_root / "catalog"
    events_dir = design_root / "events"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    wave_file = tmp_path / "data/sources/era5_waves/era5.nc"
    wave_file.parent.mkdir(parents=True)
    xr.Dataset(
        {"swh": ("valid_time", [1.0, 1.0])},
        coords={"valid_time": pd.to_datetime(["2018-01-01T17:00:00", "2018-01-07T17:00:00"])},
    ).to_netcdf(wave_file)
    pd.DataFrame([{"event_id": "evt_0001"}]).to_csv(catalog_dir / "sampled_peaks.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "coastal_template_peak_time": "2018-01-04 17:00:00",
                "coastal_valid_start_hour": -72,
                "coastal_valid_end_hour": 72,
                "snapwave_member_id": "era5_20180104",
                "snapwave_member_file": "data/sources/era5_waves/era5.nc",
                "snapwave_valid_start_time": "2018-01-01T17:00:00",
                "snapwave_valid_end_time": "2018-01-07T17:00:00",
                "rainfall_member_id": "aorc_20180104",
                "rainfall_member_file": "data/sources/aorc/aorc.nc",
                "rainfall_member_time": "2018-01-04 17:00:00",
                "soil_moisture_member_id": "nwm_20180104",
                "soil_moisture_member_file": "data/sources/nwm/nwm.nc",
                "soil_moisture_member_time": "2018-01-04 17:00:00",
                "probability_weight": 1.0,
            }
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)
    (catalog_dir / "event_catalog_audit.json").write_text(
        json.dumps({"passed": True, "issue_count": 0}),
        encoding="utf-8",
    )
    xr.Dataset(
        data_vars={
            "water_level_total": (
                ("event_id", "relative_hour"),
                [[0.1, 1.0, 0.2]],
            )
        },
        coords={"event_id": ["evt_0001"], "relative_hour": [-72, 0, 72]},
        attrs={"scenario_name": "base", "slr_offset_m": 0.0},
    ).to_netcdf(events_dir / "surge_event_members.nc")

    base = tmp_path / "base_quadtree_snapwave"
    base.mkdir()
    for name, text in {
        "sfincs.inp": "tref = 19990101 000000\nsnapwave = 1\n",
        "sfincs.bnd": "1 2\n",
        "snapwave.bnd": "1 2\n",
        "sfincs.nc": "large static grid\n",
        "sfincs_subgrid.nc": "large static subgrid\n",
        "sfincs.smax": "soil storage\n",
    }.items():
        (base / name).write_text(text, encoding="utf-8")

    def fake_snapwave(run_root, forcing, *, paths=None, config=None):
        for ext in ("bhs", "btp", "bwd", "bds"):
            (Path(run_root) / f"snapwave.{ext}").write_text("0 1\n3600 1\n", encoding="utf-8")
        return {
            "snapwave_bhsfile": "snapwave.bhs",
            "snapwave_btpfile": "snapwave.btp",
            "snapwave_bwdfile": "snapwave.bwd",
            "snapwave_bdsfile": "snapwave.bds",
        }

    def fake_precipitation(sf, run_root, forcing, *, paths, config):
        (Path(run_root) / "aorc_precip_for_sfincs.nc").write_text("prepared precip\n", encoding="utf-8")
        manifest_path = Path(run_root) / "forcing_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        times = pd.date_range(manifest["run_start"], manifest["run_stop"], freq="h")
        xr.Dataset(
            {"Precipitation": (("time", "y", "x"), [[[1.0]]] * len(times))},
            coords={"time": times, "y": [42.0], "x": [-70.0]},
        ).to_netcdf(Path(run_root) / "sfincs_netampr.nc")
        manifest.update(
            {
                "prepared_precip": str(Path(run_root) / "aorc_precip_for_sfincs.nc"),
                "netamprfile": "sfincs_netampr.nc",
                "rainfall_window_alignment": "wettest",
                "initial_soil_moisture_fraction": 0.28,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "prepared_precip": str(Path(run_root) / "aorc_precip_for_sfincs.nc"),
            "netamprfile": "sfincs_netampr.nc",
            "rainfall_window_alignment": "wettest",
            "initial_soil_moisture_fraction": 0.28,
        }

    monkeypatch.setattr(create_events.event_forcing, "stage_event_snapwave", fake_snapwave)
    monkeypatch.setattr(create_events.event_forcing, "stage_event_precipitation", fake_precipitation)

    args = parse_args(
        [
            "--design-outputs",
            str(design_root),
            "--base-dir",
            str(base),
            "--scenarios-dir",
            str(tmp_path / "scenarios"),
            "--limit",
            "1",
            "--force",
            "--include-waves",
            "--include-precip",
            "--zsini-mode",
            "boundary_t0",
        ]
    )

    report = create_events.build_scenarios(args, config={}, runtime_paths={"location_root": tmp_path}, sf_model=object())

    event_dir = tmp_path / "scenarios" / "evt_0001"
    manifest = json.loads((event_dir / "forcing_manifest.json").read_text(encoding="utf-8"))
    assert report["event_id"].tolist() == ["evt_0001"]
    assert (event_dir / "sfincs.bzs").exists()
    assert (event_dir / "snapwave.bhs").exists()
    assert (event_dir / "sfincs_netampr.nc").exists()
    assert manifest["expected_has_waves"] is True
    assert manifest["expected_has_precip"] is True
    assert manifest["forcing_variable"] == "water_level_total"
    assert manifest["expected_zsini_m"] == 0.1
    assert manifest["run_duration_hours"] == 144
    assert (event_dir / "sfincs.nc").stat().st_ino == (base / "sfincs.nc").stat().st_ino

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("resume should skip a complete existing scenario")

    monkeypatch.setattr(create_events.event_forcing, "stage_event_run", fail_if_rebuilt)
    resume_args = parse_args(
        [
            "--design-outputs",
            str(design_root),
            "--base-dir",
            str(base),
            "--scenarios-dir",
            str(tmp_path / "scenarios"),
            "--limit",
            "1",
            "--resume",
            "--include-waves",
            "--include-precip",
            "--zsini-mode",
            "boundary_t0",
        ]
    )
    resumed = create_events.build_scenarios(
        resume_args,
        config={},
        runtime_paths={"location_root": tmp_path},
        sf_model=object(),
    )

    assert resumed["scenario_status"].tolist() == ["skipped_existing"]


def test_build_scenarios_fails_fast_when_snapwave_window_precedes_wave_file(tmp_path):
    design_root = tmp_path / "event_catalog"
    catalog_dir = design_root / "catalog"
    events_dir = design_root / "events"
    wave_dir = tmp_path / "waves"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    wave_dir.mkdir(parents=True)
    wave_file = wave_dir / "era5.nc"
    xr.Dataset(
        {"swh": ("valid_time", [1.0, 1.0])},
        coords={"valid_time": pd.to_datetime(["1979-02-01", "1979-02-02"])},
    ).to_netcdf(wave_file)
    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "study_location": "test",
                "event_family": "surge_synthetic",
                "scenario_name": "base",
                "sampling_weight": 1.0,
                "sample_rp_years": 10.0,
                "sampling_region": "body",
                "snapwave_member_file": str(wave_file),
                "snapwave_valid_start_time": "1979-01-01T00:00:00",
                "snapwave_valid_end_time": "1979-01-04T17:00:00",
            }
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)
    (catalog_dir / "event_catalog_audit.json").write_text(
        json.dumps({"passed": True, "issue_count": 0}),
        encoding="utf-8",
    )
    xr.Dataset(
        data_vars={"water_level_total": (("event_id", "relative_hour"), [[0.1, 1.0, 0.2]])},
        coords={"event_id": ["evt_0001"], "relative_hour": [-72, 0, 72]},
    ).to_netcdf(events_dir / "surge_event_members.nc")
    args = parse_args(
        [
            "--config",
            "locations/marshfield/config.yaml",
            "--design-outputs",
            str(design_root),
            "--scenarios-dir",
            str(tmp_path / "scenarios"),
            "--include-waves",
            "--force",
            "--limit",
            "1",
        ]
    )

    with pytest.raises(RuntimeError, match="evt_0001.*outside"):
        create_events.build_scenarios(args)

    assert not (tmp_path / "scenarios").exists()


def test_joint_catalog_handoff_writes_sfincs_event_contract(tmp_path):
    analog_time = pd.Timestamp("1980-02-10 12:00:00")
    times = pd.date_range(analog_time - pd.Timedelta(hours=72), analog_time + pd.Timedelta(hours=72), freq="h")
    rel = ((times - analog_time) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    components = pd.DataFrame(
        {
            "msl": 0.1,
            "tide": 0.2 * np.sin(2.0 * np.pi * rel / 12.42),
            "ntr": np.exp(-0.5 * (rel / 6.0) ** 2),
        },
        index=times,
    )
    wave_file = tmp_path / "era5.nc"
    xr.Dataset(
        {"swh": ("valid_time", [1.0, 1.0])},
        coords={"valid_time": pd.to_datetime(["1980-02-01", "1980-02-20"])},
    ).to_netcdf(wave_file)
    rainfall_file = tmp_path / "aorc_sst/marshfield/72hr-events/ranked-storms.csv"
    rainfall_file.parent.mkdir(parents=True)
    rainfall_file.write_text("storm_date,mean\n1980-02-09 12:00:00,50\n", encoding="utf-8")
    joint_catalog = pd.DataFrame(
        [
            {
                "event_id": "design_0001",
                "event_family": "copula_joint_compound",
                "scenario_name": "base",
                "sample_rp_years": 25.0,
                "severity_band": "significant",
                "event_origin": "historical_tail",
                "catalog_role": "historical_reference",
                "sampling_scheme": "observed_historical_tail",
                "event_set": "historical_tail_reference",
                "selection_role": "historical_tail_reference",
                "selection_reason": "observed_joint_tail",
                "storm_type": "nor_easter",
                "sampling_region": "tail",
                "sampling_weight": 1.0,
                "probability_weight": 0.01,
                "coastal_water_level": 1.2,
                "coastal_water_level_member_id": "coastal_water_level_19800210T120000",
                "coastal_water_level_member_time": "1980-02-10T12:00:00",
                "coastal_water_level_scale_factor": 1.0,
                "rainfall_source": "aorc_sst",
                "rainfall_member_file": str(rainfall_file),
                "rainfall_member_id": "rainfall_marshfield_72h_rank0001",
                "rainfall_member_time": "1980-02-09T12:00:00",
                "rainfall_scale_factor": 0.8,
                "rainfall_pairing_policy": "copula_joint_field_preserving_analog",
                "rainfall_pairing_seed": 42,
                "forcing_pairing_policy": "copula_joint",
                "event_drivers": "coastal_water_level, rainfall",
            }
        ]
    )
    paths = {
        "location_name": "test",
        "event_members_nc": tmp_path / "event_catalog/events/surge_event_members.nc",
        "event_summary_csv": tmp_path / "event_catalog/events/surge_event_members_summary.csv",
        "event_catalog_csv": tmp_path / "event_catalog/catalog/event_catalog.csv",
        "event_catalog_audit_json": tmp_path / "event_catalog/catalog/event_catalog_audit.json",
        "event_acceptance_json": tmp_path / "event_catalog/events/surge_event_members_acceptance.json",
        "lagtimes_csv": tmp_path / "event_catalog/events/lagtimes.csv",
        "era5_waves_nc": wave_file,
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    config = {
        "project": {"name": "test"},
        "coastal_waves": True,
        "design_events": {"tide_resolving_half_window_hours": 72},
    }

    handoff = write_joint_catalog_sfincs_handoff(joint_catalog, components, config=config, paths=paths)

    written = pd.read_csv(paths["event_catalog_csv"])
    assert handoff["audit"]["passed"] is True
    assert written["event_id"].tolist() == ["design_0001"]
    assert written["event_origin"].tolist() == ["historical_tail"]
    assert written["catalog_role"].tolist() == ["historical_reference"]
    assert written["sampling_scheme"].tolist() == ["observed_historical_tail"]
    assert written["storm_type"].tolist() == ["nor_easter"]
    assert written["snapwave_valid_start_time"].tolist() == ["1980-02-07T12:00:00"]
    assert written["snapwave_valid_end_time"].tolist() == ["1980-02-13T12:00:00"]
    with xr.open_dataset(paths["event_members_nc"]) as ds:
        assert "water_level_total" in ds
        assert ds["water_level_total"].sel(event_id="design_0001").notnull().sum() == 145
    create_events._validate_snapwave_source_windows(written, paths={})


def test_joint_catalog_handoff_replaces_open_existing_member_netcdf(tmp_path):
    analog_time = pd.Timestamp("1980-02-10 12:00:00")
    times = pd.date_range(analog_time - pd.Timedelta(hours=72), analog_time + pd.Timedelta(hours=72), freq="h")
    rel = ((times - analog_time) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    components = pd.DataFrame(
        {
            "msl": 0.1,
            "tide": 0.2 * np.sin(2.0 * np.pi * rel / 12.42),
            "ntr": np.exp(-0.5 * (rel / 6.0) ** 2),
        },
        index=times,
    )
    wave_file = tmp_path / "era5.nc"
    xr.Dataset(
        {"swh": ("valid_time", [1.0, 1.0])},
        coords={"valid_time": pd.to_datetime(["1980-02-01", "1980-02-20"])},
    ).to_netcdf(wave_file)
    event_members_nc = tmp_path / "event_catalog/events/surge_event_members.nc"
    event_members_nc.parent.mkdir(parents=True)
    xr.Dataset(
        {"water_level_total": (("event_id", "relative_hour"), [[999.0]])},
        coords={"event_id": ["stale"], "relative_hour": [0]},
    ).to_netcdf(event_members_nc)
    stale_open = xr.open_dataset(event_members_nc)
    try:
        joint_catalog = pd.DataFrame(
            [
                {
                    "event_id": "design_0001",
                    "event_family": "copula_joint_compound",
                    "scenario_name": "base",
                    "sample_rp_years": 25.0,
                    "severity_band": "significant",
                    "sampling_region": "tail",
                    "sampling_weight": 1.0,
                    "probability_weight": 0.01,
                    "coastal_water_level": 1.2,
                    "coastal_water_level_member_id": "coastal_water_level_19800210T120000",
                    "coastal_water_level_member_time": "1980-02-10T12:00:00",
                    "coastal_water_level_scale_factor": 1.0,
                    "rainfall_source": "aorc_sst",
                    "rainfall_member_file": str(tmp_path / "rainfall_members.csv"),
                    "rainfall_member_id": "rainfall_0001",
                    "rainfall_member_time": "1980-02-09T12:00:00",
                    "rainfall_scale_factor": 0.8,
                    "rainfall_pairing_policy": "copula_joint_field_preserving_analog",
                    "rainfall_pairing_seed": 42,
                    "forcing_pairing_policy": "copula_joint",
                    "event_drivers": "coastal_water_level, rainfall",
                }
            ]
        )
        paths = {
            "location_name": "test",
            "event_members_nc": event_members_nc,
            "event_summary_csv": tmp_path / "event_catalog/events/surge_event_members_summary.csv",
            "event_catalog_csv": tmp_path / "event_catalog/catalog/event_catalog.csv",
            "event_catalog_audit_json": tmp_path / "event_catalog/catalog/event_catalog_audit.json",
            "event_acceptance_json": tmp_path / "event_catalog/events/surge_event_members_acceptance.json",
            "lagtimes_csv": tmp_path / "event_catalog/events/lagtimes.csv",
            "era5_waves_nc": wave_file,
            "scenario": {"name": "base", "slr_offset_m": 0.0},
        }
        config = {
            "project": {"name": "test"},
            "coastal_waves": True,
            "design_events": {"tide_resolving_half_window_hours": 72},
        }

        write_joint_catalog_sfincs_handoff(joint_catalog, components, config=config, paths=paths)
    finally:
        stale_open.close()

    with xr.open_dataset(event_members_nc) as ds:
        assert ds["event_id"].values.tolist() == ["design_0001"]


def test_scenario_is_complete_rejects_stale_all_nan_netampr(tmp_path):
    event_dir = tmp_path / "evt_0002"
    event_dir.mkdir()
    for name in create_events.required_scenario_files(include_precip=True):
        if name == "forcing_manifest.json":
            continue
        (event_dir / name).write_text("placeholder\n", encoding="utf-8")
    (event_dir / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "expected_has_precip": True,
                "run_start": "2018-11-04 15:00:00",
                "run_stop": "2018-11-10 15:00:00",
                "netamprfile": "sfincs_netampr.nc",
            }
        ),
        encoding="utf-8",
    )
    times = pd.date_range("1984-08-25 04:00:00", "1984-08-31 04:00:00", freq="h")
    xr.Dataset(
        {"Precipitation": (("time", "y", "x"), [[[float("nan")]]] * len(times))},
        coords={"time": times, "y": [42.0], "x": [-70.0]},
    ).to_netcdf(event_dir / "sfincs_netampr.nc", mode="w")

    assert not create_events.scenario_is_complete(event_dir, include_precip=True)
