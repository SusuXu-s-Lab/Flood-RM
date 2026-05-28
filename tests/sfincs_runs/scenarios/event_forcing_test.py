from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sfincs_runs.scenarios.event_forcing import (
    EventForcing,
    build_single_use_event,
    load_event_forcing,
    resolve_event_hydrology_inputs,
    run_sfincs_model,
    stage_event_snapwave,
    stage_event_precipitation,
    stage_event_run,
)
from sfincs_runs.scenarios.io import parse_sfincs_inp
from sfincs_runs.scenarios.timing import MissingTimingDescriptorsError


def test_load_event_forcing_uses_full_catalog_row_and_surge_member(tmp_path):
    root = tmp_path / "event_catalog"
    catalog_dir = root / "catalog"
    events_dir = root / "events"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "rainfall_member_id": "aorc_19910304",
                "soil_moisture_member_id": "nldas_19910304",
                "snapwave_pairing_policy": "coastal_template",
                "probability_weight": 0.25,
            },
            {
                "event_id": "evt_0002",
                "rainfall_member_id": "aorc_20180302",
                "soil_moisture_member_id": "nldas_20180302",
                "snapwave_pairing_policy": "coastal_template",
                "probability_weight": 0.75,
            },
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)

    ds = xr.Dataset(
        data_vars={
            "surge_absolute": (
                ("event_id", "relative_hour"),
                [[0.4, 0.7, 0.5], [1.1, 1.3, float("nan")]],
            ),
            "water_level_total": (
                ("event_id", "relative_hour"),
                [[0.2, 0.4, 0.1], [2.1, 2.4, float("nan")]],
            )
        },
        coords={
            "event_id": ["evt_0001", "evt_0002"],
            "relative_hour": [0, 1, 2],
        },
        attrs={"scenario_name": "base", "slr_offset_m": 0.0},
    )
    ds.to_netcdf(events_dir / "surge_event_members.nc")

    forcing = load_event_forcing(
        root,
        event_id="evt_0002",
        tref="2000-01-01 06:00:00",
        zsini_mode="boundary_t0",
    )

    assert forcing.event_id == "evt_0002"
    assert forcing.h.tolist() == [2.1, 2.4]
    assert forcing.t_start == pd.Timestamp("2000-01-01 06:00:00")
    assert forcing.t_stop == pd.Timestamp("2000-01-01 07:00:00")
    assert forcing.zsini == 2.1
    assert forcing.catalog["rainfall_member_id"] == "aorc_20180302"
    assert forcing.catalog["soil_moisture_member_id"] == "nldas_20180302"
    assert forcing.design_scenario == "base"
    assert Path(forcing.surge_dataset).name == "surge_event_members.nc"


def test_build_single_use_event_uses_plan_and_catalog_forcing(tmp_path):
    root = tmp_path / "event_catalog"
    catalog_dir = root / "catalog"
    events_dir = root / "events"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    pd.DataFrame([{"event_id": "evt_0001"}]).to_csv(
        catalog_dir / "event_catalog.csv", index=False
    )
    xr.Dataset(
        data_vars={"surge_absolute": (("event_id", "relative_hour"), [[0.2]])},
        coords={"event_id": ["evt_0001"], "relative_hour": [0]},
    ).to_netcdf(events_dir / "surge_event_members.nc")
    paths = {
        "location_name": "marshfield",
        "outputs_root": tmp_path / "sfincs_outputs",
        "design_outputs_root": root,
        "base_model_root": tmp_path / "base",
        "event_catalog_csv": catalog_dir / "event_catalog.csv",
    }

    event = build_single_use_event(
        {},
        paths,
        event_id="evt_0001",
        base_model_root=tmp_path / "base_quadtree_snapwave",
    )

    assert event.plan.event_id == "evt_0001"
    assert event.base_model_root == tmp_path / "base_quadtree_snapwave"
    assert event.forcing.h.tolist() == [0.2]


def test_stage_event_run_writes_boundary_forcing_and_manifest(tmp_path):
    root = tmp_path / "event_catalog"
    catalog_dir = root / "catalog"
    events_dir = root / "events"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "rainfall_member_id": "aorc_19910304",
                "probability_weight": 0.5,
            }
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)
    xr.Dataset(
        data_vars={
            "surge_absolute": (
                ("event_id", "relative_hour"),
                [[0.2, 0.4]],
            )
        },
        coords={"event_id": ["evt_0001"], "relative_hour": [0, 1]},
    ).to_netcdf(events_dir / "surge_event_members.nc")

    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "sfincs.bnd").write_text("1 2\n3 4\n", encoding="utf-8")
    (base_model / "sfincs.dep").write_text("static\n", encoding="utf-8")
    (base_model / "sfincs.inp").write_text(
        "tref                 = 19990101 000000\n"
        "tstart               = 19990101 000000\n"
        "tstop                = 19990102 000000\n"
        "precipfile           = old.dat\n"
        "srcfile              = old.src\n",
        encoding="utf-8",
    )

    forcing = load_event_forcing(root, event_id="evt_0001", tref="2000-01-01 00:00:00")
    staged = stage_event_run(
        base_model,
        tmp_path / "stage",
        forcing,
        force=True,
        include_waves=False,
        include_precip=False,
    )

    inp_text = (staged.run_root / "sfincs.inp").read_text(encoding="utf-8")
    inp = parse_sfincs_inp(staged.run_root / "sfincs.inp")
    bzs = (staged.run_root / "sfincs.bzs").read_text(encoding="utf-8").splitlines()
    manifest = pd.read_json(staged.run_root / "forcing_manifest.json", typ="series")

    assert staged.run_root == tmp_path / "stage" / "evt_0001"
    assert (staged.run_root / "sfincs.dep").exists()
    assert inp["tref"] == "20000101 000000"
    assert inp["tstop"] == "20000101 010000"
    assert inp["bzsfile"] == "sfincs.bzs"
    assert "precipfile" not in inp_text
    assert "srcfile" not in inp_text
    assert bzs == ["     0.0   0.200   0.200", "  3600.0   0.400   0.400"]
    assert manifest["event_id"] == "evt_0001"
    assert bool(manifest["expected_has_precip"]) is False
    assert bool(manifest["expected_has_waves"]) is False


def test_stage_event_run_does_not_mutate_base_model_template(tmp_path):
    root = tmp_path / "event_catalog"
    catalog_dir = root / "catalog"
    events_dir = root / "events"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "event_reference_time": "2000-01-02 00:00:00",
                "coastal_start_offset_hours": -6,
                "coastal_end_offset_hours": 18,
            }
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)
    xr.Dataset(
        data_vars={
            "surge_absolute": (
                ("event_id", "relative_hour"),
                [[0.2, 0.4]],
            )
        },
        coords={"event_id": ["evt_0001"], "relative_hour": [0, 1]},
    ).to_netcdf(events_dir / "surge_event_members.nc")

    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "sfincs.bnd").write_text("1 2\n", encoding="utf-8")
    (base_model / "sfincs.inp").write_text(
        "tref                 = 19990101 000000\n"
        "tstart               = 19990101 000000\n"
        "tstop                = 19990102 000000\n",
        encoding="utf-8",
    )
    base_inp_before = (base_model / "sfincs.inp").read_text(encoding="utf-8")

    forcing = load_event_forcing(root, event_id="evt_0001", tref="2000-01-01 00:00:00")
    staged = stage_event_run(
        base_model,
        tmp_path / "stage",
        forcing,
        force=True,
        include_waves=False,
        include_precip=False,
    )

    assert (base_model / "sfincs.inp").read_text(encoding="utf-8") == base_inp_before
    assert parse_sfincs_inp(staged.run_root / "sfincs.inp")["tref"] == "20000101 180000"


def test_stage_event_run_does_not_mutate_base_snapwave_placeholders(tmp_path):
    wave_nc = tmp_path / "era5_waves.nc"
    times = pd.date_range("2018-01-04 11:00", periods=2, freq="h")
    swh = np.array([[[1.0]], [[2.0]]])
    xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "pp1d": (("valid_time", "latitude", "longitude"), np.full_like(swh, 8.0)),
            "mwd": (("valid_time", "latitude", "longitude"), np.full_like(swh, 90.0)),
            "wdw": (("valid_time", "latitude", "longitude"), np.full_like(swh, np.pi / 6)),
        },
        coords={"valid_time": times, "latitude": [42.0], "longitude": [-70.0]},
    ).to_netcdf(wave_nc)

    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "sfincs.bnd").write_text("1 2\n", encoding="utf-8")
    (base_model / "sfincs.inp").write_text("epsg = 4326\n", encoding="utf-8")
    (base_model / "snapwave.bnd").write_text("-70.0 42.0\n", encoding="utf-8")
    for suffix, value in {"bhs": "1.000", "btp": "10.000", "bwd": "270.000", "bds": "20.000"}.items():
        (base_model / f"snapwave.{suffix}").write_text(
            f"0.000 {value}\n3600.000 {value}\n",
            encoding="utf-8",
        )
    base_snapwave_before = {
        path.name: path.read_text(encoding="utf-8")
        for path in base_model.glob("snapwave.*")
    }

    forcing = EventForcing(
        event_id="evt_0001",
        catalog={
            "event_reference_time": "2000-01-02 00:00:00",
            "coastal_start_offset_hours": -6,
            "coastal_end_offset_hours": 18,
            "snapwave_member_file": str(wave_nc),
            "snapwave_valid_start_time": "2018-01-04T11:00:00",
            "snapwave_valid_end_time": "2018-01-04T12:00:00",
        },
        h=pd.Series([0.0, 1.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 01:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    staged = stage_event_run(
        base_model,
        tmp_path / "stage",
        forcing,
        force=True,
        include_waves=True,
        include_precip=False,
    )

    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in base_model.glob("snapwave.*")
    } == base_snapwave_before
    assert (staged.run_root / "snapwave.bhs").read_text(encoding="utf-8") != base_snapwave_before["snapwave.bhs"]


def test_stage_event_run_uses_forcing_support_window_descriptors_in_manifest(tmp_path):
    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "sfincs.bnd").write_text("1 2\n", encoding="utf-8")
    (base_model / "sfincs.inp").write_text(
        "tref                 = 19990101 000000\n"
        "tstart               = 19990101 000000\n"
        "tstop                = 19990101 010000\n",
        encoding="utf-8",
    )
    forcing = EventForcing(
        event_id="evt_0001",
        catalog={
            "event_reference_time": "2000-01-02 00:00:00",
            "coastal_start_offset_hours": -6,
            "coastal_peak_offset_hours": 0,
            "coastal_end_offset_hours": 18,
            "rainfall_start_offset_hours": -24,
            "rainfall_peak_offset_hours": -12,
            "rainfall_end_offset_hours": 48,
        },
        h=pd.Series([0.0, 1.0, 0.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 02:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    staged = stage_event_run(
        base_model,
        tmp_path / "stage",
        forcing,
        force=True,
        timing_config={"spinup_hours": 6, "drain_down_hours": 12},
    )

    inp = parse_sfincs_inp(staged.run_root / "sfincs.inp")
    bzs = (staged.run_root / "sfincs.bzs").read_text(encoding="utf-8").splitlines()
    manifest = pd.read_json(staged.run_root / "forcing_manifest.json", typ="series")

    assert inp["tstart"] == "19991231 180000"
    assert inp["tstop"] == "20000104 120000"
    assert bzs[0].startswith(" 86400.0")
    assert manifest["timing_policy"] == "descriptors"
    assert manifest["event_reference_time"] == "2000-01-02 00:00:00"
    assert manifest["run_duration_hours"] == 90


def test_stage_event_run_rejects_legacy_timing_when_inference_is_disabled(tmp_path):
    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "sfincs.bnd").write_text("1 2\n", encoding="utf-8")
    (base_model / "sfincs.inp").write_text("tstart = 19990101 000000\n", encoding="utf-8")
    forcing = EventForcing(
        event_id="evt_0001",
        catalog={"event_id": "evt_0001"},
        h=pd.Series([0.0, 1.0, 0.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 02:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    with pytest.raises(MissingTimingDescriptorsError):
        stage_event_run(
            base_model,
            tmp_path / "stage",
            forcing,
            force=True,
            timing_config={"allow_legacy_inference": False},
        )


def test_run_sfincs_model_uses_configured_sfincs_binary(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    fake_sfincs = tmp_path / "fake_sfincs"
    fake_sfincs.write_text(
        "#!/usr/bin/env bash\n"
        "echo configured-runner\n"
        "touch sfincs_map.nc\n",
        encoding="utf-8",
    )
    fake_sfincs.chmod(0o755)

    result = run_sfincs_model(
        run_root,
        config={"scenario_run": {"sfincs_bin": str(fake_sfincs)}},
    )

    assert result.returncode == 0
    assert result.map_path == run_root / "sfincs_map.nc"
    assert "configured-runner" in result.log_path.read_text(encoding="utf-8")


def test_resolve_event_hydrology_inputs_uses_catalog_pairing_metadata(tmp_path):
    root = tmp_path / "event_catalog"
    catalog_dir = root / "catalog"
    events_dir = root / "events"
    rainfall_dir = tmp_path / "rainfall"
    event_windows = rainfall_dir / "event_windows"
    catalog_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    event_windows.mkdir(parents=True)
    rainfall_member_file = rainfall_dir / "ranked-storms.csv"
    rainfall_member_file.write_text("rank,start\n", encoding="utf-8")
    rainfall_window = event_windows / "rainfall_marshfield_72h_rank0027_20090911T00.nc"
    rainfall_window.touch()
    soil_csv = tmp_path / "soil.csv"
    pd.DataFrame(
        {
            "time": ["2009-09-10 00:00", "2009-09-11 00:00"],
            "SOIL_M": [0.2, 0.4],
        }
    ).to_csv(soil_csv, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "evt_0001",
                "rainfall_member_file": str(rainfall_member_file),
                "rainfall_member_id": "rainfall_marshfield_72h_rank0027",
                "rainfall_member_time": "2009-09-11T00:00:00",
                "soil_moisture_member_file": str(soil_csv),
                "soil_moisture_member_time": "2009-09-11T00:00:00",
            }
        ]
    ).to_csv(catalog_dir / "event_catalog.csv", index=False)
    xr.Dataset(
        data_vars={"surge_absolute": (("event_id", "relative_hour"), [[0.2]])},
        coords={"event_id": ["evt_0001"], "relative_hour": [0]},
    ).to_netcdf(events_dir / "surge_event_members.nc")

    forcing = load_event_forcing(root, event_id="evt_0001")
    hydrology = resolve_event_hydrology_inputs(
        forcing,
        paths={"location_root": tmp_path},
        config={"coastal_wave_coupling": {"hydrology": {"soil_moisture": {"lookback_hours": 24}}}},
    )

    assert hydrology["rainfall_source_nc"] == str(rainfall_window)
    assert hydrology["rainfall_storm_start"] == "2009-09-11 00:00:00"
    assert hydrology["soil_moisture_summary"]["mean_soil_moisture"] == pytest.approx(0.3)


def test_stage_event_precipitation_restages_seff_from_event_soil_moisture(tmp_path):
    run_root = tmp_path / "run"
    event_windows = tmp_path / "rainfall" / "event_windows"
    run_root.mkdir()
    event_windows.mkdir(parents=True)
    rainfall_window = event_windows / "rainfall_marshfield_72h_rank0027_20090911T00.nc"
    xr.DataArray(
        [[[1.0]], [[2.0]]],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [pd.Timestamp("2009-09-11 00:00"), pd.Timestamp("2009-09-11 01:00")],
            "latitude": [42.0],
            "longitude": [-71.0],
        },
        name="APCP_surface",
    ).to_dataset().to_netcdf(rainfall_window)
    rainfall_member_file = tmp_path / "rainfall" / "ranked-storms.csv"
    rainfall_member_file.write_text("rank,start\n", encoding="utf-8")
    soil_csv = tmp_path / "soil.csv"
    pd.DataFrame(
        {
            "time": ["2009-09-10 00:00", "2009-09-11 00:00"],
            "SOILSAT_TOP": [0.2, 0.4],
        }
    ).to_csv(soil_csv, index=False)

    smax = np.array([0.1, 0.2, 0.0], dtype="<f4")
    smax.tofile(run_root / "sfincs.smax")
    np.full_like(smax, 0.5).tofile(run_root / "sfincs.seff")
    (run_root / "forcing_manifest.json").write_text(
        '{\n'
        '  "run_start": "2000-01-01 00:00:00",\n'
        '  "run_stop": "2000-01-03 00:00:00",\n'
        '  "timing_policy": "descriptors"\n'
        '}\n',
        encoding="utf-8",
    )

    forcing = EventForcing(
        event_id="evt_0001",
        catalog={
            "rainfall_member_file": str(rainfall_member_file),
            "rainfall_member_id": "rainfall_marshfield_72h_rank0027",
            "rainfall_member_time": "2009-09-11T00:00:00",
            "soil_moisture_member_file": str(soil_csv),
            "soil_moisture_member_time": "2009-09-11T00:00:00",
        },
        h=pd.Series([0.0, 1.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 01:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    class FakeRoot:
        def __init__(self):
            self.path = tmp_path / "base"
            self.mode = "r"

        def set(self, path, mode):
            self.path = Path(path)
            self.mode = mode

    class FakeConfig:
        def __init__(self):
            self.values = {
                "tref": "19990101 000000",
                "tstart": "19990101 000000",
                "tstop": "19990102 000000",
                "precipfile": "base_precip.nc",
                "netamprfile": "base_netampr.nc",
            }

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    class FakeSf:
        def __init__(self):
            self.root = FakeRoot()
            self.config = FakeConfig()

        class data_catalog:
            @staticmethod
            def from_dict(_):
                pass

        class precipitation:
            @staticmethod
            def create(**_):
                pass

            @staticmethod
            def write():
                pass

    sf = FakeSf()
    config_before = dict(sf.config.values)
    manifest = stage_event_precipitation(
        sf,
        run_root,
        forcing,
        paths={"location_root": tmp_path},
        config={
            "coastal_wave_coupling": {
                "hydrology": {
                    "precipitation": {"window_alignment": "start"},
                    "soil_moisture": {"lookback_hours": 24},
                }
            }
        },
    )

    seff = np.fromfile(run_root / "sfincs.seff", dtype="<f4")
    with xr.open_dataset(manifest["prepared_precip"]) as ds:
        precip_times = pd.to_datetime(ds["time"].values)

    assert seff.tolist() == pytest.approx([0.03, 0.06, 0.0])
    assert precip_times[0] == pd.Timestamp("2000-01-01 00:00:00")
    assert precip_times[-1] == pd.Timestamp("2000-01-03 00:00:00")
    assert manifest["rainfall_window_alignment"] == "start"
    assert manifest["soil_moisture_summary"]["mean_soil_moisture"] == pytest.approx(0.3)
    assert sf.config.values == config_before
    assert sf.root.path == tmp_path / "base"
    assert sf.root.mode == "r"


def test_stage_event_snapwave_writes_era5_boundary_timeseries(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "sfincs.inp").write_text("epsg = 4326\n", encoding="utf-8")
    (run_root / "snapwave.bnd").write_text("-70.0 42.0\n-70.5 42.5\n", encoding="utf-8")
    wave_nc = tmp_path / "era5_waves.nc"
    times = pd.date_range("2018-01-04 11:00", periods=3, freq="h")
    swh = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "pp1d": (("valid_time", "latitude", "longitude"), np.full_like(swh, 8.0)),
            "mwd": (("valid_time", "latitude", "longitude"), np.full_like(swh, 90.0)),
            "wdw": (("valid_time", "latitude", "longitude"), np.full_like(swh, np.pi / 6)),
        },
        coords={"valid_time": times, "latitude": [42.0, 42.5], "longitude": [-70.5, -70.0]},
    ).to_netcdf(wave_nc)

    forcing = EventForcing(
        event_id="evt_0001",
        catalog={
            "snapwave_member_file": str(wave_nc),
            "snapwave_valid_start_time": "2018-01-04T11:00:00",
            "snapwave_valid_end_time": "2018-01-04T13:00:00",
        },
        h=pd.Series([0.0, 1.0, 0.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 02:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    manifest = stage_event_snapwave(run_root, forcing, paths={"location_root": tmp_path}, config={})

    bhs = pd.read_csv(run_root / "snapwave.bhs", sep=r"\s+", header=None)
    bds = pd.read_csv(run_root / "snapwave.bds", sep=r"\s+", header=None)
    assert bhs.iloc[:, 0].tolist() == [0.0, 3600.0, 7200.0]
    assert bhs.iloc[:, 1].tolist() == pytest.approx([2.0, 2.5, 3.0])
    assert bhs.iloc[:, 2].tolist() == pytest.approx([3.0, 3.5, 4.0])
    assert bds.iloc[0, 1] == pytest.approx(30.0)
    assert manifest["snapwave_bhsfile"] == "snapwave.bhs"


def test_stage_event_snapwave_offsets_times_to_staged_support_window(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "sfincs.inp").write_text("epsg = 4326\n", encoding="utf-8")
    (run_root / "snapwave.bnd").write_text("-70.0 42.0\n", encoding="utf-8")
    (run_root / "forcing_manifest.json").write_text(
        '{\n'
        '  "run_start": "2000-01-01 00:00:00",\n'
        '  "event_reference_time": "2000-01-02 00:00:00",\n'
        '  "driver_windows": [\n'
        '    {"driver": "wave", "start_offset_hours": -6, "peak_offset_hours": 0, "end_offset_hours": 18}\n'
        '  ]\n'
        '}\n',
        encoding="utf-8",
    )
    wave_nc = tmp_path / "era5_waves.nc"
    times = pd.date_range("2018-01-04 11:00", periods=2, freq="h")
    swh = np.array([[[1.0]], [[2.0]]])
    xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "pp1d": (("valid_time", "latitude", "longitude"), np.full_like(swh, 8.0)),
            "mwd": (("valid_time", "latitude", "longitude"), np.full_like(swh, 90.0)),
            "wdw": (("valid_time", "latitude", "longitude"), np.full_like(swh, np.pi / 6)),
        },
        coords={"valid_time": times, "latitude": [42.0], "longitude": [-70.0]},
    ).to_netcdf(wave_nc)
    forcing = EventForcing(
        event_id="evt_0001",
        catalog={
            "snapwave_member_file": str(wave_nc),
            "snapwave_valid_start_time": "2018-01-04T11:00:00",
            "snapwave_valid_end_time": "2018-01-04T12:00:00",
        },
        h=pd.Series([0.0, 1.0]),
        forcing_variable="surge_absolute",
        t_start=pd.Timestamp("2000-01-01 00:00"),
        t_stop=pd.Timestamp("2000-01-01 01:00"),
        zsini=0.0,
        design_scenario="base",
        design_slr_offset_m=0.0,
        surge_dataset="surge.nc",
    )

    stage_event_snapwave(run_root, forcing, paths={"location_root": tmp_path}, config={})

    bhs = pd.read_csv(run_root / "snapwave.bhs", sep=r"\s+", header=None)
    assert bhs.iloc[:, 0].tolist() == [64800.0, 68400.0]
