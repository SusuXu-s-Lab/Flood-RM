import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
import yaml

from sfincs_runs.scenarios.event_forcing import _sfincs_subprocess_env
from sfincs_runs.hydrology import validate_physics
from sfincs_runs.scenarios.inland_coupled import (
    audit_inland_coupled_batch_readiness,
    plan_example,
    stage_scenarios,
)
from sfincs_runs.scenarios.inland_initial_conditions import derive_hydrograph_initial_depth


def test_stage_scenarios_writes_wflow_sfincs_manifests(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    base_model = location_root / "data/sfincs/base"
    base_model.mkdir(parents=True)
    for name, text in {
        "sfincs.inp": "tref = 19990101 000000\n",
        "sfincs.bnd": "1 2\n",
        "sfincs.dep": "static\n",
    }.items():
        (base_model / name).write_text(text, encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "usgs_02095000_20180917T120000",
                "event_reference_time": "2018-09-17T12:00:00",
                "event_origin": "historical_tail",
                "catalog_role": "historical_reference",
                "sampling_scheme": "observed_historical_tail",
                "event_set": "historical_tail_reference",
                "selection_role": "historical_tail_reference",
                "selection_reason": "observed_joint_tail",
                "severity_band": "rare",
                "sample_rp_years": 50.0,
                "streamflow_member_id": "02095000_20180917T120000",
                "rainfall_member_id": "rainfall_greensboro_72h_rank0001",
                "soil_moisture_member_id": "nwm_20180916",
                "wflow_event_dir": "data/wflow/events/usgs_02095000_20180917T120000",
                "sfincs_scenario_dir": "data/sfincs/scenarios/usgs_02095000_20180917T120000",
                "probability_weight": 0.01,
            }
        ]
    ).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "forcing_mode": "dual_fluvial_pluvial",
                "source_variable": "river_q",
                "source_standard_name": "river_water__volume_flow_rate",
                "direct_rainfall_enabled": True,
                "events": [
                    {
                        "event_id": "usgs_02095000_20180917T120000",
                        "wflow_event_dir": "data/wflow/events/usgs_02095000_20180917T120000",
                        "discharge_forcing": "data/wflow/events/usgs_02095000_20180917T120000/sfincs_discharge.nc",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    discharge_forcing = location_root / "data/wflow/events/usgs_02095000_20180917T120000/sfincs_discharge.nc"
    discharge_forcing.parent.mkdir(parents=True, exist_ok=True)
    discharge_forcing.write_text("wflow discharge fixture\n", encoding="utf-8")
    config = {
        "paths": {
            "base_model_root": "data/sfincs/base",
            "scenarios_root": "data/sfincs/scenarios",
        },
        "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}},
        "inland_coupling": {
            "forcing_mode": "dual_fluvial_pluvial",
            "direct_rainfall": {"enabled": True},
            "discharge_forcing": {"source": "wflow"},
            "streamflow_reference_time": "dominant_streamgage_network_peak",
        },
        "event_catalog": {
            "forcing_members": {
                "rainfall": "data/sources/aorc_sst/rainfall_members.csv",
                "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
                "soil_moisture": "data/sources/nwm/soil_moisture.csv",
            }
        },
    }

    report = stage_scenarios(
        config,
        {"location_root": location_root},
        catalog_path=catalog_path,
        force=True,
    )

    scenario_root = location_root / "data/sfincs/scenarios/usgs_02095000_20180917T120000"
    manifest = json.loads((scenario_root / "forcing_manifest.json").read_text(encoding="utf-8"))
    assert report["event_id"].tolist() == ["usgs_02095000_20180917T120000"]
    assert (scenario_root / "sfincs.inp").exists()
    assert manifest["forcing_mode"] == "dual_fluvial_pluvial"
    assert manifest["wflow_discharge_forcing"] == "data/wflow/events/usgs_02095000_20180917T120000/sfincs_discharge.nc"
    assert manifest["direct_rainfall_enabled"] is True
    assert manifest["rainfall_member_id"] == "rainfall_greensboro_72h_rank0001"
    assert manifest["event_reference_time"] == "2018-09-17T12:00:00"
    assert manifest["event_origin"] == "historical_tail"
    assert manifest["sampling_scheme"] == "observed_historical_tail"
    assert manifest["selection_reason"] == "observed_joint_tail"
    assert manifest["streamflow_reference_time"] == "dominant_streamgage_network_peak"
    assert (location_root / "data/sfincs/scenarios/scenario_build_report.csv").exists()


def test_derive_hydrograph_initial_depth_uses_bounded_discharge_proxy():
    discharge = pd.DataFrame(
        {
            "inflow_01": [4000.0, 5000.0],
            "inflow_02": [6000.0, 7000.0],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="h"),
    )

    report = derive_hydrograph_initial_depth(
        discharge,
        {
            "inland_coupling": {
                "initial_conditions": {
                    "reference_discharge_m3s": 5000.0,
                    "reference_depth_m": 1.5,
                    "min_depth_m": 0.1,
                    "max_depth_m": 2.5,
                    "mean_window_hours": 0.0,
                }
            }
        },
    )

    assert report["mean_initial_discharge_m3s"] == 5000.0
    assert report["initial_depth_m"] == 1.5
    assert report["depth_method"].startswith("sqrt(")


def test_audit_inland_coupled_batch_readiness_checks_accepted_event_staged_inputs(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    event_root = location_root / "data/wflow/events/evt_001"
    event_root.mkdir(parents=True, exist_ok=True)
    discharge_nc = event_root / "sfincs_discharge.nc"
    xr.Dataset(coords={"time": pd.date_range("2018-09-15T12:00:00", periods=7, freq="D")}).to_netcdf(discharge_nc)
    (event_root / "sfincs_discharge.dynamic_handoff.json").write_text(
        json.dumps(
            {
                "event_id": "evt_001",
                "status": "accepted",
                "discharge_source": "wflow_dynamic",
                "discharge_nc": str(discharge_nc),
                "checks": [
                    {"check": "event_peak", "status": "passed", "message": ""},
                    {"check": "source_ids", "status": "passed", "message": ""},
                    {"check": "zero_rain_peak_fraction", "status": "passed", "message": ""},
                ],
                "metadata": {"streamflow_realization": "wflow_external_river_inflow"},
            }
        ),
        encoding="utf-8",
    )
    scenarios_root = location_root / "data/sfincs/scenarios"
    run_root = scenarios_root / "evt_001/greensboro_rural"
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"event_id": "evt_001", "run_root": "evt_001/greensboro_rural"}]).to_csv(
        scenarios_root / "scenario_catalog.csv",
        index=False,
    )
    catalog_path = location_root / "data/event_catalog/catalog/scenario_catalog.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"event_id": "evt_001", "event_reference_time": "2018-09-17T12:00:00"}]).to_csv(
        catalog_path,
        index=False,
    )
    for name in [
        "sfincs.inp",
        "sfincs.src",
        "sfincs.dis",
        "forcing_manifest.json",
        "sfincs_subgrid.nc",
        "sfincs_netampr.nc",
        "aorc_precip_for_sfincs.nc",
        "sfincs.ini",
        "sfincs.smax",
        "sfincs.seff",
        "sfincs.ks",
    ]:
        (run_root / name).write_text("fixture\n", encoding="utf-8")

    config = {
        "paths": {"scenarios_root": "data/sfincs/scenarios"},
        "wflow": {"events_root": "data/wflow/events"},
        "event_drivers": ["rainfall", "soil_moisture"],
        "inland_coupling": {
            "direct_rainfall": {"enabled": True},
            "initial_conditions": {"enabled": True},
            "infiltration": {"enabled": True},
            "discharge_forcing": {"source": "wflow_dynamic"},
        },
    }

    report = audit_inland_coupled_batch_readiness(
        config,
        {"location_root": location_root},
        catalog_path=catalog_path,
        event_ids=["evt_001"],
    )

    assert set(report["status"]) == {"passed"}

    (run_root / "sfincs.dis").unlink()
    report = audit_inland_coupled_batch_readiness(
        config,
        {"location_root": location_root},
        catalog_path=catalog_path,
        event_ids=["evt_001"],
    )

    failed = report[report["status"].eq("failed")]
    assert "file:sfincs.dis" in failed["check"].tolist()


def test_sfincs_subprocess_env_sets_openmp_threads_from_config(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    env = _sfincs_subprocess_env({"scenario_run": {"threads": 1}})

    assert env["OMP_NUM_THREADS"] == "1"


def test_sfincs_subprocess_env_sanitizes_non_numeric_debug(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")

    env = _sfincs_subprocess_env({})

    assert env["DEBUG"] == "0"


def test_sfincs_subprocess_env_can_use_sfincs_threads_env(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("SFINCS_THREADS", "2")

    env = _sfincs_subprocess_env({})

    assert env["OMP_NUM_THREADS"] == "2"


def test_sfincs_subprocess_env_rejects_nonpositive_threads(monkeypatch):
    monkeypatch.setenv("SFINCS_THREADS", "0")

    try:
        _sfincs_subprocess_env({})
    except ValueError as exc:
        assert "scenario_run.threads" in str(exc)
    else:
        raise AssertionError("Expected nonpositive SFINCS threads to fail")


def test_validate_physics_requires_cn_recovery_and_spatial_roughness(tmp_path):
    model_root = tmp_path / "sfincs_base"
    model_root.mkdir()
    (model_root / "sfincs.inp").write_text(
        "\n".join(
            [
                "sbgfile = sfincs_subgrid.nc",
                "qinf = 0",
                "smaxfile = sfincs.smax",
                "sefffile = sfincs.seff",
                "ksfile = sfincs.ks",
            ]
        ),
        encoding="utf-8",
    )
    for name in ["sfincs.smax", "sfincs.seff", "sfincs.ks"]:
        (model_root / name).write_text("fixture", encoding="utf-8")
    xr.Dataset(
        {
            "uv_navg": (("n", "m"), [[0.04, 0.08], [0.12, 0.04]]),
            "uv_nrep": (("n", "m"), [[0.04, 0.05], [0.06, 0.04]]),
        }
    ).to_netcdf(model_root / "sfincs_subgrid.nc")

    report = validate_physics(
        model_root,
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "inland_coupling": {"infiltration": {"enabled": True, "method": "cn_with_recovery"}},
        },
    )

    assert report["infiltration"]["status"] == "active"
    assert report["roughness"]["spatially_varying"] is True


def test_validate_physics_rejects_missing_cn_recovery_files_for_rainfall(tmp_path):
    model_root = tmp_path / "sfincs_base"
    model_root.mkdir()
    (model_root / "sfincs.inp").write_text("qinf = 0\nsbgfile = sfincs_subgrid.nc\n", encoding="utf-8")
    xr.Dataset({"uv_navg": (("n", "m"), [[0.04, 0.08]])}).to_netcdf(model_root / "sfincs_subgrid.nc")

    with pytest.raises(RuntimeError, match="lacks active native infiltration"):
        validate_physics(
            model_root,
            {
                "event_drivers": ["rainfall"],
                "inland_coupling": {"infiltration": {"enabled": True, "method": "cn_with_recovery"}},
            },
        )


def test_stage_scenarios_fans_shared_events_across_sfincs_domain_set(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    for domain_id in ["greensboro_west", "greensboro_east"]:
        base_model = location_root / f"data/sfincs/domains/{domain_id}/base"
        base_model.mkdir(parents=True, exist_ok=True)
        (base_model / "sfincs.inp").write_text("tref = 19990101 000000\n", encoding="utf-8")
        (base_model / "sfincs.dep").write_text("static\n", encoding="utf-8")
    domain_manifest = location_root / "data/sfincs/domains/domain_set.yaml"
    domain_manifest.parent.mkdir(parents=True, exist_ok=True)
    domain_manifest.write_text(
        yaml.safe_dump(
            {
                "status": "ready",
                "event_catalog_scope": "shared_across_domain_set",
                "domains": [
                    {
                        "sfincs_domain_id": "greensboro_west",
                        "base_model_root": str(location_root / "data/sfincs/domains/greensboro_west/base"),
                        "handoff_source_ids": ["south_buffalo_02095000"],
                    },
                    {
                        "sfincs_domain_id": "greensboro_east",
                        "base_model_root": str(location_root / "data/sfincs/domains/greensboro_east/base"),
                        "handoff_source_ids": ["reedy_fork_02094500"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame([{"event_id": "evt_001", "event_reference_time": "2018-09-17T12:00:00"}]).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "forcing_mode": "dual_fluvial_pluvial",
                "events": [
                    {
                        "event_id": "evt_001",
                        "wflow_event_dir": "data/wflow/events/evt_001",
                        "discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    discharge_forcing = location_root / "data/wflow/events/evt_001/sfincs_discharge.nc"
    discharge_forcing.parent.mkdir(parents=True, exist_ok=True)
    discharge_forcing.write_text("wflow discharge fixture\n", encoding="utf-8")

    report = stage_scenarios(
        {
            "paths": {"scenarios_root": "data/sfincs/scenarios"},
            "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}},
            "sfincs_domain_set": {
                "enabled": True,
                "domain_manifest": "data/sfincs/domains/domain_set.yaml",
            },
            "inland_coupling": {
                "forcing_mode": "dual_fluvial_pluvial",
                "direct_rainfall": {"enabled": True},
            },
        },
        {"location_root": location_root},
        catalog_path=catalog_path,
        force=True,
    )

    assert report["sfincs_domain_id"].tolist() == ["greensboro_west", "greensboro_east"]
    for domain_id in ["greensboro_west", "greensboro_east"]:
        scenario_root = location_root / f"data/sfincs/scenarios/evt_001/{domain_id}"
        manifest = json.loads((scenario_root / "forcing_manifest.json").read_text(encoding="utf-8"))
        assert (scenario_root / "sfincs.inp").exists()
        assert manifest["event_id"] == "evt_001"
        assert manifest["sfincs_domain_id"] == domain_id
        assert manifest["wflow_discharge_forcing"] == "data/wflow/events/evt_001/sfincs_discharge.nc"


def test_stage_scenarios_requires_matching_wflow_handoff_event(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    (location_root / "data/sfincs/base").mkdir(parents=True)
    (location_root / "data/sfincs/base/sfincs.inp").write_text("tref = 19990101 000000\n", encoding="utf-8")
    (location_root / "data/sfincs/base/sfincs.dep").write_text("dep\n", encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame([{"event_id": "evt_missing"}]).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(yaml.safe_dump({"events": []}), encoding="utf-8")

    try:
        stage_scenarios(
            {
                "paths": {"base_model_root": "data/sfincs/base", "scenarios_root": "data/sfincs/scenarios"},
                "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}},
            },
            {"location_root": location_root},
            catalog_path=catalog_path,
            force=True,
        )
    except ValueError as exc:
        assert "missing Wflow handoff events" in str(exc)
    else:
        raise AssertionError("Expected missing handoff event to fail")


def test_stage_scenarios_requires_wflow_discharge_file_before_sfincs_staging(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    base_model = location_root / "data/sfincs/base"
    base_model.mkdir(parents=True)
    (base_model / "sfincs.inp").write_text("tref = 19990101 000000\n", encoding="utf-8")
    (base_model / "sfincs.dep").write_text("dep\n", encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame([{"event_id": "evt_001"}]).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "events": [
                    {
                        "event_id": "evt_001",
                        "wflow_event_dir": "data/wflow/events/evt_001",
                        "discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    try:
        stage_scenarios(
            {
                "paths": {"base_model_root": "data/sfincs/base", "scenarios_root": "data/sfincs/scenarios"},
                "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}},
            },
            {"location_root": location_root},
            catalog_path=catalog_path,
            force=True,
        )
    except FileNotFoundError as exc:
        assert "Wflow discharge forcing is missing" in str(exc)
        assert "data/wflow/events/evt_001/sfincs_discharge.nc" in str(exc)
    else:
        raise AssertionError("Expected SFINCS staging to wait for Wflow discharge forcing")


def test_stage_scenarios_rejects_gitkeep_only_sfincs_base(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    base_model = location_root / "data/sfincs/base"
    base_model.mkdir(parents=True)
    (base_model / ".gitkeep").write_text("", encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame([{"event_id": "evt_001"}]).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "events": [
                    {
                        "event_id": "evt_001",
                        "wflow_event_dir": "data/wflow/events/evt_001",
                        "discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    discharge_forcing = location_root / "data/wflow/events/evt_001/sfincs_discharge.nc"
    discharge_forcing.parent.mkdir(parents=True, exist_ok=True)
    discharge_forcing.write_text("wflow discharge fixture\n", encoding="utf-8")

    try:
        stage_scenarios(
            {
                "paths": {"base_model_root": "data/sfincs/base", "scenarios_root": "data/sfincs/scenarios"},
                "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}},
            },
            {"location_root": location_root},
            catalog_path=catalog_path,
            force=True,
        )
    except FileNotFoundError as exc:
        assert "SFINCS base model is not built" in str(exc)
    else:
        raise AssertionError("Expected .gitkeep-only SFINCS base to fail")


def test_plan_example_selects_catalog_event_and_run_commands(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    (location_root / "data/sfincs/base").mkdir(parents=True)
    (location_root / "data/sfincs/base/sfincs.inp").write_text("tref = 19990101 000000\n", encoding="utf-8")
    (location_root / "data/sfincs/base/sfincs.dep").write_text("dep\n", encoding="utf-8")
    (location_root / "data/wflow/base/reedy_fork").mkdir(parents=True)
    (location_root / "data/wflow/base/reedy_fork/wflow_sbm.toml").write_text("[model]\n", encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "usgs_low_20180101T000000",
                "event_reference_time": "2018-01-01T00:00:00",
                "sample_rp_years": 5.0,
                "streamflow_member_id": "low",
                "rainfall_member_id": "rainfall_greensboro_72h_rank0000",
            },
            {
                "event_id": "usgs_02095000_20180917T120000",
                "event_reference_time": "2018-09-17T12:00:00",
                "sample_rp_years": 250.0,
                "streamflow_member_id": "02095000_20180917T120000",
                "rainfall_member_id": "rainfall_greensboro_72h_rank0001",
            }
        ]
    ).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "source_variable": "river_q",
                "source_standard_name": "river_water__volume_flow_rate",
                "events": [
                    {
                        "event_id": "usgs_low_20180101T000000",
                        "wflow_event_dir": "data/wflow/events/usgs_low_20180101T000000",
                        "discharge_forcing": "data/wflow/events/usgs_low_20180101T000000/sfincs_discharge.nc",
                    },
                    {
                        "event_id": "usgs_02095000_20180917T120000",
                        "wflow_event_dir": "data/wflow/events/usgs_02095000_20180917T120000",
                        "discharge_forcing": "data/wflow/events/usgs_02095000_20180917T120000/sfincs_discharge.nc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "paths": {
            "base_model_root": "data/sfincs/base",
            "scenarios_root": "data/sfincs/scenarios",
            "run_root": "data/sfincs/run_stage",
            "storage_root": "data/sfincs/run_outputs",
        },
        "wflow": {
            "handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"},
            "base_model_root": "data/wflow/base",
            "events_root": "data/wflow/events",
        },
        "scenario_run": {"workers": 1},
    }

    plan = plan_example(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.event_id == "usgs_02095000_20180917T120000"
    assert plan.event_reference_time == "2018-09-17T12:00:00"
    assert plan.wflow_event_dir == "data/wflow/events/usgs_02095000_20180917T120000"
    assert plan.wflow_discharge_forcing == "data/wflow/events/usgs_02095000_20180917T120000/sfincs_discharge.nc"
    assert plan.sfincs_scenario_dir.endswith("data/sfincs/scenarios/usgs_02095000_20180917T120000")
    assert "--event-id usgs_02095000_20180917T120000" in plan.sfincs_dry_run_command
    assert "stage_scenarios" in plan.stage_command


def test_plan_example_points_to_first_domain_scenario_for_domain_set(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    domain_root = location_root / "data/sfincs/domains/greensboro_west/base"
    domain_root.mkdir(parents=True, exist_ok=True)
    (domain_root / "sfincs.inp").write_text("tref = 19990101 000000\n", encoding="utf-8")
    (domain_root / "sfincs.dep").write_text("dep\n", encoding="utf-8")
    (location_root / "data/wflow/base/reedy_fork").mkdir(parents=True)
    (location_root / "data/wflow/base/reedy_fork/wflow_sbm.toml").write_text("[model]\n", encoding="utf-8")
    domain_manifest = location_root / "data/sfincs/domains/domain_set.yaml"
    domain_manifest.parent.mkdir(parents=True, exist_ok=True)
    domain_manifest.write_text(
        yaml.safe_dump(
            {
                "domains": [
                    {
                        "sfincs_domain_id": "greensboro_west",
                        "base_model_root": str(domain_root),
                        "handoff_source_ids": ["south_buffalo_02095000"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"event_id": "evt_001", "event_reference_time": "2018-09-17T12:00:00", "sample_rp_years": 100.0}]
    ).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "events": [
                    {
                        "event_id": "evt_001",
                        "wflow_event_dir": "data/wflow/events/evt_001",
                        "discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = plan_example(
        {
            "paths": {"scenarios_root": "data/sfincs/scenarios"},
            "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}, "base_model_root": "data/wflow/base"},
            "sfincs_domain_set": {"enabled": True, "domain_manifest": "data/sfincs/domains/domain_set.yaml"},
        },
        {"location_root": location_root},
    )

    assert plan.status == "ready"
    assert plan.sfincs_scenario_dir.endswith("data/sfincs/scenarios/evt_001/greensboro_west")
    assert plan.forcing_manifest.endswith("data/sfincs/scenarios/evt_001/greensboro_west/forcing_manifest.json")


def test_plan_example_rejects_gitkeep_only_model_dirs(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    (location_root / "data/sfincs/base").mkdir(parents=True)
    (location_root / "data/sfincs/base/.gitkeep").write_text("", encoding="utf-8")
    (location_root / "data/wflow/base").mkdir(parents=True)
    (location_root / "data/wflow/base/.gitkeep").write_text("", encoding="utf-8")
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    catalog_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"event_id": "evt_001", "event_reference_time": "2018-09-17T12:00:00"}]
    ).to_csv(catalog_path, index=False)
    handoff_path = location_root / "data/wflow/domain_set_handoff.yaml"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(
        yaml.safe_dump(
            {
                "events": [
                    {
                        "event_id": "evt_001",
                        "wflow_event_dir": "data/wflow/events/evt_001",
                        "discharge_forcing": "data/wflow/events/evt_001/sfincs_discharge.nc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = plan_example(
        {
            "paths": {"base_model_root": "data/sfincs/base", "scenarios_root": "data/sfincs/scenarios"},
            "wflow": {"handoff": {"manifest": "data/wflow/domain_set_handoff.yaml"}, "base_model_root": "data/wflow/base"},
        },
        {"location_root": location_root},
    )

    assert plan.status == "missing_inputs"
    assert any("SFINCS base model is not built" in issue for issue in plan.issues)
    assert any("Wflow base model is not built" in issue for issue in plan.issues)
