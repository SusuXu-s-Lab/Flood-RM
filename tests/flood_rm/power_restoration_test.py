import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from power.audit.onm_readiness import (
    summarize_dynagrid_smoke_readiness,
    summarize_event_bundle_readiness,
)
from power.exports.restoration import (
    AssetStateRow,
    build_onm_events,
    materialize_onm_run_bundle,
)
from power.exports.restoration import _der_generators_text
from power.resilience.der import OfflineReoptSurrogateClient


def test_build_onm_events_emits_powermodelsonm_switch_events():
    event_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        AssetStateRow("storm_a", 3, event_start, "sw1", "available"),
        AssetStateRow("storm_a", 3, event_start + timedelta(hours=2), "sw1", "failed"),
    ]

    result = build_onm_events(
        rows,
        event_id="storm_a",
        mc_draw=3,
        asset_to_dss_element={"sw1": "line.sw_feeder_1"},
        event_start_utc=event_start,
    )

    assert result.skipped_asset_ids == []
    assert result.events == [
        {
            "timestep": 3,
            "event_type": "switch",
            "affected_asset": "line.sw_feeder_1",
            "event_data": {
                "dispatchable": "NO",
                "state": "OPEN",
                "status": "ENABLED",
            },
        }
    ]


def test_der_generators_text_marks_reopt_sized_gensets_grid_forming(tmp_path):
    smart = tmp_path / "smart"
    smart.mkdir()
    pd.DataFrame(
        [
            {
                "der_id": "marshfield:der:facility-1",
                "feeder_id": "feeder_a",
                "assignment_status": "assigned",
                "gfm_capable": True,
                "phases": "3",
                "nominal_voltage_kv": 0.208,
                "genset_kw": 125.0,
                "placement_rule": "reopt_resilience_sizing",
                "bus": "feeder_a__bus1",
            }
        ]
    ).to_parquet(smart / "der_inventory.parquet", index=False)
    pd.DataFrame(
        [{"load_asset_id": "marshfield:asset:loads:bus1", "peak_kw": 100.0}]
    ).to_parquet(smart / "load_profile_assignments.parquet", index=False)

    text, counts, dss_settings = _der_generators_text(
        smart_ds_compat_dir=smart,
        feeder_ids={"feeder_a"},
    )

    assert "New Generator.marshfield_der_facility_1" in text
    assert counts == {"provisional": 0, "reopt_sized": 1}
    assert dss_settings == {
        "Generator.marshfield_der_facility_1": {"inverter": "GRID_FORMING"}
    }


def test_offline_reopt_surrogate_sizes_outage_critical_load_with_reserve():
    client = OfflineReoptSurrogateClient(reserve_margin=0.10, capacity_step_kw=5.0)

    result = client(
        {
            "ElectricLoad": {
                "loads_kw": [10.0, 20.0, 30.0],
                "critical_load_fraction": 0.5,
            },
            "ElectricUtility": {
                "outage_start_time_step": 2,
                "outage_end_time_step": 3,
            },
        }
    )

    assert result["status"] == "optimal_offline_surrogate"
    assert result["outputs"]["Generator"]["size_kw"] == 20.0
    assert result["outputs"]["Outages"]["critical_loads_met"] is True
    assert result["reopt_version"] == "offline_reopt_surrogate.v0.1"


def test_materialize_onm_run_bundle_writes_event_window_files(tmp_path):
    export_dir = tmp_path / "onm_export"
    smart = tmp_path / "smart"
    export_dir.mkdir()
    smart.mkdir()
    (export_dir / "network.dss").write_text("Clear\n", encoding="utf-8")
    (export_dir / "settings.json").write_text("{}\n", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "load_asset_id": "marshfield:asset:loads:bus1",
                "loadshape_id": "shape1",
                "profile_source": "synthetic",
                "source_building_type": "hospital",
                "source_geography": "MA",
                "source_provenance": json.dumps({"schedule_overlay": "24x7"}),
                "weather_year": 2026,
                "peak_kw": 50.0,
                "feeder_id": "feeder_a",
                "customer_class": "critical",
            }
        ]
    ).to_parquet(smart / "load_profile_assignments.parquet", index=False)
    pd.DataFrame(
        [
            {
                "block_id": "block_a",
                "buses_json": json.dumps(["bus1"]),
            }
        ]
    ).to_parquet(smart / "switch_bounded_load_blocks.parquet", index=False)

    event_start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    bundle = materialize_onm_run_bundle(
        export_dir=export_dir,
        smart_ds_compat_dir=smart,
        event_id="storm_a",
        mc_draw=0,
        event_start=event_start,
        horizon_hours=4,
        asset_states=[
            AssetStateRow("storm_a", 0, event_start + timedelta(hours=1), "sw1", "failed")
        ],
        asset_to_dss_element={"sw1": "line.sw1"},
    )

    runtime_args = json.loads(bundle.runtime_args_path.read_text(encoding="utf-8"))
    nominal = json.loads(bundle.nominal_load_window_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(bundle.load_uncertainty_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.run_manifest_path.read_text(encoding="utf-8"))

    assert runtime_args["events"] == str(bundle.events_path)
    assert nominal["timestep_count"] == 4
    assert nominal["units"] == "kW"
    assert len(nominal["loads"][0]["values"]) == 4
    assert len(uncertainty["bounds"]) == 4
    assert manifest["event_count"] == 1
    assert summarize_event_bundle_readiness(export_dir)["passed"] is True


def test_dynagrid_readiness_requires_voltages_and_der_setpoints():
    assert summarize_dynagrid_smoke_readiness(
        {
            "status": "ok",
            "tracked_voltage_count": 3,
            "der_setpoint_count": 2,
            "time_periods": 1,
            "time_steps": 3,
        }
    )["passed"] is True
    assert summarize_dynagrid_smoke_readiness({"status": "missing"})["passed"] is False
