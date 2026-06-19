from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from power.resilience import BlockInvariantViolation, build_switch_bounded_load_blocks
from power.exports import build_event_window_bundle
from power.exports import export_powermodels_onm
from power.resilience import (
    build_ssap_components,
    physical_lines_only,
    physical_switch_candidate_edges,
)
from power.audit.synthetic_validation import (
    build_audit_summary,
    plot_validation_region_report_card,
    run_operational_validation,
)


def test_build_ssap_components_bridges_transformer_windings_without_switching_them():
    buses = pd.DataFrame(
        [
            {"bus": "src", "feeder_id": "f1", "load_kw": 0.0, "source_count": 1},
            {"bus": "mv", "feeder_id": "f1", "load_kw": 10.0, "source_count": 0},
            {"bus": "lv", "feeder_id": "f1", "load_kw": 5.0, "source_count": 0},
            {"bus": "tail", "feeder_id": "f1", "load_kw": 2.0, "source_count": 0},
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "line_name": "primary",
                "feeder_id": "f1",
                "from_bus": "src",
                "to_bus": "mv",
                "line_class": "line",
                "length": 1.0,
            },
            {
                "line_name": "secondary",
                "feeder_id": "f1",
                "from_bus": "lv",
                "to_bus": "tail",
                "line_class": "line",
                "length": 1.0,
            },
        ]
    )
    transformers = pd.DataFrame(
        [
            {
                "feeder_id": "f1",
                "winding_buses": "mv,lv",
            }
        ]
    )
    sources = pd.DataFrame([{"feeder_id": "f1", "bus": "src"}])

    F, meta = build_ssap_components(
        buses,
        physical_lines_only(lines),
        sources,
        exposure_mode="homogeneous",
        transformers=transformers,
    )

    feeder = F["f1"]
    switchable = physical_switch_candidate_edges(feeder, physical_lines_only(lines))
    assert len(F) == 1
    assert len(meta) == 1
    assert len(feeder.edges) == 3
    assert len(switchable) == 2
    assert {edge.downstream for edge in switchable} == {"mv", "tail"}


def test_build_switch_bounded_load_blocks_returns_notebook_ready_frame_and_report():
    buses = pd.DataFrame(
        [
            {"bus": "src", "feeder_id": "f1", "load_kw": 0.0},
            {"bus": "a", "feeder_id": "f1", "load_kw": 10.0},
            {"bus": "b", "feeder_id": "f1", "load_kw": 20.0},
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "line_name": "l_src_a",
                "from_bus": "src",
                "to_bus": "a",
                "line_class": "line",
                "phases": 3,
            },
            {
                "line_name": "l_a_b",
                "from_bus": "a",
                "to_bus": "b",
                "line_class": "line",
                "phases": 3,
            },
        ]
    )
    loads = pd.DataFrame(
        [
            {"load_name": "load_a", "bus": "a", "kv": 12.47},
            {"load_name": "load_b", "bus": "b", "kv": 12.47},
        ]
    )
    sources = pd.DataFrame([{"bus": "src", "basekv": 12.47}])
    switches = pd.DataFrame(
        [
            {
                "switch_id": "s_l_a_b",
                "from_bus": "a",
                "to_bus": "b",
                "opens_existing_line": True,
                "associated_line_name": "l_a_b",
            }
        ]
    )
    der_inventory = pd.DataFrame(
        columns=["der_id", "bus", "gfm_capable"]
    )

    blocks, report = build_switch_bounded_load_blocks(
        buses=buses,
        lines=lines,
        loads=loads,
        sources=sources,
        switches=switches,
        der_inventory=der_inventory,
        sandbox_id="test_location",
    )

    assert len(blocks) == 2
    assert report["violations"] == []
    assert report["summary"]["block_count"] == 2
    assert set(blocks["schema_version"]) == {"stage_b_switch_bounded_load_blocks.v0.1"}
    assert any("s_l_a_b" in json.loads(value) for value in blocks["bounding_switch_ids_json"])


def test_build_switch_bounded_load_blocks_uses_transformer_bridges_for_invariant_b():
    buses = pd.DataFrame(
        [
            {"bus": "src", "feeder_id": "f1", "load_kw": 0.0},
            {"bus": "mv", "feeder_id": "f1", "load_kw": 0.0},
            {"bus": "lv", "feeder_id": "f1", "load_kw": 0.0},
            {"bus": "load", "feeder_id": "f1", "load_kw": 7.0},
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "line_name": "l_src_mv",
                "from_bus": "src",
                "to_bus": "mv",
                "line_class": "line",
                "phases": 3,
            },
            {
                "line_name": "l_lv_load",
                "from_bus": "lv",
                "to_bus": "load",
                "line_class": "line",
                "phases": 1,
            },
        ]
    )
    loads = pd.DataFrame([{"load_name": "load_1", "bus": "load", "kv": 0.208}])
    sources = pd.DataFrame([{"bus": "src", "basekv": 12.47}])
    switches = pd.DataFrame(
        [
            {
                "switch_id": "s_l_src_mv",
                "from_bus": "src",
                "to_bus": "mv",
                "opens_existing_line": True,
                "associated_line_name": "l_src_mv",
            }
        ]
    )
    transformers = pd.DataFrame(
        [
            {
                "transformer_name": "xf1",
                "winding_buses": "mv,lv",
                "max_kv": 12.47,
                "min_kv": 0.208,
            }
        ]
    )

    with pytest.raises(BlockInvariantViolation, match="invariant_b"):
        build_switch_bounded_load_blocks(
            buses=buses,
            lines=lines,
            loads=loads,
            sources=sources,
            switches=switches,
            sandbox_id="test_location",
        )

    blocks, report = build_switch_bounded_load_blocks(
        buses=buses,
        lines=lines,
        loads=loads,
        sources=sources,
        switches=switches,
        transformers=transformers,
        sandbox_id="test_location",
    )

    assert report["violations"] == []
    assert len(blocks) == 2
    assert sorted(blocks["load_kw"]) == [0.0, 7.0]


def test_build_event_window_bundle_returns_notebook_preview_slices():
    bus = "f1__load-bus"
    load_asset_token = "f1_load_bus"
    load_profiles = pd.DataFrame(
        [
            {
                "load_asset_id": f"test_location:asset:load_buses:{bus}",
                "loadshape_id": "shape_1",
                "feeder_id": "f1",
                "customer_class": "critical_facility",
                "profile_source": "comstock",
                "source_building_type": "SmallOffice",
                "source_geography": "ASHRAE_5A",
                "weather_year": 2018,
                "peak_kw": 50.0,
                "source_provenance": json.dumps({"schedule_overlay": "24x7"}),
            }
        ]
    )
    blocks = pd.DataFrame(
        [
            {
                "block_id": "block_a",
                "buses_json": json.dumps([bus]),
            }
        ]
    )
    load_profiles.loc[0, "load_asset_id"] = (
        f"test_location:asset:load_buses:{load_asset_token}"
    )

    bundle = build_event_window_bundle(
        event_start=datetime(2018, 3, 2, 12, 0),
        horizon_hours=3,
        load_profiles=load_profiles,
        blocks=blocks,
        sandbox_id="test_location",
        uncertainty_band=0.20,
    )

    assert bundle["event_start"] == "2018-03-02T12:00:00+00:00"
    assert bundle["event_end"] == "2018-03-02T15:00:00+00:00"
    assert bundle["load_profile_count"] == 1
    assert bundle["block_count"] == 1
    assert bundle["uncertainty_row_count"] == 3
    assert bundle["nodal_demand"][0]["block_id"] == "block_a"
    assert len(bundle["nodal_demand"][0]["values"]) == 3
    assert bundle["uncertainty_bands"][0]["lower_kw"] == pytest.approx(
        bundle["uncertainty_bands"][0]["nominal_kw"] * 0.8
    )


def test_audit_notebook_api_builds_summary_and_validation_region_plot(tmp_path):
    block_report = {"violations": [], "summary": {"block_count": 2}}
    stage_a1 = {"passed": True, "errors": [], "checks": {}}
    stat_results = {
        "switches_per_feeder": {
            "grade": "good",
            "feeder_count": 2,
            "typical_fraction": 1.0,
            "uncommon_fraction": 0.0,
            "rare_fraction": 0.0,
        }
    }
    op_results = run_operational_validation(
        opendss_root=tmp_path / "missing_opendss",
        registry_dir=tmp_path / "asset_registry",
        output_dir=tmp_path,
    )

    summary = build_audit_summary(
        block_report=block_report,
        stage_a1=stage_a1,
        stat_results=stat_results,
        op_results=op_results,
    )

    assert summary["gate_results"]["block_invariant"] == "pass"
    assert summary["gate_results"]["smart_ds_interface"] == "pass"
    assert summary["gate_results"]["statistical_validation"] == "pass"
    assert summary["gate_results"]["operational_validation"] == "gap"
    assert any("OpenDSS feeder cases" in gap for gap in summary["gaps"])

    report = {
        "statistical_metrics": [
            {
                "metric_id": "switches_per_feeder",
                "grade": "good",
                "feeder_count": 2,
                "min": 4.0,
                "median": 8.0,
                "max": 12.0,
                "validation_target": {
                    "typical": ((0.0, 275.0),),
                    "uncommon": ((275.0, 625.0),),
                },
            }
        ]
    }
    plot_path = plot_validation_region_report_card(
        report,
        tmp_path / "validation_region_report_card.png",
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_export_powermodels_onm_writes_notebook_manifest_from_frames(tmp_path):
    opendss_root = tmp_path / "derived_opendss"
    feeder_dir = opendss_root / "f1"
    feeder_dir.mkdir(parents=True)
    (feeder_dir / "LineCodes.dss").write_text("New LineCode.lc R1=0.1 X1=0.1\n", encoding="utf-8")
    (feeder_dir / "Transformers.dss").write_text("", encoding="utf-8")
    (feeder_dir / "Lines.dss").write_text("New Line.l1 Bus1=src Bus2=a Phases=3 LineCode=lc Length=1 units=m\n", encoding="utf-8")
    (feeder_dir / "Loads.dss").write_text("New Load.load_a Bus1=a Phases=3 kV=12.47 kW=10 kvar=2\n", encoding="utf-8")
    (feeder_dir / "BusCoords.dss").write_text("SetBusXY src -70.0 42.0\nSetBusXY a -70.1 42.1\n", encoding="utf-8")

    asset_registry_dir = tmp_path / "asset_registry"
    asset_registry_dir.mkdir()
    pd.DataFrame(
        [{"source_name": "source", "bus": "f1__src", "basekv": 12.47, "pu": 1.0, "angle": 0.0, "phases": 3}]
    ).to_csv(asset_registry_dir / "sources.csv", index=False)
    pd.DataFrame([{"load_name": "load_a", "bus": "f1__a"}]).to_csv(asset_registry_dir / "loads.csv", index=False)

    augmented_dir = tmp_path / "augmented"
    augmented_dir.mkdir()
    pd.DataFrame([{"asset_id": "asset_load_a", "bus": "f1__a"}]).to_parquet(augmented_dir / "assets.parquet", index=False)

    blocks = pd.DataFrame(
        [
            {
                "block_id": "block_1",
                "buses_json": json.dumps(["f1__src", "f1__a"]),
                "load_kw": 10.0,
                "voltage_source_reachability": "substation_pcc",
            }
        ]
    )
    switches = pd.DataFrame(
        [
            {
                "switch_id": "switch_1",
                "opendss_element": "Line.sw1",
                "from_bus": "f1__src",
                "to_bus": "f1__a",
                "phases": 3,
                "initial_state": "closed",
                "normal_state": "closed",
                "dispatchable": True,
                "status": "enabled",
                "switch_role": "sectionalizing",
                "opens_existing_line": True,
                "associated_line_name": "l1",
                "associated_linecode": "lc",
                "associated_units": "m",
                "associated_length_m": 1.0,
            }
        ]
    )
    der_inventory = pd.DataFrame(
        columns=[
            "der_id",
            "assignment_status",
            "gfm_capable",
            "bus",
            "phases",
            "nominal_voltage_kv",
            "genset_kw",
            "placement_rule",
        ]
    )
    load_profiles = pd.DataFrame(
        columns=[
            "load_asset_id",
            "loadshape_id",
            "profile_source",
            "source_building_type",
            "source_geography",
            "source_provenance",
            "peak_kw",
        ]
    )

    manifest = export_powermodels_onm(
        opendss_root=opendss_root,
        asset_registry_dir=asset_registry_dir,
        blocks=blocks,
        switches=switches,
        der_inventory=der_inventory,
        load_profiles=load_profiles,
        output_dir=tmp_path / "onm_export",
    )

    assert manifest["schema_version"] == "marshfield_powermodels_onm_export.v0.1"
    assert (tmp_path / "onm_export" / "network.dss").exists()
    assert (augmented_dir / "switches.dss").exists()
    assert (augmented_dir / "onm_settings.json").exists()
