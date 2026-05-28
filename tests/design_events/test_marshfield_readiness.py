import json

import pandas as pd

from design_events.readiness import (
    check_acquisition_dry_run,
    check_aorc_sst_collection,
    check_rainfall_catalog_smoke,
    check_wave_forcing_smoke,
    source_inventory_frame,
    write_data_acquisition_readiness,
)


def _write_aorc_sst_catalog(root, source_artifacts, *, smoke=False):
    collection = root / "marshfield" / "72hr-events"
    collection.mkdir(parents=True)
    source_artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "storm_date": ["2020-01-01T00"],
            "min": [1.0],
            "mean": [3.0],
            "max": [5.0],
            "por_rank": [1],
            "annual_rank": [1],
        }
    ).to_csv(collection / "ranked-storms.csv", index=False)
    pd.DataFrame(
        {
            "storm_date": ["2020-01-01T00", "2020-01-02T00"],
            "min": [1.0, 0.2],
            "mean": [3.0, 0.5],
            "max": [5.0, 0.8],
        }
    ).to_csv(collection / "storm-stats.csv", index=False)
    (source_artifacts / "aorc_sst_rainfall_catalog.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-31T00:00:00",
                "metadata": {"backend": "direct_aorc_sst", "smoke": smoke},
            }
        ),
        encoding="utf-8",
    )
    return collection


def _write_source_manifests(source_artifacts, *, smoke=False):
    source_artifacts.mkdir(parents=True, exist_ok=True)
    for filename in [
        "cora_boundary_water_level.json",
        "nwm_retrospective_hydrologic_state.json",
    ]:
        (source_artifacts / filename).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "start": "2020-01-01T00:00:00",
                    "end": "2020-01-31T00:00:00",
                    "metadata": {"smoke": smoke},
                }
            ),
            encoding="utf-8",
        )


def _marshfield_nwm_config():
    return {
        "streamflow": {
            "available": False,
            "feature_ids": [],
            "reason": "Marshfield coastal grid has no meaningful streamflow driver.",
        },
        "soil_moisture": {"points": [{"id": "center", "x": 1.0, "y": 2.0}]},
    }


def test_readiness_passes_direct_aorc_sst_collection_gate(tmp_path):
    aorc_sst_root = tmp_path / "aorc_sst"
    source_artifacts = tmp_path / "source_artifacts"
    _write_aorc_sst_catalog(aorc_sst_root, source_artifacts)
    paths = {
        "location_name": "marshfield",
        "aorc_sst_root": aorc_sst_root,
        "source_artifacts_root": source_artifacts,
    }
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "aorc_sst": {"storm_duration_hours": 72},
        }
    }

    gate = check_aorc_sst_collection(config, paths)

    assert gate["id"] == "aorc_sst_collection"
    assert gate["passed"] is True
    assert gate["details"]["storm_stats_rows"] == 2
    assert gate["details"]["ranked_storm_rows"] == 1


def test_source_inventory_names_cora_as_event_index_marginal_and_lists_compound_sources(tmp_path):
    source_artifacts = tmp_path / "source_artifacts"
    _write_source_manifests(source_artifacts)
    (source_artifacts / "aorc_sst_rainfall_catalog.json").write_text(
        '{"status": "complete", "start": "2020-01-01", "end": "2020-01-31"}',
        encoding="utf-8",
    )
    (source_artifacts / "era5_snapwave_boundary_forcing.json").write_text(
        '{"status": "complete", "start": "2020-01-01", "end": "2020-01-31"}',
        encoding="utf-8",
    )
    paths = {
        "source_artifacts_root": source_artifacts,
    }
    config = {
        "coastal_waves": True,
        "collection": {
            "aorc_sst": {"storm_duration_hours": 72},
            "nwm": _marshfield_nwm_config(),
        },
        "event_catalog": {
            "pairing": {
                "rainfall": {"strategy": "seasonal_window_permutation", "window_days": 45},
                "soil_moisture": {"strategy": "antecedent_to_forcing", "lead_time_hours": 24},
            }
        },
    }

    inventory = source_inventory_frame(config, paths)

    rows = inventory.set_index("driver")
    assert rows.loc["coastal_water_level", "source"] == "CORA"
    assert "event-index marginal" in rows.loc["coastal_water_level", "role"]
    assert rows.loc["rainfall", "source"] == "Direct AORC SST"
    assert rows.loc["rainfall", "pairing_policy"] == "seasonal_window_permutation"
    assert rows.loc["soil_moisture", "source"] == "NWM retrospective"
    assert rows.loc["soil_moisture", "pairing_policy"] == "antecedent_to_forcing"
    assert rows.loc["coastal_waves", "pairing_policy"] == "same_historical_analog"


def test_readiness_requires_ranked_and_stats_tables_for_aorc_sst_collection(tmp_path):
    aorc_sst_root = tmp_path / "aorc_sst"
    source_artifacts = tmp_path / "source_artifacts"
    collection = aorc_sst_root / "marshfield" / "72hr-events"
    collection.mkdir(parents=True)
    source_artifacts.mkdir()
    (source_artifacts / "aorc_sst_rainfall_catalog.json").write_text(
        '{"status": "complete", "start": "2020-01-01", "end": "2020-01-31"}',
        encoding="utf-8",
    )
    paths = {
        "location_name": "marshfield",
        "aorc_sst_root": aorc_sst_root,
        "source_artifacts_root": source_artifacts,
    }
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "aorc_sst": {"storm_duration_hours": 72},
        }
    }

    gate = check_aorc_sst_collection(config, paths)

    assert gate["passed"] is False
    assert "missing or empty AORC SST storm-stats.csv" in gate["issues"]
    assert "missing or empty AORC SST ranked-storms.csv" in gate["issues"]


def test_readiness_passes_rainfall_catalog_gate_when_members_are_paired(tmp_path):
    rainfall = tmp_path / "aorc_sst/rainfall_members.csv"
    rainfall.parent.mkdir()
    pd.DataFrame(
        {
            "member_id": ["rainfall_marshfield_72h_rank0001"],
            "member_file": ["ranked-storms.csv"],
            "source": ["aorc_sst"],
        }
    ).to_csv(rainfall, index=False)
    catalog = tmp_path / "catalog/event_catalog.csv"
    catalog.parent.mkdir()
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": [str(rainfall)],
            "rainfall_member_id": ["rainfall_marshfield_72h_rank0001"],
        }
    ).to_csv(catalog, index=False)
    audit_json = tmp_path / "catalog/event_catalog_audit.json"
    audit_json.write_text('{"passed": true, "issue_count": 0, "issues": []}', encoding="utf-8")
    paths = {
        "location_name": "marshfield",
        "aorc_sst_rainfall_members_csv": rainfall,
        "event_catalog_csv": catalog,
        "event_catalog_audit_json": audit_json,
    }
    config = {"collection": {"aorc_sst": {"storm_duration_hours": 72}}}

    gate = check_rainfall_catalog_smoke(config, paths)

    assert gate["id"] == "rainfall_catalog"
    assert gate["passed"] is True
    assert gate["details"]["rainfall_member_rows"] == 1
    assert gate["details"]["paired_event_rows"] == 1


def test_readiness_requires_wave_forcing_when_coastal_waves_are_enabled(tmp_path):
    paths = {
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "era5_waves_nc": tmp_path / "waves/era5.nc",
    }
    config = {"coastal_waves": True, "collection": {}}

    gate = check_wave_forcing_smoke(config, paths)

    assert gate["id"] == "wave_forcing"
    assert gate["passed"] is False
    assert "missing collection.era5_waves settings" in gate["issues"]


def test_readiness_rejects_smoke_limited_source_artifacts_in_production(tmp_path):
    outputs = tmp_path / "outputs"
    source_artifacts = outputs / "source_artifacts"
    nwm_root = outputs / "nwm"
    aorc_sst_root = outputs / "aorc_sst"
    for path in [outputs, nwm_root, aorc_sst_root]:
        path.mkdir(parents=True)
    _write_aorc_sst_catalog(aorc_sst_root, source_artifacts, smoke=True)
    _write_source_manifests(source_artifacts, smoke=True)
    (source_artifacts / "era5_snapwave_boundary_forcing.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start": "2020-01-01T00:00:00",
                "end": "2020-01-31T00:00:00",
                "metadata": {"smoke": True},
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "location_name": "marshfield",
        "outputs_root": outputs,
        "source_artifacts_root": source_artifacts,
        "nwm_root": nwm_root,
        "aorc_sst_root": aorc_sst_root,
        "aorc_sst_rainfall_members_csv": outputs / "aorc_sst/rainfall_members.csv",
        "event_catalog_csv": outputs / "catalog/event_catalog.csv",
        "event_catalog_audit_json": outputs / "catalog/event_catalog_audit.json",
        "era5_waves_nc": outputs / "waves/era5.nc",
        "data_acquisition_readiness_json": outputs / "readiness.json",
    }
    config = {
        "coastal_waves": True,
        "collection": {
            "start": "1979-02-01",
            "end": "2022-12-31",
            "aorc_sst": {"storm_duration_hours": 72},
            "era5_waves": {"bbox_wgs84": [-71.0, 42.0, -70.0, 42.5]},
            "nwm": _marshfield_nwm_config(),
        },
    }

    audit = write_data_acquisition_readiness(config, paths)

    issues = [issue for gate in audit["gates"] for issue in gate["issues"]]
    assert audit["passed"] is False
    assert "CORA source artifact is smoke-limited" in issues
    assert "NWM source artifact is smoke-limited" in issues
    assert "ERA5 wave source artifact is smoke-limited" in issues
    assert any("does not cover production window" in issue for issue in issues)


def test_readiness_passes_wave_forcing_gate_with_complete_era5_artifact(tmp_path):
    import xarray as xr

    output = tmp_path / "waves/era5.nc"
    output.parent.mkdir()
    xr.Dataset(
        {
            "swh": ("valid_time", [1.0]),
            "pp1d": ("valid_time", [8.0]),
            "mwd": ("valid_time", [90.0]),
            "wdw": ("valid_time", [0.4]),
        },
        coords={"valid_time": [pd.Timestamp("2018-01-01")]},
    ).to_netcdf(output)
    source_artifacts = tmp_path / "source_artifacts"
    source_artifacts.mkdir()
    (source_artifacts / "era5_snapwave_boundary_forcing.json").write_text(
        '{"status": "complete", "start": "2018-01-01T00:00:00", "end": "2018-01-01T00:00:00"}',
        encoding="utf-8",
    )
    paths = {
        "location_name": "marshfield",
        "source_artifacts_root": source_artifacts,
        "era5_waves_nc": output,
    }
    config = {
        "coastal_waves": True,
        "collection": {
            "start": "2018-01-01T00:00:00",
            "end": "2018-01-01T00:00:00",
            "era5_waves": {
                "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
                "output_path": output.as_posix(),
            }
        },
    }

    gate = check_wave_forcing_smoke(config, paths)

    assert gate["id"] == "wave_forcing"
    assert gate["passed"] is True
    assert gate["details"]["variables"] == ["swh", "pp1d", "mwd", "wdw"]


def test_readiness_rejects_wave_forcing_when_netcdf_does_not_cover_collection_window(tmp_path):
    import xarray as xr

    output = tmp_path / "waves/era5.nc"
    output.parent.mkdir()
    xr.Dataset(
        {
            "swh": ("valid_time", [1.0] * 24),
            "pp1d": ("valid_time", [8.0] * 24),
            "mwd": ("valid_time", [90.0] * 24),
            "wdw": ("valid_time", [0.4] * 24),
        },
        coords={"valid_time": pd.date_range("2018-01-01", periods=24, freq="h")},
    ).to_netcdf(output)
    source_artifacts = tmp_path / "source_artifacts"
    source_artifacts.mkdir()
    (source_artifacts / "era5_snapwave_boundary_forcing.json").write_text(
        '{"status": "complete", "start": "1979-02-01T00:00:00", "end": "2022-12-31T00:00:00"}',
        encoding="utf-8",
    )
    paths = {
        "location_name": "marshfield",
        "source_artifacts_root": source_artifacts,
        "era5_waves_nc": output,
    }
    config = {
        "coastal_waves": True,
        "collection": {
            "start": "1979-02-01T00:00:00",
            "end": "2022-12-31T00:00:00",
            "era5_waves": {
                "bbox_wgs84": [-71.0, 42.0, -70.0, 42.5],
                "output_path": output.as_posix(),
            },
        },
    }

    gate = check_wave_forcing_smoke(config, paths)

    assert gate["passed"] is False
    assert "ERA5 wave NetCDF does not cover collection window" in gate["issues"]


def test_readiness_passes_source_acquisition_gate_for_marshfield_exception(tmp_path):
    outputs = tmp_path / "outputs"
    source_artifacts = outputs / "source_artifacts"
    nwm_root = outputs / "nwm"
    aorc_sst_root = outputs / "aorc_sst"
    for path in [outputs, source_artifacts, nwm_root, aorc_sst_root]:
        path.mkdir(parents=True)
    _write_source_manifests(source_artifacts)
    paths = {
        "location_name": "marshfield",
        "outputs_root": outputs,
        "source_artifacts_root": source_artifacts,
        "nwm_root": nwm_root,
        "aorc_sst_root": aorc_sst_root,
    }
    config = {"collection": {"aorc_sst": {"storm_duration_hours": 72}, "nwm": _marshfield_nwm_config()}}

    gate = check_acquisition_dry_run(config, paths)

    assert gate["id"] == "source_acquisition"
    assert gate["passed"] is True
    assert gate["details"]["rainfall_backend"] == "aorc_sst"
    assert gate["details"]["streamflow_available"] is False
    assert gate["details"]["soil_moisture_point_count"] == 1


def test_readiness_source_acquisition_rejects_missing_aorc_sst_collection_config(tmp_path):
    outputs = tmp_path / "outputs"
    source_artifacts = outputs / "source_artifacts"
    nwm_root = outputs / "nwm"
    for path in [outputs, source_artifacts, nwm_root]:
        path.mkdir(parents=True)
    _write_source_manifests(source_artifacts)
    paths = {
        "location_name": "marshfield",
        "outputs_root": outputs,
        "source_artifacts_root": source_artifacts,
        "nwm_root": nwm_root,
        "aorc_sst_root": outputs / "aorc_sst",
    }
    config = {"collection": {"nwm": _marshfield_nwm_config()}}

    gate = check_acquisition_dry_run(config, paths)

    assert gate["passed"] is False
    assert "AORC SST rainfall collection is not configured" in gate["issues"]


def test_readiness_audit_passes_when_all_three_gates_pass(tmp_path):
    outputs = tmp_path / "outputs"
    aorc_sst_root = outputs / "aorc_sst"
    source_artifacts = outputs / "source_artifacts"
    nwm_root = outputs / "nwm"
    for path in [outputs, source_artifacts, nwm_root, aorc_sst_root]:
        path.mkdir(parents=True)
    collection = _write_aorc_sst_catalog(aorc_sst_root, source_artifacts)
    _write_source_manifests(source_artifacts)
    rainfall = aorc_sst_root / "rainfall_members.csv"
    pd.DataFrame(
        {
            "member_id": ["rainfall_marshfield_72h_rank0001"],
            "member_file": [str(collection / "ranked-storms.csv")],
            "source": ["aorc_sst"],
        }
    ).to_csv(rainfall, index=False)
    catalog = outputs / "catalog/event_catalog.csv"
    catalog.parent.mkdir()
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": [str(rainfall)],
            "rainfall_member_id": ["rainfall_marshfield_72h_rank0001"],
        }
    ).to_csv(catalog, index=False)
    audit_json = outputs / "catalog/event_catalog_audit.json"
    audit_json.write_text('{"passed": true, "issue_count": 0, "issues": []}', encoding="utf-8")
    paths = {
        "location_name": "marshfield",
        "outputs_root": outputs,
        "source_artifacts_root": source_artifacts,
        "nwm_root": nwm_root,
        "aorc_sst_root": aorc_sst_root,
        "aorc_sst_rainfall_members_csv": rainfall,
        "event_catalog_csv": catalog,
        "event_catalog_audit_json": audit_json,
        "data_acquisition_readiness_json": outputs / "readiness.json",
    }
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "aorc_sst": {"storm_duration_hours": 72},
            "nwm": _marshfield_nwm_config(),
        }
    }

    audit = write_data_acquisition_readiness(config, paths)

    assert audit["passed"] is True
    assert [gate["id"] for gate in audit["gates"]] == ["aorc_sst_collection", "rainfall_catalog", "source_acquisition"]
    assert [gate["passed"] for gate in audit["gates"]] == [True, True, True]
    assert json.loads(paths["data_acquisition_readiness_json"].read_text(encoding="utf-8"))["passed"] is True
