from pathlib import Path
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import LineString, Point, Polygon, box

from study_location import define_location
from wflow_runs import (
    build_meteo,
    build_wflow_submodel,
    build_wflow_data_catalog,
    build_wflow_build_plan,
    build_wflow_steps_for_submodel,
    plan_wflow_domain_set,
    plan_wflow_us_source_strategy,
    plan_wflow_domain_set_from_streamgages,
    wflow_catalog_source_readiness,
    write_wflow_sfincs_gauge_locations,
    write_wflow_observation_gauge_locations,
    write_wflow_subbasin_fabric_from_nhdplus,
    write_wflow_domain_set_manifest,
)
from wflow_runs.notebook import (
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
    run_wflow_event_replay,
    wflow_event_replay_plan,
)
from wflow_runs.build_plan import (
    _assert_wflow_reservoir_staticmaps_current,
    _cached_wbd_hucs,
    _observation_gauge_layer_matches,
    _sfincs_gauge_layer_matches,
    _write_wflow_sfincs_handoff_gauge_locations,
)
from wflow_runs.build_plan import (
    repair_wflow_canopy_parameters,
    repair_wflow_gauge_map,
    repair_wflow_river_width,
    repair_wflow_staticmaps_nodata,
    validate_wflow_reservoir_outlets,
    validate_wflow_reservoir_staticmaps,
    validate_staticmaps,
    write_wflow_reservoir_readiness,
)
from design_events.collect_sources.aorc_event_meteo import prepare_aorc_temp_pet_for_wflow
from design_events.collect_sources.aorc_sst import collect_warmup
from wflow_runs.replay import _prepare_replay_submodel_output_dir, _write_per_event_update_config
from wflow_runs.replay import merge_submodel_discharge, replay_inland_domain_set
from wflow_runs.replay import _zero_event_forcing
from wflow_runs.replay import write_event_streamflow_handoff_discharge
from wflow_runs.coupling_qa import validate_dynamic_handoff
from wflow_runs.dynamic_handoff import require_handoff
from wflow_runs.streamflow_realization import prepare_wflow_streamflow_realization_for_event_model
from wflow_runs.river_geometry import validate_geometry
from wflow_runs.states import (
    configure_wflow_state_paths,
    plan_warmup,
    validate_wflow_reservoir_states,
    validate_instates,
    warmup_window,
    write_cold_state_workflow,
)
from wflow_runs.visualize import _gauge_layers, _reference_rivers_for_basemap


def test_location_wflow_configs_use_installed_runner_command():
    repo_root = Path(__file__).resolve().parents[2]

    for name in ("greensboro", "austin"):
        config = define_location(repo_root / f"locations/{name}/config.yaml").config
        command = config["wflow"]["run"]["command"]

        assert command == "wflow_cli {run_config}"


@pytest.mark.parametrize("name", ("greensboro", "austin"))
def test_location_build_config_is_hydromt_v1_steps_format(name):
    repo_root = Path(__file__).resolve().parents[2]
    workflow = define_location(repo_root / f"locations/{name}/config.yaml").model_recipes["wflow_build"]

    assert list(workflow) == ["steps"]
    step_names = [next(iter(step)) for step in workflow["steps"]]
    expected = [
        "setup_config",
        "setup_basemaps",
        "setup_rivers",
    ]
    if name == "austin":
        expected.append("setup_reservoirs_no_control")
    expected.extend([
        "setup_lulcmaps",
        "setup_soilmaps",
        "setup_constant_pars",
        "setup_gauges",
    ])
    assert step_names == expected
    assert workflow["steps"][2]["setup_rivers"]["river_geom_fn"] == "nhdplus_hr_river_geometry"
    if name == "austin":
        assert workflow["steps"][3]["setup_reservoirs_no_control"]["reservoirs_fn"] == "nhdplus_hr_wflow_reservoirs"
    assert workflow["steps"][-1]["setup_gauges"]["snap_to_river"] is True


@pytest.mark.parametrize("name", ("greensboro", "austin"))
def test_location_update_forcing_config_is_hydromt_v1_steps_format(tmp_path, name):
    """Per-event Wflow update config must parse as HydroMT v1.x (a ``steps:`` section).

    Regression for the v0.x flat-format ``wflow_update_forcing.yml`` that made
    ``hydromt update wflow_sbm`` raise "does not contain a `steps` section". Exercises the
    real path: the canonical source config → :func:`_write_per_event_update_config` (which
    substitutes the event window) → hydromt's ``read_workflow_yaml`` (the function that
    raised).
    """
    from hydromt.readers import read_workflow_yaml

    repo_root = Path(__file__).resolve().parents[2]
    definition = define_location(repo_root / f"locations/{name}/config.yaml")
    build_wflow_build_plan(definition.config, {"location_root": repo_root / f"locations/{name}"})
    source = repo_root / f"locations/{name}/data/wflow/config/wflow_update_forcing.yml"
    start, end = pd.Timestamp("2020-11-12T00:00:00"), pd.Timestamp("2020-11-17T00:00:00")

    out = _write_per_event_update_config(source, tmp_path, start, end)
    _modeltype, _init, steps = read_workflow_yaml(out, modeltype="wflow_sbm")

    step_names = [next(iter(step)) for step in steps]
    assert step_names == [
        "setup_config",
        "setup_precip_forcing",
        "setup_temp_pet_forcing",
    ]
    setup_config = next(s for s in steps if "setup_config" in s)["setup_config"]["data"]
    assert setup_config["time.starttime"] == "2020-11-12T00:00:00"
    assert setup_config["time.endtime"] == "2020-11-17T00:00:00"


def test_location_update_forcing_config_uses_aorc_surface_pressure_without_pressure_correction():
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("greensboro", "austin"):
        workflow = define_location(repo_root / f"locations/{name}/config.yaml").model_recipes["wflow_update_forcing"]
        temp_pet = next(step["setup_temp_pet_forcing"] for step in workflow["steps"] if "setup_temp_pet_forcing" in step)
        assert temp_pet["pet_method"] == "makkink"
        assert temp_pet["press_correction"] is False


def test_prepare_aorc_temp_pet_for_wflow_writes_native_makkink_contract(tmp_path):
    source = tmp_path / "aorc_window.nc"
    precip = tmp_path / "precip.nc"
    out = tmp_path / "temp_pet.nc"
    provenance = tmp_path / "temp_pet_provenance.json"
    time = pd.date_range("2020-01-01", periods=3, freq="1h")
    coords = {"time": time, "latitude": [35.0, 35.1], "longitude": [-80.0, -79.9]}
    shape = (3, 2, 2)
    xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), np.full(shape, 293.15)),
            "PRES_surface": (("time", "latitude", "longitude"), np.full(shape, 101325.0)),
            "DSWRF_surface": (("time", "latitude", "longitude"), np.full(shape, 200.0)),
        },
        coords=coords,
    ).to_netcdf(source)
    xr.Dataset(
        {"precip": (("time", "y", "x"), np.zeros(shape, dtype="float32"))},
        coords={"time": time, "y": [35.0, 35.1], "x": [-80.0, -79.9]},
    ).to_netcdf(precip)

    report = prepare_aorc_temp_pet_for_wflow(
        source,
        out,
        t_start=time[0],
        t_stop=time[-1],
        precip_template=precip,
        provenance_path=provenance,
    )

    ds = xr.open_dataset(out)
    try:
        assert set(ds.data_vars) == {"temp", "press_msl", "kin"}
        assert float(ds["temp"].mean()) == pytest.approx(20.0)
        assert float(ds["press_msl"].mean()) == pytest.approx(1013.25)
        assert float(ds["kin"].mean()) == pytest.approx(200.0)
        assert ds["temp"].attrs["units"] == "degree C"
    finally:
        ds.close()
    assert provenance.exists()
    assert report["variable_mapping"]["temp"] == "TMP_2maboveground"


def test_validate_staticmaps_flags_bad_staticmaps(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    intmin = np.iinfo(np.int32).min
    xr.Dataset(
        {
            "subcatchment": (("y", "x"), np.array([[1, intmin], [1, 1]], dtype=np.int32)),
            "local_drain_direction": (("y", "x"), np.array([[5, 255], [5, 5]], dtype=np.uint8)),
            "river_mask": (("y", "x"), np.ones((2, 2), dtype=np.uint8)),
            "river_width": (("y", "x"), np.full((2, 2), 30.0, dtype=np.float32)),
            "river_depth": (("y", "x"), np.full((2, 2), 1.0, dtype=np.float32)),
            "land_slope": (("y", "x"), np.full((2, 2), 1000.0, dtype=np.float32)),
            "meta_upstream_area": (("y", "x"), np.full((2, 2), 1.0, dtype=np.float32)),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(model_root / "staticmaps.nc")

    report = validate_staticmaps(
        model_root,
        river_upa_km2=10.0,
        raise_on_error=False,
    )

    assert set(report["status"]) >= {"failed", "review_required"}
    with pytest.raises(RuntimeError):
        validate_staticmaps(model_root, river_upa_km2=10.0)


def test_validate_wflow_reservoir_staticmaps_requires_native_maps(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    xr.Dataset(
        {
            "subcatchment": (("y", "x"), np.ones((2, 2), dtype=np.int32)),
            "local_drain_direction": (("y", "x"), np.ones((2, 2), dtype=np.uint8)),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(model_root / "staticmaps.nc")

    report = validate_wflow_reservoir_staticmaps(model_root, required=True, raise_on_error=False)

    assert report.loc[0, "status"] == "failed"
    assert "reservoir_area_id" in report.loc[0, "message"]


def test_reservoir_enabled_stale_base_blocks_reuse(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    xr.Dataset(
        {
            "subcatchment": (("y", "x"), np.ones((2, 2), dtype=np.int32)),
            "local_drain_direction": (("y", "x"), np.ones((2, 2), dtype=np.uint8)),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(model_root / "staticmaps.nc")
    config = {"collection": {"national_hydrography": {"reservoirs": {"enabled": True}}}}

    with pytest.raises(RuntimeError, match="stale for enabled reservoirs"):
        _assert_wflow_reservoir_staticmaps_current(config, model_root, "austin_p5u")


def test_validate_wflow_reservoir_staticmaps_accepts_native_maps(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    reservoir_ids = np.array([[1, 1], [0, 0]], dtype=np.int32)
    xr.Dataset(
        {
            "reservoir_area_id": (("y", "x"), reservoir_ids),
            "reservoir_outlet_id": (("y", "x"), np.array([[0, 1], [0, 0]], dtype=np.int32)),
            "reservoir_initial_depth": (("y", "x"), np.where(reservoir_ids > 0, 5.0, 0.0).astype("float32")),
            "meta_reservoir_mean_outflow": (("y", "x"), np.where(reservoir_ids > 0, 10.0, 0.0).astype("float32")),
            "reservoir_b": (("y", "x"), np.where(reservoir_ids > 0, 1.0, 0.0).astype("float32")),
            "reservoir_e": (("y", "x"), np.where(reservoir_ids > 0, 2.0, 0.0).astype("float32")),
            "reservoir_rating_curve": (("y", "x"), np.where(reservoir_ids > 0, 2, 0).astype("int32")),
            "reservoir_storage_curve": (("y", "x"), np.where(reservoir_ids > 0, 1, 0).astype("int32")),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(model_root / "staticmaps.nc")

    report = validate_wflow_reservoir_staticmaps(model_root)

    assert set(report["status"]) == {"passed"}
    assert "area_cells=2" in report.loc[report["check"].eq("reservoir_area_id"), "message"].iloc[0]


def test_validate_wflow_reservoir_states_requires_reservoir_water_level(tmp_path):
    model_root = tmp_path / "wflow_model"
    instate = model_root / "instate/instates.nc"
    instate.parent.mkdir(parents=True)
    xr.Dataset({"river_h": (("y", "x"), np.ones((2, 2), dtype="float32"))}).to_netcdf(instate)

    report = validate_wflow_reservoir_states(model_root, raise_on_error=False)

    assert report.loc[0, "status"] == "failed"
    assert "missing reservoir_water_level" in report.loc[0, "message"]

    xr.Dataset({"reservoir_water_level": (("y", "x"), np.ones((2, 2), dtype="float32"))}).to_netcdf(instate)
    report = validate_wflow_reservoir_states(model_root)
    assert report.loc[0, "status"] == "passed"


def test_validate_wflow_reservoir_outlets_flags_missing_outlet_and_source_ids(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    xr.Dataset(
        {
            "reservoir_area_id": (("y", "x"), np.array([[1, 2], [0, 0]], dtype=np.int32)),
            "reservoir_outlet_id": (("y", "x"), np.array([[1, 0], [0, 0]], dtype=np.int32)),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(model_root / "staticmaps.nc")
    reservoirs = tmp_path / "reservoirs.gpkg"
    gpd.GeoDataFrame(
        {"waterbody_id": [1, 2], "waterbody_name": ["Lake Travis", "Lake Austin"]},
        geometry=[box(-0.1, 0.9, 0.1, 1.1), box(0.9, 0.9, 1.1, 1.1)],
        crs="EPSG:4326",
    ).to_file(reservoirs, driver="GPKG")

    report = validate_wflow_reservoir_outlets(model_root, reservoirs_path=reservoirs, raise_on_error=False)

    assert "failed" in set(report["status"])
    assert "review_required" in set(report["status"])
    assert "missing_outlet_ids=[2]" in "; ".join(report["message"].astype(str))


def _write_dynamic_discharge_fixture(path: Path, *, names=("inflow_01",), values=(1.0, 2.0, 3.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "discharge": (
                ("index", "time"),
                np.array([values for _ in names], dtype="float32"),
            )
        },
        coords={
            "index": np.arange(1, len(names) + 1),
            "time": pd.date_range("2020-01-01", periods=len(values), freq="1h"),
            "name": ("index", np.array(names, dtype=object)),
        },
    ).to_netcdf(path)


def test_river_geometry_readiness_rejects_constant_formula_fallback(tmp_path):
    path = tmp_path / "rivers.gpkg"
    gpd.GeoDataFrame(
        {
            "rivwth": [30.0, 30.0],
            "rivdph": [1.0, 1.0],
            "rivwth_source": ["drainage_area_formula_fallback", "drainage_area_formula_fallback"],
            "rivdph_source": ["missing_native_powlaw_fallback", "missing_native_powlaw_fallback"],
        },
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")

    report = validate_geometry(path, raise_on_error=False)

    assert set(report["status"]) >= {"failed", "review_required"}
    with pytest.raises(RuntimeError, match="river geometry QA failed"):
        validate_geometry(path)


def test_river_geometry_readiness_accepts_variable_stream_geo_fields(tmp_path):
    path = tmp_path / "rivers.gpkg"
    gpd.GeoDataFrame(
        {
            "rivwth": [12.0, 34.0],
            "rivdph": [0.8, 2.2],
            "rivwth_source": ["STREAM-geo", "STREAM-geo"],
            "rivdph_source": ["STREAM-geo", "STREAM-geo"],
        },
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")

    report = validate_geometry(path)

    assert set(report["status"]) == {"passed"}


def test_state_helpers_write_native_cold_state_workflow_and_configure_toml(tmp_path):
    workflow = write_cold_state_workflow(tmp_path / "setup_cold_states.yml", timestamp="2020-01-01T00:00:00")
    toml = tmp_path / "wflow_sbm.toml"
    toml.write_text(
        'dir_output = "run_default"\n[model]\ncold_start__flag = true\n[state]\npath_input = "old.nc"\n',
        encoding="utf-8",
    )

    configure_wflow_state_paths(toml)

    assert "setup_cold_states" in workflow.read_text(encoding="utf-8")
    text = toml.read_text(encoding="utf-8")
    assert 'path_input = "instate/instates.nc"' in text
    assert 'path_output = "outstate/outstates.nc"' in text
    assert "cold_start__flag = false" in text


def test_validate_instates_requires_native_state_file(tmp_path):
    location_root = tmp_path / "location"
    model_root = location_root / "data/wflow/base/test_submodel"
    model_root.mkdir(parents=True)
    config = {
        "wflow": {
            "base_model_root": "data/wflow/base",
            "domain_set": {"submodels": [{"wflow_submodel_id": "test_submodel"}]},
        }
    }

    report = validate_instates(config, location_root, raise_on_error=False)
    assert report.loc[0, "status"] == "failed"

    (model_root / "instate").mkdir()
    (model_root / "instate/instates.nc").write_text("placeholder", encoding="utf-8")
    report = validate_instates(config, location_root)
    assert report.loc[0, "status"] == "passed"


def test_validate_instates_requires_reservoir_state_when_enabled(tmp_path):
    location_root = tmp_path / "location"
    model_root = location_root / "data/wflow/base/test_submodel"
    instate = model_root / "instate/instates.nc"
    instate.parent.mkdir(parents=True)
    xr.Dataset({"river_h": (("y", "x"), np.ones((2, 2), dtype="float32"))}).to_netcdf(instate)
    config = {
        "wflow": {
            "base_model_root": "data/wflow/base",
            "domain_set": {"submodels": [{"wflow_submodel_id": "test_submodel"}]},
        },
        "collection": {"national_hydrography": {"reservoirs": {"enabled": True}}},
    }

    report = validate_instates(config, location_root, raise_on_error=False)

    assert report.loc[0, "status"] == "failed"
    assert "reservoir_water_level" in report.loc[0, "message"]


def test_warmup_window_defaults_to_90_days_before_reference():
    start, end = warmup_window("2020-11-14T00:00:00")

    assert start == pd.Timestamp("2020-08-16T00:00:00")
    assert end == pd.Timestamp("2020-11-13T23:00:00")


def test_shared_baseline_warmup_plan_is_event_agnostic(tmp_path):
    config = {
        "wflow": {
            "dynamic_handoff": {
                "state_policy": "shared_baseline",
                "baseline_id": "baseline_90d",
                "baseline_reference_time": "2020-11-14T00:00:00",
                "baseline_root": "data/wflow/warmup/baseline_90d",
                "warmup_days": 90,
            }
        }
    }

    plan = plan_warmup(config, tmp_path)

    assert plan["state_policy"] == "shared_baseline"
    assert plan["baseline_id"] == "baseline_90d"
    assert plan["warmup_start"] == "2020-08-16T00:00:00"
    assert plan["warmup_end"] == "2020-11-13T23:00:00"
    assert plan["warmup_precip"] == str(tmp_path / "data/wflow/warmup/baseline_90d/precip.nc")
    assert plan["warmup_temp_pet"] == str(tmp_path / "data/wflow/warmup/baseline_90d/temp_pet.nc")


def test_collect_warmup_writes_cached_precip_and_temp_pet(tmp_path):
    times = pd.date_range("2020-01-01T00:00:00", periods=6, freq="h")
    lat = [35.0, 35.1]
    lon = [-80.0, -79.9]
    ds = xr.Dataset(
        {
            "APCP_surface": (("time", "latitude", "longitude"), np.ones((6, 2, 2), dtype="float32")),
            "TMP_2maboveground": (("time", "latitude", "longitude"), np.full((6, 2, 2), 293.15, dtype="float32")),
            "PRES_surface": (("time", "latitude", "longitude"), np.full((6, 2, 2), 100000.0, dtype="float32")),
            "DSWRF_surface": (("time", "latitude", "longitude"), np.full((6, 2, 2), 200.0, dtype="float32")),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    config = {
        "collection": {"aorc_sst": {"bbox_wgs84": [-80.1, 34.9, -79.8, 35.2], "variable": "APCP_surface"}},
        "wflow": {
            "dynamic_handoff": {
                "baseline_id": "baseline_test",
                "baseline_reference_time": "2020-01-01T06:00:00",
                "baseline_root": "data/wflow/warmup/baseline_test",
                "warmup_days": 0.25,
            }
        },
    }
    paths = {"location_root": tmp_path, "repo_root": tmp_path}

    report = collect_warmup(config, paths, opener=lambda year, spec: ds, force=True)

    precip_nc = tmp_path / "data/wflow/warmup/baseline_test/precip.nc"
    temp_pet_nc = tmp_path / "data/wflow/warmup/baseline_test/temp_pet.nc"
    assert report["status"] == "collected"
    assert precip_nc.exists()
    assert temp_pet_nc.exists()
    with xr.open_dataset(precip_nc) as precip:
        assert "precip" in precip
        assert {"time", "y", "x"}.issubset(precip["precip"].dims)
    with xr.open_dataset(temp_pet_nc) as meteo:
        assert {"temp", "press_msl", "kin"}.issubset(meteo.data_vars)


def test_wflow_basemap_overlays_fallback_gages_as_usgs_layer():
    sfincs = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    usgs = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")

    class Model:
        geoms = {"gauges_sfincs": sfincs}

    layers = _gauge_layers(Model(), usgs)

    assert [name for name, _ in layers] == ["gauges_sfincs", "gauges_usgs"]


def test_wflow_basemap_adds_lower_order_rivers_near_usgs_gages():
    rivers = gpd.GeoDataFrame(
        {
            "strord": [3, 1, 1],
            "name": ["main", "near_gage", "far_low_order"],
        },
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(0.48, -0.02), (0.52, 0.02)]),
            LineString([(5.0, 5.0), (5.2, 5.0)]),
        ],
        crs="EPSG:4326",
    )
    gages = gpd.GeoDataFrame(geometry=[Point(0.5, 0.0)], crs="EPSG:4326")

    selected, _ = _reference_rivers_for_basemap(
        rivers,
        sfincs_domains=None,
        map_crs="EPSG:4326",
        gage_context=gages,
        streamorder_field="strord",
        min_river_streamorder=3,
        sfincs_domain_min_river_streamorder=None,
        gage_min_river_streamorder=1,
        gage_river_buffer=0.05,
    )

    assert set(selected["name"]) == {"main", "near_gage"}


def test_dynamic_handoff_qa_reports_zero_rain_peak_fraction_without_blocking(tmp_path):
    event = tmp_path / "event.nc"
    zero = tmp_path / "zero.nc"
    _write_dynamic_discharge_fixture(event, names=("inflow_01",), values=(1.0, 10.0, 2.0))
    _write_dynamic_discharge_fixture(zero, names=("inflow_01",), values=(1.0, 8.0, 2.0))

    report = validate_dynamic_handoff(
        event,
        zero_rain_discharge_nc=zero,
        expected_source_ids={"inflow_01"},
        max_zero_peak_fraction=0.2,
    )

    row = report[report["check"] == "zero_rain_peak_fraction"].iloc[0]
    assert row["status"] == "diagnostic"
    assert "fraction=0.8" in row["message"]


def test_zero_event_forcing_removes_precip_and_external_river_inflow(tmp_path):
    forcing = tmp_path / "inmaps-event.nc"
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    xr.Dataset(
        {
            "precip": (("time", "y", "x"), np.ones((2, 2, 2), dtype="float32")),
            "river_inflow": (("time", "y", "x"), np.full((2, 2, 2), 3.0, dtype="float32")),
            "temp": (("time", "y", "x"), np.full((2, 2, 2), 20.0, dtype="float32")),
        },
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
    ).to_netcdf(forcing)

    _zero_event_forcing(forcing)

    with xr.open_dataset(forcing) as ds:
        assert float(ds["precip"].sum()) == 0.0
        assert float(ds["river_inflow"].sum()) == 0.0
        assert float(ds["temp"].mean()) == 20.0


def test_dynamic_handoff_qa_rejects_stale_source_ids(tmp_path):
    event = tmp_path / "event.nc"
    _write_dynamic_discharge_fixture(event, names=("inflow_01", "stale_inflow"), values=(1.0, 10.0, 2.0))

    report = validate_dynamic_handoff(event, expected_source_ids={"inflow_01", "inflow_02"}, raise_on_error=False)

    row = report[report["check"] == "source_ids"].iloc[0]
    assert row["status"] == "failed"
    assert "stale_inflow" in row["message"]


def test_dynamic_handoff_qa_rejects_duplicated_source_hydrograph_shapes(tmp_path):
    event = tmp_path / "event.nc"
    _write_dynamic_discharge_fixture(event, names=("inflow_01", "inflow_02"), values=(1.0, 10.0, 2.0))

    report = validate_dynamic_handoff(event, expected_source_ids={"inflow_01", "inflow_02"}, raise_on_error=False)

    row = report[report["check"] == "source_hydrograph_shape_diversity"].iloc[0]
    assert row["status"] == "failed"
    assert "inflow_01~inflow_02" in row["message"]


def test_dynamic_handoff_qa_accepts_distinct_source_hydrograph_shapes(tmp_path):
    event = tmp_path / "event.nc"
    event.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "discharge": (
                ("index", "time"),
                np.array(
                    [
                        [0.0, 1.0, 8.0, 4.0, 1.0],
                        [0.0, 0.0, 2.0, 9.0, 5.0],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={
            "index": [1, 2],
            "time": pd.date_range("2020-01-01", periods=5, freq="1h"),
            "name": ("index", np.array(["inflow_01", "inflow_02"], dtype=object)),
        },
    ).to_netcdf(event)

    report = validate_dynamic_handoff(event, expected_source_ids={"inflow_01", "inflow_02"}, raise_on_error=False)

    row = report[report["check"] == "source_hydrograph_shape_diversity"].iloc[0]
    assert row["status"] == "passed"


def test_require_handoff_explains_missing_acceptance_with_existing_discharge(tmp_path):
    location_root = tmp_path / "location"
    event_root = location_root / "data/wflow/events/design_0398"
    event_root.mkdir(parents=True)
    _write_dynamic_discharge_fixture(event_root / "sfincs_discharge.nc", names=("inflow_01",), values=(1.0, 2.0))
    config = {"wflow": {"events_root": "data/wflow/events"}}

    with pytest.raises(FileNotFoundError, match="b_prepare_wflow_dynamic_handoff.ipynb.*design_0398"):
        require_handoff(config, location_root, "design_0398")


def test_require_handoff_rejects_missing_streamflow_realization_metadata(tmp_path):
    location_root = tmp_path / "location"
    event_root = location_root / "data/wflow/events/design_0398"
    event_root.mkdir(parents=True)
    discharge = event_root / "sfincs_discharge.nc"
    _write_dynamic_discharge_fixture(discharge, names=("inflow_01",), values=(1.0, 2.0))
    acceptance = event_root / "sfincs_discharge.dynamic_handoff.json"
    acceptance.write_text(
        json.dumps(
            {
                "event_id": "design_0398",
                "status": "accepted",
                "discharge_source": "wflow_dynamic",
                "discharge_nc": str(discharge),
                "checks": [
                    {"check": "event_peak", "status": "passed", "message": "peak_m3s=2"},
                    {"check": "zero_rain_peak_fraction", "status": "passed", "message": "fraction=0.1"},
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    config = {"wflow": {"events_root": "data/wflow/events"}}

    with pytest.raises(RuntimeError, match="USGS/POT streamflow consumed by Wflow"):
        require_handoff(config, location_root, "design_0398")


def test_locations_default_to_dynamic_wflow_handoff():
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("greensboro", "austin"):
        config = define_location(repo_root / f"locations/{name}/config.yaml").config
        assert config["inland_coupling"]["discharge_forcing"]["source"] == "wflow_dynamic"
        assert config["wflow"]["dynamic_handoff"]["state_policy"] == "shared_baseline"
        assert config["wflow"]["dynamic_handoff"]["baseline_id"] == "baseline_90d"
        assert config["wflow"]["dynamic_handoff"]["warmup_days"] == 90


def test_repair_wflow_staticmaps_nodata_retags_intmin_subcatchments(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    intmin = np.iinfo(np.int32).min
    staticmaps_path = model_root / "staticmaps.nc"
    xr.Dataset(
        {
            "subcatchment": (
                ("y", "x"),
                np.array([[1, intmin], [0, 2]], dtype=np.int32),
                {"_FillValue": np.int32(0)},
            ),
            "local_drain_direction": (
                ("y", "x"),
                np.array([[5, 255], [255, 7]], dtype=np.uint8),
                {"_FillValue": np.uint8(255)},
            ),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(staticmaps_path)

    repair_wflow_staticmaps_nodata(model_root)

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        assert raw["subcatchment"].dtype == np.int32
        assert raw["subcatchment"].attrs["_FillValue"] == 0
        assert raw["subcatchment"].values.tolist() == [[1, 0], [0, 2]]
        active = raw["subcatchment"].values != 0
        missing_ldd = raw["local_drain_direction"].values == 255
        assert int((active & missing_ldd).sum()) == 0
    finally:
        raw.close()


def test_repair_wflow_staticmaps_nodata_casts_float_subcatchments_to_wflow_contract(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    intmin = np.iinfo(np.int32).min
    staticmaps_path = model_root / "staticmaps.nc"
    xr.Dataset(
        {
            "subcatchment": (
                ("y", "x"),
                np.array([[1.0, float(intmin)], [np.nan, 2.0]], dtype=np.float64),
                {"_FillValue": np.nan},
            ),
            "local_drain_direction": (
                ("y", "x"),
                np.array([[5, 255], [255, 7]], dtype=np.uint8),
                {"_FillValue": np.uint8(255)},
            ),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(staticmaps_path)

    repair_wflow_staticmaps_nodata(model_root)

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        assert raw["subcatchment"].dtype == np.int32
        assert raw["subcatchment"].attrs["_FillValue"] == 0
        assert raw["subcatchment"].values.tolist() == [[1, 0], [0, 2]]
        active = raw["subcatchment"].values != 0
        missing_ldd = raw["local_drain_direction"].values == 255
        assert int((active & missing_ldd).sum()) == 0
    finally:
        raw.close()


def test_repair_wflow_river_width_adds_missing_staticmap_and_toml_mapping(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    staticmaps_path = model_root / "staticmaps.nc"
    xr.Dataset(
        {
            "river_mask": (
                ("y", "x"),
                np.array([[1, 0], [2, 0]], dtype=np.uint8),
                {"_FillValue": np.uint8(0)},
            ),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(staticmaps_path)
    (model_root / "wflow_sbm.toml").write_text(
        "[input.static]\nriver__length = \"river_length\"\nriver__slope = \"river_slope\"\n",
        encoding="utf-8",
    )

    repair_wflow_river_width(model_root)

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        assert raw["river_width"].dtype == np.float32
        assert raw["river_width"].attrs["_FillValue"] == np.float32(-9999.0)
        assert raw["river_width"].values.tolist() == [[30.0, -9999.0], [30.0, -9999.0]]
    finally:
        raw.close()
    toml_text = (model_root / "wflow_sbm.toml").read_text(encoding="utf-8")
    assert 'river__width = "river_width"' in toml_text


def test_repair_wflow_canopy_parameters_adds_noncyclic_staticmaps_and_toml(tmp_path):
    model_root = tmp_path / "wflow_model"
    model_root.mkdir()
    staticmaps_path = model_root / "staticmaps.nc"
    xr.Dataset(
        {
            "vegetation_kext": (
                ("y", "x"),
                np.array([[0.8, 0.0], [-999.0, 0.6]], dtype=np.float32),
                {"_FillValue": np.float32(-999.0)},
            ),
            "vegetation_leaf_storage": (
                ("y", "x"),
                np.array([[0.23, 0.0], [-999.0, 0.1]], dtype=np.float32),
                {"_FillValue": np.float32(-999.0)},
            ),
            "vegetation_wood_storage": (
                ("y", "x"),
                np.array([[0.09, 0.0], [-999.0, 0.01]], dtype=np.float32),
                {"_FillValue": np.float32(-999.0)},
            ),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(staticmaps_path)
    (model_root / "wflow_sbm.toml").write_text(
        "[input.static]\nvegetation_root__depth = \"vegetation_root_depth\"\n",
        encoding="utf-8",
    )

    repair_wflow_canopy_parameters(model_root)

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        gap = raw["vegetation_canopy_gap_fraction"]
        storage = raw["vegetation_water_storage_capacity"]
        assert gap.dtype == np.float32
        assert storage.dtype == np.float32
        assert gap.attrs["_FillValue"] == np.float32(-999.0)
        assert storage.attrs["_FillValue"] == np.float32(-999.0)
        np.testing.assert_allclose(gap.values[[0, 0, 1], [0, 1, 1]], np.exp(-np.array([0.8, 0.0, 0.6])))
        assert gap.values[1, 0] == np.float32(-999.0)
        np.testing.assert_allclose(storage.values[[0, 0, 1], [0, 1, 1]], [0.32, 0.0, 0.11])
        assert storage.values[1, 0] == np.float32(-999.0)
    finally:
        raw.close()
    toml_text = (model_root / "wflow_sbm.toml").read_text(encoding="utf-8")
    assert 'vegetation_canopy__gap_fraction = "vegetation_canopy_gap_fraction"' in toml_text
    assert 'vegetation_water__storage_capacity = "vegetation_water_storage_capacity"' in toml_text


def test_repair_wflow_gauge_map_places_all_sfincs_gauges_on_active_river_cells(tmp_path):
    model_root = tmp_path / "wflow_model"
    staticgeoms = model_root / "staticgeoms"
    staticgeoms.mkdir(parents=True)
    staticmaps_path = model_root / "staticmaps.nc"
    xr.Dataset(
        {
            "river_mask": (
                ("y", "x"),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
                {"_FillValue": np.uint8(0)},
            ),
            "subcatchment": (
                ("y", "x"),
                np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int32),
                {"_FillValue": np.int32(0)},
            ),
            "gauges_sfincs": (
                ("y", "x"),
                np.zeros((3, 3), dtype=np.int32),
                {"_FillValue": np.int32(0)},
            ),
        },
        coords={"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    ).to_netcdf(staticmaps_path)
    gpd.GeoDataFrame(
        {
            "index": [1, 2],
            "sfincs_handoff_id": ["handoff_01", "handoff_02"],
            "name": ["handoff_01", "handoff_02"],
        },
        geometry=[Point(0.0, 2.0), Point(2.0, 0.0)],
        crs="EPSG:4326",
    ).to_file(staticgeoms / "gauges_sfincs.geojson", driver="GeoJSON")

    repair_wflow_gauge_map(model_root)

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        values = raw["gauges_sfincs"].values
        active_river = raw["river_mask"].values > 0
        assert set(np.unique(values)) == {0, 1, 2}
        assert bool(np.any((values == 1) & active_river))
        assert bool(np.any((values == 2) & active_river))
        assert raw["gauges_sfincs"].dtype == np.int32
        assert raw["gauges_sfincs"].attrs["_FillValue"] == 0
    finally:
        raw.close()


def test_wflow_build_plan_reads_hydromt_configs_and_flags_review_required_bbox():
    repo_root = Path(__file__).resolve().parents[2]
    location_root = repo_root / "locations/greensboro"
    config = define_location(location_root / "config.yaml").config
    config["wflow"]["domain_set"] = {"submodels": [], "review_required": True}

    plan = build_wflow_build_plan(config, {"location_root": location_root})

    assert plan.study_location == "greensboro"
    assert plan.plugin == "wflow_sbm"
    assert plan.base_model_root == location_root / "data/wflow/base"
    assert plan.events_root == location_root / "data/wflow/events"
    assert plan.data_catalog == location_root / "data/wflow/data_catalog.yml"
    assert plan.build_steps[:3] == ("setup_config", "setup_basemaps", "setup_rivers")
    assert plan.update_steps == (
        "setup_config",
        "setup_precip_forcing",
        "setup_temp_pet_forcing",
    )
    assert plan.region_kind == "bbox"
    assert plan.review_required is True
    assert plan.domain_status == "review_required_bbox_placeholder"
    assert plan.build_command == (
        "hydromt build wflow_sbm "
        f"{location_root / 'data/wflow/base'} "
        f"-i {location_root / 'data/wflow/config/wflow_build.yml'} "
        f"-d {location_root / 'data/wflow/data_catalog.yml'} -vvv"
    )
    assert plan.update_command == (
        "hydromt update wflow_sbm "
        f"{location_root / 'data/wflow/base'} "
        f"-i {location_root / 'data/wflow/config/wflow_update_forcing.yml'} "
        f"-d {location_root / 'data/wflow/data_catalog.yml'} "
        f"-o {location_root / 'data/wflow/events/<event_id>'} -vvv"
    )


def test_austin_wflow_update_config_uses_hydromt_full_write_contract():
    repo_root = Path(__file__).resolve().parents[2]
    location_root = repo_root / "locations/austin"
    config = define_location(location_root / "config.yaml").config
    config["wflow"]["domain_set"] = {"submodels": [], "review_required": True}

    plan = build_wflow_build_plan(config, {"location_root": location_root})

    assert plan.update_steps == (
        "setup_config",
        "setup_precip_forcing",
        "setup_temp_pet_forcing",
    )


def test_prepare_replay_submodel_output_dir_cleans_stale_hydromt_model_dir(tmp_path):
    event_dir = tmp_path / "events" / "design_0398"
    out_dir = event_dir / "greensboro_rural"
    (out_dir / "run_event").mkdir(parents=True)
    (out_dir / "staticmaps.nc").write_text("partial HydroMT output\n", encoding="utf-8")
    (out_dir / "run_event" / "log.txt").write_text("partial Wflow output\n", encoding="utf-8")
    (event_dir / "precip.nc").write_text("event forcing stays\n", encoding="utf-8")

    _prepare_replay_submodel_output_dir(event_dir, out_dir)

    assert not out_dir.exists()
    assert (event_dir / "precip.nc").exists()


def test_prepare_replay_submodel_output_dir_refuses_event_root_or_external_paths(tmp_path):
    event_dir = tmp_path / "events" / "design_0398"
    event_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="refusing to clean replay output"):
        _prepare_replay_submodel_output_dir(event_dir, event_dir)
    with pytest.raises(ValueError, match="refusing to clean replay output"):
        _prepare_replay_submodel_output_dir(event_dir, tmp_path / "other")


def test_replay_inland_domain_set_plans_wflow_sbm_toml_run_config(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    event_id = "design_0398"
    base_root = location_root / "data/wflow/base/greensboro_rural"
    event_root = location_root / "data/wflow/events"
    config_dir = location_root / "data/wflow/config"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    base_root.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    catalog_path.parent.mkdir(parents=True)
    (base_root / "wflow_sbm.toml").write_text("[model]\n", encoding="utf-8")
    (config_dir / "wflow_update_forcing.yml").write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {"setup_config": {"data": {"dir_output": "run_event"}}},
                    {"setup_precip_forcing": {"precip_fn": "event_precip"}},
                    {"setup_temp_pet_forcing": {"temp_pet_fn": "event_temp_pet"}},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (location_root / "data/wflow/data_catalog.yml").parent.mkdir(parents=True, exist_ok=True)
    (location_root / "data/wflow/data_catalog.yml").write_text("{}\n", encoding="utf-8")
    catalog_path.write_text(
        "event_id,event_reference_time\n"
        "design_0398,2020-11-14T00:00:00\n",
        encoding="utf-8",
    )
    config = {
        "wflow": {
            "base_model_root": "data/wflow/base",
            "events_root": "data/wflow/events",
            "data_catalog": "data/wflow/data_catalog.yml",
            "update_forcing_config": "data/wflow/config/wflow_update_forcing.yml",
            "domain_set": {"submodels": [{"wflow_submodel_id": "greensboro_rural"}]},
            "run": {"command": "wflow_cli {run_config}"},
        }
    }

    report = replay_inland_domain_set(
        config,
        location_root,
        event_id,
        catalog_path=catalog_path,
        execute=False,
    )

    run_command = report.loc[0, "run_command"]
    assert run_command.endswith("data/wflow/events/design_0398/greensboro_rural/wflow_sbm.toml")
    assert report.loc[0, "run_output_dir"].endswith("data/wflow/events/design_0398/greensboro_rural/run_event")


def test_merge_submodel_discharge_uses_sfincs_handoff_point_override(tmp_path):
    run_output_dir = tmp_path / "run_event"
    run_output_dir.mkdir()
    pd.DataFrame(
        {"Q_5": [1.0, 2.0]},
        index=pd.to_datetime(["2020-01-01T00:00:00", "2020-01-01T01:00:00"]),
    ).to_csv(run_output_dir / "output.csv", index_label="time")
    gauges_geojson = tmp_path / "gauges_sfincs.geojson"
    gpd.GeoDataFrame(
        {
            "index": [5],
            "sfincs_handoff_id": ["greensboro_rural_inflow_05"],
        },
        geometry=[Point(10.0, 20.0)],
        crs="EPSG:32617",
    ).to_file(gauges_geojson, driver="GeoJSON")

    out_path = tmp_path / "sfincs_discharge.nc"
    merge_submodel_discharge(
        [{"run_output_dir": run_output_dir, "gauges_geojson": gauges_geojson}],
        model_crs="EPSG:32617",
        out_path=out_path,
        handoff_points={"greensboro_rural_inflow_05": (615286.0, 4000313.0)},
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds["name"].values.tolist() == ["greensboro_rural_inflow_05"]
        assert ds["x"].values.tolist() == [615286.0]
        assert ds["y"].values.tolist() == [4000313.0]
        assert ds["discharge"].values.tolist() == [[1.0, 2.0]]
    finally:
        ds.close()


def test_write_event_streamflow_handoff_discharge_uses_catalog_analog_timeseries(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    event_id = "design_0398"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    records_path = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    members_path = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    handoff_path = location_root / "data/sfincs/domains/greensboro_rural/base/gis/wflow_handoff_sources.geojson"
    stale_handoff_path = location_root / "data/sfincs/domains/greensboro_east/base/gis/wflow_handoff_sources.geojson"
    catalog_path.parent.mkdir(parents=True)
    records_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)
    stale_handoff_path.parent.mkdir(parents=True)

    catalog_path.write_text(
        "\n".join(
            [
                "event_id,event_reference_time,streamflow_member_id,streamflow_member_time,streamflow,streamflow_template_value,streamflow_scale_factor",
                "design_0398,2020-11-14T00:00:00,0212378405_20201114T000000,2020-11-14T00:00:00,10000,5000,2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    members_path.write_text(
        "\n".join(
            [
                "member_id,site_no,event_time,peak_flow_cfs,contributing_site_nos",
                '0212378405_20201114T000000,0212378405,2020-11-14T00:00:00,5000,"02094500,02095000"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "site_no": ["02094500", "02094500", "02094500", "02095000", "02095000", "02095000"],
            "time": pd.to_datetime(
                [
                    "2020-11-12T00:00:00",
                    "2020-11-14T00:00:00",
                    "2020-11-16T00:00:00",
                    "2020-11-12T00:00:00",
                    "2020-11-14T00:00:00",
                    "2020-11-16T00:00:00",
                ]
            ),
            "discharge_cfs": [100.0, 5000.0, 200.0, 50.0, 3000.0, 100.0],
            "source": ["usgs_dv"] * 6,
        }
    ).to_csv(records_path, index=False)
    gpd.GeoDataFrame(
        {
            "sfincs_domain_id": ["greensboro_rural", "greensboro_rural"],
            "sfincs_handoff_id": ["greensboro_rural_inflow_01", "greensboro_rural_inflow_02"],
            "handoff_placement": ["sfincs_native_river_inflow", "sfincs_native_river_inflow"],
            "uparea": [3.0, 1.0],
        },
        geometry=[Point(615000.0, 4000000.0), Point(616000.0, 4000000.0)],
        crs="EPSG:32617",
    ).to_file(handoff_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "sfincs_domain_id": ["greensboro_east"],
            "sfincs_handoff_id": ["greensboro_east_inflow_01"],
            "handoff_placement": ["sfincs_native_river_inflow"],
            "uparea": [99.0],
        },
        geometry=[Point(700000.0, 4000000.0)],
        crs="EPSG:32617",
    ).to_file(stale_handoff_path, driver="GeoJSON")

    out_path = location_root / "data/wflow/events/design_0398/sfincs_discharge.nc"
    write_event_streamflow_handoff_discharge(
        {
            "sfincs_domain_set": {"include_domain_ids": ["greensboro_rural"], "domains_root": "data/sfincs/domains"},
            "inland_coupling": {
                "discharge_forcing": {
                    "source": "event_streamflow_timeseries",
                    "handoff_location": "sfincs_native_river_inflow",
                }
            },
        },
        location_root,
        event_id,
        catalog_path=catalog_path,
        model_crs="EPSG:32617",
        out_path=out_path,
        start=pd.Timestamp("2020-11-12T00:00:00"),
        end=pd.Timestamp("2020-11-16T00:00:00"),
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["discharge_source"] == "event_streamflow_timeseries"
        assert ds["name"].values.tolist() == ["greensboro_rural_inflow_01", "greensboro_rural_inflow_02"]
        peak_cms = float(ds["discharge"].sum("index").max())
        assert peak_cms == pytest.approx(10000 * 0.028316846592)
        weights = ds["discharge"].isel(time=48).values / ds["discharge"].isel(time=48).sum().item()
        assert weights.tolist() == pytest.approx([0.75, 0.25])
        assert pd.Timestamp(ds["time"].values[48]).isoformat() == "2020-11-14T00:00:00"
    finally:
        ds.close()
    assert out_path.with_suffix(".provenance.json").exists()


def test_prepare_wflow_streamflow_realization_writes_native_external_inflow(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    event_id = "design_0398"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    records_path = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    members_path = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    gauges_path = location_root / "data/wflow/domain_set_gauges/greensboro_rural_observation_gauges.geojson"
    event_model_root = location_root / "data/wflow/events/design_0398/greensboro_rural"
    catalog_path.parent.mkdir(parents=True)
    records_path.parent.mkdir(parents=True)
    gauges_path.parent.mkdir(parents=True)
    event_model_root.mkdir(parents=True)

    catalog_path.write_text(
        "\n".join(
            [
                "event_id,event_reference_time,streamflow_member_id,streamflow_member_time,streamflow_template_value,streamflow_scale_factor",
                "design_0398,2020-11-14T00:00:00,02094500_20201114T000000,2020-11-14T00:00:00,100.0,2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    members_path.write_text(
        "\n".join(
            [
                "member_id,site_no,event_time,peak_flow_cfs,contributing_site_nos",
                "02094500_20201114T000000,02094500,2020-11-14T00:00:00,100.0,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "site_no": ["02094500", "02094500", "02094500"],
            "time": pd.to_datetime(["2020-11-12T00:00:00", "2020-11-14T00:00:00", "2020-11-16T00:00:00"]),
            "discharge_cfs": [10.0, 100.0, 20.0],
        }
    ).to_csv(records_path, index=False)
    gpd.GeoDataFrame(
        {"site_no": ["02094500"], "name": ["02094500"], "index": [1]},
        geometry=[Point(-79.95, 36.15)],
        crs="EPSG:4326",
    ).to_file(gauges_path, driver="GeoJSON")
    times = pd.date_range("2020-11-12T00:00:00", "2020-11-16T00:00:00", freq="h")
    xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
            "pet": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
            "temp": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
        },
        coords={"time": times, "latitude": [36.2, 36.15, 36.1], "longitude": [-80.0, -79.95, -79.9]},
    ).to_netcdf(event_model_root / "inmaps-event.nc")
    (event_model_root / "wflow_sbm.toml").write_text(
        'dir_output = "run_event"\n[input]\npath_forcing = "inmaps-event.nc"\n[input.forcing]\natmosphere_water__precipitation_volume_flux = "precip"\n',
        encoding="utf-8",
    )

    result = prepare_wflow_streamflow_realization_for_event_model(
        {},
        location_root,
        event_id,
        catalog_path=catalog_path,
        event_model_root=event_model_root,
        submodel_id="greensboro_rural",
        start=pd.Timestamp("2020-11-12T00:00:00"),
        end=pd.Timestamp("2020-11-16T00:00:00"),
    )

    assert result["streamflow_realization"] == "wflow_external_river_inflow"
    text = (event_model_root / "wflow_sbm.toml").read_text(encoding="utf-8")
    assert 'river_water__external_inflow_volume_flow_rate = "river_inflow"' in text
    with xr.open_dataset(event_model_root / "inmaps-event.nc") as ds:
        assert "river_inflow" in ds
        assert float(ds["river_inflow"].max()) == pytest.approx(200.0 * 0.028316846592)
        assert ds["river_inflow"].attrs["standard_name"] == "river_water__external_inflow_volume_flow_rate"
    provenance = json.loads((event_model_root / "streamflow_realization.provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_sites"] == ["02094500"]


def test_prepare_wflow_streamflow_realization_prefers_cached_instantaneous_usgs(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    event_id = "design_0398"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    records_path = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    iv_root = location_root / "data/sources/usgs_streamgages/event_streamflow_iv"
    members_path = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    gauges_path = location_root / "data/wflow/domain_set_gauges/greensboro_rural_observation_gauges.geojson"
    event_model_root = location_root / "data/wflow/events/design_0398/greensboro_rural"
    catalog_path.parent.mkdir(parents=True)
    records_path.parent.mkdir(parents=True)
    iv_root.mkdir(parents=True)
    gauges_path.parent.mkdir(parents=True)
    event_model_root.mkdir(parents=True)

    catalog_path.write_text(
        "\n".join(
            [
                "event_id,event_reference_time,streamflow_member_id,streamflow_member_time,streamflow_template_value,streamflow_scale_factor",
                "design_0398,2020-11-14T00:00:00,02094500_20201114T000000,2020-11-14T00:00:00,100.0,2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    members_path.write_text(
        "\n".join(
            [
                "member_id,site_no,event_time,peak_flow_cfs,contributing_site_nos",
                "02094500_20201114T000000,02094500,2020-11-14T00:00:00,100.0,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "site_no": ["02094500", "02094500", "02094500"],
            "time": pd.to_datetime(["2020-11-12T00:00:00", "2020-11-14T00:00:00", "2020-11-16T00:00:00"]),
            "discharge_cfs": [10.0, 100.0, 20.0],
            "source": ["usgs_dv", "usgs_dv", "usgs_dv"],
        }
    ).to_csv(records_path, index=False)
    cached_iv_path = iv_root / "design_0398_02094500_20201114T000000_20201112T000000_20201116T000000.csv"
    pd.DataFrame(
        {
            "site_no": ["02094500", "02094500", "02094500"],
            "time": pd.to_datetime(["2020-11-13T23:00:00", "2020-11-14T00:00:00", "2020-11-14T01:00:00"]),
            "discharge_cfs": [5.0, 60.0, 10.0],
            "source": ["usgs_iv", "usgs_iv", "usgs_iv"],
        }
    ).to_csv(cached_iv_path, index=False)
    gpd.GeoDataFrame(
        {"site_no": ["02094500"], "name": ["02094500"], "index": [1]},
        geometry=[Point(-79.95, 36.15)],
        crs="EPSG:4326",
    ).to_file(gauges_path, driver="GeoJSON")
    times = pd.date_range("2020-11-12T00:00:00", "2020-11-16T00:00:00", freq="h")
    xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
            "pet": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
            "temp": (("time", "latitude", "longitude"), np.zeros((len(times), 3, 3), dtype="float32")),
        },
        coords={"time": times, "latitude": [36.2, 36.15, 36.1], "longitude": [-80.0, -79.95, -79.9]},
    ).to_netcdf(event_model_root / "inmaps-event.nc")
    (event_model_root / "wflow_sbm.toml").write_text(
        'dir_output = "run_event"\n[input]\npath_forcing = "inmaps-event.nc"\n[input.forcing]\natmosphere_water__precipitation_volume_flux = "precip"\n',
        encoding="utf-8",
    )

    prepare_wflow_streamflow_realization_for_event_model(
        {
            "wflow": {
                "streamflow_realization": {
                    "event_records_root": "data/sources/usgs_streamgages/event_streamflow_iv",
                    "require_instantaneous_usgs": True,
                }
            }
        },
        location_root,
        event_id,
        catalog_path=catalog_path,
        event_model_root=event_model_root,
        submodel_id="greensboro_rural",
        start=pd.Timestamp("2020-11-12T00:00:00"),
        end=pd.Timestamp("2020-11-16T00:00:00"),
    )

    with xr.open_dataset(event_model_root / "inmaps-event.nc") as ds:
        assert "river_inflow" in ds
        assert float(ds["river_inflow"].max()) == pytest.approx(120.0 * 0.028316846592)
    provenance = json.loads((event_model_root / "streamflow_realization.provenance.json").read_text(encoding="utf-8"))
    assert provenance["records_path"] == str(cached_iv_path)
    assert provenance["records_source"] == "usgs_iv:3"


def test_cached_wbd_hucs_reuses_union_huc_artifact(tmp_path):
    location_root = tmp_path
    huc_root = location_root / "data/wflow/domain_huc"
    huc_root.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "huc_id": ["03030002_03010104"],
            "huc_level": [8],
            "huc_kind": ["union"],
        },
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:4326",
    ).to_file(huc_root / "greensboro_east.geojson", driver="GeoJSON")
    config = {"wflow": {"domain_set": {"huc": {"root": "data/wflow/domain_huc"}}}}

    cached = _cached_wbd_hucs(config, location_root, box(0.5, 0.5, 1.5, 1.5), 8)

    assert cached is not None
    assert cached["huc_id"].tolist() == ["03030002_03010104"]
    assert cached["huc_kind"].tolist() == ["union"]


def test_build_meteo_writes_scaled_precip_and_neutral_pet(tmp_path):
    location_root = tmp_path / "locations/test"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    storm_root = location_root / "data/sources/aorc_sst/test/72hr-events"
    event_windows = storm_root / "event_windows"
    event_windows.mkdir(parents=True)
    catalog_path.parent.mkdir(parents=True)
    member_file = storm_root / "ranked-storms.csv"
    member_file.write_text("member_id\nrainfall_test_72h_rank0001\n", encoding="utf-8")

    source_nc = event_windows / "rainfall_test_72h_rank0001_20200101T00.nc"
    source_time = pd.date_range("2020-01-01T00:00:00", periods=2, freq="h")
    source_coords = {"time": source_time, "latitude": [36.0], "longitude": [-79.0]}
    xr.Dataset(
        {
            "APCP_surface": (("time", "latitude", "longitude"), [[[2.0]], [[4.0]]]),
            "TMP_2maboveground": (("time", "latitude", "longitude"), np.full((2, 1, 1), 293.15)),
            "PRES_surface": (("time", "latitude", "longitude"), np.full((2, 1, 1), 101325.0)),
            "DSWRF_surface": (("time", "latitude", "longitude"), np.full((2, 1, 1), 120.0)),
        },
        coords=source_coords,
    ).to_netcdf(source_nc)

    pd.DataFrame(
        [
            {
                "event_id": "design_0001",
                "event_reference_time": "2020-05-01T00:00:00",
                "rainfall_member_id": "rainfall_test_72h_rank0001",
                "rainfall_member_file": str(member_file),
                "rainfall_member_time": "2020-01-01T00:00:00",
                "rainfall_start_offset_hours": -1,
                "rainfall_scale_factor": 3.0,
            }
        ]
    ).to_csv(catalog_path, index=False)
    config = {
        "wflow": {"events_root": "data/wflow/events"},
        "collection": {"aorc_sst": {"variable": "APCP_surface"}},
    }

    report = build_meteo(
        config,
        location_root,
        "design_0001",
        catalog_path=catalog_path,
        pre_event_hours=2,
        post_event_hours=1,
    )

    precip = xr.open_dataset(report["precip_path"])
    temp_pet = xr.open_dataset(report["temp_pet_path"])
    assert report["precip_written"] is True
    assert report["temp_pet_written"] is True
    assert report["rainfall_source_nc"] == str(source_nc)
    assert float(precip["precip"].sel(time="2020-04-30T22:00:00").sum()) == 0.0
    assert float(precip["precip"].sel(time="2020-04-30T23:00:00").sum()) == 6.0
    assert float(precip["precip"].sel(time="2020-05-01T00:00:00").sum()) == 12.0
    assert set(temp_pet.data_vars) == {"temp", "press_msl", "kin"}
    assert float(temp_pet["kin"].max()) == 120.0
    assert float(temp_pet["temp"].isel(time=0, y=0, x=0)) == pytest.approx(20.0)
    assert float(temp_pet["press_msl"].isel(time=0, y=0, x=0)) == pytest.approx(1013.25)
    assert Path(report["temp_pet_provenance"]).exists()

    second = build_meteo(config, location_root, "design_0001", catalog_path=catalog_path)
    assert second["precip_written"] is False
    assert second["temp_pet_written"] is False


def test_build_meteo_shifts_temp_pet_when_catalog_has_no_rainfall_offset(tmp_path):
    location_root = tmp_path / "locations/test"
    catalog_path = location_root / "data/event_catalog/catalog/probability_catalog.csv"
    storm_root = location_root / "data/sources/aorc_sst/test/72hr-events"
    event_windows = storm_root / "event_windows"
    event_windows.mkdir(parents=True)
    catalog_path.parent.mkdir(parents=True)
    member_file = storm_root / "ranked-storms.csv"
    member_file.write_text("member_id\nrainfall_test_72h_rank0001\n", encoding="utf-8")

    source_nc = event_windows / "rainfall_test_72h_rank0001_20090105T06.nc"
    source_time = pd.date_range("2009-01-05T06:00:00", periods=72, freq="h")
    source_coords = {"time": source_time, "latitude": [36.0], "longitude": [-79.0]}
    xr.Dataset(
        {
            "APCP_surface": (("time", "latitude", "longitude"), np.ones((72, 1, 1), dtype="float32")),
            "TMP_2maboveground": (("time", "latitude", "longitude"), np.full((72, 1, 1), 293.15)),
            "PRES_surface": (("time", "latitude", "longitude"), np.full((72, 1, 1), 101325.0)),
            "DSWRF_surface": (("time", "latitude", "longitude"), np.full((72, 1, 1), 120.0)),
        },
        coords=source_coords,
    ).to_netcdf(source_nc)

    pd.DataFrame(
        [
            {
                "event_id": "design_0007",
                "event_reference_time": "2014-05-16T00:00:00",
                "rainfall_member_id": "rainfall_test_72h_rank0001",
                "rainfall_member_file": str(member_file),
                "rainfall_member_time": "2009-01-05T06:00:00",
                "rainfall_scale_factor": 1.0,
            }
        ]
    ).to_csv(catalog_path, index=False)
    config = {
        "wflow": {"events_root": "data/wflow/events"},
        "collection": {"aorc_sst": {"variable": "APCP_surface"}},
    }

    report = build_meteo(
        config,
        location_root,
        "design_0007",
        catalog_path=catalog_path,
        pre_event_hours=48,
        post_event_hours=72,
    )

    with xr.open_dataset(report["temp_pet_path"]) as temp_pet:
        assert temp_pet.sizes["time"] == 121
        assert pd.Timestamp(temp_pet["time"].min().values) == pd.Timestamp("2014-05-14T00:00:00")
        assert pd.Timestamp(temp_pet["time"].max().values) == pd.Timestamp("2014-05-19T00:00:00")
        assert float(temp_pet["temp"].isel(time=0, y=0, x=0)) == pytest.approx(20.0)


def test_hydromt_command_resolver_skips_stale_project_console_script(monkeypatch, tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    location_root.mkdir(parents=True)
    stale = tmp_path / ".venv" / "bin" / "hydromt"
    stale.parent.mkdir(parents=True)
    stale.write_text("#!/missing/python\n", encoding="utf-8")
    stale.chmod(0o755)
    monkeypatch.setattr("wflow_runs.notebook.shutil.which", lambda name: None)
    monkeypatch.setattr("wflow_runs.notebook._project_hydromt_candidates", lambda _: (stale,))
    monkeypatch.setattr("wflow_runs.notebook._project_python_candidates", lambda _: ())

    command = _resolve_hydromt_command("hydromt update wflow_sbm base -i update.yml", location_root)

    assert command[:3] == [sys.executable, "-m", "hydromt.cli.main"]
    assert command[3:] == ["update", "wflow_sbm", "base", "-i", "update.yml"]


def test_hydromt_command_resolver_skips_console_script_from_other_venv(monkeypatch, tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    location_root.mkdir(parents=True)
    flood_rm_python = tmp_path / ".venv" / "bin" / "python"
    flood_rm_python.parent.mkdir(parents=True)
    flood_rm_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    flood_rm_python.chmod(0o755)
    surge_python = tmp_path / "SURGE" / ".venv" / "bin" / "python3"
    surge_python.parent.mkdir(parents=True)
    surge_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    surge_python.chmod(0o755)
    stale_hydromt = tmp_path / ".venv" / "bin" / "hydromt"
    stale_hydromt.write_text(f"#!{surge_python}\n", encoding="utf-8")
    stale_hydromt.chmod(0o755)
    monkeypatch.setattr("wflow_runs.notebook.shutil.which", lambda name: None)

    command = _resolve_hydromt_command("hydromt update wflow_sbm base -i update.yml", location_root)

    assert command[:3] == [str(flood_rm_python), "-m", "hydromt.cli.main"]
    assert command[3:] == ["update", "wflow_sbm", "base", "-i", "update.yml"]


def test_hydromt_subprocess_env_sanitizes_non_numeric_debug(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")

    env = _hydromt_subprocess_env()

    assert env["DEBUG"] == "0"
    assert env["MPLCONFIGDIR"] == "/tmp/matplotlib"


def test_wflow_event_replay_helpers_substitute_event_id_without_executing(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    config_dir = location_root / "data/wflow/config"
    config_dir.mkdir(parents=True)
    hydromt_cli = tmp_path / ".venv/bin/hydromt"
    hydromt_cli.parent.mkdir(parents=True)
    hydromt_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hydromt_cli.chmod(0o755)
    (config_dir / "wflow_build.yml").write_text(
        yaml.safe_dump(
            {"steps": [{"setup_config": {}}, {"setup_basemaps": {"region": {"bbox": [-80, 36, -79, 37]}}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (config_dir / "wflow_update_forcing.yml").write_text(
        yaml.safe_dump(
            {"steps": [{"setup_config": {}}, {"setup_precip_forcing": {}}, {"setup_temp_pet_forcing": {}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = {
        "project": {"name": "greensboro"},
        "wflow": {
            "plugin": "wflow_sbm",
            "base_model_root": "data/wflow/base",
            "events_root": "data/wflow/events",
            "data_catalog": "data/wflow/data_catalog.yml",
            "build_config": "data/wflow/config/wflow_build.yml",
            "update_forcing_config": "data/wflow/config/wflow_update_forcing.yml",
            "domain_set": {"submodels": [], "review_required": True},
        },
    }
    monkeypatch.setattr("wflow_runs.notebook.shutil.which", lambda _: None)

    plan = wflow_event_replay_plan(config, location_root, "evt_001")
    result = run_wflow_event_replay(config, location_root, "evt_001", execute=False)

    assert "data/wflow/events/evt_001" in plan["hydromt_wflow_update_command"]
    assert str(hydromt_cli) in plan["resolved_hydromt_wflow_update_command"]
    assert plan["hydromt_runner_status"] == "project_venv"
    assert result["status"] == "dry_run"
    assert result["resolved_command"].startswith(str(hydromt_cli))
    assert result["wflow_discharge_forcing"].endswith("data/wflow/events/evt_001/sfincs_discharge.nc")


def test_wflow_event_replay_execute_reports_missing_hydromt_cli(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    config_dir = location_root / "data/wflow/config"
    config_dir.mkdir(parents=True)
    (config_dir / "wflow_build.yml").write_text(
        yaml.safe_dump(
            {"steps": [{"setup_config": {}}, {"setup_basemaps": {"region": {"bbox": [-80, 36, -79, 37]}}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (config_dir / "wflow_update_forcing.yml").write_text(
        yaml.safe_dump(
            {"steps": [{"setup_config": {}}, {"setup_precip_forcing": {}}, {"setup_temp_pet_forcing": {}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = {
        "project": {"name": "greensboro"},
        "wflow": {
            "plugin": "wflow_sbm",
            "base_model_root": "data/wflow/base",
            "events_root": "data/wflow/events",
            "data_catalog": "data/wflow/data_catalog.yml",
            "build_config": "data/wflow/config/wflow_build.yml",
            "update_forcing_config": "data/wflow/config/wflow_update_forcing.yml",
            "domain_set": {"submodels": [], "review_required": True},
        },
    }
    monkeypatch.setattr("wflow_runs.notebook.shutil.which", lambda _: None)
    monkeypatch.setattr("wflow_runs.notebook._project_hydromt_candidates", lambda _: ())
    monkeypatch.setattr("wflow_runs.notebook._project_python_candidates", lambda _: ())
    monkeypatch.setattr("wflow_runs.notebook.importlib.util.find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="HydroMT CLI executable not found"):
        run_wflow_event_replay(config, location_root, "evt_001", execute=True)


def test_wflow_domain_set_plan_groups_reviewed_handoff_gages_into_subbasin_submodels(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    sfincs_manifest_path = location_root / "data/sfincs/domains/domain_set.yaml"
    network_path.parent.mkdir(parents=True)
    sfincs_manifest_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "Reedy Fork near Oak Ridge",
        "status": "active",
        "drainage_area_sqmi": 20.1,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "greensboro-main",
        "wflow_submodel_id": "reedy_fork",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "reedy_fork_inflow",
        "review_status": "accepted",
        "review_notes": "Accepted as the reviewed Wflow subbasin outlet."
      },
      "geometry": {"type": "Point", "coordinates": [-79.998, 36.175]}
    },
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095500",
        "site_name": "Buffalo Creek near Greensboro",
        "status": "active",
        "drainage_area_sqmi": 12.4,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.91,
        "roles": ["validation"],
        "frequency_basis": "greensboro-main",
        "wflow_submodel_id": "reedy_fork",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": null,
        "review_status": "accepted_with_warning",
        "review_notes": "Validation only; not a SFINCS handoff point."
      },
      "geometry": {"type": "Point", "coordinates": [-79.87, 36.09]}
    },
    {
      "type": "Feature",
      "properties": {
        "site_no": "02099000",
        "site_name": "East Fork Deep River near High Point",
        "status": "active",
        "drainage_area_sqmi": 14.8,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.99,
        "roles": ["frequency", "validation"],
        "frequency_basis": "regional-reference",
        "wflow_submodel_id": "deep_river",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": null,
        "review_status": "accepted_with_warning",
        "review_notes": "Regional frequency reference only; not a coupled domain outlet."
      },
      "geometry": {"type": "Point", "coordinates": [-80.01, 36.00]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    sfincs_manifest_path.write_text(
        yaml.safe_dump(
            {
                "domains": [
                    {
                        "sfincs_domain_id": "greensboro_east",
                        "handoff_source_ids": ["reedy_fork_inflow"],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = {
        "sfincs_domain_set": {"domain_manifest": "data/sfincs/domains/domain_set.yaml"},
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {"event_catalog_scope": "shared_across_domain_set"},
        }
    }

    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.handoff_count == 1
    assert plan.gage_count == 3
    assert plan.submodels[0]["region"] == {
        "subbasin": [-79.998, 36.175],
        "uparea": pytest.approx(52.058, rel=1.0e-3),
    }

    manifest_path = write_wflow_domain_set_manifest(plan, config, {"location_root": location_root})
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["event_catalog_scope"] == "shared_across_domain_set"
    assert manifest["submodels"][0]["hydromt_region"]["subbasin"] == [-79.998, 36.175]
    assert manifest["submodels"][0]["hydromt_region"]["uparea"] == pytest.approx(52.058, rel=1.0e-3)
    assert plan.submodels[0]["sfincs_domain_ids"] == ("greensboro_main",)
    assert manifest["submodels"][0]["sfincs_domain_ids"] == ["greensboro_east"]
    assert manifest["submodels"][0]["sfincs_handoff_ids"] == ["reedy_fork_inflow"]


def test_wflow_domain_set_plan_can_collapse_to_one_bbox_model(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    build_config = location_root / "data/wflow/config/wflow_build.yml"
    network_path.parent.mkdir(parents=True)
    build_config.parent.mkdir(parents=True)
    build_config.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {
                        "setup_basemaps": {
                            "region": {"bbox": [-80.1, 35.9, -79.7, 36.2]},
                            "hydrography_fn": "us_hydrography_basemap",
                        }
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "South Buffalo Creek near Greensboro",
        "status": "active",
        "drainage_area_sqmi": 34.0,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "south_buffalo",
        "wflow_submodel_id": "south_buffalo",
        "sfincs_domain_id": "greensboro_west",
        "sfincs_handoff_id": "south_buffalo_02095000",
        "review_status": "accepted",
        "review_notes": "handoff"
      },
      "geometry": {"type": "Point", "coordinates": [-79.75, 36.05]}
    },
    {
      "type": "Feature",
      "properties": {
        "site_no": "02094500",
        "site_name": "Reedy Fork near Gibsonville",
        "status": "active",
        "drainage_area_sqmi": 131.0,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.95,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "reedy_fork",
        "wflow_submodel_id": "reedy_fork",
        "sfincs_domain_id": "greensboro_east",
        "sfincs_handoff_id": "reedy_fork_02094500",
        "review_status": "accepted_with_warning",
        "review_notes": "handoff"
      },
      "geometry": {"type": "Point", "coordinates": [-79.61, 36.17]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    config = {
        "project": {"name": "greensboro"},
        "wflow": {
            "build_config": "data/wflow/config/wflow_build.yml",
            "domain_set": {
                "allow_multiple_submodels": False,
                "single_submodel_id": "greensboro_main",
                "event_catalog_scope": "shared_across_domain_set",
            },
            "streamgage_network": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
            },
        },
    }

    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodel_count == 1
    assert plan.handoff_count == 2
    assert plan.submodels[0]["wflow_submodel_id"] == "greensboro_main"
    assert plan.submodels[0]["region_kind"] == "bbox"
    assert plan.submodels[0]["region"] == {"bbox": [-80.1, 35.9, -79.7, 36.2]}
    assert plan.submodels[0]["sfincs_domain_ids"] == ("greensboro_east", "greensboro_west")
    assert plan.submodels[0]["sfincs_handoff_ids"] == (
        "reedy_fork_02094500",
        "south_buffalo_02095000",
    )

    manifest_path = write_wflow_domain_set_manifest(plan, config, {"location_root": location_root})
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["submodels"]) == 1
    assert manifest["submodels"][0]["hydromt_region"] == {"bbox": [-80.1, 35.9, -79.7, 36.2]}


def test_wflow_domain_set_plan_ignores_nhdplus_bounds_for_hydromt_region(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    subbasin_path = location_root / "data/wflow/domain_set_subbasins/reedy_fork.geojson"
    network_path.parent.mkdir(parents=True)
    subbasin_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "Reedy Fork near Oak Ridge",
        "status": "active",
        "drainage_area_sqmi": 20.1,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "greensboro-main",
        "wflow_submodel_id": "reedy_fork",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "reedy_fork_inflow",
        "review_status": "accepted",
        "review_notes": "Accepted as the reviewed Wflow subbasin outlet."
      },
      "geometry": {"type": "Point", "coordinates": [-79.998, 36.175]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    gpd.GeoDataFrame(
        {"wflow_submodel_id": ["reedy_fork"]},
        geometry=[
            Polygon(
                [
                    (-80.01, 36.15),
                    (-79.96, 36.15),
                    (-79.96, 36.20),
                    (-80.01, 36.20),
                    (-80.01, 36.15),
                ]
            )
        ],
        crs="EPSG:4326",
    ).to_file(subbasin_path, driver="GeoJSON")
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {"subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg"},
        }
    }

    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.submodels[0]["region_kind"] == "subbasin"
    assert plan.submodels[0]["region"] == {
        "subbasin": [-79.998, 36.175],
        "uparea": pytest.approx(52.058, rel=1.0e-3),
    }
    assert plan.submodels[0]["outlet_region"] == {"subbasin": [-79.998, 36.175]}
    assert plan.submodels[0]["subbasin_geometry"] is None

    manifest_path = write_wflow_domain_set_manifest(plan, config, {"location_root": location_root})
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["submodels"][0]["hydromt_region"]["subbasin"] == [-79.998, 36.175]
    assert "bounds" not in manifest["submodels"][0]["hydromt_region"]
    assert manifest["submodels"][0]["handoff_outlet_region"] == {"subbasin": [-79.998, 36.175]}
    assert manifest["submodels"][0]["subbasin_geometry"] is None


def test_wflow_domain_set_manifest_requires_ready_reviewed_subbasin_plan(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    config = {
        "wflow": {
            "domain_set_manifest": "data/wflow/domain_set.yaml",
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
        }
    }
    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})

    with pytest.raises(ValueError, match="not ready"):
        write_wflow_domain_set_manifest(plan, config, {"location_root": location_root})


def test_write_wflow_subbasin_fabric_from_nhdplus_preserves_boundary_metadata(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchment_path = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    network_path.parent.mkdir(parents=True)
    river_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "South Buffalo Creek near Greensboro",
        "status": "active",
        "drainage_area_sqmi": 42.0,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "south_buffalo",
        "wflow_submodel_id": "south_buffalo",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "south_buffalo_02095000",
        "review_status": "accepted",
        "review_notes": "Accepted as the reviewed Wflow-SFINCS handoff outlet."
      },
      "geometry": {"type": "Point", "coordinates": [-79.75, 36.05]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    gpd.GeoDataFrame(
        {
            "NHDPlusID": [100, 200, 300],
            "HydroSeq": [10, 20, 30],
            "DnHydroSeq": [0, 10, 20],
            "TotDASqKm": [30.0, 20.0, 10.0],
        },
        geometry=[
            LineString([(-79.76, 36.04), (-79.74, 36.06)]),
            LineString([(-79.78, 36.06), (-79.76, 36.04)]),
            LineString([(-79.80, 36.08), (-79.78, 36.06)]),
        ],
        crs="EPSG:4326",
    ).to_file(river_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"featureid": [100, 200, 300]},
        geometry=[
            Polygon([(-79.77, 36.03), (-79.73, 36.03), (-79.73, 36.07), (-79.77, 36.07), (-79.77, 36.03)]),
            Polygon([(-79.79, 36.05), (-79.75, 36.05), (-79.75, 36.09), (-79.79, 36.09), (-79.79, 36.05)]),
            Polygon([(-79.81, 36.07), (-79.77, 36.07), (-79.77, 36.11), (-79.81, 36.11), (-79.81, 36.07)]),
        ],
        crs="EPSG:4326",
    ).to_file(catchment_path, driver="GPKG")
    config = {
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
            }
        },
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {
                "subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg",
                "subbasin_fabric_diagnostics": "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
            },
            "handoff": {
                "source_variable": "river_q",
                "source_standard_name": "river_water__volume_flow_rate",
                "target": "sfincs_discharge_forcing",
            },
        },
    }

    result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})

    fabric = gpd.read_file(result["subbasin_fabric"])
    assert result["submodel_count"] == 1
    assert result["catchment_count"] == 3
    assert result["subbasin_geometry_files"] == (
        location_root / "data/wflow/domain_set_subbasins/south_buffalo.geojson",
    )
    assert result["subbasin_geometry_files"][0].exists()
    assert fabric.loc[0, "wflow_submodel_id"] == "south_buffalo"
    assert fabric.loc[0, "sfincs_boundary_id"] == "south_buffalo_02095000"
    assert fabric.loc[0, "sfincs_boundary_type"] == "discharge"
    assert fabric.loc[0, "sfincs_forcing_target"] == "sfincs_discharge_forcing"
    assert fabric.loc[0, "wflow_source_variable"] == "river_q"
    assert fabric.loc[0, "wflow_source_standard_name"] == "river_water__volume_flow_rate"
    assert fabric.loc[0, "aggregation_method"] == "routed_upstream_catchments"
    assert fabric.loc[0, "area_match_status"] == "review_required_area_mismatch"
    assert result["area_mismatch_count"] == 1
    assert result["area_mismatch_submodels"] == ("south_buffalo",)
    assert result["diagnostics_csv"].exists()


def test_boundary_handoff_watershed_fabric_unions_all_inflow_catchments(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchment_path = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    river_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "NHDPlusID": [100, 200, 300],
            "HydroSeq": [10, 20, 30],
            "DnHydroSeq": [0, 10, 0],
            "TotDASqKm": [100.0, 40.0, 80.0],
        },
        geometry=[
            LineString([(15.0, 5.0), (5.0, 5.0)]),
            LineString([(17.0, 5.0), (15.0, 5.0)]),
            LineString([(12.0, 8.0), (8.0, 8.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(river_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"featureid": [100, 200, 300]},
        geometry=[
            Polygon([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)]),
            Polygon([(6.0, 4.0), (8.0, 4.0), (8.0, 6.0), (6.0, 6.0), (6.0, 4.0)]),
            Polygon([(7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0), (7.0, 7.0)]),
        ],
        crs="EPSG:4326",
    ).to_file(catchment_path, driver="GPKG")
    config = {
        "project": {"name": "greensboro"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
            }
        },
        "wflow": {
            "domain_set": {
                "outlet_source": "boundary_handoff_watershed",
                "crossings": {"min_uparea_km2": 5.0},
                "subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg",
                "subbasin_fabric_diagnostics": "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
            },
            "handoff": {"source_variable": "river_q", "target": "sfincs_discharge_forcing"},
        },
    }

    result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})

    assert result["submodel_count"] == 1
    assert result["catchment_count"] == 3
    assert result["subbasin_geometry_files"] == (
        location_root / "data/wflow/domain_set_subbasins/greensboro_main.geojson",
    )
    fabric = gpd.read_file(result["subbasin_fabric"])
    assert len(fabric) == 1
    assert fabric["wflow_submodel_id"].tolist() == ["greensboro_main"]
    assert fabric["sfincs_boundary_id"].iloc[0] == "greensboro_main_inflow_01,greensboro_main_inflow_02"
    assert fabric["aggregation_method"].iloc[0] == "routed_upstream_catchments"


def test_boundary_handoff_watershed_region_includes_reference_gage_anchor(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    river_path.parent.mkdir(parents=True, exist_ok=True)
    network_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs="EPSG:4326",
    ).to_file(bbox_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "NHDPlusID": [100],
            "HydroSeq": [10],
            "DnHydroSeq": [0],
            "TotDASqKm": [100.0],
        },
        geometry=[LineString([(15.0, 5.0), (5.0, 5.0)])],
        crs="EPSG:4326",
    ).to_file(river_path, driver="GPKG")
    gpd.GeoDataFrame(
        {
            "site_no": ["02094500"],
            "site_name": ["Reference gage"],
            "status": ["active"],
            "drainage_area_sqmi": [131.0],
            "period_start": ["1979-02-01"],
            "period_end": ["2022-12-31"],
            "record_years": [43.9],
            "completeness_score": [0.98],
            "roles": [["frequency", "calibration", "validation"]],
            "frequency_basis": ["greensboro_main"],
            "wflow_submodel_id": ["greensboro_main"],
            "sfincs_domain_id": ["greensboro_main"],
            "sfincs_handoff_id": [None],
            "review_status": ["accepted"],
            "review_notes": ["Downstream reference gage."],
        },
        geometry=[Point(20.0, 5.0)],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    config = {
        "project": {"name": "greensboro"},
        "static_sources": {"bbox": {"output": "data/static/aoi/bbox.geojson"}},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
            }
        },
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {
                "outlet_source": "boundary_handoff_watershed",
                "crossings": {"min_uparea_km2": 5.0},
            },
        },
        "inland_coupling": {
            "amplification": {"primary_reference_gage": "02094500"},
            "baseflow": {"reference_gage": "02094500"},
        },
    }

    plan = plan_wflow_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    submodel = plan.submodels[0]
    assert submodel["gauge_site_nos"] == ("02094500",)
    assert submodel["region"]["subbasin"][0][-1] == pytest.approx(20.0)
    assert submodel["region"]["subbasin"][1][-1] == pytest.approx(5.0)
    assert submodel["outlet_region"]["subbasin"] != submodel["region"]["subbasin"]


def test_observation_gauge_layer_requires_configured_reference_gage(tmp_path):
    model_root = tmp_path / "wflow_model"
    gauges_path = model_root / "staticgeoms" / "gauges_usgs.geojson"
    gauges_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"site_no": ["02094500"]},
        geometry=[Point(-79.61, 36.17)],
        crs="EPSG:4326",
    ).to_file(gauges_path, driver="GeoJSON")
    submodel = {"gauge_site_nos": ("02094500", "02095000")}

    assert _observation_gauge_layer_matches(
        model_root,
        submodel,
        {"inland_coupling": {"amplification": {"primary_reference_gage": "02094500"}}},
    )
    assert not _observation_gauge_layer_matches(
        model_root,
        submodel,
        {"inland_coupling": {"amplification": {"primary_reference_gage": "02095000"}}},
    )


def test_subbasin_fabric_does_not_add_evaluation_coverage_region(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchment_path = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    footprint_path = location_root / "data/static/aoi/evaluation_footprint.geojson"
    power_path = location_root / "data/static/aoi/dft_power_extent.geojson"
    network_path.parent.mkdir(parents=True)
    river_path.parent.mkdir(parents=True)
    footprint_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "South Buffalo Creek near Greensboro",
        "status": "active",
        "drainage_area_sqmi": 42.0,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "calibration", "sfincs_handoff"],
        "frequency_basis": "south_buffalo",
        "wflow_submodel_id": "south_buffalo",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "south_buffalo_02095000",
        "review_status": "accepted",
        "review_notes": "Accepted as the reviewed Wflow-SFINCS handoff outlet."
      },
      "geometry": {"type": "Point", "coordinates": [-79.75, 36.05]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    gpd.GeoDataFrame(
        {
            "NHDPlusID": [100, 200, 300],
            "HydroSeq": [10, 20, 30],
            "DnHydroSeq": [0, 10, 20],
            "TotDASqKm": [30.0, 20.0, 10.0],
        },
        geometry=[
            LineString([(-79.76, 36.04), (-79.74, 36.06)]),
            LineString([(-79.78, 36.06), (-79.76, 36.04)]),
            LineString([(-79.80, 36.08), (-79.78, 36.06)]),
        ],
        crs="EPSG:4326",
    ).to_file(river_path, driver="GPKG")
    # Catchment 400 is east of the routed handoff drainage. It intersects the
    # SMART-DS footprint, but it does not drain to a SFINCS boundary handoff.
    gpd.GeoDataFrame(
        {"featureid": [100, 200, 300, 400]},
        geometry=[
            Polygon([(-79.77, 36.03), (-79.73, 36.03), (-79.73, 36.07), (-79.77, 36.07), (-79.77, 36.03)]),
            Polygon([(-79.79, 36.05), (-79.75, 36.05), (-79.75, 36.09), (-79.79, 36.09), (-79.79, 36.05)]),
            Polygon([(-79.81, 36.07), (-79.77, 36.07), (-79.77, 36.11), (-79.81, 36.11), (-79.81, 36.07)]),
            Polygon([(-79.71, 36.00), (-79.66, 36.00), (-79.66, 36.05), (-79.71, 36.05), (-79.71, 36.00)]),
        ],
        crs="EPSG:4326",
    ).to_file(catchment_path, driver="GPKG")
    # The footprint and power extent fall inside catchment 400, which is not in
    # the routed handoff drainage and should not expand the Wflow model domain.
    gpd.GeoDataFrame(
        {"region_id": ["greensboro"]},
        geometry=[Polygon([(-79.70, 36.01), (-79.67, 36.01), (-79.67, 36.04), (-79.70, 36.04), (-79.70, 36.01)])],
        crs="EPSG:4326",
    ).to_file(footprint_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"region_id": ["greensboro"]},
        geometry=[Polygon([(-79.695, 36.015), (-79.675, 36.015), (-79.675, 36.035), (-79.695, 36.035), (-79.695, 36.015)])],
        crs="EPSG:4326",
    ).to_file(power_path, driver="GeoJSON")
    stale_coverage = location_root / "data/wflow/domain_set_subbasins/evaluation_coverage.geojson"
    stale_coverage.parent.mkdir(parents=True, exist_ok=True)
    stale_coverage.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    config = {
        "smart_ds_evaluation_footprint": {"output": "data/static/aoi/evaluation_footprint.geojson"},
        "grid": {"power_extent": "data/static/aoi/dft_power_extent.geojson"},
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
            }
        },
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {
                "subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg",
                "subbasin_fabric_diagnostics": "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
            },
            "handoff": {"source_variable": "river_q", "target": "sfincs_discharge_forcing"},
        },
    }

    result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})

    # Routed handoff drainage is the whole Wflow domain: 1 submodel over the 3
    # upstream catchments. There is no separate intersecting-footprint watershed.
    assert result["submodel_count"] == 1
    assert result["catchment_count"] == 3
    assert result["subbasin_geometry_files"] == (
        location_root / "data/wflow/domain_set_subbasins/south_buffalo.geojson",
    )
    assert result["coverage_status"] == "handoff_watershed_only"
    assert result["coverage_catchment_count"] == 0
    assert result["coverage_region"] is None
    assert not stale_coverage.exists()

    fabric = gpd.read_file(result["subbasin_fabric"])
    assert set(fabric["region_role"]) == {"handoff_drainage"}
    assert set(fabric["wflow_submodel_id"]) == {"south_buffalo"}


def test_subbasin_fabric_skips_coverage_when_no_smart_ds_targets_configured(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchment_path = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    network_path.parent.mkdir(parents=True)
    river_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "South Buffalo Creek near Greensboro",
        "status": "active",
        "drainage_area_sqmi": 42.0,
        "period_start": "1979-02-01",
        "period_end": "2022-12-31",
        "record_years": 43.9,
        "completeness_score": 0.98,
        "roles": ["frequency", "sfincs_handoff"],
        "frequency_basis": "south_buffalo",
        "wflow_submodel_id": "south_buffalo",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "south_buffalo_02095000",
        "review_status": "accepted",
        "review_notes": "Accepted."
      },
      "geometry": {"type": "Point", "coordinates": [-79.75, 36.05]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    gpd.GeoDataFrame(
        {"NHDPlusID": [100], "HydroSeq": [10], "DnHydroSeq": [0], "TotDASqKm": [30.0]},
        geometry=[LineString([(-79.76, 36.04), (-79.74, 36.06)])],
        crs="EPSG:4326",
    ).to_file(river_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"featureid": [100]},
        geometry=[Polygon([(-79.77, 36.03), (-79.73, 36.03), (-79.73, 36.07), (-79.77, 36.07), (-79.77, 36.03)])],
        crs="EPSG:4326",
    ).to_file(catchment_path, driver="GPKG")
    config = {
        "collection": {
            "national_hydrography": {
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
            }
        },
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {"subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg"},
            "handoff": {"source_variable": "river_q", "target": "sfincs_discharge_forcing"},
        },
    }

    result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})

    assert result["coverage_status"] == "handoff_watershed_only"
    assert result["coverage_region"] is None
    assert result["submodel_count"] == 1
    fabric = gpd.read_file(result["subbasin_fabric"])
    assert set(fabric["region_role"]) == {"handoff_drainage"}


def test_wflow_domain_set_plan_requires_reviewed_streamgage_schema_fields(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True)
    network_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "site_no": "02095000",
        "site_name": "Reedy Fork near Oak Ridge",
        "status": "active",
        "roles": ["sfincs_handoff"],
        "wflow_submodel_id": "reedy_fork",
        "sfincs_domain_id": "greensboro_main",
        "sfincs_handoff_id": "reedy_fork_inflow",
        "review_status": "accepted"
      },
      "geometry": {"type": "Point", "coordinates": [-79.998, 36.175]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
        }
    }

    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})

    assert plan.status == "review_required"
    assert "02095000 missing reviewed schema fields" in plan.issues[0]
    assert "drainage_area_sqmi" in plan.issues[0]
    assert "frequency_basis" in plan.issues[0]


def test_wflow_build_steps_for_submodel_replace_bbox_with_reviewed_subbasin(tmp_path):
    build_config = tmp_path / "wflow_build.yml"
    build_config.write_text(
        """
steps:
  - setup_config:
      data:
        time.timestepsecs: 3600
  - setup_basemaps:
      region:
        bbox: [-80.1, 35.9, -79.7, 36.2]
      hydrography_fn: merit_hydro
  - setup_rivers:
      hydrography_fn: merit_hydro
""".strip(),
        encoding="utf-8",
    )
    submodel = {
        "wflow_submodel_id": "reedy_fork",
        "region": {"subbasin": [-79.998, 36.175]},
    }

    steps = build_wflow_steps_for_submodel(build_config, submodel)

    assert steps[1]["setup_basemaps"]["region"] == {"subbasin": [-79.998, 36.175]}
    assert steps[1]["setup_basemaps"]["hydrography_fn"] == "merit_hydro"
    assert [next(iter(step)) for step in steps] == ["setup_config", "setup_basemaps", "setup_rivers"]


def test_wflow_build_steps_accept_top_level_hydromt_recipe(tmp_path):
    build_config = tmp_path / "wflow_build.yml"
    build_config.write_text(
        """
setup_config:
  data:
    time.timestepsecs: 3600
setup_basemaps:
  region:
    bbox: [-80.1, 35.9, -79.7, 36.2]
  hydrography_fn: merit_hydro
setup_rivers:
  hydrography_fn: merit_hydro
""".strip(),
        encoding="utf-8",
    )
    submodel = {
        "wflow_submodel_id": "reedy_fork",
        "region": {"subbasin": [-79.998, 36.175]},
    }

    steps = build_wflow_steps_for_submodel(build_config, submodel)

    assert steps[1]["setup_basemaps"]["region"] == {"subbasin": [-79.998, 36.175]}
    assert [next(iter(step)) for step in steps] == ["setup_config", "setup_basemaps", "setup_rivers"]


def test_wflow_build_steps_add_sfincs_gauges_from_reviewed_handoff_points(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02095500", "02099000"],
            "site_name": ["South Buffalo", "North Buffalo", "Inactive handoff"],
            "status": ["active", "active", "inactive"],
            "drainage_area_sqmi": [20.1, 12.4, 99.0],
            "period_start": ["1979-02-01", "1979-02-01", "1979-02-01"],
            "period_end": ["2022-12-31", "2022-12-31", "2022-12-31"],
            "record_years": [43.9, 43.9, 43.9],
            "completeness_score": [0.98, 0.91, 0.88],
            "roles": [
                ["frequency", "calibration", "sfincs_handoff"],
                ["validation"],
                ["sfincs_handoff"],
            ],
            "frequency_basis": ["greensboro-main", "greensboro-main", "regional"],
            "wflow_submodel_id": ["south_buffalo", "south_buffalo", "south_buffalo"],
            "sfincs_domain_id": ["greensboro_west", "greensboro_west", "greensboro_west"],
            "sfincs_handoff_id": ["south_buffalo_02095000", None, "inactive_02099000"],
            "review_status": ["accepted", "accepted_with_warning", "accepted"],
            "review_notes": ["SFINCS inflow", "Calibration only", "Do not use"],
        },
        geometry=[
            Point(-79.721, 36.060),
            Point(-79.707, 36.122),
            Point(-79.990, 36.038),
        ],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    build_config = location_root / "data/wflow/config/wflow_build.yml"
    build_config.parent.mkdir(parents=True)
    build_config.write_text(
        """
steps:
  - setup_config:
      data:
        time.timestepsecs: 3600
  - setup_basemaps:
      region:
        bbox: [-80.1, 35.9, -79.7, 36.2]
      hydrography_fn: us_hydrography_basemap
  - setup_rivers:
      hydrography_fn: us_hydrography_basemap
""".strip(),
        encoding="utf-8",
    )
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "handoff": {"source_standard_name": "river_water__volume_flow_rate"},
            "gauges": {"root": "data/wflow/domain_set_gauges"},
        }
    }
    plan = plan_wflow_domain_set_from_streamgages(config, {"location_root": location_root})
    submodel = plan.submodels[0]

    gauge_summary = write_wflow_sfincs_gauge_locations(
        config,
        {"location_root": location_root},
        submodel,
    )
    steps = build_wflow_steps_for_submodel(
        build_config,
        submodel,
        gauges_fn=gauge_summary["gauges_fn"],
        handoff_config=config["wflow"]["handoff"],
    )

    gauges = gpd.read_file(gauge_summary["gauges_fn"])
    assert gauge_summary["gauge_count"] == 1
    assert gauges["name"].tolist() == ["south_buffalo_02095000"]
    assert gauges["site_no"].tolist() == ["02095000"]
    assert gauges["uparea"].iloc[0] == pytest.approx(52.058, rel=1.0e-3)
    assert [next(iter(step)) for step in steps] == [
        "setup_config",
        "setup_basemaps",
        "setup_rivers",
        "setup_gauges",
    ]
    assert steps[1]["setup_basemaps"]["region"] == submodel["region"]
    assert steps[-1]["setup_gauges"] == {
        "gauges_fn": str(gauge_summary["gauges_fn"]),
        "index_col": "index",
        "snap_to_river": True,
        "snap_uparea": True,
        "basename": "sfincs",
        "gauge_toml_header": ["Q"],
        "gauge_toml_param": ["river_water__volume_flow_rate"],
        "derive_subcatch": True,
    }


def test_wflow_sfincs_gauges_use_sfincs_boundary_handoff_locations_without_uparea_snap(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    handoff_path = location_root / "data/sfincs/domains/greensboro_main/base/gis/wflow_handoff_sources.geojson"
    build_config = location_root / "data/wflow/config/wflow_build.yml"
    network_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)
    build_config.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "site_name": ["South Buffalo"],
            "status": ["active"],
            "drainage_area_sqmi": [42.0],
            "period_start": ["1979-02-01"],
            "period_end": ["2022-12-31"],
            "record_years": [43.9],
            "completeness_score": [0.98],
            "roles": [["frequency", "calibration", "sfincs_handoff"]],
            "frequency_basis": ["south_buffalo"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "review_status": ["accepted"],
            "review_notes": ["Accepted as downstream review gage."],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "index": [1],
            "name": ["south_buffalo_02095000"],
            "site_no": ["02095000"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
            "uparea": [99.0],
            "handoff_placement": ["stream_boundary_intersection"],
            "handoff_location_review_status": ["review_required_stream_boundary_intersection"],
            "stream_boundary_river_source": ["hydromt_wflow_setup_rivers"],
        },
        geometry=[Point(0, 5)],
        crs="EPSG:3857",
    ).to_file(handoff_path, driver="GeoJSON")
    build_config.write_text(
        """
steps:
  - setup_basemaps:
      region:
        bbox: [-80.1, 35.9, -79.7, 36.2]
      hydrography_fn: us_hydrography_basemap
""".strip(),
        encoding="utf-8",
    )
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "gauges": {"root": "data/wflow/domain_set_gauges"},
        },
        "sfincs_domain_set": {
            "domains_root": "data/sfincs/domains",
        },
        "inland_coupling": {
            "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
        },
    }
    submodel = {
        "wflow_submodel_id": "south_buffalo",
        "region": {"subbasin": [-79.7, 36.1]},
        "sfincs_handoff_ids": ("south_buffalo_02095000",),
    }

    gauge_summary = write_wflow_sfincs_gauge_locations(
        config,
        {"location_root": location_root},
        submodel,
    )
    steps = build_wflow_steps_for_submodel(
        build_config,
        submodel,
        gauges_fn=gauge_summary["gauges_fn"],
        sfincs_snap_to_river=gauge_summary["snap_to_river"],
        sfincs_snap_uparea=gauge_summary["snap_uparea"],
    )

    gauges = gpd.read_file(gauge_summary["gauges_fn"]).to_crs("EPSG:3857")
    assert gauge_summary["snap_to_river"] is True
    assert gauge_summary["snap_uparea"] is False
    assert gauges.geometry.iloc[0].x == pytest.approx(0.0, abs=1.0e-6)
    assert gauges["gauge_location_source"].tolist() == ["sfincs_stream_boundary_intersection"]
    assert steps[-1]["setup_gauges"]["basename"] == "sfincs"
    assert steps[-1]["setup_gauges"]["snap_to_river"] is True
    assert steps[-1]["setup_gauges"]["snap_uparea"] is False


def test_wflow_sfincs_gauges_ignore_legacy_non_boundary_handoff_artifacts(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    handoff_path = location_root / "data/sfincs/domains/greensboro_main/base/gis/wflow_handoff_sources.geojson"
    network_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "site_name": ["South Buffalo"],
            "status": ["active"],
            "drainage_area_sqmi": [42.0],
            "period_start": ["1979-02-01"],
            "period_end": ["2022-12-31"],
            "record_years": [43.9],
            "completeness_score": [0.98],
            "roles": [["frequency", "calibration", "sfincs_handoff"]],
            "frequency_basis": ["south_buffalo"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "review_status": ["accepted"],
            "review_notes": ["Accepted as downstream review gage."],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "index": [1],
            "name": ["south_buffalo_02095000"],
            "site_no": ["02095000"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(0, 5)],
        crs="EPSG:3857",
    ).to_file(handoff_path, driver="GeoJSON")
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "gauges": {"root": "data/wflow/domain_set_gauges"},
        },
        "sfincs_domain_set": {
            "domains_root": "data/sfincs/domains",
        },
        "inland_coupling": {
            "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
        },
    }
    submodel = {
        "wflow_submodel_id": "south_buffalo",
        "region": {"subbasin": [-79.7, 36.1]},
        "sfincs_handoff_ids": ("south_buffalo_02095000",),
    }

    gauge_summary = write_wflow_sfincs_gauge_locations(
        config,
        {"location_root": location_root},
        submodel,
    )

    gauges = gpd.read_file(gauge_summary["gauges_fn"]).to_crs("EPSG:3857")
    assert gauge_summary["snap_uparea"] is True
    assert gauges.geometry.iloc[0].x == pytest.approx(15.0, abs=1.0e-6)
    assert "gauge_location_source" not in gauges


def test_encompassing_huc_wflow_gauges_prefer_sfincs_boundary_handoff_locations(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    handoff_path = location_root / "data/sfincs/domains/greensboro_east/base/gis/wflow_handoff_sources.geojson"
    network_path.parent.mkdir(parents=True)
    handoff_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "status": ["active"],
            "drainage_area_sqmi": [42.0],
            "wflow_submodel_id": ["greensboro_east"],
            "sfincs_domain_id": ["greensboro_east"],
            "sfincs_handoff_id": [None],
            "review_status": ["accepted"],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "index": [1],
            "name": ["greensboro_east_inflow_01"],
            "site_no": ["greensboro_east_inflow_01"],
            "sfincs_handoff_id": ["greensboro_east_inflow_01"],
            "wflow_submodel_id": ["greensboro_east"],
            "sfincs_domain_id": ["greensboro_east"],
            "handoff_placement": ["stream_boundary_intersection"],
            "handoff_location_review_status": ["review_required_stream_boundary_intersection"],
            "stream_boundary_river_source": ["hydromt_wflow_setup_rivers"],
        },
        geometry=[Point(0, 5)],
        crs="EPSG:3857",
    ).to_file(handoff_path, driver="GeoJSON")
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "gauges": {"root": "data/wflow/domain_set_gauges"},
            "domain_set": {"outlet_source": "encompassing_huc"},
        },
        "sfincs_domain_set": {"domains_root": "data/sfincs/domains"},
        "inland_coupling": {"discharge_forcing": {"handoff_location": "stream_boundary_intersection"}},
    }
    submodel = {
        "wflow_submodel_id": "greensboro_east",
        "region": {"geom": "data/wflow/domain_huc/greensboro_east.geojson"},
        "sfincs_handoff_ids": ("greensboro_east_inflow_01",),
        "handoff_points": [
            {
                "sfincs_handoff_id": "greensboro_east_inflow_01",
                "sfincs_domain_id": "greensboro_east",
                "lon": 10,
                "lat": 5,
                "uparea_km2": 99.0,
            }
        ],
    }

    summary = _write_wflow_sfincs_handoff_gauge_locations(config, {"location_root": location_root}, submodel)

    gauges = gpd.read_file(summary["gauges_fn"]).to_crs("EPSG:3857")
    assert summary["snap_uparea"] is False
    assert gauges.geometry.iloc[0].x == pytest.approx(0.0, abs=1.0e-6)
    assert gauges["gauge_location_source"].tolist() == ["sfincs_stream_boundary_intersection"]


def test_write_wflow_observation_gauges_keeps_all_accepted_gages_with_roles(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02094659", "02095500", "02099000"],
            "status": ["active", "active", "active", "inactive"],
            "drainage_area_sqmi": [34.0, 7.33, 37.1, 99.0],
            "wflow_submodel_id": ["south_buffalo", "south_buffalo", "south_buffalo", "south_buffalo"],
            "sfincs_domain_id": ["greensboro_west"] * 4,
            "sfincs_handoff_id": ["south_buffalo_02095000", None, None, "inactive_02099000"],
            "review_status": ["accepted", "accepted_with_warning", "accepted_with_warning", "accepted"],
        },
        geometry=[
            Point(-79.721, 36.060),
            Point(-79.735, 36.050),
            Point(-79.707, 36.122),
            Point(-79.990, 36.038),
        ],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    config = {
        "wflow": {
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "gauges": {"root": "data/wflow/domain_set_gauges"},
        }
    }
    submodel = {
        "wflow_submodel_id": "south_buffalo",
        "gauge_site_nos": ("02095000", "02094659", "02095500"),
        "sfincs_handoff_ids": ("south_buffalo_02095000",),
    }

    summary = write_wflow_observation_gauge_locations(config, {"location_root": location_root}, submodel)

    gauges = gpd.read_file(summary["gauges_fn"])
    # The inactive gage (02099000) is dropped; all 3 accepted active gages are kept.
    assert summary["gauge_count"] == 3
    assert set(gauges["site_no"].astype(str)) == {"02095000", "02094659", "02095500"}
    assert summary["snap_uparea"] is True
    # Roles distinguish the SFINCS source from observation-only gages.
    role_by_site = dict(zip(gauges["site_no"].astype(str), gauges["role"]))
    assert role_by_site["02095000"] == "sfincs_source"
    assert role_by_site["02094659"] == "observation"
    assert role_by_site["02095500"] == "observation"
    assert gauges["name"].tolist() == gauges["site_no"].astype(str).tolist()
    assert (gauges["uparea"] > 0).all()


def test_build_wflow_submodel_calls_hydromt_wflow_with_reviewed_local_inputs(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    for relative_path in [
        "data/static/processed/landcover_region_setup.tif",
        "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
    ]:
        path = location_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    hydrography_path = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    hydrography_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "flwdir": (("y", "x"), np.ones((2, 2), dtype="uint8")),
            "elevtn": (("y", "x"), np.ones((2, 2), dtype="float32")),
            "uparea": (("y", "x"), np.ones((2, 2), dtype="float32")),
            "strord": (("y", "x"), np.ones((2, 2), dtype="int16")),
            "basins": (("y", "x"), np.ones((2, 2), dtype="int32")),
        },
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    ).to_netcdf(hydrography_path)
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "site_name": ["South Buffalo"],
            "status": ["active"],
            "drainage_area_sqmi": [20.1],
            "period_start": ["1979-02-01"],
            "period_end": ["2022-12-31"],
            "record_years": [43.9],
            "completeness_score": [0.98],
            "roles": [["frequency", "calibration", "sfincs_handoff"]],
            "frequency_basis": ["south_buffalo_creek"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_west"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "review_status": ["accepted"],
            "review_notes": ["Reviewed handoff"],
        },
        geometry=[Point(-79.7258, 36.06)],
        crs="EPSG:4326",
    ).to_file(network_path, driver="GeoJSON")
    build_config = location_root / "data/wflow/config/wflow_build.yml"
    update_config = location_root / "data/wflow/config/wflow_update_forcing.yml"
    build_config.parent.mkdir(parents=True, exist_ok=True)
    build_config.write_text(
        """
steps:
  - setup_config:
      data:
        time.timestepsecs: 3600
  - setup_basemaps:
      region:
        bbox: [-80.1, 35.9, -79.7, 36.2]
      hydrography_fn: us_hydrography_basemap
  - setup_rivers:
      hydrography_fn: us_hydrography_basemap
""".strip(),
        encoding="utf-8",
    )
    update_config.write_text(
        """
steps:
  - setup_config:
      data:
        time.timestepsecs: 3600
  - setup_precip_forcing:
      precip_fn: event_precip
""".strip(),
        encoding="utf-8",
    )
    calls = []

    class FakeWflowModel:
        def __init__(self, root, mode, data_libs):
            calls.append(("model", "__init__", {"root": root, "mode": mode, "data_libs": data_libs}))
            self.root = Path(root)

        def build(self, *, steps):
            calls.append(("model", "build", {"steps": steps}))
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "wflow_sbm.toml").write_text("written\n", encoding="utf-8")
            staticgeoms = self.root / "staticgeoms"
            staticgeoms.mkdir(parents=True, exist_ok=True)
            for step in steps:
                if "setup_gauges" not in step:
                    continue
                gauges = step["setup_gauges"]
                basename = gauges["basename"]
                gpd.read_file(gauges["gauges_fn"]).to_file(
                    staticgeoms / f"gauges_{basename}.geojson",
                    driver="GeoJSON",
                )

    config = {
        "project": {"name": "greensboro", "country": "US", "reference_crs": "EPSG:4326"},
        "wflow": {
            "plugin": "wflow_sbm",
            "base_model_root": "data/wflow/base",
            "events_root": "data/wflow/events",
            "data_catalog": "data/wflow/data_catalog.yml",
            "build_config": "data/wflow/config/wflow_build.yml",
            "update_forcing_config": "data/wflow/config/wflow_update_forcing.yml",
            "streamgage_network": {"reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"},
            "domain_set": {"event_catalog_scope": "shared_across_domain_set"},
            "handoff": {"source_standard_name": "river_water__volume_flow_rate"},
        },
        "static_sources": {"landcover": {"output": "data/static/processed/landcover_region_setup.tif"}},
        "collection": {
            "national_hydrography": {
                "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
            }
        },
    }

    summary = build_wflow_submodel(
        config,
        {"location_root": location_root},
        submodel_id="south_buffalo",
        model_cls=FakeWflowModel,
        force=True,
    )

    assert summary["status"] == "built"
    assert summary["wflow_submodel_id"] == "south_buffalo"
    assert summary["base_model_root"] == location_root / "data/wflow/base/south_buffalo"
    assert summary["built"] is True
    assert calls[0] == (
        "model",
        "__init__",
        {
            "root": str(location_root / "data/wflow/base/south_buffalo"),
            "mode": "w+",
            "data_libs": [str(location_root / "data/wflow/data_catalog.yml")],
        },
    )
    build_steps = calls[1][2]["steps"]
    assert build_steps[1]["setup_basemaps"]["region"]["subbasin"] == [-79.7258, 36.06]
    gauge_layers = {
        step["setup_gauges"]["basename"]: step["setup_gauges"]
        for step in build_steps
        if "setup_gauges" in step
    }
    assert set(gauge_layers) == {"sfincs", "usgs"}
    assert gauge_layers["sfincs"]["gauges_fn"].endswith("south_buffalo_sfincs_gauges.geojson")
    assert gauge_layers["usgs"]["gauges_fn"].endswith("south_buffalo_observation_gauges.geojson")
    assert summary["observation_gauges_fn"].name == "south_buffalo_observation_gauges.geojson"
    assert summary["observation_gauge_count"] >= summary["gauge_count"]

    reused = build_wflow_submodel(
        config,
        {"location_root": location_root},
        submodel_id="south_buffalo",
        model_cls=FakeWflowModel,
        force=False,
    )

    assert reused["status"] == "reused"
    assert reused["built"] is True
    assert reused["model"].root == location_root / "data/wflow/base/south_buffalo"
    assert calls[-1] == (
        "model",
        "__init__",
        {
            "root": str(location_root / "data/wflow/base/south_buffalo"),
            "mode": "r",
            "data_libs": [str(location_root / "data/wflow/data_catalog.yml")],
        },
    )


def test_sfincs_gauge_layer_rejects_stale_handoff_source_geometry(tmp_path):
    location_root = tmp_path / "location"
    model_root = location_root / "data/wflow/base/south_buffalo"
    gauges_path = model_root / "staticgeoms/gauges_sfincs.geojson"
    source_path = location_root / "data/sfincs/domains/south_buffalo/base/gis/wflow_handoff_sources.geojson"
    manifest_path = location_root / "data/sfincs/domains/domain_set.yaml"
    gauges_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame(
        {"sfincs_handoff_id": ["south_buffalo_inflow_01"]},
        geometry=[Point(-79.7, 36.0)],
        crs="EPSG:4326",
    ).to_file(gauges_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"sfincs_handoff_id": ["south_buffalo_inflow_01"]},
        geometry=[Point(-79.6, 36.1)],
        crs="EPSG:4326",
    ).to_file(source_path, driver="GeoJSON")
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "domains": [
                    {
                        "sfincs_domain_id": "south_buffalo",
                        "base_model_root": "data/sfincs/domains/south_buffalo/base",
                        "handoff_source_ids": ["south_buffalo_inflow_01"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = {"sfincs_domain_set": {"domain_manifest": "data/sfincs/domains/domain_set.yaml"}}
    paths = {"location_root": location_root}
    submodel = {
        "sfincs_domain_ids": ["south_buffalo"],
        "sfincs_handoff_ids": ["south_buffalo_inflow_01"],
    }

    os.utime(gauges_path, (100, 100))
    os.utime(source_path, (200, 200))
    assert not _sfincs_gauge_layer_matches(model_root, submodel, config=config, paths=paths)

    os.utime(gauges_path, (300, 300))
    assert _sfincs_gauge_layer_matches(model_root, submodel, config=config, paths=paths)


def test_wflow_us_source_strategy_prefers_local_ldd_and_existing_ssurgo_for_greensboro():
    config = {
        "project": {"name": "greensboro", "country": "USA"},
        "static_sources": {
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            }
        },
        "wflow": {
            "source_strategy": {
                "hydrography": {
                    "preferred": "us_hydrography",
                    "hydromt_basemap": "us_hydrography_basemap",
                    "api": "hydromt_gis_flw_d8_from_dem",
                    "fallback": "merit_hydro",
                },
                "soils": {
                    "preferred": "ssurgo",
                    "wflow_parameters": "ssurgo_wflow_soil_parameters",
                    "fallback": "soilgrids",
                },
            }
        },
    }

    strategy = plan_wflow_us_source_strategy(config)

    assert strategy.status == "review_required"
    assert strategy.hydrography_policy == "us_hydrography_first"
    assert strategy.hydromt_basemap_source == "us_hydrography_basemap"
    assert strategy.river_geometry_source == "nhdplus_hr_river_geometry"
    assert strategy.catchment_source is None
    assert strategy.soil_policy == "ssurgo_first"
    assert strategy.ssurgo_inputs == (
        "data/static/soils/hsg_greensboro.tif",
        "data/static/soils/ksat_mmhr_greensboro.tif",
    )
    assert strategy.global_fallbacks == ("merit_hydro", "soilgrids")
    assert "HydroMT-Wflow setup_basemaps requires a local DEM-derived RasterDataset" in strategy.issues[0]


def test_build_wflow_data_catalog_writes_location_specific_hydromt_sources(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    config = {
        "project": {"name": "greensboro", "country": "USA", "reference_crs": "EPSG:4326"},
        "wflow": {
            "data_catalog": "data/wflow/data_catalog.yml",
            "source_strategy": {
                "hydrography": {
                    "preferred": "us_hydrography",
                    "hydromt_basemap": "us_hydrography_basemap",
                    "api": "hydromt_gis_flw_d8_from_dem",
                    "fallback": "merit_hydro",
                },
                "soils": {
                    "preferred": "ssurgo",
                    "wflow_parameters": "ssurgo_wflow_soil_parameters",
                    "fallback": "soilgrids",
                },
            },
        },
        "static_sources": {
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            },
            "landcover": {"output": "data/static/processed/landcover_region_setup.tif"},
            "wflow_collection_extent": {
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
                "hsg_output": "data/wflow/static/soils/hsg_wflow.tif",
                "ksat_output": "data/wflow/static/soils/ksat_mmhr_wflow.tif",
            },
        },
        "collection": {
            "usgs_streamgages": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
            },
            "national_hydrography": {
                "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
            },
            "aorc_sst": {
                "event_precip": "data/wflow/events/<event_id>/precip.nc",
            },
            "nwm": {
                "event_temp_pet": "data/wflow/events/<event_id>/temp_pet.nc",
            },
        },
    }

    catalog_path = build_wflow_data_catalog(config, {"location_root": location_root})

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["meta"]["roots"] == [".."]
    assert catalog["meta"]["source_policy"] == {
        "hydrography": "us_hydrography_first",
        "soils": "ssurgo_first",
    }
    assert catalog["meta"]["global_fallback_dependencies"] == ["merit_hydro", "soilgrids"]
    assert catalog["greensboro_streamgage_network"] == {
        "data_type": "GeoDataFrame",
        "driver": {"name": "pyogrio"},
        "uri": str(location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"),
        "metadata": {"crs": "EPSG:4326", "category": "hydrography"},
    }
    assert catalog["us_hydrography_basemap"]["metadata"]["required_variables"] == [
        "flwdir",
        "elevtn",
        "uparea",
        "strord",
    ]
    for source in ("us_hydrography_basemap", "esa_worldcover", "ssurgo_wflow_soil_parameters"):
        required_for_build = catalog[source]["metadata"]["required_for_build"]
        assert required_for_build == 1
        assert not isinstance(required_for_build, bool)
    assert catalog["nhdplus_hr_river_geometry"] == {
        "data_type": "GeoDataFrame",
        "driver": {"name": "pyogrio"},
        "uri": str(location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"),
        "metadata": {
            "crs": "EPSG:4326",
            "category": "hydrography",
            "required_for_build": 1,
            "required_columns": ["rivwth", "rivdph", "qbankfull"],
        },
    }
    assert "nhdplus_hr_catchments" not in catalog
    # setup_lulcmaps must resolve ESA WorldCover from region setup, not the
    # absent `globcover` global default. The source name drives HydroMT-Wflow's
    # shipped esa_worldcover_mapping_default.
    assert catalog["esa_worldcover"]["data_type"] == "RasterDataset"
    assert catalog["esa_worldcover"]["driver"]["name"] == "rasterio"
    assert catalog["esa_worldcover"]["uri"] == str(location_root / "data/wflow/static/processed/landcover_wflow_coarse.tif")
    assert catalog["esa_worldcover"]["metadata"]["category"] == "landuse"
    assert catalog["ssurgo_hydrologic_soil_group"]["uri"] == str(location_root / "data/wflow/static/soils/hsg_wflow.tif")
    assert catalog["ssurgo_saturated_conductivity"]["uri"] == str(location_root / "data/wflow/static/soils/ksat_mmhr_wflow.tif")
    assert catalog["ssurgo_wflow_soil_parameters"]["metadata"]["status"] == "review_required"
    assert catalog["event_precip"]["data_type"] == "RasterDataset"
    assert catalog["event_precip"]["driver"]["name"] == "raster_xarray"
    assert catalog["event_precip"]["uri"] == str(location_root / "data/wflow/events/<event_id>/precip.nc")
    assert catalog["event_temp_pet"]["data_type"] == "RasterDataset"
    assert catalog["event_temp_pet"]["uri"] == str(location_root / "data/wflow/events/<event_id>/temp_pet.nc")


def test_wflow_catalog_source_readiness_flags_missing_review_required_sources(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    config = {
        "wflow": {"data_catalog": "data/wflow/data_catalog.yml"},
        "collection": {
            "national_hydrography": {
                "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
            }
        },
        "static_sources": {
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            }
        },
    }
    catalog_path = build_wflow_data_catalog(config, {"location_root": location_root})

    readiness = wflow_catalog_source_readiness(catalog_path)

    by_source = {row["source"]: row for row in readiness}
    assert by_source["us_hydrography_basemap"]["exists"] is False
    assert by_source["us_hydrography_basemap"]["required_for_build"] is True
    assert by_source["ssurgo_wflow_soil_parameters"]["exists"] is False
    assert by_source["ssurgo_wflow_soil_parameters"]["required_for_build"] is True


def test_greensboro_wflow_build_resolution_matches_collected_hydrography_resolution():
    repo_root = Path(__file__).resolve().parents[2]
    location_root = repo_root / "locations/greensboro"
    config = define_location(location_root / "config.yaml").config
    build_config = config["_model_recipes"]["wflow_build"]

    target_resolution = config["collection"]["national_hydrography"]["basemap_source_resolution_degrees"]
    setup_basemaps = next(step["setup_basemaps"] for step in build_config["steps"] if "setup_basemaps" in step)

    assert target_resolution == pytest.approx(0.0009)
    assert setup_basemaps["res"] == pytest.approx(target_resolution)
