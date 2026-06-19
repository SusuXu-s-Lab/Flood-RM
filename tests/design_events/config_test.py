from pathlib import Path

from design_events.utils import build_paths, load_runtime


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[2]).as_posix()


def test_design_event_paths_include_study_location_workspace():
    paths = build_paths({"project": {"name": "sfo"}, "paths": {}})

    assert suffix(paths["location_root"]) == "locations/sfo"
    assert suffix(paths["location_config_path"]) == "locations/sfo/config.yaml"
    assert suffix(paths["notebooks_root"]) == "locations/sfo/02_flood"
    assert suffix(paths["location_data_root"]) == "locations/sfo/data"
    assert suffix(paths["outputs_root"]) == "locations/sfo/data/event_catalog"
    assert suffix(paths["data_root"]) == "locations/sfo/data/sources/cora_waterlevel"


def test_marshfield_config_yaml_preserves_design_event_paths():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    assert config["project"]["name"] == "marshfield"
    assert suffix(paths["location_config_path"]) == "locations/marshfield/config.yaml"
    assert suffix(paths["outputs_root"]) == "locations/marshfield/data/event_catalog"
    assert suffix(paths["waterlevel_csv"]) == "locations/marshfield/data/sources/cora_waterlevel/cora_mfield_boundary_hourly_msl.csv"
    assert suffix(paths["event_catalog_csv"]) == "locations/marshfield/data/event_catalog/catalog/event_catalog.csv"
    assert suffix(paths["event_catalog_audit_json"]) == "locations/marshfield/data/event_catalog/catalog/event_catalog_audit.json"
    assert suffix(paths["resilience_stress_training_catalog_csv"]) == "locations/marshfield/data/event_catalog/catalog/resilience_stress_training_catalog.csv"
    assert suffix(paths["nwm_root"]) == "locations/marshfield/data/sources/nwm"
    assert suffix(paths["nwm_streamflow_csv"]) == "locations/marshfield/data/sources/nwm/streamflow.csv"
    assert suffix(paths["usgs_streamgages_root"]) == "locations/marshfield/data/sources/usgs_streamgages"
    assert suffix(paths["usgs_streamgage_candidates_geojson"]) == "locations/marshfield/data/sources/usgs_streamgages/streamgage_candidates.geojson"
    assert suffix(paths["aorc_sst_root"]) == "locations/marshfield/data/sources/aorc_sst"
    assert suffix(paths["aorc_sst_rainfall_members_csv"]) == "locations/marshfield/data/sources/aorc_sst/rainfall_members.csv"
    assert suffix(paths["era5_waves_nc"]) == "locations/marshfield/data/sources/era5_waves/era5_mfield_offshore_hourly.nc"
    assert "sampling" not in config
    assert "resilience_stress_training" not in config
