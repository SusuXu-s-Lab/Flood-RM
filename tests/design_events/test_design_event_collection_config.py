import json
from pathlib import Path

from design_events.utils import load_runtime


repo_root = Path(__file__).resolve().parents[2]


def notebook_text(relative_path):
    notebook = json.loads((repo_root / relative_path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_marshfield_collection_windows_align_across_cora_aorc_and_nwm():
    config, _ = load_runtime("locations/marshfield/config.yaml")
    collection = config["collection"]

    assert collection["start"] == "1979-02-01"
    assert collection["end"] == "2022-12-31"
    assert collection["aorc_sst"]["start_date"] == collection["start"]
    assert collection["aorc_sst"]["end_date"] == collection["end"]
    assert collection["aorc_sst"]["zarr_year_pattern"] == "s3://noaa-nws-aorc-v1-1-1km/{year}.zarr"
    assert collection["aorc_sst"]["write_event_windows"] is True
    assert collection["aorc_sst"]["transposition_region"]["geometry_file"] == "data/sources/aorc_sst/transposition_regions/transposition_region_100km.geojson"
    assert collection["nwm"]["start"] == collection["start"]
    assert collection["nwm"]["end"] == collection["end"]
    assert collection["nwm"]["version"] == "3.0"
    assert collection["nwm"]["streamflow"]["zarr"] == "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
    assert collection["nwm"]["streamflow"]["available"] is False
    assert collection["nwm"]["streamflow"]["feature_ids"] == []
    assert collection["nwm"]["soil_moisture"]["zarr"] == "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr"
    assert collection["nwm"]["soil_moisture"]["variables"] == ["SOIL_M", "SOILSAT_TOP"]
    # Coastal locations derive sampling points from the study area; no hand-typed coordinates.
    assert collection["nwm"]["soil_moisture"]["points_source"] == "data/static/aoi/study_area.geojson"
    assert collection["nwm"]["soil_moisture"]["points_file"] == "data/static/aoi/nwm_soil_moisture_points.geojson"
    assert "points" not in collection["nwm"]["soil_moisture"]
    assert "event_catalog" not in config

    text = notebook_text("locations/marshfield/02_flood/03_build_event_catalog.ipynb")
    assert "event_policy = " in text
    assert "'rainfall': 'data/sources/aorc_sst/rainfall_members.csv'" in text
    assert "'soil_moisture': 'data/sources/nwm/soil_moisture.csv'" in text
    assert "'strategy': 'seasonal_window_permutation'" in text
    assert "'window_days': 45" in text


def test_greensboro_uses_usgs_streamflow_and_derived_nwm_soil_moisture_points():
    config, _ = load_runtime("locations/greensboro/config.yaml")
    collection = config["collection"]
    nwm = collection["nwm"]

    assert collection["usgs_streamgages"]["discovery"]["parameter_cd"] == "00060"
    assert collection["usgs_streamgages"]["discovery"]["has_data_type_cd"] == "dv"
    assert nwm["streamflow"]["available"] is False
    assert "USGS" in nwm["streamflow"]["reason"]
    assert nwm["streamflow"]["feature_ids"] == []
    assert nwm["soil_moisture"]["variables"] == ["SOIL_M", "SOILSAT_TOP"]
    assert "aggregate_points" not in nwm["soil_moisture"]
    assert nwm["soil_moisture"]["x"] == "x"
    assert nwm["soil_moisture"]["y"] == "y"
    assert "lcc" in nwm["soil_moisture"]["crs"]
    # Inland locations derive sampling points from the evaluation footprint; no coordinates in YAML.
    assert nwm["soil_moisture"]["points_source"] == "data/static/aoi/evaluation_footprint.geojson"
    assert nwm["soil_moisture"]["points_file"] == "data/static/aoi/nwm_soil_moisture_points.geojson"
    assert nwm["soil_moisture"]["selection_method"].endswith("snapped_to_nearest_nwm_cell")
    assert "points" not in nwm["soil_moisture"]
    assert config["event_catalog"]["dependence"]["driver_records"]["rainfall"]["path"] == (
        "data/sources/aorc_sst/greensboro/72hr-events/storm-stats.csv"
    )

    text = notebook_text("locations/greensboro/02_flood/03_build_event_catalog.ipynb")
    assert "event_policy = " in text
    assert "'streamflow': 'data/sources/usgs_streamgages/streamflow_members.csv'" in text
    assert "'soil_moisture': 'data/sources/nwm/soil_moisture.csv'" in text
