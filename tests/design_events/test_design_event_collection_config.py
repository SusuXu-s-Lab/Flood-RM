from design_events.config import load_runtime


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
    assert [point["id"] for point in collection["nwm"]["soil_moisture"]["points"]] == ["center", "southwest", "northwest", "west_mid", "south_mid"]
    assert config["event_catalog"]["forcing_members"]["rainfall"] == "data/sources/aorc_sst/rainfall_members.csv"
    assert config["event_catalog"]["forcing_members"]["soil_moisture"] == "data/sources/nwm/soil_moisture.csv"
    assert config["event_catalog"]["pairing"]["rainfall"]["strategy"] == "seasonal_window_permutation"
    assert config["event_catalog"]["pairing"]["rainfall"]["window_days"] == 45
