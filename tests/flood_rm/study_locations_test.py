from pathlib import Path

from study_location import (
    build_location_template,
    configured_study_locations,
    list_study_locations,
    resolve_study_location,
)


def test_study_locations_resolves_study_location():
    location = resolve_study_location(
        {
            "project": {"name": "sfo"},
            "grid_footprint": {"source": "grid_footprint.geojson"},
            "coastal_waves": True,
        },
        Path("/repo"),
    )

    assert location.name == "sfo"
    assert location.root == Path("/repo/locations/sfo")
    assert location.data_root == Path("/repo/locations/sfo/data")
    assert location.grid_footprint_source == Path("/repo/locations/sfo/grid_footprint.geojson")
    assert location.uses_coastal_water_level is True
    assert location.coastal_waves is True


def test_study_location_grid_footprint_can_be_workspace_relative():
    location = resolve_study_location(
        {
            "project": {"name": "marshfield"},
            "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        },
        Path("/repo"),
    )

    assert location.grid_footprint_source == Path("/repo/locations/marshfield/data/static/aoi/study_area.geojson")


def test_study_locations_builds_inland_template():
    template = build_location_template("greensboro", flood_setting="inland")

    assert template["flood_setting"] == "inland"
    assert template["event_drivers"] == ["rainfall", "streamflow", "soil_moisture"]
    assert template["paths"]["data_root"] == "locations/greensboro/data"
    assert template["event_catalog"]["forcing_members"]["rainfall"] == "locations/greensboro/data/sources/aorc_sst/rainfall_members.csv"


def test_study_location_listing_keeps_placeholders_unconfigured(tmp_path):
    (tmp_path / "locations" / "marshfield").mkdir(parents=True)
    (tmp_path / "locations" / "sfo").mkdir(parents=True)
    (tmp_path / "locations" / "marshfield" / "config.yaml").write_text(
        "project:\n  name: marshfield\n",
        encoding="utf-8",
    )

    assert list_study_locations(tmp_path) == ["marshfield", "sfo"]
    assert configured_study_locations(tmp_path) == ["marshfield"]
