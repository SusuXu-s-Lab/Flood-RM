import geopandas as gpd
from shapely.geometry import box

from design_events.collect_sources.ssurgo import (
    has_footprint,
    load_points,
)

# The NWM v3.0 ldasout grid CRS, as configured in the shared location bases.
NWM_CRS = (
    "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 "
    "+lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext +no_defs"
)

EXPECTED_IDS = [
    "center",
    "southwest",
    "southeast",
    "northwest",
    "northeast",
    "south_mid",
    "north_mid",
    "west_mid",
    "east_mid",
]


def _write_footprint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(-80.0, 36.0, -79.5, 36.3)], crs="EPSG:4326"
    )
    gdf.to_file(path, driver="GeoJSON")


def _spec(**overrides):
    spec = {
        "crs": NWM_CRS,
        "x": "x",
        "y": "y",
        "points_source": "data/static/aoi/evaluation_footprint.geojson",
        "points_file": "data/static/aoi/nwm_soil_moisture_points.geojson",
    }
    spec.update(overrides)
    return spec


def test_load_points_derives_centroid_corners_and_edges_from_footprint(tmp_path):
    _write_footprint(tmp_path / "data/static/aoi/evaluation_footprint.geojson")
    output = tmp_path / "data/static/aoi/nwm_soil_moisture_points.geojson"
    paths = {
        "location_root": tmp_path,
        "repo_root": tmp_path,
        "nwm_soil_moisture_points_geojson": output,
    }

    points = load_points(_spec(), paths)

    assert [point["id"] for point in points] == EXPECTED_IDS
    assert all({"id", "x", "y"} == set(point) for point in points)
    # The derivation writes the stakeholder-facing, review-required GeoJSON.
    assert output.exists()
    written = gpd.read_file(output)
    assert set(written["review_status"]) == {"review_required"}
    assert len(written) == len(EXPECTED_IDS)

    xs = {point["id"]: point["x"] for point in points}
    ys = {point["id"]: point["y"] for point in points}
    # The centroid sits inside the bounding box defined by the corner samples.
    assert xs["southwest"] <= xs["center"] <= xs["southeast"]
    assert ys["southwest"] <= ys["center"] <= ys["northwest"]


def test_explicit_points_bypass_footprint_derivation(tmp_path):
    points = load_points(
        _spec(points=[{"id": "a", "x": 1.0, "y": 2.0}]), {"location_root": tmp_path}
    )
    assert points == [{"id": "a", "x": 1.0, "y": 2.0}]


def test_missing_footprint_and_points_resolves_to_empty(tmp_path):
    spec = _spec()
    paths = {"location_root": tmp_path}
    assert has_footprint(spec, paths) is False
    assert load_points(spec, paths) == []
