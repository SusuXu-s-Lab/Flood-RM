from pathlib import Path
import json

from shapely.geometry import Point, shape

from study_location import build_study_area, define_location, study_area_bbox

repo_root = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_study_area_from_asset_registry_writes_location_artifacts(tmp_path):
    location_root = tmp_path / "locations" / "marshfield"
    registry = location_root / "data" / "static" / "power_grid" / "asset_registry"
    _write_csv(
        registry / "buses.csv",
        "bus,feeder_id,lon,lat",
        [
            "bus_1,feeder_a,-70.78,42.14",
            "bus_2,feeder_a,-70.77,42.15",
            "bus_3,feeder_a,-70.76,42.13",
        ],
    )
    _write_csv(
        registry / "transformers.csv",
        "transformer_name,feeder_id,location_lon,location_lat",
        ["xfmr_1,feeder_a,-70.790,42.144"],
    )
    _write_csv(
        registry / "lines.csv",
        "line_name,feeder_id,from_lon,from_lat,to_lon,to_lat",
        ["line_1,feeder_a,-70.781,42.143,-70.775,42.150"],
    )

    config = {
        "project": {"name": "marshfield"},
        "aoi": {
            "source": "data/static/power_grid/asset_registry",
            "source_format": "asset_registry",
            "alpha_ratio": 0.3,
            "output": "data/static/aoi/study_area.geojson",
            "metadata_output": "data/static/aoi/study_area.json",
        },
    }

    result = build_study_area(config, tmp_path)

    assert result.location_name == "marshfield"
    assert result.n_points == 6
    assert result.output_path == location_root / "data/static/aoi/study_area.geojson"
    assert result.metadata_path == location_root / "data/static/aoi/study_area.json"

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    feature = payload["features"][0]
    geom = shape(feature["geometry"])
    for lon, lat in [
        (-70.78, 42.14),
        (-70.77, 42.15),
        (-70.76, 42.13),
        (-70.790, 42.144),
        (-70.781, 42.143),
        (-70.775, 42.150),
    ]:
        assert geom.covers(Point(lon, lat))

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_format"] == "asset_registry"
    assert metadata["n_points"] == 6


def test_build_study_area_preserves_disconnected_smart_ds_subregions_from_bridged_geojson(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    source = location_root / "data/static/aoi/dft_power_extent.geojson"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"region_id": "greensboro"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.0, 0.0],
                                    [0.0, 1.0],
                                    [1.0, 1.0],
                                    [6.0, 6.0],
                                    [7.0, 6.0],
                                    [7.0, 7.0],
                                    [6.0, 7.0],
                                    [6.0, 6.0],
                                    [1.0, 1.0],
                                    [1.0, 0.0],
                                    [0.0, 0.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "project": {"name": "greensboro"},
        "aoi": {
            "source": "data/static/aoi/dft_power_extent.geojson",
            "source_format": "geojson",
            "preserve_disconnected_subregions": True,
            "subregion_bridge_max_edge_degrees": 2.0,
            "output": "data/static/aoi/study_area.geojson",
            "metadata_output": "data/static/aoi/study_area.json",
        },
    }

    result = build_study_area(config, tmp_path)

    geometry = shape(json.loads(result.output_path.read_text(encoding="utf-8"))["features"][0]["geometry"])
    assert geometry.geom_type == "MultiPolygon"
    assert len(geometry.geoms) == 2
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["subregion_count"] == 2


def test_build_study_area_writes_smart_ds_buscoords_subregion_features(tmp_path):
    location_root = tmp_path / "locations" / "austin"
    smart_ds = location_root / "data/smart_ds/2016"
    for subregion, offset in [("P1U", 0.0), ("P2U", 10.0)]:
        buscoords = smart_ds / subregion / "scenarios/base_timeseries/opendss_no_loadshapes/Buscoords.dss"
        buscoords.parent.mkdir(parents=True, exist_ok=True)
        buscoords.write_text(
            "\n".join(
                [
                    f"bus_a {offset + 0.0} 0.0",
                    f"bus_b {offset + 0.0} 1.0",
                    f"bus_c {offset + 1.0} 1.0",
                    f"bus_d {offset + 1.0} 0.0",
                ]
            ),
            encoding="utf-8",
        )
    config = {
        "project": {"name": "austin"},
        "aoi": {
            "source": "data/smart_ds/2016",
            "source_format": "smart_ds_buscoords",
            "preserve_disconnected_subregions": True,
            "alpha_ratio": 0.1,
            "output": "data/static/aoi/study_area.geojson",
            "metadata_output": "data/static/aoi/study_area.json",
        },
    }

    result = build_study_area(config, tmp_path)

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert [feature["properties"]["subregion_id"] for feature in payload["features"]] == ["P1U", "P2U"]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["subregion_count"] == 2
    assert metadata["subregion_ids"] == ["P1U", "P2U"]


def test_marshfield_location_builds_grid_footprint_from_power_aoi():
    config = define_location(repo_root / "locations/marshfield/config.yaml").config

    result = build_study_area(config, repo_root)

    assert config["grid_footprint"]["source"] == "data/static/aoi/study_area.geojson"
    assert config["aoi"]["source"] == "data/power_grid/asset_registry"
    assert result.output_path.exists()
    assert result.metadata_path.exists()
    assert result.n_points > 10_000


def test_study_area_bbox_reads_grid_footprint_with_buffer():
    config = {
        "project": {"name": "marshfield"},
        "grid_footprint": {"source": "locations/marshfield/data/static/aoi/study_area.geojson"},
    }

    west, south, east, north = study_area_bbox(config, repo_root, buffer_degrees=0.01)

    assert west < -70.77
    assert south < 42.06
    assert east > -70.64
    assert north > 42.16
