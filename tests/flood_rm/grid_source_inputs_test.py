from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

import pytest

from power.resilience import load_critical_facilities
from power.baseline_network.source_inputs import (
    SourceAnchorReviewRequired,
    source_anchors,
    source_area,
)
from study_location import define_location


repo_root = Path(__file__).resolve().parents[2]


def test_marshfield_location_defines_place_based_grid_source_inputs():
    location = define_location(repo_root / "locations/marshfield/config.yaml")

    assert location.config["project"]["place_name"] == "Marshfield, MA, USA"
    assert location.config["project"]["country"] == "US"
    assert location.grid["source_area"] == {
        "source": "osmnx_place",
        "place_name": "project.place_name",
        "extra_areas": [
            {
                "name": "humarock_barrier_strip",
                "bbox": [-70.725, 42.088, -70.655, 42.145],
            }
        ],
    }


def test_location_critical_facilities_load_from_location_local_static_input():
    location = define_location(repo_root / "locations/marshfield/config.yaml")
    source = location.root / location.grid["critical_facilities"]["source"]

    facilities = load_critical_facilities(source, location_name=location.name)

    assert source == repo_root / "locations/marshfield/data/static/grid/critical_facilities.geojson"
    assert len(facilities) >= 1
    assert set(facilities["sandbox_id"]) == {"marshfield"}
    assert facilities["facility_id"].str.startswith("marshfield:").all()
    assert {"facility_name", "lifeline", "lon", "lat", "source_provenance"}.issubset(
        facilities.columns
    )
    assert location.grid["source_anchors"] == {
        "source": "osm_power_substations",
        "place_name": "project.place_name",
        "candidate_output": "data/static/grid/source_anchor_candidates.geojson",
        "reviewed_override": "data/static/grid/source_anchors.geojson",
        "accept_unreviewed_source_anchors": False,
    }


def test_grid_source_area_resolves_place_identity_and_named_patches():
    location = define_location(repo_root / "locations/marshfield/config.yaml")

    def geocode_place(place_name):
        assert place_name == "Marshfield, MA, USA"
        return gpd.GeoDataFrame(geometry=[box(-70.8, 42.0, -70.7, 42.1)], crs="EPSG:4326")

    source_area = source_area(location.config, geocode_place=geocode_place)

    assert source_area.place_name == "Marshfield, MA, USA"
    assert source_area.patch_names == ("humarock_barrier_strip",)
    assert tuple(round(value, 3) for value in source_area.geometry.bounds) == (
        -70.8,
        42.0,
        -70.655,
        42.145,
    )


def test_grid_source_anchors_write_candidates_and_require_review(tmp_path):
    location = define_location(repo_root / "locations/marshfield/config.yaml")
    config = dict(location.config)
    config["grid"] = dict(location.config["grid"])
    config["grid"]["source_anchors"] = {
        **location.config["grid"]["source_anchors"],
        "candidate_output": "data/static/grid/source_anchor_candidates.geojson",
        "reviewed_override": "data/static/grid/source_anchors.geojson",
    }

    def fetch_candidates(_geometry):
        return [
            {
                "name": "Marshfield Substation",
                "substation_id": "osm:node:1",
                "lon": -70.72,
                "lat": 42.09,
                "source": "osm_power_substations",
            }
        ]

    with pytest.raises(SourceAnchorReviewRequired) as exc:
        source_anchors(
            config,
            location_root=tmp_path,
            source_area_geometry=box(-70.8, 42.0, -70.6, 42.2),
            fetch_candidates=fetch_candidates,
        )

    assert exc.value.candidate_path == tmp_path / "data/static/grid/source_anchor_candidates.geojson"
    assert exc.value.reviewed_path == tmp_path / "data/static/grid/source_anchors.geojson"
    assert exc.value.candidate_path.exists()


def test_grid_source_anchors_reuse_reviewed_location_artifact(tmp_path):
    location = define_location(repo_root / "locations/marshfield/config.yaml")
    reviewed_path = tmp_path / "data/static/grid/source_anchors.geojson"
    reviewed_path.parent.mkdir(parents=True)
    reviewed_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [-70.72, 42.09]},
      "properties": {
        "name": "Reviewed Substation",
        "substation_id": "reviewed:1",
        "source": "reviewed_source_anchor_artifact"
      }
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fetch_candidates(_geometry):
        raise AssertionError("reviewed source anchors should not fetch live candidates")

    anchors = source_anchors(
        location.config,
        location_root=tmp_path,
        source_area_geometry=box(-70.8, 42.0, -70.6, 42.2),
        fetch_candidates=fetch_candidates,
    )

    assert anchors == [
        {
            "name": "Reviewed Substation",
            "substation_id": "reviewed:1",
            "source": "reviewed_source_anchor_artifact",
            "lon": -70.72,
            "lat": 42.09,
        }
    ]
