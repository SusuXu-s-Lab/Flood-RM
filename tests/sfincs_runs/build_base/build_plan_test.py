from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import yaml
from shapely.geometry import LineString, Point, Polygon, MultiPolygon

from sfincs_runs.build_base import (
    add_inland_outflow_boundary,
    build_baseline_build_plan,
    build_inland_sfincs_base,
    build_inland_sfincs_domain_set,
    is_built_sfincs_base,
    plan_inland_sfincs_domain_set,
    plan_inland_sfincs_base,
    build_static_data_catalog,
    build_static_intake_plan,
    write_inland_sfincs_handoff_locations,
    write_inland_sfincs_domain_set_manifest,
)
from sfincs_runs.config import load_runtime


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_baseline_build_plan_selects_wave_coupled_notebook_for_marshfield(tmp_path):
    config = {
        "project": {"name": "marshfield"},
        "coastal_waves": True,
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "notebooks": {
            "build_sfincs": "02_flood/04/a_build_standard.ipynb",
            "build_sfincs_wave_coupled": "02_flood/04/a_build_waves.ipynb",
        },
    }
    paths = {
        "root": tmp_path / "locations/marshfield/data/sfincs",
        "base_model_root": tmp_path / "locations/marshfield/data/sfincs/base",
        "data_catalog": tmp_path / "locations/marshfield/data/static/data_catalogue.yaml",
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations/marshfield",
    }

    plan = build_baseline_build_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.build_kind == "wave_coupled"
    assert plan.truth_set_kind == "wave_coupled_truth_set"
    assert plan.notebook_path == tmp_path / "locations/marshfield/02_flood/04/a_build_waves.ipynb"
    assert plan.grid_footprint_source == tmp_path / "locations/marshfield/data/static/aoi/study_area.geojson"
    assert plan.required_sources == ("era5_waves",)


def test_baseline_build_plan_selects_regular_notebook_without_coastal_waves(tmp_path):
    config = {
        "project": {"name": "austin"},
        "coastal_waves": False,
        "grid_footprint": {"source": "grid_footprint.geojson"},
    }
    paths = {
        "root": tmp_path / "locations/austin/data/sfincs",
        "base_model_root": tmp_path / "locations/austin/data/sfincs/base",
        "data_catalog": tmp_path / "locations/austin/data/static/data_catalogue.yaml",
        "repo_root": tmp_path,
        "location_name": "austin",
        "location_root": tmp_path / "locations/austin",
    }

    plan = build_baseline_build_plan(config, paths)

    assert plan.build_kind == "regular_grid"
    assert plan.truth_set_kind == "hydrodynamic_truth_set"
    assert plan.notebook_path == tmp_path / "locations/austin/02_flood/04/a_build_standard.ipynb"
    assert plan.grid_footprint_source == tmp_path / "locations/austin/grid_footprint.geojson"
    assert plan.required_sources == ()


def test_baseline_build_plan_uses_marshfield_runtime_config():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    plan = build_baseline_build_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.build_kind == "wave_coupled"
    assert suffix(plan.notebook_path) == "locations/marshfield/02_flood/04/a_build_waves.ipynb"
    assert suffix(plan.base_model_root) == "locations/marshfield/data/sfincs/base"
    assert suffix(plan.grid_footprint_source) == "locations/marshfield/data/static/aoi/study_area.geojson"


def test_static_intake_plan_uses_location_data_workspace(tmp_path):
    config = {
        "project": {"name": "marshfield", "model_crs": "EPSG:26919"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
    }
    paths = {
        "location_name": "marshfield",
        "location_root": tmp_path / "locations/marshfield",
        "static_root": tmp_path / "locations/marshfield/data/static/processed",
        "raw_root": tmp_path / "locations/marshfield/data/static/raw",
        "data_catalog": tmp_path / "locations/marshfield/data/static/data_catalogue.yaml",
    }

    plan = build_static_intake_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.model_crs == "EPSG:26919"
    assert plan.static_root == tmp_path / "locations/marshfield/data/static/processed"
    assert plan.raw_root == tmp_path / "locations/marshfield/data/static/raw"
    assert plan.data_catalog == tmp_path / "locations/marshfield/data/static/data_catalogue.yaml"
    assert plan.required_static_inputs == ("terrain", "landcover", "coastline", "ssurgo")


def test_build_static_data_catalog_writes_inland_hydromt_catalog(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
        "static_sources": {
            "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
            "landcover": {"output": "data/static/processed/landcover_region_setup.tif"},
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            },
        },
        "event_catalog": {
            "forcing_members": {
                "rainfall": "data/sources/aorc_sst/rainfall_members.csv",
                "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
                "soil_moisture": "data/sources/nwm/soil_moisture.csv",
            }
        },
        "collection": {
            "usgs_streamgages": {
                "streamflow_records": {
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                }
            }
        },
        "wflow": {
            "handoff": {
                "manifest": "data/wflow/domain_set_handoff.yaml",
            }
        },
    }
    paths = {
        "location_root": location_root,
        "data_catalog": location_root / "data/static/data_catalogue.yaml",
    }

    catalog_path = build_static_data_catalog(config, paths)

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["dem_region"]["uri"] == "processed/dem_region_setup.tif"
    assert catalog["landcover_region"]["uri"] == "processed/landcover_region_setup.tif"
    assert catalog["hydrologic_soil_group"]["uri"] == "soils/hsg_greensboro.tif"
    assert catalog["usgs_streamflow_records"]["uri"] == "../sources/usgs_streamgages/streamflow_records.csv"
    assert catalog["wflow_domain_set_handoff"]["uri"] == "../wflow/domain_set_handoff.yaml"
    assert catalog["dem_region"]["metadata"]["crs"] == "EPSG:32617"


def test_is_built_sfincs_base_rejects_gitkeep_only_directory(tmp_path):
    base = tmp_path / "data/sfincs/base"
    base.mkdir(parents=True)
    (base / ".gitkeep").write_text("", encoding="utf-8")

    assert is_built_sfincs_base(base) is False

    (base / "sfincs.inp").write_text("tref = 20000101 000000\n", encoding="utf-8")
    (base / "sfincs.dep").write_text("dep\n", encoding="utf-8")

    assert is_built_sfincs_base(base) is True


def test_add_inland_outflow_boundary_marks_lowest_perimeter_as_outflow():
    from sfincs_runs.build_base.inland_base import outflow_zmax_from_active_mask

    # 6x6 active domain; elevation increases west->east so the western edge is the low outlet.
    dep = np.tile(np.arange(6, dtype="float64"), (6, 1)) * 10.0  # columns 0..5 -> 0..50 m
    mask = np.ones((6, 6), dtype="uint8")
    zmax = outflow_zmax_from_active_mask(mask, dep, quantile=0.2)
    # Perimeter elevations are {0,10,20,30,40,50}; the 20th percentile sits at the low (west) edge.
    assert zmax < 20.0

    captured = {}

    class FakeMask:
        def create_boundary(self, **kwargs):
            captured.update(kwargs)

    class FakeGrid:
        data = xr.Dataset({"mask": (("y", "x"), mask), "dep": (("y", "x"), dep)})

    class FakeModel:
        grid = FakeGrid()
        mask = FakeMask()

    result = add_inland_outflow_boundary(FakeModel(), quantile=0.2)
    assert result["outflow_zmax_m"] == zmax
    assert captured["btype"] == "outflow"
    assert captured["zmax"] == zmax


def test_build_inland_sfincs_base_calls_hydromt_components_and_write(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    for relative_path in [
        "data/static/aoi/study_area.geojson",
        "data/static/processed/dem_region_setup.tif",
        "data/static/processed/landcover_region_setup.tif",
        "data/static/soils/hsg_greensboro.tif",
        "data/static/soils/ksat_mmhr_greensboro.tif",
        "data/wflow/domain_set_handoff.yaml",
    ]:
        path = location_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    calls = []

    class Component:
        def __init__(self, name, data=None):
            self.name = name
            self.data = data

        def create_from_region(self, **kwargs):
            calls.append((self.name, "create_from_region", kwargs))

        def create(self, **kwargs):
            calls.append((self.name, "create", kwargs))

        def create_active(self, **kwargs):
            calls.append((self.name, "create_active", kwargs))

        def create_boundary(self, **kwargs):
            calls.append((self.name, "create_boundary", kwargs))

        def create_cn_with_recovery(self, **kwargs):
            calls.append((self.name, "create_cn_with_recovery", kwargs))

    class Catalog:
        def from_dict(self, value):
            calls.append(("data_catalog", "from_dict", value))

    class FakeSfincsModel:
        def __init__(self, root, mode, write_gis=True):
            calls.append(("model", "__init__", {"root": root, "mode": mode, "write_gis": write_gis}))
            self.root = Path(root)
            self.data_catalog = Catalog()
            grid_data = xr.Dataset(
                {
                    "mask": (("y", "x"), np.ones((6, 6), dtype="uint8")),
                    "dep": (("y", "x"), np.arange(36, dtype="float32").reshape(6, 6)),
                }
            )
            self.grid = Component("grid", data=grid_data)
            self.elevation = Component("elevation")
            self.mask = Component("mask")
            self.roughness = Component("roughness")
            self.subgrid = Component("subgrid")
            self.infiltration = Component("infiltration")
            self.quadtree_infiltration = Component("quadtree_infiltration")
            self.config = {}

        def write(self):
            calls.append(("model", "write", {}))
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "sfincs.inp").write_text("written\n", encoding="utf-8")
            (self.root / "sfincs.dep").write_text("written\n", encoding="utf-8")

    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
        "paths": {"base_model_root": "data/sfincs/base"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "static_sources": {
            "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
            "landcover": {"output": "data/static/processed/landcover_region_setup.tif"},
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            },
        },
        "sfincs": {"model_crs": "EPSG:32617", "grid_resolution_m": 30},
        "event_drivers": ["rainfall", "streamflow", "soil_moisture"],
        "inland_coupling": {
            "forcing_mode": "dual_fluvial_pluvial",
            "discharge_forcing": {"handoff_manifest": "data/wflow/domain_set_handoff.yaml"},
            "infiltration": {
                "enabled": True,
                "method": "cn_with_recovery",
                "lulc": "landcover_region",
                "hsg": "data/static/soils/hsg_greensboro.tif",
                "ksat": "data/static/soils/ksat_mmhr_greensboro.tif",
                "effective": 0.5,
            },
        },
    }

    plan = plan_inland_sfincs_base(config, {"location_root": location_root})
    assert plan.ready_to_build is True

    summary = build_inland_sfincs_base(
        config,
        {"location_root": location_root},
        model_cls=FakeSfincsModel,
        force=True,
    )

    assert summary["status"] == "built"
    assert is_built_sfincs_base(location_root / "data/sfincs/base") is True
    assert ("grid", "create_from_region") in [(name, method) for name, method, _ in calls]
    assert ("elevation", "create") in [(name, method) for name, method, _ in calls]
    assert ("mask", "create_active") in [(name, method) for name, method, _ in calls]
    outflow_call = next(
        kwargs for name, method, kwargs in calls if (name, method) == ("mask", "create_boundary")
    )
    assert outflow_call["btype"] == "outflow"
    assert "zmax" in outflow_call
    assert ("subgrid", "create") in [(name, method) for name, method, _ in calls]
    roughness_call = next(kwargs for name, method, kwargs in calls if (name, method) == ("roughness", "create"))
    subgrid_call = next(kwargs for name, method, kwargs in calls if (name, method) == ("subgrid", "create"))
    assert roughness_call["roughness_list"][0]["reclass_table"].endswith("esa_worldcover_mapping.csv")
    assert subgrid_call["roughness_list"][0]["reclass_table"].endswith("esa_worldcover_mapping.csv")
    assert ("infiltration", "create_cn_with_recovery") in [
        (name, method) for name, method, _ in calls
    ]
    assert ("quadtree_infiltration", "create_cn_with_recovery") not in [
        (name, method) for name, method, _ in calls
    ]
    assert ("model", "write") in [(name, method) for name, method, _ in calls]


def test_write_inland_sfincs_handoff_locations_uses_reviewed_handoff_gages(tmp_path):
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
        "review_status": "accepted",
        "sfincs_handoff_id": "south_buffalo_02095000",
        "wflow_submodel_id": "south_buffalo",
        "sfincs_domain_id": "greensboro_main"
      },
      "geometry": {"type": "Point", "coordinates": [-79.75, 36.05]}
    },
    {
      "type": "Feature",
      "properties": {
        "site_no": "02093877",
        "review_status": "accepted_with_warning",
        "sfincs_handoff_id": null,
        "wflow_submodel_id": "brush_creek",
        "sfincs_domain_id": "greensboro_main"
      },
      "geometry": {"type": "Point", "coordinates": [-79.90, 36.13]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    summary = write_inland_sfincs_handoff_locations(
        {
            "project": {"model_crs": "EPSG:32617"},
            "sfincs": {"model_crs": "EPSG:32617"},
            "wflow": {
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                }
            },
        },
        {"location_root": location_root},
    )

    out = summary["handoff_locations"]
    handoff = __import__("geopandas").read_file(out)
    assert summary["source_point_count"] == 1
    assert handoff["index"].tolist() == [1]
    assert handoff["name"].tolist() == ["south_buffalo_02095000"]
    assert handoff.crs.to_epsg() == 32617


def test_write_inland_sfincs_handoff_locations_can_snap_to_domain_boundary(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    region_path = location_root / "data/sfincs/domains/greensboro_west/region.geojson"
    network_path.parent.mkdir(parents=True)
    region_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_west"]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(region_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["greensboro_main"],
            "sfincs_domain_id": ["greensboro_west"],
        },
        geometry=[Point(1, 4)],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")

    summary = write_inland_sfincs_handoff_locations(
        {
            "project": {"model_crs": "EPSG:32617"},
            "sfincs": {"model_crs": "EPSG:32617"},
            "wflow": {
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                }
            },
            "inland_coupling": {
                "discharge_forcing": {"handoff_location": "sfincs_domain_boundary"}
            },
        },
        {"location_root": location_root},
        domain_region=region_path,
    )

    handoff = gpd.read_file(summary["handoff_locations"]).to_crs("EPSG:32617")
    point = handoff.geometry.iloc[0]
    assert point.x == pytest.approx(0.0, abs=1.0e-6)
    assert point.y == pytest.approx(4.0, abs=1.0e-6)
    assert handoff["site_no"].tolist() == ["02095000"]


def test_write_inland_sfincs_handoff_locations_uses_wflow_native_stream_entry(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    region_path = location_root / "data/sfincs/domains/greensboro_main/region.geojson"
    river_path = location_root / "data/wflow/base/south_buffalo/staticgeoms/rivers.geojson"
    fallback_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    network_path.parent.mkdir(parents=True)
    region_path.parent.mkdir(parents=True)
    river_path.parent.mkdir(parents=True)
    fallback_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_main"]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(region_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"idx": [7], "strord": [2]},
        geometry=[LineString([(-5, 5), (0, 5), (10, 5), (15, 5)])],
        crs="EPSG:3857",
    ).to_file(river_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"nhdplusid": [999]},
        geometry=[LineString([(-5, 8), (0, 8), (10, 8), (15, 8)])],
        crs="EPSG:3857",
    ).to_file(fallback_path, driver="GPKG")

    summary = write_inland_sfincs_handoff_locations(
        {
            "project": {"model_crs": "EPSG:3857"},
            "sfincs": {"model_crs": "EPSG:3857"},
            "wflow": {
                "base_model_root": "data/wflow/base",
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                },
            },
            "collection": {
                "national_hydrography": {
                    "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
                }
            },
            "inland_coupling": {
                "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
            },
        },
        {"location_root": location_root},
        domain_region=region_path,
    )

    handoff = gpd.read_file(summary["handoff_locations"]).to_crs("EPSG:3857")
    point = handoff.geometry.iloc[0]
    assert point.x == pytest.approx(0.0, abs=1.0e-6)
    assert point.y == pytest.approx(5.0, abs=1.0e-6)
    assert handoff["stream_boundary_river_source"].tolist() == ["hydromt_wflow_setup_rivers"]
    assert handoff["stream_boundary_river_id"].tolist() == ["7"]
    assert handoff["handoff_location_review_status"].tolist() == [
        "review_required_stream_boundary_intersection"
    ]


def test_crossing_derived_sfincs_handoff_snaps_to_nearest_boundary_intersection(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    region_path = location_root / "data/sfincs/domains/greensboro_main/region.geojson"
    plan_river_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    native_river_path = location_root / "data/wflow/base/greensboro_main_inflow_01/staticgeoms/rivers.geojson"
    for path in (network_path, bbox_path, region_path, plan_river_path, native_river_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    region = gpd.GeoDataFrame(
        {"subregion_id": ["greensboro_main"]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])],
        crs="EPSG:3857",
    )
    region.to_crs("EPSG:4326").to_file(bbox_path, driver="GeoJSON")
    region.to_crs("EPSG:4326").to_file(region_path, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857").to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"TotDASqKm": [100.0]},
        geometry=[LineString([(15, 5), (5, 5)])],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(plan_river_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"idx": [6, 7], "strord": [1, 2]},
        geometry=[
            LineString([(9.8, 5), (9.9, 5)]),
            LineString([(5, 5), (9, 5), (10, 6), (15, 6)]),
        ],
        crs="EPSG:3857",
    ).to_file(native_river_path, driver="GeoJSON")

    summary = write_inland_sfincs_handoff_locations(
        {
            "project": {"name": "greensboro", "model_crs": "EPSG:3857"},
            "sfincs": {"model_crs": "EPSG:3857"},
            "sfincs_domain_set": {"source": "data/static/aoi/bbox.geojson", "allow_multiple_domains": True},
            "wflow": {
                "base_model_root": "data/wflow/base",
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                },
                "domain_set": {
                    "outlet_source": "stream_boundary_crossings",
                    "crossings": {"min_uparea_km2": 5.0},
                },
            },
            "collection": {
                "national_hydrography": {
                    "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
                }
            },
            "inland_coupling": {
                "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
            },
        },
        {"location_root": location_root},
        handoff_source_ids=["greensboro_main_inflow_01"],
        domain_region=region_path,
        sfincs_domain_id="greensboro_main",
    )

    handoff = gpd.read_file(summary["handoff_locations"]).to_crs("EPSG:3857")
    point = handoff.geometry.iloc[0]
    assert point.x == pytest.approx(10.0, abs=1.0e-6)
    assert point.y == pytest.approx(6.0, abs=1.0e-6)
    assert handoff["stream_boundary_river_source"].tolist() == ["hydromt_wflow_setup_rivers"]


def test_write_inland_sfincs_handoff_locations_falls_back_to_review_hydrography(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    region_path = location_root / "data/sfincs/domains/greensboro_main/region.geojson"
    fallback_path = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    network_path.parent.mkdir(parents=True)
    region_path.parent.mkdir(parents=True)
    fallback_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_main"]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(region_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"nhdplusid": [999]},
        geometry=[LineString([(-5, 5), (0, 5), (10, 5), (15, 5)])],
        crs="EPSG:3857",
    ).to_file(fallback_path, driver="GPKG")

    summary = write_inland_sfincs_handoff_locations(
        {
            "project": {"model_crs": "EPSG:3857"},
            "sfincs": {"model_crs": "EPSG:3857"},
            "wflow": {
                "base_model_root": "data/wflow/base",
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                },
            },
            "collection": {
                "national_hydrography": {
                    "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
                }
            },
            "inland_coupling": {
                "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
            },
        },
        {"location_root": location_root},
        domain_region=region_path,
    )

    handoff = gpd.read_file(summary["handoff_locations"]).to_crs("EPSG:3857")
    assert handoff.geometry.iloc[0].x == pytest.approx(0.0, abs=1.0e-6)
    assert handoff["stream_boundary_river_source"].tolist() == ["review_hydrography_fallback"]
    assert handoff["stream_boundary_river_id"].tolist() == ["999"]


def test_write_inland_sfincs_handoff_locations_requires_stream_boundary_crossing(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    region_path = location_root / "data/sfincs/domains/greensboro_main/region.geojson"
    river_path = location_root / "data/wflow/base/south_buffalo/staticgeoms/rivers.geojson"
    network_path.parent.mkdir(parents=True)
    region_path.parent.mkdir(parents=True)
    river_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_main"]},
        geometry=[Polygon([(0, 0), (20, 0), (20, 10), (0, 10), (0, 0)])],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(region_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(15, 5)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"idx": [7], "strord": [2]},
        geometry=[LineString([(5, 5), (15, 5)])],
        crs="EPSG:3857",
    ).to_file(river_path, driver="GeoJSON")

    with pytest.raises(ValueError, match="No stream/SFINCS-boundary intersections"):
        write_inland_sfincs_handoff_locations(
            {
                "project": {"model_crs": "EPSG:3857"},
                "sfincs": {"model_crs": "EPSG:3857"},
                "wflow": {
                    "base_model_root": "data/wflow/base",
                    "streamgage_network": {
                        "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                    },
                },
                "inland_coupling": {
                    "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
                },
            },
            {"location_root": location_root},
            domain_region=region_path,
        )


def test_plan_inland_sfincs_domain_set_preserves_disconnected_exposure_regions_and_assigns_handoffs(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    footprint = location_root / "data/static/aoi/study_area.geojson"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    footprint.parent.mkdir(parents=True)
    network_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"name": ["smart_ds_exposure"]},
        geometry=[
            MultiPolygon(
                [
                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
                    Polygon([(30, 0), (40, 0), (40, 10), (30, 10), (30, 0)]),
                ]
            )
        ],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(footprint, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02094500", "02093877"],
            "review_status": ["accepted", "accepted_with_warning", "accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000", "reedy_fork_02094500", None],
            "wflow_submodel_id": ["south_buffalo", "reedy_fork", "brush_creek"],
            "sfincs_domain_id": ["greensboro_main", "greensboro_main", "greensboro_main"],
        },
        geometry=[Point(9, 5), Point(33, 5), Point(7, 5)],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "paths": {"base_model_root": "data/sfincs/base"},
        "sfincs": {"model_crs": "EPSG:32617", "grid_resolution_m": 30},
        "sfincs_domain_set": {
            "enabled": True,
            "domains_root": "data/sfincs/domains",
            "domain_manifest": "data/sfincs/domains/domain_set.yaml",
            "event_catalog_scope": "shared_across_domain_set",
            "evaluation_merge": "max_depth_per_asset_with_source_domain",
        },
        "wflow": {
            "streamgage_network": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
            }
        },
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.domain_count == 2
    assert [domain["sfincs_domain_id"] for domain in plan.domains] == ["greensboro_west", "greensboro_east"]
    assert plan.domains[0]["handoff_source_ids"] == ["south_buffalo_02095000"]
    assert plan.domains[1]["handoff_source_ids"] == ["reedy_fork_02094500"]
    assert plan.domains[0]["base_model_root"] == location_root / "data/sfincs/domains/greensboro_west/base"
    assert plan.domains[1]["region"] == location_root / "data/sfincs/domains/greensboro_east/region.geojson"

    manifest_path = write_inland_sfincs_domain_set_manifest(plan, config, {"location_root": location_root})
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["event_catalog_scope"] == "shared_across_domain_set"
    assert manifest["evaluation_merge"] == "max_depth_per_asset_with_source_domain"
    assert [domain["sfincs_domain_id"] for domain in manifest["domains"]] == ["greensboro_west", "greensboro_east"]
    assert Path(manifest["domains"][0]["region"]).exists()


def test_plan_inland_sfincs_domain_set_can_use_bbox_regions_for_exposure_components(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    footprint = location_root / "data/static/aoi/study_area.geojson"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    footprint.parent.mkdir(parents=True)
    network_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"name": ["smart_ds_exposure"]},
        geometry=[
            Polygon([(0, 0), (10, 0), (8, 7), (0, 10), (0, 0)]),
        ],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(footprint, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(9, 5)],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "paths": {"base_model_root": "data/sfincs/base"},
        "sfincs": {"model_crs": "EPSG:32617", "grid_resolution_m": 30},
        "sfincs_domain_set": {
            "enabled": True,
            "domains_root": "data/sfincs/domains",
            "domain_manifest": "data/sfincs/domains/domain_set.yaml",
            "region_geometry": "bounding_box",
        },
        "wflow": {
            "streamgage_network": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
            }
        },
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.domains[0]["geometry"].equals_exact(plan.domains[0]["geometry"].envelope, tolerance=0)


def test_plan_inland_sfincs_domain_set_uses_smart_ds_subregion_ids_without_handoffs(tmp_path):
    location_root = tmp_path / "locations/austin"
    footprint = location_root / "data/static/aoi/evaluation_footprint.geojson"
    footprint.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"subregion_id": ["P1U", "P2U"]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]),
            Polygon([(30, 0), (40, 0), (40, 6), (30, 6), (30, 0)]),
        ],
        crs="EPSG:32614",
    ).to_crs("EPSG:4326").to_file(footprint, driver="GeoJSON")
    config = {
        "project": {"name": "austin", "model_crs": "EPSG:32614"},
        "sfincs": {"model_crs": "EPSG:32614", "grid_resolution_m": 100},
        "sfincs_domain_set": {
            "source": "data/static/aoi/evaluation_footprint.geojson",
            "domains_root": "data/sfincs/domains",
            "domain_manifest": "data/sfincs/domains/domain_set.yaml",
            "allow_multiple_domains": True,
            "region_geometry": "bounding_box",
        },
        "wflow": {
            "streamgage_network": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
            }
        },
        "inland_coupling": {
            "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
        },
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.status == "needs_review"
    assert plan.domain_count == 2
    assert plan.handoff_count == 0
    assert [domain["sfincs_domain_id"] for domain in plan.domains] == ["austin_p1u", "austin_p2u"]
    assert [domain["exposure_subregion_id"] for domain in plan.domains] == ["P1U", "P2U"]

    manifest_path = write_inland_sfincs_domain_set_manifest(plan, config, {"location_root": location_root})
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "needs_review"
    assert [domain["sfincs_domain_id"] for domain in manifest["domains"]] == ["austin_p1u", "austin_p2u"]
    assert [domain["exposure_subregion_id"] for domain in manifest["domains"]] == ["P1U", "P2U"]


def test_plan_inland_sfincs_domain_set_keeps_stream_boundary_bbox_on_exposure(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    footprint = location_root / "data/static/aoi/evaluation_footprint.geojson"
    network_path = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    footprint.parent.mkdir(parents=True)
    network_path.parent.mkdir(parents=True)
    exposure = Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)])
    gpd.GeoDataFrame(
        {"name": ["smart_ds_exposure"]},
        geometry=[exposure],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(footprint, driver="GeoJSON")
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000"],
            "review_status": ["accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000"],
            "wflow_submodel_id": ["south_buffalo"],
            "sfincs_domain_id": ["greensboro_main"],
        },
        geometry=[Point(30, 3)],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326").to_file(network_path, driver="GeoJSON")
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
        "sfincs": {"model_crs": "EPSG:32617", "grid_resolution_m": 30},
        "sfincs_domain_set": {
            "source": "data/static/aoi/evaluation_footprint.geojson",
            "domains_root": "data/sfincs/domains",
            "domain_manifest": "data/sfincs/domains/domain_set.yaml",
            "allow_multiple_domains": False,
            "region_geometry": "bounding_box",
        },
        "wflow": {
            "streamgage_network": {
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
            }
        },
        "inland_coupling": {
            "discharge_forcing": {"handoff_location": "stream_boundary_intersection"}
        },
    }

    plan = plan_inland_sfincs_domain_set(config, {"location_root": location_root})

    assert plan.status == "ready"
    assert plan.domains[0]["handoff_source_ids"] == ["south_buffalo_02095000"]
    assert tuple(round(value, 6) for value in plan.domains[0]["geometry"].bounds) == (0, 0, 10, 6)


def test_build_inland_sfincs_domain_set_builds_each_ready_domain(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    for relative_path in [
        "data/static/processed/dem_region_setup.tif",
        "data/static/processed/landcover_region_setup.tif",
        "data/static/soils/hsg_greensboro.tif",
        "data/static/soils/ksat_mmhr_greensboro.tif",
        "data/wflow/domain_set_handoff.yaml",
    ]:
        path = location_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    domain_root = location_root / "data/sfincs/domains"
    for domain_id in ["greensboro_west", "greensboro_east"]:
        region = domain_root / domain_id / "region.geojson"
        region.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(
            {"sfincs_domain_id": [domain_id]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
            crs="EPSG:4326",
        ).to_file(region, driver="GeoJSON")
    manifest = domain_root / "domain_set.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "status": "ready",
                "domains": [
                    {
                        "sfincs_domain_id": "greensboro_west",
                        "region": str(domain_root / "greensboro_west/region.geojson"),
                        "base_model_root": str(domain_root / "greensboro_west/base"),
                        "handoff_source_ids": ["south_buffalo_02095000"],
                    },
                    {
                        "sfincs_domain_id": "greensboro_east",
                        "region": str(domain_root / "greensboro_east/region.geojson"),
                        "base_model_root": str(domain_root / "greensboro_east/base"),
                        "handoff_source_ids": ["reedy_fork_02094500"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    network = location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    network.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            "site_no": ["02095000", "02094500"],
            "review_status": ["accepted", "accepted"],
            "sfincs_handoff_id": ["south_buffalo_02095000", "reedy_fork_02094500"],
            "wflow_submodel_id": ["south_buffalo", "reedy_fork"],
            "sfincs_domain_id": ["greensboro_main", "greensboro_main"],
        },
        geometry=[Point(-79.75, 36.05), Point(-79.61, 36.17)],
        crs="EPSG:4326",
    ).to_file(network, driver="GeoJSON")
    calls = []

    class Component:
        def __init__(self, name, data=None):
            self.name = name
            self.data = data

        def create_from_region(self, **kwargs):
            calls.append((self.name, "create_from_region", kwargs))

        def create(self, **kwargs):
            calls.append((self.name, "create", kwargs))

        def create_active(self, **kwargs):
            calls.append((self.name, "create_active", kwargs))

        def create_boundary(self, **kwargs):
            calls.append((self.name, "create_boundary", kwargs))

    class Catalog:
        def from_dict(self, value):
            calls.append(("data_catalog", "from_dict", value))

    class FakeSfincsModel:
        def __init__(self, root, mode, write_gis=True):
            calls.append(("model", "__init__", {"root": root, "mode": mode, "write_gis": write_gis}))
            self.root = Path(root)
            self.data_catalog = Catalog()
            grid_data = xr.Dataset(
                {
                    "mask": (("y", "x"), np.ones((6, 6), dtype="uint8")),
                    "dep": (("y", "x"), np.arange(36, dtype="float32").reshape(6, 6)),
                }
            )
            self.grid = Component("grid", data=grid_data)
            self.elevation = Component("elevation")
            self.mask = Component("mask")
            self.roughness = Component("roughness")
            self.subgrid = Component("subgrid")
            self.config = {}

        def write(self):
            calls.append(("model", "write", {}))
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "sfincs.inp").write_text("written\n", encoding="utf-8")
            (self.root / "sfincs.dep").write_text("written\n", encoding="utf-8")

    report = build_inland_sfincs_domain_set(
        {
            "project": {"name": "greensboro", "model_crs": "EPSG:32617"},
            "static_sources": {
                "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
                "landcover": {"output": "data/static/processed/landcover_region_setup.tif"},
            },
            "sfincs": {"model_crs": "EPSG:32617", "grid_resolution_m": 30},
            "sfincs_domain_set": {"domain_manifest": "data/sfincs/domains/domain_set.yaml"},
            "wflow": {
                "streamgage_network": {
                    "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson"
                }
            },
            "inland_coupling": {
                "discharge_forcing": {"handoff_manifest": "data/wflow/domain_set_handoff.yaml"},
                "infiltration": {
                    "hsg": "data/static/soils/hsg_greensboro.tif",
                    "ksat": "data/static/soils/ksat_mmhr_greensboro.tif",
                },
            },
        },
        {"location_root": location_root},
        model_cls=FakeSfincsModel,
        force=True,
    )

    assert report["sfincs_domain_id"].tolist() == ["greensboro_west", "greensboro_east"]
    assert report["status"].tolist() == ["built", "built"]
    assert is_built_sfincs_base(domain_root / "greensboro_west/base")
    assert is_built_sfincs_base(domain_root / "greensboro_east/base")
    assert (domain_root / "greensboro_west/base/gis/wflow_handoff_sources.geojson").exists()
    assert (domain_root / "greensboro_east/base/gis/wflow_handoff_sources.geojson").exists()
    west_handoff = gpd.read_file(domain_root / "greensboro_west/base/gis/wflow_handoff_sources.geojson")
    east_handoff = gpd.read_file(domain_root / "greensboro_east/base/gis/wflow_handoff_sources.geojson")
    assert west_handoff["name"].tolist() == ["south_buffalo_02095000"]
    assert east_handoff["name"].tolist() == ["reedy_fork_02094500"]
    assert west_handoff["sfincs_domain_id"].tolist() == ["greensboro_west"]
    assert east_handoff["sfincs_domain_id"].tolist() == ["greensboro_east"]
    assert [call[2]["root"] for call in calls if call[:2] == ("model", "__init__")] == [
        str(domain_root / "greensboro_west/base"),
        str(domain_root / "greensboro_east/base"),
    ]
