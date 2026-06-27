from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import LineString, Point, box

from sfincs_runs.build_base import (
    build_static_data_catalog,
    plan_inland_sfincs_base,
    plot_sfincs_handoff_basemap,
    set_observations,
    sfincs_grid_resolution_matches,
)
from fiat_runs.build_model import fiat_model_inputs
from study_location import define_location


repo_root = Path(__file__).resolve().parents[2]


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _workflow_step(workflow, name):
    for step in workflow.get("steps", []):
        if isinstance(step, dict) and name in step:
            return step[name]
    raise AssertionError(f"missing workflow step: {name}")


def test_location_model_recipes_use_native_hydromt_shapes_from_model_yaml():
    for name in ("austin", "greensboro", "marshfield"):
        location_root = repo_root / "locations" / name
        definition = define_location(location_root / "config.yaml")

        assert definition.model_recipes
        assert "sfincs_build" in definition.model_recipes
        assert "setup_grid_from_region" in definition.model_recipes["sfincs_build"]
        if "wflow_build" in definition.model_recipes:
            assert "steps" in definition.model_recipes["wflow_build"]
        if "snapwave_build" in definition.model_recipes:
            assert "setup_snapwave" in definition.model_recipes["snapwave_build"]


def test_inland_locations_declare_wflow_build_and_update_recipes():
    for name in ("austin", "greensboro"):
        location_root = repo_root / "locations" / name
        definition = define_location(location_root / "config.yaml")

        assert definition.config["includes"]["wflow"] == "wflow.yaml"
        assert (location_root / "wflow.yaml").exists()
        assert definition.config["wflow"]["build_config"] == "data/wflow/config/wflow_build.yml"
        assert definition.config["wflow"]["update_forcing_config"] == "data/wflow/config/wflow_update_forcing.yml"
        build_steps = {
            next(iter(step))
            for step in definition.model_recipes["wflow_build"]["steps"]
        }
        assert build_steps >= {
            "setup_config",
            "setup_basemaps",
            "setup_rivers",
            "setup_lulcmaps",
            "setup_soilmaps",
            "setup_constant_pars",
            "setup_gauges",
        }
        assert (
            _workflow_step(definition.model_recipes["wflow_build"], "setup_rivers")["river_geom_fn"]
            == "nhdplus_hr_river_geometry"
        )
        update_steps = [
            next(iter(step))
            for step in definition.model_recipes["wflow_update_forcing"]["steps"]
        ]
        assert update_steps == [
            "setup_config",
            "setup_precip_forcing",
            "setup_temp_pet_forcing",
        ]


def test_inland_wflow_river_threshold_matches_boundary_handoff_threshold():
    for name in ("austin", "greensboro"):
        location_root = repo_root / "locations" / name
        definition = define_location(location_root / "config.yaml")

        handoff_uparea = definition.config["wflow"]["domain_set"]["crossings"]["min_uparea_km2"]
        river_upa = _workflow_step(definition.model_recipes["wflow_build"], "setup_rivers")["river_upa"]

        assert river_upa == handoff_uparea


def test_inland_sfincs_domains_use_high_resolution_dem_and_60m_grid():
    for name in ("austin", "greensboro"):
        location_root = repo_root / "locations" / name
        definition = define_location(location_root / "config.yaml")
        plan = plan_inland_sfincs_base(definition.config, {"location_root": location_root})

        assert definition.config["sfincs"]["grid_resolution_m"] == 60
        assert definition.model_recipes["sfincs_build"]["setup_grid_from_region"]["res"] == 60
        assert plan.grid_resolution_m == 60
        assert plan.dem == location_root / "data/static/processed/dem_region_setup.tif"
        assert "wflow" not in plan.dem.name


def test_sfincs_grid_resolution_matches_existing_regular_grid(tmp_path):
    root = tmp_path / "sfincs"
    root.mkdir()
    (root / "sfincs.inp").write_text("dx = 100\n dy = 100\n", encoding="utf-8")

    assert sfincs_grid_resolution_matches(root, 100)
    assert not sfincs_grid_resolution_matches(root, 60)


def test_wflow_basemap_renders_filled_wflow_and_sfincs_domain_areas():
    text = (repo_root / "src/wflow_runs/visualize.py").read_text(encoding="utf-8")

    assert 'label="Wflow basin"' in text
    assert 'facecolor="#d9d9d9"' in text
    assert 'label="SFINCS domain"' in text
    assert 'sfincs_domains.plot(' in text


def test_sfincs_observation_points_from_reviewed_gages_adds_name_and_filters_to_region():
    class ObservationPoints:
        def __init__(self):
            self.gdf = None

        def set(self, gdf, merge=True):
            self.gdf = gdf

    class Model:
        crs = "EPSG:4326"

        def __init__(self):
            self.region = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs=self.crs)
            self.observation_points = ObservationPoints()

    model = Model()
    gages = gpd.GeoDataFrame(
        {"site_no": ["inside", "outside"]},
        geometry=[Point(1, 1), Point(5, 5)],
        crs="EPSG:4326",
    )

    plotted = set_observations(model, gages)

    assert plotted["name"].tolist() == ["inside"]
    assert model.observation_points.gdf["name"].tolist() == ["inside"]


def test_sfincs_handoff_basemap_reports_visible_native_overlays():
    class Model:
        crs = "EPSG:4326"

        def __init__(self):
            self.region = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs=self.crs)

        def plot_basemap(self, **kwargs):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([], [], label="src")
            ax.plot([], [], label="obs")
            ax.plot([], [], label="rivers")
            self.plot_kwargs = kwargs
            return fig, ax

    model = Model()
    src = gpd.GeoDataFrame(
        geometry=[Point(0.5, 0.5), Point(0.5, 0.5), Point(1.5, 1.5)],
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        geometry=[
            LineString([(-1, 1), (1, 1)]),
            LineString([(3, 3), (4, 4)]),
        ],
        crs="EPSG:4326",
    )
    observations = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")

    fig, ax, qa = plot_sfincs_handoff_basemap(
        model,
        handoff_sources=src,
        rivers=rivers,
        observations=observations,
    )

    assert model.plot_kwargs["plot_bounds"] is True
    assert model.plot_kwargs["plot_geoms"] is True
    assert model.plot_kwargs["geom_names"] == ["src", "obs", "rivers"]
    assert "zorder" not in model.plot_kwargs["geom_kwargs"]["src"]
    assert "zorder" not in model.plot_kwargs["geom_kwargs"]["obs"]
    assert "zorder" not in model.plot_kwargs["geom_kwargs"]["rivers"]
    assert qa == {
        "discharge_sources_src": 3,
        "unique_discharge_source_locations": 2,
        "visible_wflow_native_river_features": 1,
        "reviewed_usgs_gages_visible": 1,
    }
    fig.clear()


def test_static_data_catalog_uses_actual_region_raster_crs():
    for name in ("austin", "greensboro"):
        catalog = read_yaml(repo_root / "locations" / name / "data/static/data_catalogue.yaml")

        assert catalog["dem_region"]["metadata"]["crs"] == "EPSG:4269"
        assert catalog["landcover_region"]["metadata"]["crs"] == "EPSG:4269"


def test_coastal_static_data_catalog_adds_snapwave_recipe_aliases(tmp_path):
    location_root = tmp_path / "locations/marshfield"
    catalog_path = location_root / "data/static/data_catalogue.yaml"
    config = {
        "project": {"model_crs": "EPSG:26919"},
        "coastal_waves": True,
        "static_sources": {
            "terrain": {"output": "data/static/processed/cudem_region_setup.tif"},
            "landcover": {"output": "data/static/processed/worldcover_region_setup.tif"},
            "ssurgo": {
                "hsg_output": "data/static/soils/hsg_mfield.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_mfield.tif",
            },
        },
    }

    build_static_data_catalog(config, {"location_root": location_root, "data_catalog": catalog_path})

    catalog = read_yaml(catalog_path)
    assert catalog["dem_region"]["uri"] == "processed/cudem_region_setup.tif"
    assert catalog["landcover_region"]["uri"] == "processed/worldcover_region_setup.tif"
    assert catalog["cudem_elv"]["uri"] == "processed/cudem_region_setup.tif"
    assert catalog["worldcover"]["uri"] == "processed/worldcover_region_setup.tif"


def test_marshfield_has_no_wflow_recipe_requirement():
    definition = define_location(repo_root / "locations/marshfield/config.yaml")

    assert "wflow" not in definition.config["includes"]
    assert "wflow_build" not in definition.model_recipes
    assert "snapwave_build" in definition.model_recipes


def test_marshfield_snapwave_recipe_is_explicit_location_yaml():
    snapwave_yaml = read_yaml(repo_root / "locations/marshfield/snapwave.yaml")
    raw_build = snapwave_yaml["hydromt"]["build"]
    definition = define_location(repo_root / "locations/marshfield/config.yaml")
    recipe_build = definition.model_recipes["snapwave_build"]
    recipe_update = definition.model_recipes["snapwave_update_forcing"]

    assert "setup_dep" in raw_build
    assert "setup_subgrid" in raw_build
    assert recipe_build == raw_build
    assert recipe_build["setup_grid_from_region"]["res"] == 60
    assert recipe_build["setup_grid_from_region"]["quadtree"] is True
    assert recipe_build["setup_dep"]["datasets_dep"] == [{"elevtn": "cudem_elv"}]
    assert recipe_build["setup_mask_bounds"]["btype"] == "waterlevel"
    assert recipe_build["setup_mask_bounds"]["buffer"] == 180
    assert "setup_snapwave" in recipe_build
    assert "setup_runup_gauges" in recipe_build
    assert recipe_update["setup_precip_forcing"]["precip_fn"] == "event_precip"
    assert recipe_update["setup_waterlevel_forcing"]["timeseries"] == "cora_waterlevel"
    assert recipe_update["setup_wave_forcing"]["geodataset"] == "era5_waves"


def test_marshfield_fiat_recipe_lives_at_location_root():
    location_root = repo_root / "locations" / "marshfield"
    inputs = fiat_model_inputs(
        {"coastal_wave_coupling": {"quadtree": {"base_model_root": "data/sfincs/base_quadtree_snapwave"}}},
        {"location_root": location_root},
    )

    assert inputs["config_yml"] == location_root / "fiat_config.yml"
    assert inputs["config_yml"].exists()
    assert not (location_root / "hydromt_config").exists()


def test_flood_notebook_startup_uses_location_root_cwd():
    notebooks = []
    for name in ("austin", "greensboro", "marshfield"):
        for folder in ("02_flood", "03_clean_flood"):
            root = repo_root / "locations" / name / folder
            if root.exists():
                notebooks.extend(root.rglob("*.ipynb"))

    for notebook in notebooks:
        text = notebook.read_text(encoding="utf-8")
        assert "def find_location_root" not in text
        assert "location_root = find_location_root()" not in text
        assert "def exists_table(named_paths)" not in text
