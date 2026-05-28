from pathlib import Path

from sfincs_runs.build_base import build_region_setup
from sfincs_runs.config import load_runtime


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_region_setup_uses_aoi_bbox_and_static_source_paths():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    setup = build_region_setup(config, paths, buffer_degrees=0.01)

    assert setup.bbox_wgs84[0] < -70.77
    assert setup.bbox_wgs84[1] < 42.06
    assert setup.bbox_wgs84[2] > -70.64
    assert setup.bbox_wgs84[3] > 42.16
    assert suffix(setup.dem_raw) == "locations/marshfield/data/static/raw/topo/cudem_10_marshfield.tif"
    assert suffix(setup.landcover_raw) == "locations/marshfield/data/static/raw/landcover/worldcover_bbox.tif"
    assert suffix(setup.dem_output) == "locations/marshfield/data/static/processed/cudem_region_setup.tif"
    assert suffix(setup.landcover_output) == "locations/marshfield/data/static/processed/worldcover_region_setup.tif"
    assert suffix(setup.coastal_region_output) == "locations/marshfield/data/static/processed/coastal_region.geojson"
    assert suffix(setup.ssurgo_output) == "locations/marshfield/data/static/soils/ssurgo_mapunitpoly.gpkg"
    assert suffix(setup.ssurgo_attributes_output) == "locations/marshfield/data/static/soils/ssurgo_mapunit_attributes.csv"
    assert suffix(setup.ssurgo_hsg_output) == "locations/marshfield/data/static/soils/hsg_mfield.tif"
    assert suffix(setup.ssurgo_ksat_output) == "locations/marshfield/data/static/soils/ksat_mmhr_mfield.tif"
