"""Build the Marshfield Delft-FIAT model once (orchestrated from the main env).

Runs in the main project env; the actual hydromt_fiat build happens in the isolated
conda ``fiat`` env via :func:`fiat_runs._env.run_in_fiat_env`. The built model
(settings.toml, exposure geoms/csv, vulnerability curves) is plain files that the
delft_fiat engine later reads per event, so it is environment-independent once written.

Exposure = NSI, vulnerability = HAZUS, ground elevation = SFINCS subgrid DEM. See the
declarative recipe in ``locations/<loc>/fiat_config.yml``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._env import run_in_fiat_env

_BUILD_ENTRY = Path(__file__).with_name("_build_in_fiat_env.py")


def _wave_base_root(config: dict, paths: dict) -> Path:
    rel = (
        config.get("coastal_wave_coupling", {})
        .get("quadtree", {})
        .get("base_model_root", "data/sfincs/base_quadtree_snapwave")
    )
    return Path(paths["location_root"]) / rel


def fiat_model_inputs(config: dict, paths: dict) -> dict:
    """Resolve the SFINCS-derived region + ground-elevation DEM that FIAT couples to."""
    wave_base = _wave_base_root(config, paths)
    return {
        "region": wave_base / "gis" / "region.geojson",
        "ground_elevation": wave_base / "subgrid" / "dep_subgrid_lev0.tif",
        "config_yml": Path(paths["location_root"]) / "fiat_config.yml",
    }


def model_ready(paths: dict) -> bool:
    return (Path(paths["fiat_model_root"]) / "settings.toml").exists()


def build_model(config: dict, paths: dict, *, force: bool = False) -> dict:
    """Build (or reuse) the FIAT model. Returns the build receipt dict.

    Raises FileNotFoundError if the SFINCS region/DEM inputs are missing.
    """
    model_root = Path(paths["fiat_model_root"])
    inputs = fiat_model_inputs(config, paths)
    for key in ("region", "ground_elevation", "config_yml"):
        if not Path(inputs[key]).exists():
            raise FileNotFoundError(f"FIAT build input '{key}' missing: {inputs[key]}")

    receipt_path = model_root / "fiat_build_receipt.json"
    if model_ready(paths) and not force:
        if receipt_path.exists():
            return json.loads(receipt_path.read_text(encoding="utf-8"))
        return {"model_root": str(model_root), "reused": True}

    model_root.mkdir(parents=True, exist_ok=True)

    # The NSI API requires the query polygon in EPSG:4326 (lon/lat); the SFINCS region
    # is in the model UTM CRS. Reproject once and hand the lon/lat region to the builder.
    import geopandas as gpd

    region_4326 = model_root / "_region_4326.geojson"
    gpd.read_file(inputs["region"]).to_crs(4326).to_file(region_4326, driver="GeoJSON")

    params = {
        "region": str(region_4326),
        "model_root": str(model_root),
        "config_yml": str(inputs["config_yml"]),
    }
    params_path = model_root / "_build_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    run_in_fiat_env(["python", str(_BUILD_ENTRY), str(params_path)])

    if not receipt_path.exists():
        raise RuntimeError(
            f"FIAT build finished but no receipt at {receipt_path}; check the conda 'fiat' env build log."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # Always couple ground elevation to the SFINCS DEM right after a build so the model
    # on disk is never left with NSI's (mismatched-datum) ground elevation.
    receipt["ground_coupling"] = apply_ground(config, paths)
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    return receipt


def apply_ground(config: dict, paths: dict) -> dict:
    """Overwrite exposure ground elevation with the SFINCS subgrid DEM (offline coupling).

    FIAT computes inundation as ``water_level - ground_elevation - ground_floor_height``.
    For the depths to be consistent the ground elevation must be in the SFINCS vertical
    datum, not NSI's NAVD88. We sample the high-resolution SFINCS DEM at each structure and
    write it (in feet, the model unit) into ``exposure.csv['ground_elevtn']``. This resolves
    the per-structure ground at 10 m even though the water level is on the 60 m SFINCS grid.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio

    model_root = Path(paths["fiat_model_root"])
    dep_path = fiat_model_inputs(config, paths)["ground_elevation"]
    exposure_csv = model_root / "exposure" / "exposure.csv"
    buildings = model_root / "exposure" / "buildings.gpkg"

    gdf = gpd.read_file(buildings)
    with rasterio.open(dep_path) as src:
        pts = [(geom.x, geom.y) for geom in gdf.to_crs(src.crs).geometry.representative_point()]
        ground_m = np.array([v[0] for v in src.sample(pts)], dtype=float)
    ground_ft = ground_m * 3.28084

    ex = pd.read_csv(exposure_csv)
    ground_by_id = dict(zip(gdf["object_id"].astype(str), ground_ft))
    ex["ground_elevtn"] = ex["object_id"].astype(str).map(ground_by_id)
    n_filled = int(ex["ground_elevtn"].notna().sum())
    ex.to_csv(exposure_csv, index=False)
    return {
        "exposure_csv": str(exposure_csv),
        "dem": str(dep_path),
        "structures_grounded": n_filled,
        "ground_ft_median": float(np.nanmedian(ground_ft)),
    }
