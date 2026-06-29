from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .build import open_model
from .io import model_root_path, register_raster_or_dataset
from .schema import NativeSourceConfig


def _component_gdf(component: Any) -> gpd.GeoDataFrame:
    value = getattr(component, "gdf", None)
    if isinstance(value, gpd.GeoDataFrame):
        return value
    value = getattr(component, "data", None)
    if isinstance(value, gpd.GeoDataFrame):
        return value
    raise RuntimeError(f"{component!r} does not expose a GeoDataFrame")


def _register_hydrography_if_path(sf, name_or_path: str | Path) -> str | Path:
    path = Path(name_or_path) if not isinstance(name_or_path, Path) else name_or_path
    looks_like_path = path.is_absolute() or len(path.parts) > 1 or bool(path.suffix)
    if looks_like_path and path.exists():
        return register_raster_or_dataset(sf, path.stem, path)
    return name_or_path


def _stable_source_names(domain_id: str, count: int) -> list[str]:
    return [f"{domain_id}_inflow_{index:02d}" for index in range(1, count + 1)]


def create_wflow_source_contract(
    base_root: str | Path,
    *,
    sfincs_domain_id: str,
    output: str | Path,
    source_config: NativeSourceConfig | None = None,
    data_libs: list[str | Path] | str | Path | None = None,
    wflow_submodel_id: str = "",
    write_model: bool = True,
) -> gpd.GeoDataFrame:
    """Create SFINCS-native river inflow points and export them as Wflow gauges.

    Default coupling contract:

    ``Wflow discharge NetCDF name(index) == SfincsModel.discharge_points.gdf['name']``.
    """
    cfg = source_config or NativeSourceConfig()
    sf = open_model(base_root, mode="r+", read=True, write_gis=True, data_libs=data_libs)
    hydrography = _register_hydrography_if_path(sf, cfg.hydrography)

    sf.rivers.create_river_inflow(
        hydrography=hydrography,
        buffer=float(cfg.buffer_m),
        river_upa=float(cfg.river_upa_km2),
        river_len=float(cfg.river_len_m),
        river_width=float(cfg.river_width_m),
        keep_rivers_geom=bool(cfg.keep_rivers_geom),
        merge=False,
        first_index=int(cfg.first_index),
        reverse_river_geom=bool(cfg.reverse_river_geom),
        src_type=str(cfg.src_type),
    )

    src = _component_gdf(sf.discharge_points).copy()
    if src.empty:
        raise RuntimeError(
            "HydroMT-SFINCS created no river inflow source points. Review domain, hydrography, river_upa, river_len, and buffer."
        )
    model_crs = getattr(sf, "crs", None)
    if src.crs is None and model_crs is not None:
        src = src.set_crs(model_crs)
    elif model_crs is not None:
        src = src.to_crs(model_crs)

    src = src.reset_index(drop=True)
    if cfg.max_source_points is not None and len(src) > cfg.max_source_points:
        sort_cols = [c for c in ("uparea", "uparea_km2") if c in src]
        src = src.sort_values(sort_cols, ascending=False).head(int(cfg.max_source_points)).reset_index(drop=True) if sort_cols else src.head(int(cfg.max_source_points)).reset_index(drop=True)

    if "index" in src.columns:
        src = src.drop(columns=["index"])
    src.insert(0, "index", range(int(cfg.first_index), int(cfg.first_index) + len(src)))
    names = _stable_source_names(str(sfincs_domain_id), len(src))
    src["name"] = names
    src["sfincs_handoff_id"] = names
    src["site_no"] = names
    src["sfincs_domain_id"] = str(sfincs_domain_id)
    src["wflow_submodel_id"] = str(wflow_submodel_id or sfincs_domain_id)
    src["gauge_location_source"] = "hydromt_sfincs.rivers.create_river_inflow"
    src["river_upa_km2"] = float(cfg.river_upa_km2)
    src["river_len_m"] = float(cfg.river_len_m)
    src["river_width_m"] = float(cfg.river_width_m)
    src["river_inflow_buffer_m"] = float(cfg.buffer_m)

    keep = [
        "index",
        "name",
        "sfincs_handoff_id",
        "site_no",
        "sfincs_domain_id",
        "wflow_submodel_id",
        "gauge_location_source",
        "uparea",
        "uparea_km2",
        "river_upa_km2",
        "river_len_m",
        "river_width_m",
        "river_inflow_buffer_m",
        "geometry",
    ]
    keep = [col for col in keep if col in src.columns]
    src = gpd.GeoDataFrame(src[keep], geometry="geometry", crs=src.crs)

    sf.discharge_points.set_locations(src, merge=False)
    if write_model:
        sf.write()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    src.to_file(out, driver="GeoJSON")
    return src


def read_source_contract(path: str | Path, *, crs=None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None and crs is not None:
        gdf = gdf.set_crs(crs)
    if "name" not in gdf:
        if "sfincs_handoff_id" in gdf:
            gdf["name"] = gdf["sfincs_handoff_id"].astype(str)
        else:
            raise ValueError(f"source contract lacks a 'name' column: {path}")
    return gdf


def validate_source_contract(source_contract: gpd.GeoDataFrame, discharge_names: list[str] | pd.Index | tuple[str, ...]) -> pd.Series:
    """Validate that Wflow discharge names satisfy the SFINCS source contract."""
    src_names = source_contract["name"].astype(str).tolist()
    q_names = [str(value) for value in discharge_names]
    missing_in_discharge = sorted(set(src_names) - set(q_names))
    extra_discharge = sorted(set(q_names) - set(src_names))
    return pd.Series(
        {
            "source_count": len(src_names),
            "discharge_name_count": len(q_names),
            "missing_in_discharge": ",".join(missing_in_discharge),
            "extra_discharge_names": ",".join(extra_discharge),
            "compatible": not missing_in_discharge,
        },
        name="sfincs_wflow_source_contract",
    )


def write_contract_from_existing_model(base_root: str | Path, output: str | Path) -> gpd.GeoDataFrame:
    """Export existing native SFINCS ``src`` points as a Wflow source contract."""
    sf = open_model(base_root, mode="r", read=True)
    src = _component_gdf(sf.discharge_points).copy()
    if src.empty:
        raise RuntimeError(f"No SFINCS discharge source points in {model_root_path(sf)}")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    src.to_file(out, driver="GeoJSON")
    return src
