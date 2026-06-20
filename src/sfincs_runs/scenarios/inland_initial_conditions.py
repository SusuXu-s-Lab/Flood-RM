from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def inland_initial_condition_config(config: dict) -> dict:
    """Return inland SFINCS initial-condition policy from location config."""
    coupling = config.get("inland_coupling", {}) or {}
    cfg = dict(coupling.get("initial_conditions", {}) or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("method", "hydrograph_mean_channel_inifile")
    cfg.setdefault("source", "wflow_dynamic_hydrographs")
    cfg.setdefault("statistic", "initial_window_source_mean")
    cfg.setdefault("mean_window_hours", 1.0)
    cfg.setdefault("reference_discharge_m3s", 5000.0)
    cfg.setdefault("reference_depth_m", 1.5)
    cfg.setdefault("min_depth_m", 0.1)
    cfg.setdefault("max_depth_m", 2.5)
    cfg.setdefault("river_buffer_cells", 2.0)
    cfg.setdefault("fill_value", -9999.0)
    return cfg


def derive_hydrograph_initial_depth(discharge, config: dict) -> dict:
    """Convert Wflow-SFINCS discharge hydrographs to an initial channel depth proxy.

    SFINCS initial water conditions are water levels in metres, not discharge.
    This helper intentionally records the discharge statistic and conversion so
    a run manifest can audit the hydraulic proxy used to fill the river corridor.
    """
    cfg = inland_initial_condition_config(config)
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "status": "disabled"}

    df = _as_discharge_frame(discharge)
    if df.empty:
        raise ValueError("Cannot derive SFINCS initial condition from empty discharge hydrographs")
    window = _initial_window(df, hours=float(cfg["mean_window_hours"]))
    source_means = window.mean(axis=0, skipna=True)
    mean_q = float(source_means.mean(skipna=True))
    if not np.isfinite(mean_q):
        raise ValueError("Cannot derive SFINCS initial condition from all-NaN discharge hydrographs")
    mean_q = max(mean_q, 0.0)

    reference_q = max(float(cfg["reference_discharge_m3s"]), 1.0e-6)
    reference_depth = max(float(cfg["reference_depth_m"]), 0.0)
    raw_depth = np.sqrt(mean_q / reference_q) * reference_depth
    depth = float(np.clip(raw_depth, float(cfg["min_depth_m"]), float(cfg["max_depth_m"])))

    return {
        "enabled": True,
        "status": "derived",
        "method": str(cfg["method"]),
        "source": str(cfg["source"]),
        "statistic": str(cfg["statistic"]),
        "mean_window_hours": float(cfg["mean_window_hours"]),
        "source_count": int(len(source_means)),
        "mean_initial_discharge_m3s": mean_q,
        "source_initial_discharge_m3s": {str(k): float(v) for k, v in source_means.items()},
        "depth_method": "sqrt(mean_initial_discharge / reference_discharge) * reference_depth",
        "reference_discharge_m3s": reference_q,
        "reference_depth_m": reference_depth,
        "raw_depth_m": float(raw_depth),
        "initial_depth_m": depth,
        "min_depth_m": float(cfg["min_depth_m"]),
        "max_depth_m": float(cfg["max_depth_m"]),
    }


def configure_hydrograph_initial_conditions(sf, discharge, config: dict, *, run_dir=None) -> dict:
    """Stage native HydroMT-SFINCS initial conditions from Wflow handoff hydrographs.

    The preferred inland path writes a spatial ``sfincs.ini`` through
    ``SfincsModel.initial_conditions.create``. Cells outside the river corridor
    use the SFINCS fill value so the kernel initializes them at bed level.
    """
    report = derive_hydrograph_initial_depth(discharge, config)
    if not report.get("enabled", True):
        return report

    cfg = inland_initial_condition_config(config)
    method = str(cfg.get("method", "")).lower()
    depth = float(report["initial_depth_m"])
    if method in {"zsini", "uniform_zsini"}:
        sf.config.set("zsini", f"{depth:.6f}")
        report.update({"status": "configured", "sfincs_initial_condition": "zsini"})
        return report
    if method != "hydrograph_mean_channel_inifile":
        raise ValueError(f"Unsupported inland SFINCS initial condition method: {cfg.get('method')!r}")

    if "dep" not in sf.grid.data or "mask" not in sf.grid.data:
        sf.config.set("zsini", f"{depth:.6f}")
        report.update(
            {
                "status": "configured",
                "sfincs_initial_condition": "zsini",
                "fallback_reason": "SFINCS grid lacks dep or mask map for spatial inifile",
            }
        )
        return report

    dep = sf.grid.data["dep"]
    mask = sf.grid.data["mask"]
    fill_value = float(cfg["fill_value"])
    river_mask = _river_corridor_mask(sf, cfg, run_dir=run_dir)
    if river_mask is None:
        sf.config.set("zsini", f"{depth:.6f}")
        report.update(
            {
                "status": "configured",
                "sfincs_initial_condition": "zsini",
                "fallback_reason": "No river corridor geometry available for spatial inifile",
            }
        )
        return report

    active_river = river_mask & (mask >= 1)
    if not bool(active_river.any()):
        sf.config.set("zsini", f"{depth:.6f}")
        report.update(
            {
                "status": "configured",
                "sfincs_initial_condition": "zsini",
                "fallback_reason": "River corridor geometry did not intersect active SFINCS cells",
            }
        )
        return report

    ini = xr.where(active_river, dep + depth, fill_value).astype("float32")
    ini.name = "ini"
    ini.attrs.update(
        {
            "standard_name": "initial water level",
            "unit": "m+ref",
            "source": "Wflow-SFINCS discharge hydrograph depth proxy",
        }
    )
    try:
        ini.raster.set_crs(sf.crs)
    except Exception:
        pass
    sf.initial_conditions.create(ini=ini, fill_value=fill_value)
    report.update(
        {
            "status": "configured",
            "sfincs_initial_condition": "inifile",
            "inifile": "sfincs.ini",
            "fill_value": fill_value,
            "river_initial_cell_count": int(active_river.sum()),
            "river_buffer_cells": float(cfg["river_buffer_cells"]),
        }
    )
    return report


def _as_discharge_frame(discharge) -> pd.DataFrame:
    if isinstance(discharge, pd.Series):
        return discharge.to_frame()
    if isinstance(discharge, pd.DataFrame):
        out = discharge.copy()
    elif isinstance(discharge, xr.DataArray):
        da = discharge
        if "time" not in da.dims:
            raise ValueError("Discharge DataArray must have a time dimension")
        other_dims = [dim for dim in da.dims if dim != "time"]
        if len(other_dims) > 1:
            raise ValueError(f"Discharge DataArray has too many non-time dimensions: {other_dims}")
        out = da.transpose("time", *other_dims).to_pandas()
        if isinstance(out, pd.Series):
            out = out.to_frame()
    else:
        out = pd.DataFrame(discharge)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.dropna(axis=1, how="all")


def _initial_window(df: pd.DataFrame, *, hours: float) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.head(1)
    if df.empty:
        return df
    start = df.index.min()
    stop = start + pd.to_timedelta(max(hours, 0.0), unit="h")
    out = df.loc[(df.index >= start) & (df.index <= stop)]
    return out if not out.empty else df.head(1)


def _river_corridor_mask(sf, cfg: dict, *, run_dir=None):
    geoms = _river_corridor_geoms(sf, run_dir=run_dir)
    if geoms is None or geoms.empty:
        return None
    try:
        geoms = geoms.to_crs(sf.crs)
    except Exception:
        pass
    cell_size = _grid_cell_size(sf)
    buffer_m = float(cfg.get("river_buffer_m", 0.0) or 0.0)
    if buffer_m <= 0.0:
        buffer_m = max(float(cfg.get("river_buffer_cells", 2.0)), 0.0) * cell_size
    geoms = geoms.copy()
    geoms["geometry"] = geoms.geometry.buffer(buffer_m)
    return sf.grid.data["mask"].raster.geometry_mask(geoms)


def _river_corridor_geoms(sf, *, run_dir=None):
    root = Path(run_dir) if run_dir is not None else Path(sf.root.path)
    rivers = root / "gis" / "rivers_inflow.geojson"
    if rivers.exists():
        gdf = gpd.read_file(rivers)
        if not gdf.empty:
            return gdf
    try:
        src = sf.discharge_points.gdf
    except Exception:
        return None
    if src is None or src.empty:
        return None
    return src


def _grid_cell_size(sf) -> float:
    try:
        res = sf.grid.data.raster.res
        return float(max(abs(res[0]), abs(res[1])))
    except Exception:
        dx = sf.config.get("dx")
        dy = sf.config.get("dy")
        values = [abs(float(value)) for value in (dx, dy) if value not in (None, "")]
        return max(values) if values else 60.0
