from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .build import open_model
from .io import config_set, model_root_path, register_raster_or_dataset, set_model_time, write_json
from .schema import EventManifest


def read_wflow_discharge(
    path: str | Path,
    *,
    variable: str = "discharge",
    index_dim: str = "index",
    name_var: str = "name",
) -> pd.DataFrame:
    """Read Wflow-produced discharge for SFINCS source points.

    Expected minimum schema:

    ``discharge(time, index)`` and ``name(index)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {path}")
        data = ds[variable]
        if "time" not in data.dims or index_dim not in data.dims:
            raise ValueError(f"{variable!r} must have dimensions ('time', {index_dim!r}); got {data.dims}")
        frame = data.transpose("time", index_dim).to_pandas()
        if name_var in ds:
            names = ds[name_var].values.astype(str)
        elif name_var in ds.coords:
            names = ds.coords[name_var].values.astype(str)
        else:
            raise KeyError(f"{name_var!r} coordinate/variable not found in {path}")
    frame.columns = names
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="time")
    return frame.sort_index()


def _source_points(sf) -> gpd.GeoDataFrame:
    gdf = getattr(sf.discharge_points, "gdf", None)
    if gdf is None:
        gdf = getattr(sf.discharge_points, "data", None)
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        raise RuntimeError(
            f"{model_root_path(sf)} has no native SFINCS src points. Build the base with rivers.create_river_inflow first."
        )
    if "name" not in gdf:
        raise RuntimeError("SFINCS source points do not have a 'name' column")
    return gdf.copy()


def stage_wflow_discharge(sf, discharge: pd.DataFrame | str | Path) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    """Stage upstream Wflow hydrographs through native ``discharge_points.create``."""
    q = read_wflow_discharge(discharge) if isinstance(discharge, (str, Path)) else discharge.copy()
    if q.empty:
        raise ValueError("Wflow discharge table is empty")
    q.index = pd.DatetimeIndex(pd.to_datetime(q.index), name="time")
    t0, t1 = set_model_time(sf, q.index.min(), q.index.max())

    src = _source_points(sf)
    src_names = src["name"].astype(str).tolist()
    missing = sorted(set(src_names) - set(map(str, q.columns)))
    if missing:
        raise RuntimeError(f"Wflow discharge is missing SFINCS source names: {missing}")

    q_src = q.loc[:, src_names].copy()
    q_src.columns = src["index"].astype(int).tolist() if "index" in src else src.index.astype(int).tolist()
    sf.discharge_points.create(timeseries=q_src, merge=False)
    return t0, t1, src_names


def stage_gridded_precipitation(
    sf,
    precip_nc: str | Path,
    *,
    source_name: str = "event_precip",
    cumulative_input: bool = True,
    time_label: str = "right",
    aggregate: bool | str = False,
    buffer_m: float = 30000.0,
    dst_res: float | None = None,
) -> str:
    """Stage rain-on-grid through native ``precipitation.create``."""
    precip_nc = Path(precip_nc)
    if not precip_nc.exists():
        raise FileNotFoundError(precip_nc)
    source = register_raster_or_dataset(sf, source_name, precip_nc, crs="EPSG:4326")

    # Clear stale copied-base pointers before the native component writes them.
    config_set(sf, "precipfile", None)
    config_set(sf, "netamprfile", None)

    kwargs: dict[str, Any] = {"buffer": float(buffer_m)}
    if dst_res is not None:
        kwargs["dst_res"] = float(dst_res)
    sf.precipitation.create(
        precip=source,
        cumulative_input=bool(cumulative_input),
        time_label=str(time_label),
        aggregate=aggregate,
        **kwargs,
    )
    return "sfincs_netampr.nc" if aggregate is False else "sfincs.precip"


def _water_level_locations(sf):
    component = getattr(sf, "water_level", None)
    gdf = getattr(component, "gdf", None)
    if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
        return gdf
    data = getattr(component, "data", None)
    if isinstance(data, gpd.GeoDataFrame) and not data.empty:
        return data
    return None


def _water_level_columns(sf) -> list[int]:
    gdf = _water_level_locations(sf)
    if gdf is not None and not gdf.empty:
        if "index" in gdf:
            return gdf["index"].astype(int).tolist()
        return list(range(1, len(gdf) + 1))
    bnd_path = model_root_path(sf) / "sfincs.bnd"
    if not bnd_path.exists():
        raise FileNotFoundError("No water-level boundary locations found; build bnd locations first or pass locations=...")
    count = sum(1 for line in bnd_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
    if count == 0:
        raise RuntimeError(f"Empty water-level boundary file: {bnd_path}")
    return list(range(1, count + 1))


def _absolute_series(series: pd.Series, *, run_start=None, dt_unit: str = "h") -> pd.Series:
    out = series.dropna().astype(float).copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index), name="time")
        return out.sort_index()
    if run_start is None:
        raise ValueError("run_start is required when the water-level series has relative-time index")
    out.index = pd.Timestamp(run_start) + pd.to_timedelta(pd.to_numeric(out.index), unit=dt_unit)
    out.index.name = "time"
    return out.sort_index()


def stage_water_level(
    sf,
    eta: pd.Series | pd.DataFrame,
    *,
    locations: gpd.GeoDataFrame | str | Path | None = None,
    run_start=None,
    relative_time_unit: str = "h",
    offset=None,
    merge: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Stage coastal/open-boundary water level through native ``water_level.create``."""
    if isinstance(eta, pd.Series):
        h = _absolute_series(eta, run_start=run_start, dt_unit=relative_time_unit)
        if locations is None:
            columns = _water_level_columns(sf)
        else:
            loc_gdf = gpd.read_file(locations) if isinstance(locations, (str, Path)) else locations
            columns = loc_gdf["index"].astype(int).tolist() if "index" in loc_gdf else list(range(1, len(loc_gdf) + 1))
        table = pd.DataFrame({column: h.to_numpy(dtype=float) for column in columns}, index=h.index)
    else:
        table = eta.copy()
        if not isinstance(table.index, pd.DatetimeIndex):
            if run_start is None:
                raise ValueError("run_start is required when the water-level table has a relative-time index")
            table.index = pd.Timestamp(run_start) + pd.to_timedelta(pd.to_numeric(table.index), unit=relative_time_unit)
        table.index = pd.DatetimeIndex(pd.to_datetime(table.index), name="time")
        table = table.sort_index().astype(float)

    set_model_time(sf, table.index.min(), table.index.max())
    kwargs: dict[str, Any] = {"timeseries": table, "merge": bool(merge)}
    if locations is not None:
        kwargs["locations"] = locations
    if offset is not None:
        kwargs["offset"] = offset
    sf.water_level.create(**kwargs)
    return table.index.min(), table.index.max()


def _initial_component(sf):
    if str(getattr(sf, "grid_type", "") or "").lower() == "quadtree":
        return getattr(sf, "quadtree_initial_conditions", None) or getattr(sf, "initial_conditions", None)
    return getattr(sf, "initial_conditions", None) or getattr(sf, "quadtree_initial_conditions", None)


def stage_initial_condition(
    sf,
    *,
    ini: str | Path | xr.DataArray | None = None,
    zsini_m: float | None = None,
    fill_value: float = -9999.0,
) -> dict[str, Any]:
    """Stage spatial initial water level or uniform ``zsini``.

    A spatial ``ini`` uses the native ``initial_conditions.create`` component.  A
    scalar ``zsini`` uses the native config component because SFINCS stores it in
    ``sfincs.inp`` rather than a separate raster.
    """
    if ini is not None:
        component = _initial_component(sf)
        if component is None:
            raise RuntimeError("HydroMT-SFINCS model has no initial_conditions component")
        component.create(ini=ini, fill_value=float(fill_value))
        return {"kind": "inifile", "fill_value": float(fill_value), "source": str(ini)}
    if zsini_m is not None:
        config_set(sf, "zsini", f"{float(zsini_m):.6f}")
        return {"kind": "zsini", "zsini_m": float(zsini_m)}
    return {"kind": "none"}


def stage_inland_event_forcing(
    run_root: str | Path,
    *,
    event_id: str,
    wflow_discharge_nc: str | Path,
    precip_nc: str | Path | None = None,
    direct_rainfall: bool = False,
    initial_ini: str | Path | xr.DataArray | None = None,
    initial_zsini_m: float | None = None,
    probability_weight: float | None = None,
    total_rate_per_year: float | None = None,
    annual_rate: float | None = None,
    sfincs_domain_id: str = "",
    write_manifest: bool = True,
) -> EventManifest:
    """Stage one Wflow-discharge-driven inland SFINCS event folder."""
    sf = open_model(run_root, mode="r+", read=True)
    t0, t1, src_names = stage_wflow_discharge(sf, wflow_discharge_nc)

    netamprfile = ""
    if direct_rainfall and precip_nc is not None:
        netamprfile = stage_gridded_precipitation(sf, precip_nc)

    initial = stage_initial_condition(sf, ini=initial_ini, zsini_m=initial_zsini_m)
    sf.write()

    manifest = EventManifest(
        event_id=str(event_id),
        run_root=str(run_root),
        forcing_mode="inland_wflow_discharge",
        run_start=t0.strftime("%Y-%m-%d %H:%M:%S"),
        run_stop=t1.strftime("%Y-%m-%d %H:%M:%S"),
        sfincs_domain_id=str(sfincs_domain_id),
        probability_weight=probability_weight,
        total_rate_per_year=total_rate_per_year,
        annual_rate=annual_rate,
        wflow_discharge_nc=str(wflow_discharge_nc),
        sfincs_src_names=tuple(src_names),
        precipitation_nc="" if precip_nc is None else str(precip_nc),
        netamprfile=netamprfile,
        initial_condition=initial,
    )
    if write_manifest:
        write_json(Path(run_root) / "forcing_manifest.json", manifest.to_dict())
    return manifest
