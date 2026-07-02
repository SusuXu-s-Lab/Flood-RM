from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .build import open_model
from .forcing import stage_gridded_precipitation
from .io import config_set, model_root_path, write_json
from .schema import EventManifest


def build_coastal_hydrograph_from_analog(
    components: pd.DataFrame | pd.Series,
    peak_time,
    scale_factor: float,
    *,
    window_hours: float = 72.0,
    msl_offset_m: float = 0.0,
    return_absolute_time: bool = True,
) -> pd.Series:
    """Build a tide-preserving total water-level realization.

    Scientific contract:

    ``eta_e(t) = MSL(t) + tide(t) + K_e * NTR(t) + Δz_SLR``.

    Tide and mean sea level are not scaled; only the non-tidal residual is
    scaled.  The default returns an absolute-time series so it can be passed
    directly to ``sf.water_level.create``.
    """
    scale = float(scale_factor)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale_factor must be finite and > 0, got {scale_factor!r}")
    peak = pd.Timestamp(peak_time)
    half = pd.Timedelta(hours=float(window_hours))
    window = components.loc[peak - half : peak + half]
    if window.empty:
        raise ValueError(f"no coastal components within +/-{window_hours:g} h of {peak}")

    if isinstance(window, pd.Series):
        baseline = float(window.min())
        values = baseline + scale * (window.to_numpy(dtype=float) - baseline) + float(msl_offset_m)
    else:
        required = {"msl", "tide", "ntr"}
        if not required.issubset(window.columns):
            raise ValueError(f"components must have columns {sorted(required)}")
        values = window["msl"].to_numpy(dtype=float) + window["tide"].to_numpy(dtype=float) + scale * window["ntr"].to_numpy(dtype=float) + float(msl_offset_m)

    if return_absolute_time:
        index = pd.DatetimeIndex(window.index, name="time")
    else:
        rel_hours = np.round((pd.DatetimeIndex(window.index) - peak) / pd.Timedelta(hours=1)).astype(int)
        index = pd.Index(rel_hours, name="relative_hour")
    out = pd.Series(values, index=index, name="water_level_m")
    return out[~out.index.duplicated(keep="first")].sort_index()


def coastal_timeseries_from_catalog_row(
    row: pd.Series | dict[str, Any],
    components: pd.DataFrame,
    *,
    member_time_column: str = "coastal_water_level_member_time",
    scale_column: str = "coastal_water_level_scale_factor",
    window_hours: float = 72.0,
    msl_offset_m: float = 0.0,
    return_absolute_time: bool = False,
) -> dict[str, Any]:
    """Build the legacy coastal event-forcing dict from one catalog row."""
    data = dict(row)
    peak_time = data.get(member_time_column)
    if peak_time in (None, "") or bool(pd.isna(peak_time)):
        raise ValueError(f"catalog row is missing {member_time_column!r}")
    scale = data.get(scale_column, 1.0)
    series = build_coastal_hydrograph_from_analog(
        components,
        peak_time,
        scale,
        window_hours=window_hours,
        msl_offset_m=msl_offset_m,
        return_absolute_time=return_absolute_time,
    )
    return {
        "h": series,
        "forcing_variable": "coastal_water_level",
        "analog_peak_time": str(peak_time),
        "scale_factor": float(scale),
        "msl_offset_m": float(msl_offset_m),
    }


def _set_model_time(sf, start: pd.Timestamp, stop: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    values = {
        "tref": start.to_pydatetime(),
        "tstart": start.to_pydatetime(),
        "tstop": stop.to_pydatetime(),
    }
    if hasattr(sf.config, "update"):
        sf.config.update(values)
    else:
        for key, value in values.items():
            config_set(sf, key, value)
    return start, stop


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


def _stage_water_level(sf, eta: pd.Series | pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(eta, pd.Series):
        h = _absolute_series(eta)
        table = pd.DataFrame({column: h.to_numpy(dtype=float) for column in _water_level_columns(sf)}, index=h.index)
    else:
        table = eta.copy()
        table.index = pd.DatetimeIndex(pd.to_datetime(table.index), name="time")
        table = table.sort_index().astype(float)

    _set_model_time(sf, table.index.min(), table.index.max())
    sf.water_level.create(timeseries=table, merge=False)
    return table.index.min(), table.index.max()


def _stage_initial_condition(sf, *, ini: str | Path | xr.DataArray | None = None, zsini_m: float | None = None) -> dict[str, Any]:
    if ini is not None:
        component = getattr(sf, "initial_conditions", None) or getattr(sf, "quadtree_initial_conditions", None)
        if component is None:
            raise RuntimeError("HydroMT-SFINCS model has no initial_conditions component")
        component.create(ini=ini, fill_value=-9999.0)
        return {"kind": "inifile", "fill_value": -9999.0, "source": str(ini)}
    if zsini_m is not None:
        config_set(sf, "zsini", f"{float(zsini_m):.6f}")
        return {"kind": "zsini", "zsini_m": float(zsini_m)}
    return {"kind": "none"}


def stage_coastal_event_forcing(
    run_root: str | Path,
    *,
    event_id: str,
    eta: pd.Series | pd.DataFrame,
    precip_nc: str | Path | None = None,
    include_precip: bool = False,
    initial_ini: str | Path | xr.DataArray | None = None,
    initial_zsini_m: float | None = None,
    probability_weight: float | None = None,
    total_rate_per_year: float | None = None,
    annual_rate: float | None = None,
    sfincs_domain_id: str = "",
    metadata: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> EventManifest:
    """Stage one coastal SFINCS event through native water-level forcing."""
    sf = open_model(run_root, mode="r+", read=True)
    t0, t1 = _stage_water_level(sf, eta)
    netamprfile = ""
    if include_precip and precip_nc is not None:
        netamprfile = stage_gridded_precipitation(sf, precip_nc)
    initial = _stage_initial_condition(sf, ini=initial_ini, zsini_m=initial_zsini_m)
    sf.write()

    manifest = EventManifest(
        event_id=str(event_id),
        run_root=str(run_root),
        forcing_mode="coastal_water_level",
        run_start=t0.strftime("%Y-%m-%d %H:%M:%S"),
        run_stop=t1.strftime("%Y-%m-%d %H:%M:%S"),
        sfincs_domain_id=str(sfincs_domain_id),
        probability_weight=probability_weight,
        total_rate_per_year=total_rate_per_year,
        annual_rate=annual_rate,
        coastal_water_level=True,
        precipitation_nc="" if precip_nc is None else str(precip_nc),
        netamprfile=netamprfile,
        initial_condition=initial,
        metadata=metadata or {},
    )
    if write_manifest:
        write_json(Path(run_root) / "forcing_manifest.json", manifest.to_dict())
    return manifest
