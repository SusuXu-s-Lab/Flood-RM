from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .build import open_model
from .forcing import stage_gridded_precipitation, stage_initial_condition, stage_water_level
from .io import write_json
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


def coastal_hydrograph_from_catalog_row(
    row: pd.Series | dict[str, Any],
    components: pd.DataFrame,
    *,
    member_time_column: str = "coastal_water_level_member_time",
    scale_column: str = "coastal_water_level_scale_factor",
    window_hours: float = 72.0,
    msl_offset_m: float = 0.0,
) -> tuple[pd.Series, dict[str, Any]]:
    data = dict(row)
    peak_time = data.get(member_time_column)
    if peak_time in (None, "") or bool(pd.isna(peak_time)):
        raise ValueError(f"catalog row is missing {member_time_column!r}")
    scale = data.get(scale_column, 1.0)
    eta = build_coastal_hydrograph_from_analog(
        components,
        peak_time,
        scale,
        window_hours=window_hours,
        msl_offset_m=msl_offset_m,
        return_absolute_time=True,
    )
    return eta, {
        "coastal_analog_peak_time": str(pd.Timestamp(peak_time)),
        "coastal_water_level_scale_factor": float(scale),
        "msl_offset_m": float(msl_offset_m),
        "expected_bzs_peak_max_m": float(eta.max()),
    }


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
    t0, t1 = stage_water_level(sf, eta, merge=False)
    netamprfile = ""
    if include_precip and precip_nc is not None:
        netamprfile = stage_gridded_precipitation(sf, precip_nc)
    initial = stage_initial_condition(sf, ini=initial_ini, zsini_m=initial_zsini_m)
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
