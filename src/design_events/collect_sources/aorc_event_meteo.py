from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_AORC_METEO_VARIABLES = {
    "temp": ("TMP_2maboveground", "TMP_2m", "TMP_surface", "temp"),
    "press_msl": ("PRES_surface", "PRMSL_meansealevel", "press_msl"),
    "kin": ("DSWRF_surface", "SWDOWN", "kin"),
}


def aorc_wflow_temp_pet_variables(config: dict | None = None) -> dict[str, tuple[str, ...]]:
    """Return configured AORC source-variable candidates for Wflow temp/PET forcing."""
    cfg = (
        ((config or {}).get("wflow", {}) or {})
        .get("event_forcing", {})
        .get("temp_pet", {})
        or {}
    )
    aorc_cfg = (
        ((config or {}).get("collection", {}) or {})
        .get("aorc_sst", {})
        .get("event_meteo", {})
        or {}
    )
    source_variables = {
        **(cfg.get("source_variables") or {}),
        **(aorc_cfg.get("source_variables") or {}),
    }
    variables: dict[str, tuple[str, ...]] = {}
    for target, defaults in DEFAULT_AORC_METEO_VARIABLES.items():
        configured = source_variables.get(target)
        if configured in (None, ""):
            variables[target] = defaults
        elif isinstance(configured, str):
            variables[target] = (configured, *tuple(name for name in defaults if name != configured))
        else:
            ordered = tuple(str(name) for name in configured if str(name).strip())
            variables[target] = (*ordered, *tuple(name for name in defaults if name not in ordered))
    return variables


def prepare_aorc_temp_pet_for_wflow(
    source_nc: str | Path,
    output_nc: str | Path,
    *,
    t_start: str | pd.Timestamp,
    t_stop: str | pd.Timestamp,
    precip_template: str | Path | None = None,
    variable_candidates: dict[str, tuple[str, ...]] | None = None,
    source_time_start: str | pd.Timestamp | None = None,
    freq: str = "1h",
    provenance_path: str | Path | None = None,
) -> dict:
    """Write a HydroMT-Wflow ``event_temp_pet`` dataset from AORC event fields.

    The output follows the native ``WflowSbmModel.setup_temp_pet_forcing`` contract for
    ``pet_method: makkink``: ``temp`` [degC], ``press_msl`` [hPa], and
    ``kin`` [W/m2] on ``time/y/x``. NOAA AORC v1.1 does not provide outgoing
    shortwave radiation, so De Bruin PET is not used unless a reviewed ``kout``
    source is added separately.
    """
    source_nc = Path(source_nc)
    output_nc = Path(output_nc)
    if not source_nc.exists():
        raise FileNotFoundError(source_nc)
    variable_candidates = variable_candidates or DEFAULT_AORC_METEO_VARIABLES

    full_time = pd.date_range(pd.Timestamp(t_start), pd.Timestamp(t_stop), freq=freq)
    with xr.open_dataset(source_nc) as source:
        selected = {
            target: _select_variable(source, candidates, source_nc=source_nc)
            for target, candidates in variable_candidates.items()
        }
        prepared = {
            "temp": _prepare_field(selected["temp"], full_time, target="temp", source_time_start=source_time_start, freq=freq),
            "press_msl": _prepare_field(selected["press_msl"], full_time, target="press_msl", source_time_start=source_time_start, freq=freq),
            "kin": _prepare_field(selected["kin"], full_time, target="radiation", source_time_start=source_time_start, freq=freq),
        }
        if "kout" in selected:
            prepared["kout"] = _prepare_field(selected["kout"], full_time, target="radiation", source_time_start=source_time_start, freq=freq)

    if precip_template is not None and Path(precip_template).exists():
        with xr.open_dataset(precip_template) as template:
            if "precip" in template:
                prepared = {
                    name: _align_to_template(da, template["precip"])
                    for name, da in prepared.items()
                }

    ds = xr.Dataset(prepared).astype("float32")
    ds.attrs.update(
        {
            "crs": "EPSG:4326",
            "source": str(source_nc),
            "hydromt_wflow_contract": "setup_temp_pet_forcing(pet_method=makkink)",
        }
    )
    ds["temp"].attrs.update(units="degree C", source_units="K_or_degree_C")
    ds["press_msl"].attrs.update(units="hPa", source_units="Pa_or_hPa")
    ds["kin"].attrs.update(units="W m-2")
    if "kout" in ds:
        ds["kout"].attrs.update(units="W m-2")

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_nc)

    provenance = {
        "source_nc": str(source_nc),
        "output_nc": str(output_nc),
        "time_start": pd.Timestamp(t_start).isoformat(),
        "time_stop": pd.Timestamp(t_stop).isoformat(),
        "freq": freq,
        "variable_mapping": {
            target: _selected_name(source_nc, candidates)
            for target, candidates in variable_candidates.items()
        },
        "unit_conversions": {
            "temp": "Kelvin to degree_C when values exceed 150",
            "press_msl": "Pa to hPa when values exceed 2000",
            "kin": "W m-2 unchanged",
            **({"kout": "W m-2 unchanged"} if "kout" in ds else {}),
        },
        "temporal_fill": "linear interpolation to model clock; endpoint nearest-fill where needed",
        "source_time_start": "" if source_time_start is None else pd.Timestamp(source_time_start).isoformat(),
        "spatial_template": str(precip_template) if precip_template else "",
    }
    if provenance_path is not None:
        provenance_path = Path(provenance_path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _select_variable(ds: xr.Dataset, candidates: tuple[str, ...], *, source_nc: Path) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(
        f"{source_nc} lacks required AORC meteo variable; tried {list(candidates)}. "
        "Rerun source collection with AORC event_meteo variables enabled."
    )


def _selected_name(source_nc: Path, candidates: tuple[str, ...]) -> str:
    with xr.open_dataset(source_nc) as ds:
        for name in candidates:
            if name in ds:
                return name
    return ""


def _prepare_field(
    da: xr.DataArray,
    full_time: pd.DatetimeIndex,
    *,
    target: str,
    source_time_start: str | pd.Timestamp | None,
    freq: str,
) -> xr.DataArray:
    da = _normalize_dims(da).sortby("y").sortby("x")
    if "time" not in da.dims:
        raise ValueError(f"{da.name!r} has no time dimension")
    if source_time_start is not None:
        da = da.assign_coords(
            time=pd.date_range(pd.Timestamp(source_time_start), periods=da.sizes["time"], freq=freq)
        )
    da = da.sortby("time")
    start = full_time[0]
    stop = full_time[-1]
    da = da.sel(time=slice(start, stop))
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name!r} has no data in requested event window {start}..{stop}")
    da = da.interp(time=full_time)
    da = _fill_time_endpoints(da, fill_value=0.0 if target == "radiation" else None)
    if target == "temp":
        da = xr.where(da > 150.0, da - 273.15, da)
        return da.rename("temp")
    if target == "press_msl":
        da = xr.where(da > 2000.0, da / 100.0, da)
        return da.rename("press_msl")
    return da.clip(min=0.0).rename(None)


def _normalize_dims(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "y"
    if "longitude" in da.dims:
        rename["longitude"] = "x"
    if "lat" in da.dims:
        rename["lat"] = "y"
    if "lon" in da.dims:
        rename["lon"] = "x"
    return da.rename(rename)


def _fill_time_endpoints(da: xr.DataArray, *, fill_value: float | None) -> xr.DataArray:
    if not bool(da.isnull().any()):
        return da
    if fill_value is not None:
        return da.fillna(fill_value)
    values = da.values
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"{da.name!r} has no finite values after temporal interpolation")
    fill = np.nanmedian(values)
    return da.fillna(float(fill))


def _align_to_template(da: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    if {"y", "x"} - set(template.dims):
        return da
    if (
        "y" in da.coords
        and "x" in da.coords
        and da.sizes.get("y") == template.sizes.get("y")
        and da.sizes.get("x") == template.sizes.get("x")
        and np.allclose(da["y"].values, template["y"].values)
        and np.allclose(da["x"].values, template["x"].values)
    ):
        return da
    return da.interp(
        y=template["y"],
        x=template["x"],
        kwargs={"fill_value": "extrapolate"},
    )
