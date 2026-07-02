from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


AORC_METEO = {
    "temp": ("TMP_2maboveground", "TMP_2m", "TMP_surface", "temp"),
    "press_msl": ("PRES_surface", "PRMSL_meansealevel", "press_msl"),
    "kin": ("DSWRF_surface", "SWDOWN", "kin"),
}


def aorc_variable_candidates(config: dict | None = None) -> dict[str, tuple[str, ...]]:
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
    configured = {
        **(cfg.get("source_variables") or {}),
        **(aorc_cfg.get("source_variables") or {}),
    }
    out: dict[str, tuple[str, ...]] = {}
    for target, defaults in AORC_METEO.items():
        value = configured.get(target)
        if value in (None, ""):
            out[target] = defaults
        elif isinstance(value, str):
            out[target] = (value, *tuple(name for name in defaults if name != value))
        else:
            ordered = tuple(str(name) for name in value if str(name).strip())
            out[target] = (*ordered, *tuple(name for name in defaults if name not in ordered))
    return out


def aorc_source_variables(ds: xr.Dataset, spec: dict, config: dict | None = None) -> list[str]:
    variables = [spec.get("variable", "APCP_surface")]
    if (spec.get("event_meteo") or {}).get("enabled", False):
        for candidates in aorc_variable_candidates(config).values():
            match = next((name for name in candidates if name in ds), None)
            if match:
                variables.append(match)
    return list(dict.fromkeys(variables))


def write_wflow_temp_pet(
    source_nc,
    output_nc,
    *,
    start,
    end,
    freq="1h",
    precip_template=None,
    config=None,
    variable_candidates: dict[str, tuple[str, ...]] | None = None,
    source_time_start=None,
    provenance_path=None,
) -> dict:
    """AORC event meteo -> HydroMT-Wflow temp/PET forcing fields."""
    source_nc = Path(source_nc)
    output_nc = Path(output_nc)
    if not source_nc.exists():
        raise FileNotFoundError(source_nc)
    time = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
    candidates = variable_candidates or aorc_variable_candidates(config)
    with xr.open_dataset(source_nc) as ds:
        selected = {k: _pick(ds, names, source_nc=source_nc) for k, names in candidates.items()}
        prepared = {
            "temp": _field(selected["temp"], time, "temp", source_time_start=source_time_start, freq=freq),
            "press_msl": _field(selected["press_msl"], time, "press_msl", source_time_start=source_time_start, freq=freq),
            "kin": _field(selected["kin"], time, "radiation", source_time_start=source_time_start, freq=freq),
        }
        if "kout" in selected:
            prepared["kout"] = _field(selected["kout"], time, "radiation", source_time_start=source_time_start, freq=freq)
    if precip_template is not None and Path(precip_template).exists():
        with xr.open_dataset(precip_template) as template:
            if "precip" in template:
                prepared = {name: _align_to_template(da, template["precip"]) for name, da in prepared.items()}

    out = xr.Dataset(prepared).astype("float32")
    out.attrs.update(
        {
            "crs": "EPSG:4326",
            "source": str(source_nc),
            "hydromt_wflow_contract": "setup_temp_pet_forcing(pet_method=makkink)",
        }
    )
    out["temp"].attrs.update(units="degree C", source_units="K_or_degree_C")
    out["press_msl"].attrs.update(units="hPa", source_units="Pa_or_hPa")
    out["kin"].attrs.update(units="W m-2")
    if "kout" in out:
        out["kout"].attrs.update(units="W m-2")

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_nc)

    provenance = {
        "source_nc": str(source_nc),
        "output_nc": str(output_nc),
        "time_start": pd.Timestamp(start).isoformat(),
        "time_stop": pd.Timestamp(end).isoformat(),
        "freq": freq,
        "variable_mapping": {
            target: _selected_name(source_nc, names)
            for target, names in candidates.items()
        },
        "unit_conversions": {
            "temp": "Kelvin to degree_C when values exceed 150",
            "press_msl": "Pa to hPa when values exceed 2000",
            "kin": "W m-2 unchanged",
            **({"kout": "W m-2 unchanged"} if "kout" in out else {}),
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


def _pick(ds: xr.Dataset, names: tuple[str, ...], *, source_nc: Path | None = None) -> xr.DataArray:
    for name in names:
        if name in ds:
            return ds[name]
    prefix = f"{source_nc} lacks required AORC meteo variable" if source_nc is not None else "missing AORC meteo variable"
    raise KeyError(f"{prefix}; tried {list(names)}. Rerun source collection with AORC event_meteo variables enabled.")


def _selected_name(source_nc: Path, names: tuple[str, ...]) -> str:
    with xr.open_dataset(source_nc) as ds:
        for name in names:
            if name in ds:
                return name
    return ""


def _field(
    da: xr.DataArray,
    time: pd.DatetimeIndex,
    target: str,
    *,
    source_time_start=None,
    freq="1h",
) -> xr.DataArray:
    da = _normalize_dims(da).sortby("y").sortby("x")
    if "time" not in da.dims:
        raise ValueError(f"{da.name!r} has no time dimension")
    if source_time_start is not None:
        da = da.assign_coords(time=pd.date_range(pd.Timestamp(source_time_start), periods=da.sizes["time"], freq=freq))
    da = da.sortby("time")
    start = time[0]
    stop = time[-1]
    da = da.sel(time=slice(start, stop))
    if da.sizes.get("time", 0) == 0:
        raise ValueError(f"{da.name!r} has no data in requested event window {start}..{stop}")
    da = da.interp(time=time)
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
    return da.fillna(float(np.nanmedian(values)))


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
