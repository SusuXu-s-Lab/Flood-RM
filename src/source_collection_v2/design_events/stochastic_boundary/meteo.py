from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .gridded import to_yx


AORC_METEO = {
    "temp": ("TMP_2maboveground", "TMP_2m", "TMP_surface", "temp"),
    "press_msl": ("PRES_surface", "PRMSL_meansealevel", "press_msl"),
    "kin": ("DSWRF_surface", "SWDOWN", "kin"),
}


def aorc_variable_candidates(config: dict | None = None) -> dict[str, tuple[str, ...]]:
    configured = (((config or {}).get("collection", {}) or {}).get("aorc_sst", {}) or {}).get("event_meteo", {}).get("source_variables", {}) or {}
    out = {}
    for target, defaults in AORC_METEO.items():
        value = configured.get(target)
        ordered = (value,) if isinstance(value, str) else tuple(value or ())
        out[target] = tuple(dict.fromkeys([*ordered, *defaults]))
    return out


def aorc_source_variables(ds: xr.Dataset, spec: dict, config: dict | None = None) -> list[str]:
    variables = [spec.get("variable", "APCP_surface")]
    if (spec.get("event_meteo") or {}).get("enabled", False):
        for candidates in aorc_variable_candidates(config).values():
            match = next((name for name in candidates if name in ds), None)
            if match:
                variables.append(match)
    return list(dict.fromkeys(variables))


def write_wflow_temp_pet(source_nc, output_nc, *, start, end, freq="1h", precip_template=None, config=None) -> dict:
    """AORC event meteo -> HydroMT-Wflow temp/PET forcing fields."""
    time = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
    with xr.open_dataset(source_nc) as ds:
        selected = {k: _pick(ds, names) for k, names in aorc_variable_candidates(config).items()}
        out = xr.Dataset({
            "temp": _field(selected["temp"], time, "temp"),
            "press_msl": _field(selected["press_msl"], time, "press_msl"),
            "kin": _field(selected["kin"], time, "radiation"),
        }).astype("float32")
    if precip_template and Path(precip_template).exists():
        with xr.open_dataset(precip_template) as template:
            if "precip" in template:
                out = out.interp(y=template["y"], x=template["x"], kwargs={"fill_value": "extrapolate"})
    out["temp"].attrs["units"] = "degree_C"
    out["press_msl"].attrs["units"] = "hPa"
    out["kin"].attrs["units"] = "W m-2"
    out.attrs.update(source=str(source_nc), hydromt_wflow_contract="setup_temp_pet_forcing(pet_method=makkink)")
    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_nc)
    return {"source_nc": str(source_nc), "output_nc": str(output_nc), "start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat()}


def _pick(ds: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray:
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(f"missing AORC meteo variable; tried {list(names)}")


def _field(da: xr.DataArray, time: pd.DatetimeIndex, target: str) -> xr.DataArray:
    da = to_yx(da).sortby("time").interp(time=time)
    if target == "temp":
        da = xr.where(da > 150, da - 273.15, da).rename("temp")
    elif target == "press_msl":
        da = xr.where(da > 2000, da / 100.0, da).rename("press_msl")
    else:
        da = da.clip(min=0).rename("kin")
    values = da.values
    return da.fillna(0 if target == "radiation" else float(np.nanmedian(values)))
