from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

M_TO_FT = 3.28084
DEFAULT_THRESHOLDS_FT = (0.5, 1.0, 2.0)


def annual_rate_table(events: pd.DataFrame, total_rate_per_year: float) -> pd.DataFrame:
    """Map stochastic event weights to annual rates.

    ``lambda_e = Lambda * w_e``.
    """
    table = events.copy()
    table["probability_weight"] = pd.to_numeric(table["probability_weight"], errors="coerce")
    table["total_rate_per_year"] = float(total_rate_per_year)
    table["annual_rate"] = float(total_rate_per_year) * table["probability_weight"]
    return table


def poisson_exceedance_probability(annual_rate) -> np.ndarray:
    """At-least-one annual exceedance probability for a Poisson process."""
    rate = np.asarray(annual_rate, dtype=float)
    return 1.0 - np.exp(-rate)


def completed_runs(storage_root: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(storage_root).glob("**/sfincs_map.nc")):
        event_id = path.parent.name
        domain_id = path.parent.name if path.parent.parent.name.startswith("evt_") else ""
        rows.append({"event_id": event_id if not domain_id else path.parent.parent.name, "sfincs_domain_id": domain_id, "map_path": str(path)})
    return pd.DataFrame(rows)


def masked_sfincs_depth(
    map_path: str | Path,
    *,
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
    depth_kind: str = "incremental",
) -> dict[str, np.ndarray]:
    """Return masked SFINCS flood depth in feet from ``sfincs_map.nc``."""
    with xr.open_dataset(map_path) as ds:
        if "zsmax" in ds:
            zsmax = ds["zsmax"].max("timemax") if "timemax" in ds["zsmax"].dims else ds["zsmax"]
        elif "zs" in ds:
            zsmax = ds["zs"].max("time", skipna=True) if "time" in ds["zs"].dims else ds["zs"]
        else:
            raise KeyError("sfincs_map.nc must contain zsmax or zs")
        zb = ds["zb"]
        depth_m = zsmax - zb
        if "zs" in ds and "time" in ds["zs"].dims:
            baseline_depth = (ds["zs"].isel(time=0) - zb).clip(min=0.0).fillna(0.0)
        else:
            baseline_depth = xr.zeros_like(depth_m)
        incremental_m = depth_m - baseline_depth
        flooded = incremental_m > float(huthresh_m)
        if land_min_elev_m is not None:
            flooded = flooded & (zb >= float(land_min_elev_m))
        value = depth_m if depth_kind == "total" else incremental_m
        return {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
            "depth_ft": np.asarray((value.where(flooded) * M_TO_FT).values, dtype=float),
        }


def catalog_depth_probability(
    runs: pd.DataFrame,
    event_rates: pd.DataFrame,
    *,
    thresholds_ft=DEFAULT_THRESHOLDS_FT,
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
    depth_kind: str = "incremental",
) -> xr.Dataset:
    """Catalog-weighted flood-depth annual exceedance probability rasters.

    ``p_d(x) = 1 - exp(-sum_e lambda_e I[D_e(x) > d])``.
    """
    if runs.empty:
        raise ValueError("runs is empty")
    key_cols = ["event_id"] + (["sfincs_domain_id"] if "sfincs_domain_id" in runs and "sfincs_domain_id" in event_rates else [])
    rate = event_rates.copy()
    for col in key_cols:
        rate[col] = rate[col].astype(str)
    lookup = rate.drop_duplicates(key_cols).set_index(key_cols)

    first = masked_sfincs_depth(runs.iloc[0]["map_path"], huthresh_m=huthresh_m, land_min_elev_m=land_min_elev_m, depth_kind=depth_kind)
    shape = np.asarray(first["depth_ft"]).shape
    threshold_values = tuple(float(value) for value in thresholds_ft)
    exceedance_rate = {threshold: np.zeros(shape, dtype=float) for threshold in threshold_values}
    used_weight = 0.0
    used_events = 0

    for row in runs.itertuples(index=False):
        row_dict = row._asdict()
        key = tuple(str(row_dict[col]) for col in key_cols)
        if key not in lookup.index:
            continue
        rec = lookup.loc[key]
        annual_rate = float(pd.to_numeric(rec.get("annual_rate"), errors="coerce"))
        weight = float(pd.to_numeric(rec.get("probability_weight", np.nan), errors="coerce"))
        if not np.isfinite(annual_rate) or annual_rate <= 0:
            continue
        data = masked_sfincs_depth(row_dict["map_path"], huthresh_m=huthresh_m, land_min_elev_m=land_min_elev_m, depth_kind=depth_kind)
        depth = np.asarray(data["depth_ft"], dtype=float)
        for threshold in threshold_values:
            exceedance_rate[threshold] += np.where(np.isfinite(depth) & (depth > threshold), annual_rate, 0.0)
        if np.isfinite(weight):
            used_weight += weight
        used_events += 1

    ds = xr.Dataset(coords={"n": np.arange(shape[0]), "m": np.arange(shape[1])})
    ds = ds.assign_coords(x=(("n", "m"), np.asarray(first["x"], dtype=float)), y=(("n", "m"), np.asarray(first["y"], dtype=float)))
    for threshold in threshold_values:
        token = str(threshold).replace(".", "p")
        rate_name = f"depth_gt_{token}ft_annual_rate"
        prob_name = f"depth_gt_{token}ft_aep"
        ds[rate_name] = (("n", "m"), exceedance_rate[threshold])
        ds[prob_name] = (("n", "m"), poisson_exceedance_probability(exceedance_rate[threshold]))
        ds[rate_name].attrs.update(long_name=f"Annual exceedance rate for depth > {threshold:g} ft", units="1/year")
        ds[prob_name].attrs.update(long_name=f"P(annual max flood depth > {threshold:g} ft)", units="1")
    ds.attrs.update(
        completed_event_count=int(used_events),
        covered_probability_weight=float(used_weight),
        thresholds_ft=list(threshold_values),
        probability_method="1 - exp(-sum(total_rate_per_year * probability_weight for exceeding events))",
        depth_kind=depth_kind,
    )
    return ds
