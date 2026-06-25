"""Cross-check: Delft-FIAT native return-period (risk=true) EAD.

The weighted-event EAD in :mod:`fiat_runs.risk` is the defensible figure for a *compound*
catalog. As an independent sanity bound we also run FIAT's native risk integrator, which
expects a return-period ladder of hazard maps and integrates damage log-linearly over the
exceedance probabilities (``fiat.methods.ead``).

Because the catalog carries no benchmark-RP flags, we pick, for each target return period,
the synthetic event whose *joint* ``sample_rp_years`` is nearest. This is approximate by
construction (joint RP != marginal coastal RP, one event per RP rather than the full
compound integral), so the result is a coarse bound, not ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from ._env import run_in_fiat_env
from .hazard import _NODATA
from .run import _toml_path, read_fiat_damages, write_event_settings

DEFAULT_TARGET_RPS = (10, 50, 100, 500)


def select_rp(catalog_csv, target_rps=DEFAULT_TARGET_RPS) -> dict:
    """Map each target return period to the synthetic event with the nearest joint RP."""
    cat = pd.read_csv(catalog_csv)
    cat = cat[cat["event_origin"].isin(("synthetic_body", "synthetic_tail"))]
    reps = {}
    for rp in target_rps:
        idx = (cat["sample_rp_years"] - rp).abs().idxmin()
        reps[int(rp)] = str(cat.loc[idx, "event_id"])
    return reps


def _ead_column(gdf) -> str:
    for c in gdf.columns:
        cl = c.lower()
        if "ead" in cl or ("risk" in cl and "damage" in cl):
            return c
    raise RuntimeError(f"no EAD/risk column in FIAT risk output; columns={list(gdf.columns)}")


def run_rp_risk(model_root, rasterizer, storage_root, rp_events, out_dir, hazard_root, *, srs="EPSG:4326") -> dict:
    """Export RP-band water-level rasters, run FIAT risk=true, return native EAD."""
    out_dir, hazard_root = Path(out_dir), Path(hazard_root)
    hazard_root.mkdir(parents=True, exist_ok=True)
    rps = sorted(rp_events)

    # delft_fiat risk mode wants ONE multi-band raster, band order matching return_periods.
    bands = [
        rasterizer.level_dataarray(Path(storage_root) / rp_events[rp] / "sfincs_map.nc")
        for rp in rps
    ]
    stack = xr.concat(bands, dim="band").assign_coords(band=list(range(1, len(rps) + 1)))
    stack = stack.rio.write_crs(bands[0].rio.crs).rio.write_nodata(_NODATA)
    multiband = hazard_root / "rp_stack.tif"
    stack.rio.to_raster(multiband, compress="deflate")

    settings = write_event_settings(
        out_dir / "settings.toml", model_root=model_root, hazard_files=multiband,
        out_dir=out_dir, risk=True, return_periods=rps, srs=srs,
    )
    run_in_fiat_env(["fiat", "run", _toml_path(settings)], capture_output=True)

    gdf = read_fiat_damages(out_dir)
    col = _ead_column(gdf)
    return {
        "method": "fiat_native_rp_risk",
        "return_periods": rps,
        "rp_events": {int(k): v for k, v in rp_events.items()},
        "ead": float(pd.to_numeric(gdf[col], errors="coerce").fillna(0).sum()),
        "ead_column": col,
    }
