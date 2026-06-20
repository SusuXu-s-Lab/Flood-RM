"""Per-event flood-hazard export for Delft-FIAT (runs in the main project env).

SFINCS writes a rotated/curvilinear map grid (dims ``n, m`` with 2-D ``x``/``y`` cell
centres in absolute UTM) plus a precomputed ``zsmax`` (peak water-surface elevation).

We export the peak **water level** (``zsmax``), not depth, and let FIAT subtract each
structure's own ground elevation. The ground used in the FIAT model is sampled from the
high-resolution SFINCS subgrid DEM (see ``build_model.apply_dem_ground_elevation``), so a
waterfront structure on high ground stays dry even when its coarse 60 m SFINCS cell is wet.
Doing the depth subtraction at the structure's 10 m ground resolution — rather than against
the 60 m cell bed — is what keeps damage magnitudes credible.

**Flood mask (critical).** Water level is only emitted on cells the event genuinely
inundates: ``zsmax - zb > huthresh`` (wet) AND ``zsmax - zs_t0 > huthresh`` (a real
increment above the pre-storm/tidal baseline). Without this, dry-land cells report
``zsmax == zb`` (the 60 m bed); subtracting the 10 m structure ground turns terrain noise
into spurious shallow flooding on high ground — diagnosed at ~54% of one event's damage.
The mask also removes permanent/tidal standing water. This mirrors the inland incremental
depth that ``06`` uses for its flood metrics.

``hydromt_sfincs.utils.downscale_floodmap`` cannot consume the curvilinear grid, so a
regular UTM target grid takes the water level of its nearest active *flooded* cell. Output
is a **water-level** GeoTIFF in **feet** (HAZUS curve unit), reprojected to the FIAT model
CRS, which FIAT reads with ``hazard.elevation_reference = "datum"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from scipy.spatial import cKDTree

M_TO_FT = 3.28084
_NODATA = np.float32(-9999.0)


class WaterLevelRasterizer:
    """Reusable nearest-neighbour rasteriser of SFINCS peak water level onto a regular grid.

    The SFINCS grid geometry is identical across every run, so the cell->pixel index map is
    precomputed once from a single map and reused for all events/scenarios.

    Parameters
    ----------
    map_path : any one completed ``sfincs_map.nc`` (its grid geometry is shared by all runs).
    src_crs  : CRS of the SFINCS map cell coordinates (absolute UTM, e.g. EPSG:32619).
    dst_crs  : CRS to reproject the water-level raster to (the FIAT model CRS).
    res_m    : target grid resolution in metres.
    max_dist_factor : cap nearest-cell distance at this multiple of the SFINCS cell size so
        target pixels outside the SFINCS domain stay nodata instead of extrapolating.
    huthresh_m : minimum water depth/increment in metres for a cell to count as flooded.
    """

    def __init__(self, map_path, *, src_crs="EPSG:32619", dst_crs="EPSG:4326",
                 res_m=30.0, max_dist_factor=1.5, huthresh_m=0.1):
        self.src_crs = src_crs
        self.dst_crs = dst_crs
        self.huthresh_m = float(huthresh_m)

        with xr.open_dataset(map_path) as ds:
            cx = np.asarray(ds["x"].values, dtype=float)
            cy = np.asarray(ds["y"].values, dtype=float)
            active = np.asarray(ds["msk"].values) > 0
            cell_m = float(np.nanmedian(np.abs(np.diff(cx, axis=1))))

        self._active = active.ravel()
        ax = cx.ravel()[self._active]
        ay = cy.ravel()[self._active]

        # Regular north-up target grid spanning the active SFINCS extent.
        xmin, xmax = ax.min(), ax.max()
        ymin, ymax = ay.min(), ay.max()
        self._xs = np.arange(xmin, xmax + res_m, res_m)
        self._ys = np.arange(ymax, ymin - res_m, -res_m)
        gx, gy = np.meshgrid(self._xs, self._ys)
        self._shape = gx.shape

        tree = cKDTree(np.column_stack([ax, ay]))
        dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        self._idx = idx  # index into the active-cell subset
        self._within = dist <= max_dist_factor * cell_m

    def event_level_cells(self, map_path) -> np.ndarray:
        """SFINCS peak water level (metres) over genuinely flooded active cells."""
        with xr.open_dataset(map_path) as ds:
            zsmax = _peak_water_level(ds)
            zs0 = _initial_water_level(ds)
            event_depth = zsmax - ds["zb"]
            event_increment = zsmax - zs0
            flooded = (event_depth > self.huthresh_m) & (event_increment > self.huthresh_m)
            level = np.where(flooded, zsmax, np.nan).ravel()
        return level[self._active]

    def level_dataarray(self, map_path) -> xr.DataArray:
        """Per-event water-level field (feet), reprojected to ``dst_crs``, nodata outside domain."""
        level_cells = self.event_level_cells(map_path)
        level_m = np.where(self._within, level_cells[self._idx], np.nan)
        level_ft = (level_m * M_TO_FT).reshape(self._shape).astype(np.float32)
        out = xr.DataArray(
            np.where(np.isfinite(level_ft), level_ft, _NODATA),
            dims=("y", "x"),
            coords={"y": self._ys, "x": self._xs},
        )
        out = out.rio.write_crs(self.src_crs).rio.write_nodata(_NODATA)
        return out.rio.reproject(self.dst_crs, nodata=_NODATA)

    def export(self, map_path, out_tif) -> dict:
        """Write a per-event water-level GeoTIFF (feet, datum-referenced). Returns an audit dict."""
        out = self.level_dataarray(map_path)
        Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
        out.rio.to_raster(out_tif, compress="deflate")
        valid = out.values[np.isfinite(out.values) & (out.values != _NODATA)]
        return {
            "out_tif": str(out_tif),
            "domain_pixels": int(valid.size),
            "max_level_ft": float(valid.max()) if valid.size else 0.0,
        }


def _peak_water_level(ds: xr.Dataset) -> xr.DataArray:
    if "zsmax" in ds:
        zsmax = ds["zsmax"]
        return zsmax.max("timemax") if "timemax" in zsmax.dims else zsmax
    zs = ds["zs"]
    return zs.max("time") if "time" in zs.dims else zs


def _initial_water_level(ds: xr.Dataset) -> xr.DataArray:
    if "zs" not in ds:
        raise KeyError("SFINCS map is missing 'zs'; required for incremental flood masking")
    zs = ds["zs"]
    return zs.isel(time=0) if "time" in zs.dims else zs
