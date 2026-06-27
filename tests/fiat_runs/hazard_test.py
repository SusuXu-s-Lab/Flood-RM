from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fiat_runs.hazard import M_TO_FT, _NODATA, WaterLevelRasterizer


def _write_sfincs_map(path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "x": (("n", "m"), np.array([[0.0, 10.0], [0.0, 10.0]])),
            "y": (("n", "m"), np.array([[10.0, 10.0], [0.0, 0.0]])),
            "msk": (("n", "m"), np.ones((2, 2), dtype=np.int16)),
            "zb": (("n", "m"), np.array([[0.0, 5.0], [-1.0, 0.0]])),
            "zs": (("time", "n", "m"), np.array([[[0.0, 0.0], [0.50, 0.0]]])),
            "zsmax": (("timemax", "n", "m"), np.array([[[1.0, 5.0], [0.55, 0.05]]])),
        },
        coords={"time": [0], "timemax": [0]},
    )
    ds.to_netcdf(path)


def test_hazard_raster_masks_dry_permanent_and_subthreshold_cells(tmp_path):
    map_path = tmp_path / "sfincs_map.nc"
    _write_sfincs_map(map_path)

    rasterizer = WaterLevelRasterizer(
        map_path,
        src_crs="EPSG:32619",
        dst_crs="EPSG:32619",
        res_m=10.0,
        max_dist_factor=0.1,
    )

    out = rasterizer.level_dataarray(map_path)

    assert out.values[0, 0] == pytest.approx(1.0 * M_TO_FT)
    assert out.values[0, 1] == _NODATA  # dry cell: zsmax == zb, not event flooding
    assert out.values[1, 0] == _NODATA  # permanent/tidal water: no storm increment
    assert out.values[1, 1] == _NODATA  # below huthresh
