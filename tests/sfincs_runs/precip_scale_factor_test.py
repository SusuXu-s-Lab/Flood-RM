import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sfincs_runs.hydrology import prepare_aorc_precip_for_sfincs


def _write_synthetic_aorc(path, n_time=6):
    time = pd.date_range("2018-09-15 00:00", periods=n_time, freq="1h")
    lat = np.array([35.6, 35.7, 35.8])
    lon = np.array([-79.9, -79.8, -79.7])
    rng = np.random.default_rng(0)
    data = rng.gamma(2.0, 3.0, size=(n_time, lat.size, lon.size)).astype("float32")
    ds = xr.Dataset(
        {"APCP_surface": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)
    return time


def test_scale_factor_multiplies_the_full_precip_field(tmp_path):
    src = tmp_path / "storm.nc"
    time = _write_synthetic_aorc(src)
    t0, t1 = time[0], time[-1]

    base = prepare_aorc_precip_for_sfincs(src, tmp_path / "base.nc", t_start=t0, t_stop=t1, scale_factor=1.0)
    scaled = prepare_aorc_precip_for_sfincs(src, tmp_path / "scaled.nc", t_start=t0, t_stop=t1, scale_factor=2.5)

    base_field = xr.open_dataset(base)["precip"]
    scaled_field = xr.open_dataset(scaled)["precip"]

    # the whole spatio-temporal field is scaled, structure preserved (no spatial flattening)
    np.testing.assert_allclose(scaled_field.to_numpy(), base_field.to_numpy() * 2.5, rtol=1e-5)
    assert scaled_field.shape == base_field.shape
    assert scaled_field.attrs["applied_scale_factor"] == 2.5
    assert base_field.attrs["applied_scale_factor"] == 1.0


def test_scale_factor_must_be_positive(tmp_path):
    src = tmp_path / "storm.nc"
    time = _write_synthetic_aorc(src)
    with pytest.raises(ValueError):
        prepare_aorc_precip_for_sfincs(src, tmp_path / "bad.nc", t_start=time[0], t_stop=time[-1], scale_factor=0.0)
