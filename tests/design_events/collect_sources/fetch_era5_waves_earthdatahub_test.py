import numpy as np
import pandas as pd
import xarray as xr

from design_events.collect_sources.fetch_era5_waves_earthdatahub import (
    chunk_time_windows,
    earthdatahub_era5_ocean_zarr,
    earthdatahub_storage_options,
    earthdatahub_store_url,
    fetch_era5_waves_from_earthdatahub,
    normalize_earthdatahub_zarr_url,
    read_earthdatahub_auth,
    subset_earthdatahub_waves,
)


def _source_dataset(longitudes=None, latitudes=None, times=None):
    times = times if times is not None else pd.date_range("2018-01-01", periods=4, freq="h")
    latitudes = latitudes if latitudes is not None else [42.5, 42.0]
    longitudes = longitudes if longitudes is not None else [289.0, 289.5, 290.0]
    shape = (len(times), len(latitudes), len(longitudes))
    return xr.Dataset(
        {
            "swh": (("time", "latitude", "longitude"), np.ones(shape)),
            "pp1d": (("time", "latitude", "longitude"), np.ones(shape) * 8),
            "mwd": (("time", "latitude", "longitude"), np.ones(shape) * 90),
            "wdw": (("time", "latitude", "longitude"), np.ones(shape) * 0.4),
            "wind": (("time", "latitude", "longitude"), np.ones(shape) * 12),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )


def test_earthdatahub_store_url_can_embed_token():
    url = earthdatahub_store_url(token="abc123")

    assert url.startswith("https://edh:abc123@api.earthdatahub.destine.eu/")
    assert earthdatahub_storage_options(token="abc123") is None


def test_earthdatahub_storage_options_use_netrc_without_token(monkeypatch):
    monkeypatch.delenv("EARTHDATAHUB_TOKEN", raising=False)

    assert earthdatahub_storage_options() == {"client_kwargs": {"trust_env": True}}


def test_read_earthdatahub_auth_parses_labeled_url_and_key(tmp_path):
    auth_path = tmp_path / "api-key.txt"
    auth_path.write_text(
        "URL: https://example.test/waves.zarr\n\nkey: secret-token\n",
        encoding="utf-8",
    )

    auth = read_earthdatahub_auth(auth_path)

    assert auth == {"url": "https://example.test/waves.zarr", "token": "secret-token"}


def test_normalize_earthdatahub_zarr_url_expands_api_base_url():
    url = normalize_earthdatahub_zarr_url("https://api.earthdatahub.destine.eu")

    assert url == earthdatahub_era5_ocean_zarr


def test_subset_earthdatahub_waves_uses_short_variables_and_converts_longitude():
    ds = _source_dataset()

    subset = subset_earthdatahub_waves(
        ds,
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01 01:00"), pd.Timestamp("2018-01-01 02:00")),
    )

    assert list(subset.data_vars) == ["swh", "pp1d", "mwd", "wdw"]
    assert list(subset.longitude.values) == [289.0, 289.5, 290.0]
    assert list(subset.time.values) == list(ds.time.values[1:3])


def test_subset_earthdatahub_waves_handles_ascending_latitude():
    ds = _source_dataset(latitudes=[42.0, 42.5])

    subset = subset_earthdatahub_waves(
        ds,
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01 03:00")),
    )

    assert list(subset.latitude.values) == [42.0, 42.5]


def test_chunk_time_windows_splits_inclusive_hourly_window_by_months():
    windows = chunk_time_windows("2018-01-15", "2018-04-01", chunk_months=2)

    assert windows == [
        (pd.Timestamp("2018-01-15"), pd.Timestamp("2018-03-14 23:00:00")),
        (pd.Timestamp("2018-03-15"), pd.Timestamp("2018-04-01")),
    ]


def test_fetch_era5_waves_from_earthdatahub_writes_netcdf(tmp_path):
    output_path = tmp_path / "era5_waves.nc"

    def opener(url=None, token=None):
        assert url is None
        assert token is None
        return _source_dataset()

    result = fetch_era5_waves_from_earthdatahub(
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01 03:00")),
        output_path=output_path,
        opener=opener,
    )

    assert result == output_path
    written = xr.open_dataset(output_path)
    assert list(written.data_vars) == ["swh", "pp1d", "mwd", "wdw"]


def test_fetch_era5_waves_from_earthdatahub_writes_chunked_netcdf_without_duplicate_times(tmp_path):
    output_path = tmp_path / "era5_waves.nc"
    times = pd.date_range("2018-01-01", "2018-03-01", freq="h")

    def opener(url=None, token=None):
        return _source_dataset(times=times)

    fetch_era5_waves_from_earthdatahub(
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-03-01")),
        output_path=output_path,
        opener=opener,
        chunk_months=1,
    )

    with xr.open_dataset(output_path) as written:
        assert int(written.sizes["time"]) == len(times)
        assert pd.Index(written.time.values).is_unique
        assert pd.Timestamp(written.time.values[0]) == pd.Timestamp("2018-01-01")
        assert pd.Timestamp(written.time.values[-1]) == pd.Timestamp("2018-03-01")


def test_fetch_era5_waves_from_earthdatahub_skips_existing_without_opening_zarr(tmp_path):
    output_path = tmp_path / "era5_waves.nc"
    _source_dataset().to_netcdf(output_path)

    def opener(url=None, token=None):
        raise AssertionError("existing ERA5 output should not be redownloaded")

    result = fetch_era5_waves_from_earthdatahub(
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01 03:00")),
        output_path=output_path,
        opener=opener,
        force=False,
    )

    assert result == output_path


def test_fetch_era5_waves_from_earthdatahub_uses_auth_file(tmp_path):
    output_path = tmp_path / "era5_waves.nc"
    auth_path = tmp_path / "api-key.txt"
    auth_path.write_text(
        "URL: https://example.test/waves.zarr\nkey: secret-token\n",
        encoding="utf-8",
    )
    calls = []

    def opener(url=None, token=None):
        calls.append((url, token))
        return _source_dataset()

    fetch_era5_waves_from_earthdatahub(
        bbox_wgs84=(-71.0, 42.0, -70.0, 42.5),
        time_window=(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01 03:00")),
        output_path=output_path,
        auth_path=auth_path,
        opener=opener,
    )

    assert calls == [("https://example.test/waves.zarr", "secret-token")]
