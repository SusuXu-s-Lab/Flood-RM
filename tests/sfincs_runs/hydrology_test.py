from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import xarray as xr

from sfincs_runs.hydrology import (
    aggregate_ssurgo_infiltration_fields,
    compute_cn_recovery_seff,
    find_aorc_event_window,
    prepare_ksat_raster_for_cn_recovery,
    prepare_aorc_precip_for_sfincs,
    setup_hydromt_infiltration,
    summarize_soil_moisture,
    validate_infiltration_config,
    write_ssurgo_infiltration_rasters,
)


def test_find_aorc_event_window_by_member_id(tmp_path):
    target = tmp_path / "rainfall_marshfield_72h_rank0061_20180302T06.nc"
    target.touch()
    (tmp_path / "rainfall_marshfield_72h_rank0001_20060513T06.nc").touch()

    found = find_aorc_event_window(
        tmp_path, member_id="rainfall_marshfield_72h_rank0061"
    )

    assert found == target


def test_find_aorc_event_window_intersects_member_and_start(tmp_path):
    target = tmp_path / "rainfall_marshfield_72h_rank0061_20180302T06.nc"
    target.touch()
    (tmp_path / "rainfall_marshfield_72h_rank0061_20220822T12.nc").touch()

    found = find_aorc_event_window(
        tmp_path,
        member_id="rainfall_marshfield_72h_rank0061",
        storm_start="2018-03-02 06:00",
    )

    assert found == target


def test_prepare_aorc_precip_for_sfincs_pads_and_renames(tmp_path):
    src = tmp_path / "aorc.nc"
    da = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [pd.Timestamp("2018-03-02 06:00")],
            "latitude": [42.0, 41.0],
            "longitude": [-71.0, -70.0],
        },
        name="APCP_surface",
    )
    da.to_dataset().to_netcdf(src)

    out = prepare_aorc_precip_for_sfincs(
        src,
        tmp_path / "precip.nc",
        t_start="2018-03-02 05:00",
        t_stop="2018-03-02 07:00",
    )

    ds = xr.open_dataset(out)
    assert set(ds["precip"].dims) == {"time", "y", "x"}
    assert ds["precip"].sizes["time"] == 3
    assert float(ds["precip"].sel(time="2018-03-02 05:00").sum()) == 0.0
    assert ds.attrs["crs"] == "EPSG:4326"


def test_prepare_aorc_precip_for_sfincs_can_retime_storm_to_run_window(tmp_path):
    src = tmp_path / "aorc.nc"
    da = xr.DataArray(
        [
            [[1.0]],
            [[2.0]],
        ],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [
                pd.Timestamp("2018-03-02 06:00"),
                pd.Timestamp("2018-03-02 07:00"),
            ],
            "latitude": [42.0],
            "longitude": [-71.0],
        },
        name="APCP_surface",
    )
    da.to_dataset().to_netcdf(src)

    out = prepare_aorc_precip_for_sfincs(
        src,
        tmp_path / "precip.nc",
        t_start="2000-01-01 00:00",
        t_stop="2000-01-01 02:00",
        align_start_to_run=True,
    )

    ds = xr.open_dataset(out)
    assert float(ds["precip"].sel(time="2000-01-01 00:00").sum()) == 1.0
    assert float(ds["precip"].sel(time="2000-01-01 01:00").sum()) == 2.0
    assert float(ds["precip"].sel(time="2000-01-01 02:00").sum()) == 0.0


def test_prepare_aorc_precip_for_sfincs_can_stage_wettest_run_length_window(tmp_path):
    src = tmp_path / "aorc.nc"
    times = pd.date_range("2022-12-22 06:00", periods=8, freq="h")
    da = xr.DataArray(
        np.array([[[0.0]], [[0.0]], [[1.0]], [[4.0]], [[6.0]], [[5.0]], [[0.0]], [[2.0]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [42.0], "longitude": [-71.0]},
        name="APCP_surface",
    )
    da.to_dataset().to_netcdf(src)

    out = prepare_aorc_precip_for_sfincs(
        src,
        tmp_path / "precip.nc",
        t_start="2000-01-01 00:00",
        t_stop="2000-01-01 02:00",
        align_start_to_run=True,
        window_alignment="wettest",
    )

    ds = xr.open_dataset(out)
    assert pd.to_datetime(ds["precip"].time.values).tolist() == [
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 01:00"),
        pd.Timestamp("2000-01-01 02:00"),
    ]
    assert ds["precip"].values[:, 0, 0].tolist() == [4.0, 6.0, 5.0]


def test_summarize_soil_moisture(tmp_path):
    csv = tmp_path / "soil.csv"
    pd.DataFrame(
        {
            "time": [
                "2018-03-01 00:00",
                "2018-03-02 00:00",
                "2018-03-03 00:00",
            ],
            "SOIL_M": [0.2, 0.4, 0.8],
        }
    ).to_csv(csv, index=False)

    summary = summarize_soil_moisture(
        csv, at_time="2018-03-02 00:00", lookback_hours=24
    )

    assert summary["mean_soil_moisture"] == pytest.approx(0.3)
    assert summary["row_count"] == 2


def test_summarize_soil_moisture_prefers_soilsat_top_when_available(tmp_path):
    csv = tmp_path / "soil.csv"
    pd.DataFrame(
        {
            "time": ["2018-03-01 00:00", "2018-03-02 00:00"],
            "SOIL_M": [0.2, 0.4],
            "SOILSAT_TOP": [0.6, 0.8],
        }
    ).to_csv(csv, index=False)

    summary = summarize_soil_moisture(csv, at_time="2018-03-02 00:00", lookback_hours=24)

    assert summary["soil_moisture_variable"] == "SOILSAT_TOP"
    assert summary["mean_soil_moisture"] == pytest.approx(0.7)


def test_compute_cn_recovery_seff_uses_clipped_soilsat_top_fraction():
    smax = xr.DataArray([[0.10, 0.20, 0.30]], dims=("y", "x"))
    soilsat = xr.DataArray([[-0.1, 0.5, 1.2]], dims=("y", "x"), name="SOILSAT_TOP")

    seff = compute_cn_recovery_seff(smax, soilsat)

    assert seff.name == "seff"
    assert seff.attrs["units"] == "m"
    assert seff.values.tolist() == [[0.0, 0.1, 0.3]]


def test_compute_cn_recovery_seff_accepts_percent_soilsat_top():
    smax = xr.DataArray([[0.10, 0.20]], dims=("y", "x"))
    soilsat = xr.DataArray([[25.0, 100.0]], dims=("y", "x"), name="SOILSAT_TOP")

    seff = compute_cn_recovery_seff(smax, soilsat, input_units="percent")

    assert seff.values.tolist() == [[0.025, 0.2]]


def test_aggregate_ssurgo_infiltration_fields_maps_hsg_and_harmonic_ksat():
    attrs = pd.DataFrame(
        {
            "mukey": ["1", "1", "2"],
            "hydgrp": ["B/D", "B/D", "A"],
            "ksat_r": [10.0, 30.0, 100.0],
            "hzdept_r": [0.0, 20.0, 0.0],
            "hzdepb_r": [20.0, 40.0, 40.0],
        }
    )

    out = aggregate_ssurgo_infiltration_fields(attrs, drainage_condition="undrained")

    first = out.set_index("mukey").loc["1"]
    second = out.set_index("mukey").loc["2"]
    assert first["hsg"] == "D"
    assert first["hsg_code"] == 4
    assert first["ksat_mmhr"] == pytest.approx(54.0)
    assert second["hsg"] == "A"
    assert second["ksat_mmhr"] == pytest.approx(360.0)


def test_write_ssurgo_infiltration_rasters_aligns_to_template(tmp_path):
    template = tmp_path / "template.tif"
    with rasterio.open(
        template,
        "w",
        driver="GTiff",
        width=4,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(np.ones((2, 4), dtype="uint8"), 1)

    soils = gpd.GeoDataFrame(
        {"mukey": ["1", "2"]},
        geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2)],
        crs="EPSG:26919",
    )
    attrs = pd.DataFrame(
        {
            "mukey": ["1", "2"],
            "hydgrp": ["A", "C/D"],
            "ksat_r": [100.0, 10.0],
            "hzdept_r": [0.0, 0.0],
            "hzdepb_r": [40.0, 40.0],
        }
    )

    summary = write_ssurgo_infiltration_rasters(
        soils,
        attrs,
        template,
        hsg_out=tmp_path / "hsg.tif",
        ksat_out=tmp_path / "ksat.tif",
    )

    with rasterio.open(summary["hsg"]) as src:
        hsg = src.read(1).tolist()
        assert src.crs.to_epsg() == 26919
    with rasterio.open(summary["ksat"]) as src:
        ksat = src.read(1)

    assert hsg == [[1, 1, 4, 4], [1, 1, 4, 4]]
    assert float(ksat[0, 0]) == pytest.approx(360.0)
    assert float(ksat[0, 3]) == pytest.approx(36.0)
    assert summary["rasterized_polygons"] == 2
    assert not (tmp_path / ".hsg.tmp.tif").exists()
    assert not (tmp_path / ".ksat.tmp.tif").exists()


def test_write_ssurgo_infiltration_rasters_masks_to_land_domain(tmp_path):
    template = tmp_path / "template.tif"
    with rasterio.open(
        template,
        "w",
        driver="GTiff",
        width=4,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(np.ones((2, 4), dtype="uint8"), 1)

    soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[box(0, 0, 4, 2)],
        crs="EPSG:26919",
    )
    land = gpd.GeoDataFrame(
        {"name": ["land"]},
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:26919",
    )
    attrs = pd.DataFrame(
        {
            "mukey": ["1"],
            "hydgrp": ["B"],
            "ksat_r": [20.0],
            "hzdept_r": [0.0],
            "hzdepb_r": [40.0],
        }
    )

    summary = write_ssurgo_infiltration_rasters(
        soils,
        attrs,
        template,
        hsg_out=tmp_path / "hsg_land.tif",
        ksat_out=tmp_path / "ksat_land.tif",
        land_domain=land,
    )

    with rasterio.open(summary["hsg"]) as src:
        hsg = src.read(1).tolist()
    with rasterio.open(summary["ksat"]) as src:
        ksat = src.read(1)

    assert hsg == [[2, 2, 0, 0], [2, 2, 0, 0]]
    assert np.isfinite(ksat[:, :2]).all()
    assert np.isnan(ksat[:, 2:]).all()
    assert summary["land_pixels"] == 4


def test_prepare_ksat_raster_for_cn_recovery_scales_and_caps_values(tmp_path):
    source = tmp_path / "ksat_raw.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=3,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:26919",
        transform=from_origin(0, 1, 1, 1),
        nodata=np.nan,
    ) as dst:
        dst.write(np.array([[360.0, 1000.0, np.nan]], dtype="float32"), 1)

    summary = prepare_ksat_raster_for_cn_recovery(
        source,
        tmp_path / "ksat_effective.tif",
        scale_factor=0.1,
        max_mmhr=75.0,
    )

    with rasterio.open(summary["ksat"]) as src:
        out = src.read(1)

    assert out[0, 0] == pytest.approx(36.0)
    assert out[0, 1] == pytest.approx(75.0)
    assert np.isnan(out[0, 2])
    assert summary["scale_factor"] == 0.1
    assert summary["max_mmhr"] == 75.0
    assert summary["capped_fraction"] == pytest.approx(0.5)


def test_validate_infiltration_config_requires_cn_recovery_inputs_for_rainfall():
    with pytest.raises(RuntimeError) as excinfo:
        validate_infiltration_config(
            {"method": "cn_with_recovery", "lulc": "worldcover"},
            event_drivers=["coastal_water_level", "rainfall", "soil_moisture"],
        )

    message = str(excinfo.value)
    assert "hsg" in message
    assert "ksat" in message
    assert "effective" in message


def test_validate_infiltration_config_allows_explicitly_disabled_hydrology():
    validate_infiltration_config(
        {"enabled": False, "method": "cn_with_recovery"},
        event_drivers=["rainfall", "soil_moisture"],
    )


def test_setup_hydromt_infiltration_returns_disabled_summary_without_writes(tmp_path):
    class FakeSf:
        pass

    summary = setup_hydromt_infiltration(
        FakeSf(),
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "coastal_wave_coupling": {
                "hydrology": {
                    "infiltration": {"enabled": False, "method": "cn_with_recovery"}
                }
            },
        },
        {"location_root": tmp_path},
    )

    assert summary["enabled"] is False
    assert summary["written"] is False


def test_setup_hydromt_infiltration_registers_conditioned_ksat_raster(tmp_path):
    raw_ksat = tmp_path / "ksat_raw.tif"
    with rasterio.open(
        raw_ksat,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:26919",
        transform=from_origin(0, 1, 1, 1),
        nodata=np.nan,
    ) as dst:
        dst.write(np.array([[1000.0]], dtype="float32"), 1)
    hsg = tmp_path / "hsg.tif"
    hsg.write_text("placeholder", encoding="utf-8")

    class FakeCatalog:
        def __init__(self):
            self.sources = {}

        def from_dict(self, data):
            self.sources.update(data)

    class FakeInfiltration:
        def create_cn_with_recovery(self, **kwargs):
            self.kwargs = kwargs

    class FakeSf:
        def __init__(self):
            self.data_catalog = FakeCatalog()
            self.quadtree_infiltration = FakeInfiltration()

    sf = FakeSf()
    summary = setup_hydromt_infiltration(
        sf,
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "coastal_wave_coupling": {
                "hydrology": {
                    "infiltration": {
                        "enabled": True,
                        "method": "cn_with_recovery",
                        "lulc": "worldcover",
                        "hsg": str(hsg),
                        "ksat": str(raw_ksat),
                        "ksat_effective": str(tmp_path / "ksat_effective.tif"),
                        "ksat_scale_factor": 0.1,
                        "ksat_max_mmhr": 75.0,
                        "effective": 0.5,
                    }
                }
            },
        },
        {"location_root": tmp_path},
    )

    with rasterio.open(tmp_path / "ksat_effective.tif") as src:
        assert float(src.read(1)[0, 0]) == pytest.approx(75.0)
    assert sf.data_catalog.sources["ksat"]["uri"] == str(tmp_path / "ksat_effective.tif")
    assert sf.quadtree_infiltration.kwargs["ksat"] == "ksat"
    assert sf.quadtree_infiltration.kwargs["factor_ksat"] == pytest.approx(1.0 / 3.6)
    assert summary["ksat_conditioning"]["capped_fraction"] == pytest.approx(1.0)


def test_setup_hydromt_infiltration_prefers_regular_grid_component(tmp_path):
    hsg = tmp_path / "hsg.tif"
    ksat = tmp_path / "ksat.tif"
    hsg.write_text("placeholder", encoding="utf-8")
    ksat.write_text("placeholder", encoding="utf-8")

    class FakeCatalog:
        def __init__(self):
            self.sources = {}

        def from_dict(self, data):
            self.sources.update(data)

    class FakeInfiltration:
        def __init__(self):
            self.kwargs = None

        def create_cn_with_recovery(self, **kwargs):
            self.kwargs = kwargs

    class FakeSf:
        def __init__(self):
            self.data_catalog = FakeCatalog()
            self.infiltration = FakeInfiltration()
            self.quadtree_infiltration = FakeInfiltration()

    sf = FakeSf()
    summary = setup_hydromt_infiltration(
        sf,
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "coastal_wave_coupling": {
                "hydrology": {
                    "infiltration": {
                        "enabled": True,
                        "method": "cn_with_recovery",
                        "lulc": "worldcover",
                        "hsg": str(hsg),
                        "ksat": str(ksat),
                        "effective": 0.5,
                    }
                }
            },
        },
        {"location_root": tmp_path},
    )

    assert summary["written"] is True
    assert sf.infiltration.kwargs["hsg"] == "hsg"
    assert sf.infiltration.kwargs["ksat"] == "ksat"
    assert sf.quadtree_infiltration.kwargs is None


def test_setup_hydromt_infiltration_uses_quadtree_component_for_quadtree_grid(tmp_path):
    hsg = tmp_path / "hsg.tif"
    ksat = tmp_path / "ksat.tif"
    hsg.write_text("placeholder", encoding="utf-8")
    ksat.write_text("placeholder", encoding="utf-8")

    class FakeCatalog:
        def __init__(self):
            self.sources = {}

        def from_dict(self, data):
            self.sources.update(data)

    class RegularInfiltration:
        def create_cn_with_recovery(self, **kwargs):
            raise AttributeError("'SfincsGrid' object has no attribute 'nmax'")

    class QuadtreeInfiltration:
        def __init__(self):
            self.kwargs = None

        def create_cn_with_recovery(self, **kwargs):
            self.kwargs = kwargs

    class FakeSf:
        def __init__(self):
            self.grid_type = "quadtree"
            self.data_catalog = FakeCatalog()
            self.infiltration = RegularInfiltration()
            self.quadtree_infiltration = QuadtreeInfiltration()

    sf = FakeSf()
    summary = setup_hydromt_infiltration(
        sf,
        {
            "event_drivers": ["rainfall", "soil_moisture"],
            "coastal_wave_coupling": {
                "hydrology": {
                    "infiltration": {
                        "enabled": True,
                        "method": "cn_with_recovery",
                        "lulc": "worldcover",
                        "hsg": str(hsg),
                        "ksat": str(ksat),
                        "effective": 0.5,
                    }
                }
            },
        },
        {"location_root": tmp_path},
    )

    assert summary["written"] is True
    assert sf.quadtree_infiltration.kwargs["hsg"] == "hsg"
    assert sf.quadtree_infiltration.kwargs["ksat"] == "ksat"
