import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, Polygon

from sfincs_runs.snapwave_setup import (
    derive_seaward_boundary,
    era5_spectra_to_snapwave_timeseries,
    inject_runup_config,
    repair_snapwave_directional_spreading_file,
    select_snapwave_boundary_points,
    unwrap_direction_degrees,
    validate_runup_transects,
    write_runup_gauges_file,
)


def test_derive_seaward_boundary_drops_shared_coastline():
    # 1000x1000 square domain; west half is land, east half is ocean.
    domain = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    land = Polygon([(0, 0), (500, 0), (500, 1000), (0, 1000)])

    offshore, seaward = derive_seaward_boundary(domain, land, tolerance_m=1.0)

    assert offshore.area == pytest.approx(500 * 1000)
    # Seaward total length ≈ east edge (1000) + top half (500) + bottom half (500) = 2000 m
    assert seaward.length == pytest.approx(2000, abs=10)
    # The shared coastline at x=500 must not be inside the seaward edge.
    coastline_segment = LineString([(500, 250), (500, 750)])
    assert not seaward.buffer(0.5).contains(coastline_segment)
    # The east open-ocean edge at x=1000 must be inside the seaward edge.
    east_edge_segment = LineString([(1000, 100), (1000, 900)])
    assert seaward.buffer(0.5).contains(east_edge_segment)


def test_unwrap_direction_degrees_removes_zero_360_plot_jump():
    raw = pd.Series([350.0, 355.0, 2.0, 8.0, 15.0])

    unwrapped = unwrap_direction_degrees(raw)

    assert unwrapped.tolist() == pytest.approx([350.0, 355.0, 362.0, 368.0, 375.0])
    assert float(unwrapped.diff().abs().max()) < 15.0


def test_derive_seaward_boundary_can_select_ocean_seed_side_only():
    # Same domain as above, but Marshfield should use the ocean-facing bbox
    # side only, not the lateral north/south bbox edges.
    domain = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    land = Polygon([(0, 0), (500, 0), (500, 1000), (0, 1000)])

    offshore, seaward = derive_seaward_boundary(
        domain,
        land,
        ocean_seed=Point(900, 500),
        tolerance_m=1.0,
    )

    assert offshore.area == pytest.approx(500 * 1000)
    assert seaward.length == pytest.approx(1000, abs=10)
    east_edge_segment = LineString([(1000, 100), (1000, 900)])
    north_lateral_segment = LineString([(600, 1000), (900, 1000)])
    south_lateral_segment = LineString([(600, 0), (900, 0)])
    assert seaward.buffer(0.5).contains(east_edge_segment)
    assert not seaward.buffer(0.5).contains(north_lateral_segment)
    assert not seaward.buffer(0.5).contains(south_lateral_segment)


def test_derive_seaward_boundary_handles_land_outside_domain():
    # If land is entirely outside the domain, offshore == domain and the
    # seaward edge is the full domain exterior.
    domain = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    land = Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])

    offshore, seaward = derive_seaward_boundary(domain, land, tolerance_m=1.0)

    assert offshore.area == pytest.approx(100 * 100)
    assert seaward.length == pytest.approx(400, abs=1)


def test_derive_seaward_boundary_raises_when_domain_is_all_land():
    domain = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    land = Polygon([(-10, -10), (110, -10), (110, 110), (-10, 110)])

    with pytest.raises(ValueError, match="no offshore region"):
        derive_seaward_boundary(domain, land)


def _toy_era5(times, lats, lons, fills):
    # fills keyed by variable name; broadcast to (time, lat, lon)
    shape = (len(times), len(lats), len(lons))
    return xr.Dataset(
        {name: (("valid_time", "latitude", "longitude"), np.full(shape, value))
         for name, value in fills.items()},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )


def test_select_snapwave_boundary_points_places_points_at_spacing():
    # straight 1000 m offshore edge; polygon covers a wide strip around it
    edge = LineString([(0.0, 0.0), (1000.0, 0.0)])
    polygon = Polygon([(-50.0, -100.0), (1050.0, -100.0), (1050.0, 100.0), (-50.0, 100.0)])

    points = select_snapwave_boundary_points(edge, polygon, spacing_m=100.0, crs="EPSG:26919")

    xs = points.geometry.x.values
    assert len(points) == 11
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(1000.0)
    diffs = xs[1:] - xs[:-1]
    assert all(d == pytest.approx(100.0) for d in diffs)


def test_select_snapwave_boundary_points_uses_zero_padded_names():
    edge = LineString([(0.0, 0.0), (300.0, 0.0)])
    polygon = Polygon([(-50.0, -100.0), (350.0, -100.0), (350.0, 100.0), (-50.0, 100.0)])

    points = select_snapwave_boundary_points(edge, polygon, spacing_m=100.0, crs="EPSG:26919")

    assert points["name"].tolist() == ["0001", "0002", "0003", "0004"]


def test_select_snapwave_boundary_points_drops_points_outside_polygon():
    # edge spans x=0..400, but offshore polygon only covers x>=200
    edge = LineString([(0.0, 0.0), (400.0, 0.0)])
    polygon = Polygon([(200.0, -50.0), (400.0, -50.0), (400.0, 50.0), (200.0, 50.0)])

    points = select_snapwave_boundary_points(edge, polygon, spacing_m=100.0, crs="EPSG:26919")

    xs = points.geometry.x.values
    assert xs.min() >= 200.0
    assert points["name"].tolist() == ["0001", "0002", "0003"]


def test_era5_spectra_to_snapwave_timeseries_returns_four_dataframes():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    ds = _toy_era5(
        times=times,
        lats=[42.0, 42.5],
        lons=[-70.5, -70.0],
        fills={"swh": 1.5, "pp1d": 8.0, "mwd": 90.0, "wdw": 30.0},
    )
    points = gpd.GeoDataFrame(
        {"name": ["0001", "0002"], "geometry": [Point(-70.0, 42.0), Point(-70.5, 42.5)]},
        crs="EPSG:4326",
    )

    result = era5_spectra_to_snapwave_timeseries(ds, points, event_window=(times[0], times[-1]))

    assert set(result.keys()) == {"bhs", "btp", "bwd", "bds"}
    for key in ("bhs", "btp", "bwd", "bds"):
        assert list(result[key].columns) == ["0001", "0002"]
        assert len(result[key]) == 3


def test_era5_spectra_to_snapwave_timeseries_picks_nearest_grid_cell():
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    # build a 2x2 grid with distinct swh values per cell so we can tell which cell was picked
    swh = np.array(
        [[[1.0, 2.0],   # lat=42.0:  lon=-70.5 -> 1.0,  lon=-70.0 -> 2.0
          [3.0, 4.0]],  # lat=42.5:  lon=-70.5 -> 3.0,  lon=-70.0 -> 4.0
         [[1.0, 2.0],
          [3.0, 4.0]]]
    )
    ds = xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "pp1d": (("valid_time", "latitude", "longitude"), np.zeros_like(swh)),
            "mwd": (("valid_time", "latitude", "longitude"), np.zeros_like(swh)),
            "wdw": (("valid_time", "latitude", "longitude"), np.zeros_like(swh)),
        },
        coords={"valid_time": times, "latitude": [42.0, 42.5], "longitude": [-70.5, -70.0]},
    )
    # point 0001 is closest to (lat=42.5, lon=-70.0) -> swh=4.0
    # point 0002 is closest to (lat=42.0, lon=-70.5) -> swh=1.0
    points = gpd.GeoDataFrame(
        {"name": ["0001", "0002"], "geometry": [Point(-70.05, 42.49), Point(-70.49, 42.01)]},
        crs="EPSG:4326",
    )

    result = era5_spectra_to_snapwave_timeseries(ds, points, event_window=(times[0], times[-1]))

    assert result["bhs"]["0001"].iloc[0] == pytest.approx(4.0)
    assert result["bhs"]["0002"].iloc[0] == pytest.approx(1.0)


def test_era5_spectra_to_snapwave_timeseries_falls_back_to_nearest_finite_cell():
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    swh = np.array(
        [
            [[np.nan, 2.0], [3.0, 4.0]],
            [[np.nan, 2.5], [3.5, 4.5]],
        ]
    )
    ds = xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "pp1d": (("valid_time", "latitude", "longitude"), np.where(np.isnan(swh), np.nan, 8.0)),
            "mwd": (("valid_time", "latitude", "longitude"), np.where(np.isnan(swh), np.nan, 90.0)),
            "wdw": (("valid_time", "latitude", "longitude"), np.where(np.isnan(swh), np.nan, 0.5)),
        },
        coords={"valid_time": times, "latitude": [42.0, 42.5], "longitude": [-70.5, -70.0]},
    )
    points = gpd.GeoDataFrame(
        {"name": ["0001"], "geometry": [Point(-70.49, 42.01)]},
        crs="EPSG:4326",
    )

    result = era5_spectra_to_snapwave_timeseries(ds, points, event_window=(times[0], times[-1]))

    assert result["bhs"]["0001"].tolist() == pytest.approx([2.0, 2.5])


def test_era5_spectra_to_snapwave_timeseries_slices_event_window():
    # 10-hour ERA5 record; ask for a 3-hour slice in the middle
    times = pd.date_range("2020-01-01", periods=10, freq="h")
    ds = _toy_era5(
        times=times,
        lats=[42.0],
        lons=[-70.0],
        fills={"swh": 1.0, "pp1d": 1.0, "mwd": 1.0, "wdw": 1.0},
    )
    points = gpd.GeoDataFrame(
        {"name": ["0001"], "geometry": [Point(-70.0, 42.0)]},
        crs="EPSG:4326",
    )

    result = era5_spectra_to_snapwave_timeseries(
        ds, points, event_window=(times[3], times[5])
    )

    assert len(result["bhs"]) == 3
    assert list(result["bhs"].index) == list(times[3:6])


def test_era5_spectra_to_snapwave_timeseries_raises_on_missing_variable():
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    # mwd is intentionally absent so the helper fails before writing SnapWave input.
    ds = _toy_era5(
        times=times,
        lats=[42.0],
        lons=[-70.0],
        fills={"swh": 1.0, "pp1d": 1.0, "wdw": 1.0},
    )
    points = gpd.GeoDataFrame(
        {"name": ["0001"], "geometry": [Point(-70.0, 42.0)]},
        crs="EPSG:4326",
    )

    with pytest.raises(KeyError, match="mwd"):
        era5_spectra_to_snapwave_timeseries(ds, points, event_window=(times[0], times[-1]))


def test_write_runup_gauges_file_writes_single_transect(tmp_path):
    path = tmp_path / "sfincs.rug"
    transects = [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))]

    write_runup_gauges_file(path, transects)

    lines = path.read_text().splitlines()
    assert lines[0] == "beach_north"
    assert lines[1] == "2 2"
    x0, y0 = lines[2].split()
    x1, y1 = lines[3].split()
    assert float(x0) == pytest.approx(354500.0)
    assert float(y0) == pytest.approx(4657800.0)
    assert float(x1) == pytest.approx(354520.0)
    assert float(y1) == pytest.approx(4657820.0)


def test_write_runup_gauges_file_concatenates_multiple_transects(tmp_path):
    path = tmp_path / "sfincs.rug"
    transects = [
        ("north", (100.123456, 200.654321), (110.0, 210.0)),
        ("south", (300.0, 400.0), (310.987654, 420.123456)),
    ]

    write_runup_gauges_file(path, transects)

    lines = path.read_text().splitlines()
    assert len(lines) == 8
    assert lines[0] == "north"
    assert lines[4] == "south"
    # precision preserved to ~6 decimals (sub-micron in UTM — plenty)
    x0, y0 = lines[2].split()
    assert float(x0) == pytest.approx(100.123456, abs=1e-6)
    assert float(y0) == pytest.approx(200.654321, abs=1e-6)
    x1, y1 = lines[7].split()
    assert float(x1) == pytest.approx(310.987654, abs=1e-6)
    assert float(y1) == pytest.approx(420.123456, abs=1e-6)


def test_validate_runup_transects_rejects_lines_outside_model_region():
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    transects = [
        ("inside", (10.0, 10.0), (90.0, 90.0)),
        ("outside", (150.0, 10.0), (180.0, 10.0)),
    ]

    with pytest.raises(ValueError, match="outside"):
        validate_runup_transects(transects, region)


def test_validate_runup_transects_accepts_structure_adjacent_region_crossing_line():
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    transects = [("cross_shore", (-10.0, 50.0), (80.0, 50.0))]

    validate_runup_transects(transects, region)


def test_inject_runup_config_appends_to_inp_lacking_keys(tmp_path):
    path = tmp_path / "sfincs.inp"
    path.write_text("mmax = 544\nnmax = 435\n")
    write_runup_gauges_file(
        tmp_path / "sfincs.rug",
        [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))],
    )

    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)

    text = path.read_text()
    assert "mmax = 544" in text  # original lines preserved
    assert "rugfile              = sfincs.rug" in text
    assert "rugdepth             = 0.05" in text
    assert "obsfile              = sfincs.rug.obs" in text
    assert "runupfile" not in text
    assert (tmp_path / "sfincs.rug.obs").read_text() == (
        "354500.000000 4657800.000000 'runup_gauge_workaround'\n"
    )


def test_inject_runup_config_is_idempotent(tmp_path):
    path = tmp_path / "sfincs.inp"
    path.write_text("mmax = 544\nnmax = 435\n")
    write_runup_gauges_file(
        tmp_path / "sfincs.rug",
        [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))],
    )

    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)
    once = path.read_text()
    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)
    twice = path.read_text()

    assert once == twice
    # exactly one official rugfile/rugdepth/obsfile set — no duplication
    assert twice.count("runupfile") == 0
    assert twice.count("rugfile") == 1
    assert twice.count("rugdepth") == 1
    assert twice.count("obsfile") == 1


def test_inject_runup_config_updates_existing_keys_in_place(tmp_path):
    path = tmp_path / "sfincs.inp"
    # rugfile is in the middle of the file with a stale value
    path.write_text(
        "mmax = 544\n"
        "rugfile = stale.rug\n"
        "rugdepth = 0.01\n"
        "nmax = 435\n"
    )
    write_runup_gauges_file(
        tmp_path / "sfincs.rug",
        [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))],
    )

    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)

    lines = path.read_text().splitlines()
    # surrounding lines preserved in original positions
    assert lines[0] == "mmax = 544"
    assert lines[3] == "nmax = 435"
    assert lines[1] == "rugfile              = sfincs.rug"
    assert lines[2] == "rugdepth             = 0.05"
    assert lines[4] == "obsfile              = sfincs.rug.obs"


def test_inject_runup_config_deduplicates_legacy_runupfile_key(tmp_path):
    path = tmp_path / "sfincs.inp"
    path.write_text(
        "rugfile = stale.rug\n"
        "runupfile = stale.rug\n"
        "rugdepth = 0.01\n"
    )
    write_runup_gauges_file(
        tmp_path / "sfincs.rug",
        [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))],
    )

    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)

    text = path.read_text()
    assert text.count("runupfile") == 0
    assert text.count("rugfile") == 1
    assert "rugfile              = sfincs.rug" in text
    assert "rugdepth             = 0.05" in text


def test_inject_runup_config_preserves_existing_obsfile(tmp_path):
    path = tmp_path / "sfincs.inp"
    path.write_text("obsfile = sfincs.obs\n")
    write_runup_gauges_file(
        tmp_path / "sfincs.rug",
        [("beach_north", (354500.0, 4657800.0), (354520.0, 4657820.0))],
    )

    inject_runup_config(path, rugfile="sfincs.rug", rugdepth=0.05)

    text = path.read_text()
    assert "obsfile = sfincs.obs" in text
    assert "sfincs.rug.obs" not in text
    assert not (tmp_path / "sfincs.rug.obs").exists()


def test_repair_snapwave_directional_spreading_file_uses_bhs_time_axis(tmp_path):
    (tmp_path / "snapwave.bhs").write_text(
        "0.000 1.000 1.000\n"
        "86400.000 1.000 1.000\n"
    )
    # hydromt-sfincs 2.0.0rc1 writes this malformed file by reusing bwd data.
    (tmp_path / "snapwave.bds").write_text(
        "-1779753600.000 270.000 270.000\n"
        "-1779753600.000 270.000 270.000\n"
    )

    repair_snapwave_directional_spreading_file(tmp_path, spread_degrees=20.0)

    assert (tmp_path / "snapwave.bds").read_text() == (
        "0.000 20.000 20.000\n"
        "86400.000 20.000 20.000\n"
    )
