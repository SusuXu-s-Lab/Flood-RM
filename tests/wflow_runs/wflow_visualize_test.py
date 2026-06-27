import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import hydromt  # noqa: F401  (registers the .raster accessor)
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, Polygon

import pandas as pd

from wflow_runs import (
    animate_wflow_output,
    event_peak_discharge_table,
    plot_event_precipitation_peak_discharge,
    plot_wflow_basemap,
    plot_wflow_event_handoff,
    plot_wflow_ldd_components,
)
from wflow_runs.visualize import _reference_rivers_for_basemap


class _Staticmaps:
    def __init__(self, data):
        self.data = data


class _FakeModel:
    def __init__(self, data, rivers, basins, geoms=None):
        self.staticmaps = _Staticmaps(data)
        self._rivers = rivers
        self._basins = basins
        self.geoms = geoms or {}

    @property
    def rivers(self):
        return self._rivers

    @property
    def basins(self):
        return self._basins


def _staticmaps():
    ys = np.linspace(36.10, 36.00, 6)
    xs = np.linspace(-79.80, -79.70, 6)
    rng = np.random.default_rng(0)
    elev = rng.uniform(150, 320, size=(6, 6)).astype("float32")
    sub = np.ones((6, 6), dtype="int32")
    riv = np.zeros((6, 6), dtype="int32")
    riv[3, :] = 1
    ds = xr.Dataset(
        {
            "land_elevation": (("y", "x"), elev),
            "subcatchment": (("y", "x"), sub),
            "river_mask": (("y", "x"), riv),
        },
        coords={"y": ys, "x": xs},
    )
    ds.raster.set_crs("EPSG:4326")
    return ds


def _fake_model():
    rivers = gpd.GeoDataFrame(
        {"strord": [3, 5]},
        geometry=[
            LineString([(-79.79, 36.05), (-79.75, 36.05)]),
            LineString([(-79.75, 36.05), (-79.71, 36.05)]),
        ],
        crs="EPSG:4326",
    )
    basins = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Polygon([(-79.80, 36.00), (-79.70, 36.00), (-79.70, 36.10), (-79.80, 36.10)])],
        crs="EPSG:4326",
    )
    gauges = gpd.GeoDataFrame(
        {"site_no": ["02095000"]}, geometry=[Point(-79.74, 36.05)], crs="EPSG:4326"
    )
    return _FakeModel(_staticmaps(), rivers, basins, geoms={"gauges_sfincs": gauges})


def test_plot_wflow_basemap_returns_reference_style_panel_with_gauges_and_sfincs_domain():
    model = _fake_model()
    sfincs_domain = gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_west"]},
        geometry=[Polygon([(-79.76, 36.03), (-79.72, 36.03), (-79.72, 36.08), (-79.76, 36.08)])],
        crs="EPSG:4326",
    )

    fig, ax = plot_wflow_basemap(model, sfincs_domains=sfincs_domain, title="south_buffalo")

    assert ax.get_title() == "south_buffalo"
    assert ax.get_xlabel() == "longitude [degree east]"
    assert ax.get_ylabel() == "latitude [degree north]"
    assert len(ax.collections) >= 4
    assert ax.get_legend() is not None
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert "SFINCS domain" in labels
    assert "gauges_sfincs" in labels
    assert "river" in labels
    assert "river cells" not in labels
    assert "strord=1" not in labels
    assert "strord=2" not in labels
    sfincs_handle = labels.index("SFINCS domain")
    assert legend.legend_handles[sfincs_handle].get_color() == "red"


def test_plot_wflow_basemap_can_return_diagnostic_two_panel_view():
    model = _fake_model()

    fig, axes = plot_wflow_basemap(model, title="south_buffalo", diagnostic=True)

    assert len(axes) == 2
    assert axes[0].get_title() == "south_buffalo"
    assert axes[1].get_title() == "subcatchment + river mask"


def test_reference_basemap_rivers_add_low_order_streams_only_inside_sfincs_domains():
    rivers = gpd.GeoDataFrame(
        {"river_id": ["inside_order1", "outside_order1", "inside_order2", "outside_order3"], "strord": [1, 1, 2, 3]},
        geometry=[
            LineString([(-79.76, 36.04), (-79.74, 36.06)]),
            LineString([(-79.79, 36.09), (-79.78, 36.10)]),
            LineString([(-79.75, 36.05), (-79.73, 36.06)]),
            LineString([(-79.79, 36.01), (-79.77, 36.02)]),
        ],
        crs="EPSG:4326",
    )
    sfincs_domain = gpd.GeoDataFrame(
        {"sfincs_domain_id": ["greensboro_west"]},
        geometry=[Polygon([(-79.77, 36.03), (-79.72, 36.03), (-79.72, 36.08), (-79.77, 36.08)])],
        crs="EPSG:4326",
    )

    selected, order = _reference_rivers_for_basemap(
        rivers,
        sfincs_domain,
        "EPSG:4326",
        streamorder_field="strord",
        min_river_streamorder=2,
        sfincs_domain_min_river_streamorder=1,
    )

    assert selected["river_id"].tolist() == ["inside_order1", "inside_order2", "outside_order3"]
    assert order.tolist() == [1.0, 2.0, 3.0]


def test_plot_wflow_basemap_raises_on_missing_elevation():
    model = _fake_model()
    model.staticmaps.data = model.staticmaps.data.drop_vars("land_elevation")
    with pytest.raises(KeyError, match="land_elevation"):
        plot_wflow_basemap(model)


def test_plot_wflow_ldd_components_masks_fill_values_and_labels_ldd_codes():
    ys = np.linspace(36.10, 36.00, 4)
    xs = np.linspace(-79.80, -79.70, 4)
    ds = xr.Dataset(
        {
            "land_elevation": (("y", "x"), np.arange(16, dtype="float32").reshape(4, 4) + 200),
            "meta_upstream_area": (
                ("y", "x"),
                np.array(
                    [
                        [-9999.0, -9999.0, -9999.0, -9999.0],
                        [0.01, 0.02, 0.03, -9999.0],
                        [0.04, 0.05, 0.06, -9999.0],
                        [0.07, 0.08, 0.09, -9999.0],
                    ],
                    dtype="float32",
                ),
            ),
            "meta_streamorder": (
                ("y", "x"),
                np.array([[255, 255, 255, 255], [1, 1, 2, 255], [1, 3, 4, 255], [2, 3, 4, 255]], dtype="uint8"),
            ),
            "local_drain_direction": (
                ("y", "x"),
                np.array([[255, 255, 255, 255], [1, 2, 3, 255], [4, 5, 6, 255], [7, 8, 9, 255]], dtype="uint8"),
            ),
        },
        coords={"y": ys, "x": xs},
    )
    for name, fill_value in {
        "meta_upstream_area": -9999.0,
        "meta_streamorder": 255,
        "local_drain_direction": 255,
    }.items():
        ds[name].attrs["_FillValue"] = fill_value
    ds.raster.set_crs("EPSG:4326")
    model = _FakeModel(ds, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))

    fig, axes = plot_wflow_ldd_components(model)

    assert axes[0, 0].get_title() == "Selected stream order"
    assert "elevation [m]" not in [axis.get_ylabel() for axis in fig.axes]
    assert len(axes[0, 0].collections) >= 2
    assert axes[1, 1].get_title() == "local_drain_direction"
    legend = axes[0, 0].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert "strord=1" not in labels
    assert "strord=4" in labels
    ldd_colorbar = fig.axes[-1]
    assert ldd_colorbar.get_ylabel() == "LDD code"
    assert 255 not in [tick for tick in ldd_colorbar.get_yticks()]


def test_animate_wflow_output_builds_animation_over_time(tmp_path):
    times = np.arange("2016-01-01", "2016-01-05", dtype="datetime64[D]")
    data = np.random.default_rng(1).uniform(0, 10, size=(len(times), 4, 4))
    ds = xr.Dataset(
        {"q_river": (("time", "y", "x"), data)},
        coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
    )
    out = tmp_path / "output.nc"
    ds.to_netcdf(out)

    from matplotlib.animation import FuncAnimation

    anim = animate_wflow_output(out, "q_river")

    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == len(times)  # one frame per timestep


def test_animate_wflow_output_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the model"):
        animate_wflow_output(tmp_path / "missing.nc", "q_river")

    ds = xr.Dataset(
        {"static_only": (("y", "x"), np.zeros((3, 3)))},
        coords={"y": np.arange(3), "x": np.arange(3)},
    )
    out = tmp_path / "static.nc"
    ds.to_netcdf(out)
    with pytest.raises(KeyError, match="q_river"):
        animate_wflow_output(out, "q_river")
    with pytest.raises(ValueError, match="no 'time' dimension"):
        animate_wflow_output(out, "static_only")


def test_wflow_event_response_plots_catalog_and_handoff_outputs(tmp_path):
    event_dir = tmp_path / "data/wflow/events/design_0001"
    event_dir.mkdir(parents=True)
    times = pd.date_range("2020-01-01", periods=4, freq="1h")
    discharge = xr.Dataset(
        {
            "discharge": (
                ("index", "time"),
                np.array([[1.0, 3.0, 2.0, 1.5], [0.5, 1.0, 2.5, 1.0]], dtype="float32"),
            )
        },
        coords={
            "index": [1, 2],
            "time": times,
            "name": ("index", ["src_1", "src_2"]),
            "x": ("index", [100.0, 110.0]),
            "y": ("index", [40.0, 45.0]),
        },
    )
    discharge_nc = event_dir / "sfincs_discharge.nc"
    discharge.to_netcdf(discharge_nc)
    precip_nc = event_dir / "precip.nc"
    xr.Dataset(
        {"precip": (("time", "y", "x"), np.ones((4, 2, 2), dtype="float32"))},
        coords={"time": times, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(precip_nc)
    catalog = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "rainfall_mm": [75.0],
            "soil_moisture_metric": [0.31],
            "wflow_event_dir": ["data/wflow/events/design_0001"],
        }
    )

    table = event_peak_discharge_table(catalog, location_root=tmp_path)
    assert table.loc[0, "wflow_discharge_exists"]
    assert table.loc[0, "peak_discharge_m3s"] == pytest.approx(4.5)
    assert table.loc[0, "max_source_peak_discharge_m3s"] == pytest.approx(3.0)

    fig, ax, frame = plot_event_precipitation_peak_discharge(catalog, location_root=tmp_path)
    assert ax.get_xlabel() == "Event precipitation [mm]"
    assert frame.loc[0, "peak_discharge_m3s"] == pytest.approx(4.5)

    fig, axes = plot_wflow_event_handoff(discharge_nc, precipitation_nc=precip_nc, event_label="design_0001")
    assert axes[0].get_title() == "Wflow handoff peaks"
    assert axes[-1].get_title() == "Handoff hydrographs"
