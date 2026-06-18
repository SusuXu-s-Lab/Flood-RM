"""Visualization helpers for built and run HydroMT-Wflow models.

`plot_wflow_basemap` renders the *static* model (a built model has no time axis):
elevation + river network + basins + gauges, with an optional SFINCS domain
overlay.
`animate_wflow_output` renders a *dynamic* gridded output variable (discharge,
water level, ...) over time and is only meaningful after the model has been run.

Both helpers are duck-typed: `plot_wflow_basemap` needs a model exposing
``staticmaps.data`` (xarray.Dataset) and ``rivers`` / ``basins`` GeoDataFrames,
so it can be exercised without instantiating a full ``WflowSbmModel``.
"""

from __future__ import annotations

from pathlib import Path


def plot_wflow_basemap(
    model,
    *,
    gages=None,
    sfincs_domains=None,
    ax=None,
    elevation_var: str = "land_elevation",
    subcatchment_var: str = "subcatchment",
    river_mask_var: str = "river_mask",
    streamorder_field: str = "strord",
    title: str | None = None,
    figsize=(8, 6),
    diagnostic: bool = False,
    streamorder_levels: tuple[int, ...] = (1, 2),
):
    """Plot a built HydroMT-Wflow model as a reference-style base map.

    Parameters
    ----------
    model
        A built Wflow model exposing ``staticmaps.data`` (xarray.Dataset) and
        ``rivers`` / ``basins`` GeoDataFrame properties.
    gages : geopandas.GeoDataFrame, optional
        Fallback point geometries to overlay when the model does not expose
        a ``gauges*`` geometry layer.
    sfincs_domains : geopandas.GeoDataFrame, optional
        SFINCS domain polygons to overlay as dashed red boundaries.
    ax : matplotlib Axes, optional
        Axis to draw into. A new figure is created when omitted.
    elevation_var, subcatchment_var, river_mask_var : str
        Staticmap variable names (HydroMT-Wflow v1 defaults).
    streamorder_field : str
        River GeoDataFrame column used to scale line width.
    streamorder_levels : tuple[int, ...]
        Gridded ``meta_streamorder`` classes to overlay for stream-order QA.
    title : str, optional
        Title for the left (elevation) panel.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.lines import Line2D

    if diagnostic:
        return _plot_wflow_diagnostic_basemap(
            model,
            gages=gages,
            elevation_var=elevation_var,
            subcatchment_var=subcatchment_var,
            river_mask_var=river_mask_var,
            streamorder_field=streamorder_field,
            title=title,
        )

    staticmaps = model.staticmaps.data
    if elevation_var not in staticmaps:
        raise KeyError(
            f"staticmaps has no {elevation_var!r}; available: {sorted(staticmaps.data_vars)}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    elevation = staticmaps[elevation_var].raster.mask_nodata()
    elevation.attrs.update(long_name="elevation", units="m")
    vmin, vmax = _finite_quantiles(elevation, (0.0, 0.98))
    cmap = colors.LinearSegmentedColormap.from_list(
        "wflow_dem",
        plt.cm.terrain(np.linspace(0.25, 1.0, 256)),
    )
    elevation.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs=dict(aspect=30, shrink=0.8, label="elevation [m]"),
    )

    map_crs = staticmaps.raster.crs
    legend_handles = []
    if river_mask_var in staticmaps:
        staticmaps[river_mask_var].where(staticmaps[river_mask_var] > 0).plot(
            ax=ax,
            cmap=colors.ListedColormap(["blue"]),
            add_colorbar=False,
            zorder=3,
        )
        legend_handles.append(Line2D([0], [0], color="blue", linewidth=2.0, label="river cells"))

    streamorder_handles = _plot_streamorder_levels(
        staticmaps,
        ax,
        levels=streamorder_levels,
        streamorder_var="meta_streamorder",
    )

    rivers = _safe_property(model, "rivers")
    if rivers is not None and not rivers.empty:
        rivers = _to_map_crs(rivers, map_crs)
        if streamorder_field in rivers:
            order = rivers[streamorder_field].astype(float)
            linewidth = order / max(float(order.max()), 1.0) * 1.0 + 0.5
        else:
            linewidth = 1.0
        rivers.plot(ax=ax, color="blue", linewidth=linewidth, zorder=4, label="river")
        legend_handles.append(Line2D([0], [0], color="blue", linewidth=2.0, label="river vector"))

    basins = _safe_property(model, "basins")
    if basins is not None and not basins.empty:
        _to_map_crs(basins, map_crs).boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=5)
        legend_handles.append(Line2D([0], [0], color="black", linewidth=0.8, label="Wflow basin"))

    # Distinct styling per gauge layer so the coupled SFINCS-source gauges and the reviewed
    # USGS gages are visually separable. Styles mirror the SFINCS basemap (crimson diamonds
    # for Wflow->SFINCS sources, open black circles for reviewed USGS gages).
    gauge_styles = {
        "gauges_sfincs": dict(marker="D", markersize=45, facecolor="crimson", edgecolor="black", linewidth=0.6),
        "gauges_usgs": dict(marker="o", markersize=35, facecolor="none", edgecolor="black", linewidth=1.0),
    }
    gauge_labels = {
        "gauges_sfincs": "Wflow gauge at SFINCS source",
        "gauges_usgs": "reviewed USGS gage",
    }
    default_gauge_style = dict(marker="d", markersize=35, facecolor="red", edgecolor="red", linewidth=0.6)
    gauge_layers = _gauge_layers(model, gages)
    for name, layer in gauge_layers:
        style = gauge_styles.get(str(name), default_gauge_style)
        label = gauge_labels.get(str(name), str(name).replace("_", " "))
        layer = _to_map_crs(layer, map_crs)
        layer.plot(ax=ax, zorder=6, **style)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor="none" if style["facecolor"] == "none" else style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markersize=7,
                label=label,
            )
        )

    if sfincs_domains is not None and not sfincs_domains.empty:
        _to_map_crs(sfincs_domains, map_crs).boundary.plot(
            ax=ax,
            color="red",
            linewidth=1.0,
            linestyle="--",
            zorder=7,
        )
        legend_handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="SFINCS domain"))

    xlim, ylim = _dataarray_bounds(elevation)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(_x_label(map_crs))
    ax.set_ylabel(_y_label(map_crs))
    ax.set_title(title or "wflow base map")
    _set_unique_legend(
        ax,
        extra_patches=[*legend_handles, *streamorder_handles, *_waterbody_patches(model, map_crs, ax)],
    )
    return fig, ax


def plot_wflow_ldd_components(
    model,
    *,
    streamorder_levels: tuple[int, ...] = (1, 2),
    figsize=(13, 10),
):
    """Plot HydroMT-Wflow LDD QA components for a built model.

    The layout follows the intent of the HydroMT-Wflow prepare-LDD example:
    inspect elevation, upstream area, stream order, and local drain direction
    together before trusting a routed Wflow-SFINCS handoff.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.lines import Line2D

    staticmaps = model.staticmaps.data
    required = ("land_elevation", "meta_upstream_area", "meta_streamorder", "local_drain_direction")
    missing = [name for name in required if name not in staticmaps]
    if missing:
        raise KeyError(f"staticmaps missing LDD QA variables: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    elevation = staticmaps["land_elevation"].raster.mask_nodata()
    cmap = colors.LinearSegmentedColormap.from_list(
        "wflow_dem",
        plt.cm.terrain(np.linspace(0.25, 1.0, 256)),
    )
    vmin, vmax = _finite_quantiles(elevation, (0.0, 0.98))
    elevation.plot(
        ax=axes[0, 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs=dict(shrink=0.75, label="elevation [m]"),
    )
    axes[0, 0].set_title("DEM + selected stream order")
    handles = _plot_streamorder_levels(
        staticmaps,
        axes[0, 0],
        levels=streamorder_levels,
        streamorder_var="meta_streamorder",
    )
    if handles:
        axes[0, 0].legend(handles=handles, title="Stream order", loc="lower right")

    uparea = _mask_positive(staticmaps["meta_upstream_area"])
    np.log10(uparea).plot(
        ax=axes[0, 1],
        cmap="viridis",
        cbar_kwargs=dict(shrink=0.75, label="log10 upstream area [km2]"),
    )
    axes[0, 1].set_title("upstream area")

    streamorder = _mask_positive(staticmaps["meta_streamorder"])
    streamorder.plot(
        ax=axes[1, 0],
        cmap="magma",
        cbar_kwargs=dict(shrink=0.75, label="stream order"),
    )
    axes[1, 0].set_title("meta_streamorder")

    ldd = _mask_positive(staticmaps["local_drain_direction"])
    ldd_codes = _present_integer_values(ldd)
    if ldd_codes and max(ldd_codes) <= 9:
        ldd_cmap = colors.ListedColormap(plt.cm.tab10(np.linspace(0, 1, 9)))
        ldd_norm = colors.BoundaryNorm(np.arange(0.5, 10.5, 1), ldd_cmap.N)
        ldd_ticks = range(1, 10)
        ldd_label = "LDD code"
    else:
        ldd_cmap = colors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, 8)))
        ldd_norm = colors.BoundaryNorm([0.5, 1.5, 3, 6, 12, 24, 48, 96, 192], ldd_cmap.N)
        ldd_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
        ldd_label = "D8 code"
    ldd.plot(
        ax=axes[1, 1],
        cmap=ldd_cmap,
        norm=ldd_norm,
        cbar_kwargs=dict(shrink=0.75, label=ldd_label, ticks=list(ldd_ticks)),
    )
    axes[1, 1].set_title("local_drain_direction")

    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(_x_label(staticmaps.raster.crs))
        ax.set_ylabel(_y_label(staticmaps.raster.crs))
    return fig, axes


def _mask_positive(data_array):
    try:
        masked = data_array.raster.mask_nodata()
    except Exception:
        fill_value = data_array.attrs.get("_FillValue")
        masked = data_array.where(data_array != fill_value) if fill_value is not None else data_array
    return masked.where(masked > 0)


def _present_integer_values(data_array) -> list[int]:
    import numpy as np

    values = np.asarray(data_array.values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    return sorted({int(value) for value in values if float(value).is_integer()})


def _plot_wflow_diagnostic_basemap(
    model,
    *,
    gages=None,
    elevation_var: str,
    subcatchment_var: str,
    river_mask_var: str,
    streamorder_field: str,
    title: str | None,
):
    import matplotlib.pyplot as plt

    staticmaps = model.staticmaps.data
    if elevation_var not in staticmaps:
        raise KeyError(
            f"staticmaps has no {elevation_var!r}; available: {sorted(staticmaps.data_vars)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    left, right = axes[0], axes[1]

    elevation = staticmaps[elevation_var].raster.mask_nodata()
    elevation.plot(
        ax=left,
        cmap="terrain",
        cbar_kwargs=dict(shrink=0.7, label="elevation [m]"),
    )
    rivers = _safe_property(model, "rivers")
    if rivers is not None and not rivers.empty:
        if streamorder_field in rivers:
            order = rivers[streamorder_field].astype(float)
            linewidth = order / max(float(order.max()), 1.0) * 1.8 + 0.4
        else:
            linewidth = 0.8
        rivers.plot(ax=left, color="#1f4e79", linewidth=linewidth, zorder=3)
    basins = _safe_property(model, "basins")
    if basins is not None and not basins.empty:
        basins.boundary.plot(ax=left, color="k", linewidth=0.6, zorder=4)
    if gages is not None and not gages.empty:
        gages_p = gages.to_crs(staticmaps.raster.crs) if gages.crs else gages
        gages_p.plot(
            ax=left, marker="*", color="red", markersize=140,
            edgecolor="k", linewidth=0.6, zorder=5,
        )
    left.set_title(title or "Wflow model")
    left.set_aspect("equal", adjustable="box")

    if subcatchment_var in staticmaps:
        staticmaps[subcatchment_var].where(staticmaps[subcatchment_var] > 0).plot(
            ax=right, cmap="tab20", add_colorbar=False, alpha=0.6,
        )
    if river_mask_var in staticmaps:
        staticmaps[river_mask_var].where(staticmaps[river_mask_var] > 0).plot(
            ax=right, cmap="Blues", add_colorbar=False,
        )
    right.set_title("subcatchment + river mask")
    right.set_aspect("equal", adjustable="box")

    return fig, axes


def _plot_streamorder_levels(staticmaps, ax, *, levels, streamorder_var):
    from matplotlib.lines import Line2D

    if not levels or streamorder_var not in staticmaps:
        return []
    colors = {
        1: "#00A6D6",
        2: "#0057FF",
        3: "#0040B8",
        4: "#2E2E8B",
    }
    handles = []
    streamorder = staticmaps[streamorder_var]
    for level in levels:
        color = colors.get(int(level), "#003399")
        streamorder.where(streamorder == int(level)).plot(
            ax=ax,
            cmap=_single_color_cmap(color),
            add_colorbar=False,
            zorder=4 + int(level),
        )
        handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"strord={int(level)}"))
    return handles


def _single_color_cmap(color):
    from matplotlib import colors

    return colors.ListedColormap([color])


def animate_wflow_output(
    output_path,
    variable: str,
    *,
    cmap: str = "Blues",
    quantile: float = 0.98,
    interval_ms: int = 200,
    save_path=None,
    fps: int = 8,
    dpi: int = 120,
):
    """Animate a time-varying gridded Wflow output variable.

    Only meaningful after the model has been run (Section 6 / ``run_wflow_replay``),
    which writes an output dataset with a ``time`` dimension.

    Parameters
    ----------
    output_path : str or Path
        Wflow run output NetCDF (e.g. ``run_default/output.nc``).
    variable : str
        Gridded variable to animate (e.g. ``q_river``, ``h``, ``vwc``).
    cmap, quantile, interval_ms
        Colormap, upper colour clip quantile, and frame interval.
    save_path : str or Path, optional
        If given, save the animation (``.mp4`` needs ffmpeg; ``.gif`` needs pillow).
    fps, dpi
        Encoding options used when ``save_path`` is set.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Wrap with ``IPython.display.HTML(anim.to_jshtml())`` to scrub inline.
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.animation import FuncAnimation

    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Wflow output not found: {output_path}. Run the model (Section 6) first."
        )
    dataset = xr.open_dataset(output_path)
    if variable not in dataset:
        raise KeyError(
            f"{variable!r} not in output; available: {sorted(dataset.data_vars)}"
        )
    data = dataset[variable]
    if "time" not in data.dims:
        raise ValueError(f"{variable!r} has no 'time' dimension; nothing to animate.")

    vmin = float(data.min())
    vmax = float(data.quantile(quantile))
    fig, ax = plt.subplots(figsize=(7, 6))

    def _draw(frame: int):
        ax.clear()
        data.isel(time=frame).plot(
            ax=ax, cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{variable} — {str(data['time'].values[frame])[:16]}")
        ax.set_aspect("equal", adjustable="datalim")

    _draw(0)
    anim = FuncAnimation(fig, _draw, frames=int(data.sizes["time"]), interval=interval_ms)
    if save_path is not None:
        save_path = Path(save_path)
        anim.save(save_path, fps=fps, dpi=dpi)
    return anim


def _safe_property(model, name):
    try:
        return getattr(model, name)
    except Exception:
        return None


def _finite_quantiles(data, quantiles):
    values = data.quantile(list(quantiles)).values
    return tuple(float(value) for value in values)


def _to_map_crs(frame, crs):
    if frame is None or frame.empty or crs is None or frame.crs is None:
        return frame
    return frame.to_crs(crs)


def _gauge_layers(model, fallback_gages):
    layers = []
    for name, layer in _iter_geoms(model):
        if str(name).startswith("gauges") and layer is not None and not layer.empty:
            layers.append((str(name), layer))
    if not layers and fallback_gages is not None and not fallback_gages.empty:
        layers.append(("gauges_sfincs", fallback_gages))
    return layers


def _waterbody_patches(model, crs, ax):
    from matplotlib.patches import Patch

    patches = []
    geoms = dict(_iter_geoms(model))
    styles = {
        "lakes": dict(facecolor="lightblue", edgecolor="black", linewidth=1, label="lakes"),
        "reservoirs": dict(facecolor="white", edgecolor="black", linewidth=1, label="reservoirs"),
        "glaciers": dict(facecolor="grey", edgecolor="grey", linewidth=1, label="glaciers"),
    }
    for name, kwargs in styles.items():
        layer = geoms.get(name)
        if layer is None or layer.empty:
            continue
        _to_map_crs(layer, crs).plot(ax=ax, zorder=4, **kwargs)
        patches.append(Patch(**kwargs))
    return patches


def _iter_geoms(model):
    geoms = _safe_property(model, "geoms") or {}
    data = getattr(geoms, "data", geoms)
    if hasattr(data, "items"):
        return list(data.items())
    return []


def _set_unique_legend(ax, *, extra_patches=()):
    handles, labels = ax.get_legend_handles_labels()
    ordered = []
    seen = set()
    for handle, label in [*zip(handles, labels), *((patch, patch.get_label()) for patch in extra_patches)]:
        if label.startswith("_") or label in seen:
            continue
        seen.add(label)
        ordered.append((handle, label))
    if ordered:
        ax.legend(
            handles=[handle for handle, _ in ordered],
            labels=[label for _, label in ordered],
            title="Legend",
            loc="lower right",
            frameon=True,
            framealpha=0.7,
            edgecolor="black",
            facecolor="white",
        )


def _dataarray_bounds(data):
    x_name = "x" if "x" in data.coords else data.dims[-1]
    y_name = "y" if "y" in data.coords else data.dims[-2]
    xs = data[x_name].values
    ys = data[y_name].values
    dx = _spacing(xs)
    dy = _spacing(ys)
    xmin = float(min(xs[0], xs[-1]) - dx / 2)
    xmax = float(max(xs[0], xs[-1]) + dx / 2)
    ymin = float(min(ys[0], ys[-1]) - dy / 2)
    ymax = float(max(ys[0], ys[-1]) + dy / 2)
    xpad = (xmax - xmin) * 0.02
    ypad = (ymax - ymin) * 0.02
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def _spacing(values):
    if len(values) < 2:
        return 0.0
    return abs(float(values[1]) - float(values[0]))


def _x_label(crs):
    return "longitude [degree east]" if _is_geographic(crs) else "x"


def _y_label(crs):
    return "latitude [degree north]" if _is_geographic(crs) else "y"


def _is_geographic(crs):
    if crs is None:
        return False
    try:
        return bool(crs.is_geographic)
    except AttributeError:
        return str(crs).upper() == "EPSG:4326"
