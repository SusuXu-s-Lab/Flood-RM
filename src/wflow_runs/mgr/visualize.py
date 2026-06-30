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

from wflow_runs.qa import (
    event_peak_discharge_table as _v2_event_peak_discharge_table,
    plot_event_precipitation_peak_discharge as _v2_plot_event_precipitation_peak_discharge,
)


def plot_wflow_basemap(
    model,
    *,
    gages=None,
    handoff_sources=None,
    sfincs_domains=None,
    background_elevation=None,
    ax=None,
    elevation_var: str = "land_elevation",
    subcatchment_var: str = "subcatchment",
    river_mask_var: str | None = None,
    streamorder_field: str = "strord",
    min_river_streamorder: int | None = 2,
    sfincs_domain_min_river_streamorder: int | None = 1,
    gage_min_river_streamorder: int | None = 1,
    gage_river_buffer: float | None = None,
    title: str | None = None,
    figsize=(8, 6),
    diagnostic: bool = False,
    streamorder_levels: tuple[int, ...] | None = None,
    legend_outside: bool = True,
):
    """Plot a built HydroMT-Wflow model as a reference-style base map.

    Parameters
    ----------
    model
        A built Wflow model exposing ``staticmaps.data`` (xarray.Dataset) and
        ``rivers`` / ``basins`` GeoDataFrame properties.
    gages : geopandas.GeoDataFrame, optional
        Reviewed USGS/observation gages to overlay as open circles. This is
        useful when the model geoms only expose the SFINCS handoff gauges.
    handoff_sources : geopandas.GeoDataFrame or path, optional
        Persisted SFINCS handoff source points to overlay as the intended
        Wflow-to-SFINCS source locations. When supplied, the model's
        ``gauges_sfincs`` layer is labelled as post-snap Wflow gauges.
    sfincs_domains : geopandas.GeoDataFrame, optional
        SFINCS domain polygons to overlay as dashed red boundaries.
    background_elevation : xarray.DataArray or iterable of DataArray, optional
        Additional DEM/elevation rasters to draw below the Wflow DEM, useful
        when the SFINCS coverage extends outside the Wflow staticmap footprint.
    ax : matplotlib Axes, optional
        Axis to draw into. A new figure is created when omitted.
    elevation_var, subcatchment_var, river_mask_var : str
        Staticmap variable names (HydroMT-Wflow v1 defaults).
    streamorder_field : str
        River GeoDataFrame column used to scale line width.
    min_river_streamorder : int, optional
        Minimum vector river stream order to draw on the full reference map.
    sfincs_domain_min_river_streamorder : int, optional
        Minimum vector river stream order to draw where rivers intersect SFINCS
        domains. This keeps lower-order handoff/crossing context visible without
        drawing every low-order stream across the full Wflow basin.
    gage_min_river_streamorder : int, optional
        Minimum vector river stream order to draw near reviewed USGS gages.
        This links gage markers to nearby streams without crowding the full map.
    gage_river_buffer : float, optional
        Buffer around reviewed USGS gages in map CRS units. Defaults to a small
        degree buffer for geographic maps and a 2500 m buffer for projected maps.
    streamorder_levels : tuple[int, ...]
        Optional gridded ``meta_streamorder`` classes to overlay for stream-order QA.
    legend_outside : bool
        Place the map legend outside the plotting area so it does not obscure
        SFINCS domains, gauges, rivers, or reservoirs.
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
    from matplotlib.patches import Patch

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
    map_crs = staticmaps.raster.crs
    _plot_background_elevation(
        background_elevation,
        ax=ax,
        dst_crs=map_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    elevation.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs=dict(aspect=30, shrink=0.8, label="elevation [m]"),
    )

    legend_handles = []
    if river_mask_var and river_mask_var in staticmaps:
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

    gage_context = _reviewed_gage_layer(model, gages)
    rivers = _safe_property(model, "rivers")
    if rivers is not None and not rivers.empty:
        rivers, order = _reference_rivers_for_basemap(
            rivers,
            sfincs_domains,
            map_crs,
            gage_context,
            streamorder_field=streamorder_field,
            min_river_streamorder=min_river_streamorder,
            sfincs_domain_min_river_streamorder=sfincs_domain_min_river_streamorder,
            gage_min_river_streamorder=gage_min_river_streamorder,
            gage_river_buffer=gage_river_buffer,
        )
        if order is not None:
            linewidth = order / max(float(order.max()), 1.0) * 1.4 + 0.4
        else:
            linewidth = 1.0
        if not rivers.empty:
            rivers.plot(ax=ax, color="blue", linewidth=linewidth, zorder=4, label="river")
            legend_handles.append(Line2D([0], [0], color="blue", linewidth=1.2, label="river"))

    basins = _safe_property(model, "basins")
    if basins is not None and not basins.empty:
        basins = _to_map_crs(basins, map_crs)
        basins.plot(
            ax=ax,
            facecolor="white",
            edgecolor="black",
            alpha=0.12,
            linewidth=0.4,
            zorder=2,
        )
        basins.boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=5)
        legend_handles.append(Patch(facecolor="white", edgecolor="black", alpha=0.35, label="Wflow basin"))

    # Distinct styling per gauge layer so the coupled SFINCS-source gauges and the reviewed
    # USGS gages are visually separable. Styles mirror the SFINCS basemap (crimson diamonds
    # for Wflow->SFINCS sources, open black circles for reviewed USGS gages).
    source_layer = _handoff_source_layer(handoff_sources, map_crs)
    gauge_styles = {
        "gauges_sfincs": (
            dict(marker="X", markersize=45, facecolor="darkorange", edgecolor="black", linewidth=0.6)
            if source_layer is not None and not source_layer.empty
            else dict(marker="D", markersize=45, facecolor="crimson", edgecolor="black", linewidth=0.6)
        ),
        "gauges_usgs": dict(marker="o", markersize=35, facecolor="none", edgecolor="black", linewidth=1.0),
    }
    gauge_labels = {
        "gauges_sfincs": (
            "Wflow gauge (post-snap)"
            if source_layer is not None and not source_layer.empty
            else "gauges_sfincs"
        ),
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
    if source_layer is not None and not source_layer.empty:
        source_style = dict(marker="D", markersize=48, facecolor="crimson", edgecolor="black", linewidth=0.7)
        source_layer.plot(ax=ax, zorder=8, **source_style)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=source_style["marker"],
                color="none",
                markerfacecolor=source_style["facecolor"],
                markeredgecolor=source_style["edgecolor"],
                markersize=7,
                label="SFINCS source",
            )
        )

    if sfincs_domains is not None and not sfincs_domains.empty:
        sfincs_domains = _to_map_crs(sfincs_domains, map_crs)
        sfincs_domains.plot(
            ax=ax,
            facecolor="#d9d9d9",
            edgecolor="none",
            alpha=0.45,
            zorder=3,
        )
        sfincs_domains.boundary.plot(
            ax=ax,
            color="red",
            linewidth=1.0,
            linestyle="--",
            zorder=7,
        )
        legend_handles.append(Line2D([0], [0], color="red", linewidth=1.0, linestyle="--", label="SFINCS domain"))

    xlim, ylim = _combined_axis_limits(
        elevation,
        (
            frame
            for frame in (basins, sfincs_domains, source_layer)
            if frame is not None and not frame.empty
        ),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(_x_label(map_crs))
    ax.set_ylabel(_y_label(map_crs))
    ax.set_title(title or "wflow base map")
    _set_unique_legend(
        ax,
        extra_patches=[*legend_handles, *streamorder_handles, *_waterbody_patches(model, map_crs, ax)],
        outside=legend_outside,
    )
    return fig, ax


def plot_wflow_ldd_components(
    model,
    *,
    streamorder_levels: tuple[int, ...] | None = None,
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
    axes[0, 0].set_facecolor("white")
    _plot_watershed_boundary(model, staticmaps, axes[0, 0])
    handles = _plot_streamorder_levels(
        staticmaps,
        axes[0, 0],
        levels=_streamorder_overlay_levels(staticmaps["meta_streamorder"], streamorder_levels),
        streamorder_var="meta_streamorder",
    )
    axes[0, 0].set_title("Selected stream order")
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


def event_peak_discharge_table(
    catalog,
    *,
    location_root,
    events_root=None,
    discharge_filename: str = "sfincs_discharge.nc",
):
    """Attach Wflow handoff peak discharge metrics to Event Catalog rows.

    The dynamic Wflow handoff writes one SFINCS-ready discharge GeoDataset per
    event. For catalog-scale diagnostics, the most useful scalar response is
    the peak of the summed handoff hydrograph, with the largest individual
    source peak retained for QA.
    """
    return _v2_event_peak_discharge_table(
        catalog,
        location_root=location_root,
        events_root=events_root,
        discharge_filename=discharge_filename,
    )


def plot_event_precipitation_peak_discharge(
    catalog,
    *,
    location_root,
    events_root=None,
    rainfall_column: str | None = None,
    soil_moisture_column: str | None = None,
    log_y: bool = True,
    ax=None,
):
    """Plot Event Catalog rainfall against Wflow-generated peak discharge."""
    return _v2_plot_event_precipitation_peak_discharge(
        catalog,
        location_root=location_root,
        events_root=events_root,
        rainfall_column=rainfall_column,
        soil_moisture_column=soil_moisture_column,
        log_y=log_y,
        ax=ax,
    )


def plot_wflow_event_handoff(
    discharge_nc,
    *,
    precipitation_nc=None,
    event_label: str | None = None,
    figsize=(13, 4.5),
):
    """Plot one dynamic Wflow-to-SFINCS handoff event."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    discharge_nc = Path(discharge_nc)
    if not discharge_nc.exists():
        raise FileNotFoundError(discharge_nc)

    with xr.open_dataset(discharge_nc) as opened:
        ds = opened.load()
    if "discharge" not in ds:
        raise ValueError(f"{discharge_nc} lacks discharge")
    discharge = ds["discharge"]
    if "time" not in discharge.dims:
        raise ValueError("discharge has no time dimension")

    has_precip = precipitation_nc is not None and Path(precipitation_nc).exists()
    ncols = 3 if has_precip else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    if ncols == 2:
        map_ax, hydro_ax = axes
    else:
        map_ax, precip_ax, hydro_ax = axes

    source_dim = next((dim for dim in discharge.dims if dim != "time"), None)
    if source_dim is None:
        source_peak = discharge.max(dim="time", skipna=True).values.reshape(1)
        x = np.array([0.0])
        y = np.array([0.0])
        names = np.array(["total"])
    else:
        source_peak = discharge.max(dim="time", skipna=True)
        x = np.asarray(ds.coords.get("x", np.arange(source_peak.size)), dtype=float).ravel()
        y = np.asarray(ds.coords.get("y", np.zeros(source_peak.size)), dtype=float).ravel()
        names = np.asarray(ds.coords.get("name", np.arange(source_peak.size)), dtype=str).ravel()
        source_peak = np.asarray(source_peak.values, dtype=float).ravel()

    points = map_ax.scatter(
        x,
        y,
        c=source_peak,
        cmap="plasma",
        s=np.clip(source_peak / max(np.nanmax(source_peak), 1.0) * 180.0, 45.0, 180.0),
        edgecolors="black",
        linewidths=0.55,
    )
    fig.colorbar(points, ax=map_ax, shrink=0.78, label="Peak discharge [m3 s$^{-1}$]")
    for xi, yi, name in zip(x, y, names, strict=False):
        map_ax.annotate(str(name), (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)
    map_ax.set_title("Wflow handoff peaks")
    map_ax.set_aspect("equal", adjustable="datalim")
    map_ax.set_xlabel("x")
    map_ax.set_ylabel("y")

    if has_precip:
        with xr.open_dataset(precipitation_nc) as precip_opened:
            precip_ds = precip_opened.load()
        precip_name = "precip" if "precip" in precip_ds else next(iter(precip_ds.data_vars))
        rainfall = precip_ds[precip_name]
        total_rainfall = rainfall.sum(dim="time", skipna=True) if "time" in rainfall.dims else rainfall
        total_rainfall.plot(
            ax=precip_ax,
            cmap="Blues",
            cbar_kwargs=dict(shrink=0.78, label="Storm total rainfall [mm]"),
        )
        precip_ax.set_title("Staged rainfall total")
        precip_ax.set_aspect("equal", adjustable="datalim")

    times = pd.DatetimeIndex(pd.to_datetime(discharge["time"].values))
    if source_dim is None:
        hydro_ax.plot(times, np.asarray(discharge.values, dtype=float), color="#225ea8", linewidth=1.8)
    else:
        frame = discharge.transpose(source_dim, "time").to_pandas().T
        if len(names) == len(frame.columns):
            frame.columns = names
        for column in frame.columns:
            hydro_ax.plot(times, frame[column].to_numpy(dtype=float), linewidth=1.25, alpha=0.86)
    hydro_ax.set_title("Handoff hydrographs")
    hydro_ax.set_xlabel("time")
    hydro_ax.set_ylabel("discharge [m3 s$^{-1}$]")
    hydro_ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    hydro_ax.tick_params(axis="x", labelrotation=30)

    title = event_label or discharge_nc.parent.name
    fig.suptitle(f"{title} Wflow dynamic handoff", y=1.03)
    return fig, axes


def _first_present(frame, candidates):
    return next((column for column in candidates if column in frame), None)


def _mask_positive(data_array):
    try:
        masked = data_array.raster.mask_nodata()
    except Exception:
        fill_value = data_array.attrs.get("_FillValue")
        masked = data_array.where(data_array != fill_value) if fill_value is not None else data_array
    return masked.where(masked > 0)


def _plot_background_elevation(background_elevation, *, ax, dst_crs, cmap, vmin, vmax) -> None:
    if background_elevation is None:
        return
    if isinstance(background_elevation, (list, tuple)):
        rasters = background_elevation
    else:
        rasters = [background_elevation]

    for raster in rasters:
        if raster is None:
            continue
        try:
            elevation = raster.raster.mask_nodata()
        except Exception:
            elevation = raster
        try:
            if elevation.raster.crs != dst_crs:
                elevation = elevation.raster.reproject(dst_crs=dst_crs, method="bilinear")
        except Exception:
            continue
        elevation.attrs.update(long_name="elevation", units="m")
        elevation.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            zorder=0,
        )


def _present_integer_values(data_array) -> list[int]:
    import numpy as np

    values = np.asarray(data_array.values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    return sorted({int(value) for value in values if float(value).is_integer()})


def _streamorder_overlay_levels(streamorder, requested_levels: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested_levels is not None:
        return tuple(int(level) for level in requested_levels)

    present = _present_integer_values(_mask_positive(streamorder))
    if not present:
        return ()
    largest = max(present)
    first_visible = max(2, largest - 3)
    return tuple(level for level in present if level >= first_visible)


def _reference_rivers_for_basemap(
    rivers,
    sfincs_domains,
    map_crs,
    gage_context=None,
    *,
    streamorder_field: str,
    min_river_streamorder: int | None,
    sfincs_domain_min_river_streamorder: int | None,
    gage_min_river_streamorder: int | None = 1,
    gage_river_buffer: float | None = None,
):
    rivers = _to_map_crs(rivers, map_crs)
    if streamorder_field not in rivers:
        return rivers, None

    order = rivers[streamorder_field].astype(float)
    keep = _streamorder_keep_mask(order, min_river_streamorder)

    if sfincs_domains is not None and not sfincs_domains.empty and sfincs_domain_min_river_streamorder is not None:
        domain_keep = _streamorder_keep_mask(order, sfincs_domain_min_river_streamorder)
        if domain_keep.any():
            domains = _to_map_crs(sfincs_domains, map_crs)
            if domains is not None and not domains.empty:
                domain_geometry = _union_geometries(domains.geometry)
                keep = keep | (domain_keep & rivers.geometry.intersects(domain_geometry))

    if gage_context is not None and not gage_context.empty and gage_min_river_streamorder is not None:
        gage_keep = _streamorder_keep_mask(order, gage_min_river_streamorder)
        if gage_keep.any():
            gages = _to_map_crs(gage_context, map_crs)
            if gages is not None and not gages.empty:
                distance = _gage_river_buffer_distance(map_crs, gage_river_buffer)
                gage_geometry = _union_geometries([geometry.buffer(distance) for geometry in gages.geometry])
                keep = keep | (gage_keep & rivers.geometry.intersects(gage_geometry))

    selected = rivers.loc[keep].copy()
    return selected, order.loc[selected.index]


def _reviewed_gage_layer(model, fallback_gages):
    for name, layer in _iter_geoms(model):
        if str(name) == "gauges_usgs" and layer is not None and not layer.empty:
            return layer
    return fallback_gages


def _gage_river_buffer_distance(crs, configured):
    if configured is not None:
        return float(configured)
    is_geographic = getattr(crs, "is_geographic", None)
    if is_geographic is None and crs is not None:
        try:
            from pyproj import CRS

            is_geographic = CRS.from_user_input(crs).is_geographic
        except Exception:
            is_geographic = False
    if bool(is_geographic):
        return 0.025
    return 2500.0


def _streamorder_keep_mask(order, minimum):
    if minimum is None:
        return order.notna()
    return order >= float(minimum)


def _union_geometries(geometries):
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    if hasattr(geometries, "unary_union"):
        return geometries.unary_union
    from shapely.ops import unary_union

    return unary_union(list(geometries))


def _plot_watershed_boundary(model, staticmaps, ax) -> None:
    basins = _safe_property(model, "basins")
    if basins is not None and not basins.empty:
        boundary = basins.to_crs(staticmaps.raster.crs) if basins.crs else basins
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=20)
        return

    if "subcatchment" not in staticmaps:
        return
    subcatchment = _mask_positive(staticmaps["subcatchment"])
    try:
        subcatchment.where(subcatchment > 0).plot.contour(
            ax=ax,
            levels=[0.5],
            colors="black",
            linewidths=0.8,
            add_colorbar=False,
            zorder=20,
        )
    except Exception:
        pass


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


def _handoff_source_layer(handoff_sources, map_crs):
    if handoff_sources is None or (isinstance(handoff_sources, str) and handoff_sources == ""):
        return None
    import geopandas as gpd

    layer = gpd.read_file(handoff_sources) if not hasattr(handoff_sources, "geometry") else handoff_sources.copy()
    if layer.empty:
        return layer
    if layer.crs is None and map_crs is not None:
        layer = layer.set_crs(map_crs)
    return _to_map_crs(layer, map_crs)


def _gauge_layers(model, fallback_gages):
    layers = []
    names = set()
    for name, layer in _iter_geoms(model):
        if str(name).startswith("gauges") and layer is not None and not layer.empty:
            layer_name = str(name)
            layers.append((layer_name, layer))
            names.add(layer_name)
    if fallback_gages is not None and not fallback_gages.empty and "gauges_usgs" not in names:
        layers.append(("gauges_usgs", fallback_gages))
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


def _set_unique_legend(ax, *, extra_patches=(), outside: bool = True):
    handles, labels = ax.get_legend_handles_labels()
    ordered = []
    seen = set()
    for handle, label in [*zip(handles, labels), *((patch, patch.get_label()) for patch in extra_patches)]:
        if label.startswith("_") or label in seen:
            continue
        seen.add(label)
        ordered.append((handle, label))
    if ordered:
        kwargs = dict(
            handles=[handle for handle, _ in ordered],
            labels=[label for _, label in ordered],
            title="Legend",
            frameon=True,
            framealpha=0.86,
            edgecolor="black",
            facecolor="white",
        )
        if outside:
            kwargs.update(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=min(3, len(ordered)),
                borderaxespad=0.0,
                columnspacing=1.25,
                handlelength=1.8,
            )
        else:
            kwargs.update(loc="lower right")
        ax.legend(**kwargs)


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


def _combined_axis_limits(data, frames=()):
    xlim, ylim = _dataarray_bounds(data)
    xmin, xmax = xlim
    ymin, ymax = ylim
    crs = data.raster.crs
    for frame in frames:
        frame = _to_map_crs(frame, crs)
        if frame is None or frame.empty:
            continue
        fxmin, fymin, fxmax, fymax = frame.total_bounds
        xmin = min(xmin, float(fxmin))
        ymin = min(ymin, float(fymin))
        xmax = max(xmax, float(fxmax))
        ymax = max(ymax, float(fymax))

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
