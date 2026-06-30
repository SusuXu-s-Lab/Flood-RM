from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sfincs_runs.snapwave import unwrap_direction_degrees

def derive_seaward_boundary(
    domain: BaseGeometry,
    land: BaseGeometry,
    *,
    ocean_seed: BaseGeometry | None = None,
    tolerance_m: float = 1.0,
    edge_selection_margin_m: float = 1.0,
) -> tuple[BaseGeometry, BaseGeometry]:
    """Carve the seaward (open-ocean-facing) boundary out of a SFINCS domain.

    Subtracts the land polygon from the SFINCS domain to get the offshore
    portion, then drops the parts of its exterior that are shared with the
    coastline. What remains is the open-ocean-facing edge — the right place to
    anchor the SFINCS waterlevel boundary and the SnapWave wave boundary.

    Parameters
    ----------
    domain : shapely geometry
        SFINCS domain polygon (usually the buffered AOI bbox) in the model CRS.
    land : shapely geometry
        Land polygon in the model CRS.
    tolerance_m : float
        Buffer applied to the land boundary when subtracting from the offshore
        exterior. Should be small (~1 m) to clip away the shared coastline
        without eating into the genuinely seaward edge.
    ocean_seed : shapely geometry, optional
        Point in the offshore water. When provided, only the open-domain edge
        nearest this point is retained. This prevents lateral bbox sides from
        becoming water-level and SnapWave boundaries.
    edge_selection_margin_m : float
        Additional distance tolerance when selecting the nearest open-domain
        edge to `ocean_seed`.

    Returns
    -------
    offshore_polygon, seaward_edge
        Both in the same CRS as the inputs.
    """
    offshore_polygon = domain.difference(land)
    if offshore_polygon.is_empty:
        raise ValueError(
            "domain - land is empty; no offshore region to anchor a wave boundary"
        )

    if offshore_polygon.geom_type == "Polygon":
        offshore_rings = [offshore_polygon.exterior]
    elif offshore_polygon.geom_type == "MultiPolygon":
        offshore_rings = [g.exterior for g in offshore_polygon.geoms]
    else:
        raise TypeError(
            f"unexpected offshore geometry type: {offshore_polygon.geom_type}"
        )

    offshore_exterior = unary_union([LineString(r.coords) for r in offshore_rings])
    seaward_edge = offshore_exterior.difference(land.boundary.buffer(float(tolerance_m)))
    if ocean_seed is not None:
        seaward_edge = _nearest_line_parts(
            seaward_edge,
            ocean_seed,
            margin_m=float(edge_selection_margin_m),
        )
    if seaward_edge.is_empty:
        raise ValueError(
            "seaward edge is empty; check that land actually intersects the domain"
        )
    return offshore_polygon, seaward_edge

def _line_parts(geometry: BaseGeometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        if len(coords) <= 2:
            return [geometry]
        return [
            LineString([start, stop])
            for start, stop in zip(coords[:-1], coords[1:])
            if start != stop
        ]
    if geometry.geom_type == "MultiLineString":
        parts: list[LineString] = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    if geometry.geom_type == "GeometryCollection":
        parts: list[LineString] = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    raise TypeError(f"unexpected line geometry type: {geometry.geom_type}")

def _nearest_line_parts(
    geometry: BaseGeometry,
    seed: BaseGeometry,
    *,
    margin_m: float,
) -> BaseGeometry:
    parts = _line_parts(geometry)
    if not parts:
        return geometry

    distances = np.array([part.distance(seed) for part in parts], dtype=float)
    min_distance = float(distances.min())
    keep = [
        part
        for part, distance in zip(parts, distances)
        if distance <= min_distance + margin_m
    ]
    return unary_union(keep)

def select_snapwave_boundary_points(
    offshore_edge: LineString | MultiLineString,
    offshore_polygon: Polygon,
    spacing_m: float,
    crs,
) -> gpd.GeoDataFrame:
    # place points along the edge every spacing_m, keep those inside polygon
    length = offshore_edge.length
    n = int(np.floor(length / spacing_m)) + 1
    distances = np.arange(n) * spacing_m
    raw_points = [offshore_edge.interpolate(d) for d in distances]
    kept = [p for p in raw_points if offshore_polygon.contains(p) or offshore_polygon.touches(p)]
    names = [f"{i + 1:04d}" for i in range(len(kept))]
    return gpd.GeoDataFrame({"name": names, "geometry": kept}, crs=crs)

def write_runup_gauges_file(path, transects) -> None:
    # SFINCS rug file: per transect, name line, "2 2" header, then x0 y0 and x1 y1.
    # SFINCS only reads the first two vertices; emitting more would trigger a warning.
    lines = []
    for name, (x0, y0), (x1, y1) in transects:
        lines.append(name)
        lines.append("2 2")
        lines.append(f"{x0:.6f} {y0:.6f}")
        lines.append(f"{x1:.6f} {y1:.6f}")
    Path(path).write_text("\n".join(lines) + "\n")

def validate_runup_transects(transects, region: BaseGeometry, *, min_length_m: float = 1.0) -> None:
    """Reject runup transects that cannot sample the model region."""
    if region is None or region.is_empty:
        raise ValueError("runup gauge validation requires a non-empty model region")
    invalid = []
    for name, (x0, y0), (x1, y1) in transects:
        line = LineString([(float(x0), float(y0)), (float(x1), float(y1))])
        if line.length < float(min_length_m):
            invalid.append(f"{name} is shorter than {float(min_length_m):g} m")
        elif not line.intersects(region):
            invalid.append(f"{name} does not intersect the model region")
    if invalid:
        raise ValueError("Invalid runup transect(s): " + "; ".join(invalid))

def derive_snapwave_boundary_points(
    sf,
    *,
    min_dist: float | None = None,
    bnd_dist: float = 5000.0,
) -> gpd.GeoDataFrame:
    """Build SnapWave open-boundary points from `snapwave_mask == 2` cells.

    Workaround for hydromt-sfincs 2.0.0rc1 bug at
    `snapwave_boundary_conditions.SnapWaveBoundaryConditions.get_boundary_points_from_mask`
    which calls `self.model.quadtree_grid.face_coordinates()` with parens —
    but `face_coordinates` is a `@property` on the quadtree grid component.
    Every other call site in the package (water_level, elevation, infiltration,
    snapwave_quadtree_mask.build) accesses it without parens. Once upstream is
    patched this helper can be deleted and the notebook can call the upstream
    method directly.

    Algorithm matches the upstream `get_boundary_points_from_mask`: walk the
    mask==2 cells into polylines, then interpolate `bnd_dist`-spaced points
    along each polyline. The result is written back to
    `sf.snapwave_boundary_conditions.data` and a constant placeholder
    timeseries is set via `set_timeseries`.
    """
    grid = sf.quadtree_grid
    if min_dist is None:
        min_dist = float(grid.data.attrs["dx"]) * 2
    mask = grid.data["snapwave_mask"]
    ibnd = np.where(mask == 2)
    xz, yz = grid.face_coordinates  # property — no parens (the upstream bug)
    xp = xz[ibnd]
    yp = yz[ibnd]
    used = np.full(xp.shape, False, dtype=bool)
    polylines: list[list[int]] = []
    while True:
        if np.all(used):
            break
        i1 = int(np.where(~used)[0][0])
        used[i1] = True
        polyline = [i1]
        # Walk forward from i1.
        while True:
            xpu = xp[~used]
            ypu = yp[~used]
            unused_indices = np.where(~used)[0]
            dst = np.sqrt((xpu - xp[i1]) ** 2 + (ypu - yp[i1]) ** 2)
            if np.all(np.isnan(dst)):
                break
            inear = int(np.nanargmin(dst))
            inearall = int(unused_indices[inear])
            if dst[inear] < min_dist:
                polyline.append(inearall)
                used[inearall] = True
                i1 = inearall
            else:
                break
        # Walk backward from polyline[0].
        i1 = polyline[0]
        while True:
            if np.all(used):
                break
            xpu = xp[~used]
            ypu = yp[~used]
            unused_indices = np.where(~used)[0]
            dst = np.sqrt((xpu - xp[i1]) ** 2 + (ypu - yp[i1]) ** 2)
            inear = int(np.nanargmin(dst))
            inearall = int(unused_indices[inear])
            if dst[inear] < min_dist:
                polyline.insert(0, inearall)
                used[inearall] = True
                i1 = inearall
            else:
                break
        if len(polyline) > 1:
            polylines.append(polyline)
    gdf_rows: list[dict] = []
    transformer = (
        Transformer.from_crs(sf.crs, 3857, always_xy=True)
        if sf.crs.is_geographic
        else None
    )
    ip = 0
    for polyline in polylines:
        x = xp[polyline]
        y = yp[polyline]
        line = LineString(list(zip(x.ravel(), y.ravel())))
        if transformer is not None:
            xm, ym = transformer.transform(x, y)
            line_m = LineString(list(zip(xm.ravel(), ym.ravel())))
            num_points = int(line_m.length / bnd_dist) + 2
        else:
            num_points = int(line.length / bnd_dist) + 2
        new_points = [
            line.interpolate(i / float(num_points - 1), normalized=True)
            for i in range(num_points)
        ]
        for point in new_points:
            ip += 1
            gdf_rows.append({
                "name": f"{ip:04d}",
                "timeseries": pd.DataFrame(),
                "geometry": point,
            })
    bnd_gdf = gpd.GeoDataFrame(gdf_rows, crs=sf.crs)
    sf.snapwave_boundary_conditions.data = bnd_gdf
    sf.snapwave_boundary_conditions.set_timeseries(
        shape="constant",
        timestep=600.0,
        hs=1.0,
        tp=10.0,
        wd=270.0,
        ds=20.0,
    )
    return bnd_gdf

def inject_runup_config(
    inp_path,
    rugfile: str,
    rugdepth: float = 0.05,
    obsfile: str | None = None,
) -> None:
    """Inject mt_Faber run-up gauge config into ``sfincs.inp``.

    ``rugfile`` and ``rugdepth`` are official mt_Faber SFINCS keys, but this
    branch has a source bug in ``read_rug_file``: it checks ``obsfile`` while
    reporting a missing rug file. When no real observation file exists, we write
    a tiny one-point obsfile beside the model to let run-up gauges initialize.
    """
    path = Path(inp_path)
    updates = {"rugfile": rugfile, "rugdepth": rugdepth}
    support_obsfile = obsfile or f"{Path(rugfile).name}.obs"
    lines = path.read_text().splitlines()
    seen = set()
    has_real_obsfile = False
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        value = line.split("=", 1)[1].strip() if "=" in line else ""
        if key == "runupfile":
            continue
        if key == "rugfile":
            if "rugfile" not in seen:
                out.append(_sfincs_inp_line("rugfile", updates["rugfile"]))
                seen.add("rugfile")
            continue
        if key == "rugdepth":
            out.append(_sfincs_inp_line("rugdepth", updates["rugdepth"]))
            seen.add("rugdepth")
        elif key == "obsfile":
            if value and value[:4] != "none" and "obsfile" not in seen:
                out.append(line)
                seen.add("obsfile")
                has_real_obsfile = True
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(_sfincs_inp_line(key, value))
    if not has_real_obsfile:
        _write_runup_obs_workaround(path.parent, rugfile, support_obsfile)
        out.append(_sfincs_inp_line("obsfile", support_obsfile))
    path.write_text("\n".join(out) + "\n")

def _sfincs_inp_line(key: str, value) -> str:
    return f"{key:<21}= {value}"

def _write_runup_obs_workaround(model_root: Path, rugfile: str, obsfile: str) -> None:
    x, y = _first_runup_gauge_xy(model_root, rugfile)
    obs_path = model_root / obsfile
    obs_path.write_text(f"{x:.6f} {y:.6f} 'runup_gauge_workaround'\n")

def _first_runup_gauge_xy(model_root: Path, rugfile: str) -> tuple[float, float]:
    rug_path = Path(rugfile)
    if not rug_path.is_absolute():
        rug_path = model_root / rug_path
    lines = [line.strip() for line in rug_path.read_text().splitlines() if line.strip()]
    i = 0
    while i + 2 < len(lines):
        i += 1  # gauge name
        _nrows = int(lines[i].split()[0])
        i += 1
        x, y = (float(value) for value in lines[i].split()[:2])
        return x, y
    raise ValueError(f"no run-up gauge coordinate found in {rug_path}")

def repair_snapwave_directional_spreading_file(
    model_root,
    *,
    spread_degrees: float = 20.0,
    reference_filename: str = "snapwave.bhs",
    bds_filename: str = "snapwave.bds",
) -> Path:
    """Rewrite ``snapwave.bds`` after hydromt-sfincs writes invalid values.

    hydromt-sfincs 2.0.0rc1 mutates the previous wave-direction dataframe while
    writing ``snapwave.bwd`` and then reuses it for ``snapwave.bds``. The result
    is a negative nanosecond-derived time axis and direction values where
    directional spreading should be. Use the stable ``snapwave.bhs`` time axis
    and boundary-point count, then fill directional spreading explicitly.
    """
    root = Path(model_root)
    reference_path = root / reference_filename
    bds_path = root / bds_filename
    reference = pd.read_csv(reference_path, sep=r"\s+", header=None)
    if reference.shape[1] < 2:
        raise ValueError(
            f"{reference_path} must contain a time column and at least one boundary point"
        )

    n_boundary_points = reference.shape[1] - 1
    values = np.column_stack(
        [
            reference.iloc[:, 0].to_numpy(dtype=float),
            np.full((len(reference), n_boundary_points), float(spread_degrees)),
        ]
    )
    lines = [
        " ".join(f"{value:.3f}" for value in row)
        for row in values
    ]
    bds_path.write_text("\n".join(lines) + "\n")
    return bds_path
