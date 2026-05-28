import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/surge-mpl-config")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu

from sfincs_runs.config import build_paths, load_config
from sfincs_runs.scenarios.io import parse_sfincs_inp


def read_points(path):
    if not Path(path).exists():
        return np.empty((0, 2), float)
    rows = [line.split()[:2] for line in Path(path).read_text().splitlines() if line.strip()]
    return np.asarray(rows, float) if rows else np.empty((0, 2), float)


def project_crs():
    return load_config().get("project", {}).get("model_crs", "EPSG:26919")


def sfincs_context(crs):
    base = build_paths()["base_model_root"]
    inp = parse_sfincs_inp(base / "sfincs.inp")
    if inp.get("epsg") and not str(crs).upper().endswith(inp["epsg"]):
        crs = f"EPSG:{inp['epsg']}"
    return crs, read_points(base / inp.get("bndfile", "sfincs.bnd")), read_points(base / inp.get("srcfile", "sfincs.src"))


def signed_area(xy):
    return 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])


def centroid(xy):
    area2 = 2.0 * signed_area(xy)
    if abs(area2) < 1e-12:
        return xy.mean(axis=0)
    f = xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]
    return np.array([
        np.sum((xy[:, 0] + np.roll(xy[:, 0], -1)) * f) / (3.0 * area2),
        np.sum((xy[:, 1] + np.roll(xy[:, 1], -1)) * f) / (3.0 * area2),
    ])


def clean_faces(raw_faces, node_xy):
    faces, keep, areas = [], [], []
    for face_id, raw in enumerate(raw_faces):
        idx = raw[raw >= 0].astype(np.int32)
        if idx.size < 3:
            continue
        area = signed_area(node_xy[idx])
        if abs(area) < 1e-8:
            continue
        faces.append(idx if area > 0 else idx[::-1])
        keep.append(face_id)
        areas.append(abs(area))
    out = np.full((len(faces), max(len(f) for f in faces)), -1, np.int32)
    for i, face in enumerate(faces):
        out[i, : len(face)] = face
    return out, np.asarray(keep, np.int32), np.asarray(areas, float)


def graph_edges(face_cell, n_cells):
    edges, boundary = [], []
    for left, right in face_cell:
        left_ok, right_ok = left < n_cells, right < n_cells
        if left_ok and right_ok:
            edges.append((int(left), int(right)))
        elif left_ok or right_ok:
            boundary.append(int(left if left_ok else right))
    return np.asarray(edges, np.int32), np.unique(np.asarray(boundary, np.int32))


def remap_graph(edges, boundary, keep, n_cells):
    if len(keep) == n_cells:
        return edges, boundary
    if edges.size == 0:
        return edges.reshape(0, 2), boundary
    mask = np.zeros(n_cells, bool)
    mask[keep] = True
    remap = np.full(n_cells, -1, np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    return remap[edges[mask[edges[:, 0]] & mask[edges[:, 1]]]], remap[boundary[mask[boundary]]]


def components(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen, out = np.zeros(n, bool), []
    for start in range(n):
        if seen[start]:
            continue
        q, comp = deque([start]), []
        seen[start] = True
        while q:
            i = q.popleft()
            comp.append(i)
            for j in adj[i]:
                if not seen[j]:
                    seen[j] = True
                    q.append(j)
        out.append(np.asarray(comp, np.int32))
    return sorted(out, key=len, reverse=True)


def graph_distance(n, edges, seed):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    dist, q = np.full(n, -1, np.int32), deque()
    for i in seed:
        dist[int(i)] = 0
        q.append(int(i))
    while q:
        i = q.popleft()
        for j in adj[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    return dist


def nearest_faces(centers, xy):
    if xy.size == 0:
        return np.empty(0, np.int32)
    return np.argmin(((centers[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2), axis=0).astype(np.int32)


def polyline_distance(points, line):
    if len(line) < 2:
        return np.full(len(points), np.inf)
    best = np.full(len(points), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        den = float(ab @ ab)
        t = 0 if den == 0 else np.clip(((points - a) @ ab) / den, 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return best


def sample_polyline(xy, spacing=2000.0):
    if len(xy) < 2:
        return xy.copy()
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dist = np.r_[0.0, np.cumsum(seg)]
    if dist[-1] == 0:
        return xy[:1].copy()
    samples = np.arange(0, dist[-1] + spacing * 0.5, spacing)
    return np.asarray([xy[min(np.searchsorted(dist, s, "right") - 1, len(seg) - 1)] + ((s - dist[min(np.searchsorted(dist, s, "right") - 1, len(seg) - 1)]) / max(seg[min(np.searchsorted(dist, s, "right") - 1, len(seg) - 1)], 1e-12)) * np.diff(xy, axis=0)[min(np.searchsorted(dist, s, "right") - 1, len(seg) - 1)] for s in samples])


def east_line(perimeter):
    i, out = int(np.argmax(perimeter[:, 0])), []
    for _ in range(len(perimeter)):
        out.append(perimeter[i])
        j = (i + 1) % len(perimeter)
        step = perimeter[j] - perimeter[i]
        if out and abs(step[0]) > abs(step[1]):
            break
        i = j
    return np.asarray(out, float)


def recommend_east_boundary_faces(centers, boundary, perimeter, areas):
    line = east_line(perimeter)
    d = polyline_distance(centers[boundary], line)
    typical = float(np.sqrt(np.median(areas[boundary])))
    east = boundary[d <= max(1.75 * typical, 225.0)]
    east = east[np.argsort(centers[east, 1])]
    samples = sample_polyline(centers[east], 2000.0)
    return east.astype(np.int32), line, samples


def load_unstructured_mesh_from_hdf(mesh_path, area_name="2D Area 1", clean_mesh_path=None):
    mesh_path = Path(mesh_path)
    clean_mesh_path = Path(clean_mesh_path or mesh_path.with_name(f"{mesh_path.stem}_clean.ugrid.nc"))
    crs = project_crs()

    with h5py.File(mesh_path, "r") as hdf:
        area = hdf[f"Geometry/2D Flow Areas/{area_name}"]
        n_cells = int(hdf["Geometry/2D Flow Areas/Attributes"][0]["Cell Count"])
        node_xy = area["FacePoints Coordinate"][:]
        raw_faces = area["Cells FacePoint Indexes"][:n_cells]
        raw_centers = area["Cells Center Coordinate"][:n_cells]
        face_cell = area["Faces Cell Indexes"][:]
        perimeter = area["Perimeter"][:]
        faces, keep, areas = clean_faces(raw_faces, node_xy)
        centers = raw_centers[keep].copy()
        if len(keep) != n_cells or np.linalg.norm(centers - raw_centers[keep], axis=1).mean() > 25:
            centers = np.vstack([centroid(node_xy[row[row >= 0]]) for row in faces])
        edges, boundary = remap_graph(*graph_edges(face_cell, n_cells), keep, n_cells)
        comps = components(len(faces), edges)
        component_id = np.full(len(faces), -1, np.int32)
        for i, comp in enumerate(comps):
            component_id[comp] = i
        boundary_dist = graph_distance(len(faces), edges, boundary)
        east, east_xy, east_samples = recommend_east_boundary_faces(centers, boundary, perimeter, areas)
        east_dist = graph_distance(len(faces), edges, east)

        grid = xu.Ugrid2d(node_x=node_xy[:, 0], node_y=node_xy[:, 1], fill_value=-1, face_node_connectivity=faces, name="mesh2d", is_projected=True, crs=crs, start_index=0)
        ds = grid.to_dataset()
        ds.attrs.update(source_file=str(mesh_path), source_format="HEC-RAS Geometry HDF", Conventions="CF-1.9 UGRID-1.0")
        dim = grid.face_dimension
        for name, values in {
            "mesh2d_face_x": centers[:, 0],
            "mesh2d_face_y": centers[:, 1],
            "mesh2d_face_area": areas,
            "mesh2d_boundary_flag": np.isin(np.arange(len(faces)), boundary).astype(np.int8),
            "mesh2d_east_boundary_flag": np.isin(np.arange(len(faces)), east).astype(np.int8),
            "mesh2d_component_id": component_id,
            "mesh2d_boundary_distance": boundary_dist,
            "mesh2d_east_boundary_distance": east_dist,
        }.items():
            ds[name] = xr.DataArray(values, dims=[dim])

        def attr(name, default=""):
            value = hdf.attrs.get(name, default)
            return value.decode() if hasattr(value, "decode") else str(value)

        report = NS(
            mesh_path=mesh_path,
            clean_mesh_path=clean_mesh_path,
            area_name=area_name,
            file_type=attr("File Type"),
            file_version=attr("File Version"),
            units_system=attr("Units System"),
            crs=crs,
            raw_center_count=int(area["Cells Center Coordinate"].shape[0]),
            declared_cell_count=n_cells,
            dropped_ghost_cells=int(area["Cells Center Coordinate"].shape[0]) - n_cells,
            valid_cell_count=len(faces),
            facepoint_count=len(node_xy),
            face_count=int(face_cell.shape[0]),
            boundary_face_count=int(len(boundary)),
            min_nodes_per_face=int(np.min(np.sum(faces >= 0, axis=1))),
            max_nodes_per_face=int(np.max(np.sum(faces >= 0, axis=1))),
            min_face_area=float(areas.min()),
            mean_face_area=float(areas.mean()),
            max_face_area=float(areas.max()),
            mesh_is_valid=bool(len(faces) and np.all(areas > 0)),
        )

    crs, boundary_xy, source_xy = sfincs_context(crs)
    return NS(
        report=report, grid=grid, dataset=ds, node_x=node_xy[:, 0], node_y=node_xy[:, 1],
        face_node_connectivity=faces, face_centers=centers, face_areas=areas,
        adjacency_edges=edges, boundary_faces=boundary, perimeter_xy=perimeter,
        east_boundary_line_xy=east_xy, east_boundary_faces=east, east_boundary_distance=east_dist,
        east_boundary_sample_xy=east_samples, component_id=component_id, boundary_distance=boundary_dist,
        boundary_xy=boundary_xy, legacy_source_xy=source_xy, legacy_boundary_face_ids=nearest_faces(centers, boundary_xy),
        legacy_source_face_id=(int(nearest_faces(centers, source_xy)[0]) if source_xy.size else None),
    )


def export_clean_mesh(mesh):
    ds = mesh.dataset.copy()
    for name, xy, dim in [
        ("perimeter", mesh.perimeter_xy, "mesh2d_nPerimeter"),
        ("east_line", mesh.east_boundary_line_xy, "mesh2d_nEastLine"),
        ("east_sample", mesh.east_boundary_sample_xy, "mesh2d_nEastSample"),
    ]:
        if xy.size:
            ds[f"mesh2d_{name}_x"] = xr.DataArray(xy[:, 0], dims=[dim])
            ds[f"mesh2d_{name}_y"] = xr.DataArray(xy[:, 1], dims=[dim])
    mesh.report.clean_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(mesh.report.clean_mesh_path)
    return mesh.report.clean_mesh_path


def _perimeter_from_ugrid(ds):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    node_x, node_y = ds["mesh2d_node_x"].values, ds["mesh2d_node_y"].values
    fill = ds["mesh2d_face_nodes"].attrs.get("_FillValue", -1)
    polys = []
    for row in ds["mesh2d_face_nodes"].values:
        idx = np.asarray(row)
        idx = idx[np.isfinite(idx) & (idx != fill) & (idx >= 0)].astype(int)
        if len(idx) >= 3:
            polys.append(Polygon(np.c_[node_x[idx], node_y[idx]]))
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return np.asarray(merged.exterior.coords, float)


def derive_east_boundary_from_ugrid(ugrid_path, *, sample_spacing_m=2000.0):
    ds = xr.open_dataset(ugrid_path).load()
    get_xy = lambda stem: np.c_[ds[f"mesh2d_{stem}_x"].values, ds[f"mesh2d_{stem}_y"].values].astype(float)
    perimeter = get_xy("perimeter") if "mesh2d_perimeter_x" in ds else _perimeter_from_ugrid(ds)
    line = get_xy("east_line") if "mesh2d_east_line_x" in ds else east_line(perimeter)
    if "mesh2d_east_sample_x" in ds:
        samples = get_xy("east_sample")
    else:
        flag = ds["mesh2d_east_boundary_flag"].values.astype(bool)
        centers = np.c_[ds["mesh2d_face_x"].values, ds["mesh2d_face_y"].values][flag]
        samples = sample_polyline(centers[np.argsort(centers[:, 1])], sample_spacing_m)
    return perimeter, line, samples


def validation_summary(mesh):
    r = mesh.report
    return {
        "mesh_path": str(r.mesh_path), "clean_mesh_path": str(r.clean_mesh_path), "crs": r.crs,
        "mesh_is_valid": r.mesh_is_valid, "declared_cell_count": r.declared_cell_count,
        "valid_cell_count": r.valid_cell_count, "dropped_ghost_cells": r.dropped_ghost_cells,
        "boundary_face_count": r.boundary_face_count, "east_boundary_face_count": int(len(mesh.east_boundary_faces)),
        "connected_components": int(mesh.component_id.max() + 1),
        "max_boundary_distance": int(mesh.boundary_distance.max()),
        "max_east_boundary_distance": int(mesh.east_boundary_distance.max()),
        "legacy_boundary_point_count": int(len(mesh.boundary_xy)),
        "legacy_source_point_count": int(len(mesh.legacy_source_xy)),
        "legacy_source_face_id": mesh.legacy_source_face_id,
    }


def forcing_recommendations(mesh):
    return {
        "recommended_keep_legacy_source_active": False,
        "legacy_source_face_id": mesh.legacy_source_face_id,
        "legacy_boundary_point_count": int(len(mesh.boundary_xy)),
        "recommended_east_boundary_face_count": int(len(mesh.east_boundary_faces)),
        "recommended_east_boundary_sample_count": int(len(mesh.east_boundary_sample_xy)),
        "recommended_forcing_direction": "Apply eastern external-edge boundary forcing westward through graph adjacency.",
    }


def plot_points(mesh, values=None, ax=None, title="Mesh"):
    fig, ax = plt.subplots(figsize=(9, 8)) if ax is None else (ax.figure, ax)
    values = np.zeros(len(mesh.face_centers)) if values is None else values
    sc = ax.scatter(mesh.face_centers[:, 0], mesh.face_centers[:, 1], c=values, s=4, cmap="viridis")
    ax.scatter(mesh.face_centers[mesh.east_boundary_faces, 0], mesh.face_centers[mesh.east_boundary_faces, 1], s=10, c="#2a9d8f")
    if mesh.boundary_xy.size:
        ax.scatter(mesh.boundary_xy[:, 0], mesh.boundary_xy[:, 1], s=20, facecolors="none", edgecolors="black")
    if mesh.east_boundary_sample_xy.size:
        ax.scatter(mesh.east_boundary_sample_xy[:, 0], mesh.east_boundary_sample_xy[:, 1], s=24, marker="s", c="white", edgecolors="#2a9d8f")
    ax.plot(mesh.east_boundary_line_xy[:, 0], mesh.east_boundary_line_xy[:, 1], color="#006d5b")
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    return fig, ax


def plot_mesh_overview(mesh, ax=None):
    return plot_points(mesh, None, ax, "Imported mesh and east forcing edge")


def plot_component_map(mesh, ax=None):
    return plot_points(mesh, mesh.component_id, ax, "Connected components")


def plot_boundary_distance(mesh, ax=None):
    return plot_points(mesh, mesh.boundary_distance, ax, "Boundary graph distance")


def plot_local_graph(mesh, ax=None, radius_steps=7):
    if mesh.legacy_source_face_id is not None:
        source = int(mesh.legacy_source_face_id)
    else:
        source = int(mesh.east_boundary_faces[len(mesh.east_boundary_faces) // 2])
    keep = np.linalg.norm(mesh.face_centers - mesh.face_centers[source], axis=1) < 1200
    fig, ax = plt.subplots(figsize=(8, 8)) if ax is None else (ax.figure, ax)
    ax.scatter(mesh.face_centers[keep, 0], mesh.face_centers[keep, 1], c=mesh.boundary_distance[keep], s=18)
    ax.scatter(mesh.face_centers[source, 0], mesh.face_centers[source, 1], s=120, marker="*", c="#c1121f")
    ax.set(title=f"Local graph around face {source}", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    return fig, ax


def plot_validation_panel(mesh):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, fn in zip(axes.ravel(), [plot_mesh_overview, plot_component_map, plot_boundary_distance, plot_local_graph]):
        fn(mesh, ax=ax)
    fig.tight_layout()
    return fig, axes


def save_default_figures(mesh, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, fn in {"mesh_overview": plot_mesh_overview, "components": plot_component_map, "boundary_distance": plot_boundary_distance, "local_graph": plot_local_graph, "validation_panel": plot_validation_panel}.items():
        fig, _ = fn(mesh)
        outputs[name] = outdir / f"unstructured_mesh_{name}.png"
        fig.savefig(outputs[name], dpi=180, bbox_inches="tight")
        plt.close(fig)
    return outputs


def main(mesh_path=None, clean_path=None, figures_dir=None):
    paths = build_paths()
    project = load_config().get("project", {}).get("name", "marshfield")
    mesh_path = Path(mesh_path or paths["raw_root"] / "mesh" / "mfield_mesh_v2.hdf")
    clean_path = Path(clean_path or paths["static_root"] / "unstructured_mfield_mesh_ugrid.nc")
    figures_dir = Path(figures_dir or paths["outputs_root"] / "figures" / "mesh_validation")
    mesh = load_unstructured_mesh_from_hdf(mesh_path, clean_mesh_path=clean_path)
    export_clean_mesh(mesh)
    save_default_figures(mesh, figures_dir)
    for k, v in validation_summary(mesh).items():
        print(f"{k}: {v}")
    print(f"project: {project}")


if __name__ == "__main__":
    main()
